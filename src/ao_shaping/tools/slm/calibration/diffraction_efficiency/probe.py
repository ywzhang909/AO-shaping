"""闪耀光栅衍射效率的**轻量探针**后端 (单个/少数灰度深度, 无标定 .npz 依赖)。

与 :mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.measure` 的区别
------------------------------------------------------------------------------
``measure`` 是全量扫描: 必须先有光路标定 ``.npz`` (零级中心 + Kx/Ky), 必须先过
焦面门禁, 扫 21 点再拟合相位响应。本模块是**快速探针**: 只下发一个平场基线 +
一个闪耀光栅灰度图, 从帧内 ``argmax`` 定位零级 (2f 光路的 0 级 = 全局最大, 不是
相机几何中心), 按周期/pupil 几何给出**未定标的粗略** ±1 级落点, 积分三处能量,
输出 0 级 / ±1 级的占比。没有几何标定时 ±1 级落点只能靠光栅周期估算, 所以
结果**只作为相对指标** (标签 ``粗略/未定标``), 不冒充绝对衍射效率。

实验纪律 (AGENTS.md)
--------------------
- SLM 只用内存模式 (``video_mode=0``); DVI 模式 ``open()`` 会挂起。
- 相邻两次显示必须用不同内存槽: 统一经 ``SlotRotator.display_data`` —— 它内部
  选一个新槽并 ``time.sleep(wait_time_s)``, 调用方**不得**再重复 sleep。
- 平场灰度图用 ``np.full((h, w), gray, dtype=np.uint16)`` 直接走 raw 路径,
  **严禁** ``create_phase_from_array`` (它做 mod 2π, 会破坏 uint16 灰度)。
- 相机一律通过 ``BaseCamera`` 注册表
  (``from ao_shaping.drivers.ccd.common import create_camera, full_scale``) 构造,
  支持 ``--cam-type daheng|miicam|sim|...``。
- 满量程/饱和检查经 ``full_scale(frame, camera=cam)`` (相机生效位深优先),
  防止 12-bit 装进 uint16 容器的假满量程误判。
- 暗帧/背景按相机序列号 + **同曝光**从 ``BaseCamera.get_dark_frame`` 读取
  (激光关闭时采集的库存暗帧; 探针无权也不能关激光), 无可用暗帧时告警并
  在 JSON 里记 ``dark.available=false``, 不中断采集。
- 两个设备都用 ``with`` 上下文管理器独占打开, 且**先拿台面跨进程互斥锁**
  (``DeviceLock``) 再开设备: SLM 的 USB 句柄独占, 第二个进程 open 会被
  驱动拒绝 (``-10002``/``-200``), 现象却像"设备没插"。
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    BlazeSpec,
    blaze_gray,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame

__all__ = [
    "ProbeConfig",
    "probe_diffraction_efficiency",
    "locate_zero_order",
    "integrate_lobes",
    "measure_lobe_offsets",
]


#: 零级峰值占满量程的比例上限 (超过即告警: 过曝会把效率严重低估)。
SATURATION_WARN_FRACTION: float = 0.8

#: 粗略 ±1 级 ROI 半径相对估算间距的上限倍率; 不能再大, 否则三个 ROI 相交。
_ROI_RADIUS_FRACTION: float = 0.35

#: 台面跨进程互斥锁的等待上限 (s): 够等一次正常实验收尾, 又不至于无限挂住。
_LOCK_TIMEOUT_S: float = 1800.0


@contextmanager
def _bench_lock_context(
    *,
    lock_factory: Callable[[], Any] | None,
    slm_factory: Callable[[], Any] | None,
    cam_factory: Callable[..., Any] | None,
) -> Iterator[Any]:
    """真实硬件路径拿台面跨进程互斥锁; 注入工厂 (= 离线/测试路径) 不碰真锁。

    SLM 的 USB 句柄是**独占**的: 两个进程同时 open 会直接被驱动拒绝
    (``-10002``/``-200``), 而且失败方式看起来像"设备没插", 很难查。所以凡是
    真开设备的入口都必须先拿 :class:`DeviceLock`, 并在 ``finally`` 里释放。
    """
    if lock_factory is None and (slm_factory is not None or cam_factory is not None):
        # 调用方注入了设备工厂 ⇒ 这次不碰真实硬件, 也就不该占台面锁。
        yield None
        return

    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    make_lock = lock_factory or (lambda: DeviceLock.default())
    lock = make_lock()
    if not lock.acquire(timeout=_LOCK_TIMEOUT_S):
        raise RuntimeError(
            f"台面设备被其它进程占用 (holder={lock.holder()}), "
            f"锁文件 {lock.path}; 等它释放后再重试"
        )
    logger.info("衍射效率探针: 已获得台面互斥锁 {} (holder=本进程)", lock.path)
    try:
        yield lock
    finally:
        lock.release()
        logger.info("衍射效率探针: 已释放台面互斥锁")


@dataclass(frozen=True)
class ProbeConfig:
    """轻量衍射效率探针的配置。

    Attributes:
        out_dir: 输出目录 (写 ``summary.json`` / ``summary.csv``, 可选原始帧)。
        slm_number: SLM 设备编号 (1-8)。
        slm_wavelength: 工作波长 (nm), 仅用于日志/元数据。
        cam_type: ``create_camera`` 的注册名 (``daheng`` / ``miicam`` / ``sim`` ...)。
        cam_id: 相机设备号。
        exposure_ms: 曝光时间 (ms)。
        gray_depth: 闪耀斜坡最大灰度 (器件本征 gray 路径的 2π 灰度)。
            0 = 从设备读 ``slm._max_gray`` (SLM-200 @1064nm 典型读数 993),
            再回落到 ``MAX_GRAYSCALE_VALUE`` (1023, 硬件满量程, **不是** 2π)。
            显式给正值则按该值下限幅到硬件满量程。
        period_px: 光栅周期 (面板像元)。
        axis: 光栅方向 (``"x"`` = 相位沿列变化, ``"y"`` = 沿行)。
        separation_px: 已标定的 ±1 级间距 (相机像素); ``0`` = 按周期估算。
            🔴 估算式 ``span/(2·period)`` 只是量级, 在本台 f=125mm 2f 光路上
            误差可达 6 倍 (估算 81px vs 实测 ~471px), 故真实测量必须给这个值。
        lobe_axis: ±1 级在**相机**上的位移方向; 空串 = 与 ``axis`` 相同。
            轴的对应关系取决于 SLM 在光路里的安装朝向: 2026-11-21 实测本台
            ``axis="x"`` 的光栅在相机上沿**行**位移, 即需 ``lobe_axis="y"``。
            方向弄反时 ±1 窗口落在空区, 能量为 0 ⇒ 比值退化成纯噪声。
        frames: 每图案平均帧数 (``get_numpy_image(n_sample=frames, skip_first=True)``)。
        settle_s: 显示后 LC 稳定等待 (s), 交给 ``SlotRotator.display_data``。
        save_frames: 是否把每幅平均帧存成 ``.npy`` (用于离线复核)。
        subtract_dark: 是否扣暗帧。暗帧按相机序列号 + 曝光从
            :meth:`BaseCamera.get_dark_frame` 读取 (严格同曝光匹配, 不跨曝光
            套用); 没有可用暗帧时只告警、按不扣处理, 并在 JSON 里注明原因。
        slot_min / slot_max: 内存槽轮换范围。
    """

    out_dir: Path = Path("data/diffraction_efficiency/probe")
    slm_number: int = 1
    slm_wavelength: int = 1064
    cam_type: str = "daheng"
    cam_id: int = 0
    exposure_ms: float = 5.0
    gray_depth: int = 0
    period_px: int = 32
    axis: str = "x"
    separation_px: float = 0.0
    lobe_axis: str = ""
    frames: int = 3
    settle_s: float = 0.25
    save_frames: bool = False
    subtract_dark: bool = True
    slot_min: int = 2
    slot_max: int = 125

    def __post_init__(self) -> None:
        if int(self.period_px) < 2:
            raise ValueError(f"period_px 必须 >= 2, 得到 {self.period_px}")
        if str(self.axis).lower() not in ("x", "y"):
            raise ValueError(f"axis 必须是 'x' 或 'y', 得到 {self.axis!r}")
        if str(self.lobe_axis).lower() not in ("", "x", "y"):
            raise ValueError(
                f"lobe_axis 必须是 '' / 'x' / 'y', 得到 {self.lobe_axis!r}"
            )
        if float(self.separation_px) < 0.0:
            raise ValueError(
                f"separation_px 不能为负, 得到 {self.separation_px}"
            )
        if int(self.frames) < 1:
            raise ValueError(f"frames 必须 >= 1, 得到 {self.frames}")
        if int(self.gray_depth) < 0:
            raise ValueError(f"gray_depth 不能为负, 得到 {self.gray_depth}")
        if int(self.slot_min) < 1 or int(self.slot_max) < int(self.slot_min):
            raise ValueError(f"非法内存槽范围 [{self.slot_min}, {self.slot_max}]")
        if float(self.settle_s) < 0.0:
            raise ValueError(f"settle_s 不能为负, 得到 {self.settle_s}")


def locate_zero_order(frame: np.ndarray) -> tuple[float, float, float]:
    """在去背景画面里以 ``argmax`` 定位零级, 返回 ``(cy, cx, peak)``。

    2f 傅里叶光路的零级是**帧全局最大**, 不是相机几何中心 (实测帧中心
    (1344,760) vs 0 级 (1441,705))。先扣中位数并截断非负再 argmax, 避免
    对称读出噪声/热像素把定位带偏 (与 ``measure._locate_zero_order`` 同口径)。
    """
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    if clean.size == 0 or not np.isfinite(clean).all() or float(clean.max()) <= 0.0:
        raise RuntimeError("帧内无有效信号 (检查曝光 / 光路 / 图案是否上屏)")
    index = int(np.argmax(clean))
    cy, cx = divmod(index, clean.shape[1])
    return float(cy), float(cx), float(np.asarray(frame).max())


def _roi_energy(
    clean: np.ndarray,
    cy: float,
    cx: float,
    radius: float,
    *,
    annulus: tuple[float, float] = (2.0, 3.2),
) -> tuple[float, float, tuple[float, float]]:
    """圆 ROI 内积分能量并扣**本窗环带背景**; 返回 ``(能量, 背景中位数, 重心)``。

    ⚠️ 直接对"已扣整帧中位数、负值已截断"的帧积分, 会给**空窗**积出正的噪声
    基座: 半径 165 px 的空窗实测能积出约 5 万 counts, 与真实级次同量级, 于是
    空窗口会伪装成一个占 44% 能量的级次 (2026-11-21 实测踩过)。每个窗用自身
    环带 (``annulus`` × radius) 的中位数扣背景, 空窗才会诚实地报 ~0。
    """
    height, width = clean.shape
    radius = max(1.0, float(radius))
    r_out = max(radius + 1.0, float(annulus[1]) * radius)
    y0 = max(0, int(math.floor(cy - r_out)))
    y1 = min(height, int(math.ceil(cy + r_out)) + 1)
    x0 = max(0, int(math.floor(cx - r_out)))
    x1 = min(width, int(math.ceil(cx + r_out)) + 1)
    # ROI 完全落在画面外 (小帧/边缘瓣): 能量为 0, 不去构造负维度的网格。
    if y1 <= y0 or x1 <= x0:
        return 0.0, 0.0, (float(cy), float(cx))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    patch = clean[y0:y1, x0:x1]
    inside = dist2 <= radius * radius
    ring = (dist2 >= (float(annulus[0]) * radius) ** 2) & (
        dist2 <= (float(annulus[1]) * radius) ** 2
    )
    background = float(np.median(patch[ring])) if bool(ring.any()) else 0.0
    core = patch[inside]
    total = float(core.sum() - background * float(core.size))
    weights = np.where(inside, np.clip(patch - background, 0.0, None), 0.0)
    wsum = float(weights.sum())
    if wsum > 0.0:
        wy = float((weights * yy).sum() / wsum)
        wx = float((weights * xx).sum() / wsum)
    else:
        wy, wx = float(cy), float(cx)
    return total, background, (wy, wx)


def _spot_fwhm(frame: np.ndarray, cy: float, cx: float, *, half: int = 80) -> float:
    """粗估光斑 FWHM (px): 取 ``half`` 半径内行/列投影较宽的一边。

    只用来给 ROI 半径定标: 焦距窗口若按「级次间距的 35%」取 (本台 471 px ⇒ 165 px),
    半径会是实测光斑 (FWHM ≈15 px) 的 10 倍, 环带外的弥散晕就会主导积分。
    """
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    y0 = max(0, int(round(cy)) - int(half))
    y1 = min(clean.shape[0], int(round(cy)) + int(half) + 1)
    x0 = max(0, int(round(cx)) - int(half))
    x1 = min(clean.shape[1], int(round(cx)) + int(half) + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    patch = clean[y0:y1, x0:x1]
    widths: list[float] = []
    for profile in (patch.sum(axis=1), patch.sum(axis=0)):
        p = np.clip(np.asarray(profile, dtype=np.float64), 0.0, None)
        if p.size == 0 or float(p.max()) <= 0.0:
            continue
        above = np.where(p >= 0.5 * float(p.max()))[0]
        if above.size:
            widths.append(float(above[-1] - above[0] + 1))
    return max(widths) if widths else 0.0


def _axis_profile(frame: np.ndarray, cy: float, cx: float, axis: str) -> np.ndarray:
    """取穿过 0 级的 1-D 剖面 (``axis`` = 位移方向)。"""
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    if str(axis).lower() == "x":
        return np.clip(clean[int(round(cy)), :], 0.0, None)
    return np.clip(clean[:, int(round(cx))], 0.0, None)


def _peak_offset(profile: np.ndarray, center: int, lo: int, hi: int) -> float | None:
    """在 ``[lo, hi]`` 内找剖面最大值, 返回相对 ``center`` 的偏移 (三点抛物线亚像素)。"""
    lo = max(0, int(lo))
    hi = min(profile.size - 1, int(hi))
    if hi <= lo:
        return None
    window = profile[lo : hi + 1]
    if window.size == 0 or float(window.max()) <= 0.0:
        return None
    idx = int(np.argmax(window)) + lo
    left = float(profile[idx - 1]) if idx - 1 >= 0 else float(profile[idx])
    mid = float(profile[idx])
    right = float(profile[idx + 1]) if idx + 1 < profile.size else float(profile[idx])
    denom = left - 2.0 * mid + right
    shift = 0.5 * (left - right) / denom if denom != 0.0 else 0.0
    shift = float(np.clip(shift, -0.5, 0.5))
    return float(idx + shift - float(center))


def measure_lobe_offsets(
    frame: np.ndarray,
    *,
    cy: float,
    cx: float,
    axis: str = "x",
    search_px: float,
    min_offset_px: float = 0.0,
) -> dict[str, float | None]:
    """沿位移轴分别实测两侧最强瓣的**带符号偏移** (px)。

    **不假设 ±1 对称**: 闪耀光栅本身就是非对称器件 —— 相位斜坡把能量主要推进
    单一级次, 另一侧的共轭级次被闪耀包络压掉。本台 2026-11-21 实测 `--axis x`
    的光栅只在相机行方向出现**一个**一级次 (零级 + 单瓣), 另一侧窗口内没有任何
    光斑。所以"三窗各占 1/3"对闪耀光栅是错的读法; 这里改成实测落点 + 泄漏比。
    """
    profile = _axis_profile(frame, cy, cx, axis)
    center = int(round(cx)) if str(axis).lower() == "x" else int(round(cy))
    span = profile.size - 1
    lo_off = max(float(min_offset_px), 1.0)
    plus = _peak_offset(
        profile, center, center + int(lo_off), min(span, center + int(search_px))
    )
    minus = _peak_offset(
        profile, center, max(0, center - int(search_px)), center - int(lo_off)
    )
    return {"positive": plus, "negative": minus}


def integrate_lobes(
    frame: np.ndarray,
    *,
    cy: float,
    cx: float,
    period_px: int,
    axis: str = "x",
    separation_scale: float | None = None,
    lobe_axis: str | None = None,
    spot_fwhm: float | None = None,
) -> dict[str, Any]:
    """按 0 级位置 + 光栅周期积分 0 / 一级次能量, 并实测两侧落点。

    这是**未定标的粗略归属**: 级次间距按 ``K ≈ 相机跨度 / (2·周期)`` 的量级估算
    (2f 光路里一级位移 ~ f·λ/(P·d)), 没有几何标定时只能给一个量级估计。若调用方
    给了 ``separation_scale`` (例如来自已有标定的 K), 则直接用它当间距。

    每个窗口扣**自身环带背景** (空窗不再积出噪声基座), ROI 半径由 ``spot_fwhm``
    定标 (不传时退回 ``0.35·separation``)。

    Args:
        axis: 光栅方向 (SLM 图案侧)。
        lobe_axis: 级次在**相机**上的位移方向; ``None`` = 与 ``axis`` 相同。
            SLM 在光路里转了 90° 时二者不同 (2026-11-21 实测本台
            ``axis="x"`` → 相机行方向), 弄反会把窗口放进空区。
        spot_fwhm: 实测光斑 FWHM; 用来把 ROI 半径压到光斑量级。
    """
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    height, width = clean.shape
    disp_axis = str(lobe_axis or axis).lower()
    if separation_scale is None or float(separation_scale) <= 0.0:
        span = width if disp_axis == "x" else height
        separation = float(span) / (2.0 * float(period_px))
    else:
        separation = float(separation_scale)
    separation = max(8.0, separation)
    radius_cap = _ROI_RADIUS_FRACTION * separation
    if spot_fwhm is not None and float(spot_fwhm) > 0.0:
        radius = min(radius_cap, max(6.0, 1.5 * float(spot_fwhm)))
    else:
        radius = max(4.0, radius_cap)

    e_zero, bg_zero, zero_c = _roi_energy(clean, cy, cx, radius)
    if disp_axis == "x":
        plus_c = (float(cy), float(cx + separation))
        minus_c = (float(cy), float(cx - separation))
    else:
        plus_c = (float(cy + separation), float(cx))
        minus_c = (float(cy - separation), float(cx))
    e_plus, bg_plus, plus_centroid = _roi_energy(clean, *plus_c, radius)
    e_minus, bg_minus, minus_centroid = _roi_energy(clean, *minus_c, radius)
    total = e_zero + e_plus + e_minus
    ratios = (
        {
            "zero": e_zero / total,
            "plus_one": e_plus / total,
            "minus_one": e_minus / total,
        }
        if total > 0.0
        else {"zero": None, "plus_one": None, "minus_one": None}
    )
    # 闪耀光栅的级次是**非对称**的: 强的一侧是闪耀级次, 弱的一侧只是泄漏 floor。
    stronger_is_plus = e_plus >= e_minus
    blaze_energy = max(e_plus, e_minus)
    mirror_energy = min(e_plus, e_minus)
    orders = {
        "blaze_order": "plus_one" if stronger_is_plus else "minus_one",
        "blaze_energy": blaze_energy,
        "mirror_energy": mirror_energy,
        "mirror_leakage": _safe_ratio(mirror_energy, blaze_energy),
        "asymmetry": _safe_ratio(e_plus - e_minus, e_plus + e_minus),
    }
    measured = measure_lobe_offsets(
        frame,
        cy=cy,
        cx=cx,
        axis=disp_axis,
        search_px=1.6 * separation,
        min_offset_px=0.5 * separation,
    )
    offsets = [abs(v) for v in measured.values() if v is not None]
    measured_sep = max(offsets) if offsets else None
    calibrated = separation_scale is not None and float(separation_scale) > 0.0
    separation_check: dict[str, Any] = {
        "given_px": separation if calibrated else None,
        "measured_px": measured_sep,
        "consistent": None,
    }
    if calibrated and measured_sep is not None:
        separation_check["consistent"] = bool(
            abs(measured_sep - separation) <= 0.25 * separation
        )
    return {
        "energies": {"zero": e_zero, "plus_one": e_plus, "minus_one": e_minus},
        "ratios": ratios,
        "backgrounds": {"zero": bg_zero, "plus_one": bg_plus, "minus_one": bg_minus},
        "orders": orders,
        "roi_radius_px": radius,
        "separation_px": separation,
        "lobe_axis": disp_axis,
        "measured_offsets": measured,
        "separation_check": separation_check,
        "centroids": {
            "zero": zero_c,
            "plus_one": plus_centroid,
            "minus_one": minus_centroid,
        },
        "calibrated": calibrated,
        "note": (
            "±1 落点取自标定间距 (separation_px), 并已实测两侧瓣位置; "
            "闪耀光栅非对称, 弱侧是泄漏 floor 而非共轭级次; "
            "比值仅作相对指标, 不代表绝对衍射效率"
            if calibrated
            else "粗略/未定标: 落点由相机跨度与光栅周期估算, 无几何标定; "
            "本台估算式误差可达数倍, 给 --separation-px 后才是可信读数; "
            "比值仅作相对指标, 不代表绝对衍射效率"
        ),
    }



def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """安全比值: 分母非正/非有限时返回 ``None`` (JSON 里是 null)。"""
    num = float(numerator)
    den = float(denominator)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return None
    return num / den


def _average_frames(camera: Any, frames: int) -> np.ndarray:
    """按 ``BaseCamera.get_numpy_image`` 契约取多帧平均 (丢首帧防陈旧)。"""
    return np.asarray(
        camera.get_numpy_image(n_sample=int(frames), skip_first=True), dtype=np.float64
    )


def _load_dark(
    camera: Any, exposure_ms: float, *, enabled: bool
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """按序列号 + 曝光取暗帧 (严格同曝光匹配), 失败只告警不中断。

    暗帧来自 :mod:`ao_shaping.drivers.ccd.dark_frames` 的序列号存档库, 本身是
    **激光关闭**时采集的, 因此这里不需要也没法关激光 —— 探针只负责按曝光匹配
    并扣掉。没有可用暗帧 (未采集 / 曝光不匹配 / 后端无序列号) 时返回 ``None``,
    并把原因写进 JSON, 结果明确标注 ``dark_subtracted: false``。
    """
    info: dict[str, Any] = {
        "requested": bool(enabled),
        "available": False,
        "source": None,
        "exposure_ms": float(exposure_ms),
        "mean_counts": None,
        "reason": None,
    }
    if not enabled:
        info["reason"] = "disabled (--no-subtract-dark)"
        return None, info
    get_dark = getattr(camera, "get_dark_frame", None)
    if not callable(get_dark):
        info["reason"] = f"{type(camera).__name__} 没有 get_dark_frame 入口"
        return None, info
    try:
        dark = get_dark(float(exposure_ms))
    except Exception as exc:  # 暗帧不可用不应中断采集
        info["reason"] = f"get_dark_frame 失败: {type(exc).__name__}: {exc}"
        logger.warning("暗帧不可用, 按未扣暗处理: {}", info["reason"])
        return None, info
    if dark is None:
        info["reason"] = "无同曝光暗帧 (先用 tools/ccd/capture_darks 采集)"
        logger.warning("暗帧不可用, 按未扣暗处理: {}", info["reason"])
        return None, info
    mean = np.asarray(dark.mean, dtype=np.float64)
    info.update(
        {
            "available": True,
            "source": "BaseCamera.get_dark_frame (serial-keyed library)",
            "mean_counts": float(mean.mean()) if mean.size else None,
            "reason": None,
            "dark_exposure_ms": float(getattr(dark, "exposure_ms", exposure_ms)),
            "dark_frames": int(getattr(dark, "frames", 0) or 0),
        }
    )
    return mean, info


def _subtract_dark(frame: np.ndarray, dark: np.ndarray | None) -> np.ndarray:
    """扣暗并截断非负; 形状不一致时告警并原样返回 (换 ROI 的暗帧不可复用)。"""
    clean = np.asarray(frame, dtype=np.float64)
    if dark is None:
        return clean
    if dark.shape != clean.shape:
        logger.warning(
            "暗帧形状 {} 与画面 {} 不一致, 跳过扣暗 (换过 ROI / binning?)",
            dark.shape,
            clean.shape,
        )
        return clean
    return np.clip(clean - dark, 0.0, None)


def _saturation_warn(
    frame: np.ndarray, camera: Any, full_scale_fn: Callable[..., float]
) -> float:
    """用相机感知的满量程检查过曝, 返回满量程值。"""
    detector_full_scale = float(full_scale_fn(frame, camera=camera))
    peak = float(np.asarray(frame).max()) if np.asarray(frame).size else 0.0
    if (
        detector_full_scale > 0.0
        and peak / detector_full_scale > SATURATION_WARN_FRACTION
    ):
        logger.warning(
            "峰值 {:.0f} 达满量程 {:.0f} 的 {:.0f}% (> {:.0f}%): "
            "过曝会让衍射效率严重低估, 请降低曝光",
            peak,
            detector_full_scale,
            100.0 * peak / detector_full_scale,
            100.0 * SATURATION_WARN_FRACTION,
        )
    return detector_full_scale


def _write_outputs(
    result: dict[str, Any], payloads: dict[str, np.ndarray], out_dir: Path
) -> dict[str, str]:
    """写紧凑的 ``summary.json`` / ``summary.csv`` (可选 ``.npy`` 原始帧)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lobe = result["lobe"]
    relative = result.get("relative_efficiency", {})
    base_zero = relative.get("baseline_zero_window_counts")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["region", "energy_counts", "lobe_ratio", "relative_to_baseline_zero"]
        )
        for name in ("zero", "plus_one", "minus_one"):
            energy = lobe["energies"][name]
            writer.writerow(
                [
                    name,
                    energy,
                    lobe["ratios"][name],
                    (energy / base_zero) if base_zero else None,
                ]
            )
        # 闪耀级次汇总: 闪耀光栅两侧本就不对称, 逐窗三行读不出哪个是真正的一级。
        writer.writerow(
            ["blaze", lobe["orders"]["blaze_energy"], lobe["orders"]["blaze_order"],
             relative.get("blaze_energy_share")]
        )
        writer.writerow(
            ["mirror", lobe["orders"]["mirror_energy"], "leakage_floor",
             relative.get("mirror_leakage")]
        )
    written: dict[str, str] = {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
    }
    for name, frame in payloads.items():
        path = out_dir / f"{name}.npy"
        np.save(path, frame)
        written[name] = str(path)
    return written


def probe_diffraction_efficiency(
    config: ProbeConfig,
    *,
    slm_factory: Callable[[], Any] | None = None,
    cam_factory: Callable[..., Any] | None = None,
    lock_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """在单个灰度深度上做一次轻量衍射效率探针。

    Args:
        config: :class:`ProbeConfig`。
        slm_factory: 测试注入的 SLM 上下文管理器工厂 (默认真实 ``Santec``)。
        cam_factory: 测试注入的相机工厂 (默认 ``create_camera`` 注册表);
            调用形态 ``cam_factory(cam_type, cam_id, exposure_time_ms=...)``,
            返回值须支持 ``with`` 与 ``get_numpy_image``。
        lock_factory: 测试注入的台面锁工厂 (默认 ``DeviceLock.default()``)。
            只有注入它才会在注入设备工厂的同时仍去拿锁。

    Returns:
        紧凑的结果字典 (同时写入 ``out_dir/summary.json`` 与 ``summary.csv``)。
    """
    from ao_shaping.drivers.ccd.common import create_camera, full_scale
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator

    make_slm = slm_factory or (
        lambda: Santec(
            slm_number=int(config.slm_number),
            video_mode=0,  # DVI 模式 open() 会挂起
            wavelength=int(config.slm_wavelength),
        )
    )
    make_camera = cam_factory or create_camera

    with ExitStack() as stack:
        # 锁先拿、设备后开; 退出时按相反顺序先关设备再放锁 (ExitStack LIFO)。
        stack.enter_context(
            _bench_lock_context(
                lock_factory=lock_factory,
                slm_factory=slm_factory,
                cam_factory=cam_factory,
            )
        )
        slm = stack.enter_context(make_slm())
        camera = stack.enter_context(
            make_camera(
                config.cam_type,
                config.cam_id,
                exposure_time_ms=float(config.exposure_ms),
            )
        )
        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        max_gray = int(config.gray_depth) or int(
            getattr(slm, "_max_gray", None) or getattr(slm, "MAX_GRAYSCALE_VALUE", 1023)
        )
        max_gray = min(max_gray, int(getattr(slm, "MAX_GRAYSCALE_VALUE", max_gray)))
        rotator = SlotRotator(
            slm, slot_min=int(config.slot_min), slot_max=int(config.slot_max)
        )
        logger.info(
            "衍射效率探针: SLM#{} ({}×{}, 2π 灰度={}) 相机 {}#{} 曝光={}ms "
            "周期={}px 方向={} ±1 间距={} 级次方向={} 帧数={}",
            config.slm_number,
            panel_shape[1],
            panel_shape[0],
            max_gray,
            config.cam_type,
            config.cam_id,
            config.exposure_ms,
            config.period_px,
            config.axis,
            (
                f"{float(config.separation_px):.1f}px(标定)"
                if float(config.separation_px) > 0.0
                else "估算"
            ),
            str(config.lobe_axis) or str(config.axis),
            config.frames,
        )

        # 暗帧: 按序列号 + 同曝光取库存暗帧 (激光关闭时采集), 无则告警并跳过。
        dark_mean, dark_info = _load_dark(
            camera, float(config.exposure_ms), enabled=bool(config.subtract_dark)
        )

        # A: 平场基线 (raw uint16, 严禁 create_phase_from_array)。SlotRotator
        # 内部选新槽并 sleep(settle_s), 此处不再重复 sleep。
        flat = np.full(panel_shape, 0, dtype=np.uint16)
        flat_slot = rotator.display_data(slm, flat, float(config.settle_s))
        baseline_raw = _average_frames(camera, config.frames)
        baseline = _subtract_dark(baseline_raw, dark_mean)
        baseline_full_scale = _saturation_warn(baseline_raw, camera, full_scale)
        # 零级中心从**平场**帧的 argmax 取: 灰度深度接近 2π 时零级几乎被熄灭,
        # 闪耀帧的全局最大会落在 +1 级上, 用它定位零级会整体偏一个阶次间距。
        # 平场下所有能量都在零级, argmax 正是 0 级位置 (2f 光路的硬事实)。
        cy, cx, baseline_peak = locate_zero_order(baseline)

        # B: 闪耀光栅 (depth=2π ⇒ gray 深度 = 设备 2π 灰度)。
        spec = BlazeSpec(
            period_px=int(config.period_px),
            axis=str(config.axis),
            depth_rad=2.0 * math.pi,
        )
        blaze = blaze_gray(spec, panel_shape, max_gray)
        blaze_slot = rotator.display_data(slm, blaze, float(config.settle_s))
        blazed_raw = _average_frames(camera, config.frames)
        blazed = _subtract_dark(blazed_raw, dark_mean)
        blazed_full_scale = _saturation_warn(blazed_raw, camera, full_scale)

        blazed_peak = float(np.asarray(blazed).max())
        # 光斑 FWHM 用**平场**帧量 (此时全部能量都在零级, 是最干净的孤立方斑)。
        # ROI 半径由它定标: 按间距 35% 取会到 165px, 是实测光斑 (FWHM≈15px) 的
        # 十倍, 弥散晕与背景噪声就会主导积分。
        spot_fwhm = _spot_fwhm(prepare_frame(baseline), cy, cx)
        lobe = integrate_lobes(
            blazed,
            cy=cy,
            cx=cx,
            period_px=int(config.period_px),
            axis=str(config.axis),
            separation_scale=(
                float(config.separation_px) if float(config.separation_px) > 0.0 else None
            ),
            lobe_axis=(str(config.lobe_axis) or None),
            spot_fwhm=spot_fwhm,
        )
        plus_one = float(lobe["energies"]["plus_one"])
        minus_one = float(lobe["energies"]["minus_one"])
        # 空瓣自检: **闪耀级次那一侧**窗口里没能量, 说明落点整体放错了 (间距/方向
        # 错)。另一侧本来就只是泄漏 floor, 空是正常的, 不告警 (2026-11-21 实测:
        # 用默认估算间距 81px 时两侧全空, 而那正是落点错的信号)。
        if max(plus_one, minus_one) <= 0.0:
            logger.warning(
                "衍射效率探针: ±1 两侧窗口内都没有能量 —— 落点 (间距 {}px, 方向 {}) "
                "多半不对; 本台实测一级位移沿相机行方向约 471px, "
                "用 --separation-px 471 --lobe-axis y 重测 (或先做几何标定)。",
                lobe["separation_px"],
                lobe["lobe_axis"],
            )
        # 归一化效率: 以**平场零级同半径窗口**的能量为入射基准。它仍受窗口
        # 截断影响, 所以只作相对指标 (与 lobe["note"] 同口径)。
        radius_px = float(lobe["roi_radius_px"])
        baseline_zero = _roi_energy(prepare_frame(baseline), cy, cx, radius_px)[0]
        orders = dict(lobe["orders"])
        relative_efficiency = {
            "blaze_order": orders["blaze_order"],
            "blaze_energy_share": _safe_ratio(
                orders["blaze_energy"], baseline_zero
            ),
            "mirror_leakage": orders["mirror_leakage"],
            "first_orders_sum": _safe_ratio(plus_one + minus_one, baseline_zero),
            "baseline_zero_window_counts": baseline_zero,
            "note": (
                "相对平场零级窗口能量; 闪耀级次 = 两侧较强的一侧, 弱侧是泄漏 floor; "
                "窗口截断/无几何标定, 非绝对衍射效率"
            ),
        }

        result: dict[str, Any] = {
            "probe": "diffraction_efficiency",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "slm_number": config.slm_number,
                "slm_wavelength_nm": config.slm_wavelength,
                "cam_type": config.cam_type,
                "cam_id": config.cam_id,
                "exposure_ms": config.exposure_ms,
                "gray_depth": max_gray,
                "period_px": config.period_px,
                "axis": config.axis,
                "separation_px": float(config.separation_px),
                "lobe_axis": str(config.lobe_axis) or str(config.axis),
                "frames": config.frames,
                "settle_s": config.settle_s,
                "subtract_dark": bool(config.subtract_dark),
                "panel_shape": list(panel_shape),
            },
            "zero_order": {
                "cy": cy,
                "cx": cx,
                "from": "baseline argmax",
                "baseline_peak": baseline_peak,
                "blazed_frame_peak": blazed_peak,
            },
            "lobe": lobe,
            "spot_fwhm_px": spot_fwhm,
            "relative_efficiency": relative_efficiency,
            "peaks": {
                "baseline_counts": float(np.asarray(baseline).max()),
                "blazed_counts": blazed_peak,
            },
            "dark": dark_info,
            "full_scale": {
                "baseline": baseline_full_scale,
                "blazed": blazed_full_scale,
            },
            "slots": {"baseline": flat_slot, "blaze": blaze_slot},
            "saturation_warn_fraction": SATURATION_WARN_FRACTION,
            "calibration": (
                f"separation={lobe['separation_px']:.1f}px 来自 --separation-px; "
                "0 级 argmax, 无焦面门禁 (粗略/未定标)"
                if lobe["calibrated"]
                else "none (argmax 0 级 + 周期估算 ±1 级; 粗略/未定标)"
            ),
        }
        payloads: dict[str, np.ndarray] = {}
        if config.save_frames:
            payloads = {
                "baseline": baseline,
                "blazed": blazed,
                "baseline_raw": baseline_raw,
                "blazed_raw": blazed_raw,
            }
        result["artifacts"] = _write_outputs(result.copy(), payloads, config.out_dir)
        logger.info(
            "探针完成: 0级=({:.0f},{:.0f}) 峰值={:.0f}; 能量占比 0/+1/-1 = "
            "{:.3f}/{:.3f}/{:.3f} ({})",
            cy,
            cx,
            baseline_peak,
            lobe["ratios"]["zero"] or float("nan"),
            lobe["ratios"]["plus_one"] or float("nan"),
            lobe["ratios"]["minus_one"] or float("nan"),
            lobe["note"],
        )
        return result
