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
- 两个设备都用 ``with`` 上下文管理器独占打开。
"""

from __future__ import annotations

import csv
import json
import math
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
]


#: 零级峰值占满量程的比例上限 (超过即告警: 过曝会把效率严重低估)。
SATURATION_WARN_FRACTION: float = 0.8

#: 粗略 ±1 级 ROI 半径相对估算间距的上限倍率; 不能再大, 否则三个 ROI 相交。
_ROI_RADIUS_FRACTION: float = 0.35


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
    clean: np.ndarray, cy: float, cx: float, radius: float
) -> tuple[float, tuple[float, float]]:
    """在圆 ROI 内积分能量, 同时给出强度加权重心 ``(cy, cx)``。"""
    height, width = clean.shape
    radius = max(1.0, float(radius))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(height, int(math.ceil(cy + radius)) + 1)
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(width, int(math.ceil(cx + radius)) + 1)
    # ROI 完全落在画面外 (小帧/边缘瓣): 能量为 0, 不去构造负维度的网格。
    if y1 <= y0 or x1 <= x0:
        return 0.0, (float(cy), float(cx))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = clean[y0:y1, x0:x1]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    weights = np.where(inside, patch, 0.0)
    total = float(weights.sum())
    if total > 0.0:
        wy = float((weights * yy).sum() / total)
        wx = float((weights * xx).sum() / total)
    else:
        wy, wx = float(cy), float(cx)
    return total, (wy, wx)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """安全比值: 分母非正/非有限时返回 ``None`` (JSON 里是 null)。"""
    num = float(numerator)
    den = float(denominator)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return None
    return num / den


def integrate_lobes(
    frame: np.ndarray,
    *,
    cy: float,
    cx: float,
    period_px: int,
    axis: str = "x",
    separation_scale: float | None = None,
) -> dict[str, Any]:
    """按 0 级位置 + 光栅周期积分 0 / +1 / -1 三处 Fourier 瓣能量。

    这是**未定标的粗略归属**: ±1 级间距按 ``K ≈ 相机宽度 / (4·周期)`` 的量级
    估算 (2f 光路里一级位移 ~ f·λ/(P·d), 与 CCD 宽度同数量级时被周期数整除),
    没有几何标定时只能给一个量级估计。若调用方给了 ``separation_scale`` (例如
    来自已有标定的 K), 则直接用它当间距。结果中的 ``note`` 明确标注粗略性。
    """
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    height, width = clean.shape
    if separation_scale is None or float(separation_scale) <= 0.0:
        span = width if str(axis).lower() == "x" else height
        separation = float(span) / (2.0 * float(period_px))
    else:
        separation = float(separation_scale)
    separation = max(8.0, separation)
    radius = max(4.0, _ROI_RADIUS_FRACTION * separation)

    e_zero, zero_c = _roi_energy(clean, cy, cx, radius)
    if str(axis).lower() == "x":
        plus_c = (float(cy), float(cx + separation))
        minus_c = (float(cy), float(cx - separation))
    else:
        plus_c = (float(cy + separation), float(cx))
        minus_c = (float(cy - separation), float(cx))
    e_plus, plus_centroid = _roi_energy(clean, *plus_c, radius)
    e_minus, minus_centroid = _roi_energy(clean, *minus_c, radius)
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
    return {
        "energies": {"zero": e_zero, "plus_one": e_plus, "minus_one": e_minus},
        "ratios": ratios,
        "roi_radius_px": radius,
        "separation_px": separation,
        "centroids": {
            "zero": zero_c,
            "plus_one": plus_centroid,
            "minus_one": minus_centroid,
        },
        "calibrated": separation_scale is not None and float(separation_scale) > 0.0,
        "note": (
            "粗略/未定标: ±1 级落点由光栅周期估算, 无几何标定 .npz; "
            "比值仅作相对指标, 不代表绝对衍射效率"
        ),
    }


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
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["region", "energy_counts", "lobe_ratio", "relative_to_baseline_zero"]
        )
        for name in ("zero", "plus_one", "minus_one"):
            writer.writerow(
                [
                    name,
                    lobe["energies"][name],
                    lobe["ratios"][name],
                    relative.get(name),
                ]
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
) -> dict[str, Any]:
    """在单个灰度深度上做一次轻量衍射效率探针。

    Args:
        config: :class:`ProbeConfig`。
        slm_factory: 测试注入的 SLM 上下文管理器工厂 (默认真实 ``Santec``)。
        cam_factory: 测试注入的相机工厂 (默认 ``create_camera`` 注册表);
            调用形态 ``cam_factory(cam_type, cam_id, exposure_time_ms=...)``,
            返回值须支持 ``with`` 与 ``get_numpy_image``。

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

    with (
        make_slm() as slm,
        make_camera(
            config.cam_type,
            config.cam_id,
            exposure_time_ms=float(config.exposure_ms),
        ) as camera,
    ):
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
            "周期={}px 方向={} 帧数={}",
            config.slm_number,
            panel_shape[1],
            panel_shape[0],
            max_gray,
            config.cam_type,
            config.cam_id,
            config.exposure_ms,
            config.period_px,
            config.axis,
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
        lobe = integrate_lobes(
            blazed,
            cy=cy,
            cx=cx,
            period_px=int(config.period_px),
            axis=str(config.axis),
        )
        # 归一化效率: 以**平场零级同半径窗口**的能量为入射基准。它仍受窗口
        # 截断影响, 所以只作相对指标 (与 lobe["note"] 同口径)。
        radius_px = float(lobe["roi_radius_px"])
        baseline_zero = _roi_energy(prepare_frame(baseline), cy, cx, radius_px)[0]
        plus_one = float(lobe["energies"]["plus_one"])
        minus_one = float(lobe["energies"]["minus_one"])
        relative_efficiency = {
            "plus_one": _safe_ratio(plus_one, baseline_zero),
            "minus_one": _safe_ratio(minus_one, baseline_zero),
            "first_orders_sum": _safe_ratio(plus_one + minus_one, baseline_zero),
            "baseline_zero_window_counts": baseline_zero,
            "note": "相对平场零级窗口能量; 窗口截断/无几何标定, 非绝对衍射效率",
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
            "calibration": "none (argmax 0 级 + 周期估算 ±1 级; 粗略/未定标)",
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
