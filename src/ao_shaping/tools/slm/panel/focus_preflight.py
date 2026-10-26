"""采集前置: 焦面 (离焦) 补偿 + 去转向自标切, 作为采集流程的**一等前置阶段**。

为什么需要它
------------

2f 实验系统的 CCD **常驻离焦**。瞳面菲涅尔矫正之后, 平场光斑的 FWHM 从
~115 px 收到 ~15 px, 峰值/曝光提升约 29-31 倍 (见
``report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261022.md`` §0.3
与 ``data/debug/physical_forward_defocus/alignment_20261022_1812/focus.json``:
``best_focal_mm = 1783.54``, ``gain_vs_flat = 31.36``)。

在**未矫正焦面**的实验系统上拟合物理预测模型, 模型会把光斑预测得**更散**
(宽度比 1.328), CV 偏差 −0.3009 且 51/51 同向 —— 那不是模型误差, 是
**实验系统离焦**被模型当成了像差 (见
``report/shape_trials/experiment_summary_20261022.md`` §2)。所以焦面标定
必须发生在**采集之前**, 补偿相位必须叠加到**每一次**下发的瞳面相位上。

本模块的职责边界
----------------

* **只做纯计算**: 生成瞳面焦面相位、给出它的 Zernike 分解、算去转向斜坡、
  落盘可审计的元数据。**不开设备** (现场标定是唯一会开设备的路径, 且必须
  由调用方显式放行)。
* **只调用既有入口**: 焦面相位来自
  :func:`ao_shaping.tools.kernels.bench_kernels.fresnel_panel` (仓库里唯一的
  瞳面二次相位实现), 斜坡来自同模块的 :func:`ramp_panel`。本模块**不**
  重写任何 Zernike / 菲涅尔 / 斜坡数学。

叠加契约 (顺序与符号)
----------------------

补偿相位与原始瞳面相位都是 **raw 未包裹弧度** (唯一一次 mod 2π 在驱动的
弧度→灰度换算里)。叠加的**顺序**固定为::

    total = ((focus_phase + tilt_phase) + panel_phase)

即 **焦面 → 去转向 → 原始瞳面相位**。加法的结合顺序在浮点下会差最后一个
ULP, 所以顺序是被写死并被单元测试锁住的, 不是随意的。

**符号**同样被写死:

* 焦面相位是**会聚**的: :func:`fresnel_panel` 把它代进 ``pi·r²/(λ·f)``,
  所以相位在照明圆盘上是**正**的 (中心为 0, 边缘最大), 盘外为 0。
  ``focus.best_focal_mm`` 必须为**正** —— 负焦距 (发散支) 与 0 焦距都不是
  合法补偿, 本模块**显式报错**而不是静默用零相位 (见
  :func:`load_focus_calibration`)。
* 去转向斜坡的符号由**实测的面板→相机响应**决定 (见
  :func:`build_tilt_phase`): 斜坡取负即相位梯度反向, 光斑往反方向走。

默认**关闭**
------------

补偿**默认关闭**, 且关闭时 :meth:`FocusPreflight.apply_to_panel_phase`
返回**同一个对象** (不是「加了 0 的新数组」), 所以调用方在关闭态下的
输出与接入前**逐位一致**。

术语
----

本模块的中文叙述遵守 ``docs/terminology_glossary.md``: 光学口径一律写「瞳面」,
实验设备一律写「实验系统」, A/B 对照语境一律写「变体」。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ao_shaping.tools.kernels.bench_kernels import fresnel_panel, ramp_panel

__all__ = [
    "FocusPreflight",
    "FocusPreflightError",
    "FocusCalibration",
    "TiltAxisResponse",
    "TiltRampSpec",
    "build_focus_phase",
    "build_focus_preflight",
    "build_tilt_phase",
    "load_focus_calibration",
    "load_tilt_axis_map",
    "plan_live_focus_probe",
    "resolve_focus_calibration",
    "write_preflight_record",
    "zernike_decomposition",
]


#: 采集前置阶段写的证据文件名 (落在**本次运行**的目录里)。
PREFLIGHT_FILENAME = "focus_preflight.json"

#: 焦面标定证据的文件名 (在 ``--focus-dir`` 指向的目录里)。
FOCUS_EVIDENCE_FILENAME = "focus.json"

#: 允许的最小斜坡周期 (panel px)。与
#: ``scripts/slm/hw_gs_defocus_probe.py`` 的 ``MIN_RAMP_PERIOD_PX`` 同值。
MIN_RAMP_PERIOD_PX = 2

#: 采集时用的瞳面中心与标定时的照明中心相差超过这么多 (panel px) 就提醒:
#: 焦面标定是在**那个**照明位置上做的, 光束挪走了标定就过期了。
PUPIL_CENTRE_TOLERANCE_PX = 20.0


class FocusPreflightError(RuntimeError):
    """焦面标定证据不可用。

    采集前置阶段**必须显式失败**: 静默用零补偿等于把「未矫正焦面」的实验
    系统当成「已矫正」的, 而那正是本模块要消灭的错误。错误信息必须说明
    该做什么 (重新标定 / 换一个 ``--focus-dir``), 不能只说「读不出来」。
    """


# ---------------------------------------------------------------------------
# 标定结果
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FocusCalibration:
    """一次焦面标定的结果 (从 ``focus.json`` 读出)。

    Attributes:
        focal_mm: 瞳面菲涅尔焦距 (mm), **正的会聚焦距**, 直接代进
            :func:`fresnel_panel`。
        diopter_m1: 屈光度 1/f (m⁻¹), 用于报告对照。
        gain_vs_flat: 相对平场的峰值/曝光增益 (实测), ``None`` 表示没测到。
        radius_px: 标定时用的照明半径 (panel px)。
        center_x: 标定时用的照明中心 x (panel px)。
        center_y: 标定时用的照明中心 y (panel px)。
        wavelength_nm: 标定时的波长 (nm)。
        slm_number: 标定用的面板编号。
        cam_type: 标定用的相机类型。
        cam_id: 标定用的相机编号。
        source_path: 证据文件的绝对路径。
        source_mtime_utc: 证据文件的最后修改时间 (UTC, ISO 8601)。
    """

    focal_mm: float
    diopter_m1: float | None = None
    gain_vs_flat: float | None = None
    radius_px: int = 0
    center_x: int = 0
    center_y: int = 0
    wavelength_nm: float = 1064.0
    slm_number: int = 1
    cam_type: str = ""
    cam_id: int = 0
    source_path: str = ""
    source_mtime_utc: str = ""

    def describe(self) -> dict[str, Any]:
        """JSON-able 的标定摘要 (进 ``focus_preflight.json``)。"""
        return {
            "focal_mm": float(self.focal_mm),
            "diopter_m1": (
                None if self.diopter_m1 is None else float(self.diopter_m1)
            ),
            "gain_vs_flat": (
                None if self.gain_vs_flat is None else float(self.gain_vs_flat)
            ),
            "radius_px": int(self.radius_px),
            "center_x": int(self.center_x),
            "center_y": int(self.center_y),
            "wavelength_nm": float(self.wavelength_nm),
            "slm_number": int(self.slm_number),
            "cam_type": str(self.cam_type),
            "cam_id": int(self.cam_id),
            "source_path": str(self.source_path),
            "source_mtime_utc": str(self.source_mtime_utc),
        }


@dataclass(frozen=True)
class TiltAxisResponse:
    """面板某条轴的斜坡 → 相机位移的**实测**响应。

    与 ``scripts/slm/hw_gs_defocus_probe.py`` 的 ``RampAxisResponse`` 同形,
    但**不共享那个模块** (``src/`` 不许 import ``scripts/``)。两边的数值
    口径一致, 且本类的用法被一个对拍测试锁住 (见
    ``tests/ao_shaping/tools/slm/test_focus_preflight.py``)。
    """

    panel_axis: int
    camera_axis: int
    shift_px: float
    period_px: int

    @property
    def slope(self) -> float:
        """相机 px × panel px —— 位移 ∝ 1/period 的斜率。"""
        return float(self.shift_px) * float(self.period_px)

    def describe(self) -> dict[str, Any]:
        return {
            "panel_axis": int(self.panel_axis),
            "camera_axis": int(self.camera_axis),
            "shift_px": float(self.shift_px),
            "period_px": int(self.period_px),
            "slope_px_per_inv_period": float(self.slope),
        }


@dataclass(frozen=True)
class TiltRampSpec:
    """一条被叠加进瞳面相位的斜坡的规格。

    ``sign`` 是相位梯度的方向: ``+1`` 是 :func:`ramp_panel` 本身, ``-1`` 是
    它的负 (相位梯度反向 ⇒ 光斑往反方向走)。
    """

    panel_axis: int
    period_px: int
    sign: float = 1.0

    def describe(self) -> dict[str, Any]:
        return {
            "panel_axis": int(self.panel_axis),
            "period_px": int(self.period_px),
            "sign": float(self.sign),
        }


# ---------------------------------------------------------------------------
# 证据读取
# ---------------------------------------------------------------------------
def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_float(
    value: Any, label: str, *, allow_none: bool = False, default: float | None = None
) -> float | None:
    if value is None:
        if allow_none:
            return None
        if default is not None:
            return float(default)
        raise FocusPreflightError(f"焦面标定证据缺字段 {label!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FocusPreflightError(
            f"焦面标定证据的 {label!r} 不是数值: {value!r}"
        ) from exc
    if not math.isfinite(out):
        raise FocusPreflightError(
            f"焦面标定证据的 {label!r} 不是有限值: {value!r}"
        )
    return out


def _as_int(value: Any, label: str, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise FocusPreflightError(f"焦面标定证据缺字段 {label!r}")
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FocusPreflightError(
            f"焦面标定证据的 {label!r} 不是整数: {value!r}"
        ) from exc


def load_focus_calibration(focus_dir: str | Path) -> FocusCalibration:
    """从 ``--focus-dir`` 读一次焦面标定。

    支持两种证据布局 (仓库里两种都真实存在):

    * **嵌套式** (推荐, 由
      :mod:`ao_shaping.tools.slm.panel.alignment_probe` 写): 顶层 ``focus``
      里有 ``best_focal_mm`` / ``best_diopter_m1`` / ``gain_vs_flat``, 顶层
      ``settings`` 里有 ``center_x`` / ``center_y`` / ``radius`` /
      ``wavelength`` / ``slm_number`` / ``cam_type`` / ``cam_id``。
    * **扁平式** (由 :func:`ao_shaping.tools.slm.panel.focus_probe.probe_session_focus`
      写): ``best_focal_mm`` 等直接在顶层, 几何在 ``radius_px`` /
      ``centre_panel_px``。

    Args:
        focus_dir: 含 ``focus.json`` 的目录。

    Returns:
        读出的标定。

    Raises:
        FocusPreflightError: 目录/文件不存在、JSON 不可读、缺关键字段, 或
            焦度不合理 (``best_focal_mm`` 缺失 / 非有限 / 为 0)。
    """
    directory = Path(focus_dir)
    path = directory / FOCUS_EVIDENCE_FILENAME
    if not path.is_file():
        raise FocusPreflightError(
            f"焦面标定证据不存在: {path}。请先用 "
            "`python -m ao_shaping.tools.slm.panel.alignment_probe` "
            "(或 focus_probe) 标定焦面, 再把 --focus-dir 指向那个目录。"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FocusPreflightError(
            f"焦面标定证据不可读: {path} ({exc})。请重新标定焦面。"
        ) from exc
    if not isinstance(payload, dict):
        raise FocusPreflightError(
            f"焦面标定证据格式不对: {path} 的顶层不是对象。"
        )

    focus = payload.get("focus")
    if not isinstance(focus, dict):
        focus = payload
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        settings = payload

    focal = _as_float(
        _first_present(focus, "best_focal_mm", "focal_mm"), "focus.best_focal_mm"
    )
    assert focal is not None
    if focal <= 0.0:
        raise FocusPreflightError(
            f"焦面标定证据的 focus.best_focal_mm = {focal!r} 不是正的会聚焦距: "
            "本阶段只接受**会聚**矫正 (focal_mm > 0)。焦距为 0 不是透镜, 负焦距"
            "意味着标定的最佳点落在了发散支上 —— 两者都会把未矫正焦面的实验"
            "系统搞得更糟, 所以这里显式报错而不是静默用零补偿。请重新标定焦面, "
            "并确认 --focus-dir 指向的是**最佳点**那一轮的证据, 而不是扫描表里"
            "的某一个测点。"
        )

    centre = settings.get("centre_panel_px")
    centre_x = settings.get("center_x")
    centre_y = settings.get("center_y")
    if isinstance(centre, (list, tuple)) and len(centre) == 2:
        centre_x = centre[0]
        centre_y = centre[1]

    stat = path.stat()
    return FocusCalibration(
        focal_mm=float(focal),
        diopter_m1=_as_float(
            _first_present(focus, "best_diopter_m1"), "focus.best_diopter_m1",
            allow_none=True,
        ),
        gain_vs_flat=_as_float(
            _first_present(focus, "gain_vs_flat"), "focus.gain_vs_flat",
            allow_none=True,
        ),
        radius_px=_as_int(
            _first_present(settings, "radius", "radius_px"), "settings.radius",
            default=0,
        ),
        center_x=_as_int(centre_x, "settings.center_x", default=0),
        center_y=_as_int(centre_y, "settings.center_y", default=0),
        wavelength_nm=float(
            _as_float(
                _first_present(settings, "wavelength", "wavelength_nm"),
                "settings.wavelength", default=1064.0,
            )
        ),
        slm_number=_as_int(settings.get("slm_number"), "settings.slm_number", default=1),
        cam_type=str(settings.get("cam_type") or ""),
        cam_id=_as_int(settings.get("cam_id"), "settings.cam_id", default=0),
        source_path=str(path.resolve()),
        source_mtime_utc=datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
    )


def plan_live_focus_probe(**kwargs: Any) -> dict[str, Any]:
    """描述一次**现场**焦面标定会做什么 (不开设备)。

    离线 / 假设备模式下只能打印这份计划: 现场标定要开面板和相机, 而采集
    前置阶段在离线复算与单元测试里必须完全无设备。

    Returns:
        JSON-able 的计划描述 (参数 + 一句人话说明)。
    """
    return {
        "action": "probe_session_focus",
        "module": "ao_shaping.tools.slm.panel.focus_probe",
        "params": {k: v for k, v in kwargs.items()},
        "note": (
            "现场焦面标定会打开面板与相机并扫描 1/f; "
            "离线模式下只输出这份计划, 不开设备。"
        ),
    }


def resolve_focus_calibration(
    focus_dir: str | Path | None,
    *,
    allow_live_probe: bool = False,
    live_probe_kwargs: dict[str, Any] | None = None,
) -> FocusCalibration | None:
    """拿到一次焦面标定: 优先读证据, 没有才 (可选) 现场标定。

    Args:
        focus_dir: 含 ``focus.json`` 的目录; ``None`` 表示没有可用证据。
        allow_live_probe: 允许现场标定。**默认 False** —— 现场标定会开
            设备, 必须是调用方显式放行的。
        live_probe_kwargs: 传给
            :func:`ao_shaping.tools.slm.panel.focus_probe.probe_session_focus`
            的参数。

    Returns:
        标定结果; 没有证据且不允许现场标定时返回 ``None``。

    Raises:
        FocusPreflightError: 给了 ``focus_dir`` 但证据不可用 (不静默降级),
            或现场标定失败。
    """
    if focus_dir is not None and str(focus_dir).strip():
        return load_focus_calibration(focus_dir)
    if not allow_live_probe:
        return None
    from ao_shaping.tools.slm.panel.focus_probe import probe_session_focus

    evidence = probe_session_focus(**(live_probe_kwargs or {}))
    if "error" in evidence:
        raise FocusPreflightError(
            f"现场焦面标定失败: {evidence['error']}"
        )
    focal = evidence.get("best_focal_mm")
    if focal is None:
        raise FocusPreflightError(
            "现场焦面标定没有找到焦面 (best_focal_mm 缺失): 请检查光路与曝光, "
            "或改用 --focus-dir 指向一份已有的标定证据。"
        )
    centre = evidence.get("centre_panel_px") or [0, 0]
    return FocusCalibration(
        focal_mm=float(focal),
        diopter_m1=_as_float(
            evidence.get("best_diopter_m1"), "best_diopter_m1", allow_none=True
        ),
        gain_vs_flat=_as_float(
            evidence.get("gain_vs_flat"), "gain_vs_flat", allow_none=True
        ),
        radius_px=_as_int(evidence.get("radius_px"), "radius_px", default=0),
        center_x=_as_int(centre[0], "centre_panel_px[0]", default=0),
        center_y=_as_int(centre[1], "centre_panel_px[1]", default=0),
        wavelength_nm=float(evidence.get("wavelength_nm", 1064.0)),
        slm_number=_as_int(evidence.get("slm_number"), "slm_number", default=1),
        cam_type=str(evidence.get("cam_type") or ""),
        cam_id=_as_int(evidence.get("cam_id"), "cam_id", default=0),
        source_path="<live probe_session_focus>",
        source_mtime_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# 焦面相位
# ---------------------------------------------------------------------------
def build_focus_phase(
    calibration: FocusCalibration,
    *,
    panel_shape: tuple[int, int],
    pupil_center: tuple[int, int] | None = None,
    radius_px: int | None = None,
    wavelength_nm: float | None = None,
) -> np.ndarray:
    """生成瞳面焦面相位 (raw 未包裹弧度, 盘外为 0)。

    直接调用 :func:`ao_shaping.tools.kernels.bench_kernels.fresnel_panel`
    —— 仓库里唯一的瞳面二次相位实现, 形式 ``pi·r²/(λ·f)``。焦距是**正的
    会聚**焦距 (读入时就已校验), 所以相位在照明圆盘上是正的、中心为 0、
    边缘最大。

    Args:
        calibration: 读出的标定。
        panel_shape: 面板形状 ``(height, width)``。
        pupil_center: 二次相位的中心 (panel px)。默认取标定里的照明中心。
        radius_px: 照明半径 (panel px)。默认取标定里的半径。
        wavelength_nm: 波长 (nm)。默认取标定里的波长。

    Returns:
        ``panel_shape`` 的 float64 相位 (弧度, 未包裹)。
    """
    centre = (
        (int(calibration.center_x), int(calibration.center_y))
        if pupil_center is None
        else (int(pupil_center[0]), int(pupil_center[1]))
    )
    radius = int(radius_px if radius_px is not None else calibration.radius_px)
    if radius <= 0:
        raise FocusPreflightError(
            "焦面标定证据里没有照明半径 (settings.radius), 无法生成焦面相位: "
            "请重新标定焦面, 或在调用方显式给出半径。"
        )
    wavelength = float(
        wavelength_nm if wavelength_nm is not None else calibration.wavelength_nm
    )
    return fresnel_panel(
        float(calibration.focal_mm),
        radius=radius,
        pupil_center=centre,
        panel_shape=(int(panel_shape[0]), int(panel_shape[1])),
        wavelength_nm=wavelength,
    )


def zernike_decomposition(
    phase_rad: np.ndarray,
    *,
    radius_px: int,
    pupil_center: tuple[int, int],
    n_max: int = 6,
) -> dict[str, Any]:
    """把一片瞳面相位分解成 Zernike 系数 (unit-RMS Noll 归一)。

    分解在**照明圆盘**上做 (圆盘外的相位是 0, 参与拟合会把系数拉偏)。
    Noll 4 (离焦) 是焦面矫正最该被报告的那一项: 二次相位 ``pi·r²/(λf)``
    在圆盘上恰好等于「活塞 + Noll 4」, 所以 Noll 4 系数就是这片相位里
    真正的离焦量。

    Args:
        phase_rad: 面板分辨率的相位 (弧度)。
        radius_px: 照明半径 (panel px)。
        pupil_center: 照明中心 (panel px)。
        n_max: 拟合的最高径向阶。

    Returns:
        ``{"noll_1_piston_rad", "noll_4_defocus_rad", "modes": {...}}``。
    """
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    radius = int(radius_px)
    if radius <= 0:
        raise FocusPreflightError("Zernike 分解需要正的照明半径")
    height, width = phase_rad.shape
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, width)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, height)
    patch = np.asarray(phase_rad[y0:y1, x0:x1], dtype=np.float64)
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        raise FocusPreflightError(
            f"Zernike 分解的照明圆盘被面板裁掉了 (取到 {patch.shape}, 期望 "
            f"{(2 * radius + 1, 2 * radius + 1)}); 请减小半径或换一个中心。"
        )
    gen = ZernikeGenerator((patch.shape[1], patch.shape[0]), radius=radius, n_orders=n_max)
    coeffs = np.asarray(gen.fit(patch), dtype=np.float64).ravel()
    modes: dict[str, float] = {}
    for index, value in enumerate(coeffs):
        noll = index + 1
        try:
            nm = gen.noll_to_nm(noll)
        except Exception:  # 超出基底的 Noll 索引没有名字
            continue
        modes[f"noll{noll}_n{nm[0]}m{nm[1]}"] = float(value)
    return {
        "noll_1_piston_rad": float(coeffs[0]) if coeffs.size else 0.0,
        "noll_4_defocus_rad": float(coeffs[3]) if coeffs.size > 3 else 0.0,
        "radius_px": radius,
        "pupil_center": [cx, cy],
        "n_max": int(n_max),
        "modes": modes,
    }


# ---------------------------------------------------------------------------
# 去转向斜坡
# ---------------------------------------------------------------------------
def load_tilt_axis_map(path: str | Path) -> tuple[TiltAxisResponse, ...]:
    """读一份「面板斜坡 → 相机位移」的实测映射。

    接受 ``scripts/slm/hw_gs_defocus_probe.py`` 的 ``axis_map_to_dict``
    写出的那套键 (``panel_axis_0`` / ``panel_axis_1``), 也接受一个更简单的
    列表形式::

        {"axis_map": [{"panel_axis": 0, "camera_axis": 1,
                       "shift_px": 468.0, "period_px": 240}, ...]}

    Args:
        path: JSON 文件路径。

    Returns:
        实测响应 (每条面板轴一条)。

    Raises:
        FocusPreflightError: 文件不可读 / 格式不对 / 没有可用响应。
    """
    file = Path(path)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FocusPreflightError(f"去转向标定文件不可读: {file} ({exc})") from exc
    if not isinstance(payload, dict):
        raise FocusPreflightError(f"去转向标定文件的顶层不是对象: {file}")
    raw = payload.get("axis_map", payload)
    entries: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not key.startswith("panel_axis_") or not isinstance(value, dict):
                continue
            entry = dict(value)
            entry.setdefault("panel_axis", int(key.rsplit("_", 1)[1]))
            entries.append(entry)
    elif isinstance(raw, list):
        entries = [dict(item) for item in raw if isinstance(item, dict)]
    if not entries:
        raise FocusPreflightError(
            f"去转向标定文件里没有可用的实测响应: {file}"
        )
    responses: list[TiltAxisResponse] = []
    for entry in entries:
        responses.append(
            TiltAxisResponse(
                panel_axis=_as_int(entry.get("panel_axis"), "panel_axis"),
                camera_axis=_as_int(entry.get("camera_axis"), "camera_axis"),
                shift_px=float(_as_float(entry.get("shift_px"), "shift_px")),
                period_px=_as_int(entry.get("period_px"), "period_px"),
            )
        )
    return tuple(responses)


def build_tilt_phase(
    axis_map: tuple[TiltAxisResponse, ...],
    desired_shift_px: tuple[float, float],
    panel_shape: tuple[int, int],
    *,
    min_period_px: int = MIN_RAMP_PERIOD_PX,
) -> tuple[np.ndarray, tuple[TiltRampSpec, ...]]:
    """按**期望的相机位移**反解出去转向斜坡。

    与 ``scripts/slm/hw_gs_defocus_probe.py`` 的 ``compensating_ramp`` 同一套
    公式 (周期由实测斜率反解 ``period = |斜率| / |期望位移|``, 符号取「让
    实测响应与期望同向」的那一侧), 但**不 import 那个脚本** —— ``src/``
    不许依赖 ``scripts/``。两边的等价性由一个对拍测试锁住。

    Args:
        axis_map: 实测的面板→相机响应。
        desired_shift_px: 期望的**相机**位移 (x, y)。
        panel_shape: 面板形状 ``(height, width)``。
        min_period_px: 允许的最小周期 (panel px)。

    Returns:
        ``(叠加相位, 斜坡规格列表)``。无可用响应时返回全零相位与空列表。
    """
    phase: np.ndarray | None = None
    specs: list[TiltRampSpec] = []
    for camera_axis in (0, 1):
        desired = float(desired_shift_px[camera_axis])
        if not math.isfinite(desired) or abs(desired) < 1e-9:
            continue
        response = next(
            (r for r in axis_map if int(r.camera_axis) == camera_axis), None
        )
        if response is None:
            continue
        slope = response.slope
        if not math.isfinite(slope) or abs(slope) <= 0.0:
            continue
        period = max(int(round(abs(slope) / abs(desired))), int(min_period_px))
        sign = 1.0 if (slope * desired) > 0.0 else -1.0
        spec = TiltRampSpec(int(response.panel_axis), period, sign)
        specs.append(spec)
        ramp = ramp_panel(period, int(response.panel_axis), panel_shape) * sign
        phase = ramp if phase is None else phase + ramp
    if phase is None:
        phase = np.zeros(panel_shape, dtype=np.float64)
    return phase, tuple(specs)


# ---------------------------------------------------------------------------
# 采集前置阶段
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FocusPreflight:
    """一次采集的焦面补偿状态。

    Attributes:
        requested: 调用方**是否要求**补偿 (与 ``enabled`` 分开: 「要求了但
            没生效」和「没要求」是两件事, 事后看记录必须能分清)。
        enabled: 补偿是否**真的**生效 (关着的时候
            :meth:`apply_to_panel_phase` 返回原对象)。
        calibration: 读出的焦面标定 (``None`` = 没有标定)。
        focus_phase_rad: 瞳面焦面相位 (raw 弧度), ``None`` = 没有。
        tilt_phase_rad: 去转向斜坡 (raw 弧度), ``None`` = 没有。
        tilt_specs: 被叠加的斜坡规格。
        tilt_source: 斜坡的来源说明 (文件路径 / "未标定")。
        disable_reason: 关闭原因 (人话), 落盘用。
        warnings: 不致命但要记下来的提醒。
        live_probe_plan: 现场标定的计划 (离线模式下只打印, 不开设备)。
    """

    requested: bool = False
    enabled: bool = False
    calibration: FocusCalibration | None = None
    focus_phase_rad: np.ndarray | None = field(default=None, repr=False)
    tilt_phase_rad: np.ndarray | None = field(default=None, repr=False)
    tilt_specs: tuple[TiltRampSpec, ...] = ()
    tilt_source: str = ""
    disable_reason: str = ""
    warnings: tuple[str, ...] = ()
    live_probe_plan: dict[str, Any] | None = None

    # -- 叠加契约 --------------------------------------------------------
    def apply_to_panel_phase(self, panel_phase_rad: np.ndarray) -> np.ndarray:
        """把补偿相位叠加到一片瞳面相位上。

        契约 (``total = ((focus + tilt) + panel)``):

        * 三者都是 **raw 未包裹弧度**, 唯一一次 mod 2π 在驱动的弧度→灰度
          换算里。
        * 顺序固定为 **焦面 → 去转向 → 原始瞳面相位**。
        * 补偿关闭时返回**同一个对象** (调用方在关闭态下的输出与接入前
          逐位一致)。

        Args:
            panel_phase_rad: 原始瞳面相位 (面板分辨率, 弧度)。

        Returns:
            总相位 (面板分辨率, 弧度)。
        """
        if not self.enabled:
            return panel_phase_rad
        total: np.ndarray | None = None
        if self.focus_phase_rad is not None:
            total = np.asarray(self.focus_phase_rad, dtype=np.float64).copy()
        if self.tilt_phase_rad is not None:
            tilt = np.asarray(self.tilt_phase_rad, dtype=np.float64)
            total = tilt.copy() if total is None else total + tilt
        if total is None:
            return panel_phase_rad
        return total + np.asarray(panel_phase_rad, dtype=np.float64)

    # -- 元数据 ----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """可落盘的元数据 (进 ``focus_preflight.json``)。

        无论补偿开没开都能调用: 关着的时候它明确写 ``enabled: false`` 与
        关闭原因, 免得事后分不清是「没开」还是「没记」。
        """
        focus_stats = None
        if self.focus_phase_rad is not None:
            arr = np.asarray(self.focus_phase_rad, dtype=np.float64)
            nonzero = arr[arr != 0.0]
            focus_stats = {
                "shape": [int(arr.shape[0]), int(arr.shape[1])],
                "max_abs_rad": float(np.max(np.abs(arr))) if arr.size else 0.0,
                "min_rad": float(np.min(arr)) if arr.size else 0.0,
                "max_rad": float(np.max(arr)) if arr.size else 0.0,
                "nonzero_px": int(nonzero.size),
                "total_px": int(arr.size),
            }
        tilt_stats = None
        if self.tilt_phase_rad is not None:
            arr = np.asarray(self.tilt_phase_rad, dtype=np.float64)
            tilt_stats = {
                "max_abs_rad": float(np.max(np.abs(arr))) if arr.size else 0.0,
                "min_rad": float(np.min(arr)) if arr.size else 0.0,
                "max_rad": float(np.max(arr)) if arr.size else 0.0,
            }
        return {
            "requested": bool(self.requested),
            "enabled": bool(self.enabled),
            "disable_reason": str(self.disable_reason),
            "overlay_order": ["focus", "tilt", "panel"],
            "overlay_contract": (
                "total = ((focus_phase_rad + tilt_phase_rad) + panel_phase_rad), "
                "raw 未包裹弧度; 唯一一次 mod 2pi 在驱动的弧度->灰度换算里"
            ),
            "calibration": (
                None if self.calibration is None else self.calibration.describe()
            ),
            "focus_phase_stats": focus_stats,
            "tilt_phase_stats": tilt_stats,
            "tilt_specs": [spec.describe() for spec in self.tilt_specs],
            "tilt_source": str(self.tilt_source),
            "warnings": list(self.warnings),
            "live_probe_plan": self.live_probe_plan,
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def build_focus_preflight(
    *,
    focus_dir: str | Path | None = None,
    enabled: bool = True,
    allow_live_probe: bool = False,
    live_probe_kwargs: dict[str, Any] | None = None,
    panel_shape: tuple[int, int] = (1200, 1920),
    pupil_center: tuple[int, int] | None = None,
    radius_px: int | None = None,
    wavelength_nm: float | None = None,
    run_pupil_center: tuple[int, int] | None = None,
    tilt_axis_map_path: str | Path | None = None,
    desired_shift_px: tuple[float, float] | None = None,
) -> FocusPreflight:
    """组装一次采集的焦面前置阶段。

    Args:
        focus_dir: 含 ``focus.json`` 的目录。
        enabled: 补偿是否生效。``False`` ⇒ 返回一个**关闭**的前置阶段
            (不读证据、不生成相位), 但仍会带上关闭原因。
        allow_live_probe: 没有证据时允许现场标定 (会开设备)。
        live_probe_kwargs: 现场标定的参数。
        panel_shape: 面板形状 ``(height, width)``。
        pupil_center: 焦面相位的中心 (panel px), 默认取标定里的照明中心。
        radius_px: 焦面相位的照明半径 (panel px), 默认取标定里的半径。
        wavelength_nm: 焦面相位的波长 (nm), 默认取标定里的波长。
        run_pupil_center: 本次采集用的瞳面中心 (panel px)。只用来跟标定里的
            照明中心对账 (差太多就提醒标定可能过期), 不影响相位本身。
        tilt_axis_map_path: 去转向实测映射的 JSON 文件。
        desired_shift_px: 要抵消的**相机**位移 (x, y)。

    Returns:
        前置阶段对象。

    Raises:
        FocusPreflightError: 给了 ``focus_dir`` 但证据不可用, 或去转向
            标定不可用。**不静默降级成零补偿**。
    """
    if not enabled:
        reason = "未开启焦面补偿 (--focus-compensation 未给出)"
        if focus_dir is not None and str(focus_dir).strip():
            reason += "; 已给出 --focus-dir 但补偿被显式关闭"
        return FocusPreflight(
            requested=False, enabled=False, disable_reason=reason
        )

    calibration = resolve_focus_calibration(
        focus_dir,
        allow_live_probe=allow_live_probe,
        live_probe_kwargs=live_probe_kwargs,
    )
    if calibration is None:
        plan = plan_live_focus_probe(**(live_probe_kwargs or {}))
        return FocusPreflight(
            requested=True,
            enabled=False,
            disable_reason=(
                "要求了焦面补偿, 但没有焦面标定证据 (--focus-dir 未给出) 且"
                "未放行现场标定; 采集将在未矫正焦面的实验系统上进行"
            ),
            live_probe_plan=plan,
        )

    warnings: list[str] = []
    focus_phase = build_focus_phase(
        calibration,
        panel_shape=panel_shape,
        pupil_center=pupil_center,
        radius_px=radius_px,
        wavelength_nm=wavelength_nm,
    )
    if run_pupil_center is not None:
        offset = math.hypot(
            float(run_pupil_center[0]) - float(calibration.center_x),
            float(run_pupil_center[1]) - float(calibration.center_y),
        )
        if offset > PUPIL_CENTRE_TOLERANCE_PX:
            warnings.append(
                f"本次采集的瞳面中心 {tuple(int(v) for v in run_pupil_center)} 与"
                f"焦面标定的照明中心 ({calibration.center_x}, "
                f"{calibration.center_y}) 相差 {offset:.1f} panel px "
                f"(> {PUPIL_CENTRE_TOLERANCE_PX:.0f}): 标定可能已过期, "
                "焦面相位也会带来额外的转向。"
            )

    tilt_phase: np.ndarray | None = None
    tilt_specs: tuple[TiltRampSpec, ...] = ()
    tilt_source = ""
    if tilt_axis_map_path is not None and desired_shift_px is not None:
        axis_map = load_tilt_axis_map(tilt_axis_map_path)
        tilt_phase, tilt_specs = build_tilt_phase(
            axis_map, desired_shift_px, panel_shape
        )
        tilt_source = str(Path(tilt_axis_map_path).resolve())
    elif tilt_axis_map_path is not None or desired_shift_px is not None:
        raise FocusPreflightError(
            "去转向补偿需要同时给出实测映射 (tilt_axis_map_path) 与要抵消的"
            "相机位移 (desired_shift_px); 只给一个无法反解斜坡。"
        )

    return FocusPreflight(
        requested=True,
        enabled=True,
        calibration=calibration,
        focus_phase_rad=focus_phase,
        tilt_phase_rad=tilt_phase,
        tilt_specs=tilt_specs,
        tilt_source=tilt_source,
        warnings=tuple(warnings),
    )


def write_preflight_record(
    out_dir: str | Path,
    preflight: FocusPreflight,
    *,
    filename: str = PREFLIGHT_FILENAME,
) -> Path:
    """把 :meth:`FocusPreflight.describe` 落盘到本次运行的目录。

    补偿**没开**时也写一份 (内容写明 ``enabled: false`` 与关闭原因), 免得
    事后分不清是「没开」还是「没记」。

    Args:
        out_dir: 本次运行的目录。
        preflight: 前置阶段对象。
        filename: 证据文件名。

    Returns:
        写出的文件路径。
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        json.dumps(preflight.describe(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
