"""远场尺寸换算助手：望远镜远场光斑 ↔ 诊断相机像素（issue #204 的 GUI 步骤）。

为什么单独一个模块
------------------
换算只有一份实现：:mod:`ao_shaping.utils.wavefront.link_design`。本模块**不重新
推导**任何尺度律（``W_ccd = f_c·M_t·W_far/L``），只做两件事：

1. 把操作员读得懂的单位（cm / mm / µm / nm）与 issue #204 的锁定设计点收进
   :data:`DESIGN_POINT` / :data:`LINK_FIELDS`，让 popover 的控件表与换算的入参
   **同源** —— 默认值只写一份，控件和换算不会各自漂移；
2. :func:`compute_link_summary` 一次算出 popover 要显示的全部数字，
   :func:`summary_rows` 给出成对的「标签 / 数值」文本。

这样换算与文案都能在没有 Streamlit、没有设备的情况下单测（本模块不 import
streamlit）。Streamlit 那一层只剩控件与按钮（``shaping_controls`` 的
``ShapingMethod.render_link_assist``）。

锁定设计点（issue #204，勿改）
------------------------------
远场 10 cm @ 1 km → 相机 200 µm = 91 px（像元 2.2 µm）；零级 Airy 斑半径
28.8 µm = 13 px；分辨单元 ``N = D_exit·α/λ`` = 8.46（出瞳 90 mm）；
换算系数 ``L/(f_c·M_t) = 500``。这些数字由 ``link_design`` 的 golden 值钉住，
本模块只做单位换算与呈现。
"""

from __future__ import annotations

from dataclasses import dataclass

from ao_shaping.utils.wavefront.link_design import (
    TelescopeLinkParams,
    airy_first_zero,
    ccd_to_far,
    far_to_ccd,
)

__all__ = [
    "DEFAULT_FAR_SIDE",
    "DESIGN_POINT",
    "FAR_UNITS",
    "LINK_FIELDS",
    "LinkField",
    "build_params",
    "compute_link_summary",
    "compute_reverse",
    "far_side_to_m",
    "format_ccd_length",
    "format_far_length",
    "summary_rows",
]

#: issue #204 的锁定设计点，单位取操作员读得懂的那一档（popover 的默认值）。
#: 键名同时是 :class:`LinkField` 的 ``name``，也是 ``link_design`` 入参的来源。
DESIGN_POINT: dict[str, float] = {
    "target_distance_m": 1.0e3,  # L = 1 km
    "focus_focal_length_mm": 200.0,  # f_c
    "telescope_magnification": 10.0,  # M_t
    "wavelength_nm": 1064.0,
    "beam_diameter_mm": 9.0,  # SLM 有效窗口
    "exit_pupil_diameter_mm": 90.0,
    "ccd_pixel_um": 2.2,
}

#: popover 第一行「远场边长」的默认值与单位。
DEFAULT_FAR_SIDE: dict[str, float | str] = {"value": 10.0, "unit": "cm"}

#: 允许的远场边长单位（第一项是默认）。
FAR_UNITS: tuple[str, ...] = ("cm", "mm", "m")

_UNIT_TO_M: dict[str, float] = {"cm": 1.0e-2, "mm": 1.0e-3, "m": 1.0}


@dataclass(frozen=True)
class LinkField:
    """popover「光路参数」里的一行可编辑字段。

    ``name`` 是 :data:`DESIGN_POINT` 的键，同时是 widget key 的后缀与
    :func:`build_params` 的入参名 —— 三者同源，改一处不会漏另一处。
    """

    name: str
    label: str
    lo: float
    hi: float
    step: float

    @property
    def default(self) -> float:
        """锁定设计点上的取值（``DESIGN_POINT`` 的唯一入口）。"""
        return DESIGN_POINT[self.name]


#: 「光路参数」的字段表，按渲染顺序排列。
LINK_FIELDS: tuple[LinkField, ...] = (
    LinkField("target_distance_m", "目标距离 (m)", 1.0, 1.0e5, 100.0),
    LinkField("focus_focal_length_mm", "焦距 f_c (mm)", 1.0, 2000.0, 10.0),
    LinkField("telescope_magnification", "望远镜放大率 M_t", 1.0, 100.0, 1.0),
    LinkField("wavelength_nm", "波长 (nm)", 100.0, 2000.0, 1.0),
    LinkField("beam_diameter_mm", "入射光束直径 (mm)", 0.1, 500.0, 1.0),
    LinkField("exit_pupil_diameter_mm", "出瞳直径 (mm)", 1.0, 500.0, 1.0),
    LinkField("ccd_pixel_um", "CCD 像元间距 (µm)", 0.1, 100.0, 0.1),
)


def far_side_to_m(value: float, unit: str) -> float:
    """把「远场边长」输入换算成米（``cm`` / ``mm`` / ``m``）。"""
    try:
        factor = _UNIT_TO_M[unit]
    except KeyError:
        raise ValueError(
            f"未知的远场边长单位: {unit!r}（可选 {', '.join(FAR_UNITS)}）"
        ) from None
    return float(value) * factor


def build_params(**fields: float) -> TelescopeLinkParams:
    """用「光路参数」的字段值构造 :class:`TelescopeLinkParams`。

    缺省键回落到 :data:`DESIGN_POINT`；未知键直接报错 —— 字段改名后静默沿用
    默认值会让换算与控件显示的参数不一致，那比报错难查得多。
    """
    unknown = sorted(set(fields) - set(DESIGN_POINT))
    if unknown:
        raise TypeError(f"未知的光路字段: {unknown}")
    values = {**DESIGN_POINT, **{name: float(v) for name, v in fields.items()}}
    return TelescopeLinkParams(
        wavelength=values["wavelength_nm"] * 1.0e-9,
        target_distance=values["target_distance_m"],
        telescope_magnification=values["telescope_magnification"],
        focus_focal_length=values["focus_focal_length_mm"] * 1.0e-3,
        beam_diameter=values["beam_diameter_mm"] * 1.0e-3,
        exit_pupil_diameter=values["exit_pupil_diameter_mm"] * 1.0e-3,
        ccd_pixel=values["ccd_pixel_um"] * 1.0e-6,
    )


def compute_link_summary(
    far_side_m: float,
    *,
    window_side_px: int,
    params: TelescopeLinkParams,
) -> dict[str, float]:
    """一次「远场 → 诊断相机」换算，外加 popover 要显示的全部数字。

    所有长度都从 :mod:`link_design` 来（``far_to_ccd`` / ``ccd_to_far`` /
    ``airy_first_zero`` / ``params.scale``）；本函数不重算 ``L/(f_c·M_t)``。
    ``window_side_px`` 只用来把当前读出窗折成它能覆盖的远场边长，让操作员看到
    「相机视场有多大」。
    """
    far_side_m = float(far_side_m)
    alpha = far_side_m / params.target_distance
    w_ccd_m, w_ccd_px = far_to_ccd(far_side_m, params)
    airy_m = airy_first_zero(
        params.wavelength,
        params.effective_focal_length,
        params.exit_pupil_diameter,
    )
    window_px = int(window_side_px)
    window_m = ccd_to_far(window_px * params.ccd_pixel, params)
    return {
        "far_side_m": far_side_m,
        "far_angle_rad": alpha,
        "scale": params.scale,
        "effective_focal_length_m": params.effective_focal_length,
        "ccd_side_m": w_ccd_m,
        "ccd_side_um": w_ccd_m * 1.0e6,
        "ccd_side_px": w_ccd_px,
        "coherent_cells": params.exit_pupil_diameter * alpha / params.wavelength,
        "airy_first_zero_m": airy_m,
        "airy_first_zero_um": airy_m * 1.0e6,
        "airy_first_zero_px": airy_m / params.ccd_pixel,
        "window_side_px": window_px,
        "window_far_m": window_m,
    }


def compute_reverse(
    ccd_side_px: float, *, params: TelescopeLinkParams
) -> dict[str, float]:
    """反向：相机像素 → 远场边长（同一个 ``ccd_to_far``）。"""
    px = float(ccd_side_px)
    ccd_m = px * params.ccd_pixel
    return {
        "ccd_side_px": px,
        "ccd_side_m": ccd_m,
        "ccd_side_um": ccd_m * 1.0e6,
        "far_side_m": ccd_to_far(ccd_m, params),
    }


def format_far_length(m: float) -> str:
    """远场长度按 1 km 尺度该读的单位显示（m / cm / mm）。"""
    m = float(m)
    if m >= 1.0:
        return f"{m:.3f} m"
    if m >= 1.0e-2:
        return f"{m * 1.0e2:.2f} cm"
    return f"{m * 1.0e3:.2f} mm"


def format_ccd_length(m: float) -> str:
    """相机平面长度一律用 µm（诊断相机自己的尺度）。"""
    return f"{float(m) * 1.0e6:.1f} µm"


def summary_rows(summary: dict[str, float]) -> tuple[tuple[str, str], ...]:
    """popover 结果块的「标签 / 数值」文本，顺序固定。"""
    return (
        (
            "CCD 边长",
            f"{summary['ccd_side_um']:.1f} µm = "
            f"{round(summary['ccd_side_px'])} px",
        ),
        (
            "分辨单元 N",
            f"{summary['coherent_cells']:.2f}（出瞳直径上的相干单元数）",
        ),
        (
            "零级 Airy 斑半径",
            f"{summary['airy_first_zero_um']:.1f} µm = "
            f"{round(summary['airy_first_zero_px'])} px（到第一暗环）",
        ),
        (
            f"读出窗 {int(summary['window_side_px'])} px 覆盖的远场",
            format_far_length(summary["window_far_m"]),
        ),
    )
