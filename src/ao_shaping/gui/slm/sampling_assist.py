"""采样成像助手：目标方边长 ``S`` → 采样相机像素，外加物理可达性判据。

为什么单独一个模块
------------------
与 :mod:`ao_shaping.gui.slm.link_design_assist` 同一分工：换算是
:mod:`ao_shaping.utils.wavefront.sampling_limit` 的**唯一**实现，本模块只做
两件事 ——

1. 把操作员读得懂的单位（mm / µm / nm、扩束倍率）收进 :data:`SAMPLING_FIELDS`，
   让 popover 的控件表与换算的入参**同源**；
2. :func:`compute_sampling_summary` 一次算出 popover 要显示的全部数字，
   :func:`summary_rows` 给出成对的「标签 / 数值」文本，:func:`violation_rows`
   把越界事实翻成「操作员该改哪一个旋钮」。

因此本模块不 import streamlit，也不打开设备 —— 可以在无服务器环境里逐条单测。
Streamlit 那一层只剩控件与按钮（``shaping_controls.render_link_assist``）。

与另外两条尺度律**不可互换**
-----------------------------
远场换算（``W_ccd = f_c·M_t·W_far/L``）量的是 1 km 处目标的角尺度；GS 的
``D·f/(d·p_cam)`` 量的是照明瞳的几何像。这里量的是**主路焦面上已有的方斑**
被采样透镜以 ``f_sample/f_main`` 再次成像后落在相机上的大小。三者在同一个
popover 里并列两段，各自标清口径，绝不互相代入。
"""

from __future__ import annotations

from dataclasses import dataclass

from ao_shaping.utils.wavefront.sampling_limit import (
    DEFAULT_SAMPLING,
    SamplingGeometry,
    SamplingSetup,
    evaluate_sampling,
)

__all__ = [
    "DEFAULT_SAMPLING",
    "SAMPLING_FIELDS",
    "SamplingField",
    "build_setup",
    "compute_sampling_summary",
    "format_length_mm",
    "summary_rows",
    "violation_rows",
]

#: 长度/倍率字段在操作员那一侧的单位换算。``_SI`` 由
#: :func:`build_setup` 逐项套用，因此 popover 的控件与 ``SamplingSetup`` 之间
#: 只有一处单位知识。
_MM = 1.0e-3
_UM = 1.0e-6
_NM = 1.0e-9

#: 目标方边长（主路焦面）在 popover 里的初值 (mm)。**不是**光路参数：同一台面
#: 每次实验的目标都不一样，所以它只作初值，不进「恢复锁定参数」。
DEFAULT_TARGET_SIDE_MM = 10.0


@dataclass(frozen=True)
class SamplingField:
    """popover「采样光路」里的一行可编辑字段。

    ``name`` 是 :data:`DEFAULT_SAMPLING` 或 :func:`build_setup` 的入参名，
    同时是 widget key 的后缀 —— 二者同源，改一处不会漏另一处。
    """

    name: str
    label: str
    lo: float
    hi: float
    step: float
    unit_scale: float
    #: :data:`DEFAULT_SAMPLING` 里的键名。空串表示这一格不是光路参数（目标边长
    #: ``S`` 每次实验都不同）。**显式声明**而不是靠去 ``_mm`` 后缀猜：靠猜的话
    #: 名字差一个字符就会静默回落到初值，而控件看上去一切正常。
    si_name: str = ""
    help: str = ""

    @property
    def default(self) -> float:
        """默认显示值（本字段单位下的数）。

        光路参数直接取 :data:`DEFAULT_SAMPLING` 并换算到本字段单位 —— 默认值
        只有一份，改物理常数不会漏掉 popover。非光路参数（``main_side``）回落
        :data:`DEFAULT_TARGET_SIDE_MM`：与 ``link_design_assist.DEFAULT_FAR_SIDE``
        同理，只是初值。
        """
        if not self.is_optical:
            return DEFAULT_TARGET_SIDE_MM
        return float(DEFAULT_SAMPLING[self.si_name]) / self.unit_scale

    @property
    def is_optical(self) -> bool:
        """是否属于「光路参数」（默认值来自 :data:`DEFAULT_SAMPLING`）。

        ``main_side_mm`` 不是光路参数 —— 同一台面每次实验的目标边长都不同，
        所以它回落 :data:`DEFAULT_TARGET_SIDE_MM` 并且不进「恢复锁定参数」。
        """
        return bool(self.si_name)


#: 「采样光路」的字段表，按渲染顺序。默认值来源见
#: :data:`~ao_shaping.utils.wavefront.sampling_limit.DEFAULT_SAMPLING` 的注释。
SAMPLING_FIELDS: tuple[SamplingField, ...] = (
    SamplingField(
        "main_side_mm",
        "方斑边长 S (mm)",
        0.01,
        500.0,
        1.0,
        _MM,
        si_name="",  # 非光路参数：每次实验的目标都不一样
        help="主路焦面上的目标方边长，也就是「目标方边长」换算前的物理量。",
    ),
    SamplingField(
        "f_main_mm",
        "主路焦距 f_main (mm)",
        1.0,
        5000.0,
        10.0,
        _MM,
        si_name="f_main_m",
        help="2f 主路透镜焦距。",
    ),
    SamplingField(
        "f_sample_mm",
        "采样焦距 f_sample (mm)",
        1.0,
        2000.0,
        5.0,
        _MM,
        si_name="f_sample_m",
        help="二次成像到采样相机的透镜焦距；S_CCD = S·f_sample/f_main。",
    ),
    SamplingField(
        "expansion",
        "扩束倍率 N",
        1.0,
        40.0,
        0.5,
        1.0,
        si_name="expansion",
        help="只影响照明直径与判据，不进入 S_CCD。",
    ),
    SamplingField(
        "waist_mm",
        "扩束前束腰半径 w (mm)",
        0.01,
        50.0,
        0.1,
        _MM,
        si_name="waist_m",
        help="只影响照明直径与判据，不进入 S_CCD。",
    ),
)


def _setup_from(fields: dict[str, float]) -> SamplingSetup:
    """把 popover 字段（mm / 倍率）补全成 :class:`SamplingSetup`（SI）。

    面板尺寸 / 波长 / 像元间距这些**不是操作员输入**的量走各自的单一事实源
    （``bench_constants`` 与相机参数），在这里补齐，避免 popover 再抄一份。
    """
    from ao_shaping.utils.slm.bench_constants import (
        LASER_WAVELENGTH_M,
        SLM_PANEL_H,
        SLM_PANEL_W,
        SLM_PITCH_M,
    )

    values = {name: float(value) for name, value in fields.items()}
    return SamplingSetup(
        main_side_m=values["main_side_mm"] * _MM,
        f_main_m=values["f_main_mm"] * _MM,
        f_sample_m=values["f_sample_mm"] * _MM,
        expansion=values["expansion"],
        waist_m=values["waist_mm"] * _MM,
        pixel_pitch_m=SLM_PITCH_M,
        wavelength_m=LASER_WAVELENGTH_M,
        camera_pixel_m=values["camera_pixel_m"],
        panel_size_m=(SLM_PANEL_W * SLM_PITCH_M, SLM_PANEL_H * SLM_PITCH_M),
    )


def build_setup(**fields: float) -> SamplingSetup:
    """用「采样光路」的字段值构造 :class:`SamplingSetup`。

    缺省键回落到 :data:`DEFAULT_SAMPLING`，未知键直接报错（与
    ``link_design_assist.build_params`` 同一约定：改名后静默沿用默认值，会让
    控件显示与换算结果不一致）。``camera_pixel_m`` 没有默认项，必须由调用方
    从侧栏的相机像元设置传进来 —— 那是相机侧的唯一事实源。
    """
    known = {field.name for field in SAMPLING_FIELDS} | {"camera_pixel_m"}
    unknown = sorted(set(fields) - known)
    if unknown:
        raise TypeError(f"未知的采样光路字段: {unknown}")
    if "camera_pixel_m" not in fields:
        raise TypeError("build_setup 需要 camera_pixel_m（相机像元的单一事实源）")
    values = {field.name: field.default for field in SAMPLING_FIELDS}
    values.update({name: float(v) for name, v in fields.items()})
    return _setup_from(values)


def compute_sampling_summary(
    geometry: SamplingGeometry, *, window_side_px: int | None = None
) -> dict[str, float | bool]:
    """一次换算，外加 popover 要显示的全部数字。

    全部数值来自 :func:`~ao_shaping.utils.wavefront.sampling_limit.evaluate_sampling`
    已算好的派生量；本函数不重算任何尺度律，只整理成「标签 → 数值」需要的形状。
    ``window_side_px`` 给定时额外给出「换算结果是否装得进读出窗」，那是本页
    目标方边长控件的上界。
    """
    limits = geometry.limits
    summary: dict[str, float | bool] = {
        "main_side_m": geometry.setup.main_side_m,
        "main_side_mm": geometry.setup.main_side_m * 1.0e3,
        "side_on_camera_m": limits.side_on_camera_m,
        "side_on_camera_mm": limits.side_on_camera_m * 1.0e3,
        "side_on_camera_um": limits.side_on_camera_m * 1.0e6,
        "side_on_camera_px": limits.side_on_camera_px,
        "sampling_magnification": limits.sampling_magnification,
        "beam_diameter_mm": limits.beam_diameter_m * 1.0e3,
        "effective_pixels": limits.effective_pixels,
        "main_resolution_um": limits.main_resolution_m * 1.0e6,
        "sampling_ratio": limits.sampling_ratio,
        "nyquist_focal_length_mm": limits.nyquist_focal_length_m * 1.0e3,
        "panel_fov_mm": limits.panel_fov_m * 1.0e3,
        "target_fov_mm": limits.target_fov_m * 1.0e3,
        "ok": limits.ok,
        "violations": list(limits.violations),
    }
    if window_side_px is not None:
        side_px = limits.side_on_camera_px
        summary["window_side_px"] = int(window_side_px)
        summary["fits_window"] = side_px <= float(window_side_px)
        summary["overflow_px"] = max(0.0, side_px - float(window_side_px))
    return summary


def format_length_mm(value_m: float) -> str:
    """焦面长度：≥1 mm 用 mm，更小用 µm（操作员按 10 mm 量级思考）。"""
    m = float(value_m)
    if abs(m) >= 1.0e-3:
        return f"{m * 1.0e3:.2f} mm"
    return f"{m * 1.0e6:.0f} µm"


def summary_rows(
    summary: dict[str, float | bool],
) -> tuple[tuple[str, str], ...]:
    """popover 结果块的「标签 / 数值」文本，顺序固定。"""
    rows: list[tuple[str, str]] = [
        (
            "采样相机上的方斑 S_CCD",
            f"{summary['side_on_camera_mm']:.3f} mm = "
            f"{float(summary['side_on_camera_px']):.0f} px",
        ),
        (
            "角倍率 f_sample/f_main",
            f"{float(summary['sampling_magnification']):.4f}"
            "（只由这两个焦距决定，与束径无关）",
        ),
        (
            "照明直径 D_SLM",
            f"{float(summary['beam_diameter_mm']):.2f} mm"
            f"（有效像素 {float(summary['effective_pixels']):.0f} px）",
        ),
        (
            "主路分辨单元 Δ_min",
            f"{float(summary['main_resolution_um']):.0f} µm"
            "（λ·f_main/D_SLM）",
        ),
        (
            "采样倍数 M_samp",
            f"{float(summary['sampling_ratio']):.2f}"
            f"（需 ≥ 2；要 f_sample ≥ "
            f"{float(summary['nyquist_focal_length_mm']):.1f} mm）",
        ),
        (
            "面板物理边长 / 无混叠视场",
            f"{float(summary['panel_fov_mm']):.2f} mm / "
            f"{float(summary['target_fov_mm']):.1f} mm",
        ),
    ]
    if "fits_window" in summary:
        side_px = float(summary["side_on_camera_px"])
        window = int(summary["window_side_px"])
        rows.append(
            (
                "读出窗",
                f"{window} px —— " + (
                    f"装得下（占 {side_px / window:.0%}）"
                    if summary["fits_window"]
                    else f"装不下，超出 {float(summary['overflow_px']):.0f} px"
                ),
            )
        )
    return tuple(rows)


def violation_rows(summary: dict[str, float | bool]) -> tuple[tuple[str, str], ...]:
    """每条越界事实 + **建议改哪个旋钮**。

    事实文本来自 ``SamplingLimits.violations``（单一实现）；这里只补「那该怎么办」，
    因为同一个越界在不同页面上的处置不同，建议话术属于呈现层。
    """
    hints = {
        "M_samp": (
            "欠采样（相机看不出主路分辨单元）",
            f"把采样焦距加大到 ≥ {float(summary['nyquist_focal_length_mm']):.1f} mm，"
            "或缩束腰 / 降扩束倍率以减小 D_SLM",
        ),
        "无混叠视场": (
            "目标超出无混叠视场（方斑会锯齿）",
            "减小目标方边长 S，或换更小的面板像素间距 —— 这条是面板采样极限",
        ),
        "面板物理尺寸": (
            "目标顶到面板物理尺寸",
            "减小目标方边长 S：比面板还大的方斑，任何相位都合不出来",
        ),
        "超出面板": (
            "照明直径超出面板（边缘被挡）",
            "降低扩束倍率 N 或束腰 w，使 2·N·w 落在面板短边以内",
        ),
    }
    out: list[tuple[str, str]] = []
    for text in summary["violations"]:  # type: ignore[union-attr]
        for key, (label, advice) in hints.items():
            if key in text:
                out.append((label, f"{text}。建议：{advice}"))
                break
        else:  # 新加的判据没有建议话术时，事实本身也必须显示出来
            out.append(("越界", str(text)))
    return tuple(out)
