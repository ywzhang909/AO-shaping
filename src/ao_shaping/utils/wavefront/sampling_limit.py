"""目标边长 → 采样相机像素的尺度律，以及由它导出的**物理可达性**判据。

为什么单独一个模块
------------------
*实验系统里的三条尺度律互不相同*，混用会让「目标方边长填多少像素」这件事
前后矛盾。本模块只负责其中**第三条**，并把它写成可在无设备、无 Streamlit 环境
下逐条断言的纯函数：

=========================================  ==========================================
尺度律                                     描述的对象
=========================================  ==========================================
``D·f/(d·p_cam)`` (``slm_gs_refine``)      照明瞳的**几何像**有多大
``W_ccd = f_c·M_t·W_far/L`` (``link_design``)  望远镜远场光斑 → 诊断相机（issue #204）
``S_CCD = S·f_sample/f_main`` (本模块)      主路焦面上的**目标方斑**被采样透镜再次成像
=========================================  ==========================================

三者不可互换：GS 那条量的是瞳像（与目标图案无关），link_design 那条量的是
目标处（1 km）的角尺度，而这里量的是「主路焦面 40 mm 的方斑，以 50/1000 的
角倍率落到采样相机上就是 2 mm」。所以本模块**不**复用 ``link_design`` 的
``params.scale``，也不复用 ``OpticalLimits.slm_fov_m``（那条是瞳面采样极限）。

公式
----
主路焦面 → 采样相机（二次成像::

    S_CCD = S · f_sample / f_main

SLM 侧与采样侧（用于判断「这条设置物理上做不做得到」）::

    D_SLM   = 2·N·w                    照明直径（N = 扩束倍率，w = 扩束前束腰半径）
    N_eff   = D_SLM / p                被照亮的 SLM 像素数（有效像素）
    Δ_min   = λ·f_main / D_SLM         主路焦面的最小可分辨尺度
    M_samp  = λ·f_sample / (D_SLM·p)   目标在采样相机上的采样倍数（需 ≥ 2，Nyquist）

``N`` 与 ``w`` 只以 ``D_SLM`` 的形式进入判据 —— **绝不进入 ``S_CCD``**：方斑在
主路焦面上的物理位置与多远、SLM 被照多宽都无关，只由主路焦距与采样焦距的比
决定。把 ``N``/``w`` 塞进 ``S_CCD`` 会把「束径」和「像面尺度」两件事搅在一起。

Nyquist 的等价形式对操作员更有用：给定 ``N``/``w``/``p``，采样焦距的下限是

    f_sample ≥ 4·N·w·p / λ          （由 M_samp ≥ 2 解出）

可加性
------
本模块**只报事实**（哪些量越界、差多少），不改写输入、不夹取、不抛异常取代表。
是否阻断运行由调用方决定 —— GUI 里「早于运行按钮的校验」与「协议脚本的预检」
对同一个越界的处置不同，把策略写进本模块会让两者只能有一个。

单位
----
对外一律 SI（米 / 弧度），与 ``link_design`` 一致；GUI 负责 mm / µm 的换算。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_SAMPLING",
    "SamplingGeometry",
    "SamplingLimits",
    "SamplingSetup",
    "evaluate_sampling",
    "nyquist_focal_length_m",
    "sampled_side_m",
]

#: 采样倍数下限（Nyquist：目标盒内每个分辨单元至少要有 2 个相机像元）。
NYQUIST_MIN_SAMPLING = 2.0

#: 本仓库 2f 实验系统的默认取值。**来源逐项注明**，因为这些默认值会直接决定
#: 操作员看到的告警，不能凭空取整：
#:
#: * ``f_main_m`` = 0.125 —— 2f 主路透镜；与 ``target_size.DEFAULT_SYSTEM.focal_m``
#:   同值（那条是瞳面采样极限用的同一个 f）。
#: * ``f_sample_m`` = 0.050 —— 采样（二次成像）透镜，台面标称值。
#: * ``expansion`` = 4.0 / ``waist_m`` = 1.0 mm —— 扩束倍率与扩束前束腰；
#:   ``D_SLM = 2·4·1 mm = 8 mm`` 与 ``BEAM_RADIUS_PANEL``(=450 px × 8 µm ≈ 3.6 mm
#:   半径，即 7.2 mm 直径) 同量级，故这是**台面上真实成立**的一组值。
#: * ``panel_size_m`` —— 取 ``SLM_PANEL_W``/``SLM_PANEL_H`` × ``SLM_PITCH_M``
#:   的成品面板尺寸（15.36 × 9.6 mm），不是照明区。
DEFAULT_SAMPLING = {
    "f_main_m": 0.125,
    "f_sample_m": 0.050,
    "expansion": 4.0,
    "waist_m": 1.0e-3,
}


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (value > 0.0) or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} 必须是正的有限值: {value!r}")
    return value


@dataclass(frozen=True)
class SamplingSetup:
    """一次采样成像的输入：主路焦面 → 采样相机。

    所有长度是米。``expansion`` 与 ``waist_m`` **不参与** :attr:`side_on_camera_m`
    —— 它们只在 :func:`evaluate_sampling` 的物理可达性判据里以 ``2·expansion·waist``
    的形式出现。这个分工是本模块的口径，见模块 docstring。

    Attributes:
        main_side_m: 主路焦面上的目标方斑边长 ``S`` (m)。
        f_main_m: 主路透镜焦距 ``f_main`` (m)。
        f_sample_m: 采样透镜焦距 ``f_sample`` (m)。
        expansion: 扩束倍率 ``N``（无量纲，> 0）。
        waist_m: 扩束**前**的束腰半径 ``w`` (m)。
        pixel_pitch_m: SLM 像素间距 ``p`` (m)。
        wavelength_m: 激光波长 ``λ`` (m)。
        camera_pixel_m: 采样相机像元间距 (m)。
        panel_size_m: SLM 面板物理尺寸 ``(宽, 高)`` (m)，用于截断判据。
    """

    main_side_m: float
    f_main_m: float
    f_sample_m: float
    expansion: float
    waist_m: float
    pixel_pitch_m: float
    wavelength_m: float
    camera_pixel_m: float
    panel_size_m: tuple[float, float]

    def __post_init__(self) -> None:
        for name in (
            "main_side_m",
            "f_main_m",
            "f_sample_m",
            "expansion",
            "waist_m",
            "pixel_pitch_m",
            "wavelength_m",
            "camera_pixel_m",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        width, height = (float(v) for v in self.panel_size_m)
        object.__setattr__(
            self, "panel_size_m", (_positive("panel_size_m[0]", width), _positive("panel_size_m[1]", height))
        )

    # -- 唯一进入 S_CCD 的量 ------------------------------------------------
    @property
    def sampling_magnification(self) -> float:
        """采样相机相对主路焦面的**角倍率** ``f_sample / f_main``。"""
        return self.f_sample_m / self.f_main_m

    @property
    def side_on_camera_m(self) -> float:
        """``S_CCD = S · f_sample / f_main`` —— 方斑在采样相机上的边长 (m)。"""
        return sampled_side_m(self.main_side_m, self.f_main_m, self.f_sample_m)

    @property
    def side_on_camera_px(self) -> float:
        """同一长度换算成采样相机像元数（可能不是整数，这是正常的）。"""
        return self.side_on_camera_m / self.camera_pixel_m

    # -- 由 SLM 侧导出的量（S_CCD 不消费） ---------------------------------
    @property
    def beam_diameter_m(self) -> float:
        """照明直径 ``D_SLM = 2·N·w`` (m)。"""
        return 2.0 * self.expansion * self.waist_m

    @property
    def effective_pixels(self) -> float:
        """有效像素数 ``N_eff = D_SLM / p``（可能不是整数：照明区不一定整像素）。"""
        return self.beam_diameter_m / self.pixel_pitch_m

    @property
    def main_resolution_m(self) -> float:
        """主路焦面最小可分辨尺度 ``Δ_min = λ·f_main / D_SLM`` (m)。

        等价于：孔径 ``D_SLM`` 的艾里斑第一暗环半径量级，或 DFT 中相邻谱线在
        焦面的间隔（``λf/W``，见 ``target_size.OpticalLimits.far_field_sample_m``
        的推导，那里用面板整宽，这里用**被照亮的**宽度 —— 照明区外的相位无光，
        不贡献分辨率）。
        """
        return self.wavelength_m * self.f_main_m / self.beam_diameter_m

    @property
    def sampling_ratio(self) -> float:
        """采样倍数 ``M_samp = λ·f_sample / (D_SLM·p)``。

        物理含义：主路焦面一个分辨单元 ``Δ_min = λ·f_main/D_SLM`` 被采样透镜按
        ``f_sample/f_main`` 放大后，落到几个**相机**像元上::

            M_samp = Δ_min · (f_sample / f_main) / p_cam

        注意两处容易混的口径：``p`` 在这里是 **SLM 像素间距**（决定 ``Δ_min``
        的那个采样尺度），``p_cam`` 才是相机像元。所以 ``M_samp`` **不是**
        ``Δ_min/p_cam`` —— 少了 ``f_sample/f_main`` 这个角倍率，「采样相机把
        焦面放大了多少」就被漏掉了（同一 formula 里 ``p`` 与 ``p_cam`` 恰好
        等值时两者才会碰巧相同）。
        """
        return self.wavelength_m * self.f_sample_m / (self.beam_diameter_m * self.pixel_pitch_m)

    @property
    def nyquist_focal_length_m(self) -> float:
        """满足 ``M_samp ≥ 2`` 的采样焦距下限 ``4·N·w·p / λ`` (m)。"""
        return nyquist_focal_length_m(
            beam_diameter_m=self.beam_diameter_m,
            pixel_pitch_m=self.pixel_pitch_m,
            wavelength_m=self.wavelength_m,
        )

    @property
    def panel_fov_m(self) -> float:
        """面板物理尺寸决定的**主路焦面**可达边长上限 ``min(宽, 高)`` (m)。

        方斑在焦面上比面板还大时，面板边缘的相位无论如何都覆盖不到它 —— 这与
        衍射极限无关，是纯几何截断。
        """
        return min(self.panel_size_m)

    @property
    def target_fov_m(self) -> float:
        """主路焦面在无混叠下能表达的边长 ``λ·f_main/p`` (m)。

        与 ``target_size.OpticalLimits.slm_fov_m`` 同一个式子：面板像素间距 p 的
        采样在焦面上的 Nyquist 视场。方斑超过它，必然出现锯齿/混叠而不是均匀
        方斑。
        """
        return self.wavelength_m * self.f_main_m / self.pixel_pitch_m


@dataclass(frozen=True)
class SamplingLimits:
    """一次评估的派生量（全部 SI 或纯数）。

    Attributes:
        side_on_camera_m: ``S_CCD``。
        side_on_camera_px: ``S_CCD`` 折成采样相机像元。
        sampling_magnification: ``f_sample/f_main``。
        beam_diameter_m: ``D_SLM``。
        effective_pixels: ``N_eff``。
        main_resolution_m: ``Δ_min``。
        sampling_ratio: ``M_samp``。
        nyquist_focal_length_m: ``M_samp ≥ 2`` 所需的 ``f_sample`` 下限。
        panel_fov_m: 面板物理尺寸给出的焦面边长上限。
        target_fov_m: 无混叠视场 ``λ·f_main/p``。
        panel_truncated: ``D_SLM`` 是否超出面板物理尺寸（照明被挡）。
        under_sampled: ``M_samp < 2``（采样相机看不出主路分辨单元）。
        beyond_target_fov: ``S > λ·f_main/p``（方斑超出无混叠视场）。
        beyond_panel: ``S`` 是否超出面板物理尺寸。
    """

    side_on_camera_m: float
    side_on_camera_px: float
    sampling_magnification: float
    beam_diameter_m: float
    effective_pixels: float
    main_resolution_m: float
    sampling_ratio: float
    nyquist_focal_length_m: float
    panel_fov_m: float
    target_fov_m: float
    panel_truncated: bool
    under_sampled: bool
    beyond_target_fov: bool
    beyond_panel: bool

    @property
    def violations(self) -> tuple[str, ...]:
        """按固定顺序给出**事实**描述，空元组表示这条设置没有越界。

        顺序即严重度：先是「做不到」（照明被挡 / 面板装不下），再是「做得到但
        表达不出来」（混叠），最后是「相机看不出」（欠采样）。调用方可以只取
        前几条，但顺序本身是契约（GUI 与脚本的告警文本同序）。
        """
        out: list[str] = []
        if self.panel_truncated:
            out.append(
                f"照明直径 {self.beam_diameter_m * 1e3:.2f} mm 超出面板 "
                f"{self.panel_fov_m * 1e3:.2f} mm：相位边缘被挡，方斑边缘必然失真"
            )
        if self.beyond_panel:
            out.append(
                f"目标边长 {self.panel_fov_m * 1e3:.2f} mm 已顶到面板物理尺寸："
                "比面板更大的方斑，任何相位都合不出来"
            )
        if self.beyond_target_fov:
            out.append(
                f"目标边长超出无混叠视场 λ·f/p = {self.target_fov_m * 1e3:.2f} mm："
                "面板采样不足，方斑会出现锯齿而不是均匀边"
            )
        if self.under_sampled:
            out.append(
                f"采样倍数 M_samp = {self.sampling_ratio:.2f} < {NYQUIST_MIN_SAMPLING:g}："
                f"采样相机看不出主路 {self.main_resolution_m * 1e6:.0f} µm 的分辨单元，"
                f"需要 f_sample ≥ {self.nyquist_focal_length_m * 1e3:.1f} mm"
            )
        return tuple(out)

    @property
    def ok(self) -> bool:
        """四项判据全部通过。"""
        return not self.violations


@dataclass(frozen=True)
class SamplingGeometry:
    """求值结果：输入 + 派生量（一次算完，避免调用方重算任何一项）。"""

    setup: SamplingSetup
    limits: SamplingLimits


def sampled_side_m(main_side_m: float, f_main_m: float, f_sample_m: float) -> float:
    """``S_CCD = S·f_sample/f_main`` (m) —— 本模块唯一的尺度律实现。

    ``expansion`` 与 ``waist_m`` 有意不在签名里：它们不是这条换算的输入
    （见模块 docstring）。把它们加进来会让「换个束径方斑就变大」这种错误
    顺手写出来。
    """
    return (
        _positive("main_side_m", main_side_m)
        * _positive("f_sample_m", f_sample_m)
        / _positive("f_main_m", f_main_m)
    )


def nyquist_focal_length_m(
    *, beam_diameter_m: float, pixel_pitch_m: float, wavelength_m: float, minimum: float = NYQUIST_MIN_SAMPLING
) -> float:
    """满足 ``λ·f_sample/(D·p) ≥ minimum`` 的采样焦距下限 (m)。

    ``minimum = 2`` 即 Nyquist（每个分辨单元 2 个像元）；保守取 3–4 是操作员
    常做的事，所以阈值是参数而不是常量。
    """
    return (
        _positive("minimum", minimum)
        * _positive("beam_diameter_m", beam_diameter_m)
        * _positive("pixel_pitch_m", pixel_pitch_m)
        / _positive("wavelength_m", wavelength_m)
    )


def evaluate_sampling(setup: SamplingSetup) -> SamplingGeometry:
    """把 :class:`SamplingSetup` 一次展开成输入 + 全部派生量与判据。

    纯函数：同一个 setup 永远给同一个 geometry，不读环境、不改输入。
    """
    return SamplingGeometry(
        setup=setup,
        limits=SamplingLimits(
            side_on_camera_m=setup.side_on_camera_m,
            side_on_camera_px=setup.side_on_camera_px,
            sampling_magnification=setup.sampling_magnification,
            beam_diameter_m=setup.beam_diameter_m,
            effective_pixels=setup.effective_pixels,
            main_resolution_m=setup.main_resolution_m,
            sampling_ratio=setup.sampling_ratio,
            nyquist_focal_length_m=setup.nyquist_focal_length_m,
            panel_fov_m=setup.panel_fov_m,
            target_fov_m=setup.target_fov_m,
            panel_truncated=setup.beam_diameter_m > setup.panel_fov_m,
            under_sampled=setup.sampling_ratio < NYQUIST_MIN_SAMPLING,
            beyond_target_fov=setup.main_side_m > setup.target_fov_m,
            beyond_panel=setup.main_side_m > setup.panel_fov_m,
        ),
    )
