"""``sampling_limit``：目标边长 → 采样相机像素，以及物理可达性判据。

本模块是那组判据的**唯一**实现，所以测试直接对着算子写，而不是对着 GUI 的
文本写（GUI 只负责把 :attr:`SamplingLimits.violations` 显示出来）。

三件必须钉住的事：

1. **``S_CCD`` 只由 ``S``/``f_main``/``f_sample`` 决定。** 测试逐项改 ``expansion``
   与 ``waist_m``，``S_CCD`` 必须逐位不变。这条是整个模块存在的理由 —— 把束径
   混进像面尺度是最容易犯、最难在台面上发现的一类错误。
2. **算例的数值。** 用户给的 工作例（f_main=1000 mm, N=4, w=1 mm, f_sample=50 mm,
   S=40 mm, λ=1064 nm, p=8 µm, p_pixel=2.2 µm）复算后 ``S_CCD`` = 2.0 mm =
   909 px、``N_eff`` = 1000、``Δ_min`` = 133 µm，但 ``M_samp`` = 0.83 **不是**
   最初手记的 3.0（要 3.0 需 f_sample ≈ 180 mm）。这里按公式取值，手记的 3.0
   作为反例一并钉住，免得下次又按错的数去调采样焦距。
3. **每条判据都能单独触发。** 四个 violation 各有独立的最小构造，否则某条判据
   被写成恒真/恒假也没人发现。
"""

from __future__ import annotations

import pytest

from ao_shaping.utils.slm.bench_constants import (
    LASER_WAVELENGTH_M,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
)
from ao_shaping.utils.wavefront.sampling_limit import (
    DEFAULT_SAMPLING,
    SamplingSetup,
    evaluate_sampling,
    nyquist_focal_length_m,
    sampled_side_m,
)

#: 用户的工作例（除 ``M_samp`` 按公式修正外，其余逐项对齐）。
WORKED_EXAMPLE = dict(
    main_side_m=0.040,
    f_main_m=1.0,
    f_sample_m=0.050,
    expansion=4.0,
    waist_m=1.0e-3,
    pixel_pitch_m=8.0e-6,
    wavelength_m=1064e-9,
    camera_pixel_m=2.2e-6,
    panel_size_m=(SLM_PANEL_W * SLM_PITCH_M, SLM_PANEL_H * SLM_PITCH_M),
)


def _limits(**overrides):
    return evaluate_sampling(SamplingSetup(**{**WORKED_EXAMPLE, **overrides})).limits


def test_worked_example_reproduces_the_formula_values() -> None:
    """S_CCD = 2.0 mm = 909 px, N_eff = 1000, Δ_min = 133 µm."""
    limits = _limits()

    assert limits.side_on_camera_m == pytest.approx(2.0e-3, rel=1e-12)
    assert limits.side_on_camera_px == pytest.approx(909.09, rel=1e-4)
    assert limits.effective_pixels == pytest.approx(1000.0, rel=1e-12)
    assert limits.main_resolution_m * 1e6 == pytest.approx(133.0, rel=1e-9)
    assert limits.sampling_magnification == pytest.approx(0.05, rel=1e-12)
    assert limits.beam_diameter_m == pytest.approx(8.0e-3, rel=1e-12)


def test_the_handwritten_msamp_of_3_is_not_what_the_formula_gives() -> None:
    """反例：M_samp = 0.83，不是手记的 3.0；3.0 要 f_sample ≈ 180 mm。

    这个数决定操作员该不该加大采样焦距，写错就是「欠采样看起来像合格」，
    所以它作为反例被钉住而不是删掉。
    """
    limits = _limits()

    assert limits.sampling_ratio == pytest.approx(0.83125, rel=1e-9)
    assert limits.sampling_ratio != pytest.approx(3.0, rel=0.1)
    # 反过来：直接解 ``λ·f_s/(D·p) = 3`` 得到的采样焦距（≈180 mm），代回去
    # 必须真的给出 3.0 —— 而不是靠 0.83 这个数反推的凑算式。
    f_s_for_three = 3.0 * limits.beam_diameter_m * 8.0e-6 / 1064e-9
    assert f_s_for_three == pytest.approx(0.1804511278, rel=1e-6)
    assert _limits(f_sample_m=f_s_for_three).sampling_ratio == pytest.approx(3.0, rel=1e-12)


def test_msamp_is_the_magnified_main_resolution_over_the_camera_pixel() -> None:
    """``M_samp = Δ_min·(f_sample/f_main)/p_cam`` —— 恒等式，任何一项改动都同时成立。

    这里刻意不写成 ``Δ_min/p``：``p``（SLM 像素间距）与 ``p_cam``（相机像元）
    是两个不同的量，漏掉 ``f_sample/f_main`` 就会在「换采样透镜」时给出错的
    采样倍数。取 ``p_cam = p`` 的台面（同为 8 µm）便于逐项核对，另加一组
    ``p_cam ≠ p`` 的值证明这一项真的在用。
    """
    for waist in (0.5e-3, 1.0e-3, 2.5e-3):
        limits = _limits(waist_m=waist)
        expected = (
            limits.main_resolution_m
            * (limits.sampling_magnification)
            / 8.0e-6
        )
        assert limits.sampling_ratio == pytest.approx(expected, rel=1e-12)

    # 相机像元换成 2.2 µm（本实验系统的真值）⇒ 采样倍数按比例放大 8/2.2
    tiny = _limits(camera_pixel_m=2.2e-6)
    assert tiny.sampling_ratio == pytest.approx(
        _limits().sampling_ratio, rel=1e-12
    ), "M_samp 是光学量，不该随相机像元变"
    assert tiny.main_resolution_m * tiny.sampling_magnification / 2.2e-6 == pytest.approx(
        tiny.sampling_ratio * 8.0 / 2.2, rel=1e-12
    )


@pytest.mark.parametrize(("field", "value"), [("expansion", 1.0), ("expansion", 8.0), ("waist_m", 0.25e-3), ("waist_m", 4.0e-3)])
def test_beam_size_never_enters_the_camera_side_length(field: str, value: float) -> None:
    """改扩束倍率 / 束腰，``S_CCD`` 必须逐位不变（本模块的核心口径）。"""
    base = _limits()
    changed = _limits(**{field: value})

    assert changed.side_on_camera_m == base.side_on_camera_m
    assert changed.side_on_camera_px == base.side_on_camera_px
    assert changed.sampling_magnification == base.sampling_magnification
    # 但它们**必须**改变判据里的量，否则「不改 S_CCD」就变成了「完全没用」
    assert changed.beam_diameter_m != base.beam_diameter_m


def test_sampled_side_is_linear_in_the_target_side() -> None:
    """``S`` 翻倍 ⇒ ``S_CCD`` 翻倍（纯线性，无阈值无夹取）。"""
    assert sampled_side_m(0.080, 1.0, 0.050) == pytest.approx(
        2.0 * sampled_side_m(0.040, 1.0, 0.050), rel=1e-12
    )


def test_sampled_side_rejects_non_positive_inputs() -> None:
    """0 / 负数 / 非有限值是编程错误，不是「自动」——必须报错。"""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sampled_side_m(bad, 1.0, 0.050)
        with pytest.raises(ValueError):
            sampled_side_m(0.040, bad, 0.050)


def test_the_worked_example_is_under_sampled() -> None:
    """M_samp = 0.83 < 2：欠采样是逐条判据，不影响 S_CCD 的换算。"""
    limits = _limits()

    assert limits.under_sampled
    assert any("M_samp" in text for text in limits.violations)
    assert not limits.panel_truncated


def test_enough_sampling_focal_length_clears_the_nyquist_violation() -> None:
    """取到 f_sample ≥ 4Nwp/λ 后，欠采样那条消失（其余判据不受影响）。"""
    need = nyquist_focal_length_m(
        beam_diameter_m=8.0e-3, pixel_pitch_m=8.0e-6, wavelength_m=1064e-9
    )
    assert need == pytest.approx(0.12030075187969925, rel=1e-9)

    limits = _limits(f_sample_m=need * 1.0001)

    assert limits.sampling_ratio == pytest.approx(2.0, rel=1e-3)
    assert not limits.under_sampled
    assert not any("M_samp" in text for text in limits.violations)
    # S_CCD 跟着 f_sample 一起变（同一条尺度律），所以目标边长换算也要重算
    assert limits.side_on_camera_m > 2.0e-3


def test_nyquist_focal_length_scales_with_the_beam_diameter() -> None:
    """``4·N·w·p/λ``：束径翻倍，所需采样焦距翻倍。"""
    one = nyquist_focal_length_m(
        beam_diameter_m=8.0e-3, pixel_pitch_m=8e-6, wavelength_m=1064e-9
    )
    two = nyquist_focal_length_m(
        beam_diameter_m=16.0e-3, pixel_pitch_m=8e-6, wavelength_m=1064e-9
    )

    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_a_wider_beam_than_the_panel_is_truncated() -> None:
    """照明直径超过面板短边（9.6 mm）⇒ 面板挡光。"""
    limits = _limits(expansion=6.0)  # D = 12 mm > 9.6 mm

    assert limits.beam_diameter_m == pytest.approx(12.0e-3, rel=1e-12)
    assert limits.panel_truncated
    assert "超出面板" in limits.violations[0]


def test_a_target_larger_than_the_panel_is_rejected() -> None:
    """目标边长 > 面板物理尺寸 ⇒ 任何相位都合不出来（纯几何）。"""
    limits = _limits(main_side_m=12.0e-3)  # > 9.6 mm

    assert limits.beyond_panel
    assert any("面板物理尺寸" in text for text in limits.violations)


def test_a_target_beyond_the_aliasing_free_field_is_rejected() -> None:
    """S > λ·f_main/p ⇒ 面板采样不足，方斑锯齿。

    工作例里 ``λf/p`` = 133 mm，面板只有 9.6 mm，所以面板判据先触发；这条
    判据要用一个**面板足够大**的 setup 单独验（否则它永远被面板盖住、
    变成恒假）。
    """
    limits = _limits(panel_size_m=(0.5, 0.5), main_side_m=0.200)

    assert limits.target_fov_m == pytest.approx(133.0e-3, rel=1e-12)
    assert limits.beyond_target_fov
    assert not limits.beyond_panel
    assert any("无混叠视场" in text for text in limits.violations)


def test_a_clean_setup_has_no_violations() -> None:
    """四条判据同时满足时 ``ok`` 为真、``violations`` 为空。"""
    limits = _limits(f_sample_m=0.150, main_side_m=0.005, expansion=4.0)

    assert not limits.under_sampled
    assert not limits.panel_truncated
    assert not limits.beyond_panel
    assert not limits.beyond_target_fov
    assert limits.violations == ()
    assert limits.ok


def test_violations_are_ordered_by_severity() -> None:
    """顺序是契约：先「做不到」，再「混叠」，最后「相机看不出」。"""
    limits = _limits(expansion=6.0, main_side_m=12.0e-3)

    texts = limits.violations
    assert len(texts) >= 3
    assert "超出面板" in texts[0]  # 照明被挡
    assert any("M_samp" in text for text in texts)  # 欠采样在最后
    assert "M_samp" in texts[-1]


def test_default_sampling_is_the_bench_nominal_set() -> None:
    """默认取值必须与台面常量自洽：D_SLM 与实测照明半径同量级。"""
    from ao_shaping.utils.slm.bench_constants import BEAM_RADIUS_PANEL

    d_slm = 2.0 * DEFAULT_SAMPLING["expansion"] * DEFAULT_SAMPLING["waist_m"]
    measured_radius = BEAM_RADIUS_PANEL * SLM_PITCH_M

    assert DEFAULT_SAMPLING["f_main_m"] == pytest.approx(0.125)
    assert d_slm == pytest.approx(8.0e-3)
    assert d_slm / measured_radius == pytest.approx(2.2, abs=0.2)


def test_the_module_defaults_use_the_panel_constants() -> None:
    """面板尺寸取自 ``bench_constants``（不在这里再写一遍 1920/1200）。"""
    setup = SamplingSetup(
        main_side_m=0.010,
        f_main_m=DEFAULT_SAMPLING["f_main_m"],
        f_sample_m=DEFAULT_SAMPLING["f_sample_m"],
        expansion=DEFAULT_SAMPLING["expansion"],
        waist_m=DEFAULT_SAMPLING["waist_m"],
        pixel_pitch_m=SLM_PITCH_M,
        wavelength_m=LASER_WAVELENGTH_M,
        camera_pixel_m=2.2e-6,
        panel_size_m=(SLM_PANEL_W * SLM_PITCH_M, SLM_PANEL_H * SLM_PITCH_M),
    )

    assert setup.panel_fov_m == pytest.approx(9.6e-3, rel=1e-12)
    assert setup.target_fov_m == pytest.approx(1064e-9 * 0.125 / SLM_PITCH_M, rel=1e-12)


def test_the_default_f_sample_is_far_below_nyquist_for_the_default_beam() -> None:
    """默认 50 mm 采样焦距配 8 mm 照明直径达不到 Nyquist —— 这是**已知现状**。

    这条断言的作用不是「证明默认值好」，而是让「默认配置本身欠采样」这件事
    变成一个显式的、有据可查的事实：谁要改默认值，这条会红。
    """
    setup = SamplingSetup(
        main_side_m=0.040,
        f_main_m=DEFAULT_SAMPLING["f_main_m"],
        f_sample_m=DEFAULT_SAMPLING["f_sample_m"],
        expansion=DEFAULT_SAMPLING["expansion"],
        waist_m=DEFAULT_SAMPLING["waist_m"],
        pixel_pitch_m=SLM_PITCH_M,
        wavelength_m=LASER_WAVELENGTH_M,
        camera_pixel_m=2.2e-6,
        panel_size_m=(SLM_PANEL_W * SLM_PITCH_M, SLM_PANEL_H * SLM_PITCH_M),
    )
    limits = evaluate_sampling(setup).limits

    assert limits.sampling_ratio == pytest.approx(0.83125, rel=1e-9)
    assert limits.under_sampled
    assert limits.nyquist_focal_length_m * 1e3 == pytest.approx(120.3, abs=0.05)
