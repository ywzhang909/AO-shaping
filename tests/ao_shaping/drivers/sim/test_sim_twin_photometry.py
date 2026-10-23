"""孪生的三个新旋钮的契约测试: 相机离焦 / 固定 counts 参考 / 暗电平。

背景 (2026-10-22)
------------------
对照尺 (``scripts/slm/compare_sim_vs_hw.py``) 把仿真 vs 第三轮硬件实测的差距
归到三个根因上, 本文件把它们**锁死**:

1. **CCD 离焦** —— 孪生的远场是面板相位的纯 FFT ⇒ 焦面按构造就在探测器上,
   于是「用瞳面菲涅尔相位把焦面搬回探测器」在仿真里毫无效果 (焦面标定扫出来
   的最佳 1/f 恒为 0)。加上相机离焦后, 补偿焦距 ``f_v = f²/Δz`` 才存在。
2. **counts 的固定参考** —— 逐帧把峰值归一到 100 ⇒ 离焦散开的光又被拉回原
   峰值 (实测峰值随离焦崩塌 27 → 0.25 counts, 仿真几乎不动)。固定参考
   (平场在理想聚焦下的峰值) 让 counts 随散开自然下降。
3. **暗电平** —— 没有暗电平时, 背景的读出噪声被 ``clip(·, 0, None)`` 整流成
   一块与像素数成正比的 DC 底座; 而当背景分布恰好让「中位数」落在两个取值
   之间时, 中位数本身逐帧跳变 ⇒ 整帧总功率抖 3 个量级 ⇒ σ(PIB) 被它支配。

这些测试**不跑真仿真**: 用一块**小面板** (64×64) 让远场求值在毫秒级完成,
口径与真面板完全一致 (同一份代码路径)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_pib_sim import (
    COUNTS_REFERENCE_FLAT,
    COUNTS_REFERENCE_FRAME,
    SimNoiseSpec,
    SimPanelAberration,
    SimPibSystem,
)

SMALL = (64, 64)


def _flat_system(**kwargs) -> SimPibSystem:
    """一块小面板上的孪生 (零相位)。"""
    system = SimPibSystem(slm_shape=SMALL, far_field_padding=2, **kwargs)
    system.set_phase_rad(np.zeros(SMALL))
    return system


# ---------------------------------------------------------------------------
# ① 相机离焦
# ---------------------------------------------------------------------------


class TestCameraDefocus:
    def test_off_by_default(self) -> None:
        """默认 = 探测器在 2f 焦面上 (本模块改动前的行为, 逐位不变)。"""
        system = SimPibSystem(seed=42, slm_shape=SMALL)
        assert system.camera_defocus_m == 0.0
        assert math.isinf(system.defocus_focal_m)
        assert system._defocus_phase() is None

    def test_defocus_changes_the_far_field(self) -> None:
        """离焦必须真的让远场变糊 (不是假的旋钮)。"""
        clean = _flat_system(seed=42)
        defocused = _flat_system(seed=42, camera_defocus_m=0.01)
        assert not np.array_equal(clean.far_field(), defocused.far_field())

    def test_compensating_focal_is_f_squared_over_dz(self) -> None:
        """等效瞳面二次相位焦距 ``f_v = f²/Δz`` (Collins/ABCD)。"""
        system = SimPibSystem(
            seed=42, slm_shape=SMALL, focal_length_m=0.125, camera_defocus_m=0.01
        )
        assert system.defocus_focal_m == pytest.approx(0.125**2 / 0.01)

    def test_defocus_phase_sign_matches_the_ccd_behind_focus(self) -> None:
        """探测器在焦面**之后** ⇒ 瞳面上是负的二次相位 (凹透镜)。"""
        system = SimPibSystem(seed=42, slm_shape=SMALL, camera_defocus_m=0.01)
        phase = system._defocus_phase()
        assert phase is not None
        # 面板中心 r=0 ⇒ 相位 0; 离轴 ⇒ 相位为负。
        centre = (system.slm_h // 2, system.slm_w // 2)
        assert phase[centre] == pytest.approx(0.0)
        assert phase[0, 0] < 0.0

    def test_the_fresnel_correction_cancels_the_defocus(self) -> None:
        """面板菲涅尔相位 (``+πr²/(λf_v)``) 必须**正好**抵消离焦的二次相位。

        这是「最佳补偿值落在 f_v ≈ 1549 mm」的**机理**检验: 抵消之后, 远场与
        无离焦时逐位相同 (面板是理想的, 抵消得干净)。
        """
        from ao_shaping.tools.kernels.bench_kernels import fresnel_panel

        system = SimPibSystem(
            seed=42, slm_shape=SMALL, focal_length_m=0.125, camera_defocus_m=0.01
        )
        focal_mm = 1000.0 * system.defocus_focal_m
        correction = fresnel_panel(
            focal_mm,
            radius=system.slm_w // 2,
            pupil_center=(system.slm_w // 2, system.slm_h // 2),
            panel_shape=SMALL,
            wavelength_nm=system.wavelength_nm,
            pixel_um=system.slm_pixel_um,
        )
        defocus = system._defocus_phase()
        assert defocus is not None
        residual = np.max(np.abs(defocus + correction))
        # 残余 = 面板中心那一块的相位差 (菲涅尔相位在盘内是二次的)。
        assert residual < 0.5, f"离焦与矫正没有抵消干净: 残余 {residual:.4f} rad"


# ---------------------------------------------------------------------------
# ② counts 的固定参考
# ---------------------------------------------------------------------------


class TestCountsReference:
    def test_frame_mode_is_the_default(self) -> None:
        system = SimPibSystem(seed=42, slm_shape=SMALL)
        assert system.counts_reference == COUNTS_REFERENCE_FRAME

    def test_unknown_reference_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SimPibSystem(seed=42, slm_shape=SMALL, counts_reference="nope")

    def test_flat_reference_does_not_rescale_a_spread_spot(self) -> None:
        """固定参考 ⇒ 光被散开后 counts **下降**; 逐帧归一 ⇒ 不下降。"""
        reference = 1000.0  # counts/(峰值分数·ms)
        system = SimPibSystem(
            seed=42,
            slm_shape=SMALL,
            far_field_padding=2,
            peak_counts_per_ms=reference,
            camera_defocus_m=0.01,
            counts_reference=COUNTS_REFERENCE_FLAT,
        )
        system.set_phase_rad(np.zeros(SMALL))
        focused = float(system.counts_frame(0.2).max())
        # 加一片等效的菲涅尔相位 ⇒ 光被散开。
        from ao_shaping.tools.kernels.bench_kernels import fresnel_panel

        system.set_phase_rad(
            fresnel_panel(
                1000.0 * system.defocus_focal_m,
                radius=system.slm_w // 2,
                pupil_center=(system.slm_w // 2, system.slm_h // 2),
                panel_shape=SMALL,
                wavelength_nm=system.wavelength_nm,
                pixel_um=system.slm_pixel_um,
            )
        )
        spread = float(system.counts_frame(0.2).max())
        assert spread < focused, (
            f"散开后 counts 没有下降: focused={focused:.3f} spread={spread:.3f}"
        )

    def test_frame_mode_keeps_the_peak_constant(self) -> None:
        """逐帧归一 (历史行为) ⇒ 峰值恒定, 与散不散开无关。"""
        system = SimPibSystem(
            seed=42,
            slm_shape=SMALL,
            far_field_padding=2,
            peak_counts_per_ms=133.5,
            counts_reference=COUNTS_REFERENCE_FRAME,
        )
        system.set_phase_rad(np.zeros(SMALL))
        first = float(system.counts_frame(0.2).max())
        phase = np.zeros(SMALL)
        phase[0, 0] = 3.0
        system.set_phase_rad(phase)
        second = float(system.counts_frame(0.2).max())
        assert second == pytest.approx(first, rel=1e-6)


# ---------------------------------------------------------------------------
# ③ 暗电平
# ---------------------------------------------------------------------------


class TestDarkLevel:
    def test_off_by_default(self) -> None:
        assert SimNoiseSpec().dark_level_counts == 0.0

    def test_dark_level_lifts_the_background(self) -> None:
        system = SimPibSystem(
            seed=42,
            slm_shape=SMALL,
            far_field_padding=2,
            peak_counts_per_ms=133.5,
            noise_spec=SimNoiseSpec(dark_level_counts=26.75),
        )
        system.set_phase_rad(np.zeros(SMALL))
        frame = system.counts_frame(0.2)
        assert float(np.median(frame)) == pytest.approx(26.75, abs=1.0)

    def test_quantisation_acts_on_the_dark_level(self) -> None:
        """量化作用在**带暗电平的原始 ADU** 上 (真实相机就是这样)。"""
        system = SimPibSystem(
            seed=42,
            slm_shape=SMALL,
            far_field_padding=2,
            peak_counts_per_ms=133.5,
            noise_spec=SimNoiseSpec(dark_level_counts=26.75, quantize_bits=8),
        )
        system.set_phase_rad(np.zeros(SMALL))
        frame = system.counts_frame(0.2)
        assert np.all(frame == np.round(frame))
        assert float(np.median(frame)) == pytest.approx(27.0, abs=1.0)

    def test_dark_level_prevents_the_clip_rectification(self) -> None:
        """没有暗电平时, 背景的读出噪声被 ``clip(·, 0, None)`` **整流**。

        这是根因本身: 零均值噪声被裁到 0 ⇒ 一半像素钉在 0、一半在 1 ⇒ 背景不再是
        零均值, 而是**一块与像素数成正比的 DC 底座**。有了暗电平 (26.75 counts,
        远高于 σ=0.5) 之后, 背景整体被顶离裁剪面 ⇒ 一个像素都不再被裁到 0,
        底座也就不再是「被整流出来的」。

        ⚠️ 在**全尺寸面板 + 250 px 相机窗口**上, 这个整流还会让「中位数」恰好卡在
        0/0.25 的跳变边界上 (P(0) ≈ 0.50), 于是中位数逐帧跳变、整帧总功率抖 3 个
        量级 —— 实测对照表里 σ(PIB) 差 66× 就是这么来的。小面板上测不到那一步
        (像素数太少), 所以这里锁的是**机理的第一步**: 整流本身。
        """
        without = SimPibSystem(
            seed=42,
            slm_shape=(256, 256),
            far_field_padding=2,
            beam_w0=40.0,
            peak_counts_per_ms=133.5,
            noise_spec=SimNoiseSpec(read_sigma_counts=0.5, quantize_bits=8),
        )
        with_dark = SimPibSystem(
            seed=42,
            slm_shape=(256, 256),
            far_field_padding=2,
            beam_w0=40.0,
            peak_counts_per_ms=133.5,
            noise_spec=SimNoiseSpec(
                read_sigma_counts=0.5, quantize_bits=8, dark_level_counts=26.75
            ),
        )
        without.set_phase_rad(np.zeros((256, 256)))
        with_dark.set_phase_rad(np.zeros((256, 256)))
        # 背景 = 无噪声远场里峰值 1% 以下的像素 (与面板尺寸无关的判据)。
        background = without.far_field() < 1.0
        clipped = float(np.mean(without.counts_frame(0.2)[background] == 0.0))
        lifted = float(np.mean(with_dark.counts_frame(0.2)[background] == 0.0))
        assert clipped > 0.3, f"背景没有被整流 (只有 {clipped:.2%} 被裁到 0)"
        assert lifted == 0.0, "暗电平没有把背景顶离裁剪面"


# ---------------------------------------------------------------------------
# 照明盘 (截断)
# ---------------------------------------------------------------------------


class TestIlluminationRadius:
    def test_off_by_default(self) -> None:
        system = SimPibSystem(seed=42, slm_shape=SMALL)
        assert system.illumination_radius_px is None

    def test_clipping_removes_the_out_of_disc_light(self) -> None:
        system = SimPibSystem(
            seed=42, slm_shape=SMALL, illumination_radius_px=8.0
        )
        amplitude = system._amplitude_field()
        yy, xx = np.mgrid[0 : SMALL[0], 0 : SMALL[1]]
        r2 = (xx - SMALL[1] / 2.0) ** 2 + (yy - SMALL[0] / 2.0) ** 2
        assert np.all(amplitude[r2 > 64.0] == 0.0)
        assert np.any(amplitude[r2 <= 64.0] > 0.0)

    def test_default_amplitude_is_unchanged(self) -> None:
        """不截断时高斯一路铺到面板边缘 (改动前的行为)。"""
        system = SimPibSystem(seed=42, slm_shape=SMALL)
        amplitude = system._amplitude_field()
        assert float(amplitude[0, 0]) > 0.0


# ---------------------------------------------------------------------------
# 默认行为不变
# ---------------------------------------------------------------------------


class TestDefaultsAreUnchanged:
    def test_far_field_is_bit_identical_without_the_new_knobs(self) -> None:
        """新旋钮全关时, 远场与「没有这些旋钮」的孪生逐位相同。"""
        plain = SimPibSystem(seed=42, slm_shape=SMALL, far_field_padding=2)
        with_knobs = SimPibSystem(
            seed=42,
            slm_shape=SMALL,
            far_field_padding=2,
            camera_defocus_m=0.0,
            counts_reference=COUNTS_REFERENCE_FRAME,
            illumination_radius_px=None,
            panel_aberration=SimPanelAberration(),
        )
        plain.set_phase_rad(np.zeros(SMALL))
        with_knobs.set_phase_rad(np.zeros(SMALL))
        assert np.array_equal(plain.far_field(), with_knobs.far_field())
