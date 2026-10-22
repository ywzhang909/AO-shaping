"""2f 实验系统数字孪生核心 (``bench_twin``) 的物理断言。

被测对象: ``ao_shaping/drivers/sim/bench_twin.py``。本文件只断言**物理口径**,
不锁死实现细节:

* **默认 = 理想** —— 默认配置下远场与「同一约定下手算的补零 FFT 强度」逐元素一致,
  且两次调用逐位相同;
* **LUT** —— Gamma 非线性按解析律偏离, 灰度量化精确, 实测 LUT 表优先于 Gamma;
* **面板相差** —— 注入纯离焦 (Noll 4) 后远场可测量地变化且随系数单调, 系数全 0 时
  与默认逐位相同;
* **相机离焦** —— Δz≠0 时光斑二阶矩变大, Δz=0 是恒等, 无 NaN/Inf;
* **光路倾斜** —— 0 级光斑沿 x 平移 ``f·tilt`` (换算成相机像元), 容差 10%;
* **透镜像差** —— 非零频域像差改变远场, 零 → 恒等;
* **噪声** —— ``noise_adu=0`` 确定性, ``noise_adu>0`` 均值不变、标准差 > 0;
* **相机标度** —— ``camera_px_per_model_px`` 等于解析式 ``p_fft / d_ccd``;
* **焦面标度口径** —— 由实测标度反推的等效焦距与手算一致 (311.32 mm), 往返后孪生的
  解析标度与传入的实测标度一致; 默认路径的解析标度逐位不变; 非法输入抛 ValueError;
* **序列化** —— ``to_dict`` → JSON → ``from_dict`` → 远场逐位相同;
* **预补偿** —— LUT 往返与面板 Zernike 相差被消除 (残余 ≤ 一个量化级);
* **可用性** —— import 期无副作用, 且不 import 任何驱动包。
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
    equivalent_focal_length_m,
)
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "src" / "ao_shaping" / "drivers" / "sim" / "bench_twin.py"

#: 2026-10-19 那次实验系统的实测焦面标度 (`bench_geometry.json` 的相位斜坡扫描):
#: 「相机像元 / 模型采样点」。它反推出的等效焦距是 311.32 mm, 而标称焦距
#: 122.4060 mm 算出的解析标度只有 1.027778 (差 2.54 倍)。
MEASURED_SCALE_20261019 = 2.614022035287756

#: 与上面那个实测标度配套的几何量 (`run_config.json`)。
MEASURED_GEOMETRY_20261019: dict = {
    "wavelength_m": 1064e-9,
    "slm_pixel_um": 8.0,
    "ccd_pixel_um": 2.2,
    "region": 128,
    "region_span_px": 900.0,
    "far_field_padding": 8,
}


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _reference_far_field(
    config: BenchTwinConfig, phase_rad: np.ndarray
) -> np.ndarray:
    """独立手算的「补零 FFT 强度」参考值 (同一约定)。

    约定: 瞳面网格 ``region × region`` 覆盖 ``region_span_px`` 面板像元 → 采样
    节距 ``region_span_px / region`` 个像元; 高斯幅度峰值归一; 居中补零到
    ``region * far_field_padding``; ``fftshift(fft2(ifftshift(·)))``; 取模平方;
    居中裁剪到 ``region``; 峰值归一到 ``peak_adu``。

    相位链按同一约定走 canonical 灰度入口 (SLM 是 10 bit 器件, 量化永远存在)。
    """
    region = int(config.region)
    m = region * int(config.far_field_padding)
    pitch = config.region_span_px / region * config.slm_pixel_um * 1e-6
    w0 = config.beam_w0_px * config.slm_pixel_um * 1e-6

    idx = np.arange(region, dtype=np.float64) - (region - 1) / 2.0
    x = idx * pitch
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    amplitude = np.exp(-r2 / (2.0 * w0 * w0))
    amplitude = amplitude / amplitude.max()

    gray = phase_to_slm_grayscale(
        np.asarray(phase_rad, dtype=np.float64),
        max_grayscale=int(config.errors.gray_levels),
    )
    actual = (
        config.errors.lut_phase_max_rad
        * (gray.astype(np.float64) / int(config.errors.gray_levels))
        ** config.errors.lut_gamma
    )

    buffer = np.zeros((m, m), dtype=np.complex128)
    half = (m - region) // 2
    buffer[half : half + region, half : half + region] = amplitude * np.exp(
        1j * actual
    )
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(buffer), norm="ortho"))
    intensity = np.abs(focal) ** 2
    side = region if config.crop_px is None else int(config.crop_px)
    start = (m - side) // 2
    intensity = intensity[start : start + side, start : start + side]
    return intensity * (config.peak_adu / intensity.max())


def _second_moment(image: np.ndarray) -> float:
    """光斑的强度加权二阶矩 (x² + y² 的和, 单位: 像素²)。"""
    weights = image.astype(np.float64)
    total = weights.sum()
    rows = np.arange(image.shape[0], dtype=np.float64)
    cols = np.arange(image.shape[1], dtype=np.float64)
    cy = float((weights.sum(axis=1) * rows).sum() / total)
    cx = float((weights.sum(axis=0) * cols).sum() / total)
    var_y = float((weights * (rows[:, None] - cy) ** 2).sum() / total)
    var_x = float((weights * (cols[None, :] - cx) ** 2).sum() / total)
    return var_x + var_y


def _peak_sharpness(image: np.ndarray) -> float:
    """峰值锐度 = max / sum (归一化不变; 像差越大越低)。"""
    return float(image.max() / image.sum())


def _argmax_xy(image: np.ndarray) -> tuple[int, int]:
    """全局最大值的位置 (行, 列) —— 0 级光斑的锚点。"""
    return tuple(int(v) for v in np.unravel_index(int(image.argmax()), image.shape))


def _tilt_phase(region: int = 128, amplitude: float = 0.7) -> np.ndarray:
    """一条温和的测试相位 (低阶倾斜 + 离焦), 让远场不是完美的点。"""
    y, x = np.mgrid[0:region, 0:region]
    cx = cy = (region - 1) / 2.0
    return amplitude * (
        0.6 * (x - cx) / region + 0.4 * ((x - cx) ** 2 + (y - cy) ** 2) / region**2
    )


# ---------------------------------------------------------------------------
# 1. 默认 = 理想
# ---------------------------------------------------------------------------


class TestIdealDefault:
    def test_matches_hand_computed_padded_fft(self) -> None:
        config = BenchTwinConfig()
        twin = BenchTwin(config)
        phase = _tilt_phase(config.region)
        expected = _reference_far_field(config, phase)
        actual = twin.far_field(phase)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)

    def test_two_calls_are_bitwise_identical(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region)
        first = twin.far_field(phase)
        second = twin.far_field(phase)
        assert np.array_equal(first, second)

    def test_default_is_ideal_geometry(self) -> None:
        """默认配置不含任何误差项 (三族全 0)。"""
        errors = BenchTwinConfig().errors
        assert errors.lut_gamma == 1.0
        assert errors.lut_table is None
        assert not errors.slm_zernike
        assert errors.slm_roughness_sigma_rad == 0.0
        assert errors.defocus_m == 0.0
        assert errors.tilt_x_rad == 0.0 and errors.tilt_y_rad == 0.0
        assert not errors.lens_zernike

    def test_zero_order_sits_at_the_centre(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        image = twin.far_field(np.zeros((twin.region, twin.region)))
        row, col = _argmax_xy(image)
        assert abs(row - (twin.crop_side - 1) / 2.0) <= 1.0
        assert abs(col - (twin.crop_side - 1) / 2.0) <= 1.0


# ---------------------------------------------------------------------------
# 2. LUT 非线性 + 量化
# ---------------------------------------------------------------------------


class TestLut:
    def test_gamma_deviation_follows_the_analytic_law(self) -> None:
        gamma = 1.6
        twin = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(lut_gamma=gamma))
        )
        levels = twin.errors.gray_levels
        phase = np.linspace(0.0, 2.0 * math.pi, 257).reshape(-1, 1) * np.ones(
            (1, twin.region)
        )
        gray = phase_to_slm_grayscale(phase, max_grayscale=levels)
        expected = (
            twin.errors.lut_phase_max_rad * (gray.astype(np.float64) / levels) ** gamma
        )
        actual = twin.apply_lut(phase)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

        # Gamma > 1 → 半灰度以下的相位被压低, 且偏差随灰度单调变化。
        ideal = twin.errors.lut_phase_max_rad * (gray.astype(np.float64) / levels)
        deviation = ideal - actual
        mask = (gray > 0) & (gray < levels)
        assert np.all(deviation[mask] > 0.0)
        assert deviation[mask].min() > 0.0

    def test_gamma_one_round_trip_adds_no_nonlinearity(self) -> None:
        """gamma=1 且无 LUT 表: 往返只允许 mod 2π 与灰度量化误差。"""
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region) * 3.0
        actual = twin.apply_lut(phase)
        residual = np.mod(actual - np.mod(phase, 2.0 * math.pi) + math.pi, 2.0 * math.pi) - math.pi
        assert np.max(np.abs(residual)) <= 2.0 * math.pi / twin.errors.gray_levels

    def test_quantisation_is_exact(self) -> None:
        """往返后的相位与「查表值」逐位一致 (量化精确, 无隐藏插值)。"""
        twin = BenchTwin(BenchTwinConfig(errors=TwinErrorModel(lut_gamma=1.3)))
        phase = _tilt_phase(twin.region) * 2.0
        gray = phase_to_slm_grayscale(phase, max_grayscale=twin.errors.gray_levels)
        expected = (
            twin.errors.lut_phase_max_rad
            * (gray.astype(np.float64) / twin.errors.gray_levels) ** 1.3
        )
        np.testing.assert_array_equal(twin.apply_lut(phase), expected)

    def test_lut_table_takes_priority_over_gamma(self) -> None:
        levels = 1023
        table = 2.0 * math.pi * (np.arange(levels + 1) / levels) ** 0.7
        twin = BenchTwin(
            BenchTwinConfig(
                errors=TwinErrorModel(lut_gamma=0.4, lut_table=table.copy())
            )
        )
        phase = _tilt_phase(twin.region)
        gray = phase_to_slm_grayscale(phase, max_grayscale=levels)
        np.testing.assert_allclose(twin.apply_lut(phase), table[gray], rtol=0, atol=0)

    def test_gamma_changes_the_far_field(self) -> None:
        phase = _tilt_phase(128)
        ideal = BenchTwin(BenchTwinConfig()).far_field(phase)
        nonlinear = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(lut_gamma=1.5))
        ).far_field(phase)
        assert not np.allclose(ideal, nonlinear, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. SLM 面板静态相差
# ---------------------------------------------------------------------------


class TestPanelAberration:
    def test_zero_coefficient_is_bitwise_default(self) -> None:
        phase = _tilt_phase(128)
        default = BenchTwin(BenchTwinConfig()).far_field(phase)
        with_zernike = BenchTwin(
            BenchTwinConfig(
                errors=TwinErrorModel(slm_zernike={4: 0.0, 5: 0.0})
            )
        ).far_field(phase)
        assert np.array_equal(default, with_zernike)

    def test_defocus_coefficient_changes_the_far_field(self) -> None:
        phase = _tilt_phase(128)
        default = BenchTwin(BenchTwinConfig()).far_field(phase)
        aberrated = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(slm_zernike={4: 0.4}))
        ).far_field(phase)
        delta = float(np.max(np.abs(default - aberrated)))
        assert delta > 1.0  # 峰值标度 100 ADU → 变化必须可测量

    def test_defocus_coefficient_is_monotonic(self) -> None:
        phase = _tilt_phase(128)
        coefficients = [0.0, 0.05, 0.1, 0.2, 0.4]
        sharpness = []
        for value in coefficients:
            twin = BenchTwin(
                BenchTwinConfig(errors=TwinErrorModel(slm_zernike={4: value}))
            )
            sharpness.append(_peak_sharpness(twin.far_field(phase)))
        assert sharpness == sorted(sharpness, reverse=True)
        assert sharpness[0] - sharpness[-1] > 0.0

    def test_noll4_is_defocus_by_the_canonical_convention(self) -> None:
        """Noll 4 = (2, 0) 离焦 (仓库 canonical Noll 1976 约定)。"""
        from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm

        assert noll_to_nm(4) == (2, 0)

    def test_roughness_is_seeded_and_has_the_requested_rms(self) -> None:
        errors = TwinErrorModel(slm_roughness_sigma_rad=0.02, slm_roughness_seed=7)
        twin = BenchTwin(BenchTwinConfig(errors=errors))
        field = twin._roughness_phase()
        assert math.isclose(float(np.sqrt(np.mean(field**2))), 0.02, rel_tol=1e-9)
        again = BenchTwin(BenchTwinConfig(errors=errors))._roughness_phase()
        np.testing.assert_array_equal(field, again)


# ---------------------------------------------------------------------------
# 4. 相机离焦
# ---------------------------------------------------------------------------


class TestCameraDefocus:
    def test_zero_defocus_is_identity(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region)
        baseline = twin.far_field(phase)
        focused = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(defocus_m=0.0))
        ).far_field(phase)
        assert np.array_equal(baseline, focused)

    def test_defocus_widens_the_spot(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region)
        baseline = _second_moment(twin.far_field(phase))
        for dz in (5e-4, 1e-3, 2e-3):
            blurred = BenchTwin(
                BenchTwinConfig(errors=TwinErrorModel(defocus_m=dz))
            ).far_field(phase)
            assert _second_moment(blurred) > baseline
        # 单调: 离焦越大, 二阶矩越大。
        moments = [
            _second_moment(
                BenchTwin(
                    BenchTwinConfig(errors=TwinErrorModel(defocus_m=dz))
                ).far_field(phase)
            )
            for dz in (5e-4, 1e-3, 2e-3, 5e-3)
        ]
        assert moments == sorted(moments)

    def test_no_nan_or_inf(self) -> None:
        twin = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(defocus_m=2e-3))
        )
        image = twin.far_field(_tilt_phase(twin.region))
        assert np.all(np.isfinite(image))

    def test_apply_defocus_is_identity_for_zero(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        pupil = twin.pupil_amplitude() * np.exp(
            1j * _tilt_phase(twin.region)
        )
        focal = twin._propagate(pupil)
        assert np.array_equal(twin.apply_defocus(focal), focal)


# ---------------------------------------------------------------------------
# 5. 光路倾斜
# ---------------------------------------------------------------------------


class TestTilt:
    def test_zero_order_shifts_by_f_times_tilt(self) -> None:
        tilt = 5e-4
        config = BenchTwinConfig()
        twin = BenchTwin(config)
        phase = _tilt_phase(twin.region)
        reference = twin.far_field(phase)
        tilted = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(tilt_x_rad=tilt))
        ).far_field(phase)

        ref_row, ref_col = _argmax_xy(reference)
        row, col = _argmax_xy(tilted)
        shift_model_px = col - ref_col
        shift_camera_px = shift_model_px * twin.camera_px_per_model_px()
        expected_camera_px = (
            config.focal_length_m * tilt / twin.ccd_pixel_m
        )
        assert shift_camera_px == pytest.approx(expected_camera_px, rel=0.10)
        assert abs(row - ref_row) <= 1

    def test_zero_tilt_is_identity(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region)
        base = BenchTwin(BenchTwinConfig()).far_field(phase)
        assert np.array_equal(base, twin.far_field(phase))


# ---------------------------------------------------------------------------
# 6. 透镜像差
# ---------------------------------------------------------------------------


class TestLensAberration:
    def test_zero_is_identity(self) -> None:
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region)
        base = BenchTwin(BenchTwinConfig()).far_field(phase)
        with_lens = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(lens_zernike={4: 0.0}))
        ).far_field(phase)
        assert np.array_equal(base, with_lens)

    def test_non_zero_aberration_changes_the_far_field(self) -> None:
        phase = _tilt_phase(128)
        base = BenchTwin(BenchTwinConfig()).far_field(phase)
        aberrated = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(lens_zernike={4: 0.5}))
        ).far_field(phase)
        assert float(np.max(np.abs(base - aberrated))) > 1.0

    def test_aberration_only_touches_the_pupil_field(self) -> None:
        """像差是**瞳面**相位掩膜: 透镜在傅里叶变换之前。"""
        twin = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(lens_zernike={4: 0.3}))
        )
        pupil = twin.pupil_amplitude() * np.exp(1j * _tilt_phase(twin.region))
        aberrated = twin.apply_lens_aberration(pupil)
        assert aberrated.shape == pupil.shape
        # 相位掩膜不改变模长 → 强度只改变分布, 不改变总量级。
        assert float(np.abs(np.abs(aberrated) - np.abs(pupil)).max()) < 1e-9

    def test_focal_plane_phase_mask_could_not_change_the_intensity(self) -> None:
        """记录「焦面频域单位模掩膜不可观测」这一实测事实 (口径选择的依据)。

        ``|E·e^{iφ}|² ≡ |E|²`` —— 所以透镜像差若放在焦面, 远场强度逐位不变,
        该误差族将完全不可用。这正是本模块把它放在瞳面的理由。
        """
        twin = BenchTwin(BenchTwinConfig())
        pupil = twin.pupil_amplitude() * np.exp(1j * _tilt_phase(twin.region))
        focal = twin._propagate(pupil)
        mask = np.exp(1j * 0.4 * (focal.real**2 + focal.imag**2))
        plain = np.abs(focal) ** 2
        masked = np.abs(focal * mask) ** 2
        # 逐位不成立 (浮点), 但相对偏差只在舍入量级 —— 强度实质不变。
        assert float(np.max(np.abs(masked - plain))) <= 1e-9 * float(
            np.max(plain)
        )


# ---------------------------------------------------------------------------
# 7. 噪声
# ---------------------------------------------------------------------------


class TestNoise:
    def test_zero_noise_is_deterministic(self) -> None:
        twin = BenchTwin(BenchTwinConfig(errors=TwinErrorModel(), ))
        config = BenchTwinConfig(noise_adu=0.0)
        twin = BenchTwin(config)
        phase = _tilt_phase(twin.region)
        assert np.array_equal(twin.frame(phase), twin.frame(phase))

    def test_noise_preserves_the_mean_and_adds_variance(self) -> None:
        config = BenchTwinConfig(noise_adu=5.0, seed=3)
        twin = BenchTwin(config)
        phase = _tilt_phase(twin.region)
        clean = twin.far_field(phase)
        noisy = twin.frame(phase)
        assert not np.array_equal(noisy, twin.frame(phase))
        assert float(np.std(noisy)) > 0.0
        # 均值的抽样标准误: 散粒噪声 sum(I) 的方差 + 读出噪声 N·σ²。
        n_pixels = clean.size
        mean_sigma = math.sqrt(
            float(clean.sum()) / n_pixels**2
            + config.noise_adu**2 / n_pixels
        )
        assert abs(float(noisy.mean()) - float(clean.mean())) < 5.0 * mean_sigma


# ---------------------------------------------------------------------------
# 8. 相机标度
# ---------------------------------------------------------------------------


class TestCameraScale:
    def test_camera_px_per_model_px_matches_the_analytic_value(self) -> None:
        config = BenchTwinConfig()
        twin = BenchTwin(config)
        n_pad = config.region * config.far_field_padding
        pupil_pitch = (
            config.region_span_px / config.region * config.slm_pixel_um * 1e-6
        )
        p_fft = config.wavelength_m * config.focal_length_m / (n_pad * pupil_pitch)
        expected = p_fft / (config.ccd_pixel_um * 1e-6)
        assert twin.camera_px_per_model_px() == pytest.approx(expected, rel=1e-12)
        assert twin.p_fft == pytest.approx(p_fft, rel=1e-12)

    def test_model_grid_is_finer_than_the_camera_pixel(self) -> None:
        """默认配置: 模型采样点比相机像元细 (相机欠采样)。"""
        twin = BenchTwin(BenchTwinConfig())
        assert 0.1 < twin.camera_px_per_model_px() < 1.0


# ---------------------------------------------------------------------------
# 9. 序列化
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_json_round_trip_preserves_the_far_field(self) -> None:
        config = BenchTwinConfig(
            errors=TwinErrorModel(
                lut_gamma=1.2,
                slm_zernike={4: 0.15, 5: -0.05},
                slm_roughness_sigma_rad=0.003,
                defocus_m=5e-4,
                tilt_x_rad=2e-4,
                lens_zernike={4: 0.1},
            )
        )
        twin = BenchTwin(config)
        phase = _tilt_phase(twin.region)
        expected = twin.far_field(phase)

        payload = json.loads(json.dumps(twin.to_dict()))
        restored = BenchTwin.from_dict(payload)
        assert np.array_equal(restored.far_field(phase), expected)

    def test_lut_table_survives_the_json_round_trip(self) -> None:
        levels = 1023
        table = 2.0 * math.pi * (np.arange(levels + 1) / levels) ** 0.85
        config = BenchTwinConfig(
            errors=TwinErrorModel(lut_table=table.copy(), lut_gamma=0.5)
        )
        twin = BenchTwin(config)
        phase = _tilt_phase(twin.region)
        expected = twin.far_field(phase)
        payload = json.loads(json.dumps(twin.to_dict()))
        restored = BenchTwin.from_dict(payload)
        assert restored.errors.lut_table is not None
        np.testing.assert_array_equal(restored.errors.lut_table, table)
        assert np.array_equal(restored.far_field(phase), expected)

    def test_noll_keys_survive_the_json_round_trip(self) -> None:
        twin = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(slm_zernike={4: 0.2, 11: -0.1}))
        )
        restored = BenchTwin.from_dict(json.loads(json.dumps(twin.to_dict())))
        assert restored.errors.slm_zernike == {4: 0.2, 11: -0.1}


# ---------------------------------------------------------------------------
# 10. 预补偿
# ---------------------------------------------------------------------------


class TestCompensation:
    def test_lut_nonlinearity_is_compensated(self) -> None:
        gamma = 1.6
        twin = BenchTwin(
            BenchTwinConfig(errors=TwinErrorModel(lut_gamma=gamma))
        )
        desired = np.mod(_tilt_phase(twin.region) * 2.0, 2.0 * math.pi)
        command = twin.compensate(desired)
        actual = twin.apply_lut(twin.apply_panel_aberration(command))
        residual = np.mod(actual - desired + math.pi, 2.0 * math.pi) - math.pi
        # 量化不可反解 ⇒ 残余 ≤ Gamma·(一个量化级)。
        bound = gamma * 2.0 * math.pi / twin.errors.gray_levels
        assert float(np.max(np.abs(residual))) <= bound

    def test_panel_zernike_is_cancelled(self) -> None:
        errors = TwinErrorModel(slm_zernike={4: 0.4, 5: 0.1})
        twin = BenchTwin(BenchTwinConfig(errors=errors))
        desired = np.mod(_tilt_phase(twin.region) * 2.0, 2.0 * math.pi)

        uncompensated = twin.apply_lut(twin.apply_panel_aberration(desired))
        raw_residual = np.mod(uncompensated - desired + math.pi, 2.0 * math.pi) - math.pi

        command = twin.compensate(desired)
        compensated = twin.apply_lut(twin.apply_panel_aberration(command))
        residual = np.mod(compensated - desired + math.pi, 2.0 * math.pi) - math.pi

        assert float(np.max(np.abs(residual))) < 0.2 * float(
            np.max(np.abs(raw_residual))
        )
        assert float(np.max(np.abs(residual))) <= 2.0 * math.pi / twin.errors.gray_levels

    def test_compensation_improves_the_far_field(self) -> None:
        errors = TwinErrorModel(slm_zernike={4: 0.3})
        twin = BenchTwin(BenchTwinConfig(errors=errors))
        desired = np.mod(_tilt_phase(twin.region) * 2.0, 2.0 * math.pi)
        raw = twin.far_field(desired)
        corrected = twin.far_field(twin.compensate(desired))
        assert _peak_sharpness(corrected) > _peak_sharpness(raw)


# ---------------------------------------------------------------------------
# 11. 可用性
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_module_does_not_import_any_driver_package(self) -> None:
        """静态检查: bench_twin 不 import 任何 ``ao_shaping.drivers.*`` 模块。"""
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8-sig"))
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("ao_shaping.drivers")
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("ao_shaping.drivers"):
                    offenders.append(module)
        assert not offenders, f"bench_twin must not import driver packages: {offenders}"

    def test_import_has_no_side_effects(self) -> None:
        """子进程里**按文件路径**加载该模块: 成功, 且不加载任何驱动包。

        按路径加载是为了绕开 ``ao_shaping.drivers.sim`` 包的 ``__init__``
        (它本身会 eager import 若干仿真设备, 与本模块无关)。
        """
        code = (
            "import sys, importlib.util;"
            f"spec = importlib.util.spec_from_file_location('bench_twin_iso', r'{MODULE_PATH}');"
            "m = importlib.util.module_from_spec(spec);"
            "sys.modules[spec.name] = m;"
            "spec.loader.exec_module(m);"
            "bad = [k for k in sys.modules if k.startswith('ao_shaping.drivers')"
            " or 'gxipy' in k or 'Drv_UDPST' in k];"
            "print('OK' if not bad else 'BAD:' + ','.join(bad))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "OK", result.stdout

    def test_no_device_is_required(self) -> None:
        """构造并跑一次完整正向: 不需要任何设备上下文。"""
        twin = BenchTwin(BenchTwinConfig())
        image = twin.far_field(np.zeros((twin.region, twin.region)))
        assert image.shape == (twin.crop_side, twin.crop_side)
        assert float(image.max()) == pytest.approx(twin.config.peak_adu, rel=1e-9)


# ---------------------------------------------------------------------------
# 12. 焦面标度口径 (实测标度 ↔ 等效焦距)
# ---------------------------------------------------------------------------


class TestMeasuredScale:
    """由实测标度反推等效焦距的入口。

    孪生的远场标度只通过焦距进入 (``p_fft = λ·f/(N_pad·pupil_pitch_m)``),
    所以「实测标度 → 等效焦距 → 孪生」这条链必须往返自洽。
    """

    def test_round_trip_reproduces_the_measured_scale(self) -> None:
        """实测标度 → 反推 f → 孪生解析标度 = 传入值。"""
        config = BenchTwinConfig.from_measured_scale(
            MEASURED_SCALE_20261019, **MEASURED_GEOMETRY_20261019
        )
        twin = BenchTwin(config)
        assert twin.camera_px_per_model_px() == pytest.approx(
            MEASURED_SCALE_20261019, rel=1e-12
        )
        # 反推公式与解析式共用同一对模块级辅助函数 ⇒ 逐位相同。
        assert twin.camera_px_per_model_px() == MEASURED_SCALE_20261019

    def test_equivalent_focal_length_matches_the_hand_computed_value(self) -> None:
        """反推的 f 与手算一致 (2026-10-19 那组数字 → 311.32 mm)。"""
        geo = MEASURED_GEOMETRY_20261019
        focal = equivalent_focal_length_m(MEASURED_SCALE_20261019, **geo)

        n_pad = geo["region"] * geo["far_field_padding"]
        pupil_pitch = (
            geo["region_span_px"] / geo["region"] * geo["slm_pixel_um"] * 1e-6
        )
        hand = (
            MEASURED_SCALE_20261019
            * (geo["ccd_pixel_um"] * 1e-6)
            * n_pad
            * pupil_pitch
            / geo["wavelength_m"]
        )
        assert focal == pytest.approx(hand, abs=1e-9)
        # 报告里印的是两位小数: 311.32 mm。
        assert round(focal * 1e3, 2) == pytest.approx(311.32, abs=1e-9)

    def test_nominal_focal_length_would_be_off_by_the_measured_ratio(self) -> None:
        """按标称焦距算的解析标度与实测差 2.54 倍 (提醒, 不是断言口径)。"""
        geo = MEASURED_GEOMETRY_20261019
        nominal = BenchTwinConfig(
            wavelength_m=geo["wavelength_m"],
            slm_pixel_um=geo["slm_pixel_um"],
            ccd_pixel_um=geo["ccd_pixel_um"],
            focal_length_m=0.12240601503759398,
            region=geo["region"],
            region_span_px=geo["region_span_px"],
            far_field_padding=geo["far_field_padding"],
        )
        nominal_scale = BenchTwin(nominal).camera_px_per_model_px()
        assert nominal_scale == pytest.approx(1.027778, rel=1e-5)
        ratio = MEASURED_SCALE_20261019 / nominal_scale
        assert ratio == pytest.approx(2.5434, rel=1e-4)

    def test_default_path_is_bitwise_unchanged(self) -> None:
        """默认配置 (标称焦距口径) 的解析标度逐位不变。"""
        twin = BenchTwin(BenchTwinConfig())
        assert twin.far_field_size == 1280
        assert twin.pupil_pitch_m == 5.625e-05
        assert twin.p_fft == 1.8472222222222225e-06
        assert twin.camera_px_per_model_px() == 0.8396464646464648

    def test_default_far_field_is_bitwise_unchanged(self) -> None:
        """默认路径的远场逐位不变 (等效焦距入口不改变默认行为)。"""
        twin = BenchTwin(BenchTwinConfig())
        phase = _tilt_phase(twin.region)
        expected = _reference_far_field(BenchTwinConfig(), phase)
        np.testing.assert_array_equal(twin.far_field(phase), expected)

    @pytest.mark.parametrize(
        "scale", [0.0, -1.0, float("nan"), float("inf"), -2.614022035287756]
    )
    def test_invalid_measured_scale_raises(self, scale: float) -> None:
        with pytest.raises(ValueError):
            equivalent_focal_length_m(
                scale, **MEASURED_GEOMETRY_20261019
            )
        with pytest.raises(ValueError):
            BenchTwinConfig.from_measured_scale(
                scale, **MEASURED_GEOMETRY_20261019
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("region", 1),
            ("region", 0),
            ("region", -3),
            ("far_field_padding", 0),
            ("far_field_padding", -2),
            ("wavelength_m", 0.0),
            ("wavelength_m", -1e-9),
            ("slm_pixel_um", 0.0),
            ("ccd_pixel_um", 0.0),
            ("region_span_px", 0.0),
            ("region_span_px", -900.0),
        ],
    )
    def test_invalid_geometry_raises(self, field: str, value: object) -> None:
        kwargs = dict(MEASURED_GEOMETRY_20261019)
        kwargs[field] = value
        with pytest.raises(ValueError):
            equivalent_focal_length_m(MEASURED_SCALE_20261019, **kwargs)
        with pytest.raises(ValueError):
            BenchTwinConfig.from_measured_scale(
                MEASURED_SCALE_20261019, **kwargs
            )
