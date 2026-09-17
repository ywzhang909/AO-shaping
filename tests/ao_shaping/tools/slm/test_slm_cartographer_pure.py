from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.cartographer.cosine_pattern import (
    CosinePatternConfig,
    estimate_phase_from_pattern,
    generate_center_cosine_pattern,
    generate_traditional_gradient_pattern,
    get_pattern_peak_position,
)
from ao_shaping.tools.slm.cartographer.phase_grayscale_lut import (
    LUTCalibrationResult,
)
from ao_shaping.tools.slm.cartographer.wavefront_reconstruction import (
    FourierReconstructionConfig,
    FourierWavefrontReconstructor,
    interpolate_sparse_to_dense,
)


def test_center_cosine_pattern_is_bounded_and_centered() -> None:
    config = CosinePatternConfig(
        center_x=10,
        center_y=10,
        radius_pixels=4.0,
        max_phase_2pi=0.5,
        background_grayscale=7,
        smooth_edge_width=0.0,
    )

    pattern = generate_center_cosine_pattern(config, (20, 20))

    assert pattern.shape == (20, 20)
    assert pattern.dtype == np.uint16
    assert pattern.min() == 0
    assert pattern.max() == 511
    assert pattern[10, 10] == pattern.max()
    assert pattern[0, 0] == 7


def test_traditional_gradient_runs_along_requested_axis() -> None:
    config = CosinePatternConfig(
        center_x=2,
        center_y=2,
        radius_pixels=2.0,
        max_phase_2pi=1.0,
        background_grayscale=0,
        smooth_edge_width=0.0,
    )

    pattern = generate_traditional_gradient_pattern(config, "x", (5, 5))

    assert pattern[2, 0] == 0
    assert pattern[2, 4] == 1023


def test_phase_estimation_uses_linear_gray_to_radian_map() -> None:
    grayscale = np.array([[0, 512, 1023]], dtype=np.uint16)

    phase = estimate_phase_from_pattern(grayscale, max_grayscale=1023)

    assert phase[0, 0] == 0.0
    assert phase[0, 1] == pytest.approx(2.0 * np.pi * 512 / 1023)
    assert phase[0, 2] == pytest.approx(2.0 * np.pi)


def test_pattern_peak_position_uses_row_column_order() -> None:
    pattern = np.zeros((4, 6), dtype=np.uint16)
    pattern[1, 3] = 9

    assert get_pattern_peak_position(pattern) == (1, 3)
    assert get_pattern_peak_position(np.zeros((4, 6), dtype=np.uint16)) == (2, 3)


def test_fourier_reconstruction_returns_finite_zero_mean_wavefront() -> None:
    config = FourierReconstructionConfig(
        interpolate_dense=False,
        filter_type="none",
        pad_factor=1.0,
    )
    reconstructor = FourierWavefrontReconstructor(config, (4, 4))
    displacements = np.zeros((4, 4), dtype=np.float64)

    wavefront = reconstructor.reconstruct(displacements, displacements)

    assert wavefront.shape == (4, 4)
    assert np.isfinite(wavefront).all()
    assert wavefront == pytest.approx(np.zeros((4, 4)), abs=1e-12)


def test_fourier_reconstruction_rejects_non_2d_input() -> None:
    config = FourierReconstructionConfig(interpolate_dense=False)
    reconstructor = FourierWavefrontReconstructor(config, (2, 2))

    with pytest.raises(ValueError):
        reconstructor.reconstruct(np.zeros(4), np.zeros((2, 2)))


def test_sparse_interpolation_increases_resolution() -> None:
    sparse_x = np.arange(9, dtype=np.float64).reshape(3, 3)
    sparse_y = np.arange(9, dtype=np.float64).reshape(3, 3)[::-1]

    dense_x, dense_y = interpolate_sparse_to_dense(sparse_x, sparse_y, (6, 6))

    assert dense_x.shape == (6, 6)
    assert dense_y.shape == (6, 6)
    assert np.isfinite(dense_x).all()
    assert np.isfinite(dense_y).all()


def test_lut_result_interpolates_and_round_trips(tmp_path) -> None:
    result = LUTCalibrationResult(
        grayscale_values=[0, 512, 1023],
        measured_phases=[0.0, np.pi, 2.0 * np.pi],
        measured_phases_2pi=[0.0, 0.5, 1.0],
        lut={0: 0.0, 512: np.pi, 1023: 2.0 * np.pi},
    )

    assert result.get_phase(256) == pytest.approx(np.pi / 2)
    assert result.get_grayscale_for_phase(np.pi / 2) == 256

    path = result.save(tmp_path / "lut.json")
    restored = LUTCalibrationResult.load(path)

    assert restored.grayscale_values == result.grayscale_values
    assert restored.measured_phases == pytest.approx(result.measured_phases)
    assert restored.lut == result.lut
