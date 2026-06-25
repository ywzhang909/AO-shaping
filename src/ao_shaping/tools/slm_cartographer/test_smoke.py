"""Smoke test for slm_cartographer package.

Tests each module using direct sys.path injection to avoid circular imports.
Run with:  .venv\\Scripts\\python.exe src/ao_shaping/tools/slm_cartographer/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from loguru import logger

# __file__ is relative or absolute; resolve to absolute then walk up
_file = Path(__file__).resolve()
# file: .../src/ao_shaping/tools/slm_cartographer/test_smoke.py
# parents: [..., cartographer, tools, ao_shaping, src, AO-shaping, ...]
# We want "src" on sys.path:
sys.path.insert(0, str(_file.parents[3]))


def test_cosine_pattern() -> None:
    from ao_shaping.tools.slm_cartographer.cosine_pattern import (
        CosinePatternConfig,
        generate_center_cosine_pattern,
        generate_traditional_gradient_pattern,
        get_pattern_peak_position,
        estimate_phase_from_pattern,
    )

    logger.info("=== Cosine Pattern Tests ===")
    config = CosinePatternConfig(center_x=960, center_y=540, radius_pixels=40.0)
    pattern = generate_center_cosine_pattern(config=config)
    assert pattern.shape == (1200, 1920)
    assert pattern.dtype == np.uint16
    assert pattern.min() >= 0 and pattern.max() <= 1023
    logger.info(
        f"  Pattern OK: shape={pattern.shape}, range=[{pattern.min()}, {pattern.max()}]"
    )

    gradient = generate_traditional_gradient_pattern(config=config, direction="x")
    assert gradient.shape == pattern.shape
    logger.info(f"  Gradient OK: shape={gradient.shape}")

    py, px = get_pattern_peak_position(pattern)
    assert pattern[py, px] == pattern.max()
    logger.info(f"  Peak at (x={px}, y={py}), value={pattern[py, px]}")

    phase_rad = estimate_phase_from_pattern(pattern)
    assert abs(phase_rad[py, px] - 2 * np.pi) < 0.01
    logger.info("  Phase estimation OK")


def test_wavefront_reconstruction() -> None:
    from ao_shaping.tools.slm_cartographer.wavefront_reconstruction import (
        FourierReconstructionConfig,
        FourierWavefrontReconstructor,
        reconstruct_from_displacements,
    )

    logger.info("\n=== Wavefront Reconstruction Tests ===")
    np.random.seed(42)
    num_spots = (10, 10)
    dx = np.random.randn(*num_spots) * 0.1
    dy = np.random.randn(*num_spots) * 0.1
    reconstructor = FourierWavefrontReconstructor(
        config=FourierReconstructionConfig(),
        num_spots=num_spots,
    )
    wf = reconstructor.reconstruct(dx, dy)
    assert wf.ndim == 2
    assert np.isfinite(wf).all()
    logger.info(
        f"  Reconstructed wavefront: shape={wf.shape}, RMS={np.std(wf):.6f} rad"
    )


def test_lut_data_structures() -> None:
    from ao_shaping.tools.slm_cartographer.phase_grayscale_lut import (
        LUTCalibrationConfig,
        LUTCalibrationResult,
    )

    logger.info("\n=== LUT Data Structure Tests ===")
    result = LUTCalibrationResult(
        grayscale_values=[0, 256, 512, 768, 1023],
        measured_phases=[0.0, 1.57, 3.14, 4.71, 6.28],
        measured_phases_2pi=[0.0, 0.25, 0.5, 0.75, 1.0],
        lut={0: 0.0, 256: 1.57, 512: 3.14, 768: 4.71, 1023: 6.28},
    )

    phase = result.get_phase(512)
    assert abs(phase - 3.14) < 0.5
    logger.info(f"  Phase at gs=512: {phase:.4f} rad (expected ~3.14)")

    gs = result.get_grayscale_for_phase(np.pi)
    assert isinstance(gs, int)
    logger.info(f"  Grayscale for phase=π: {gs} (expected ~400-550)")

    lut_copy = result.to_dict()
    restored = LUTCalibrationResult.from_dict(lut_copy)
    assert restored.grayscale_values == result.grayscale_values
    logger.info("  Serialize/deserialize OK")


def test_compensation_data_structures() -> None:
    from ao_shaping.tools.slm_cartographer.dynamic_compensation import (
        CompensationConfig,
        CompensationResult,
    )

    logger.info("\n=== Compensation Data Structure Tests ===")
    config = CompensationConfig(slm_wavelength_nm=532, n_correction_iterations=1)

    result = CompensationResult(
        initial_wavefront_rms=0.1,
        final_wavefront_rms=0.05,
        converged=True,
        iterations_used=2,
    )
    d = result.to_dict()
    assert d["initial_wavefront_rms"] == 0.1
    assert d["final_wavefront_rms"] == 0.05
    assert d["converged"] is True
    logger.info(
        f"  Compensation result: RMS {d['initial_wavefront_rms']:.3f} -> {d['final_wavefront_rms']:.3f}"
    )


def test_fourier_reconstruction_function() -> None:
    from ao_shaping.tools.slm_cartographer.wavefront_reconstruction import (
        interpolate_sparse_to_dense,
    )

    logger.info("\n=== Interpolation Test ===")
    sparse_x = np.random.randn(5, 5).astype(np.float32)
    sparse_y = np.random.randn(5, 5).astype(np.float32)
    dx, dy = interpolate_sparse_to_dense(sparse_x, sparse_y)
    assert dx.shape[0] > sparse_x.shape[0]
    assert dx.shape[1] > sparse_x.shape[1]
    logger.info(f"  Interpolated: {sparse_x.shape} -> {dx.shape}")


if __name__ == "__main__":
    test_cosine_pattern()
    test_wavefront_reconstruction()
    test_lut_data_structures()
    test_compensation_data_structures()
    test_fourier_reconstruction_function()
    logger.info("\n=== All tests passed ===")
