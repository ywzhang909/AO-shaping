"""Simulation-based correctness tests for GS square beam shaping.

These tests verify that the phase produced by ``generate_gs_square_phase``
(used by the "GS方形整形" SLM pattern) actually shapes a beam into a square
at the focal plane. All verification uses angular-spectrum propagation with no
hardware required.

Workflow under test:
    1. A raw far-field beam spot image (synthetic Gaussian) is measured to
       auto-compute the square side (side = factor x spot diameter).
    2. ``generate_gs_square_phase`` runs Gerchberg-Saxton at SLM resolution
       and returns uint16 grayscale phase.
    3. The gray phase is decoded back to radians and propagated to the focal
       plane; the resulting intensity must be a uniform square, not a Gaussian.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from ao_shaping.algorithm.gerchberg_saxton import angular_spectrum_propagate
from ao_shaping.gui.slm.multi_slm_controller import (
    build_square_target_amplitude,
    compute_square_side,
    generate_gs_square_phase,
)
from ao_shaping.utils.spots_calc import centroid, radius


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_slm(
    width: int = 256,
    height: int = 256,
    bits: int = 10,
    pixel_pitch_um: float = 8.0,
    wavelength: int = 1064,
) -> MagicMock:
    """Create a small mock SLM with a realistic phase->grayscale mapping.

    ``create_phase_from_array`` mimics the Santec SLM200: wraps radians into
    [0, 2pi) then maps to [0, max_gray] as uint16.
    """
    slm = MagicMock()
    slm.Panel_Res = (width, height)
    slm.Pitch_um = pixel_pitch_um
    slm.Gray_Scale_bits = bits
    slm.wavelength = wavelength
    max_gray = 2**bits - 1

    def _create_phase_from_array(
        phase_rad: np.ndarray, max_grayscale: int | None = None
    ) -> np.ndarray:
        mg = max_gray if max_grayscale is None else int(max_grayscale)
        wrapped = np.mod(np.asarray(phase_rad, dtype=np.float64), 2 * np.pi)
        return np.round(wrapped / (2 * np.pi) * mg).astype(np.uint16)

    slm.create_phase_from_array = _create_phase_from_array
    return slm


def _gaussian_spot(size: int = 256, sigma: float = 10.0) -> np.ndarray:
    """Synthetic far-field beam spot: a centered 2D Gaussian intensity."""
    y, x = np.mgrid[0:size, 0:size]
    c = size / 2
    return np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))


def _recover_phase(gray: np.ndarray, bits: int) -> np.ndarray:
    """Invert the SLM uint16 grayscale mapping back to radians."""
    max_gray = 2**bits - 1
    return gray.astype(np.float64) / max_gray * (2 * np.pi)


def _propagate_to_focal(
    phase_rad: np.ndarray,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
) -> np.ndarray:
    """Propagate a unit-amplitude, phase-modulated field to the focal plane."""
    field = np.ones_like(phase_rad) * np.exp(1j * phase_rad)
    return np.abs(angular_spectrum_propagate(field, cell_spacing, distance, wavelength)) ** 2


def _square_metrics(
    intensity: np.ndarray,
    square_mask: np.ndarray,
    bright_fraction: float = 0.2,
) -> dict[str, float]:
    """Compute the quantitative squareness metrics for an intensity map.

    Args:
        intensity: Focal-plane intensity (2D).
        square_mask: Boolean mask of the intended square region.
        bright_fraction: Fraction of peak used as the "bright" threshold.

    Returns:
        dict with keys: energy_in, flat_top_area, flat_top_cv, squareness.
    """
    energy_in = float(np.sum(intensity * square_mask) / np.sum(intensity))
    peak = float(intensity.max())
    bright_in_square = intensity[square_mask]
    above = bright_in_square > bright_fraction * peak
    flat_top_area = float(np.mean(above))
    cv = float(
        bright_in_square[above].std()
        / (bright_in_square[above].mean() + 1e-9)
        if above.any()
        else 0.0
    )

    mask = intensity > bright_fraction * peak
    ys, xs = np.nonzero(mask)
    aspect = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) / max(
        min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), 1
    )
    squareness = float(abs(1.0 - aspect))
    return {
        "energy_in": energy_in,
        "flat_top_area": flat_top_area,
        "flat_top_cv": cv,
        "squareness": squareness,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGSBeamShapingToSquare:
    """End-to-end check that GS square-shaping phase yields a square."""

    GRID = 256
    FACTOR = 1.5
    FOCAL_M = 0.1
    ITERATIONS = 200

    @classmethod
    def _build(cls) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        slm = _mock_slm(cls.GRID, cls.GRID)
        spot = _gaussian_spot(cls.GRID, sigma=10.0)
        gray = generate_gs_square_phase(
            slm,
            spot,
            factor=cls.FACTOR,
            focal_length_m=cls.FOCAL_M,
            iterations=cls.ITERATIONS,
            energy=0.90,
        )
        assert gray.shape == (cls.GRID, cls.GRID)

        cx, cy = centroid(spot)
        r = radius(spot, center=(cx, cy), energy=0.90, use_aotools=False)
        side = compute_square_side(
            2.0 * r, factor=cls.FACTOR, p_cam=8e-6, d_slm=8e-6
        )
        square_mask = build_square_target_amplitude(
            cls.GRID, cls.GRID, side
        ).astype(bool)

        phase = _recover_phase(gray, slm.Gray_Scale_bits)
        intensity = _propagate_to_focal(phase)
        metrics = _square_metrics(intensity, square_mask)
        return intensity, square_mask, metrics

    def test_most_energy_lands_inside_square(self):
        """More than half the focal energy must fall inside the square."""
        _, _, m = self._build()
        assert m["energy_in"] > 0.5, f"energy_in={m['energy_in']:.3f}"

    def test_square_is_filled_flat_top(self):
        """The square region must be mostly bright (not just an outline)."""
        _, _, m = self._build()
        assert m["flat_top_area"] > 0.5, f"flat_top_area={m['flat_top_area']:.3f}"

    def test_flat_top_uniformity(self):
        """Intensity across the bright square must be reasonably uniform."""
        _, _, m = self._build()
        assert m["flat_top_cv"] < 0.4, f"flat_top_cv={m['flat_top_cv']:.3f}"

    def test_output_is_square_not_elongated(self):
        """Bright region aspect ratio must be close to 1 (a square)."""
        _, _, m = self._build()
        assert m["squareness"] < 0.25, f"squareness={m['squareness']:.3f}"

    def test_gaussian_control_is_not_flat_top(self):
        """A plain Gaussian spot must NOT look like a filled square.

        This guards against the metrics passing vacuously for any roundish
        blob: the shaped output is flat-topped, an unshaped Gaussian is not.
        """
        slm = _mock_slm(self.GRID, self.GRID)
        # Broad Gaussian that fills most of the frame.
        spot = _gaussian_spot(self.GRID, sigma=30.0)
        gray = generate_gs_square_phase(
            slm,
            spot,
            factor=self.FACTOR,
            focal_length_m=self.FOCAL_M,
            iterations=60,
            energy=0.90,
        )
        phase = _recover_phase(gray, slm.Gray_Scale_bits)
        intensity = _propagate_to_focal(phase)

        peak = intensity.max()
        flat_top_area = float(np.mean(intensity > 0.8 * peak))
        assert flat_top_area < 0.3, f"gaussian flat_top_area={flat_top_area:.3f}"
