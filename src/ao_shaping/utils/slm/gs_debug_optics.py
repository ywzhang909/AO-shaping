"""Idealized, phase-only optics for inspecting saved GS debug frames.

The result is a normalized FFT image, not a calibrated CCD prediction: the
incident amplitude, SLM phase LUT, aberrations and camera sampling are unknown.
"""

from __future__ import annotations

import numpy as np


def grayscale_to_radians(gray: np.ndarray, max_gray: int = 1023) -> np.ndarray:
    """Map recorded panel gray to an ideal linear 0..2pi phase LUT."""
    if max_gray <= 0:
        raise ValueError("max_gray must be positive")
    array = np.asarray(gray)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("gray must be a 2D integer panel array")
    if np.any(array < 0) or np.any(array > max_gray):
        raise ValueError("gray values are outside the phase LUT range")
    return array.astype(np.float64) * (2.0 * np.pi / max_gray)


def idealized_far_field(
    phase_rad: np.ndarray,
    *,
    pupil_center: tuple[float, float],
    beam_radius_px: float,
) -> np.ndarray:
    """Return peak-normalized Fraunhofer intensity for an off-center disk pupil.

    ``pupil_center`` is (x, y) in SLM panel pixels. No claim is made about CCD
    pixel scale or absolute efficiency.
    """
    phase = np.asarray(phase_rad, dtype=np.float64)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("phase_rad must be a finite 2D array")
    cx, cy = pupil_center
    if not np.isfinite((cx, cy, beam_radius_px)).all() or beam_radius_px <= 0:
        raise ValueError("pupil center and radius must be finite; radius > 0")
    yy, xx = np.ogrid[: phase.shape[0], : phase.shape[1]]
    pupil = (xx - cx) ** 2 + (yy - cy) ** 2 <= beam_radius_px**2
    if not pupil.any():
        raise ValueError("pupil does not intersect the panel")
    field = pupil * np.exp(1j * phase)
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    return intensity / intensity.max()
