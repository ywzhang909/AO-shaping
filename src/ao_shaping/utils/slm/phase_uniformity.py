"""Circular phase smoothing and gradient diagnostics for SLM pupils."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def circular_smooth_phase(
    phase: np.ndarray, sigma_px: float, pupil: np.ndarray
) -> np.ndarray:
    """Smooth unit phasors inside a pupil without blurring across 2-pi wraps."""
    raw = np.asarray(phase, dtype=np.float64)
    mask = np.asarray(pupil, dtype=bool)
    if raw.ndim != 2 or raw.shape != mask.shape or not np.isfinite(raw).all():
        raise ValueError("phase must be a finite 2D array matching pupil")
    if not np.isfinite(sigma_px) or sigma_px < 0:
        raise ValueError("sigma_px must be finite and nonnegative")
    if sigma_px == 0:
        return raw.copy()
    real = gaussian_filter(np.where(mask, np.cos(raw), 0.0), sigma=sigma_px, mode="constant")
    imag = gaussian_filter(np.where(mask, np.sin(raw), 0.0), sigma=sigma_px, mode="constant")
    result = raw.copy()
    result[mask] = np.arctan2(imag[mask], real[mask])
    return result


def wrapped_gradient_rms(phase: np.ndarray, pupil: np.ndarray) -> float:
    """RMS nearest-neighbour phase step in radians within the pupil."""
    raw = np.asarray(phase, dtype=np.float64)
    mask = np.asarray(pupil, dtype=bool)
    if raw.ndim != 2 or raw.shape != mask.shape or not np.isfinite(raw).all():
        raise ValueError("phase must be a finite 2D array matching pupil")
    dx = raw[:, 1:] - raw[:, :-1]
    dy = raw[1:, :] - raw[:-1, :]
    x = np.arctan2(np.sin(dx), np.cos(dx))[mask[:, 1:] & mask[:, :-1]]
    y = np.arctan2(np.sin(dy), np.cos(dy))[mask[1:, :] & mask[:-1, :]]
    if x.size + y.size == 0:
        raise ValueError("pupil has no adjacent pixels")
    return float(np.sqrt((np.sum(x * x) + np.sum(y * y)) / (x.size + y.size)))
