"""Bounded manual exposure control for quantitative CCD measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExposureDecision:
    next_ms: float
    accepted: bool
    peak_fraction: float


def choose_exposure(
    frame: np.ndarray,
    exposure_ms: float,
    *,
    max_ms: float = 0.4,
    target_fraction: float = 0.55,
    saturation_fraction: float = 0.9,
) -> ExposureDecision:
    """Choose a bounded next exposure from an unprocessed integer CCD frame.

    A dim frame at the hardware exposure limit is accepted; a saturated frame
    at the camera's minimum exposure is rejected by the caller after retrying.
    """
    image = np.asarray(frame)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError("auto exposure needs one raw integer CCD frame")
    if exposure_ms <= 0 or max_ms <= 0 or exposure_ms > max_ms + 1e-9:
        raise ValueError("exposure must be positive and no greater than max_ms")
    if not 0 < target_fraction < saturation_fraction < 1:
        raise ValueError("require 0 < target_fraction < saturation_fraction < 1")
    full_scale = float(np.iinfo(image.dtype).max)
    peak_fraction = float(np.max(image)) / full_scale
    if peak_fraction >= saturation_fraction:
        next_ms = max(0.001, exposure_ms * target_fraction / max(peak_fraction, 1e-9))
        return ExposureDecision(min(next_ms, max_ms), False, peak_fraction)
    if peak_fraction < target_fraction * 0.7 and exposure_ms < max_ms:
        next_ms = min(max_ms, exposure_ms * target_fraction / max(peak_fraction, 0.01))
        return ExposureDecision(next_ms, False, peak_fraction)
    return ExposureDecision(exposure_ms, True, peak_fraction)


def exposure_normalized_frame(
    frame: np.ndarray, exposure_ms: float, *, reference_ms: float = 0.1
) -> np.ndarray:
    """Convert raw counts to counts at a fixed reference exposure.

    Background subtraction remains with the caller, after this linear scaling.
    Camera gain must stay fixed; nonlinear or clipped frames must be rejected.
    """
    if exposure_ms <= 0 or reference_ms <= 0:
        raise ValueError("exposure and reference must be positive")
    return np.asarray(frame, dtype=np.float64) * (reference_ms / exposure_ms)
