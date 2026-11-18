"""Shared leaf helpers for the bench calibration modules.

``_pupil_amplitude`` and ``_spot_fwhm`` are used by both the speckle geometry
solve (:mod:`ao_shaping.optimizer.wfless.bench_geometry`) and the sweep fit
(:mod:`ao_shaping.optimizer.wfless.bench_sweep_fit`).  ``CalibrationRecord`` and
:func:`square_metrics_at_zero_order` are consumed directly by the runbook's
shared-aberration route.  They live here so those modules can import one another
strictly downward (common <- geometry / sweep / sweep_fit / shared_aberration)
without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.utils.image.beam_metrics import (
    compute_square_metrics,
    zero_order_center,
)
from ao_shaping.utils.image.frame_analysis import spot_fwhm

#: ``spot_fwhm`` was sunk to the leaf ``utils/image`` layer (R-48, #64); this alias
#: keeps the private name used by the calibration modules and their tests intact.
_spot_fwhm = spot_fwhm

#: Smallest pupil semi-axis, in model pixels, the geometry calibration accepts. Below it the
#: aperture is too small for the tilt and defocus fits to mean anything.
_MIN_PUPIL_SEMI_AXIS_PX = 8

#: A sweep needs at least this many usable points before a response curve can be fitted:
#: with one, a slope is undefined rather than merely noisy, so that axis/mode/offset is
#: skipped or reported as dropped instead of fitted.
_MIN_SWEEP_POINTS = 2


def _has_both_signs(values: np.ndarray) -> bool:
    """True when the coefficients straddle zero, so an offset is observable."""
    v = np.asarray(values, dtype=np.float64)
    return bool(np.any(v > 0) and np.any(v < 0))


def square_metrics_at_zero_order(
    intensity: np.ndarray, target_side: int
) -> tuple[dict[str, float], tuple[int, int]]:
    """Compute the square metrics with the 0-order located by global argmax.

    The optical axis is the frame's global maximum, never the geometric frame
    centre, so the anchor comes from
    :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center` with
    ``refine=False`` (literally ``np.unravel_index(np.argmax(frame))``) and is
    handed straight to
    :func:`~ao_shaping.utils.image.beam_metrics.compute_square_metrics`, which
    shares the ``(x, y)`` convention.

    Args:
        intensity: 2D far-field intensity frame.
        target_side: Target square side, pixels.

    Returns:
        A tuple of the metric dict and the ``(x, y)`` centre used.
    """
    center = zero_order_center(intensity, refine=False)
    return compute_square_metrics(intensity, target_side, center), center


def _pupil_amplitude(
    region: int, waist_model_px: float, semi_axis_model_px: int
) -> np.ndarray:
    """Gaussian illumination masked to the aperture inscribed in the grid."""
    yy, xx = np.mgrid[0:region, 0:region]
    r = int(min(int(semi_axis_model_px), region // 2))
    r2 = (xx - region / 2.0) ** 2 + (yy - region / 2.0) ** 2
    amp = np.exp(-r2 / (2.0 * max(float(waist_model_px), 1e-6) ** 2))
    amp[r2 > float(r) ** 2] = 0.0
    return amp


@dataclass
class CalibrationRecord:
    """One ``(far field, pupil phase)`` pair for the shared fit.

    Attributes:
        image: Measured far field. Any non-empty 2D shape; cropped to the model
            grid by the calibration.
        phase: Pupil phase that produced ``image``, raw unwrapped radians. Any
            non-empty 2D shape; resampled to the model grid.
        label: Human-readable provenance, echoed in the result.
        coefficients: Optional per-record ground truth in Noll order, used only
            to report the per-record fit error.
    """

    image: np.ndarray
    phase: np.ndarray
    label: str = "explicit"
    coefficients: np.ndarray | None = None
