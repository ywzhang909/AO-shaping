"""Physical forward-model and shaping metrics.

The functions in this module deliberately keep *model prediction quality* and
*physical shaping quality* separate.  Pearson correlation alone is blind to
gain, offset and some geometric errors; CV alone is sensitive to exposure and
background.  Every metric is computed on an explicit core mask and can retain
the zero-order exclusion used by the experiment protocol.

This module is offline and NumPy-only.  It never opens a camera or an SLM.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ao_shaping.utils.image.beam_metrics import masked_cv, masked_energy_fraction


def _finite_pair(predicted: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predicted, dtype=np.float64)
    m = np.asarray(measured, dtype=np.float64)
    if p.shape != m.shape or p.ndim != 2:
        raise ValueError(f"predicted and measured must be matching 2D arrays, got {p.shape} and {m.shape}")
    if not p.size or not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError("images must be nonempty and finite; invalid captures cannot be repaired silently")
    return p, m


def _mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    out = np.asarray(mask)
    if out.dtype != np.bool_ or out.shape != shape or not out.any():
        raise ValueError(f"core mask must match image shape and be non-empty, got {out.shape}")
    return out.copy()


def affine_radiometry(predicted: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Fit ``measured ~= gain * predicted + offset`` on *mask*.

    The fit is a diagnostic nuisance elimination, not a calibration result.  A
    non-negative gain is enforced so an inverted intensity response cannot be
    silently accepted as a good model.
    """
    p, m = _finite_pair(predicted, measured)
    support = _mask(mask, p.shape)
    x, y = p[support].ravel(), m[support].ravel()
    xc, yc = x - x.mean(), y - y.mean()
    variance = float(xc @ xc)
    gain = max(float(xc @ yc) / variance, 0.0) if variance > 0 else 0.0
    return gain, float(y.mean() - gain * x.mean())


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = x.ravel(), y.ravel()
    sx, sy = float(x.std()), float(y.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _centroid(image: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, None) * mask
    total = float(values.sum())
    if total <= 0.0:
        return None
    yy, xx = np.indices(values.shape, dtype=np.float64)
    return (float((values * xx).sum() / total), float((values * yy).sum() / total))


def compute_forward_metrics(
    predicted: np.ndarray,
    measured: np.ndarray,
    core: np.ndarray,
    *,
    exposure_ms: float | None = None,
    full_scale: float | None = None,
    zero_order_exclusion: np.ndarray | None = None,
    camera_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return exposure-aware, mask-explicit model validation metrics.

    ``measured`` remains in the returned diagnostics as raw counts through the
    saturation fraction; when ``exposure_ms`` is provided, all shape metrics
    use counts per millisecond.  ``zero_order_exclusion`` is an additional
    boolean mask of pixels to remove from the core, so its use is auditable.
    Prediction must already be in the units of the measurement (counts/ms if
    exposure is supplied, otherwise counts). This function does not calibrate
    exposure response. EE is the core fraction of this supplied window only;
    negative corrected values are explicitly clipped for EE and moments, not CV.
    Affine fitting on this same frame is diagnostic only, never validation of a
    calibrated radiometry model. Undefined ratios/correlations return None.
    """
    p, m = _finite_pair(predicted, measured)
    valid_core = _mask(core, p.shape)
    if zero_order_exclusion is not None:
        excluded = np.asarray(zero_order_exclusion)
        if excluded.dtype != np.bool_ or excluded.shape != p.shape:
            raise ValueError("zero_order_exclusion must match image shape")
        valid_core &= ~excluded
    if not valid_core.any():
        raise ValueError("core is empty after zero-order exclusion")
    if exposure_ms is not None:
        exposure_ms = float(exposure_ms)
        if not np.isfinite(exposure_ms) or exposure_ms <= 0:
            raise ValueError("exposure_ms must be positive and finite")
        m_shape = m / exposure_ms
        p_shape = p
    else:
        m_shape, p_shape = m, p
    gain, offset = affine_radiometry(p_shape, m_shape, valid_core)
    affine = gain * p_shape + offset
    y, q = p_shape[valid_core], m_shape[valid_core]
    centroid_p = _centroid(p_shape, valid_core)
    centroid_m = _centroid(m_shape, valid_core)
    # Convert sampled array indices to native CCD coordinates when available.
    if camera_xy is not None:
        coordinates = np.asarray(camera_xy, dtype=float)
        if coordinates.shape != (*p.shape, 2) or not np.isfinite(coordinates).all():
            raise ValueError("camera_xy must be a finite HxWx2 array of native CCD pixel coordinates")

        def physical_centroid(image):
            weight = np.maximum(image, 0) * valid_core
            total = weight.sum()
            return tuple((weight[..., None] * coordinates).sum((0, 1)) / total) if total > 0 else None

        centroid_p, centroid_m = physical_centroid(p_shape), physical_centroid(m_shape)
    centroid_error = (float(np.linalg.norm(np.subtract(centroid_p, centroid_m)))
                      if centroid_p is not None and centroid_m is not None else None)
    full_scale_value = float(full_scale) if full_scale is not None else None
    if full_scale_value is not None and (not np.isfinite(full_scale_value) or full_scale_value <= 0):
        raise ValueError("full_scale must be positive and finite")
    saturation_fraction = float(np.mean(m >= full_scale_value)) if full_scale_value else None
    cv_p = masked_cv(p_shape, valid_core) if y.mean() > 0 else None
    cv_m = masked_cv(m_shape, valid_core) if q.mean() > 0 else None
    denominator = float(np.linalg.norm(q))
    ee_p = masked_energy_fraction(p_shape, valid_core, nonnegative=True) if np.maximum(p_shape, 0).sum() > 0 else None
    ee_m = masked_energy_fraction(m_shape, valid_core, nonnegative=True) if np.maximum(m_shape, 0).sum() > 0 else None
    return {
        "n_core": int(valid_core.sum()),
        "pearson_ncc": _correlation(y, q),
        "relative_l2_raw": float(np.linalg.norm(y - q) / denominator) if denominator > 0 else None,
        "relative_l2_affine_diagnostic": float(np.linalg.norm(affine[valid_core] - q) / denominator) if denominator > 0 else None,
        "radiometry_gain": gain,
        "radiometry_offset": offset,
        "cv_prediction": cv_p,
        "cv_measured": cv_m,
        "cv_abs_error": abs(cv_p - cv_m) if cv_p is not None and cv_m is not None else None,
        "core_mean_prediction": float(y.mean()),
        "core_mean_measured": float(q.mean()),
        "ee_core_window_prediction": ee_p,
        "ee_core_window_measured": ee_m,
        "ee_abs_error": abs(ee_p - ee_m) if ee_p is not None and ee_m is not None else None,
        "centroid_error_px": centroid_error,
        "centroid_coordinate_unit": "native_ccd_px" if camera_xy is not None else "array_px",
        "saturation_fraction": saturation_fraction,
        "core_saturation_fraction": float(np.mean(m[valid_core] >= full_scale_value)) if full_scale_value else None,
        "exposure_ms": exposure_ms,
        "intensity_unit": "counts_per_ms" if exposure_ms is not None else "counts",
    }


def compute_directional_metrics(
    predicted_plus: np.ndarray,
    predicted_minus: np.ndarray,
    measured_plus: np.ndarray,
    measured_minus: np.ndarray,
    core: np.ndarray,
) -> dict[str, float | None]:
    """Compare a model finite-difference direction with the CCD direction.

    All four images must use the same intensity units, exposure, geometry and
    perturbation amplitude. No offset/gain is fitted to make a direction agree.
    Zero response returns undefined metrics rather than perfect agreement;
    callers must compare response amplitude against repeated-frame noise.
    """
    pp, pm = _finite_pair(predicted_plus, predicted_minus)
    mp, mm = _finite_pair(measured_plus, measured_minus)
    if pp.shape != mp.shape or pp.shape != np.asarray(core).shape:
        raise ValueError("direction arrays and core must have one matching 2D shape")
    support = _mask(core, pp.shape)
    dp, dm = (pp - pm)[support], (mp - mm)[support]
    npred, nmeas = float(np.linalg.norm(dp)), float(np.linalg.norm(dm))
    return {
        "direction_correlation": _correlation(dp, dm),
        "direction_relative_l2": float(np.linalg.norm(dp - dm) / nmeas) if nmeas > 0 else None,
        "direction_cosine": float(dp @ dm / (npred * nmeas)) if npred > 0 and nmeas > 0 else None,
        "direction_gain": float(dp @ dm / (npred**2)) if npred > 0 else None,
        "direction_prediction_norm": npred,
        "direction_measured_norm": nmeas,
        "direction_sign_agreement": float(np.mean(np.sign(dp) == np.sign(dm))) if npred > 0 and nmeas > 0 else None,
    }

