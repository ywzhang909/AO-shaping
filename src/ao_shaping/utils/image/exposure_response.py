"""Camera-independent low-parameter exposure-response diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def response_fits(exposures: np.ndarray, signals: np.ndarray) -> dict:
    """Fit low-parameter candidates and test held-out exposure levels."""
    t = np.asarray(exposures, dtype=np.float64)
    y = np.asarray(signals, dtype=np.float64)
    if t.shape != y.shape or t.ndim != 1 or len(t) < 4:
        raise ValueError("need at least four matched exposure and signal values")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(t <= 0):
        raise ValueError("exposure and signal must be finite; exposure must be positive")
    levels = np.unique(t)
    if len(levels) < 3:
        raise ValueError("need at least three distinct exposure levels")
    results = {}
    for degree, name in ((1, "affine"), (2, "quadratic")):
        if len(levels) <= degree + 1:
            continue
        coeff = np.polyfit(t, y, degree)
        prediction = np.polyval(coeff, t)
        cv_errors = []
        for level in levels:
            train = t != level
            if np.unique(t[train]).size <= degree:
                continue
            cv_coeff = np.polyfit(t[train], y[train], degree)
            cv_errors.extend((np.polyval(cv_coeff, t[~train]) - y[~train]).tolist())
        results[name] = {
            "coefficients_high_to_low": coeff.tolist(),
            "relative_rms": float(np.sqrt(np.mean((prediction - y) ** 2)) / y.mean()),
            "leave_one_level_out_relative_rms": (
                float(np.sqrt(np.mean(np.square(cv_errors))) / y.mean())
                if cv_errors else None
            ),
            "monotone_on_measured_range": bool(
                np.all(np.diff(np.polyval(coeff, np.linspace(t.min(), t.max(), 100))) > 0)
            ),
        }
        if degree == 1:
            results[name]["intercept_over_slope_ms"] = float(coeff[1] / coeff[0])
    means = np.array([y[t == level].mean() for level in levels])
    if len(levels) >= 3 and np.all(np.diff(means) > 0):
        lut = PchipInterpolator(levels, means)
        results["monotone_cubic_lut"] = {
            "exposures_ms": levels.tolist(),
            "mean_roi_signal": means.tolist(),
            "relative_rms_on_repeats": float(
                np.sqrt(np.mean((lut(t) - y) ** 2)) / y.mean()
            ),
            "status": "interpolates measured levels; no independent exposure-level validation",
        }
    results["double_exponential"] = {
        "status": "not fitted: independent levels and repeats insufficient for four-plus parameters"
    }
    results["logistic"] = {
        "status": "not fitted: independent levels and repeats insufficient for four-plus parameters"
    }
    return results

