"""Shared optical gain, irradiance background and exposure-independent counts."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def count_rate(image, row):
    """Measured irradiance in native counts per millisecond.

    This is the single normalization for every stage that compares a measured
    frame against the frozen model: ``fit``, ``blur``, ``validate``,
    ``direction``, ``local_response`` and ``axial``.

    Bit depth is deliberately *not* folded in. Folding it in per call site is
    what broke the 2026-10-29 runs: ``fit``/``blur`` divide by exposure only,
    so the frozen ``gain`` is expressed in native counts, but ``validate``
    divided its anchor frame by ``exposure * counts_scale`` (16 for Mono12).
    ``anchored_intensity`` then added a native-counts model difference to an
    anchor 16x smaller than it, so every model step predicted a ~70% loss
    change that the CCD could not see: 0/6 model blocks accepted, and the
    anchored blocks drove the RMS cap instead of converging (measured ratio
    model/measured = 16.1 on both runs, all three shapes).
    """
    exposure = float(row["exposure_ms"])
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"positive finite exposure required, got {row['exposure_ms']!r}")
    return np.asarray(image) / exposure


def fit_radiometry(optical, measured, exposure_ms, train, *, mode="exposure"):
    """Fit training pixels only in counts/ms: g*optical + b + bias_counts/t.

    The counts intercept is compatible with electronic bias but cannot be
    uniquely attributed to electronics without a dark/response experiment.
    """
    optical, measured = np.asarray(optical), np.asarray(measured)
    exposure = np.asarray(exposure_ms, dtype=float)
    train = np.asarray(train, dtype=bool)
    if optical.shape != measured.shape or optical.ndim != 3 or exposure.shape != (len(optical),):
        raise ValueError("aligned images and one exposure per phase required")
    if not np.isfinite(exposure).all() or np.any(exposure <= 0) or not train.any():
        raise ValueError("positive finite exposures and training phases required")
    columns = [optical, np.ones_like(optical)]
    if mode == "exposure":
        columns.append(np.broadcast_to(1/exposure[:, None, None], optical.shape))
    elif mode != "shared":
        raise ValueError("unknown radiometry mode")
    design = np.stack(columns, axis=-1)
    coefficients = nnls(design[train].reshape(-1, len(columns)), measured[train].ravel())[0]
    params = dict(gain=float(coefficients[0]), background=float(coefficients[1]),
                  bias_counts=float(coefficients[2]) if mode == "exposure" else 0.)
    return design @ coefficients, params
