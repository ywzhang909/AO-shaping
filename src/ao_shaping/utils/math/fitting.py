"""Least-squares fitting helpers.

Pure numpy; no device, no panel geometry, no image I/O. Moved verbatim from
``tools/kernels/bench_kernels.py`` (R-49 / #65) so the fitting math has a
single leaf home that any probe, runner or script can import without dragging
in the bench kernel surface.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def fit_linear_slope(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)``; ``(nan, nan)`` if degenerate.

    Degenerate means: fewer than two points, mismatched lengths, non-finite
    ``y``, or zero variance in ``x``. Inputs are coerced through ``list`` so a
    generator yields its items instead of freezing into the 0-d object array
    ``np.asarray`` would build from it (which would silently read as a
    one-point, hence degenerate, sample).
    """
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    if xs.size < 2 or xs.size != ys.size or not np.isfinite(ys).all():
        return float("nan"), float("nan")
    var = float(np.sum((xs - xs.mean()) ** 2))
    if var <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / var)
    return slope, float(ys.mean() - slope * xs.mean())
