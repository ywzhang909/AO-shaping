"""Least-squares fitting helpers.

Pure numpy; no device, no panel geometry, no image I/O. Moved verbatim from
``tools/kernels/bench_kernels.py`` (R-49 / #65) so the fitting math has a
single leaf home that any probe, runner or script can import without dragging
in the bench kernel surface.
"""

from __future__ import annotations

import numpy as np


def fit_linear_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)``; ``(nan, nan)`` if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or not np.isfinite(y).all():
        return float("nan"), float("nan")
    var = float(np.sum((x - x.mean()) ** 2))
    if var <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / var)
    return slope, float(y.mean() - slope * x.mean())
