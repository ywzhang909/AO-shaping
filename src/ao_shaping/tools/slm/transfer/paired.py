"""Fixed-exposure paired decisions with an explicit measured acceptance metric."""
from __future__ import annotations

import math


def paired_decision(a1, b1, b2, a2, *, metric="loss",
                    rel_tol: float = 0.0, abs_tol: float = 0.0):
    if metric not in {"loss", "cv"}:
        raise ValueError("acceptance metric must be loss or cv")
    values = [float(row["metrics"]["cv"] if metric == "cv" else row["loss"])
              for row in (a1, b1, b2, a2)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite paired metric")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")
    baseline = .5*(values[0]+values[3])
    candidate = .5*(values[1]+values[2])
    drift = abs(values[3]-values[0])
    tolerance = max(abs(baseline) * float(rel_tol), float(abs_tol))
    return dict(accept_metric=metric, baseline=baseline, candidate=candidate,
                drift=drift, tolerance=tolerance,
                accepted=candidate < baseline - drift - tolerance)
