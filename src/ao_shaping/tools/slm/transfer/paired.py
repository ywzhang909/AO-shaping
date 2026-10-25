"""Fixed-exposure paired decisions with an explicit measured acceptance metric."""
from __future__ import annotations

import math


def paired_decision(a1, b1, b2, a2, *, metric="loss"):
    if metric not in {"loss", "cv"}:
        raise ValueError("acceptance metric must be loss or cv")
    values = [float(row["metrics"]["cv"] if metric == "cv" else row["loss"])
              for row in (a1, b1, b2, a2)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite paired metric")
    baseline = .5*(values[0]+values[3])
    candidate = .5*(values[1]+values[2])
    drift = abs(values[3]-values[0])
    return dict(accept_metric=metric, baseline=baseline, candidate=candidate,
                drift=drift, accepted=candidate < baseline-drift)
