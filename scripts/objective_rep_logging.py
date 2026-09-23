"""Per-repeat LIGHT-STABILITY logging for the shaping-objective repeat study.

The bench laser drifts. At a fixed 2 ms exposure the 0-order frame peak was
measured at 180 and then at 121 within a single minute, i.e. the illumination can
fall by roughly a third while the SLM phase is unchanged. That drift - not the
objective - is the dominant residual variance between repeats: a round taken
during a bright or dim spell is not comparable with the others, so a plain median
over all rounds silently mixes incomparable illumination levels.

These helpers make the drift explicit and auditable:

* :func:`flag_drift_reps` splits the per-repeat rows into ``(kept, dropped)``
  around the **median** ``frame_peak`` - a round whose peak deviates from that
  median by more than ``max_rel_dev`` (default 25 %) is dropped as a drift
  outlier. Rounds with no usable peak (missing / zero / non-finite) cannot be
  judged: they are kept but tagged ``drift_unknown`` so the caller can see they
  were not evidence of stability.
* :func:`median_over` is a nan-safe column median.
* :func:`summarise_logged` reduces a set of columns over the KEPT rows.

Pure Python + NumPy: no hardware, no camera, no SLM.
"""

from __future__ import annotations

import math

import numpy as np


def _as_positive(value: object) -> float | None:
    """Return ``value`` as a finite positive float, else ``None``.

    ``None`` is returned for missing / non-numeric / zero / negative / non-finite
    input - exactly the cases that carry no light-stability information.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def _finite(value: object) -> float | None:
    """Return ``value`` as a finite float, else ``None`` (missing / nan / inf)."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def median_over(rows: list[dict], key: str) -> float:
    """Nan-safe median of the ``key`` column over ``rows``.

    Non-finite / missing / non-numeric values are ignored; an empty or entirely
    invalid column yields ``nan``. Zero is a legitimate value here - use
    :func:`flag_drift_reps` for the "missing / zero = unknown" semantics.
    """
    vals = [v for v in (_finite(r.get(key)) for r in rows) if v is not None]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def flag_drift_reps(
    rows: list[dict],
    key: str = "frame_peak",
    max_rel_dev: float = 0.25,
) -> tuple[list[dict], list[dict]]:
    """Split ``rows`` into ``(kept, dropped)`` drift groups around the median.

    A row is dropped when ``abs(v - median) / median > max_rel_dev``. Rows whose
    ``key`` is missing, zero or non-finite cannot be judged: they are **kept** and
    tagged ``row["drift_unknown"] = True``. With no usable value at all (or a
    non-positive median) every row is kept and tagged unknown.

    Rows are tagged in place; the two returned lists reference the same dicts.
    """
    if not rows:
        return [], []
    known = [_as_positive(r.get(key)) for r in rows]
    vals = [v for v in known if v is not None]
    median = (
        float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")
    )
    if not math.isfinite(median) or median <= 0.0:
        for row in rows:
            row["drift_unknown"] = True
        return list(rows), []
    kept: list[dict] = []
    dropped: list[dict] = []
    for row, value in zip(rows, known):
        if value is None:
            row["drift_unknown"] = True
            kept.append(row)
            continue
        if abs(value - median) / median > max_rel_dev:
            dropped.append(row)
        else:
            kept.append(row)
    return kept, dropped


def summarise_logged(
    rows: list[dict], keys: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Per-key ``{median, min, max, n}`` over ``rows`` (nan-safe).

    ``n`` counts the finite values actually used, so a column with missing entries
    reports a smaller ``n`` than ``len(rows)``. An empty column yields ``nan`` for
    median / min / max and ``n = 0``.
    """
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [v for v in (_finite(r.get(key)) for r in rows) if v is not None]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = {
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": float(len(vals)),
            }
        else:
            out[key] = {
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n": 0.0,
            }
    return out
