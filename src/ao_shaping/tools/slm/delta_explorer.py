"""Explore the SPGD perturbation amplitude (``delta``) by measuring convergence.

**Pure analysis over recorded traces — no device is constructed here.** The
caller supplies a ``run_one(delta)`` callable that performs one optimisation and
returns its recorder, so the same judging logic works for the Daheng bench, the
simulation bench, or a synthetic trace in a unit test.

Why this exists (and why it does not use "final vs first")
-----------------------------------------------------------
Picking ``delta`` by *how much the objective improved from the first to the last
epoch* is **wrong**, and this module exists because that mistake was made and
nearly shipped. A random walk reads as a large win whenever it happens to stop on
a low sample. Bench-verified example (200 epochs, n_max=9, delta=0.02):

    0.529 -> 0.400 -> 0.522 -> 0.527 -> 0.533 -> 0.534 -> ... -> 0.456 -> 0.402

First-vs-last reads **+24 %**, yet the trace never descended — it dipped, fully
recovered, and stopped low. Across 51 recorded runs, **32** had a
decreasing-step fraction within 0.05 of 0.5, i.e. were coin flips whose
"improvement" was an artefact of where the run ended.

Two robust statistics are used instead:

``frac_decreasing``
    Fraction of steps that improve. ``0.5`` is a coin flip; a converging run
    sits clearly above it.
``late_gain``
    Improvement between the mean of the first third and the mean of the last
    third of the trace. Robust to where the run stopped.

Public symbols
--------------
- ``TraceStats``   — per-delta convergence statistics
- ``analyse_trace``— convergence metrics for one objective series
- ``explore_delta``— run one optimisation per candidate delta and rank them
- ``DeltaScanResult`` / ``verdict_for`` — dataclass + acceptance rule

The physical reasoning (why ``delta`` cannot fix this bench) is documented in
``scripts/explore_delta.py``; in short: the gradient signal scales with
``delta`` while the dominant noise — slow intensity drift between the two
consecutive samples ``J(+d)`` and ``J(-d)`` — does not. Bench-verified: sweeping
``delta`` over 1000x (5e-4 … 0.5) left the decreasing-step fraction at ~0.5
throughout, so no amplitude produces a real descent. Widening the gradient by
scaling ``delta`` therefore buys signal and drift in equal measure.

Typical use::

    def run_one(delta):
        return optimize_slm_zernike_pib(make_config(epochs=200, delta=delta))

    scan = explore_delta(run_one, deltas=(0.02, 0.05, 0.1))
    print(scan.table())
    print(scan.recommended)     # None when nothing actually converges
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: A trace is treated as a random walk (not converged) when the fraction of
#: improving steps is this close to 0.5.
FRAC_DECIDING_BAND = 0.05

#: Minimum number of scored samples before a verdict is meaningful. A run whose
#: accepted updates are this rare tells us nothing about convergence.
MIN_SAMPLES = 10

#: Below this the ``late_gain`` is indistinguishable from run-to-run scatter.
#: Calibrated on the bench where converged candidates scored >10 % and coin
#: flips landed within +-10 %.
LATE_GAIN_FLOOR = 10.0

#: Sentinel J used by the energy guard for an abandoned evaluation. Rows at or
#: beyond this magnitude are "rejected", not scores.
GUARD_SENTINEL = 100.0


@dataclass
class TraceStats:
    """Convergence statistics for one candidate ``delta``."""

    delta: float
    n_samples: int
    #: Fraction of steps that improve (0.5 == random walk).
    frac_decreasing: float
    #: Improvement (%) between first-third and last-third means.
    late_gain_pct: float
    #: First and last sample (reported for context only, NOT a verdict).
    first: float
    final: float
    best: float
    #: Objective range covered by the trace (a proxy for how much it moved).
    span_pct: float
    #: Fraction of rows rejected by the energy guard.
    guard_fraction: float = 0.0
    #: Rows whose update was actually adopted, when the recorder carries gates.
    accepted: int | None = None
    total_rows: int | None = None

    @property
    def converged(self) -> bool:
        """True only when the trace descends consistently *and* by a margin."""
        return (
            self.n_samples >= MIN_SAMPLES
            and self.frac_decreasing > 0.5 + FRAC_DECIDING_BAND
            and self.late_gain_pct >= LATE_GAIN_FLOOR
        )

    @property
    def random_walk(self) -> bool:
        return abs(self.frac_decreasing - 0.5) <= FRAC_DECIDING_BAND

    def verdict(self) -> str:
        if self.n_samples < MIN_SAMPLES:
            return "too-few-samples"
        if self.converged:
            return "converged"
        if self.random_walk:
            return "random-walk"
        if self.late_gain_pct >= LATE_GAIN_FLOOR:
            return "noisy-but-rising"
        return "not-converging"


@dataclass
class DeltaScanResult:
    """Outcome of :func:`explore_delta`."""

    stats: list[TraceStats] = field(default_factory=list)

    @property
    def converged(self) -> list[TraceStats]:
        return sorted(
            (s for s in self.stats if s.converged),
            key=lambda s: -s.late_gain_pct,
        )

    @property
    def recommended(self) -> float | None:
        """Best converging ``delta``, or ``None`` if nothing converged.

        Returning ``None`` is the point: it refuses to crown a random walk.
        """
        best = self.converged
        return best[0].delta if best else None

    def table(self) -> str:
        """Markdown table of every candidate."""
        head = (
            "| delta | verdict | dec | late % | span % | guard % | accepted |"
            " first | final |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|"
        lines = [head, sep]
        for s in sorted(self.stats, key=lambda s: s.delta):
            acc = "-" if s.accepted is None else f"{s.accepted}/{s.total_rows}"
            lines.append(
                f"| {s.delta:g} | {s.verdict()} | {s.frac_decreasing:.2f} | "
                f"{s.late_gain_pct:+.1f} | {s.span_pct:.1f} | "
                f"{100 * s.guard_fraction:.1f} | {acc} | "
                f"{s.first:.4f} | {s.final:.4f} |"
            )
        return "\n".join(lines)


def verdict_for(stats: TraceStats) -> str:
    """Standalone verdict for one :class:`TraceStats`."""
    return stats.verdict()


def _as_rows(result: Any) -> list[dict[str, Any]]:
    """Normalise a recorder / dict / list-of-dicts into a list of row dicts."""
    if result is None:
        return []
    if hasattr(result, "history"):  # ao_shaping Recorder
        return [dict(r) for r in result.history]
    if isinstance(result, dict):
        # {epoch: row} as written by save_recorder_debug_artifacts
        return [dict(v) for v in result.values() if isinstance(v, dict)]
    if isinstance(result, (list, tuple)):
        return [dict(r) for r in result]
    raise TypeError(
        "run_one must return a Recorder, a {epoch: row} dict or a list of dicts; "
        f"got {type(result).__name__}"
    )


def _pick_objective_key(
    rows: Sequence[dict[str, Any]], requested: str | None
) -> str | None:
    if requested is not None:
        return requested if requested in rows[0] else None
    for candidate in ("pearson", "shape", "roi_pib", "rms_pib", "pib", "rmse"):
        if candidate in rows[0]:
            return candidate
    return "J" if "J" in rows[0] else None


def analyse_trace(
    values: Sequence[float],
    *,
    delta: float = float("nan"),
    lower_is_better: bool | None = None,
    guard_rows: int = 0,
    total_rows: int | None = None,
    accepted: int | None = None,
) -> TraceStats:
    """Convergence statistics for one objective series.

    Args:
        values: Objective per epoch, in run order.
        delta: The perturbation amplitude this trace was produced with.
        lower_is_better: Polarity. Inferred from ``delta``'s companion when
            ``None`` via :func:`_infer_polarity` if the key name is known, else
            assumed higher-is-better.
        guard_rows: Rows discarded because the energy guard rejected them.
        total_rows: Rows recorded before guard filtering.
        accepted: Rows whose update was adopted (from the gate column).

    Returns:
        A populated :class:`TraceStats`.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        return TraceStats(
            delta=delta, n_samples=0, frac_decreasing=0.0, late_gain_pct=0.0,
            first=float("nan"), final=float("nan"), best=float("nan"),
            span_pct=0.0, guard_fraction=1.0 if total_rows else 0.0,
            accepted=accepted, total_rows=total_rows,
        )

    if lower_is_better is None:
        lower_is_better = False
    # Orient so that IMPROVING IS ALWAYS AN INCREASE, regardless of polarity.
    # For a loss, improvement is a decrease, so negate it; for a score the raw
    # values already increase on improvement. Getting this backwards silently
    # reports frac_decreasing ~0.15 for a clean descent.
    sign = -1.0 if lower_is_better else 1.0
    work = sign * arr

    steps = np.diff(work)
    frac_improving = float(np.mean(steps > 0)) if steps.size else 0.0

    third = max(arr.size // 3, 1)
    head = float(np.mean(work[:third]))
    tail = float(np.mean(work[-third:]))
    denom = abs(head) if abs(head) > 1e-12 else 1.0
    late = 100.0 * (tail - head) / denom

    # ``best`` is reported in the objective's own units, not the oriented ones.
    best = float(arr.min() if lower_is_better else arr.max())
    span = 100.0 * float(np.max(arr) - np.min(arr)) / denom

    return TraceStats(
        delta=delta,
        n_samples=int(arr.size),
        frac_decreasing=frac_improving,
        late_gain_pct=late,
        first=float(arr[0]),
        final=float(arr[-1]),
        best=best,
        span_pct=span,
        guard_fraction=(guard_rows / total_rows) if total_rows else 0.0,
        accepted=accepted,
        total_rows=total_rows,
    )


def _infer_polarity(key: str | None) -> bool:
    """Pearson is a loss; every other objective in this project is a score."""
    return (key or "").lower() == "pearson"


def explore_delta(
    run_one: Callable[[float], Any],
    deltas: Sequence[float],
    *,
    objective_key: str | None = None,
    lower_is_better: bool | None = None,
    progress: Callable[[float, int, int], None] | None = None,
) -> DeltaScanResult:
    """Run one optimisation per candidate ``delta`` and rank by convergence.

    Args:
        run_one: ``delta -> recorder``. Performs one optimisation and returns
            its recorder (or a ``{epoch: row}`` dict, or a list of rows).
            Raising is treated as a failed candidate and recorded as such.
        deltas: Candidate perturbation amplitudes to try.
        objective_key: Objective column to score (default: auto-detect,
            preferring ``pearson``/``shape``/... then ``J``).
        lower_is_better: Override the polarity inference.
        progress: Optional ``(delta, index, total)`` callback.

    Returns:
        A :class:`DeltaScanResult`. Check :attr:`DeltaScanResult.recommended` —
        it is ``None`` when no candidate actually converged.
    """
    deltas = [float(d) for d in deltas]
    if not deltas:
        raise ValueError("deltas must not be empty")

    out: list[TraceStats] = []
    for i, delta in enumerate(deltas, start=1):
        if progress is not None:
            progress(delta, i, len(deltas))
        try:
            raw = run_one(delta)
        except Exception:  # a failed candidate must not abort the scan
            out.append(
                TraceStats(
                    delta=delta, n_samples=0, frac_decreasing=0.0,
                    late_gain_pct=0.0, first=float("nan"), final=float("nan"),
                    best=float("nan"), span_pct=0.0, guard_fraction=1.0,
                )
            )
            continue

        rows = _as_rows(raw)
        if not rows:
            out.append(
                TraceStats(
                    delta=delta, n_samples=0, frac_decreasing=0.0,
                    late_gain_pct=0.0, first=float("nan"), final=float("nan"),
                    best=float("nan"), span_pct=0.0, guard_fraction=1.0,
                )
            )
            continue

        key = _pick_objective_key(rows, objective_key)
        if key is None:
            continue

        guarded = [r for r in rows if abs(float(r.get("J", 0.0))) > GUARD_SENTINEL]
        scored = [r for r in rows if abs(float(r.get("J", 0.0))) <= GUARD_SENTINEL]
        accepted = None
        if any("_gate" in r for r in rows):
            accepted = sum(1 for r in rows if str(r.get("_gate")) == "applied")

        vals = [float(r[key]) for r in scored if key in r]
        if not vals:
            continue

        out.append(
            analyse_trace(
                vals,
                delta=delta,
                lower_is_better=(
                    _infer_polarity(key)
                    if lower_is_better is None
                    else lower_is_better
                ),
                guard_rows=len(guarded),
                total_rows=len(rows),
                accepted=accepted,
            )
        )

    return DeltaScanResult(stats=out)