"""Offline tests for the delta explorer — no hardware required.

The tool's whole reason to exist is that "final vs first" misjudges a random
walk, so the critical tests here feed it traces whose behaviour is known *by
construction*:

* a genuine descent must be accepted;
* a pure random walk must be rejected, **including** one engineered to end on a
  low sample so that first-vs-last looks like a large win;
* a trace with almost no samples must not be crowned.

That last family is the regression that matters: the bench produced a trace
reading ``0.529 -> 0.400 -> 0.534 -> 0.402`` (+24 % first-vs-last, zero descent).
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.delta_explorer import (
    FRAC_DECIDING_BAND,
    LATE_GAIN_FLOOR,
    MIN_SAMPLES,
    TraceStats,
    analyse_trace,
    explore_delta,
    verdict_for,
)

# ---------------------------------------------------------------------------
# Synthetic traces with known behaviour
# ---------------------------------------------------------------------------


def _descending(n: int = 120, start: float = 0.5, floor: float = 0.1) -> list[float]:
    """A genuine noisy descent: values go DOWN.

    This is an improving trace only for a **loss** (``lower_is_better=True``),
    e.g. ``1 - Pearson``. For a score it is the opposite.
    """
    t = np.linspace(0.0, 1.0, n)
    noise = np.random.default_rng(0).normal(0, 0.004, n)
    return list(start - (start - floor) * t + noise)


def _ascending(n: int = 120, start: float = 0.1, top: float = 0.5) -> list[float]:
    """A genuine noisy ascent: values go UP — improving for a **score**."""
    t = np.linspace(0.0, 1.0, n)
    noise = np.random.default_rng(0).normal(0, 0.004, n)
    return list(start + (top - start) * t + noise)


def _random_walk(n: int = 120, seed: int = 1) -> list[float]:
    return list(np.random.default_rng(seed).normal(0.5, 0.05, n))


def _endpoint_luck(n: int = 200, seed: int = 2) -> list[float]:
    """Random walk whose FINAL sample is far below its own mean.

    Reproduces the bench case: first-vs-last reads a big improvement while the
    trace is a coin flip.
    """
    rng = np.random.default_rng(seed)
    walk = rng.normal(0.5, 0.05, n)
    walk[0] = 0.53
    walk[1] = 0.40      # early dip
    walk[-1] = 0.40     # ends low -> naive metric says "+24 %"
    return list(walk)


# ---------------------------------------------------------------------------
# analyse_trace
# ---------------------------------------------------------------------------


class TestAnalyseTrace:
    def test_ascending_score_is_converged(self) -> None:
        """A score that rises consistently is a converged run."""
        stats = analyse_trace(_ascending(), delta=0.1)

        assert stats.converged
        assert stats.verdict() == "converged"
        assert stats.frac_decreasing > 0.5 + FRAC_DECIDING_BAND
        assert stats.late_gain_pct >= LATE_GAIN_FLOOR

    def test_descending_loss_is_converged(self) -> None:
        """Pearson is a loss: the same descent is improving."""
        stats = analyse_trace(_descending(), delta=0.1, lower_is_better=True)

        assert stats.converged
        assert stats.verdict() == "converged"
        assert stats.frac_decreasing > 0.5 + FRAC_DECIDING_BAND

    def test_descending_score_is_not_converged(self) -> None:
        """Polarity is not cosmetic: the same numbers must read differently."""
        stats = analyse_trace(_descending(), delta=0.1, lower_is_better=False)

        assert not stats.converged
        assert stats.late_gain_pct < 0

    def test_random_walk_is_rejected(self) -> None:
        stats = analyse_trace(_random_walk(), delta=0.1)

        assert not stats.converged
        assert stats.random_walk
        assert stats.verdict() == "random-walk"

    def test_endpoint_luck_is_rejected(self) -> None:
        """The regression: naive first-vs-last says +24 %, verdict must not."""
        trace = _endpoint_luck()
        naive = (trace[0] - trace[-1]) / abs(trace[0]) * 100

        stats = analyse_trace(trace, delta=0.02)

        assert naive > 20.0, "fixture must look like a big win naively"
        assert not stats.converged, "must NOT be crowned despite the big naive gain"
        assert stats.verdict() == "random-walk"
        # The robust statistic is what exposes it.
        assert abs(stats.late_gain_pct) < LATE_GAIN_FLOOR

    def test_rising_is_converged_when_lower_is_better(self) -> None:
        """Pearson is a loss, so a decreasing trace is the improving one."""
        stats = analyse_trace(
            _descending(start=0.9, floor=0.2), delta=0.1, lower_is_better=True
        )

        assert stats.converged
        assert stats.late_gain_pct > 0

    def test_polarity_is_symmetric(self) -> None:
        """Same numbers, opposite polarity, opposite verdict.

        A descending trace converges only when the objective is a loss.
        """
        as_loss = analyse_trace(_descending(), lower_is_better=True)
        as_score = analyse_trace(_descending(), lower_is_better=False)

        assert as_loss.converged
        assert not as_score.converged
        # Improving direction flips sign between the two readings.
        assert as_loss.late_gain_pct > 0
        assert as_score.late_gain_pct < 0

    def test_too_few_samples_not_converged(self) -> None:
        stats = analyse_trace([0.5, 0.4, 0.3], delta=0.1)

        assert stats.n_samples < MIN_SAMPLES
        assert not stats.converged
        assert stats.verdict() == "too-few-samples"

    def test_empty_trace(self) -> None:
        stats = analyse_trace([], delta=0.1)

        assert stats.n_samples == 0
        assert stats.verdict() == "too-few-samples"

    def test_noisy_but_rising_verdict(self) -> None:
        """Improving overall but with a coin-flip step fraction."""
        rng = np.random.default_rng(3)
        t = np.linspace(0, 1, 120)
        values = list(0.6 - 0.2 * t + rng.normal(0, 0.09, 120))
        stats = analyse_trace(values, delta=0.05)

        assert stats.random_walk or stats.late_gain_pct >= LATE_GAIN_FLOOR
        assert not stats.converged

    def test_flat_trace_is_random_walk(self) -> None:
        stats = analyse_trace([0.5] * 60, delta=0.1)

        assert stats.frac_decreasing == 0.0
        assert not stats.converged

    def test_span_reports_actual_movement(self) -> None:
        wide = analyse_trace(_ascending(start=0.1, top=0.9), delta=0.1)
        narrow = analyse_trace(_ascending(start=0.4, top=0.45), delta=0.1)

        assert wide.span_pct > narrow.span_pct

    def test_guard_fraction_recorded(self) -> None:
        stats = analyse_trace(
            _descending(), delta=0.1, guard_rows=40, total_rows=200,
            lower_is_better=True,
        )

        assert stats.guard_fraction == pytest.approx(0.2)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            analyse_trace([[1.0, 2.0]], delta=0.1)


# ---------------------------------------------------------------------------
# explore_delta
# ---------------------------------------------------------------------------


def _rows_from(values, key="pearson", guard_every: int = 0, gate: str | None = None):
    """Build recorder-like rows, optionally with guard rejections and gates."""
    rows = []
    for i, v in enumerate(values):
        r = {"J": float(v), key: float(v), "_epoch": i}
        if guard_every and i % guard_every == 0 and i:
            r["J"] = 1e3  # guard sentinel
        if gate is not None:
            r["_gate"] = gate
        rows.append(r)
    return rows


class TestExploreDelta:
    def test_picks_the_converging_candidate(self) -> None:
        plan = {
            0.01: _random_walk(seed=1),
            0.05: _descending(),
            0.10: _random_walk(seed=4),
        }
        scan = explore_delta(lambda d: _rows_from(plan[d]), deltas=plan.keys())

        assert scan.recommended == 0.05

    def test_returns_none_when_nothing_converges(self) -> None:
        plan = {0.01: _random_walk(seed=1), 0.1: _random_walk(seed=2)}
        scan = explore_delta(lambda d: _rows_from(plan[d]), deltas=plan.keys())

        assert scan.recommended is None
        assert scan.converged == []

    def test_never_crowns_an_endpoint_luck_run(self) -> None:
        """End-to-end version of the regression that caused the retraction."""
        plan = {0.02: _endpoint_luck(), 0.05: _endpoint_luck(seed=9)}
        scan = explore_delta(lambda d: _rows_from(plan[d]), deltas=plan.keys())

        assert scan.recommended is None

    def test_accepts_dict_payload(self) -> None:
        """save_recorder_debug_artifacts writes {epoch: row}, not a Recorder."""
        rows = _rows_from(_descending())
        scan = explore_delta(lambda d: dict(enumerate(rows)), deltas=[0.05])

        assert scan.recommended == 0.05

    def test_accepts_list_payload(self) -> None:
        scan = explore_delta(lambda d: _rows_from(_descending()), deltas=[0.05])

        assert scan.recommended == 0.05

    def test_fake_recorder_object(self) -> None:
        class Rec:
            def __init__(self, rows):
                self.history = rows

        scan = explore_delta(lambda d: Rec(_rows_from(_descending())), deltas=[0.05])

        assert scan.recommended == 0.05

    def test_guard_rows_excluded_from_scoring(self) -> None:
        """J=1e3 rows must not destroy the statistics."""
        clean = explore_delta(lambda d: _rows_from(_descending()), deltas=[0.05])
        dirty = explore_delta(
            lambda d: _rows_from(_descending(), guard_every=3), deltas=[0.05]
        )

        assert clean.recommended == 0.05
        assert dirty.stats[0].guard_fraction > 0.2

    def test_accepted_count_from_gate_column(self) -> None:
        scan = explore_delta(
            lambda d: _rows_from(_descending(), gate="applied"), deltas=[0.05]
        )

        assert scan.stats[0].accepted == 120
        assert scan.stats[0].total_rows == 120

    def test_failed_candidate_does_not_abort_scan(self) -> None:
        def run_one(d: float):
            if d == 0.01:
                raise RuntimeError("bench disconnected")
            return _rows_from(_descending())

        scan = explore_delta(run_one, deltas=[0.01, 0.05])

        assert scan.recommended == 0.05
        assert scan.stats[0].verdict() == "too-few-samples"

    def test_polarity_inferred_from_key(self) -> None:
        """The SAME descending numbers must be good for a loss, bad for a score.

        ``pearson`` is a loss (lower is better) so a descent converges;
        ``shape`` is a score (higher is better) so the identical trace does not.
        """
        desc = _descending(start=0.9, floor=0.2)
        pearson = explore_delta(
            lambda d: _rows_from(desc, key="pearson"), deltas=[0.05]
        )
        shape = explore_delta(lambda d: _rows_from(desc, key="shape"), deltas=[0.05])

        assert pearson.recommended == 0.05, "descent must converge for a loss"
        assert shape.recommended is None, "descent must NOT converge for a score"

    def test_explicit_polarity_overrides_inference(self) -> None:
        """Forcing loss polarity on a score key flips the verdict."""
        desc = _descending(start=0.9, floor=0.2)
        forced = explore_delta(
            lambda d: _rows_from(desc, key="shape"),
            deltas=[0.05],
            lower_is_better=True,
        )
        inferred = explore_delta(
            lambda d: _rows_from(desc, key="shape"), deltas=[0.05]
        )

        assert forced.recommended == 0.05
        assert inferred.recommended is None

    def test_progress_callback_fires_per_delta(self) -> None:
        seen: list[tuple[float, int, int]] = []
        explore_delta(
            lambda d: _rows_from(_random_walk()),
            deltas=[0.01, 0.05, 0.1],
            progress=lambda d, i, n: seen.append((d, i, n)),
        )

        assert [s[1] for s in seen] == [1, 2, 3]
        assert all(s[2] == 3 for s in seen)

    def test_rejects_empty_deltas(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            explore_delta(lambda d: [], deltas=[])

    def test_table_is_markdown(self) -> None:
        scan = explore_delta(lambda d: _rows_from(_descending()), deltas=[0.05])
        table = scan.table()

        assert table.startswith("| delta |")
        assert "converged" in table
        assert "0.05" in table

    def test_bad_payload_type_raises(self) -> None:
        with pytest.raises(TypeError, match="run_one must return"):
            explore_delta(lambda d: 42, deltas=[0.05])


class TestTraceStatsAndVerdict:
    def test_verdict_for_matches_method(self) -> None:
        stats = analyse_trace(_ascending(), delta=0.05)

        assert verdict_for(stats) == stats.verdict() == "converged"

    def test_dataclass_defaults(self) -> None:
        s = TraceStats(
            delta=0.1, n_samples=100, frac_decreasing=0.8, late_gain_pct=40.0,
            first=1.0, final=0.5, best=0.5, span_pct=50.0,
        )

        assert s.converged
        assert s.accepted is None and s.guard_fraction == 0.0