"""Offline tests for the per-repeat light-stability helpers (no hardware).

``scripts/objective_rep_logging.py`` is loaded by *file path* (``importlib``)
because ``scripts/`` is not an importable package. The tests cover the drift
split (no drift / one outlier / missing-zero / boundary), the nan-safe column
median and the kept-row summary.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "scripts" / "objective_rep_logging.py"


@pytest.fixture(scope="module")
def rep_log():
    """Load objective_rep_logging.py directly by file path."""
    spec = importlib.util.spec_from_file_location("objective_rep_logging", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(*peaks: float | None) -> list[dict]:
    return [{"slug": "shape", "rep": i + 1, "frame_peak": p} for i, p in enumerate(peaks)]


class TestFlagDriftReps:
    def test_no_drift_keeps_all(self, rep_log) -> None:
        rows = _rows(180, 178, 182, 179)

        kept, dropped = rep_log.flag_drift_reps(rows)

        assert len(kept) == 4
        assert dropped == []

    def test_one_clear_outlier_dropped(self, rep_log) -> None:
        rows = _rows(180, 182, 178, 181, 60)

        kept, dropped = rep_log.flag_drift_reps(rows)

        assert [r["frame_peak"] for r in kept] == [180, 182, 178, 181]
        assert [r["frame_peak"] for r in dropped] == [60]

    def test_missing_and_zero_kept_and_marked(self, rep_log) -> None:
        rows = _rows(180, None, 0, 182, 178)

        kept, dropped = rep_log.flag_drift_reps(rows)

        assert len(kept) == 5
        assert dropped == []
        marked = [r["frame_peak"] for r in kept if r.get("drift_unknown") is True]
        assert marked == [None, 0]

    def test_boundary_is_strict_greater_than(self, rep_log) -> None:
        # median 100: 125 deviates by exactly 0.25 (kept), 126 by 0.26 (dropped).
        rows = _rows(100, 100, 100, 125, 126)

        kept, dropped = rep_log.flag_drift_reps(rows)

        assert 125 in [r["frame_peak"] for r in kept]
        assert [r["frame_peak"] for r in dropped] == [126]

    def test_all_unknown_when_no_usable_peak(self, rep_log) -> None:
        rows = _rows(0, 0, None)

        kept, dropped = rep_log.flag_drift_reps(rows)

        assert len(kept) == 3
        assert dropped == []
        assert all(r["drift_unknown"] is True for r in kept)

    def test_empty_input(self, rep_log) -> None:
        assert rep_log.flag_drift_reps([]) == ([], [])


class TestMedianOver:
    def test_nan_safe(self, rep_log) -> None:
        rows = [{"frame_peak": 1.0}, {"frame_peak": float("nan")}, {"frame_peak": 3.0}]

        assert rep_log.median_over(rows, "frame_peak") == pytest.approx(2.0)

    def test_missing_column_is_nan(self, rep_log) -> None:
        assert math.isnan(rep_log.median_over([{"frame_peak": 1.0}], "absent"))

    def test_empty_input_is_nan(self, rep_log) -> None:
        assert math.isnan(rep_log.median_over([], "frame_peak"))


class TestSummariseLogged:
    def test_reduces_over_rows(self, rep_log) -> None:
        rows = [{"frame_peak": 180}, {"frame_peak": 200}, {"frame_peak": 160}]

        out = rep_log.summarise_logged(rows, ("frame_peak",))

        assert out["frame_peak"]["median"] == pytest.approx(180.0)
        assert out["frame_peak"]["min"] == pytest.approx(160.0)
        assert out["frame_peak"]["max"] == pytest.approx(200.0)
        assert out["frame_peak"]["n"] == 3.0

    def test_missing_values_reduce_n(self, rep_log) -> None:
        rows = [{"frame_peak": 180}, {"frame_peak": None}, {"frame_peak": float("nan")}]

        out = rep_log.summarise_logged(rows, ("frame_peak",))

        assert out["frame_peak"]["median"] == pytest.approx(180.0)
        assert out["frame_peak"]["n"] == 1.0

    def test_empty_input(self, rep_log) -> None:
        out = rep_log.summarise_logged([], ("frame_peak",))

        assert out["frame_peak"]["n"] == 0.0
        assert math.isnan(out["frame_peak"]["median"])
