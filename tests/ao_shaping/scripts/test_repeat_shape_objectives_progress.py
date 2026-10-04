"""Tests for the sweep progress line of ``scripts/repeat_shape_objectives.py``.

The hardware sweep is a ``variant x repeat`` matrix whose variants have wildly
different runtimes, so the only way an operator can tell how far the *whole*
sweep is from the log tail is a **global** run counter.  The per-variant
``rep 1/3`` alone cannot: at variant 3 rep 1/1 the tail looks identical whether
that is the 7th run of 9 or the 9th of 9.

These tests are deliberately offline -- the counter arithmetic and the label
format are pure, and the loop wiring is checked against ``main``'s source so a
future edit cannot quietly move the increment out of the loop.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from repeat_shape_objectives import _progress_label  # noqa: E402


def test_progress_label_reports_all_three_counters():
    label = _progress_label(
        run_idx=7,
        n_runs=45,
        variant_idx=3,
        n_variants=5,
        slug="pib_square",
        rep=2,
        n_reps=3,
    )
    assert "run 7/45" in label
    assert "variant 3/5 'pib_square'" in label
    assert "rep 2/3" in label


def test_progress_walk_over_matrix_is_monotonic_and_bounded():
    """Walk the same nested loop ``main`` uses; the labels must tile 1..N."""
    variants = ["a", "b"]
    n_reps = 3
    n_runs = len(variants) * n_reps

    seen: list[str] = []
    run_idx = 0
    for idx, slug in enumerate(variants, start=1):
        for rep in range(1, n_reps + 1):
            run_idx += 1
            seen.append(
                _progress_label(
                    run_idx, n_runs, idx, len(variants), slug, rep, n_reps
                )
            )

    assert len(seen) == n_runs
    for expected, label in enumerate(seen, start=1):
        assert f"run {expected}/{n_runs}" in label
    assert seen[0] == "=== run 1/6 | variant 1/2 'a' | rep 1/3 ==="
    assert seen[-1] == "=== run 6/6 | variant 2/2 'b' | rep 3/3 ==="


def test_main_derives_total_from_variants_and_increments_inside_the_loop():
    src = inspect.getsource(sys.modules["repeat_shape_objectives"].main)
    assert "n_runs = len(variants) * args.repeats" in src
    assert "for idx, (slug, label, kwargs) in enumerate(variants, start=1):" in src
    # The increment must precede the log call, otherwise the first run would
    # report 0/N and the last would report N-1/N.
    assert "run_idx += 1" in src, "global run counter is never incremented"
    assert "logger.info(" in src
    assert src.index("run_idx += 1") < src.index("_progress_label(")