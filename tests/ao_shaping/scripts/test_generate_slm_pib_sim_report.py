"""Regression tests for ``scripts/generate_slm_pib_sim_report.py``.

Motivation (real defect found 2026-10-01): the generated report labelled the
**last epoch's** objective as "最终 J" and described the ``shape`` objective as
"最小化" (minimize). Both are wrong:

* ``slm_zernike_pib.py`` sets ``objective_mode = "max"`` for ``shape`` (and
  ``pib``/``avg_radiu``/``roi_pib``/``rms_pib``) — so higher J is better.
* on exit the optimizer leaves the SLM at the **best** phase found, not the last
  one, so the physically meaningful "final" state is the best objective. On the
  2026-10-01 sim run the best was ``-1.8600`` while the last epoch was
  ``-1.9315`` — the report headline understated the result by 0.07 *and*
  asserted the wrong optimisation direction.

These tests pin the corrected contract: the report must state the direction,
and must show initial / best / last separately so the best-vs-sustained gap is
visible rather than hidden behind one number.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_generator():
    """Import the generator script by path (it is not an importable package)."""
    script = ROOT / "scripts" / "generate_slm_pib_sim_report.py"
    spec = importlib.util.spec_from_file_location("_slm_pib_sim_report", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Objectives the optimizer ascends (slm_zernike_pib.py:698-701).
MAXIMIZE_OBJECTIVES = ("pib", "avg_radiu", "shape", "roi_pib", "rms_pib")


def _write_run(tmp_path: Path, objective: str, js: list[float], ps: list[float]) -> Path:
    """Write a minimal recorder PKL + JSON sidecar; return the artifact dir."""
    art = tmp_path / f"slm_pib_{objective}_20260101_000000"
    art.mkdir(parents=True)
    data = {
        i: {
            "J": float(j),
            "_p%": float(p),
            "_epoch": float(i),
            "_c": np.zeros(15),
            "_img": np.zeros((8, 8)),
        }
        for i, (j, p) in enumerate(zip(js, ps))
    }
    (art / f"run_{objective}.pkl").write_bytes(pickle.dumps(data))
    (art / "run.json").write_text(
        '{"objective": "%s", "epochs": %d}' % (objective, len(js))
    )
    return art


class TestObjectiveDirection:
    """The report must not misstate the optimisation direction."""

    def test_shape_objective_is_reported_as_maximize(self, tmp_path):
        """`shape` ascends (objective_mode="max"), so the report must say so.

        Guards the 2026-10-01 defect: the report called `shape` "最小化".
        """
        gen = _load_generator()
        art = _write_run(tmp_path, "shape", [-1.88, -1.86, -1.93], [0.24, 0.26, 0.28])
        run = gen.load_run(art)
        out = tmp_path / "report.md"
        gen.build_markdown([run], ["run0"], {}, {}, out)
        text = out.read_text(encoding="utf-8")

        assert "最小化" not in text, "shape ascends; calling it 最小化 is false"
        assert "max" in text.lower() or "越大越好" in text or "越大" in text

    @pytest.mark.parametrize("objective", MAXIMIZE_OBJECTIVES)
    def test_every_maximize_objective_is_flagged_maximized(
        self, tmp_path, objective
    ):
        """No ascending objective may be described as a minimisation."""
        gen = _load_generator()
        art = _write_run(tmp_path, objective, [0.1, 0.5, 0.3], [0.1, 0.2, 0.15])
        run = gen.load_run(art)
        out = tmp_path / f"{objective}.md"
        gen.build_markdown([run], ["run0"], {}, {}, out)
        text = out.read_text(encoding="utf-8")
        assert "最小化" not in text, f"{objective} ascends but is called 最小化"


class TestBestVersusSustained:
    """Initial / best / last must be reported separately, never conflated."""

    def test_best_and_last_are_both_reported(self, tmp_path):
        """The best objective must appear, distinct from the last epoch's."""
        gen = _load_generator()
        # ascending: -1.88 (init) -> -1.86 (best) -> -1.93 (last, a regression)
        art = _write_run(tmp_path, "shape", [-1.88, -1.86, -1.93], [0.24, 0.26, 0.28])
        run = gen.load_run(art)
        out = tmp_path / "report.md"
        gen.build_markdown([run], ["run0"], {}, {}, out)
        text = out.read_text(encoding="utf-8")

        assert "-1.8600" in text, "the best objective (-1.86) must be reported"
        assert "-1.9300" in text, "the last epoch (-1.93) must be shown too"
        assert "-1.8800" in text, "the initial objective must be reported"

    def test_best_is_the_extreme_in_the_ascending_direction(self, tmp_path):
        """`best` must be max(J) for an ascending objective, not the last row."""
        gen = _load_generator()
        art = _write_run(tmp_path, "shape", [-1.0, -0.5, -0.9], [0.1, 0.9, 0.3])
        run = gen.load_run(art)
        out = tmp_path / "report.md"
        gen.build_markdown([run], ["run0"], {}, {}, out)
        text = out.read_text(encoding="utf-8")

        # max(J) == -0.5 ; the last row is -0.9 and must not be called "best"
        assert "-0.5000" in text
        best_line = next(
            (ln for ln in text.splitlines() if "最佳" in ln and "J" in ln), ""
        )
        assert best_line, "a 最佳 (best) row is required"
        assert "-0.5000" in best_line, (
            f"best row must carry max(J)=-0.5, got: {best_line!r}"
        )

    def test_sustained_regression_is_not_reported_as_success(self, tmp_path):
        """When the last epoch is worse than the best, the gap must be visible.

        This is the "best vs sustained" discipline the bench reports already use
        (docs/slm_pib_bench): reporting only the best hides drift.
        """
        gen = _load_generator()
        # ascending, with a large last-epoch regression
        art = _write_run(tmp_path, "shape", [-1.0, -0.2, -5.0], [0.1, 0.9, 0.05])
        run = gen.load_run(art)
        out = tmp_path / "report.md"
        gen.build_markdown([run], ["run0"], {}, {}, out)
        text = out.read_text(encoding="utf-8")

        # a "最终/末轮" row must exist and must carry the LAST value, not the best
        last_line = next(
            (ln for ln in text.splitlines() if "末轮" in ln and "J" in ln), ""
        )
        assert last_line, "a 末轮 (last-epoch) row is required"
        assert "-5.0000" in last_line, (
            f"last-epoch row must carry the last value -5.0, got: {last_line!r}"
        )

    def test_monotonic_improvement_needs_no_gap_warning(self, tmp_path):
        """A clean monotonic run must not be flagged as a regression."""
        gen = _load_generator()
        art = _write_run(tmp_path, "shape", [-1.0, -0.6, -0.2], [0.1, 0.5, 0.9])
        run = gen.load_run(art)
        out = tmp_path / "report.md"
        gen.build_markdown([run], ["run0"], {}, {}, out)
        text = out.read_text(encoding="utf-8")

        last_line = next(
            (ln for ln in text.splitlines() if "末轮" in ln and "J" in ln), ""
        )
        assert "-0.2000" in last_line
        # the best equals the last epoch here, so no drift language is warranted
        assert "回退" not in text and "末轮劣于最佳" not in text
