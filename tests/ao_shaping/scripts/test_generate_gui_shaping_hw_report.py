"""Offline test for the GUI shaping report generator.

Guards the two things that silently corrupt this report: a run directory
whose CSV/replay pairing does not match the expected ``flat``/``GS``/``best``
triples, and a ``best`` metric read from the wrong stage. The generator is
pure numpy + matplotlib over saved artefacts, so it runs without hardware.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "generate_gui_shaping_hw_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gui_shaping_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    if not SCRIPT.is_file():
        pytest.skip("generator script not present")
    return _load_module()


class _Run:
    """Minimal stand-in for the generator's Run dataclass."""

    def __init__(self, history: list[dict]) -> None:
        self.ts = "t"
        self.label = "L"
        self.history = history
        self.replay: list[dict] = []
        self.best = None
        self.flat = None


class TestBestRowSelection:
    def test_best_metric_comes_from_the_record_row(self, gen) -> None:
        """Per-column max would mix stages: flat has the largest CV while the
        best row has the smallest, so a max-based read reports a combination no
        measured frame produced."""
        run = _Run(
            [
                {"stage": "flat", "score": "0.55", "cv": "3.77", "pib": "0.89"},
                {"stage": "gs", "score": "0.71", "cv": "0.71", "pib": "0.82"},
                {"stage": "refine", "score": "0.74", "cv": "0.67", "pib": "0.83", "epoch": "7"},
            ]
        )
        assert gen._best_row(run)["stage"] == "refine"
        assert gen._best_metric(run, "cv") == pytest.approx(0.67)
        assert gen._best_metric(run, "pib") == pytest.approx(0.83)

    def test_tie_keeps_the_earlier_stage(self, gen) -> None:
        """Conservative on purpose: when SPGD only ties GS, the GS row stays
        ``best`` so a tie is never credited to the refinement stage."""
        run = _Run(
            [
                {"stage": "flat", "score": "0.55", "cv": "3.77"},
                {"stage": "gs", "score": "0.71", "cv": "0.70"},
                {"stage": "refine", "score": "0.71", "cv": "0.70", "epoch": "3"},
            ]
        )
        assert gen._best_row(run)["stage"] == "gs"
        assert gen._best_metric(run, "cv") == pytest.approx(0.70)

    def test_missing_metric_returns_none(self, gen) -> None:
        run = _Run([{"stage": "flat", "score": "0.5"}])
        assert gen._best_metric(run, "cv") is None


class TestArtefactPairing:
    def test_replay_needs_all_three_phases(self, gen) -> None:
        rows = [
            {"phase": "flat", "score": 0.53, "cv": 3.7, "pib": 0.86},
            {"phase": "gs", "score": 0.77, "cv": 0.79, "pib": 0.98},
        ]
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["phase"], []).append(row)
        assert not {"flat", "gs", "best"} <= set(grouped)
        grouped["best"] = rows[:1]
        assert {"flat", "gs", "best"} <= set(grouped)

    def test_session_keys_are_full_run_directory_names(self, gen) -> None:
        """Regression: keys were once short timestamps, so every lookup missed
        the ``data/debug/slm_gs_refine/<ts>/`` directories."""
        for ts in gen.SESSIONS:
            assert ts.startswith("20"), ts
            assert len(ts) == 15, ts  # YYYYMMDD_HHMMSS


class TestSummarise:
    def test_summary_json_records_both_calibrations_and_replay(self, gen, tmp_path) -> None:
        """End-to-end on synthetic artefacts: the summary must carry the
        exposure ladder (both preflights) and per-session in-run + replay
        aggregates, with best-row metrics taken from one row."""
        gen.OUT = tmp_path  # type: ignore[misc]
        run = gen.Run(
            ts="20261014_000000",
            label="T 00:00 test",
            history=[
                {"stage": "flat", "score": "0.55", "cv": "3.77", "pib": "0.89"},
                {"stage": "gs", "score": "0.71", "cv": "0.71", "pib": "0.82"},
                {"stage": "refine", "score": "0.74", "cv": "0.67", "pib": "0.83", "epoch": "7"},
            ],
            replay=[
                {"phase": "flat", "score": 0.530, "cv": 3.73, "pib": 0.86},
                {"phase": "flat", "score": 0.534, "cv": 3.73, "pib": 0.86},
                {"phase": "gs", "score": 0.769, "cv": 0.798, "pib": 0.983},
                {"phase": "best", "score": 0.770, "cv": 0.797, "pib": 0.984},
            ],
            best=np.zeros((4, 4)),
            flat=np.ones((4, 4)),
        )
        summary = gen.summarise([run])
        payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

        assert payload["exposure_ladder"]["saturation_guard"] == 250
        ladder = {row["exposure_ms"]: row for row in payload["exposure_ladder"]["rows"]}
        assert ladder[0.05]["peak_1003"] == 146.7
        assert ladder[0.05]["peak_1429"] == 125.7
        assert ladder[0.4]["peak_1429"] is None  # second ladder stopped at 0.2 ms

        entry = payload["runs"][0]
        assert entry["in_run"]["best_stage"] == "refine"
        assert entry["in_run"]["cv_best"] == pytest.approx(0.67)
        assert entry["in_run"]["gs_adopted"] is True
        # Replay aggregates: means over the alternated repeats, spread kept.
        assert entry["replay"]["flat"]["n"] == 2
        assert entry["replay"]["flat"]["score_mean"] == pytest.approx(0.532)
        assert entry["replay"]["flat"]["score_spread"] == pytest.approx(0.004)
        assert entry["replay"]["best"]["score_mean"] == pytest.approx(0.770)
        assert summary["runs"][0]["ts"] == "20261014_000000"

    def test_session_keys_are_full_run_directory_names(self, gen) -> None:
        """Regression: keys were once short timestamps, so every lookup missed
        the ``data/debug/slm_gs_refine/<ts>/`` directories."""
        for ts in gen.SESSIONS:
            assert ts.startswith("20"), ts
            assert len(ts) == 15, ts  # YYYYMMDD_HHMMSS