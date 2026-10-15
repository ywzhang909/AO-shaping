"""Offline checks for the GS-refine replay tool."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ao_shaping.tools.slm.gs_refine_replay import load_replay_plan, parse_xy


def _save_run(tmp_path, *, bad_shape=False):
    run_dir = tmp_path / "debug" / "slm_gs_refine" / "20261012_180222"
    run_dir.mkdir(parents=True)
    phase = np.zeros((4, 6), dtype=np.uint16)
    records = {
        0: {"score": 0.5, "_phase": phase, "cx": 2, "cy": 1, "target_side": 3},
        1: {"score": 0.7, "_phase": phase + 1},
        2: {
            "score": 0.8,
            "_phase": np.ones((5, 6), dtype=np.uint16) if bad_shape else phase + 2,
        },
    }
    with (run_dir / "slm_gs_refine_20261012_180222.pkl").open("wb") as handle:
        pickle.dump(records, handle)
    return run_dir


def test_load_plan_from_run_directory(tmp_path):
    run_dir = _save_run(tmp_path)
    plan = load_replay_plan(run_dir, exposure_ms=0.02)

    assert plan.output_dir == run_dir / "replay"
    assert plan.best_record_key == 2
    assert plan.debug_pkl.parent == run_dir
    assert plan.target_center == (2, 1)
    assert plan.target_side == 3
    assert plan.exposure_ms == 0.02
    np.testing.assert_array_equal(plan.phases["best"], np.full((4, 6), 2))


def test_replay_exposure_limit_and_phase_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
    run_dir = _save_run(tmp_path, bad_shape=True)
    with pytest.raises(ValueError, match="exposure"):
        load_replay_plan(run_dir, exposure_ms=0.401)
    with pytest.raises(ValueError, match="matching 2D"):
        load_replay_plan(run_dir, exposure_ms=0.02)


def test_parse_xy():
    assert parse_xy("886,706") == (886, 706)
    with pytest.raises(ValueError, match="x,y"):
        parse_xy("886")
