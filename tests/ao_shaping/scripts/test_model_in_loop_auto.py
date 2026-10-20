"""Offline contract for the portable hardware-session wrapper."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.model_in_loop_summary import summarize_session


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/slm/model_in_loop_auto.py"
SPEC = importlib.util.spec_from_file_location("model_in_loop_auto", SCRIPT)
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)


def _profile():
    return dict(cam_type="daheng", cam_id=0, slm_number=1, slm_wavelength=1064,
                target_shape="square", target_side=60, region=128,
                far_field_padding=8, n_rounds=2, max_cv=0.5, min_ee=0.4,
                min_iou=0.5)


def test_build_command_requires_fresh_geometry_and_valid_exposure():
    profile = _profile()
    command = auto.build_command(profile)
    assert "--geometry-file" not in command
    assert command[command.index("--cam-type") + 1] == "daheng"
    with pytest.raises(ValueError, match="geometry_file"):
        auto.build_command({**profile, "geometry_file": "old.json"})
    with pytest.raises(ValueError, match="exposure"):
        auto.build_command({**profile, "exposure_time_ms": 1, "max_exposure_ms": 0.5})


def test_exposure_token_is_accepted_and_a_bare_zero_is_not():
    """The runner rejects a bare 0, so the profile must not smuggle one in.

    ``auto`` / ``default`` are the words that carry the legacy "no fixed
    exposure" meaning; a numeric exposure is still bounded by the ceiling.
    """
    profile = _profile()
    command = auto.build_command({**profile, "exposure_time_ms": "auto"})
    assert command[command.index("--exposure-time-ms") + 1] == "auto"
    default = auto.build_command({**profile, "exposure_time_ms": "default"})
    assert default[default.index("--exposure-time-ms") + 1] == "default"
    with pytest.raises(ValueError, match="exposure"):
        auto.build_command({**profile, "exposure_time_ms": 0})
    with pytest.raises(ValueError, match="exposure"):
        auto.build_command({**profile, "exposure_time_ms": "bright"})


def test_the_ceiling_only_bounds_a_numeric_exposure():
    profile = _profile()
    command = auto.build_command(
        {**profile, "exposure_time_ms": "auto", "max_exposure_ms": 0.2}
    )
    assert command[command.index("--exposure-time-ms") + 1] == "auto"
    with pytest.raises(ValueError, match="exposure"):
        auto.build_command({**profile, "exposure_time_ms": 1.0, "max_exposure_ms": 0.2})


def test_dry_run_does_not_launch_subprocess(monkeypatch, tmp_path, capsys):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    monkeypatch.setattr(auto.subprocess, "Popen", lambda *_a, **_k: pytest.fail("opened device"))
    assert auto.main([str(path), "--dry-run"]) == 0
    assert "slm-model-in-loop" in capsys.readouterr().out


def test_summary_checks_each_metric_and_identifies_best_recorded_round(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps({
        "target_shape": "square", "target_side": 60,
        "min_geometry_correlation": 0.5, "min_holdout_correlation": 0.5,
    }), encoding="utf-8")
    (tmp_path / "bench_geometry.json").write_text(json.dumps({"correlation": 0.8}), encoding="utf-8")
    np.save(tmp_path / "best_phase.npy", np.zeros((2, 2)))
    with (tmp_path / "slm_model_in_loop_123.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "round", "accepted", "score_after",
                                                     "cv_after", "ee_after", "shape_iou_after",
                                                     "holdout_correlation"])
        writer.writeheader()
        writer.writerow(dict(stage="calibration", round=0, accepted=True,
                             score_after=0.2))
        writer.writerow(dict(stage="round", round=1, accepted=True, score_after=0.6,
                             cv_after=0.4, ee_after=0.6, shape_iou_after=0.2,
                             holdout_correlation=0.8))
        writer.writerow(dict(stage="round", round=2, accepted=True, score_after=0.55,
                             cv_after=0.3, ee_after=0.6, shape_iou_after=0.8,
                             holdout_correlation=0.9))
    summary = summarize_session(tmp_path, max_cv=0.5, min_ee=0.4, min_iou=0.5)
    assert summary["best_recorded_round"] == 1
    assert summary["checks"]["iou"] is False
    assert summary["checks"]["beats_flat"] is True
    assert summary["verdict"] == "review_required"


def test_summary_never_claims_success_without_rounds(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps({"target_shape": "circle", "target_side": 20}), encoding="utf-8")
    (tmp_path / "slm_model_in_loop_123.csv").write_text("stage,round,accepted\n", encoding="utf-8")
    summary = summarize_session(tmp_path, max_cv=0.5, min_ee=0.4, min_iou=0.5)
    assert summary["verdict"] == "review_required"
    assert summary["best_recorded_round"] is None
