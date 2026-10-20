"""Offline contract for the model-in-loop hardware A/B campaign.

Every check here runs without a device and without a network: the campaign's
decisions (profiles, command lines, expectations, manifest) are pure logic, so
they can be pinned down while the bench is offline.
"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/slm/model_in_loop_hw_ab.py"
SPEC = importlib.util.spec_from_file_location("model_in_loop_hw_ab", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def _args(*extra: str):
    return campaign.parse_args(list(extra))


def _write_rows(rows: list[dict]) -> list[dict]:
    """Round-trip the history rows through a CSV so the reader is exercised."""
    fieldnames = [
        "stage", "round", "accepted", "reason", "score_before", "score_after",
        "cv_after", "ee_after", "shape_iou_after", "holdout_correlation",
        "probe_count", "step_a_lr", "timestamp_utc",
    ]
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_profiles_are_accepted_by_the_existing_command_builder():
    profiles = campaign.build_profiles(_args())
    assert set(profiles) == {"baseline_a", "baseline_b", "large_1000", "large_1600"}
    for name, profile in profiles.items():
        unknown = set(profile) - campaign.PROFILE_FIELDS
        assert not unknown, f"{name} carries unknown fields: {sorted(unknown)}"
        # The existing builder raises ValueError on anything it cannot use.
        command = campaign.build_command(profile)
        assert "slm-model-in-loop" in command


def test_a_arm_keeps_the_holdout_gate_and_b_arm_forces_it_open():
    args = _args("--gate-holdout", "0.5", "--forced-holdout", "0")
    profiles = campaign.build_profiles(args)
    assert profiles["baseline_a"]["min_holdout_correlation"] == 0.5
    assert profiles["baseline_b"]["min_holdout_correlation"] == 0.0
    for name, profile in profiles.items():
        if name.startswith("large_"):
            assert profile["min_holdout_correlation"] == 0.0


def test_large_arm_uses_a_square_target_that_fits_the_readout_window():
    profiles = campaign.build_profiles(_args("--large-sides", "1000,1600"))
    for side in (1000, 1600):
        profile = profiles[f"large_{side}"]
        assert profile["target_shape"] == "square"
        assert profile["target_side"] == side
        assert profile["cam_size"] >= side + 100
    assert profiles["large_1600"]["cam_size"] == 1700


def test_exposure_is_never_the_bare_zero_the_runner_rejects():
    profiles = campaign.build_profiles(_args())
    for profile in profiles.values():
        exposure = profile["exposure_time_ms"]
        assert exposure != 0
        assert exposure != 0.0
        assert str(exposure) == "auto"
    numeric = campaign.build_profiles(_args("--exposure-time-ms", "0.8"))
    assert numeric["baseline_a"]["exposure_time_ms"] == 0.8
    with pytest.raises(SystemExit):
        _args("--exposure-time-ms", "0")


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_plan_reports_one_light_command_and_the_mil_commands(tmp_path, capsys):
    args = _args("--plan", "--out-root", str(tmp_path), "--large-sides", "1000")
    plan = campaign.build_plan(args, campaign.create_session_dir(args.out_root))
    mil_stages = [stage for stage in plan["stages"] if stage["kind"] != "light"]
    assert plan["light_command"]
    assert len(mil_stages) == 3
    assert [stage["name"] for stage in mil_stages] == [
        "baseline_a",
        "baseline_b",
        "large_1000",
    ]
    for stage in mil_stages:
        command = stage["command"]
        assert "--min-holdout-correlation" in command
        assert "slm-model-in-loop" in command
    assert campaign.main(["--plan", "--out-root", str(tmp_path), "--large-sides", "1000"]) == 0
    printed = capsys.readouterr().out
    assert "slm-model-in-loop" in printed
    assert "diagnose_light_level.py" in printed
    # The plan must not create anything.
    assert not list(tmp_path.iterdir())


def test_plan_command_carries_the_holdout_gate_value_of_each_arm(tmp_path):
    args = _args("--plan", "--out-root", str(tmp_path), "--large-sides", "1000")
    plan = campaign.build_plan(args, campaign.create_session_dir(args.out_root))
    values = {}
    for stage in plan["stages"]:
        if stage["kind"] == "light":
            continue
        command = stage["command"]
        values[stage["name"]] = command[command.index("--min-holdout-correlation") + 1]
    assert values == {
        "baseline_a": "0.5",
        "baseline_b": "0.0",
        "large_1000": "0.0",
    }


# ---------------------------------------------------------------------------
# Expectation judgement
# ---------------------------------------------------------------------------


def test_holdout_gate_that_stops_the_run_meets_the_a_arm_expectation():
    stdout = (
        "Status              : aborted_unidentifiable\n"
        "Rounds accepted     : 0 / 1\n"
        "Geometry            : panel_scan, correlation 0.8123, camera/model px 0.91\n"
        "Flat baseline score : 0.7102\n"
        "Best shaped score   : 0.7102\n"
    )
    rows = _write_rows(
        [
            {
                "stage": "round", "round": 1, "accepted": "False",
                "reason": "holdout_forward_mismatch", "holdout_correlation": 0.21,
                "probe_count": 6,
            }
        ]
    )
    observed = campaign.collect_observed(stdout, {"rows": rows})
    judgment = campaign.classify_stage("baseline_a", 2, observed)
    assert judgment["expectation_met"] is True
    assert judgment["outcome"] == "held_out_as_expected"


def test_a_gate_that_lets_the_run_through_is_reported_as_an_unexpected_pass():
    stdout = (
        "Status              : completed\n"
        "Rounds accepted     : 1 / 1\n"
        "Best phase saved    : C:/runs/best_phase.npy (raw radians)\n"
    )
    rows = _write_rows(
        [{"stage": "round", "round": 1, "accepted": "True", "score_after": 0.7}]
    )
    observed = campaign.collect_observed(stdout, {"rows": rows})
    judgment = campaign.classify_stage("baseline_a", 0, observed)
    assert judgment["expectation_met"] is False
    assert judgment["outcome"] == "unexpected_pass"


def test_completed_step_b_with_a_best_phase_meets_the_b_arm_expectation():
    stdout = (
        "Status              : completed\n"
        "Rounds accepted     : 1 / 1\n"
        "Flat baseline score : 0.7102\n"
        "Best shaped score   : 0.8200\n"
        "Best phase saved    : C:/runs/best_phase.npy (raw radians)\n"
        "Best image saved    : C:/runs/best_far_field.png\n"
    )
    rows = _write_rows(
        [
            {
                "stage": "round", "round": 1, "accepted": "True",
                "score_after": 0.82, "cv_after": 0.31, "ee_after": 0.62,
                "shape_iou_after": 0.55, "holdout_correlation": 0.42,
            }
        ]
    )
    observed = campaign.collect_observed(stdout, {"rows": rows})
    judgment = campaign.classify_stage("baseline_b", 0, observed)
    assert judgment["expectation_met"] is True
    assert judgment["outcome"] == "completed_as_expected"
    assert observed["cv"] == 0.31
    assert observed["ee"] == 0.62
    assert observed["iou"] == 0.55
    assert observed["holdout_correlation"] == [0.42]


def test_geometry_bake_off_rejection_is_separated_from_an_ordinary_abort():
    stdout = (
        "Preflight evidence  : C:/runs/slm_model_in_loop/20261020_101112\n"
        "运行中止, 设备已释放: panel scan response 0.000\n"
    )
    observed = campaign.collect_observed(stdout, {"failure_reason": "panel scan response 0.000"})
    judgment = campaign.classify_stage("baseline_b", 2, observed)
    assert judgment["outcome"] == "preflight_rejected"
    assert judgment["expectation_met"] is False


def test_a_crash_without_a_summary_is_a_failure_not_an_expectation():
    judgment = campaign.classify_stage("baseline_b", 1, {"status": None})
    assert judgment["outcome"] == "failed"
    assert judgment["expectation_met"] is False


def test_history_reader_tolerates_missing_columns(tmp_path):
    path = tmp_path / "slm_model_in_loop_20261020_101112.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "round", "accepted"])
        writer.writeheader()
        writer.writerow({"stage": "round", "round": 1, "accepted": "True"})
    rows = campaign.read_history_rows(path)
    summary = campaign.summarize_history_rows(rows)
    assert summary["rounds_total"] == 1
    assert summary["rounds_accepted"] == 1
    assert summary["cv"] is None
    assert summary["holdout_correlation"] == []


def test_light_verdict_blocks_the_campaign():
    judgment = campaign.classify_stage("light", 0, {"light_state": "no_light"})
    assert judgment["outcome"] == "blocked_no_light"
    assert judgment["expectation_met"] is False
    ok = campaign.classify_stage("light", 0, {"light_state": "ok"})
    assert ok["expectation_met"] is True


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_carries_the_required_keys_and_serialises(tmp_path):
    args = _args("--out-root", str(tmp_path))
    manifest = campaign.build_manifest(args, campaign.create_session_dir(args.out_root))
    for key in (
        "session_id", "created_utc", "git_commit", "host", "python", "argv",
        "campaign_state", "stages",
    ):
        assert key in manifest
    manifest["stages"].append(
        {
            "name": "baseline_a",
            "kind": "baseline_a",
            "command": ["python", "main.py", "slm-model-in-loop"],
            "exit_code": 2,
            "duration_s": 12.5,
            "started_utc": "2026-10-20T10:11:12+00:00",
            "ended_utc": "2026-10-20T10:11:24+00:00",
            "expected": campaign.EXPECTED_TEXT["baseline_a"],
            "expectation_met": True,
            "outcome": "held_out_as_expected",
            "observed": {"status": "aborted_unidentifiable", "rounds_accepted": 0},
            "log_file": "stage.log",
            "notes": [],
        }
    )
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    assert json.loads(payload)["campaign_state"] == "completed"
    summary = campaign.render_summary(manifest)
    assert "baseline_a" in summary
    assert "campaign state" in summary


# ---------------------------------------------------------------------------
# End to end with a faked runner
# ---------------------------------------------------------------------------


def _fake_runner(monkeypatch, calls: list, light_state: str = "ok"):
    """Replace the subprocess call with a stand-in that records a run."""

    def run_command(command, cwd, timeout_s):
        calls.append(command)
        if any("diagnose_light_level.py" in part for part in command):
            out_root = Path(command[command.index("--out-root") + 1])
            stamp = out_root / "20261020_101112"
            stamp.mkdir(parents=True, exist_ok=True)
            (stamp / "light_level.json").write_text(
                json.dumps({"verdict": {"state": light_state, "reasons": ["peak 120/255"]}}),
                encoding="utf-8",
            )
            return (2 if light_state == "no_light" else 0), f"verdict: {light_state}\n", 1.5

        stage_dir = Path(command[command.index("--dir") + 1])
        session = stage_dir / "debug" / "slm_model_in_loop" / "20261020_101112"
        session.mkdir(parents=True, exist_ok=True)
        (session / "run_config.json").write_text(
            json.dumps(
                {
                    "target_shape": "rectangle",
                    "target_side": 60,
                    "min_geometry_correlation": 0.5,
                    "min_holdout_correlation": 0.5,
                }
            ),
            encoding="utf-8",
        )
        if stage_dir.name == "baseline_a":
            (session / "slm_model_in_loop_20261020_101112.csv").write_text(
                "stage,round,accepted,reason,score_after,holdout_correlation\n"
                "round,1,False,holdout_forward_mismatch,,0.21\n",
                encoding="utf-8",
            )
            stdout = (
                "Status              : aborted_unidentifiable\n"
                "Rounds accepted     : 0 / 1\n"
                "Flat baseline score : 0.7102\n"
                "Best shaped score   : 0.7102\n"
                f"Recorder pkl/json   : {session}\n"
            )
            return 2, stdout, 42.0
        (session / "slm_model_in_loop_20261020_101112.csv").write_text(
            "stage,round,accepted,reason,score_after,cv_after,ee_after,"
            "shape_iou_after,holdout_correlation\n"
            "round,1,True,accepted,0.82,0.31,0.62,0.55,0.42\n",
            encoding="utf-8",
        )
        (session / "best_phase.npy").write_bytes(b"\x00" * 16)
        stdout = (
            "Status              : completed\n"
            "Rounds accepted     : 1 / 1\n"
            "Flat baseline score : 0.7102\n"
            "Best shaped score   : 0.8200\n"
            f"Best phase saved    : {session / 'best_phase.npy'} (raw radians)\n"
            f"Recorder pkl/json   : {session}\n"
        )
        return 0, stdout, 60.0

    monkeypatch.setattr(campaign, "run_command", run_command)


def test_campaign_runs_end_to_end_offline(tmp_path, monkeypatch):
    calls: list = []
    _fake_runner(monkeypatch, calls)
    code = campaign.main(
        ["--out-root", str(tmp_path), "--large-sides", "1000"]
    )
    assert code == 0
    sessions = list(tmp_path.iterdir())
    assert len(sessions) == 1
    session = sessions[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["campaign_state"] == "completed"
    names = [stage["name"] for stage in manifest["stages"]]
    assert names == ["light", "baseline_a", "baseline_b", "large_1000"]
    outcomes = {stage["name"]: stage["outcome"] for stage in manifest["stages"]}
    assert outcomes["baseline_a"] == "held_out_as_expected"
    assert outcomes["baseline_b"] == "completed_as_expected"
    # The session directory is lifted next to the stage log.
    assert (session / "baseline_b" / "slm_model_in_loop" / "20261020_101112").is_dir()
    assert (session / "baseline_b" / "stage.log").is_file()
    assert (session / "baseline_b" / "profile.json").is_file()
    summary = (session / "summary.md").read_text(encoding="utf-8")
    assert "baseline_a" in summary


def test_a_dark_bench_stops_the_campaign_before_any_run(tmp_path, monkeypatch):
    calls: list = []
    _fake_runner(monkeypatch, calls, light_state="no_light")
    code = campaign.main(["--out-root", str(tmp_path), "--large-sides", "1000"])
    assert code == 2
    session = list(tmp_path.iterdir())[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["campaign_state"] == "blocked_no_light"
    assert [stage["name"] for stage in manifest["stages"]] == ["light"]
    # No model-in-loop command was ever launched.
    assert not any("main.py" in part for command in calls for part in command)


def test_the_light_gate_can_be_switched_off(tmp_path, monkeypatch):
    calls: list = []
    _fake_runner(monkeypatch, calls)
    code = campaign.main(
        ["--out-root", str(tmp_path), "--large-sides", "1000", "--skip-light-gate"]
    )
    assert code == 0
    session = list(tmp_path.iterdir())[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    light = next(stage for stage in manifest["stages"] if stage["name"] == "light")
    assert light["outcome"] == "skipped"
    assert light["expectation_met"] is None
    # Only the light stage is switched off; the runs themselves still happen.
    assert not any("diagnose_light_level.py" in part for command in calls for part in command)
    assert any("main.py" in part for command in calls for part in command)

