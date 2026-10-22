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
        # The window must be **larger** than the target (a truncated box biases
        # EE / CV) and carry a margin for the 0-order's off-centre position.
        assert profile["cam_size"] > side
        assert profile["cam_size"] >= side + max(100, int(round(0.10 * side)))
    assert profiles["large_1600"]["cam_size"] == 1760


def test_a_window_that_cannot_hold_the_target_is_refused_before_any_device():
    """The guard runs while the profiles are built, so no device is opened."""
    with pytest.raises(ValueError, match="must be larger than the target side"):
        campaign._assert_window_fits_target(
            "bad", {"cam_size": 900, "target_side": 1000}
        )


def test_window_for_target_keeps_the_base_window_when_it_already_fits():
    assert campaign.window_for_target(60, 300) == 300
    assert campaign.window_for_target(1000, 300) == 1100
    assert campaign.window_for_target(1600, 300) == 1760


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


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------


def _geometry_payload() -> str:
    """A sweep geometry, as the session calibration writes it."""
    return json.dumps(
        {
            "panel_disc_radius": 450,
            "region": 128,
            "beam_waist_panel_px": 24.0,
            "far_field_size": 1024,
            "camera_px_per_model_px": 2.614022,
            "spot_fwhm_camera_px": 13.19,
            "spot_fwhm_model_px": 8.64,
            "correlation": 0.848,
            "method": "sweep",
            "calibration_notes": "measured sweep",
        }
    )


def _fake_geometry_runner(monkeypatch, calls: list, *, produce_geometry: bool):
    """A stand-in runner whose donor arm may or may not produce a geometry."""
    _fake_runner(monkeypatch, calls)
    original = campaign.run_command

    def run_command(command, cwd, timeout_s):
        code, output, duration = original(command, cwd, timeout_s)
        if "main.py" not in " ".join(command):
            return code, output, duration
        stage_dir = Path(command[command.index("--dir") + 1])
        session = stage_dir / "debug" / "slm_model_in_loop" / "20261020_101112"
        if produce_geometry and stage_dir.name == "baseline_a":
            calibration = session / "session_geometry"
            calibration.mkdir(parents=True, exist_ok=True)
            (calibration / "bench_geometry.json").write_text(
                _geometry_payload(), encoding="utf-8"
            )
        return code, output, duration

    monkeypatch.setattr(campaign, "run_command", run_command)


def test_the_later_arms_run_with_the_donors_geometry(tmp_path, monkeypatch):
    """The B arm must be handed the geometry the donor arm measured.

    Two session calibrations on this bench produced camera/model scales of
    1.3043 and 2.6531, so arms that each calibrate on their own compare two
    different experiments. The campaign therefore measures the geometry once
    and shares it.
    """
    calls: list = []
    _fake_geometry_runner(monkeypatch, calls, produce_geometry=True)
    code = campaign.main(["--out-root", str(tmp_path), "--large-sides", "1000"])
    assert code == 0
    session = list(tmp_path.iterdir())[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in manifest["stages"]}

    shared = session / campaign.SHARED_GEOMETRY_NAME
    assert shared.is_file(), "the donor's geometry must be copied to the session"
    assert manifest["shared_geometry"] == str(shared)
    assert manifest["shared_geometry_exists"] is True

    # The donor measured it; the later arms were given the file.
    assert stages["baseline_a"]["geometry_donor"] is True
    assert stages["baseline_a"]["geometry_file"] is None
    for name in ("baseline_b", "large_1000"):
        assert stages[name]["geometry_file"] == str(shared)
        assert stages[name]["geometry_fallback"] is False
        assert "--geometry-file" in stages[name]["command"]
        assert stages[name]["command"][
            stages[name]["command"].index("--geometry-file") + 1
        ] == str(shared)

    # The B arm really was launched with that file.
    b_command = next(
        command
        for command in calls
        if "main.py" in " ".join(command) and "baseline_b" in " ".join(command)
    )
    assert "--geometry-file" in b_command
    assert b_command[b_command.index("--geometry-file") + 1] == str(shared)


def test_without_a_donor_geometry_the_arms_fall_back_and_say_so(tmp_path, monkeypatch):
    """A silent fallback would make the arms incomparable without a trace."""
    calls: list = []
    _fake_geometry_runner(monkeypatch, calls, produce_geometry=False)
    code = campaign.main(["--out-root", str(tmp_path), "--large-sides", "1000"])
    assert code == 0
    session = list(tmp_path.iterdir())[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in manifest["stages"]}

    assert not (session / campaign.SHARED_GEOMETRY_NAME).is_file()
    assert manifest["shared_geometry_exists"] is False
    for name in ("baseline_b", "large_1000"):
        assert stages[name]["geometry_file"] is None
        assert stages[name]["geometry_fallback"] is True
        assert "--geometry-file" not in stages[name]["command"]
    # The donor's record says why the hand-off did not happen.
    donor_notes = " ".join(stages["baseline_a"]["notes"])
    assert "produced no bench_geometry.json" in donor_notes


def test_an_external_geometry_file_is_used_by_every_arm(tmp_path, monkeypatch):
    """``--geometry-file`` short-circuits the donor: every arm uses it."""
    calls: list = []
    _fake_runner(monkeypatch, calls)
    external = tmp_path / "measured_geometry.json"
    external.write_text(_geometry_payload(), encoding="utf-8")
    code = campaign.main(
        [
            "--out-root",
            str(tmp_path / "runs"),
            "--large-sides",
            "1000",
            "--geometry-file",
            str(external),
        ]
    )
    assert code == 0
    session = list((tmp_path / "runs").iterdir())[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in manifest["stages"]}
    for name in ("baseline_a", "baseline_b", "large_1000"):
        assert stages[name]["geometry_file"] == str(external)
        assert "--geometry-file" in stages[name]["command"]


def test_the_error_params_reach_every_arm(tmp_path, monkeypatch):
    """The forward model's error terms are handed to every arm."""
    calls: list = []
    _fake_runner(monkeypatch, calls)
    errors = tmp_path / "model_error_terms.json"
    errors.write_text(json.dumps({"lut_gamma": 2.07}), encoding="utf-8")
    code = campaign.main(
        [
            "--out-root",
            str(tmp_path),
            "--large-sides",
            "1000",
            "--error-params",
            str(errors),
        ]
    )
    assert code == 0
    session = list(tmp_path.iterdir())[0]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    for stage in manifest["stages"]:
        if stage["kind"] == "light":
            continue
        assert "--error-params" in stage["command"]
        assert (
            stage["command"][stage["command"].index("--error-params") + 1]
            == str(errors)
        )
        # The profile written next to the stage log carries it too.
        profile = json.loads(
            (session / stage["name"] / "profile.json").read_text(encoding="utf-8")
        )
        assert profile["error_params"] == str(errors)


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def _shaping_stdout(status: str, flat: float, best: float) -> str:
    return (
        f"Status              : {status}\n"
        "Rounds accepted     : 1 / 1\n"
        f"Flat baseline score : {flat:.4f}\n"
        f"Best shaped score   : {best:.4f}\n"
        "Best phase saved    : C:/runs/best_phase.npy (raw radians)\n"
    )


def test_a_shaped_phase_that_beats_flat_meets_the_b_arm_expectation():
    stdout = _shaping_stdout("completed", 0.7102, 0.8200)
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


def test_a_bake_off_rejection_is_not_a_pass():
    """``bake_off_rejected`` must not be counted as "completed + a best phase".

    The old rule ("completed + an accepted round + a best phase file") called
    that a pass, which is exactly the misjudgement this criterion removes: the
    phase was *not* adopted, the flat baseline was kept.
    """
    stdout = _shaping_stdout("bake_off_rejected", 0.7102, 0.7102)
    rows = _write_rows(
        [
            {
                "stage": "round", "round": 1, "accepted": "True",
                "score_after": 0.71, "cv_after": 0.42, "ee_after": 0.51,
                "shape_iou_after": 0.30, "holdout_correlation": 0.61,
            }
        ]
    )
    observed = campaign.collect_observed(stdout, {"rows": rows})
    judgment = campaign.classify_stage("baseline_b", 0, observed)
    assert judgment["expectation_met"] is False
    assert judgment["outcome"] == "shaping_did_not_beat_flat"
    assert "did not beat flat" in " ".join(judgment["notes"])


def test_a_completed_run_that_only_matches_flat_is_not_a_pass():
    stdout = _shaping_stdout("completed", 0.7102, 0.7102)
    rows = _write_rows(
        [
            {
                "stage": "round", "round": 1, "accepted": "True",
                "score_after": 0.7102, "cv_after": 0.42, "ee_after": 0.51,
                "shape_iou_after": 0.30, "holdout_correlation": 0.61,
            }
        ]
    )
    observed = campaign.collect_observed(stdout, {"rows": rows})
    judgment = campaign.classify_stage("baseline_b", 0, observed)
    assert judgment["expectation_met"] is False
    assert judgment["outcome"] == "shaping_did_not_beat_flat"


def test_a_completed_run_that_loses_to_flat_is_not_a_pass():
    stdout = _shaping_stdout("completed", 0.7102, 0.6800)
    rows = _write_rows(
        [
            {
                "stage": "round", "round": 1, "accepted": "True",
                "score_after": 0.68, "cv_after": 0.42, "ee_after": 0.51,
                "shape_iou_after": 0.30, "holdout_correlation": 0.61,
            }
        ]
    )
    observed = campaign.collect_observed(stdout, {"rows": rows})
    judgment = campaign.classify_stage("baseline_b", 0, observed)
    assert judgment["expectation_met"] is False
    assert judgment["outcome"] == "shaping_did_not_beat_flat"


def test_the_a_arm_expectation_says_the_gate_is_not_moved():
    """The A arm's expectation must describe the gate arm honestly."""
    text = campaign.EXPECTED_TEXT["baseline_a"]
    assert "gate" in text
    assert "bake-off" in text or "flat" in text
    assert "completed" not in text


def test_the_b_arm_expectation_demands_beating_flat():
    text = campaign.EXPECTED_TEXT["baseline_b"]
    assert "flat" in text
    assert "greater" in text

