"""Tests for the device-online campaign planner (device-free; writes to tmp_path).

The planner's whole job is to make protocol violations *unrepresentable* rather
than to be reviewed by a human at 2 a.m. Each assertion below pins one of the
mistakes the 2026-10-28 multishape review actually found in the previous
hand-written plan.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "plan_multishape_campaign.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_multishape_campaign_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner = _load_planner()


def _plan(**overrides) -> dict:
    fields = dict(sizes=[200, 600], shapes=["circle", "square"],
                  routes=["strict_cold", "analytic_init"], model_session="s1200",
                  campaign_dir=Path("/tmp/campaign"), frames=3,
                  replay_exposures=[20.0, 50.0, 90.0], repeats=6)
    fields.update(overrides)
    return planner.build_plan(**fields)


# -- route separation ------------------------------------------------------

def test_the_two_routes_never_share_a_session_or_a_directory():
    plan = _plan()
    riding = [lane for lane in plan["lanes"] if lane["route"] != "refit"]
    sessions = [lane["session"] for lane in plan["lanes"]]
    assert len(sessions) == len(set(sessions)) == 9  # 8 route lanes + the refit lane
    outs = [lane["out"] for lane in plan["lanes"]]
    assert len(outs) == len(set(outs))
    # The route is visible in the session name, so a report table cannot pool the
    # two by accident.
    assert {lane["route"] for lane in riding} == {"strict_cold", "analytic_init"}
    for lane in riding:
        assert lane["route"] in lane["session"]


def test_strict_cold_uses_only_an_uninformed_initial_phase():
    # ``random`` -> cold_start.cold_phase: reads neither the target nor the model.
    plan = _plan(routes=["strict_cold"])
    assert {lane["initial_phase"] for lane in plan["lanes"]} == {"random"}


def test_analytic_route_still_spends_no_gs_call():
    plan = _plan(routes=["analytic_init"])
    for lane in plan["lanes"]:
        if lane["route"] == "refit":
            continue  # lane 00 uses the cold route's phase; it is not an analytic lane
        assert lane["initial_phase"] == "backprojection"
        assert lane["gs_calls_allowed"] == 0
        assert "--no-gs-seed" in lane["commands"][-1]


# -- commands --------------------------------------------------------------

def test_every_lane_carries_the_preflight_gate_first():
    for lane in _plan()["lanes"]:
        joined = "\n".join(lane["commands"])
        assert planner.PROBE in lane["commands"][0]
        assert "LASTEXITCODE" in joined and "exit 1" in joined


def test_repeated_replay_exposures_are_expanded_one_flag_per_value():
    # click's ``multiple=True`` rejects ``--replay-exposures 20 50``; the planner
    # must emit one flag per value or the lane dies on its first real call.
    lane = _plan()["lanes"][0]
    command = lane["commands"][-1]
    for value in ("20.0", "50.0", "90.0"):
        assert f"--replay-exposures {value}" in command
    assert "--execute" in command


def test_result_classes_are_the_three_the_review_requires():
    lane = _plan()["lanes"][0]
    assert set(lane["required_results"]) == {"fixed_core_cv", "shape_agreement", "relative_energy"}
    assert set(planner.REQUIRED_AUDIT) == {"zero_excluded_fraction", "saturated_fraction",
                                           "phase_provenance", "model_checkpoint_sha256",
                                           "exposure_readback_ms"}


def test_window_span_follows_the_target_size():
    for lane in _plan()["lanes"]:
        assert lane["window_span"] == planner.SIZE_WINDOWS[lane["size"]]
        assert lane["window_span"] > lane["size"]


# -- ordering --------------------------------------------------------------

def test_lanes_run_route_then_size_then_shape():
    plan = _plan()
    assert plan["lanes"][0]["route"] == "refit"
    riding = plan["lanes"][1:]
    order = [(lane["route"], lane["size"], plan["shapes"].index(lane["shape"]))
             for lane in riding]
    assert order == sorted(order)
    assert [lane["order_index"] for lane in plan["lanes"]] == list(range(len(plan["lanes"])))


# -- the model may not be reused across an optical-power change -------------

def test_a_refit_lane_always_precedes_the_lanes_that_reuse_a_model():
    plan = _plan()
    refit, *rest = plan["lanes"]
    assert refit["order_index"] == 0
    assert refit["session"] == "s1200_refit"
    # The refit is the only lane without --model-dir: it *produces* the model.
    assert "--model-dir" not in " ".join(refit["commands"])
    for lane in rest:
        assert "--model-dir" in lane["commands"][-1]
        assert "<campaign>/s1200" in lane["commands"][-1]


def test_the_refit_writes_the_directory_the_other_lanes_reuse():
    plan = _plan()
    refit = plan["lanes"][0]
    assert Path(refit["out"]) == Path(plan["campaign_dir"]) / plan["model_session"]
    assert plan["model_power_check"]["refit_lane"] == refit["session"]
    assert plan["model_power_check"]["reuse_session"] == plan["model_session"]
    assert plan["model_power_check"]["rule"]


def test_a_plan_that_reuses_a_model_without_a_refit_is_rejected():
    plan = _plan()
    plan["lanes"] = [lane for lane in plan["lanes"] if lane["route"] != "refit"]
    for index, lane in enumerate(plan["lanes"]):
        lane["order_index"] = index
    problems = planner.validate_plan(plan)
    assert any("no refit lane" in problem for problem in problems)


def test_a_refit_that_is_not_first_or_wrongly_placed_is_rejected():
    plan = _plan()
    plan["lanes"][0]["order_index"] = 3          # something else would run first
    plan["lanes"][0]["out"] = "/tmp/elsewhere"   # lanes would reuse an empty dir
    problems = planner.validate_plan(plan)
    assert any("must run first" in problem for problem in problems)
    assert any("must write" in problem for problem in problems)


def test_the_refit_lane_may_not_itself_reuse_a_model():
    plan = _plan()
    refit = plan["lanes"][0]
    refit["commands"][-1] += " --model-dir /tmp/somewhere"
    problems = planner.validate_plan(plan)
    assert any("must not reuse a model dir" in problem for problem in problems)


# -- repeat budget ---------------------------------------------------------

def test_default_budget_can_reach_significance():
    assert planner.validate_plan(_plan()) == []


def test_three_repeats_are_rejected_as_unable_to_test_anything():
    problems = planner.validate_plan(_plan(repeats=3))
    assert problems
    assert all("cannot reach p<0.05" in problem for problem in problems)


# -- refusal cases ---------------------------------------------------------

def test_validate_rejects_a_gs_lane():
    plan = _plan()
    plan["lanes"][0]["initial_phase"] = "gs"
    plan["lanes"][0]["gs_calls_allowed"] = 80
    problems = planner.validate_plan(plan)
    assert any("may not use GS" in problem for problem in problems)


def test_validate_rejects_a_lane_without_preflight():
    plan = _plan()
    plan["lanes"][0]["commands"] = ["python src/ao_shaping/main.py slm-transfer-shapes"]
    assert any("preflight missing" in problem for problem in planner.validate_plan(plan))


def test_validate_rejects_missing_result_classes():
    plan = _plan()
    del plan["lanes"][0]["required_results"]["relative_energy"]
    assert any("result classes incomplete" in problem for problem in planner.validate_plan(plan))


def test_validate_rejects_an_unknown_size():
    plan = _plan()
    plan["lanes"][0]["window_span"] = None
    assert any("unknown target size" in problem for problem in planner.validate_plan(plan))


# -- CLI -------------------------------------------------------------------

def test_check_mode_writes_nothing_and_exits_zero(tmp_path, capsys):
    assert planner.main(["--check", "--campaign-dir", str(tmp_path / "c"),
                         "--sizes", "600", "--shapes", "circle"]) == 0
    assert not (tmp_path / "c").exists()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["sizes"] == [600]


def test_write_mode_emits_plan_and_one_lane_per_condition(tmp_path):
    campaign = tmp_path / "c"
    assert planner.main(["--campaign-dir", str(campaign), "--sizes", "600",
                         "--shapes", "circle", "square"]) == 0
    plan = json.loads((campaign / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["lanes"]) == 5  # refit lane + 2 routes x 1 size x 2 shapes
    lanes = sorted((campaign / "lanes").glob("*.ps1"))
    assert len(lanes) == 5
    assert lanes[0].name.startswith("00_")
    assert "refit" in lanes[0].name
    body = lanes[0].read_text(encoding="utf-8")
    assert planner.PROBE in body and "--no-gs-seed" in body
    assert "--model-dir" not in body


def test_check_mode_exits_nonzero_on_a_bad_budget(tmp_path):
    assert planner.main(["--check", "--campaign-dir", str(tmp_path / "c"),
                         "--sizes", "600", "--shapes", "circle", "--repeats", "3"]) == 1
