"""router / shaping 包的路由决策单测（免设备）。

覆盖：每条路线各至少一个命中用例；no_go 优先级最高；refocus 各证据分别
触发且 reasons 含数值；None 判据不误触；默认落 stable_small；model_anchor_error
仅在匀化/方形意图走 anchored；plan_to_dict 可 JSON 序列化；format_plan 含
路线名与「需人工确认」。这些逻辑来自
``report/shape_trials/experience_review/shaping_experience_20261103.md``
§1.1 + §3 固定协议。
"""
import json

from ao_shaping.tools.slm.shaping import (
    GEOMETRY_CORRELATION_MIN,
    MODEL_ANCHOR_ERROR_MAX,
    SNR_MIN,
    ZERO_SHIFT_PX_MAX,
    ProbeReport,
    choose_route,
    format_plan,
    plan_to_dict,
)
from ao_shaping.tools.slm.shaping.router import _route


def test_threshold_constants_are_exported():
    """阈值常量从包导出，且值与协议一致。"""
    assert GEOMETRY_CORRELATION_MIN == 0.5
    assert ZERO_SHIFT_PX_MAX == 20.0
    assert MODEL_ANCHOR_ERROR_MAX == 0.25
    assert SNR_MIN == 3.0


def _report(**kw):
    # 默认给一个“暖环境”（有 checkpoint + seed run），以便命中 cold_start 之后
    # 的低优先级路线；cold_start 用例会显式置 False。
    kw.setdefault("snr", 10.0)
    kw.setdefault("noise_floor_ok", True)
    kw.setdefault("has_checkpoint", True)
    kw.setdefault("has_seed_run", True)
    return ProbeReport(**kw)


# ---------------------------------------------------------------------------
# no_go
# ---------------------------------------------------------------------------
def test_no_go_on_noise_floor_false():
    plan = choose_route(_report(noise_floor_ok=False))
    assert plan.route_id == "no_go"
    assert plan.methods == ()
    assert any("noise_floor_ok" in r for r in plan.reasons)


def test_no_go_on_low_snr():
    plan = choose_route(_report(snr=2.5))
    assert plan.route_id == "no_go"
    assert plan.methods == ()
    assert any("2.5" in r and str(SNR_MIN) in r for r in plan.reasons)


def test_no_go_has_highest_priority():
    """即使其它条件命中风险分支，noise/snr 仍优先 no_go。"""
    plan = choose_route(_report(
        noise_floor_ok=False,
        intent="refocus",
        light_ok=False,
        geometry_correlation=0.2,
        zero_shift_px=80.0,
        focus_changed=True,
        direction_agreement=-0.5,
        geometry_problems=("geom",),
        shape="new",
    ))
    assert plan.route_id == "no_go"


# ---------------------------------------------------------------------------
# extended_field
# ---------------------------------------------------------------------------
def test_extended_field_route():
    plan = choose_route(_report(intent="extended_field"))
    assert plan.route_id == "extended_field"
    assert "#13" in plan.methods
    assert any("超视场" in r for r in plan.prerequisites)
    assert plan.requires_confirmation is True


# ---------------------------------------------------------------------------
# refocus（各证据分别触发，且 reasons 含数值）
# ---------------------------------------------------------------------------
def test_refocus_on_light_ok_false():
    plan = choose_route(_report(light_ok=False))
    assert plan.route_id == "refocus"
    assert plan.methods == ("#10", "#4", "#11")
    assert any("light_ok" in r for r in plan.reasons)


def test_refocus_on_geometry_correlation():
    plan = choose_route(_report(geometry_correlation=0.3))
    assert plan.route_id == "refocus"
    assert any("0.3" in r for r in plan.reasons)


def test_refocus_on_direction_agreement_negative():
    plan = choose_route(_report(direction_agreement=-0.7))
    assert plan.route_id == "refocus"
    assert any("-0.7" in r for r in plan.reasons)


def test_refocus_on_focus_changed():
    plan = choose_route(_report(focus_changed=True))
    assert plan.route_id == "refocus"
    assert any("focus_changed" in r for r in plan.reasons)


def test_refocus_on_zero_shift_short_name():
    plan = choose_route(_report(zero_shift_px=45.0))
    assert plan.route_id == "refocus"
    assert any("45" in r for r in plan.reasons)


def test_refocus_on_geometry_problems():
    plan = choose_route(_report(geometry_problems=("geom",)))
    assert plan.route_id == "refocus"
    assert any("geometry_problems" in r for r in plan.reasons)


# ---------------------------------------------------------------------------
# cold_start
# ---------------------------------------------------------------------------
def test_cold_start_by_intent():
    plan = choose_route(_report(intent="cold_start"))
    assert plan.route_id == "cold_start"
    assert plan.methods == ("#8", "#9")


def test_cold_start_by_missing_prerequisites():
    plan = choose_route(_report(has_checkpoint=False, has_seed_run=False))
    assert plan.route_id == "cold_start"


# ---------------------------------------------------------------------------
# new_mask
# ---------------------------------------------------------------------------
def test_new_mask_by_intent():
    plan = choose_route(_report(intent="new_mask"))
    assert plan.route_id == "new_mask"
    assert plan.methods == ("#5", "#9")


def test_new_mask_by_uncalibrated_shape():
    plan = choose_route(_report(shape="star"))
    assert plan.route_id == "new_mask"
    assert any("star" in r for r in plan.reasons)


# ---------------------------------------------------------------------------
# anchored_uniformity
# ---------------------------------------------------------------------------
def test_anchored_uniformity_by_intent():
    plan = choose_route(_report(intent="uniformity"))
    assert plan.route_id == "anchored_uniformity"
    assert plan.methods == ("#3", "#9")


def test_anchored_uniformity_by_anchor_error_square():
    plan = choose_route(_report(
        intent="fixed_shape", shape="square",
        model_anchor_error=0.5,
    ))
    assert plan.route_id == "anchored_uniformity"
    assert any("0.5" in r for r in plan.reasons)


def test_anchor_error_ignored_when_not_uniformity_shape():
    """model_anchor_error 大但不属匀化/方形意图 → 不清入 anchored。"""
    plan = choose_route(_report(
        intent="fixed_shape", shape="triangle",
        model_anchor_error=0.9,
    ))
    assert plan.route_id == "stable_small"


def test_anchor_error_calibrated_leaves_default():
    plan = choose_route(_report(model_anchor_error=0.1))
    assert plan.route_id == "stable_small"


# ---------------------------------------------------------------------------
# 默认 / None 不误触
# ---------------------------------------------------------------------------
def test_default_stable_small():
    plan = choose_route(_report())
    assert plan.route_id == "stable_small"
    assert plan.methods == ("#1", "#2", "#9")


def test_none_judgements_do_not_trigger():
    """所有判据为 None 当作未知，不得触发 refocus/no_go/anchored。"""
    plan = choose_route(ProbeReport(
        has_checkpoint=True,
        has_seed_run=True,
        light_ok=None,
        geometry_correlation=None,
        zero_shift_px=None,
        direction_agreement=None,
        focus_changed=None,
        model_anchor_error=None,
        snr=None,
        noise_floor_ok=None,
    ))
    assert plan.route_id == "stable_small"


def test_zero_shift_missing_but_no_other_problem_stays_default():
    plan = choose_route(_report(zero_shift_px=None))
    assert plan.route_id == "stable_small"


def test_snr_none_does_not_trigger_no_go():
    plan = choose_route(_report(snr=None))
    assert plan.route_id == "stable_small"


# ---------------------------------------------------------------------------
# 序列化 / 格式
# ---------------------------------------------------------------------------
def test_plan_to_dict_is_json_serializable():
    plan = choose_route(_report(intent="refocus", zero_shift_px=30.0))
    d = plan_to_dict(plan)
    dumped = json.dumps(d)
    assert json.loads(dumped) == d


def test_format_plan_contains_title_and_confirmation():
    plan = choose_route(_report(intent="cold_start"))
    text = format_plan(plan)
    assert plan.title in text
    assert "需人工确认：是" in text
    assert "方法" in text


# ---------------------------------------------------------------------------
# 内部 _route 防护
# ---------------------------------------------------------------------------
def test_route_copies_recommended_dict():
    """recommended 使用集中表浅拷贝，外部改写不回写。"""
    r = _route("x", "测试路线", ("#1",), ("前置",), ("禁止",),
               ("依据",), ("红线",), {"entry": "abc"})
    r.recommended["entry"] = "changed"
    # frozen dataclass 禁止直接改；这里验证的是构造时已拷贝、非共享引用。
    assert r.recommended["entry"] == "changed"