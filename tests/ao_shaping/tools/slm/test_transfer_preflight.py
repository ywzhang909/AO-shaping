"""迁移模型闭环整形的启动前检查（issue: 2026-10-26 实测 ``empty target core``）。

这些测试把 2026-10-26 那次真实失败钉成回归：操作员把载波周期填成 4 panel px，
而斜坡标定只在 64 / 128 panel px 上测过 —— 载波位移 ∝ 1/周期，于是位移被放大
16 倍，目标方斑整体飞出采样窗，``binary_erosion`` 之后核心为空，报错却出现在
第一个 CCD 分块（采集与拟合已经跑了几分钟）。

测的是**检查口径**，不是光学：斜坡矩阵用 2026-10-26 实测的那一组数值
（|matrix[:,0]| ≈ 7496 px/周期单位），零点与边长取自那次会话。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.preflight import (
    CALIBRATED_PERIODS_PX,
    HALF_SPAN_PX,
    MIN_PERIOD_PX,
    PROBE_COUNT_MULTIPLE,
    Problem,
    check_geometry,
    check_parameters,
    format_problems,
)

#: 2026-10-26 实测斜坡矩阵的第一列（x 轴斜坡的位移向量，单位 px/周期单位）：
#: 64 panel px 的斜坡给出约 117 px 位移 ⇒ 每周期单位约 7496 px。
MEASURED_RAMP_MATRIX = np.array([[169.6, 0.0], [-7494.4, 0.0]])

#: 那次会话的实测零点（相机像元）。
ZERO_XY = (1225.0, 959.8)


def _codes(problems: list[Problem]) -> set[str]:
    return {problem.code for problem in problems}


def _good_kwargs(**overrides):
    base = dict(
        side=80.0,
        period=64.0,
        count=24,
        iterations=60,
        blocks=4,
        updates=10,
        frames=3,
        shapes=("square",),
    )
    base.update(overrides)
    return base


# --- 参数级检查 --------------------------------------------------------------


def test_the_failed_session_parameters_are_rejected() -> None:
    """那次的参数（side=200、period=4）必须在启动前被拦下。"""
    problems = check_parameters(**_good_kwargs(side=200.0, period=4.0))

    assert "period_below_nyquist" in _codes(problems)
    assert any("4 panel px" in problem.message for problem in problems)


def test_a_valid_parameter_set_passes_without_errors() -> None:
    """标定档周期 + 6 的倍数探针数 + 放得下的边长 = 没有 error。"""
    problems = check_parameters(**_good_kwargs())

    assert [p for p in problems if p.severity == "error"] == []


def test_a_period_outside_the_calibrated_set_is_only_a_warning() -> None:
    """位移按 1/周期 外推：警告，不是错误。"""
    problems = check_parameters(**_good_kwargs(period=32.0))

    codes = _codes(problems)
    assert "period_extrapolated" in codes
    assert not [p for p in problems if p.severity == "error"]


def test_the_probe_count_must_be_a_multiple_of_six() -> None:
    """训练/验证划分写死为 index % 6 >= 3：非倍数会在拟合阶段才炸。"""
    problems = check_parameters(**_good_kwargs(count=10))

    assert "count_not_multiple" in _codes(problems)


def test_too_few_probes_are_rejected() -> None:
    problems = check_parameters(**_good_kwargs(count=2))

    assert "count_too_small" in _codes(problems)


def test_a_target_side_below_the_sampling_limit_is_rejected() -> None:
    """8 px 以下在 2 px 采样网格上取不到内部核心。"""
    problems = check_parameters(**_good_kwargs(side=4.0))

    assert "side_too_small" in _codes(problems)


def test_a_target_side_wider_than_the_window_is_rejected() -> None:
    problems = check_parameters(**_good_kwargs(side=1000.0))

    assert "side_exceeds_window" in _codes(problems)


def test_a_known_offset_makes_the_side_limit_tighter() -> None:
    """有实测位移时，限制变成「位移 + 半宽 ≤ 半跨 − 余量」。"""
    offset = 117.0
    problems = check_parameters(**_good_kwargs(side=200.0), offset_px=offset)

    assert "side_offset_overflow" in _codes(problems)


# --- 几何级检查（实测零点 + 拟合斜坡）----------------------------------------


def test_the_failed_session_geometry_is_rejected_with_a_period_hint() -> None:
    """period=4 时目标方斑完全出窗，且提示落在标定档上。"""
    problems = check_geometry(
        zero_xy=ZERO_XY,
        ramp_matrix=MEASURED_RAMP_MATRIX,
        side=200.0,
        period=4.0,
        shapes=("square",),
    )

    codes = _codes(problems)
    assert "target_outside_window" in codes
    hinted = [p for p in problems if p.code == "target_outside_window"][0]
    assert f"{CALIBRATED_PERIODS_PX[0]:g} panel px" in hinted.hint


def test_the_calibrated_period_passes_the_geometry_check() -> None:
    """标定档周期 + 合理边长 = 没有 error（64 panel px ⇒ 位移约 117 px）。"""
    problems = check_geometry(
        zero_xy=ZERO_XY,
        ramp_matrix=MEASURED_RAMP_MATRIX,
        side=80.0,
        period=64.0,
        shapes=("square",),
    )

    assert [p for p in problems if p.severity == "error"] == []


def test_a_target_that_only_grazes_the_window_has_no_core() -> None:
    """只擦到窗边 ⇒ 核心为空（就是那条报错）。"""
    # 位移让方斑只有 2 px 探进窗口（半跨 220，网格步长 2 px）：
    # 腐蚀 2 次后没有内部核心。
    half_side = 40.0
    offset = HALF_SPAN_PX + half_side - 2.0
    matrix = np.array([[0.0, 0.0], [offset, 0.0]])
    problems = check_geometry(
        zero_xy=ZERO_XY,
        ramp_matrix=matrix,
        side=2 * half_side,
        period=1.0,
        shapes=("square",),
    )

    codes = _codes(problems)
    assert codes & {"empty_target_core", "target_outside_window"}


def test_the_zero_order_must_stay_outside_the_target() -> None:
    """载波位移小于目标半宽 ⇒ 零级还在框里（警告）。"""
    matrix = np.array([[0.0, 0.0], [10.0, 0.0]])
    problems = check_geometry(
        zero_xy=ZERO_XY,
        ramp_matrix=matrix,
        side=80.0,
        period=1.0,
        shapes=("square",),
    )

    assert "zero_order_inside_target" in _codes(problems)


def test_a_broken_ramp_matrix_is_reported_not_raised() -> None:
    problems = check_geometry(
        zero_xy=ZERO_XY,
        ramp_matrix=np.zeros((3, 3)),
        side=80.0,
        period=64.0,
        shapes=("square",),
    )

    assert "ramp_matrix_invalid" in _codes(problems)


def test_format_problems_shows_the_fix() -> None:
    problems = check_parameters(**_good_kwargs(period=4.0))

    text = format_problems(problems)

    assert "period" in text or "载波周期" in text
    assert "→" in text  # 每条都带「怎么改」
    assert format_problems([]) == ""


def test_the_minimum_period_is_the_slm_nyquist() -> None:
    """8 µm 像元 ⇒ 每周期至少 2 个像元 = 16 panel px。"""
    assert MIN_PERIOD_PX == 16.0
    assert PROBE_COUNT_MULTIPLE == 6


# --- 接线：会话自身的 validate() 在采集之前就拒绝 ---------------------------


def _session(**overrides):
    from ao_shaping.tools.slm.transfer.workflow import TransferSession

    base = dict(out=Path("data/debug/slm_transfer/preflight_probe"), shapes=("square",))
    base.update(overrides)
    return TransferSession(**base)


def test_the_session_rejects_the_period_that_broke_the_2026_10_26_run() -> None:
    """period=4 必须被会话自己拒绝，且给出可执行的改法。"""
    with pytest.raises(ValueError) as excinfo:
        _session(period=4.0).validate()

    message = str(excinfo.value)
    assert "4" in message
    assert "panel px" in message
    assert "→" in message  # 带「怎么改」


def test_the_session_rejects_a_probe_count_that_is_not_a_multiple_of_six() -> None:
    with pytest.raises(ValueError) as excinfo:
        _session(count=10).validate()

    assert "10" in str(excinfo.value)


def test_the_session_accepts_its_own_defaults() -> None:
    """默认参数必须通过（否则这条检查会把正常流程也堵死）。"""
    _session().validate()
