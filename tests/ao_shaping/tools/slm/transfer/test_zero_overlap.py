"""零级剔除圆盘 ∩ 目标方框的重叠面积（issue #214 的验收项，免设备）。

三个验收数字来自 2026-10-28 那次 multishape campaign 的实测量：

* 载波位移实测 ``ramp_matrix[:, 0] / 48 = (14.76, −129.70)``
  （``report/shape_trials/multishape_scale/multishape_scale_planning_20261028.md:37``，
  §1 H3 勘误）；
* 零级位置与 s1200 目标框中心 ``(1294.8, 982.4)`` / ``(1306.5, 780.7)``
  （``.../artifacts/geometry_plan.json``）；
* 剔除面积 1200 → 0.79 %、600 → 3.14 %、200 → 5.61 %，200 档「圆心离框边 29.70 px、
  穿入 30.30 px」（同报告 ``:71`` 与 ``multishape_scale_live_20261028.md:50``）。

实现里**没有**对这三个值做特判：它们必须从几何算出来。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.preflight import (
    ZeroOverlap,
    _disc_box_area,
    check_zero_overlap,
)

#: ``geometry_plan.json`` 的零级位置（sensor px）。
ZERO_XY = (1294.8, 982.4)

#: ``geometry_plan.json`` 的 s1200 目标框中心（sensor px，= 零级 + 规划位移 (11.74, −201.7)）。
TARGET_XY_PLANNED = (1306.5, 780.7)

#: 拟合完成后实测的载波位移（规划报告 §1 H3）。200 档的目标框中心 = 零级 + 该位移。
MEASURED_SHIFT = (14.76, -129.70)


@pytest.mark.parametrize(
    "side,target,expected_pct",
    [
        (1200.0, TARGET_XY_PLANNED, 0.79),
        (600.0, TARGET_XY_PLANNED, 3.14),
        (200.0, (ZERO_XY[0] + MEASURED_SHIFT[0], ZERO_XY[1] + MEASURED_SHIFT[1]), 5.61),
    ],
)
def test_acceptance_numbers(side, target, expected_pct) -> None:
    """侧 1200 / 600 圆盘完全在框内，200 压进 29.70 px 外的框边 —— 逐位对上报告。"""
    result = check_zero_overlap(target[0], target[1], side, ZERO_XY)

    assert isinstance(result, ZeroOverlap)
    assert result.intersects is True
    assert abs(result.pct - expected_pct) <= 0.01, f"{side}: {result.pct} ≠ {expected_pct}"


def test_smallest_side_penetrates_thirty_px() -> None:
    """200 px 档：圆心在框外 29.70 px、圆盘穿入 30.30 px（半径 60 − 29.70）。"""
    target_x = ZERO_XY[0] + MEASURED_SHIFT[0]
    target_y = ZERO_XY[1] + MEASURED_SHIFT[1]

    result = check_zero_overlap(target_x, target_y, 200.0, ZERO_XY)

    assert result.depth_px == pytest.approx(30.30, abs=0.01)
    assert "29.70" in result.note
    assert "5.61" in result.note


def test_large_sides_are_fully_inside_and_remove_the_whole_disc() -> None:
    """1200 / 600 档圆盘整个落在框内：面积就是 πr²，穿入深度 0。"""
    for side in (1200.0, 600.0):
        result = check_zero_overlap(TARGET_XY_PLANNED[0], TARGET_XY_PLANNED[1], side, ZERO_XY)

        assert result.area_px == pytest.approx(math.pi * 60.0**2)
        assert result.depth_px == 0.0
        assert "完全落在目标框内" in result.note


def test_disc_fully_outside_has_zero_area() -> None:
    """完全不相交时面积为 0、深度为 0（不是"负穿入"）。"""
    result = check_zero_overlap(0.0, 0.0, 200.0, (500.0, 500.0))

    assert result.intersects is False
    assert result.area_px == 0.0
    assert result.pct == 0.0
    assert result.depth_px == 0.0
    assert "不相交" in result.note


def test_disc_touching_the_edge_from_below() -> None:
    """圆心在框外 < 半径时部分相交；深度 = 半径 − 距框距离。"""
    # 框 y ∈ [0, 200]（中心 100），圆心在下方 220 → 距框 20，半径 60 → 穿入 40
    result = check_zero_overlap(100.0, 100.0, 200.0, (100.0, 220.0))

    assert result.intersects is True
    assert result.depth_px == pytest.approx(40.0, abs=1e-9)
    # 解析弦段面积：高 h = 40 的圆缺 = r² acos(d/r) − d√(r²−d²), d = r − h
    d = 20.0
    expected = 60.0**2 * math.acos(d / 60.0) - d * math.sqrt(60.0**2 - d**2)
    assert result.area_px == pytest.approx(expected, rel=1e-9)


def test_corner_straddle_matches_dense_integration() -> None:
    """圆角相交（两个方向同时裁剪）时与稠密数值积分一致 —— 防止只对直边特化。"""
    cases = [
        # (centre, radius, box)
        ((210.0, 210.0), 60.0, (0.0, 200.0, 0.0, 200.0)),
        ((150.0, 250.0), 80.0, (0.0, 200.0, 0.0, 200.0)),
        ((-25.0, 30.0), 90.0, (0.0, 200.0, 0.0, 200.0)),
        ((0.0, 0.0), 120.0, (-50.0, 50.0, -50.0, 50.0)),
        ((1294.8, 982.4), 60.0, (706.56, 1906.56, 180.7, 1380.7)),
    ]
    for centre, radius, box in cases:
        exact = _disc_box_area(centre[0], centre[1], radius, *box)
        numeric = _numeric_disc_box_area(centre[0], centre[1], radius, *box)
        assert exact == pytest.approx(numeric, abs=0.01 * math.pi * radius**2), (centre, radius, box)


def test_box_smaller_than_the_disc_is_not_mistaken_for_a_full_disc() -> None:
    """框比圆盘小时圆心在框内，但只有框面积属于目标（不能按 πr² 报）。"""
    result = check_zero_overlap(0.0, 0.0, 40.0, (0.0, 0.0))

    assert result.area_px == pytest.approx(40.0 * 40.0)
    assert result.pct == pytest.approx(100.0)
    assert result.depth_px == 0.0


def test_unmeasured_zero_position_degrades_gracefully() -> None:
    """没有实测零级时给"未测"结论而不是抛异常（计划页必须先能画出来）。"""
    result = check_zero_overlap(100.0, 100.0, 600.0, None)

    assert result.intersects is False
    assert result.pct == 0.0
    assert "零级位置未测" in result.note


@pytest.mark.parametrize("side,radius", [(0.0, 60.0), (-1.0, 60.0), (600.0, 0.0)])
def test_invalid_lengths_raise(side, radius) -> None:
    """边长/半径非正直接报错（静默返回 0 会把配置错误读成"没有重叠"）。"""
    with pytest.raises(ValueError):
        check_zero_overlap(100.0, 100.0, side, ZERO_XY, zero_radius=radius)


def _numeric_disc_box_area(cx, cy, radius, x0, x1, y0, y1, samples: int = 40001) -> float:
    """数值参照：沿 x 积「圆被框裁过的弦长」（梯形法，误差 ≪ 1 % 圆面积）。"""
    xs = np.linspace(x0, x1, samples)
    half = np.sqrt(np.maximum(radius * radius - (xs - cx) ** 2, 0.0))
    upper = np.minimum(y1, cy + half)
    lower = np.maximum(y0, cy - half)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(np.clip(upper - lower, 0.0, None), xs))
