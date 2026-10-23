"""载波位移判定的回归测试。

锁三件事：
1. 位移**方向与量值**从两帧实测得到（不是从 K/周期 假定轴向）；
2. 面板 x 向斜坡 + 竖向安装 ⇒ CCD **y** 向位移这种情形能被正确判定（本系统实况）；
3. 没有可辨识位移时返回 ``None``，**不许**瞎给一个方向。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.displacement import measure_displacement, shift_axis_label


def _gauss(shape: tuple[int, int], cx: float, cy: float, amp: float = 100.0, sigma: float = 8.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))


def test_detects_pure_y_shift() -> None:
    """竖向安装实况：光斑整体从 y=957 移到 y=723（dx≈0）。"""
    shape = (1944, 2592)
    best = _gauss(shape, 1220, 957)
    carrier = _gauss(shape, 1226, 723)
    result = measure_displacement(best, carrier)
    assert result is not None
    assert result["dominant_axis"] == "y"
    assert result["dy"] == pytest.approx(-234.0, abs=2.0)
    assert abs(result["dx"]) < 12.0
    assert result["magnitude"] == pytest.approx(234.0, abs=2.0)


def test_detects_pure_x_shift() -> None:
    shape = (600, 800)
    best = _gauss(shape, 400, 300)
    carrier = _gauss(shape, 500, 302)
    result = measure_displacement(best, carrier)
    assert result is not None
    assert result["dominant_axis"] == "x"
    assert result["dx"] == pytest.approx(100.0, abs=3.0)


def test_shift_axis_label_names_the_measured_axis() -> None:
    y_case = {"dx": 6.6, "dy": -234.0, "magnitude": 234.1, "dominant_axis": "y"}
    label = shift_axis_label(y_case)
    assert label.startswith("−y 方向")
    assert "234" in label
    assert "x 方向" not in label

    x_case = {"dx": 120.0, "dy": 2.0, "magnitude": 120.0, "dominant_axis": "x"}
    assert shift_axis_label(x_case).startswith("+x 方向")


def test_undetectable_shift_returns_none() -> None:
    """载波帧只是整体变亮/变暗（无位移）⇒ 必须返回 None，不能编一个方向。"""
    shape = (400, 400)
    best = _gauss(shape, 200, 200)
    carrier = 1.02 * best
    assert measure_displacement(best, carrier, min_positive_fraction=0.02) is None


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        measure_displacement(np.zeros((10, 10)), np.zeros((10, 12)))
