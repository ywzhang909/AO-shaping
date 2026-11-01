"""自动曝光 / 多帧平均的离线测试 (纯 numpy, 不碰 Qt, 不碰相机)。"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.gui.ccd.beam_quality.exposure_control import (
    AUTO_STATISTIC_KEYS,
    AUTO_STATISTICS,
    DEFAULT_TARGETS,
    FrameAverage,
    frame_statistic,
    is_converged,
    next_exposure,
)


def test_statistics_keys_cover_default_targets() -> None:
    """下拉框顺序、内部键、默认目标三者对齐 (界面靠 index 取值)。"""
    assert len(AUTO_STATISTICS) == len(AUTO_STATISTIC_KEYS) == 3
    assert set(AUTO_STATISTIC_KEYS) == set(DEFAULT_TARGETS)
    assert AUTO_STATISTIC_KEYS == ("max", "median", "mean")


def test_frame_statistic_max_median_mean() -> None:
    """三种统计量各取各的: 最大值 / 中位数 / 均值 (含背景)。"""
    img = np.array([[0, 10], [200, 30]], dtype=np.uint8)
    assert frame_statistic(img, "max") == 200.0
    assert frame_statistic(img, "median") == 20.0
    assert frame_statistic(img, "mean") == pytest.approx(60.0)


def test_frame_statistic_rejects_unknown_kind_and_empty_frame() -> None:
    with pytest.raises(ValueError):
        frame_statistic(np.zeros((2, 2), dtype=np.uint8), "p99")
    with pytest.raises(ValueError):
        frame_statistic(np.empty((0, 0), dtype=np.uint8), "max")


def test_is_converged_uses_relative_tolerance() -> None:
    assert is_converged(203.0, 200.0)  # 1.5%
    assert not is_converged(180.0, 200.0)  # 10%
    assert not is_converged(float("nan"), 200.0)
    assert not is_converged(10.0, 0.0)


def test_next_exposure_returns_none_inside_tolerance() -> None:
    """已进容差带就不再动相机 (MiiCam 改一次曝光要重拉流 ~1s)。"""
    assert next_exposure(201.0, 200.0, 20.0, min_ms=0.011, max_ms=10000.0) is None


def test_next_exposure_halves_and_doubles() -> None:
    """曝光是线性旋钮: 目标/当前 就是一步到位的倍率。"""
    assert next_exposure(
        100.0, 200.0, 20.0, min_ms=0.011, max_ms=10000.0
    ) == pytest.approx(40.0)
    assert next_exposure(
        200.0, 100.0, 20.0, min_ms=0.011, max_ms=10000.0
    ) == pytest.approx(10.0)


def test_next_exposure_clamps_single_step() -> None:
    """过暗 (10 → 200) 时单次最多 4 倍, 免得一次冲过头再往回拉。"""
    assert next_exposure(
        10.0, 200.0, 20.0, min_ms=0.011, max_ms=10000.0
    ) == pytest.approx(80.0)


def test_next_exposure_clamps_to_limits_and_stops_at_limit() -> None:
    """贴到曝光上限后返回 None, 让界面提示"已到曝光极限"而不是空转。"""
    assert next_exposure(
        10.0, 200.0, 100.0, min_ms=0.011, max_ms=250.0
    ) == pytest.approx(250.0)
    assert next_exposure(10.0, 200.0, 250.0, min_ms=0.011, max_ms=250.0) is None
    assert next_exposure(200.0, 10.0, 0.011, min_ms=0.011, max_ms=250.0) is None


def test_next_exposure_rejects_invalid_input() -> None:
    """当前统计量为 0 (全黑) 或 NaN 时无法定倍率, 不动作。"""
    assert next_exposure(0.0, 200.0, 20.0, min_ms=0.011, max_ms=10000.0) is None
    assert (
        next_exposure(float("nan"), 200.0, 20.0, min_ms=0.011, max_ms=10000.0) is None
    )
    assert (
        next_exposure(100.0, 200.0, float("nan"), min_ms=0.011, max_ms=10000.0) is None
    )


def test_frame_average_rolling_window_keeps_last_n() -> None:
    """滑动窗口平均只算最近 size 帧; 用"累加和 - 最旧帧"实现, 不能漏减。"""
    averager = FrameAverage(3)
    for value in (0, 10, 20, 30):
        out = averager.push(np.full((2, 2), value, dtype=np.uint8))
    # 最近 3 帧 = 10, 20, 30
    assert averager.filled == 3
    assert np.allclose(out, 20.0)


def test_frame_average_size_one_is_passthrough() -> None:
    averager = FrameAverage(1)
    frame = np.full((2, 2), 7, dtype=np.uint8)
    assert averager.push(frame) is frame
    assert averager.filled == 0


def test_frame_average_rebuilds_on_shape_change() -> None:
    """换分辨率 (换相机 / binning 开关) 时丢掉旧历史, 不能拿旧尺寸相加。"""
    averager = FrameAverage(4)
    averager.push(np.zeros((8, 8), dtype=np.uint8))
    out = averager.push(np.full((4, 4), 5, dtype=np.uint8))
    assert out.shape == (4, 4)
    assert np.allclose(out, 5.0)
    assert averager.filled == 1


def test_frame_average_reset_clears_history() -> None:
    averager = FrameAverage(3)
    averager.push(np.full((2, 2), 9, dtype=np.uint8))
    averager.reset()
    assert averager.filled == 0
    out = averager.push(np.full((2, 2), 4, dtype=np.uint8))
    assert np.allclose(out, 4.0)
