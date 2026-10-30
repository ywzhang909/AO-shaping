"""实测帧 vs 目标掩膜的整形相似度 (``target_pearson``)。

为什么需要这个读法: ``cv`` 只描述目标内的均匀程度, ``ee_roi`` 只描述目标内
占比, 两者都不回答「远场整体像不像这个形状」——一个把光全部堆在最中心的
正方形目标可以把 ``ee_roi`` 做高、把 ``cv`` 做低, 却明显不是方斑。Pearson
在**全窗口**上算, 目标之外的光也进入分母, 因此对「没填满」和「四角漏光」
都有响应。

守卫该读法的两条退化行为:
1. 全暗帧 / 常数目标 → 方差为 0 → 返回 ``None``, 不是 0.0;
2. 形状网格不一致 → 报错, 不做隐式裁剪。
"""

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.multishape_objective import (
    ShapeObjective,
    image_metrics,
    target_pearson,
)


def _square_target(window: int = 64, half_side: int = 10) -> np.ndarray:
    yy, xx = np.mgrid[:window, :window]
    centre = (window - 1) / 2.0
    inside = (np.abs(xx - centre) <= half_side) & (np.abs(yy - centre) <= half_side)
    return inside.astype(np.float32)


def test_matches_identical_shape() -> None:
    target = _square_target()
    assert target_pearson(target, target) == pytest.approx(1.0, abs=1e-6)


def test_is_invariant_to_affine_rescaling() -> None:
    """归一化/增益/本底偏置不改变 Pearson —— 它只量形状, 不量亮度。"""
    target = _square_target()
    frame = target * 37.0 + 5.0
    assert target_pearson(frame, target) == pytest.approx(1.0, abs=1e-6)


def test_peak_only_inside_target_scores_lower_than_filled_box() -> None:
    """集中在目标中心的亮斑「像方斑」的程度, 必须低于填满的目标。"""
    target = _square_target()
    filled = target.copy()
    peak_only = np.zeros_like(target)
    peak_only[31:33, 31:33] = 1.0

    filled_score = target_pearson(filled, target)
    peak_score = target_pearson(peak_only, target)

    assert filled_score > 0.99
    assert peak_score < filled_score


def test_light_outside_target_lowers_score() -> None:
    target = _square_target()
    frame = target.copy()
    frame[::4, ::4] += 3.0  # 稀疏但明显的目标外漏光
    assert target_pearson(frame, target) < 0.99


def test_degenerate_inputs_return_none_not_zero() -> None:
    target = _square_target()
    assert target_pearson(np.zeros_like(target), target) is None
    assert target_pearson(target, np.zeros_like(target)) is None


def test_grid_mismatch_raises() -> None:
    target = _square_target()
    with pytest.raises(ValueError):
        target_pearson(target[:-1], target)


def test_image_metrics_exposes_the_key() -> None:
    """``image_metrics`` 是 summary/results.json 这类记录的单一来源。"""
    target = _square_target()
    objective = ShapeObjective(target)
    metrics = image_metrics((target * 100.0).astype(np.float32), objective)

    assert set(metrics) == {"cv", "mean", "ee_roi", "target_peak", "target_pearson"}
    assert metrics["target_pearson"] == pytest.approx(1.0, abs=1e-6)
