"""CV/EE 指标两层结构的一致性契约 (issue #199 R3)。

R3 把 "框内 CV + ROI 能量" 收敛为 :mod:`ao_shaping.utils.image.beam_metrics`
的两个 numpy 原语 (:func:`masked_cv` / :func:`masked_energy_fraction`), 四个
numpy 消费者改为薄封装, torch 侧保留可微的薄封装
(:class:`~ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity.UniformityObjective`
的 ``cv`` / :class:`~ao_shaping.algorithm.signal_processing.multishape_objective.ShapeObjective`
的 ``cv2``)。本文件钉住:

* torch 损失 == numpy 原语的**平方** (CV) / 同值 (EE) —— float32 与 float64;
* ``image_metrics`` 与 ``hw_square_carrier_probe.metrics`` 相对 R3 前的内联
  口径逐位一致 (孪生参照实现逐字内联在下面, 防未来漂移);
* 探针盒/核几何经边衬委派后与旧内联盒核逐位一致 (含盒被画幅裁剪的情形),
  以及唯一保留的行为差异 (边长 < 8 px 时旧的 ``max(1, half-3)`` 钳位)。

纯 numpy + torch; 无硬件 / 无驱动。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    UniformityObjective,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    ShapeObjective,
    image_metrics,
)
from ao_shaping.utils.image.beam_metrics import (
    masked_cv,
    masked_energy_fraction,
)
from ao_shaping.utils.image.target.masks import build_box_core_masks
from scripts.slm.hw_square_carrier_probe import metrics as probe_metrics

#: torch 与 numpy 的求和/除法次序不同: float32 相对容差 1e-4, float64 1e-9。
RTOL = {np.float32: 1e-4, np.float64: 1e-9}


def _frame(shape, dtype, seed):
    """正定随机帧 (均值远大于 torch 侧的 clamp_min(eps))。"""
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 100.0 + 1.0).astype(dtype)


# --------------------------------------------------------------------------- #
# torch == numpy (R3 两层结构)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_uniformity_objective_cv_equals_masked_cv_squared(dtype):
    image = _frame((24, 32), dtype, seed=11)
    support = np.random.default_rng(12).random((24, 32)) > 0.35
    objective = UniformityObjective(support, support, w_phase=0.0)

    torch_cv2 = float(objective.cv(torch.as_tensor(image)).detach())
    numpy_cv = masked_cv(image, support)

    assert torch_cv2 == pytest.approx(numpy_cv**2, rel=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shape_objective_losses_equal_numpy_primitives(dtype):
    target = np.zeros((32, 32), dtype=np.float32)
    target[8:24, 8:24] = 1.0
    objective = ShapeObjective(target)
    image = _frame((32, 32), dtype, seed=21)

    _, parts = objective.evaluate(torch.as_tensor(image), torch.zeros((32, 32)))

    assert parts["l_cv"] == pytest.approx(
        masked_cv(image, objective.core) ** 2, rel=RTOL[dtype]
    )
    assert parts["ee_roi"] == pytest.approx(
        masked_energy_fraction(image, objective.box, floor=1e-9), rel=RTOL[dtype]
    )


# --------------------------------------------------------------------------- #
# numpy 薄封装 vs R3 前的内联口径 (孪生参照, 逐字复制)
# --------------------------------------------------------------------------- #
def _pre_r3_image_metrics(image, objective):
    """R3 前 ``multishape_objective.image_metrics`` 的内联口径 (逐字复制)。"""
    core = np.asarray(image)[objective.core]
    mean = float(core.mean())
    return dict(
        cv=float(core.std() / mean) if mean > 0 else None,
        mean=mean,
        ee_roi=float(image[objective.box].sum() / max(float(image.sum()), 1e-9)),
        target_peak=float(image[objective.box].max()),
    )


def _pre_r3_probe_metrics(frame, centre_xy, side_px, zero_xy):
    """R3 前 ``hw_square_carrier_probe.metrics`` 的内联口径 (逐字复制)。"""
    height, width = frame.shape
    col, row = centre_xy
    half = side_px / 2.0
    yy, xx = np.mgrid[:height, :width]
    box = (np.abs(xx - col) <= half) & (np.abs(yy - row) <= half)
    inner = (np.abs(xx - col) <= max(1.0, half - 3.0)) & (
        np.abs(yy - row) <= max(1.0, half - 3.0)
    )
    values = frame[inner]
    mean = float(values.mean()) if values.size else float("nan")
    cv = float(np.sqrt((((values / mean) - 1) ** 2).mean())) if mean > 0 else float("nan")
    zero_col, zero_row = zero_xy
    zero_box = (np.abs(xx - zero_col) <= 8) & (np.abs(yy - zero_row) <= 8)
    return dict(
        target_mean=mean,
        target_peak=float(frame[box].max()) if box.any() else float("nan"),
        cv=float(cv),
        ee=float(frame[box].sum() / max(frame.sum(), 1.0)),
        zero_peak=float(frame[zero_box].max()) if zero_box.any() else float("nan"),
        separation_px=float(np.hypot(col - zero_col, row - zero_row)),
        zero_in_target=float(frame[box & zero_box].sum() / max(frame[box].sum(), 1.0)),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [3, 4])
def test_image_metrics_matches_pre_r3_inline_reference(dtype, seed):
    target = np.zeros((40, 40), dtype=np.float32)
    target[10:30, 10:30] = 1.0
    objective = ShapeObjective(target)
    image = _frame((40, 40), dtype, seed=seed)

    old = _pre_r3_image_metrics(image, objective)
    new = image_metrics(image, objective)

    assert new["mean"] == old["mean"]
    assert new["ee_roi"] == old["ee_roi"]
    assert new["target_peak"] == old["target_peak"]
    assert new["cv"] == old["cv"]  # 正定帧: 精确相等 (含 float32 除法)


@pytest.mark.parametrize("side", [8.0, 42.425, 84.0, 84.85])
@pytest.mark.parametrize("centre", [(64.0, 48.0), (64.5, 47.75), (13.7, 71.3), (0.0, 0.0)])
def test_probe_metrics_matches_pre_r3_inline_reference(side, centre):
    frame = _frame((120, 160), np.float64, seed=5)
    zero_xy = (64.0, 48.0)

    old = _pre_r3_probe_metrics(frame, centre, side, zero_xy)
    new = probe_metrics(frame, centre, side, zero_xy)

    assert new["target_mean"] == old["target_mean"]
    assert new["target_peak"] == old["target_peak"]
    assert new["ee"] == old["ee"]
    assert new["zero_peak"] == old["zero_peak"]
    assert new["separation_px"] == old["separation_px"]
    assert new["zero_in_target"] == old["zero_in_target"]
    # cv: 旧式 sqrt(mean((v/m-1)**2)) 与规范 std/mean 代数等价, 仅末位 ULP。
    assert new["cv"] == pytest.approx(old["cv"], rel=1e-12)


@pytest.mark.parametrize("side", [8.0, 42.425, 84.0, 84.85])
@pytest.mark.parametrize("centre", [(64.0, 48.0), (64.5, 47.75), (13.7, 71.3), (0.0, 0.0)])
def test_probe_box_and_core_match_old_distance_formula_bitwise(side, centre):
    """边衬委派与旧内联盒核逐位一致 (含盒被画幅裁剪、中心为半整数的情形)。

    ``_erode`` 除盒边界外还腐蚀数组边界, 因此探针按内缩量边衬后再委托 ——
    否则画幅裁剪会与内缩叠加, 比旧公式多删 3 px。
    """
    height, width = 120, 160
    col, row = centre
    half = side / 2.0
    yy, xx = np.mgrid[:height, :width]
    old_box = (np.abs(xx - col) <= half) & (np.abs(yy - row) <= half)
    old_inner = (np.abs(xx - col) <= max(1.0, half - 3.0)) & (
        np.abs(yy - row) <= max(1.0, half - 3.0)
    )

    pad = 3
    box_p, inner_p = build_box_core_masks(
        (height + 2 * pad, width + 2 * pad),
        (col + pad, row + pad),
        side,
        float(pad),
        (1.0, 1.0),
    )

    assert np.array_equal(box_p[pad : pad + height, pad : pad + width], old_box)
    assert np.array_equal(inner_p[pad : pad + height, pad : pad + width], old_inner)


@pytest.mark.parametrize("side", [5.0, 6.0, 7.0])
def test_probe_small_side_divergence_is_documented(side):
    """边长 < 8 px: 旧的 ``max(1, half-3)`` 钳位不再保留 (腐蚀语义)。

    这是 R3 在探针上唯一保留的行为差异; 实际目标量级是 PSF FWHM 24 px。
    """
    frame = _frame((120, 160), np.float64, seed=6)
    centre, zero_xy = (80.0, 60.0), (80.0, 60.0)

    old = _pre_r3_probe_metrics(frame, centre, side, zero_xy)
    new = probe_metrics(frame, centre, side, zero_xy)

    assert new["target_mean"] != old["target_mean"]
    if side <= 5.0:
        assert math.isnan(new["target_mean"])
        assert math.isnan(new["cv"])
        assert math.isfinite(new["target_peak"])
