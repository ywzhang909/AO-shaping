"""录制模型上的 FFT 可达窗验收（离线，不开 SLM / 相机 / 激光）。

- ``hw_20261106_daheng/model``（grid=640）：side=1800 的三角形**必须**能被种出来，
  且 FFT 目标不是"整窗全 1"——全 1 正是目标被裁到窗外的症状。
- ``20261107_174854/fit``（grid=128）：同一个目标必须被硬保护拒绝。

两个目录不在时整组 skip（数据不进仓库的机器上不应失败）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    load_frozen_parameters,
    recentre_window,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    assert_target_within_fft_window,
    fft_grid_target,
)
from ao_shaping.tools.slm.transfer.geometric_seed import target_backprojection_phase

ROOT = Path(__file__).resolve().parents[5]
WIDE_MODEL = ROOT / "data/debug/slm_transfer/diff_square_grow/hw_20261106_daheng/model"
NARROW_MODEL = ROOT / "data/debug/slm_transfer/20261107_174854/fit"
SIDE = 1800.0
PERIOD = 120.0

# 真实 640/2048 物理预测模型的构造与反投影约需二三十秒：不进默认 fast 路径。
pytestmark = pytest.mark.slow


def _forward(directory: Path):
    if not (directory / "frozen_parameters.json").is_file():
        pytest.skip(f"录制模型不在本机: {directory}")
    params = load_frozen_parameters(directory)
    origin = tuple(float(v) for v in params.origin_xy)
    xy = recentre_window(origin, half_span_px=220, step_px=4)
    return FrozenDefocusForward(params, xy, device="cpu")


def _target_centre(forward):
    origin = np.asarray(forward.params.origin_xy, dtype=float)
    ramp = np.asarray(forward.params.ramp_matrix, dtype=float)
    return origin + ramp[:, 0] / PERIOD


def test_wide_recorded_model_shapes_triangle_1800():
    forward = _forward(WIDE_MODEL)
    assert forward.params.grid_size == 640
    centre = _target_centre(forward)
    target = fft_grid_target(forward, centre, SIDE, "triangle")
    coverage = float(np.mean(target > 0.5))
    # 三角形样：有真实占比，且**绝不是**整窗覆盖（被裁出窗时 coverage≈1）。
    assert 0.05 < coverage < 0.5
    assert coverage < 0.95
    phase, diagnostics = target_backprojection_phase(forward, centre, SIDE, "triangle")
    assert phase.shape == (640, 640)
    assert np.all(np.isfinite(phase))
    assert diagnostics["target_amplitude_sum"] > 0.0


def test_narrow_recorded_model_rejects_triangle_1800():
    forward = _forward(NARROW_MODEL)
    assert forward.params.grid_size == 128
    with pytest.raises(ValueError, match="FFT 网格可达窗口"):
        assert_target_within_fft_window(forward, _target_centre(forward), SIDE, "triangle")
