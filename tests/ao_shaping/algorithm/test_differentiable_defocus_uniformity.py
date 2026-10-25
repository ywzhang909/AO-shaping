"""核心可微模块的离线验证（CPU，不碰硬件）。

* 冻结前向复现存档预测（相对 L1 ≤ 1e-5；实测 9.976e-08）；
* CV 目标有解析值；掩膜几何可核对；
* 相位增量按**照明加权 RMS** 限制；去 piston/倾斜；
* 坐标变换与拟合 ``ramp_matrix`` 的 90° 交换必须能被判出来。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    CoordinateTransform,
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    UniformityObjective,
    load_frozen_parameters,
)
from ao_shaping.utils.image.target.masks import build_box_core_masks

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "data/debug/physical_forward_defocus/uniformity_refit_20261022_r1"
RUN_DIR = ROOT / "data/debug/defocus_differentiable/20261023_101822"

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="need the fitted checkpoint on this machine")


@pytest.fixture(scope="module")
def checkpoint():
    with np.load(CHECKPOINT / "inputs.npz") as data:
        inputs = {key: data[key] for key in data.files}
    with np.load(CHECKPOINT / "prediction.npz") as data:
        inputs["predicted"] = data["predicted"]
    inputs["parameters"] = load_frozen_parameters(CHECKPOINT)
    return inputs


@pytest.fixture(scope="module")
def forward(checkpoint):
    return FrozenDefocusForward(checkpoint["parameters"], checkpoint["camera_xy"], device="cpu")


def test_frozen_forward_reproduces_stored_prediction(checkpoint, forward):
    import torch

    with torch.no_grad():
        recon = forward.predict(checkpoint["phases"][:6]).numpy()
    stored = checkpoint["predicted"][:6]
    relative = float(np.abs(recon - stored).sum() / np.abs(stored).sum())
    assert relative < 1e-5, f"frozen forward drift {relative:.3e}"


def test_uniformity_objective_analytic_values():
    import torch

    mask = np.ones((4, 4), dtype=bool)
    objective = UniformityObjective(mask, mask, w_phase=0.0)
    uniform = torch.full((4, 4), 3.5)
    assert float(objective.cv(uniform)) == pytest.approx(0.0, abs=1e-12)

    small = np.ones((2, 2), dtype=bool)
    objective2 = UniformityObjective(small, small, w_phase=0.0)
    pattern = torch.tensor([[1.0, 3.0], [1.0, 3.0]])
    expected = ((0.5 - 1.0) ** 2 + (1.5 - 1.0) ** 2) / 2.0
    assert float(objective2.cv(pattern)) == pytest.approx(float(expected), rel=1e-6)


def test_build_box_core_masks_uses_frozen_geometry():
    box, core = build_box_core_masks((121, 121), (96, 33), box_px=60, inner_shrink_px=3, step_ccd_px=(2.0, 2.0))
    assert int(box.sum()) == 31 * 31
    assert int(core.sum()) == 27 * 27
    assert box[33, 96] and core[33, 96]


def test_coordinate_transform_round_trip_and_verdict():
    transform = CoordinateTransform(rows=((8083.34, 0.17), (1.08, -8136.20)), sign=(1.0, -1.0))
    vector = np.array([0.0, 1.0])
    assert np.allclose(transform.to_panel(transform.to_camera(vector)), vector, atol=1e-9)

    fitted = [[465.146, 7174.691], [-7533.176, -30.278]]
    verdict = transform.compare_with_ramp_matrix(fitted)
    assert verdict["same_axes"] is False
    assert "90-degree swap" in verdict["verdict"]

    aligned = CoordinateTransform(rows=((8000.0, 10.0), (5.0, -8100.0)), sign=(1.0, -1.0))
    assert aligned.compare_with_ramp_matrix([[900.0, 1.0], [0.0, -950.0]])["same_axes"] is True


def test_increment_projection_respects_illumination_weighted_rms(checkpoint, forward):
    col, row = forward.sample_grid_index(1432.0, 968.0)
    box, core = build_box_core_masks(forward.camera_xy.shape[:2], (col, row), 60, 3, forward.sample_step_ccd())
    objective = UniformityObjective(box, core, w_phase=0.0)
    seed = checkpoint["phases"][0].astype(np.float32)
    optimizer = DifferentiableUniformityOptimizer(forward, objective, seed, lr=0.5, step_rms_cap=1e-4,
                                                 device="cpu")
    optimizer.update()
    record = optimizer.loss_history[-1]
    assert record["increment_rms"] <= 1e-4 + 1e-9
    assert record["increment_rms_before_projection"] > 1e-4


def test_optimizer_removes_piston_and_tilt(checkpoint, forward):
    import torch

    col, row = forward.sample_grid_index(1432.0, 968.0)
    box, core = build_box_core_masks(forward.camera_xy.shape[:2], (col, row), 60, 3, forward.sample_step_ccd())
    objective = UniformityObjective(box, core, w_phase=0.0)
    optimizer = DifferentiableUniformityOptimizer(forward, objective, checkpoint["phases"][0], lr=0.5,
                                                 step_rms_cap=0.5, device="cpu")
    with torch.no_grad():
        optimizer._delta.add_(torch.linspace(-1.0, 1.0, optimizer._delta.shape[0])[:, None])
        optimizer._delta.add_(0.7)
    optimizer._remove_piston_and_tilt()
    delta = optimizer.delta_phase
    weights = forward.illumination_weights()
    assert abs(float((weights * delta).sum())) < 1e-6
    w = weights.sum(1)
    active = w > 0
    profile = (delta * weights).sum(1)[active] / w[active]
    grid = np.arange(delta.shape[0], dtype=float)[active]
    w = w[active]
    slope = float(np.sum(w * (grid - np.average(grid, weights=w)) * profile)
                  / np.sum(w * (grid - np.average(grid, weights=w)) ** 2))
    assert abs(slope) < 1e-6


def test_optimizer_decreases_cv_on_model_prediction(checkpoint, forward):
    col, row = forward.sample_grid_index(1432.0, 968.0)
    box, core = build_box_core_masks(forward.camera_xy.shape[:2], (col, row), 60, 3, forward.sample_step_ccd())
    objective = UniformityObjective(box, core, w_phase=0.0)
    optimizer = DifferentiableUniformityOptimizer(forward, objective, checkpoint["phases"][0], lr=0.02,
                                                 step_rms_cap=0.05, device="cpu")
    for _ in range(3):
        optimizer.update()
    losses = [record["l_cv"] for record in optimizer.loss_history]
    assert losses[-1] < losses[0], losses


def test_checkpoint_axis_map_disagreement_is_reported():
    config_path = RUN_DIR / "stage_a/chirp_tilt/stage_a/run_config.json"
    if not config_path.exists():
        pytest.skip("chirp-tilt run not present")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    transform = CoordinateTransform.from_axis_map(config["chirp_tilt_axis_map"])
    fitted = [[465.146, 7174.691], [-7533.176, -30.278]]
    verdict = transform.compare_with_ramp_matrix(fitted)
    assert verdict["verdict"].startswith("axis map DISAGREES")
