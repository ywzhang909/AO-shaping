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
    FrozenDefocusParameters,
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


def _small_forward():
    """CPU 友好的小型冻结前向（32 网格 / FFT 64 / 13×13 相机窗）。"""
    params = FrozenDefocusParameters(
        defocus_diopter=-0.1, origin_xy=(1360.0, 1020.0),
        amplitude_center_xy=(960.0, 600.0), amplitude_width_xy=(400.0, 300.0),
        residual_noll_radians={5: 0.3}, gain=25.0, background=1.0, sigma_ccd_px=2.0,
        ramp_matrix=((8000.0, 0.0), (0.0, -8000.0)),
        fft_size=64, pixel_samples=1, grid_size=32,
    )
    offsets = np.arange(-12, 13, 2.0)
    xx, yy = np.meshgrid(1360.0 + offsets, 1020.0 + offsets, indexing="xy")
    camera_xy = np.stack([xx, yy], axis=-1)
    return FrozenDefocusForward(params, camera_xy, device="cpu"), params


def _small_masks():
    box = np.zeros((13, 13), dtype=bool)
    box[2:11, 2:11] = True
    core = np.zeros((13, 13), dtype=bool)
    core[4:9, 4:9] = True
    return box, core


def test_directional_derivative_matches_finite_difference():
    """方向导数：autograd 与中心差分（方案 §7(2)），相对误差 < 1e-3。

    方向取梯度最陡方向（随机方向的内积太小，相对误差被 float32 精度主导）。
    """
    import torch

    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    rng = np.random.default_rng(0)
    phi_np = (rng.standard_normal((32, 32)) * 0.1).astype(np.float32)

    phi = torch.tensor(phi_np, requires_grad=True)
    loss = objective.cv(fwd.predict(phi)[0])
    loss.backward()
    grad = phi.grad
    direction = grad / grad.norm()

    image_leaf = fwd.predict(torch.tensor(phi_np))[0].detach().clone().requires_grad_(True)
    objective.cv(image_leaf).backward()
    grad_image = image_leaf.grad
    step = 1e-3
    with torch.no_grad():
        plus = fwd.predict(torch.tensor(phi_np) + step * direction)[0].flatten()
        minus = fwd.predict(torch.tensor(phi_np) - step * direction)[0].flatten()
    finite_diff = float(((plus - minus) / (2 * step)).dot(grad_image.flatten()))
    autograd_dir = float(grad.flatten().dot(direction.flatten()))
    relative = abs(finite_diff - autograd_dir) / max(abs(autograd_dir), 1e-9)
    assert relative < 1e-3, f"directional derivative drift {relative:.3e}"


def test_coordinate_transform_direction_flag_flips_mapped_direction():
    """方向标记（方案 §7(2)）：映射主元符号翻转 ⇒ 面板倾斜的相机映射方向翻转。"""
    positive = CoordinateTransform(rows=((8000.0, 0.1), (1.0, -8100.0)), sign=(1.0, -1.0))
    flipped = CoordinateTransform(rows=((-8000.0, 0.1), (1.0, -8100.0)), sign=(-1.0, -1.0))
    vector = np.array([1.0, 0.0])
    assert positive.to_camera(vector)[0] > 0
    assert flipped.to_camera(vector)[0] < 0
    assert np.allclose(flipped.to_panel(flipped.to_camera(vector)), vector, atol=1e-9)


def test_measured_anchoring_uses_frozen_model_jacobian():
    """测量锚定同一完整模型 Jacobian（方案 §7(7)）。

    相位梯度 ≠ 0，且与冻结前向（含离焦/展宽/辐照度标度/增益/背景）的有限差一致；
    与**裸理想 FFT**（无离焦/展宽/标度）的 Jacobian 显著不同 —— 证明反传没有
    回落到裸 FFT 路径。
    """
    import torch

    from ao_shaping.algorithm.signal_processing import defocus_forward_core as core
    from ao_shaping.utils.wavefront.fraunhofer import focal_intensity

    fwd, params = _small_forward()
    box, core_mask = _small_masks()
    objective = UniformityObjective(box, core_mask, w_phase=0.0)
    rng = np.random.default_rng(1)
    phi_np = (rng.standard_normal((32, 32)) * 0.1).astype(np.float32)
    with torch.no_grad():
        prediction = fwd.predict(torch.tensor(phi_np))[0].numpy()
    frame = prediction + 0.05 * np.abs(prediction) + 0.1

    phi = torch.tensor(phi_np, requires_grad=True)
    prediction_t = fwd.predict(phi)[0]
    measured = torch.tensor(frame.astype(np.float32)).requires_grad_(True)
    loss, _ = objective.evaluate(measured, torch.zeros((32, 32)))
    loss.backward()
    prediction_t.backward(gradient=measured.grad)
    grad = phi.grad
    assert grad is not None and float(grad.norm()) > 1e-8

    grad_image = measured.grad
    direction = grad / grad.norm()
    step = 1e-3
    with torch.no_grad():
        plus = fwd.predict(phi.detach() + step * direction)[0].flatten()
        minus = fwd.predict(phi.detach() - step * direction)[0].flatten()
    finite_diff = float(((plus - minus) / (2 * step)).dot(grad_image.flatten()))
    autograd_dir = float(grad.flatten().dot(direction.flatten()))
    relative = abs(finite_diff - autograd_dir) / max(abs(autograd_dir), 1e-9)
    assert relative < 1e-3, f"measured-anchored Jacobian drift {relative:.3e}"

    grids = core.camera_sample_grids(
        torch, fwd._xy_t, fwd.origin, fwd.inverse, fwd._grid_offsets,
        fft_size=params.fft_size, pitch=fwd.pitch)

    def raw_sampled(phase):
        field = fwd.amplitude[None] * torch.exp(1j * phase[None])
        return core.sample_intensity(torch, focal_intensity(field, params.fft_size), grids)[0]

    raw_phi = torch.tensor(phi_np, requires_grad=True)
    raw_sampled(raw_phi).backward(gradient=grad_image)
    raw_dir = float(raw_phi.grad.flatten().dot(direction.flatten()))
    contrast = abs(autograd_dir - raw_dir) / max(abs(autograd_dir), 1e-9)
    assert contrast > 0.05, f"raw-FFT Jacobian too close: contrast {contrast:.3e}"


def _anchored_opt(fwd, objective, phi_np, **kwargs):
    return DifferentiableUniformityOptimizer(fwd, objective, phi_np, device="cpu", **kwargs)


def test_anchored_mode_cancels_static_model_offset():
    """Anchored update uses I_anchor + F(phi+delta) - F(phi) as the loss image.

    With a constant additive residual R between the measured frame and the model
    prediction at the anchor, the anchored irradiance at delta=0 equals the
    measured frame, and its phase gradient comes from the frozen model Jacobian.
    """
    import torch

    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    rng = np.random.default_rng(3)
    phi_np = (rng.standard_normal((32, 32)) * 0.1).astype(np.float32)
    with torch.no_grad():
        prediction = fwd.predict(torch.tensor(phi_np))[0].numpy()
    residual = 0.2 * np.abs(prediction) + 0.3
    measured = prediction + residual

    opt = _anchored_opt(fwd, objective, phi_np, lr=0.01, step_rms_cap=0.1)
    opt.update(measured)
    record = opt.loss_history[-1]
    assert record["mode"] == "anchored"
    assert record["gradient_norm"] > 1e-8

    # At delta=0 the anchored image is exactly the measured frame, so the loss
    # matches a direct evaluation on the measurement.
    with torch.no_grad():
        direct, _ = objective.evaluate(
            torch.tensor(measured, dtype=torch.float32), torch.zeros((32, 32)))
    assert record["loss"] == pytest.approx(float(direct), rel=1e-5)


def test_anchored_mode_matches_model_mode_without_residual():
    """When measured == model prediction at the anchor, anchored == model update."""
    import torch

    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    rng = np.random.default_rng(5)
    phi_np = (rng.standard_normal((32, 32)) * 0.1).astype(np.float32)
    with torch.no_grad():
        prediction = fwd.predict(torch.tensor(phi_np))[0].numpy()

    anchored = _anchored_opt(fwd, objective, phi_np, lr=0.01, step_rms_cap=0.1)
    model = _anchored_opt(fwd, objective, phi_np, lr=0.01, step_rms_cap=0.1)
    anchored.update(prediction)
    model.update(None)
    a_rec, m_rec = anchored.loss_history[-1], model.loss_history[-1]
    assert a_rec["mode"] == "anchored" and m_rec["mode"] == "model"
    assert a_rec["loss"] == pytest.approx(m_rec["loss"], rel=1e-5)
    np.testing.assert_allclose(anchored.delta_phase, model.delta_phase, rtol=1e-4, atol=1e-6)
