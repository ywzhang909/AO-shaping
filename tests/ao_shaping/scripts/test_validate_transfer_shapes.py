"""Physical-coordinate target masks and differentiable objective checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/slm/validate_transfer_shapes.py"
spec = importlib.util.spec_from_file_location("validate_transfer_shapes", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("shape", ["square", "rectangle", "triangle", "star"])
def test_target_is_at_camera_coordinates_and_separated_from_zero(shape):
    yy, xx = np.mgrid[700:1101:2, 1000:1401:2]
    xy = np.stack((xx, yy), axis=-1)
    target = module.target_at(xy, shape, (1220., 820.), 80.)
    assert target.sum() > 20
    assert target[131, 110] == 0  # zero order at x=1220, y=962
    active_y, active_x = np.nonzero(target)
    assert abs(np.mean(xx[active_y, active_x])-1220) < 3
    assert abs(np.mean(yy[active_y, active_x])-820) < 15


def test_uniform_target_scores_better_than_uniform_background():
    import torch

    yy, xx = np.mgrid[:80, :80]
    target = ((abs(xx-40) < 15) & (abs(yy-40) < 15)).astype(np.float32)
    objective = module.ShapeObjective(target)
    delta = torch.zeros((32, 32))
    perfect, _ = objective.evaluate(torch.tensor(target), delta)
    background, _ = objective.evaluate(torch.ones((80, 80)), delta)
    assert float(perfect) < 1e-6
    assert float(background) > 1


def test_shape_loss_gradient_matches_finite_difference():
    import torch

    yy, xx = np.mgrid[:40, :40]
    target = ((abs(xx-20) < 10) & (abs(yy-20) < 10)).astype(np.float32)
    objective = module.ShapeObjective(target)
    rng = np.random.default_rng(44)
    image = torch.tensor(rng.uniform(.5, 2., (40, 40)), dtype=torch.float64, requires_grad=True)
    direction = torch.tensor(rng.normal(size=(40, 40)), dtype=torch.float64)
    delta = torch.zeros((16, 16), dtype=torch.float64)
    loss, _ = objective.evaluate(image, delta)
    gradient = torch.autograd.grad(loss, image)[0]
    predicted = float((gradient*direction).sum())
    with torch.no_grad():
        plus, _ = objective.evaluate(image+.0001*direction, delta)
        minus, _ = objective.evaluate(image-.0001*direction, delta)
    assert float((plus-minus)/.0002) == pytest.approx(predicted, rel=1e-5, abs=1e-7)


def test_measured_anchor_preserves_reference_and_model_increment():
    measured = np.array([[4., 9.], [3., 6.]], dtype=np.float32)
    prediction = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    np.testing.assert_allclose(module.anchored_intensity(measured, prediction, prediction), measured)
    step = np.array([[.1, -.2], [.3, -.4]], dtype=np.float32)
    np.testing.assert_allclose(module.anchored_intensity(measured, prediction+step, prediction), measured+step)
    with pytest.raises(ValueError, match="identical camera grid"):
        module.anchored_intensity(measured, prediction[:1], prediction)


def test_replay_refuses_different_target_before_device_use(tmp_path):
    import json

    xy = np.zeros((5, 5, 2))
    target = np.ones((5, 5))
    phase = np.ones((8, 8)) * .2
    np.savez(tmp_path / "saved.npz", camera_xy=xy, target=target, model_phase=phase)
    (tmp_path / "manifest.json").write_text(json.dumps([dict(label="square_final", file="saved.npz")]))
    np.testing.assert_array_equal(module.replay_seed(tmp_path, "square", xy, target), phase)
    with pytest.raises(ValueError, match="grid/target differs"):
        module.replay_seed(tmp_path, "square", xy, target * 0)
