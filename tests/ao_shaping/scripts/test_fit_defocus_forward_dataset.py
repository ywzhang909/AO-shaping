"""Simulation first: fixed-axis calibration and differentiable residual fit."""
import numpy as np
import pytest
import torch

from scripts.slm.fit_defocus_forward_dataset import PhysicalForwardFit, ramp_matrix


@pytest.fixture(autouse=True)
def _four_threads_then_restore():
    """Pin 4 threads for determinism, then hand the conftest budget back.

    The suite asserts the process-wide thread budget (tests/conftest.py) from a
    later module, so a bare ``set_num_threads(4)`` here silently reddens that
    check. Scoped rather than global: ``torch`` has no per-test isolation.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield
    finally:
        torch.set_num_threads(previous)

def test_ramp_matrix_recovers_axis_exchange():
    rows = []
    matrix = np.array([[120., 7400.], [-7500., 80.]])
    for index, axis in enumerate(("x", "y")):
        for period in (480, 960):
            for sign in (-1, 1):
                point = np.array([1300., 1000.]) + sign * matrix[:, index] / period
                rows.append(dict(label=f"ramp_{axis}_{period}_{sign}",
                                 spot=dict(centroid_x=point[0], centroid_y=point[1])))
    np.testing.assert_allclose(ramp_matrix(rows), matrix)


def test_physical_fit_improves_synthetic_defocus_without_test_frames():
    rng = np.random.default_rng(19)
    phases = rng.normal(0, .3, (3, 128, 128)).astype(np.float32)
    y, x = np.mgrid[-40:41:2, -40:41:2]
    xy = np.stack((x + 1360, y + 1020), axis=-1)
    kwargs = dict(camera_xy=xy, matrix=np.array([[0., 7400.], [-7500., 0.]]),
                  focus_diopter=.1, train_mask=[True, True, False], amplitude_center=[960., 960.])
    truth = PhysicalForwardFit(phases, np.ones((3, 41, 41)), **kwargs, initial_defocus=-.05)
    images = truth.prediction().detach().numpy()
    model = PhysicalForwardFit(phases, images, **kwargs)
    first = model.update()
    for _ in range(35):
        last = model.update()
    assert np.isfinite(last)
    assert last < first * .5
    assert abs(model.d.item() + .05) < .015


def test_residual_basis_has_no_second_defocus_and_receives_gradient():
    y, x = np.mgrid[-30:31:2, -30:31:2]
    model = PhysicalForwardFit(np.zeros((1, 128, 128), np.float32), np.ones((1, 31, 31)),
                               np.stack((x+1360, y+1020), axis=-1),
                               np.array([[0., 7400.], [-7500., 0.]]), .1, [True],
                               [1000., 800.], learn_residual=True)
    assert model.residual_modes == list(range(5, 12))
    with torch.no_grad():
        model.residual[2] = .4
    model.prediction().square().sum().backward()
    assert torch.isfinite(model.residual.grad).all()
    assert model.residual.grad.abs().max() > 0
