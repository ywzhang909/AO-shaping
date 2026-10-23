"""Physical invariants for sampling and exposure-offset diagnostics."""
import json

import numpy as np
import pytest
import torch

from scripts.slm.diagnose_defocus_sampling import response_bias
from scripts.slm.fit_defocus_forward_dataset import PhysicalForwardFit
from scripts.slm.refit_defocus_uniformity_model import supervised_loss


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

def test_fft_refinement_preserves_physical_peak_irradiance():
    kwargs = dict(phases=np.zeros((1, 128, 128), np.float32), images=np.ones((1, 1, 1)),
                  camera_xy=np.array([[[1360., 1020.]]]), matrix=np.array([[0., 7400.], [-7500., 0.]]),
                  focus_diopter=0., train_mask=[True], amplitude_center=[950., 750.])
    a = PhysicalForwardFit(**kwargs, fft_size=512)
    b = PhysicalForwardFit(**kwargs, fft_size=1024)
    np.testing.assert_allclose(a.prediction().detach(), b.prediction().detach(), rtol=1e-5)


def test_pixel_integral_reduces_peak_with_no_change_to_command():
    phases = np.zeros((1, 64, 64), np.float32)
    kwargs = dict(phases=phases, images=np.ones((1, 1, 1)), camera_xy=np.array([[[1360., 1020.]]]),
                  matrix=np.array([[0., 7400.], [-7500., 0.]]), focus_diopter=0.,
                  train_mask=[True], amplitude_center=[950., 750.], fft_size=1024)
    point = PhysicalForwardFit(**kwargs)
    integral = PhysicalForwardFit(**kwargs, pixel_samples=3)
    assert integral.prediction().item() < point.prediction().item()
    np.testing.assert_array_equal(phases, 0)


def test_shared_response_intercept_predicts_repeated_exposure(tmp_path):
    rows = []
    offset = np.arange(9).reshape(3, 3)*.05+.3
    for b, branch in enumerate(("C", "U")):
        for j, exposure in enumerate((1., 2., .5, 1.)):
            number = 4*b+j
            frame = offset+exposure*(2+b)
            file = f"{number}.npz"
            np.savez(tmp_path/file, raw_frames=np.stack([frame]*3))
            rows.append(dict(index=number, branch=branch, exposure_ms=exposure, file=file))
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    y, x = np.mgrid[:3, :3]
    result = response_bias(tmp_path, np.stack((x, y), axis=-1))
    np.testing.assert_allclose(result["intercept"], offset, atol=1e-10)
    np.testing.assert_allclose(result["predicted"][result["validation_mask"]],
                               result["frames"][result["validation_mask"]], atol=1e-10)


def test_uniformity_loss_rejects_blank_and_has_finite_gradient():
    target = torch.tensor([[[1., 2.], [3., 4.]]])
    masks = torch.ones_like(target)
    exact, _, _ = supervised_loss(target, target, masks)
    predicted = torch.full_like(target, .1, requires_grad=True)
    wrong, _, _ = supervised_loss(predicted, target, masks)
    wrong.backward()
    assert exact.item() == 0
    assert wrong.item() > 1
    assert torch.isfinite(predicted.grad).all()
