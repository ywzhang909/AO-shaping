"""Synthetic response checks; no device construction is permitted here."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

PATH = (
    Path(__file__).resolve().parents[3] / "scripts/ml/sim2f_local_forward_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("local_forward_experiment", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_autodiff_update_improves_independent_nonlinear_response() -> None:
    target, support = MODULE._target_mask("ring", 32)
    yy, xx = np.indices((32, 32))
    reference = (3 + 30 * np.exp(-((xx - 15.5) ** 2 + (yy - 15.5) ** 2) / 8)).astype(
        np.float32
    )
    # A known command moves power from the centre to the requested ring.
    response = 8 * support.astype(np.float32) - (reference - 3) * 0.4
    instrument = lambda c: reference + np.tanh(c) * response
    step = 0.1
    measured_jacobian = (instrument(step) - instrument(-step)) / (2 * step)
    model = MODULE.LocalForwardModel(reference, measured_jacobian[None])
    basis = np.ones((1, 16, 16), dtype=np.float32)
    coefficients, _ = model.optimise(target, support, basis, radius=0.4)
    assert 0 < coefficients[0] <= 0.40001
    before = MODULE.metrics(instrument(0), target, support)
    after = MODULE.metrics(instrument(coefficients[0]), target, support)
    assert after["objective"] < before["objective"] - 0.1
    assert after["ee"] > before["ee"]


def test_forward_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(31)
    reference = np.full((8, 8), 20, dtype=np.float32)
    jacobian = rng.normal(size=(3, 8, 8)).astype(np.float32)
    model = MODULE.LocalForwardModel(reference, jacobian)
    command = torch.tensor([0.02, -0.03, 0.04], requires_grad=True)
    weights = torch.as_tensor(rng.normal(size=(8, 8)), dtype=torch.float32)
    loss = (model.forward(command) * weights).sum()
    loss.backward()
    np.testing.assert_allclose(
        command.grad.numpy(),
        np.einsum("khw,hw->k", jacobian, weights.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


def test_gs_seed_uses_camera_pitch_and_continuous_panel_phase() -> None:
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f
    from scripts.ml.sim2f_fov_hw_session import _gs_phase

    torch.set_num_threads(4)
    bench = Sim2f(Sim2fConfig(), device="cpu")
    target, support = MODULE._target_mask("ring", 256)
    phase = _gs_phase(target, bench)
    image = bench.forward(torch.as_tensor(phase)[None], detector=False)[0, 0].numpy()
    result = MODULE.metrics(image, target, support)
    assert result["ee"] > 0.4, result
    assert result["ncc"] > 0.3, result
    # A phase restricted to [-pi, pi] cannot survive scalar interpolation at
    # wrap edges. The generated seed must preserve its continuous branches.
    assert np.ptp(phase) > 2 * np.pi
