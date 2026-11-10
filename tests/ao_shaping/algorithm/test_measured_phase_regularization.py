"""Anchored update retains both direct phase and model image gradients."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (  # noqa: E402
    DifferentiableUniformityOptimizer,
    UniformityObjective,
)


class _Forward:
    """A nonzero model VJP, without data files or an optical checkpoint."""
    class params:
        grid_size = 8

    _device = torch.device("cpu")

    def illumination_weights(self):
        return np.ones((8, 8), dtype=np.float32) / 64

    def predict(self, phase, **kwargs):
        return (2 + .1 * phase)[None]

    def total_phase_grid(self, phase, **kwargs):
        return phase


@pytest.mark.parametrize("measured_mode", [False, True])
def test_regularizer_and_image_gradients_are_both_accumulated(measured_mode):
    mask = np.ones((8, 8), dtype=bool)
    objective = UniformityObjective(mask, mask, w_phase=2.)
    optimizer = DifferentiableUniformityOptimizer(
        _Forward(), objective, np.zeros((8, 8)), lr=.001, step_rms_cap=1.)
    delta = torch.tensor(np.random.default_rng(3).normal(0, .2, (8, 8)),
                         dtype=torch.float32)
    with torch.no_grad():
        optimizer._delta.copy_(delta)
    frame = 2 + .1 * delta.numpy()
    expected_delta = delta.clone().requires_grad_(True)
    if measured_mode:
        # Anchored: I = frame + F(seed+delta) - F(seed+accepted). With accepted=0
        # and F(phi)=2+.1*phi, baseline=2, so I = (2+.1*d)+(2+.1*d)-2 = 2+.2*d.
        # Build it along the same float32 path as the implementation so the
        # gradient comparison is exact rather than dominated by rounding.
        prediction = (2 + .1 * expected_delta)[None][0]
        baseline = (2 + .1 * torch.zeros((8, 8)))[None][0]
        expected_image = (torch.tensor(frame) + prediction - baseline).clamp_min(1e-8)
    else:
        expected_image = 2 + .1 * expected_delta
    expected, _ = objective.evaluate(expected_image, expected_delta)
    expected.backward()
    optimizer.update(frame if measured_mode else None)
    torch.testing.assert_close(optimizer._delta.grad, expected_delta.grad)
    assert optimizer._delta.grad.norm() > .01
    record = optimizer.loss_history[-1]
    for key in ("anchor_model_loss", "candidate_model_loss",
                "candidate_model_loss_change", "raw_to_final_step_cosine"):
        assert np.isfinite(record[key])
    assert record["candidate_model_loss_change"] == pytest.approx(
        record["candidate_model_loss"] - record["anchor_model_loss"])


class _CountingForward(_Forward):
    """``_Forward`` that records how often ``predict`` is called.

    Returns a 3D ``(1, 8, 8)`` prediction, the shape that exercises the
    post-step anchor/candidate squeeze path in ``update``.
    """

    def __init__(self):
        self.calls = 0
        super().__init__()

    def predict(self, phase, **kwargs):
        self.calls += 1
        return (2 + .1 * phase)[None]


def test_anchored_reuses_baseline_forward_not_a_fourth_call():
    """Regression: anchored mode must reuse ``anchor_cache``, not recompute it.

    A single ``update(measured)`` runs exactly **3** forward passes — main
    (seed+delta), no-grad baseline (seed+accepted), and post-step candidate
    (seed+delta) — and the post-step anchor comes from the cached baseline.
    The 4→3 forward reuse must not add a fourth ``predict`` (the old bug).

    This also pins the squeeze fix: a 3D ``(1, 8, 8)`` candidate must be
    squeezed independently of the (already-2D) cached anchor, otherwise
    ``objective.evaluate`` raises on the shape mismatch and the test fails.
    """
    forward = _CountingForward()
    mask = np.ones((8, 8), dtype=bool)
    objective = UniformityObjective(mask, mask, w_phase=2.)
    optimizer = DifferentiableUniformityOptimizer(
        forward, objective, np.zeros((8, 8)), lr=.001, step_rms_cap=1.)
    delta = torch.tensor(np.random.default_rng(7).normal(0, .2, (8, 8)),
                         dtype=torch.float32)
    with torch.no_grad():
        optimizer._delta.copy_(delta)
    frame = (2 + .1 * delta.numpy()).astype(np.float32)

    optimizer.update(frame)

    assert forward.calls == 3, f"anchored update ran {forward.calls} forwards, want 3"
    record = optimizer.loss_history[-1]
    for key in ("anchor_model_loss", "candidate_model_loss",
                "candidate_model_loss_change", "raw_to_final_step_cosine"):
        assert np.isfinite(record[key]), f"{key} not finite"
    assert record["candidate_model_loss_change"] == pytest.approx(
        record["candidate_model_loss"] - record["anchor_model_loss"])
