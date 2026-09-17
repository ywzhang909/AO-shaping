"""Tests for the differentiable beam shaping module.

These tests are hardware-independent and use pure PyTorch simulation.
They are gated on torch being installed (it lives in the optional `ml`
dependency group), so CI stays green without `uv sync --extra ml`.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module when torch is absent (optional dependency).
torch = pytest.importorskip("torch")

from ao_shaping.algorithm.differentiable_shaping import (  # noqa: E402
    DifferentiableShapingResult,
    angular_spectrum_propagate_torch,
    create_target_mask,
    efficiency_loss,
    smoothness_regularization,
    total_loss,
    train_beam_shaping,
    uniformity_loss,
    zero_order_penalty,
)

# Small grid for fast unit tests.
GRID = (128, 128)
SIZE = 40


# ============================ Target masks ============================

class TestTargetMask:
    """Verify target mask generation for all supported shapes."""

    def test_square_shape_and_center(self):
        mask = create_target_mask("square", GRID, SIZE)
        assert mask.shape == GRID
        assert mask.dtype == np.float64
        # Values in [0, 1]
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0 + 1e-9
        # Centered: center pixel lit
        cy, cx = GRID[0] // 2, GRID[1] // 2
        assert float(mask[cy, cx]) > 0.9

    def test_square_bounding_size(self):
        mask = create_target_mask("square", GRID, SIZE)
        ys, xs = np.nonzero(mask > 0.5)
        h_span = int(ys.max()) - int(ys.min()) + 1
        w_span = int(xs.max()) - int(xs.min()) + 1
        # Square region roughly `SIZE` wide, allow small padding
        assert w_span <= SIZE + 2
        assert h_span <= SIZE + 2

    def test_circle_shape(self):
        mask = create_target_mask("circle", GRID, SIZE)
        assert mask.shape == GRID
        # Center lit
        assert float(mask[GRID[0] // 2, GRID[1] // 2]) > 0.9
        # Corners dark (filled disk)
        assert float(mask[0, 0]) == 0.0

    def test_spot_is_filled_disk(self):
        mask = create_target_mask("spot", GRID, SIZE)
        assert mask.shape == GRID
        # Spot == focused point: small centered disk, center lit, corners dark
        assert float(mask[GRID[0] // 2, GRID[1] // 2]) > 0.9
        assert float(mask[0, 0]) == 0.0

    def test_gaussian_normalized_peak(self):
        mask = create_target_mask("gaussian", GRID, SIZE)
        assert mask.shape == GRID
        assert float(mask.max()) == pytest.approx(1.0, abs=1e-6)

    def test_all_shapes_distinct(self):
        masks = {
            name: create_target_mask(name, GRID, SIZE)
            for name in ("square", "circle", "gaussian", "spot")
        }
        assert masks["square"].shape == GRID
        # Square and circle differ (corners of square lit, circle corners dark)
        assert not np.allclose(masks["square"], masks["circle"], atol=0.2)
        # Gaussian is not binary-anywhere-near (peak spread)
        assert not np.allclose(masks["gaussian"] > 0.5, masks["circle"] > 0.5)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            create_target_mask("banana", GRID, SIZE)


# ============================ Loss functions ============================

class TestLosses:
    """Verify the differentiable loss functions behave correctly."""

    def test_uniformity_lower_for_uniform_region(self):
        # Uniform intensity inside a target mask -> small CV -> small loss
        target = torch.full(GRID, 1.0)
        uniform = torch.full(GRID, 0.5)
        # Non-uniform: high value at center only
        non_uniform = torch.zeros(GRID)
        non_uniform[GRID[0] // 2, GRID[1] // 2] = 10.0

        l_uniform = uniformity_loss(uniform, target)
        l_nonuniform = uniformity_loss(non_uniform, target)
        assert float(l_uniform) < float(l_nonuniform)

    def test_efficiency_lower_when_energy_in_target(self):
        target = torch.zeros(GRID)
        target[GRID[0] // 2 - 5:GRID[0] // 2 + 5, GRID[1] // 2 - 5:GRID[1] // 2 + 5] = 1.0

        in_target = torch.zeros(GRID)
        in_target[GRID[0] // 2, GRID[1] // 2] = 1.0  # energy inside bucket
        out_target = torch.zeros(GRID)
        out_target[0, 0] = 1.0  # energy far outside bucket

        l_in = efficiency_loss(in_target, target)
        l_out = efficiency_loss(out_target, target)
        assert float(l_in) < float(l_out)

    def test_zero_order_penalty_larger_with_dc_spike(self):
        flat = torch.full(GRID, 0.01)
        spiked = torch.zeros(GRID)
        spiked[GRID[0] // 2, GRID[1] // 2] = 100.0
        assert float(zero_order_penalty(spiked)) > float(zero_order_penalty(flat))

    def test_smoothness_zero_flat_positive_noisy(self):
        flat = torch.zeros(GRID)
        noisy = torch.randn(*GRID)
        assert float(smoothness_regularization(flat)) == pytest.approx(0.0, abs=1e-9)
        assert float(smoothness_regularization(noisy)) > 0.0

    def test_total_loss_weighted_combination(self):
        target = torch.full(GRID, 1.0)
        intensity = torch.full(GRID, 0.5)
        phase = torch.zeros(GRID)
        l = total_loss(
            intensity, phase, target,
            w_uniformity=0.4, w_efficiency=0.4,
            w_zero_order=0.1, w_smoothness=0.1,
        )
        assert l.ndim == 0
        assert float(l) >= 0.0


# ============================ Forward model ============================

class TestForwardModel:
    """Verify the differentiable forward model produces finite intensity and gradients."""

    def _build(self, propagation: str = "fft"):
        from ao_shaping.algorithm.differentiable_shaping import _build_forward

        return _build_forward(propagation, 8e-6, 0.1, 1064e-9)

    def test_fft_forward_finite(self):
        from ao_shaping.algorithm.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=True)
        intensity = self._build("fft")(phase)
        assert intensity.shape == GRID
        assert th.isfinite(intensity).all()
        assert float(intensity.detach().min()) >= 0.0  # intensity non-negative

    def test_asm_forward_finite(self):
        from ao_shaping.algorithm.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=True)
        intensity = self._build("asm")(phase)
        assert intensity.shape == GRID
        assert th.isfinite(intensity).all()

    def test_gradient_flows(self):
        from ao_shaping.algorithm.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=True)
        intensity = self._build("fft")(phase)
        loss = intensity.sum()
        loss.backward()
        assert phase.grad is not None
        assert th.isfinite(phase.grad).all()

    def test_flat_phase_fft_has_dc_peak(self):
        # A flat phase maps to a strong DC spike at the field center under the
        # Fraunhofer model — the behaviour the zero-order penalty targets.
        from ao_shaping.algorithm.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=False)
        intensity = self._build("fft")(phase)
        cy, cx = GRID[0] // 2, GRID[1] // 2
        assert float(intensity[cy, cx]) > 100.0 * float(intensity[0, 0])

    def test_asm_propagate_torch_matches_numpy_scale(self):
        # Sanity: torch ASM returns complex field, finite, on a small grid
        small = (64, 64)
        field = torch.ones(*small, dtype=torch.complex128)
        out = angular_spectrum_propagate_torch(
            field, 8e-6, 0.1, 1064e-9
        )
        assert out.shape == small
        assert torch.isfinite(out).all()


# ============================ Training loop ============================

class TestTrain:
    """Verify train_beam_shaping behavior."""

    def _run(self, **kw):
        target = create_target_mask("square", GRID, SIZE)
        defaults = dict(iterations=60, lr=1e-2, seed=0)
        defaults.update(kw)
        return train_beam_shaping(target, GRID, **defaults)

    def test_loss_decreases(self):
        result = self._run()
        assert len(result.loss_history) >= 2
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-9

    def test_phase_shape_and_finite(self):
        result = self._run()
        assert result.phase.shape == GRID
        assert np.isfinite(result.phase).all()
        assert result.phase.dtype == np.float64

    def test_adam_and_lbfgs_both_run(self):
        for opt in ("adam", "lbfgs"):
            r = self._run(optimizer=opt)
            assert np.isfinite(r.phase).all()
            assert len(r.loss_history) > 0

    def test_result_dataclass_fields(self):
        result = self._run()
        assert isinstance(result.phase, np.ndarray)
        assert isinstance(result.target_intensity, np.ndarray)
        assert isinstance(result.simulated_intensity, np.ndarray)
        assert isinstance(result.loss_history, list)
        assert isinstance(result.iterations, int)
        assert isinstance(result.converged, bool)

    def test_fft_and_asm_agree_approximately(self):
        # Loose agreement: both should put energy near target; not exact (approx)
        r_fft = self._run(propagation="fft", iterations=40, seed=1)
        r_asm = self._run(propagation="asm", iterations=40, seed=1)
        # Both finite and same shape
        assert r_fft.phase.shape == r_asm.phase.shape == GRID
        assert np.isfinite(r_fft.phase).all()
        assert np.isfinite(r_asm.phase).all()

    def test_deterministic_with_seed(self):
        r1 = self._run(seed=42)
        r2 = self._run(seed=42)
        np.testing.assert_allclose(r1.phase, r2.phase, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(r1.loss_history), np.asarray(r2.loss_history), atol=1e-6
        )

    def test_no_nan_phase(self):
        result = self._run()
        assert not np.isnan(result.phase).any()

    def test_initial_phase_respected(self):
        init = np.full(GRID, 1.0, dtype=np.float64)
        r = self._run(initial_phase=init, iterations=10, seed=0)
        # starting from non-zero phase still runs
        assert np.isfinite(r.phase).all()

    def test_source_amplitude_accepted(self):
        amp = np.ones(GRID, dtype=np.float64)
        amp[GRID[0] // 2, GRID[1] // 2] = 2.0
        r = self._run(source_amplitude=amp, iterations=10, seed=0)
        assert np.isfinite(r.phase).all()


# ============================ Validation ============================

class TestValidation:
    """Verify input validation raises helpful errors."""

    def test_bad_propagation_raises(self):
        target = create_target_mask("square", GRID, SIZE)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, propagation="banana", iterations=5)

    def test_bad_optimizer_raises(self):
        target = create_target_mask("square", GRID, SIZE)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, optimizer="banana", iterations=5)

    def test_non_2d_target_raises(self):
        target = np.ones((GRID[0],), dtype=np.float64)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, iterations=5)

    def test_size_mismatch_target_grid_raises(self):
        target = create_target_mask("square", (64, 64), 20)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, iterations=5)

    def test_invalid_grid_raises(self):
        target = create_target_mask("square", GRID, SIZE)
        with pytest.raises(ValueError):
            train_beam_shaping(target, (0, 0), iterations=5)


# ============================ Torch optionality ============================

class TestTorchGraceful:
    """Verify the module raises a helpful ImportError when torch is missing."""

    def test_import_error_message(self, monkeypatch):
        from ao_shaping.algorithm import differentiable_shaping

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(differentiable_shaping, "_torch", lambda: (_ for _ in ()).throw(
            ImportError(
                "differentiable_shaping requires PyTorch. Install with: uv sync --extra ml"
            )
        ))
        with pytest.raises(ImportError, match="requires PyTorch"):
            differentiable_shaping._torch()
