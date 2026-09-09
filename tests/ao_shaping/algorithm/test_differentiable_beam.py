"""Tests for the differentiable (backprop) beam-shaping algorithm."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.algorithm.differentiable_beam import (
    BeamOptimizeResult,
    differentiable_beam_optimize,
    differentiable_far_field,
    far_field_intensity,
)


def _gaussian_target(size: int = 32) -> np.ndarray:
    """A smooth 2D gaussian target intensity (peak-normalized)."""
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r2 = x**2 + y**2
    g = np.exp(-r2 / (size / 4.0))
    return g.astype(np.float32) / g.max()


class TestDifferentiableFarField:
    """Test the differentiable FFT far-field model."""

    def test_flat_phase_concentrates_energy_in_dc(self):
        """A zero phase (flat) input puts all energy at the DC pixel.

        The uniform field ``E_near = ones`` has FFT magnitude 1 at DC and 0
        elsewhere, so after fftshift the peak is the center pixel and the
        center-pixel intensity fraction is ~1.0.
        """
        amp = np.ones((16, 16), dtype=np.float32)
        phase = torch.zeros(16, 16, dtype=torch.float32)
        i_far = far_field_intensity(amp, phase)
        frac_center = (
            i_far[8, 8] / (i_far.sum() + 1e-12)
        ).item()
        assert frac_center > 0.99

    def test_far_field_is_differentiable(self):
        """The intensity output must carry a grad_fn linking to phase."""
        amp = np.ones((16, 16), dtype=np.float32)
        phase = torch.zeros(16, 16, dtype=torch.float32, requires_grad=True)
        i_far = far_field_intensity(amp, phase)
        assert i_far.grad_fn is not None

    def test_output_shape_matches_input(self):
        amp = np.ones((24, 20), dtype=np.float32)
        phase = torch.randn(24, 20, dtype=torch.float32)
        i_far = far_field_intensity(amp, phase)
        assert i_far.shape == torch.Size((24, 20))

    def test_zero_phase_field_matches_numpy_fft(self):
        """Sanity-check the model against a hand-computed NumPy FFT."""
        n = 16
        amp = np.ones((n, n), dtype=np.float32)
        phase = torch.zeros(n, n, dtype=torch.float32)
        e_far = differentiable_far_field(amp, phase)
        expected = np.fft.fftshift(np.fft.fft2(amp))
        assert np.allclose(e_far.cpu().numpy(), expected, atol=1e-3)


class TestDifferentiableBeamOptimize:
    """Test the full Adam optimization loop."""

    def test_loss_decreases(self):
        """Over enough steps the MSE loss must go down."""
        target = _gaussian_target(32)
        result = differentiable_beam_optimize(
            target_intensity=target,
            lr=0.01,
            epochs=40,
            device="cpu",
            seed=0,
        )
        assert result.loss_history[0] > result.loss_history[-1]
        assert result.final_loss == result.loss_history[-1]

    def test_result_fields(self):
        """BeamOptimizeResult must expose the documented fields."""
        target = _gaussian_target(32)
        result = differentiable_beam_optimize(
            target_intensity=target,
            epochs=5,
            device="cpu",
            seed=0,
        )
        assert isinstance(result, BeamOptimizeResult)
        assert isinstance(result.phase, np.ndarray)
        assert result.phase.shape == target.shape
        assert isinstance(result.loss_history, list)
        assert len(result.loss_history) == 5
        assert isinstance(result.final_loss, float)
        assert isinstance(result.converged, bool)
        assert result.steps == 5
        assert result.device == "cpu"

    def test_reproducibility_with_seed(self):
        """Same seed must give the same final loss."""
        target = _gaussian_target(32)
        r1 = differentiable_beam_optimize(
            target_intensity=target, epochs=10, device="cpu", seed=42
        )
        r2 = differentiable_beam_optimize(
            target_intensity=target, epochs=10, device="cpu", seed=42
        )
        assert np.allclose(r1.phase, r2.phase)
        assert r1.final_loss == r2.final_loss

    def test_rejects_non_2d_target(self):
        with pytest.raises(ValueError):
            differentiable_beam_optimize(
                target_intensity=np.ones((10,)),
                epochs=1,
                device="cpu",
                seed=0,
            )

    def test_rejects_negative_target(self):
        bad = _gaussian_target(16)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError):
            differentiable_beam_optimize(
                target_intensity=bad,
                epochs=1,
                device="cpu",
                seed=0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
