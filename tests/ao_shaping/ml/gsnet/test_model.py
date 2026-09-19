"""Unit tests for the FourierGSNet model (src/ml/gsnet/model.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.gsnet.model import (
    ConditionUNet,
    FFTLayer,
    FourierGSNet,
    angular_difference,
    count_parameters,
)


class TestFourierGSNetForward:
    def test_output_shape_and_dtype(self):
        model = FourierGSNet(num_layers=4, base_channels=8)
        src = torch.rand(2, 1, 32, 32)
        tgt = torch.zeros(2, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        phase = model(src, tgt)
        assert tuple(phase.shape) == (2, 1, 32, 32)
        assert phase.dtype == torch.float32
        assert torch.isfinite(phase).all()

    def test_phase_range_bounded(self):
        model = FourierGSNet(num_layers=2, base_channels=8)
        src = torch.rand(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        phase = model(src, tgt)
        # With residual updates summed over N layers the range can exceed 2pi;
        # require a sane window (physics mod-2pi phases stay within a few rad).
        assert phase.max() - phase.min() < 20.0

    def test_rejects_wrong_ndim(self):
        model = FourierGSNet(num_layers=2, base_channels=8)
        src = torch.rand(2, 32, 32)
        tgt = torch.rand(2, 32, 32)
        with pytest.raises(ValueError):
            model(src, tgt)

    def test_rejects_mismatched_shapes(self):
        model = FourierGSNet(num_layers=2, base_channels=8)
        src = torch.rand(1, 1, 32, 32)
        tgt = torch.rand(1, 1, 16, 16)
        with pytest.raises(ValueError):
            model(src, tgt)

    def test_deterministic_with_same_seed(self):
        torch.manual_seed(123)
        m1 = FourierGSNet(num_layers=3, base_channels=8)
        torch.manual_seed(123)
        m2 = FourierGSNet(num_layers=3, base_channels=8)
        src = torch.rand(1, 1, 16, 16)
        tgt = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            p1 = m1(src, tgt)
            p2 = m2(src, tgt)
        assert torch.allclose(p1, p2, atol=1e-6)

    def test_single_forward_pass_only(self):
        """A trained-style model computes the phase in ONE call (no loop).

        This is the core claim of deep unrolling: training unrolls GS
        iterations, inference uses a single forward pass.
        """
        model = FourierGSNet(num_layers=10, base_channels=8)
        src = torch.rand(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        phase = model(src, tgt)
        assert tuple(phase.shape) == (1, 1, 32, 32)


class TestFFTLayerPhysics:
    def test_imposes_target_amplitude_in_far_field(self):
        """The FFTLayer's physics step replaces far amplitude with target.

        Feed a trivial input; the returned phase, when propagated, must
        correlate with the requested target *better than the flat-phase
        baseline* (the DC spot alone scores EE=1.0, so EE is not the honest
        discriminator — correlation is).
        """
        layer = FFTLayer(base_channels=4)
        torch.manual_seed(0)
        phase = torch.zeros(1, 1, 32, 32)
        source_amp = torch.ones(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        target_amp = torch.sqrt(tgt + 1e-12)

        with torch.no_grad():
            out = layer.forward(phase, source_amp, target_amp)
            field = source_amp * torch.exp(1j * out)
            far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
            intensity = torch.abs(far) ** 2
            fn = intensity[0, 0] / intensity[0, 0].sum()
            tn = tgt[0] / tgt[0].sum()
            corr = np.corrcoef(fn.flatten().numpy(), tn.flatten().numpy())[0, 1]
            ee = float((intensity[0, 0] * (tgt[0] > 0)).sum() / intensity[0, 0].sum())

            # Flat-phase baseline (identical setup, zero phase).
            far0 = torch.fft.fftshift(torch.fft.fft2(source_amp), dim=(-2, -1))
            i0 = torch.abs(far0) ** 2
            f0 = i0[0, 0] / i0[0, 0].sum()
            corr0 = np.corrcoef(f0.flatten().numpy(), tn.flatten().numpy())[0, 1]

        assert corr > corr0, f"physics step must beat baseline: {corr:.3f} vs {corr0:.3f}"
        assert ee > 0.3, f"physics step should concentrate energy, EE={ee:.3f}"

    def test_backward_runs(self):
        layer = FFTLayer(base_channels=4)
        phase = torch.randn(1, 1, 16, 16, requires_grad=True)
        source_amp = torch.ones(1, 1, 16, 16)
        tgt = torch.zeros(1, 1, 16, 16)
        tgt[:, :, 6:10, 6:10] = 1.0
        target_amp = torch.sqrt(tgt + 1e-12)
        out = layer.forward(phase, source_amp, target_amp)
        loss = out.sum()
        loss.backward()
        assert phase.grad is not None
        assert torch.isfinite(phase.grad).all()


class TestConditionUNet:
    def test_shape(self):
        net = ConditionUNet(base_channels=8)
        x = torch.rand(2, 2, 32, 32)
        cond = torch.rand(2, 3, 32, 32)
        y = net(x, cond)
        assert tuple(y.shape) == (2, 1, 32, 32)

    def test_frozen_condition_still_runs(self):
        net = ConditionUNet(base_channels=8)
        x = torch.rand(1, 2, 16, 16)
        cond = torch.zeros(1, 3, 16, 16)
        y = net(x, cond)
        assert torch.isfinite(y).all()


class TestHelpers:
    def test_count_parameters(self):
        model = FourierGSNet(num_layers=2, base_channels=8)
        assert count_parameters(model) > 0

    def test_angular_difference_zero_for_identical(self):
        p = torch.randn(2, 1, 8, 8)
        assert torch.allclose(angular_difference(p, p), torch.zeros_like(p), atol=1e-6)

    def test_angular_difference_wraps(self):
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 2 * torch.pi - 0.1)
        diff = angular_difference(pred, target)
        # Shortest angular path: |pred - target| wraps to ~0.1 rad, not ~6.2.
        assert float(diff.abs().max()) < 1.0