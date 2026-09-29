"""Unit tests for the consolidated loss namespace (src/ml/gsnet/losses.py)."""

from __future__ import annotations

import math

import torch

from ml.gsnet import evaluate as evaluate_mod
from ml.gsnet import train as train_mod
from ml.gsnet.losses import ShapingLosses
from ml.gsnet.model import angular_difference


def _wrapped(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Closed-form wrapped signed difference, written out independently."""
    raw = (pred - target + math.pi) % (2 * math.pi)
    return raw - math.pi


class TestCircularMSE:
    def test_matches_closed_form_wrapped_difference(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 3, 8, 8) * 4.0
        target = torch.randn(2, 3, 8, 8) * 4.0

        expected = torch.mean(_wrapped(pred, target) ** 2)
        assert torch.allclose(ShapingLosses.circular_mse(pred, target), expected)

    def test_wraps_across_branch_cut(self):
        """A ~2*pi offset must be counted as a small error, not a huge one."""
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 2 * math.pi - 0.2)

        assert float(ShapingLosses.circular_mse(pred, target)) < 0.1

    def test_zero_for_identical(self):
        pred = torch.randn(2, 1, 8, 8)
        assert float(ShapingLosses.circular_mse(pred, pred)) < 1e-6

    def test_delegates_angular_difference(self):
        pred = torch.randn(1, 1, 5, 5)
        target = torch.randn(1, 1, 5, 5)

        assert torch.equal(
            ShapingLosses.angular_difference(pred, target),
            angular_difference(pred, target),
        )


class TestPhaseMAE:
    def test_matches_closed_form_wrapped_difference(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 1, 6, 6) * 3.0
        target = torch.randn(2, 1, 6, 6) * 3.0

        expected = float(_wrapped(pred, target).abs().mean())
        assert ShapingLosses.phase_mae(pred, target) == expected

    def test_returns_python_float(self):
        pred = torch.randn(1, 1, 4, 4)
        assert isinstance(ShapingLosses.phase_mae(pred, pred), float)

    def test_wraps_across_branch_cut(self):
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 2 * math.pi - 0.5)

        assert ShapingLosses.phase_mae(pred, target) < 1.0


class TestIntensityMSE:
    def test_value_on_tiny_synthetic_input(self):
        # 1x1 grid: fft of a single sample returns that sample, so the
        # normalized far field is [1.0] and the loss is a closed form.
        source_amp = torch.full((1, 1, 1, 1), 2.0)
        phase = torch.full((1, 1, 1, 1), 0.3)
        target = torch.full((1, 1, 1, 1), 5.0)

        # intensity = |2 * exp(i*0.3)|^2 = 4 -> pred_n = 4 / (4 + 1e-12) ~ 1
        # tgt_n = 5 / (5 + 1e-12) ~ 1  ->  loss ~ 0
        loss = ShapingLosses.intensity_mse(source_amp, phase, target)
        assert torch.isfinite(loss)
        assert float(loss) < 1e-10

    def test_matches_explicit_fft_reference(self):
        torch.manual_seed(2)
        source_amp = torch.rand(2, 1, 8, 8) + 0.25
        phase = torch.randn(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)

        far = torch.fft.fftshift(
            torch.fft.fft2(source_amp * torch.exp(1j * phase)), dim=(-2, -1)
        )
        intensity = torch.abs(far) ** 2
        pred_n = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        tgt_n = target / (target.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        expected = torch.mean((pred_n - tgt_n) ** 2)

        assert torch.allclose(
            ShapingLosses.intensity_mse(source_amp, phase, target), expected
        )

    def test_scale_invariant(self):
        source_amp = torch.ones(1, 1, 16, 16)
        phase = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 6:10, 6:10] = 1.0

        base = ShapingLosses.intensity_mse(source_amp, phase, target)
        scaled = ShapingLosses.intensity_mse(source_amp, phase, 100.0 * target)
        assert torch.allclose(base, scaled, atol=1e-6)


class TestShapingLoss:
    def test_value_on_tiny_synthetic_input(self):
        # 1x1 grid: the single far-field sample is constant across the image,
        # so its centered vector is exactly zero -> denom ~ 1e-12 and the
        # correlation is 0/1e-12 = 0 -> loss = 1.0.
        source_amp = torch.full((1, 1, 1, 1), 1.0)
        phase = torch.zeros(1, 1, 1, 1)
        target = torch.full((1, 1, 1, 1), 1.0)

        assert torch.allclose(
            ShapingLosses.shaping_loss(source_amp, phase, target),
            torch.tensor(1.0),
        )

    def test_matches_explicit_pearson_reference(self):
        torch.manual_seed(3)
        source_amp = torch.rand(3, 1, 8, 8) + 0.5
        phase = torch.randn(3, 1, 8, 8)
        target = torch.rand(3, 1, 8, 8)

        far = torch.fft.fftshift(
            torch.fft.fft2(source_amp * torch.exp(1j * phase)), dim=(-2, -1)
        )
        pred = (torch.abs(far) ** 2).flatten(1)
        tgt = target.flatten(1)
        p = pred - pred.mean(dim=1, keepdim=True)
        t = tgt - tgt.mean(dim=1, keepdim=True)
        denom = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
        expected = torch.mean(1.0 - (p * t).sum(dim=1) / denom)

        assert torch.allclose(
            ShapingLosses.shaping_loss(source_amp, phase, target), expected
        )

    def test_bounded_and_finite(self):
        torch.manual_seed(4)
        source_amp = torch.rand(2, 1, 16, 16) + 0.5
        phase = torch.randn(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)

        loss = ShapingLosses.shaping_loss(source_amp, phase, target)
        assert torch.isfinite(loss)
        assert 0.0 <= float(loss) <= 2.0


class TestCoefficientL1:
    def test_value(self):
        c_hat = torch.tensor([[0.1, -0.4, 0.9]])
        c_gt = torch.tensor([[0.1, -0.1, 0.6]])
        # |0|, |−0.3|, |0.3| -> mean 0.2
        assert torch.allclose(
            ShapingLosses.coefficient_l1(c_hat, c_gt), torch.tensor(0.2)
        )

    def test_zero_for_exact_match(self):
        c = torch.randn(4, 8)
        assert float(ShapingLosses.coefficient_l1(c, c)) < 1e-7

    def test_is_differentiable(self):
        c_hat = torch.zeros(1, 3, requires_grad=True)
        c_gt = torch.full((1, 3), 0.5)
        ShapingLosses.coefficient_l1(c_hat, c_gt).backward()
        assert c_hat.grad is not None
        # d mean(|c - c_gt|) / dc = sign(c - c_gt) / N = -1/3 here.
        assert torch.allclose(c_hat.grad, -torch.full_like(c_hat, 1.0 / 3.0))


class TestBackwardCompatibleAliases:
    def test_train_reexports_are_the_same_callables(self):
        assert train_mod.circular_mse is ShapingLosses.circular_mse
        assert train_mod.intensity_mse is ShapingLosses.intensity_mse
        assert train_mod.shaping_loss is ShapingLosses.shaping_loss

    def test_evaluate_reexports_are_the_same_callables(self):
        assert evaluate_mod.phase_mae is ShapingLosses.phase_mae

    def test_aliases_produce_identical_values(self):
        torch.manual_seed(5)
        pred = torch.randn(2, 1, 8, 8)
        target = torch.randn(2, 1, 8, 8)
        source_amp = torch.rand(2, 1, 8, 8) + 0.25

        assert torch.equal(
            train_mod.circular_mse(pred, target),
            ShapingLosses.circular_mse(pred, target),
        )
        assert evaluate_mod.phase_mae(pred, target) == ShapingLosses.phase_mae(
            pred, target
        )
        assert torch.equal(
            train_mod.intensity_mse(source_amp, pred, target),
            ShapingLosses.intensity_mse(source_amp, pred, target),
        )
        assert torch.equal(
            train_mod.shaping_loss(source_amp, pred, target),
            ShapingLosses.shaping_loss(source_amp, pred, target),
        )
