"""Unit tests for the training loop (src/ml/gsnet/train.py)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from ml.gsnet.dataset import GSShapingDataset
from ml.gsnet.model import FourierGSNet
from ml.gsnet.train import (
    TrainResult,
    circular_mse,
    intensity_mse,
    shaping_loss,
    train_gsnet,
)


class TestCircularMSE:
    def test_zero_for_identical(self):
        p = torch.randn(2, 1, 8, 8)
        assert circular_mse(p, p) < 1e-6

    def test_nonnegative(self):
        p = torch.randn(2, 1, 8, 8)
        t = torch.randn(2, 1, 8, 8)
        assert circular_mse(p, t) >= 0

    def test_wraps_correctly(self):
        """Phase difference near 2*pi must count as a small error."""
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 2 * torch.pi - 0.2)
        loss = circular_mse(pred, target)
        assert loss < 0.1  # ~0.2^2 if wrapping works, ~6.1^2 if not


class TestIntensityMSE:
    def test_positive_definite(self):
        source_amp = torch.ones(1, 1, 32, 32)
        phase = torch.zeros(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        loss = intensity_mse(source_amp, phase, tgt)
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_scale_invariant(self):
        """Doubling the target's absolute scale must not change the loss."""
        source_amp = torch.ones(1, 1, 32, 32)
        phase = torch.zeros(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        l1 = intensity_mse(source_amp, phase, tgt)
        l2 = intensity_mse(source_amp, phase, 100.0 * tgt)
        assert torch.allclose(l1, l2, atol=1e-6)


class TestShapingLoss:
    def test_zero_for_perfect_match(self):
        """Phase that reproduces the target exactly yields loss ~0."""
        source_amp = torch.ones(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        target_amp = torch.sqrt(tgt + 1e-12)
        from ml.gsnet.dataset import compute_gs_phase

        gt_phase = compute_gs_phase(source_amp, target_amp, iterations=60)
        loss = shaping_loss(source_amp, gt_phase, tgt)
        assert loss < 0.1  # GS reproduces the target; correlation ~0.98+

    def test_bounded_scale(self):
        """Loss is 1 - correlation, so it must stay in [0, 2]."""
        source_amp = torch.rand(2, 1, 32, 32) + 0.5
        phase = torch.randn(2, 1, 32, 32)
        tgt = torch.rand(2, 1, 32, 32)
        loss = shaping_loss(source_amp, phase, tgt)
        assert 0.0 <= loss <= 2.0
        assert torch.isfinite(loss)

    def test_scale_invariant(self):
        """Beam and target absolute scales must not change the loss."""
        source_amp = torch.ones(1, 1, 32, 32)
        phase = torch.zeros(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        l1 = shaping_loss(2.0 * source_amp, phase, tgt)
        l2 = shaping_loss(source_amp, phase, 100.0 * tgt)
        assert torch.allclose(l1, l2, atol=1e-4)

    def test_random_is_worse_than_gs(self):
        """Random phase should score worse (higher loss) than GS phase."""
        source_amp = torch.ones(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        target_amp = torch.sqrt(tgt + 1e-12)
        from ml.gsnet.dataset import compute_gs_phase

        torch.manual_seed(0)
        gt_phase = compute_gs_phase(source_amp, target_amp, iterations=60)
        random_phase = torch.randn_like(gt_phase) * 2.0
        l_gs = shaping_loss(source_amp, gt_phase, tgt)
        l_rand = shaping_loss(source_amp, random_phase, tgt)
        assert l_gs < l_rand


class TestTrainGSnet:
    def test_loss_decreases_over_epochs(self):
        """Training a few epochs must reduce the total loss (convergence)."""
        torch.manual_seed(0)
        ds = GSShapingDataset(n_samples=16, grid=32, seed=1, gs_iterations=20)
        dl = DataLoader(ds, batch_size=8, shuffle=True)
        model = FourierGSNet(num_layers=2, base_channels=8)
        res = train_gsnet(model, dl, epochs=3, lr=1e-3, device="cpu")
        assert res.history[-1]["loss"] < res.history[0]["loss"]
        assert res.best_epoch >= 1

    def test_result_fields(self):
        ds = GSShapingDataset(n_samples=8, grid=32, seed=2, gs_iterations=15)
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        model = FourierGSNet(num_layers=2, base_channels=8)
        res = train_gsnet(model, dl, epochs=2, device="cpu")
        assert isinstance(res, TrainResult)
        assert len(res.history) == 2
        for key in ("loss", "phase_loss", "shaping_loss", "intensity_loss"):
            assert key in res.history[0]
        assert isinstance(res.seconds, float) and res.seconds >= 0
        assert res.checkpoint_path is None

    def test_checkpoint_saved(self, tmp_path):
        ds = GSShapingDataset(n_samples=8, grid=32, seed=3, gs_iterations=15)
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        model = FourierGSNet(num_layers=2, base_channels=8)
        res = train_gsnet(
            model, dl, epochs=2, device="cpu",
            checkpoint_dir=str(tmp_path),
        )
        assert res.checkpoint_path is not None
        assert res.checkpoint_path.exists()
        state = torch.load(res.checkpoint_path, weights_only=False)
        assert "model_state_dict" in state

    def test_model_parameters_updated(self):
        """After training, weights must differ from the initial state."""
        torch.manual_seed(0)
        ds = GSShapingDataset(n_samples=16, grid=32, seed=4, gs_iterations=20)
        dl = DataLoader(ds, batch_size=8, shuffle=True)
        model = FourierGSNet(num_layers=2, base_channels=8)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        train_gsnet(model, dl, epochs=3, lr=1e-3, device="cpu")
        changed = any(
            not torch.allclose(before[k], v) for k, v in model.state_dict().items()
        )
        assert changed

    def test_invalid_epochs(self):
        ds = GSShapingDataset(n_samples=4, grid=32, seed=5, gs_iterations=10)
        dl = DataLoader(ds, batch_size=2)
        model = FourierGSNet(num_layers=2, base_channels=8)
        with pytest.raises(ValueError):
            train_gsnet(model, dl, epochs=0, device="cpu")