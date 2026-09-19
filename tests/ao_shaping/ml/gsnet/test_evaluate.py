"""Unit tests for the evaluation metrics (src/ml/gsnet/evaluate.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ml.gsnet.dataset import GSShapingDataset
from ml.gsnet.evaluate import (
    EvalSummary,
    SampleMetrics,
    compute_sample_metrics,
    divergence_metric,
    evaluate_model,
    phase_mae,
)
from ml.gsnet.model import FourierGSNet


class TestPhaseMAE:
    def test_zero_for_identical(self):
        p = torch.randn(1, 1, 16, 16)
        assert phase_mae(p, p) < 1e-6

    def test_wraps(self):
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 2 * torch.pi - 0.5)
        assert phase_mae(pred, target) < 1.0


class TestSampleMetrics:
    def _sample(self):
        ds = GSShapingDataset(n_samples=1, grid=32, seed=0, gs_iterations=30)
        src, tgt, gt = ds[0]
        # Use the GS ground truth as "prediction": should score excellently.
        return src, tgt, gt

    def test_metrics_on_gt_phase(self):
        src, tgt, gt = self._sample()
        m = compute_sample_metrics(gt, gt, src, tgt)
        assert isinstance(m, SampleMetrics)
        assert m.phase_mae < 1e-6
        assert m.far_correlation > 0.9
        assert m.encircled_energy > 0.8
        assert m.far_rmse >= 0
        assert m.uniformity_cv >= 0

    def test_metrics_on_flat_phase(self):
        """A flat phase must score poorly on correlation (physics honest)."""
        src, tgt, gt = self._sample()
        flat = torch.zeros_like(gt)
        m = compute_sample_metrics(flat, gt, src, tgt)
        assert m.far_correlation < 0.95  # flat ≠ shaped; correlation far from GT target quality


class TestEvaluateModel:
    def test_summary_structure(self):
        torch.manual_seed(0)
        ds = GSShapingDataset(n_samples=6, grid=32, seed=6, gs_iterations=20)
        dl = DataLoader(ds, batch_size=3)
        model = FourierGSNet(num_layers=2, base_channels=8)
        summary = evaluate_model(model, dl, device="cpu")
        assert isinstance(summary, EvalSummary)
        assert summary.n_samples == 6
        assert len(summary.full) == 6
        for key in (
            "phase_mae", "far_correlation", "far_rmse",
            "uniformity_cv", "encircled_energy",
        ):
            assert key in summary.means

    def test_metrics_bounded(self):
        torch.manual_seed(0)
        ds = GSShapingDataset(n_samples=4, grid=32, seed=7, gs_iterations=20)
        dl = DataLoader(ds, batch_size=2)
        model = FourierGSNet(num_layers=2, base_channels=8)
        summary = evaluate_model(model, dl, device="cpu")
        for m in summary.full:
            assert 0.0 <= m.encircled_energy <= 1.0
            assert -1.0 <= m.far_correlation <= 1.0
            assert m.far_rmse >= 0.0


class TestDivergenceMetric:
    def test_returns_finite_scalar(self):
        ds = GSShapingDataset(n_samples=1, grid=32, seed=0, gs_iterations=20)
        model = FourierGSNet(num_layers=2, base_channels=8)
        d = divergence_metric(
            model,
            ds.source_intensity[0:1],
            ds.target_intensity[0:1],
            device="cpu",
        )
        assert torch.isfinite(d)
        assert float(d) >= 0.0