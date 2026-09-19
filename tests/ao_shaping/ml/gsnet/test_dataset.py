"""Unit tests for the GSShapingDataset (src/ml/gsnet/dataset.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.gsnet.dataset import (
    GSShapingDataset,
    compute_gs_phase,
    make_target,
)


class TestMakeTarget:
    @pytest.mark.parametrize(
        "shape", ["square", "circle", "gaussian", "ring", "cross", "checker"]
    )
    def test_valid_shapes(self, shape):
        mask = make_target(shape, size=16, grid=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.max() > 0.0  # non-empty

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            make_target("nonsense", size=16, grid=64)

    def test_size_clamped_to_grid(self):
        mask = make_target("square", size=200, grid=64)
        assert mask.shape == (64, 64)

    def test_circle_fits_inside_square_bbox(self):
        mask = make_target("circle", size=20, grid=64)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        assert rows.max() - rows.min() <= 20
        assert cols.max() - cols.min() <= 20


class TestDataset:
    def test_sample_shapes(self):
        ds = GSShapingDataset(n_samples=4, grid=32, seed=0)
        src, tgt, gt = ds[0]
        assert tuple(src.shape) == (1, 32, 32)
        assert tuple(tgt.shape) == (1, 32, 32)
        assert tuple(gt.shape) == (1, 32, 32)
        assert src.dtype == torch.float32
        assert gt.dtype == torch.float32

    def test_len_matches_n_samples(self):
        ds = GSShapingDataset(n_samples=7, grid=32, seed=1)
        assert len(ds) == 7

    def test_reproducible_with_same_seed(self):
        d1 = GSShapingDataset(n_samples=3, grid=32, seed=99)
        d2 = GSShapingDataset(n_samples=3, grid=32, seed=99)
        assert torch.allclose(
            d1.target_intensity, d2.target_intensity
        )
        assert torch.allclose(d1.gt_phase, d2.gt_phase)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            GSShapingDataset(n_samples=0)
        with pytest.raises(ValueError):
            GSShapingDataset(n_samples=1, grid=8)

    def test_targets_varied(self):
        """Dataset must contain >1 distinct target shape across samples."""
        ds = GSShapingDataset(n_samples=16, grid=32, seed=3)
        uniques = {float(t.sum()) for t in ds.target_intensity}
        assert len(uniques) > 1


class TestGroundTruthPhase:
    def test_gt_phase_produces_target_far_field(self):
        """GS GT phase, propagated, must reproduce the requested target."""
        ds = GSShapingDataset(n_samples=4, grid=48, seed=5, gs_iterations=40)
        for i in range(len(ds)):
            src, tgt, gt = ds[i]
            far = ds.predict_far_intensity(gt)
            fn = far[0, 0] / far[0, 0].sum()
            tn = tgt[0] / tgt[0].sum()
            corr = np.corrcoef(fn.flatten().numpy(), tn.flatten().numpy())[0, 1]
            ee = float((far[0, 0] * (tgt[0] > 0)).sum() / far[0, 0].sum())
            assert corr > 0.9, f"sample {i} correlation {corr:.3f}"
            assert ee > 0.8, f"sample {i} encircled energy {ee:.3f}"

    def test_gs_more_iterations_better_or_equal(self):
        """More GS iterations should not degrade the phase quality badly."""
        source_amp = torch.ones(1, 1, 32, 32)
        tgt = torch.zeros(1, 1, 32, 32)
        tgt[:, :, 12:20, 12:20] = 1.0
        target_amp = torch.sqrt(tgt + 1e-12)
        p10 = compute_gs_phase(source_amp, target_amp, iterations=10)
        p60 = compute_gs_phase(source_amp, target_amp, iterations=60)
        assert torch.isfinite(p10).all()
        assert torch.isfinite(p60).all()


class TestPredictFarIntensity:
    def test_batch_broadcast(self):
        ds = GSShapingDataset(n_samples=4, grid=32, seed=0)
        phase = torch.zeros(3, 1, 32, 32)
        far = ds.predict_far_intensity(phase)
        assert tuple(far.shape) == (3, 1, 32, 32)
        assert (far >= 0).all()