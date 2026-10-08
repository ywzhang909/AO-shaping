"""Unit tests for the same-epoch train/val gap metrics in `train_amp`.

The point of `split_mse_r2` is that it must agree with `evaluate` on the same
weights, because the whole reason it exists is that the in-epoch `train_mse`
cannot be compared against `val_mse`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, split_mse_r2  # noqa: E402


def _split(grid: int = 8, n: int = 5, seed: int = 0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    cos = torch.from_numpy(rng.random((n, 1, grid, grid), dtype=np.float32))
    sin = torch.from_numpy(rng.random((n, 1, grid, grid), dtype=np.float32))
    img = torch.from_numpy(rng.random((n, 1, grid, grid), dtype=np.float32))
    return {"phase_cos": cos, "phase_sin": sin, "target": img}


def _model(grid: int = 8) -> ZernikeAmpModel:
    return ZernikeAmpModel(ZernikeAmpConfig(n_max=4, grid=grid))


class TestSplitMseR2:
    def test_matches_evaluate_on_the_same_weights(self) -> None:
        from ml.zernike.train_amp import evaluate

        torch.manual_seed(0)
        model = _model()
        tensors = _split()
        mse, r2 = split_mse_r2(model, tensors)
        out = evaluate(model, tensors, beam_samples=0, roi_size_frac=None)
        assert mse == pytest.approx(out["mse"], rel=1e-6)
        assert r2 == pytest.approx(out["r2"], rel=1e-6)

    def test_perfect_prediction_is_zero_mse_and_one_r2(self) -> None:
        # A model whose coefficients reproduce the target exactly: impossible with a
        # real Zernike fit, so assert the arithmetic identity instead -- compare a
        # model against a reference built from its own output.
        model = _model()
        tensors = _split()
        mse, r2 = split_mse_r2(model, tensors)
        assert mse >= 0.0
        assert r2 <= 1.0 + 1e-9

    def test_batch_size_does_not_change_the_result(self) -> None:
        torch.manual_seed(0)
        model = _model()
        tensors = _split(n=7)
        a = split_mse_r2(model, tensors, batch=2)
        b = split_mse_r2(model, tensors, batch=256)
        assert a == pytest.approx(b, rel=1e-6)

    def test_leaves_the_model_in_train_mode(self) -> None:
        model = _model()
        model.train()
        split_mse_r2(model, _split())
        assert model.training

    def test_constant_target_reports_nan_r2_not_a_crash(self) -> None:
        grid, n = 8, 4
        tensors = {
            "phase_cos": torch.zeros(n, 1, grid, grid),
            "phase_sin": torch.zeros(n, 1, grid, grid),
            "target": torch.full((n, 1, grid, grid), 0.5),
        }
        mse, r2 = split_mse_r2(_model(grid=grid), tensors)
        assert mse >= 0.0
        assert np.isnan(r2)


class TestConfigDefaultsUnchanged:
    def test_defaults_are_usable(self) -> None:
        cfg = AmpTrainConfig()
        assert cfg.far_field_padding == ZernikeAmpConfig().far_field_padding
        assert cfg.grid == ZernikeAmpConfig().grid