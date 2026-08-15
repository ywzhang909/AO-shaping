"""Tests for ml.zernike_prediction.trainer / cli (T6-T7) — offline, fast, CPU-only.

Covers: training smoke, seed determinism, evaluation, prediction shape, the
wandb run-name convention (input_mode always embedded), and CLI help wiring.

No real data, no GPU, no wandb: synthetic 8x8 samples resized to 32x32, tiny
``simple_cnn`` model, 2 epochs. All training runs here set ``use_wandb=False``
so no wandb API is ever touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from ml.zernike_prediction.cli import main as cli_main
from ml.zernike_prediction.dataset import create_zernike_loaders
from ml.zernike_prediction.trainer import (
    evaluate_regressor,
    make_run_name,
    predict_coeffs,
    train_regressor,
)
from ml.zernike.models import build_model

_IMG_SIZE = 8
_TARGET_SIZE = (32, 32)
_BATCH_SIZE = 8
_EPOCHS = 2
_SEED = 7


# ---------------------------------------------------------------------------
# Synthetic data fixtures (no conftest.py in this project — helpers inline)
# ---------------------------------------------------------------------------


def _make_fake_run(tmp_path: Path, run_id: str, n: int = 40, size: int = _IMG_SIZE) -> Path:
    """Create a run dir of n synthetic samples (8x8 uint16/uint8 + 66-coeff metadata)."""
    run_dir = tmp_path / run_id
    for i in range(n):
        d = run_dir / f"sample_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(i)
        np.save(d / "daheng_frame.npy", rng.integers(0, 65536, (size, size), dtype=np.uint16))
        np.save(d / "miicam_frame.npy", rng.integers(0, 256, (size, size), dtype=np.uint8))
        coeffs = [1.0] + [float(0.05 * (i + 1) + 0.01 * rng.normal()) for _ in range(65)]
        meta = {"sample_idx": i, "phase_params": {"n_max": 10, "coefficients": coeffs}}
        (d / "metadata.json").write_text(json.dumps(meta))
    return run_dir


def _make_loaders(tmp_path: Path, seed: int = _SEED):
    # Layout mirrors the real data root: <data_root>/<run_id>/sample_XXXX/
    _make_fake_run(tmp_path, "runA")
    loaders = create_zernike_loaders(
        tmp_path,
        batch_size=_BATCH_SIZE,
        val_split=0.15,
        test_split=0.15,
        seed=seed,
        input_mode="combined",
        target_size=_TARGET_SIZE,
        run_ids=["runA"],
        num_workers=0,
    )
    return loaders, tmp_path


def _make_model(seed: int = _SEED, in_channels: int = 2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_model("simple_cnn", in_channels=in_channels, n_coeffs=65, device="cpu")


def _train_kwargs(checkpoint_dir: str | Path) -> dict:
    return {
        "epochs": _EPOCHS,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "device": "cpu",
        "model_name": "simple_cnn",
        "input_mode": "combined",
        "run_id": "test",
        "use_wandb": False,
        "seed": _SEED,
        "checkpoint_dir": str(checkpoint_dir),
        "log_every": 10,
        "phase_size": (32, 32),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_regressor_smoke(tmp_path: Path) -> None:
    """End-to-end 2-epoch CPU training on synthetic data produces a checkpoint."""
    loaders, _ = _make_loaders(tmp_path)
    model = _make_model()
    ckpt_dir = tmp_path / "ckpt"

    result = train_regressor(model, loaders["train"], loaders["val"], **_train_kwargs(ckpt_dir))

    assert Path(result["best_model_path"]).exists()
    assert result["best_model_path"].endswith("simple_cnn-combined-test-best.pt")
    assert result["best_epoch"] >= 1
    assert np.isfinite(result["best_val_mae"])
    history = result["history"]
    assert len(history["train_mae"]) == _EPOCHS
    assert len(history["val_mae"]) == _EPOCHS
    assert len(history["lr"]) == _EPOCHS
    assert all(np.isfinite(v) for v in history["train_mae"])
    assert all(np.isfinite(v) for v in history["val_mae"])


def test_trainer_deterministic_seed(tmp_path: Path) -> None:
    """Same seed + same synthetic data -> identical best val_mae (within 1e-6)."""
    loaders1, _ = _make_loaders(tmp_path, seed=_SEED)
    model1 = _make_model(_SEED)
    result1 = train_regressor(model1, loaders1["train"], loaders1["val"], **_train_kwargs(tmp_path / "c1"))

    loaders2, _ = _make_loaders(tmp_path, seed=_SEED)
    model2 = _make_model(_SEED)
    result2 = train_regressor(model2, loaders2["train"], loaders2["val"], **_train_kwargs(tmp_path / "c2"))

    assert abs(result1["best_val_mae"] - result2["best_val_mae"]) < 1e-6
    assert result1["history"]["val_mae"] == pytest.approx(result2["history"]["val_mae"], abs=1e-6)


def test_evaluate_regressor(tmp_path: Path) -> None:
    """evaluate_regressor returns the full metrics dict + (N, 65) arrays + plots."""
    loaders, _ = _make_loaders(tmp_path)
    model = _make_model()
    train_regressor(model, loaders["train"], loaders["val"], **_train_kwargs(tmp_path / "ckpt"))

    out = evaluate_regressor(
        model, loaders["test"], "cpu", phase_size=(32, 32), out_dir=str(tmp_path / "plots")
    )
    metrics = out["metrics"]
    for key in (
        "mae", "rmse", "mse", "r2", "per_coeff_mae",
        "per_order_mae", "phase_mae", "phase_rmse", "n_samples",
    ):
        assert key in metrics
    n_test = len(loaders["test"].dataset)
    assert out["pred"].shape == (n_test, 65)
    assert out["true"].shape == (n_test, 65)
    assert "plots" in out and set(out["plots"]) == {"scatter", "order_mae", "coeff_mae", "phase_error"}


def test_predict_coeffs_shape(tmp_path: Path) -> None:
    """predict_coeffs returns (N, 65) float32 arrays in loader order."""
    loaders, _ = _make_loaders(tmp_path)
    model = _make_model()
    train_regressor(model, loaders["train"], loaders["val"], **_train_kwargs(tmp_path / "ckpt"))

    preds = predict_coeffs(model, loaders["test"], "cpu")
    assert preds.shape == (len(loaders["test"].dataset), 65)
    assert preds.dtype == np.float32
    assert np.isfinite(preds).all()


def test_wandb_run_name_embeds_input_mode() -> None:
    """The run-name helper always embeds input_mode; no real wandb is touched."""
    assert make_run_name("resnet18", "combined", "sweep1") == "resnet18-combined-sweep1"
    for mode in ("combined", "focus", "pupil"):
        name = make_run_name("resnet18", mode, "run2026")
        assert mode in name
        assert name.startswith("resnet18-")


def test_cli_train_help() -> None:
    """`ao-zernike train --help` exits 0 and exposes --input-mode."""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--input-mode" in result.output
