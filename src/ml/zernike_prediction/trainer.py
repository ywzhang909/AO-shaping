"""Training loop for Zernike-coefficient regression (T6).

Implements the dual-camera regression training pipeline:

- ``train_regressor``: full train/val loop with MSE loss on the 65 non-piston
  coefficients, AdamW + ``ReduceLROnPlateau``, optional linear LR warmup,
  best-checkpointing (by val MAE), optional early stopping and optional wandb
  logging (the run name ALWAYS embeds ``input_mode``, e.g.
  ``resnet18-combined-20260414_171241``).
- ``evaluate_regressor``: full test-set evaluation (metrics + plots).
- ``predict_coeffs``: batch prediction returning ``(N, 65)`` arrays.

Torch-only (no click); the CLI lives in :mod:`ml.zernike_prediction.cli`.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from ml.zernike_prediction.metrics import mae, metrics_summary
from ml.zernike_prediction import plots

try:  # wandb is optional (tests and offline environments run without it)
    import wandb
except ImportError:  # pragma: no cover - depends on env
    wandb = None

__all__ = [
    "evaluate_regressor",
    "make_run_name",
    "predict_coeffs",
    "train_regressor",
]

_PHASE_SIZE_DEFAULT = (192, 192)
_N_TARGET = 65


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def make_run_name(model_name: str, input_mode: str, run_id: str) -> str:
    """Compose the checkpoint/wandb run name.

    The name always embeds ``input_mode`` (``{model_name}-{input_mode}-{run_id}``,
    e.g. ``resnet18-combined-sweep1``) so runs are distinguishable across
    camera configurations.

    Args:
        model_name: Backbone name (e.g. "resnet18", "simple_cnn").
        input_mode: One of "combined", "focus", "pupil".
        run_id: Run identifier (timestamp, sweep id, ...).

    Returns:
        ``f"{model_name}-{input_mode}-{run_id}"``.
    """
    return f"{model_name}-{input_mode}-{run_id}"


def _resolve_device(device: str | torch.device) -> torch.device:
    """Normalize a device spec; ``"auto"`` prefers cuda:1 > cuda:0 > cpu."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        return torch.device("cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    """Seed python/numpy/torch and enable deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _collect_pred_true(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Eval-mode prediction + ground-truth arrays over a labeled loader.

    Returns:
        ``(pred, true)`` numpy arrays in loader order, shape ``(N, 65)``.
        Empty loaders yield ``(0, n_coeffs)`` arrays.
    """
    dev = _resolve_device(device)
    model.to(dev).eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images, labels = batch[0], batch[1]
            out = model(images.to(dev))
            preds.append(out.detach().cpu().numpy())
            if labels is None:
                raise ValueError(
                    "evaluation requires labeled data, but a batch carried no labels "
                    "(create the loader with require_labels=True)"
                )
            if isinstance(labels, list):
                raise ValueError("evaluation does not support mixed labeled/unlabeled batches")
            trues.append(labels.detach().cpu().numpy())
    if not preds:
        return np.zeros((0, _N_TARGET), dtype=np.float32), np.zeros((0, _N_TARGET), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32), np.concatenate(trues, axis=0).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_coeffs(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict ``(N, 65)`` coefficient arrays in loader order (eval mode, no grad).

    Works with any collate layout (labeled, unlabeled, or with meta) since only
    the image tensor at ``batch[0]`` is used.

    Args:
        model: Regression model producing ``(B, 65)`` outputs.
        loader: Any DataLoader over ``ZernikeDualDataset``.
        device: Device spec or torch.device.

    Returns:
        ``(N, 65)`` float32 predictions.
    """
    dev = _resolve_device(device)
    model.to(dev).eval()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch[0]
            out = model(images.to(dev))
            preds.append(out.detach().cpu().numpy())
    if not preds:
        return np.zeros((0, _N_TARGET), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_regressor(
    model: nn.Module,
    test_loader: DataLoader,
    device: str | torch.device = "cpu",
    phase_size: tuple[int, int] = _PHASE_SIZE_DEFAULT,
    out_dir: str | Path | None = None,
    prefix: str = "eval",
    return_arrays: bool = True,
) -> dict[str, Any]:
    """Full test-set evaluation: metrics summary + optional plots.

    Args:
        model: Trained regression model.
        test_loader: Labeled DataLoader to evaluate.
        device: Device spec or torch.device.
        phase_size: ``(H, W)`` grid used for the circular phase metrics.
        out_dir: When given, renders the standard plot set here.
        prefix: File prefix for the rendered plots.
        return_arrays: Include ``pred``/``true`` numpy arrays in the result.

    Returns:
        Dict with ``metrics`` (from ``metrics_summary``), plus ``pred``/``true``
        when ``return_arrays`` and ``plots`` (path map) when ``out_dir``.
    """
    pred, true = _collect_pred_true(model, test_loader, device)
    summary = metrics_summary(pred, true, phase_size)
    result: dict[str, Any] = {"metrics": summary}
    if out_dir is not None:
        result["plots"] = plots.all_plots(pred, true, out_dir, prefix=prefix, phase_size=phase_size)
    if return_arrays:
        result["pred"] = pred
        result["true"] = true
    return result


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 150,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    model_name: str = "resnet18",
    input_mode: str = "combined",
    run_id: str = "local",
    use_wandb: bool = False,
    wandb_project: str = "ao-zernike",
    wandb_entity: str | None = None,
    seed: int = 42,
    checkpoint_dir: str | Path = "data/zernike_pred/checkpoints",
    log_every: int = 10,
    phase_size: tuple[int, int] = _PHASE_SIZE_DEFAULT,
    early_stop_patience: int | None = None,
    num_warmup_epochs: int = 1,
    test_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Train a Zernike-coefficient regressor with best-checkpointing.

    Loss is plain MSE on the 65 non-piston coefficients. Optimizer is AdamW;
    the scheduler is ``ReduceLROnPlateau(patience=10, factor=0.5)`` on val MAE.
    An optional linear warmup scales the LR up over the first
    ``num_warmup_epochs`` epochs. The best model (lowest val MAE) is saved to
    ``checkpoint_dir/{model_name}-{input_mode}-{run_id}-best.pt``.

    Args:
        model: Regression model (already built; moved to ``device`` internally).
        train_loader / val_loader: Labeled DataLoaders.
        epochs: Maximum number of epochs (bounded — no infinite loops).
        lr: Peak learning rate (AdamW).
        weight_decay: AdamW weight decay.
        device: Device spec (``"auto"`` -> cuda:1 > cuda:0 > cpu).
        model_name: Backbone name, embedded in the run/checkpoint name.
        input_mode: "combined" | "focus" | "pupil" — embedded in the run name.
        run_id: Run identifier for the checkpoint/wandb run name.
        use_wandb: Log to wandb (import-guarded; no-op without wandb installed).
        wandb_project / wandb_entity: wandb target project/entity.
        seed: Determinism seed (python/numpy/torch + cuDNN flags).
        checkpoint_dir: Directory for the best-model checkpoint.
        log_every: Log batch loss every N batches.
        phase_size: ``(H, W)`` grid for end-of-training phase metrics/grids.
        early_stop_patience: Stop after this many epochs without val MAE
            improvement (``None`` disables early stopping).
        num_warmup_epochs: Linear LR warmup length (1 = standard start).
        test_loader: Optional labeled loader for end-of-training test metrics
            (wandb only); falls back to ``val_loader``.

    Returns:
        Dict with ``best_val_mae``, ``best_epoch``, ``history``
        (``train_mae``/``val_mae``/``lr`` lists per epoch) and
        ``best_model_path``.
    """
    dev = _resolve_device(device)
    _set_seed(seed)
    model.to(dev)

    run_name = make_run_name(model_name, input_mode, run_id)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}-best.pt"

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )
    base_lr = float(lr)

    wandb_config: dict[str, Any] = {
        "model_name": model_name,
        "input_mode": input_mode,
        "run_id": run_id,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "early_stop_patience": early_stop_patience,
        "num_warmup_epochs": num_warmup_epochs,
        "phase_size": phase_size,
    }
    if use_wandb and wandb is not None:
        if wandb.run is None:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,  # run name MUST embed input_mode
                config=wandb_config,
            )
        else:
            # Already inside an active run (e.g. a wandb sweep agent): keep the
            # run but make sure the name embeds input_mode for traceability.
            wandb.run.name = run_name
            wandb.config.update(wandb_config, allow_val_change=True)
        logger.info("wandb run '{}' active (project={})", run_name, wandb_project)

    history: dict[str, list[float]] = {"train_mae": [], "val_mae": [], "lr": []}
    best_val_mae = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        # Optional linear LR warmup: scale lr up from 0 -> base over the first
        # num_warmup_epochs epochs (after warmup the plateau scheduler owns it).
        if epoch <= num_warmup_epochs:
            scale = epoch / max(1, num_warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * scale

        model.train()
        epoch_losses: list[float] = []
        train_preds: list[np.ndarray] = []
        train_trues: list[np.ndarray] = []
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            images, labels = batch[0], batch[1]
            if labels is None:
                raise ValueError("train loader must provide labels (require_labels=True)")
            labels = labels.to(dev)
            optimizer.zero_grad(set_to_none=True)
            out = model(images.to(dev))
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            train_preds.append(out.detach().cpu().numpy())
            train_trues.append(labels.detach().cpu().numpy())
            if step % log_every == 0 or step == n_batches:
                logger.info(
                    "epoch {}/{} batch {}/{} loss {:.6f}",
                    epoch, epochs, step, n_batches, loss.item(),
                )

        pred_np = np.concatenate(train_preds, axis=0).astype(np.float32)
        true_np = np.concatenate(train_trues, axis=0).astype(np.float32)
        train_mae = float(mae(pred_np, true_np))

        val_pred, val_true = _collect_pred_true(model, val_loader, dev)
        val_mae = float(mae(val_pred, val_true)) if val_true.size else float("nan")
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_mae)

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        logger.info(
            "epoch {}/{} train_mae {:.6f} val_mae {:.6f} lr {:.2e}",
            epoch, epochs, train_mae, val_mae, current_lr,
        )
        if use_wandb and wandb is not None:
            wandb.log({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae, "lr": current_lr})

        if np.isfinite(val_mae) and val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_path)
            logger.info("saved best model (val_mae {:.6f}) to {}", best_val_mae, best_path)
        else:
            epochs_since_improve += 1

        if early_stop_patience is not None and epochs_since_improve >= early_stop_patience:
            logger.info(
                "early stopping at epoch {} (no val_mae improvement for {} epochs)",
                epoch, epochs_since_improve,
            )
            break

    if not best_path.exists():
        # Degenerate case (e.g. empty val loader): checkpoint the final weights.
        best_val_mae = float("nan")
        torch.save(model.state_dict(), best_path)
        logger.warning("no improving val_mae observed; saved final weights to {}", best_path)

    # ------------------------------------------------------------------
    # End-of-training: wandb test metrics, best-model artifact, phase grids
    # ------------------------------------------------------------------
    eval_loader = test_loader if test_loader is not None else val_loader
    if use_wandb and wandb is not None and eval_loader is not None:
        model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
        final_pred, final_true = _collect_pred_true(model, eval_loader, dev)
        if final_true.size:
            final_metrics = metrics_summary(final_pred, final_true, phase_size)
            scalar_metrics = {
                f"final/{key}": value
                for key, value in final_metrics.items()
                if isinstance(value, (int, float, np.floating))
            }
            wandb.log(scalar_metrics)
            logger.info("logged final metrics to wandb: {}", scalar_metrics)

            artifact = wandb.Artifact(name=f"model-{run_name}", type="model")
            artifact.add_file(str(best_path))
            wandb.log_artifact(artifact)
            logger.info("uploaded best-model artifact '{}'", artifact.name)

            n_show = min(4, int(final_pred.shape[0]))
            if n_show > 0:
                with tempfile.TemporaryDirectory(prefix="ao-zernike-") as tmpdir:
                    coefs: list[np.ndarray] = []
                    labels: list[str] = []
                    for i in range(n_show):
                        coefs.append(final_pred[i])
                        labels.append(f"pred {i}")
                        coefs.append(final_true[i])
                        labels.append(f"true {i}")
                    grid_path = plots.phase_grid(
                        coefs, labels, Path(tmpdir) / "phase_grid.png", size=phase_size,
                        title=f"{run_name} sample predictions",
                    )
                    wandb.log({"phase_grid": wandb.Image(str(grid_path))})

                    scatter_path = plots.predict_true_scatter(
                        final_pred, final_true,
                        Path(tmpdir) / "predict_true_scatter.png",
                        title=f"{run_name} predict vs true (per Zernike term)",
                    )
                    wandb.log({"predict_true_scatter": wandb.Image(str(scatter_path))})

                    coeff_mae_path = plots.per_coeff_mae_bar(
                        final_pred, final_true,
                        Path(tmpdir) / "per_coeff_mae.png",
                        title=f"{run_name} per-coefficient MAE",
                    )
                    wandb.log({"per_coeff_mae": wandb.Image(str(coeff_mae_path))})

                    order_mae_path = plots.per_order_mae_bar(
                        final_pred, final_true,
                        Path(tmpdir) / "per_order_mae.png",
                        title=f"{run_name} per-order MAE",
                    )
                    wandb.log({"per_order_mae": wandb.Image(str(order_mae_path))})
        wandb.finish()

    return {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "history": history,
        "best_model_path": str(best_path),
    }
