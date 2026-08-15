"""Click CLI for the Zernike-coefficient regression pipeline (T7).

Entry point: the ``ao-zernike`` console script (``main`` group) or
``python -m ml.zernike_prediction.cli``.

Subcommands:
    train            Train a regressor (wandb-aware) and evaluate on the test split.
    sweep            Hyperparameter sweep over lr/batch_size/weight_decay/model_type.
    predict          Predict (N, 65) coefficients from a checkpoint -> npy + csv.
    eval             Evaluate a checkpoint on labeled data -> metrics + plots.
    recover-labels   Recover missing labels for a run (tolerant placeholder).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from ml.zernike_prediction.dataset import (
    ZernikeDualDataset,
    _INPUT_MODES,
    collate_with_meta,
    create_zernike_loaders,
)
from ml.zernike_prediction.metrics import coefficient_names
from ml.zernike_prediction.trainer import (
    evaluate_regressor,
    make_run_name,
    train_regressor,
)
from ml.zernike.models import build_model

try:  # wandb is optional (sweep falls back to random search without it)
    import wandb
except ImportError:  # pragma: no cover - depends on env
    wandb = None

__all__ = ["main"]

_N_COEFFS = 65
_MODEL_TYPES = ("resnet18", "resnet34", "simple_cnn")
_PHASE_SIZE = (192, 192)

# Hyperparameter sweep space (shared by wandb sweep + random-search fallback).
_SWEEP_LR_RANGE = (5e-4, 5e-3)
_SWEEP_BATCH_SIZES = (16, 32, 64)
_SWEEP_WEIGHT_DECAY_RANGE = (1e-5, 1e-3)
_SWEEP_MODEL_TYPES = ("resnet18", "resnet34")
_SWEEP_EPOCHS = 60


# ---------------------------------------------------------------------------
# Option parsing helpers
# ---------------------------------------------------------------------------


def _parse_size(ctx: click.Context, param: click.Parameter, value: str) -> tuple[int, int]:
    """Parse ``--target-size`` values like ``256,256`` or ``(256x256)``."""
    if value is None:
        return (256, 256)
    cleaned = str(value).replace("(", "").replace(")", "").replace(" ", "")
    cleaned = cleaned.replace("x", ",")
    parts = [p for p in cleaned.split(",") if p]
    if len(parts) != 2:
        raise click.BadParameter(f"expected 'H,W' (e.g. 256,256), got {value!r}")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise click.BadParameter(f"expected two integers, got {value!r}") from exc


def _parse_run_ids(run_ids: str) -> list[str] | None:
    """Split a comma-separated run-id string into a list (None when empty)."""
    parts = [p.strip() for p in run_ids.split(",") if p.strip()]
    return parts or None


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` -> cuda:1 if available, else cuda:0, else cpu."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _infer_model_type(checkpoint: Path) -> str:
    """Infer the model type from a ``{model}-{input_mode}-{run}-best.pt`` name."""
    stem = checkpoint.stem
    for mt in ("simple_cnn", "resnet34", "resnet18"):
        if stem.startswith(mt):
            return mt
    raise click.ClickException(
        f"cannot infer model_type from {checkpoint.name}; pass --model-type explicitly"
    )


def _metrics_to_json(metrics: dict[str, Any]) -> dict[str, Any]:
    """Make a metrics dict JSON-serializable (arrays -> lists, nan -> None)."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, dict):
            out[key] = {str(k): float(v) for k, v in value.items()}
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = None if not np.isfinite(value) else float(value)
        elif isinstance(value, float):
            out[key] = None if not math.isfinite(value) else value
        else:
            out[key] = value
    return out


def _common_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Attach the data/model/training options shared by train and sweep."""
    f = click.option("--data-root", type=str, default="data/slm_dual_spot", show_default=True,
                     help="Root dir containing <run>/sample_XXXX/ subdirs.")(f)
    f = click.option("--run-ids", type=str, default="20260414_171241", show_default=True,
                     help="Comma-separated run dir ids.")(f)
    f = click.option("--model-type", type=click.Choice(_MODEL_TYPES), default="resnet18",
                     show_default=True, help="Backbone model type.")(f)
    f = click.option("--input-mode", type=click.Choice(_INPUT_MODES), default="combined",
                     show_default=True, help="Input channels: combined (focus+pupil), focus or pupil.")(f)
    f = click.option("--batch-size", type=int, default=32, show_default=True,
                     help="Training batch size.")(f)
    f = click.option("--lr", type=float, default=2e-3, show_default=True,
                     help="Peak learning rate (AdamW).")(f)
    f = click.option("--weight-decay", type=float, default=1e-4, show_default=True,
                     help="AdamW weight decay.")(f)
    f = click.option("--val-split", type=float, default=0.15, show_default=True,
                     help="Validation split fraction.")(f)
    f = click.option("--test-split", type=float, default=0.15, show_default=True,
                     help="Test split fraction.")(f)
    f = click.option("--seed", type=int, default=42, show_default=True,
                     help="Random seed (splits + training).")(f)
    f = click.option("--target-size", type=str, default="256,256", show_default=True,
                     callback=_parse_size, help="Resized (H,W) as 'H,W'.")(f)
    f = click.option("--num-workers", type=int, default=4, show_default=True,
                     help="DataLoader worker processes.")(f)
    f = click.option("--use-wandb/--no-use-wandb", default=False, show_default=True,
                     help="Enable wandb logging.")(f)
    f = click.option("--wandb-project", type=str, default="ao-zernike", show_default=True,
                     help="wandb project name.")(f)
    f = click.option("--wandb-entity", type=str, default=None,
                     help="wandb entity/team (default: logged-in user).")(f)
    f = click.option("--checkpoint-dir", type=str, default="data/zernike_pred/checkpoints",
                     show_default=True, help="Directory for checkpoints/results.")(f)
    f = click.option("--device", type=str, default="auto", show_default=True,
                     help="auto | cuda:N | cpu (auto: cuda:1 if available else cuda:0 else cpu).")(f)
    f = click.option("--split-mode", type=click.Choice(["sample", "run"]), default="sample",
                     show_default=True, help="Split by random sample or whole runs.")(f)
    return f


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(verbose: bool) -> None:
    """Zernike-coefficient regression pipeline (dual-camera)."""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command()
@_common_options
@click.option("--epochs", type=int, default=150, show_default=True,
              help="Number of training epochs.")
@click.option("--wandb-run-id", type=str, default=None,
              help="Override the auto-generated wandb/checkpoint run id.")
def train(
    data_root: str,
    run_ids: str,
    model_type: str,
    input_mode: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_split: float,
    test_split: float,
    seed: int,
    target_size: tuple[int, int],
    num_workers: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    checkpoint_dir: str,
    device: str,
    split_mode: str,
    epochs: int,
    wandb_run_id: str | None,
) -> None:
    """Train a regressor and evaluate the best checkpoint on the test split."""
    rids = _parse_run_ids(run_ids)
    dev = _resolve_device(device)
    in_channels = 2 if input_mode in ("combined", "fft") else 1
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loaders = create_zernike_loaders(
        data_root, batch_size=batch_size, val_split=val_split, test_split=test_split,
        seed=seed, input_mode=input_mode, target_size=target_size, run_ids=rids,
        split_mode=split_mode, num_workers=num_workers,
    )
    model = build_model(model_type, in_channels=in_channels, n_coeffs=_N_COEFFS, device=dev)

    run_id = wandb_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        "training {} ({}) on {} train / {} val / {} test samples",
        model_type, input_mode,
        loaders["split_sizes"]["train"], loaders["split_sizes"]["val"],
        loaders["split_sizes"]["test"],
    )
    result = train_regressor(
        model, loaders["train"], loaders["val"],
        epochs=epochs, lr=lr, weight_decay=weight_decay, device=dev,
        model_name=model_type, input_mode=input_mode, run_id=run_id,
        use_wandb=use_wandb, wandb_project=wandb_project, wandb_entity=wandb_entity,
        seed=seed, checkpoint_dir=str(ckpt_dir), test_loader=loaders["test"],
    )

    # Evaluate the BEST checkpoint (train_regressor leaves the final weights in
    # the model, which may be worse than the best epoch).
    best_path = Path(result["best_model_path"])
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    ev = evaluate_regressor(model, loaders["test"], dev, out_dir=str(ckpt_dir), prefix="eval")
    test_metrics = ev["metrics"]

    summary = {
        "run_name": make_run_name(model_type, input_mode, run_id),
        "model_type": model_type,
        "input_mode": input_mode,
        "run_id": run_id,
        "best_val_mae": float(result["best_val_mae"]),
        "best_epoch": int(result["best_epoch"]),
        "best_model_path": result["best_model_path"],
        "split_sizes": loaders["split_sizes"],
        "test_metrics": _metrics_to_json(test_metrics),
    }
    summary_path = ckpt_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    click.echo(f"best val_mae: {result['best_val_mae']:.6f} (epoch {result['best_epoch']})")
    click.echo(
        f"test mae: {test_metrics['mae']:.6f}  rmse: {test_metrics['rmse']:.6f}  "
        f"r2: {test_metrics['r2']:.6f}"
    )
    click.echo(f"checkpoint: {result['best_model_path']}")
    click.echo(f"eval summary: {summary_path}")


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


@main.command()
@_common_options
@click.option("--count", type=int, default=10, show_default=True,
              help="Number of sweep runs.")
@click.option("--sweep-method", type=click.Choice(["bayes", "grid"]), default="bayes",
              show_default=True, help="wandb sweep method.")
@click.option("--sweep-metric", type=str, default="val_mae", show_default=True,
              help="Metric to optimize.")
def sweep(
    count: int,
    sweep_method: str,
    sweep_metric: str,
    data_root: str,
    run_ids: str,
    model_type: str,
    input_mode: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_split: float,
    test_split: float,
    seed: int,
    target_size: tuple[int, int],
    num_workers: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    checkpoint_dir: str,
    device: str,
    split_mode: str,
) -> None:
    """Hyperparameter sweep over lr / batch_size / weight_decay / model_type."""
    rids = _parse_run_ids(run_ids)
    dev = _resolve_device(device)
    in_channels = 2 if input_mode in ("combined", "fft") else 1
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not use_wandb or wandb is None:
        _random_sweep(
            count=count, data_root=data_root, rids=rids, input_mode=input_mode,
            in_channels=in_channels, val_split=val_split, test_split=test_split,
            seed=seed, target_size=target_size, num_workers=num_workers,
            dev=dev, ckpt_dir=ckpt_dir, split_mode=split_mode,
        )
        return

    if sweep_method == "grid":
        parameters: dict[str, Any] = {
            "lr": {"values": [5e-4, 1e-3, 2e-3, 5e-3]},
            "batch_size": {"values": [16, 32, 64]},
            "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "model_type": {"values": list(_SWEEP_MODEL_TYPES)},
            "epochs": {"value": _SWEEP_EPOCHS},
        }
    else:
        parameters = {
            "lr": {"distribution": "log_uniform_values", "min": _SWEEP_LR_RANGE[0],
                   "max": _SWEEP_LR_RANGE[1]},
            "batch_size": {"values": list(_SWEEP_BATCH_SIZES)},
            "weight_decay": {"distribution": "log_uniform_values",
                             "min": _SWEEP_WEIGHT_DECAY_RANGE[0],
                             "max": _SWEEP_WEIGHT_DECAY_RANGE[1]},
            "model_type": {"values": list(_SWEEP_MODEL_TYPES)},
            "epochs": {"value": _SWEEP_EPOCHS},
        }
    sweep_config: dict[str, Any] = {
        "method": sweep_method,
        "metric": {"name": sweep_metric, "goal": "minimize"},
        "parameters": parameters,
    }

    def _sweep_run() -> None:
        cfg = wandb.config
        mt = cfg["model_type"]
        bs = int(cfg["batch_size"])
        lr_v = float(cfg["lr"])
        wd_v = float(cfg["weight_decay"])
        ep = int(cfg["epochs"])
        loaders = create_zernike_loaders(
            data_root, batch_size=bs, val_split=val_split, test_split=test_split,
            seed=seed, input_mode=input_mode, target_size=target_size, run_ids=rids,
            split_mode=split_mode, num_workers=num_workers,
        )
        model = build_model(mt, in_channels=in_channels, n_coeffs=_N_COEFFS, device=dev)
        result = train_regressor(
            model, loaders["train"], loaders["val"],
            epochs=ep, lr=lr_v, weight_decay=wd_v, device=dev,
            model_name=mt, input_mode=input_mode, run_id=f"sweep-{wandb.run.id}",
            use_wandb=False, seed=seed, checkpoint_dir=str(ckpt_dir),
        )
        # Lightweight per-run reporting: val_mae + history curves.
        wandb.log({
            sweep_metric: result["best_val_mae"],
            "train_mae": result["history"]["train_mae"],
            "val_mae_curve": result["history"]["val_mae"],
            "lr": result["history"]["lr"],
        })
        logger.info("sweep run {}: val_mae {:.6f}", wandb.run.id, result["best_val_mae"])

    sweep_id = wandb.sweep(sweep_config, project=wandb_project, entity=wandb_entity)
    logger.info("starting wandb sweep {} (method={}, {} runs)", sweep_id, sweep_method, count)
    wandb.agent(sweep_id, function=_sweep_run, count=count)


def _random_sweep(
    *,
    count: int,
    data_root: str,
    rids: list[str] | None,
    input_mode: str,
    in_channels: int,
    val_split: float,
    test_split: float,
    seed: int,
    target_size: tuple[int, int],
    num_workers: int,
    dev: str,
    ckpt_dir: Path,
    split_mode: str,
) -> None:
    """wandb-free random-search fallback over the same hyperparameter space."""
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for i in range(count):
        cfg = {
            "lr": float(10 ** rng.uniform(math.log10(_SWEEP_LR_RANGE[0]), math.log10(_SWEEP_LR_RANGE[1]))),
            "batch_size": int(rng.choice(_SWEEP_BATCH_SIZES)),
            "weight_decay": float(10 ** rng.uniform(
                math.log10(_SWEEP_WEIGHT_DECAY_RANGE[0]), math.log10(_SWEEP_WEIGHT_DECAY_RANGE[1])
            )),
            "model_type": str(rng.choice(_SWEEP_MODEL_TYPES)),
            "epochs": _SWEEP_EPOCHS,
        }
        run_id = f"sweep{i + 1}"
        logger.info("sweep run {}/{}: {}", i + 1, count, cfg)
        loaders = create_zernike_loaders(
            data_root, batch_size=cfg["batch_size"], val_split=val_split, test_split=test_split,
            seed=seed, input_mode=input_mode, target_size=target_size, run_ids=rids,
            split_mode=split_mode, num_workers=num_workers,
        )
        model = build_model(cfg["model_type"], in_channels=in_channels, n_coeffs=_N_COEFFS, device=dev)
        result = train_regressor(
            model, loaders["train"], loaders["val"],
            epochs=cfg["epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], device=dev,
            model_name=cfg["model_type"], input_mode=input_mode, run_id=run_id,
            use_wandb=False, seed=seed, checkpoint_dir=str(ckpt_dir),
        )
        configs.append(cfg)
        results.append({
            "run_id": run_id,
            "val_mae": float(result["best_val_mae"]),
            "best_epoch": int(result["best_epoch"]),
        })
        logger.info("sweep run {}/{} val_mae {:.6f}", i + 1, count, result["best_val_mae"])

    summary_path = ckpt_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps({"configs": configs, "results": results}, indent=2))
    click.echo(f"sweep summary written to {summary_path}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@main.command()
@click.option("--checkpoint", type=str, required=True, help="Path to best-model .pt checkpoint.")
@click.option("--data-root", type=str, default="data/slm_dual_spot", show_default=True,
              help="Root dir containing run/sample subdirs.")
@click.option("--run-ids", type=str, default="20260414_171241", show_default=True,
              help="Comma-separated run dir ids.")
@click.option("--input-mode", type=click.Choice(_INPUT_MODES), default="combined",
              show_default=True, help="Input channels used by the model.")
@click.option("--target-size", type=str, default="256,256", show_default=True,
              callback=_parse_size, help="Resized (H,W) as 'H,W'.")
@click.option("--model-type", type=str, default=None,
              help="Model type; inferred from the checkpoint filename when omitted.")
@click.option("--output", type=str, default="data/zernike_pred/predictions.npy",
              show_default=True, help="Output .npy path (predictions.csv written alongside).")
@click.option("--device", type=str, default="auto", show_default=True,
              help="auto | cuda:N | cpu.")
def predict(
    checkpoint: str,
    data_root: str,
    run_ids: str,
    input_mode: str,
    target_size: tuple[int, int],
    model_type: str | None,
    output: str,
    device: str,
) -> None:
    """Predict (N, 65) coefficients from a checkpoint for the selected runs."""
    rids = _parse_run_ids(run_ids)
    dev = _resolve_device(device)
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise click.ClickException(f"checkpoint not found: {ckpt}")
    mt = model_type or _infer_model_type(ckpt)
    in_channels = 2 if input_mode in ("combined", "fft") else 1

    model = build_model(mt, in_channels=in_channels, n_coeffs=_N_COEFFS, device=dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    model.eval()

    ds = ZernikeDualDataset(
        data_root, run_ids=rids, target_size=target_size, input_mode=input_mode,
        require_labels=False, return_meta=True, seed=0,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_with_meta)

    preds: list[np.ndarray] = []
    sample_dirs: list[str] = []
    with torch.inference_mode():
        for imgs, _labels, dirs in loader:
            out = model(imgs.to(dev))
            preds.append(out.detach().cpu().numpy())
            sample_dirs.extend(dirs)
    pred_np = np.concatenate(preds, axis=0).astype(np.float32)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pred_np)
    csv_path = out_path.with_suffix(".csv")
    cols = coefficient_names()
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_dir", *cols])
        for sample_dir, row in zip(sample_dirs, pred_np):
            writer.writerow([sample_dir, *[f"{v:.8g}" for v in row]])

    click.echo(f"wrote {pred_np.shape[0]} predictions ({pred_np.shape[1]} coeffs) to {out_path}")
    click.echo(f"csv: {csv_path}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@main.command()
@click.option("--checkpoint", type=str, required=True, help="Path to best-model .pt checkpoint.")
@click.option("--data-root", type=str, default="data/slm_dual_spot", show_default=True,
              help="Root dir containing run/sample subdirs.")
@click.option("--run-ids", type=str, default="20260414_171241", show_default=True,
              help="Comma-separated run dir ids.")
@click.option("--input-mode", type=click.Choice(_INPUT_MODES), default="combined",
              show_default=True, help="Input channels used by the model.")
@click.option("--target-size", type=str, default="256,256", show_default=True,
              callback=_parse_size, help="Resized (H,W) as 'H,W'.")
@click.option("--model-type", type=str, default=None,
              help="Model type; inferred from the checkpoint filename when omitted.")
@click.option("--out-dir", type=str, default="data/zernike_pred/eval", show_default=True,
              help="Directory for plots + eval_summary.json.")
@click.option("--device", type=str, default="auto", show_default=True,
              help="auto | cuda:N | cpu.")
def eval(
    checkpoint: str,
    data_root: str,
    run_ids: str,
    input_mode: str,
    target_size: tuple[int, int],
    model_type: str | None,
    out_dir: str,
    device: str,
) -> None:
    """Evaluate a checkpoint on all labeled samples of the selected runs."""
    rids = _parse_run_ids(run_ids)
    dev = _resolve_device(device)
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise click.ClickException(f"checkpoint not found: {ckpt}")
    mt = model_type or _infer_model_type(ckpt)
    in_channels = 2 if input_mode in ("combined", "fft") else 1

    model = build_model(mt, in_channels=in_channels, n_coeffs=_N_COEFFS, device=dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))

    ds = ZernikeDualDataset(
        data_root, run_ids=rids, target_size=target_size, input_mode=input_mode,
        require_labels=True, seed=0,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = evaluate_regressor(model, loader, dev, out_dir=str(out), prefix="eval")
    metrics = ev["metrics"]

    summary_path = out / "eval_summary.json"
    summary_path.write_text(json.dumps(_metrics_to_json(metrics), indent=2))

    click.echo(f"evaluated {metrics['n_samples']} samples:")
    click.echo(
        f"  mae: {metrics['mae']:.6f}  rmse: {metrics['rmse']:.6f}  "
        f"mse: {metrics['mse']:.6f}  r2: {metrics['r2']:.6f}"
    )
    click.echo(f"  phase_mae: {metrics['phase_mae']:.6f}  phase_rmse: {metrics['phase_rmse']:.6f}")
    click.echo(f"plots + summary: {out}")


# ---------------------------------------------------------------------------
# recover-labels
# ---------------------------------------------------------------------------


@main.command("recover-labels")
@click.option("--data-root", type=str, default="data/slm_dual_spot", show_default=True,
              help="Root dir containing run/sample subdirs.")
@click.option("--run-ids", type=str, default="20260414_171241", show_default=True,
              help="Comma-separated run dir ids.")
@click.option("--refine/--no-refine", default=False, show_default=True,
              help="Refine recovered labels.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Overwrite existing labels.npy sidecars.")
def recover_labels(
    data_root: str,
    run_ids: str,
    refine: bool,
    overwrite: bool,
) -> None:
    """Recover missing labels for the selected runs (labels.py, when available)."""
    rids = _parse_run_ids(run_ids)
    try:
        from ml.zernike_prediction.labels import recover_run
    except ImportError:
        recover_run = None
    if recover_run is None:
        click.echo("labels module not yet available")
        return
    click.echo(
        f"recovering labels under {data_root} runs={rids} "
        f"refine={refine} overwrite={overwrite}"
    )
    summaries = []
    for rid in rids:
        click.echo(f"--- run {rid} ---")
        summaries.append(recover_run(data_root=data_root, run_id=rid, refine=refine, overwrite=overwrite))
    for s in summaries:
        click.echo(
            f"run {s['run_id']}: {s['recovered']} recovered, "
            f"{len(s['failed'])} failed, {s['skipped']} skipped"
        )


if __name__ == "__main__":
    main()
