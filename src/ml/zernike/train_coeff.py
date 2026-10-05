"""Train the learned Zernike-coefficient -> CCD far-field forward model.

This is the trainer for :mod:`ml.zernike.forward_model`: it fits the
deterministic physical map

    coefficients c (Noll order, radians)
        -> pupil phase  sum_j c_j Z_j
        -> pupil field  exp(i * phi)
        -> FFT
        -> |F|^2   (INTENSITY -- the CCD measures intensity, not field amplitude)

so that the network becomes a fast differentiable surrogate for
closed-loop beam shaping. The data comes from
:mod:`ml.hwdataset.zernike_dataset`.

============================  Why this trainer looks like this  ============================

**The fold is a FILE, never a record.** Records inside one pickle are
consecutive epochs of a single optimisation run, so a record-level split puts
near-identical inputs on both sides of the boundary and inflates every
validation number. :func:`file_folds` holds out one whole pickle at a time.

**Standardisation is fitted on the TRAIN positions only.** The returned vectors
are applied to every split, so statistics fitted on the whole corpus would let
validation inputs contribute to their own scaling -- and that optimism cannot be
corrected after the fact. This is the single most important line in the file.

**Selection is on R^2 alone.** ``val_psnr`` / ``val_ssim`` / ``val_mse`` are
computed and logged, but they are *diagnostics*. They are degenerate under a
per-frame normalised target: PSNR is ``10*log10(L^2/MSE)`` and SSIM's stabilisers
``C1=(0.01L)^2`` / ``C2=(0.015L)^2`` both reward shrinking the image values
toward the stabilisers, independently of fit quality. On the sibling task a
total-energy ("sum") normalisation reported MSE 0.00000 and SSIM 0.9996 while its
R^2 was *worse* than a constant predictor. Never let those three choose a
checkpoint.

**The output is peak-normalised before the loss.** ``ZernikeCoeffConvNet.forward``
is deliberately raw and unbounded (a hard-wired ``sigmoid`` -- as
``ml.phase.unet.UNetGenerator`` has -- cannot represent the dynamic range), so
:func:`~ml.zernike.forward_model.peak_normalize` is applied here to put the
prediction on the same [0, 1] scale as the target.

============================  Reproducibility  ============================
The seed pins the search and nothing else, because the training data is a fixed
on-disk corpus rather than a live measurement: two same-seed runs here ARE
bit-identical. That is the opposite of a hardware closed loop, where the seed
pins the perturbation signs and every far-field read still carries device noise.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Subset

from ml.hwdataset.dataset import FileGroupedSampler
from ml.hwdataset.index import PhaseSource, build_hw_index
from ml.hwdataset.records import MaterialiserConfig
from ml.hwdataset.zernike_dataset import (
    DEFAULT_N_MAX,
    ZERNIKE_COEFF_SOURCES,
    ZernikeCoeffDataset,
    fit_coeff_stats,
)
from ml.zernike.forward_model import (
    ZernikeCoeffConfig,
    build_forward_model,
    count_parameters,
    peak_normalize,
)

__all__ = [
    "CoeffTrainConfig",
    "CoeffTrainResult",
    "FileFold",
    "file_folds",
    "r2_score",
    "select_fold",
    "train",
    "main",
]

#: Index cache. Reading the 265-file corpus costs ~35 s cold, ~0.09 s warm, so
#: every entry point goes through it rather than re-walking ``data/debug``.
INDEX_CACHE = "data/hw_index_cache.json"

#: SSIM stabilisers, from Wang et al. 2004 with ``L = 1`` (our target is
#: normalised to a peak of 1.0). Diagnostic only -- see the module docstring.
_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.015**2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoeffTrainConfig:
    """Every knob one training run needs.

    Mirrors the ``train_amp.AmpTrainConfig`` convention: a flat dataclass whose
    fields are the argparse defaults, so a value cannot drift between the two.
    """

    # -- data / geometry (forwarded to ZernikeCoeffDataset + MaterialiserConfig)
    n_max: int = DEFAULT_N_MAX
    grid: int = 64
    image_mode: str = "robust"
    use_cache: bool = False

    # -- model (forwarded to ZernikeCoeffConfig)
    architecture: Literal["conv", "mlp"] = "conv"
    bottleneck: int = 4
    features: tuple[int, ...] = (256, 128, 64, 32)
    latent_channels: int = 256
    hidden: int = 512
    norm_layer: Literal["group", "batch", "none"] = "group"
    norm_groups: int = 8

    # -- optimisation
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    momentum: float = 0.9
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 0

    # -- fold selection
    fold: int | None = None
    held_out_path: str | None = None

    # -- bookkeeping
    out_dir: str = "logs/zernike_coeff"
    log_every: int = 1
    image_every: int = 10
    save_checkpoint: bool = True
    max_samples: int | None = None


@dataclass(frozen=True)
class CoeffTrainResult:
    """What one run produced."""

    best_val_r2: float
    best_epoch: int
    history: list[dict[str, float]]
    checkpoint: Path | None
    summary: Path | None
    held_out: str
    n_train: int
    n_val: int
    final_metrics: dict[str, float]


@dataclass(frozen=True)
class FileFold:
    """One leave-one-pickle-out fold."""

    index: int
    held_out: Path
    train_positions: tuple[int, ...]
    val_positions: tuple[int, ...]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def r2_score(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Coefficient of determination, **per sample**, against its own mean.

    ``R^2 = 1 - SS_res / SS_tot`` where ``SS_tot`` is computed from the target's
    own per-sample pixel mean.

    The per-sample denominator is the contract, not a detail. An earlier version
    in this repo pooled a pixel-only quantity against a sample+pixel variance --
    mismatched dimensions, which inflated the denominator and produced a
    **spurious negative R^2** that was briefly reported as "worse than
    predicting a constant". Flattens each sample independently so the two
    sums share their units.

    Args:
        pred: Predictions, ``(N, ...)`` or ``(N,)``.
        target: Targets, same shape as ``pred``.

    Returns:
        ``float32`` array of shape ``(N,)``, or a Python float when a single
        sample is supplied. ``nan`` where ``SS_tot == 0`` (a constant target has
        no variance to explain) rather than raising.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"pred and target must share a shape, got {p.shape} vs {t.shape}")
    single = p.ndim == 1
    if single:
        p = p[None, :]
        t = t[None, :]
    ss_res = np.sum((t - p) ** 2, axis=tuple(range(1, t.ndim)))
    ss_tot = np.sum((t - t.mean(axis=tuple(range(1, t.ndim)), keepdims=True)) ** 2,
                    axis=tuple(range(1, t.ndim)))
    out = np.full(ss_tot.shape, np.nan, dtype=np.float64)
    nonzero = ss_tot > 0.0
    out[nonzero] = 1.0 - ss_res[nonzero] / ss_tot[nonzero]
    return float(out[0]) if single else out.astype(np.float32)


def _pearson_per_sample(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-sample Pearson correlation over flattened pixels.

    ``nan`` when either sample is flat (zero variance), which a real far field
    essentially never is -- but a constant predictor must not crash the run.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(len(pred), -1)
    t = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    denom = np.sqrt((p**2).sum(axis=1) * (t**2).sum(axis=1))
    good = denom > 0.0
    if not np.any(good):
        return float("nan")
    corr = (p[good] * t[good]).sum(axis=1) / denom[good]
    return float(np.mean(corr))


def _psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    """PSNR for a target normalised to ``peak``.

    ``10*log10(peak^2 / mse)``. Diagnostic only.
    """
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


def _ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-sample SSIM (11x11 Gaussian, sigma 1.5, ``L = peak``).

    Diagnostic only. A compact, dependency-free implementation so the number is
    reproducible from this file alone. See the module docstring for why it must
    never select a checkpoint.
    """
    window = torch.hann_window(11, periodic=False)  # 1-D, shape (11,)
    kernel_2d = torch.outer(window, window)
    kernel_2d = kernel_2d / kernel_2d.sum()

    p = torch.as_tensor(np.asarray(pred, dtype=np.float32).reshape(-1, 1, *pred.shape[-2:]))
    t = torch.as_tensor(np.asarray(target, dtype=np.float32).reshape(-1, 1, *target.shape[-2:]))
    if p.shape[-1] < 11 or p.shape[-2] < 11:
        return float("nan")
    pad = 5
    mu_p = F.conv2d(p, kernel_2d[None, None], padding=pad)
    mu_t = F.conv2d(t, kernel_2d[None, None], padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sigma_p = F.conv2d(p * p, kernel_2d[None, None], padding=pad) - mu_p2
    sigma_t = F.conv2d(t * t, kernel_2d[None, None], padding=pad) - mu_t2
    sigma_pt = F.conv2d(p * t, kernel_2d[None, None], padding=pad) - mu_pt
    # Drop the padded border: it is not real data and would bias the mean.
    inner = (slice(None), slice(None), slice(pad, -pad), slice(pad, -pad))
    lum = (2.0 * mu_pt + _SSIM_C1) / (mu_p2 + mu_t2 + _SSIM_C1)
    cst = (2.0 * sigma_pt + _SSIM_C2) / (sigma_p + sigma_t + _SSIM_C2)
    return float((lum * cst)[inner].mean())


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def file_folds(dataset: ZernikeCoeffDataset) -> list[FileFold]:
    """Leave-one-pickle-out folds over ``dataset``.

    Every position appears in exactly one fold's validation half, and the folds
    are returned in first-appearance order of the source pickle so ``fold=i`` is
    stable across runs on the same corpus.

    Args:
        dataset: A :class:`ZernikeCoeffDataset`, already filtered to
            :data:`ZERNIKE_COEFF_SOURCES`.

    Returns:
        One :class:`FileFold` per distinct pickle.

    Raises:
        ValueError: If ``dataset`` holds no records.
    """
    records = dataset.records
    if not records:
        raise ValueError("file_folds: the dataset holds no records")
    order: list[Path] = []
    grouped: dict[Path, list[int]] = {}
    for position, record in enumerate(records):
        if record.path not in grouped:
            grouped[record.path] = []
            order.append(record.path)
        grouped[record.path].append(position)
    total = len(records)
    folds: list[FileFold] = []
    for index, path in enumerate(order):
        val = tuple(grouped[path])
        val_set = set(val)
        train = tuple(i for i in range(total) if i not in val_set)
        folds.append(
            FileFold(index=index, held_out=path, train_positions=train, val_positions=val)
        )
    return folds


def select_fold(folds: list[FileFold], cfg: CoeffTrainConfig) -> FileFold:
    """Choose one fold from ``cfg.held_out_path`` or ``cfg.fold``.

    ``held_out_path`` is tried first because it is unambiguous. Matching accepts
    either the full ``str(record.path)`` or the bare file name, so a caller need
    not reconstruct a platform-specific prefix.

    Raises:
        ValueError: If ``held_out_path`` matches no fold, naming the candidates;
            or if ``fold`` is out of range.
    """
    if cfg.held_out_path is not None:
        wanted = str(cfg.held_out_path)
        for fold in folds:
            if wanted in (str(fold.held_out), fold.held_out.name):
                return fold
        available = ", ".join(sorted(f.held_out.name for f in folds))
        raise ValueError(
            f"held_out_path {wanted!r} matches no fold. Available pickles: {available}"
        )
    index = 0 if cfg.fold is None else int(cfg.fold)
    if not -len(folds) <= index < len(folds):
        raise ValueError(
            f"fold {index} is out of range for {len(folds)} folds "
            f"(valid: 0..{len(folds) - 1}, or negative indices)"
        )
    return folds[index]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def _build_optimizer(model: torch.nn.Module, cfg: CoeffTrainConfig) -> torch.optim.Optimizer:
    """Dispatch on ``cfg.optimizer``.

    ``weight_decay == 0`` makes ``adam`` and ``adamw`` mathematically identical,
    so a sweep over the two at that setting measures nothing. That is a property
    of the optimisers, not a bug in this dispatch.
    """
    common = {"lr": float(cfg.lr), "weight_decay": float(cfg.weight_decay)}
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **common)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), momentum=float(cfg.momentum), **common
        )
    raise ValueError(
        f"optimizer must be one of 'adam', 'adamw', 'sgd'; got {cfg.optimizer!r}"
    )


def _predict(model: torch.nn.Module, coeffs: torch.Tensor) -> torch.Tensor:
    """Raw forward pass plus the mandatory peak normalisation.

    Split out so the training step, the evaluation and the figure cannot drift
    apart on whether the output is normalised. Deliberately NOT decorated with
    ``no_grad``: the training step needs the graph. Evaluation wraps its own call
    in :func:`torch.no_grad`.
    """
    return peak_normalize(model(coeffs))


def _evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Score one split. ``r2`` is the selection metric; the rest are diagnostics."""
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            coeffs = batch["coeffs"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            pred = _predict(model, coeffs)
            preds.append(pred.detach().float().cpu().numpy())
            targets.append(image.detach().float().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1, 1, 1), np.float32)
    target_arr = np.concatenate(targets, axis=0)
    mse = float(np.mean((pred_arr - target_arr) ** 2))
    per_sample_r2 = r2_score(pred_arr, target_arr)
    finite = per_sample_r2[np.isfinite(per_sample_r2)]
    return {
        "r2": float(np.mean(finite)) if finite.size else float("nan"),
        "pearson_r": _pearson_per_sample(pred_arr, target_arr),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "psnr": _psnr_from_mse(mse),
        "ssim": _ssim(pred_arr, target_arr),
    }


def _save_comparison(
    path: Path,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    caption: str,
) -> None:
    """Write a true-vs-prediction strip. Never fatal to the run."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # BEFORE pyplot: no display on a training box
        import matplotlib.pyplot as plt

        model.eval()
        count = min(4, batch["coeffs"].shape[0])
        pred = _predict(model, batch["coeffs"][:count].to(device)).detach().cpu().numpy()
        true = batch["image"][:count].detach().cpu().numpy()
        fig, axes = plt.subplots(2, count, figsize=(3.0 * count, 6.4))
        for column in range(count):
            axes[0, column].imshow(true[column, 0], cmap="inferno", vmin=0.0, vmax=1.0)
            # ravel to 1-D so r2_score returns a scalar rather than a length-1 array
            sample_r2 = r2_score(pred[column, 0].ravel(), true[column, 0].ravel())
            axes[0, column].set_title(f"true R^2={sample_r2:.3f}")
            axes[1, column].imshow(pred[column, 0], cmap="inferno", vmin=0.0, vmax=1.0)
            axes[1, column].set_title("predicted")
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        fig.suptitle(caption)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except (ImportError, OSError, ValueError) as exc:
        # A figure must never kill a training run that is otherwise producing a
        # checkpoint, so this degrades to a warning.
        logger.warning("Could not write the comparison figure {}: {}", path, exc)


def train(cfg: CoeffTrainConfig) -> CoeffTrainResult:
    """Train one fold and return its result.

    Args:
        cfg: The run's configuration.

    Returns:
        A :class:`CoeffTrainResult` whose ``checkpoint`` / ``summary`` are set
        when ``cfg.save_checkpoint`` is on.
    """
    started = time.perf_counter()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.image_mode == "sum":
        # Not a style preference: "sum" is the measured trap (R^2 worse than a
        # constant predictor while PSNR/SSIM look perfect).
        raise ValueError(
            "image_mode='sum' is not permitted: total-energy normalisation makes "
            "PSNR and SSIM degenerate while degrading R^2. Use 'robust' or, as "
            "an ablation only, 'peak'."
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter ONCE, then use the same index everywhere. ZernikeCoeffDataset also
    # filters internally, but fit_coeff_stats and file_folds address
    # `index.records` by position -- so an unfiltered index here would silently
    # mismatch the dataset's own positional order.
    index = build_hw_index(index_cache=INDEX_CACHE, progress_every=0)
    index = index.filter(sources=list(ZERNIKE_COEFF_SOURCES))

    materialiser_config = MaterialiserConfig(grid=cfg.grid, image_mode=cfg.image_mode)
    dataset = ZernikeCoeffDataset(
        index,
        config=materialiser_config,
        n_max=cfg.n_max,
        use_cache=cfg.use_cache,
    )

    folds = file_folds(dataset)
    fold = select_fold(folds, cfg)
    train_positions = list(fold.train_positions)
    val_positions = list(fold.val_positions)
    if cfg.max_samples is not None:
        # Deterministic head slice; enough for a smoke run, never for a
        # conclusion.
        train_positions = train_positions[: cfg.max_samples]
        val_positions = val_positions[: cfg.max_samples]

    # The leakage guard. Train positions only, then rebuild the dataset carrying
    # the fitted vectors.
    coeff_mean, coeff_std = fit_coeff_stats(
        index, train_positions, n_max=cfg.n_max
    )
    dataset = ZernikeCoeffDataset(
        index,
        config=materialiser_config,
        n_max=cfg.n_max,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        use_cache=cfg.use_cache,
    )
    logger.info(
        "Fold {} of {}: holding out {} ({} train / {} val records)",
        fold.index,
        len(folds),
        fold.held_out.name,
        len(train_positions),
        len(val_positions),
    )

    workers = max(0, int(cfg.num_workers))
    pin = bool(torch.cuda.is_available() and workers > 0)
    generator_kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": pin}
    if workers > 0:
        generator_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        Subset(dataset, train_positions),
        batch_size=cfg.batch_size,
        sampler=FileGroupedSampler(
            [dataset.records[i] for i in train_positions], seed=cfg.seed
        ),
        **generator_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_positions),
        batch_size=cfg.batch_size,
        shuffle=False,
        **generator_kwargs,
    )

    model_config = ZernikeCoeffConfig(
        n_coeffs=dataset.n_terms,
        grid=cfg.grid,
        architecture=cfg.architecture,
        bottleneck=cfg.bottleneck,
        features=tuple(cfg.features),
        latent_channels=cfg.latent_channels,
        hidden=cfg.hidden,
        norm_layer=cfg.norm_layer,
        norm_groups=cfg.norm_groups,
    )
    model = build_forward_model(model_config).to(device)
    logger.info(
        "Model {} ({} parameters), {} train / {} val, device {}",
        cfg.architecture,
        count_parameters(model),
        len(train_positions),
        len(val_positions),
        device,
    )

    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: list[dict[str, float]] = []
    best_r2 = -float("inf")
    best_epoch = -1
    checkpoint_path: Path | None = None
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            coeffs = batch["coeffs"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _predict(model, coeffs)
            loss = F.mse_loss(prediction, image)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()
            running += float(loss.detach()) * coeffs.shape[0]
            seen += coeffs.shape[0]
        scheduler.step()

        metrics = _evaluate(model, val_loader, device)
        row: dict[str, float] = {
            "epoch": float(epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            "train_mse": running / max(1, seen),
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(row)

        # Selection on R^2 ONLY -- see the module docstring.
        if metrics["r2"] == metrics["r2"] and metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_epoch = epoch
            if cfg.save_checkpoint:
                checkpoint_path = out_dir / "best_coefficients.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_config": asdict(model_config),
                        "train_config": {
                            k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in asdict(cfg).items()
                        },
                        "coeff_mean": np.asarray(coeff_mean),
                        "coeff_std": np.asarray(coeff_std),
                        "n_max": cfg.n_max,
                        "n_terms": dataset.n_terms,
                        "best_val_r2": best_r2,
                        "best_epoch": best_epoch,
                        "held_out": str(fold.held_out),
                    },
                    checkpoint_path,
                )

        if epoch % max(1, cfg.log_every) == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "epoch {:3d} | lr {:.2e} | train_mse {:.6f} | val_r2 {:+.4f} | "
                "val_pearson {:+.4f} | val_psnr {:.2f} | val_ssim {:.4f}",
                epoch,
                row["lr"],
                row["train_mse"],
                row["val_r2"],
                row["val_pearson_r"],
                row["val_psnr"],
                row["val_ssim"],
            )
        if epoch % max(1, cfg.image_every) == 0 or epoch == cfg.epochs - 1:
            _save_comparison(
                out_dir / f"compare_epoch{epoch:03d}.png",
                model,
                next(iter(val_loader)),
                device,
                f"fold {fold.index} (held out {fold.held_out.name}) epoch {epoch}",
            )

    summary_path = out_dir / "summary.json"
    final_metrics = history[-1] if history else {}
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(cfg).items()
                },
                "held_out": str(fold.held_out),
                "fold_index": fold.index,
                "n_train": len(train_positions),
                "n_val": len(val_positions),
                "n_terms": dataset.n_terms,
                "parameters": count_parameters(model),
                "best_val_r2": best_r2,
                "best_epoch": best_epoch,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "history": history,
                "wall_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Done: best val R^2 {:+.4f} at epoch {} | checkpoint {} | summary {} | {:.1f}s",
        best_r2,
        best_epoch,
        checkpoint_path,
        summary_path,
        time.perf_counter() - started,
    )
    return CoeffTrainResult(
        best_val_r2=best_r2,
        best_epoch=best_epoch,
        history=history,
        checkpoint=checkpoint_path,
        summary=summary_path,
        held_out=str(fold.held_out),
        n_train=len(train_positions),
        n_val=len(val_positions),
        final_metrics=final_metrics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the parser.

    Every numeric default is read off :class:`CoeffTrainConfig` rather than
    repeated as a literal. That convention is load-bearing here: ``train_amp``
    once hardcoded ``--far-field-padding default=1``, silently overrode the
    calibrated dataclass value of 10, and made every run start at R^2 = -0.38.
    A dataclass field is the single place a default lives.
    """
    cfg_default = CoeffTrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m ml.zernike.train_coeff",
        description="Train the Zernike-coefficient -> far-field forward model.",
    )
    parser.add_argument("--n-max", type=int, default=cfg_default.n_max)
    parser.add_argument("--grid", type=int, default=cfg_default.grid)
    parser.add_argument("--image-mode", default=cfg_default.image_mode)
    parser.add_argument("--use-cache", action="store_true", default=cfg_default.use_cache)
    parser.add_argument("--architecture", choices=["conv", "mlp"], default=cfg_default.architecture)
    parser.add_argument("--bottleneck", type=int, default=cfg_default.bottleneck)
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=list(cfg_default.features),
        help="decoder channel widths, coarse -> fine",
    )
    parser.add_argument("--latent-channels", type=int, default=cfg_default.latent_channels)
    parser.add_argument("--hidden", type=int, default=cfg_default.hidden)
    parser.add_argument(
        "--norm-layer", choices=["group", "batch", "none"], default=cfg_default.norm_layer
    )
    parser.add_argument("--norm-groups", type=int, default=cfg_default.norm_groups)
    parser.add_argument("--epochs", type=int, default=cfg_default.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg_default.batch_size)
    parser.add_argument("--lr", type=float, default=cfg_default.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg_default.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg_default.grad_clip)
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], default=cfg_default.optimizer
    )
    parser.add_argument("--momentum", type=float, default=cfg_default.momentum)
    parser.add_argument("--seed", type=int, default=cfg_default.seed)
    parser.add_argument("--device", default=cfg_default.device)
    parser.add_argument("--num-workers", type=int, default=cfg_default.num_workers)
    parser.add_argument(
        "--fold", type=int, default=cfg_default.fold, help="leave-one-pickle-out fold index"
    )
    parser.add_argument("--held-out-path", default=cfg_default.held_out_path)
    parser.add_argument("--out-dir", default=cfg_default.out_dir)
    parser.add_argument("--log-every", type=int, default=cfg_default.log_every)
    parser.add_argument("--image-every", type=int, default=cfg_default.image_every)
    parser.add_argument(
        "--no-checkpoint",
        dest="save_checkpoint",
        action="store_false",
        default=cfg_default.save_checkpoint,
    )
    parser.add_argument("--max-samples", type=int, default=cfg_default.max_samples)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    cfg = CoeffTrainConfig(
        n_max=args.n_max,
        grid=args.grid,
        image_mode=args.image_mode,
        use_cache=bool(args.use_cache),
        architecture=args.architecture,
        bottleneck=args.bottleneck,
        features=tuple(args.features),
        latent_channels=args.latent_channels,
        hidden=args.hidden,
        norm_layer=args.norm_layer,
        norm_groups=args.norm_groups,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        optimizer=args.optimizer,
        momentum=args.momentum,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        fold=args.fold,
        held_out_path=args.held_out_path,
        out_dir=args.out_dir,
        log_every=args.log_every,
        image_every=args.image_every,
        save_checkpoint=bool(args.save_checkpoint),
        max_samples=args.max_samples,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())