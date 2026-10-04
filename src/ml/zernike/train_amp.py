"""Train :class:`~ml.zernike.models.ZernikeAmpModel` on the real hardware corpus.

Drives the physics forward model from :mod:`ml.zernike.models` with a
:class:`~ml.hwdataset.dataset.HwPhaseImageDataset` DataLoader, records the
diagnostics needed to tune it, renders true-vs-pred comparisons, and logs to
Weights & Biases.

Run it::

    python -m ml.zernike.train_amp --families slm_zernike_shaping --epochs 40

Why the corpus is filtered before training
------------------------------------------
:mod:`ml.hwdataset` documents that the families have **genuinely different camera
fields of view** (a 64 px ``region=32`` window, 248/320 px windows, the full
2592x1944 sensor). One ``grid`` therefore means a different physical angular scale
per family, so a single coefficient vector fitted across mixed families would be
fitting an average of incompatible geometries. ``--families`` and ``--fov-px``
exist for that reason and default to one internally-consistent family.

What "perplexity" means here
----------------------------
Perplexity is ``exp(per-token cross-entropy)`` and has **no exact counterpart in
an MSE regression**. It was requested, so :func:`regression_perplexity` reports
``exp(NMSE)`` where ``NMSE = MSE / Var(target)`` -- a perplexity-shaped reading of
the normalised error, where lower is better and ``exp(0) = 1`` is a perfect fit.
It is *not* language-model perplexity; :func:`regression_perplexity` says so in
its own docstring. The trustworthy quality metrics here are ``r2`` and ``psnr``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from ml.hwdataset import (
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_dataloader,
    build_hw_index,
)
from ml.zernike.losses import LossConfig, composite_loss, roi_mask
from ml.zernike.metrics import (
    available_perceptual_metrics,
    batch_image_metrics,
    per_sample_beam_metrics,
    summarise_beam_metrics,
)
from ml.zernike.models import (
    Normalization,
    Observable,
    ZernikeAmpConfig,
    ZernikeAmpFitConfig,
    ZernikeAmpModel,
)

__all__ = [
    "AmpTrainConfig",
    "AmpTrainResult",
    "collect_split",
    "evaluate",
    "grad_statistics",
    "regression_perplexity",
    "render_comparison",
    "train",
]

#: Family used when the caller does not restrict the corpus. One family = one FOV
#: = one physical angular scale, which is what makes a single global coefficient
#: vector meaningful.
DEFAULT_FAMILY: str = "slm_zernike_shaping"

#: Cap on records pulled into GPU memory for one split. The full corpus is ~11.4k
#: records (~0.6 GB as float32 grids); a few hundred is plenty to fit 14
#: coefficients and keeps the run interactive.
DEFAULT_MAX_RECORDS: int = 512

#: ``wandb.init(mode=...)`` accepts these; kept local so the annotation below does
#: not need wandb imported at module scope.
WandbMode = Literal["online", "offline", "disabled", "shared"]


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------
#: Objectives ``train`` knows how to optimise. ``"mse"`` is the incumbent pixel
#: MSE on the normalised frame (bit-identical to the pre-``losses.py`` path);
#: ``"physical"`` optimises the differentiable ROI terms.
LOSS_CHOICES = ("mse", "physical")


@dataclass
class AmpTrainConfig:
    """Everything the training run needs. Every field is a plain value."""

    families: tuple[str, ...] = (DEFAULT_FAMILY,)
    fov_px: int | None = None
    grid: int = 64
    n_max: int = 4
    observable: Observable = "intensity"
    normalization: Normalization = "peak"
    far_field_padding: int = 10
    center_crop: bool = True

    max_train: int = DEFAULT_MAX_RECORDS
    max_val: int = 128
    val_fraction: float = 0.25
    beam_samples: int = 64

    epochs: int = 40
    batch_size: int = 64
    lr: float = 0.02
    l2_penalty: float = 0.0
    grad_clip: float | None = None
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 0

    #: Training objective. ``"mse"`` is the incumbent pixel MSE on the
    #: normalised frame and is bit-identical to the pre-``losses.py`` behaviour.
    #: ``"physical"`` optimises the differentiable ROI terms in
    #: :mod:`ml.zernike.losses` instead, which can see *where* the light lands
    #: rather than only how close each pixel is.
    loss: str = "mse"
    #: Term weights for ``loss="physical"``. Defaults to pib + uniformity with
    #: **no** fidelity term, so selecting ``loss="physical"`` actually changes
    #: the objective; add ``w_mse`` back to optimise the blend instead. Note this
    #: deliberately differs from :class:`~ml.zernike.losses.LossConfig`'s own
    #: default (which is the incumbent ``w_mse=1.0``), because inside the loss
    #: module the neutral default is right, whereas here it would make the
    #: physical switch a no-op.
    loss_weights: LossConfig = field(
        default_factory=lambda: LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0)
    )
    #: ROI side as a fraction of the (centre-cropped) output grid edge. The
    #: default 0.375 reproduces the 24/64 grid the bench tooling uses; it is a
    #: free parameter of the objective, not a fitted constant.
    target_size_frac: float = 0.375
    #: ROI aspect ratio, forwarded verbatim to ``target_shape_roi``.
    target_aspect_ratio: float = 4.0 / 3.0

    out_dir: str = "logs/zernike_amp"
    log_every: int = 1
    image_every: int = 5
    use_wandb: bool = True
    wandb_project: str = "ao-shaping-zernike-amp"
    wandb_name: str | None = None
    wandb_mode: WandbMode = "offline"
    save_checkpoint: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    def optimizer_fit_config(self) -> ZernikeAmpFitConfig:
        """Build the :class:`ZernikeAmpFitConfig` this run's optimiser needs.

        Returns:
            A fit config carrying only the optimiser-relevant fields; ``epochs`` and
            ``lr`` are irrelevant to the factory and are left at their defaults.
        """
        return ZernikeAmpFitConfig(
            optimizer=self.optimizer,  # type: ignore[arg-type]
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )


@dataclass
class AmpTrainResult:
    """Outcome of a training run, also written to ``summary.json``."""

    coefficients: list[float]
    n_modes: int
    n_max: int
    best_epoch: int
    best_val_mse: float
    best_val_r2: float
    best_val_psnr: float
    final_train_mse: float
    grad_norm_first: float
    grad_norm_last: float
    dead_modes: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint: str | None = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_perplexity(mse: float, target_variance: float) -> float:
    """Perplexity-shaped reading of the normalised MSE.

    Args:
        mse: Mean squared error of the prediction.
        target_variance: Variance of the target over the evaluated set.

    Returns:
        ``exp(MSE / Var(target))``. Lower is better; ``1.0`` is a perfect fit in
        the ``NMSE -> 0`` limit.

    .. warning::
        This is **not** language-model perplexity. Perplexity is
        ``exp(mean cross-entropy)`` and cross-entropy has no exact MSE analogue;
        this is a monotone rescaling of the normalised error chosen because it was
        requested and is comparable *across epochs of the same run* only. Trust
        :func:`evaluate`'s ``r2`` / ``psnr`` over this number.
    """
    if not math.isfinite(mse) or target_variance <= 0.0:
        return float("nan")
    try:
        return float(math.exp(min(mse / target_variance, 700.0)))
    except OverflowError:  # pragma: no cover - guarded by the clamp above
        return float("inf")


def grad_statistics(model: ZernikeAmpModel) -> dict[str, float]:
    """Summarise the gradient of the single coefficient vector.

    Per-mode norms are the useful part: a mode with a ~zero gradient is a mode the
    optimiser cannot see, which is how a dead (unidentifiable) direction hides.

    Args:
        model: A model whose ``.coefficients.grad`` was populated by ``backward``.

    Returns:
        ``{"total", "max", "min", "dead"}`` -- the L2 norm over all modes, the
        largest and smallest single-mode norms, and how many modes are below a
        negligible threshold relative to the largest.
    """
    grad = model.coefficients.grad
    if grad is None:
        return {"total": 0.0, "max": 0.0, "min": 0.0, "dead": 0}
    norms = grad.detach().abs().to(torch.float64).cpu().numpy()
    total = float(np.linalg.norm(norms))
    largest = float(norms.max()) if norms.size else 0.0
    # "Dead" = below 1e-6 of the largest mode's gradient.
    dead = int((norms < max(largest * 1e-6, 1e-12)).sum()) if norms.size else 0
    return {
        "total": total,
        "max": largest,
        "min": float(norms.min()) if norms.size else 0.0,
        "dead": float(dead),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def collect_split(
    dataset: HwPhaseImageDataset,
    indices: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Materialise a split into device tensors.

    The DataLoader is the source of truth -- this walks it -- but the resulting
    grid corpus for one family is tens of MB, so it is uploaded once and reused
    every epoch instead of paying the host-to-device copy 40 times.

    Args:
        dataset: The dataset to pull from.
        indices: Dataset positions forming the split.
        device: Where to place the tensors.

    Returns:
        ``{"phase_cos", "phase_sin", "target"}``, each ``(N, 1, grid, grid)``.
    """
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    img_parts: list[torch.Tensor] = []
    for batch in loader:
        cos_parts.append(batch["phase_cos"])
        sin_parts.append(batch["phase_sin"])
        img_parts.append(batch["image"])
    return {
        "phase_cos": torch.cat(cos_parts).to(device),
        "phase_sin": torch.cat(sin_parts).to(device),
        "target": torch.cat(img_parts).to(device),
    }


def _select_records(
    dataset: HwPhaseImageDataset, cfg: AmpTrainConfig
) -> tuple[list[int], list[int]]:
    """Split positions into train/val, optionally filtered by ``fov_px``.

    The split is **by file**, mirroring
    :func:`~ml.hwdataset.dataset.create_hw_dataloaders`: records inside one pickle
    are consecutive epochs of the same optimisation run, so a record-level split
    would put near-duplicates on both sides.
    """
    records = dataset.records
    if cfg.fov_px is not None:
        keep = [
            i
            for i, r in enumerate(records)
            if dataset._fov_px(r) == cfg.fov_px  # noqa: SLF001 - same package
        ]
        if not keep:
            raise SystemExit(f"no records match fov_px={cfg.fov_px}")
        records = [records[i] for i in keep]
        positions = keep
    else:
        positions = list(range(len(records)))

    by_file: dict[str, list[int]] = {}
    for position, record in zip(positions, records, strict=True):
        by_file.setdefault(str(record.path), []).append(position)

    files = sorted(by_file)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(files))
    n_val_files = max(1, int(round(len(files) * cfg.val_fraction)))
    if len(files) - n_val_files < 1:
        raise SystemExit(f"only {len(files)} file(s); cannot split train/val")

    val: list[int] = []
    train: list[int] = []
    for rank, file_index in enumerate(order):
        bucket = val if rank < n_val_files else train
        bucket.extend(by_file[files[file_index]])

    rng.shuffle(train)
    if cfg.max_train > 0:
        train = train[: cfg.max_train]
    if cfg.max_val > 0:
        val = val[: cfg.max_val]
    if not train or not val:
        raise SystemExit(f"empty split: train={len(train)} val={len(val)}")
    return train, val


# ---------------------------------------------------------------------------
# Evaluation / imaging
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: ZernikeAmpModel,
    tensors: dict[str, torch.Tensor],
    batch: int = 256,
    *,
    beam_samples: int = 64,
) -> dict[str, float]:
    """Score the model on a materialised split.

    Reports the img2img-standard set (MSE/RMSE/MAE/NRMSE/PSNR/SSIM/R²) batched on
    the GPU, plus the beam-domain set (centroid offset, spot diameter, correlation,
    efficiency, peak ratio) on the first ``beam_samples`` -- the beam metrics need a
    per-sample numpy pass and are far too slow to run on all 11k records.

    The target is put on the same scale as the prediction using the model's own
    normalisation, exactly as :meth:`ZernikeAmpModel.fit` does, so the numbers are
    directly comparable to the training loss.

    Args:
        model: The model to score.
        tensors: Output of :func:`collect_split`.
        batch: Chunk size, to bound peak memory.
        beam_samples: How many validation samples to run beam metrics on.

    Returns:
        One flat dict of scalars.
    """
    model.eval()
    target = tensors["target"]
    reference = model._normalize(target.clone())  # noqa: SLF001 - same package
    variance = float(torch.var(reference).item())

    predictions = []
    n = target.shape[0]
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        predictions.append(model(tensors["phase_cos"][start:stop], tensors["phase_sin"][start:stop]))
    prediction = torch.cat(predictions)
    model.train()

    out = batch_image_metrics(prediction, reference)
    out["perplexity"] = regression_perplexity(out["mse"], variance)

    rows = [
        per_sample_beam_metrics(
            prediction[i, 0].cpu().numpy(), reference[i, 0].cpu().numpy()
        )
        for i in range(min(beam_samples, n))
    ]
    out.update(summarise_beam_metrics(rows))
    return out


def render_comparison(
    true_img: np.ndarray,
    pred_img: np.ndarray,
    phase: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Write a true-vs-prediction comparison PNG.

    Args:
        true_img: ``(g, g)`` measured CCD frame.
        pred_img: ``(g, g)`` predicted observable.
        phase: ``(g, g)`` commanded pupil phase in radians.
        path: Destination PNG.
        title: Figure title, usually the epoch and the metric.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        (true_img, "true CCD", "inferno"),
        (pred_img, "pred far field", "inferno"),
        (np.abs(pred_img - true_img), "|diff|", "magma"),
        (phase, "pupil phase (rad)", "twilight_shifted"),
    )
    for axis, (data, label, cmap) in zip(axes, panels, strict=True):
        axis.imshow(np.asarray(data, dtype=np.float64), cmap=cmap)
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: AmpTrainConfig) -> AmpTrainResult:
    """Run the full fit and log every diagnostic.

    Args:
        cfg: The run configuration.

    Returns:
        An :class:`AmpTrainResult` with the best coefficients and the full history.
    """
    torch.manual_seed(cfg.seed)
    # Validate the objective name up front. Falling through to MSE on an
    # unrecognised value would make a typo (e.g. "physcial") silently reproduce
    # the incumbent run, which is the one outcome this switch must not have.
    if cfg.loss not in LOSS_CHOICES:
        raise ValueError(f"loss must be one of {LOSS_CHOICES}, got {cfg.loss!r}")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    if cfg.families:
        index = index.filter(families=list(cfg.families))
    dataset = HwPhaseImageDataset(
        index,
        config=MaterialiserConfig(grid=cfg.grid),
        use_cache=True,
    )
    train_idx, val_idx = _select_records(dataset, cfg)
    logger.info(
        "split: train={} val={} from {} records (families={}, fov_px={})",
        len(train_idx),
        len(val_idx),
        len(dataset),
        list(cfg.families) or "all",
        cfg.fov_px,
    )
    train_t = collect_split(dataset, train_idx, device)
    val_t = collect_split(dataset, val_idx, device)

    model = ZernikeAmpModel(
        ZernikeAmpConfig(
            n_max=cfg.n_max,
            grid=cfg.grid,
            observable=cfg.observable,
            normalization=cfg.normalization,
            far_field_padding=cfg.far_field_padding,
        )
    ).to(device)
    fit_target = model._normalize(train_t["target"].clone())  # noqa: SLF001
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run = _init_wandb(cfg, model, len(train_idx), len(val_idx))
    perceptual = available_perceptual_metrics()
    if not perceptual:
        logger.info(
            "no perceptual metric available (lpips/torchvision absent); "
            "reporting SSIM + beam metrics instead"
        )
    history: list[dict[str, float]] = []
    best = {"mse": float("inf"), "epoch": -1, "r2": -float("inf"), "psnr": -float("inf")}
    best_coefficients = model.coefficients_array()
    started = time.perf_counter()
    n_train = train_t["phase_cos"].shape[0]

    # The physical objective needs one ROI mask on the model's own output grid,
    # built by the canonical shape helper so the loss is scored on exactly the
    # region the bench optimiser reports.
    grid = int(train_t["target"].shape[-1])
    loss_mask = None
    if cfg.loss == "physical":
        loss_mask = roi_mask(
            (grid, grid),
            (grid / 2.0, grid / 2.0),
            "rectangle",
            cfg.target_size_frac * grid,
            cfg.target_aspect_ratio,
        )
        logger.info(
            "physical loss: ROI {:.1f}x{:.1f} px of a {}x{} grid, weights={}",
            cfg.target_size_frac * grid,
            cfg.target_size_frac * grid,
            grid,
            grid,
            cfg.loss_weights,
        )

    for epoch in range(cfg.epochs):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        se = 0.0
        penalty = 0.0
        for start in range(0, n_train, cfg.batch_size):
            idx = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_t["phase_cos"][idx], train_t["phase_sin"][idx])
            # Always measured, for every loss: model selection and the reported
            # history stay on val MSE so runs remain comparable across objectives.
            data_mse = torch.mean((prediction - fit_target[idx]) ** 2)
            if loss_mask is not None:
                terms = composite_loss(
                    prediction, fit_target[idx], loss_mask, cfg.loss_weights
                )
                loss = terms["_mean_total"]
            else:
                loss = data_mse
            if cfg.l2_penalty:
                loss = loss + cfg.l2_penalty * torch.sum(model.coefficients**2)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([model.coefficients], cfg.grad_clip)
            optimizer.step()
            se += float(data_mse.detach()) * idx.numel()
            penalty += float(loss.detach()) * idx.numel()
        scheduler.step()

        grads = grad_statistics(model)
        coefficients = model.coefficients_array()
        metrics = evaluate(model, val_t, beam_samples=cfg.beam_samples)
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_mse": se / n_train,
            "train_obj": penalty / n_train,
            **{f"val_{k}": v for k, v in metrics.items()},
            "grad_total": grads["total"],
            "grad_max": grads["max"],
            "grad_min": grads["min"],
            "grad_dead": grads["dead"],
            "coef_abs_max": float(np.abs(coefficients).max()),
            "coef_abs_mean": float(np.abs(coefficients).mean()),
            "coef_l2": float(np.linalg.norm(coefficients)),
        }
        history.append(row)

        if metrics["mse"] < best["mse"]:
            best = {
                "mse": metrics["mse"],
                "epoch": epoch,
                "r2": metrics["r2"],
                "psnr": metrics["psnr"],
            }
            best_coefficients = coefficients

        if epoch % max(cfg.log_every, 1) == 0:
            logger.info(
                "epoch {}/{} train={:.5f} val_mse={:.5f} r2={:+.4f} psnr={:.2f}dB "
                "ssim={:.4f} nrmse={:.4f} ppl={:.3f} | corr={:.3f} eff={:.3f} "
                "d_offset={:.2f}px d_ratio={:.3f} |g|={:.2e} dead={:.0f} "
                "max|c|={:.3f}",
                epoch + 1,
                cfg.epochs,
                row["train_mse"],
                row["val_mse"],
                row["val_r2"],
                row["val_psnr"],
                row["val_ssim"],
                row["val_nrmse"],
                row["val_perplexity"],
                row["val_correlation"],
                row["val_efficiency"],
                row["val_centroid_offset_px"],
                row["val_spot_diameter_ratio"],
                row["grad_total"],
                row["grad_dead"],
                row["coef_abs_max"],
            )
        if run is not None:
            run.log(row, step=epoch)
        if cfg.image_every and (epoch % cfg.image_every == 0 or epoch == cfg.epochs - 1):
            with torch.no_grad():
                sample = model(val_t["phase_cos"][:1], val_t["phase_sin"][:1])
            true_img = model._normalize(val_t["target"][:1].clone())  # noqa: SLF001
            figure_path = out_dir / f"compare_epoch{epoch:03d}.png"
            caption = (
                f"epoch {epoch}  val_mse={metrics['mse']:.5f}  "
                f"r2={metrics['r2']:.4f}  psnr={metrics['psnr']:.2f}dB  "
                f"max|c|={row['coef_abs_max']:.3f} rad"
            )
            render_comparison(
                true_img[0, 0].cpu().numpy(),
                sample[0, 0].cpu().numpy(),
                model.correction_phase().detach().cpu().numpy(),
                figure_path,
                caption,
            )
            if run is not None:
                _log_comparison_image(run, figure_path, epoch, caption)

    seconds = time.perf_counter() - started
    checkpoint = None
    if cfg.save_checkpoint:
        checkpoint_path = out_dir / "best_coefficients.pt"
        torch.save(
            {
                "coefficients": torch.from_numpy(best_coefficients),
                "n_max": cfg.n_max,
                "grid": cfg.grid,
                "observable": cfg.observable,
                "normalization": cfg.normalization,
                "far_field_padding": cfg.far_field_padding,
                "config": {k: v for k, v in asdict(cfg).items() if k != "extra"},
            },
            checkpoint_path,
        )
        checkpoint = str(checkpoint_path)

    result = AmpTrainResult(
        coefficients=[float(v) for v in best_coefficients],
        n_modes=model.K,
        n_max=cfg.n_max,
        best_epoch=best["epoch"],
        best_val_mse=best["mse"],
        best_val_r2=best["r2"],
        best_val_psnr=best["psnr"],
        final_train_mse=history[-1]["train_mse"] if history else float("nan"),
        grad_norm_first=history[0]["grad_total"] if history else 0.0,
        grad_norm_last=history[-1]["grad_total"] if history else 0.0,
        dead_modes=int(history[-1]["grad_dead"]) if history else 0,
        seconds=seconds,
        history=history,
        checkpoint=checkpoint,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "best epoch {} val_mse={:.5f} r2={:.4f} psnr={:.2f}dB in {:.1f}s -> {}",
        result.best_epoch,
        result.best_val_mse,
        result.best_val_r2,
        result.best_val_psnr,
        seconds,
        out_dir,
    )
    if run is not None:
        _log_wandb_summary(run, result)
        run.finish()
    return result


def _build_optimizer(
    model: ZernikeAmpModel, cfg: AmpTrainConfig
) -> torch.optim.Optimizer:
    """Build the optimizer named by ``cfg.optimizer`` over the coefficients.

    Delegating to the model's own factory keeps one definition. This used to be a
    hard-coded ``torch.optim.Adam(...)``, so the ``optimizer`` lever of the sweep
    reported byte-identical numbers for adam / adamw / sgd and quietly measured
    nothing.

    Args:
        model: The model whose parameters are optimised.
        cfg: Run configuration naming the optimiser.

    Returns:
        A configured torch optimizer.

    Raises:
        ValueError: If ``cfg.optimizer`` is not a supported name.
    """
    if cfg.optimizer not in ("adam", "adamw", "sgd"):
        raise ValueError(
            f"optimizer must be one of ['adam', 'adamw', 'sgd'], got {cfg.optimizer!r}"
        )
    fit = cfg.optimizer_fit_config()
    return model._build_optimizer(fit)  # noqa: SLF001 - same package


def _init_wandb(cfg: AmpTrainConfig, model: ZernikeAmpModel, n_train: int, n_val: int):
    """Start a wandb run, degrading to a local log if wandb is unavailable.

    Offline mode is the default: this machine has no ``WANDB_API_KEY``, so an
    online run would fail. The offline directory syncs later with
    ``wandb sync <dir>``.
    """
    if not cfg.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; continuing without it")
        return None
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            mode=cfg.wandb_mode,
            dir=str(Path(cfg.out_dir)),
            config={
                **{k: v for k, v in asdict(cfg).items() if k != "extra"},
                "n_modes": model.K,
                "n_train_records": n_train,
                "n_val_records": n_val,
            },
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001 - never let telemetry kill a run
        logger.warning("wandb.init failed ({}); continuing offline-only", exc)
        return None
    # One fixed random coefficient vector, so the "true vs pred" panel has a
    # reproducible meaning across runs.
    wandb.log({"model/n_modes": model.K})
    return run


def _log_comparison_image(run, path: Path, epoch: int, caption: str) -> None:
    """Attach one true-vs-pred comparison figure to the wandb run.

    The PNG is already on disk, so it is attached by path rather than re-rendered
    into a ``wandb.Image`` in memory. Any wandb-side failure is logged and
    swallowed: telemetry must never take down a training run that has already
    produced its artefacts.
    """
    try:
        import wandb

        run.log(
            {
                "compare/true_vs_pred": wandb.Image(
                    str(path), caption=caption, file_type="png"
                )
            },
            step=epoch,
        )
    except Exception as exc:  # noqa: BLE001 - never let telemetry kill a run
        logger.warning("could not log {} to wandb: {}", path.name, exc)


def _log_wandb_summary(run, result: AmpTrainResult) -> None:
    """Push the end-of-run summary and the final comparison image."""
    run.summary["best_epoch"] = result.best_epoch
    run.summary["best_val_mse"] = result.best_val_mse
    run.summary["best_val_r2"] = result.best_val_r2
    run.summary["best_val_psnr"] = result.best_val_psnr
    run.summary["grad_norm_first"] = result.grad_norm_first
    run.summary["grad_norm_last"] = result.grad_norm_last
    run.summary["dead_modes"] = result.dead_modes
    run.summary["coefficients"] = result.coefficients


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Numeric defaults are read off :class:`ZernikeAmpConfig` rather than repeated
    as literals. A hardcoded ``--far-field-padding default=1`` silently overrode
    the calibrated dataclass default of 10 and made every run start at R^2 = -0.38,
    which is exactly the sort of default drift the repo's AGENTS.md warns about.
    """
    model_default = ZernikeAmpConfig()
    train_default = AmpTrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m ml.zernike.train_amp",
        description="Train ZernikeAmpModel on the real hardware corpus.",
    )
    parser.add_argument("--families", nargs="*", default=[DEFAULT_FAMILY])
    parser.add_argument("--fov-px", type=int, default=None)
    parser.add_argument("--grid", type=int, default=model_default.grid)
    parser.add_argument("--n-max", type=int, default=model_default.n_max)
    parser.add_argument("--observable", default=model_default.observable)
    parser.add_argument("--normalization", default=model_default.normalization)
    parser.add_argument(
        "--far-field-padding", type=int, default=model_default.far_field_padding
    )
    parser.add_argument(
        "--no-center-crop",
        dest="center_crop",
        action="store_false",
        default=model_default.center_crop,
        help="emit the full far field instead of the centre-cropped grid",
    )
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--max-val", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--beam-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2-penalty", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--loss",
        choices=list(LOSS_CHOICES),
        default=train_default.loss,
        help=(
            "Training objective. 'mse' is the incumbent pixel MSE on the "
            "normalised far field; 'physical' optimises the differentiable ROI "
            "terms (pib + uniformity), which can see where the light lands."
        ),
    )
    parser.add_argument(
        "--target-size-frac",
        type=float,
        default=train_default.target_size_frac,
        help="ROI side as a fraction of the output grid edge (physical loss only).",
    )
    parser.add_argument(
        "--target-aspect-ratio",
        type=float,
        default=train_default.target_aspect_ratio,
        help="ROI aspect ratio (physical loss only).",
    )
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], default="adam"
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="logs/zernike_amp")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--image-every", type=int, default=5)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ao-shaping-zernike-amp")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    cfg = AmpTrainConfig(
        families=tuple(args.families),
        fov_px=args.fov_px,
        grid=args.grid,
        n_max=args.n_max,
        observable=args.observable,
        normalization=args.normalization,
        far_field_padding=args.far_field_padding,
        center_crop=args.center_crop,
        max_train=args.max_train,
        max_val=args.max_val,
        val_fraction=args.val_fraction,
        beam_samples=args.beam_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_penalty=args.l2_penalty,
    loss=args.loss,
    target_size_frac=args.target_size_frac,
    target_aspect_ratio=args.target_aspect_ratio,
        grad_clip=args.grad_clip,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=0,
        out_dir=args.out_dir,
        log_every=args.log_every,
        image_every=args.image_every,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode,
        save_checkpoint=not args.no_checkpoint,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())