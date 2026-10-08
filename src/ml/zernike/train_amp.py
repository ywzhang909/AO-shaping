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
import re
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
from ml.zernike.losses import (
    LossConfig,
    composite_loss,
    roi_mask,
    spot_moment_gap_term,
)
from ml.zernike.metrics import (
    available_perceptual_metrics,
    batch_image_metrics,
    constant_baseline_metrics,
    per_sample_beam_metrics,
    roi_shape_terms,
    summarise_beam_metrics,
    total_variation_ratio,
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
    "objective_of",
    "per_objective_r2",
    "regression_perplexity",
    "render_comparison",
    "split_mse_r2",
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

#: Pickle-name shape for the ``slm_zernike_shaping`` family. The family packs 10
#: pickles from four optimisation objectives (``rms_pib`` / ``rmse_out`` /
#: ``shape`` / ``roi_pib``) -- four distinct bench states behind one family name.
#: A stem looks like ``slm_zernike_shaping_<objective>_<YYYYMMDD>_<HHMMSS>[_...].pkl``
#: and carries a timestamp (sometimes two), so the objective group is matched
#: **lazily** -- the first ``_<objective>`` before the first date stamp -- mirroring
#: ``scripts/compare_models_cv.py`` so the two never drift.
_OBJECTIVE_RE = re.compile(r"^slm_zernike_shaping_(?P<objective>.+?)_\d{8}_\d{6}")


def objective_of(path: Any) -> str:
    """Return the optimisation objective encoded in a ``slm_zernike_shaping`` path.

    The four objectives are very unevenly represented (``roi_pib`` has a single
    file), which is exactly why the train/val split must be stratified by objective
    instead of shuffling records. Paths that do not carry the objective (other
    families) fall back to ``"?"`` so the stratification simply treats them as one
    group.

    Args:
        path: A ``Path`` or ``str`` to the corpus pickle.

    Returns:
        The objective substring, or ``"?"`` when it cannot be parsed.
    """
    match = _OBJECTIVE_RE.match(Path(path).stem)
    return match.group("objective") if match else "?"

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
    #: Substring filter on the artefact path, applied before ``fov_px``. Lets a
    #: run cover one slice of a family -- needed because
    #: ``slm_zernike_shaping`` mixes four optimisation objectives, hence four
    #: distinct bench states, behind one family name.
    file_contains: str | None = None
    grid: int = 64
    #: Dataset-level image mode, forwarded to ``MaterialiserConfig``. ``"abs255"``
    #: (default) keeps the absolute CCD brightness, which is what makes the exposure
    #: readable from the target. ``"peak"`` / ``"robust"`` normalise per frame instead,
    #: which discards the absolute level; both are legitimate but answer a different
    #: question, so the mode is recorded in the run's config rather than implied.
    image_mode: str = "abs255"
    n_max: int = 4
    observable: Observable = "intensity"
    normalization: Normalization = "peak"
    far_field_padding: int = 10
    center_crop: bool = True

    #: Trainable self-attention refinement after the far field (see
    #: :class:`~ml.zernike.models.ZernikeAmpConfig.attention`). Off by default:
    #: the pure-physics path is a closed-form function of its coefficients, so
    #: the phase it implies is directly implementable on the SLM, and an
    #: attention block is not invertible back to a phase. When enabled it is
    #: identity-initialised, so training starts from the physics rather than from
    #: noise, and it is optimised alongside the coefficients.
    attention: bool = False
    attention_grid: int = 16
    attention_dim: int = 32
    attention_heads: int = 1

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
    #: Term weights for ``loss="physical"``. Defaults to the **anchored** physical
    #: term: MSE for fidelity plus ``w_shape_gap`` for the ROI shape statistics,
    #: relative-normalised so the two are comparable weights.
    #:
    #: It deliberately does NOT default to ``w_pib``/``w_uniformity``. Those are
    #: unanchored -- they read only the prediction -- so on a *fitting* task their
    #: optimum is "ignore the data and emit an ideal spot". Measured here:
    #: ``w_pib=1, w_uniformity=1`` drove val R2 from +0.78 to **-0.86** while
    #: ``shape_sum`` reached 1.496 against a measured 1.113, i.e. 34% "better"
    #: than the physics being predicted. Even ``w_mse=1`` alongside them left
    #: R2 at -0.28, because the physical terms are ``O(1)`` and a peak-normalised
    #: MSE is ``O(0.003)`` -- roughly 450x, so the fidelity anchor carried about
    #: 0.2% of the gradient. Keep the unanchored pair for *shaping* (inverting a
    #: model to produce a phase), not for training one.
    loss_weights: LossConfig = field(
        default_factory=lambda: LossConfig(w_mse=1.0, w_shape_gap=1.0)
    )
    #: Rescale the model's output so its intensity sum equals the input phasor's.
    #: Off by default (see ``ZernikeAmpConfig.conserve_energy``); enable it when the
    #: loss should be able to anchor against absolute energy.
    conserve_energy: bool = False
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


def _stratified_cap(
    positions: list[int], file_to_positions: dict[str, list[int]], cap: int
) -> list[int]:
    """Return at most ``cap`` of ``positions`` without collapsing the objective mix.

    A bare ``positions[:cap]`` cuts in file order, so on the default 128-record cap
    it would drop one of the two objectives present in the validation split. This
    instead allocates the cap across the files (largest-remainder proportional) so
    every objective that has files in the split is retained -- which is the whole
    point of the stratified split.

    Args:
        positions: All positions of one split (file order preserved).
        file_to_positions: ``str(path) -> positions`` for the files in the split.
        cap: Maximum number of positions to keep.

    Returns:
        A subset of ``positions`` of size ``min(cap, len(positions))``.
    """
    if cap <= 0 or len(positions) <= cap:
        return list(positions)
    weights = [len(v) for v in file_to_positions.values()]
    total = float(sum(weights))
    # Largest-remainder so the integers sum to exactly ``cap``.
    raw = [cap * w / total for w in weights]
    quotas = [int(math.floor(r)) for r in raw]
    deficit = cap - sum(quotas)
    # Hand the leftover slots to the files with the largest fractional remainders,
    # tie-broken by file order for determinism.
    for i, _ in sorted(
        enumerate(raw), key=lambda t: (-(t[1] - quotas[t[0]]), t[0])
    )[:deficit]:
        quotas[i] += 1
    out: list[int] = []
    for (fname, fpos), q in zip(file_to_positions.items(), quotas, strict=True):
        out.extend(fpos[:q])
    return out


def _select_records(
    dataset: HwPhaseImageDataset, cfg: AmpTrainConfig
) -> tuple[list[int], list[int]]:
    """Split positions into train/val, optionally filtered by ``fov_px``.

    The split is **by file**, mirroring
    :func:`~ml.hwdataset.dataset.create_hw_dataloaders`: records inside one pickle
    are consecutive epochs of the same optimisation run, so a record-level split
    would put near-duplicates on both sides.

    It is additionally **stratified by optimisation objective**. The
    ``slm_zernike_shaping`` family packs 10 pickles from four objectives
    (``rms_pib`` / ``rmse_out`` / ``shape`` / ``roi_pib``) -- four distinct bench
    states behind one family name. A plain 25%-of-files draw (plus the 128-record
    cap) happened to land both validation files on one objective, so the single
    global coefficient vector was judged on one bench state while trained on a
    mixture of four -- the cause of the large train/val gap. Stratifying assigns
    validation files per objective so every objective that has >= 2 files is
    represented on both sides, and the size caps are applied proportionally
    (see :func:`_stratified_cap`) so the retained mix survives truncation.
    """
    records = dataset.records
    if cfg.file_contains is not None:
        # Substring match on the artefact path. The ``slm_zernike_shaping`` family
        # packs 10 pickles from **4 different optimisation objectives** (measured:
        # rms_pib 404 / rmse_out 303 / shape 202 / roi_pib 101 records), and each
        # objective's run leaves the SLM at a different aberration -- so the
        # family is four distinct bench states sharing one family name, and a
        # single global coefficient vector has to compromise across all four.
        # This filter is what makes "train on one objective" expressible.
        needle = cfg.file_contains
        keep = [i for i, r in enumerate(records) if needle in str(r.path)]
        if not keep:
            raise SystemExit(f"no records match file_contains={needle!r}")
        records = [records[i] for i in keep]
        positions = keep
    elif cfg.fov_px is not None:
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
    file_objective: dict[str, str] = {}
    for position, record in zip(positions, records, strict=True):
        key = str(record.path)
        by_file.setdefault(key, []).append(position)
        file_objective[key] = objective_of(key)

    # Group files by objective so the train/val draw is stratified.
    by_objective: dict[str, list[str]] = {}
    for fname in by_file:
        by_objective.setdefault(file_objective[fname], []).append(fname)

    rng = np.random.default_rng(cfg.seed)
    val_files: set[str] = set()
    train_files: set[str] = set()
    for obj, fns in by_objective.items():
        fns_sorted = sorted(fns)
        if len(fns_sorted) >= 2:
            order = rng.permutation(len(fns_sorted))
            n_val = max(1, int(round(len(fns_sorted) * cfg.val_fraction)))
            for rank, idx in enumerate(order):
                (val_files if rank < n_val else train_files).add(fns_sorted[idx])
        else:
            # A single-file objective cannot be split, so it stays in train only;
            # that objective is simply absent from validation, which is honest.
            train_files.update(fns_sorted)

    if not val_files or not train_files:
        raise SystemExit(f"cannot split {len(by_file)} file(s) across objectives")

    train: list[int] = []
    val: list[int] = []
    train_files_sorted: list[str] = []
    val_files_sorted: list[str] = []
    for fname in sorted(by_file):
        if fname in val_files:
            val_files_sorted.append(fname)
            val.extend(by_file[fname])
        else:
            train_files_sorted.append(fname)
            train.extend(by_file[fname])

    rng.shuffle(train)
    rng.shuffle(val)
    if cfg.max_train > 0:
        train = _stratified_cap(train, {f: by_file[f] for f in train_files_sorted}, cfg.max_train)
    if cfg.max_val > 0:
        val = _stratified_cap(val, {f: by_file[f] for f in val_files_sorted}, cfg.max_val)
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
    roi_size_frac: float | None = None,
    roi_aspect: float = 1.0,
) -> dict[str, float]:
    """Score the model on a materialised split.

    Reports the img2img-standard set (MSE/RMSE/MAE/NRMSE/PSNR/SSIM/R²) batched on
    the GPU, plus ``tv_ratio`` (detail retention, 1.0 = same as the target) and the
    beam-domain set (centroid offset, spot diameter, correlation,
    efficiency, peak ratio) on the first ``beam_samples`` -- the beam metrics need a
    per-sample numpy pass and are far too slow to run on all 11k records.

    ``tv_ratio`` is here because SSIM alone is misleading for this model: on the strongest
    arm of the architecture search, SSIM ranked the blurriest prediction first.
    See :func:`~ml.zernike.metrics.total_variation_ratio`.

    When ``roi_size_frac`` is given, the ROI shape terms (``pib_term`` /
    ``uniformity`` / ``shape_sum``, plus the measured frame's own) are added via
    :func:`~ml.zernike.metrics.roi_shape_terms`. Those are the quantities
    ``loss="physical"`` actually optimises; without them a loss change is
    unmeasurable, because the beam set alone never says where the light landed
    inside the target box. Pass the *same* fraction the loss was configured with
    so the objective and the score refer to one region.

    The target is put on the same scale as the prediction using the model's own
    normalisation, exactly as :meth:`ZernikeAmpModel.fit` does, so the numbers are
    directly comparable to the training loss.

    Args:
        model: The model to score.
        tensors: Output of :func:`collect_split`.
        batch: Chunk size, to bound peak memory.
        beam_samples: How many validation samples to run beam metrics on.
        roi_size_frac: ROI short side as a fraction of the grid edge, or ``None``
            to skip the ROI terms.

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
    # The metric floor, made explicit. On this corpus a *constant* image scores
    # R^2 ~= +0.910, so the R^2 above cannot be read as model quality on its own;
    # `skill` says how much of that constant baseline the model actually removes
    # (0.0 = no better than one image for everything, negative = worse).
    # See `report/zernike_r2_baseline/report.md`.
    out.update(constant_baseline_metrics(prediction, reference))
    out["perplexity"] = regression_perplexity(out["mse"], variance)
    # Detail retention. Included because SSIM cannot see over-smoothing: the U-Net arm won on
    # SSIM while being the blurriest prediction we trained, so an SSIM-only report would have
    # ranked that failure as a success. 1.0 = same detail as the target, <1 smoother,
    # >1 rougher. See `ml.zernike.metrics.total_variation_ratio` for the sweep that chose it.
    out["tv_ratio"] = total_variation_ratio(prediction, reference)

    rows = [
        per_sample_beam_metrics(
            prediction[i, 0].cpu().numpy(), reference[i, 0].cpu().numpy()
        )
        for i in range(min(beam_samples, n))
    ]
    out.update(summarise_beam_metrics(rows))

    if roi_size_frac:
        grid = int(target.shape[-1])
        side = float(roi_size_frac) * grid
        centre = (grid / 2.0, grid / 2.0)
        roi_rows = [
            roi_shape_terms(
                prediction[i, 0].cpu().numpy(),
                reference[i, 0].cpu().numpy(),
                center=centre,
                size=side,
            )
            for i in range(min(beam_samples, n))
        ]
        out.update(summarise_beam_metrics(roi_rows))
        # The anchored spot-size gap the new term optimises, on the same ROI, so a run
        # that enables the term is observable rather than assumed. One batched call on
        # the concatenated tensor -- `prediction` is (N, 1, g, g); `predictions` is the
        # list of per-batch chunks and indexing that indexes BATCHES, not samples.
        k = min(beam_samples, n)
        out["spot_moment_gap"] = float(
            spot_moment_gap_term(
                prediction[:k],
                reference[:k],
                roi_mask((grid, grid), centre, "rectangle", side, roi_aspect),
            ).mean()
        )
    return out


@torch.no_grad()
def per_objective_r2(
    model: ZernikeAmpModel,
    tensors: dict[str, torch.Tensor],
    groups: list[str],
    batch: int = 256,
) -> dict[str, float]:
    """Per-group R² on one materialised split, keyed ``r2_<group>``.

    The aggregate R² in :func:`evaluate` averages every sample together, so on the
    ``slm_zernike_shaping`` family it hides how well each objective's bench state is
    predicted. With the train/val gap dominated by objective heterogeneity, the
    per-objective breakdown is the number that actually explains a run.

    Args:
        model: The model to score.
        tensors: Output of :func:`collect_split`.
        groups: One group label (see :func:`objective_of`) per sample, in the same
            order as ``tensors["target"]``.
        batch: Chunk size for the forward pass.

    Returns:
        ``{f"r2_{group}": value}`` for every distinct group with >= 2 samples;
        smaller groups map to ``nan`` so their absence is visible, not silent.
    """
    model.eval()
    target = tensors["target"]
    reference = model._normalize(target.clone())  # noqa: SLF001 - same package
    n = target.shape[0]
    chunks = [
        model(tensors["phase_cos"][start : start + batch], tensors["phase_sin"][start : start + batch])
        for start in range(0, n, batch)
    ]
    prediction = torch.cat(chunks)
    model.train()
    out: dict[str, float] = {}
    for group in dict.fromkeys(groups):
        mask = torch.as_tensor([g == group for g in groups])
        if int(mask.sum()) < 2:
            out[f"r2_{group}"] = float("nan")
            continue
        # Scored with the canonical metric on the subgroup, so a per-group R^2 and the
        # aggregate ``val_r2`` are the same quantity -- not two definitions that happen
        # to look alike (``batch_image_metrics`` uses ``1 - mse/var`` with ``torch.var``'s
        # n-1 denominator; re-deriving it as ``1 - ss_res/ss_tot`` shifts R^2 by ~1e-2
        # at n=320 and made train/val gaps meaningless).
        out[f"r2_{group}"] = batch_image_metrics(prediction[mask], reference[mask])["r2"]
    return out


@torch.no_grad()
def split_mse_r2(
    model: ZernikeAmpModel,
    tensors: dict[str, torch.Tensor],
    batch: int = 256,
) -> tuple[float, float]:
    """Whole-split MSE and R² for one materialised split, at the current weights.

    Why this exists: the ``train_mse`` in the history is an average of the
    **in-epoch** losses, i.e. measured while the weights were still moving, whereas
    ``val_mse`` is always measured with the **end-of-epoch** weights. Dividing one by
    the other -- the obvious "train/val gap" -- therefore mixes two epochs, and it also
    compares a running average against a point measurement. These two numbers are the
    apples-to-apples pair: same weights, same peak normalisation, same forward path as
    :func:`evaluate`.

    Measured on real ``slm_zernike_shaping`` data (6 seeds, n_max=4, 40 epochs), the
    mixed-epoch ratio sits at ~1.15 with a seed-to-seed spread of 0.63-2.32; the
    same-epoch R² difference is ~0.00-0.05, i.e. **there is no meaningful
    generalisation gap at this model size** (14 coefficients vs 512 samples). A large
    apparent gap is a split-composition artefact, not overfitting.

    Args:
        model: The model to score.
        tensors: Output of :func:`collect_split`.
        batch: Chunk size for the forward pass.

    Returns:
        ``(mse, r2)``, taken straight from
        :func:`~ml.zernike.metrics.batch_image_metrics` so the definition cannot drift
        from :func:`evaluate`'s ``val_mse`` / ``val_r2``.
    """
    model.eval()
    target = tensors["target"]
    reference = model._normalize(target.clone())  # noqa: SLF001 - same package
    n = target.shape[0]
    prediction = torch.cat(
        [
            model(tensors["phase_cos"][start : start + batch], tensors["phase_sin"][start : start + batch])
            for start in range(0, n, batch)
        ]
    )
    model.train()
    out = batch_image_metrics(prediction, reference)
    return out["mse"], out["r2"]


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
        config=MaterialiserConfig(grid=cfg.grid, image_mode=cfg.image_mode),
        use_cache=True,
    )
    train_idx, val_idx = _select_records(dataset, cfg)
    # Per-objective labels for the validation samples, aligned to ``val_idx`` order,
    # so the per-objective R^2 breakdown can be reported every epoch.
    val_objectives = [objective_of(dataset.records[i].path) for i in val_idx]
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
    conserve_energy=cfg.conserve_energy,
            far_field_padding=cfg.far_field_padding,
            center_crop=cfg.center_crop,
            attention=cfg.attention,
            attention_grid=cfg.attention_grid,
            attention_dim=cfg.attention_dim,
            attention_heads=cfg.attention_heads,
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
    if (
        cfg.loss == "physical"
        or cfg.loss_weights.w_spot_moment > 0.0
        or cfg.loss_weights.w_ellipse > 0.0
    ):
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
                torch.nn.utils.clip_grad_norm_(_trainable_params(model), cfg.grad_clip)
            optimizer.step()
            se += float(data_mse.detach()) * idx.numel()
            penalty += float(loss.detach()) * idx.numel()
        scheduler.step()

        grads = grad_statistics(model)
        coefficients = model.coefficients_array()
        metrics = evaluate(
            model,
            val_t,
            beam_samples=cfg.beam_samples,
            roi_size_frac=cfg.target_size_frac,
            roi_aspect=cfg.target_aspect_ratio,
        )
        # The aggregate val R^2 hides how well each objective's bench state is
        # predicted -- the very axis that drives the train/val gap. A second,
        # cheap forward over the (small) validation split keeps the per-objective
        # breakdown available without polluting evaluate()'s float-only metrics dict.
        metrics.update(per_objective_r2(model, val_t, val_objectives))
        # Same-epoch train score: same weights, same normalisation, same code path as
        # the val metrics above. This -- not `train_mse` -- is what a train/val gap has
        # to be computed from (see split_mse_r2).
        train_eval_mse, train_eval_r2 = split_mse_r2(model, train_t)
        val_mse = float(metrics["mse"])
        val_r2 = float(metrics["r2"])
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_mse": se / n_train,
            "train_obj": penalty / n_train,
            "train_eval_mse": train_eval_mse,
            "train_eval_r2": train_eval_r2,
            #: Same-epoch val/train MSE ratio. 1.0 = no generalisation gap. This is the
            #: only gap number comparable across runs; `best_val_mse / final_train_mse`
            #: mixes epochs and is not.
            "gap_mse": val_mse / train_eval_mse if train_eval_mse > 0.0 else float("nan"),
            #: Same-epoch R^2 difference (val minus train). 0.0 = no gap.
            "gap_r2": val_r2 - train_eval_r2,
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
                "max|c|={:.3f} | tr_eval={:.5f} tr_r2={:+.4f} gap_mse={:.2f} gap_r2={:+.3f}",
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
                row["train_eval_mse"],
                row["train_eval_r2"],
                row["gap_mse"],
                row["gap_r2"],
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


def _trainable_params(model: ZernikeAmpModel) -> list[torch.nn.Parameter]:
    """Every trainable tensor on the model, not just the coefficients.

    ``_build_optimizer`` already optimises ``model.parameters()``, so an optional
    attention block is trained. Gradient *clipping* was still hard-wired to
    ``[model.coefficients]``, which would have left the attention's gradients
    unclipped while the physics was clipped -- silently applying two different
    effective learning rates to two parts of the same model.
    """
    return [p for p in model.parameters() if p.requires_grad]


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
    parser.add_argument(
        "--file-contains",
        type=str,
        default=None,
        help=(
            "Substring filter on the artefact path, applied before the split. On the "
            "slm_zernike_shaping family this is what makes 'train on one objective' "
            "expressible, e.g. --file-contains rms_pib."
        ),
    )
    parser.add_argument("--fov-px", type=int, default=None)
    parser.add_argument("--grid", type=int, default=model_default.grid)
    parser.add_argument(
        "--image-mode",
        default=train_default.image_mode,
        help=(
            "Dataset-level image mode: abs255 (default, keeps absolute CCD brightness "
            "so the exposure stays readable) | peak | robust (per-frame normalisation, "
            "discards the absolute level)."
        ),
    )
    parser.add_argument("--n-max", type=int, default=model_default.n_max)
    parser.add_argument("--observable", default=model_default.observable)
    parser.add_argument("--normalization", default=model_default.normalization)
    parser.add_argument(
        "--conserve-energy",
        action="store_true",
        help="Rescale the prediction so its intensity sum equals the input phasor's",
    )
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
        "--w-spot-moment",
        type=float,
        default=0.0,
        help=(
            "Weight on the anchored spot-size term |var_r(pred)-var_r(ref)|/var_r(ref), "
            "var_r = second moment about the intensity centroid. >0 switches the "
            "objective to mse + this term, and makes the val moment gap visible in the "
            "history. Trades fidelity for spot size -- measure before enabling."
        ),
    )
    parser.add_argument(
        "--w-ellipse",
        type=float,
        default=0.0,
        help=(
            "Weight on the anchored ellipse-fit gap (centroid x/y, var_x, var_y, "
            "covariance). Richer than --w-spot-moment: sees spot position and tilt, "
            "which the radial term scores as zero."
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
        file_contains=args.file_contains,
        fov_px=args.fov_px,
        grid=args.grid,
        image_mode=args.image_mode,
        n_max=args.n_max,
        observable=args.observable,
        normalization=args.normalization,
        conserve_energy=args.conserve_energy,
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
        loss_weights=LossConfig(
            w_mse=1.0,
            w_shape_gap=1.0,
            w_spot_moment=args.w_spot_moment,
            w_ellipse=args.w_ellipse,
        ),
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