"""Train the amortising U-Net: requested spot -> aberration-free pupil phase.

Run it as ``python -m ml.sim2f train --cache <dir>`` (or
``python -m ml.sim2f.train``).  The corpus is the archive written by
:mod:`ml.sim2f.synth`; :func:`load_corpus` is the single reader of it and both
this module and :mod:`ml.sim2f.evaluate` use that one, so a checkpoint and its
evaluation cannot disagree about the data or about the optics that produced it.

============================  Why the parser reads the dataclasses  ============================

Every default below is taken from a constructed :class:`TrainConfig` (or
``SynthConfig`` / ``TeacherConfig`` / ``Sim2fConfig`` in :mod:`ml.sim2f.__main__`)
rather than written as a literal.  This repository shipped exactly that bug: the
``train_amp`` parser hard-wrote ``default=1`` for ``far_field_padding`` and
silently overrode a calibrated value of ``10`` on every run
(``report/zernike_phase2amp/report.md``).  Constructing the frozen dataclass once
and reading from it makes "the CLI default" and "the code default" the same
object, so they cannot drift.

============================  Selection metric  ============================

The checkpoint is selected on the mean validation **physics objective** -- the
teacher's own ``L_shape + w_ee * (1 - EE)`` -- because that is the quantity the
report compares ("flat / net / teacher"), and unlike ``ml.shaping`` there is no
undefined subset here: every sample of this corpus has a bench, so selecting on
it introduces no family bias.  The supervised phasor L1 is logged alongside but
is not the selection metric: it can be lowered by copying the teacher's arbitrary
global piston, which says nothing about shaping quality.

A corpus may legally have **no validation samples** (a smoke run of a handful of
pairs is all train).  That is detected once, before the loop: validation is
switched off with a single log line, ``history.csv`` carries train columns only
(empty val cells, never ``nan``), and ``model_best.pt`` is the final state, so
downstream ``evaluate`` still finds the conventional name.  The alternative --
letting the loop "validate" an empty split -- writes a row of NaNs every
``val_every`` steps and leaves every checkpoint unselectable.

============================  Determinism  ============================

:meth:`TrainConfig.cudnn_deterministic` (default on) sets
``torch.backends.cudnn.deterministic``, ``benchmark = False`` and
``torch.use_deterministic_algorithms(True, warn_only=True)``.  The repository
measured a 0.006 R2 spread between same-seed runs without the flags, which is
larger than several effects the reports compare.  Wall clock is deliberately
*not* written into ``history.csv``: a rerun with the same seed must produce a
byte-identical file, and a timing column would make that impossible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ml.sim2f.config import OpticsErrors, Sim2fConfig, TeacherConfig, TrainConfig
from ml.sim2f._device import _resolve_device
from ml.sim2f.losses import (
    piston_free_phasor_l1,
    physics_loss,
    to_phasor,
    tv_phasor,
)
from ml.sim2f.model import Sim2fUNet, build_unet, network_input
from ml.sim2f.schema import (
    CACHE_FILE,
    META_FILE,
    SPLIT_NAMES,
    validate_cache,
)

__all__ = [
    "CHECKPOINT_BEST",
    "CHECKPOINT_LAST",
    "Corpus",
    "build_parser",
    "context_from_corpus",
    "load_corpus",
    "main",
    "seed_everything",
    "train",
    "training_objective",
]

#: Written every time validation improves on the physics objective.
CHECKPOINT_BEST = "model_best.pt"
#: Written at the end of the run (and every validation event, so a killed run
#: still leaves the latest weights).
CHECKPOINT_LAST = "model_last.pt"
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"
HISTORY_NAME = "history.csv"

HISTORY_COLUMNS: tuple[str, ...] = (
    "step",
    "lr",
    "train_total",
    "train_sup",
    "train_phy",
    "train_tv",
    "val_total",
    "val_sup",
    "val_phy",
    "val_tv",
)


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------
@dataclass
class Corpus:
    """The cached supervised corpus, in memory, as tensors.

    Attributes:
        target: ``(N, 1, ccd_px, ccd_px)`` requested distributions, unit sum.
        support: ``(N, 1, ccd_px, ccd_px)`` analytic supports, ``0.0``/``1.0``.
        phase: ``(N, grid, grid)`` teacher phases, raw radians, aberration-free.
        spot: ``(N, 1, ccd_px, ccd_px)`` ideal flat-phase frames.
        teacher_metrics: ``(N, 4)`` cached teacher columns
            ``[objective, l1, ee, ncc]``.
        kind: ``(N,)`` int64 index into :data:`ml.sim2f.config.TARGET_KINDS`.
        split: ``(N,)`` int64 split code (see :data:`ml.sim2f.schema.SPLIT_NAMES`).
        sim_cfg: The bench geometry that produced the labels.
        teacher_cfg: The teacher's objective settings (``w_ee`` / ``metric``
            must be the ones the labels were made with -- see
            :func:`training_objective`).
        meta: The raw ``meta.json`` payload (provenance only).
        root: The cache directory.
    """

    target: Tensor
    support: Tensor
    phase: Tensor
    spot: Tensor
    teacher_metrics: Tensor
    kind: Tensor
    split: Tensor
    sim_cfg: Sim2fConfig
    teacher_cfg: TeacherConfig
    meta: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=Path)

    @property
    def n_samples(self) -> int:
        """Total number of cached pairs."""
        return int(self.target.shape[0])

    def indices(self, split: str) -> Tensor:
        """Indices belonging to one split.

        Args:
            split: ``"train"``, ``"val"``, ``"test"`` or ``"all"``.

        Returns:
            A 1-D int64 tensor of sample indices, in cache order.

        Raises:
            ValueError: If ``split`` is not one of the names above.
        """
        name = str(split).lower()
        if name == "all":
            return torch.arange(self.n_samples, dtype=torch.long)
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return torch.nonzero(self.split == SPLIT_NAMES.index(name), as_tuple=False)[
            :, 0
        ]

    def kinds(self) -> tuple[str, ...]:
        """Per-sample target-kind names, in cache order.

        Returns:
            A tuple of :data:`ml.sim2f.config.TARGET_KINDS` names.
        """
        from ml.sim2f.config import TARGET_KINDS

        return tuple(TARGET_KINDS[int(value)] for value in self.kind.tolist())


def _section(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    """Return the first mapping stored under any of ``names``.

    The cache metadata is written by :mod:`ml.sim2f.synth`, which is a parallel
    change; this reads the sections by the obvious names and falls back to the
    whole payload, so a metadata key rename degrades to "the sim fields are at
    the top level" rather than to a crash.

    Args:
        payload: The parsed ``meta.json``.
        names: Candidate section keys, in preference order.

    Returns:
        The section, or an empty dict when none of ``names`` is a mapping.
    """
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _filtered(cls: type, section: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys that are fields of the dataclass ``cls``.

    ``to_dict()`` round-trips settings *plus* derived quantities (``fft_length``,
    ``psf_scale_m``, ...), so a naive ``cls(**section)`` would reject a metadata
    file this package itself wrote.

    Args:
        cls: A frozen dataclass from :mod:`ml.sim2f.config`.
        section: Candidate keyword arguments.

    Returns:
        The subset of ``section`` that ``cls`` accepts.
    """
    known = {item.name for item in fields(cls)}
    return {key: value for key, value in section.items() if key in known}


def _errors_from_meta(section: dict[str, Any]) -> OpticsErrors | None:
    """Rebuild the twin-error block from its JSON dump.

    ``OpticsErrors.as_dict`` writes the two Zernike mode families as lists of
    ``[noll, radians]`` pairs, while the dataclass requires tuples (it is frozen
    and hashable).  A naive ``OpticsErrors(**section)`` therefore rejects a
    metadata file this package itself wrote, so the tuple-valued fields are
    re-tupled here.

    Args:
        section: The ``errors`` mapping from ``meta.json``, or ``{}``.

    Returns:
        The reconstructed block, or ``None`` when the metadata has none (the
        caller then keeps the dataclass default, which is the identity).
    """
    if not section:
        return None
    known = {item.name: item for item in fields(OpticsErrors)}
    kwargs: dict[str, Any] = {}
    for name, value in section.items():
        if name not in known:
            continue
        default = known[name].default
        if isinstance(default, tuple):
            kwargs[name] = tuple(
                tuple(item) if isinstance(item, (list, tuple)) else item
                for item in (value or ())
            )
        elif isinstance(default, bool):
            kwargs[name] = bool(value)
        else:
            kwargs[name] = value
    try:
        return OpticsErrors(**kwargs)
    except (TypeError, ValueError) as exc:
        logger.warning("meta.json optics errors unusable ({}); using the identity", exc)
        return None


def _sim_cfg_from_meta(payload: dict[str, Any], *, grid: int, ccd_px: int) -> Sim2fConfig:
    """Rebuild the bench geometry, taking the grid sizes from the arrays.

    Args:
        payload: Parsed ``meta.json`` (may be empty).
        grid: Pupil grid side, read from the cached ``phase`` array.
        ccd_px: Camera grid side, read from the cached ``target`` array.

    Returns:
        The bench geometry.  A metadata file that is missing or silent about the
        optics yields the dataclass defaults with the two grid sizes pinned to
        what the archive actually holds.

    Raises:
        ValueError: If the resulting configuration is invalid (propagated from
            :class:`~ml.sim2f.config.Sim2fConfig`).
    """
    section = _filtered(Sim2fConfig, _section(payload, "sim", "sim_config", "optics"))
    if not section:
        section = _filtered(Sim2fConfig, payload)
    # The arrays are the ground truth about their own shape: a metadata file
    # cannot make a `grid`-wide phase map be something else.
    section["grid"] = int(grid)
    section["ccd_px"] = int(ccd_px)
    errors = _errors_from_meta(section.get("errors") or {})
    if errors is not None:
        section["errors"] = errors
    else:
        section.pop("errors", None)
    return Sim2fConfig(**section)


def _teacher_from_meta(payload: dict[str, Any]) -> TeacherConfig:
    """Rebuild the teacher objective settings from the metadata.

    Args:
        payload: Parsed ``meta.json`` (may be empty).

    Returns:
        The teacher settings the labels were produced with, or the dataclass
        defaults when the metadata is silent.
    """
    section = _filtered(TeacherConfig, _section(payload, "teacher", "teacher_config"))
    if not section:
        return TeacherConfig()
    return TeacherConfig(**section)


def load_corpus(cache_dir: str | Path) -> Corpus:
    """Read the supervised archive and validate it against the schema.

    Args:
        cache_dir: Directory holding ``pairs.npz`` (and optionally
            ``meta.json``).

    Returns:
        The in-memory corpus.

    Raises:
        FileNotFoundError: If ``pairs.npz`` is missing.
        ValueError: If the archive fails :func:`ml.sim2f.schema.validate_cache`
            or its metadata describes an invalid bench.
    """
    root = Path(cache_dir)
    path = root / CACHE_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- build the corpus first with "
            "`python -m ml.sim2f synth --out <dir>`"
        )
    meta_path = root / META_FILE
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read {} ({}); using defaults", meta_path, exc)
            meta = {}

    with np.load(path) as archive:
        grid = int(archive["phase"].shape[1])
        ccd_px = int(archive["target"].shape[2])
        validate_cache(archive, grid=grid, ccd_px=ccd_px)
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    sim_cfg = _sim_cfg_from_meta(meta, grid=grid, ccd_px=ccd_px)
    teacher_cfg = _teacher_from_meta(meta)
    corpus = Corpus(
        target=torch.from_numpy(arrays["target"].astype(np.float32, copy=False)).clone(),
        support=torch.from_numpy(arrays["support"].astype(np.float32, copy=False)).clone(),
        phase=torch.from_numpy(arrays["phase"].astype(np.float32, copy=False)).clone(),
        spot=torch.from_numpy(arrays["spot"].astype(np.float32, copy=False)).clone(),
        teacher_metrics=torch.from_numpy(
            arrays["teacher_metrics"].astype(np.float32, copy=False)
        ).clone(),
        kind=torch.from_numpy(arrays["kind"].astype(np.int64, copy=False)).clone(),
        split=torch.from_numpy(arrays["split"].astype(np.int64, copy=False)).clone(),
        sim_cfg=sim_cfg,
        teacher_cfg=teacher_cfg,
        meta=meta,
        root=root,
    )
    counts = {
        name: int(corpus.indices(name).numel()) for name in SPLIT_NAMES
    }
    logger.info(
        "corpus {}: {} pairs ({} train / {} val / {} test), grid={} ccd_px={} "
        "padding={} kinds={}",
        root,
        corpus.n_samples,
        counts["train"],
        counts["val"],
        counts["test"],
        sim_cfg.grid,
        sim_cfg.ccd_px,
        sim_cfg.far_field_padding,
        len(set(corpus.kind.tolist())),
    )
    return corpus


def context_from_corpus(corpus: Corpus) -> Tensor:
    """The static PSF context channel, averaged over the cached flat frames.

    Every sample stores the same flat-phase frame of the *same* bench (the
    corpus' own ``Sim2fConfig``, detector off), which is why one frame describes
    the whole corpus: it carries the chain's diffraction scale and its 0-order
    pedestal.  The rows are averaged rather than picked one-by-one so that any
    per-sample noise the generator left behind is averaged out instead of
    becoming part of the network's input.

    Args:
        corpus: The loaded corpus.

    Returns:
        ``(1, 1, ccd_px, ccd_px)`` peak-normalised frame.

    Note:
        Chunked accumulation rather than ``spot.mean(0)``: a 600-sample archive
        at 512x512 is 629 MB, and the mean is only used at 1/600 of that size.
    """
    total: Tensor | None = None
    chunk = 32
    for start in range(0, corpus.spot.shape[0], chunk):
        piece = corpus.spot[start : start + chunk].sum(dim=0, keepdim=True)
        total = piece if total is None else total + piece
    assert total is not None  # validate_cache guarantees at least one sample
    frame = total / float(corpus.spot.shape[0])
    peak = frame.amax()
    if float(peak) <= 1e-8:
        logger.warning(
            "cached `spot` frames are empty; the PSF-context channel carries no "
            "information (was the detector model switched off before caching?)"
        )
        return frame.detach()
    return (frame / peak).detach()


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
def training_objective(
    phasor: Tensor,
    gt_phase: Tensor,
    target: Tensor,
    support: Tensor,
    *,
    sim: torch.nn.Module,
    cfg: TrainConfig,
    teacher: TeacherConfig,
    mask: Tensor | None = None,
    prepare: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    """Compose the training objective from its two legs, in one place.

    Args:
        phasor: ``(B, 2, grid, grid)`` network output (unit phasor on the pupil).
        gt_phase: ``(B, grid, grid)`` teacher phase in raw radians.
        target: ``(B, 1, ccd_px, ccd_px)`` requested distribution.
        support: ``(B, 1, ccd_px, ccd_px)`` analytic support.
        sim: The differentiable bench (the calibrated twin) for the physics leg.
        cfg: Supplies ``w_sup`` / ``w_phy`` / ``w_tv``.
        teacher: Supplies ``w_ee`` and ``metric`` -- the same values the labels
            were optimised with, so the terms are like-for-like.
        mask: Optional pupil mask for the supervised leg (the teacher's phase
            outside the pupil is arbitrary because the field there is dark).
        prepare: Background removal before any ratio in the physics leg.  On by
            default; the twin's zero-order background and dark offset mean a raw
            ``I.sum()`` is not the modulated energy.

    Returns:
        ``(total, terms)`` where ``terms`` holds ``total``, ``sup``, ``phy`` and
        ``tv`` as floats.  A leg whose weight is zero is skipped, not computed
        and multiplied by zero: the physics leg costs one propagation per step.
    """
    device = phasor.device
    zero = torch.zeros((), dtype=phasor.dtype, device=device)

    if float(cfg.w_sup) > 0.0:
        reference = to_phasor(gt_phase.to(device=device, dtype=phasor.dtype))
        sup = piston_free_phasor_l1(phasor, reference, mask)
    else:
        sup = zero

    if float(cfg.w_phy) > 0.0:
        phy, _terms = physics_loss(
            sim,
            phasor,
            target.to(device=device, dtype=phasor.dtype),
            support.to(device=device, dtype=phasor.dtype),
            w_ee=teacher.w_ee,
            metric=teacher.metric,
            prepare=prepare,
        )
    else:
        phy = zero

    tv = tv_phasor(phasor) if float(cfg.w_tv) > 0.0 else zero

    total = float(cfg.w_sup) * sup + float(cfg.w_phy) * phy + float(cfg.w_tv) * tv
    return total, {
        "total": float(total.detach()),
        "sup": float(sup.detach()),
        "phy": float(phy.detach()),
        "tv": float(tv.detach()),
    }


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed every RNG the training loop can reach.

    Args:
        seed: Seed value.  Negative values are mapped into ``uint32`` because
            :mod:`numpy` rejects them and the failure would surface deep inside
            the run.
    """
    value = int(seed)
    if value < 0:
        value += 1 << 32
    random.seed(value)
    np.random.seed(value % (1 << 32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    logger.debug("seeded python/numpy/torch with {}", value)


def _apply_determinism(enabled: bool) -> None:
    """Set the cuDNN / autograd determinism switches.

    Args:
        enabled: Whether determinism was requested.
    """
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = not bool(enabled)
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(
            "determinism enabled (cudnn.deterministic=True, benchmark=False, "
            "use_deterministic_algorithms=True, warn_only)"
        )


def _build_sim(sim_cfg: Sim2fConfig, device: torch.device) -> torch.nn.Module:
    """Construct the differentiable 2f bench.

    Imported lazily so this module can be imported (and its parser tested)
    without the optics module or a CUDA device.  Tests monkeypatch this name to
    substitute a hand-written forward model.

    Args:
        sim_cfg: Bench geometry.
        device: Device the bench should compute on.

    Returns:
        The bench, ready for ``forward(phase)``.
    """
    from ml.sim2f.optics import Sim2f

    return Sim2f(sim_cfg, device=str(device))


def _batch_stream(
    indices: Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[Tensor]:
    """Yield shuffled index batches for ever, reshuffling every pass.

    Args:
        indices: Sample indices of the split.
        batch_size: Maximum batch size.
        generator: Seeded generator, so the order is reproducible and does not
            consume the global RNG stream the model initialisation used.

    Yields:
        1-D index tensors.
    """
    size = int(indices.numel())
    if size == 0:
        raise ValueError("cannot stream batches from an empty split")
    while True:
        order = indices[torch.randperm(size, generator=generator)]
        for start in range(0, size, batch_size):
            yield order[start : start + batch_size]


def _forward_batches(
    model: Sim2fUNet,
    corpus: Corpus,
    indices: Tensor,
    *,
    context: Tensor,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Yield ``(phasor, target, support, gt_phase)`` for each batch of a split.

    Args:
        model: The network (used for device/dtype and the shared context).
        corpus: The corpus.
        indices: Sample indices to visit, in order.
        context: ``(1, 1, ccd_px, ccd_px)`` PSF context.
        batch_size: Batch size.
        device: Device to move the batch to.

    Yields:
        Tuples of tensors, all on ``device``; ``phasor`` is the network output.
    """
    for start in range(0, int(indices.numel()), batch_size):
        batch = indices[start : start + batch_size]
        target = corpus.target[batch].to(device=device, dtype=torch.float32)
        support = corpus.support[batch].to(device=device, dtype=torch.float32)
        gt_phase = corpus.phase[batch].to(device=device, dtype=torch.float32)
        phasor = model(network_input(target, context))
        yield phasor, target, support, gt_phase


def _evaluate_split(
    model: Sim2fUNet,
    corpus: Corpus,
    indices: Tensor,
    *,
    context: Tensor,
    sim: torch.nn.Module,
    cfg: TrainConfig,
    teacher: TeacherConfig,
    mask: Tensor | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Score one split without touching the optimiser or the RNG.

    Args:
        model: The network.
        corpus: The corpus.
        indices: Sample indices of the split (empty returns NaNs).
        context: PSF context.
        sim: The bench.
        cfg: Loss weights.
        teacher: Teacher objective settings.
        mask: Pupil mask for the supervised leg.
        batch_size: Batch size.
        device: Device.

    Returns:
        Mean ``total`` / ``sup`` / ``phy`` / ``tv`` over the split.  All NaN when
        the split is empty, so "not measured" is visibly distinct from "zero".
    """
    if int(indices.numel()) == 0:
        return {"total": float("nan"), "sup": float("nan"),
                "phy": float("nan"), "tv": float("nan")}
    was_training = model.training
    model.eval()
    sums = {"total": 0.0, "sup": 0.0, "phy": 0.0, "tv": 0.0}
    seen = 0
    try:
        with torch.no_grad():
            for phasor, target, support, gt_phase in _forward_batches(
                model, corpus, indices, context=context, batch_size=batch_size,
                device=device,
            ):
                _total, terms = training_objective(
                    phasor, gt_phase, target, support, sim=sim, cfg=cfg,
                    teacher=teacher, mask=mask,
                )
                batch = int(phasor.shape[0])
                for key in sums:
                    sums[key] += terms[key] * batch
                seen += batch
    finally:
        model.train(was_training)
    return {key: value / seen for key, value in sums.items()}


def _save_checkpoint(
    path: Path,
    model: Sim2fUNet,
    *,
    step: int,
    cfg: TrainConfig,
    sim_cfg: Sim2fConfig,
    context: Tensor,
    metrics: dict[str, Any],
) -> None:
    """Write one checkpoint.

    The checkpoint carries the network weights, both configs (as objects for
    this package and as dicts for tools that should not unpickle dataclasses),
    the PSF context and the metrics that selected it -- an evaluation is not
    reproducible from weights alone.

    Args:
        path: Destination file.
        model: The network.
        step: Optimiser step the weights were taken at.
        cfg: Training configuration.
        sim_cfg: Bench geometry.
        context: PSF context tensor.
        metrics: Selection metrics for this checkpoint.
    """
    torch.save(
        {
            "format": 1,
            "state_dict": model.state_dict(),
            "train_config": cfg,
            "sim_config": sim_cfg,
            "train_config_dict": cfg.to_dict(),
            "sim_config_dict": sim_cfg.to_dict(),
            "context": context.detach().cpu(),
            "step": int(step),
            "metrics": metrics,
        },
        path,
    )


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    """Rewrite ``history.csv`` from scratch, deterministically.

    A column the row does not carry is written as an **empty cell**, not as
    ``nan``: a run with no validation split has no validation number, and an
    empty field is the CSV convention for "not measured" while ``nan`` reads as
    "measured and broken".

    Args:
        path: Destination file.
        rows: One dict per validation event, keyed by :data:`HISTORY_COLUMNS`.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HISTORY_COLUMNS)
        for row in rows:
            cells = [
                "" if row.get(column) is None else f"{row[column]:.10g}"
                for column in HISTORY_COLUMNS
            ]
            writer.writerow(cells)


def _checkpoint_metrics(row: dict[str, float]) -> dict[str, float | None]:
    """Project a history row onto the values a checkpoint records.

    A run without a validation split carries no ``val_*`` entry at all, and
    ``None`` records that honestly -- a checkpoint metric of ``nan`` would be
    indistinguishable from a diverged run.

    Args:
        row: One history row (possibly train-only).

    Returns:
        The four selection metrics, ``None`` where the row has none.
    """
    return {
        key: row.get(key)
        for key in ("val_phy", "val_sup", "val_total", "train_total")
    }


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the JSON stays standard.

    ``json.dumps`` writes ``NaN`` / ``Infinity`` by default, which is not JSON
    and which strict readers reject.  A split that was never validated (e.g.
    ``steps < val_every``) legitimately has no number, and ``null`` says that.

    Args:
        value: Any JSON-shaped object (dict / list / float / scalar).

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(
    cfg: TrainConfig,
    *,
    cache_dir: Path | str,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the training loop described by ``cfg``.

    Args:
        cfg: A validated configuration.  ``cfg.out_dir`` receives the artefacts.
        cache_dir: Directory holding ``pairs.npz``.
        device: Concrete device, or ``None`` to resolve ``cfg.device``.

    Returns:
        A summary with ``best_step``, ``best_val_phy`` (``None`` when the corpus
        has no validation split), ``val_enabled``, ``final``, ``history``,
        ``n_parameters`` and the written paths.

    Raises:
        FileNotFoundError: If the corpus is missing.
        ValueError: If the corpus has no training samples, or the architecture
            cannot be built for this bench (e.g. ``ccd_px % 2 ** depth``).
    """
    corpus = load_corpus(cache_dir)
    sim_cfg = corpus.sim_cfg
    resolved = _resolve_device(cfg.device) if device is None else device
    _apply_determinism(cfg.cudnn_deterministic)
    seed_everything(cfg.seed)

    train_indices = corpus.indices("train")
    val_indices = corpus.indices("val")
    if int(train_indices.numel()) == 0:
        raise ValueError(
            f"the train split of {Path(cache_dir) / CACHE_FILE} is empty; "
            "re-synthesise with a non-zero SynthConfig.train_frac"
        )
    # Decided once, up front. An empty val split is a legal corpus (a smoke run of
    # four pairs is all train), and letting it reach the loop would write a row of
    # NaNs every `val_every` steps, leave `best_val_phy = inf` so no checkpoint
    # ever qualifies as "best", and make `evaluate` fail on a missing
    # `model_best.pt`. None of that is a measurement, so validation is switched
    # off explicitly and the final state is the selected one.
    val_enabled = int(val_indices.numel()) > 0
    if not val_enabled:
        logger.info(
            "no val split (0 samples): validation disabled, selecting on the last step"
        )

    context = context_from_corpus(corpus).to(device=resolved, dtype=torch.float32)
    model = build_unet(cfg, sim_cfg, device=resolved, context=context)
    mask = model.pupil_mask
    sim = _build_sim(sim_cfg, resolved)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    n_parameters = model.n_parameters()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "sim2f train: device={} params={} steps={} batch={} lr={} seed={} "
        "w_sup={} w_phy={} w_tv={} out={}",
        resolved, n_parameters, cfg.steps, cfg.batch_size, cfg.lr, cfg.seed,
        cfg.w_sup, cfg.w_phy, cfg.w_tv, out_dir,
    )

    generator = torch.Generator().manual_seed(int(cfg.seed))
    stream = _batch_stream(train_indices, cfg.batch_size, generator)
    autocast_enabled = bool(cfg.amp) and resolved.type == "cuda"
    if cfg.amp and not autocast_enabled:
        logger.info("--amp ignored on {}: autocast needs a CUDA device", resolved)

    history: list[dict[str, float]] = []
    running = {"total": 0.0, "sup": 0.0, "phy": 0.0, "tv": 0.0}
    running_count = 0
    best_val_phy: float | None = float("inf")
    best_step = -1
    final: dict[str, float] = {}

    for step in range(1, int(cfg.steps) + 1):
        batch = next(stream)
        target = corpus.target[batch].to(device=resolved, dtype=torch.float32)
        support = corpus.support[batch].to(device=resolved, dtype=torch.float32)
        gt_phase = corpus.phase[batch].to(device=resolved, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        context_manager = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        with context_manager:
            phasor = model(network_input(target, context))
            loss, terms = training_objective(
                phasor, gt_phase, target, support, sim=sim, cfg=cfg,
                teacher=corpus.teacher_cfg, mask=mask,
            )
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(
                f"non-finite training objective ({float(loss.detach())}) at step {step}"
            )
        loss.backward()
        optimizer.step()

        for key in running:
            running[key] += terms[key]
        running_count += 1

        last = step == int(cfg.steps)
        if step % int(cfg.val_every) and not last:
            continue

        averages = {key: value / max(running_count, 1) for key, value in running.items()}
        row = {
            "step": float(step),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_total": averages["total"],
            "train_sup": averages["sup"],
            "train_phy": averages["phy"],
            "train_tv": averages["tv"],
        }
        improved = False
        if val_enabled:
            val = _evaluate_split(
                model, corpus, val_indices, context=context, sim=sim, cfg=cfg,
                teacher=corpus.teacher_cfg, mask=mask, batch_size=cfg.batch_size,
                device=resolved,
            )
            row.update(
                {
                    "val_total": val["total"],
                    "val_sup": val["sup"],
                    "val_phy": val["phy"],
                    "val_tv": val["tv"],
                }
            )
            improved = math.isfinite(val["phy"]) and val["phy"] < best_val_phy
            if improved:
                best_val_phy = val["phy"]
                best_step = step
        history.append(row)
        running = {"total": 0.0, "sup": 0.0, "phy": 0.0, "tv": 0.0}
        running_count = 0
        final = row

        if val_enabled:
            logger.info(
                "step {}/{} train_total={:.5f} train_sup={:.5f} train_phy={:.5f} "
                "val_total={:.5f} val_sup={:.5f} val_phy={:.5f}{}",
                step, cfg.steps, averages["total"], averages["sup"], averages["phy"],
                val["total"], val["sup"], val["phy"], "  <- best" if improved else "",
            )
        else:
            logger.info(
                "step {}/{} train_total={:.5f} train_sup={:.5f} train_phy={:.5f}{}",
                step, cfg.steps, averages["total"], averages["sup"], averages["phy"],
                "  <- final" if last else "",
            )

        _write_history(out_dir / HISTORY_NAME, history)
        _save_checkpoint(
            out_dir / (CHECKPOINT_BEST if improved else CHECKPOINT_LAST),
            model, step=step, cfg=cfg, sim_cfg=sim_cfg, context=context,
            metrics=_checkpoint_metrics(row),
        )

    # The last checkpoint is always written, even when the final step was not a
    # validation step (steps < val_every), so `evaluate` always has weights.
    _write_history(out_dir / HISTORY_NAME, history)
    final_metrics = _checkpoint_metrics(final)
    _save_checkpoint(
        out_dir / CHECKPOINT_LAST, model, step=int(cfg.steps), cfg=cfg,
        sim_cfg=sim_cfg, context=context, metrics=final_metrics,
    )
    if not val_enabled:
        # Nothing ever improved on anything, so "best" is defined as the final
        # state. It still has to exist under the conventional name: `evaluate`
        # and every downstream consumer load `model_best.pt`.
        best_step = int(cfg.steps)
        # `inf` is not a selection metric, it is the absence of one: report the
        # absence (`None`, `null` in JSON) rather than a number no run produced.
        best_val_phy = None
        _save_checkpoint(
            out_dir / CHECKPOINT_BEST, model, step=int(cfg.steps), cfg=cfg,
            sim_cfg=sim_cfg, context=context, metrics=final_metrics,
        )

    counts = {name: int(corpus.indices(name).numel()) for name in SPLIT_NAMES}
    config_payload = {
        "train": cfg.to_dict(),
        "sim": sim_cfg.to_dict(),
        "teacher": corpus.teacher_cfg.to_dict(),
        "synth": _section(corpus.meta, "synth", "synth_config") or None,
        "cache": str(Path(cache_dir)),
        "counts": counts,
        "n_parameters": n_parameters,
        "device": str(resolved),
        "history_columns": list(HISTORY_COLUMNS),
    }
    (out_dir / CONFIG_NAME).write_text(
        json.dumps(_json_safe(config_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_payload = {
        "steps": int(cfg.steps),
        "n_parameters": n_parameters,
        "counts": counts,
        "val_enabled": bool(val_enabled),
        "best_step": int(best_step),
        "best_val_phy": best_val_phy,
        "final": final,
        "history": history,
    }
    (out_dir / METRICS_NAME).write_text(
        json.dumps(_json_safe(metrics_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if val_enabled:
        logger.info(
            "done: {} step(s), best val_phy={:.6f} at step {} -> {}",
            cfg.steps, best_val_phy, best_step, out_dir,
        )
    else:
        logger.info(
            "done: {} step(s), no validation: selected the final state at step {} -> {}",
            cfg.steps, best_step, out_dir,
        )
    return {
        "best_step": int(best_step),
        "best_val_phy": best_val_phy,
        "val_enabled": bool(val_enabled),
        "final": final,
        "history": history,
        "n_parameters": n_parameters,
        "counts": counts,
        "paths": {
            "config": str(out_dir / CONFIG_NAME),
            "metrics": str(out_dir / METRICS_NAME),
            "history": str(out_dir / HISTORY_NAME),
            "best": str(out_dir / CHECKPOINT_BEST),
            "last": str(out_dir / CHECKPOINT_LAST),
        },
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, every default read from :class:`TrainConfig`.

    Returns:
        The configured parser.
    """
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m ml.sim2f train",
        description=(
            "Train the sim2f U-Net: requested camera spot -> aberration-free pupil "
            "phase."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="corpus directory containing pairs.npz (written by `ml.sim2f synth`)",
    )
    parser.add_argument("--steps", type=int, default=cfg.steps)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--width", type=int, default=cfg.width)
    parser.add_argument("--depth", type=int, default=cfg.depth)
    parser.add_argument("--w-sup", type=float, default=cfg.w_sup)
    parser.add_argument("--w-phy", type=float, default=cfg.w_phy)
    parser.add_argument("--w-tv", type=float, default=cfg.w_tv)
    parser.add_argument("--val-every", type=int, default=cfg.val_every)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default=cfg.device
    )
    parser.add_argument("--out-dir", default=cfg.out_dir)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.amp)
    parser.add_argument(
        "--cudnn-deterministic",
        action=argparse.BooleanOptionalAction,
        default=cfg.cudnn_deterministic,
        help="bit-identical reruns for the same seed (~4%% wall clock)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Rebuild a :class:`TrainConfig` from parsed arguments.

    Args:
        args: The parsed namespace.

    Returns:
        A validated configuration (validation lives in the dataclass, so an
        invalid combination fails here rather than mid-run).
    """
    return TrainConfig(
        width=args.width,
        depth=args.depth,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_sup=args.w_sup,
        w_phy=args.w_phy,
        w_tv=args.w_tv,
        val_every=args.val_every,
        seed=args.seed,
        device=args.device,
        cudnn_deterministic=args.cudnn_deterministic,
        out_dir=args.out_dir,
        amp=args.amp,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` on a rejected configuration.
    """
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        logger.error("invalid configuration: {}", exc)
        return 2
    try:
        train(cfg, cache_dir=Path(args.cache))
    except (FileNotFoundError, ValueError, FloatingPointError) as exc:
        logger.error("training failed: {}", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
