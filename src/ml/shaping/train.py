"""Training entry point for the inverse beam-shaping model.

Run it as ``python -m ml.shaping.train``. The objective is the two-term one
documented in :mod:`ml.shaping.losses`; this module only owns the loop, the
checkpointing and the logging.

============================  Why selection uses validation ``sup``  ============================

The checkpoint is selected on the mean validation **supervised** L1, not on the
physics term. ``sup`` is defined for *every* record, while ``phy`` is undefined
for the records whose ``fov_px`` has no calibrated forward model -- roughly a
fifth of the corpus has an unknown ``fov_px``, and ``fov_px == 1944``
(``slm_gsnet_square``) has no usable padding at any value because it stores
freeform phase cells rather than Zernike coefficients. Selecting on ``phy``
would therefore silently restrict model selection to whichever families happen to
be geometrically comparable, which is a selection bias, not a preference.

============================  Why the parser reads the dataclass  ============================

Every default in :func:`_build_parser` is taken from :class:`ShapingConfig`
rather than written as a literal. This repository shipped exactly that bug: the
``train_amp`` parser hard-wrote ``default=1`` for ``far_field_padding`` and
silently overrode the calibrated value of 10 on every single run
(see ``report/zernike_phase2amp/report.md``). Constructing the frozen dataclass
once and reading from it makes "the CLI default" and "the code default" the same
object, so they cannot drift. ``tests/ao_shaping/ml/shaping/test_train.py``
parametrises over every field to keep it that way.

============================  Determinism  ============================

``config.cudnn_deterministic`` defaults to ``True``. The repo's existing trainers
were measured **non-reproducible** without it -- the same arm, same seed, same
code gave val R2 0.8510 on one run and 0.8573 on the next, with fitted
coefficients differing by up to 0.065 -- because cuDNN convolution backward uses
nondeterministic atomics. Turning the flags on costs about 4% wall clock and was
verified to give bit-identical reruns.

That fixes *compute* nondeterminism only. Closed-loop measurements on real
hardware still drift, so a reproducible run here means "the same numbers given
the same inputs", never "the bench produced the same light twice".
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from ml.hwdataset.records import MaterialiserConfig
from ml.shaping.config import IMAGE_MODES, ShapingConfig, parse_csv_tuple
from ml.shaping.dataset import build_shaping_loaders, corpus_summary, network_input
from ml.shaping.viz import build_comparison, render_comparison
from ml.shaping.losses import ShapingLoss
from ml.shaping.models import build_phasor_unet
from ml.shaping.physics import ForwardBank

__all__ = [
    "TrainResult",
    "main",
    "seed_everything",
    "train",
]

#: Name of the checkpoint written by :func:`train`. A fixed name (rather than a
#: timestamp) keeps ``--checkpoint-dir`` self-contained: the newest run in a
#: directory is always ``last.pt``, and the history sits beside it.
CHECKPOINT_NAME = "last.pt"
HISTORY_NAME = "history.json"


@dataclass(frozen=True)
class TrainResult:
    """What one :func:`train` call produced.

    Attributes:
        epochs_run: Epochs actually executed (may be fewer than
            ``config.epochs`` when ``config.max_steps`` capped the run).
        global_step: Optimiser steps taken.
        best_val_sup: Lowest mean validation supervised L1 seen, i.e. the
            selection metric.
        best_epoch: Epoch that produced :attr:`best_val_sup`.
        checkpoint: Path of the written checkpoint.
        history: Per-epoch rows, one dict per epoch.
        final: The last epoch's aggregate metrics, or ``{}`` when the run
            executed no steps.
    """

    epochs_run: int
    global_step: int
    best_val_sup: float
    best_epoch: int
    checkpoint: str
    history: list[dict[str, float]]
    final: dict[str, float]


def seed_everything(seed: int) -> None:
    """Seed every RNG the training loop can reach.

    Args:
        seed: Seed value. Any integer; negative values are accepted by torch
            but rejected by :mod:`numpy`, so they are mapped rather than
            allowed to raise deep inside the run.
    """
    value = int(seed)
    if value < 0:
        value += 1 << 32
    random.seed(value)
    np.random.seed(value % (1 << 32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _resolve_device(spec: str) -> torch.device:
    """Turn a :data:`~ml.shaping.config.DeviceSpec` into a concrete device.

    Args:
        spec: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved :class:`torch.device`.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _features(value: str) -> tuple[int, ...]:
    """Parse a comma-separated feature-width list.

    Args:
        value: e.g. ``"16,32,64,128,256"``.

    Returns:
        The widths as a tuple of ints.

    Raises:
        argparse.ArgumentTypeError: If any entry is not a positive integer.
    """
    try:
        widths = tuple(int(part) for part in str(value).split(",") if part.strip())
    except ValueError as exc:  # pragma: no cover - argparse formats the message
        raise argparse.ArgumentTypeError(
            f"--features must be comma-separated integers, got {value!r}"
        ) from exc
    if not widths:
        raise argparse.ArgumentTypeError("--features must not be empty")
    return widths


def _roots(value: str) -> tuple[str, ...]:
    """Parse a comma-separated corpus-root list.

    Args:
        value: e.g. ``"data/debug"`` or ``"a,b"``.

    Returns:
        The roots as a tuple of strings.
    """
    roots = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not roots:
        raise argparse.ArgumentTypeError("--roots must not be empty")
    return roots


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Every default is read from a single :class:`ShapingConfig` instance so the
    parser and the dataclass cannot drift apart -- see the module docstring for
    the bug that motivates this.

    Returns:
        The configured parser.
    """
    cfg = ShapingConfig()
    parser = argparse.ArgumentParser(
        prog="python -m ml.shaping.train",
        description=(
            "Train the inverse beam-shaping U-Net: far-field spot -> SLM phase."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--grid", type=int, default=cfg.grid)
    parser.add_argument("--features", type=_features, default=cfg.features)
    parser.add_argument("--in-channels", type=int, default=cfg.in_channels)

    parser.add_argument("--w-sup", type=float, default=cfg.w_sup)
    parser.add_argument("--w-phy", type=float, default=cfg.w_phy)
    parser.add_argument("--w-smooth", type=float, default=cfg.w_smooth)

    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)

    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=cfg.device)
    parser.add_argument(
        "--cudnn-deterministic",
        action=argparse.BooleanOptionalAction,
        default=cfg.cudnn_deterministic,
    )

    parser.add_argument("--limit-files", type=int, default=cfg.limit_files)
    parser.add_argument("--max-steps", type=int, default=cfg.max_steps)

    parser.add_argument("--checkpoint-dir", default=cfg.checkpoint_dir)
    parser.add_argument("--resume", default=cfg.resume)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=cfg.wandb)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled", "shared"),
        default=cfg.wandb_mode,
        help="offline by default: this machine has no WANDB_API_KEY and no "
        "~/.netrc, so an online run would fail; the offline directory syncs "
        "later with `wandb sync <dir>`",
    )
    parser.add_argument("--wandb-project", default=cfg.wandb_project)
    parser.add_argument("--wandb-name", default=cfg.wandb_name)
    parser.add_argument(
        "--image-every",
        type=int,
        default=cfg.image_every,
        help="render the pred-vs-true montage every N epochs (0 disables the "
        "figure; metrics are still logged)",
    )
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=cfg.viz_samples,
        help="rows in the pred-vs-true montage",
    )
    parser.add_argument("--roots", type=_roots, default=cfg.roots)
    parser.add_argument("--forward-checkpoint", default=cfg.forward_checkpoint)
    parser.add_argument(
        "--index-cache",
        default=cfg.index_cache,
        help="pin the corpus to a JSON index snapshot; strongly recommended, "
        "because data/debug grows between runs and two runs would otherwise "
        "train on different data (see ShapingConfig.index_cache)",
    )

    # -- data preprocessing (report/hwdataset_corpus/report.md) ------------
    parser.add_argument("--image-mode", choices=IMAGE_MODES, default=cfg.image_mode)
    parser.add_argument(
        "--balance-alpha",
        type=float,
        default=cfg.balance_alpha,
        help=(
            "per-family balancing strength in [0,1]; 0 keeps corpus proportions, "
            "1 gives every family equal files per epoch"
        ),
    )
    parser.add_argument(
        "--max-file-repeats",
        type=int,
        default=cfg.max_file_repeats,
        help="cap on one file's repetitions per balanced epoch",
    )
    parser.add_argument("--families", type=parse_csv_tuple, default=cfg.families)
    parser.add_argument("--sources", type=parse_csv_tuple, default=cfg.sources)
    parser.add_argument(
        "--drop-exposure-channel",
        action=argparse.BooleanOptionalAction,
        default=cfg.drop_exposure_channel,
        help=(
            "feed the far-field image alone instead of "
            "[image, exposure_log10]; the ablation that separates learned optics "
            "from family identification, since exposure is stratified by family"
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> ShapingConfig:
    """Rebuild a :class:`ShapingConfig` from parsed arguments.

    Args:
        args: The parsed namespace.

    Returns:
        A validated config. Validation happens in the dataclass, so an invalid
        CLI combination raises :class:`ValueError` here rather than deep inside
        the training loop.
    """
    return ShapingConfig(
        grid=args.grid,
        features=tuple(args.features),
        in_channels=args.in_channels,
        w_sup=args.w_sup,
        w_phy=args.w_phy,
        w_smooth=args.w_smooth,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        cudnn_deterministic=args.cudnn_deterministic,
        limit_files=args.limit_files,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        wandb=args.wandb,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        image_every=args.image_every,
        viz_samples=args.viz_samples,
        roots=tuple(args.roots),
        forward_checkpoint=args.forward_checkpoint,
        index_cache=args.index_cache,
        image_mode=args.image_mode,
        balance_alpha=args.balance_alpha,
        max_file_repeats=args.max_file_repeats,
        families=tuple(args.families) if args.families else None,
        sources=tuple(args.sources) if args.sources else None,
        drop_exposure_channel=args.drop_exposure_channel,
    )


def _apply_determinism(enabled: bool) -> None:
    """Set the cuDNN / autograd determinism switches.

    Args:
        enabled: Whether determinism was requested.
    """
    if not enabled:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(
        "determinism enabled (cudnn.deterministic=True, benchmark=False, "
        "use_deterministic_algorithms=True); costs ~4% wall clock"
    )


def _load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Restore model/optimizer state from a checkpoint.

    Args:
        path: Checkpoint file.
        model: Module to load weights into.
        optimizer: Optimizer to load state into.

    Returns:
        The checkpoint dict, so the caller can recover ``epoch`` / ``global_step``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {target}")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise KeyError(f"resume checkpoint {target} has no 'model_state' entry")
    model.load_state_dict(state)
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    logger.info(
        "resumed from {} (epoch={} global_step={})",
        target,
        payload.get("epoch"),
        payload.get("global_step"),
    )
    return payload


def _epoch_pass(
    model: torch.nn.Module,
    loader: Any,
    criterion: ShapingLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    grid: int,
    in_channels: int,
    step_budget: list[int] | None = None,
) -> dict[str, float]:
    """Run one pass over ``loader``.

    Args:
        model: The network.
        loader: A DataLoader yielding shaping batches.
        criterion: The loss.
        device: Device the model lives on.
        optimizer: Optimiser to step, or ``None`` for an evaluation pass.
        grid: Grid side length, used to assemble the network input.
        in_channels: Network input channels.
        step_budget: Optional single-element list holding the remaining global
            step allowance. When it reaches zero the loop stops early, which is
            how ``config.max_steps`` caps a run mid-epoch.

    Returns:
        Mean of each loss term over the pass, plus ``physics_excluded`` summed
        across batches and the number of optimiser steps taken.
    """
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "sup": 0.0, "phy": 0.0, "smooth": 0.0}
    excluded = 0
    batches = 0
    steps = 0

    for batch in loader:
        if step_budget is not None and step_budget[0] <= 0:
            break
        images = network_input(batch, grid, in_channels).to(device)
        target = batch["fov_px"]
        if torch.is_tensor(target):
            target = target.to(device)
        losses = criterion(model(images), batch)

        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            steps += 1
            if step_budget is not None:
                step_budget[0] -= 1

        totals["total"] += float(losses["total"].detach())
        totals["sup"] += float(losses["sup"].detach())
        totals["phy"] += float(losses["phy"].detach())
        totals["smooth"] += float(losses["smooth"].detach())
        excluded += int(losses["physics_excluded"])
        batches += 1

    if batches == 0:
        return {"total": float("nan"), "sup": float("nan"), "phy": float("nan"),
                "smooth": float("nan"), "physics_excluded": 0, "steps": 0}
    return {
        "total": totals["total"] / batches,
        "sup": totals["sup"] / batches,
        "phy": totals["phy"] / batches,
        "smooth": totals["smooth"] / batches,
        "physics_excluded": excluded,
        "steps": steps,
    }


def train(cfg: ShapingConfig) -> TrainResult:
    """Run the training loop described by ``cfg``.

    Args:
        cfg: A validated configuration.

    Returns:
        A :class:`TrainResult`.

    Raises:
        FileNotFoundError: If ``cfg.resume`` points at a missing file.
        ValueError: Propagated from :class:`ShapingConfig` validation.
    """
    _apply_determinism(cfg.cudnn_deterministic)
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    logger.info(
        "training on {} | grid={} features={} w_sup={} w_phy={} w_smooth={} "
        "lr={} batch={} epochs={} seed={} max_steps={} roots={}",
        device,
        cfg.grid,
        list(cfg.features),
        cfg.w_sup,
        cfg.w_phy,
        cfg.w_smooth,
        cfg.lr,
        cfg.batch_size,
        cfg.epochs,
        cfg.seed,
        cfg.max_steps,
        list(cfg.roots),
    )

    train_loader, val_loader, test_loader = build_shaping_loaders(
        cfg.roots,
        config=MaterialiserConfig(grid=cfg.grid, image_mode=cfg.image_mode),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        limit_files=cfg.limit_files,
        index_cache=cfg.index_cache,
        use_cache=True,
        families=cfg.families,
        sources=cfg.sources,
        balance_alpha=cfg.balance_alpha,
        max_file_repeats=cfg.max_file_repeats,
    )

    # `drop_exposure_channel` changes the network's input arity, so the config
    # itself is rewritten rather than passing `in_channels` alongside it --
    # `build_phasor_unet` lets `config` win, and handing it both would silently
    # drop `config.features` and rebuild the default-width U-Net instead.
    model_config = (
        replace(cfg, in_channels=1) if cfg.drop_exposure_channel else cfg
    )
    model = build_phasor_unet(device=str(device), config=model_config)
    bank = ForwardBank(
        grid=cfg.grid,
        device=str(device),
        forward_checkpoint=cfg.forward_checkpoint,
    )
    criterion = ShapingLoss(cfg, bank)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    global_step = 0
    best_val_sup = float("inf")
    best_epoch = -1
    if cfg.resume is not None:
        payload = _load_checkpoint(cfg.resume, model, optimizer)
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        best_val_sup = float(payload.get("best_val_sup", float("inf")))
        best_epoch = int(payload.get("best_epoch", -1))

    # Which corpus this run actually saw. A stale index is not hypothetical --
    # see ml.shaping.dataset.corpus_summary -- so it is written next to the
    # weights rather than left implicit. `index_cache` names the snapshot the
    # numbers are attributable to; `None` means "whatever was on disk", which is
    # exactly the condition a reader needs to be warned about.
    summary = {
        "index_cache": cfg.index_cache,
        "train": corpus_summary(train_loader),
        "val": corpus_summary(val_loader),
        "test": corpus_summary(test_loader),
    }
    logger.info(
        "corpus: {} train / {} val records over {} / {} / {} files; "
        "{} families, dominant share {:.1%}; index_cache={}",
        summary["train"]["n_records"],
        summary["val"]["n_records"],
        summary["train"]["n_files"],
        summary["val"]["n_files"],
        summary["test"]["n_files"],
        summary["train"]["n_families"],
        summary["train"]["dominant_family_share"],
        cfg.index_cache or "<none: rescanned, not pinned>",
    )

    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    history_path = out_dir / HISTORY_NAME

    run = _init_wandb(cfg, summary)

    # One fixed validation batch for the whole run. The val loader uses a
    # sequential sampler, so ``next(iter(...))`` always yields the same samples:
    # the montage means the same thing at every epoch and across reruns, which a
    # randomly-drawn batch would not.
    viz_batch: dict[str, Any] | None = None
    if cfg.image_every:
        try:
            viz_batch = next(iter(val_loader))
        except StopIteration:
            logger.warning("validation loader is empty; pred-vs-true figures disabled")

    budget = [cfg.epochs * max(1, len(train_loader))] if cfg.max_steps is None else [
        int(cfg.max_steps)
    ]

    history: list[dict[str, float]] = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read {} ({}); starting a fresh history",
                           history_path, exc)
            history = []

    for epoch in range(start_epoch, cfg.epochs):
        started = time.time()
        train_stats = _epoch_pass(
            model, train_loader, criterion, device,
            optimizer=optimizer, grid=cfg.grid, in_channels=model_config.in_channels,
            step_budget=budget,
        )
        global_step += int(train_stats["steps"])
        val_stats = _epoch_pass(
            model, val_loader, criterion, device,
            optimizer=None, grid=cfg.grid, in_channels=model_config.in_channels,
        )

        val_sup = float(val_stats["sup"])
        improved = val_sup == val_sup and val_sup < best_val_sup
        if improved:
            best_val_sup = val_sup
            best_epoch = epoch

        row = {
            "epoch": float(epoch),
            "train_total": train_stats["total"],
            "train_sup": train_stats["sup"],
            "train_phy": train_stats["phy"],
            "train_smooth": train_stats["smooth"],
            "val_sup": val_sup,
            "val_phy": val_stats["phy"],
            "physics_excluded": float(train_stats["physics_excluded"]),
            "global_step": float(global_step),
            "seconds": time.time() - started,
        }
        history.append(row)
        logger.info(
            "epoch {}/{} step={} train_total={:.5f} train_sup={:.5f} "
            "train_phy={:.5f} val_sup={:.5f} excluded={} secs={:.1f}{}",
            epoch + 1, cfg.epochs, global_step, train_stats["total"],
            train_stats["sup"], train_stats["phy"], val_sup,
            int(train_stats["physics_excluded"]), row["seconds"],
            "  <- best" if improved else "",
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_sup": best_val_sup,
                "best_epoch": best_epoch,
                "config": asdict(cfg),
                "corpus": summary,
                "val_metrics": val_stats,
            },
            checkpoint_path,
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if run is not None:
            run.log(row, step=epoch)

        every = max(int(cfg.image_every), 1)
        if viz_batch is not None and cfg.image_every and (
            epoch % every == 0 or epoch == cfg.epochs - 1
        ):
            _render_and_log(
                run, model, viz_batch, bank, cfg,
                device=device,
                in_channels=model_config.in_channels,
                epoch=epoch,
                out_dir=out_dir,
                caption=(
                    f"epoch {epoch}  val_sup={val_sup:.5f}  "
                    f"val_phy={float(val_stats['phy']):.5f}  "
                    f"excluded={int(train_stats['physics_excluded'])}"
                ),
            )

        if budget[0] <= 0:
            logger.info("max_steps={} reached; stopping", cfg.max_steps)
            break

    epochs_run = len(history)
    final = history[-1] if history else {}
    logger.info(
        "done: {} epoch(s), {} step(s), best val_sup={:.6f} at epoch {} -> {}",
        epochs_run, global_step, best_val_sup, best_epoch, checkpoint_path,
    )
    if run is not None:
        try:
            run.summary["best_epoch"] = best_epoch
            run.summary["best_val_sup"] = best_val_sup
            run.summary["global_step"] = global_step
            run.summary["corpus"] = summary
        except Exception as exc:  # never fail a finished run
            logger.warning("could not write wandb summary: {}", exc)
        run.finish()
    return TrainResult(
        epochs_run=epochs_run,
        global_step=global_step,
        best_val_sup=best_val_sup,
        best_epoch=best_epoch,
        checkpoint=str(checkpoint_path),
        history=history,
        final=final,
    )


def _init_wandb(cfg: ShapingConfig, summary: dict[str, Any]):
    """Start a Weights & Biases run, degrading to a local run when unavailable.

    Mirrors :func:`ml.zernike.train_amp._init_wandb`: the default mode is
    ``"offline"`` because this machine has no ``WANDB_API_KEY`` and no
    ``~/.netrc`` -- ``wandb init`` writes only ``[default] entity/project`` into
    ``wandb/settings``, so an online run would still fail. Every failure path
    returns ``None`` so telemetry can never take down a training run.

    ``reinit`` is deliberately not passed. A shaping run starts exactly one run
    per process, so it is unnecessary, and wandb >= 0.27 deprecates the boolean
    form -- passing either ``True`` or ``False`` warns on every launch.

    The corpus summary is injected into the W&B config because a run is not
    reproducible without knowing which corpus produced it -- see
    :func:`ml.shaping.dataset.corpus_summary`.

    Args:
        cfg: The run configuration.
        summary: Corpus provenance for the train/val splits.

    Returns:
        A ``wandb.Run``, or ``None`` when disabled, unavailable, or failing.
    """
    if not cfg.wandb:
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
            dir=str(Path(cfg.checkpoint_dir)),
            config={
                **{
                    key: value
                    for key, value in asdict(cfg).items()
                    if key != "roots"
                },
                "roots": list(cfg.roots),
                "corpus": summary,
            },
        )
    except Exception as exc:  # telemetry must never kill a run
        logger.warning("wandb.init failed ({}); continuing offline-only", exc)
        return None
    logger.info(
        "wandb run {} (project={}, mode={})",
        getattr(run, "id", None) or getattr(run, "name", None) or "<unnamed>",
        cfg.wandb_project,
        cfg.wandb_mode,
    )
    return run


def _log_comparison_image(run: Any, path: Path, epoch: int, caption: str) -> None:
    """Attach one pred-vs-true montage to the run.

    The PNG is already on disk, so it is attached by path rather than re-rendered
    into a ``wandb.Image`` in memory. Failures are logged and swallowed for the
    same reason as :func:`_init_wandb`.
    """
    try:
        import wandb

        run.log(
            {"compare/pred_vs_true": wandb.Image(str(path), caption=caption)},
            step=epoch,
        )
    except Exception as exc:  # telemetry must never kill a run
        logger.warning("could not log {} to wandb: {}", path.name, exc)


def _render_and_log(
    run: Any,
    model: torch.nn.Module,
    batch: dict[str, Any],
    bank: Any,
    cfg: ShapingConfig,
    *,
    device: torch.device,
    in_channels: int,
    epoch: int,
    out_dir: Path,
    caption: str,
) -> Path | None:
    """Render the pred-vs-true montage to disk and attach it to W&B.

    The figure is written to disk whether or not W&B is enabled, so an offline
    run still leaves a visual record next to the checkpoint.

    Returns:
        The written path, or ``None`` when the batch yielded no rows.
    """
    figure = build_comparison(
        model,
        batch,
        bank,
        device=device,
        grid=cfg.grid,
        in_channels=in_channels,
        max_samples=cfg.viz_samples,
    )
    if not figure.rows:
        logger.debug("no comparison rows available at epoch {}", epoch)
        return None
    if figure.n_physics_skipped:
        logger.info(
            "far-field panels cover {}/{} samples; {} dropped for uncalibrated fov",
            len(figure.rows), figure.n_candidates, figure.n_physics_skipped,
        )
    path = render_comparison(
        figure, out_dir / f"compare_epoch{epoch:03d}.png", caption
    )
    if run is not None:
        _log_comparison_image(run, path, epoch, caption)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` on a usage error (the argparse convention, and
        the same code ``main.py`` uses for a rejected flag combination).
    """
    args = _build_parser().parse_args(argv)
    try:
        cfg = _config_from_args(args)
    except ValueError as exc:
        logger.error("invalid configuration: {}", exc)
        return 2
    if cfg.wandb:
        logger.info(
            "wandb enabled (project={}, mode={}); a missing key or an init "
            "failure degrades to an offline/local run rather than aborting",
            cfg.wandb_project, cfg.wandb_mode,
        )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())