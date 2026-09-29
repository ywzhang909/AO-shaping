"""Offline FourierGSNet training on the recorded debug corpus (no hardware).

This module wires the three halves of the offline GSNet feature together:

* :mod:`ao_shaping.runners.gsnet_offline` — the numpy transforms that turn a
  debug record into ``(source, target, gt_phase)`` grids;
* :mod:`ao_shaping.runners.gsnet_dataset` — the lazy ``Dataset`` + ``DataLoader``
  that streams those records without ever holding the corpus in RAM;
* :mod:`ml.gsnet` — the network, the losses and the training loop.

**No hardware is opened anywhere on this path.** There is no camera, no SLM and
no DM import: the model is trained purely on the pickled records the optimizers
already wrote to ``data/debug/``. The reconstructed far-field uses exactly the
FFT physics the network itself unrolls, so the evaluation numbers are
simulation-side by construction (see :mod:`ml.gsnet.evaluate`).

Typical use (through the CLI group)::

    python src/ao_shaping/main.py slm-gsnet train --epochs 1 --max-samples 32

Artifacts land in ``<dir>/gsnet_train/run-<timestamp>/``:

============================  =================================================
``summary.json``              config + per-epoch history + evaluation means
``comparison.png``            six-panel predicted-vs-ground-truth montage
``train_history.png``         per-epoch loss curves
``fourier_gsnet_best.pt``     best-state checkpoint (from ``ml.gsnet.train``)
============================  =================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (deliberately after Agg)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from ao_shaping.runners.gsnet_cache import (  # noqa: E402
    GSNetCacheError,
    prepare_gsnet_cache,
)
from ao_shaping.runners.gsnet_dataset import (
    GSNetDebugDataset,
    build_gsnet_dataloader,
)  # noqa: E402
from ao_shaping.runners.gsnet_offline import (  # noqa: E402
    DEFAULT_ROOTS,
    RecordIndex,
    build_record_index,
)
from ao_shaping.runners.runner_common import RunParams, option  # noqa: E402
from ao_shaping.utils.io.cli_helpers import setup_coredumpy  # noqa: E402
from ml.gsnet.evaluate import evaluate_model  # noqa: E402
from ml.gsnet.model import FourierGSNet, count_parameters  # noqa: E402
from ml.gsnet.train import train_gsnet  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "PANEL_KEYS",
    "SCHEDULER_GAMMA",
    "SCHEDULER_STEP_SIZE",
    "GsnetTrainParams",
    "TrainResult",
    "render_comparison",
    "reconstruct_far_intensity",
    "resolve_device",
    "run_offline_training",
]

#: Panel order of the comparison montage. Fixed, because the W&B image and the
#: PNG must show the same columns in the same order.
PANEL_KEYS: tuple[str, ...] = (
    "source",
    "target",
    "gt_phase",
    "pred_phase",
    "far_pred",
    "far_diff",
)

#: ``StepLR`` schedule of :func:`ml.gsnet.train.train_gsnet`, mirrored here only
#: so the learning rate can be logged to W&B from the returned history (the
#: training loop owns its optimizer and does not expose the scheduler).
SCHEDULER_STEP_SIZE: int = 30
SCHEDULER_GAMMA: float = 0.5

#: Sample name of the montage written next to the requested ``out_path``.
HISTORY_FIGURE_NAME: str = "train_history.png"

#: Sample name of the comparison montage.
COMPARISON_FIGURE_NAME: str = "comparison.png"

#: Sample name of the JSON run report.
SUMMARY_NAME: str = "summary.json"

#: W&B modes this CLI accepts, mapped to the literals ``wandb.init`` expects. The
#: value type is what keeps the lookup result precisely typed without a cast.
_WANDB_MODES: dict[str, Literal["online", "offline", "disabled"]] = {
    "online": "online",
    "offline": "offline",
    "disabled": "disabled",
}


def _resolve_wandb_mode(spec: str) -> Literal["online", "offline", "disabled"]:
    """Validate ``--wandb-mode`` and narrow it to a mode ``wandb.init`` accepts.

    Args:
        spec: The raw flag value; case and surrounding whitespace are ignored.

    Returns:
        ``"online"``, ``"offline"`` or ``"disabled"``.

    Raises:
        ValueError: If ``spec`` is not one of the three supported modes.
    """
    key = (spec or "").strip().lower()
    try:
        return _WANDB_MODES[key]
    except KeyError as exc:
        raise ValueError(
            f"wandb_mode must be one of {sorted(_WANDB_MODES)}, got {spec!r}"
        ) from exc


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class GsnetTrainParams:
    """Every knob of the offline FourierGSNet training run.

    ``seed`` deliberately carries **no** ``option()``: the ``--seed`` flag on the
    command belongs to :class:`~ao_shaping.runners.runner_common.RunParams`,
    and two options of the same name on one click command is a hard error. The
    value here is the programmatic default, and
    :func:`run_offline_training` prefers ``run.seed`` whenever it is set.
    """

    epochs: Annotated[
        int, option("--epochs", help="Number of training epochs (default: 50).")
    ] = 50
    batch_size: Annotated[
        int, option("--batch-size", help="Samples per batch (default: 16).")
    ] = 16
    lr: Annotated[
        float, option("--lr", help="Adam learning rate (default: 1e-3).")
    ] = 1e-3
    w_phase: Annotated[
        float,
        option(
            "--w-phase", help="Weight of the GS phase-regression loss (default: 1.0)."
        ),
    ] = 1.0
    w_shaping: Annotated[
        float,
        option(
            "--w-shaping",
            help="Weight of the far-field shaping loss (default: 40.0).",
        ),
    ] = 40.0
    num_layers: Annotated[
        int, option("--num-layers", help="Unrolled GS layers (default: 10).")
    ] = 10
    base_channels: Annotated[
        int, option("--base-channels", help="CNN width per layer (default: 32).")
    ] = 32
    grid: Annotated[
        int, option("--grid", help="Sample grid side length (default: 64).")
    ] = 64
    num_workers: Annotated[
        int,
        option(
            "--num-workers",
            help="DataLoader worker processes; 0 keeps RAM at one pickle (default: 0).",
        ),
    ] = 0
    device: Annotated[
        str,
        option(
            "--device",
            help="Compute device: auto / cpu / cuda (default: auto).",
        ),
    ] = "auto"
    max_samples: Annotated[
        int,
        option(
            "--max-samples",
            help="Samples per epoch; 0 uses every indexed record (default: 0).",
        ),
    ] = 0
    roots: Annotated[
        str,
        option(
            "--roots",
            help=(
                "Glob of debug pickles to train on; empty uses "
                f"{', '.join(DEFAULT_ROOTS)} (default: empty)."
            ),
        ),
    ] = ""
    wandb_project: Annotated[
        str, option("--wandb-project", help="W&B project (default: gsnet-offline-train).")
    ] = "gsnet-offline-train"
    wandb_entity: Annotated[
        str, option("--wandb-entity", help="W&B entity/team (default: empty).")
    ] = ""
    wandb_name: Annotated[
        str, option("--wandb-name", help="W&B run name (default: empty = auto).")
    ] = ""
    wandb_mode: Annotated[
        str,
        option(
            "--wandb-mode",
            help=(
                "W&B mode: offline / online / disabled. Defaults to 'offline', so the "
                "command works with no credentials. For a live run you must first "
                "run 'wandb login' and then pass '--wandb-mode online'."
            ),
        ),
    ] = "offline"
    out_dir: Annotated[
        str,
        option(
            "--out-dir",
            help="Artifact directory; empty uses <dir>/gsnet_train/run-<timestamp>.",
        ),
    ] = ""
    n_compare: Annotated[
        int,
        option(
            "--n-compare",
            help="Samples rendered in the comparison montage (default: 6).",
        ),
    ] = 6
    seed: int = 0


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """Everything one offline training run produced.

    Attributes:
        out_dir: Directory holding every artifact.
        history: Per-epoch ``{loss, phase_loss, shaping_loss, intensity_loss}``.
        best_epoch: 1-based epoch with the lowest total loss.
        best_loss: Lowest total loss achieved.
        checkpoint_path: Best-state checkpoint, or ``None`` when none improved.
        seconds: Wall-clock training time.
        eval_means: Mean simulation-side metrics over the corpus.
        n_records: Number of indexed records the loader can serve.
        n_parameters: Trainable parameter count of the model.
        device: Resolved torch device string.
        seed: Seed actually used.
        artifacts: ``{name: path}`` of every written file.
        wandb_url: Run URL, or ``None`` when W&B was disabled/unavailable.
    """

    out_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_loss: float = float("inf")
    checkpoint_path: Path | None = None
    seconds: float = 0.0
    eval_means: dict[str, float] = field(default_factory=dict)
    n_records: int = 0
    n_parameters: int = 0
    device: str = "cpu"
    seed: int = 0
    artifacts: dict[str, Path] = field(default_factory=dict)
    wandb_url: str | None = None


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(spec: str) -> str:
    """Resolve a ``--device`` string to a concrete torch device.

    Args:
        spec: ``"auto"``, ``"cpu"`` or ``"cuda"`` (case/whitespace tolerant).

    Returns:
        ``"cuda"`` or ``"cpu"``. ``"auto"`` picks CUDA when available.
        An explicit ``"cuda"`` on a CPU-only box degrades to ``"cpu"`` with a
        warning rather than raising, so a run configured on a GPU workstation
        still finishes on a laptop.

    Raises:
        ValueError: If ``spec`` is not one of the three accepted values.
    """
    key = (spec or "auto").strip().lower()
    if key in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if key == "cuda":
        if not torch.cuda.is_available():
            logger.warning("device='cuda' requested but CUDA is unavailable; using CPU")
            return "cpu"
        return "cuda"
    if key == "cpu":
        return "cpu"
    raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {spec!r}")


# --------------------------------------------------------------------------- #
# Far-field reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_far_intensity(
    model: FourierGSNet,
    source: torch.Tensor,
    target: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict a phase and propagate it to the far field.

    The propagation body is byte-identical to the physics the network unrolls
    (``ml.gsnet.model.FFTLayer``) and to the evaluation metrics, so the montage
    shows exactly what the loss saw.

    Args:
        model: The network (switched to ``eval`` and restored afterwards).
        source: Source-plane intensity ``(B, 1, H, W)``.
        target: Requested far-field intensity ``(B, 1, H, W)``.
        device: Device to run on.

    Returns:
        ``(pred_phase, far_intensity)``, both ``(B, 1, H, W)`` on ``device``.
        ``far_intensity`` is unnormalised (arbitrary absolute scale).
    """
    model.to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            src = source.to(device)
            tgt = target.to(device)
            pred_phase = model(src, tgt)
            source_amp = torch.sqrt(src.clamp_min(0.0) + 1e-12)
            field = source_amp * torch.exp(1j * pred_phase)
            far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
            far_intensity = torch.abs(far) ** 2
    finally:
        model.train(was_training)
    return pred_phase, far_intensity


def _unit_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise a field to unit total, per image.

    Byte-identical to the normalisation inside
    :meth:`ml.gsnet.losses.ShapingLosses.intensity_mse` and
    :func:`ml.gsnet.evaluate.compute_sample_metrics`, so the montage's residual
    panel is the very quantity ``intensity_mse`` scores. It is required because
    an FFT intensity is only defined up to an arbitrary constant.

    Args:
        tensor: ``(B, 1, H, W)`` (extra leading dims are tolerated).

    Returns:
        The same tensor divided by its per-image plane sum.
    """
    return tensor / (tensor.sum(dim=(-2, -1), keepdim=True) + 1e-12)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _panel_cmap(key: str) -> str:
    """Return the colormap for one montage panel."""
    if key in ("gt_phase", "pred_phase"):
        # Cyclic map: phase is a circle, not a scalar.
        return "twilight_shifted"
    if key == "far_diff":
        return "RdBu_r"
    return "magma"


def _render_history(history: Sequence[dict[str, float]], out_path: Path) -> Path:
    """Plot the per-epoch loss curves to ``out_path``.

    Args:
        history: Per-epoch dicts carrying at least ``loss``.
        out_path: PNG destination; parent directories are created.

    Returns:
        ``out_path``. An empty history still yields a valid placeholder PNG so
        the artifact set stays complete.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    if history:
        epochs = list(range(1, len(history) + 1))
        for key in ("loss", "phase_loss", "shaping_loss", "intensity_loss"):
            series = [float(entry.get(key, float("nan"))) for entry in history]
            if all(np.isfinite(v) for v in series):
                ax.plot(epochs, series, marker="o", markersize=3, label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if all(v > 0 for v in [float(e.get("loss", 0.0)) for e in history]):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no epochs recorded", ha="center", va="center")

    fig.suptitle("FourierGSNet offline training history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_comparison(
    out_path: str | Path,
    rows: Sequence[dict[str, np.ndarray]],
    history: Sequence[dict[str, float]],
    *,
    title: str,
) -> Path:
    """Render the six-panel montage and the loss history.

    One row per sample, six columns in :data:`PANEL_KEYS` order: the two inputs
    (``source``, ``target``), the two phases (``gt_phase``, ``pred_phase``), the
    predicted far field, and its difference against the target.

    Args:
        out_path: Destination PNG (usually ``comparison.png``). The loss history
            is written next to it as :data:`HISTORY_FIGURE_NAME`.
        rows: Per-sample ``{panel_key: (H, W) ndarray}`` mappings.
        history: Per-epoch loss history for the companion figure.
        title: Suptitle of the montage.

    Returns:
        The montage path actually written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    fig, axes = plt.subplots(max(n, 1), len(PANEL_KEYS), figsize=(3.0 * len(PANEL_KEYS), 3.0 * max(n, 1)), squeeze=False)

    if n == 0:
        axes[0][0].text(0.5, 0.5, "no comparison samples", ha="center", va="center")

    for r, row in enumerate(rows):
        for c, key in enumerate(PANEL_KEYS):
            ax = axes[r][c]
            data = np.asarray(row[key], dtype=np.float64)
            if key == "far_diff":
                # Symmetric limits so zero sits at the middle of the diverging
                # map; otherwise the peak would wash out the structure.
                bound = float(np.max(np.abs(data))) or 1.0
                ax.imshow(data, cmap=_panel_cmap(key), vmin=-bound, vmax=bound)
            else:
                ax.imshow(data, cmap=_panel_cmap(key))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(key, fontsize=11)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    _render_history(history, out.parent / HISTORY_FIGURE_NAME)
    return out


# --------------------------------------------------------------------------- #
# W&B (optional — a missing wandb must never fail a run)
# --------------------------------------------------------------------------- #
def _load_wandb_logger() -> Any:
    """Import :mod:`ml.wandb_logger` lazily, or return ``None``.

    W&B is an optional extra. Importing the module eagerly would make the whole
    training command fail on a machine without the dependency, which contradicts
    the "training must still complete" contract.

    Returns:
        The ``ml.wandb_logger`` module, or ``None`` when wandb is unavailable.
    """
    try:
        from ml import wandb_logger
    except ImportError as exc:
        logger.warning("wandb is unavailable ({}); training without W&B", exc)
        return None
    return wandb_logger


def _scheduled_lr(base_lr: float, epoch: int) -> float:
    """Learning rate of the 1-based ``epoch`` under the training loop's schedule.

    Mirrors ``StepLR(step_size=30, gamma=0.5)``: the loop steps once per epoch,
    and torch decays when ``last_epoch % step_size == 0``, so epoch ``e`` runs
    at ``base_lr * gamma ** (e // 30)``.
    """
    return float(base_lr) * SCHEDULER_GAMMA ** (int(epoch) // SCHEDULER_STEP_SIZE)


def _log_epoch_metrics(
    wandb_run: Any,
    history: Sequence[dict[str, float]],
    base_lr: float,
) -> None:
    """Push every epoch's losses and its learning rate to W&B."""
    for offset, entry in enumerate(history):
        epoch = offset + 1
        payload = {
            "train/loss": float(entry.get("loss", float("nan"))),
            "train/phase_loss": float(entry.get("phase_loss", float("nan"))),
            "train/shaping_loss": float(entry.get("shaping_loss", float("nan"))),
            "train/intensity_loss": float(entry.get("intensity_loss", float("nan"))),
            "train/lr": _scheduled_lr(base_lr, epoch),
        }
        wandb_run.log(payload, step=epoch)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _seed_everything(seed: int) -> None:
    """Seed torch and numpy before the model is constructed.

    :func:`ml.gsnet.train.train_gsnet` seeds the *training* stream, but the
    network's initial weights are drawn when :class:`FourierGSNet` is built.
    Seeding here too is what makes two runs with the same ``--seed`` produce
    identical loss histories.

    Args:
        seed: The seed actually in force for this run.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def _resolve_roots(roots: str) -> tuple[str, ...]:
    """Return the glob(s) to index: the user glob, or the debug defaults."""
    text = (roots or "").strip()
    return (text,) if text else tuple(DEFAULT_ROOTS)


def _resolve_out_dir(run: RunParams, params: GsnetTrainParams) -> Path:
    """Return the artifact directory for this run."""
    text = (params.out_dir or "").strip()
    if text:
        return Path(text)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(run.dir) / "gsnet_train" / f"run-{stamp}"


def _to_plane(tensor: torch.Tensor) -> np.ndarray:
    """Detach a prediction tensor down to a plain 2-D plane.

    Prediction tensors arrive as ``(1, 1, H, W)`` and ``(1, H, W)`` depending on
    the caller, so leading singleton axes are squeezed until a 2-D plane is left.

    Args:
        tensor: Any tensor whose trailing two dims are the image.

    Returns:
        A ``(H, W)`` float64 numpy array safe for :func:`matplotlib.pyplot.imshow`.
    """
    plane = tensor.detach().float().cpu()
    while plane.ndim > 2:
        plane = plane.squeeze(0)
    return plane.numpy()


def _comparison_rows(
    model: FourierGSNet,
    dataset: GSNetDebugDataset,
    count: int,
    device: str,
) -> list[dict[str, np.ndarray]]:
    """Build one montage row per sample from the first ``count`` records.

    Args:
        model: Trained model.
        dataset: The training loader's dataset (indexed directly, so no extra
            epoch is streamed).
        count: Maximum number of rows.
        device: Compute device.

    Returns:
        ``[{panel_key: (H, W) ndarray}]`` in dataset order.
    """
    rows: list[dict[str, np.ndarray]] = []
    total = min(int(count), len(dataset))
    for i in range(total):
        source, target, gt_phase = dataset[i]
        pred_phase, far_pred = reconstruct_far_intensity(
            model, source.unsqueeze(0), target.unsqueeze(0), device
        )
        residual = _unit_sum(far_pred) - _unit_sum(target.unsqueeze(0).to(device))
        rows.append(
            {
                "source": _to_plane(source),
                "target": _to_plane(target),
                "gt_phase": _to_plane(gt_phase),
                "pred_phase": _to_plane(pred_phase),
                "far_pred": _to_plane(_unit_sum(far_pred)),
                "far_diff": _to_plane(residual),
            }
        )
    return rows


def _ensure_lean_cache(index: RecordIndex) -> None:
    """Build the lean mmap cache for every pickle in ``index`` (best effort).

    The debug corpus is ~9.6 GB of raw records but only ~0.20 GB of ``_c`` /
    ``_img``; the ``_phase`` field dominates and is never used for training.
    :func:`prepare_gsnet_cache` writes the trimmed, mmap-able twin of each
    pickle next to it, which is what turns a full re-read of a multi-GB pickle
    (once per record per epoch) into a few MiB of mapped reads. The saving is
    per-file and scales with how much of that pickle is ``_phase``: a
    ``_img``-dominated dump caches at roughly 1:1, while a ``_phase``-dominated
    one shrinks by orders of magnitude (across the full debug corpus the raw
    records are ~9.6 GB and the trimmed set ~0.20 GB).

    The call is idempotent -- an up-to-date cache directory is reused without
    opening the source pickle -- so a second run costs nothing. It is also
    strictly best effort: a file the cacher cannot handle raises
    :class:`GSNetCacheError`, and :class:`GSNetDebugDataset` transparently
    falls back to ``pickle.load`` for that file, so caching is an optimisation
    and never a precondition for training.
    """
    try:
        directories = prepare_gsnet_cache(index)
    except GSNetCacheError as exc:
        logger.warning(
            "lean cache build failed ({}); the dataset will fall back to "
            "reading pickles directly",
            exc,
        )
        return
    logger.info(
        "lean cache ready for {} of {} pickles (mmap-backed reads; the saving "
        "is per-file and scales with how much of the pickle is _phase)",
        len(directories),
        len({pkl for pkl, _key in index.entries}),
    )


def run_offline_training(run: RunParams, params: GsnetTrainParams) -> TrainResult:
    """Train FourierGSNet offline and write every artifact.

    Sequence: index the debug pickles -> stream them through the lazy dataset ->
    train -> evaluate in simulation -> render the montage + history -> write
    ``summary.json``. W&B is initialised *before* training and always finished
    in a ``finally`` block; a missing or failing wandb degrades to a warning and
    never prevents training or the PNGs.

    Args:
        run: Run-wide options (``dir``, ``seed``). ``run.seed``, when set,
            overrides :attr:`GsnetTrainParams.seed`.
        params: Training hyper-parameters and output controls.

    Returns:
        A populated :class:`TrainResult`.

    Raises:
        ValueError: For a non-positive ``epochs`` / negative ``max_samples`` /
            negative ``n_compare``, or an unknown ``device``. An empty corpus
            raises ``ValueError`` from :func:`build_gsnet_dataloader` (the
            indexer only warns), and a malformed record raises
            :class:`~ao_shaping.runners.gsnet_dataset.GSNetRecordError`.
    """
    setup_coredumpy()

    if params.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {params.epochs}")
    if params.max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {params.max_samples}")
    if params.n_compare < 0:
        raise ValueError(f"n_compare must be >= 0, got {params.n_compare}")

    device = resolve_device(params.device)
    seed = int(run.seed) if run.seed is not None else int(params.seed)
    out_dir = _resolve_out_dir(run, params)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_record_index(roots=_resolve_roots(params.roots))
    _ensure_lean_cache(index)
    loader = build_gsnet_dataloader(
        index,
        grid=params.grid,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        num_samples=params.max_samples or None,
        seed=seed,
    )

    _seed_everything(seed)
    model = FourierGSNet(
        num_layers=params.num_layers, base_channels=params.base_channels
    )
    logger.info(
        "FourierGSNet: {} layers, base_channels={}, {} parameters, device={}",
        params.num_layers,
        params.base_channels,
        count_parameters(model),
        device,
    )

    wandb_logger = _load_wandb_logger()
    wandb_run: Any = None
    wandb_url: str | None = None
    if wandb_logger is not None:
        try:
            wandb_run = wandb_logger.init_wandb(
                project=params.wandb_project,
                name=params.wandb_name or None,
                config=asdict(params),
                entity=params.wandb_entity or None,
                mode=_resolve_wandb_mode(params.wandb_mode),
            )
            wandb_url = getattr(wandb_run, "url", None)
        except Exception as exc:  # noqa: BLE001 - W&B must never break a run
            logger.warning("wandb.init failed ({}); training without W&B", exc)
            wandb_run = None

    try:
        trained = train_gsnet(
            model,
            loader,
            epochs=params.epochs,
            lr=params.lr,
            w_phase=params.w_phase,
            w_shaping=params.w_shaping,
            device=device,
            checkpoint_dir=out_dir,
            seed=seed,
        )
        if wandb_run is not None:
            _log_epoch_metrics(wandb_run, trained.history, params.lr)

        summary = evaluate_model(model, loader, device=device)
        if wandb_run is not None:
            wandb_run.log({f"eval/{k}": float(v) for k, v in summary.means.items()})

        dataset = loader.dataset
        if not isinstance(dataset, GSNetDebugDataset):
            raise TypeError(
                "expected the GSNet dataloader to wrap a GSNetDebugDataset, got "
                f"{type(dataset).__name__}"
            )
        rows = _comparison_rows(model, dataset, params.n_compare, device)
        comparison_path = render_comparison(
            out_dir / COMPARISON_FIGURE_NAME,
            rows,
            trained.history,
            title=(
                f"FourierGSNet offline - {len(index.entries)} records, "
                f"best epoch {trained.best_epoch} (loss {trained.best_loss:.4f})"
            ),
        )
        if wandb_run is not None and wandb_logger is not None and rows:
            wandb_run.log(
                {
                    "shaping_comparison": wandb_logger.log_shaping_comparison(
                        rows, PANEL_KEYS
                    )
                }
            )

        artifacts: dict[str, Path] = {
            "summary": out_dir / SUMMARY_NAME,
            "comparison": comparison_path,
            "history": out_dir / HISTORY_FIGURE_NAME,
        }
        if trained.checkpoint_path is not None:
            artifacts["checkpoint"] = trained.checkpoint_path

        payload = {
            "config": asdict(params),
            "resolved": {
                "device": device,
                "seed": seed,
                "n_records": len(index.entries),
                "roots": list(_resolve_roots(params.roots)),
                "out_dir": str(out_dir),
            },
            "model": {
                "num_layers": params.num_layers,
                "base_channels": params.base_channels,
                "n_parameters": count_parameters(model),
            },
            "training": {
                "history": trained.history,
                "best_epoch": trained.best_epoch,
                "best_loss": trained.best_loss,
                "seconds": trained.seconds,
            },
            "evaluation": {
                "n_samples": summary.n_samples,
                "means": summary.means,
            },
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "wandb_url": wandb_url,
        }
        artifacts["summary"].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        for name, path in artifacts.items():
            logger.info("wrote {} artifact: {}", name, path)
        logger.info(
            "FourierGSNet offline training done: best_epoch={} best_loss={:.6f} "
            "far_correlation={:.4f} encircled_energy={:.4f} ({:.1f}s)",
            trained.best_epoch,
            trained.best_loss,
            summary.means.get("far_correlation", float("nan")),
            summary.means.get("encircled_energy", float("nan")),
            trained.seconds,
        )

        return TrainResult(
            out_dir=out_dir,
            history=trained.history,
            best_epoch=trained.best_epoch,
            best_loss=trained.best_loss,
            checkpoint_path=trained.checkpoint_path,
            seconds=trained.seconds,
            eval_means=dict(summary.means),
            n_records=len(index.entries),
            n_parameters=count_parameters(model),
            device=device,
            seed=seed,
            artifacts=artifacts,
            wandb_url=wandb_url,
        )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:  # noqa: BLE001 - teardown must not mask errors
                logger.warning("wandb.finish failed: {}", exc)
