"""Frozen configuration contract for :mod:`ml.shaping`.

This module is the **single shared interface** that ``models`` / ``dataset`` /
``physics`` / ``losses`` / ``train`` / ``evaluate`` all import. It deliberately
imports **stdlib only** -- no ``torch``, no ``ml.*``, no device code -- so that
any track can be developed, imported and unit-tested without pulling the
optional ``ml`` dependency group or a native SDK into the process.

Why a frozen dataclass
----------------------
``train._build_parser`` reads **every** CLI default from this dataclass rather
than writing literals into ``add_argument``. That is not a style preference: this
repository shipped a bug where ``train_amp``'s parser hard-wrote
``default=1`` for ``far_field_padding`` and silently overrode the calibrated
value of 10 on every single run (see ``report/zernike_phase2amp/report.md``).
A frozen dataclass makes "the parser default" and "the dataclass default" the
same object, so they cannot drift.

Why these particular defaults
-----------------------------
* ``grid=64`` -- the corpus materialiser's default; every measured number in the
  repo's reports is at this grid.
* ``w_sup = w_phy = 1.0`` -- both terms are load-bearing. ``L_phy`` alone is
  blind to piston (a global phase offset leaves the far-field intensity
  unchanged), so it cannot pin a phase on its own; ``L_sup`` is what anchors it.
* ``w_smooth = 0.0`` -- ``ml/zernike/losses.py``'s phase-smoothness penalty was
  **measured harmful** (paired dR2 -0.0264 at w=0.003 and -0.1670 at w=0.03, 0/3
  seeds negative) and removed from the codebase. It stays here as an *off-by-
  default* knob with that history recorded, not as a recommendation.
* ``cudnn_deterministic = True`` -- the repo's existing trainers were measured
  **non-reproducible** (val R2 0.8510 vs 0.8573 run-to-run, fitted coefficients
  differing by 0.065) because cudnn.benchmark defaults to False but
  nondeterministic atomics are still used in convolution backward. The fix costs
  about 4% wall clock and is verified to give bit-identical reruns. This module
  is new, so turning it on does not perturb any existing baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DeviceSpec",
    "IMAGE_MODES",
    "ShapingConfig",
    "UNKNOWN_EXPOSURE",
    "UNKNOWN_FOV",
]

#: Sentinel written by :func:`ml.shaping.dataset.shaping_collate` when a record
#: carries no usable exposure. ``torch.utils.data.default_collate`` raises
#: ``TypeError`` on a Python ``None`` inside a batch, so the sentinel is what
#: keeps a mixed batch collatable. It is negative because a real exposure is
#: always strictly positive.
UNKNOWN_EXPOSURE: float = -1.0

#: Sentinel written by :func:`ml.shaping.dataset.shaping_collate` when a record
#: carries no derivable ``fov_px``. ``fov_px`` is ``None`` for roughly a fifth of
#: the corpus and there is no calibrated ``far_field_padding`` for it, so such a
#: record contributes to the supervised term but is **excluded** from the
#: physics term (see :mod:`ml.shaping.physics`).
UNKNOWN_FOV: int = -1

#: ``"auto"`` resolves to CUDA when available and CPU otherwise.
DeviceSpec = Literal["auto", "cuda", "cpu"]

#: Far-field intensity modes accepted by
#: :class:`ml.hwdataset.records.MaterialiserConfig`, re-declared here only so
#: the CLI can validate before the corpus is touched. ``abs255`` keeps the
#: absolute 0-255 detector scale, ``peak`` divides by the frame maximum,
#: ``raw`` applies no scaling.
#:
#: Why ``abs255`` is the default: ``report/hwdataset_corpus/report.md`` §4
#: measured the corpus exposure spanning 0.1-80 ms (2.9 orders of magnitude),
#: and §10 measured frame brightness tracking exposure almost exactly. Keeping
#: the absolute scale preserves that signal, and it is supplied separately as
#: the standardised ``exposure_log10`` input channel. Peak-normalising instead
#: would make the target exposure-invariant -- useful when the target really
#: should be brightness-independent, at the cost of throwing the exposure cue
#: away from the regression target. That is a deliberate choice, so it is a flag
#: rather than a hard-coded assumption.
#:
#: ``sum``-style normalisation is deliberately NOT offered: it was measured to
#: report PSNR 72 dB / SSIM 0.9996 while its R^2 was *worse* than a constant
#: predictor, i.e. it measures the normalisation rather than the fit.
IMAGE_MODES = ("abs255", "peak", "raw")


def parse_csv_tuple(value: str) -> tuple[str, ...] | None:
    """Parse a comma-separated CLI value into a tuple, preserving empty as ``None``.

    Lives in :mod:`ml.shaping.config` rather than in either CLI so that
    ``train`` and ``evaluate`` cannot drift apart: the two must accept exactly the
    same ``--families`` / ``--sources`` syntax, since evaluation has to reproduce
    the training run's corpus selection verbatim.

    Args:
        value: Comma-separated text, possibly empty.

    Returns:
        The stripped parts as a tuple, or ``None`` when ``value`` is empty or
        whitespace -- which the config reads as "no restriction".
    """
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    return parts or None


@dataclass(frozen=True)
class ShapingConfig:
    """Every knob of the inverse-shaping trainer, in one frozen record.

    Attributes:
        grid: Side length ``g`` of the square pupil/CCD grid. Must be >= 8 so the
            U-Net's five pooling stages leave a non-degenerate bottleneck. The
            corpus materialiser is configured to match.
        features: U-Net encoder widths. Must be non-empty and strictly
            non-decreasing is *not* required, but every entry must be >= 1 and
            ``len(features) >= 2`` because ``UNetGenerator`` builds one decoder
            block per level.
        in_channels: Network input channels. 2 = ``[far-field image,
            standardised log-exposure broadcast]``; exposure is an *input*
            covariate in this corpus (9 values spanning 0.1-80 ms, i.e. 2.9
            orders of magnitude), and the dataset's ``image`` deliberately keeps
            the absolute brightness that encodes it.
        w_sup: Weight of the supervised phasor L1 term.
        w_phy: Weight of the physics round-trip L1 term.
        w_smooth: Weight of the phasor total-variation term. **Default 0** --
            see the module docstring.
        lr: Adam learning rate.
        batch_size: Samples per batch.
        epochs: Maximum training epochs. The loop also honours ``max_steps``.
        seed: Seed for weight init, the grouped sampler and any augmentation.
        num_workers: DataLoader worker processes. On Windows each worker
            re-pickles the Dataset, which the corpus Dataset supports by
            contract (``__getstate__`` drops the payload LRU and rebuilds it).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        cudnn_deterministic: Force deterministic cuDNN. See module docstring.
        limit_files: Cap on scanned pickles, for smoke runs. ``None`` scans all.
        max_steps: Hard cap on optimiser steps, overriding ``epochs`` when set.
            Used by the tests to keep a smoke run to a couple of steps.
        checkpoint_dir: Directory for checkpoints and the training history JSON.
        resume: Path to a checkpoint to resume from. ``None`` starts fresh.
        wandb: Attempt a Weights & Biases run. Falls back to offline mode when no
            key is configured, and is wrapped so an import failure is a warning
            rather than a crash.
        roots: Corpus roots handed to
            :func:`~ml.hwdataset.index.build_hw_index`.
        forward_checkpoint: Optional path to a trained forward-model checkpoint
            for :func:`~ml.zernike.amp_checkpoint.load_trained_forward_model`.
            ``None`` means "use a zero-coefficient model", i.e. a pure calibrated
            Fraunhofer propagation with **no** learned bench aberration. That is
            a strictly weaker physics term and the report must say which was
            used -- it is not a silent default.
        image_mode: Far-field intensity scaling, one of :data:`IMAGE_MODES`.
            See that constant for why ``abs255`` is the default and why ``sum``
            normalisation is deliberately absent.
        balance_alpha: Strength of per-family balancing in the train sampler,
            in ``[0, 1]``. ``0`` reproduces the corpus proportions exactly; ``1``
            gives every family the same number of files per epoch; ``0.5``
            (default) is the standard sqrt-inverse-frequency compromise.
            See :class:`ml.shaping.dataset.FamilyBalancedGroupedSampler` for why
            balancing resamples whole files rather than records.
        max_file_repeats: Hard cap on how many times one file may appear in a
            single epoch under balancing. Balancing a corpus whose families are
            350x apart would otherwise replay the smallest family's one or two
            files hundreds of times, which converts a sampling bias into an
            overfitting one. The corpus report states this limit explicitly --
            weighting mitigates imbalance, it does not replace missing data.
        families: Restrict the corpus to these families, or ``None`` for all.
        sources: Restrict the corpus to these phase sources, or ``None`` for all.
            Both are forwarded to
            :meth:`~ml.hwdataset.index.HwCorpusIndex.filter`.
        drop_exposure_channel: When ``True`` the network takes the far-field
            image alone instead of ``[image, exposure_log10]``. This exists
            because exposure is almost perfectly stratified **by family** in
            this corpus (§4: each family uses a near-constant exposure), so the
            exposure channel can act as a family label rather than as optical
            information. Dropping it is the ablation that separates "learned the
            optics" from "identified the family".
    """

    grid: int = 64
    features: tuple[int, ...] = (16, 32, 64, 128, 256)
    in_channels: int = 2

    w_sup: float = 1.0
    w_phy: float = 1.0
    w_smooth: float = 0.0

    lr: float = 3e-4
    batch_size: int = 16
    epochs: int = 40
    seed: int = 42
    num_workers: int = 4

    device: DeviceSpec = "auto"
    cudnn_deterministic: bool = True

    limit_files: int | None = None
    max_steps: int | None = None

    checkpoint_dir: str = "logs/shaping"
    resume: str | None = None
    wandb: bool = True
    #: ``wandb.init(mode=...)`` values. ``"offline"`` is the default because this
    #: machine has no ``WANDB_API_KEY`` and no ``~/.netrc`` -- ``wandb init``
    #: writes only ``[default] entity/project`` into ``wandb/settings``, so an
    #: online run would still fail. The offline directory syncs later with
    #: ``wandb sync <dir>``. Mirrors ``ml.zernike.train_amp.WandbMode``.
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "offline"
    #: Matches the ``[default] project`` in ``wandb/settings``, created via
    #: ``wandb init``. Passed explicitly so a run does not depend on the working
    #: directory; ``entity`` is left unset so W&B reads it from that same file.
    wandb_project: str = "ao-shaping"
    wandb_name: str | None = None
    #: Render the pred-vs-true montage every N epochs. ``0`` disables the
    #: figure entirely (metrics are still logged).
    image_every: int = 1
    #: Rows in the pred-vs-true montage.
    viz_samples: int = 4
    roots: tuple[str, ...] = ("data/debug",)
    forward_checkpoint: str | None = None
    #: Optional JSON index snapshot. **Set this for any run whose numbers will be
    #: quoted.** The corpus under ``data/debug`` is a moving target: during one
    #: session it went from 11,393 records / 261 pickles (the shipped cache) to
    #: 14,335 / 165, then to 14,350 / 167 as new runs landed. Without a pinned
    #: snapshot two runs a day apart silently train on different data, and a
    #: reported metric cannot be attributed to a dataset. ``None`` rescans on
    #: every invocation (~35 s), which is fine for smoke runs and wrong for
    #: anything reproducible.
    index_cache: str | None = None

    # -- data preprocessing (see report/hwdataset_corpus/report.md) --------
    image_mode: str = "abs255"
    balance_alpha: float = 0.5
    max_file_repeats: int = 8
    families: tuple[str, ...] | None = None
    sources: tuple[str, ...] | None = None
    drop_exposure_channel: bool = False

    def __post_init__(self) -> None:
        """Reject a configuration that cannot produce a valid run.

        Raises:
            ValueError: On a grid too small for the U-Net, an empty or
                non-positive feature list, fewer than two U-Net levels, a
                negative loss weight, a non-positive learning rate / batch size
                / epoch count, an unknown device spec, a negative
                ``limit_files`` / ``max_steps``, or a negative worker count.
        """
        if int(self.grid) < 8:
            raise ValueError(
                f"grid must be >= 8 for the U-Net bottleneck, got {self.grid}"
            )
        if not self.features:
            raise ValueError("features must be a non-empty sequence")
        if any(int(f) < 1 for f in self.features):
            raise ValueError(f"every feature width must be >= 1, got {self.features}")
        if len(self.features) < 2:
            raise ValueError(
                f"features needs >= 2 levels for a U-Net, got {len(self.features)}"
            )
        if int(self.in_channels) < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        for name in ("w_sup", "w_phy", "w_smooth"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if float(self.lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if int(self.batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if int(self.epochs) < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(
                f"device must be one of ('auto', 'cuda', 'cpu'), got {self.device!r}"
            )
        if self.limit_files is not None and int(self.limit_files) < 1:
            raise ValueError(
                f"limit_files must be >= 1 when given, got {self.limit_files}"
            )
        if self.max_steps is not None and int(self.max_steps) < 1:
            raise ValueError(f"max_steps must be >= 1 when given, got {self.max_steps}")
        if int(self.num_workers) < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if not self.roots:
            raise ValueError("roots must contain at least one corpus root")
        if self.image_mode not in IMAGE_MODES:
            raise ValueError(
                f"image_mode must be one of {IMAGE_MODES}, got {self.image_mode!r}"
            )
        alpha = float(self.balance_alpha)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"balance_alpha must be in [0, 1], got {self.balance_alpha!r}")
        if int(self.max_file_repeats) < 1:
            raise ValueError(
                f"max_file_repeats must be >= 1, got {self.max_file_repeats}"
            )
        if self.families is not None and not self.families:
            raise ValueError("families must be None or a non-empty sequence")
        if self.sources is not None and not self.sources:
            raise ValueError("sources must be None or a non-empty sequence")