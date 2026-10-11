"""Pred-vs-true visualisation for :mod:`ml.shaping` training.

The trainer supervises a **unit phasor** (the continuous ``(cos, sin)`` pair) and
adds a physics term that propagates it through the calibrated forward models. A
scalar loss therefore hides two very different failure modes: a network that
gets the right *phase distribution* but the wrong *far field*, and one that
slightly degrades the far field while its phasor error barely moves. This module
renders both, side by side, so a run can be judged by eye rather than by a
number that moved 0.02.

What is shown, and why each panel exists
----------------------------------------

============================  ==========================================
``image``                     The network input: the measured far field.
``true_phase``                Ground-truth pupil phase as an angle.
``pred_phase``                Predicted phase as an angle.
``phasor_err``                ``|pred_phasor - true_phasor|`` per pixel.
``far_pred``                  Physics forward of the **predicted** phasor.
``far_true``                  Physics forward of the **true** phasor.
``far_diff``                  ``far_pred - far_true``.
============================  ==========================================

``phasor_err`` is deliberately the **phasor** distance and not an angle
difference. The supervised loss is an L1 on the phasor pair
(:mod:`ml.shaping.losses`), so the phasor distance is the quantity actually being
optimised, and it has no branch cut. An angle difference would wrap at +/-pi and
hide a large error as a small one along the cut -- the same failure the corpus
report warns about for the *inputs*, reproduced here in the diagnostics.

``far_*`` panels exist only when the sample's FOV has a calibrated forward model
(:data:`ml.zernike.models.PADDING_BY_FOV_PX`). The rebuilt corpus is dominated by
``slm_pib`` whose FOV is 1944 px and therefore *uncalibrated*, so a batch can be
entirely un-propagatable. Rather than render zero-filled far-field panels that
look like a failed prediction, this module reports the shortfall and drops the
far-field columns when nothing in the batch can be propagated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - backend must be set first
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.shaping.dataset import network_input, phase_target  # noqa: E402
from ml.shaping.models import unit_phasor  # noqa: E402

__all__ = [
    "PANEL_KEYS_FULL",
    "PANEL_KEYS_PHASE_ONLY",
    "ComparisonFigure",
    "build_comparison",
    "render_comparison",
]

#: Columns rendered when at least one sample in the batch has a calibrated FOV.
PANEL_KEYS_FULL: tuple[str, ...] = (
    "image",
    "true_phase",
    "pred_phase",
    "phasor_err",
    "far_true",
    "far_pred",
    "far_diff",
)

#: Columns rendered when no sample can be propagated through the forward bank.
#: Far-field panels are omitted rather than filled with zeros, which would be
#: indistinguishable from a genuinely black prediction.
PANEL_KEYS_PHASE_ONLY: tuple[str, ...] = (
    "image",
    "true_phase",
    "pred_phase",
    "phasor_err",
)


@dataclass(frozen=True)
class ComparisonFigure:
    """A rendered-ready pred-vs-true montage.

    Attributes:
        rows: Per-sample ``{panel_key: (grid, grid) ndarray}`` mappings.
        panel_keys: Column order; every key is present in every row.
        n_candidates: Samples offered to the montage before the cap.
        n_physics_skipped: Samples excluded because their FOV has no calibrated
            forward model, counted over the whole batch rather than over the
            rendered rows -- so a run over an uncalibrated corpus reports the
            shortfall instead of quietly rendering fewer columns.
    """

    rows: list[dict[str, np.ndarray]]
    panel_keys: tuple[str, ...]
    n_candidates: int
    n_physics_skipped: int


def _angle(cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Render a phasor pair as an angle in ``[-pi, pi]``.

    Display only. The network never sees an angle -- it consumes and emits the
    continuous pair -- but ``twilight_shifted`` is a cyclic map, so showing the
    angle is the only way to make a phase panel legible to a human.

    Args:
        cos: Cosine channel.
        sin: Sine channel.

    Returns:
        The angle array, same shape as the inputs.
    """
    return np.arctan2(sin, cos)


def build_comparison(
    model: torch.nn.Module,
    batch: dict[str, Any],
    bank: Any,
    *,
    device: torch.device,
    grid: int,
    in_channels: int,
    max_samples: int = 4,
) -> ComparisonFigure:
    """Run the model over one batch and assemble the pred-vs-true panels.

    Args:
        model: The shaping network; called as ``model(image) -> (B, 2, g, g)``.
        batch: One batch from a shaping loader.
        bank: A :class:`~ml.shaping.physics.ForwardBank`.
        device: Device the model and bank live on.
        grid: Spatial size of every panel.
        in_channels: Network input arity (1 when the exposure channel is dropped).
        max_samples: Cap on rendered rows.

    Returns:
        A :class:`ComparisonFigure`. ``rows`` may be empty when the batch is
        smaller than the collate window; callers must handle that.
    """
    image = network_input(batch, grid, in_channels).to(device)
    target = phase_target(batch).to(device)
    fov = batch["fov_px"]

    with torch.no_grad():
        pred = unit_phasor(model(image))

    n_candidates = int(image.shape[0])
    limit = max(0, int(max_samples))

    # Propagate the whole batch once; the bank marks uncalibrated rows itself
    # rather than raising, so a batch dominated by fov=1944 costs nothing.
    far_pred, excluded = bank.forward(pred, fov)
    far_true, _ = bank.forward(target, fov)

    # Keep only rows that can actually be propagated when any of them can;
    # otherwise keep everything and fall back to the phase-only columns.
    # `forward` always returns tensors (unpropagatable rows are zero-filled and
    # flagged in `excluded`), so `have_far` -- not a None sentinel -- decides
    # whether the far-field columns are meaningful.
    n_excluded = int(excluded.sum())
    propagatable = [i for i in range(n_candidates) if not bool(excluded[i])]
    have_far = bool(propagatable)
    if have_far:
        indices = propagatable[:limit]
    else:
        indices = list(range(min(n_candidates, limit)))

    rows: list[dict[str, np.ndarray]] = []
    for i in indices:
        img = image[i, 0].detach().cpu().numpy()
        true_cos = target[i, 0].detach().cpu().numpy()
        true_sin = target[i, 1].detach().cpu().numpy()
        pred_cos = pred[i, 0].detach().cpu().numpy()
        pred_sin = pred[i, 1].detach().cpu().numpy()

        row: dict[str, np.ndarray] = {
            "image": img,
            "true_phase": _angle(true_cos, true_sin),
            "pred_phase": _angle(pred_cos, pred_sin),
            "phasor_err": np.hypot(pred_cos - true_cos, pred_sin - true_sin),
        }
        if have_far:
            true_far = far_true[i, 0].detach().cpu().numpy()
            pred_far = far_pred[i, 0].detach().cpu().numpy()
            row["far_true"] = true_far
            row["far_pred"] = pred_far
            row["far_diff"] = pred_far - true_far
        rows.append(row)

    return ComparisonFigure(
        rows=rows,
        panel_keys=PANEL_KEYS_FULL if have_far else PANEL_KEYS_PHASE_ONLY,
        n_candidates=n_candidates,
        n_physics_skipped=n_excluded,
    )


def _cmap_for(key: str) -> str:
    """Colormap for one panel key.

    Residual panels are checked **first**: a key such as ``phase_diff`` contains
    both markers, and matching ``phase`` first would paint a signed residual with
    a cyclic map that cannot represent sign.
    """
    if "diff" in key or "err" in key:
        return "RdBu_r" if "diff" in key else "magma"
    if "phase" in key:
        return "twilight_shifted"
    return "magma"


def _residual_bound(data: np.ndarray) -> float:
    """Symmetric colour limits for a signed residual panel."""
    return float(np.max(np.abs(data))) or 1.0


def render_comparison(
    figure: ComparisonFigure,
    path: Path | str,
    caption: str = "",
) -> Path:
    """Render a :class:`ComparisonFigure` to a PNG.

    The panel order is taken from ``figure.panel_keys``, so the on-disk image and
    the W&B upload are guaranteed to show the same columns in the same order.

    Args:
        figure: Rows and column order from :func:`build_comparison`.
        path: Destination PNG.
        caption: Suptitle, typically carrying the epoch and val metrics.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    keys = list(figure.panel_keys)
    rows = figure.rows[: max(0, len(figure.rows))]
    n = max(len(rows), 1)

    fig, axes = plt.subplots(
        n, len(keys), figsize=(3.0 * len(keys), 3.0 * n), squeeze=False
    )
    if not rows:
        axes[0][0].text(
            0.5, 0.5, "no comparison samples", ha="center", va="center"
        )

    for r, row in enumerate(rows):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            data = np.asarray(row[key], dtype=np.float64)
            if "diff" in key:
                bound = _residual_bound(data)
                ax.imshow(data, cmap=_cmap_for(key), vmin=-bound, vmax=bound)
            elif key == "phasor_err":
                # Bounded by construction: |unit - unit| <= 2.
                ax.imshow(data, cmap=_cmap_for(key), vmin=0.0, vmax=2.0)
            else:
                ax.imshow(data, cmap=_cmap_for(key))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(key, fontsize=11)

    fig.suptitle(caption)
    fig.tight_layout()
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    return destination