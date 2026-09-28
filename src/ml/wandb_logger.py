"""Wandb logger for phase prediction training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize


def init_wandb(
    project: str = "ao-shaping-phase",
    name: str | None = None,
    config: dict | None = None,
    entity: str | None = None,
    mode: Literal["online", "offline", "disabled", "shared"] | None = None,
) -> wandb.Run:
    """Initialize wandb run.

    Args:
        project: Project name.
        run_name: Run name.
        config: Config dict.
        entity: Entity/team name.
        mode: W&B run mode, e.g. ``"offline"``, ``"online"`` or ``"disabled"``.
            ``None`` (default) leaves W&B's own default in place. Added for the
            offline FourierGSNet training CLI, which defaults to ``"offline"`` so
            a run works with no credentials; a live run needs ``wandb login`` and
            ``mode="online"``.

    Returns:
        wandb.Run
    """
    return wandb.init(
        project=project, name=name, config=config, entity=entity, mode=mode
    )


def log_phase_comparison(
    true_phase: torch.Tensor,
    pred_phase: torch.Tensor,
    step: int,
    title: str = "Phase Comparison",
) -> wandb.Image:
    """Log phase comparison image.

    Args:
        true_phase: Ground truth phase (1, H, W) or (H, W)
        pred_phase: Predicted phase (1, H, W) or (H, W)
        step: Current step.
        title: Title for the image.

    Returns:
        wandb.Image
    """
    # Squeeze to 2D
    if true_phase.ndim == 3:
        true_phase = true_phase.squeeze(0)
    if pred_phase.ndim == 3:
        pred_phase = pred_phase.squeeze(0)

    # Move to CPU and numpy
    true_np = true_phase.detach().cpu().numpy()
    pred_np = pred_phase.detach().cpu().numpy()

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common color normalization
    vmin = min(true_np.min(), pred_np.min())
    vmax = max(true_np.max(), pred_np.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    # True phase
    im0 = axes[0].imshow(true_np, cmap="viridis", norm=norm)
    axes[0].set_title("True Phase")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Predicted phase
    im1 = axes[1].imshow(pred_np, cmap="viridis", norm=norm)
    axes[1].set_title("Predicted Phase")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Difference
    diff = pred_np - true_np
    im2 = axes[2].imshow(diff, cmap="RdBu", vmin=-0.2, vmax=0.2)
    axes[2].set_title("Difference (Pred - True)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[0].set_title(f"True Phase (MAE: {np.abs(diff).mean():.4f})")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    # Convert to wandb
    image = wandb.Image(fig)
    plt.close(fig)

    return image


def log_phase_grid(
    true_phases: list[torch.Tensor],
    pred_phases: list[torch.Tensor],
    step: int,
    max_samples: int = 4,
) -> wandb.Image:
    """Log grid of phase comparisons.

    Args:
        true_phases: List of ground truth phases.
        pred_phases: List of predicted phases.
        step: Current step.
        max_samples: Maximum samples to show.

    Returns:
        wandb.Image
    """
    import matplotlib.pyplot as plt

    n = min(len(true_phases), max_samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        true = true_phases[i]
        pred = pred_phases[i]

        if true.ndim == 3:
            true = true.squeeze(0)
        if pred.ndim == 3:
            pred = pred.squeeze(0)

        true_np = true.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        diff = pred_np - true_np

        # Common norm
        vmin = min(true_np.min(), pred_np.min())
        vmax = max(true_np.max(), pred_np.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        axes[i, 0].imshow(true_np, cmap="viridis", norm=norm)
        axes[i, 0].set_title(f"True #{i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_np, cmap="viridis", norm=norm)
        axes[i, 1].set_title(f"Pred #{i}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff, cmap="RdBu", vmin=-0.2, vmax=0.2)
        axes[i, 2].set_title(f"Diff (MAE: {np.abs(diff).mean():.4f})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)

    return image


def _shaping_panel_cmap(key: str) -> str:
    """Pick a colormap for one shaping-comparison panel.

    Phase panels use a cyclic map (phase is a circle, not a scalar) and any
    residual panel uses a diverging map centred on zero.

    Args:
        key: Panel key, e.g. ``"pred_phase"`` or ``"far_diff"``.

    Returns:
        A matplotlib colormap name.
    """
    if "phase" in key:
        return "twilight_shifted"
    if "diff" in key:
        return "RdBu_r"
    return "magma"


def log_shaping_comparison(
    rows: Sequence[Mapping[str, np.ndarray]],
    panel_keys: Sequence[str],
    *,
    max_samples: int = 6,
    title: str = "FourierGSNet shaping comparison",
) -> wandb.Image:
    """Log a predicted-vs-ground-truth shaping montage.

    One row per sample, one column per key in ``panel_keys``. The caller owns the
    panel order, so the W&B image and the on-disk PNG are guaranteed to show the
    same columns in the same order without this module having to know the
    runner's key names.

    Args:
        rows: Per-sample ``{panel_key: (H, W) ndarray}`` mappings.
        panel_keys: Column order; every key must be present in every row.
        max_samples: Cap on the number of rows rendered.
        title: Suptitle of the figure.

    Returns:
        A ``wandb.Image`` wrapping the rendered figure. The figure is closed
        before returning, so the caller never leaks a pyplot figure.
    """
    keys = list(panel_keys)
    if not keys:
        raise ValueError("panel_keys must not be empty")

    selected = [dict(row) for row in rows[: max(0, int(max_samples))]]
    n = max(len(selected), 1)
    fig, axes = plt.subplots(n, len(keys), figsize=(3.0 * len(keys), 3.0 * n), squeeze=False)

    if not selected:
        axes[0][0].text(0.5, 0.5, "no comparison samples", ha="center", va="center")

    for r, row in enumerate(selected):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            data = np.asarray(row[key], dtype=np.float64)
            cmap = _shaping_panel_cmap(key)
            if "diff" in key:
                # Symmetric limits keep zero in the middle of the diverging map.
                bound = float(np.max(np.abs(data))) or 1.0
                ax.imshow(data, cmap=cmap, vmin=-bound, vmax=bound)
            else:
                ax.imshow(data, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(key, fontsize=11)

    fig.suptitle(title)
    fig.tight_layout()

    image = wandb.Image(fig)
    plt.close(fig)
    return image


class WandbLogger:
    """Wandb logger for training."""

    def __init__(
        self,
        project: str = "ao-shaping-phase",
        name: str | None = None,
        config: dict | None = None,
        entity: str | None = None,
        log_frequency: int = 100,
    ):
        self.project = project
        self.name = name
        self.config = config
        self.entity = entity
        self.log_frequency = log_frequency
        self.run = None

    def init(self) -> None:
        """Initialize wandb."""
        self.run = init_wandb(
            project=self.project,
            name=self.name,
            config=self.config,
            entity=self.entity,
        )

    def log_metrics(
        self,
        metrics: dict,
        step: int,
    ) -> None:
        """Log metrics."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_phase_comparisons(
        self,
        true_phases: torch.Tensor,
        pred_phases: torch.Tensor,
        step: int,
    ) -> None:
        """Log phase comparison images.

        Args:
            true_phases: Batch of true phases (B, 1, H, W)
            pred_phases: Batch of predicted phases (B, 1, H, W)
            step: Current step
        """
        if not self.run:
            return

        # Log first sample in detail
        true_single = true_phases[0:1]
        pred_single = pred_phases[0:1]
        image = log_phase_comparison(true_single, pred_single, step, "Phase Comparison")
        self.run.log({"phase_comparison": image}, step=step)

    def finish(self) -> None:
        """Finish wandb run."""
        if self.run:
            self.run.finish()
