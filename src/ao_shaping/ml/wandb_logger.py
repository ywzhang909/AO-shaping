"""Wandb logger for phase prediction training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import wandb
from matplotlib import colormaps
from matplotlib.colors import Normalize


def init_wandb(
    project: str = "ao-shaping-phase",
    name: str | None = None,
    config: dict | None = None,
    entity: str | None = None,
) -> wandb.Run:
    """Initialize wandb run.

    Args:
        project: Project name.
        run_name: Run name.
        config: Config dict.
        entity: Entity/team name.

    Returns:
        wandb.Run
    """
    return wandb.init(project=project, name=name, config=config, entity=entity)


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
