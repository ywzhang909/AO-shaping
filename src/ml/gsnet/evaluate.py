"""Evaluation metrics for the FourierGSNet beam-shaping model.

All metrics are computed in *simulation* — the predicted phase is propagated
with the same FFT physics the model was trained with, and the resulting
far-field intensity is compared against the requested target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from loguru import logger

from ml.gsnet.losses import ShapingLosses
from ml.gsnet.model import FourierGSNet

# Backward-compatible module-level alias: the body now lives in the single
# ``ShapingLosses`` namespace, so ``from ml.gsnet.evaluate import phase_mae``
# keeps working unchanged.
phase_mae = ShapingLosses.phase_mae


@dataclass
class SampleMetrics:
    """Per-sample shaping metrics (simulation-side).

    Attributes:
        phase_mae: Circular MAE of the predicted phase vs GS ground truth (rad).
        far_correlation: Pearson correlation between reconstructed far-field
            intensity and the requested target.
        far_rmse: RMSE between sum-normalized far-field and target.
        uniformity_cv: Coefficient of variation of far-field intensity *inside
            the target support* (lower is more uniform).
        encircled_energy: Fraction of far-field power inside the target mask.
    """

    phase_mae: float
    far_correlation: float
    far_rmse: float
    uniformity_cv: float
    encircled_energy: float


def compute_sample_metrics(
    pred_phase: torch.Tensor,
    gt_phase: torch.Tensor,
    source_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
) -> SampleMetrics:
    """Compute all simulation-side metrics for a single sample prediction.

    Args:
        pred_phase: Predicted phase ``(1, H, W)`` or ``(H, W)`` radians.
        gt_phase: GS ground-truth phase (same shape).
        source_intensity: Source intensity ``(1, H, W)`` or ``(H, W)``.
        target_intensity: Target intensity (same shape as source).

    Returns:
        A :class:`SampleMetrics` instance.
    """
    pred_phase = pred_phase.detach().float().cpu()
    if pred_phase.ndim == 2:
        pred_phase = pred_phase.unsqueeze(0)
    gt_phase = gt_phase.detach().float().cpu()
    source_intensity = source_intensity.detach().float().cpu()
    target_intensity = target_intensity.detach().float().cpu()

    # Reconstruct the far field with the same physics as the model.
    source_amp = torch.sqrt(source_intensity.clamp_min(0.0) + 1e-12)
    field = source_amp * torch.exp(1j * pred_phase)
    far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    intensity = torch.abs(far) ** 2

    target_support = target_intensity > 0
    pred_n = intensity / (intensity.sum() + 1e-12)
    tgt_n = target_intensity / (target_intensity.sum() + 1e-12)

    # Metrics.
    corr = np.corrcoef(
        pred_n.flatten().numpy(), tgt_n.flatten().numpy()
    )[0, 1]
    rmse = float(torch.sqrt(((pred_n - tgt_n) ** 2).mean()))
    inside = intensity[target_support]
    cv = float((inside.std() / (inside.mean() + 1e-12)).cpu()) if inside.numel() else float("nan")
    ee = float((intensity * target_support).sum() / (intensity.sum() + 1e-12))
    pmae = phase_mae(pred_phase, gt_phase)

    return SampleMetrics(
        phase_mae=pmae,
        far_correlation=float(corr),
        far_rmse=rmse,
        uniformity_cv=cv,
        encircled_energy=ee,
    )


@dataclass
class EvalSummary:
    """Aggregated evaluation summary over a dataset.

    Attributes:
        n_samples: Number of evaluated samples.
        means: Mean of every :class:`SampleMetrics` field.
        full: Per-sample metric lists.
    """

    n_samples: int
    means: dict[str, float]
    full: list[SampleMetrics]

    def __str__(self) -> str:
        return " | ".join(
            f"{k}={v:.4f}" for k, v in self.means.items()
        )


def evaluate_model(
    model: FourierGSNet,
    dataloader: DataLoader,
    device: str = "cpu",
) -> EvalSummary:
    """Evaluate a trained FourierGSNet on a dataset (single forward pass each).

    Args:
        model: Trained (or untrained) FourierGSNet.
        dataloader: DataLoader yielding ``(source_intensity, target_intensity,
            gt_phase)`` tuples.
        device: Compute device.

    Returns:
        Aggregated :class:`EvalSummary`.
    """
    model.to(device)
    model.eval()

    full: list[SampleMetrics] = []
    logger.info("Evaluating FourierGSNet on {} samples", len(dataloader.dataset))

    with torch.no_grad():
        for source, target, gt_phase in dataloader:
            source = source.to(device)
            target = target.to(device)
            gt_phase = gt_phase.to(device)

            pred_phase = model(source, target)
            for b in range(source.shape[0]):
                full.append(
                    compute_sample_metrics(
                        pred_phase[b], gt_phase[b], source[b], target[b]
                    )
                )

    fields = [
        "phase_mae", "far_correlation", "far_rmse", "uniformity_cv", "encircled_energy",
    ]
    means: dict[str, float] = {}
    for f in fields:
        values = [getattr(m, f) for m in full if not math.isnan(getattr(m, f))]
        means[f] = float(np.mean(values)) if values else float("nan")

    summary = EvalSummary(n_samples=len(full), means=means, full=full)
    logger.info("Evaluation: {}", summary)
    return summary


def divergence_metric(
    model: FourierGSNet,
    source_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Gradient of the intensity loss w.r.t. the predicted phase.

    Useful for diagnosing whether a trained model has reached a local optimum
    of the physics-consistent objective: a converged model should have a small
    gradient magnitude since small phase perturbations leave the intensity
    basically unchanged.

    Args:
        model: The FourierGSNet.
        source_intensity: Source intensity ``(1, 1, H, W)``.
        target_intensity: Target intensity ``(1, 1, H, W)``.
        device: Compute device.

    Returns:
        Mean absolute gradient of intensity-MSE w.r.t. input phase.
    """
    model.to(device)
    model.eval()
    source = source_intensity.to(device)
    target = target_intensity.to(device)

    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    pred_phase = model(source, target)
    pred_phase.requires_grad_(True)

    loss = ShapingLosses.intensity_mse(source_amp, pred_phase, target)

    grad = torch.autograd.grad(loss, pred_phase)[0]
    return grad.abs().mean().detach()