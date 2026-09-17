"""Training loop for the FourierGSNet beam-shaping model.

Loss = circular-MSE(predicted_phase, GS ground-truth phase)
     + w_intensity * MSE(normalized predicted far-field intensity, target)

The intensity term is a *physics-consistent auxiliary loss*: it encourages the
predicted phase to produce the requested far-field even when the GS ground truth
is only one of many valid phase solutions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from loguru import logger

from ml.gsnet.model import FourierGSNet


def circular_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared shortest angular distance between two phase maps."""
    diff = torch.remainder(pred - target + math.pi, 2 * math.pi) - math.pi
    return torch.mean(diff**2)


def intensity_mse(
    source_amp: torch.Tensor,
    phase: torch.Tensor,
    target_intensity: torch.Tensor,
) -> torch.Tensor:
    """MSE between normalized far-field intensity and normalized target.

    Applies the same physics as the model (FFT + fftshift), normalizes both
    distributions to unit sum, and returns the per-image mean squared
    difference. Scale-invariant so absolute beam power does not matter.

    Note: once the physics unrolling already matches the target (far_rmse
    ~1e-3), this term saturates near zero and carries no training signal —
    use :func:`shaping_loss` as the shaping objective instead.
    """
    field = source_amp * torch.exp(1j * phase)
    far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    intensity = torch.abs(far) ** 2
    pred_n = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    tgt_n = target_intensity / (
        target_intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12
    )
    return torch.mean((pred_n - tgt_n) ** 2)


def shaping_loss(
    source_amp: torch.Tensor,
    phase: torch.Tensor,
    target_intensity: torch.Tensor,
) -> torch.Tensor:
    """1 - mean Pearson correlation between predicted and target far-field.

    Unlike :func:`intensity_mse` — which saturates near zero the moment the
    physics amplitude exchange already matches the target — returns a loss on
    a O(1) scale (0 = perfect correlation) that stays differentiable well past
    the MSE saturation point. Scale-invariant: the predicted intensity has
    arbitrary absolute scale, so both distributions are centered and
    normalized per image before correlation.
    """
    field = source_amp * torch.exp(1j * phase)
    far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    intensity = torch.abs(far) ** 2
    pred = intensity.flatten(1)
    tgt = target_intensity.flatten(1)
    p = pred - pred.mean(dim=1, keepdim=True)
    t = tgt - tgt.mean(dim=1, keepdim=True)
    denom = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
    corr = (p * t).sum(dim=1) / denom
    return torch.mean(1.0 - corr)


@dataclass
class TrainResult:
    """Result of a FourierGSNet training run.

    Attributes:
        history: Per-epoch ``{"loss", "phase_loss", "intensity_loss"}`` dicts.
        best_epoch: Epoch index with the lowest total loss.
        best_loss: Lowest total loss achieved.
        checkpoint_path: Path of the saved best-state checkpoint (or None).
        seconds: Wall-clock training time.
    """

    history: list[dict[str, float]]
    best_epoch: int
    best_loss: float
    checkpoint_path: Path | None = None
    seconds: float = 0.0


def train_gsnet(
    model: FourierGSNet,
    dataloader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    w_phase: float = 1.0,
    w_shaping: float = 40.0,
    device: str = "cpu",
    checkpoint_dir: str | Path | None = None,
    seed: int = 0,
) -> TrainResult:
    """Train a FourierGSNet model on the synthetic beam-shaping dataset.

    Args:
        model: The FourierGSNet to train (in-place).
        dataloader: DataLoader of ``(source_intensity, target_intensity,
            gt_phase)`` tuples.
        epochs: Number of full training epochs.
        lr: Adam learning rate.
        w_phase: Weight of the GS phase-regression term. The phase loss is a
            noisy many-to-one target (GS returns one specific phase among many
            equally good ones), so it acts as a regularizer.
        w_shaping: Weight of the far-field shaping loss (1 - Pearson corr).
            This is the *primary* shaping objective; its raw scale is ~0.04 at
            the physics-unrolled baseline, ~40x smaller than the phase term,
            so it needs a higher weight to actually steer training.
        device: Compute device (``"cpu"`` or ``"cuda"``).
        checkpoint_dir: If given, save best-state checkpoint there.
        seed: RNG seed for the optimizer.

    Returns:
        A :class:`TrainResult` with loss history and best state.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    torch.manual_seed(seed)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    ckpt_path: Path | None = None
    start = time.monotonic()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_phase = 0.0
        epoch_shaping = 0.0
        epoch_inten = 0.0
        n_batches = 0

        for source, target, gt_phase in dataloader:
            source = source.to(device)
            target = target.to(device)
            gt_phase = gt_phase.to(device)

            source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)

            pred_phase = model(source, target)

            l_phase = circular_mse(pred_phase, gt_phase)
            l_shaping = shaping_loss(source_amp, pred_phase, target)
            l_inten = intensity_mse(source_amp, pred_phase, target)
            loss = w_phase * l_phase + w_shaping * l_shaping

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_phase += float(l_phase.detach())
            epoch_shaping += float(l_shaping.detach())
            epoch_inten += float(l_inten.detach())
            n_batches += 1

        scheduler.step()

        avg = {
            "loss": epoch_loss / n_batches,
            "phase_loss": epoch_phase / n_batches,
            "shaping_loss": epoch_shaping / n_batches,
            "intensity_loss": epoch_inten / n_batches,
        }
        history.append(avg)
        logger.info(
            "Epoch {}/{} loss={:.6f} phase={:.6f} shaping={:.6f} lr={:.2e}",
            epoch, epochs, avg["loss"], avg["phase_loss"], avg["shaping_loss"],
            scheduler.get_last_lr()[0],
        )

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            best_epoch = epoch
            if checkpoint_dir is not None:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "fourier_gsnet_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "loss": best_loss,
                        "config": {
                            "num_layers": model.num_layers,
                            "base_channels": model.layers[0].phase_net.down1[0].out_channels,
                        },
                    },
                    ckpt_path,
                )

    seconds = time.monotonic() - start
    logger.info(
        "Training complete: best_epoch={} best_loss={:.6f} time={:.1f}s",
        best_epoch, best_loss, seconds,
    )
    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_loss=best_loss,
        checkpoint_path=ckpt_path,
        seconds=seconds,
    )