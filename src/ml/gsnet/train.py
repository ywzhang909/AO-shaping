"""Training loop for the FourierGSNet beam-shaping model.

Loss = circular-MSE(predicted_phase, GS ground-truth phase)
     + w_intensity * MSE(normalized predicted far-field intensity, target)

The intensity term is a *physics-consistent auxiliary loss*: it encourages the
predicted phase to produce the requested far-field even when the GS ground truth
is only one of many valid phase solutions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from loguru import logger

from ml.gsnet.losses import ShapingLosses
from ml.gsnet.model import FourierGSNet

# Backward-compatible module-level aliases: the loss bodies now live in the
# single ``ShapingLosses`` namespace, so ``from ml.gsnet.train import
# shaping_loss`` (and friends) keeps working unchanged.
circular_mse = ShapingLosses.circular_mse
intensity_mse = ShapingLosses.intensity_mse
shaping_loss = ShapingLosses.shaping_loss


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