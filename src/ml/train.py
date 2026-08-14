"""Phase prediction training and inference CLI.

Usage:
    # Train phase map prediction (U-Net+GAN)
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --epochs 100 --output-mode phase

    # Train Zernike coefficient prediction
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --epochs 100 --output-mode coeffs --input-mode combined

    # Train with specific input mode
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --input-mode focus --output-mode coeffs

    # Run wandb sweep
    uv run python src/ao_shaping/ml/train.py sweep --data-dir data/slm_capture --output-mode coeffs

    # Inference
    uv run python src/ao_shaping/ml/train.py predict --checkpoint checkpoints/best.pt --data-dir data/slm_capture

Note:
    - UNet: phase map output (requires --target-phase-dir for training data)
    - Zernike models (resnet18/resnet34/simple_cnn): coefficient output
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader

# Set matplotlib backend to Agg to avoid threading issues
matplotlib.use("Agg")

from ml.phase.dataset import (
    PhasePredictionDataset,
    create_dataloaders,
)
from ml.zernike.dataset import (
    create_zernike_loaders,
)
from ml.zernike.models import (
    build_model,
)
from ml.phase import build_unet, build_discriminator
from ml.phase.trainer import PhaseGANTrainer


# =============================================================================
# Model Type Definitions
# =============================================================================

# Model types that output Zernike coefficients (regression)
ZERNIKE_MODEL_TYPES = {"resnet18", "resnet34", "simple_cnn"}

# Model types that output phase maps (image-to-image)
PHASE_MODEL_TYPES = {"unet"}

# All supported model types
ALL_MODEL_TYPES = ZERNIKE_MODEL_TYPES | PHASE_MODEL_TYPES


# =============================================================================
# WandB Sweep Configuration (Universal - supports ALL model types)
# =============================================================================

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"values": [1e-4, 5e-4, 1e-3, 5e-3]},
        "model_type": {"values": ["resnet18", "resnet34", "simple_cnn", "unet"]},
        # Zernike-specific parameters (ignored for unet)
        "input_mode": {"values": ["focus", "pupil", "combined"]},
        "output_mode": {"values": ["coeffs", "phase"]},
        "n_zernike_terms": {"values": [28, 55]},
        # UNet-specific parameters (ignored for zernike models)
        "lambda_l1": {"values": [50.0, 100.0, 200.0]},
        "gan_mode": {"values": ["lsgan", "vanilla"]},
    },
}


@click.group()
def cli():
    """Phase prediction training and inference CLI."""
    pass


# =============================================================================
# Unified Training Function (supports ALL model types)
# =============================================================================


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    # Model-type specific args
    model_type: str = "resnet18",
    is_phase_model: bool = False,
    lambda_l1: float = 100.0,
    lambda_adv: float = 1.0,
    gan_mode: str = "lsgan",
    loss_type: str = "angular",
    checkpoint_dir: str = "checkpoints",
    use_wandb: bool = False,
    wandb_project: str = "ao-shaping",
    wandb_name: str | None = None,
    log_images: bool = True,
    resume: str | None = None,
) -> dict:
    """Unified training function for both UNet (phase) and Zernike (coefficient) models.
    
    Args:
        model: The model to train (UNetGenerator or regression model)
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional, for zernike models)
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        model_type: Model type identifier
        is_phase_model: Whether this is a UNet phase prediction model (with GAN training)
        lambda_l1: L1 loss weight (only for phase models)
        lambda_adv: Adversarial loss weight (only for phase models)
        gan_mode: GAN mode - "vanilla" or "lsgan" (only for phase models)
        loss_type: Loss type - "l1" or "angular" (only for phase models)
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use wandb logging
        wandb_project: wandb project name
        wandb_name: wandb run name
        log_images: Whether to log images to wandb
        resume: Checkpoint path to resume from
        
    Returns:
        Training history dict
    """
    # Build optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Initialize wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=wandb_project, name=wandb_name)
        except ImportError:
            logger.warning("wandb not installed, skipping")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if is_phase_model:
            # GAN model checkpoint
            model.load_state_dict(ckpt["generator"])
        else:
            model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # Create criterion
    if is_phase_model:
        # For phase prediction, use PhaseGANTrainer
        from ml.phase.trainer import PhaseGANTrainer

        # Need to build discriminator for GAN
        in_channels = model.in_channels if hasattr(model, "in_channels") else 2
        generator = build_unet(in_channels=in_channels, device=device)
        discriminator = build_discriminator(in_channels=in_channels + 1, device=device)

        trainer = PhaseGANTrainer(
            generator=model,
            discriminator=discriminator,
            device=device,
            lr_gen=lr,
            lr_disc=lr,
            lambda_l1=lambda_l1,
            lambda_adv=lambda_adv,
            gan_mode=gan_mode,
            checkpoint_dir=checkpoint_dir,
            loss_type=loss_type,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_name,
            log_images=log_images,
        )

        # GAN training
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            save_every=10,
        )

        if wandb_run:
            wandb_run.finish()

        return history
    else:
        # Regression training (Zernike coefficients)
        criterion = nn.MSELoss()
        history = {"train_loss": [], "val_loss": [], "test_loss": []}

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset) if val_loader else 0
        n_test = len(test_loader.dataset) if test_loader else 0

        for epoch in range(start_epoch, epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_processed = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device)
                targets = batch["coefficients"].to(device)

                optimizer.zero_grad()
                pred = model(images)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                n_processed += images.size(0)

                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Step {batch_idx + 1}/{len(train_loader)}, batch_loss={loss.item():.6f}")

            train_loss /= n_train

            # Validate
            model.eval()
            val_loss = 0.0
            if val_loader:
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        targets = batch["coefficients"].to(device)
                        pred = model(images)
                        loss = criterion(pred, targets)
                        val_loss += loss.item() * images.size(0)

                val_loss /= n_val

            # Update scheduler
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            lr_info = f", lr={new_lr:.2e}" if new_lr != old_lr else ""
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}{lr_info}"
            )

            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": new_lr,
                })

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"  -> New best model! val_loss={val_loss:.6f}")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                }, Path(checkpoint_dir) / "best.pt")

        # Final test evaluation
        if test_loader and len(test_loader.dataset) > 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["image"].to(device)
                    targets = batch["coefficients"].to(device)
                    pred = model(images)
                    test_loss += criterion(pred, targets).item() * images.size(0)

            test_loss /= n_test
            history["test_loss"] = test_loss
            logger.info(f"Test loss: {test_loss:.6f}")

        # Save final model
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epochs,
            "val_loss": best_val_loss,
            "test_loss": history.get("test_loss", None),
        }, Path(checkpoint_dir) / "final.pt")

        if wandb_run:
            wandb_run.finish()

        logger.info(f"Done! Best val={best_val_loss:.6f}")
        return history


@cli.command("train")
@click.option("--data-dir", required=True, help="数据目录")
@click.option("--output-dir", default="checkpoints", help="模型保存目录")
@click.option(
    "--model-type",
    "model_type",
    default="resnet18",
    type=click.Choice(["unet", "resnet18", "resnet34", "simple_cnn"]),
    help="模型类型: unet=相位图, 其他=Zernike系数",
)
@click.option("--epochs", default=100, help="训练轮数")
@click.option("--batch-size", default=8, help="批次大小")
@click.option("--lr", default=1e-3, help="学习率")
@click.option("--weight-decay", default=1e-4, help="权重衰减")
@click.option("--train-split", default=0.7, help="训练集比例")
@click.option("--val-split", default=0.15, help="验证集比例")
@click.option("--num-workers", default=0, help="DataLoader工作进程")
@click.option("--device", default=None, help="设备 (auto)")
@click.option("--seed", default=42, help="随机种子")
@click.option("--target-size", default="256x256", help="目标尺寸 HxW")
@click.option(
    "--input-mode",
    "input_mode",
    type=click.Choice(["focus", "pupil", "combined"]),
    default="combined",
    help="输入模式: focus(Daheng远场), pupil(MiiCam近场), combined(二者结合)",
)
@click.option(
    "--output-mode",
    type=click.Choice(["phase", "coeffs"]),
    default="phase",
    help="输出模式: phase(二维相位图), coeffs(Zernike系数)",
)
@click.option("--n-zernike-terms", "n_zernike_terms", default=55, help="Zernike项数 (coeffs模式)")
@click.option("--n-max", "n_max", default=10, help="Zernike径向阶 (UNet)")
@click.option("--lambda-l1", default=100.0, help="L1损失权重 (UNet)")
@click.option("--lambda-adv", default=1.0, help="对抗损失权重 (UNet)")
@click.option(
    "--gan-mode",
    type=click.Choice(["vanilla", "lsgan"]),
    default="lsgan",
    help="GAN模式 (UNet)",
)
@click.option(
    "--loss-type",
    type=click.Choice(["l1", "angular"]),
    default="angular",
    help="损失类型 (UNet)",
)
@click.option("--use-daheng/--no-daheng", default=True, help="使用Daheng相机数据")
@click.option("--use-miicam/--no-miicam", default=True, help="使用MiiCam相机数据")
@click.option("--log-images/--no-log-images", default=True, help="记录图像到wandb")
@click.option("--use-wandb/--no-wandb", default=False, help="使用wandb")
@click.option("--wandb-project", default="ao-shaping", help="项目名")
@click.option("--wandb-name", default=None, help="run名")
@click.option("--resume", default=None, help="恢复训练")
def train(
    data_dir: str,
    output_dir: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_split: float,
    val_split: float,
    num_workers: int,
    input_mode: str,
    output_mode: str,
    n_zernike_terms: int,
    use_daheng: bool,
    use_miicam: bool,
    device: str | None,
    seed: int,
    target_size: str,
    n_max: int,
    lambda_l1: float,
    lambda_adv: float,
    gan_mode: str,
    loss_type: str,
    log_images: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    resume: str | None,
):
    """Train model for phase prediction (phase map or Zernike coefficients)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Input mode: {input_mode}, Output mode: {output_mode}")

    # Parse target size
    if target_size:
        h, w = map(int, target_size.split("x"))
        target_size_tuple = (h, w)
    else:
        target_size_tuple = (512, 512) if output_mode == "phase" else (256, 256)

    # Validate input_mode with legacy camera flags
    if input_mode == "focus" and not use_daheng:
        raise click.BadParameter("input_mode='focus' requires --use-daheng")
    if input_mode == "pupil" and not use_miicam:
        raise click.BadParameter("input_mode='pupil' requires --use-miicam")
    if input_mode == "combined" and (not use_daheng or not use_miicam):
        raise click.BadParameter(
            "input_mode='combined' requires both --use-daheng and --use-miicam"
        )

    # Determine input channels
    in_channels = 2 if input_mode == "combined" else 1
    logger.info(f"Input channels: {in_channels} (input_mode={input_mode})")

    if output_mode == "coeffs":
        # Zernike coefficient prediction training
        _train_coeffs_mode(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            train_split=train_split,
            val_split=val_split,
            num_workers=num_workers,
            input_mode=input_mode,
            model_type=model_type,
            n_zernike_terms=n_zernike_terms,
            target_size=target_size_tuple,
            device=device,
            seed=seed,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            resume=resume,
        )
    else:
        # Phase map prediction training (legacy U-Net+GAN)
        _train_phase_mode(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_l1=lambda_l1,
            lambda_adv=lambda_adv,
            target_size=target_size_tuple,
            train_split=train_split,
            num_workers=num_workers,
            use_daheng=use_daheng,
            use_miicam=use_miicam,
            device=device,
            seed=seed,
            gan_mode=gan_mode,
            loss_type=loss_type,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            log_images=log_images,
            resume=resume,
        )


def _train_phase_mode(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_l1: float,
    lambda_adv: float,
    target_size: tuple[int, int],
    train_split: float,
    num_workers: int,
    use_daheng: bool,
    use_miicam: bool,
    device: torch.device,
    seed: int,
    gan_mode: str,
    loss_type: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    log_images: bool,
    resume: str | None,
):
    """Train U-Net+GAN for phase map prediction."""
    # Determine input channels
    in_channels = int(use_daheng) + int(use_miicam)
    if in_channels == 0:
        raise click.BadParameter("At least one camera must be enabled.")
    logger.info(
        f"Input channels: {in_channels} (daheng={use_daheng}, miicam={use_miicam})"
    )

    # Create dataloaders
    logger.info(f"Loading data from {data_dir}")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=train_split,
        target_size=target_size,
        num_workers=num_workers,
        use_daheng=use_daheng,
        use_miicam=use_miicam,
        seed=seed,
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    # Build models
    generator, discriminator = build_gan_models(
        in_channels=in_channels,
        n_coeffs=n_zernike_terms,
        device=device,
    )
    logger.info(f"Model: {model_type}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Save config
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "input_mode": input_mode,
        "n_zernike_terms": n_zernike_terms,
        "n_max": n_max,
        "target_size": target_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_split": train_split,
        "val_split": val_split,
        "seed": seed,
    }
    if is_phase_model:
        config.update({
            "lambda_l1": lambda_l1,
            "lambda_adv": lambda_adv,
            "gan_mode": gan_mode,
            "loss_type": loss_type,
        })
    with (Path(output_dir) / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Train using unified training function
    logger.info(f"Starting {model_type} training for {epochs} epochs...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        model_type=model_type,
        is_phase_model=is_phase_model,
        lambda_l1=lambda_l1,
        lambda_adv=lambda_adv,
        gan_mode=gan_mode,
        loss_type=loss_type,
        checkpoint_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        resume=resume,
    )

    # Resume if requested
    if resume:
        logger.info(f"Resuming from {resume}")
        trainer.load_checkpoint(resume)

    # Save config
    config = {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_l1": lambda_l1,
        "lambda_adv": lambda_adv,
        "target_size": target_size,
        "train_split": train_split,
        "in_channels": in_channels,
        "use_daheng": use_daheng,
        "use_miicam": use_miicam,
        "gan_mode": gan_mode,
        "loss_type": loss_type,
        "seed": seed,
        "output_mode": "phase",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with (Path(output_dir) / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Train
    logger.info("Starting phase map training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_every=10,
    )

    logger.info("Training complete!")
    logger.info(f"Best val loss: {min(history.get('val_loss', [float('inf')])):.4f}")


def _train_coeffs_mode(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_split: float,
    val_split: float,
    num_workers: int,
    input_mode: str,
    model_type: str,
    n_zernike_terms: int,
    target_size: tuple[int, int],
    device: torch.device,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    resume: str | None,
):
    """Train model for Zernike coefficient prediction."""
    in_channels = 2 if input_mode == "combined" else 1

    # Create dataloaders (3-way split)
    logger.info(f"Loading data from {data_dir}")
    train_loader, val_loader, test_loader = create_zernike_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        input_mode=input_mode,
        n_zernike_terms=n_zernike_terms,
        target_size=target_size,
        num_workers=num_workers,
        seed=seed,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    n_batches = len(train_loader)
    logger.info(f"Dataset: train={n_train}, val={n_val}, test={n_test} samples")
    logger.info(f"Batch size: {batch_size}, steps per epoch: {n_batches}")

    model = build_model(
        model_type,
        in_channels=in_channels,
        n_coeffs=n_zernike_terms,
        device=device,
    )
    logger.info(
        f"Model: {model_type}, params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    # WandB
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "model_type": model_type,
                    "input_mode": input_mode,
                    "n_zernike_terms": n_zernike_terms,
                    "target_size": target_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "train_split": train_split,
                    "val_split": val_split,
                    "seed": seed,
                    "output_mode": "coeffs",
                },
            )
        except ImportError:
            logger.warning("wandb not installed, skipping")

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # Save config
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "input_mode": input_mode,
        "n_zernike_terms": n_zernike_terms,
        "target_size": target_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_split": train_split,
        "val_split": val_split,
        "seed": seed,
        "output_mode": "coeffs",
    }
    with (Path(output_dir) / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    logger.info(f"Starting Zernike coefficient training for {epochs} epochs...")
    history = {"train_loss": [], "val_loss": [], "test_loss": []}

    val_loss = np.inf
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_processed = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            coeffs = batch["coefficients"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, coeffs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            n_processed += images.size(0)

            # Log every 50 steps
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Step {batch_idx + 1}/{n_batches}, batch_loss={loss.item():.6f}"
                )

        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_processed = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                coeffs = batch["coefficients"].to(device)
                pred = model(images)
                loss = criterion(pred, coeffs)
                val_loss += loss.item() * images.size(0)
                n_val_processed += images.size(0)

        val_loss /= n_val

        # Update scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Log epoch
        lr_info = f", lr={new_lr:.2e}" if new_lr != old_lr else ""
        logger.info(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}{lr_info}"
        )

        # Log test images to wandb every 10 epochs or on best
        if wandb_run and (epoch + 1) % 10 == 0:
            # Log a random sample from test set
            test_sample_idx = np.random.randint(0, len(test_loader.dataset))
            test_sample = test_loader.dataset[test_sample_idx]

            # Get prediction
            model.eval()
            with torch.no_grad():
                test_img = test_sample["image"].unsqueeze(0).to(device)
                test_coeff = test_sample["coefficients"]
                pred_coeff = model(test_img).cpu().squeeze(0).numpy()

            # Generate phase maps
            from ml.phase.dataset import coefficients_to_phase_map

            target_phase = coefficients_to_phase_map(
                test_coeff.numpy(), size=target_size
            )
            pred_phase = coefficients_to_phase_map(pred_coeff, size=target_size)

            # Log image grid
            wandb_run.log(
                {
                    "test_sample": wandb.Image(
                        np.concatenate([target_phase, pred_phase], axis=1),
                        caption=f"Epoch {epoch + 1}: target vs prediction",
                    )
                }
            )

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": new_lr,
                }
            )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  -> New best model! val_loss={val_loss:.6f}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "config": config,
                },
                Path(output_dir) / "best.pt",
            )
            logger.info(f"Saved best (val={val_loss:.6f})")

    # Final test
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            coeffs = batch["coefficients"].to(device)
            pred = model(images)
            test_loss += criterion(pred, coeffs).item() * images.size(0)

    test_loss /= n_test
    history["test_loss"] = test_loss
    logger.info(f"Test loss: {test_loss:.6f}")

    if wandb_run:
        wandb_run.log(
            {
                "test_loss": test_loss,
                "best_val_loss": best_val_loss,
            }
        )

    # Save final with scheduler state
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epochs,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "config": config,
        },
        Path(output_dir) / "final.pt",
    )

    with (Path(output_dir) / "history.json").open("w") as f:
        json.dump(history, f)

    if wandb_run:
        wandb_run.finish()

    logger.info(f"Done! Best val={best_val_loss:.6f}, test={test_loss:.6f}")


@cli.command()
@click.option("--checkpoint", required=True, help="模型检查点路径")
@click.option("--data-dir", default=None, help="数据集目录 (批量预测)")
@click.option("--image", default=None, help="单张图像 .npy 文件路径")
@click.option(
    "--output",
    default=None,
    help="输出目录或文件 (default: 预测结果保存在输入同级目录)",
)
@click.option("--device", default=None, help="设备 (default: auto)")
@click.option("--use-daheng/--no-daheng", default=True, help="使用Daheng通道")
@click.option("--use-miicam/--no-miicam", default=True, help="使用MiiCam通道")
def predict(
    checkpoint: str,
    data_dir: str | None,
    image: str | None,
    output: str | None,
    device: str | None,
    use_daheng: bool,
    use_miicam: bool,
):
    """Predict phase from camera image(s)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load checkpoint to get model config
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Determine input channels from checkpoint or args
    in_channels = int(use_daheng) + int(use_miicam)

    # Build models
    generator, _ = build_gan_models(in_channels=in_channels, device=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    if data_dir:
        # Batch prediction on dataset
        dataset = PhasePredictionDataset(
            data_dir=data_dir,
            use_daheng=use_daheng,
            use_miicam=use_miicam,
        )
        out_dir = Path(output) if output else Path(data_dir) / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                image_tensor = sample["image"].unsqueeze(0).to(device)
                pred_phase = generator(image_tensor).cpu().squeeze(0)

                # Save prediction
                sample_name = Path(sample["path"]).name
                np.save(
                    out_dir / f"{sample_name}_predicted_phase.npy", pred_phase.numpy()
                )
                logger.info(f"Saved prediction for {sample_name}")

        logger.info(f"All predictions saved to {out_dir}")

    elif image:
        # Single image prediction
        img_data = np.load(image)
        img_tensor = torch.from_numpy(img_data.astype(np.float32))

        # Normalize to [-1, 1]
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        img_min = img_tensor.amin(dim=(1, 2), keepdim=True)
        img_max = img_tensor.amax(dim=(1, 2), keepdim=True)
        img_range = img_max - img_min
        img_range = torch.where(img_range == 0, torch.ones_like(img_range), img_range)
        img_tensor = 2.0 * (img_tensor - img_min) / img_range - 1.0

        with torch.no_grad():
            pred_phase = generator(img_tensor.unsqueeze(0).to(device)).cpu().squeeze(0)

        out_path = Path(output) if output else Path(image).with_suffix("_phase.npy")
        np.save(out_path, pred_phase.numpy())
        logger.info(f"Prediction saved to {out_path}")
    else:
        raise click.BadParameter("Provide either --data-dir or --image")




@cli.command("sweep")
@click.option("--data-dir", required=True, help="数据目录")
@click.option("--project", default="ao-shaping-sweep", help="wandb项目名")
@click.option("--batch-size", default=8, help="批次大小")
@click.option("--epochs", default=30, help="每次run轮数")
@click.option("--count", default=10, help="sweep次数")
@click.option(
    "--input-mode",
    type=click.Choice(["focus", "pupil", "combined"]),
    default=None,
    help="输入模式 (默认: 从sweep config随机选择)",
)
@click.option(
    "--output-mode",
    type=click.Choice(["coeffs", "phase"]),
    default=None,
    help="输出模式 (默认: 从sweep config随机选择)",
)
@click.option(
    "--n-zernike-terms",
    default=None,
    type=int,
    help="Zernike项数量 (默认: 从sweep config随机选择)",
)
@click.option("--target-size", default="256x256", help="图像尺寸 HxW")
@click.option("--use-wandb/--no-use-wandb", default=True, help="使用wandb记录")
def sweep(
    data_dir: str,
    project: str,
    batch_size: int,
    epochs: int,
    count: int,
    input_mode: str | None,
    output_mode: str | None,
    n_zernike_terms: int | None,
    target_size: str,
    use_wandb: bool,
):
    """Run wandb hyperparameter sweep."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        return

    # Parse target size
    h, w = map(int, target_size.split("x"))
    target_size_tuple = (h, w)

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    logger.info(f"Sweep: {sweep_id}")
    logger.info(f"Fixed batch_size: {batch_size}, epochs: {epochs}")

    def agent():
        """Universal sweep agent that handles ALL model types."""
        wandb.init()
        cfg = wandb.config

        # Use CLI overrides or sweep config values
        cfg_input_mode = input_mode if input_mode else cfg.get("input_mode", "combined")
        cfg_output_mode = (
            output_mode if output_mode else cfg.get("output_mode", "coeffs")
        )
        cfg_n_terms = (
            n_zernike_terms if n_zernike_terms else cfg.get("n_zernike_terms", 55)
        )
        cfg_model_type = cfg.get("model_type", "resnet18")
        cfg_lr = cfg.get("lr", 1e-3)

        logger.info(
            f"Sweep config: input_mode={cfg_input_mode}, output_mode={cfg_output_mode}"
        )
        logger.info(f"Model: {cfg_model_type}, lr={cfg_lr}, n_terms={cfg_n_terms}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg_output_mode == "coeffs":
            _sweep_coeffs_agent(
                data_dir=data_dir,
                batch_size=batch_size,
                epochs=epochs,
                input_mode=cfg_input_mode,
                n_zernike_terms=cfg_n_terms,
                model_type=cfg_model_type,
                lr=cfg_lr,
                target_size=target_size_tuple,
                device=device,
                use_wandb=use_wandb,
            )
        else:  # phase mode
            _sweep_phase_agent(
                data_dir=data_dir,
                batch_size=batch_size,
                epochs=epochs,
                input_mode=cfg_input_mode,
                target_size=target_size_tuple,
                device=device,
                use_wandb=use_wandb,
            )

    wandb.agent(sweep_id, agent, count=count)
    logger.info("Sweep complete!")


def _sweep_coeffs_agent(
    data_dir: str,
    batch_size: int,
    epochs: int,
    input_mode: str,
    n_zernike_terms: int,
    model_type: str,
    lr: float,
    target_size: tuple[int, int],
    device: torch.device,
    use_wandb: bool,
):
    """Sweep agent for Zernike coefficient prediction."""
    import wandb

    in_ch = 2 if input_mode == "combined" else 1
    model = build_model(
        model_type, in_channels=in_ch, n_coeffs=n_zernike_terms, device=device
    )

    train_loader, val_loader, test_loader = create_zernike_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        input_mode=input_mode,
        n_zernike_terms=n_zernike_terms,
        target_size=target_size,
        num_workers=0,
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            cs = batch["coefficients"].to(device)
            opt.zero_grad()
            loss = crit(model(imgs), cs)
            loss.backward()
            opt.step()

        model.eval()
        v_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                cs = batch["coefficients"].to(device)
                v_loss += crit(model(imgs), cs).item() * imgs.size(0)
                n_val += imgs.size(0)

        v_loss /= n_val
        best_val = min(best_val, v_loss)

        if use_wandb:
            wandb.log({"val_loss": v_loss, "epoch": epoch + 1})

        # Log test samples every 10 epochs
        if use_wandb and (epoch + 1) % 10 == 0 and len(test_loader.dataset) > 0:
            from ml.phase.dataset import coefficients_to_phase_map

            sample_idx = np.random.randint(0, len(test_loader.dataset))
            sample = test_loader.dataset[sample_idx]

            with torch.no_grad():
                test_img = sample["image"].unsqueeze(0).to(device)
                test_coeff = sample["coefficients"].numpy()
                pred_coeff = model(test_img).cpu().squeeze(0).numpy()

            target_phase = coefficients_to_phase_map(test_coeff, size=target_size)
            pred_phase = coefficients_to_phase_map(pred_coeff, size=target_size)

            # Log phase comparison
            wandb.log(
                {
                    "phase_comparison": wandb.Image(
                        np.concatenate([target_phase, pred_phase], axis=1),
                        caption=f"Epoch {epoch + 1}: target (left) vs pred (right)",
                    )
                }
            )

            # Log coefficient comparison
            x = np.arange(len(test_coeff))
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].bar(x - 0.2, test_coeff, 0.4, label="Ground Truth", alpha=0.8)
            axes[0].bar(x + 0.2, pred_coeff, 0.4, label="Prediction", alpha=0.8)
            axes[0].set_xlabel("Zernike Index")
            axes[0].set_ylabel("Coefficient Value")
            axes[0].set_title(f"Zernike Coefficients (Epoch {epoch + 1})")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            diff = pred_coeff - test_coeff
            colors = ["green" if d >= 0 else "red" for d in diff]
            axes[1].bar(x, diff, color=colors, alpha=0.7)
            axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[1].set_xlabel("Zernike Index")
            axes[1].set_ylabel("Difference")
            axes[1].set_title("Prediction Error")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            with tempfile.TemporaryDirectory() as temp_dir:
                coeff_path = Path(temp_dir) / "coeff_comparison.png"
                plt.savefig(coeff_path)
                plt.close()
                wandb.log(
                    {
                        "coeff_comparison": wandb.Image(
                            Image.open(coeff_path),
                            caption=f"Epoch {epoch + 1}: coefficients",
                        )
                    }
                )

    if use_wandb:
        wandb.log({"best_val_loss": best_val})


def _sweep_phase_agent(
    data_dir: str,
    batch_size: int,
    epochs: int,
    input_mode: str,
    target_size: tuple[int, int],
    device: torch.device,
    use_wandb: bool,
):
    """Sweep agent for phase map prediction (U-Net+GAN)."""
    import wandb

    # Determine camera usage from input_mode
    use_daheng = input_mode in ("focus", "combined")
    use_miicam = input_mode in ("pupil", "combined")
    in_channels = int(use_daheng) + int(use_miicam)

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.8,
        target_size=target_size,
        num_workers=0,
        use_daheng=use_daheng,
        use_miicam=use_miicam,
        seed=42,
    )

    generator, discriminator = build_gan_models(
        in_channels=in_channels,
        device=device,
    )

    trainer = PhaseGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr_gen=2e-4,
        lr_disc=2e-4,
        lambda_l1=100.0,
        lambda_adv=1.0,
        gan_mode="lsgan",
        checkpoint_dir=None,  # Don't save checkpoints in sweep
        loss_type="angular",
        use_wandb=False,  # Manual logging
        wandb_project="",
        wandb_run_name=None,
        log_images=False,
    )

    best_val = float("inf")
    for epoch in range(epochs):
        # Training loop
        train_metrics = {"gen_loss": 0.0, "disc_loss": 0.0}
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            phases = batch["phase"].to(device)
            metrics = trainer.train_step(images, phases)
            train_metrics["gen_loss"] += metrics["gen_loss"]
            train_metrics["disc_loss"] += metrics["disc_loss"]
            n_batches += 1

        train_metrics["gen_loss"] /= n_batches
        train_metrics["disc_loss"] /= n_batches

        # Validation
        val_metrics = trainer.validate(val_loader)
        val_loss = val_metrics.get("val_loss", val_metrics.get("l1_loss", 0.0))
        best_val = min(best_val, val_loss)

        if use_wandb:
            wandb.log(
                {
                    "val_loss": val_loss,
                    "train_gen_loss": train_metrics["gen_loss"],
                    "train_disc_loss": train_metrics["disc_loss"],
                    "epoch": epoch + 1,
                }
            )

        # Log sample images every 10 epochs
        if use_wandb and (epoch + 1) % 10 == 0:
            sample = val_loader.dataset[0]
            img = sample["image"].unsqueeze(0).to(device)
            target_phase = sample["phase"].cpu().numpy()

            with torch.no_grad():
                pred_phase = generator(img).cpu().squeeze(0).numpy()

            # Normalize for visualization
            pred_phase = pred_phase[0] if pred_phase.ndim > 2 else pred_phase
            target_phase = target_phase[0] if target_phase.ndim > 2 else target_phase

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(target_phase, cmap="gray")
            axes[0].set_title("Target Phase")
            axes[0].axis("off")
            axes[1].imshow(pred_phase, cmap="gray")
            axes[1].set_title("Predicted Phase")
            axes[1].axis("off")
            plt.tight_layout()

            with tempfile.TemporaryDirectory() as temp_dir:
                phase_path = Path(temp_dir) / "phase_comparison.png"
                plt.savefig(phase_path)
                plt.close()
                wandb.log(
                    {
                        "phase_comparison": wandb.Image(
                            Image.open(phase_path),
                            caption=f"Epoch {epoch + 1}: phase map",
                        )
                    }
                )

    if use_wandb:
        wandb.log({"best_val_loss": best_val})


if __name__ == "__main__":
    cli()
