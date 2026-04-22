"""Phase prediction training and inference CLI.

Usage:
    # Unified training - auto-detect model type
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --epochs 100 --model-type unet
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --epochs 100 --model-type resnet18

    # Run wandb sweep (supports ALL model types: unet, resnet18, resnet34, simple_cnn)
    uv run python src/ao_shaping/ml/train.py sweep --data-dir data/slm_capture/20260414_171241

    # Inference
    uv run python src/ao_shaping/ml/train.py predict --checkpoint checkpoints/best.pt --data-dir data/slm_capture

Note:
    - UNet: phase map output (requires --target-phase-dir for training data)
    - Zernike models (resnet18/resnet34/simple_cnn): coefficient output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# Set matplotlib backend to Agg to avoid threading issues
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

from loguru import logger

from ao_shaping.ml.dataset import (
    PhasePredictionDataset,
    ZernikeCoefficientDataset,
    create_dataloaders,
    create_zernike_loaders,
)
from ao_shaping.ml.models import (
    MODEL_REGISTRY,
    BasePhasePredictor,
    build_model,
    build_gan_models,
)
from ao_shaping.ml.trainer import PhaseGANTrainer


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
        "n_zernike_terms": {"values": [28, 55]},
        # UNet-specific parameters (ignored for zernike models)
        "lambda_l1": {"values": [50.0, 100.0, 200.0]},
        "gan_mode": {"values": ["lsgan", "vanilla"]},
    },
}


@click.group()
def cli():
    """Phase prediction training and inference."""
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
        from ao_shaping.ml.trainer import PhaseGANTrainer
        
        # Need to build discriminator for GAN
        in_channels = model.in_channels if hasattr(model, "in_channels") else 2
        _, discriminator = build_gan_models(in_channels=in_channels, device=device)
        
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
# Zernike-specific options
@click.option(
    "--input-mode",
    "input_mode",
    type=click.Choice(["focus", "pupil", "combined"]),
    default="combined",
    help="输入模式 (Zernike模型)",
)
@click.option("--n-zernike-terms", "n_zernike_terms", default=55, help="Zernike项数")
@click.option("--n-max", "n_max", default=10, help="Zernike径向阶")
# UNet-specific options
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
    help="损���函��� (UNet)",
)
# Logging options
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
    device: str | None,
    seed: int,
    target_size: str,
    input_mode: str,
    n_zernike_terms: int,
    n_max: int,
    lambda_l1: float,
    lambda_adv: float,
    gan_mode: str,
    loss_type: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    resume: str | None,
):
    """Unified training for all model types.
    
    Model types:
    - unet: Phase map output (U-Net+GAN)
    - resnet18/resnet34/simple_cnn: Zernike coefficient output
    
    Examples:
        # Train Zernike model
        uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture \\
            --model-type resnet18 --epochs 100 --lr 1e-3
        
        # Train UNet phase model
        uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture \\
            --model-type unet --epochs 100 --lr 2e-4
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parse target size
    h, w = map(int, target_size.split("x"))
    target_size_tuple = (h, w)
    
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Determine model category
    is_phase_model = model_type in PHASE_MODEL_TYPES
    
    # Create dataloaders based on model type
    if is_phase_model:
        # UNet: phase prediction (use old dataloaders)
        in_channels = 2 if input_mode == "combined" else 1
        logger.info(f"Loading phase prediction data from {data_dir}")
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_split=train_split,
            target_size=target_size_tuple,
            num_workers=num_workers,
            use_daheng=True,
            use_miicam=True,
            seed=seed,
        )
        test_loader = None
    else:
        # Zernike: coefficient prediction
        in_channels = 2 if input_mode == "combined" else 1
        logger.info(f"Loading Zernike coefficient data from {data_dir}")
        train_loader, val_loader, test_loader = create_zernike_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            input_mode=input_mode,
            n_zernike_terms=n_zernike_terms,
            n_max=n_max,
            target_size=target_size_tuple,
            num_workers=num_workers,
            seed=seed,
        )
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset) if val_loader else 0
    n_test = len(test_loader.dataset) if test_loader else 0
    logger.info(f"Dataset: train={n_train}, val={n_val}, test={n_test}")
    
    # Build model
    model = build_model(
        model_type,
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
    
    logger.info(f"Training complete! Model saved to {output_dir}")


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
@click.option("--target-size", default="256x256", help="目标尺寸")
def sweep(data_dir: str, project: str, batch_size: int, epochs: int, count: int, target_size: str):
    """Run wandb hyperparameter sweep for ALL model types.
    
    Supports: unet (phase), resnet18, resnet34, simple_cnn (Zernike coefficients)
    
    The sweep config includes:
    - model_type: All supported model types
    - lr: Learning rate
    - input_mode: focus/pupil/combined (Zernike models)
    - n_zernike_terms: 28 or 55 (Zernike models)
    - lambda_l1: L1 loss weight (UNet)
    - gan_mode: GAN mode (UNet)
    """
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
    logger.info(f"Fixed batch_size: {batch_size}, target_size: {target_size}")

    def agent():
        """Universal sweep agent that handles ALL model types."""
        wandb.init()
        cfg = wandb.config

        # Determine model category
        model_type = cfg.model_type
        is_phase_model = model_type in PHASE_MODEL_TYPES
        
        # Determine input channels
        if is_phase_model:
            in_ch = 2  # UNet uses both cameras
        else:
            in_ch = 2 if cfg.get("input_mode", "combined") == "combined" else 1
        
        # Build model
        n_coeffs = cfg.get("n_zernike_terms", 55)
        model = build_model(model_type, in_channels=in_ch, n_coeffs=n_coeffs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create dataloaders based on model type
        if is_phase_model:
            # UNet: phase prediction
            train_loader, val_loader = create_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                train_split=0.7,
                target_size=target_size_tuple,
                num_workers=0,
                use_daheng=True,
                use_miicam=True,
                seed=42,
            )
            test_loader = None
        else:
            # Zernike: coefficient prediction
            train_loader, val_loader, test_loader = create_zernike_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                train_split=0.7,
                val_split=0.15,
                input_mode=cfg.get("input_mode", "combined"),
                n_zernike_terms=n_coeffs,
                n_max=10,
                target_size=target_size_tuple,
                num_workers=0,
                seed=42,
            )

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        # Setup training based on model type
        if is_phase_model:
            # UNet: GAN training
            from ao_shaping.ml.trainer import PhaseGANTrainer
            
            _, discriminator = build_gan_models(in_channels=in_ch, device=device)
            
            lambda_l1 = cfg.get("lambda_l1", 100.0)
            lambda_adv = cfg.get("lambda_adv", 1.0)
            gan_mode = cfg.get("gan_mode", "lsgan")
            
            trainer = PhaseGANTrainer(
                generator=model,
                discriminator=discriminator,
                device=device,
                lr_gen=cfg.lr,
                lr_disc=cfg.lr,
                lambda_l1=lambda_l1,
                lambda_adv=lambda_adv,
                gan_mode=gan_mode,
                checkpoint_dir=None,  # No checkpoint saving during sweep
                loss_type="angular",
                use_wandb=False,
            )
            
            best_val = float("inf")
            for epoch in range(epochs):
                # Train one epoch (GAN)
                for batch in train_loader:
                    images = batch["image"]
                    phases = batch["phase"]
                    trainer.train_step(images, phases)
                
                # Validate
                val_info = trainer.validate(val_loader)
                v_loss = val_info.get("val_loss", float('inf'))
                best_val = min(best_val, v_loss)
                wandb.log({"val_loss": v_loss})
        else:
            # Zernike: Regression training
            opt = optim.Adam(model.parameters(), lr=cfg.lr)
            crit = nn.MSELoss()

            best_val = float("inf")
            for epoch in range(epochs):
                model.train()
                for batch in train_loader:
                    imgs = batch["image"].to(device)
                    cs = batch["coefficients"].to(device)
                    opt.zero_grad()
                    pred = model(imgs)
                    loss = crit(pred, cs)
                    loss.backward()
                    opt.step()

                model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch["image"].to(device)
                        cs = batch["coefficients"].to(device)
                        pred = model(imgs)
                        v_loss += crit(pred, cs).item() * imgs.size(0)

                v_loss /= n_val
                best_val = min(best_val, v_loss)
                wandb.log({"val_loss": v_loss})
                
                # Log test visualization every 10 epochs
                if (epoch + 1) % 10 == 0 and test_loader and len(test_loader.dataset) > 0:
                    import matplotlib.pyplot as plt
                    from ao_shaping.ml.dataset import coefficients_to_phase_map

                    # Get random test sample
                    sample_idx = np.random.randint(0, len(test_loader.dataset))
                    sample = test_loader.dataset[sample_idx]

                    with torch.no_grad():
                        test_img = sample["image"].unsqueeze(0).to(device)
                        test_coeff = sample["coefficients"].numpy()
                        pred_coeff = model(test_img).cpu().squeeze(0).numpy()

                    # Generate phase maps
                    target_phase = coefficients_to_phase_map(test_coeff, size=target_size_tuple)
                    pred_phase = coefficients_to_phase_map(pred_coeff, size=target_size_tuple)

                    # Create visualization
                    diff = pred_coeff - test_coeff
                    x = np.arange(len(test_coeff))

                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

                # Save images to temporary files and load with PIL
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save phase comparison
                    phase_path = Path(temp_dir) / "phase_comparison.png"
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(target_phase[0], cmap="gray")
                    plt.title("Target Phase")
                    plt.subplot(1, 2, 2)
                    plt.imshow(pred_phase[0], cmap="gray")
                    plt.title("Predicted Phase")
                    plt.tight_layout()
                    plt.savefig(phase_path)
                    plt.close()

                    # Load with PIL and log to wandb
                    phase_image = Image.open(phase_path)
                    wandb.log(
                        {
                            "phase_target_vs_pred": wandb.Image(
                                phase_image,
                                caption=f"Epoch {epoch + 1}: phase target vs prediction",
                            )
                        }
                    )

                    # Save coefficient comparison
                    coeff_path = Path(temp_dir) / "coeff_comparison.png"
                    plt.figure(figsize=(12, 8))

                    # Top: target vs prediction bar chart
                    plt.subplot(2, 1, 1)
                    x = np.arange(len(test_coeff))
                    plt.bar(x - 0.2, test_coeff, 0.4, label="Ground Truth", alpha=0.8)
                    plt.bar(x + 0.2, pred_coeff, 0.4, label="Prediction", alpha=0.8)
                    plt.xlabel("Zernike Index")
                    plt.ylabel("Coefficient Value")
                    plt.title(f"Zernike Coefficients Comparison (Epoch {epoch + 1})")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Bottom: difference bar chart
                    plt.subplot(2, 1, 2)
                    diff = pred_coeff - test_coeff
                    colors = ["green" if d >= 0 else "red" for d in diff]
                    plt.bar(x, diff, color=colors, alpha=0.7)
                    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
                    plt.xlabel("Zernike Index")
                    plt.ylabel("Difference (Pred - GT)")
                    plt.title("Prediction Error")
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(coeff_path)
                    plt.close()

                    # Load with PIL and log to wandb
                    coeff_image = Image.open(coeff_path)
                    wandb.log(
                        {
                            "coeff_comparison": wandb.Image(
                                coeff_image,
                                caption=f"Epoch {epoch + 1}: coefficients target vs prediction",
                            )
                        }
                    )

        wandb.log({"best_val_loss": best_val})

    wandb.agent(sweep_id, agent, count=count)
    logger.info("Sweep complete!")


if __name__ == "__main__":
    cli()
