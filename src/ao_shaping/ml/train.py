"""Phase prediction training and inference CLI.

Usage:
    # Train legacy U-Net+GAN (phase map)
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --epochs 100

    # Train Zernike coefficient predictor
    uv run python src/ao_shaping/ml/train.py zernike --data-dir data/slm_capture/20260414_171241

    # Run wandb sweep
    uv run python src/ao_shaping/ml/train.py sweep --data-dir data/slm_capture/20260414_171241

    # Inference
    uv run python src/ao_shaping/ml/train.py predict --checkpoint checkpoints/best.pt --data-dir data/slm_capture
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from loguru import logger

from ao_shaping.ml.dataset import (
    PhasePredictionDataset,
    ZernikeCoefficientDataset,
    create_dataloaders,
    create_zernike_loaders,
)
from ao_shaping.ml.models import (
    MODEL_REGISTRY,
    build_model,
    build_gan_models,
)
from ao_shaping.ml.trainer import PhaseGANTrainer


# =============================================================================
# WandB Sweep Configuration
# =============================================================================

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"values": [1e-4, 5e-4, 1e-3, 5e-3]},
        "batch_size": {"values": [4, 8, 16]},
        "model_type": {"values": ["resnet18", "simple_cnn"]},
        "input_mode": {"values": ["focus", "pupil", "combined"]},
        "n_zernike_terms": {"values": [28, 55]},
    },
}


@click.group()
def cli():
    """Phase prediction U-Net+GAN training and inference."""
    pass


@cli.command()
@click.option("--data-dir", required=True, help="训练数据目录 (包含sample_XXXX子目录)")
@click.option(
    "--output-dir", default="checkpoints", help="模型保存目录 (default: checkpoints)"
)
@click.option("--epochs", default=100, help="训练轮数 (default: 100)")
@click.option("--batch-size", default=8, help="批次大小 (default: 8)")
@click.option("--lr", default=2e-4, help="学习率 (default: 2e-4)")
@click.option("--lambda-l1", default=100.0, help="L1损失权重 (default: 100.0)")
@click.option("--lambda-adv", default=1.0, help="对抗损失权重 (default: 1.0)")
@click.option("--target-size", default=None, help="目标尺寸 HxW (default: 原始尺寸)")
@click.option("--train-split", default=0.8, help="训练集比例 (default: 0.8)")
@click.option("--num-workers", default=0, help="DataLoader工作进程数 (default: 0)")
@click.option("--use-daheng/--no-daheng", default=True, help="使用Daheng相机数据")
@click.option("--use-miicam/--no-miicam", default=True, help="使用MiiCam相机数据")
@click.option("--device", default=None, help="设备 (default: auto cuda/cpu)")
@click.option("--seed", default=42, help="随机种子 (default: 42)")
@click.option("--gan-mode", type=click.Choice(["vanilla", "lsgan"]), default="lsgan")
@click.option(
    "--loss-type",
    type=click.Choice(["l1", "angular"]),
    default="angular",
    help="损失函数类型: l1或angular (default: angular)",
)
@click.option("--use-wandb/--no-wandb", default=False, help="使用 wandb 记录训练")
@click.option("--wandb-project", default="ao-shaping", help="wandb 项目名称")
@click.option("--wandb-name", default=None, help="wandb run 名称")
@click.option("--log-images/--no-log-images", default=True, help="记录验证图像到 wandb")
@click.option("--resume", default=None, help="从检查点恢复训练")
def train(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_l1: float,
    lambda_adv: float,
    target_size: str | None,
    train_split: float,
    num_workers: int,
    use_daheng: bool,
    use_miicam: bool,
    device: str | None,
    seed: int,
    gan_mode: str,
    loss_type: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    log_images: bool,
    resume: str | None,
):
    """Train U-Net+GAN for phase prediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Parse target size
    target_size_tuple = None
    if target_size:
        h, w = map(int, target_size.split("x"))
        target_size_tuple = (h, w)

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
        target_size=target_size_tuple,
        num_workers=num_workers,
        use_daheng=use_daheng,
        use_miicam=use_miicam,
        seed=seed,
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    # Build models
    generator, discriminator = build_model(
        in_channels=in_channels,
        device=device,
    )
    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(
        f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}"
    )

    # Create trainer
    trainer = PhaseGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr_gen=lr,
        lr_disc=lr,
        lambda_l1=lambda_l1,
        lambda_adv=lambda_adv,
        gan_mode=gan_mode,
        checkpoint_dir=output_dir,
        loss_type=loss_type,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_name,
        log_images=log_images,
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
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    import json

    with (Path(output_dir) / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_every=10,
    )

    logger.info("Training complete!")
    logger.info(f"Best val loss: {min(history.get('val_loss', [float('inf')])):.4f}")


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
    generator, _ = build_model(in_channels=in_channels, device=device)
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


# =============================================================================
# Zernike Coefficient Training
# =============================================================================


@cli.command("zernike")
@click.option("--data-dir", required=True, help="数据目录 (含sample_XXXX子目录)")
@click.option("--output-dir", default="checkpoints_zernike", help="输出目录")
@click.option(
    "--model-type",
    "model_type",
    default="resnet18",
    help="模型类型: resnet18, simple_cnn",
)
@click.option(
    "--input-mode",
    type=click.Choice(["focus", "pupil", "combined"]),
    default="combined",
    help="输入模式",
)
@click.option("--n-zernike-terms", "n_zernike_terms", default=55, help="Zernike项数量")
@click.option("--n-max", "n_max", default=10, help="Zernike径向阶")
@click.option("--target-size", "target_size", default="256x256", help="图像尺寸 HxW")
@click.option("--epochs", default=100, help="训练轮数")
@click.option("--batch-size", default=8, help="批次大小")
@click.option("--lr", default=1e-3, help="学习率")
@click.option("--weight-decay", default=1e-4, help="权重衰减")
@click.option("--train-split", default=0.7, help="训练集比例")
@click.option("--val-split", default=0.15, help="验证集比例")
@click.option("--num-workers", default=0, help="DataLoader工作进程")
@click.option("--device", default=None, help="设备")
@click.option("--seed", default=42, help="随机种子")
@click.option("--use-wandb/--no-wandb", default=False, help="使用wandb")
@click.option("--wandb-project", default="ao-shaping-zernike", help="项目名")
@click.option("--wandb-name", default=None, help="run名")
@click.option("--resume", default=None, help="恢复训练")
def zernike_train(
    data_dir: str,
    output_dir: str,
    model_type: str,
    input_mode: str,
    n_zernike_terms: int,
    n_max: int,
    target_size: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_split: float,
    val_split: float,
    num_workers: int,
    device: str | None,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    resume: str | None,
):
    """Train Zernike coefficient predictor."""
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

    # Input channels
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
        n_max=n_max,
        target_size=target_size_tuple,
        num_workers=num_workers,
        seed=seed,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Build model
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
                    "n_max": n_max,
                    "target_size": target_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "train_split": train_split,
                    "val_split": val_split,
                    "seed": seed,
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
    with (Path(output_dir) / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    logger.info("Starting training...")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            coeffs = batch["coefficients"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, coeffs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                coeffs = batch["coefficients"].to(device)
                pred = model(images)
                loss = criterion(pred, coeffs)
                val_loss += loss.item() * images.size(0)

        val_loss /= n_val
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
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

    # Save final
    torch.save(
        {
            "model": model.state_dict(),
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


# =============================================================================
# WandB Sweep
# =============================================================================


@cli.command("sweep")
@click.option("--data-dir", required=True, help="数据目录")
@click.option("--project", default="ao-shaping-sweep", help="wandb项目名")
@click.option("--epochs", default=30, help="每次run轮数")
@click.option("--count", default=10, help="sweep次数")
def sweep(data_dir: str, project: str, epochs: int, count: int):
    """Run wandb hyperparameter sweep."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Run: pip install wandb")
        return

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    logger.info(f"Sweep: {sweep_id}")

    def agent():
        wandb.init()
        cfg = wandb.config

        in_ch = 2 if cfg.input_mode == "combined" else 1
        model = build_model(
            cfg.model_type, in_channels=in_ch, n_coeffs=cfg.n_zernike_terms
        )

        train_loader, val_loader, _ = create_zernike_loaders(
            data_dir=data_dir,
            batch_size=cfg.batch_size,
            input_mode=cfg.input_mode,
            n_zernike_terms=cfg.n_zernike_terms,
            target_size=(256, 256),
            num_workers=0,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        crit = nn.MSELoss()

        best_val = float("inf")
        for _ in range(epochs):
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
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    cs = batch["coefficients"].to(device)
                    v_loss += crit(model(imgs), cs).item()

            v_loss /= len(val_loader.dataset)
            best_val = min(best_val, v_loss)
            wandb.log({"val_loss": v_loss})

        wandb.log({"best_val_loss": best_val})

    wandb.agent(sweep_id, agent, count=count)
    logger.info("Sweep complete!")


if __name__ == "__main__":
    cli()
