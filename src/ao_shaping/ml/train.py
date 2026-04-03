"""Phase prediction training and inference CLI.

Usage:
    # Train a model
    uv run python src/ao_shaping/ml/train.py train --data-dir data/slm_capture --epochs 100

    # Inference on a single image
    uv run python src/ao_shaping/ml/train.py predict --checkpoint checkpoints/best.pt --image input.npy

    # Inference on a dataset
    uv run python src/ao_shaping/ml/train.py predict --checkpoint checkpoints/best.pt --data-dir data/slm_capture
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loguru import logger

from ao_shaping.ml.dataset import PhasePredictionDataset, create_dataloaders
from ao_shaping.ml.models import build_model
from ao_shaping.ml.trainer import PhaseGANTrainer


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
@click.option("--gan-mode", type=click.Choice(["vanilla", "lsgan"]), default="vanilla")
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
        "seed": seed,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    import json

    with open(Path(output_dir) / "config.json", "w") as f:
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


if __name__ == "__main__":
    cli()
