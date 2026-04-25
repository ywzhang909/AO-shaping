"""GAN trainer for phase prediction with U-Net generator + PatchGAN discriminator."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Internal imports - using new ml package paths
from ml.phase.unet import UNetGenerator
from ml.phase.discriminator import PatchGANDiscriminator


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Circular/angular loss for phase prediction.

    Phase values wrap around 2*pi, so we need special handling.
    This computes the shortest angular distance between predicted and target phase.

    Args:
        pred: Predicted phase (B, 1, H, W) or (B, H, W)
        target: Target phase (B, 1, H, W) or (B, H, W)

    Returns:
        Mean squared angular difference
    """if pred.shape[1] != target.shape[1]:
    # Ensure same shape
    if pred.shape != target.shape and pred.ndim == 4 and target.ndim == 4:
        if pred.shape[1] == 1:
            pred = pred.squeeze(1)
        elif target.shape[1] == 1:
            target = target.squeeze(1)

    # Compute angular difference (shortest path)
    diff = torch.remainder(pred - target + torch.pi, 2 * torch.pi) - torch.pi
    return torch.mean(diff**2)


class GANLoss(nn.Module):
    """GAN loss for discriminator and generator."""

    def __init__(self, gan_mode: str = "vanilla"):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        if self.gan_mode in {"vanilla", "lsgan"}:
            target = (
                torch.ones_like(prediction)
                if target_is_real
                else torch.zeros_like(prediction)
            )
        else:
            raise ValueError(f"Unsupported GAN mode: {self.gan_mode}")
        return self.loss(prediction, target)


class PhaseGANTrainer:
    """Train U-Net + PatchGAN for phase prediction from camera images.

    Loss = AngularLoss(pred_phase, true_phase) + lambda_adv * BCE(GAN_loss)
    """

    def __init__(
        self,
        generator: UNetGenerator,
        discriminator: PatchGANDiscriminator,
        device: torch.device,
        lr_gen: float = 2e-4,
        lr_disc: float = 2e-4,
        lambda_l1: float = 100.0,
        lambda_adv: float = 1.0,
        beta1: float = 0.5,
        beta2: float = 0.999,
        gan_mode: str = "vanilla",
        checkpoint_dir: str | Path | None = None,
        loss_type: str = "angular",
        use_wandb: bool = False,
        wandb_project: str = "ao-shaping",
        wandb_run_name: str | None = None,
        log_images: bool = True,
    ):
        self.gen = generator
        self.disc = discriminator
        self.device = device

        self.lambda_l1 = lambda_l1
        self.lambda_adv = lambda_adv
        self.loss_type = loss_type
        self.use_wandb = use_wandb
        self.log_images = log_images

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "lr_gen": lr_gen,
                    "lr_disc": lr_disc,
                    "lambda_l1": lambda_l1,
                    "lambda_adv": lambda_adv,
                    "beta1": beta1,
                    "beta2": beta2,
                    "gan_mode": gan_mode,
                    "loss_type": loss_type,
                },
            )

        # Optimizers
        self.opt_gen = torch.optim.Adam(
            self.gen.parameters(), lr=lr_gen, betas=(beta1, beta2)
        )
        self.opt_disc = torch.optim.Adam(
            self.disc.parameters(), lr=lr_disc, betas=(beta1, beta2)
        )

        # Learning rate schedulers
        self.sched_gen = torch.optim.lr_scheduler.StepLR(
            self.opt_gen, step_size=50, gamma=0.5
        )
        self.sched_disc = torch.optim.lr_scheduler.StepLR(
            self.opt_disc, step_size=50, gamma=0.5
        )

        # Loss functions
        self.gan_loss = GANLoss(gan_mode).to(device)
        self.l1_loss = nn.L1Loss()
        self.angular_loss = angular_loss

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "gen_loss": [],
            "disc_loss": [],
            "main_loss": [],
            "adv_loss": [],
            "val_loss": [],
        }

    def train_step(self, images: torch.Tensor, phases: torch.Tensor) -> dict:
        """Single training step."""
        images = images.to(self.device)
        phases = phases.to(self.device)

        # Train Discriminator
        self.opt_disc.zero_grad()

        fake_phase = self.gen(images).detach()
        fake_input = torch.cat([images, fake_phase], dim=1)
        pred_fake = self.disc(fake_input)
        loss_disc_fake = self.gan_loss(pred_fake, target_is_real=False)

        real_input = torch.cat([images, phases], dim=1)
        pred_real = self.disc(real_input)
        loss_disc_real = self.gan_loss(pred_real, target_is_real=True)

        loss_disc = (loss_disc_fake + loss_disc_real) * 0.5
        loss_disc.backward()
        self.opt_disc.step()

        # Train Generator
        self.opt_gen.zero_grad()

        fake_phase = self.gen(images)
        fake_input = torch.cat([images, fake_phase], dim=1)
        pred_fake = self.disc(fake_input)

        loss_adv = self.gan_loss(pred_fake, target_is_real=True)

        if self.loss_type == "angular":
            loss_main = self.angular_loss(fake_phase, phases)
        else:
            loss_main = self.l1_loss(fake_phase, phases)

        loss_gen = loss_main * self.lambda_l1 + loss_adv * self.lambda_adv

        loss_gen.backward()
        self.opt_gen.step()

        return {
            "gen_loss": loss_gen.item(),
            "disc_loss": loss_disc.item(),
            "main_loss": loss_main.item(),
            "adv_loss": loss_adv.item(),
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Run validation pass."""
        self.gen.eval()
        self.disc.eval()

        total_main = 0.0
        total_samples = 0

        for batch in val_loader:
            images = batch["image"].to(self.device)
            phases = batch["phase"].to(self.device)

            pred_phase = self.gen(images)

            if self.loss_type == "angular":
                loss = self.angular_loss(pred_phase, phases)
            else:
                loss = self.l1_loss(pred_phase, phases)

            total_main += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_main / max(total_samples, 1)

        self.gen.train()
        self.disc.train()
        return {"val_loss": avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
        save_every: int = 10,
    ) -> dict:
        """Full training loop."""
        self.gen.train()
        self.disc.train()

        best_val_loss = float("inf")
        loss_key = "angular" if self.loss_type == "angular" else "l1"

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_losses = {
                "gen_loss": 0,
                "disc_loss": 0,
                "main_loss": 0,
                "adv_loss": 0,
            }
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch in pbar:
                images = batch["image"]
                phases = batch["phase"]

                losses = self.train_step(images, phases)
                for k in epoch_losses:
                    epoch_losses[k] += losses[k]
                n_batches += 1

                pbar.set_postfix({
                    "G": f"{losses['gen_loss']:.4f}",
                    "D": f"{losses['disc_loss']:.4f}",
                    loss_key: f"{losses['main_loss']:.4f}",
                })

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            # Validation
            val_info = {}
            if val_loader is not None:
                val_info = self.validate(val_loader)
                epoch_losses.update(val_info)

                if val_info.get("val_loss", float("inf")) < best_val_loss:
                    best_val_loss = val_info["val_loss"]
                    self.save_checkpoint("best.pt")

            # Record history
            for k in self.history:
                if k in epoch_losses:
                    self.history[k].append(epoch_losses[k])

            # Step schedulers
            self.sched_gen.step()
            self.sched_disc.step()

            epoch_time = time.time() - epoch_start
            lr = self.opt_gen.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"G: {epoch_losses['gen_loss']:.4f} | "
                f"D: {epoch_losses['disc_loss']:.4f} | "
                f"{loss_key}: {epoch_losses['main_loss']:.4f} | "
                f"Val: {epoch_losses.get('val_loss', 'N/A'):.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/gen_loss": epoch_losses["gen_loss"],
                    "train/disc_loss": epoch_losses["disc_loss"],
                    f"train/{loss_key}_loss": epoch_losses["main_loss"],
                    "train/adv_loss": epoch_losses["adv_loss"],
                    "val/loss": epoch_losses.get("val_loss", 0),
                    "train/lr": lr,
                    "train/time": epoch_time,
                })

                if self.log_images and val_loader is not None:
                    self._log_val_images(val_loader)

            if (epoch + 1) % save_every == 0 and self.checkpoint_dir:
                self.save_checkpoint(f"checkpoint_{epoch + 1}.pt")

        if self.checkpoint_dir:
            self.save_checkpoint("final.pt")
            self.save_history()

        return self.history

    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        path = self.checkpoint_dir / filename
        torch.save({
            "generator": self.gen.state_dict(),
            "discriminator": self.disc.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
            "sched_gen": self.sched_gen.state_dict(),
            "sched_disc": self.sched_disc.state_dict(),
            "history": self.history,
            "lambda_l1": self.lambda_l1,
            "lambda_adv": self.lambda_adv,
        }, path)

    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.gen.load_state_dict(checkpoint["generator"])
        self.disc.load_state_dict(checkpoint["discriminator"])
        self.opt_gen.load_state_dict(checkpoint["opt_gen"])
        self.opt_disc.load_state_dict(checkpoint["opt_disc"])
        if "sched_gen" in checkpoint:
            self.sched_gen.load_state_dict(checkpoint["sched_gen"])
        if "sched_disc" in checkpoint:
            self.sched_disc.load_state_dict(checkpoint["sched_disc"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "lambda_l1" in checkpoint:
            self.lambda_l1 = checkpoint["lambda_l1"]
        if "lambda_adv" in checkpoint:
            self.lambda_adv = checkpoint["lambda_adv"]

    def save_history(self):
        """Save training history as JSON."""
        if not self.checkpoint_dir:
            return
        serializable = {}
        for k, v in self.history.items():
            serializable[k] = [
                float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v
            ]
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(serializable, f, indent=2)

        if self.use_wandb:
            wandb.finish()

    def _log_val_images(self, val_loader: DataLoader) -> None:
        """Log validation images to wandb."""
        self.gen.eval()

        batch = next(iter(val_loader))
        images = batch["image"].to(self.device)
        phases = batch["phase"].to(self.device)

        with torch.no_grad():
            pred_phase = self.gen(images)

        idx = 0
        daheng = images[idx, 0].cpu().numpy()
        miicam = images[idx, 1].cpu().numpy()
        true_phase = phases[idx, 0].cpu().numpy()
        pred = pred_phase[idx, 0].cpu().numpy()

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(daheng, cmap="gray")
        axes[0].set_title("Daheng")
        axes[0].axis("off")

        axes[1].imshow(miicam, cmap="gray")
        axes[1].set_title("MiiCam")
        axes[1].axis("off")

        im2 = axes[2].imshow(true_phase, cmap="hsv", vmin=0, vmax=1)
        axes[2].set_title("True Phase")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2])

        im3 = axes[3].imshow(pred, cmap="hsv", vmin=0, vmax=1)
        axes[3].set_title("Predicted")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3])

        plt.tight_layout()

        wandb.log({"val/images": wandb.Image(fig)})
        plt.close(fig)

        self.gen.train()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict phase from camera image(s)."""
        self.gen.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        phase = self.gen(image)
        return phase.squeeze(0) if phase.size(0) == 1 else phase