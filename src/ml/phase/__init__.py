"""Phase prediction submodule - U-Net + PatchGAN for phase map prediction."""

from .unet import (
    UNetGenerator,
    DoubleConv,
    build_unet,
)
from .discriminator import (
    PatchGANDiscriminator,
    build_discriminator,
)
from .dataset import (
    PhasePredictionDataset,
    coefficients_to_phase_map,
    load_zernike_coefficients,
    create_dataloaders,
)
from .trainer import (
    PhaseGANTrainer,
    angular_loss,
    GANLoss,
)

__all__ = [
    # UNet Generator
    "UNetGenerator",
    "DoubleConv",
    "build_unet",
    # Discriminator
    "PatchGANDiscriminator",
    "build_discriminator",
    # Dataset
    "PhasePredictionDataset",
    "coefficients_to_phase_map",
    "load_zernike_coefficients",
    "create_dataloaders",
    # Trainer
    "PhaseGANTrainer",
    "angular_loss",
    "GANLoss",
]