"""ML - Phase prediction models for AO-Shaping.

This package contains machine learning models for adaptive optics phase prediction:

- ml.phase: Phase map prediction (U-Net + PatchGAN, image-to-image)
- ml.zernike: Physics-based global Zernike forward model (optimises one
  coefficient vector through a real Fraunhofer propagation)

Example usage:
    # Phase map prediction
    from ml.phase import UNetGenerator, PhaseGANTrainer

    # Physics-based Zernike forward model
    from ml.zernike import ZernikeAmpModel, ZernikeAmpConfig

    # Or use convenience functions
    from ml.phase import build_unet, create_dataloaders
    from ml.zernike import create_zernike_loaders

Note:
    ``ml.zernike`` no longer depends on torchvision: the forward model is pure
    torch, so its exports are unconditional. The previous ResNet/SimpleCNN
    coefficient *regressors* were removed -- they predicted coefficients from
    an image without any forward model, which is not what the bench needs.
"""

# Phase submodule exports
from ml.phase import (
    GANLoss,
    PatchGANDiscriminator,
    PhaseGANTrainer,
    PhasePredictionDataset,
    UNetGenerator,
    angular_loss,
    build_discriminator,
    build_unet,
    create_dataloaders,
)

# Zernike submodule exports (pure torch, no optional dependency)
from ml.zernike import (
    ZernikeAmpConfig,
    ZernikeAmpFitConfig,
    ZernikeAmpModel,
    ZernikeAmpResult,
    ZernikeBasis,
    ZernikeCoefficientDataset,
    create_zernike_loaders,
)

# Shared utilities
from ml.phase.dataset import coefficients_to_phase_map

__all__ = [
    # Phase prediction
    "UNetGenerator",
    "PatchGANDiscriminator",
    "PhasePredictionDataset",
    "PhaseGANTrainer",
    "create_dataloaders",
    "angular_loss",
    "GANLoss",
    "build_unet",
    "build_discriminator",
    # Zernike physics-based forward model
    "ZernikeBasis",
    "ZernikeAmpConfig",
    "ZernikeAmpFitConfig",
    "ZernikeAmpModel",
    "ZernikeAmpResult",
    "ZernikeCoefficientDataset",
    "create_zernike_loaders",
    # Utilities
    "coefficients_to_phase_map",
]