"""ML - Phase prediction models for AO-Shaping.

This package contains machine learning models for adaptive optics phase prediction:

- ml.phase: Phase map prediction (U-Net + PatchGAN, image-to-image)
- ml.zernike: Zernike coefficient prediction (ResNet/SimpleCNN, regression)

Example usage:
    # Phase map prediction
    from ml.phase import UNetGenerator, PhaseGANTrainer
    
    # Zernike coefficient prediction  
    from ml.zernike import ResNetRegression, build_model
    
    # Or use convenience functions
    from ml.phase import build_unet, create_dataloaders
    from ml.zernike import build_zernike_model, create_zernike_loaders
"""

# Phase submodule exports
from ml.phase import (
    UNetGenerator,
    PatchGANDiscriminator,
    PhasePredictionDataset,
    PhaseGANTrainer,
    create_dataloaders,
    angular_loss,
    GANLoss,
    build_unet,
    build_discriminator,
)

# Zernike submodule exports  
from ml.zernike import (
    ResNetRegression,
    SimpleCNNRegression,
    ZernikeCoefficientDataset,
    create_zernike_loaders,
    build_model,
    build_zernike_model,
    BasePhasePredictor,
    MODEL_REGISTRY,
)

# Shared utilities
from ml.phase.dataset import coefficients_to_phase_map
from ml.zernike.dataset import coefficients_to_phase_map as _phase_map_helper

# For backwards compatibility with old imports (ao_shaping.ml.*)
# These are also provided by the submodules above

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
    # Zernike prediction
    "ResNetRegression",
    "SimpleCNNRegression",
    "ZernikeCoefficientDataset",
    "create_zernike_loaders",
    "build_model",
    "build_zernike_model",
    "BasePhasePredictor",
    "MODEL_REGISTRY",
    # Utilities
    "coefficients_to_phase_map",
]