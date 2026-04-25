"""Zernike coefficient prediction submodule - ResNet/SimpleCNN for regression."""

from ml.zernike.models import (
    BasePhasePredictor,
    ResNetRegression,
    SimpleCNNRegression,
    MODEL_REGISTRY,
    build_model,
    build_zernike_model,
)
from ml.zernike.dataset import (
    ZernikeCoefficientDataset,
    coefficients_to_phase_map,
    load_zernike_coefficients,
    create_zernike_loaders,
)

__all__ = [
    # Models
    "BasePhasePredictor",
    "ResNetRegression",
    "SimpleCNNRegression",
    "MODEL_REGISTRY",
    "build_model",
    "build_zernike_model",
    # Dataset
    "ZernikeCoefficientDataset",
    "coefficients_to_phase_map",
    "load_zernike_coefficients",
    "create_zernike_loaders",
]