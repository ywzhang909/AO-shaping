"""Zernike submodule - physics-based global Zernike forward model.

Exports the canonical ``ZernikeBasis`` / ``ZernikeAmpModel`` pair (see
``ml.zernike.models``) plus the dataset helpers, which are unchanged.

Example:
    from ml.zernike import ZernikeAmpModel, ZernikeAmpConfig

    model = ZernikeAmpModel(ZernikeAmpConfig(n_max=4, grid=64))
    far_field = model(phase_cos, phase_sin)
"""

from ml.zernike.dataset import (
    ZernikeCoefficientDataset,
    coefficients_to_phase_map,
    create_zernike_loaders,
    load_zernike_coefficients,
)
from ml.zernike.models import (
    ZernikeAmpConfig,
    ZernikeAmpFitConfig,
    ZernikeAmpModel,
    ZernikeAmpResult,
    ZernikeBasis,
)

__all__ = [
    # Physics-based Zernike forward model
    "ZernikeBasis",
    "ZernikeAmpConfig",
    "ZernikeAmpFitConfig",
    "ZernikeAmpModel",
    "ZernikeAmpResult",
    # Dataset
    "ZernikeCoefficientDataset",
    "coefficients_to_phase_map",
    "load_zernike_coefficients",
    "create_zernike_loaders",
]