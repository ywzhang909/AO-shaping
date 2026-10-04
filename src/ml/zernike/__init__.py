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
from ml.zernike.forward_model import (
    DEFAULT_N_COEFFS,
    ZernikeCoeffConfig,
    ZernikeCoeffConvNet,
    ZernikeCoeffMLP,
    build_forward_model,
    count_parameters,
    peak_normalize,
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
    # Learned coefficient -> far-field forward model (conv decoder + MLP baseline)
    "DEFAULT_N_COEFFS",
    "ZernikeCoeffConfig",
    "ZernikeCoeffConvNet",
    "ZernikeCoeffMLP",
    "peak_normalize",
    "build_forward_model",
    "count_parameters",
    # Dataset
    "ZernikeCoefficientDataset",
    "coefficients_to_phase_map",
    "load_zernike_coefficients",
    "create_zernike_loaders",
]