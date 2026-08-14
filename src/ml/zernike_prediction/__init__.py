"""Zernike phase generation and unwrapping for the SLM phase-prediction dataset.

This package implements capture-exact Zernike phase math (reproduces stored
``phase.csv`` files with a 100% exact match) and DCT-based 2D least-squares
phase unwrapping, forming the data-preparation layer (T1) of the
Zernike-coefficient regression pipeline. Also exposes the T2 regression-metrics
API (``metrics.py``) for evaluating predicted vs. true coefficient vectors.
"""

from ml.zernike_prediction.metrics import (
    alignment_ok,
    coefficient_names,
    mae,
    metrics_summary,
    mse,
    per_coeff_mae,
    per_order_mae,
    phase_mae,
    phase_rmse,
    r2,
    rmse,
)
from ml.zernike_prediction.phase_gen import (
    COEFF_ORDER_NAMES,
    build_basis_maps,
    coefficients_to_phase_radians,
    coefficients_to_wrapped_gray,
    count_zernike_terms,
    gray_to_wrapped,
    iter_nm_terms,
    load_stored_gray,
    metadata_order_to_noll,
    non_piston_indices,
    unwrap_phase_lsq,
    zernike_radial,
)
from ml.zernike_prediction.dataset import (
    ZernikeDualDataset,
    build_label_manifest,
    collate_with_meta,
    create_zernike_loaders,
)
from ml.zernike_prediction.plots import (
    all_plots,
    loss_curves,
    per_coeff_mae_bar,
    per_order_mae_bar,
    phase_error_map,
    phase_grid,
    predict_true_scatter,
)

__all__ = [
    "COEFF_ORDER_NAMES",
    "ZernikeDualDataset",
    "alignment_ok",
    "all_plots",
    "build_basis_maps",
    "build_label_manifest",
    "coefficient_names",
    "coefficients_to_phase_radians",
    "coefficients_to_wrapped_gray",
    "collate_with_meta",
    "count_zernike_terms",
    "create_zernike_loaders",
    "gray_to_wrapped",
    "iter_nm_terms",
    "load_stored_gray",
    "loss_curves",
    "mae",
    "metadata_order_to_noll",
    "metrics_summary",
    "mse",
    "non_piston_indices",
    "per_coeff_mae",
    "per_coeff_mae_bar",
    "per_order_mae",
    "per_order_mae_bar",
    "phase_error_map",
    "phase_grid",
    "phase_mae",
    "phase_rmse",
    "predict_true_scatter",
    "r2",
    "rmse",
    "unwrap_phase_lsq",
    "zernike_radial",
]
