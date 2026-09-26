"""Beam-shaping target and objective utilities, split by type.

The former flat :mod:`ao_shaping.utils.image.targets` module was split into this
package so each concern lives in a focused module:

* :mod:`~ao_shaping.utils.image.target.patterns` - parametric target patterns
  on the SLM / image grid (``create_target_shape`` etc.).
* :mod:`~ao_shaping.utils.image.target.metrics` - objective / metric
  computations on CCD frames (ROI PIB, RMS, RMSE, shape metrics, stage weights).
* :mod:`~ao_shaping.utils.image.target.square` - square beam-shaping target
  sizing and amplitude masks.
* :mod:`~ao_shaping.utils.image.target.ccd` - CCD-frame driven targets
  (crop/resize, arbitrary-centre masks, frame -> SLM-grid pipeline).
* :mod:`~ao_shaping.utils.image.target.objective` - the ``ShapingObjective``
  selector, its parameter dataclasses and the adaptive ``rms_pib`` weights.

This package is a leaf of :mod:`ao_shaping.utils`: it must not import
``algorithm`` / ``drivers`` / ``optimizer`` / ``runners`` or torch. The legacy
``targets`` module re-exports everything here for backward compatibility.
"""

from __future__ import annotations

from ao_shaping.utils.image.target.ccd import (
    _rectangle_mask,
    _resize_bilinear,
    _SHAPE_ALIASES,
    build_ccd_target,
    build_target_from_frame,
    crop_resize_to_grid,
    generate_target_mask,
    square_target_from_measurement,
)
from ao_shaping.utils.image.target.metrics import (
    SHAPE_STAGE_WEIGHTS,
    TARGET_SHAPE_CHOICES,
    TargetShape,
    rmse_shape_metric,
    rms_pib_terms,
    roi_energy_loss,
    roi_pib_metric,
    shape_metric,
    shape_stage,
    shape_stage_from_energy,
    spot_waist_sigma,
    target_shape_roi,
)
from ao_shaping.utils.image.target.objective import (
    GUARDED_OBJECTIVES,
    SHAPING_OBJECTIVE_CHOICES,
    ObjectiveResult,
    ShapeScoringParams,
    ShapingObjective,
    ShapingObjectiveParams,
    _resolve_init_weights,
    _update_dynamic_weights,
)
from ao_shaping.utils.image.target.patterns import (
    create_target_mask,
    create_target_shape,
    load_target_image,
)
from ao_shaping.utils.image.target.square import (
    build_square_target_amplitude,
    compute_square_side,
)

__all__ = [
    # patterns
    "create_target_shape",
    "create_target_mask",
    "load_target_image",
    # metrics
    "TARGET_SHAPE_CHOICES",
    "TargetShape",
    "target_shape_roi",
    "spot_waist_sigma",
    "roi_energy_loss",
    "roi_pib_metric",
    "rms_pib_terms",
    "rmse_shape_metric",
    "SHAPE_STAGE_WEIGHTS",
    "shape_stage",
    "shape_stage_from_energy",
    "shape_metric",
    # square
    "compute_square_side",
    "build_square_target_amplitude",
    # ccd
    "crop_resize_to_grid",
    "square_target_from_measurement",
    "generate_target_mask",
    "build_ccd_target",
    "build_target_from_frame",
    # objective
    "ShapeScoringParams",
    "ShapingObjectiveParams",
    "ObjectiveResult",
    "SHAPING_OBJECTIVE_CHOICES",
    "GUARDED_OBJECTIVES",
    "ShapingObjective",
    # private helpers kept reachable for the legacy shim / existing tests
    "_update_dynamic_weights",
    "_resolve_init_weights",
    "_resize_bilinear",
    "_rectangle_mask",
    "_SHAPE_ALIASES",
]
