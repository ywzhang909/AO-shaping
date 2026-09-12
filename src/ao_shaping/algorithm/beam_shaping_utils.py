"""Shared beam-shaping utilities for SLM+CCD closed-loop runners (compat layer).

.. deprecated::
   This module is now a **backward-compatibility re-export facade**. All
   implementations have migrated to leaf ``ao_shaping.utils`` modules.
   New code must import from the new locations; this module keeps the old
   import paths working for existing runners/tests/scripts.

Migration map:
- Target-pattern generation (``create_target_shape`` / ``create_target_mask`` /
  ``load_target_image`` / ``square_target_from_measurement`` / ...) ->
  :mod:`ao_shaping.utils.targets`.
- Quality metrics (``compute_metrics`` / ``compute_shaping_metrics`` /
  ``compute_square_metrics`` / ``compute_quality_score`` /
  ``measure_spot_diameter_cam`` / ``measure_bright_span`` /
  ``normalize_pattern`` / ``intensity_to_amplitude`` / ``clamp_side``) ->
  :mod:`ao_shaping.utils.beam_metrics`.
- SLM phase conversion / memory-slot rotation (``phase_to_slm_grayscale`` /
  ``pick_slm_slot`` / ``display_phase`` + physical constants) ->
  :mod:`ao_shaping.utils.slm_utils`.
- Hardware helpers (``capture_amplitude`` / ``call_with_timeout`` /
  auto-exposure / frame recording) -> :mod:`ao_shaping.utils.hardware_utils`.
"""

from __future__ import annotations

# fmt: off
# Target-pattern generation -> ``ao_shaping.utils.targets``
from ao_shaping.utils.targets import (
    build_square_target_amplitude,
    compute_square_side,
    create_target_mask,
    create_target_shape,
    crop_resize_to_grid,
    load_target_image,
    square_target_from_measurement,
)
# Quality metrics -> ``ao_shaping.utils.beam_metrics``
from ao_shaping.utils.beam_metrics import (
    clamp_side,
    compute_metrics,
    compute_quality_score,
    compute_shaping_metrics,
    compute_square_metrics,
    intensity_to_amplitude,
    measure_bright_span,
    measure_spot_diameter_cam,
    normalize_pattern,
)
# SLM phase / slot rotation + physical constants -> ``ao_shaping.utils.slm_utils``
from ao_shaping.utils.slm_utils import (
    DEFAULT_DISTANCE,
    DEFAULT_MAX_GRAYSCALE,
    DEFAULT_SLM_PIXEL_SIZE,
    DEFAULT_WAVELENGTH,
    display_phase,
    phase_to_slm_grayscale,
    pick_slm_slot,
)
# Hardware helpers -> ``ao_shaping.utils.hardware_utils``
from ao_shaping.utils.hardware_utils import (
    apply_auto_exposure,
    auto_exposure_possible,
    auto_exposure_target_ms,
    call_with_timeout,
    capture_amplitude,
    init_frame_recording,
    record_frame,
    save_frame_png,
)
# fmt: on

# The legacy ``parse_tuple(value)`` helper had no callers after the CLI parse
# helper (:func:`ao_shaping.utils.cli_helpers.parse_tuple`, click callback)
# became the standard; it was removed during migration.

__all__ = [
    # targets
    "build_square_target_amplitude",
    "compute_square_side",
    "create_target_mask",
    "create_target_shape",
    "crop_resize_to_grid",
    "load_target_image",
    "square_target_from_measurement",
    # beam_metrics
    "clamp_side",
    "compute_metrics",
    "compute_quality_score",
    "compute_shaping_metrics",
    "compute_square_metrics",
    "intensity_to_amplitude",
    "measure_bright_span",
    "measure_spot_diameter_cam",
    "normalize_pattern",
    # slm_utils
    "DEFAULT_DISTANCE",
    "DEFAULT_MAX_GRAYSCALE",
    "DEFAULT_SLM_PIXEL_SIZE",
    "DEFAULT_WAVELENGTH",
    "display_phase",
    "phase_to_slm_grayscale",
    "pick_slm_slot",
    # hardware_utils
    "apply_auto_exposure",
    "auto_exposure_possible",
    "auto_exposure_target_ms",
    "call_with_timeout",
    "capture_amplitude",
    "init_frame_recording",
    "record_frame",
    "save_frame_png",
]