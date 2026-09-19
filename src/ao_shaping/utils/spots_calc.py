"""Backward-compat shim for :mod:`ao_shaping.utils.spots_calc`.

Real module moved to :mod:`ao_shaping.utils.image.spots_calc`.
"""

from __future__ import annotations

from ao_shaping.utils.image.spots_calc import *  # noqa: F403,F401
from ao_shaping.utils.image.spots_calc import (  # noqa: F401
    calculate_sharpness,
    calculate_sharpness_cupy,
    calculate_sharpness_numba,
    center_of_brightness,
    center_of_brightness_cupy,
    center_of_brightness_numba,
    center_of_mass_cupy,
    center_of_mass_numba,
    center_of_mass_numpy,
    centroid,
    crop,
    crop_cupy,
    crop_numba,
    diffraction_limit,
    disp,
    effective_radius,
    jitter_diameter,
    make_coord,
    peak_position,
    pib_ratio_mask,
    power_bucket,
    power_in_bucket_mask,
    radius,
)