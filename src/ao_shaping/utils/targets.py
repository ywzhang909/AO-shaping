"""Backward-compat shim for :mod:`ao_shaping.utils.targets`.

Real module moved to :mod:`ao_shaping.utils.image.targets`.
"""

from __future__ import annotations

from ao_shaping.utils.image.targets import *  # noqa: F403,F401
from ao_shaping.utils.image.targets import (  # noqa: F401
    build_ccd_target,
    build_square_target_amplitude,
    build_target_from_frame,
    compute_square_side,
    create_target_mask,
    create_target_shape,
    crop_resize_to_grid,
    load_target_image,
    square_target_from_measurement,
)