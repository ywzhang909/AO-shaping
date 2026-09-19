"""Backward-compat shim for :mod:`ao_shaping.utils.slm_lut`.

Real module moved to :mod:`ao_shaping.utils.slm.slm_lut`.
"""

from __future__ import annotations

from ao_shaping.utils.slm.slm_lut import *  # noqa: F403,F401
from ao_shaping.utils.slm.slm_lut import (  # noqa: F401
    LUTData,
    blaze_ramp_gray,
    build_inverse_lut,
    depth_pattern,
    invert_depth_scan,
    invert_offset_scan,
    load_lut,
    normalize_efficiency,
    offset_pattern,
    save_lut,
    sinc_inv,
    stack_halves,
)