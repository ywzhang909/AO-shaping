"""Backward-compat shim for :mod:`ao_shaping.utils.resample`.

Real module moved to :mod:`ao_shaping.utils.image.resample`.
"""

from __future__ import annotations

from ao_shaping.utils.image.resample import *  # noqa: F403,F401
from ao_shaping.utils.image.resample import (  # noqa: F401
    crop_to_aspect,
    resample_to_grid,
)