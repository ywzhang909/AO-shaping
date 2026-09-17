"""Backward-compat shim for :mod:`ao_shaping.utils.hadamard_calc`.

Real module moved to :mod:`ao_shaping.utils.wavefront.hadamard_calc`.
"""

from __future__ import annotations

from ao_shaping.utils.wavefront.hadamard_calc import *  # noqa: F403,F401
from ao_shaping.utils.wavefront.hadamard_calc import (  # noqa: F401
    HadamardGenerator,
    calc_n_hadamard_modes,
    hadamard_mode_2d,
    is_hadamard_order,
)