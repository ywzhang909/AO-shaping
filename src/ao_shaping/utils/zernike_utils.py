"""Backward-compat shim for :mod:`ao_shaping.utils.zernike_utils`.

Real module moved to :mod:`ao_shaping.utils.wavefront.zernike_utils`.
"""

from __future__ import annotations

from ao_shaping.utils.wavefront.zernike_utils import *  # noqa: F403,F401
from ao_shaping.utils.wavefront.zernike_utils import (  # noqa: F401
    coefficients_to_array,
    generate_zernike_phase,
    list_zernike_modes,
    parse_zernike_coefficients,
)