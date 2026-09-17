"""Backward-compat shim for :mod:`ao_shaping.utils.zernike_calc`.

Real module moved to :mod:`ao_shaping.utils.wavefront.zernike_calc`.
"""

from __future__ import annotations

from ao_shaping.utils.wavefront.zernike_calc import *  # noqa: F403,F401
from ao_shaping.utils.wavefront.zernike_calc import (  # noqa: F401
    ZernikeGenerator,
    calc_n_zernike_terms,
    fit_zernike,
    generate_noll_polynomial,
    get_zernike_name,
    noll_to_nm,
    zernike_radial,
)