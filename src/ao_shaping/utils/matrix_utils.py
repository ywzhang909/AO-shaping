"""Backward-compat shim for :mod:`ao_shaping.utils.matrix_utils`.

Real module moved to :mod:`ao_shaping.utils.wavefront.matrix_utils`.
"""

from __future__ import annotations

from ao_shaping.utils.wavefront.matrix_utils import *  # noqa: F403,F401
from ao_shaping.utils.wavefront.matrix_utils import (  # noqa: F401
    calc_n_zernike_terms,
    compute_lstsq,
    compute_pinv,
    index_to_noll,
    noll_to_index,
)