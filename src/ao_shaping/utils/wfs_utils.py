"""Backward-compat shim for :mod:`ao_shaping.utils.wfs_utils`.

Real module moved to :mod:`ao_shaping.utils.wavefront.wfs_utils`.
"""

from __future__ import annotations

from ao_shaping.utils.wavefront.wfs_utils import *  # noqa: F403,F401
from ao_shaping.utils.wavefront.wfs_utils import (  # noqa: F401
    DitheredReference,
    compute_snr,
    flatten_slopes,
    make_actuator_debug_callback,
    make_mode_debug_callback,
    save_debug_data,
    unflatten_slopes,
)