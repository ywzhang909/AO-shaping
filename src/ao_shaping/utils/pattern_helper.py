"""Backward-compat shim for :mod:`ao_shaping.utils.pattern_helper`.

Real module moved to :mod:`ao_shaping.utils.slm.pattern_helper`.
"""

from __future__ import annotations

from ao_shaping.utils.slm.pattern_helper import *  # noqa: F403,F401
from ao_shaping.utils.slm.pattern_helper import (  # noqa: F401
    UNWRAP_STRATEGY,
    WRAP_STRATEGY,
    PatternHelper,
    PhaseUnwrapperHelper,
    PhaseWrapOptimizerHelper,
    calc_blazed_grating_period,
)