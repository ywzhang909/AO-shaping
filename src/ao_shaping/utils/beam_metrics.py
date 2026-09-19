"""Backward-compat shim for :mod:`ao_shaping.utils.beam_metrics`.

Real module moved to :mod:`ao_shaping.utils.image.beam_metrics`.
"""

from __future__ import annotations

from ao_shaping.utils.image.beam_metrics import *  # noqa: F403,F401
from ao_shaping.utils.image.beam_metrics import (  # noqa: F401
    clamp_side,
    compute_metrics,
    compute_quality_score,
    compute_shaping_metrics,
    compute_square_metrics,
    intensity_to_amplitude,
    measure_bright_span,
    measure_spot_diameter_cam,
    normalize_pattern,
)