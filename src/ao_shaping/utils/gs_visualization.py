"""Backward-compat shim for :mod:`ao_shaping.utils.gs_visualization`.

Real module moved to :mod:`ao_shaping.utils.image.gs_visualization`.
"""

from __future__ import annotations

from ao_shaping.utils.image.gs_visualization import *  # noqa: F403,F401
from ao_shaping.utils.image.gs_visualization import (  # noqa: F401
    GSVizCallback,
    create_gs_iteration_frame,
    gerchberg_saxton_with_visualization,
    render_gs_animation,
    save_frames_as_gif,
)