"""Backward-compat shim for :mod:`ao_shaping.utils.display`.

Real module moved to :mod:`ao_shaping.utils.image.display`.
"""

from __future__ import annotations

from ao_shaping.utils.image.display import *  # noqa: F403,F401
from ao_shaping.utils.image.display import (  # noqa: F401
    BACKGROUND_COLOR,
    ImageVoltagesDisplay,
    LINE_COLOR,
    LOG_J_HEIGHT,
    VOLT_HEIGHT,
    ZERN_BAR_DEFAULT_COLOR,
    ZERN_BG_COLOR,
    ZERN_MODERATE_COLOR,
    ZERN_PROGRESS_BG,
    ZERN_PROGRESS_FILL,
    ZERN_STABLE_COLOR,
    ZERN_TEXT_COLOR,
    ZERN_UNSTABLE_COLOR,
    ZernikeCalibrationDisplay,
    plot_funcs,
    plot_img,
    plot_log_j,
    plot_pib_history,
    plot_rms_history,
    plot_voltages,
    plot_voltage_comparison,
    plot_voltage_heatmap,
    plot_wavefront,
)