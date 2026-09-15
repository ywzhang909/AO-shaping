"""Santec SLM 驱动子包

Santec 空间光调制器 (SLM-200 / SLM-300 等型号共用) 驱动的统一入口。
``driver`` 提供驱动类 :class:`Santec` 与异常 :class:`SantecError`;
``constants`` 提供驱动层 (非设备型号专属) 常量; ``slm200_constants``
提供 SLM-200 专属设备硬件参数; ``_slm_win`` 为 Windows SDK bindings;
``wavefront_correction`` 提供波前误差校正管线。
"""

from __future__ import annotations

from ao_shaping.drivers.slm.santec.constants import (
    DEFAULT_SHIFT_X,
    DEFAULT_SHIFT_Y,
    DEFAULT_WAVELENGTH,
    DVI_MODE,
    FLAGS_RATE120,
    GRAYSCALE_MAX,
    GRAYSCALE_MIN,
    MAX_MEM_SLOTS,
    MEMORY_MODE_INTERNAL,
    MEMORY_NUMBER_MAX,
    MEMORY_NUMBER_MIN,
    SLM_ERROR_MESSAGES,
    SLM_NUMBER_MAX,
    SLM_NUMBER_MIN,
    SLM_OK,
    SLMErrorCode,
    VideoMode,
    WAVELENGTH_MAX,
    WAVELENGTH_MIN,
    get_slm_error_message,
)
from ao_shaping.drivers.slm.santec.driver import (
    MAX_PIXEL_FLIP_TIME_S,
    Santec,
    SantecError,
    SLM_CONFIG,
    SLMParams,
    apply_lut_remap,
)
from ao_shaping.drivers.slm.santec.slm200_constants import (
    GRAY_SCALE_BITS,
    MAX_PIXEL_FLIP_TIME_MS,
    PANEL_RES,
    PANEL_SIZE_MM,
    PITCH_UM,
    PIXEL_SIZE_UM,
    RESPONSE_TIME_MS,
    get_max_grayscale,
)
from ao_shaping.drivers.slm.santec.wavefront_correction import WavefrontCorrection

__all__ = [
    "DEFAULT_SHIFT_X",
    "DEFAULT_SHIFT_Y",
    "DEFAULT_WAVELENGTH",
    "DVI_MODE",
    "FLAGS_RATE120",
    "GRAY_SCALE_BITS",
    "GRAYSCALE_MAX",
    "GRAYSCALE_MIN",
    "MAX_MEM_SLOTS",
    "MAX_PIXEL_FLIP_TIME_MS",
    "MAX_PIXEL_FLIP_TIME_S",
    "MEMORY_MODE_INTERNAL",
    "MEMORY_NUMBER_MAX",
    "MEMORY_NUMBER_MIN",
    "PANEL_RES",
    "PANEL_SIZE_MM",
    "PITCH_UM",
    "PIXEL_SIZE_UM",
    "RESPONSE_TIME_MS",
    "SLM_CONFIG",
    "SLM_ERROR_MESSAGES",
    "SLM_NUMBER_MAX",
    "SLM_NUMBER_MIN",
    "SLM_OK",
    "SLMErrorCode",
    "SLMParams",
    "Santec",
    "SantecError",
    "VideoMode",
    "WAVELENGTH_MAX",
    "WAVELENGTH_MIN",
    "WavefrontCorrection",
    "apply_lut_remap",
    "get_max_grayscale",
    "get_slm_error_message",
]