"""SLM+CCD 闭环 runner 共用的光束整形工具 (兼容层)。

.. deprecated::
   本模块现在是一个 **向后兼容的 re-export 门面**。所有实现都已迁移到叶子层
   ``ao_shaping.utils`` 模块。新代码必须从新位置导入; 本模块保留旧导入路径,
   使现有 runner / 测试 / 脚本继续可用。

迁移对照表:
- 目标图形生成 (``create_target_shape`` / ``create_target_mask`` /
  ``load_target_image`` / ``square_target_from_measurement`` / ...) ->
  :mod:`ao_shaping.utils.targets`.
- 质量指标 (``compute_beam_metrics`` / ``compute_shaping_metrics`` /
  ``compute_square_metrics`` / ``compute_quality_score`` /
  ``measure_spot_diameter_cam`` / ``measure_bright_span`` /
  ``normalize_pattern`` / ``intensity_to_amplitude`` / ``clamp_side``) ->
  :mod:`ao_shaping.utils.beam_metrics`.
- SLM 相位转换 / 存储槽轮转 (``phase_to_slm_grayscale`` /
  ``pick_slm_slot`` / ``display_phase`` + 物理常量) ->
  :mod:`ao_shaping.utils.slm.phase_display`.
- 硬件辅助 (``capture_amplitude`` / ``call_with_timeout`` /
  自动曝光 / 帧录制) -> :mod:`ao_shaping.utils.image.hardware_utils`.
"""

from __future__ import annotations

# fmt: off
# 目标图形生成 -> ``ao_shaping.utils.targets``
from ao_shaping.utils.image.targets import (
    build_square_target_amplitude,
    compute_square_side,
    create_target_mask,
    create_target_shape,
    crop_resize_to_grid,
    load_target_image,
    square_target_from_measurement,
)
# 质量指标 -> ``ao_shaping.utils.beam_metrics``
from ao_shaping.utils.image.beam_metrics import (
    clamp_side,
    compute_beam_metrics,
    compute_quality_score,
    compute_shaping_metrics,
    compute_square_metrics,
    intensity_to_amplitude,
    measure_bright_span,
    measure_spot_diameter_cam,
    normalize_pattern,
)
# SLM 相位 / 存储槽轮转 + 物理常量 -> ``ao_shaping.utils.slm.phase_display``
from ao_shaping.utils.slm.phase_display import (
    DEFAULT_DISTANCE,
    DEFAULT_MAX_GRAYSCALE,
    DEFAULT_SLM_PIXEL_SIZE,
    DEFAULT_WAVELENGTH,
    display_phase,
    phase_to_slm_grayscale,
    pick_slm_slot,
)
# 硬件辅助 -> ``ao_shaping.utils.image.hardware_utils``
from ao_shaping.utils.image.hardware_utils import (
    apply_auto_exposure,
    auto_exposure_possible,
    auto_exposure_target_ms,
    call_with_timeout,
    capture_amplitude,
    init_frame_recording,
    record_frame,
    save_frame_png,
)
# fmt: on

# 遗留的 ``parse_tuple(value)`` 辅助函数在 CLI 解析辅助
# (:func:`ao_shaping.utils.io.cli_helpers.parse_tuple`, click 回调)
# 成为标准做法后就没有调用方了; 已在迁移期间删除。

__all__ = [
    # targets
    "build_square_target_amplitude",
    "compute_square_side",
    "create_target_mask",
    "create_target_shape",
    "crop_resize_to_grid",
    "load_target_image",
    "square_target_from_measurement",
    # beam_metrics
    "clamp_side",
    "compute_beam_metrics",
    "compute_quality_score",
    "compute_shaping_metrics",
    "compute_square_metrics",
    "intensity_to_amplitude",
    "measure_bright_span",
    "measure_spot_diameter_cam",
    "normalize_pattern",
    # phase_display
    "DEFAULT_DISTANCE",
    "DEFAULT_MAX_GRAYSCALE",
    "DEFAULT_SLM_PIXEL_SIZE",
    "DEFAULT_WAVELENGTH",
    "display_phase",
    "phase_to_slm_grayscale",
    "pick_slm_slot",
    # hardware_utils
    "apply_auto_exposure",
    "auto_exposure_possible",
    "auto_exposure_target_ms",
    "call_with_timeout",
    "capture_amplitude",
    "init_frame_recording",
    "record_frame",
    "save_frame_png",
]
