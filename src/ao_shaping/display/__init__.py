"""显示与可视化包 (display & visualization).

统一暴露所有面向用户的帧渲染与交互窗口：相机/相位/截面三件套
(:mod:`ao_shaping.display.beam_profile`)、GUI 帧对象
(:mod:`~ao_shaping.display.frames`)、GS 动画 (:mod:`~ao_shaping.display.gs_visualization`)、
Zernike 校准与电压面板 (:mod:`~ao_shaping.display.image_display`) 与
各 Slm* 显示窗口 (:mod:`~ao_shaping.display.windows`)。
"""

from __future__ import annotations
from ao_shaping.display.beam_profile import (
    ProfileCut,
    cut_point,
    describe_artifact,
    frame_artifact_paths,
    plot_xy_profile,
    save_far_field,
    save_frame_artifacts,
    save_phase,
    save_xy_profile,
    select_epochs,
)
from ao_shaping.display.frames import (
    EpochCurveFrame,
    Image2DFrame,
    Image2DWithBucketFrame,
    MultiCurveFrame,
    VoltageFrame,
    save_best_image,
    to_display_uint8,
)
from ao_shaping.display.gs_visualization import (
    GSVizCallback,
    create_gs_iteration_frame,
    gerchberg_saxton_with_visualization,
    render_gs_animation,
    save_frames_as_gif,
)
from ao_shaping.display.image_display import (
    ImageVoltagesDisplay,
    plot_funcs,
    VOLT_HEIGHT,
    LOG_J_HEIGHT,
    BACKGROUND_COLOR,
    ZERN_UNSTABLE_COLOR,
    ZERN_MODERATE_COLOR,
    ZERN_STABLE_COLOR,
    ZERN_PROGRESS_BG,
    ZERN_PROGRESS_FILL,
    ZERN_TEXT_COLOR,
    ZERN_BG_COLOR,
    ZERN_BAR_DEFAULT_COLOR,
    ZernikeCalibrationDisplay,
)
from ao_shaping.display.windows import (
    AutoDisplay,
    DisplayClosedError,
    FrameInfo,
    SlmModelInLoopDisplay,
    SlmSquareDisplay,
    SlmZernikeDisplay,
    phase_for_display,
)

__all__ = (
    "AutoDisplay",
    "DisplayClosedError",
    "EpochCurveFrame",
    "FrameInfo",
    "GSVizCallback",
    "Image2DFrame",
    "Image2DWithBucketFrame",
    "MultiCurveFrame",
    "ImageVoltagesDisplay",
    "SlmModelInLoopDisplay",
    "SlmSquareDisplay",
    "SlmZernikeDisplay",
    "phase_for_display",
    "ProfileCut",
    "VoltageFrame",
    "beam_profile",
    "cut_point",
    "describe_artifact",
    "frame_artifact_paths",
    "plot_xy_profile",
    "save_far_field",
    "save_frame_artifacts",
    "save_phase",
    "save_xy_profile",
    "select_epochs",
    "plot_funcs",
    "VOLT_HEIGHT",
    "LOG_J_HEIGHT",
    "BACKGROUND_COLOR",
    "ZERN_UNSTABLE_COLOR",
    "ZERN_MODERATE_COLOR",
    "ZERN_STABLE_COLOR",
    "ZERN_PROGRESS_BG",
    "ZERN_PROGRESS_FILL",
    "ZERN_TEXT_COLOR",
    "ZERN_BG_COLOR",
    "ZERN_BAR_DEFAULT_COLOR",
    "ZernikeCalibrationDisplay",
    "create_gs_iteration_frame",
    "gerchberg_saxton_with_visualization",
    "render_gs_animation",
    "save_best_image",
    "save_frames_as_gif",
    "to_display_uint8",
    "windows",
    "beam_profile",
    "frames",
    "gs_visualization",
    "image_display",
)
