"""SLM 相位变化速度计算工具 (phase-change-speed calculator).

用 SLM 发一个相位阶跃 (平场两灰度, 或闪耀光栅深度阶跃), 逐帧采集 CCD 远场
图像, 跟踪 0 级强度直到稳定, 提取响应时间 / 稳定时间 / 稳定帧数.

分层:
    * :mod:`core` —— 纯算法 (0 级指标 + 阶跃时序解算), 无设备依赖;
    * :mod:`measure` —— 硬件后端 (Santec SLM + 相机注册表), 落盘产物;
    * :mod:`__main__` —— ``python -m ao_shaping.tools.slm.phase_speed`` CLI.

⚠️ 测量的是 **SLM + CCD 组合系统** 的响应. 时间分辨率受相机帧率限制
(报告实测 fps 与单帧量化间隔), 不是孤立的 SLM 上升时间.
"""

from __future__ import annotations

from .core import PhaseStepResult, analyze_step, far_field_metric
from .measure import (
    SpeedMeasurementResult,
    build_pattern,
    run_speed_measurement,
    write_pattern,
)

__all__ = [
    "PhaseStepResult",
    "SpeedMeasurementResult",
    "analyze_step",
    "build_pattern",
    "far_field_metric",
    "run_speed_measurement",
    "write_pattern",
]
