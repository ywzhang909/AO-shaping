"""整形任务的纯路由决策：探测报告 → 推荐整形路线。

纯离线：只依赖标准库，无设备 / torch / Qt，可在无硬件环境单测。

导出 ``ProbeReport`` / ``RoutePlan`` / ``choose_route`` / ``plan_to_dict`` /
``format_plan`` 与路由阈值常量。
"""
from __future__ import annotations

from ao_shaping.tools.slm.shaping.router import (
    GEOMETRY_CORRELATION_MIN,
    MODEL_ANCHOR_ERROR_MAX,
    SNR_MIN,
    ZERO_SHIFT_PX_MAX,
    ProbeReport,
    RoutePlan,
    choose_route,
    format_plan,
    plan_to_dict,
)

__all__ = [
    "GEOMETRY_CORRELATION_MIN",
    "MODEL_ANCHOR_ERROR_MAX",
    "SNR_MIN",
    "ZERO_SHIFT_PX_MAX",
    "ProbeReport",
    "RoutePlan",
    "choose_route",
    "format_plan",
    "plan_to_dict",
]