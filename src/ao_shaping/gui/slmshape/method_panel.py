"""「参数 → 校验 → 调试开关 + 运行按钮」这一段的共同编排。

顺序是契约：校验必须早于按钮，按钮才能 ``disabled``。本模块只负责通用控件；
调试开关的说明文案由调用方传入（两页的 help 不同）。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ao_shaping.gui.slm import bench_settings
from ao_shaping.gui.slm.shaping_controls import ShapingMethod

__all__ = [
    "render_method_inputs",
    "render_method_validation",
    "render_run_controls",
]


def render_method_inputs(method: ShapingMethod, layout: Any) -> dict[str, Any]:
    """目标与预算 → 高级参数，并并入侧栏光路值（光路控件只在侧栏）。"""
    with layout:
        values = method.render_inputs()
    bench_settings.merge_bench_values(
        values, method.prefix, method.slm_num, method.key
    )
    method.hint_bench_calibration(values)
    return values


def render_method_validation(
    method: ShapingMethod, values: dict[str, Any], layout: Any
) -> list[str]:
    """校验区必须在按钮之前，按钮才能 ``disabled``。"""
    errors: list[str] = []
    bench_error = bench_settings.validate_bench_inputs()
    if bench_error:
        errors.append(bench_error)
    method_error = method.validate(values)
    if method_error:
        errors.append(method_error)
    with layout:
        for error in errors:
            st.error(error)
    return errors


def render_run_controls(
    method: ShapingMethod, errors: list[str], layout: Any, *, debug_help: str
) -> tuple[bool, bool]:
    """调试开关 + 运行按钮。返回 ``(clicked, debug)``。

    Args:
        debug_help: 调试开关的 help 文案（两页不同）。
    """
    with layout:
        debug = st.checkbox(
            "保存调试产物 (data/debug/…)",
            value=False,
            key=f"{method.prefix}_shp_debug",
            disabled=bool(errors),
            help=debug_help,
        )
        clicked = st.button(
            method.action,
            key=f"{method.prefix}_shp_run_{method.key}",
            type="primary",
            disabled=bool(errors) or bench_settings.bench_busy(),
            use_container_width=True,
        )
    return bool(clicked), bool(debug)
