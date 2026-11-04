"""``PatternHelper`` 相位生成器在独立 tab 下的直接暴露（直通 API，不建控件树）。

为什么单独成模块
----------------
``PatternHelper``（:mod:`ao_shaping.utils.wavefront.pattern_helper`）是纯相位
发生器，已被控制器经 ``multi_slm_controller`` 的 ``PATTERN_REGISTRY`` /
``PatternControl`` 控件树使用。那个控件树仍在「相位控制」tab 里，本模块**不碰**它，
而是把 ``PatternHelper`` 的生成方法**直接**暴露成一张「相位生成」tab：选图案类型 →
调参数 → 「生成预览」（``helper.to_uint16`` + ``st.image``）→ 「下发到 SLM」
（``slm.create_phase_from_array`` 弧度→灰度 + shift/矫正/LUT → ``slm.display_data``），
与控制器同一条下发路径，**不**二次施加平移。

设计契约
--------
* 入口 :func:`render_pattern_helper_tab`，父 ``main()`` 重构后的 tab 调用它。
* 纯/离线部分（类型注册表 :data:`PATTERN_TYPES`、参数规格 :data:`ParamSpec`、
  :func:`generate_phase_radians`、:data:`CONNECT_HINT`）与 ``st.*`` 调用隔离，
  可脱离 Streamlit 做单元测试。这套纯规格的**单一真源**抽在
  :mod:`~ao_shaping.gui.slm.pattern_specs`（不 import Streamlit），PySide6 桌面页
  与本 tab 共享同一份；本模块透传并保留公开名。
* 遵循 gui/slm 三纪律：不调 ``st.divider()``；本模块不建 ``type="primary"`` 按钮；
  空状态文案一律点名下一步（侧重栏表格的「连接」：:data:`CONNECT_HINT`）。
* 每个 SLM 复用 ``session_state["slm{N}_pattern_helper"]`` 上的缓存
  （与控制器 ``_get_slm_helper`` 同一键、同一面板几何），不重复建坐标网格。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.gui.slm.pattern_specs import (
    MethodCall,
    ParamSpec,
    PatternTypeSpec,
    PATTERN_TYPES,
    PARAM_SPECS,
    generate_phase_radians,
    specs_for,
)

# 侧重栏「SLM 设备」表格的引导语（与 multi_slm_controller.CONNECT_HINT 同文）。
CONNECT_HINT = "在左侧「SLM 设备」表格里点「连接」"

#: 未连接时的默认面板几何（与控制器 ``_get_slm_helper`` 的回退一致）。
_DEFAULT_PANEL_RES: tuple[int, int] = (1920, 1200)
_DEFAULT_BITS: int = 10

#: 每个 SLM 复用 / 重建的 PatternHelper 在 session_state 里的缓存键。
HELPER_CACHE_KEY = "slm{slm_num}_pattern_helper"

#: 重导出给既有消费者（测试 / 其他模块直接 ``from helper_tab import ...``）。
#: 纯规格单一真源在 :mod:`~ao_shaping.gui.slm.pattern_specs`，本模块只透传。
#: ``_specs_for`` 是本地渲染用别名，保留原名以免移动破坏既有调用。
__all__ = [
    "CONNECT_HINT",
    "MethodCall",
    "ParamSpec",
    "PatternTypeSpec",
    "PATTERN_TYPES",
    "PARAM_SPECS",
    "HELPER_CACHE_KEY",
    "generate_phase_radians",
    "render_pattern_helper_tab",
]


def _specs_for(key: str) -> tuple[ParamSpec, ...]:
    """返回图案类型 key 的参数规格（缺省空元组）。"""
    return specs_for(key)


def _slm_panel_geometry(slm: Any) -> tuple[int, int, int]:
    """从 SLM 对象取面板几何，返回 ``(width, height, bits)``。

    与控制器 ``_build_control`` 一致：``Panel_Res`` 为 ``(width, height)``，
    位深取 ``Gray_Scale_bits``，缺省回退到 `_DEFAULT_PANEL_RES` / `_DEFAULT_BITS`。
    """
    panel_res = getattr(slm, "Panel_Res", _DEFAULT_PANEL_RES)
    width, height = int(panel_res[0]), int(panel_res[1])
    bits = int(getattr(slm, "Gray_Scale_bits", _DEFAULT_BITS))
    return width, height, bits


def _get_helper(slm_num: int, slm: Any) -> PatternHelper:
    """返回 SLM 的缓存 :class:`PatternHelper`（复用 ``session_state`` 缓存键）。

    与控制器 :func:`_get_slm_helper` 同键 ``slm{N}_pattern_helper``，面板几何或
    位深变化（例如重连不同设备）时重建。
    """
    width, height, bits = _slm_panel_geometry(slm)
    key = HELPER_CACHE_KEY.format(slm_num=slm_num)
    helper = st.session_state.get(key)
    if helper is None or helper.resolution != (width, height) or helper.bits != bits:
        helper = PatternHelper((width, height), bits=bits)
        st.session_state[key] = helper
    return helper


def _render_param(
    spec: ParamSpec,
    prefix: str,
) -> float | int | str | None:
    """渲染单个参数 widget，返回其当前值（随 Streamlit 默认逻辑读 session_state）。"""
    key = f"{prefix}_helper_{spec.name}"
    if spec.kind == "select":
        return st.selectbox(
            spec.label, options=spec.options, index=_index_of(spec), key=key
        )
    if spec.kind == "seed":
        value: Any = st.session_state.get(key, spec.default)
        return st.number_input(
            spec.label,
            value=value,
            min_value=spec.min,
            step=1,
            key=key,
        )
    if spec.kind == "int":
        return st.number_input(
            spec.label,
            value=int(spec.default) if spec.default is not None else None,
            min_value=spec.min,
            max_value=spec.max,
            step=int(spec.step) if spec.step is not None else 1,
            key=key,
        )
    return st.number_input(
        spec.label,
        value=float(spec.default) if spec.default is not None else None,
        min_value=spec.min,
        max_value=spec.max,
        step=spec.step,
        key=key,
    )


def _index_of(spec: ParamSpec) -> int | None:
    default = spec.default
    assert isinstance(default, str), "select 参数 default 必须是 str"
    if default in spec.options:
        return spec.options.index(default)
    return None


def _render_params(prefix: str, pattern_type: str) -> dict[str, Any]:
    """按注册表渲染 ``pattern_type`` 的全部参数 widget。"""
    specs = _specs_for(pattern_type)
    return {spec.name: _render_param(spec, prefix) for spec in specs}


def _display_phase(slm: Any, phase_radians: np.ndarray, slm_num: int) -> None:
    """把弧度相位经驱动下发到面板（与控制器同一条路径，不二次施加平移）。

    用 ``slm.create_phase_from_array`` 完成 弧度→灰度（内部含 shift + 矫正 +
    LUT），再 ``display_data`` 写入并立即显示 —— 平移只在这里应用一次。
    """
    phase_gray = slm.create_phase_from_array(phase_radians)
    slm.display_data(phase_gray)


def render_pattern_helper_tab(slm_num: int) -> None:
    """tab 入口：直接暴露 :class:`PatternHelper` 的相位生成器。

    Args:
        slm_num: SLM 序号，session_state 前缀 ``slm{slm_num}``。
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is None:
        st.info(f"尚未连接 SLM —— {CONNECT_HINT}, 连上后这里出现相位生成。")
        return

    helper = _get_helper(slm_num, slm)
    width, height, bits = _slm_panel_geometry(slm)
    st.caption(
        f"PatternHelper @ {width}×{height} ×{bits} bit（分辨率 {helper.resolution}）"
    )

    pattern_keys = list(PATTERN_TYPES)
    pattern_type: str = st.selectbox(
        "图案类型",
        options=pattern_keys,
        index=0,
        key=f"{prefix}_helper_pattern_type",
        format_func=lambda k: PATTERN_TYPES[k].label,
    )

    params = _render_params(prefix, pattern_type)

    generated: np.ndarray | None = None
    if st.button("生成预览", key=f"{prefix}_helper_preview_btn", type="secondary"):
        try:
            phase = generate_phase_radians(helper, pattern_type, params)
            if pattern_type == "turbulence":
                st.info("湍流相位屏已按当前 r0/L0 重新初始化解偶生成。")
            gray = helper.to_uint16(phase)
            st.image(
                gray,
                caption=f"图案「{PATTERN_TYPES[pattern_type].label}」预览",
                clamp=True,
            )
            generated = phase
        except Exception as e:
            st.error(f"生成相位失败: {e}")
            logger.exception(f"Pattern helper preview failed for SLM {slm_num}: {e}")

    if st.button("下发到 SLM", key=f"{prefix}_helper_display_btn", type="secondary"):
        phase = (
            generated
            if generated is not None
            else generate_phase_radians(helper, pattern_type, params)
        )
        try:
            _display_phase(slm, phase, slm_num)
            st.success(
                f"相位已下发到 SLM {slm_num} 并显示（灰度经 create_phase_from_array 编码）。"
            )
        except Exception as e:
            st.error(f"下发失败: {e} —— 请先回到「生成预览」确认参数后重试。")
            logger.exception(f"Pattern helper display failed for SLM {slm_num}: {e}")
