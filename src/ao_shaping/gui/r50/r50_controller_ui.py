"""R50 控制器控制面板 (Streamlit) — 薄编排层。

面向 ``MicroDM.py`` 中 :class:`R50Controller` / :class:`MicroDM` 的 UI。

可测试逻辑位于兄弟模块:
- r50_channel_select: 配置 / CSV 索引 / 通道选择 (纯逻辑)
- r50_connection:     连接工厂 / 仿真设备 / 下电安全 (纯逻辑)
- r50_voltage_send:   裁剪 / 批量下发 / 发送循环 (纯逻辑)

UI 渲染与动作逻辑按功能域拆分到兄弟模块:
- r50_debug:     调试 TCP 客户端 / 指令日志 / 操作日志 / 本地调试服务器
- r50_common:    反馈 / session_state 初始化 / 通道标签 / 发送循环管理
- r50_single:    单控制器 (连通性 / 连接 / 继电器 / 单次下发)
- r50_joint:     联合控制 (MicroDM 矩阵 / 编辑 / 可视化)
- r50_group:     分组控制 (组连接 / 下发 / 批量上下电)
- r50_units:     单单元控制 Tab
- r50_tabs:      三个控制 Tab 的渲染 (单控制器 / 单组 / 全部控制)
- r50_sidebar:   Sidebar (调试面板 / 连接配置 / 三种模式连接界面)

本文件只负责组装: session_state 初始化、循环反馈排空与 Tab 布局。

使用方式:
    streamlit run src/ao_shaping/gui/r50/r50_controller_ui.py
"""

from __future__ import annotations

import streamlit as st

from ao_shaping.gui.r50.r50_common import (
    _drain_loop_feedback,
    _initialize_state,
)
from ao_shaping.gui.r50.r50_debug import _drain_local_debug_buffer
from ao_shaping.gui.r50.r50_sidebar import _sidebar_connection_config
from ao_shaping.gui.r50.r50_tabs import (
    render_tab_all_control,
    render_tab_single_controller,
    render_tab_single_group,
)
from ao_shaping.gui.r50.r50_units import render_tab_single_unit


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="R50 控制器控制面板",
        page_icon="🔌",
        layout="wide",
    )

    _initialize_state()
    _drain_loop_feedback()
    _drain_local_debug_buffer()

    _sidebar_connection_config()

    tab_su, tab_sc, tab_sg, tab_ac = st.tabs([
        "💠 单单元控制",
        "🔌 单控制器控制",
        "🧩 单组控制",
        "🔗 全部控制",
    ])

    with tab_su:
        render_tab_single_unit()

    with tab_sc:
        render_tab_single_controller()

    with tab_sg:
        render_tab_single_group()

    with tab_ac:
        render_tab_all_control()


if __name__ == "__main__":
    main()
