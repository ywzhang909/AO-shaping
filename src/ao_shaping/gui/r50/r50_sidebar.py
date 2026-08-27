"""R50 Sidebar 模块 — 调试面板 / 连接配置 / 三种模式的连接界面。

被入口 ``r50_controller_ui.main`` 调用。
"""

from __future__ import annotations

import collections

import pandas as pd
import streamlit as st

from ao_shaping.gui.r50.r50_channel_select import (
    HW_VOLTAGE_MAX,
    HW_VOLTAGE_MIN,
    P,
)
from ao_shaping.gui.r50.r50_group import (
    _gc_batch_power_off,
    _gc_batch_power_on,
    _gc_connect,
    _gc_disconnect,
    _gc_set_relay,
)
from ao_shaping.gui.r50.r50_joint import (
    _jc_batch_power_off,
    _jc_batch_power_on,
    _jc_connect,
    _jc_disconnect,
    _jc_set_relay,
)
from ao_shaping.gui.r50.r50_single import (
    connect,
    disconnect,
    set_relay_power,
    test_connectivity,
)
from ao_shaping.gui.r50.r50_units import _channel_info_to_dict


# =============================================================================
# Sidebar: 调试面板
# =============================================================================


def _sidebar_debug_panel() -> None:
    """Sidebar 调试面板: 仿真状态 + 指令日志 + 操作日志。"""
    with st.container(border=True):
        st.markdown("##### 🐛 调试面板")

        sim_single = st.session_state.get(f"{P}_simulate", False)
        sim_joint = st.session_state.get(f"{P}_jc_simulate", False)
        sim_group = st.session_state.get(f"{P}_gc_simulate", False)
        if sim_single or sim_joint or sim_group:
            parts = []
            if sim_single:
                parts.append("单控制器")
            if sim_joint:
                parts.append("联合控制")
            if sim_group:
                parts.append("分组控制")
            st.info(f"🟡 仿真模式: {', '.join(parts)}")
        else:
            st.caption("仿真模式未启用")

        _debug_enabled = st.checkbox(
            "指令日志",
            value=st.session_state[f"{P}_debug"],
            key=f"{P}_debug_pkt_enable_sb",
            help="显示下发的指令包十六进制内容",
        )
        st.session_state[f"{P}_debug"] = _debug_enabled

        with st.expander("操作日志", expanded=False):
            op_log: collections.deque = st.session_state.get(
                f"{P}_debug_op_log", collections.deque()
            )
            if op_log:
                st.code("\n".join(op_log), language="text")
            else:
                st.caption("暂无操作日志")
            if st.button(
                "清空日志", key=f"{P}_debug_op_clear_sb", use_container_width=True
            ):
                st.session_state[f"{P}_debug_op_log"].clear()
                st.rerun()

        with st.expander("指令日志 (下发包记录)", expanded=False):
            log_lines = list(st.session_state.get(f"{P}_debug_log", []))
            if log_lines:
                st.code("\n".join(log_lines), language="text")
            else:
                st.caption("暂无指令记录")
            if st.button(
                "清空指令日志", key=f"{P}_debug_log_clear_sb", use_container_width=True
            ):
                st.session_state[f"{P}_debug_log"].clear()
                st.rerun()


# =============================================================================
# Sidebar: 连接配置 (三种模式统一入口)
# =============================================================================


def _sidebar_connection_config() -> None:
    """Sidebar 连接配置: 三种连接模式的统一入口。"""
    with st.sidebar:
        with st.container(border=True):
            st.markdown("##### 连接配置")
            mode = st.radio(
                "连接模式",
                options=["single", "joint", "group"],
                format_func={
                    "single": "🔌 单控制器",
                    "joint": "🔗 联合控制 (所有控制器)",
                    "group": "🧩 分组控制",
                }.get,
                index=["single", "joint", "group"].index(
                    st.session_state.get(f"{P}_connection_mode", "single")
                ),
                key=f"{P}_conn_mode_radio",
            )
            st.session_state[f"{P}_connection_mode"] = mode
        st.divider()

        if mode == "single":
            _sidebar_single_connection()
        elif mode == "joint":
            _sidebar_joint_connection()
        elif mode == "group":
            _sidebar_group_connection()

        st.divider()
        with st.container(border=True):
            st.markdown("##### 电压安全范围 (允许范围)")
            col_min, col_max = st.columns(2)
            with col_min:
                vmin = st.number_input(
                    "下限 (V)",
                    min_value=HW_VOLTAGE_MIN,
                    max_value=HW_VOLTAGE_MAX,
                    value=st.session_state[f"{P}_vmin"],
                    step=1.0,
                    format="%.1f",
                    key=f"{P}_vmin_input_sb",
                )
            with col_max:
                vmax = st.number_input(
                    "上限 (V)",
                    min_value=HW_VOLTAGE_MIN,
                    max_value=HW_VOLTAGE_MAX,
                    value=st.session_state[f"{P}_vmax"],
                    step=1.0,
                    format="%.1f",
                    key=f"{P}_vmax_input_sb",
                )
            if vmin >= vmax:
                st.warning("⚠️ 电压下限必须小于上限")
            st.session_state[f"{P}_vmin"] = vmin
            st.session_state[f"{P}_vmax"] = vmax

        _sidebar_debug_panel()


def _sidebar_single_connection() -> None:
    """Sidebar 单控制器连接配置。"""
    with st.sidebar:
        with st.container(border=True):
            _connected = st.session_state[f"{P}_connected"]
            st.markdown("##### 当前状态")
            if _connected:
                _ip = st.session_state.get(f"{P}_ip", "")
                _port = st.session_state.get(f"{P}_port", 0)
                _ctrl_num = st.session_state.get(f"{P}_controller_num", 1)
                st.success(f"✅ 已连接  {_ip}:{_port} (控制器 #{_ctrl_num})")
            else:
                st.error("❌ 未连接")
            if st.session_state[f"{P}_relay_on"]:
                st.success("⚡ 继电器上电 (输出接通)")
            else:
                st.warning("⏻ 继电器下电 (输出断开)")
            if st.session_state[f"{P}_connection_error"]:
                st.caption(f"错误: {st.session_state[f'{P}_connection_error']}")

        with st.container(border=True):
            st.markdown("##### 连接")
            _controller_num = st.number_input(
                "控制器序号 (1-26)",
                min_value=1,
                max_value=26,
                value=1,
                step=1,
                disabled=_connected,
                key=f"{P}_controller_num_input_sb",
                help="控制器序号 1-26，对应 IP 192.168.0.101 ~ 192.168.0.126，端口 10101 ~ 10126",
            )
            st.session_state[f"{P}_controller_num"] = int(_controller_num)
            _ip = f"192.168.0.{100 + int(_controller_num)}"
            _port = 10100 + int(_controller_num)
            st.session_state[f"{P}_ip"] = _ip
            st.session_state[f"{P}_port"] = _port
            st.caption(f"自动设置 IP: **{_ip}**  端口: **{_port}**")
            _sim = st.checkbox(
                "🟡 仿真模式 (无硬件)",
                value=st.session_state.get(f"{P}_simulate", False),
                disabled=_connected,
                help="启用后连接/上电/下发均使用模拟设备，不连接真实硬件",
                key=f"{P}_simulate_sb",
            )
            st.session_state[f"{P}_simulate"] = _sim
            col_test, col_conn = st.columns(2)
            with col_test:
                if st.button(
                    "📡 检测连通性", use_container_width=True, key=f"{P}_test_btn_sb"
                ):
                    test_connectivity()
                    st.rerun()
            with col_conn:
                if not _connected:
                    if st.button(
                        "🔌 连接",
                        type="primary",
                        use_container_width=True,
                        key=f"{P}_connect_sb",
                    ):
                        with st.spinner("连接中..."):
                            connect()
                        st.rerun()
                else:
                    if st.button(
                        "⏏ 断开", use_container_width=True, key=f"{P}_disconnect_sb"
                    ):
                        if st.session_state[f"{P}_relay_on"]:
                            st.session_state[f"{P}_confirm_disconnect"] = True
                            st.rerun()
                        else:
                            disconnect()
                            st.rerun()

            if st.session_state[f"{P}_confirm_disconnect"]:
                st.warning(
                    "⚠️ 继电器仍处于**上电**状态, 断开连接前会先自动下电。确认继续?"
                )
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button(
                        "确认断开",
                        type="primary",
                        use_container_width=True,
                        key=f"{P}_disconnect_confirm_sb",
                    ):
                        disconnect()
                        st.rerun()
                with col_n:
                    if st.button(
                        "取消",
                        use_container_width=True,
                        key=f"{P}_disconnect_cancel_sb",
                    ):
                        st.session_state[f"{P}_confirm_disconnect"] = False
                        st.rerun()

        with st.container(border=True):
            st.markdown("##### 继电器上下电")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button(
                    "⚡ 上电 (接通输出)",
                    type="primary",
                    use_container_width=True,
                    disabled=not _connected or st.session_state[f"{P}_relay_on"],
                    key=f"{P}_relay_on_btn_sb",
                ):
                    set_relay_power(True)
                    st.rerun()
            with col_r2:
                if st.button(
                    "⏻ 下电 (断开输出)",
                    use_container_width=True,
                    disabled=not _connected or not st.session_state[f"{P}_relay_on"],
                    key=f"{P}_relay_off_btn_sb",
                ):
                    set_relay_power(False)
                    st.rerun()


def _sidebar_joint_connection() -> None:
    """Sidebar 联合控制连接配置 (批量Ping/连接/上下电)。"""
    jc = f"{P}_jc"
    with st.sidebar:
        jc_connected = st.session_state.get(f"{jc}_connected", False)
        jc_relay_on = st.session_state.get(f"{jc}_relay_on", False)

        with st.container(border=True):
            st.markdown("##### 当前状态")
            if jc_connected:
                n_ctrl = st.session_state.get(f"{jc}_controller_count", 0)
                n_ips = len(st.session_state.get(f"{jc}_sorted_ips", []))
                st.success(f"✅ MicroDM 已连接 ({n_ctrl} 控制器, {n_ips} IP)")
                if jc_relay_on:
                    st.success("⚡ 继电器已上电 (输出接通)")
                else:
                    st.warning("⏻ 继电器已下电 (输出断开)")
            else:
                st.error("❌ MicroDM 未连接")
            if st.session_state.get(f"{jc}_connection_error", ""):
                st.caption(f"错误: {st.session_state[f'{jc}_connection_error']}")

        with st.container(border=True):
            _jc_sim = st.checkbox(
                "🟡 仿真模式 (无硬件)",
                value=st.session_state.get(f"{P}_jc_simulate", False),
                disabled=jc_connected,
                help="启用后连接/上电/下发均使用模拟设备",
                key=f"{P}_jc_simulate_sb",
            )
            st.session_state[f"{P}_jc_simulate"] = _jc_sim
            st.markdown("##### 操作")
            if not jc_connected:
                if st.button(
                    "🔌 连接 MicroDM",
                    type="primary",
                    use_container_width=True,
                    key=f"{jc}_connect_btn_sb",
                ):
                    with st.spinner("连接所有控制器..."):
                        _jc_connect()
                    st.rerun()
            else:
                col_r1, col_r2, col_disc = st.columns(3)
                with col_r1:
                    if st.button(
                        "⚡ 上电",
                        type="primary",
                        use_container_width=True,
                        disabled=jc_relay_on,
                        key=f"{jc}_relay_on_btn_sb",
                    ):
                        _jc_set_relay(True)
                        st.rerun()
                with col_r2:
                    if st.button(
                        "⏻ 下电",
                        use_container_width=True,
                        disabled=not jc_relay_on,
                        key=f"{jc}_relay_off_btn_sb",
                    ):
                        _jc_set_relay(False)
                        st.rerun()
                with col_disc:
                    if st.button(
                        "⏏ 断开",
                        use_container_width=True,
                        key=f"{jc}_disconnect_btn_sb",
                    ):
                        _jc_disconnect()
                        st.rerun()

        with st.container(border=True):
            st.markdown("##### 批量上下电 (Ping 测试)")
            col_bon, col_boff = st.columns(2)
            with col_bon:
                if st.button(
                    "⚡ 批量上电 (先Ping)",
                    type="primary",
                    use_container_width=True,
                    disabled=jc_relay_on,
                    key=f"{jc}_batch_on_btn_sb",
                ):
                    _jc_batch_power_on()
                    st.rerun()
            with col_boff:
                if st.button(
                    "⏻ 批量下电",
                    use_container_width=True,
                    disabled=not jc_relay_on,
                    key=f"{jc}_batch_off_btn_sb",
                ):
                    _jc_batch_power_off()
                    st.rerun()
            st.caption("上电前自动 Ping 测试所有控制器连通性")


def _sidebar_group_connection() -> None:
    """Sidebar 分组控制连接配置。"""
    gc = f"{P}_gc"
    with st.sidebar:
        groups = st.session_state.get(f"{gc}_groups", {})
        group_names = sorted(groups.keys())
        gc_connected = st.session_state.get(f"{gc}_connected", False)
        gc_relay_on = st.session_state.get(f"{gc}_relay_on", False)

        if not group_names:
            st.warning("未找到 1300-5 组别定义 (CSV 加载失败)")
            return

        selected = st.session_state.get(f"{gc}_selected_group", group_names[0])
        if selected not in group_names:
            selected = group_names[0]

        with st.container(border=True):
            st.markdown("##### 组别选择")
            sel_idx = group_names.index(selected) if selected in group_names else 0
            selected = st.selectbox(
                "选择组别",
                options=group_names,
                index=sel_idx,
                key=f"{gc}_group_select_sb",
            )
            st.session_state[f"{gc}_selected_group"] = selected

            if selected and selected in groups:
                group_def = groups[selected]
                st.caption(
                    f"**{selected}** — {len(group_def.channels_by_ip)} 个控制器, "
                    f"{group_def.total_channels} 个通道"
                )
                rows = []
                for ip_suffix in sorted(group_def.channels_by_ip.keys()):
                    for ch_info in group_def.channels_by_ip[ip_suffix]:
                        rows.append(_channel_info_to_dict(ch_info))
                if rows:
                    with st.expander("📋 通道详情", expanded=False):
                        st.dataframe(
                            pd.DataFrame(rows),
                            width="stretch",
                            hide_index=True,
                        )

        with st.container(border=True):
            st.markdown("##### 当前状态")
            if gc_connected:
                n_ctrl = len(st.session_state.get(f"{gc}_controllers", {}))
                st.success(f"✅ 已连接 {selected} ({n_ctrl} 个控制器)")
                if gc_relay_on:
                    st.success("⚡ 继电器已上电 (输出接通)")
                else:
                    st.warning("⏻ 继电器已下电 (输出断开)")
            else:
                st.error("❌ 未连接")

        with st.container(border=True):
            _gc_sim = st.checkbox(
                "🟡 仿真模式 (无硬件)",
                value=st.session_state.get(f"{P}_gc_simulate", False),
                disabled=gc_connected,
                help="启用后连接/上电/下发均使用模拟设备",
                key=f"{P}_gc_simulate_sb",
            )
            st.session_state[f"{P}_gc_simulate"] = _gc_sim
            st.markdown("##### 操作")
            if not gc_connected:
                if st.button(
                    "🔌 连接组控制器",
                    type="primary",
                    use_container_width=True,
                    key=f"{gc}_connect_btn_sb",
                ):
                    with st.spinner("连接中..."):
                        _gc_connect()
                    st.rerun()
            else:
                col_r1, col_r2, col_disc = st.columns(3)
                with col_r1:
                    if st.button(
                        "⚡ 上电",
                        type="primary",
                        use_container_width=True,
                        disabled=gc_relay_on,
                        key=f"{gc}_relay_on_btn_sb",
                    ):
                        _gc_set_relay(True)
                        st.rerun()
                with col_r2:
                    if st.button(
                        "⏻ 下电",
                        use_container_width=True,
                        disabled=not gc_relay_on,
                        key=f"{gc}_relay_off_btn_sb",
                    ):
                        _gc_set_relay(False)
                        st.rerun()
                with col_disc:
                    if st.button(
                        "⏏ 断开",
                        use_container_width=True,
                        key=f"{gc}_disconnect_btn_sb",
                    ):
                        _gc_disconnect()
                        st.rerun()

        with st.container(border=True):
            st.markdown("##### 批量上下电 (Ping 测试)")
            col_bon, col_boff = st.columns(2)
            with col_bon:
                if st.button(
                    "⚡ 批量上电 (先Ping)",
                    type="primary",
                    use_container_width=True,
                    disabled=gc_relay_on,
                    key=f"{gc}_batch_on_btn_sb",
                ):
                    _gc_batch_power_on()
                    st.rerun()
            with col_boff:
                if st.button(
                    "⏻ 批量下电",
                    use_container_width=True,
                    disabled=not gc_relay_on,
                    key=f"{gc}_batch_off_btn_sb",
                ):
                    _gc_batch_power_off()
                    st.rerun()
            st.caption("上电前自动 Ping 测试组内所有控制器连通性")
