"""R50 控制 Tab 模块 — 单控制器 / 单组 / 全部控制 (联合) 三个 Tab 的渲染。

纯渲染层: 所有动作逻辑委托给 ``r50_single`` / ``r50_joint`` / ``r50_group`` /
``r50_units`` / ``r50_common``。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from ao_shaping.gui.r50.r50_channel_select import (
    DM_NUM_ACTUATORS,
    GRID_SIZE,
    HW_VOLTAGE_MAX,
    HW_VOLTAGE_MIN,
    P,
    REFRESH_INTERVAL,
    SINGLE_CHANNELS,
    ChannelSelection,
    build_position_ip_table,
)
from ao_shaping.gui.r50.r50_common import (
    _channel_label,
    _get_cached_csv_df,
    _get_channel_info,
    _loop_start,
    _loop_stop_all,
    _reload_csv,
    _set_feedback,
    _show_feedback,
)
from ao_shaping.gui.r50.r50_group import (
    _gc_apply_voltage,
    _gc_show_feedback,
)
from ao_shaping.gui.r50.r50_joint import (
    _jc_apply_matrix,
    _jc_disconnect,
    _jc_fill_all,
    _jc_fill_col,
    _jc_fill_rect,
    _jc_fill_row,
    _jc_refresh_from_hardware,
    _jc_render_stats,
    _jc_render_styled_matrix,
    _jc_reset_matrix,
    _jc_reset_to_applied,
    _jc_set_cell,
    _jc_sync_matrix_from_global_state,
    _jc_disconnect,
)
from ao_shaping.gui.r50.r50_single import (
    _require_relay_on,
    _send_channels,
)
from ao_shaping.gui.r50.r50_units import (
    _channel_info_to_dict,
    _render_current_voltages,
)
from ao_shaping.gui.r50.r50_voltage_send import (
    alt_tick,
    hold_tick,
    seq_tick,
    sine_tick,
)


# =============================================================================
# 单控制器 Tab — 子模块渲染器
# =============================================================================


def _get_single_tab_connection_state() -> tuple[str, bool, str]:
    """返回单控制器 Tab 的连接状态 (mode, is_connected, error_msg)。"""
    mode = st.session_state.get(f"{P}_connection_mode", "single")
    single_connected = st.session_state.get(f"{P}_connected", False)
    jc_connected = st.session_state.get(f"{P}_jc_connected", False)

    if mode == "single" and not single_connected:
        return mode, False, "请先在侧边栏「单控制器」连接模式下连接控制器。"
    if mode == "joint" and not jc_connected:
        return mode, False, "请先在侧边栏「联合控制」模式下连接 MicroDM。"
    if mode not in ("single", "joint"):
        return (
            mode,
            False,
            "当前未在「单控制器」或「联合控制」连接模式。请在侧边栏切换。",
        )

    is_connected = (mode == "single" and single_connected) or (
        mode == "joint" and jc_connected
    )
    return mode, is_connected, ""


def _render_joint_ip_selector() -> str | None:
    """在联合控制模式下渲染控制器 IP 选择器，返回选中的 IP 或 None。"""
    jc = f"{P}_jc"
    sorted_ips = st.session_state.get(f"{jc}_sorted_ips", [])
    dm = st.session_state.get(f"{jc}_dm")

    available_ips: list[str] = []
    if dm is not None:
        available_ips = [
            ctrl.ip for ctrl in getattr(dm, "_controllers", []) if hasattr(ctrl, "ip")
        ]
    if not available_ips:
        available_ips = sorted_ips

    if not available_ips:
        st.warning("无可用控制器 IP")
        return None

    current_selected = st.session_state.get(f"{P}_jc_selected_ip", "")
    if current_selected not in available_ips:
        current_selected = available_ips[0]
        st.session_state[f"{P}_jc_selected_ip"] = current_selected

    sel_idx = available_ips.index(current_selected)
    selected_ip = st.selectbox(
        "选择控制器 IP",
        options=available_ips,
        index=sel_idx,
        key=f"{P}_jc_single_ctrl_select",
    )
    st.session_state[f"{P}_jc_selected_ip"] = selected_ip
    st.session_state[f"{P}_ip"] = selected_ip
    st.caption(f"当前操作控制器: **{selected_ip}**")
    return selected_ip


def _render_voltage_send_form(is_connected: bool) -> None:
    """渲染单次 / 持续保持 电压下发表单。"""
    with st.container(border=True):
        st.markdown("##### 电压下发")
        st.checkbox(
            "全部单元 (50)",
            value=st.session_state[f"{P}_all_mode"],
            help="选中则下发到全部 50 个单元；否则下发到下方「指定单元」(可多选)",
            key=f"{P}_all_mode_input",
        )
        st.session_state[f"{P}_all_mode"] = st.session_state[f"{P}_all_mode_input"]

        if not st.session_state[f"{P}_all_mode"]:
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                sel = st.multiselect(
                    "指定单元 (可多选, 0-49)",
                    options=list(range(SINGLE_CHANNELS)),
                    default=st.session_state[f"{P}_channels"],
                    format_func=_channel_label,
                )
                st.session_state[f"{P}_channels"] = [int(c) for c in sel]
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                b_all, b_inv = st.columns(2)
                with b_all:
                    if st.button("全选", width="stretch", key=f"{P}_sel_all"):
                        st.session_state[f"{P}_channels"] = list(range(SINGLE_CHANNELS))
                        st.rerun()
                with b_inv:
                    if st.button("反选", width="stretch", key=f"{P}_sel_inv"):
                        cur = set(st.session_state[f"{P}_channels"])
                        st.session_state[f"{P}_channels"] = [
                            i for i in range(SINGLE_CHANNELS) if i not in cur
                        ]
                        st.rerun()

            if st.session_state[f"{P}_channels"]:
                _infos = []
                for _ch in st.session_state[f"{P}_channels"]:
                    _ci = _get_channel_info(int(_ch))
                    _infos.append(
                        _channel_label(int(_ch)) if _ci else f"ch{_ch}: 无映射"
                    )
                st.caption("针脚映射: " + " ｜ ".join(_infos))

        voltage = st.number_input(
            "电压 (V)",
            min_value=st.session_state[f"{P}_vmin"],
            max_value=st.session_state[f"{P}_vmax"],
            value=st.session_state[f"{P}_voltage"],
            step=1.0,
            format="%.1f",
            key=f"{P}_voltage_input",
        )
        st.session_state[f"{P}_voltage"] = float(voltage)

        col_send1, col_send2 = st.columns(2)
        with col_send1:
            if st.button(
                "⚡ 发送一次",
                type="primary",
                width="stretch",
                disabled=not is_connected,
                key=f"{P}_send_once",
            ):
                if _require_relay_on():
                    if (
                        not st.session_state[f"{P}_all_mode"]
                        and not st.session_state[f"{P}_channels"]
                    ):
                        _set_feedback("未选择任何指定单元", "warning")
                    else:
                        try:
                            _send_channels(voltage)
                        except Exception as e:
                            _set_feedback(f"发送失败: {e}", "error")
                    st.rerun()
        with col_send2:
            if not st.session_state[f"{P}_hold"]:
                if st.button(
                    "🔁 持续保持",
                    width="stretch",
                    type="secondary",
                    disabled=not is_connected
                    or st.session_state[f"{P}_sine_running"]
                    or st.session_state[f"{P}_seq_running"],
                    key=f"{P}_hold_start",
                ):
                    if _require_relay_on():
                        if (
                            not st.session_state[f"{P}_all_mode"]
                            and not st.session_state[f"{P}_channels"]
                        ):
                            _set_feedback("未选择任何指定单元", "warning")
                        else:
                            selection = ChannelSelection(
                                all_mode=st.session_state[f"{P}_all_mode"],
                                channels=list(st.session_state[f"{P}_channels"]),
                            )
                            _loop_start(
                                hold_tick,
                                {
                                    "voltage": float(voltage),
                                    "dt": 0.1,
                                    "selection": selection,
                                },
                            )
                            st.session_state[f"{P}_hold"] = True
                            _set_feedback("持续下发中", "success")
                            st.rerun()
            else:
                if st.button(
                    "⏹ 停止",
                    width="stretch",
                    type="secondary",
                    key=f"{P}_hold_stop",
                ):
                    _loop_stop_all()
                    _set_feedback("已停止持续下发", "info")
                    st.rerun()


def _render_sine_controls(is_connected: bool) -> None:
    """渲染正弦电压控制表单。"""
    with st.container(border=True):
        st.markdown("##### 正弦电压")
        col_a, col_o, col_f = st.columns(3)
        with col_a:
            amp = st.number_input(
                "振幅 (V)",
                min_value=0.0,
                max_value=140.0,
                value=st.session_state[f"{P}_sine_amp"],
                step=1.0,
                format="%.1f",
                key=f"{P}_sine_amp_input",
            )
            st.session_state[f"{P}_sine_amp"] = float(amp)
        with col_o:
            offset = st.number_input(
                "偏置 (V)",
                min_value=st.session_state[f"{P}_vmin"],
                max_value=st.session_state[f"{P}_vmax"],
                value=st.session_state[f"{P}_sine_offset"],
                step=1.0,
                format="%.1f",
                key=f"{P}_sine_offset_input",
            )
            st.session_state[f"{P}_sine_offset"] = float(offset)
        with col_f:
            freq = st.number_input(
                "频率 (Hz)",
                min_value=0.01,
                max_value=50.0,
                value=st.session_state[f"{P}_sine_freq"],
                step=0.05,
                format="%.2f",
                key=f"{P}_sine_freq_input",
            )
            st.session_state[f"{P}_sine_freq"] = float(freq)

        vmax_wave = offset + amp
        vmin_wave = offset - amp
        if (
            vmax_wave > st.session_state[f"{P}_vmax"]
            or vmin_wave < st.session_state[f"{P}_vmin"]
        ):
            st.warning(
                f"⚠️ 正弦范围 [{vmin_wave:.1f}, {vmax_wave:.1f}] V 超出安全范围, "
                "将自动截断到允许范围"
            )

        st.checkbox(
            "应用到全部单元 (50 通道)",
            value=st.session_state[f"{P}_sine_apply_all"],
            key=f"{P}_sine_apply_all_input",
        )
        st.session_state[f"{P}_sine_apply_all"] = st.session_state[
            f"{P}_sine_apply_all_input"
        ]

        if not st.session_state[f"{P}_sine_apply_all"]:
            sine_ch = st.number_input(
                "指定单元 (0-49)",
                min_value=0,
                max_value=SINGLE_CHANNELS - 1,
                value=st.session_state[f"{P}_channel"],
                step=1,
                key=f"{P}_sine_channel_input",
            )
            st.session_state[f"{P}_channel"] = int(sine_ch)
            _sine_ci = _get_channel_info(int(sine_ch))
            if _sine_ci:
                st.caption(f"针脚映射: {_channel_label(int(sine_ch))}")

        if not st.session_state[f"{P}_sine_running"]:
            if st.button(
                "▶ 开始正弦下发",
                type="primary",
                width="stretch",
                disabled=not is_connected
                or st.session_state[f"{P}_hold"]
                or st.session_state[f"{P}_seq_running"],
                key=f"{P}_sine_start",
            ):
                if _require_relay_on():
                    if st.session_state[f"{P}_sine_apply_all"]:
                        selection = ChannelSelection(all_mode=True)
                    else:
                        selection = ChannelSelection(
                            all_mode=False,
                            channels=[st.session_state[f"{P}_channel"]],
                        )
                    _loop_start(
                        sine_tick,
                        {
                            "amp": float(amp),
                            "offset": float(offset),
                            "freq": float(freq),
                            "dt": 0.05,
                            "t0": time.time(),
                            "selection": selection,
                        },
                    )
                    st.session_state[f"{P}_sine_running"] = True
                    _set_feedback(
                        f"正弦下发中: amp={amp}V, offset={offset}V, f={freq}Hz",
                        "success",
                    )
                    st.rerun()
        else:
            if st.button(
                "⏹ 停止",
                type="primary",
                width="stretch",
                key=f"{P}_sine_stop",
            ):
                _loop_stop_all()
                _set_feedback("正弦下发已停止", "info")
                st.rerun()


def _render_alternating_controls(is_connected: bool) -> None:
    """渲染交替电压控制表单。"""
    with st.container(border=True):
        st.markdown("##### 交替电压 (0V ↔ Input)")
        st.caption("在 0V 和设定电压之间循环交替发送到全部 50 个单元")

        col_alt_v, col_alt_f = st.columns(2)
        with col_alt_v:
            alt_voltage = st.number_input(
                "Input 电压 (V)",
                min_value=st.session_state[f"{P}_vmin"],
                max_value=st.session_state[f"{P}_vmax"],
                value=st.session_state[f"{P}_alt_voltage"],
                step=1.0,
                format="%.1f",
                key=f"{P}_alt_voltage_input",
            )
            st.session_state[f"{P}_alt_voltage"] = float(alt_voltage)
        with col_alt_f:
            alt_freq = st.number_input(
                "交替频率 (Hz)",
                min_value=0.01,
                max_value=50.0,
                value=st.session_state[f"{P}_alt_freq"],
                step=0.05,
                format="%.2f",
                key=f"{P}_alt_freq_input",
            )
            st.session_state[f"{P}_alt_freq"] = float(alt_freq)

        if not st.session_state[f"{P}_alt_running"]:
            if st.button(
                "▶ 开始交替下发",
                type="primary",
                width="stretch",
                disabled=(
                    not is_connected
                    or st.session_state[f"{P}_hold"]
                    or st.session_state[f"{P}_sine_running"]
                    or st.session_state[f"{P}_seq_running"]
                ),
                key=f"{P}_alt_start",
            ):
                if _require_relay_on():
                    _loop_start(
                        alt_tick,
                        {
                            "voltage": float(alt_voltage),
                            "freq": float(alt_freq),
                            "dt": 0.01,
                            "t0": time.time(),
                            "selection": ChannelSelection(all_mode=True),
                        },
                    )
                    st.session_state[f"{P}_alt_running"] = True
                    _set_feedback(
                        f"交替下发中: 0V ↔ {alt_voltage:.1f}V, f={alt_freq:.2f}Hz",
                        "success",
                    )
                    st.rerun()
        else:
            if st.button(
                "⏹ 停止交替",
                type="primary",
                width="stretch",
                key=f"{P}_alt_stop",
            ):
                _loop_stop_all()
                _set_feedback("交替下发已停止", "info")
                st.rerun()


def _render_sequential_controls(is_connected: bool) -> None:
    """渲染逐序下发控制表单。"""
    with st.container(border=True):
        st.markdown("##### 逐序下发 (按通道序号)")
        st.caption("依次对选中通道: 发送X电压 → 发送0V → 下一通道 (每步间隔T秒)")

        col_seq_v, col_seq_t = st.columns(2)
        with col_seq_v:
            seq_voltage = st.number_input(
                "电压 (V)",
                min_value=st.session_state[f"{P}_vmin"],
                max_value=st.session_state[f"{P}_vmax"],
                value=st.session_state[f"{P}_seq_voltage"],
                step=1.0,
                format="%.1f",
                key=f"{P}_seq_voltage_input",
            )
            st.session_state[f"{P}_seq_voltage"] = float(seq_voltage)
        with col_seq_t:
            seq_interval = st.number_input(
                "间隔 (s)",
                min_value=0.01,
                max_value=60.0,
                value=st.session_state[f"{P}_seq_interval"],
                step=0.1,
                format="%.2f",
                key=f"{P}_seq_interval_input",
            )
            st.session_state[f"{P}_seq_interval"] = float(seq_interval)

        st.checkbox(
            "完成后自动循环",
            value=st.session_state[f"{P}_seq_auto_loop"],
            help="一轮全部通道扫描完成后自动从头开始下一轮",
            key=f"{P}_seq_auto_loop_input",
        )
        st.session_state[f"{P}_seq_auto_loop"] = st.session_state[
            f"{P}_seq_auto_loop_input"
        ]

        if not st.session_state[f"{P}_seq_running"]:
            if st.button(
                "▶ 开始逐序下发",
                type="primary",
                width="stretch",
                disabled=(
                    not is_connected
                    or st.session_state[f"{P}_hold"]
                    or st.session_state[f"{P}_sine_running"]
                    or st.session_state[f"{P}_alt_running"]
                ),
                key=f"{P}_seq_start",
            ):
                if _require_relay_on():
                    if (
                        not st.session_state[f"{P}_all_mode"]
                        and not st.session_state[f"{P}_channels"]
                    ):
                        _set_feedback("未选择任何指定单元", "warning")
                    else:
                        channels = (
                            list(range(SINGLE_CHANNELS))
                            if st.session_state[f"{P}_all_mode"]
                            else list(st.session_state[f"{P}_channels"])
                        )
                        auto_loop = st.session_state[f"{P}_seq_auto_loop"]
                        _loop_start(
                            seq_tick,
                            {
                                "voltage": float(seq_voltage),
                                "seq_interval": float(seq_interval),
                                "seq_channels": channels,
                                "seq_index": 0,
                                "seq_phase": 0,
                                "seq_last_tick": time.time(),
                                "seq_done": False,
                                "seq_auto_loop": auto_loop,
                                "seq_round": 0,
                                "dt": 0.01,
                                "selection": ChannelSelection(all_mode=True),
                            },
                        )
                        st.session_state[f"{P}_seq_running"] = True
                        mode_label = "循环扫描" if auto_loop else "单次扫描"
                        _set_feedback(
                            f"逐序下发中 ({mode_label}): {len(channels)} 通道, V={seq_voltage:.1f}V, T={seq_interval:.2f}s",
                            "success",
                        )
                        st.rerun()
        else:
            st.caption("运行中 — 请等待或点击停止")
            if st.button(
                "⏹ 停止逐序下发",
                type="primary",
                width="stretch",
                key=f"{P}_seq_stop",
            ):
                _loop_stop_all()
                _set_feedback("逐序下发已停止", "info")
                st.rerun()


def _render_voltage_summary() -> None:
    """渲染当前各单元电压摘要 (柱状图 + 统计)。"""
    st.divider()
    st.markdown("##### 当前各单元电压 (50 路)")
    df = _render_current_voltages()
    st.bar_chart(df, height=300, width="stretch")
    vols = np.asarray(st.session_state[f"{P}_current_voltages"], dtype=np.float64)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("最小", f"{vols.min():.1f} V")
    with col_m2:
        st.metric("最大", f"{vols.max():.1f} V")
    with col_m3:
        st.metric("均值", f"{vols.mean():.1f} V")
    st.caption(
        f"安全范围: [{st.session_state[f'{P}_vmin']:.1f}, "
        f"{st.session_state[f'{P}_vmax']:.1f}] V ｜ "
        f"未连接时显示上次下发值"
    )


# =============================================================================
# 单控制器控制 Tab
# =============================================================================


def render_tab_single_controller() -> None:
    """单控制器控制 Tab: 完整 50 通道电压下发、正弦/交替/保持。"""
    st.title("🔌 单控制器控制")
    st.caption(
        "单个 R50Controller (50 通道) 电压控制 | 持续保持 · 正弦 · 交替 · 可视化"
    )

    mode, is_connected, error_msg = _get_single_tab_connection_state()
    if error_msg:
        st.info(f"💡 {error_msg}")
        return

    _show_feedback()

    if mode == "joint":
        _render_joint_ip_selector()

    _render_voltage_send_form(is_connected)
    _render_sine_controls(is_connected)
    _render_alternating_controls(is_connected)
    _render_sequential_controls(is_connected)
    _render_voltage_summary()

    if (
        st.session_state[f"{P}_hold"]
        or st.session_state[f"{P}_sine_running"]
        or st.session_state[f"{P}_alt_running"]
        or st.session_state[f"{P}_seq_running"]
    ):
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


# =============================================================================
# 单组控制 Tab
# =============================================================================


def render_tab_single_group() -> None:
    """单组控制 Tab: 按 wiring map 组别选择控制器并下发电压。"""
    st.title("🧩 单组控制")
    st.caption("按 wiring map 组别同时控制多个控制器")

    gc = f"{P}_gc"

    if st.session_state.get(f"{P}_connection_mode") != "group":
        st.info("💡 当前未在「分组控制」连接模式。请在侧边栏切换到「分组控制」并连接。")
        return
    if not st.session_state.get(f"{gc}_connected", False):
        st.info("💡 请先在侧边栏「分组控制」模式下连接控制器。")
        return

    relay_on = st.session_state.get(f"{gc}_relay_on", False)
    if not relay_on:
        st.warning("⚠️ 继电器未上电，请在侧边栏先上电")
        return

    _gc_show_feedback()

    groups = st.session_state.get(f"{gc}_groups", {})
    group_names = sorted(groups.keys())

    if not group_names:
        st.warning("未找到 1300-5 组别定义 (CSV 加载失败)")
        return

    selected = st.session_state.get(f"{gc}_selected_group", group_names[0])
    if selected not in group_names:
        selected = group_names[0]

    with st.container(border=True):
        st.markdown("##### 组别选择")
        sel_idx = group_names.index(selected) if selected in group_names else 0
        prev_selected = selected
        selected = st.selectbox(
            "选择组别",
            options=group_names,
            index=sel_idx,
            key=f"{gc}_group_select_main",
        )
        st.session_state[f"{gc}_selected_group"] = selected

        # Reset channels when group changes
        if selected != prev_selected:
            new_group = groups.get(selected)
            new_positions = new_group.all_payload_positions if new_group else []
            st.session_state[f"{gc}_selected_channels"] = new_positions.copy()
            st.session_state[f"{gc}_all_mode"] = True

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

    st.divider()
    st.markdown("##### 电压控制")

    group_def = groups.get(selected)
    all_payload_positions = group_def.all_payload_positions if group_def else []

    # Ensure selected_channels initialized for current group
    if (
        f"{gc}_selected_channels" not in st.session_state
        or not st.session_state[f"{gc}_selected_channels"]
    ):
        st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()
    st.session_state.setdefault(f"{gc}_all_mode", True)

    col_v, col_mode = st.columns(2)
    with col_v:
        voltage = st.number_input(
            "电压 (V)",
            min_value=HW_VOLTAGE_MIN,
            max_value=HW_VOLTAGE_MAX,
            value=float(st.session_state.get(f"{gc}_voltage", 0.0)),
            step=1.0,
            format="%.1f",
            key=f"{gc}_voltage_input",
        )
        st.session_state[f"{gc}_voltage"] = float(voltage)

    with col_mode:
        all_mode = st.checkbox(
            "全部通道模式",
            value=st.session_state[f"{gc}_all_mode"],
            help="开启后下发到组内全部通道; 关闭后仅下发到下方选中的通道",
            key=f"{gc}_all_mode_input",
        )
        st.session_state[f"{gc}_all_mode"] = all_mode

    # Channel selector (visible always for reference, but only used when all_mode=False)
    ch_labels: dict[int, str] = {}
    for ip_suffix in sorted(group_def.channels_by_ip.keys()) if group_def else []:
        for ch_info in group_def.channels_by_ip[ip_suffix]:
            pp = ch_info.payload_position
            desc = f"ch{pp}"
            if ch_info.needle_id:
                desc += f" 针脚#{ch_info.needle_id}"
            if ch_info.physical_label:
                desc += f" ({ch_info.physical_label})"
            desc += f" [192.168.0.{ip_suffix}]"
            ch_labels[pp] = desc

    selected_chs = st.multiselect(
        "选择通道 (payload_position) — 仅「指定通道」模式生效",
        options=all_payload_positions,
        default=st.session_state.get(f"{gc}_selected_channels", all_payload_positions),
        format_func=lambda pp: ch_labels.get(pp, str(pp)),
        key=f"{gc}_channel_select",
        disabled=all_mode,
    )
    if not all_mode:
        st.session_state[f"{gc}_selected_channels"] = selected_chs

    col_apply, col_sel, col_desel = st.columns(3)
    with col_apply:
        if st.button(
            "⚡ 下发电压",
            type="primary",
            width="stretch",
            key=f"{gc}_apply_btn",
            disabled=not relay_on,
        ):
            _gc_apply_voltage()
            st.rerun()
    with col_sel:
        if st.button("全选通道", width="stretch", key=f"{gc}_select_all_btn"):
            st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()
            st.rerun()
    with col_desel:
        if st.button("清空选择", width="stretch", key=f"{gc}_deselect_all_btn"):
            st.session_state[f"{gc}_selected_channels"] = []
            st.rerun()

    st.divider()
    st.markdown("##### 通道统计")
    n_selected = (
        len(all_payload_positions)
        if all_mode
        else len(st.session_state.get(f"{gc}_selected_channels", []))
    )
    n_total = len(all_payload_positions)
    st.metric("已选通道", f"{n_selected} / {n_total}")
    st.caption(
        f"安全范围: [{HW_VOLTAGE_MIN}, {HW_VOLTAGE_MAX}] V | "
        f"当前设定电压: {st.session_state.get(f'{gc}_voltage', 0.0):.1f} V"
    )


# =============================================================================
# 全部控制 Tab (联合控制)
# =============================================================================


def render_tab_all_control() -> None:
    """全部控制 Tab: 36×36 联合矩阵全量编辑与下发。"""
    st.title("🔗 全部控制")
    st.caption("MicroDM 36×36 压电陶瓷矩阵 · 全量联合编辑与下发")

    _jc_sync_matrix_from_global_state()

    jc = f"{P}_jc"
    matrix: np.ndarray | None = st.session_state.get(f"{jc}_matrix")
    applied: np.ndarray | None = st.session_state.get(f"{jc}_applied_matrix")
    connected = st.session_state.get(f"{jc}_connected", False)
    relay_on = st.session_state.get(f"{jc}_relay_on", False)

    if st.session_state.get(f"{P}_connection_mode") != "joint":
        st.info("💡 当前未在「联合控制」连接模式。请在侧边栏切换到「联合控制」并连接。")
        return
    if not connected:
        st.info("💡 请先在侧边栏「联合控制」模式下连接 MicroDM。")
        return
    if not relay_on:
        st.warning("⚠️ 继电器未上电，请在侧边栏先上电")
        return
    if matrix is None:
        st.info("💡 矩阵尚未初始化，请先连接 MicroDM 后重试。")
        return

    _show_feedback(prefix="jc")

    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)

    with st.container(border=True):
        st.markdown("##### 行列选择")
        col_row, col_col = st.columns(2)
        with col_row:
            sel_row = st.selectbox(
                "查看行 (0=全部)",
                options=list(range(0, GRID_SIZE)),
                format_func=lambda r: f"全部" if r == 0 else f"行 {r}",
                key=f"{jc}_view_row_select",
            )
        with col_col:
            sel_col = st.selectbox(
                "查看列 (0=全部)",
                options=list(range(0, GRID_SIZE)),
                format_func=lambda c: f"全部" if c == 0 else f"列 {c}",
                key=f"{jc}_view_col_select",
            )

    col_img, col_edit = st.columns([3, 1])

    with col_img:
        _jc_render_styled_matrix(matrix, applied, vmin, vmax)

    with col_edit:
        with st.container(border=True):
            st.markdown("###### 编辑矩阵")

            st.markdown("**全部填充**")
            fill_all_v = st.number_input(
                "电压 (V)",
                min_value=HW_VOLTAGE_MIN,
                max_value=HW_VOLTAGE_MAX,
                value=0.0,
                step=1.0,
                format="%.1f",
                key=f"{jc}_fill_all_input",
            )
            if st.button("填充全部", width="stretch", key=f"{jc}_fill_all_btn"):
                _jc_fill_all(fill_all_v)
                st.rerun()

            st.divider()

            st.markdown("**单个单元**")
            col_e_r, col_e_c = st.columns(2)
            with col_e_r:
                edit_row = st.number_input(
                    "行 (0-35)", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_edit_row_input"
                )
            with col_e_c:
                edit_col = st.number_input(
                    "列 (0-35)", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_edit_col_input"
                )
            edit_v = st.number_input(
                "电压 (V)",
                min_value=HW_VOLTAGE_MIN,
                max_value=HW_VOLTAGE_MAX,
                value=0.0,
                step=1.0,
                format="%.1f",
                key=f"{jc}_edit_v_input",
            )
            if st.button("设置单元", width="stretch", key=f"{jc}_set_cell_btn"):
                _jc_set_cell(int(edit_row), int(edit_col), edit_v)
                st.rerun()

            st.divider()

            st.markdown("**行/列填充**")
            fill_v = st.number_input(
                "电压 (V)",
                min_value=HW_VOLTAGE_MIN,
                max_value=HW_VOLTAGE_MAX,
                value=0.0,
                step=1.0,
                format="%.1f",
                key=f"{jc}_fill_v_input",
            )
            col_fr, col_fc = st.columns(2)
            with col_fr:
                fill_row = st.number_input(
                    "目标行", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_fill_row_input"
                )
                if st.button("填充行", width="stretch", key=f"{jc}_fill_row_btn"):
                    _jc_fill_row(int(fill_row), fill_v)
                    st.rerun()
            with col_fc:
                fill_col = st.number_input(
                    "目标列", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_fill_col_input"
                )
                if st.button("填充列", width="stretch", key=f"{jc}_fill_col_btn"):
                    _jc_fill_col(int(fill_col), fill_v)
                    st.rerun()

            st.divider()

            st.markdown("**矩形区域**")
            rect_v = st.number_input(
                "电压 (V)",
                min_value=HW_VOLTAGE_MIN,
                max_value=HW_VOLTAGE_MAX,
                value=0.0,
                step=1.0,
                format="%.1f",
                key=f"{jc}_rect_v_input",
            )
            col_rx1, col_ry1 = st.columns(2)
            with col_rx1:
                rx1 = st.number_input(
                    "列起始", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_rx1_input"
                )
            with col_ry1:
                ry1 = st.number_input(
                    "行起始", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_ry1_input"
                )
            col_rx2, col_ry2 = st.columns(2)
            with col_rx2:
                rx2 = st.number_input(
                    "列结束", 0, GRID_SIZE - 1, GRID_SIZE - 1, 1, key=f"{jc}_rx2_input"
                )
            with col_ry2:
                ry2 = st.number_input(
                    "行结束", 0, GRID_SIZE - 1, GRID_SIZE - 1, 1, key=f"{jc}_ry2_input"
                )
            if st.button("填充矩形", width="stretch", key=f"{jc}_rect_btn"):
                _jc_fill_rect(int(rx1), int(ry1), int(rx2), int(ry2), rect_v)
                st.rerun()

    st.divider()
    st.markdown("##### 硬件操作")
    col_apply, col_reset, col_disconnect = st.columns(3)
    with col_apply:
        if st.button(
            "⚡ 下发全部电压到硬件",
            type="primary",
            width="stretch",
            disabled=not connected,
            key=f"{jc}_apply_btn",
        ):
            if not relay_on:
                st.warning("⚠️ 请先继电器上电")
            else:
                _jc_apply_matrix()
                st.rerun()
    with col_reset:
        if st.button("🔄 重置编辑", width="stretch", key=f"{jc}_reset_btn"):
            _jc_reset_to_applied()
            st.rerun()
    with col_disconnect:
        if st.button("🔌 断开并归零", width="stretch", key=f"{jc}_disconnect_btn"):
            _jc_disconnect()
            st.rerun()

    st.divider()
    _jc_render_stats(matrix)

    st.divider()
    with st.container(border=True):
        st.markdown("##### 矩阵说明")
        pos_to_hw = st.session_state.get(f"{jc}_pos_to_hw", {})
        st.caption(
            f"36×36 矩阵共 {DM_NUM_ACTUATORS} 个压电陶瓷单元 · "
            f"1300-5 映射 {len(pos_to_hw)} 个物理位置 · "
            f"排序顺序: {', '.join(st.session_state.get(f'{jc}_sorted_ips', []))[:80]}..."
        )
        st.caption(
            f"电压安全范围: [{HW_VOLTAGE_MIN}, {HW_VOLTAGE_MAX}] V<br>"
            "矩阵坐标: 行=(物理位置-1)//36, 列=(物理位置-1)%36<br>"
            "<b>粗体</b> = 已下发到硬件 (仿真/正式均加粗)",
            unsafe_allow_html=True,
        )

    st.divider()
    with st.container(border=True):
        st.markdown("##### 📋 位置 ↔ IP+序号 对应表")
        col_rc, col_info = st.columns([1, 3])
        with col_rc:
            if st.button("🔄 重新加载 CSV", width="stretch", key=f"{jc}_reload_csv_btn"):
                _reload_csv()
                st.rerun()
        with col_info:
            st.caption("来源: data/1300-5-enriched.csv · 内存缓存, 仅重载时重新读取")
        pos_table = build_position_ip_table(_get_cached_csv_df())
        if pos_table.empty:
            st.warning("⚠️ 1300-5-enriched.csv 加载失败或为空")
        else:
            st.dataframe(pos_table, height=450, width="stretch")
