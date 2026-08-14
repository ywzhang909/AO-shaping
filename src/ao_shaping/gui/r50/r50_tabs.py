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
)
from ao_shaping.gui.r50.r50_common import (
    _channel_label,
    _get_channel_info,
    _loop_start,
    _loop_stop_all,
    _set_feedback,
    _show_feedback,
)
from ao_shaping.gui.r50.r50_group import (
    _gc_apply_voltage,
    _gc_show_feedback,
)
from ao_shaping.gui.r50.r50_joint import (
    _jc_apply_matrix,
    _jc_fill_all,
    _jc_fill_col,
    _jc_fill_rect,
    _jc_fill_row,
    _jc_refresh_from_hardware,
    _jc_render_matrix_dataframe,
    _jc_render_matrix_image,
    _jc_render_profile,
    _jc_render_stats,
    _jc_reset_matrix,
    _jc_set_cell,
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
    sine_tick,
)


# =============================================================================
# 单控制器控制 Tab
# =============================================================================

def render_tab_single_controller() -> None:
    """单控制器控制 Tab: 完整 50 通道电压下发、正弦/交替/保持。"""
    st.title("🔌 单控制器控制")
    st.caption("单个 R50Controller (50 通道) 电压控制 | 持续保持 · 正弦 · 交替 · 可视化")

    if st.session_state.get(f"{P}_connection_mode") != "single":
        st.info("💡 当前未在「单控制器」连接模式。请在侧边栏切换到「单控制器」并连接。")
        return
    if not st.session_state.get(f"{P}_connected", False):
        st.info("💡 请先在侧边栏「单控制器」连接模式下连接控制器。")
        return

    _show_feedback()

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
                    if st.button("全选", width='stretch', key=f"{P}_sel_all"):
                        st.session_state[f"{P}_channels"] = list(range(SINGLE_CHANNELS))
                        st.rerun()
                with b_inv:
                    if st.button("反选", width='stretch', key=f"{P}_sel_inv"):
                        cur = set(st.session_state[f"{P}_channels"])
                        st.session_state[f"{P}_channels"] = [
                            i for i in range(SINGLE_CHANNELS) if i not in cur
                        ]
                        st.rerun()

            if st.session_state[f"{P}_channels"]:
                _infos = []
                for _ch in st.session_state[f"{P}_channels"]:
                    _ci = _get_channel_info(int(_ch))
                    _infos.append(_channel_label(int(_ch)) if _ci else f"ch{_ch}: 无映射")
                st.caption("针脚映射: " + " ｜ ".join(_infos))

        voltage = st.number_input(
            "电压 (V)", min_value=st.session_state[f"{P}_vmin"],
            max_value=st.session_state[f"{P}_vmax"],
            value=st.session_state[f"{P}_voltage"], step=1.0, format="%.1f",
            key=f"{P}_voltage_input",
        )
        st.session_state[f"{P}_voltage"] = float(voltage)

        col_send1, col_send2 = st.columns(2)
        with col_send1:
            if st.button(
                "⚡ 发送一次", type="primary", width='stretch',
                disabled=not st.session_state[f"{P}_connected"], key=f"{P}_send_once",
            ):
                if _require_relay_on():
                    if not st.session_state[f"{P}_all_mode"] and not st.session_state[f"{P}_channels"]:
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
                    "🔁 持续保持", width='stretch', type="secondary",
                    disabled=not st.session_state[f"{P}_connected"] or st.session_state[f"{P}_sine_running"],
                    key=f"{P}_hold_start",
                ):
                    if _require_relay_on():
                        if not st.session_state[f"{P}_all_mode"] and not st.session_state[f"{P}_channels"]:
                            _set_feedback("未选择任何指定单元", "warning")
                        else:
                            selection = ChannelSelection(
                                all_mode=st.session_state[f"{P}_all_mode"],
                                channels=list(st.session_state[f"{P}_channels"]),
                            )
                            _loop_start(hold_tick, {"voltage": float(voltage), "dt": 0.1, "selection": selection})
                            st.session_state[f"{P}_hold"] = True
                            _set_feedback("持续下发中", "success")
                            st.rerun()
            else:
                if st.button(
                    "⏹ 停止", width='stretch', type="secondary",
                    key=f"{P}_hold_stop",
                ):
                    _loop_stop_all()
                    _set_feedback("已停止持续下发", "info")
                    st.rerun()

    with st.container(border=True):
        st.markdown("##### 正弦电压")
        col_a, col_o, col_f = st.columns(3)
        with col_a:
            amp = st.number_input(
                "振幅 (V)", min_value=0.0, max_value=140.0,
                value=st.session_state[f"{P}_sine_amp"], step=1.0, format="%.1f",
                key=f"{P}_sine_amp_input",
            )
            st.session_state[f"{P}_sine_amp"] = float(amp)
        with col_o:
            offset = st.number_input(
                "偏置 (V)", min_value=st.session_state[f"{P}_vmin"],
                max_value=st.session_state[f"{P}_vmax"],
                value=st.session_state[f"{P}_sine_offset"], step=1.0, format="%.1f",
                key=f"{P}_sine_offset_input",
            )
            st.session_state[f"{P}_sine_offset"] = float(offset)
        with col_f:
            freq = st.number_input(
                "频率 (Hz)", min_value=0.01, max_value=50.0,
                value=st.session_state[f"{P}_sine_freq"], step=0.05, format="%.2f",
                key=f"{P}_sine_freq_input",
            )
            st.session_state[f"{P}_sine_freq"] = float(freq)

        vmax_wave = offset + amp
        vmin_wave = offset - amp
        if vmax_wave > st.session_state[f"{P}_vmax"] or vmin_wave < st.session_state[f"{P}_vmin"]:
            st.warning(
                f"⚠️ 正弦范围 [{vmin_wave:.1f}, {vmax_wave:.1f}] V 超出安全范围, "
                "将自动截断到允许范围"
            )

        st.checkbox(
            "应用到全部单元 (50 通道)", value=st.session_state[f"{P}_sine_apply_all"],
            key=f"{P}_sine_apply_all_input",
        )
        st.session_state[f"{P}_sine_apply_all"] = st.session_state[f"{P}_sine_apply_all_input"]

        if not st.session_state[f"{P}_sine_apply_all"]:
            sine_ch = st.number_input(
                "指定单元 (0-49)", min_value=0, max_value=SINGLE_CHANNELS - 1,
                value=st.session_state[f"{P}_channel"], step=1, key=f"{P}_sine_channel_input",
            )
            st.session_state[f"{P}_channel"] = int(sine_ch)
            _sine_ci = _get_channel_info(int(sine_ch))
            if _sine_ci:
                st.caption(f"针脚映射: {_channel_label(int(sine_ch))}")

        if not st.session_state[f"{P}_sine_running"]:
            if st.button(
                "▶ 开始正弦下发", type="primary", width='stretch',
                disabled=not st.session_state[f"{P}_connected"] or st.session_state[f"{P}_hold"],
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
                            "amp": float(amp), "offset": float(offset),
                            "freq": float(freq), "dt": 0.05,
                            "t0": time.time(), "selection": selection,
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
                "⏹ 停止", type="primary", width='stretch',
                key=f"{P}_sine_stop",
            ):
                _loop_stop_all()
                _set_feedback("正弦下发已停止", "info")
                st.rerun()

    with st.container(border=True):
        st.markdown("##### 交替电压 (0V ↔ Input)")
        st.caption("在 0V 和设定电压之间循环交替发送到全部 50 个单元")

        col_alt_v, col_alt_f = st.columns(2)
        with col_alt_v:
            alt_voltage = st.number_input(
                "Input 电压 (V)", min_value=st.session_state[f"{P}_vmin"],
                max_value=st.session_state[f"{P}_vmax"],
                value=st.session_state[f"{P}_alt_voltage"], step=1.0, format="%.1f",
                key=f"{P}_alt_voltage_input",
            )
            st.session_state[f"{P}_alt_voltage"] = float(alt_voltage)
        with col_alt_f:
            alt_freq = st.number_input(
                "交替频率 (Hz)", min_value=0.01, max_value=50.0,
                value=st.session_state[f"{P}_alt_freq"], step=0.05, format="%.2f",
                key=f"{P}_alt_freq_input",
            )
            st.session_state[f"{P}_alt_freq"] = float(alt_freq)

        if not st.session_state[f"{P}_alt_running"]:
            if st.button(
                "▶ 开始交替下发", type="primary", width='stretch',
                disabled=(
                    not st.session_state[f"{P}_connected"]
                    or st.session_state[f"{P}_hold"]
                    or st.session_state[f"{P}_sine_running"]
                ),
                key=f"{P}_alt_start",
            ):
                if _require_relay_on():
                    _loop_start(
                        alt_tick,
                        {
                            "voltage": float(alt_voltage), "freq": float(alt_freq),
                            "dt": 0.01, "t0": time.time(),
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
                "⏹ 停止交替", type="primary", width='stretch',
                key=f"{P}_alt_stop",
            ):
                _loop_stop_all()
                _set_feedback("交替下发已停止", "info")
                st.rerun()

    st.divider()
    st.markdown("##### 当前各单元电压 (50 路)")
    df = _render_current_voltages()
    st.bar_chart(df, height=300, width='stretch')
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

    if st.session_state[f"{P}_debug"]:
        st.divider()
        st.markdown("##### 调试日志 (指令 / 下发包)")
        log_lines = list(st.session_state[f"{P}_debug_log"])
        st.code("\n".join(log_lines) if log_lines else "(无记录)", language="text")

    if (
        st.session_state[f"{P}_hold"]
        or st.session_state[f"{P}_sine_running"]
        or st.session_state[f"{P}_alt_running"]
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
        selected = st.selectbox(
            "选择组别",
            options=group_names,
            index=sel_idx,
            key=f"{gc}_group_select_main",
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
                        pd.DataFrame(rows), width='stretch', hide_index=True,
                    )

    st.divider()
    st.markdown("##### 电压控制")

    group_def = groups.get(selected)
    all_payload_positions = group_def.all_payload_positions if group_def else []

    if not st.session_state.get(f"{gc}_selected_channels"):
        st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()

    col_v, col_ch = st.columns([1, 2])
    with col_v:
        voltage = st.number_input(
            "电压 (V)",
            min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
            value=float(st.session_state.get(f"{gc}_voltage", 0.0)),
            step=1.0, format="%.1f",
            key=f"{gc}_voltage_input",
        )
        st.session_state[f"{gc}_voltage"] = float(voltage)

    with col_ch:
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
            "选择通道 (payload_position)",
            options=all_payload_positions,
            default=st.session_state.get(f"{gc}_selected_channels", all_payload_positions),
            format_func=lambda pp: ch_labels.get(pp, str(pp)),
            key=f"{gc}_channel_select",
        )
        st.session_state[f"{gc}_selected_channels"] = selected_chs

    col_apply, col_apply_all, col_sel, col_desel = st.columns(4)
    with col_apply:
        if st.button(
            "⚡ 下发电压", type="primary", width='stretch',
            key=f"{gc}_apply_btn", disabled=not relay_on,
        ):
            _gc_apply_voltage()
            st.rerun()
    with col_apply_all:
        if st.button(
            "⚡ 全部通道下发", type="secondary", width='stretch',
            key=f"{gc}_apply_all_btn", disabled=not relay_on,
        ):
            st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()
            _gc_apply_voltage(all_channels=True)
            st.rerun()
    with col_sel:
        if st.button("全选通道", width='stretch', key=f"{gc}_select_all_btn"):
            st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()
            st.rerun()
    with col_desel:
        if st.button("清空选择", width='stretch', key=f"{gc}_deselect_all_btn"):
            st.session_state[f"{gc}_selected_channels"] = []
            st.rerun()

    st.divider()
    st.markdown("##### 通道统计")
    n_selected = len(st.session_state.get(f"{gc}_selected_channels", []))
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
    """全部控制 Tab: 36×36 联合矩阵全量控制。"""
    st.title("🔗 全部控制")
    st.caption("MicroDM 36×36 压电陶瓷矩阵 · 全量联合编辑与下发")

    jc = f"{P}_jc"
    matrix: np.ndarray = st.session_state[f"{jc}_matrix"]
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

    _show_feedback(prefix="jc")

    st.divider()
    st.markdown("##### 36×36 电压矩阵 (Streamlit 原生控件)")

    col_img, col_edit = st.columns([3, 1])

    with col_img:
        _jc_render_matrix_image(matrix)
        _jc_render_matrix_dataframe(matrix)

    with col_edit:
        with st.container(border=True):
            st.markdown("###### 编辑矩阵")

            st.markdown("**全部填充**")
            fill_all_v = st.number_input(
                "电压 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                value=0.0, step=1.0, format="%.1f",
                key=f"{jc}_fill_all_input",
            )
            if st.button("填充全部", width='stretch', key=f"{jc}_fill_all_btn"):
                _jc_fill_all(fill_all_v)
                st.rerun()

            st.divider()

            st.markdown("**单个单元**")
            col_e_r, col_e_c = st.columns(2)
            with col_e_r:
                edit_row = st.number_input("行 (0-35)", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_edit_row_input")
            with col_e_c:
                edit_col = st.number_input("列 (0-35)", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_edit_col_input")
            edit_v = st.number_input(
                "电压 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                value=0.0, step=1.0, format="%.1f",
                key=f"{jc}_edit_v_input",
            )
            if st.button("设置单元", width='stretch', key=f"{jc}_set_cell_btn"):
                _jc_set_cell(int(edit_row), int(edit_col), edit_v)
                st.rerun()

            st.divider()

            st.markdown("**行/列填充**")
            fill_v = st.number_input(
                "电压 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                value=0.0, step=1.0, format="%.1f",
                key=f"{jc}_fill_v_input",
            )
            col_fr, col_fc = st.columns(2)
            with col_fr:
                fill_row = st.number_input("目标行", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_fill_row_input")
                if st.button("填充行", width='stretch', key=f"{jc}_fill_row_btn"):
                    _jc_fill_row(int(fill_row), fill_v)
                    st.rerun()
            with col_fc:
                fill_col = st.number_input("目标列", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_fill_col_input")
                if st.button("填充列", width='stretch', key=f"{jc}_fill_col_btn"):
                    _jc_fill_col(int(fill_col), fill_v)
                    st.rerun()

            st.divider()

            st.markdown("**矩形区域**")
            rect_v = st.number_input(
                "电压 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                value=0.0, step=1.0, format="%.1f",
                key=f"{jc}_rect_v_input",
            )
            col_rx1, col_ry1 = st.columns(2)
            with col_rx1:
                rx1 = st.number_input("列起始", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_rx1_input")
            with col_ry1:
                ry1 = st.number_input("行起始", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_ry1_input")
            col_rx2, col_ry2 = st.columns(2)
            with col_rx2:
                rx2 = st.number_input("列结束", 0, GRID_SIZE - 1, GRID_SIZE - 1, 1, key=f"{jc}_rx2_input")
            with col_ry2:
                ry2 = st.number_input("行结束", 0, GRID_SIZE - 1, GRID_SIZE - 1, 1, key=f"{jc}_ry2_input")
            if st.button("填充矩形", width='stretch', key=f"{jc}_rect_btn"):
                _jc_fill_rect(int(rx1), int(ry1), int(rx2), int(ry2), rect_v)
                st.rerun()

    st.divider()
    st.markdown("##### 硬件操作")
    col_send, col_reset, col_refresh = st.columns(3)
    with col_send:
        if st.button("⚡ 下发全部电压到硬件", type="primary", width='stretch',
                     disabled=not connected, key=f"{jc}_apply_btn"):
            if not relay_on:
                st.warning("⚠️ 请先继电器上电")
            else:
                _jc_apply_matrix()
                st.rerun()
    with col_reset:
        if st.button("🔄 清零矩阵", width='stretch',
                     key=f"{jc}_reset_btn"):
            _jc_reset_matrix()
            st.rerun()
    with col_refresh:
        if st.button("📡 从硬件刷新", width='stretch',
                     disabled=not connected, key=f"{jc}_refresh_btn"):
            _jc_refresh_from_hardware()
            st.rerun()

    st.divider()
    _jc_render_stats(matrix)
    _jc_render_profile(matrix)

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
            "矩阵坐标: 行=(物理位置-1)//36, 列=(物理位置-1)%36",
            unsafe_allow_html=True,
        )
