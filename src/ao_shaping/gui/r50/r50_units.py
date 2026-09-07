"""R50 单单元控制模块 — 跨控制器选择物理单元并下发电压。

包含 ``render_tab_single_unit`` 及其辅助函数。复用方:
``r50_tabs`` (``_render_current_voltages`` / ``_channel_info_to_dict``) 与
``r50_sidebar`` (``_channel_info_to_dict``)。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ao_shaping.drivers.dm.MicroDM import (
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    voltages_to_payload,
)
from ao_shaping.gui.r50.r50_channel_select import (
    P,
    SINGLE_CHANNELS,
    ChannelInfo,
    build_all_units,
)
from ao_shaping.gui.r50.r50_command import (
    INF,
    Packet,
    get_unit_voltage,
    r50_command,
)
from ao_shaping.gui.r50.r50_debug import _debug_add_op
from ao_shaping.gui.r50.r50_voltage_send import (
    SendResult,
    apply_joint,
    apply_units_via_controller,
    clip_voltage,
)
from ao_shaping.utils.network import ip_last_octet


INFO_DISPLAY_COLS = ["控制器 IP", "通道号", "组别", "针脚 ID", "物理标签", "物理位置"]


def _channel_info_to_dict(ci: ChannelInfo) -> dict[str, Any]:
    """Convert ChannelInfo to display dict for DataFrame."""
    return {
        "控制器 IP": ci.ip,
        "通道号": ci.payload_position,
        "组别": ci.group,
        "针脚 ID": ci.needle_id,
        "物理标签": ci.physical_label,
        "物理位置": ci.physical_position,
    }


def _render_current_voltages() -> pd.DataFrame:
    """构建 50 个单元当前电压的 DataFrame (供 ``st.bar_chart`` 使用)。"""
    vols = np.asarray(st.session_state[f"{P}_current_voltages"], dtype=np.float64)
    return pd.DataFrame(
        {"单元": list(range(SINGLE_CHANNELS)), "电压 (V)": vols}
    ).set_index("单元")


def _apply_units_group_mode(
    controllers: dict[int, Any],
    units: list[ChannelInfo],
    voltage: float,
    current_map: dict[int, np.ndarray] | None = None,
) -> SendResult:
    """单单元 Tab (分组模式): 按 IP 聚合, 每控制器一个批量包下发。"""
    if current_map is None:
        current_map = {}
    by_ip: dict[int, list[ChannelInfo]] = {}
    for u in units:
        by_ip.setdefault(u.ip_suffix, []).append(u)
    result = SendResult()
    for ip_suffix, unit_list in sorted(by_ip.items()):
        ctrl = controllers.get(int(ip_suffix))
        if ctrl is None:
            result.fail += len(unit_list)
            result.failed_targets.append(f"192.168.0.{ip_suffix} (未连接)")
            continue
        cur = current_map.setdefault(
            int(ip_suffix), np.zeros(SINGLE_CHANNELS, dtype=np.float64)
        )
        r = apply_units_via_controller(
            ctrl,
            cur,
            unit_list,
            voltage,
            st.session_state[f"{P}_vmin"],
            st.session_state[f"{P}_vmax"],
        )
        result.ok += r.ok
        result.fail += r.fail
        result.failed_targets.extend(r.failed_targets)
    return result


def _show_units_result(mode: str, result: SendResult, voltage: float) -> None:
    """单单元 Tab 下发结果反馈。"""
    clipped = clip_voltage(voltage)
    if result.fail:
        st.error(f"⚠️ 下发失败 {result.fail} 个单元: {', '.join(result.failed_targets[:10])}")
        return
    if result.ok:
        st.success(f"✅ 已向 {result.ok} 个单元下发 {clipped:.1f} V")
        _debug_add_op("set_voltage", f"single_unit {mode} {result.ok}ch {clipped:.1f}V", "all")


def render_tab_single_unit() -> None:
    """单单元控制 Tab: 跨控制器选择个别物理单元并下发电压。"""
    st.title("💠 单单元控制")
    st.caption("从 1300-5 映射表中选择单个物理单元并设置电压 (支持跨控制器)")

    if "r50c_single_unit_list" not in st.session_state:
        st.session_state["r50c_single_unit_list"] = build_all_units()
    all_units: list[ChannelInfo] = st.session_state["r50c_single_unit_list"]

    if not all_units:
        st.warning("⚠️ 1300-5-enriched.csv 加载失败或无有效物理单元数据")
        return

    group_names = sorted(set(u.group for u in all_units if u.group))
    selected_group = st.selectbox(
        "按组别筛选", ["全部"] + group_names,
        key="r50c_su_group_filter",
    )

    filtered = all_units
    if selected_group != "全部":
        filtered = [u for u in filtered if u.group == selected_group]

    ip_suffixes = sorted(set(u.ip_suffix for u in filtered))
    conn_ip = st.session_state.get(f"{P}_ip", "").strip()
    conn_suffix = ip_last_octet(conn_ip)
    default_ips = (
        [s for s in ip_suffixes if s == conn_suffix]
        if conn_suffix is not None
        else []
    )
    selected_ips = st.multiselect(
        "按控制器 IP 筛选",
        options=ip_suffixes,
        default=default_ips,
        format_func=lambda s: f"192.168.0.{s}",
        key="r50c_su_ip_filter",
    )
    if selected_ips:
        filtered = [u for u in filtered if u.ip_suffix in selected_ips]

    search = st.text_input("🔍 搜索针脚 ID / 物理标签", "", key="r50c_su_search")
    if search.strip():
        q = search.strip().lower()
        filtered = [
            u for u in filtered
            if q in str(u.needle_id).lower()
            or q in str(u.physical_label).lower()
            or q in str(u.ip_suffix)
        ]

    if not filtered:
        st.info("无匹配的物理单元")
        return

    st.markdown(f"**匹配 {len(filtered)} 个单元**")
    df_display = pd.DataFrame([_channel_info_to_dict(u) for u in filtered])

    with st.container(border=True):
        selected_indices = st.dataframe(
            df_display[INFO_DISPLAY_COLS],
            width='stretch',
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="r50c_su_unit_select",
        )

    sel_rows = selected_indices.get("rows", []) if selected_indices else []
    if not sel_rows:
        st.info("请在上方表格中选择单元")
        return

    selected_units: list[ChannelInfo] = [filtered[i] for i in sel_rows]

    st.divider()
    st.markdown(f"**已选 {len(selected_units)} 个单元**")
    df_sel = pd.DataFrame([_channel_info_to_dict(u) for u in selected_units])
    df_sel["电压状态"] = [
        "未上电"
        if get_unit_voltage(u.ip_suffix, u.payload_position) == INF
        else f"{get_unit_voltage(u.ip_suffix, u.payload_position):.1f} V"
        for u in selected_units
    ]
    st.dataframe(df_sel[INFO_DISPLAY_COLS[:4] + ["电压状态"]], width='stretch', hide_index=True)

    voltage = st.number_input(
        "电压 (V)",
        min_value=st.session_state[f"{P}_vmin"],
        max_value=st.session_state[f"{P}_vmax"],
        value=0.0, step=1.0, format="%.1f",
        key="r50c_su_voltage",
    )

    mode = st.session_state.get(f"{P}_connection_mode", "single")
    jc_connected = st.session_state.get(f"{P}_jc_connected", False)
    single_connected = st.session_state.get(f"{P}_connected", False)
    gc_connected = st.session_state.get(f"{P}_gc_connected", False)

    has_connection = jc_connected or single_connected or gc_connected
    if not has_connection:
        st.warning("⚠️ 请先在侧边栏连接控制器")
        return

    relay_key = {"single": f"{P}_relay_on", "joint": f"{P}_jc_relay_on", "group": f"{P}_gc_relay_on"}
    conn_key = {"single": single_connected, "joint": jc_connected, "group": gc_connected}
    relay_ok = st.session_state.get(relay_key.get(mode, ""), False) if conn_key.get(mode, False) else False
    if not relay_ok:
        st.warning("⚠️ 继电器未上电, 请先在侧边栏上电")
        return

    if mode == "single" and not st.session_state.get(f"{P}_simulate", False):
        current_ip = st.session_state.get(f"{P}_ip", "")
        try:
            current_suffix = int(current_ip.split(".")[-1]) if current_ip else -1
        except ValueError:
            current_suffix = -1
        valid = [u for u in selected_units if u.ip_suffix == current_suffix]
        if len(valid) < len(selected_units):
            st.warning(
                f"当前为单控制器模式 (IP: {current_ip}), "
                f"仅有 {len(valid)}/{len(selected_units)} 个单元属于此控制器"
            )
        if not valid:
            st.error("所选单元均不属于当前控制器, 无法下发")
            return
        selected_units = valid

    if not selected_units:
        return

    if st.button(
        "⚡ 下发电压到所选单元", type="primary", width='stretch',
        key="r50c_su_apply",
    ):
        vmin = st.session_state[f"{P}_vmin"]
        vmax = st.session_state[f"{P}_vmax"]
        units = [(u.ip_suffix, u.payload_position) for u in selected_units]

        def _send() -> tuple[SendResult, list[Packet]]:
            if mode == "joint" and jc_connected:
                dm = st.session_state.get(f"{P}_jc_dm")
                pos_to_hw = st.session_state.get(f"{P}_jc_pos_to_hw", {})
                ip_to_ctrl = st.session_state.get(f"{P}_jc_ip_to_controller_idx", {})
                flat, result = apply_joint(
                    dm,
                    st.session_state[f"{P}_jc_current_flat"],
                    selected_units,
                    voltage, vmin, vmax,
                    pos_to_hw, ip_to_ctrl,
                )
                st.session_state[f"{P}_jc_current_flat"] = flat
                pkt = (
                    HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(flat) + FOOTER
                )
                ips = ",".join(sorted(f"192.168.0.{s}" for s in ip_to_ctrl))
                return result, [("SINGLE_UNIT_SET_VOLTAGE", ips, pkt)]
            if mode == "single" and single_connected:
                ctrl = st.session_state.get(f"{P}_controller")
                result = apply_units_via_controller(
                    ctrl,
                    st.session_state[f"{P}_current_voltages"],
                    selected_units,
                    voltage, vmin, vmax,
                )
                arr = np.asarray(st.session_state[f"{P}_current_voltages"], dtype=np.float64)
                pkt = (
                    HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(arr) + FOOTER
                )
                return result, [("SINGLE_UNIT_SET_VOLTAGE", getattr(ctrl, "ip", "single"), pkt)]
            if mode == "group" and gc_connected:
                controllers = st.session_state.get(f"{P}_gc_controllers", {})
                current_map = st.session_state.setdefault(f"{P}_gc_current_map", {})
                result = _apply_units_group_mode(controllers, selected_units, voltage, current_map)
                packets: list[Packet] = []
                for ip_suffix in sorted(controllers):
                    arr = current_map.get(int(ip_suffix))
                    if arr is not None and np.any(arr):
                        pkt = (
                            HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(arr) + FOOTER
                        )
                        packets.append(("SINGLE_UNIT_SET_VOLTAGE", f"192.168.0.{ip_suffix}", pkt))
                return result, packets
            return SendResult(fail=1, failed_targets=["未知模式"]), []

        result = r50_command(
            mode, "SINGLE_UNIT_SET_VOLTAGE", units, voltage, vmin, vmax, _send
        )
        _show_units_result(mode, result, voltage)
        st.rerun()
