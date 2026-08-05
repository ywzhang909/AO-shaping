"""R50 controller control UI (regenerated, dual-process architecture).

Thin Streamlit frontend: spawns :func:`start_service` (hardware-owning child
process), sends :class:`ServiceCommand` objects via the client and renders
:class:`ServiceStatus` snapshots. No direct hardware access in this process.

Run: ``streamlit run src/ao_shaping/gui/streamlit_helper/r50_controller/r50_controller_ui.py``
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st

from loguru import logger

from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import (
    CFG,
    GRID_SIZE,
    SINGLE_CHANNELS,
    build_all_units,
    build_groups,
    channel_label,
    load_csv,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import (
    WaveformConfig,
    WaveformType,
    start_service,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_service_client import R50ServiceClient

ALL_IP_SUFFIXES = list(range(101, 127))
DEFAULT_SIMULATE = False


def _ensure_service() -> tuple[R50ServiceClient, Any]:
    client: R50ServiceClient | None = st.session_state.get("r50c_client")
    proc: Any = st.session_state.get("r50c_proc")
    if client is None or proc is None or not proc.is_alive():
        if client is not None:
            logger.warning("Service process died, respawning")
        cmd_q, status_q, proc = start_service()
        client = R50ServiceClient(cmd_q, status_q)
        st.session_state["r50c_client"] = client
        st.session_state["r50c_proc"] = proc
    return client, st.session_state["r50c_proc"]


def _status() -> Any:
    """Return the status cached by the monitor fragment.

    The service publishes into a single status queue that the monitor
    fragment drains; reading the fragment's session_state cache here
    (instead of polling the queue again) avoids racing that drain.
    """
    return st.session_state.get("r50c_status")


def _selected_units() -> list[tuple[int, int]]:
    return st.session_state.get("r50c_selected_units", [])


# =============================================================================
# Sidebar: connection, relay, service lifecycle
# =============================================================================


def _render_sidebar(client: R50ServiceClient, proc: Any) -> None:
    with st.sidebar:
        st.header("设备管理")
        alive = proc.is_alive()
        st.caption(f"服务进程: {'运行中' if alive else '已退出'} (PID {proc.pid})")
        if st.button("停止服务并下电", use_container_width=True):
            client.stop_service()
            st.session_state["r50c_proc"] = None
            st.rerun()

        st.divider()
        st.subheader("单控制器")
        simulate_single = st.checkbox("模拟", key="sim_single", value=DEFAULT_SIMULATE)
        ip_suffix = st.selectbox("IP", ALL_IP_SUFFIXES, key="single_ip")
        status = _status()
        st.caption(_conn_badge(status, int(ip_suffix)))
        c1, c2 = st.columns(2)
        if c1.button("连接", key="conn_single", use_container_width=True):
            client.connect_single(f"192.168.0.{ip_suffix}", simulate=simulate_single)
        if c2.button("断开", key="disc_single", use_container_width=True):
            client.disconnect_single()
        if st.button("继电器开", key="relay_single_on", use_container_width=True):
            client.set_relay(True, mode="single")
        if st.button("继电器关", key="relay_single_off", use_container_width=True):
            client.set_relay(False, mode="single")

        st.divider()
        st.subheader("联合控制 (MicroDM)")
        simulate_joint = st.checkbox("模拟", key="sim_joint", value=DEFAULT_SIMULATE)
        st.caption("联合控制: " + ("已连接" if (status and status.joint_connected) else "未连接"))
        c1, c2 = st.columns(2)
        if c1.button("连接", key="conn_joint", use_container_width=True):
            client.connect_joint(simulate=simulate_joint)
        if c2.button("断开", key="disc_joint", use_container_width=True):
            client.disconnect_joint()
        if st.button("继电器开", key="relay_joint_on", use_container_width=True):
            client.set_relay(True, mode="joint")
        if st.button("继电器关", key="relay_joint_off", use_container_width=True):
            client.set_relay(False, mode="joint")

        st.divider()
        st.subheader("组控制")
        simulate_group = st.checkbox("模拟", key="sim_group", value=DEFAULT_SIMULATE)
        group_names = list(build_groups().keys())
        group_name = st.selectbox("组别", group_names, key="group_name") if group_names else None
        st.caption("组控制: " + ((f"{status.group_name} 已连接") if (status and status.group_connected) else "未连接"))
        c1, c2 = st.columns(2)
        if c1.button("连接", key="conn_group", use_container_width=True, disabled=not group_name):
            if group_name:
                client.connect_group(group_name, simulate=simulate_group)
        if c2.button("断开", key="disc_group", use_container_width=True):
            client.disconnect_group()
        if st.button("继电器开", key="relay_group_on", use_container_width=True):
            client.set_relay(True, mode="group")
        if st.button("继电器关", key="relay_group_off", use_container_width=True):
            client.set_relay(False, mode="group")


def _conn_badge(status: Any, ip_suffix: int) -> str:
    if status is None:
        return f"{ip_suffix}: 未知"
    info = status.controllers.get(ip_suffix)
    if info is None:
        return f"{ip_suffix}: 未连接"
    relay = "继电器开" if info.get("relay_on") else "继电器关"
    sim = " (模拟)" if info.get("simulate") else ""
    return f"{ip_suffix}: {relay}{sim}"


# =============================================================================
# Tab 1: 设备管理
# =============================================================================


def _render_devices_tab(client: R50ServiceClient) -> None:
    st.subheader("全部控制器")
    status = _status()
    rows: list[dict[str, Any]] = []
    for ip_suffix in ALL_IP_SUFFIXES:
        info = (status.controllers.get(ip_suffix) if status else None) or {}
        http_ports = info.get("http_ports") or {}
        http_display = http_ports.get(ip_suffix) if http_ports else info.get("http_port")
        rows.append(
            {
                "IP": f"192.168.0.{ip_suffix}",
                "状态": "已连接" if info else "未连接",
                "模式": info.get("mode", "-"),
                "继电器": "开" if info.get("relay_on") else "关",
                "模拟": "是" if info.get("simulate") else "否",
                "HTTP端口": http_display if http_display else "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("全部上电", key="dev_all_on", use_container_width=True):
        client.set_relay(True, mode="all")
    if c2.button("全部下电", key="dev_all_off", use_container_width=True):
        client.set_relay(False, mode="all")
    if c3.button("刷新状态", key="dev_refresh", use_container_width=True):
        client.refresh()
    if status and status.last_error:
        st.error(f"最近错误: {status.last_error}")


# =============================================================================
# Tab 2: 单元选择
# =============================================================================


def _build_units_df() -> pd.DataFrame:
    units = build_all_units()
    selected = set(_selected_units())
    rows = [
        {
            "选中": (u.ip_suffix, u.payload_position) in selected,
            "IP": u.ip_suffix,
            "通道": u.payload_position,
            "组": u.group,
            "针脚": u.needle_id,
            "连接器": u.physical_label,
        }
        for u in units
    ]
    return pd.DataFrame(rows)


def _render_units_tab(client: R50ServiceClient) -> None:
    st.subheader("单元选择")
    groups = build_groups()
    group_filter = st.multiselect("按组筛选", list(groups.keys()), key="unit_group_filter")
    df = _build_units_df()
    if group_filter:
        df = df[df["组"].isin(group_filter)]
    edited = st.data_editor(
        df,
        key="unit_table",
        hide_index=True,
        use_container_width=True,
        column_config={
            "选中": st.column_config.CheckboxColumn("选中", default=False),
            "IP": st.column_config.NumberColumn("IP", format="%d"),
            "通道": st.column_config.NumberColumn("通道", format="%d"),
            "针脚": st.column_config.NumberColumn("针脚", format="%d"),
        },
        disabled=["IP", "通道", "组", "针脚", "连接器"],
    )
    if edited is not None:
        edited_df = cast(pd.DataFrame, edited)
        if not edited_df.empty:
            mask = edited_df["选中"].to_numpy(dtype=bool)
            ips = edited_df["IP"].to_numpy()
            chs = edited_df["通道"].to_numpy()
            st.session_state["r50c_selected_units"] = [
                (int(a), int(b)) for a, b in zip(ips[mask], chs[mask])
            ]
    selected = _selected_units()
    st.caption(f"已选 {len(selected)} 个单元")
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("清空选择", key="unit_clear", use_container_width=True):
        st.session_state["r50c_selected_units"] = []
        st.session_state.pop("unit_table", None)
        st.rerun()
    voltage = c2.number_input("下发电压 (V)", min_value=float(CFG.HW_VOLTAGE_MIN), max_value=float(CFG.HW_VOLTAGE_MAX), value=0.0, key="unit_voltage")
    c3_1, c3_2 = st.columns(2)
    if c3_1.button("下发到选中单元", key="unit_send", use_container_width=True, disabled=not selected):
        client.set_voltage_direct(voltage, selected)
    if c3_2.button("清零选中单元", key="unit_zero", use_container_width=True, disabled=not selected):
        client.set_voltage_direct(0.0, selected)


# =============================================================================
# Tab 3: 波形配置
# =============================================================================


def _waveform_params() -> dict[str, Any]:
    wtype = st.radio(
        "波形类型",
        [t.name for t in (WaveformType.DC, WaveformType.SINE, WaveformType.SQUARE, WaveformType.ALT)],
        horizontal=True,
        key="wf_type",
    )
    params: dict[str, Any] = {"type": WaveformType[wtype]}
    if wtype in ("DC", "ALT"):
        params["voltage"] = st.number_input("电压 (V)", -20.0, 120.0, 10.0, key="wf_voltage")
    elif wtype == "SINE":
        params["amp"] = st.number_input("幅值 (V)", 0.0, 120.0, 10.0, key="wf_amp")
        params["offset"] = st.number_input("偏置 (V)", -20.0, 120.0, 0.0, key="wf_offset")
        params["freq"] = st.number_input("频率 (Hz)", 0.01, 100.0, 1.0, key="wf_sine_freq")
    elif wtype == "SQUARE":
        params["voltage_a"] = st.number_input("高电平 (V)", -20.0, 120.0, 10.0, key="wf_sq_a")
        params["voltage_b"] = st.number_input("低电平 (V)", -20.0, 120.0, 0.0, key="wf_sq_b")
        params["freq"] = st.number_input("频率 (Hz)", 0.01, 100.0, 1.0, key="wf_sq_freq")
    params["dt"] = st.number_input("步长 dt (s)", 0.01, 1.0, 0.05, key="wf_dt")
    params["duration"] = st.number_input(
        "发送持续时长 (s, 0 = 持续运行)", 0.0, 3600.0, 0.0, 1.0, key="wf_duration",
        help="波形连续发送该时长后自动停止并下发 0V；0 表示持续运行直到手动停止",
    )
    return params


def _build_targets(status: Any) -> list[tuple[int, int]]:
    scope = st.radio(
        "目标范围",
        ["选中单元", "单控制器全部通道", "组全部通道", "全部已连接"],
        horizontal=True,
        key="wf_scope",
    )
    if scope == "选中单元":
        return _selected_units()
    if scope == "单控制器全部通道":
        ips = st.multiselect("控制器", ALL_IP_SUFFIXES, key="wf_single_ips")
        return [(int(ip), pp) for ip in ips for pp in range(1, SINGLE_CHANNELS + 1)]
    if scope == "组全部通道":
        groups = build_groups()
        gname = st.selectbox("组别", list(groups.keys()), key="wf_group_name") if groups else None
        if not gname:
            return []
        gdef = groups[gname]
        return [(int(ip), ch.payload_position) for ip, chs in gdef.channels_by_ip.items() for ch in chs]
    if status is None:
        return []
    return [(int(ip), pp) for ip in status.controllers for pp in range(1, SINGLE_CHANNELS + 1)]


@st.fragment(run_every=0.5)
def _render_waveform_tab(client: R50ServiceClient) -> None:
    st.subheader("波形配置")
    params = _waveform_params()
    status = _status()
    targets = _build_targets(status)
    st.caption(f"目标 {len(targets)} 个单元")
    if targets:
        t = np.arange(0.0, 2.0, 0.01)
        cfg = WaveformConfig(targets=targets, **params)
        preview = np.array([_waveform_value(cfg, float(x)) for x in t])
        st.line_chart(pd.DataFrame({"V": preview}))
    c1, c2 = st.columns(2)
    if c1.button("启动波形", key="wf_start", use_container_width=True, disabled=not targets):
        client.start_waveform(WaveformConfig(targets=targets, **params))
    if c2.button("停止波形", key="wf_stop", use_container_width=True, disabled=not (status and status.waveform_running)):
        client.stop_waveform()
    if status and status.waveform_running:
        if status.waveform_duration > 0.0:
            st.success(f"波形运行中: {status.waveform_type}, 剩余 {status.waveform_remaining:.1f}s / {status.waveform_duration:.1f}s")
        else:
            st.success(f"波形运行中: {status.waveform_type} (持续运行)")


def _waveform_value(cfg: WaveformConfig, t: float) -> float:
    from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import WaveformEngine

    return WaveformEngine.compute(cfg, t)


# =============================================================================
# Tab 4: 联合控制 (36x36 matrix)
# =============================================================================


def _default_matrix() -> np.ndarray:
    status = _status()
    if status and status.joint_matrix is not None:
        return np.asarray(status.joint_matrix, dtype=np.float64)
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)


def _render_joint_tab(client: R50ServiceClient) -> None:
    st.subheader("联合控制 36×36")
    status = _status()
    if not (status and status.joint_connected):
        st.info("联合控制未连接, 请在左侧连接")
        return
    matrix_df = pd.DataFrame(np.asarray(_default_matrix(), dtype=np.float64))
    matrix_df.columns = [str(i) for i in range(GRID_SIZE)]
    matrix = st.data_editor(
        matrix_df,
        key="joint_matrix",
        height=420,
        column_config={
            str(i): st.column_config.NumberColumn(str(i), format="%.1f") for i in range(GRID_SIZE)
        },
        disabled=False,
    )
    m = np.asarray(matrix, dtype=np.float64)
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("应用到硬件", key="joint_apply", use_container_width=True):
        client.set_joint_matrix(m)
    if c2.button("全零", key="joint_zero", use_container_width=True):
        client.set_joint_matrix(np.zeros((GRID_SIZE, GRID_SIZE)))
        st.rerun()
    fill = c3.number_input("填充值 (V)", -20.0, 120.0, 0.0, key="joint_fill")
    if c4.button("填充", key="joint_fill_btn", use_container_width=True):
        client.set_joint_matrix(np.full((GRID_SIZE, GRID_SIZE), fill))
        st.rerun()
    st.metric("min / max / mean", f"{m.min():.1f} / {m.max():.1f} / {m.mean():.1f} V")
    _render_heatmap(m)


def _render_heatmap(m: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(m, cmap="RdBu_r", vmin=min(m.min(), -1), vmax=max(m.max(), 1))
        fig.colorbar(im, ax=ax, label="V")
        ax.set_title("36×36 电压矩阵")
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.dataframe(pd.DataFrame(m), use_container_width=True)


# =============================================================================
# Tab 5: 监控面板 (auto-refresh)
# =============================================================================


@st.fragment(run_every=0.5)
def _monitor_fragment(client: R50ServiceClient) -> None:
    status = client.poll_status()
    if status is None:
        st.info("等待服务状态...")
        return
    st.session_state["r50c_status"] = status
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("单控制器", "已连接" if status.controllers else "未连接")
    c2.metric("联合控制", f"{'已连接' if status.joint_connected else '未连接'} ({status.joint_controller_count}台)")
    c3.metric("组控制", status.group_name if status.group_connected else "未连接")
    c4.metric("波形", "运行中" if status.waveform_running else "停止")
    if status.waveform_type:
        c4.caption(f"类型: {status.waveform_type}")
    if status.last_error:
        st.error(status.last_error)
    st.divider()
    st.subheader("当前电压 (V)")
    if status.current_voltages:
        rows = []
        for ip, volts in sorted(status.current_voltages.items()):
            arr = np.asarray(volts, dtype=np.float64)
            rows.append(
                {
                    "IP": ip,
                    "min": arr.min(),
                    "max": arr.max(),
                    "mean": arr.mean(),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("无已连接控制器")
    if status.joint_matrix is not None:
        st.subheader("联合矩阵")
        _render_heatmap(np.asarray(status.joint_matrix, dtype=np.float64))


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="R50 控制器", page_icon="🔧", layout="wide")
    st.title("R50 Power 控制器 (双进程架构)")
    st.caption("UI 只发送命令, 所有硬件 IO 由独立服务进程执行")
    st.session_state.setdefault("r50c_selected_units", [])
    st.session_state.setdefault("r50c_status", None)

    client, proc = _ensure_service()
    _render_sidebar(client, proc)

    tabs = st.tabs(["设备管理", "单元选择", "波形配置", "联合控制", "监控面板"])
    with tabs[0]:
        _render_devices_tab(client)
    with tabs[1]:
        _render_units_tab(client)
    with tabs[2]:
        _render_waveform_tab(client)
    with tabs[3]:
        _render_joint_tab(client)
    with tabs[4]:
        _monitor_fragment(client)


if __name__ == "__main__":
    main()
