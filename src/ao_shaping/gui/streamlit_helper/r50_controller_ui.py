"""
R50Power 单控制器控制 UI (Streamlit)

面向 ``MicroDM.py`` 中单个 IP 对应的 :class:`R50Controller` 的专用控制面板。

功能:
1. 设置控制器 IP 与端口
2. 检测连通性 (ICMP ping + TCP 端口)
3. 继电器上下电 (relay): 上电 = 闭合输出 (0x06, relay ON), 下电 = 断开输出 (0x07, relay OFF)
4. 在继电器上电状态下下发电压:
   - 指定单元 (可多选 0-49, 含全选/反选) 或 全部 50 单元
   - 持续保持 (单通道组 / 全部)
   - 正弦值电压 (持续循环, 单通道或全通道)

安全约束:
- 所有电压下发操作仅在「继电器上电」后可用, 未上电时发送会被拦截。
- 电压自动截断到安全范围 [vmin, vmax], 且不超越硬件极限 [-20, 120] V。

使用方式:
    streamlit run src/ao_shaping/gui/streamlit_helper/r50_controller_ui.py
"""

from __future__ import annotations

import collections
import socket
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import (
    R50Controller,
    HEADER,
    FOOTER,
    CMD_SET_CHANNEL_VOLTAGE,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_RELAY_ON,
    CMD_RELAY_OFF,
    voltages_to_payload,
)

# 调试日志最大行数
DEBUG_LOG_MAX = 300

# =============================================================================
# Constants
# =============================================================================

SINGLE_CHANNELS = 50  # 单个 R50Power 控制器通道数

# 硬件物理极限 (不可超越)
HW_VOLTAGE_MIN = -20.0
HW_VOLTAGE_MAX = 120.0

# 实时可视化刷新间隔 (s)
REFRESH_INTERVAL = 0.15

DEFAULT_PORT = 10101

# session_state key 前缀, 避免与 micro_dm_ui.py 冲突
P = "r50c"


# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量。"""

    # ---- 连接配置 ----
    st.session_state.setdefault(f"{P}_ip", "192.168.0.101")
    st.session_state.setdefault(f"{P}_port", DEFAULT_PORT)
    st.session_state.setdefault(f"{P}_controller", None)
    st.session_state.setdefault(f"{P}_connected", False)
    st.session_state.setdefault(f"{P}_connection_error", "")

    # ---- 电压安全范围 ----
    st.session_state.setdefault(f"{P}_vmin", HW_VOLTAGE_MIN)
    st.session_state.setdefault(f"{P}_vmax", HW_VOLTAGE_MAX)

    # ---- 继电器状态 ----
    st.session_state.setdefault(f"{P}_relay_on", False)
    st.session_state.setdefault(f"{P}_confirm_disconnect", False)

    # ---- 单元选择 / 电压下发 (指定单元 + 全部单元 合并) ----
    st.session_state.setdefault(f"{P}_channel", 0)       # 正弦单通道目标
    st.session_state.setdefault(f"{P}_channels", [0])    # 指定单元多选
    st.session_state.setdefault(f"{P}_all_mode", False)  # 全部单元(50)开关
    st.session_state.setdefault(f"{P}_voltage", 0.0)
    st.session_state.setdefault(f"{P}_hold", False)

    # ---- 正弦电压发送 ----
    st.session_state.setdefault(f"{P}_sine_amp", 20.0)
    st.session_state.setdefault(f"{P}_sine_offset", 50.0)
    st.session_state.setdefault(f"{P}_sine_freq", 1.0)
    st.session_state.setdefault(f"{P}_sine_apply_all", True)
    st.session_state.setdefault(f"{P}_sine_running", False)

    # ---- 反馈 ----
    st.session_state.setdefault(f"{P}_feedback", "")
    st.session_state.setdefault(f"{P}_feedback_type", "")

    # ---- 调试模式 ----
    st.session_state.setdefault(f"{P}_debug", False)
    st.session_state.setdefault(
        f"{P}_debug_log", collections.deque(maxlen=DEBUG_LOG_MAX)
    )

    # ---- 各单元当前电压 (可视化, 本地跟踪) ----
    st.session_state.setdefault(
        f"{P}_current_voltages", np.zeros(SINGLE_CHANNELS, dtype=np.float64)
    )


# =============================================================================
# Feedback Helpers
# =============================================================================

def set_feedback(message: str, msg_type: str = "info") -> None:
    """设置反馈信息。"""
    st.session_state[f"{P}_feedback"] = message
    st.session_state[f"{P}_feedback_type"] = msg_type


def show_and_clear_feedback() -> None:
    """显示反馈并清除。"""
    message = st.session_state.get(f"{P}_feedback", "")
    msg_type = st.session_state.get(f"{P}_feedback_type", "")
    if message:
        if msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
        elif msg_type == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state[f"{P}_feedback"] = ""
        st.session_state[f"{P}_feedback_type"] = ""


def _debug_log_packet(cmd_name: str, packet: bytes) -> None:
    """记录一条调试日志: 指令名 + 下发数据包的十六进制内容。"""
    if not st.session_state.get(f"{P}_debug"):
        return
    ts = time.strftime("%H:%M:%S", time.localtime())
    hexstr = " ".join(f"{b:02X}" for b in packet)
    st.session_state[f"{P}_debug_log"].append(f"[{ts}] {cmd_name}: {hexstr}")


# =============================================================================
# 连通性检测
# =============================================================================

def _tcp_reachable(ip: str, port: int, timeout: float = 2.0) -> bool:
    """TCP 端口连通性测试。"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except OSError:
        return False


def _ping_reachable(ip: str, timeout: float = 2.0) -> bool:
    """ICMP ping 可达性测试。"""
    param = "-n" if subprocess.os.name == "nt" else "-c"
    timeout_arg = (
        str(int(timeout * 1000))
        if subprocess.os.name == "nt"
        else str(int(timeout))
    )
    try:
        result = subprocess.run(
            ["ping", param, "1", "-W", timeout_arg, ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 1,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def test_connectivity() -> None:
    """检测单个控制器 IP/端口连通性, 写入反馈。"""
    ip = st.session_state[f"{P}_ip"].strip()
    port = int(st.session_state[f"{P}_port"])
    tcp_ok = _tcp_reachable(ip, port)
    ping_ok = _ping_reachable(ip)
    if tcp_ok:
        msg = f"✅ TCP {ip}:{port} 可连通" + ("" if ping_ok else " (ICMP ping 未响应)")
        set_feedback(msg, "success")
    else:
        detail = "TCP 端口不可达" + ("" if ping_ok else "，且 ICMP ping 未响应")
        set_feedback(f"❌ {ip}:{port} {detail}", "error")


# =============================================================================
# 连接 / 断开
# =============================================================================

def connect() -> None:
    """连接单个 R50Power 控制器。"""
    try:
        # 清理旧连接
        if st.session_state[f"{P}_controller"] is not None:
            try:
                st.session_state[f"{P}_controller"].close()
            except Exception as e:
                logger.warning(f"close warning: {e}")
            st.session_state[f"{P}_controller"] = None
        st.session_state[f"{P}_connected"] = False
        st.session_state[f"{P}_relay_on"] = False
        st.session_state[f"{P}_confirm_disconnect"] = False

        ip = st.session_state[f"{P}_ip"].strip()
        port = int(st.session_state[f"{P}_port"])
        ctrl = R50Controller(controller_id=1, ip=ip, port=port)
        if not ctrl.open():
            raise ConnectionError(f"无法建立 TCP 连接到 {ip}:{port}")
        st.session_state[f"{P}_controller"] = ctrl
        st.session_state[f"{P}_connected"] = True
        st.session_state[f"{P}_relay_on"] = False
        st.session_state[f"{P}_connection_error"] = ""
        logger.info(f"R50Controller connected: {ip}:{port}")
        set_feedback(f"已连接到 {ip}:{port}", "success")
    except Exception as e:
        st.session_state[f"{P}_connection_error"] = f"连接失败: {e}"
        st.session_state[f"{P}_connected"] = False
        st.session_state[f"{P}_controller"] = None
        set_feedback(f"连接失败: {e}", "error")
        logger.exception(f"R50Controller connect failed: {e}")


def disconnect() -> None:
    """断开控制器连接 (若继电器仍上电则先自动下电)。"""
    ctrl = st.session_state[f"{P}_controller"]
    st.session_state[f"{P}_sine_running"] = False
    st.session_state[f"{P}_hold"] = False
    # 安全保护: 断开前先下电, 避免带电断开
    if st.session_state[f"{P}_relay_on"] and ctrl is not None:
        try:
            ctrl.set_relay(False)
            logger.info("Relay powered OFF before disconnect")
        except Exception as e:
            logger.warning(f"relay off before disconnect failed: {e}")
    try:
        if ctrl is not None:
            ctrl.close()
    except Exception as e:
        logger.exception(f"disconnect warning: {e}")
    st.session_state[f"{P}_controller"] = None
    st.session_state[f"{P}_connected"] = False
    st.session_state[f"{P}_relay_on"] = False
    st.session_state[f"{P}_confirm_disconnect"] = False
    st.session_state[f"{P}_connection_error"] = ""
    logger.info("R50Controller disconnected")
    set_feedback("已断开连接 (已先下电)", "info")


# =============================================================================
# 继电器上下电
# =============================================================================

def set_relay_power(on: bool) -> None:
    """继电器上电 (on=True) / 下电 (on=False)。

    上电 = 闭合输出 (CMD 0x06, relay ON); 下电 = 断开输出 (CMD 0x07, relay OFF)。
    与 MATLAB ``SetRelayState`` 及驱动 ``set_relay`` 语义一致。
    """
    ctrl = st.session_state[f"{P}_controller"]
    if ctrl is None:
        set_feedback("设备未连接", "error")
        return
    try:
        packet = HEADER + bytes([CMD_RELAY_ON if on else CMD_RELAY_OFF]) + FOOTER
        if ctrl.set_relay(on):
            st.session_state[f"{P}_relay_on"] = on
            _debug_log_packet(f"RELAY {'ON(上电)' if on else 'OFF(下电)'}", packet)
            if on:
                set_feedback("✅ 继电器已上电 (输出接通)", "success")
                logger.info("Relay powered ON")
            else:
                set_feedback("⏻ 继电器已下电 (输出断开)", "info")
                logger.info("Relay powered OFF")
        else:
            set_feedback("继电器指令发送失败", "error")
    except Exception as e:
        set_feedback(f"继电器操作失败: {e}", "error")
        logger.exception(f"relay set failed: {e}")


# =============================================================================
# 电压下发 (仅在继电器上电时允许)
# =============================================================================

def _clip_voltage(voltage: float) -> float:
    """将电压截断到安全范围, 并约束在硬件极限内。"""
    v = float(np.clip(voltage, st.session_state[f"{P}_vmin"], st.session_state[f"{P}_vmax"]))
    v = float(np.clip(v, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX))
    return v


def _send_single(channel: int, voltage: float) -> None:
    """给指定单元 (单通道) 发送一次电压, 并更新本地当前电压跟踪。"""
    ctrl = st.session_state[f"{P}_controller"]
    if ctrl is None:
        return
    v = _clip_voltage(voltage)
    payload = voltages_to_payload(voltage)
    hv, lv = payload[0], payload[1]
    packet = HEADER + bytes([CMD_SET_CHANNEL_VOLTAGE, channel, hv, lv]) + FOOTER
    ctrl.set_channel_voltage(channel, v)
    st.session_state[f"{P}_current_voltages"][channel] = v
    _debug_log_packet(f"SET_CH ch={channel} {v:.1f}V", packet)


def _send_all(voltage: float) -> None:
    """给所有单元 (50 通道) 发送电压, 并更新本地当前电压跟踪。"""
    ctrl = st.session_state[f"{P}_controller"]
    if ctrl is None:
        return
    v = _clip_voltage(voltage)
    payload = voltages_to_payload(voltage)
    hv, lv = payload[0], payload[1]
    packet = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, hv, lv]) + FOOTER
    ctrl.set_all_channel_voltage(v)
    st.session_state[f"{P}_current_voltages"][:] = v
    _debug_log_packet(f"SET_ALL {v:.1f}V", packet)


def _require_relay_on() -> bool:
    """检查继电器是否已上电, 未上电时给出反馈并返回 False。"""
    if not st.session_state[f"{P}_connected"]:
        set_feedback("设备未连接", "error")
        return False
    if not st.session_state[f"{P}_relay_on"]:
        set_feedback("⚠️ 请先继电器上电后再下发电压", "error")
        return False
    return True


def _send_channels(voltage: float) -> None:
    """下发电压: 全部单元(50)模式 -> 全部通道; 否则 -> 指定单元(多选)。"""
    if st.session_state[f"{P}_all_mode"]:
        _send_all(voltage)
    else:
        for ch in st.session_state[f"{P}_channels"]:
            _send_single(int(ch), voltage)


def _hold_loop(voltage: float, interval: float) -> None:
    """电压持续下发线程 (根据当前模式: 全部 / 指定单元)。"""
    try:
        while (
            st.session_state[f"{P}_hold"]
            and st.session_state[f"{P}_relay_on"]
            and not st.session_state[f"{P}_sine_running"]
        ):
            _send_channels(voltage)
            time.sleep(interval)
    except Exception as e:
        st.session_state[f"{P}_hold"] = False
        logger.exception(f"持续下发异常: {e}")
        set_feedback(f"持续下发异常: {e}", "error")


def _sine_loop(amp: float, offset: float, freq: float, apply_all: bool, dt: float) -> None:
    """正弦电压持续下发线程。"""
    ctrl = st.session_state[f"{P}_controller"]
    if ctrl is None:
        return
    freq = max(freq, 0.01)
    omega = 2.0 * np.pi * freq
    t0 = time.time()
    try:
        while (
            st.session_state[f"{P}_sine_running"]
            and st.session_state[f"{P}_relay_on"]
            and not st.session_state[f"{P}_hold"]
        ):
            elapsed = time.time() - t0
            v = offset + amp * np.sin(omega * elapsed)
            if apply_all:
                _send_all(v)
            else:
                _send_single(int(st.session_state[f"{P}_channel"]), v)
            time.sleep(dt)
    except Exception as e:
        st.session_state[f"{P}_sine_running"] = False
        logger.exception(f"正弦下发异常: {e}")
        set_feedback(f"正弦下发异常: {e}", "error")


# =============================================================================
# 电压历史可视化
# =============================================================================

def _render_current_voltages() -> pd.DataFrame:
    """构建 50 个单元当前电压的 DataFrame (供 ``st.bar_chart`` 使用)。"""
    vols = np.asarray(st.session_state[f"{P}_current_voltages"], dtype=np.float64)
    return pd.DataFrame(
        {"单元": list(range(SINGLE_CHANNELS)), "电压 (V)": vols}
    ).set_index("单元")


# =============================================================================
# Main App
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="R50 单控制器控制面板",
        page_icon="🔌",
        layout="wide",
    )

    st.title("🔌 R50Power 单控制器控制面板")
    st.caption("单个 R50Controller (单 IP) 控制 | 连通性检测 · 继电器上下电 · 电压下发")

    _initialize_state()

    # =================== Sidebar: 当前状态 / 连接 / 继电器 / 安全范围 ===================
    with st.sidebar:
        # ---- 当前状态 ----
        with st.container(border=True):
            st.markdown("##### 当前状态")
            if st.session_state[f"{P}_connected"]:
                st.success(
                    f"✅ 已连接  {st.session_state[f'{P}_ip']}:{st.session_state[f'{P}_port']}"
                )
            else:
                st.error("❌ 未连接")
            if st.session_state[f"{P}_relay_on"]:
                st.success("⚡ 继电器上电 (输出接通)")
            else:
                st.warning("⏻ 继电器下电 (输出断开)")
            if st.session_state[f"{P}_connection_error"]:
                st.caption(f"错误: {st.session_state[f'{P}_connection_error']}")

        # ---- 连接配置 ----
        with st.container(border=True):
            st.markdown("##### 连接配置")
            st.text_input("IP 地址", value=st.session_state[f"{P}_ip"], key=f"{P}_ip_input")
            st.session_state[f"{P}_ip"] = st.session_state[f"{P}_ip_input"]
            st.number_input(
                "端口", min_value=1, max_value=65535,
                value=st.session_state[f"{P}_port"], step=1, key=f"{P}_port_input",
            )
            st.session_state[f"{P}_port"] = int(st.session_state[f"{P}_port_input"])
            col_test, col_conn = st.columns(2)
            with col_test:
                if st.button("📡 检测连通性", use_container_width=True, key=f"{P}_test_btn"):
                    test_connectivity()
                    st.rerun()
            with col_conn:
                if not st.session_state[f"{P}_connected"]:
                    if st.button("🔌 连接", type="primary", use_container_width=True, key=f"{P}_connect"):
                        with st.spinner("连接中..."):
                            connect()
                        st.rerun()
                else:
                    if st.button("⏏ 断开", use_container_width=True, key=f"{P}_disconnect"):
                        if st.session_state[f"{P}_relay_on"]:
                            # 继电器仍上电: 需二次确认
                            st.session_state[f"{P}_confirm_disconnect"] = True
                            st.rerun()
                        else:
                            disconnect()
                            st.rerun()

            # 断开确认 (继电器仍上电保护)
            if st.session_state[f"{P}_confirm_disconnect"]:
                st.warning("⚠️ 继电器仍处于**上电**状态, 断开连接前会先自动下电。确认继续?")
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("确认断开", type="primary", use_container_width=True, key=f"{P}_disconnect_confirm"):
                        disconnect()
                        st.rerun()
                with col_n:
                    if st.button("取消", use_container_width=True, key=f"{P}_disconnect_cancel"):
                        st.session_state[f"{P}_confirm_disconnect"] = False
                        st.rerun()

        # ---- 继电器上下电 ----
        with st.container(border=True):
            st.markdown("##### 继电器上下电")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button(
                    "⚡ 上电 (接通输出)", type="primary", use_container_width=True,
                    disabled=not st.session_state[f"{P}_connected"] or st.session_state[f"{P}_relay_on"],
                    key=f"{P}_relay_on_btn",
                ):
                    set_relay_power(True)
                    st.rerun()
            with col_r2:
                if st.button(
                    "⏻ 下电 (断开输出)", use_container_width=True,
                    disabled=not st.session_state[f"{P}_connected"] or not st.session_state[f"{P}_relay_on"],
                    key=f"{P}_relay_off_btn",
                ):
                    set_relay_power(False)
                    st.rerun()
            relay_ok = st.session_state[f"{P}_relay_on"]
            st.caption(
                "上电后输出接通, 方可下发电压; 下电立即断开高压输出。"
                if not relay_ok
                else "✅ 继电器已上电, 可下发电压。"
            )

        # ---- 电压安全范围 ----
        with st.container(border=True):
            st.markdown("##### 电压安全范围 (允许范围)")
            col_min, col_max = st.columns(2)
            with col_min:
                vmin = st.number_input(
                    "下限 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                    value=st.session_state[f"{P}_vmin"], step=1.0, format="%.1f",
                    key=f"{P}_vmin_input",
                )
            with col_max:
                vmax = st.number_input(
                    "上限 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                    value=st.session_state[f"{P}_vmax"], step=1.0, format="%.1f",
                    key=f"{P}_vmax_input",
                )
            if vmin >= vmax:
                st.warning("⚠️ 电压下限必须小于上限")
            st.session_state[f"{P}_vmin"] = vmin
            st.session_state[f"{P}_vmax"] = vmax

        # ---- 调试模式 ----
        with st.container(border=True):
            st.markdown("##### 调试模式")
            st.checkbox(
                "显示指令与下发包日志",
                value=st.session_state[f"{P}_debug"],
                key=f"{P}_debug_input",
            )
            st.session_state[f"{P}_debug"] = st.session_state[f"{P}_debug_input"]
            if st.button("清空日志", use_container_width=True, key=f"{P}_debug_clear"):
                st.session_state[f"{P}_debug_log"].clear()
                st.rerun()

    # =================== Main: 反馈 + 电压下发 ===================
    show_and_clear_feedback()

    # ---- 电压下发 (指定单元 / 全部单元 合并) ----
    with st.container(border=True):
        st.markdown("##### 电压下发")

        # 全部单元 (50) 开关
        st.checkbox(
            "全部单元 (50)",
            value=st.session_state[f"{P}_all_mode"],
            help="选中则下发到全部 50 个单元；否则下发到下方「指定单元」(可多选)",
            key=f"{P}_all_mode_input",
        )
        st.session_state[f"{P}_all_mode"] = st.session_state[f"{P}_all_mode_input"]

        # 指定单元多选 + 全选 / 反选
        if not st.session_state[f"{P}_all_mode"]:
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                # 不使用 key: 每次重渲染以 default 重建, 便于「全选/反选」修改选择
                sel = st.multiselect(
                    "指定单元 (可多选, 0-49)",
                    options=list(range(SINGLE_CHANNELS)),
                    default=st.session_state[f"{P}_channels"],
                )
                st.session_state[f"{P}_channels"] = [int(c) for c in sel]
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                b_all, b_inv = st.columns(2)
                with b_all:
                    if st.button("全选", use_container_width=True, key=f"{P}_sel_all"):
                        st.session_state[f"{P}_channels"] = list(range(SINGLE_CHANNELS))
                        st.rerun()
                with b_inv:
                    if st.button("反选", use_container_width=True, key=f"{P}_sel_inv"):
                        cur = set(st.session_state[f"{P}_channels"])
                        st.session_state[f"{P}_channels"] = [
                            i for i in range(SINGLE_CHANNELS) if i not in cur
                        ]
                        st.rerun()

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
                "⚡ 发送一次", type="primary", use_container_width=True,
                disabled=not st.session_state[f"{P}_connected"], key=f"{P}_send_once",
            ):
                if _require_relay_on():
                    if not st.session_state[f"{P}_all_mode"] and not st.session_state[f"{P}_channels"]:
                        set_feedback("未选择任何指定单元", "warning")
                    else:
                        try:
                            _send_channels(voltage)
                            if st.session_state[f"{P}_all_mode"]:
                                set_feedback(f"已向全部 50 通道下发 {voltage:.1f} V", "success")
                            else:
                                set_feedback(
                                    f"已向 {len(st.session_state[f'{P}_channels'])} 个指定单元下发 {voltage:.1f} V",
                                    "success",
                                )
                        except Exception as e:
                            set_feedback(f"发送失败: {e}", "error")
                    st.rerun()
        with col_send2:
            if not st.session_state[f"{P}_hold"]:
                if st.button(
                    "🔁 持续保持", use_container_width=True, type="secondary",
                    disabled=not st.session_state[f"{P}_connected"] or st.session_state[f"{P}_sine_running"],
                    key=f"{P}_hold_start",
                ):
                    if _require_relay_on():
                        if not st.session_state[f"{P}_all_mode"] and not st.session_state[f"{P}_channels"]:
                            set_feedback("未选择任何指定单元", "warning")
                        else:
                            # 避免与正弦下发线程竞争同一 socket
                            st.session_state[f"{P}_sine_running"] = False
                            st.session_state[f"{P}_hold"] = True
                            threading.Thread(
                                target=_hold_loop, args=(voltage, 0.1), daemon=True,
                            ).start()
                            set_feedback("持续下发中", "success")
                            st.rerun()
            else:
                if st.button(
                    "⏹ 停止", use_container_width=True, type="secondary",
                    key=f"{P}_hold_stop",
                ):
                    st.session_state[f"{P}_hold"] = False
                    set_feedback("已停止持续下发", "info")
                    st.rerun()

    # ---- 正弦电压 ----
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

        if not st.session_state[f"{P}_sine_running"]:
            if st.button(
                "▶ 开始正弦下发", type="primary", use_container_width=True,
                disabled=not st.session_state[f"{P}_connected"] or st.session_state[f"{P}_hold"],
                key=f"{P}_sine_start",
            ):
                if _require_relay_on():
                    # 避免与持续保持线程竞争同一 socket
                    st.session_state[f"{P}_hold"] = False
                    st.session_state[f"{P}_sine_running"] = True
                    threading.Thread(
                        target=_sine_loop,
                        args=(amp, offset, freq, st.session_state[f"{P}_sine_apply_all"], 0.05),
                        daemon=True,
                    ).start()
                    set_feedback(
                        f"正弦下发中: amp={amp}V, offset={offset}V, f={freq}Hz",
                        "success",
                    )
                    st.rerun()
        else:
            if st.button(
                "⏹ 停止", type="primary", use_container_width=True,
                key=f"{P}_sine_stop",
            ):
                st.session_state[f"{P}_sine_running"] = False
                set_feedback("正弦下发已停止", "info")
                st.rerun()

    # ---- 各单元当前电压 ----
    st.divider()
    st.markdown("##### 当前各单元电压 (50 路)")
    df = _render_current_voltages()
    st.bar_chart(df, height=300, use_container_width=True)
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

    # ---- 调试日志 (指令 / 下发包) ----
    if st.session_state[f"{P}_debug"]:
        st.divider()
        st.markdown("##### 调试日志 (指令 / 下发包)")
        log_lines = list(st.session_state[f"{P}_debug_log"])
        st.code("\n".join(log_lines) if log_lines else "(无记录)", language="text")

    if (
        st.session_state[f"{P}_hold"]
        or st.session_state[f"{P}_sine_running"]
    ):
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


if __name__ == "__main__":
    main()
