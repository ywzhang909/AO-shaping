"""
Micro DM 微驱动器控制 UI (Streamlit)

功能:
1. Tab 1 - 单个微驱动器单元: 配置 IP/端口、测试连通性、下发固定电压、
   周期波形发送 (正弦/方波)、实时可视化当前下发电压
2. Tab 2 - 联合控制所有单元: 39×39 矩阵批量控制、热力图可视化、一键应用

两个 Tab 均可独立设置电压的上下限 (安全范围)。

使用方式:
    streamlit run src/ao_shaping/gui/dm/micro_dm_ui.py
"""

from __future__ import annotations

import collections
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import R50Controller
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT
from ao_shaping.utils.network import ping_reachable, tcp_reachable
# =============================================================================
# Constants
# =============================================================================

GRID_SIZE = 39
TOTAL_CHANNELS = GRID_SIZE * GRID_SIZE  # 1521
SINGLE_CHANNELS = 50  # 单个 R50Power 控制器通道数

# 硬件物理极限 (不可超越)
HW_VOLTAGE_MIN = -20.0
HW_VOLTAGE_MAX = 120.0

# 实时可视化刷新间隔 (s)
REFRESH_INTERVAL = 0.15
# 电压历史缓存最大长度
HISTORY_LEN = 600

DEFAULT_PORT = 10101

# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量。"""

    # ---- 全局电压安全范围 (两个 Tab 共享) ----
    st.session_state.setdefault("mdm_vmin", HW_VOLTAGE_MIN)
    st.session_state.setdefault("mdm_vmax", HW_VOLTAGE_MAX)

    # ---- Tab 1: 单单元连接 ----
    st.session_state.setdefault("mdm_single_ip", "192.168.0.101")
    st.session_state.setdefault("mdm_single_port", DEFAULT_PORT)
    st.session_state.setdefault("mdm_single_controller", None)
    st.session_state.setdefault("mdm_single_connected", False)
    st.session_state.setdefault("mdm_single_test", "")
    st.session_state.setdefault("mdm_single_test_type", "info")
    st.session_state.setdefault("mdm_single_connection_error", "")

    # ---- Tab 1: 单单元发送参数 ----
    st.session_state.setdefault("mdm_single_channel", 0)
    st.session_state.setdefault("mdm_single_apply_all", False)
    st.session_state.setdefault("mdm_single_fixed_voltage", 0.0)
    st.session_state.setdefault("mdm_single_fixed_hold", False)
    st.session_state.setdefault("mdm_single_mode", "fixed")  # fixed / periodic
    st.session_state.setdefault("mdm_single_wave_type", "sine")  # sine / square
    st.session_state.setdefault("mdm_single_period", 2.0)
    st.session_state.setdefault("mdm_single_amp", 40.0)
    st.session_state.setdefault("mdm_single_offset", 50.0)
    st.session_state.setdefault("mdm_single_high", 100.0)
    st.session_state.setdefault("mdm_single_low", 0.0)
    st.session_state.setdefault("mdm_single_running", False)
    st.session_state.setdefault(
        "mdm_single_history", collections.deque(maxlen=HISTORY_LEN)
    )
    st.session_state.setdefault(
        "mdm_single_history_t", collections.deque(maxlen=HISTORY_LEN)
    )

    # ---- 反馈 (Tab 独立) ----
    st.session_state.setdefault("mdm_tab1_feedback", "")
    st.session_state.setdefault("mdm_tab1_feedback_type", "")
    st.session_state.setdefault("mdm_tab2_feedback", "")
    st.session_state.setdefault("mdm_tab2_feedback_type", "")

    # ---- Tab 2: 39×39 矩阵参数 ----
    st.session_state.setdefault("mdm_dm", None)
    st.session_state.setdefault("mdm_connected", False)
    st.session_state.setdefault("mdm_connection_error", "")
    st.session_state.setdefault("mdm_matrix", np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64))
    st.session_state.setdefault("mdm_matrix_loaded", False)
    st.session_state.setdefault("mdm_heatmap_vmin", HW_VOLTAGE_MIN)
    st.session_state.setdefault("mdm_heatmap_vmax", HW_VOLTAGE_MAX)
    st.session_state.setdefault("mdm_colormap", "coolwarm")
    st.session_state.setdefault("mdm_edit_mode", "single")
    st.session_state.setdefault("mdm_fill_value", 0.0)
    st.session_state.setdefault("mdm_matrix_continuous_voltage", 0.0)
    st.session_state.setdefault("mdm_matrix_continuous_interval", 0.1)
    st.session_state.setdefault("mdm_matrix_continuous_running", False)
    st.session_state.setdefault("mdm_use_wiring_map", True)
    st.session_state.setdefault("mdm_custom_ips", "")
    st.session_state.setdefault("mdm_confirm_action", None)


# =============================================================================
# Feedback Helpers
# =============================================================================

def set_tab1_feedback(message: str, msg_type: str = "info") -> None:
    """设置 Tab 1 反馈信息。"""
    st.session_state.mdm_tab1_feedback = message
    st.session_state.mdm_tab1_feedback_type = msg_type


def set_tab2_feedback(message: str, msg_type: str = "info") -> None:
    """设置 Tab 2 反馈信息。"""
    st.session_state.mdm_tab2_feedback = message
    st.session_state.mdm_tab2_feedback_type = msg_type


def show_and_clear_feedback(tab: str = "tab1") -> None:
    """显示反馈并清除。"""
    fb_key = f"mdm_{tab}_feedback"
    ft_key = f"mdm_{tab}_feedback_type"
    message = st.session_state.get(fb_key, "")
    msg_type = st.session_state.get(ft_key, "")
    if message:
        if msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
        elif msg_type == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state[fb_key] = ""
        st.session_state[ft_key] = ""


# =============================================================================
# 电压范围 (两个 Tab 共享)
# =============================================================================

def render_voltage_limits(key_prefix: str) -> None:
    """在两个 Tab 中渲染可编辑的电压上下限控件，写入共享 session_state。"""
    with st.container(border=True):
        st.markdown("##### 电压上下限 (安全范围)")
        col_min, col_max = st.columns(2)
        with col_min:
            vmin = st.number_input(
                "下限 (V)",
                min_value=HW_VOLTAGE_MIN,
                max_value=HW_VOLTAGE_MAX,
                value=st.session_state.mdm_vmin,
                step=1.0,
                format="%.1f",
                key=f"{key_prefix}_vmin",
            )
        with col_max:
            vmax = st.number_input(
                "上限 (V)",
                min_value=HW_VOLTAGE_MIN,
                max_value=HW_VOLTAGE_MAX,
                value=st.session_state.mdm_vmax,
                step=1.0,
                format="%.1f",
                key=f"{key_prefix}_vmax",
            )
        if vmin >= vmax:
            st.warning("⚠️ 电压下限必须小于上限")
        st.session_state.mdm_vmin = vmin
        st.session_state.mdm_vmax = vmax


# =============================================================================
# 连通性测试 / 单单元连接
# =============================================================================

def test_single_connectivity() -> None:
    """测试单个单元 IP/端口连通性，写入反馈。"""
    ip = st.session_state.mdm_single_ip.strip()
    port = int(st.session_state.mdm_single_port)
    tcp_ok = tcp_reachable(ip, port)
    ping_ok = ping_reachable(ip)
    if tcp_ok:
        msg = f"✅ TCP {ip}:{port} 可连通" + ("" if ping_ok else " (ICMP ping 未响应)")
        set_tab1_feedback(msg, "success")
    else:
        detail = "TCP 端口不可达" + ("" if ping_ok else "，且 ICMP ping 未响应")
        set_tab1_feedback(f"❌ {ip}:{port} {detail}", "error")


def connect_single() -> None:
    """连接单个微驱动器单元 (单台 R50Power 控制器)。"""
    try:
        if st.session_state.mdm_single_controller is not None:
            try:
                st.session_state.mdm_single_controller.close()
            except Exception as e:
                logger.warning(f"close warning: {e}")
            st.session_state.mdm_single_controller = None
            st.session_state.mdm_single_connected = False

        ip = st.session_state.mdm_single_ip.strip()
        port = int(st.session_state.mdm_single_port)
        ctrl = R50Controller(controller_id=1, ip=ip, port=port)
        if not ctrl.open():
            raise ConnectionError(f"无法建立 TCP 连接到 {ip}:{port}")
        st.session_state.mdm_single_controller = ctrl
        st.session_state.mdm_single_connected = True
        st.session_state.mdm_single_connection_error = ""
        logger.info(f"Single unit connected: {ip}:{port}")
    except Exception as e:
        st.session_state.mdm_single_connection_error = f"连接失败: {e}"
        st.session_state.mdm_single_connected = False
        st.session_state.mdm_single_controller = None
        logger.exception(f"Single unit connect failed: {e}")


def disconnect_single() -> None:
    """断开单个单元连接。"""
    st.session_state.mdm_single_running = False
    try:
        if st.session_state.mdm_single_controller is not None:
            st.session_state.mdm_single_controller.close()
    except Exception as e:
        logger.exception(f"disconnect warning: {e}")
    st.session_state.mdm_single_controller = None
    st.session_state.mdm_single_connected = False
    st.session_state.mdm_single_connection_error = ""
    logger.info("Single unit disconnected")


# =============================================================================
# Tab 1: 单单元发送线程
# =============================================================================

def _single_send_v(ctrl, channel: int, voltage: float, apply_all: bool) -> None:
    """向单单元下发电压 (单通道或全部通道)。"""
    v = float(np.clip(voltage, st.session_state.mdm_vmin, st.session_state.mdm_vmax))
    v = float(np.clip(v, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX))
    if ctrl is None:
        return
    if apply_all:
        ctrl.set_all_channel_voltage(v)
    else:
        ctrl.set_channel_voltage(channel, v)
    st.session_state.mdm_single_history.append(v)
    st.session_state.mdm_single_history_t.append(time.time())


def _single_fixed_loop(channel: int, voltage: float, apply_all: bool, interval: float) -> None:
    """固定电压持续下发线程。"""
    ctrl = st.session_state.mdm_single_controller
    try:
        while st.session_state.mdm_single_running:
            _single_send_v(ctrl, channel, voltage, apply_all)
            time.sleep(interval)
    except Exception as e:
        st.session_state.mdm_single_running = False
        logger.exception(f"固定电压下发异常: {e}")
        set_tab1_feedback(f"固定电压下发异常: {e}", "error")


def _single_periodic_loop(
    channel: int, apply_all: bool, wave_type: str,
    amp: float, offset: float, high: float, low: float,
    period: float, dt: float,
) -> None:
    """周期波形下发线程，并记录下发电压用于可视化。"""
    ctrl = st.session_state.mdm_single_controller
    if ctrl is None:
        return
    period = max(period, 0.05)
    t0 = time.time()
    try:
        while st.session_state.mdm_single_running:
            elapsed = time.time() - t0
            phase = elapsed % period
            if wave_type == "sine":
                v = offset + amp * np.sin(2 * np.pi * phase / period)
            else:  # square
                v = high if phase < period / 2 else low
            _single_send_v(ctrl, channel, v, apply_all)
            time.sleep(dt)
    except Exception as e:
        st.session_state.mdm_single_running = False
        logger.exception(f"周期波形下发异常: {e}")
        set_tab1_feedback(f"周期波形下发异常: {e}", "error")


# =============================================================================
# Tab 1: 下发电压可视化
# =============================================================================

def _render_voltage_history() -> None:
    """绘制当前下发电压的实时历史曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    history = list(st.session_state.mdm_single_history)
    t_hist = list(st.session_state.mdm_single_history_t)
    vmin = st.session_state.mdm_vmin
    vmax = st.session_state.mdm_vmax

    fig, ax = plt.subplots(figsize=(7, 3))
    if len(history) > 1:
        t0 = t_hist[0]
        t_rel = [tt - t0 for tt in t_hist]
        ax.plot(t_rel, history, linewidth=1.5, color="#1f77b4", label="下发电压")
    else:
        ax.plot([0], history if history else [0], "o", color="#1f77b4")
    ax.axhline(y=vmax, color="red", linestyle=":", alpha=0.5, label=f"上限 {vmax:.0f}V")
    ax.axhline(y=vmin, color="red", linestyle=":", alpha=0.5, label=f"下限 {vmin:.0f}V")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("电压 (V)")
    ax.set_ylim(vmin - 2, vmax + 2)
    ax.set_title("当前下发电压")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _render_waveform_preview(period: float, wave_type: str,
                             amp: float, offset: float, high: float, low: float) -> None:
    """渲染周期波形预览图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6, 2.5))
    t = np.linspace(0, period * 1.5, 500)
    if wave_type == "sine":
        v = offset + amp * np.sin(2 * np.pi * t / period)
    else:
        v = np.where((t % period) < period / 2, high, low)
    ax.plot(t, v, linewidth=1.5, color="#1f77b4")
    ax.axhline(y=st.session_state.mdm_vmax, color="red", linestyle=":", alpha=0.4, label=f"上限 {st.session_state.mdm_vmax:.0f}V")
    ax.axhline(y=st.session_state.mdm_vmin, color="red", linestyle=":", alpha=0.4, label=f"下限 {st.session_state.mdm_vmin:.0f}V")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("电压 (V)")
    ax.set_title("周期波形预览")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# Tab 1: 单单元控制
# =============================================================================

def render_tab_single_unit() -> None:
    """渲染 Tab 1: 单个微驱动器单元控制。"""

    # ---- 顶部状态栏 ----
    col_status1, col_status2 = st.columns([1, 4])
    with col_status1:
        if st.session_state.mdm_single_connected:
            st.success("✅ 已连接")
        else:
            st.error("❌ 未连接")
    with col_status2:
        if st.session_state.mdm_single_connection_error:
            st.caption(f"错误: {st.session_state.mdm_single_connection_error}")

    show_and_clear_feedback("tab1")

    # ---- 连接配置 ----
    with st.container(border=True):
        st.markdown("##### 连接配置")
        col_ip, col_port, col_test, col_conn = st.columns([3, 1, 1, 1])
        with col_ip:
            st.text_input("IP 地址", value=st.session_state.mdm_single_ip,
                          key="mdm_single_ip")
        with col_port:
            st.number_input("端口", min_value=1, max_value=65535,
                            value=st.session_state.mdm_single_port, step=1,
                            key="mdm_single_port")
        with col_test:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📡 测试连通性", use_container_width=True, key="mdm_single_test_btn"):
                test_single_connectivity()
                st.rerun()
        with col_conn:
            st.markdown("<br>", unsafe_allow_html=True)
            if not st.session_state.mdm_single_connected:
                if st.button("🔌 连接", type="primary", use_container_width=True, key="mdm_single_connect"):
                    with st.spinner("连接中..."):
                        connect_single()
                    st.rerun()
            else:
                if st.button("⏏ 断开", use_container_width=True, key="mdm_single_disconnect"):
                    disconnect_single()
                    st.rerun()

    # ---- 电压上下限 (两个 Tab 共享) ----
    with st.container(border=True):
        render_voltage_limits("single")

    # ---- 通道选择 ----
    col_ch1, col_ch2 = st.columns([2, 2])
    with col_ch1:
        channel = st.number_input(
            "通道号", min_value=0, max_value=SINGLE_CHANNELS - 1,
            value=st.session_state.mdm_single_channel, step=1, key="mdm_single_channel",
        )
    with col_ch2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.checkbox("应用到全部通道 (50)", value=st.session_state.mdm_single_apply_all,
                    key="mdm_single_apply_all")

    # ---- 发送模式 ----
    with st.container(border=True):
        st.markdown("##### 发送模式")
        mode = st.segmented_control(
            "模式", options=["fixed", "periodic"], default="fixed",
            selection_mode="single", key="mdm_single_mode_seg",
        )
        if isinstance(mode, list):
            mode = mode[0] if mode else "fixed"
        st.session_state.mdm_single_mode = mode

        if mode == "fixed":
            voltage = st.number_input(
                "固定电压 (V)",
                min_value=st.session_state.mdm_vmin, max_value=st.session_state.mdm_vmax,
                value=st.session_state.mdm_single_fixed_voltage, step=1.0, format="%.1f",
                key="mdm_single_fixed_voltage",
            )
            col_send1, col_send2 = st.columns(2)
            with col_send1:
                if st.button("⚡ 发送一次", type="primary", use_container_width=True,
                             disabled=not st.session_state.mdm_single_connected,
                             key="mdm_single_send_once"):
                    if not st.session_state.mdm_single_connected:
                        set_tab1_feedback("设备未连接", "error")
                    else:
                        try:
                            _single_send_v(st.session_state.mdm_single_controller,
                                           int(channel), voltage, st.session_state.mdm_single_apply_all)
                            set_tab1_feedback(f"已下发 {voltage:.1f} V", "success")
                        except Exception as e:
                            set_tab1_feedback(f"发送失败: {e}", "error")
                    st.rerun()
            with col_send2:
                running = st.session_state.mdm_single_running
                if not running:
                    if st.button("🔁 持续保持", use_container_width=True, type="secondary",
                                 disabled=not st.session_state.mdm_single_connected,
                                 key="mdm_single_hold_start"):
                        if not st.session_state.mdm_single_connected:
                            set_tab1_feedback("设备未连接", "error")
                        else:
                            st.session_state.mdm_single_running = True
                            threading.Thread(
                                target=_single_fixed_loop,
                                args=(int(channel), voltage,
                                      st.session_state.mdm_single_apply_all, 0.1),
                                daemon=True,
                            ).start()
                            set_tab1_feedback(f"持续下发 {voltage:.1f} V", "success")
                            st.rerun()
                else:
                    if st.button("⏹ 停止", use_container_width=True, type="secondary",
                                 key="mdm_single_hold_stop"):
                        st.session_state.mdm_single_running = False
                        set_tab1_feedback("已停止持续下发", "info")
                        st.rerun()
        else:
            wave_type = st.segmented_control(
                "波形", options=["sine", "square"], default="sine",
                selection_mode="single", key="mdm_single_wave_seg",
            )
            if isinstance(wave_type, list):
                wave_type = wave_type[0] if wave_type else "sine"
            st.session_state.mdm_single_wave_type = wave_type

            period = st.number_input(
                "周期 (s)", min_value=0.1, max_value=60.0,
                value=st.session_state.mdm_single_period, step=0.1, format="%.1f",
                key="mdm_single_period",
            )

            if wave_type == "sine":
                col_a, col_o = st.columns(2)
                with col_a:
                    amp = st.number_input("振幅 (V)", min_value=0.0, max_value=140.0,
                                          value=st.session_state.mdm_single_amp, step=1.0,
                                          format="%.1f", key="mdm_single_amp")
                with col_o:
                    offset = st.number_input("偏置 (V)", min_value=st.session_state.mdm_vmin,
                                             max_value=st.session_state.mdm_vmax,
                                             value=st.session_state.mdm_single_offset, step=1.0,
                                             format="%.1f", key="mdm_single_offset")
                high = low = 0.0
                vmax_wave = offset + amp
                vmin_wave = offset - amp
                if vmax_wave > st.session_state.mdm_vmax or vmin_wave < st.session_state.mdm_vmin:
                    st.warning(f"⚠️ 波形范围 [{vmin_wave:.1f}, {vmax_wave:.1f}] V 超出安全范围，将自动截断")
            else:
                col_h, col_l = st.columns(2)
                with col_h:
                    high = st.number_input("高电平 (V)", min_value=st.session_state.mdm_vmin,
                                           max_value=st.session_state.mdm_vmax,
                                           value=st.session_state.mdm_single_high, step=1.0,
                                           format="%.1f", key="mdm_single_high")
                with col_l:
                    low = st.number_input("低电平 (V)", min_value=st.session_state.mdm_vmin,
                                          max_value=st.session_state.mdm_vmax,
                                          value=st.session_state.mdm_single_low, step=1.0,
                                          format="%.1f", key="mdm_single_low")
                amp = offset = 0.0

            _render_waveform_preview(period, wave_type, amp, offset, high, low)

            running = st.session_state.mdm_single_running
            if not running:
                if st.button("▶ 开始周期下发", type="primary", use_container_width=True,
                             disabled=not st.session_state.mdm_single_connected,
                             key="mdm_single_period_start"):
                    if not st.session_state.mdm_single_connected:
                        set_tab1_feedback("设备未连接", "error")
                    else:
                        st.session_state.mdm_single_running = True
                        threading.Thread(
                            target=_single_periodic_loop,
                            args=(int(channel), st.session_state.mdm_single_apply_all,
                                  wave_type, amp, offset, high, low, period, 0.05),
                            daemon=True,
                        ).start()
                        set_tab1_feedback(f"周期下发中: {wave_type}, 周期{period}s", "success")
                        st.rerun()
            else:
                if st.button("⏹ 停止", type="primary", use_container_width=True,
                             key="mdm_single_period_stop"):
                    st.session_state.mdm_single_running = False
                    set_tab1_feedback("周期下发已停止", "info")
                    st.rerun()

    # ---- 实时电压可视化 ----
    st.divider()
    st.markdown("##### 当前下发电压")
    chart_placeholder = st.empty()
    fig = _render_voltage_history()
    chart_placeholder.pyplot(fig)
    plt_close_safe(fig)

    if st.session_state.mdm_single_running:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


def plt_close_safe(fig) -> None:
    """关闭 matplotlib 图像释放内存。"""
    import matplotlib.pyplot as plt
    plt.close(fig)


# =============================================================================
# Tab 2: 联合控制所有单元 (39×39 矩阵)
# =============================================================================

def connect_dm() -> bool:
    """连接 MicroDM (所有单元)。"""
    try:
        if st.session_state.mdm_dm is not None:
            try:
                st.session_state.mdm_dm.close()
            except Exception as e:
                logger.warning(f"MicroDM close warning: {e}")
            st.session_state.mdm_dm = None
            st.session_state.mdm_connected = False

        from ao_shaping.drivers.dm.MicroDM import MicroDM

        use_wiring_map = st.session_state.mdm_use_wiring_map
        ips = None
        custom_ips_str = st.session_state.mdm_custom_ips.strip()
        if not use_wiring_map and custom_ips_str:
            ips = [ip.strip() for ip in custom_ips_str.split(",") if ip.strip()]
            if not ips:
                ips = None

        dm = MicroDM(ips=ips, use_wiring_map=use_wiring_map)
        dm.open()

        st.session_state.mdm_dm = dm
        st.session_state.mdm_connected = True
        st.session_state.mdm_connection_error = ""

        voltages = dm.get_actuator_positions()
        st.session_state.mdm_matrix = voltages.reshape((GRID_SIZE, GRID_SIZE))
        st.session_state.mdm_matrix_loaded = True

        logger.info("MicroDM connected successfully")
        return True
    except Exception as e:
        error_msg = f"MicroDM 连接失败: {e}"
        st.session_state.mdm_connection_error = error_msg
        st.session_state.mdm_connected = False
        st.session_state.mdm_dm = None
        logger.exception(error_msg)
        return False


def disconnect_dm() -> None:
    """断开 MicroDM 连接。"""
    try:
        if st.session_state.mdm_dm is not None:
            st.session_state.mdm_dm.close()
        st.session_state.mdm_dm = None
        st.session_state.mdm_connected = False
        st.session_state.mdm_connection_error = ""
        logger.info("MicroDM disconnected")
    except Exception as e:
        st.session_state.mdm_connection_error = f"断开连接失败: {e}"
        logger.exception(f"MicroDM disconnect failed: {e}")


def _matrix_continuous_loop(voltage: float, interval: float) -> None:
    """矩阵持续电压发送后台线程。"""
    dm = st.session_state.mdm_dm
    if dm is None:
        return
    try:
        voltage_clipped = float(np.clip(voltage, st.session_state.mdm_vmin, st.session_state.mdm_vmax))
        voltage_clipped = float(np.clip(voltage_clipped, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX))
        voltages = np.full(TOTAL_CHANNELS, voltage_clipped, dtype=np.float64)
        while st.session_state.mdm_matrix_continuous_running:
            dm.send_voltages(voltages)
            time.sleep(interval)
    except Exception as e:
        st.session_state.mdm_matrix_continuous_running = False
        logger.exception(f"矩阵持续电压发送异常: {e}")
        set_tab2_feedback(f"矩阵持续发送异常: {e}", "error")


def _render_heatmap(matrix, vmin, vmax, colormap):
    """渲染热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(matrix, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xlabel("X (列)")
    ax.set_ylabel("Y (行)")
    ax.set_title("DM 39×39 电压分布")
    fig.colorbar(im, ax=ax, label="电压 (V)")
    ax.grid(False)
    fig.tight_layout()
    return fig


def _hw_reset_action() -> None:
    """硬件归零回调。"""
    try:
        st.session_state.mdm_dm.reset_all()
        st.session_state.mdm_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        set_tab2_feedback("硬件已归零", "success")
    except Exception as e:
        set_tab2_feedback(f"归零失败: {e}", "error")


def render_tab_matrix() -> None:
    """渲染 Tab 2: 39×39 矩阵联合控制。"""
    vmin = st.session_state.mdm_vmin
    vmax = st.session_state.mdm_vmax

    if not st.session_state.mdm_connected:
        st.warning("⚠️ 设备未连接，矩阵操作仅在本地预览")

    show_and_clear_feedback("tab2")

    # ---- 电压上下限 (两个 Tab 共享) ----
    with st.container(border=True):
        render_voltage_limits("matrix")

    # ---- 确认对话框 ----
    if st.session_state.mdm_confirm_action:
        action = st.session_state.mdm_confirm_action
        st.warning(f"⚠️ **确认操作**: {action['message']}")
        col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 3])
        with col_confirm1:
            if st.button("✅ 确认", type="primary", use_container_width=True, key="mdm_confirm_yes"):
                action["callback"]()
                st.session_state.mdm_confirm_action = None
                st.rerun()
        with col_confirm2:
            if st.button("❌ 取消", use_container_width=True, key="mdm_confirm_no"):
                st.session_state.mdm_confirm_action = None
                st.rerun()
        st.divider()

    # ---- 左侧热力图 + 右侧控制面板 ----
    col_left, col_right = st.columns([3, 2])

    with col_left:
        matrix = st.session_state.mdm_matrix.copy()
        fig = _render_heatmap(matrix, st.session_state.mdm_heatmap_vmin,
                              st.session_state.mdm_heatmap_vmax, st.session_state.mdm_colormap)
        st.pyplot(fig)
        plt_close_safe(fig)

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("最小", f"{np.min(matrix):.1f} V")
        with col_stat2:
            st.metric("最大", f"{np.max(matrix):.1f} V")
        with col_stat3:
            st.metric("均值", f"{np.mean(matrix):.1f} V")
        with col_stat4:
            st.metric("非零", f"{np.count_nonzero(matrix)}")

    with col_right:
        with st.container(border=True):
            st.markdown("##### 热力图设置")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.number_input(
                    "最小值", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                    value=st.session_state.mdm_heatmap_vmin, step=5.0, format="%.1f",
                    key="mdm_vmin",
                )
            with col_h2:
                st.number_input(
                    "最大值", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                    value=st.session_state.mdm_heatmap_vmax, step=5.0, format="%.1f",
                    key="mdm_vmax",
                )
            st.session_state.mdm_heatmap_vmin = st.session_state.mdm_vmin
            st.session_state.mdm_heatmap_vmax = st.session_state.mdm_vmax

            st.selectbox(
                "颜色映射",
                options=["viridis", "plasma", "coolwarm", "RdBu_r", "jet"],
                index=2, key="mdm_colormap",
            )

        with st.container(border=True):
            st.markdown("##### 编辑模式")
            edit_mode_raw = st.segmented_control(
                "模式", options=["single", "rect", "row", "col"],
                default="single", selection_mode="single", key="mdm_edit_mode_seg",
            )
            if isinstance(edit_mode_raw, list):
                edit_mode = edit_mode_raw[0] if edit_mode_raw else "single"
            else:
                edit_mode = edit_mode_raw if edit_mode_raw else "single"

            fill_value = st.number_input(
                "填充值 (V)", min_value=vmin, max_value=vmax,
                value=st.session_state.mdm_fill_value, step=1.0, format="%.1f",
                key="mdm_fill_value_input",
            )
            st.session_state.mdm_fill_value = fill_value

            if edit_mode == "single":
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    row = st.number_input("行", 0, 38, 0, key="mdm_edit_row")
                with col_s2:
                    col = st.number_input("列", 0, 38, 0, key="mdm_edit_col")
                if st.button("设置", type="primary", use_container_width=True, key="mdm_btn_set_cell"):
                    st.session_state.mdm_matrix[row, col] = fill_value
                    set_tab2_feedback(f"[{row},{col}] → {fill_value} V", "success")
                    st.rerun()
            elif edit_mode == "rect":
                st.markdown("**区域范围**")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    r1 = st.number_input("起始行", 0, 38, 0, key="mdm_rect_r1")
                    c1 = st.number_input("起始列", 0, 38, 0, key="mdm_rect_c1")
                with col_r2:
                    r2 = st.number_input("结束行", 0, 38, 10, key="mdm_rect_r2")
                    c2 = st.number_input("结束列", 0, 38, 10, key="mdm_rect_c2")
                if st.button("设置区域", type="primary", use_container_width=True, key="mdm_btn_set_rect"):
                    rs, re = min(r1, r2), max(r1, r2)
                    cs, ce = min(c1, c2), max(c1, c2)
                    st.session_state.mdm_matrix[rs:re + 1, cs:ce + 1] = fill_value
                    cnt = (re - rs + 1) * (ce - cs + 1)
                    set_tab2_feedback(f"区域 [{rs}:{re+1}, {cs}:{ce+1}] ({cnt}单元) → {fill_value} V", "success")
                    st.rerun()
            elif edit_mode == "row":
                row_idx = st.number_input("行号", 0, 38, 0, key="mdm_edit_row_idx")
                if st.button(f"设置第 {row_idx} 行", type="primary", use_container_width=True, key="mdm_btn_set_row"):
                    st.session_state.mdm_matrix[row_idx, :] = fill_value
                    set_tab2_feedback(f"第 {row_idx} 行 ({GRID_SIZE}单元) → {fill_value} V", "success")
                    st.rerun()
            elif edit_mode == "col":
                col_idx = st.number_input("列号", 0, 38, 0, key="mdm_edit_col_idx")
                if st.button(f"设置第 {col_idx} 列", type="primary", use_container_width=True, key="mdm_btn_set_col"):
                    st.session_state.mdm_matrix[:, col_idx] = fill_value
                    set_tab2_feedback(f"第 {col_idx} 列 ({GRID_SIZE}单元) → {fill_value} V", "success")
                    st.rerun()

    # ---- 批量操作 ----
    st.divider()
    st.markdown("##### 图案生成")
    col_batch1, col_batch2, col_batch3, col_batch4, col_batch5, col_batch6 = st.columns(6)
    with col_batch1:
        if st.button("🔄 归零", use_container_width=True, key="mdm_btn_reset_zero"):
            st.session_state.mdm_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
            set_tab2_feedback("矩阵已归零", "success")
            st.rerun()
    with col_batch2:
        if st.button("📋 全填充", use_container_width=True, key="mdm_btn_fill_all"):
            st.session_state.mdm_matrix[:, :] = fill_value
            set_tab2_feedback(f"全部 {TOTAL_CHANNELS} 单元 → {fill_value} V", "success")
            st.rerun()
    with col_batch3:
        if st.button("🎲 随机", use_container_width=True, key="mdm_btn_random"):
            st.session_state.mdm_matrix = np.random.uniform(vmin, vmax, (GRID_SIZE, GRID_SIZE))
            set_tab2_feedback("矩阵已随机生成", "success")
            st.rerun()
    with col_batch4:
        if st.button("🌊 正弦", use_container_width=True, key="mdm_btn_sine_pattern"):
            x = np.linspace(0, 4 * np.pi, GRID_SIZE)
            y = np.linspace(0, 4 * np.pi, GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            pattern = np.clip(((vmin + vmax) / 2.0) + ((vmax - vmin) / 2.0) * np.sin(X) * np.cos(Y), vmin, vmax)
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("正弦图案已生成", "success")
            st.rerun()
    with col_batch5:
        if st.button("🔺 高斯", use_container_width=True, key="mdm_btn_gaussian"):
            x = np.linspace(-3, 3, GRID_SIZE)
            y = np.linspace(-3, 3, GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            pattern = np.clip(vmax - (vmax - vmin) * (1.0 - np.exp(-(X ** 2 + Y ** 2) / 2.0)), vmin, vmax)
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("高斯图案已生成", "success")
            st.rerun()
    with col_batch6:
        if st.button("📐 渐变", use_container_width=True, key="mdm_btn_gradient"):
            pattern = np.tile(np.linspace(vmin, vmax, GRID_SIZE), (GRID_SIZE, 1)).T
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("渐变图案已生成", "success")
            st.rerun()

    # ---- 硬件操作 ----
    st.divider()
    st.markdown("##### 硬件操作")
    col_hw1, col_hw2, col_hw3 = st.columns(3)
    with col_hw1:
        if st.button("⬆ 应用到硬件", type="primary", use_container_width=True, key="mdm_btn_apply"):
            if not st.session_state.mdm_connected:
                set_tab2_feedback("设备未连接", "error")
            else:
                try:
                    st.session_state.mdm_dm.send_voltages(st.session_state.mdm_matrix.flatten())
                    set_tab2_feedback(f"矩阵已发送 ({TOTAL_CHANNELS} 通道)", "success")
                except Exception as e:
                    set_tab2_feedback(f"应用失败: {e}", "error")
    with col_hw2:
        if st.button("⬇ 从硬件读取", use_container_width=True, key="mdm_btn_read_matrix"):
            if not st.session_state.mdm_connected:
                set_tab2_feedback("设备未连接", "error")
            else:
                try:
                    voltages = st.session_state.mdm_dm.get_actuator_positions()
                    st.session_state.mdm_matrix = voltages.reshape((GRID_SIZE, GRID_SIZE))
                    set_tab2_feedback("矩阵已刷新", "success")
                    st.rerun()
                except Exception as e:
                    set_tab2_feedback(f"读取失败: {e}", "error")
    with col_hw3:
        if st.button("⚡ 硬件归零", type="primary", use_container_width=True, key="mdm_btn_hw_reset"):
            if not st.session_state.mdm_connected:
                set_tab2_feedback("设备未连接", "error")
            else:
                st.session_state.mdm_confirm_action = {
                    "message": "将所有通道电压归零。此操作将立即发送到硬件。",
                    "callback": lambda: _hw_reset_action(),
                }
                st.rerun()

    # ---- 数据导入/导出 ----
    st.divider()
    col_io1, col_io2 = st.columns(2)
    with col_io1:
        uploaded = st.file_uploader("上传 .npy 文件", type=["npy"], label_visibility="collapsed", key="mdm_file_upload")
        if uploaded is not None:
            try:
                data = np.load(uploaded)
                if data.shape == (GRID_SIZE, GRID_SIZE):
                    st.session_state.mdm_matrix = np.clip(data.astype(np.float64), vmin, vmax)
                    set_tab2_feedback(f"矩阵已加载 ({data.shape})", "success")
                    st.rerun()
                else:
                    set_tab2_feedback(f"形状不匹配: 期望 ({GRID_SIZE},{GRID_SIZE}), 实际 {data.shape}", "error")
            except Exception as e:
                set_tab2_feedback(f"加载失败: {e}", "error")
    with col_io2:
        buffer = io.BytesIO()
        np.save(buffer, st.session_state.mdm_matrix)
        buffer.seek(0)
        st.download_button(
            label="💾 下载 .npy",
            data=buffer,
            file_name="dm_matrix.npy",
            mime="application/octet-stream",
            use_container_width=True,
            key="mdm_download_btn",
        )

    # ---- 动态持续电压 ----
    st.divider()
    with st.expander("🔄 动态持续电压", expanded=False):
        st.markdown("持续将当前矩阵电压发送到硬件，适用于需要稳定电压输出的场景。")
        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            continuous_voltage = st.number_input(
                "持续电压值 (V)", min_value=vmin, max_value=vmax,
                value=st.session_state.mdm_matrix_continuous_voltage, step=1.0, format="%.1f",
                key="mdm_cv_voltage",
            )
        with col_cv2:
            continuous_interval = st.number_input(
                "发送间隔 (s)", min_value=0.01, max_value=10.0,
                value=st.session_state.mdm_matrix_continuous_interval, step=0.01, format="%.2f",
                key="mdm_cv_interval",
            )
        col_cv_btn1, col_cv_btn2 = st.columns(2)
        with col_cv_btn1:
            if st.button("▶ 开始持续发送", type="primary",
                         disabled=st.session_state.mdm_matrix_continuous_running,
                         use_container_width=True, key="mdm_cv_start"):
                if not st.session_state.mdm_connected:
                    set_tab2_feedback("设备未连接", "error")
                else:
                    st.session_state.mdm_matrix_continuous_voltage = continuous_voltage
                    st.session_state.mdm_matrix_continuous_interval = continuous_interval
                    st.session_state.mdm_matrix_continuous_running = True
                    threading.Thread(
                        target=_matrix_continuous_loop,
                        args=(continuous_voltage, continuous_interval), daemon=True,
                    ).start()
                    set_tab2_feedback(f"矩阵持续发送中: {continuous_voltage}V, 间隔{continuous_interval}s", "success")
                    st.rerun()
        with col_cv_btn2:
            if st.button("⏹ 停止发送",
                         disabled=not st.session_state.mdm_matrix_continuous_running,
                         use_container_width=True, type="secondary", key="mdm_cv_stop"):
                st.session_state.mdm_matrix_continuous_running = False
                set_tab2_feedback("矩阵持续发送已停止", "info")
                st.rerun()
        if st.session_state.mdm_matrix_continuous_running:
            st.success(f"🔵 持续发送中: {continuous_voltage} V, 间隔 {continuous_interval} s")


# =============================================================================
# Sidebar: 矩阵设备连接 (Tab 2)
# =============================================================================

def render_sidebar() -> None:
    """渲染侧边栏: 矩阵设备 (MicroDM) 连接与全局设置。"""
    with st.sidebar:
        st.header("🔌 矩阵设备连接 (所有单元)")

        with st.container(border=True):
            st.session_state.mdm_use_wiring_map = st.checkbox(
                "使用 Wiring Map",
                value=st.session_state.mdm_use_wiring_map,
                help="从 wiring_map.json 自动加载控制器 IP",
            )
            if not st.session_state.mdm_use_wiring_map:
                st.session_state.mdm_custom_ips = st.text_input(
                    "自定义 IP (逗号分隔)",
                    value=st.session_state.mdm_custom_ips,
                    placeholder="192.168.0.101,192.168.0.102",
                )

        if not st.session_state.mdm_connected:
            if st.button("连接 MicroDM", type="primary", use_container_width=True, key="mdm_connect_btn"):
                with st.spinner("连接中..."):
                    if connect_dm():
                        st.rerun()
        else:
            st.success("✅ 已连接")
            if st.button("断开连接", use_container_width=True, key="mdm_disconnect_btn"):
                disconnect_dm()
                st.rerun()
            try:
                info = st.session_state.mdm_dm.get_hardware_info()
                st.caption(f"控制器: {info.get('connected_controllers', 0)}/{info.get('n_controllers', 0)}")
                st.caption(f"通道: {info.get('total_channels', 0)}")
            except Exception:
                pass

        if st.session_state.mdm_connection_error:
            st.error(st.session_state.mdm_connection_error)

        with st.container(border=True):
            st.markdown("##### 硬件规格")
            st.caption(f"矩阵: {GRID_SIZE} × {GRID_SIZE}")
            st.caption(f"通道: {TOTAL_CHANNELS}")
            st.caption(f"单控制器通道: {SINGLE_CHANNELS}")
            st.caption(f"硬件电压极限: [{HW_VOLTAGE_MIN}, {HW_VOLTAGE_MAX}] V")
            st.caption("安全范围可在各 Tab 内调整")


# =============================================================================
# Main App
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="Micro DM 控制面板",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Micro DM 微驱动器控制面板")
    st.caption("自适应光学变形镜控制 | 单单元控制 · 联合矩阵控制 · 实时电压可视化")

    _initialize_state()
    render_sidebar()

    tab1, tab2 = st.tabs(["单驱动器单元", "联合控制所有单元"])

    with tab1:
        render_tab_single_unit()

    with tab2:
        render_tab_matrix()


if __name__ == "__main__":
    main()
