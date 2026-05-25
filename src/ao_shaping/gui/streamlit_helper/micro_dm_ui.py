"""
Micro DM 39×39 矩阵控制 UI (Streamlit) - 优化版

功能:
1. Tab 1 - 单控制单元连接: 参数配置、连接状态显示、持续发送正弦电压/01电压
2. Tab 2 - 39×39 矩阵直接控制: 交互式网格界面、单元格点击修改、区域批量调整、
   数值热力图可视化、一键应用/重置操作

优化点:
- 连接状态置顶，操作反馈集中管理
- 使用 expander 折叠连续发送区域，减少页面长度
- 使用 segmented_control 替代 radio，更紧凑的编辑模式选择
- 危险操作增加确认对话框
- 使用 st.form 减少不必要的 rerun
- 侧边栏精简，移除重复操作
- 热力图渲染优化

使用方式:
    streamlit run src/ao_shaping/gui/streamlit_helper/micro_dm_ui.py
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
from loguru import logger

# =============================================================================
# Path Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # AO-shaping/
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# =============================================================================
# Constants
# =============================================================================

GRID_SIZE = 39
VOLTAGE_MIN = -20.0
VOLTAGE_MAX = 120.0
TOTAL_CHANNELS = GRID_SIZE * GRID_SIZE  # 1521


# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量。"""
    # ---- 设备连接状态 ----
    st.session_state.setdefault("mdm_dm", None)
    st.session_state.setdefault("mdm_connected", False)
    st.session_state.setdefault("mdm_connection_error", "")

    # ---- Tab 1: 单控制单元参数 ----
    st.session_state.setdefault("mdm_single_channel", 0)
    st.session_state.setdefault("mdm_single_voltage", 0.0)
    st.session_state.setdefault("mdm_sine_amplitude", 50.0)
    st.session_state.setdefault("mdm_sine_frequency", 1.0)
    st.session_state.setdefault("mdm_sine_offset", 50.0)
    st.session_state.setdefault("mdm_sine_running", False)
    st.session_state.setdefault("mdm_01_running", False)
    st.session_state.setdefault("mdm_01_high_voltage", 100.0)
    st.session_state.setdefault("mdm_01_low_voltage", 0.0)
    st.session_state.setdefault("mdm_01_interval", 1.0)

    # ---- 反馈 (Tab 独立) ----
    st.session_state.setdefault("mdm_tab1_feedback", "")
    st.session_state.setdefault("mdm_tab1_feedback_type", "")
    st.session_state.setdefault("mdm_tab2_feedback", "")
    st.session_state.setdefault("mdm_tab2_feedback_type", "")

    # ---- Tab 2: 39×39 矩阵参数 ----
    st.session_state.setdefault("mdm_matrix", np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64))
    st.session_state.setdefault("mdm_matrix_loaded", False)
    st.session_state.setdefault("mdm_heatmap_vmin", VOLTAGE_MIN)
    st.session_state.setdefault("mdm_heatmap_vmax", VOLTAGE_MAX)
    st.session_state.setdefault("mdm_colormap", "coolwarm")
    st.session_state.setdefault("mdm_edit_mode", "single")
    st.session_state.setdefault("mdm_fill_value", 0.0)

    # ---- Tab 2: 矩阵动态持续电压 ----
    st.session_state.setdefault("mdm_matrix_continuous_voltage", 0.0)
    st.session_state.setdefault("mdm_matrix_continuous_interval", 0.1)
    st.session_state.setdefault("mdm_matrix_continuous_running", False)

    # ---- 连接参数 ----
    st.session_state.setdefault("mdm_use_wiring_map", True)
    st.session_state.setdefault("mdm_custom_ips", "")

    # ---- 确认对话框状态 ----
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
        # 清除反馈
        st.session_state[fb_key] = ""
        st.session_state[ft_key] = ""


# =============================================================================
# Device Connection
# =============================================================================

def connect_dm() -> bool:
    """连接 MicroDM 设备。"""
    try:
        # 先断开已有连接
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

        # 加载当前电压到矩阵
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
    """断开 MicroDM 设备连接。"""
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


# =============================================================================
# Tab 1: 单控制单元控制
# =============================================================================

def _sine_wave_loop(channel: int, amplitude: float, frequency: float, offset: float) -> None:
    """正弦电压发送后台线程。"""
    dm = st.session_state.mdm_dm
    if dm is None:
        return

    t = 0.0
    dt = 0.05  # 50ms 更新间隔
    try:
        while st.session_state.mdm_sine_running:
            voltage = offset + amplitude * np.sin(2 * np.pi * frequency * t)
            voltage = np.clip(voltage, VOLTAGE_MIN, VOLTAGE_MAX)
            dm.set_channel_voltage(channel, voltage)
            t += dt
            time.sleep(dt)
    except Exception as e:
        st.session_state.mdm_sine_running = False
        logger.exception(f"正弦电压发送异常: {e}")
        set_tab1_feedback(f"正弦发送异常: {e}", "error")


def _01_voltage_loop(channel: int, high: float, low: float, interval: float) -> None:
    """01电压切换后台线程。"""
    dm = st.session_state.mdm_dm
    if dm is None:
        return

    state = True
    try:
        while st.session_state.mdm_01_running:
            voltage = high if state else low
            voltage = np.clip(voltage, VOLTAGE_MIN, VOLTAGE_MAX)
            dm.set_channel_voltage(channel, voltage)
            state = not state
            time.sleep(interval)
    except Exception as e:
        st.session_state.mdm_01_running = False
        logger.exception(f"01电压切换异常: {e}")
        set_tab1_feedback(f"01切换异常: {e}", "error")


def _render_waveform_preview(amplitude: float, frequency: float, offset: float) -> None:
    """渲染正弦波形预览图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6, 2.5))
    t = np.linspace(0, 3 / max(frequency, 0.1), 500)
    v = offset + amplitude * np.sin(2 * np.pi * frequency * t)
    ax.plot(t, v, linewidth=1.5, color="#1f77b4")
    ax.axhline(y=offset, color="gray", linestyle="--", alpha=0.5, label=f"偏置 {offset:.0f}V")
    ax.axhline(y=VOLTAGE_MAX, color="red", linestyle=":", alpha=0.4, label=f"上限 {VOLTAGE_MAX:.0f}V")
    ax.axhline(y=VOLTAGE_MIN, color="red", linestyle=":", alpha=0.4, label=f"下限 {VOLTAGE_MIN:.0f}V")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("电压 (V)")
    ax.set_title("正弦波形预览")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_tab_single_unit() -> None:
    """渲染 Tab 1: 单控制单元连接与控制。"""

    # ---- 顶部状态栏 ----
    col_status1, col_status2, col_status3 = st.columns([1, 2, 2])
    with col_status1:
        if st.session_state.mdm_connected:
            st.success("✅ 已连接")
        else:
            st.error("❌ 未连接")
    with col_status2:
        if st.session_state.mdm_connected:
            try:
                info = st.session_state.mdm_dm.get_hardware_info()
                st.caption(
                    f"控制器 {info.get('connected_controllers', 0)}/{info.get('n_controllers', 0)} | "
                    f"通道 {info.get('total_channels', 0)} | "
                    f"电压 [{info.get('voltage_range', [VOLTAGE_MIN, VOLTAGE_MAX])[0]:.0f}, "
                    f"{info.get('voltage_range', [VOLTAGE_MIN, VOLTAGE_MAX])[1]:.0f}] V"
                )
            except Exception:
                pass
    with col_status3:
        if st.session_state.mdm_connection_error:
            st.caption(f"错误: {st.session_state.mdm_connection_error}")

    show_and_clear_feedback("tab1")

    # ---- 参数配置 + 手动发送 (Form 减少 rerun) ----
    with st.form("mdm_single_form", border=True):
        st.markdown("##### 参数配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            channel = st.number_input(
                "通道号",
                min_value=0,
                max_value=TOTAL_CHANNELS - 1,
                value=st.session_state.mdm_single_channel,
                step=1,
                key="mdm_input_channel",
            )
        with col2:
            voltage = st.number_input(
                "目标电压 (V)",
                min_value=VOLTAGE_MIN,
                max_value=VOLTAGE_MAX,
                value=st.session_state.mdm_single_voltage,
                step=1.0,
                format="%.1f",
                key="mdm_input_voltage",
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            send_clicked = st.form_submit_button("⚡ 发送电压", type="primary", use_container_width=True)

        if send_clicked:
            if not st.session_state.mdm_connected:
                set_tab1_feedback("设备未连接", "error")
            else:
                try:
                    st.session_state.mdm_dm.set_channel_voltage(channel, voltage)
                    st.session_state.mdm_single_channel = channel
                    st.session_state.mdm_single_voltage = voltage
                    set_tab1_feedback(f"通道 {channel} → {voltage} V", "success")
                except Exception as e:
                    set_tab1_feedback(f"发送失败: {e}", "error")
            st.rerun()

    # ---- 读取当前电压 ----
    col_read1, col_read2 = st.columns([1, 4])
    with col_read1:
        if st.button("📖 读取电压", use_container_width=True, key="mdm_btn_read_voltage"):
            if not st.session_state.mdm_connected:
                set_tab1_feedback("设备未连接", "error")
            else:
                try:
                    voltages = st.session_state.mdm_dm.get_actuator_positions()
                    current_v = voltages[channel]
                    set_tab1_feedback(f"通道 {channel} 当前: {current_v:.2f} V", "info")
                except Exception as e:
                    set_tab1_feedback(f"读取失败: {e}", "error")
            st.rerun()

    # ---- 连续发送区域 (Expander 折叠) ----
    st.divider()

    # 正弦电压
    with st.expander("🌊 正弦电压发送", expanded=False):
        col_sin1, col_sin2, col_sin3 = st.columns(3)
        with col_sin1:
            amplitude = st.number_input(
                "振幅 (V)", min_value=0.0, max_value=70.0,
                value=st.session_state.mdm_sine_amplitude, step=1.0, format="%.1f",
                key="mdm_sin_amplitude",
            )
        with col_sin2:
            frequency = st.number_input(
                "频率 (Hz)", min_value=0.1, max_value=10.0,
                value=st.session_state.mdm_sine_frequency, step=0.1, format="%.1f",
                key="mdm_sin_frequency",
            )
        with col_sin3:
            offset = st.number_input(
                "偏置 (V)", min_value=VOLTAGE_MIN, max_value=VOLTAGE_MAX,
                value=st.session_state.mdm_sine_offset, step=1.0, format="%.1f",
                key="mdm_sin_offset",
            )

        # 波形预览
        _render_waveform_preview(amplitude, frequency, offset)

        # 参数验证
        sin_max = offset + amplitude
        sin_min = offset - amplitude
        if sin_max > VOLTAGE_MAX or sin_min < VOLTAGE_MIN:
            st.warning(f"⚠️ 电压范围 [{sin_min:.1f}, {sin_max:.1f}] V 超出硬件限制，将自动截断")

        col_sin_btn1, col_sin_btn2 = st.columns(2)
        with col_sin_btn1:
            if st.button("▶ 开始", type="primary", disabled=st.session_state.mdm_sine_running,
                         use_container_width=True, key="mdm_sin_start"):
                if not st.session_state.mdm_connected:
                    set_tab1_feedback("设备未连接", "error")
                else:
                    st.session_state.mdm_sine_running = True
                    threading.Thread(
                        target=_sine_wave_loop,
                        args=(channel, amplitude, frequency, offset), daemon=True,
                    ).start()
                    set_tab1_feedback(f"正弦发送中: Ch{channel}, {amplitude}V, {frequency}Hz, 偏置{offset}V", "success")
                    st.rerun()
        with col_sin_btn2:
            if st.button("⏹ 停止", disabled=not st.session_state.mdm_sine_running,
                         use_container_width=True, type="secondary", key="mdm_sin_stop"):
                st.session_state.mdm_sine_running = False
                set_tab1_feedback("正弦发送已停止", "info")
                st.rerun()

        if st.session_state.mdm_sine_running:
            st.success("🔵 正弦发送中...")

    # 01 电压切换
    with st.expander("🔲 01 电压切换", expanded=False):
        col_01_1, col_01_2, col_01_3 = st.columns(3)
        with col_01_1:
            high_v = st.number_input(
                "高电平 (V)", min_value=VOLTAGE_MIN, max_value=VOLTAGE_MAX,
                value=st.session_state.mdm_01_high_voltage, step=1.0, format="%.1f",
                key="mdm_01_high",
            )
        with col_01_2:
            low_v = st.number_input(
                "低电平 (V)", min_value=VOLTAGE_MIN, max_value=VOLTAGE_MAX,
                value=st.session_state.mdm_01_low_voltage, step=1.0, format="%.1f",
                key="mdm_01_low",
            )
        with col_01_3:
            interval = st.number_input(
                "间隔 (s)", min_value=0.1, max_value=60.0,
                value=st.session_state.mdm_01_interval, step=0.1, format="%.1f",
                key="mdm_01_interval_input",
            )

        col_01_btn1, col_01_btn2 = st.columns(2)
        with col_01_btn1:
            if st.button("▶ 开始", type="primary", disabled=st.session_state.mdm_01_running,
                         use_container_width=True, key="mdm_01_start"):
                if not st.session_state.mdm_connected:
                    set_tab1_feedback("设备未连接", "error")
                else:
                    st.session_state.mdm_01_running = True
                    threading.Thread(
                        target=_01_voltage_loop,
                        args=(channel, high_v, low_v, interval), daemon=True,
                    ).start()
                    set_tab1_feedback(f"01切换中: Ch{channel}, 高{high_v}V/低{low_v}V, {interval}s", "success")
                    st.rerun()
        with col_01_btn2:
            if st.button("⏹ 停止", disabled=not st.session_state.mdm_01_running,
                         use_container_width=True, type="secondary", key="mdm_01_stop"):
                st.session_state.mdm_01_running = False
                set_tab1_feedback("01切换已停止", "info")
                st.rerun()

        if st.session_state.mdm_01_running:
            st.success("🔵 01切换中...")


# =============================================================================
# Tab 2: 39×39 矩阵控制
# =============================================================================

def _matrix_continuous_loop(voltage: float, interval: float) -> None:
    """矩阵持续电压发送后台线程。"""
    dm = st.session_state.mdm_dm
    if dm is None:
        return

    try:
        voltage_clipped = np.clip(voltage, VOLTAGE_MIN, VOLTAGE_MAX)
        voltages = np.full(TOTAL_CHANNELS, voltage_clipped, dtype=np.float64)
        while st.session_state.mdm_matrix_continuous_running:
            dm.send_voltages(voltages)
            time.sleep(interval)
    except Exception as e:
        st.session_state.mdm_matrix_continuous_running = False
        logger.exception(f"矩阵持续电压发送异常: {e}")
        set_tab2_feedback(f"矩阵持续发送异常: {e}", "error")


def _render_heatmap(matrix, vmin, vmax, colormap):
    """渲染热力图（独立函数便于缓存）。"""
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


def render_tab_matrix() -> None:
    """渲染 Tab 2: 39×39 矩阵直接控制。"""

    # ---- 顶部状态提示 ----
    if not st.session_state.mdm_connected:
        st.warning("⚠️ 设备未连接，矩阵操作仅在本地预览")

    show_and_clear_feedback("tab2")

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
        # 热力图
        matrix = st.session_state.mdm_matrix.copy()
        fig = _render_heatmap(matrix, st.session_state.mdm_heatmap_vmin,
                              st.session_state.mdm_heatmap_vmax, st.session_state.mdm_colormap)
        st.pyplot(fig)

        # 统计信息
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
        # 热力图参数
        with st.container(border=True):
            st.markdown("##### 热力图设置")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.number_input(
                    "最小值", min_value=VOLTAGE_MIN, max_value=VOLTAGE_MAX,
                    value=st.session_state.mdm_heatmap_vmin, step=5.0, format="%.1f",
                    key="mdm_vmin",
                )
            with col_h2:
                st.number_input(
                    "最大值", min_value=VOLTAGE_MIN, max_value=VOLTAGE_MAX,
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

        # 编辑模式 (segmented_control)
        with st.container(border=True):
            st.markdown("##### 编辑模式")

            edit_mode_raw = st.segmented_control(
                "模式",
                options=["single", "rect", "row", "col"],
                default="single",
                selection_mode="single",
                key="mdm_edit_mode_seg",
            )
            # segmented_control 返回字符串列表，取第一个
            if isinstance(edit_mode_raw, list):
                edit_mode = edit_mode_raw[0] if edit_mode_raw else "single"
            else:
                edit_mode = edit_mode_raw if edit_mode_raw else "single"

            fill_value = st.number_input(
                "填充值 (V)", min_value=VOLTAGE_MIN, max_value=VOLTAGE_MAX,
                value=st.session_state.mdm_fill_value, step=1.0, format="%.1f",
                key="mdm_fill_value_input",
            )
            st.session_state.mdm_fill_value = fill_value

            # 根据模式显示不同输入
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
            st.session_state.mdm_matrix = np.random.uniform(VOLTAGE_MIN, VOLTAGE_MAX, (GRID_SIZE, GRID_SIZE))
            set_tab2_feedback("矩阵已随机生成", "success")
            st.rerun()
    with col_batch4:
        if st.button("🌊 正弦", use_container_width=True, key="mdm_btn_sine_pattern"):
            x = np.linspace(0, 4 * np.pi, GRID_SIZE)
            y = np.linspace(0, 4 * np.pi, GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            pattern = np.clip(50.0 + 50.0 * np.sin(X) * np.cos(Y), VOLTAGE_MIN, VOLTAGE_MAX)
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("正弦图案已生成", "success")
            st.rerun()
    with col_batch5:
        if st.button("🔺 高斯", use_container_width=True, key="mdm_btn_gaussian"):
            x = np.linspace(-3, 3, GRID_SIZE)
            y = np.linspace(-3, 3, GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            pattern = np.clip(50.0 + 70.0 * np.exp(-(X ** 2 + Y ** 2) / 2.0), VOLTAGE_MIN, VOLTAGE_MAX)
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("高斯图案已生成", "success")
            st.rerun()
    with col_batch6:
        if st.button("📐 渐变", use_container_width=True, key="mdm_btn_gradient"):
            pattern = np.tile(np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, GRID_SIZE), (GRID_SIZE, 1)).T
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
                # 使用确认对话框
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
                    st.session_state.mdm_matrix = data.astype(np.float64)
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
                "持续电压值 (V)",
                min_value=VOLTAGE_MIN,
                max_value=VOLTAGE_MAX,
                value=st.session_state.mdm_matrix_continuous_voltage,
                step=1.0,
                format="%.1f",
                key="mdm_cv_voltage",
            )
        with col_cv2:
            continuous_interval = st.number_input(
                "发送间隔 (s)",
                min_value=0.01,
                max_value=10.0,
                value=st.session_state.mdm_matrix_continuous_interval,
                step=0.01,
                format="%.2f",
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


def _hw_reset_action() -> None:
    """硬件归零回调。"""
    try:
        st.session_state.mdm_dm.reset_all()
        st.session_state.mdm_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        set_tab2_feedback("硬件已归零", "success")
    except Exception as e:
        set_tab2_feedback(f"归零失败: {e}", "error")


# =============================================================================
# Sidebar: 设备连接
# =============================================================================

def render_sidebar() -> None:
    """渲染侧边栏: 设备连接与全局设置。"""
    with st.sidebar:
        st.header("🔌 设备连接")

        # 连接设置
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

        # 连接/断开
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

            # 硬件信息
            try:
                info = st.session_state.mdm_dm.get_hardware_info()
                st.caption(f"控制器: {info.get('connected_controllers', 0)}/{info.get('n_controllers', 0)}")
                st.caption(f"通道: {info.get('total_channels', 0)}")
            except Exception:
                pass

        # 错误信息
        if st.session_state.mdm_connection_error:
            st.error(st.session_state.mdm_connection_error)

        # 硬件信息
        with st.container(border=True):
            st.markdown("##### 硬件规格")
            st.caption(f"矩阵: {GRID_SIZE} × {GRID_SIZE}")
            st.caption(f"通道: {TOTAL_CHANNELS}")
            st.caption(f"电压: [{VOLTAGE_MIN}, {VOLTAGE_MAX}] V")


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

    st.title("🔬 Micro DM 39×39 矩阵控制面板")
    st.caption("自适应光学变形镜控制 | 单单元控制 · 矩阵批量操作 · 实时可视化")

    _initialize_state()
    render_sidebar()

    tab1, tab2 = st.tabs(["单控制单元", "39×39 矩阵"])

    with tab1:
        render_tab_single_unit()

    with tab2:
        render_tab_matrix()


if __name__ == "__main__":
    main()
