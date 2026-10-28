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
import queue
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.dm.micro import R50Controller, VOLTAGE_MAX, VOLTAGE_MIN
from ao_shaping.utils.io.network import ping_reachable, tcp_reachable
# =============================================================================
# Constants
# =============================================================================

GRID_SIZE = 39
TOTAL_CHANNELS = GRID_SIZE * GRID_SIZE  # 1521
SINGLE_CHANNELS = 50  # 单个 R50Power 控制器通道数

# 硬件物理极限 (不可超越)。唯一来源: drivers/dm/micro/micro_constants.py
# (R50Power 数据手册), 与 gui/r50/r50_channel_select.py 同一写法。
HW_VOLTAGE_MIN = VOLTAGE_MIN
HW_VOLTAGE_MAX = VOLTAGE_MAX

#: offset±amp 须同时落在 [MIN, MAX] 内 ⇒ 2*amp <= 峰峰值; 更大振幅必被截断。
SINE_AMP_MAX = (HW_VOLTAGE_MAX - HW_VOLTAGE_MIN) / 2.0

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


# =============================================================================
# 后台下发循环的线程边界 (约定见 AGENTS.md "Streamlit background threads")
#
# 线程体只看传入的 ctrl / 参数快照 / threading.Event / queue.Queue;
# st.session_state 自始至终只由主线程读写。停止信号是 Event (不是被轮询的
# session_state 布尔标志), 回传走队列, 由 fragment 在 rerun 时排空。
# 与 gui/r50/r50_voltage_send.py 的 run_loop/start_loop/stop_loop 同构。
# =============================================================================

def _start_bg_loop(
    loop_fn: Callable[..., None],
    args: tuple,
    *,
    running_key: str,
    stop_key: str,
    queue_key: str,
    thread_name: str,
) -> None:
    """主线程启动后台循环: Event + Queue 建在这里, 句柄存进 session_state。"""
    stop_event = threading.Event()
    out_q: queue.Queue = queue.Queue()
    st.session_state[stop_key] = stop_event
    st.session_state[queue_key] = out_q
    st.session_state[running_key] = True
    threading.Thread(
        target=loop_fn,
        args=(*args, stop_event, out_q),
        daemon=True,
        name=thread_name,
    ).start()


def _stop_bg_loop(running_key: str, stop_key: str) -> None:
    """主线程请求停止: set Event, 循环在下一个 tick 退出。"""
    stop_event = st.session_state.get(stop_key)
    if stop_event is not None:
        stop_event.set()
    st.session_state[running_key] = False


def _drain_bg_loop(
    running_key: str,
    stop_key: str,
    queue_key: str,
    feedback_setter: Callable[[str, str], None],
    sample_sink: Callable[[float, float], None] | None = None,
    bounds_ref: Sequence[float] | None = None,
) -> None:
    """主线程每次 rerun 排空回传队列, 并把已自退的循环对账为停止。

    循环在 ``finally`` 里 set 自己的 Event 表示退出; 运行标志只由主线程清,
    所以界面不会和线程抢同一个标志。``bounds_ref`` 在运行期间刷新成当前软限值,
    等价于线程原本每 tick 读 ``mdm_vmin/mdm_vmax`` 的行为。
    """
    stop_event = st.session_state.get(stop_key)
    if stop_event is not None and stop_event.is_set():
        st.session_state[running_key] = False
    if bounds_ref is not None and st.session_state.get(running_key):
        bounds_ref[0] = float(st.session_state.mdm_vmin)
        bounds_ref[1] = float(st.session_state.mdm_vmax)
    out_q = st.session_state.get(queue_key)
    if out_q is None:
        return
    while True:
        try:
            kind, payload = out_q.get_nowait()
        except queue.Empty:
            break
        if kind == "sample" and sample_sink is not None:
            sample_sink(payload[0], payload[1])
        else:
            feedback_setter(str(payload), "error")


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

def render_voltage_limits(key_prefix: str, write_back: bool = True) -> None:
    """在两个 Tab 中渲染可编辑的电压上下限控件，写入共享 session_state。

    Args:
        key_prefix: 控件 key 前缀 (``single`` / ``matrix``)。
        write_back: 是否把控件值写回 ``mdm_vmin`` / ``mdm_vmax``。tab1 的
            fragment 每 0.15s 轮询, 若也写回会与 tab2 的写回互相覆盖
            (B1); 因此 tab1 传 ``write_back=False``, 仅 tab2 负责写回。
    """
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
        if write_back:
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
            # policy: closing the stale handle before a reconnect is best effort: a failure here must not block establishing the new connection
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
    # policy: a failed connect becomes operator-visible state (error string plus disconnected flag), and the half-open controller is dropped so a retry starts clean
    except Exception as e:
        st.session_state.mdm_single_connection_error = f"连接失败: {e}"
        st.session_state.mdm_single_connected = False
        st.session_state.mdm_single_controller = None
        logger.exception(f"Single unit connect failed: {e}")


def disconnect_single() -> None:
    """断开单个单元连接。"""
    _stop_bg_loop(_SINGLE_RUNNING_KEY, _SINGLE_STOP_KEY)
    try:
        if st.session_state.mdm_single_controller is not None:
            st.session_state.mdm_single_controller.close()
    # policy: teardown is best effort: the controller reference is cleared either way, so a refusing close must not leave the UI holding a dead handle
    except Exception as e:
        logger.exception(f"disconnect warning: {e}")
    st.session_state.mdm_single_controller = None
    st.session_state.mdm_single_connected = False
    st.session_state.mdm_single_connection_error = ""
    logger.info("Single unit disconnected")


# =============================================================================
# Tab 1: 单单元发送线程
# =============================================================================

def _clamp_voltage(voltage: float, bounds: Sequence[float]) -> float:
    """先夹到软限 (侧栏 vmin/vmax), 再夹到驱动硬上限。"""
    v = float(np.clip(float(voltage), float(bounds[0]), float(bounds[1])))
    return float(np.clip(v, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX))


def _apply_single_voltage(ctrl, channel: int, v: float, apply_all: bool) -> bool:
    """向单单元下发电压 (单通道或全部通道); 未连接时返回 False, 调用方据此不记历史。"""
    if ctrl is None:
        return False
    if apply_all:
        ctrl.set_all_channel_voltage(v)
    else:
        ctrl.set_channel_voltage(channel, v)
    return True


def _record_single_sample(voltage: float, t: float) -> None:
    """主线程把一次下发记进可视化历史 (deque 有 maxlen, 无需手工裁剪)。"""
    st.session_state.mdm_single_history.append(voltage)
    st.session_state.mdm_single_history_t.append(t)


#: 线程边界句柄所在的 session_state 键 (Event/Queue 由主线程创建与持有)。
_SINGLE_RUNNING_KEY = "mdm_single_running"
_SINGLE_STOP_KEY = "mdm_single_stop"
_SINGLE_Q_KEY = "mdm_single_q"
_SINGLE_BOUNDS_KEY = "mdm_single_bounds_ref"
_MATRIX_RUNNING_KEY = "mdm_matrix_continuous_running"
_MATRIX_STOP_KEY = "mdm_matrix_stop"
_MATRIX_Q_KEY = "mdm_matrix_q"


def _single_fixed_loop(
    channel: int, voltage: float, apply_all: bool, interval: float,
    ctrl, bounds_ref: Sequence[float],
    stop_event: threading.Event, out_q: queue.Queue,
) -> None:
    """固定电压持续下发线程。"""
    try:
        while not stop_event.is_set():
            v = _clamp_voltage(voltage, bounds_ref)
            if not _apply_single_voltage(ctrl, channel, v, apply_all):
                out_q.put(("error", "设备未连接，持续下发已停止"))
                break
            out_q.put(("sample", (v, time.time())))
            time.sleep(interval)
    # policy: a fixed-voltage send failure is reported to the UI thread through the queue; raising out of the loop thread would stop the hardware with no explanation
    except Exception as e:
        out_q.put(("error", f"固定电压下发异常: {e}"))
        logger.exception(f"固定电压下发异常: {e}")
    finally:
        stop_event.set()


def _single_periodic_loop(
    channel: int, apply_all: bool, wave_type: str,
    amp: float, offset: float, high: float, low: float,
    period: float, dt: float,
    ctrl, bounds_ref: Sequence[float],
    stop_event: threading.Event, out_q: queue.Queue,
) -> None:
    """周期波形下发线程，并记录下发电压用于可视化。"""
    period = max(period, 0.05)
    t0 = time.time()
    try:
        while not stop_event.is_set():
            elapsed = time.time() - t0
            phase = elapsed % period
            if wave_type == "sine":
                v = offset + amp * np.sin(2 * np.pi * phase / period)
            else:  # square
                v = high if phase < period / 2 else low
            v = _clamp_voltage(v, bounds_ref)
            if not _apply_single_voltage(ctrl, channel, v, apply_all):
                out_q.put(("error", "设备未连接，周期下发已停止"))
                break
            out_q.put(("sample", (v, time.time())))
            time.sleep(dt)
    # policy: a periodic waveform failure is reported through the queue and the finally clears the running flag -- an unhandled raise would leave the panel convinced it is still sweeping
    except Exception as e:
        out_q.put(("error", f"周期波形下发异常: {e}"))
        logger.exception(f"周期波形下发异常: {e}")
    finally:
        stop_event.set()


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

@st.fragment(run_every=0.15)
def _tab1_fragment():
    """Tab 1 单单元控制 fragment — 按钮自动刷新 + run_every 实时电压可视化."""
    _drain_bg_loop(
        _SINGLE_RUNNING_KEY, _SINGLE_STOP_KEY, _SINGLE_Q_KEY,
        set_tab1_feedback,
        sample_sink=_record_single_sample,
        bounds_ref=st.session_state.get(_SINGLE_BOUNDS_KEY),
    )

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
        # tab1 fragment 每 0.15s 轮询, 不写回共享状态 (由 tab2 负责写回)
        render_voltage_limits("single", write_back=False)

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
                            v = _clamp_voltage(
                                voltage,
                                (st.session_state.mdm_vmin, st.session_state.mdm_vmax),
                            )
                            if _apply_single_voltage(
                                st.session_state.mdm_single_controller,
                                int(channel), v, st.session_state.mdm_single_apply_all,
                            ):
                                _record_single_sample(v, time.time())
                            set_tab1_feedback(f"已下发 {voltage:.1f} V", "success")
                        # policy: a single-channel send failure is surfaced as panel feedback: the operator must know the channel was not changed
                        except Exception as e:
                            set_tab1_feedback(f"发送失败: {e}", "error")
            with col_send2:
                running = st.session_state.mdm_single_running
                if not running:
                    if st.button("🔁 持续保持", use_container_width=True, type="secondary",
                                 disabled=not st.session_state.mdm_single_connected,
                                 key="mdm_single_hold_start"):
                        if not st.session_state.mdm_single_connected:
                            set_tab1_feedback("设备未连接", "error")
                        else:
                            st.session_state[_SINGLE_BOUNDS_KEY] = [
                                float(st.session_state.mdm_vmin),
                                float(st.session_state.mdm_vmax),
                            ]
                            _start_bg_loop(
                                _single_fixed_loop,
                                (int(channel), voltage,
                                 st.session_state.mdm_single_apply_all, 0.1,
                                 st.session_state.mdm_single_controller,
                                 st.session_state[_SINGLE_BOUNDS_KEY]),
                                running_key=_SINGLE_RUNNING_KEY,
                                stop_key=_SINGLE_STOP_KEY,
                                queue_key=_SINGLE_Q_KEY,
                                thread_name="mdm-fixed-loop",
                            )
                            set_tab1_feedback(f"持续下发 {voltage:.1f} V", "success")
                else:
                    if st.button("⏹ 停止", use_container_width=True, type="secondary",
                                 key="mdm_single_hold_stop"):
                        _stop_bg_loop(_SINGLE_RUNNING_KEY, _SINGLE_STOP_KEY)
                        set_tab1_feedback("已停止持续下发", "info")
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
                    amp = st.number_input("振幅 (V)", min_value=0.0, max_value=SINE_AMP_MAX,
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
                        st.session_state[_SINGLE_BOUNDS_KEY] = [
                            float(st.session_state.mdm_vmin),
                            float(st.session_state.mdm_vmax),
                        ]
                        _start_bg_loop(
                            _single_periodic_loop,
                            (int(channel), st.session_state.mdm_single_apply_all,
                             wave_type, amp, offset, high, low, period, 0.05,
                             st.session_state.mdm_single_controller,
                             st.session_state[_SINGLE_BOUNDS_KEY]),
                            running_key=_SINGLE_RUNNING_KEY,
                            stop_key=_SINGLE_STOP_KEY,
                            queue_key=_SINGLE_Q_KEY,
                            thread_name="mdm-periodic-loop",
                        )
                        set_tab1_feedback(f"周期下发中: {wave_type}, 周期{period}s", "success")
            else:
                if st.button("⏹ 停止", type="primary", use_container_width=True,
                             key="mdm_single_period_stop"):
                    _stop_bg_loop(_SINGLE_RUNNING_KEY, _SINGLE_STOP_KEY)
                    set_tab1_feedback("周期下发已停止", "info")

    # ---- 实时电压可视化 ----
    st.divider()
    st.markdown("##### 当前下发电压")
    chart_placeholder = st.empty()
    fig = _render_voltage_history()
    chart_placeholder.pyplot(fig)
    plt_close_safe(fig)


def render_tab_single_unit() -> None:
    """渲染 Tab 1: 单个微驱动器单元控制。"""
    _tab1_fragment()


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
            # policy: same contract as the single-unit path -- dropping the previous handle is best effort before opening a new one
            except Exception as e:
                logger.warning(f"MicroDM close warning: {e}")
            st.session_state.mdm_dm = None
            st.session_state.mdm_connected = False

        from ao_shaping.drivers.dm.micro import MicroDM

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
    # policy: connect failure is a reported result (False plus a session error) for the caller to act on, not an exception a Streamlit rerun would swallow
    except Exception as e:
        error_msg = f"MicroDM 连接失败: {e}"
        st.session_state.mdm_connection_error = error_msg
        st.session_state.mdm_connected = False
        st.session_state.mdm_dm = None
        logger.exception(error_msg)
        return False


def disconnect_dm() -> None:
    """断开 MicroDM 连接。"""
    _stop_bg_loop(_MATRIX_RUNNING_KEY, _MATRIX_STOP_KEY)
    try:
        if st.session_state.mdm_dm is not None:
            st.session_state.mdm_dm.close()
        st.session_state.mdm_dm = None
        st.session_state.mdm_connected = False
        st.session_state.mdm_connection_error = ""
        logger.info("MicroDM disconnected")
    # policy: a refused close leaves the devices possibly still held, so the failure is recorded for the operator instead of pretending the session is disconnected
    except Exception as e:
        st.session_state.mdm_connection_error = f"断开连接失败: {e}"
        logger.exception(f"MicroDM disconnect failed: {e}")


def _matrix_continuous_loop(
    voltage: float, interval: float, dm, vmin: float, vmax: float,
    stop_event: threading.Event, out_q: queue.Queue,
) -> None:
    """矩阵持续电压发送后台线程。"""
    try:
        if dm is None:
            out_q.put(("error", "设备未连接，矩阵持续发送已停止"))
            return
        voltage_clipped = _clamp_voltage(voltage, (vmin, vmax))
        voltages = np.full(TOTAL_CHANNELS, voltage_clipped, dtype=np.float64)
        while not stop_event.is_set():
            dm.send_voltages(voltages)
            time.sleep(interval)
    # policy: a continuous matrix send failure is reported through the queue: the operator must learn the channels may still be driven, rather than losing the thread silently
    except Exception as e:
        out_q.put(("error", f"矩阵持续发送异常: {e}"))
        logger.exception(f"矩阵持续电压发送异常: {e}")
    finally:
        stop_event.set()


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
    # policy: if reset_all fails the actuators may still sit at their last voltages -- safety-relevant feedback, so it has to reach the panel and not raise
    except Exception as e:
        set_tab2_feedback(f"归零失败: {e}", "error")


@st.fragment()
def _tab2_fragment():
    """Tab 2 矩阵联合控制 fragment — 按钮点击自动刷新."""
    _drain_bg_loop(
        _MATRIX_RUNNING_KEY, _MATRIX_STOP_KEY, _MATRIX_Q_KEY,
        set_tab2_feedback,
    )
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
        with col_confirm2:
            if st.button("❌ 取消", use_container_width=True, key="mdm_confirm_no"):
                st.session_state.mdm_confirm_action = None
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
            elif edit_mode == "row":
                row_idx = st.number_input("行号", 0, 38, 0, key="mdm_edit_row_idx")
                if st.button(f"设置第 {row_idx} 行", type="primary", use_container_width=True, key="mdm_btn_set_row"):
                    st.session_state.mdm_matrix[row_idx, :] = fill_value
                    set_tab2_feedback(f"第 {row_idx} 行 ({GRID_SIZE}单元) → {fill_value} V", "success")
            elif edit_mode == "col":
                col_idx = st.number_input("列号", 0, 38, 0, key="mdm_edit_col_idx")
                if st.button(f"设置第 {col_idx} 列", type="primary", use_container_width=True, key="mdm_btn_set_col"):
                    st.session_state.mdm_matrix[:, col_idx] = fill_value
                    set_tab2_feedback(f"第 {col_idx} 列 ({GRID_SIZE}单元) → {fill_value} V", "success")

    # ---- 批量操作 ----
    st.divider()
    st.markdown("##### 图案生成")
    col_batch1, col_batch2, col_batch3, col_batch4, col_batch5, col_batch6 = st.columns(6)
    with col_batch1:
        if st.button("🔄 归零", use_container_width=True, key="mdm_btn_reset_zero"):
            st.session_state.mdm_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
            set_tab2_feedback("矩阵已归零", "success")
    with col_batch2:
        if st.button("📋 全填充", use_container_width=True, key="mdm_btn_fill_all"):
            st.session_state.mdm_matrix[:, :] = fill_value
            set_tab2_feedback(f"全部 {TOTAL_CHANNELS} 单元 → {fill_value} V", "success")
    with col_batch3:
        if st.button("🎲 随机", use_container_width=True, key="mdm_btn_random"):
            st.session_state.mdm_matrix = np.random.uniform(vmin, vmax, (GRID_SIZE, GRID_SIZE))
            set_tab2_feedback("矩阵已随机生成", "success")
    with col_batch4:
        if st.button("🌊 正弦", use_container_width=True, key="mdm_btn_sine_pattern"):
            x = np.linspace(0, 4 * np.pi, GRID_SIZE)
            y = np.linspace(0, 4 * np.pi, GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            pattern = np.clip(((vmin + vmax) / 2.0) + ((vmax - vmin) / 2.0) * np.sin(X) * np.cos(Y), vmin, vmax)
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("正弦图案已生成", "success")
    with col_batch5:
        if st.button("🔺 高斯", use_container_width=True, key="mdm_btn_gaussian"):
            x = np.linspace(-3, 3, GRID_SIZE)
            y = np.linspace(-3, 3, GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            pattern = np.clip(vmax - (vmax - vmin) * (1.0 - np.exp(-(X ** 2 + Y ** 2) / 2.0)), vmin, vmax)
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("高斯图案已生成", "success")
    with col_batch6:
        if st.button("📐 渐变", use_container_width=True, key="mdm_btn_gradient"):
            pattern = np.tile(np.linspace(vmin, vmax, GRID_SIZE), (GRID_SIZE, 1)).T
            st.session_state.mdm_matrix = pattern
            set_tab2_feedback("渐变图案已生成", "success")

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
                # policy: a failed apply is reported in-panel and the session matrix is deliberately left as edited, so the operator can retry without re-entering it
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
                # policy: a failed read-back is reported and the displayed matrix is kept -- half-read voltages are worse than stale ones
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
                else:
                    set_tab2_feedback(f"形状不匹配: 期望 ({GRID_SIZE},{GRID_SIZE}), 实际 {data.shape}", "error")
            # policy: a rejected upload is a user-correctable input error: report it in the panel and keep the existing matrix untouched
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
                    _start_bg_loop(
                        _matrix_continuous_loop,
                        (continuous_voltage, continuous_interval, st.session_state.mdm_dm,
                         float(st.session_state.mdm_vmin), float(st.session_state.mdm_vmax)),
                        running_key=_MATRIX_RUNNING_KEY,
                        stop_key=_MATRIX_STOP_KEY,
                        queue_key=_MATRIX_Q_KEY,
                        thread_name="mdm-matrix-loop",
                    )
                    set_tab2_feedback(f"矩阵持续发送中: {continuous_voltage}V, 间隔{continuous_interval}s", "success")
        with col_cv_btn2:
            if st.button("⏹ 停止发送",
                         disabled=not st.session_state.mdm_matrix_continuous_running,
                         use_container_width=True, type="secondary", key="mdm_cv_stop"):
                _stop_bg_loop(_MATRIX_RUNNING_KEY, _MATRIX_STOP_KEY)
                set_tab2_feedback("矩阵持续发送已停止", "info")
        if st.session_state.mdm_matrix_continuous_running:
            st.success(f"🔵 持续发送中: {continuous_voltage} V, 间隔 {continuous_interval} s")


def render_tab_matrix() -> None:
    """渲染 Tab 2: 39×39 矩阵联合控制。"""
    _tab2_fragment()


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
            # policy: the sidebar caption is decorative device info; a capability read that fails must not break the whole page rerun
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
