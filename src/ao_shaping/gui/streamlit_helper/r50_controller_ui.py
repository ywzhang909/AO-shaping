"""R50 控制器控制面板 (Streamlit) — 薄编排层。

面向 ``MicroDM.py`` 中 :class:`R50Controller` / :class:`MicroDM` 的 UI。

可测试逻辑位于兄弟模块:
- r50_channel_select: 配置 / CSV 索引 / 通道选择 (纯逻辑)
- r50_connection:     连接工厂 / 仿真设备 / 下电安全 (纯逻辑)
- r50_voltage_send:   裁剪 / 批量下发 / 发送循环 (纯逻辑)

本文件只负责 session_state、st 渲染、反馈与调试日志。

使用方式:
    streamlit run src/ao_shaping/gui/streamlit_helper/r50_controller_ui.py
"""

from __future__ import annotations

import collections
import json
import queue
import socket
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    MicroDM,
    voltages_to_payload,
)
from ao_shaping.gui.streamlit_helper.r50_channel_select import (
    CFG,
    DEBUG_HOST,
    DEBUG_LOG_MAX,
    DEBUG_PORT,
    DM_NUM_ACTUATORS,
    GRID_SIZE,
    HW_VOLTAGE_MAX,
    HW_VOLTAGE_MIN,
    P,
    REFRESH_INTERVAL,
    SINGLE_CHANNELS,
    ChannelInfo,
    ChannelSelection,
    build_all_units,
    build_groups,
    get_channel_info,
    jc_build_ip_index,
    jc_build_wiring_index,
    jc_matrix_to_flat,
    load_csv,
)
from ao_shaping.gui.streamlit_helper.r50_connection import (
    SimulatedMicroDM,
    create_controller,
    ping_reachable,
    power_off_and_close,
    set_relay,
    tcp_reachable,
)
from ao_shaping.gui.streamlit_helper.r50_voltage_send import (
    SendResult,
    alt_tick,
    apply_group_controllers,
    apply_joint,
    apply_single_controller,
    apply_units_via_controller,
    build_bulk_array,
    clip_voltage,
    hold_tick,
    sine_tick,
    start_loop,
    stop_loop,
)


# =============================================================================
# 调试 TCP 客户端 (调试日志转发到外部 TCP 监听器)
# =============================================================================

class DebugTcpClient:
    """外部 TCP 调试客户端: 将调试日志 JSON 发送到 ``127.0.0.1:9999``。"""

    def __init__(self, host: str = DEBUG_HOST, port: int = DEBUG_PORT) -> None:
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self._connected = False
        self._last_attempt = 0.0
        self._reconnect_interval = 5.0

    def configure(self, host: str, port: int) -> None:
        """配置新的目标地址并断开旧连接。"""
        if host != self.host or port != self.port:
            self.disconnect()
            self.host = host
            self.port = port

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """建立连接 (带 5s 退避, 失败静默)。"""
        now = time.time()
        if now - self._last_attempt < self._reconnect_interval:
            return
        self._last_attempt = now
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            self.sock.connect((self.host, self.port))
            self._connected = True
            logger.debug(f"[DebugTcpClient] 已连接 {self.host}:{self.port}")
        except OSError:
            self._connected = False
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def send(self, data: dict[str, Any]) -> None:
        """发送一行 JSON 日志。"""
        if not self._connected:
            return
        try:
            payload = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
            self.sock.sendall(payload)
        except OSError:
            self._connected = False
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def disconnect(self) -> None:
        """断开连接。"""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._connected = False


# =============================================================================
# 反馈助手 (单控制器 / 联合 / 分组共用, 前缀区分)
# =============================================================================

def _set_feedback(message: str, msg_type: str = "info", prefix: str = "") -> None:
    """写入反馈消息。prefix ("jc"/"gc") 为空时使用单控制器前缀。"""
    p = f"{P}_{prefix}" if prefix else P
    st.session_state[f"{p}_feedback"] = message
    st.session_state[f"{p}_feedback_type"] = msg_type


def _show_feedback(prefix: str = "") -> None:
    """渲染当前反馈消息 (显示后清除)。"""
    p = f"{P}_{prefix}" if prefix else P
    msg = st.session_state.get(f"{p}_feedback", "")
    msg_type = st.session_state.get(f"{p}_feedback_type", "info")
    if msg:
        if msg_type == "success":
            st.success(msg)
        elif msg_type == "error":
            st.error(msg)
        elif msg_type == "warning":
            st.warning(msg)
        else:
            st.info(msg)
        st.session_state[f"{p}_feedback"] = ""
        st.session_state[f"{p}_feedback_type"] = "info"


# =============================================================================
# 调试日志助手 (外部 TCP 转发 + 本地捕获 + 操作日志)
# =============================================================================

_local_debug_lock = threading.Lock()
_local_debug_buffer: collections.deque[str] = collections.deque(maxlen=256)


def _debug_log_packet(cmd_name: str, packet: bytes) -> None:
    """将原始数据包写入调试日志 (指令日志 + 操作日志 + 外部 TCP)。"""
    hex_str = packet.hex()
    ts = time.strftime("%H:%M:%S", time.localtime())
    st.session_state[f"{P}_debug_log"].append(f"[{ts}] {cmd_name} {hex_str}")
    _debug_add_op(cmd_name, f"packet={hex_str}")
    if st.session_state.get(f"{P}_debug", False):
        st.session_state[f"{P}_debug_tcp_client"].send(
            {"cmd": cmd_name, "hex": hex_str, "ts": time.time()}
        )


def _debug_tcp_connect() -> None:
    """(重新)连接外部 TCP 调试通道。"""
    if st.session_state.get(f"{P}_debug", False):
        st.session_state[f"{P}_debug_tcp_client"].connect()


def _debug_tcp_disconnect() -> None:
    """断开外部 TCP 调试通道。"""
    st.session_state[f"{P}_debug_tcp_client"].disconnect()


def _debug_add_op(operation: str, detail: str, ip: str = "") -> None:
    """向操作日志追加一行 (仅 UI 层记录)。"""
    ts = time.strftime("%H:%M:%S", time.localtime(time.time()))
    line = f"[{ts}] {operation}"
    if ip:
        line += f" @{ip}"
    if detail:
        line += f" | {detail}"
    st.session_state[f"{P}_debug_op_log"].append(line)


# ---------------------------------------------------------------------------
# 本地调试 TCP 服务器 (捕获外部客户端发来的调试日志)
# 注意: 后台线程不读取 session_state — 用 threading.Event 控制启停。
# ---------------------------------------------------------------------------

_local_debug_server_event = threading.Event()
_local_debug_server_error: str | None = None


def _drain_local_debug_buffer() -> None:
    """将后台线程捕获的调试行搬运到 session_state (主线程执行)。"""
    with _local_debug_lock:
        lines = list(_local_debug_buffer)
        _local_debug_buffer.clear()
    if lines:
        st.session_state[f"{P}_local_debug_logs"].extend(lines)


def _debug_tcp_start_local_server() -> None:
    """启动本地调试 TCP 服务器 (后台线程, 127.0.0.1:port)。"""
    global _local_debug_server_error
    port = int(st.session_state.get(f"{P}_debug_tcp_port", DEBUG_PORT))
    _local_debug_server_error = None
    _local_debug_server_event.set()
    st.session_state[f"{P}_local_debug_server"] = True
    st.session_state[f"{P}_local_debug_logs"].clear()
    st.session_state[f"{P}_debug_tcp_host"] = "127.0.0.1"
    st.session_state[f"{P}_debug_tcp_port"] = port
    client = st.session_state[f"{P}_debug_tcp_client"]
    client.configure("127.0.0.1", port)
    client.connect()

    def _run() -> None:
        server_sock: socket.socket | None = None
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("127.0.0.1", port))
            server_sock.listen(1)
            server_sock.settimeout(0.2)
            while _local_debug_server_event.is_set():
                try:
                    conn, _addr = server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    conn.settimeout(0.2)
                    while _local_debug_server_event.is_set():
                        data = conn.recv(4096)
                        if not data:
                            break
                        for line in data.decode("utf-8", errors="replace").splitlines():
                            if line.strip():
                                with _local_debug_lock:
                                    _local_debug_buffer.append(line)
                except socket.timeout:
                    pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        except OSError as exc:
            _local_debug_server_error = str(exc)
            logger.error(f"[DebugTcpServer] 绑定失败: {exc}")
        finally:
            _local_debug_server_event.clear()
            if server_sock is not None:
                try:
                    server_sock.close()
                except OSError:
                    pass

    threading.Thread(target=_run, name="debug-tcp-server", daemon=True).start()


def _debug_tcp_stop_local_server() -> None:
    """停止本地调试 TCP 服务器。"""
    _local_debug_server_event.clear()
    st.session_state[f"{P}_local_debug_server"] = False
    with _local_debug_lock:
        _local_debug_buffer.clear()


# =============================================================================
# 通道标签助手 (单控制器 IP 下)
# =============================================================================

def _current_ip_suffix() -> int | None:
    """当前单控制器 IP 的末段 (int), 解析失败返回 None。"""
    ip = st.session_state.get(f"{P}_ip", "").strip()
    try:
        return int(ip.split(".")[-1])
    except (ValueError, IndexError):
        return None


def _get_channel_info(channel: int) -> ChannelInfo | None:
    """当前 IP 下 channel (0-based) 的 ChannelInfo。"""
    suffix = _current_ip_suffix()
    if suffix is None:
        return None
    return get_channel_info(suffix, channel)


def _channel_label(ch: int) -> str:
    """带针脚信息的通道标签。"""
    info = _get_channel_info(ch)
    return f"{ch} | {info.short_info()}" if info else str(ch)


# =============================================================================
# Session State 初始化
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量 (幂等, 每次 rerun 调用)。"""
    # ---- 单控制器 ----
    st.session_state.setdefault(f"{P}_connection_mode", "single")  # "single" | "joint" | "group"
    st.session_state.setdefault(f"{P}_ip", "192.168.0.101")
    st.session_state.setdefault(f"{P}_port", CFG.DEFAULT_PORT)
    st.session_state.setdefault(f"{P}_controller", None)
    st.session_state.setdefault(f"{P}_connected", False)
    st.session_state.setdefault(f"{P}_connection_error", "")
    st.session_state.setdefault(f"{P}_simulate", False)
    st.session_state.setdefault(f"{P}_vmin", HW_VOLTAGE_MIN)
    st.session_state.setdefault(f"{P}_vmax", HW_VOLTAGE_MAX)
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

    # ---- 交替电压 (0V ↔ Input) ----
    st.session_state.setdefault(f"{P}_alt_running", False)
    st.session_state.setdefault(f"{P}_alt_voltage", 20.0)
    st.session_state.setdefault(f"{P}_alt_freq", 1.0)

    # ---- 方波电压 (A/B) ----
    st.session_state.setdefault(f"{P}_square_running", False)
    st.session_state.setdefault(f"{P}_square_voltage_a", 20.0)
    st.session_state.setdefault(f"{P}_square_voltage_b", 0.0)
    st.session_state.setdefault(f"{P}_square_freq", 1.0)

    # ---- 下发模式选择 ----
    st.session_state.setdefault(f"{P}_send_mode", "clear")

    # ---- 反馈 ----
    st.session_state.setdefault(f"{P}_current_voltages", np.zeros(SINGLE_CHANNELS, dtype=np.float64))
    st.session_state.setdefault(f"{P}_feedback", "")
    st.session_state.setdefault(f"{P}_feedback_type", "info")
    # ---- 调试日志 ----
    st.session_state.setdefault(f"{P}_debug", False)
    st.session_state.setdefault(f"{P}_debug_pkt_enable_sb", False)
    st.session_state.setdefault(f"{P}_debug_log", collections.deque(maxlen=DEBUG_LOG_MAX))
    st.session_state.setdefault(f"{P}_debug_op_log", collections.deque(maxlen=DEBUG_LOG_MAX))
    st.session_state.setdefault(f"{P}_debug_tcp_client", DebugTcpClient())
    st.session_state.setdefault(f"{P}_debug_tcp_enabled", False)
    st.session_state.setdefault(f"{P}_debug_tcp_host", DEBUG_HOST)
    st.session_state.setdefault(f"{P}_debug_tcp_port", DEBUG_PORT)
    st.session_state.setdefault(f"{P}_local_debug_server", False)
    st.session_state.setdefault(f"{P}_local_debug_logs", collections.deque(maxlen=100))
    # ---- 单控制器通道选择 / 波形 ----
    st.session_state.setdefault(f"{P}_all_mode", True)
    st.session_state.setdefault(f"{P}_channels", [])
    st.session_state.setdefault(f"{P}_channel", 0)
    st.session_state.setdefault(f"{P}_voltage", 0.0)
    st.session_state.setdefault(f"{P}_hold", False)
    st.session_state.setdefault(f"{P}_sine_running", False)
    st.session_state.setdefault(f"{P}_alt_running", False)
    st.session_state.setdefault(f"{P}_sine_apply_all", True)
    st.session_state.setdefault(f"{P}_sine_channel_input", 0)
    st.session_state.setdefault(f"{P}_sine_offset", 50.0)
    st.session_state.setdefault(f"{P}_sine_amp", 20.0)
    st.session_state.setdefault(f"{P}_sine_freq", 1.0)
    st.session_state.setdefault(f"{P}_alt_voltage", 20.0)
    st.session_state.setdefault(f"{P}_alt_freq", 1.0)
    st.session_state.setdefault(f"{P}_loop_stop_event", None)
    st.session_state.setdefault(f"{P}_loop_feedback_q", None)
    # ---- 联合控制 (JC) ----
    _init_jc_state()
    # ---- 分组控制 (GC) ----
    _init_gc_state()


def _init_jc_state() -> None:
    """初始化联合控制相关状态。"""
    p = f"{P}_jc"
    st.session_state.setdefault(f"{p}_simulate", False)
    st.session_state.setdefault(f"{p}_dm", None)
    st.session_state.setdefault(f"{p}_connected", False)
    st.session_state.setdefault(f"{p}_connection_error", "")
    st.session_state.setdefault(f"{p}_relay_on", False)
    st.session_state.setdefault(f"{p}_matrix", None)
    st.session_state.setdefault(f"{p}_row_count", 0)
    st.session_state.setdefault(f"{p}_col_count", 0)
    st.session_state.setdefault(f"{p}_current_flat", np.zeros(0, dtype=np.float64))
    st.session_state.setdefault(f"{p}_current_value", 0.0)
    st.session_state.setdefault(f"{p}_controller_count", 0)
    st.session_state.setdefault(f"{p}_pos_to_hw", {})
    st.session_state.setdefault(f"{p}_ip_to_controller_idx", {})
    st.session_state.setdefault(f"{p}_sorted_ips", [])
    st.session_state.setdefault(f"{p}_dm_num", 0)
    st.session_state.setdefault(f"{p}_matrix_init", False)
    st.session_state.setdefault(f"{p}_feedback", "")
    st.session_state.setdefault(f"{p}_feedback_type", "info")


def _init_gc_state() -> None:
    """初始化分组控制相关状态 (组信息由 CSV 构建一次)。"""
    p = f"{P}_gc"
    st.session_state.setdefault(f"{p}_simulate", False)
    st.session_state.setdefault(f"{p}_controllers", {})
    st.session_state.setdefault(f"{p}_connected", False)
    st.session_state.setdefault(f"{p}_connection_error", "")
    st.session_state.setdefault(f"{p}_relay_on", False)
    st.session_state.setdefault(f"{p}_groups", None)
    st.session_state.setdefault(f"{p}_selected_group", None)
    st.session_state.setdefault(f"{p}_voltage", 0.0)
    st.session_state.setdefault(f"{p}_selected_channels", [])
    st.session_state.setdefault(f"{p}_feedback", "")
    st.session_state.setdefault(f"{p}_feedback_type", "info")
    st.session_state.setdefault(f"{p}_current_map", {})
    if st.session_state.get(f"{p}_groups") is None:
        groups = build_groups()
        # 与旧版一致: 每组内按 payload_position 排序
        for g in groups.values():
            for ip_s, chs in g.channels_by_ip.items():
                chs.sort(key=lambda c: c.payload_position)
        st.session_state[f"{p}_groups"] = groups
    if st.session_state.get(f"{p}_selected_group") is None:
        names = sorted(st.session_state[f"{p}_groups"].keys())
        st.session_state[f"{p}_selected_group"] = names[0] if names else None


# =============================================================================
# 发送循环管理 (后台线程不触碰 session_state; 反馈经 queue 回传)
# =============================================================================

def _loop_stop_all() -> None:
    """停止所有运行中的发送循环。"""
    st.session_state[f"{P}_hold"] = False
    st.session_state[f"{P}_sine_running"] = False
    st.session_state[f"{P}_alt_running"] = False
    ev = st.session_state.get(f"{P}_loop_stop_event")
    if ev is not None:
        stop_loop(ev)
        st.session_state[f"{P}_loop_stop_event"] = None


def _loop_start(loop_fn: Any, params: dict[str, Any]) -> None:
    """启动一个发送循环 (先停旧循环, 换新 Event)。"""
    ctrl = st.session_state.get(f"{P}_controller")
    if ctrl is None:
        return
    _loop_stop_all()
    ev = threading.Event()
    q: queue.Queue[tuple[str, str]] = queue.Queue()
    st.session_state[f"{P}_loop_stop_event"] = ev
    st.session_state[f"{P}_loop_feedback_q"] = q
    params = dict(params)
    params.setdefault("vmin", st.session_state[f"{P}_vmin"])
    params.setdefault("vmax", st.session_state[f"{P}_vmax"])
    params.setdefault("selection", ChannelSelection(all_mode=True))
    start_loop(
        loop_fn,
        ctrl,
        st.session_state[f"{P}_current_voltages"],
        params,
        ev,
        q,
    )


def _drain_loop_feedback() -> None:
    """主线程在每次 rerun 时消费循环线程的反馈。"""
    q = st.session_state.get(f"{P}_loop_feedback_q")
    if q is None:
        return
    items: list[tuple[str, str]] = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    for msg_type, msg in items:
        _set_feedback(msg, msg_type)


# =============================================================================
# 连通性测试
# =============================================================================

def test_connectivity() -> None:
    """测试与 192.168.0.x 控制器网络的连通性 (Ping + TCP)。"""
    ip = st.session_state.get(f"{P}_ip", "192.168.0.101").strip()
    port = int(st.session_state.get(f"{P}_port", CFG.DEFAULT_PORT))
    st.write(f"#### 测试目标: {ip}:{port}")
    ping_ok = ping_reachable(ip, timeout=1.0)
    st.write(f"**Ping {ip}** → {'✅ 可达' if ping_ok else '❌ 不可达'}")
    tcp_ok = tcp_reachable(ip, port, timeout=1.0)
    st.write(f"**TCP {ip}:{port}** → {'✅ 可达' if tcp_ok else '❌ 不可达'}")
    if ping_ok and tcp_ok:
        st.success("网络连通性正常, 可以连接控制器。")
    else:
        st.warning("请检查网线连接、IP 段与控制器电源。")


# =============================================================================
# 连接 / 断开 / 继电器 (单控制器)
# =============================================================================

def connect() -> None:
    """连接单控制器 (真实或仿真)。"""
    ip = st.session_state[f"{P}_ip"].strip()
    port = int(st.session_state[f"{P}_port"])
    simulate = st.session_state.get(f"{P}_simulate", False)
    # 清理旧连接
    old = st.session_state.get(f"{P}_controller")
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
        st.session_state[f"{P}_controller"] = None
        st.session_state[f"{P}_connected"] = False
    st.session_state[f"{P}_connection_error"] = ""
    try:
        ctrl = create_controller(controller_id=1, ip=ip, port=port, simulate=simulate)
    except ConnectionError as exc:
        st.session_state[f"{P}_connection_error"] = f"无法建立 TCP 连接到 {ip}:{port} ({exc})"
        st.session_state[f"{P}_connected"] = False
        logger.error(st.session_state[f"{P}_connection_error"])
        return
    st.session_state[f"{P}_controller"] = ctrl
    st.session_state[f"{P}_connected"] = True
    st.session_state[f"{P}_feedback"] = f"已连接 {ip}:{port} {'(仿真)' if simulate else ''}"
    st.session_state[f"{P}_feedback_type"] = "success"
    logger.info(f"R50 控制器已连接: {ip}:{port} simulate={simulate}")


def disconnect() -> None:
    """断开单控制器 (先停循环, 下电继电器, 再关闭连接)。"""
    _loop_stop_all()
    ctrl = st.session_state.get(f"{P}_controller")
    if ctrl is not None:
        try:
            power_off_and_close(ctrl)
        except Exception as exc:
            logger.error(f"关闭控制器异常: {exc}")
    st.session_state[f"{P}_controller"] = None
    st.session_state[f"{P}_connected"] = False
    st.session_state[f"{P}_relay_on"] = False
    st.session_state[f"{P}_feedback"] = "已断开连接 (继电器已下电)"
    st.session_state[f"{P}_feedback_type"] = "info"


def set_relay_power(on: bool) -> None:
    """控制继电器 (真实: 发 0x1A/0x1B 命令; 仿真: 记录状态)。"""
    ctrl = st.session_state.get(f"{P}_controller")
    if ctrl is None:
        _set_feedback("请先连接控制器", "error")
        return
    try:
        ok = set_relay(ctrl, on)
    except Exception as exc:
        _set_feedback(f"继电器操作失败: {exc}", "error")
        logger.error(f"set_relay_power({on}) 异常: {exc}")
        return
    if ok:
        st.session_state[f"{P}_relay_on"] = on
        cmd_name = "CMD_RELAY_ON" if on else "CMD_RELAY_OFF"
        cmd_byte = CMD_RELAY_ON if on else CMD_RELAY_OFF
        _debug_log_packet(cmd_name, HEADER + bytes([cmd_byte]) + FOOTER)
        _set_feedback(f"继电器已{'开启' if on else '关闭'}", "success")
    else:
        _set_feedback("继电器操作失败 (未收到确认)", "error")


# =============================================================================
# 下发保护 / 单次下发
# =============================================================================

def _require_relay_on() -> bool:
    """下发前强制检查继电器状态。"""
    if not st.session_state.get(f"{P}_relay_on", False):
        _set_feedback("⚠️ 请先开启继电器 (否则控制器不会输出)", "warning")
        return False
    return True


def _send_success_feedback(voltage: float, result: SendResult) -> None:
    """根据 SendResult 生成下发反馈。"""
    if result.fail:
        targets = ", ".join(str(t) for t in result.failed_targets[:10])
        _set_feedback(f"⚠️ 下发失败 {result.fail} 个目标: {targets}", "error")
        return
    all_mode = st.session_state.get(f"{P}_all_mode", True)
    if all_mode:
        _set_feedback(f"✅ 已发送 全选 50 通道 {voltage:.1f}V", "success")
    else:
        n = len(st.session_state.get(f"{P}_channels", []))
        _set_feedback(f"✅ 已发送 {n} 通道 {voltage:.1f}V", "success")


def _log_all_packet(voltage: float) -> None:
    """调试日志: 全选 0x08 数据包。"""
    payload = voltages_to_payload(np.asarray([voltage], dtype=np.float64))
    packet = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, int(payload[0]), int(payload[1])]) + FOOTER
    _debug_log_packet("CMD_SET_ALL_CHANNEL_VOLTAGE", packet)


def _log_bulk_packet(voltage: float, selection: ChannelSelection) -> None:
    """调试日志: 批量 0x09 数据包。"""
    current = st.session_state[f"{P}_current_voltages"]
    arr = build_bulk_array(
        current,
        selection.normalized(len(current)),
        voltage,
        st.session_state[f"{P}_vmin"],
        st.session_state[f"{P}_vmax"],
    )
    packet = HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(arr) + FOOTER
    _debug_log_packet("CMD_SET_ALL_VOLTAGE_BY_ARR", packet)


def _send_channels(voltage: float) -> SendResult:
    """单次下发当前选择 (全选或勾选通道), 更新当前电压表并反馈。"""
    ctrl = st.session_state.get(f"{P}_controller")
    if ctrl is None:
        _set_feedback("请先连接控制器", "error")
        return SendResult(fail=1, failed_targets=["未连接"])
    if not _require_relay_on():
        return SendResult(fail=1, failed_targets=["继电器未开启"])
    all_mode = st.session_state.get(f"{P}_all_mode", True)
    selection = ChannelSelection(
        all_mode=all_mode,
        channels=list(st.session_state.get(f"{P}_channels", [])),
    )
    if selection.is_empty:
        _set_feedback("未选择任何通道", "warning")
        return SendResult(fail=1, failed_targets=["无选中通道"])
    if st.session_state.get(f"{P}_debug", False):
        if all_mode:
            _log_all_packet(voltage)
        else:
            _log_bulk_packet(voltage, selection)
    current, result = apply_single_controller(
        ctrl,
        st.session_state[f"{P}_current_voltages"],
        selection,
        voltage,
        st.session_state[f"{P}_vmin"],
        st.session_state[f"{P}_vmax"],
    )
    st.session_state[f"{P}_current_voltages"] = current
    _send_success_feedback(voltage, result)
    return result


# =============================================================================
# 联合控制 (JC): 矩阵读取 / 连接 / 继电器 / 下发
# =============================================================================

def _jc_read_matrix_from_dm(
    dm: Any,
    pos_to_hw: dict[int, tuple[int, int]],
    ip_to_ctrl_idx: dict[int, int],
) -> np.ndarray:
    """从 MicroDM 读取当前电压并构建 36×36 矩阵。"""
    flat = dm.get_actuator_positions()
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for physical_pos, (ip_suffix, payload_pos) in pos_to_hw.items():
        row = (physical_pos - 1) // GRID_SIZE
        col = (physical_pos - 1) % GRID_SIZE
        ctrl_idx = ip_to_ctrl_idx.get(ip_suffix)
        if ctrl_idx is not None:
            flat_idx = ctrl_idx * SINGLE_CHANNELS + (payload_pos - 1)
            if flat_idx < len(flat):
                matrix[row, col] = flat[flat_idx]
    return matrix


def _jc_connect() -> None:
    """连接 MicroDM (所有控制器, 支持仿真模式)。"""
    jc = f"{P}_jc"
    try:
        simulate = st.session_state.get(f"{P}_jc_simulate", False)
        csv_df = load_csv()
        if csv_df.empty:
            st.session_state[f"{jc}_connection_error"] = "1300-5-enriched.csv 加载失败"
            st.session_state[f"{jc}_connected"] = False
            return
        ip_suffixes = sorted(int(ip) for ip in csv_df["IP组"].unique())
        if simulate:
            dm: Any = SimulatedMicroDM(ips=ip_suffixes)
            dm.open()
            feedback_prefix = "🟡 [仿真] "
        else:
            dm = MicroDM(use_wiring_map=True)
            dm.open()
            feedback_prefix = ""
        st.session_state[f"{jc}_dm"] = dm
        st.session_state[f"{jc}_connected"] = True
        st.session_state[f"{jc}_relay_on"] = False
        st.session_state[f"{jc}_connection_error"] = ""
        st.session_state[f"{jc}_controller_count"] = len(dm._controllers)
        pos_to_hw = jc_build_wiring_index(csv_df)
        ip_to_ctrl_idx = jc_build_ip_index(csv_df)
        st.session_state[f"{jc}_pos_to_hw"] = pos_to_hw
        st.session_state[f"{jc}_ip_to_controller_idx"] = ip_to_ctrl_idx
        st.session_state[f"{jc}_sorted_ips"] = [f"192.168.0.{s}" for s in ip_suffixes]
        st.session_state[f"{jc}_dm_num"] = getattr(
            dm, "DM_Num", len(ip_suffixes) * SINGLE_CHANNELS
        )
        st.session_state[f"{jc}_matrix_init"] = True
        if not simulate:
            matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl_idx)
        else:
            matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        st.session_state[f"{jc}_matrix"] = matrix
        st.session_state[f"{jc}_current_flat"] = matrix.flatten().copy()
        n_ctrl = st.session_state[f"{jc}_controller_count"]
        st.session_state[f"{jc}_feedback"] = f"{feedback_prefix}已连接 MicroDM: {n_ctrl} 个控制器"
        st.session_state[f"{jc}_feedback_type"] = "success"
        _debug_add_op("connect", f"joint ({n_ctrl} controllers)", "all")
        logger.info(f"MicroDM connected: {n_ctrl} controllers")
    except Exception as e:
        st.session_state[f"{jc}_connection_error"] = f"连接失败: {e}"
        st.session_state[f"{jc}_connected"] = False
        st.session_state[f"{jc}_dm"] = None
        st.session_state[f"{jc}_feedback"] = f"连接失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM connect failed: {e}")


def _jc_disconnect() -> None:
    """断开 MicroDM 连接 (先下电)。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is not None:
        try:
            if st.session_state.get(f"{jc}_relay_on", False):
                dm.set_relay_state(False)
            dm.close()
        except Exception as e:
            logger.warning(f"MicroDM disconnect warning: {e}")
    st.session_state[f"{jc}_dm"] = None
    st.session_state[f"{jc}_connected"] = False
    st.session_state[f"{jc}_relay_on"] = False
    st.session_state[f"{jc}_connection_error"] = ""
    st.session_state[f"{jc}_feedback"] = "已断开连接 (已先下电)"
    st.session_state[f"{jc}_feedback_type"] = "info"
    _debug_add_op("disconnect", "joint", "all")
    logger.info("MicroDM disconnected")


def _jc_set_relay(on: bool) -> None:
    """所有控制器继电器上下电。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        dm.set_relay_state(on)
        st.session_state[f"{jc}_relay_on"] = on
        if on:
            st.session_state[f"{jc}_feedback"] = "✅ 所有控制器继电器已上电 (输出接通)"
            st.session_state[f"{jc}_feedback_type"] = "success"
            _debug_add_op("relay_on", "joint", "all")
        else:
            st.session_state[f"{jc}_feedback"] = "⏻ 所有控制器继电器已下电 (输出断开)"
            st.session_state[f"{jc}_feedback_type"] = "info"
            _debug_add_op("relay_off", "joint", "all")
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"继电器操作失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM relay failed: {e}")


def _jc_apply_matrix() -> None:
    """将当前 36×36 矩阵电压下发到所有控制器。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    if not st.session_state.get(f"{jc}_relay_on", False):
        st.session_state[f"{jc}_feedback"] = "⚠️ 请先继电器上电后再下发电压"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        matrix = st.session_state[f"{jc}_matrix"]
        pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
        ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
        dm_num = st.session_state[f"{jc}_dm_num"]
        flat = jc_matrix_to_flat(matrix, pos_to_hw, ip_to_ctrl, dm_num)
        dm.send_voltages(flat)
        st.session_state[f"{jc}_current_flat"] = flat.copy()
        non_zero = np.count_nonzero(matrix)
        st.session_state[f"{jc}_feedback"] = (
            f"✅ 已下发 36×36 矩阵电压 (非零通道: {non_zero}/{DM_NUM_ACTUATORS})"
        )
        st.session_state[f"{jc}_feedback_type"] = "success"
        _debug_add_op("set_voltage", f"matrix {non_zero} non-zero channels", "all")
        logger.info(f"MicroDM voltage applied: {non_zero} non-zero channels")
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"电压下发失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM apply failed: {e}")


def _jc_reset_matrix() -> None:
    """将矩阵清零并下发。"""
    jc = f"{P}_jc"
    st.session_state[f"{jc}_matrix"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    if st.session_state.get(f"{jc}_connected", False) and st.session_state.get(f"{jc}_relay_on", False):
        _jc_apply_matrix()
    else:
        st.session_state[f"{jc}_feedback"] = "矩阵已清零 (未下发到硬件)"
        st.session_state[f"{jc}_feedback_type"] = "info"


def _jc_refresh_from_hardware() -> None:
    """从硬件读取当前电压并刷新矩阵显示。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        return
    if st.session_state.get(f"{P}_jc_simulate", False):
        st.session_state[f"{jc}_feedback"] = "仿真模式: 矩阵保持当前值"
        st.session_state[f"{jc}_feedback_type"] = "info"
        return
    try:
        pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
        ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
        matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl)
        st.session_state[f"{jc}_matrix"] = matrix
        st.session_state[f"{jc}_current_flat"] = matrix.flatten().copy()
        st.session_state[f"{jc}_feedback"] = "已从硬件刷新电压矩阵"
        st.session_state[f"{jc}_feedback_type"] = "info"
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"刷新失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM refresh failed: {e}")


# =============================================================================
# 联合控制 (JC): 批量上下电 (Ping 测试)
# =============================================================================

def _jc_batch_power_on() -> None:
    """批量上电: 先 ping 测试所有控制器, 再继电器上电。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    sorted_ips = st.session_state.get(f"{jc}_sorted_ips", [])
    if not sorted_ips:
        st.session_state[f"{jc}_feedback"] = "无控制器 IP 信息"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    simulate = st.session_state.get(f"{P}_jc_simulate", False)
    reachable: list[str] = []
    unreachable: list[str] = []
    for ip in sorted_ips:
        if simulate or ping_reachable(ip, timeout=1.0):
            reachable.append(ip)
        else:
            unreachable.append(ip)
    if not reachable:
        st.session_state[f"{jc}_feedback"] = f"❌ 所有控制器均不可达: {', '.join(unreachable)}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        dm.set_relay_state(True)
        st.session_state[f"{jc}_relay_on"] = True
        if not unreachable:
            st.session_state[f"{jc}_feedback"] = (
                f"✅ 批量上电成功 ({len(reachable)} 个控制器全部可达)"
            )
            st.session_state[f"{jc}_feedback_type"] = "success"
        else:
            st.session_state[f"{jc}_feedback"] = (
                f"⚠️ 部分上电: {len(reachable)} 可达并已上电, "
                f"{len(unreachable)} 不可达 ({', '.join(unreachable)})"
            )
            st.session_state[f"{jc}_feedback_type"] = "warning"
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"批量上电失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"Batch power on failed: {e}")


def _jc_batch_power_off() -> None:
    """批量下电: 所有控制器继电器下电。"""
    _jc_set_relay(False)


# =============================================================================
# 联合控制 (JC): 矩阵编辑
# =============================================================================

def _jc_set_cell(row: int, col: int, voltage: float) -> None:
    """设置矩阵中单个单元电压。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    matrix[row, col] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_row(row: int, voltage: float) -> None:
    """填充整行。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    matrix[row, :] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_col(col: int, voltage: float) -> None:
    """填充整列。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    matrix[:, col] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_rect(x1: int, y1: int, x2: int, y2: int, voltage: float) -> None:
    """填充矩形区域。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    r1, r2 = min(y1, y2), max(y1, y2)
    c1, c2 = min(x1, x2), max(x1, x2)
    matrix[r1:r2 + 1, c1:c2 + 1] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_all(voltage: float) -> None:
    """填充整个矩阵。"""
    jc = f"{P}_jc"
    matrix = np.full((GRID_SIZE, GRID_SIZE), voltage, dtype=np.float64)
    st.session_state[f"{jc}_matrix"] = matrix


# =============================================================================
# 联合控制 (JC): 可视化 (Streamlit 原生控件)
# =============================================================================

def _jc_colormap_image(matrix: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """将电压矩阵转换为彩色图像 (numpy, 无 matplotlib 依赖)。

    coolwarm 风格: 蓝色 (vmin) → 白色 (0) → 红色 (vmax)。
    """
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((matrix - vmin) / (vmax - vmin), 0, 1)
    h, w = matrix.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    mask_low = normalized <= 0.5
    mask_high = normalized > 0.5
    img[mask_low, 0] = normalized[mask_low] * 2.0
    img[mask_low, 1] = normalized[mask_low] * 2.0
    img[mask_low, 2] = 1.0
    img[mask_high, 0] = 1.0
    img[mask_high, 1] = 2.0 - normalized[mask_high] * 2.0
    img[mask_high, 2] = 2.0 - normalized[mask_high] * 2.0
    return img


def _jc_render_matrix_image(matrix: np.ndarray) -> None:
    """Streamlit ``st.image`` 显示彩色热力图。"""
    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)
    img = _jc_colormap_image(matrix, vmin, vmax)
    st.image(img, caption="36×36 电压分布 (蓝色低 · 红色高)", width='stretch')


def _jc_render_matrix_dataframe(matrix: np.ndarray) -> None:
    """分块显示带颜色的 36×36 数值矩阵 (6 块 × 6 列)。"""
    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)
    blocks = 6
    cols_per_block = 6
    for block_idx in range(blocks):
        start_col = block_idx * cols_per_block
        end_col = min(start_col + cols_per_block, GRID_SIZE)
        col_labels = [str(c + 1) for c in range(start_col, end_col)]
        df_block = pd.DataFrame(
            matrix[:, start_col:end_col],
            index=[f"行{r + 1}" for r in range(GRID_SIZE)],
            columns=col_labels,
        )
        with st.expander(f"📍 第 {start_col + 1}–{end_col} 列", expanded=(block_idx == 0)):
            col_config = {}
            for i, c in enumerate(col_labels):
                col_config[c] = st.column_config.NumberColumn(
                    label=c,
                    min_value=vmin,
                    max_value=vmax,
                    format="%.1f",
                )
            st.dataframe(
                df_block,
                column_config=col_config,
                height=min(36 * 35 + 40, 800),
                width='stretch',
            )


def _jc_render_profile(matrix: np.ndarray) -> None:
    """行/列均值剖面图表。"""
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    with st.container(border=True):
        st.markdown("##### 电压剖面")
        tab_r, tab_c = st.tabs(["📊 行均值", "📊 列均值"])
        with tab_r:
            df_row = pd.DataFrame(
                {"行号": list(range(1, GRID_SIZE + 1)), "均值 (V)": row_means}
            ).set_index("行号")
            st.bar_chart(df_row, height=200, width='stretch')
        with tab_c:
            df_col = pd.DataFrame(
                {"列号": list(range(1, GRID_SIZE + 1)), "均值 (V)": col_means}
            ).set_index("列号")
            st.bar_chart(df_col, height=200, width='stretch')


def _jc_render_stats(matrix: np.ndarray) -> None:
    """显示矩阵统计指标。"""
    vals = matrix.flatten()
    non_zero_count = np.count_nonzero(matrix)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("最小值", f"{np.min(vals):.1f} V")
    with col2:
        st.metric("最大值", f"{np.max(vals):.1f} V")
    with col3:
        st.metric("均值", f"{np.mean(vals):.1f} V")
    with col4:
        st.metric("标准差", f"{np.std(vals):.1f} V")
    with col5:
        st.metric("非零通道", f"{non_zero_count}/{DM_NUM_ACTUATORS}")


# =============================================================================
# 分组控制 (GC): 连接 / 断开 / 继电器 / 下发 / 批量上下电
# =============================================================================

def _gc_show_feedback() -> None:
    """显示分组控制反馈。"""
    _show_feedback(prefix="gc")


def _gc_set_feedback(message: str, msg_type: str = "info") -> None:
    """写入分组控制反馈。"""
    _set_feedback(message, msg_type, prefix="gc")


# =============================================================================
# Group Control: Connect / Disconnect
# =============================================================================

def _gc_connect() -> None:
    """连接所选组的所有控制器 (支持仿真模式)。"""
    gc = f"{P}_gc"
    selected = st.session_state.get(f"{gc}_selected_group")
    groups = st.session_state.get(f"{gc}_groups", {})

    if not selected or selected not in groups:
        _gc_set_feedback("未选择有效组别", "error")
        return

    simulate = st.session_state.get(f"{P}_gc_simulate", False)
    group_def = groups[selected]
    controllers: dict[int, Any] = {}
    connected_count = 0
    total_count = len(group_def.channels_by_ip)
    errors: list[str] = []

    for ip_suffix in sorted(group_def.channels_by_ip.keys()):
        ip = f"192.168.0.{ip_suffix}"
        port = CFG.DEFAULT_PORT
        try:
            ctrl = create_controller(
                controller_id=ip_suffix, ip=ip, port=port, simulate=simulate
            )
        except Exception as e:
            errors.append(f"{ip}:{port} {e}")
            logger.exception(f"Group control connect failed for {ip}:{port}: {e}")
            continue
        controllers[ip_suffix] = ctrl
        connected_count += 1
        logger.info(f"Group control connected: {ip}:{port}")

    st.session_state[f"{gc}_controllers"] = controllers
    st.session_state[f"{gc}_connected"] = connected_count > 0
    st.session_state[f"{gc}_relay_on"] = False
    st.session_state[f"{gc}_connection_error"] = ""

    prefix = "🟡 [仿真] " if simulate else ""
    if connected_count == total_count:
        _gc_set_feedback(
            f"{prefix}已连接 {selected} 全部 {connected_count} 个控制器",
            "success",
        )
    elif connected_count > 0:
        error_detail = "; ".join(errors) if errors else ""
        _gc_set_feedback(
            f"⚠️ {prefix}已连接 {connected_count}/{total_count} 个控制器"
            + (f" ({error_detail})" if error_detail else ""),
            "warning",
        )
    else:
        error_detail = "; ".join(errors) if errors else "未知错误"
        _gc_set_feedback(f"❌ 连接失败: {error_detail}", "error")

    if selected:
        _debug_add_op("connect", f"group={selected} ({connected_count}/{total_count})", "")


def _gc_disconnect() -> None:
    """断开所选组的所有控制器连接 (先下电)。"""
    gc = f"{P}_gc"
    controllers: dict[int, Any] = st.session_state.get(f"{gc}_controllers", {})

    for ip_suffix, ctrl in controllers.items():
        try:
            power_off_and_close(ctrl)
        except Exception as e:
            logger.warning(f"Group control disconnect warning for ip={ip_suffix}: {e}")

    st.session_state[f"{gc}_controllers"] = {}
    st.session_state[f"{gc}_connected"] = False
    st.session_state[f"{gc}_relay_on"] = False
    st.session_state[f"{gc}_connection_error"] = ""
    _gc_set_feedback("已断开所有控制器 (已先下电)", "info")
    _debug_add_op("disconnect", "group", "")
    logger.info("Group control disconnected all controllers")


def _gc_set_relay(on: bool) -> None:
    """所有控制器继电器上下电 (逐台下发, 统计成败)。"""
    gc = f"{P}_gc"
    controllers: dict[int, Any] = st.session_state.get(f"{gc}_controllers", {})

    if not controllers:
        _gc_set_feedback("设备未连接", "error")
        return

    success_count = 0
    error_count = 0
    for ip_suffix, ctrl in controllers.items():
        try:
            if set_relay(ctrl, on):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            logger.exception(f"Group control relay failed for ip={ip_suffix}: {e}")

    if error_count == 0:
        st.session_state[f"{gc}_relay_on"] = on
        label = "上电 (输出接通)" if on else "下电 (输出断开)"
        _gc_set_feedback(
            f"✅ 所有控制器继电器已{label} ({success_count} 个控制器)",
            "success" if on else "info",
        )
        _debug_add_op("relay_on" if on else "relay_off", f"group, {success_count} ok", "")
        logger.info(f"Group control relay {'ON' if on else 'OFF'}: {success_count} controllers")
    else:
        _gc_set_feedback(
            f"⚠️ 继电器操作: {success_count} 成功, {error_count} 失败",
            "warning",
        )


def _gc_apply_voltage(all_channels: bool = False) -> None:
    """向所选组下发电压 (每控制器一个 0x09 批量包, 一次点击全部送达)。"""
    gc = f"{P}_gc"
    controllers: dict[int, Any] = st.session_state.get(f"{gc}_controllers", {})
    if not controllers:
        _gc_set_feedback("设备未连接", "error")
        return
    if not st.session_state.get(f"{gc}_relay_on", False):
        _gc_set_feedback("⚠️ 请先继电器上电后再下发电压", "error")
        return

    voltage = float(st.session_state.get(f"{gc}_voltage", 0.0))
    clipped = clip_voltage(voltage)

    selected_group = st.session_state.get(f"{gc}_selected_group")
    groups = st.session_state.get(f"{gc}_groups", {})
    if not selected_group or selected_group not in groups:
        _gc_set_feedback("未选择有效组别", "error")
        return

    group_def = groups[selected_group]
    selected_channels = st.session_state.get(f"{gc}_selected_channels", [])
    if all_channels or not selected_channels:
        selected_payloads = group_def.all_payload_positions
    else:
        selected_payloads = [int(c) for c in selected_channels]

    current_map = st.session_state.setdefault(f"{gc}_current_map", {})
    result = apply_group_controllers(
        controllers,
        group_def,
        selected_payloads,
        clipped,
        st.session_state[f"{P}_vmin"],
        st.session_state[f"{P}_vmax"],
        current_map,
    )
    st.session_state[f"{gc}_current_map"] = current_map

    if result.fail:
        targets = ", ".join(str(t) for t in result.failed_targets[:10])
        _gc_set_feedback(f"⚠️ 下发失败 {result.fail} 个控制器: {targets}", "warning")
        _debug_add_op(
            "set_voltage", f"group {selected_group} fail={result.fail}", ""
        )
        return

    sel_set = set(selected_payloads)
    n_ch = sum(
        1
        for chs in group_def.channels_by_ip.values()
        for ci in chs
        if ci.payload_position in sel_set
    )
    label = "全部 " if all_channels else ""
    _gc_set_feedback(
        f"✅ 已向 {selected_group} {label}{n_ch} 个通道下发 {clipped:.1f} V",
        "success",
    )
    _debug_add_op(
        "set_voltage", f"group {selected_group} {label}{n_ch}ch {clipped:.1f}V", ""
    )


def _gc_batch_power_on() -> None:
    """批量上电: 先 ping 测试组内所有控制器, 再继电器上电。"""
    gc = f"{P}_gc"
    controllers: dict[int, Any] = st.session_state.get(f"{gc}_controllers", {})
    if not controllers:
        _gc_set_feedback("设备未连接", "error")
        return

    simulate = st.session_state.get(f"{P}_gc_simulate", False)
    reachable: list[int] = []
    unreachable: list[int] = []
    for ip_suffix in sorted(controllers.keys()):
        ip = f"192.168.0.{ip_suffix}"
        if simulate or ping_reachable(ip, timeout=1.0):
            reachable.append(ip_suffix)
        else:
            unreachable.append(ip_suffix)

    if not reachable:
        _gc_set_feedback("❌ 所有控制器均不可达", "error")
        return

    success_count = 0
    error_count = 0
    for ip_suffix in reachable:
        ctrl = controllers.get(ip_suffix)
        if ctrl is None:
            continue
        try:
            if set_relay(ctrl, True):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            logger.exception(f"Batch relay on failed for ip={ip_suffix}: {e}")

    if error_count == 0:
        st.session_state[f"{gc}_relay_on"] = True
        if not unreachable:
            _gc_set_feedback(
                f"✅ 批量上电成功 ({success_count} 个控制器全部可达)", "success"
            )
        else:
            _gc_set_feedback(
                f"⚠️ 部分上电: {success_count} 可达并已上电, "
                f"{len(unreachable)} 不可达",
                "warning",
            )
    else:
        _gc_set_feedback(
            f"⚠️ 批量上电: {success_count} 成功, {error_count} 失败", "warning"
        )


def _gc_batch_power_off() -> None:
    """批量下电: 所有控制器继电器下电。"""
    _gc_set_relay(False)


# =============================================================================
# 单单元控制 Tab: 跨控制器选择物理单元并下发
# =============================================================================

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
    try:
        conn_suffix = int(conn_ip.rsplit(".", 1)[-1])
        default_ips = [s for s in ip_suffixes if s == conn_suffix]
    except (ValueError, IndexError):
        default_ips = []
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
    st.dataframe(df_sel[INFO_DISPLAY_COLS[:4]], width='stretch', hide_index=True)

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
            _show_units_result("joint", result, voltage)
        elif mode == "single" and single_connected:
            ctrl = st.session_state.get(f"{P}_controller")
            result = apply_units_via_controller(
                ctrl,
                st.session_state[f"{P}_current_voltages"],
                selected_units,
                voltage, vmin, vmax,
            )
            _show_units_result("single", result, voltage)
        elif mode == "group" and gc_connected:
            controllers = st.session_state.get(f"{P}_gc_controllers", {})
            current_map = st.session_state.setdefault(f"{P}_gc_current_map", {})
            result = _apply_units_group_mode(controllers, selected_units, voltage, current_map)
            _show_units_result("group", result, voltage)
        st.rerun()


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

        st.checkbox(
            "指令日志",
            value=st.session_state[f"{P}_debug"],
            key=f"{P}_debug_pkt_enable_sb",
            help="显示下发的指令包十六进制内容",
        )
        st.session_state[f"{P}_debug"] = st.session_state[f"{P}_debug_pkt_enable_sb"]

        with st.expander("操作日志", expanded=False):
            op_log: collections.deque = st.session_state.get(f"{P}_debug_op_log", collections.deque())
            if op_log:
                st.code("\n".join(op_log), language="text")
            else:
                st.caption("暂无操作日志")
            if st.button("清空日志", key=f"{P}_debug_op_clear_sb", use_container_width=True):
                st.session_state[f"{P}_debug_op_log"].clear()
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
                    "下限 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                    value=st.session_state[f"{P}_vmin"], step=1.0, format="%.1f",
                    key=f"{P}_vmin_input_sb",
                )
            with col_max:
                vmax = st.number_input(
                    "上限 (V)", min_value=HW_VOLTAGE_MIN, max_value=HW_VOLTAGE_MAX,
                    value=st.session_state[f"{P}_vmax"], step=1.0, format="%.1f",
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

        with st.container(border=True):
            st.markdown("##### 连接")
            st.text_input(
                "IP 地址", value=st.session_state[f"{P}_ip"],
                disabled=_connected, key=f"{P}_ip_input_sb",
            )
            st.session_state[f"{P}_ip"] = st.session_state[f"{P}_ip_input_sb"]
            st.number_input(
                "端口", min_value=1, max_value=65535,
                value=st.session_state[f"{P}_port"], step=1,
                disabled=_connected, key=f"{P}_port_input_sb",
            )
            st.session_state[f"{P}_port"] = int(st.session_state[f"{P}_port_input_sb"])
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
                if st.button("📡 检测连通性", use_container_width=True, key=f"{P}_test_btn_sb"):
                    test_connectivity()
                    st.rerun()
            with col_conn:
                if not _connected:
                    if st.button("🔌 连接", type="primary", use_container_width=True, key=f"{P}_connect_sb"):
                        with st.spinner("连接中..."):
                            connect()
                        st.rerun()
                else:
                    if st.button("⏏ 断开", use_container_width=True, key=f"{P}_disconnect_sb"):
                        if st.session_state[f"{P}_relay_on"]:
                            st.session_state[f"{P}_confirm_disconnect"] = True
                            st.rerun()
                        else:
                            disconnect()
                            st.rerun()

            if st.session_state[f"{P}_confirm_disconnect"]:
                st.warning("⚠️ 继电器仍处于**上电**状态, 断开连接前会先自动下电。确认继续?")
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("确认断开", type="primary", use_container_width=True, key=f"{P}_disconnect_confirm_sb"):
                        disconnect()
                        st.rerun()
                with col_n:
                    if st.button("取消", use_container_width=True, key=f"{P}_disconnect_cancel_sb"):
                        st.session_state[f"{P}_confirm_disconnect"] = False
                        st.rerun()

        with st.container(border=True):
            st.markdown("##### 继电器上下电")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button(
                    "⚡ 上电 (接通输出)", type="primary", use_container_width=True,
                    disabled=not _connected or st.session_state[f"{P}_relay_on"],
                    key=f"{P}_relay_on_btn_sb",
                ):
                    set_relay_power(True)
                    st.rerun()
            with col_r2:
                if st.button(
                    "⏻ 下电 (断开输出)", use_container_width=True,
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
                if st.button("🔌 连接 MicroDM", type="primary", use_container_width=True, key=f"{jc}_connect_btn_sb"):
                    with st.spinner("连接所有控制器..."):
                        _jc_connect()
                    st.rerun()
            else:
                col_r1, col_r2, col_disc = st.columns(3)
                with col_r1:
                    if st.button("⚡ 上电", type="primary", use_container_width=True,
                                 disabled=jc_relay_on, key=f"{jc}_relay_on_btn_sb"):
                        _jc_set_relay(True)
                        st.rerun()
                with col_r2:
                    if st.button("⏻ 下电", use_container_width=True,
                                 disabled=not jc_relay_on, key=f"{jc}_relay_off_btn_sb"):
                        _jc_set_relay(False)
                        st.rerun()
                with col_disc:
                    if st.button("⏏ 断开", use_container_width=True, key=f"{jc}_disconnect_btn_sb"):
                        _jc_disconnect()
                        st.rerun()

        with st.container(border=True):
            st.markdown("##### 批量上下电 (Ping 测试)")
            col_bon, col_boff = st.columns(2)
            with col_bon:
                if st.button("⚡ 批量上电 (先Ping)", type="primary", use_container_width=True,
                             disabled=jc_relay_on, key=f"{jc}_batch_on_btn_sb"):
                    _jc_batch_power_on()
                    st.rerun()
            with col_boff:
                if st.button("⏻ 批量下电", use_container_width=True,
                             disabled=not jc_relay_on, key=f"{jc}_batch_off_btn_sb"):
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
                            pd.DataFrame(rows), width='stretch', hide_index=True,
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
                if st.button("🔌 连接组控制器", type="primary", use_container_width=True, key=f"{gc}_connect_btn_sb"):
                    with st.spinner("连接中..."):
                        _gc_connect()
                    st.rerun()
            else:
                col_r1, col_r2, col_disc = st.columns(3)
                with col_r1:
                    if st.button("⚡ 上电", type="primary", use_container_width=True,
                                 disabled=gc_relay_on, key=f"{gc}_relay_on_btn_sb"):
                        _gc_set_relay(True)
                        st.rerun()
                with col_r2:
                    if st.button("⏻ 下电", use_container_width=True,
                                 disabled=not gc_relay_on, key=f"{gc}_relay_off_btn_sb"):
                        _gc_set_relay(False)
                        st.rerun()
                with col_disc:
                    if st.button("⏏ 断开", use_container_width=True, key=f"{gc}_disconnect_btn_sb"):
                        _gc_disconnect()
                        st.rerun()

        with st.container(border=True):
            st.markdown("##### 批量上下电 (Ping 测试)")
            col_bon, col_boff = st.columns(2)
            with col_bon:
                if st.button("⚡ 批量上电 (先Ping)", type="primary", use_container_width=True,
                             disabled=gc_relay_on, key=f"{gc}_batch_on_btn_sb"):
                    _gc_batch_power_on()
                    st.rerun()
            with col_boff:
                if st.button("⏻ 批量下电", use_container_width=True,
                             disabled=not gc_relay_on, key=f"{gc}_batch_off_btn_sb"):
                    _gc_batch_power_off()
                    st.rerun()
            st.caption("上电前自动 Ping 测试组内所有控制器连通性")


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


