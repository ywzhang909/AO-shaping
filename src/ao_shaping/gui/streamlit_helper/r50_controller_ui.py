"""
R50Power 控制器控制面板 (Streamlit)

面向 ``MicroDM.py`` 中 :class:`R50Controller` / :class:`MicroDM` 的 UI。

使用方式:
    streamlit run src/ao_shaping/gui/streamlit_helper/r50_controller_ui.py
"""

from __future__ import annotations

import collections
import json
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    MicroDM,
)
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT


# =============================================================================
# Configuration Constants
# =============================================================================

@dataclass(frozen=True)
class Cfg:
    """Application configuration constants."""
    # session_state key prefix
    PREFIX: str = "r50c"
    # single controller
    SINGLE_CHANNELS: int = 50
    DEFAULT_PORT: int = 10101
    # voltage hardware limits (never exceeded)
    HW_VOLTAGE_MIN: float = -20.0
    HW_VOLTAGE_MAX: float = 120.0
    # joint control 36x36 grid
    GRID_SIZE: int = 36
    DM_NUM_ACTUATORS: int = 96  # computed via __post_init__
    # UI refresh interval (s)
    REFRESH_INTERVAL: float = 0.15
    # debug
    DEBUG_LOG_MAX: int = 300
    DEBUG_TCP_HOST: str = "127.0.0.1"
    DEBUG_TCP_PORT: int = 9999
    # CSV data source
    CSV_PATH: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "1300-5-enriched.csv")

    def __post_init__(self) -> None:
        # Can't modify frozen dataclass, so use object.__setattr__
        object.__setattr__(self, "DM_NUM_ACTUATORS", self.GRID_SIZE * self.GRID_SIZE)


CFG = Cfg()
P = CFG.PREFIX

# Module-level aliases for CFG (used extensively — single source of truth remains CFG)
GRID_SIZE = CFG.GRID_SIZE
SINGLE_CHANNELS = CFG.SINGLE_CHANNELS
HW_VOLTAGE_MIN = CFG.HW_VOLTAGE_MIN
HW_VOLTAGE_MAX = CFG.HW_VOLTAGE_MAX
REFRESH_INTERVAL = CFG.REFRESH_INTERVAL
DEBUG_LOG_MAX = CFG.DEBUG_LOG_MAX
DEBUG_HOST = CFG.DEBUG_TCP_HOST
DEBUG_PORT = CFG.DEBUG_TCP_PORT
DM_NUM_ACTUATORS = CFG.DM_NUM_ACTUATORS


# =============================================================================
# Simulated Controller (无硬件测试)
# =============================================================================


class SimulatedR50Controller:
    """模拟 R50Controller，无硬件连接时用于 UI 流程测试。

    支持完整生命周期: open / close / set_relay / set_channel_voltage / set_all_channel_voltage。
    所有调用返回成功，电压记录在实例内部便于追踪。
    """

    def __init__(self, controller_id: int = 1, ip: str = "127.0.0.1", port: int = 0):
        self.controller_id = controller_id
        self._ip = ip
        self._port = port
        self._relay_on = False
        self._voltages: list[float] = [0.0] * 50
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        logger.info(f"[Sim] R50Controller(#{self.controller_id}) opened ({self._ip}:{self._port})")
        return True

    def close(self) -> None:
        self._opened = False
        self._relay_on = False
        logger.info(f"[Sim] R50Controller(#{self.controller_id}) closed")

    def set_relay(self, state: bool) -> bool:
        self._relay_on = state
        logger.info(f"[Sim] R50Controller(#{self.controller_id}) relay {'ON' if state else 'OFF'}")
        return True

    def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        if 0 <= channel < len(self._voltages):
            self._voltages[channel] = voltage
        logger.info(f"[Sim] R50Controller(#{self.controller_id}) ch{channel} = {voltage:.1f}V")
        return True

    def set_all_channel_voltage(self, voltage: float) -> bool:
        self._voltages = [voltage] * 50
        logger.info(f"[Sim] R50Controller(#{self.controller_id}) all channels = {voltage:.1f}V")
        return True


class SimulatedMicroDM:
    """模拟 MicroDM，无硬件连接时用于联合控制 UI 流程测试。

    管理多个 SimulatedR50Controller，支持 open/close/set_relay_state/send_voltages。
    """

    DM_Num: int = 36 * 36
    DM_NUM: int = 36 * 36
    V_Min: float = -20.0
    V_Max: float = 120.0
    max_neibor_diff: float = float("inf")

    def __init__(self, ips: list[str] | None = None, **_kwargs):
        if ips:
            self._ips = ips
        else:
            self._ips = ["192.168.0.101", "192.168.0.102"]
        self._controllers: list[SimulatedR50Controller] = []
        self._relay_state = False
        self._last_voltages: np.ndarray = np.zeros(self.DM_Num, dtype=np.float64)
        for i, ip_str in enumerate(self._ips, start=1):
            port = 10000 + int(ip_str.rsplit(".", 1)[-1])
            ctrl = SimulatedR50Controller(controller_id=i, ip=ip_str, port=port)
            self._controllers.append(ctrl)

    def open(self) -> None:
        for ctrl in self._controllers:
            ctrl.open()
        logger.info(f"[Sim] MicroDM opened ({len(self._controllers)} controllers)")

    def close(self) -> None:
        for ctrl in self._controllers:
            ctrl.close()
        logger.info("[Sim] MicroDM closed")

    def set_relay_state(self, state: bool) -> None:
        self._relay_state = state
        for ctrl in self._controllers:
            ctrl.set_relay(state)
        logger.info(f"[Sim] MicroDM relay {'ON' if state else 'OFF'}")

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        vs = np.asarray(vs, dtype=np.float64)
        vs = np.clip(vs, self.V_Min, self.V_Max)
        ch_per_ctrl = 50
        for i, ctrl in enumerate(self._controllers):
            start = i * ch_per_ctrl
            chunk = vs[start:start + ch_per_ctrl]
            padded = np.zeros(ch_per_ctrl, dtype=np.float64)
            padded[:len(chunk)] = chunk
            ctrl._voltages = padded.tolist()
        self._last_voltages = vs.copy()
        logger.info(f"[Sim] MicroDM sent voltages ({len(vs)} channels)")
        return self._last_voltages


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ChannelInfo:
    """Physical unit info from 1300-5-enriched.csv."""
    ip_suffix: int
    payload_position: int           # 1-based
    physical_position: int          # 1-based 36x36 position
    group: str
    needle_id: int
    physical_label: str

    @property
    def ip(self) -> str:
        return f"192.168.0.{self.ip_suffix}"

    @property
    def description(self) -> str:
        desc = f"ch{self.payload_position}"
        if self.needle_id:
            desc += f" 针脚#{self.needle_id}"
        if self.physical_label:
            desc += f" ({self.physical_label})"
        desc += f" [{self.ip}]"
        return desc

    def short_info(self) -> str:
        """Brief needle info for channel labels: '一组 针脚#277 (3-3-1)'."""
        if self.needle_id:
            return f"{self.group} 针脚#{self.needle_id} ({self.physical_label})"
        return f"{self.group} ({self.physical_label})"


@dataclass
class GroupDef:
    """A named group of channels from CSV, keyed by controller IP suffix."""
    name: str
    channels_by_ip: dict[int, list[ChannelInfo]]

    @property
    def total_channels(self) -> int:
        return sum(len(chs) for chs in self.channels_by_ip.values())

    @property
    def all_payload_positions(self) -> list[int]:
        return sorted(
            ch.payload_position
            for chs in self.channels_by_ip.values()
            for ch in chs
        )


# =============================================================================
# Debug Client
# =============================================================================

class DebugTcpClient:
    """TCP 调试客户端: 向远程调试服务器推送 JSON 操作日志。

    自动重连 (退避 5s) + 存储于 session_state。

    Usage::

        client = DebugTcpClient.from_session(prefix)
        client.send("connect", detail="port=10101", ip="192.168.0.101")
        client.disconnect()
    """

    def __init__(self, host: str = DEBUG_HOST, port: int = DEBUG_PORT) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._last_fail: float = 0.0
        self._retry_interval: float = 5.0

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def configure(self, host: str, port: int) -> None:
        self.disconnect()
        self._host = host
        self._port = port

    def connect(self) -> bool:
        """建立 TCP 连接 (含 5s 退避防止频繁重试)。"""
        if self._sock is not None:
            return True
        if time.monotonic() - self._last_fail < self._retry_interval:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self._host, self._port))
            self._sock = sock
            self._last_fail = 0.0
            logger.debug(f"Debug client connected to {self._host}:{self._port}")
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Debug client connect failed: {e}")
            self._sock = None
            self._last_fail = time.monotonic()
            return False

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            self._last_fail = 0.0

    def send(self, operation: str, detail: str = "", ip: str = "") -> bool:
        """发送 JSON 操作日志到调试服务器。失败时自动断开。"""
        if self._sock is None:
            if not self.connect():
                return False
        msg = json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "operation": operation,
                "ip": ip,
                "detail": detail,
            },
            ensure_ascii=False,
        )
        try:
            self._sock.sendall((msg + "\n").encode("utf-8"))
            return True
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"Debug client send failed: {e}")
            self.disconnect()
            return False

    @staticmethod
    def from_session(prefix: str) -> DebugTcpClient:
        key = f"{prefix}_debug_client"
        if key not in st.session_state:
            host = st.session_state.get(f"{prefix}_debug_tcp_host", DEBUG_HOST)
            port = st.session_state.get(f"{prefix}_debug_tcp_port", DEBUG_PORT)
            st.session_state[key] = DebugTcpClient(host, port)
        else:
            client = st.session_state[key]
            ui_host = st.session_state.get(f"{prefix}_debug_tcp_host", DEBUG_HOST)
            ui_port = st.session_state.get(f"{prefix}_debug_tcp_port", DEBUG_PORT)
            if client._host != ui_host or client._port != ui_port:
                client.configure(ui_host, ui_port)
        return st.session_state[key]

    @staticmethod
    def remove_from_session(prefix: str) -> None:
        key = f"{prefix}_debug_client"
        if key in st.session_state:
            client = st.session_state[key]
            client.disconnect()
            del st.session_state[key]


# =============================================================================
# CSV Loading
# =============================================================================

def _load_csv() -> pd.DataFrame:
    """Load 1300-5-enriched.csv, returns empty DataFrame on failure."""
    if not CFG.CSV_PATH.exists():
        logger.warning(f"CSV not found: {CFG.CSV_PATH}")
        return pd.DataFrame()
    try:
        return pd.read_csv(CFG.CSV_PATH)
    except Exception as e:
        logger.warning(f"Failed to load CSV: {e}")
        return pd.DataFrame()


def _row_to_channel_info(row) -> ChannelInfo:
    """Convert a CSV row to ChannelInfo."""
    ip_suffix = int(row["IP组"])
    pp = int(row["序号"]) + 1          # 0-based -> 1-based
    r = int(row["36×36行"])
    c = int(row["36×36列"])
    pos = r * CFG.GRID_SIZE + c + 1    # 1-based
    return ChannelInfo(
        ip_suffix=ip_suffix,
        payload_position=pp,
        physical_position=pos,
        group=str(row["组"]),
        needle_id=int(row["引脚编号"]),
        physical_label=str(row["连接器"]),
    )


def _build_csv_index(df: pd.DataFrame | None = None) -> dict[tuple[int, int], ChannelInfo]:
    """Build (ip_suffix, payload_position) -> ChannelInfo from CSV."""
    if df is None:
        df = _load_csv()
    index: dict[tuple[int, int], ChannelInfo] = {}
    if df.empty:
        return index
    try:
        for _, row in df.iterrows():
            ci = _row_to_channel_info(row)
            key = (ci.ip_suffix, ci.payload_position)
            if key in index:
                continue
            index[key] = ci
    except Exception as e:
        logger.warning(f"Failed to build CSV index: {e}")
    return index


def _get_wiring_index() -> dict[tuple[int, int], ChannelInfo]:
    """Lazy-loaded CSV index, cached in function attribute (avoids module-level global)."""
    if not hasattr(_get_wiring_index, "_cache"):
        _get_wiring_index._cache = _build_csv_index()  # type: ignore[attr-defined]
    return _get_wiring_index._cache  # type: ignore[attr-defined]


def _get_channel_info(channel: int) -> ChannelInfo | None:
    """Look up ChannelInfo for a given 0-based channel of the current single controller."""
    ip = st.session_state.get(f"{P}_ip", "").strip()
    idx = _get_wiring_index()
    if not ip or not idx:
        return None
    try:
        ip_suffix = int(ip.split(".")[-1])
    except (ValueError, IndexError):
        return None
    payload_pos = channel + 1
    return idx.get((ip_suffix, payload_pos))


def _channel_label(ch: int) -> str:
    """Formatted channel label with needle info for multiselect."""
    info = _get_channel_info(ch)
    return f"{ch} | {info.short_info()}" if info else str(ch)



# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量。"""

    # ---- CSV 索引 (仅构建一次) ----
    _get_wiring_index()  # trigger lazy load

    # ---- 连接配置 ----
    st.session_state.setdefault(f"{P}_ip", "192.168.0.101")
    st.session_state.setdefault(f"{P}_port", CFG.DEFAULT_PORT)
    st.session_state.setdefault(f"{P}_controller", None)
    st.session_state.setdefault(f"{P}_connected", False)
    st.session_state.setdefault(f"{P}_connection_error", "")
    st.session_state.setdefault(f"{P}_simulate", False)
    st.session_state.setdefault(f"{P}_jc_simulate", False)
    st.session_state.setdefault(f"{P}_gc_simulate", False)

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
    st.session_state.setdefault(f"{P}_feedback", "")
    st.session_state.setdefault(f"{P}_feedback_type", "")

    # ---- 连接模式 (sidebar) ----
    st.session_state.setdefault(f"{P}_connection_mode", "single")  # "single" | "joint" | "group"

    # ---- 调试模式 (packet hex log) ----
    st.session_state.setdefault(f"{P}_debug", False)
    st.session_state.setdefault(
        f"{P}_debug_log", collections.deque(maxlen=DEBUG_LOG_MAX)
    )

    # ---- TCP 调试客户端 ----
    st.session_state.setdefault(f"{P}_debug_tcp_enabled", False)
    st.session_state.setdefault(f"{P}_debug_tcp_host", DEBUG_HOST)
    st.session_state.setdefault(f"{P}_debug_tcp_port", DEBUG_PORT)
    st.session_state.setdefault(
        f"{P}_debug_op_log", collections.deque(maxlen=DEBUG_LOG_MAX)
    )
    st.session_state.setdefault(
        f"{P}_local_debug_logs", collections.deque(maxlen=100)
    )
    # DebugTcpClient is created lazily by DebugTcpClient.from_session()

    # ---- 各单元当前电压 (可视化, 本地跟踪) ----
    st.session_state.setdefault(
        f"{P}_current_voltages", np.zeros(SINGLE_CHANNELS, dtype=np.float64)
    )

    # ---- Joint Control (JC) session state ----
    _init_jc_state()

    # ---- Group Control (GC) session state ----
    _init_gc_state()


def _init_jc_state() -> None:
    """初始化联合控制 (36×36 矩阵) 的 session_state 变量。"""
    jc = f"{P}_jc"

    # 连接
    st.session_state.setdefault(f"{jc}_dm", None)
    st.session_state.setdefault(f"{jc}_connected", False)
    st.session_state.setdefault(f"{jc}_connection_error", "")
    st.session_state.setdefault(f"{jc}_relay_on", False)
    st.session_state.setdefault(f"{jc}_controller_count", 0)
    st.session_state.setdefault(f"{jc}_ip_list", [])

    # 36×36 矩阵数据
    st.session_state.setdefault(
        f"{jc}_matrix", np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    )

    # 物理位置 → (ip_suffix, payload_position) 索引
    st.session_state.setdefault(f"{jc}_pos_to_hw", {})          # int → (ip_suffix, payload_position)
    st.session_state.setdefault(f"{jc}_ip_to_controller_idx", {})  # ip_suffix → controller index
    st.session_state.setdefault(f"{jc}_sorted_ips", [])
    st.session_state.setdefault(f"{jc}_dm_num", 0)
    st.session_state.setdefault(f"{jc}_matrix_init", False)

    # 编辑控制
    st.session_state.setdefault(f"{jc}_edit_row", 0)
    st.session_state.setdefault(f"{jc}_edit_col", 0)
    st.session_state.setdefault(f"{jc}_edit_voltage", 0.0)
    st.session_state.setdefault(f"{jc}_fill_voltage", 0.0)
    st.session_state.setdefault(f"{jc}_fill_all_voltage", 0.0)

    # 矩形区域
    st.session_state.setdefault(f"{jc}_rect_x1", 0)
    st.session_state.setdefault(f"{jc}_rect_y1", 0)
    st.session_state.setdefault(f"{jc}_rect_x2", GRID_SIZE - 1)
    st.session_state.setdefault(f"{jc}_rect_y2", GRID_SIZE - 1)

    # 反馈
    st.session_state.setdefault(f"{jc}_feedback", "")
    st.session_state.setdefault(f"{jc}_feedback_type", "")


# =============================================================================
# Joint Control: CSV Wiring Index (1300-5-enriched.csv)
# =============================================================================

def _jc_build_wiring_index(df: pd.DataFrame | None = None) -> dict[int, tuple[int, int]]:
    """从 1300-5-enriched.csv 构建 physical_position → (ip_suffix, payload_position) 映射。

    physical_position (1-1296) 对应 36×36 矩阵位置: row=36×36行, col=36×36列。
    payload_position 输出为 1-based (序号 0-based → +1)。
    """
    if df is None:
        df = _load_csv()
    pos_to_hw: dict[int, tuple[int, int]] = {}
    if df.empty:
        return pos_to_hw
    try:
        for _, row in df.iterrows():
            ip_s = int(row["IP组"])
            pp = int(row["序号"]) + 1
            r = int(row["36×36行"])
            c = int(row["36×36列"])
            pos = r * GRID_SIZE + c + 1
            pos_to_hw[pos] = (ip_s, pp)
    except Exception as e:
        logger.warning(f"Failed to build joint control CSV index: {e}")
    return pos_to_hw


def _jc_build_ip_index(df: pd.DataFrame | None = None) -> dict[int, int]:
    """从 CSV 构建 ip_suffix → controller_index 映射 (flat array 顺序)。"""
    if df is None:
        df = _load_csv()
    if df.empty:
        return {}
    sorted_ips = sorted(df["IP组"].unique())
    return {ip: idx for idx, ip in enumerate(sorted_ips)}


# =============================================================================
# Joint Control: Matrix ↔ Flat Array Conversion
# =============================================================================

def _jc_matrix_to_flat(
    matrix: np.ndarray,
    pos_to_hw: dict[int, tuple[int, int]],
    ip_to_ctrl_idx: dict[int, int],
    dm_num: int,
) -> np.ndarray:
    """将 36×36 矩阵转换为 MicroDM.send_voltages() 所需的 flat array。

    MicroDM 的 flat array 顺序: controller[0]→vs[0:50], controller[1]→vs[50:100], ...
    """
    flat = np.zeros(dm_num, dtype=np.float64)
    for physical_pos, (ip_suffix, payload_pos) in pos_to_hw.items():
        row = (physical_pos - 1) // GRID_SIZE
        col = (physical_pos - 1) % GRID_SIZE
        voltage = matrix[row, col]
        ctrl_idx = ip_to_ctrl_idx.get(ip_suffix)
        if ctrl_idx is not None:
            flat_idx = ctrl_idx * 50 + (payload_pos - 1)
            if flat_idx < dm_num:
                flat[flat_idx] = voltage
    return flat


def _jc_read_matrix_from_dm(
    dm: MicroDM,
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
            flat_idx = ctrl_idx * 50 + (payload_pos - 1)
            if flat_idx < len(flat):
                matrix[row, col] = flat[flat_idx]
    return matrix


# =============================================================================
# Joint Control: Connect / Disconnect
# =============================================================================

def _jc_connect() -> None:
    """连接 MicroDM (所有控制器, 支持仿真模式)。"""
    jc = f"{P}_jc"
    try:
        simulate = st.session_state.get(f"{P}_jc_simulate", False)
        if simulate:
            dm = SimulatedMicroDM(use_wiring_map=True)  # type: ignore[call-arg]
            dm.open()
            feedback_prefix = "🟡 [仿真] "
        else:
            dm = MicroDM(use_wiring_map=True)  # type: ignore[call-arg]
            dm.open()
            feedback_prefix = ""

        st.session_state[f"{jc}_dm"] = dm
        st.session_state[f"{jc}_connected"] = True
        st.session_state[f"{jc}_relay_on"] = False
        st.session_state[f"{jc}_connection_error"] = ""
        st.session_state[f"{jc}_controller_count"] = len(dm._controllers)  # type: ignore

        # 从 CSV 构建索引
        csv_df = _load_csv()
        if csv_df.empty:
            st.session_state[f"{jc}_connection_error"] = "1300-5-enriched.csv 加载失败"
            st.session_state[f"{jc}_connected"] = False
            return
        pos_to_hw = _jc_build_wiring_index(csv_df)
        ip_to_ctrl_idx = _jc_build_ip_index(csv_df)
        sorted_ips = [f"192.168.0.{s}" for s in sorted(csv_df["IP组"].unique())]
        st.session_state[f"{jc}_pos_to_hw"] = pos_to_hw
        st.session_state[f"{jc}_ip_to_controller_idx"] = ip_to_ctrl_idx
        st.session_state[f"{jc}_sorted_ips"] = sorted_ips
        st.session_state[f"{jc}_dm_num"] = dm.DM_Num
        st.session_state[f"{jc}_matrix_init"] = True

        # 仿真模式跳过硬件读取, 矩阵保持全零
        if not simulate:
            matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl_idx)
            st.session_state[f"{jc}_matrix"] = matrix
        else:
            st.session_state[f"{jc}_matrix"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

        n_ctrl = st.session_state[f"{jc}_controller_count"]
        st.session_state[f"{jc}_feedback"] = (
            f"{feedback_prefix}已连接 MicroDM: {n_ctrl} 个控制器"
        )
        st.session_state[f"{jc}_feedback_type"] = "success"
        logger.info(f"MicroDM connected: {n_ctrl} controllers")
        _debug_add_op("connect", f"joint ({n_ctrl} controllers)", "all")
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
    _dm = st.session_state[f"{jc}_dm"]
    if _dm is not None:
        try:
            if st.session_state[f"{jc}_relay_on"]:
                _dm.set_relay_state(False)
            _dm.close()
        except Exception as e:
            logger.warning(f"MicroDM disconnect warning: {e}")
    st.session_state[f"{jc}_dm"] = None
    st.session_state[f"{jc}_connected"] = False
    st.session_state[f"{jc}_relay_on"] = False
    st.session_state[f"{jc}_connection_error"] = ""
    st.session_state[f"{jc}_feedback"] = "已断开连接 (已先下电)"
    st.session_state[f"{jc}_feedback_type"] = "info"
    logger.info("MicroDM disconnected")
    _debug_add_op("disconnect", "joint", "all")


# =============================================================================
# Joint Control: Relay / Apply
# =============================================================================

def _jc_set_relay(on: bool) -> None:
    """所有控制器继电器上下电。"""
    jc = f"{P}_jc"
    _dm = st.session_state[f"{jc}_dm"]
    if _dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        _dm.set_relay_state(on)
        st.session_state[f"{jc}_relay_on"] = on
        if on:
            st.session_state[f"{jc}_feedback"] = "✅ 所有控制器继电器已上电 (输出接通)"
            st.session_state[f"{jc}_feedback_type"] = "success"
            logger.info("MicroDM relay ON (all controllers)")
            _debug_add_op("relay_on", "joint", "all")
        else:
            st.session_state[f"{jc}_feedback"] = "⏻ 所有控制器继电器已下电 (输出断开)"
            st.session_state[f"{jc}_feedback_type"] = "info"
            logger.info("MicroDM relay OFF (all controllers)")
            _debug_add_op("relay_off", "joint", "all")
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"继电器操作失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM relay failed: {e}")


def _jc_apply_matrix() -> None:
    """将当前 36×36 矩阵电压下发到所有控制器。"""
    jc = f"{P}_jc"
    dm: MicroDM | None = st.session_state[f"{jc}_dm"]
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    if not st.session_state[f"{jc}_relay_on"]:
        st.session_state[f"{jc}_feedback"] = "⚠️ 请先继电器上电后再下发电压"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        matrix = st.session_state[f"{jc}_matrix"]
        pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
        ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
        dm_num = st.session_state[f"{jc}_dm_num"]
        flat = _jc_matrix_to_flat(matrix, pos_to_hw, ip_to_ctrl, dm_num)
        dm.send_voltages(flat)
        # 统计非零通道数
        non_zero = np.count_nonzero(matrix)
        st.session_state[f"{jc}_feedback"] = (
            f"✅ 已下发 36×36 矩阵电压 (非零通道: {non_zero}/{DM_NUM_ACTUATORS})"
        )
        st.session_state[f"{jc}_feedback_type"] = "success"
        logger.info(f"MicroDM voltage applied: {non_zero} non-zero channels")
        _debug_add_op("set_voltage", f"matrix {non_zero} non-zero channels", "all")
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"电压下发失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM apply failed: {e}")


def _jc_reset_matrix() -> None:
    """将矩阵清零并下发。"""
    jc = f"{P}_jc"
    st.session_state[f"{jc}_matrix"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    if st.session_state[f"{jc}_connected"] and st.session_state[f"{jc}_relay_on"]:
        _jc_apply_matrix()
    else:
        st.session_state[f"{jc}_feedback"] = "矩阵已清零 (未下发到硬件)"
        st.session_state[f"{jc}_feedback_type"] = "info"


def _jc_refresh_from_hardware() -> None:
    """从硬件读取当前电压并刷新矩阵显示。"""
    jc = f"{P}_jc"
    dm = st.session_state[f"{jc}_dm"]
    if dm is None:
        return
    # 仿真模式: 矩阵不变
    if st.session_state.get(f"{P}_jc_simulate", False):
        st.session_state[f"{jc}_feedback"] = "仿真模式: 矩阵保持当前值"
        st.session_state[f"{jc}_feedback_type"] = "info"
        return
    pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
    ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
    matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl)
    st.session_state[f"{jc}_matrix"] = matrix
    st.session_state[f"{jc}_feedback"] = "已从硬件刷新电压矩阵"
    st.session_state[f"{jc}_feedback_type"] = "info"


# =============================================================================
# Joint Control: Batch Power On/Off (with Ping Test)
# =============================================================================


def _jc_batch_power_on() -> None:
    """批量上电: 先 ping 测试所有控制器, 再继电器上电。"""
    jc = f"{P}_jc"
    dm = st.session_state[f"{jc}_dm"]
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
    reachable = []
    unreachable = []
    for ip in sorted_ips:
        if simulate or _ping_reachable(ip, timeout=1.0):
            reachable.append(ip)
        else:
            unreachable.append(ip)

    if not reachable:
        st.session_state[f"{jc}_feedback"] = (
            f"❌ 所有控制器均不可达: {', '.join(unreachable)}"
        )
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
# Joint Control: Matrix Editing
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
# Joint Control: Visualization (Streamlit Native)
# =============================================================================

def _jc_colormap_image(matrix: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """将电压矩阵转换为彩色图像 (numpy, 无 matplotlib 依赖)。

    使用 coolwarm 风格的配色:
    - 蓝色:  低电压 (vmin)
    - 白色:  中间 (0)
    - 红色:  高电压 (vmax)
    """
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = (matrix - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    h, w = matrix.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Coolwarm-like colormap
    # Blue → White → Red
    # Blue region: n ∈ [0, 0.5] → R=0, G=n*2, B=1
    # Red region: n ∈ [0.5, 1] → R=1, G=2-2n, B=0
    mask_low = normalized <= 0.5
    mask_high = normalized > 0.5

    img[mask_low, 0] = normalized[mask_low] * 2.0          # R
    img[mask_low, 1] = normalized[mask_low] * 2.0           # G
    img[mask_low, 2] = 1.0                                   # B

    img[mask_high, 0] = 1.0                                   # R
    img[mask_high, 1] = 2.0 - normalized[mask_high] * 2.0    # G
    img[mask_high, 2] = 2.0 - normalized[mask_high] * 2.0    # B

    return img


def _jc_render_matrix_image(matrix: np.ndarray) -> None:
    """Streamlit ``st.image`` 显示彩色热力图。"""
    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)
    img = _jc_colormap_image(matrix, vmin, vmax)
    st.image(img, caption="36×36 电压分布 (蓝色低 · 红色高)", width='stretch')


def _jc_render_matrix_dataframe(matrix: np.ndarray) -> None:
    """Streamlit ``st.dataframe`` 显示带颜色的 36×36 数值矩阵。

    使用 st.column_config.NumberColumn 配置。
    将 36 列拆分为 6 组 × 6 列, 通过 expander 展示。
    """
    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)

    # 拆分为 6 块 6 列显示 (36 列在一屏太宽)
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
    """行/列剖面图表。"""
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    with st.container(border=True):
        st.markdown("##### 电压剖面")
        tab_r, tab_c = st.tabs(["📊 行均值", "📊 列均值"])
        with tab_r:
            df_row = pd.DataFrame({"行号": list(range(1, GRID_SIZE + 1)), "均值 (V)": row_means}).set_index("行号")
            st.bar_chart(df_row, height=200, width='stretch')
        with tab_c:
            df_col = pd.DataFrame({"列号": list(range(1, GRID_SIZE + 1)), "均值 (V)": col_means}).set_index("列号")
            st.bar_chart(df_col, height=200, width='stretch')


def _jc_render_stats(matrix: np.ndarray) -> None:
    """显示矩阵统计指标。"""
    vals = matrix.flatten()
    non_zero_count = np.count_nonzero(matrix)
    vmin = np.min(vals)
    vmax = np.max(vals)
    vmean = np.mean(vals)
    vstd = np.std(vals)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("最小值", f"{vmin:.1f} V")
    with col2:
        st.metric("最大值", f"{vmax:.1f} V")
    with col3:
        st.metric("均值", f"{vmean:.1f} V")
    with col4:
        st.metric("标准差", f"{vstd:.1f} V")
    with col5:
        st.metric("非零通道", f"{non_zero_count}/{DM_NUM_ACTUATORS}")



# =============================================================================
# Joint Control: Main Tab Renderer
# =============================================================================

def render_tab_all_control() -> None:
    """全部控制 Tab: 36×36 联合矩阵全量控制。"""
    st.title("🔗 全部控制")
    st.caption("MicroDM 36×36 压电陶瓷矩阵 · 全量联合编辑与下发")

    jc = f"{P}_jc"
    matrix: np.ndarray = st.session_state[f"{jc}_matrix"]
    connected = st.session_state.get(f"{jc}_connected", False)
    relay_on = st.session_state.get(f"{jc}_relay_on", False)

    # 检查连接模式
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

    # ===== 矩阵可视化 =====
    st.divider()
    st.markdown("##### 36×36 电压矩阵 (Streamlit 原生控件)")

    col_img, col_edit = st.columns([3, 1])

    with col_img:
        # 彩色热力图
        _jc_render_matrix_image(matrix)

        # 数值矩阵 (分块)
        _jc_render_matrix_dataframe(matrix)

    with col_edit:
        with st.container(border=True):
            st.markdown("###### 编辑矩阵")

            # 全部填充
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

            # 单个单元
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

            # 行/列填充
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

            # 矩形区域
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

    # ===== 发送/硬件操作 =====
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

    # ===== 剖面 & 统计 =====
    st.divider()
    _jc_render_stats(matrix)
    _jc_render_profile(matrix)

    # ===== 针脚信息 =====
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
# Feedback Helpers
# =============================================================================

def _set_feedback(message: str, msg_type: str = "info", prefix: str = "") -> None:
    """设置反馈信息 (prefix: 留空用通用, "jc"/"gc" 等)."""
    if prefix:
        st.session_state[f"{P}_{prefix}_feedback"] = message
        st.session_state[f"{P}_{prefix}_feedback_type"] = msg_type
    else:
        st.session_state[f"{P}_feedback"] = message
        st.session_state[f"{P}_feedback_type"] = msg_type


def _show_feedback(prefix: str = "") -> None:
    """显示反馈并清除 (prefix: 留空用通用)."""
    if prefix:
        fb = f"{P}_{prefix}_feedback"
        ft = f"{P}_{prefix}_feedback_type"
    else:
        fb = f"{P}_feedback"
        ft = f"{P}_feedback_type"
    message = st.session_state.get(fb, "")
    msg_type = st.session_state.get(ft, "")
    if message:
        if msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
        elif msg_type == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state[fb] = ""
        st.session_state[ft] = ""


def _debug_log_packet(cmd_name: str, packet: bytes) -> None:
    """记录一条调试日志: 指令名 + 下发数据包的十六进制内容。"""
    if not st.session_state.get(f"{P}_debug"):
        return
    ts = time.strftime("%H:%M:%S", time.localtime())
    hexstr = " ".join(f"{b:02X}" for b in packet)
    st.session_state[f"{P}_debug_log"].append(f"[{ts}] {cmd_name}: {hexstr}")


def _debug_tcp_connect() -> DebugTcpClient | None:
    """从 session_state 获取 DebugTcpClient 并触发连接。"""
    if not st.session_state.get(f"{P}_debug_tcp_enabled"):
        return None
    client = DebugTcpClient.from_session(P)
    client.connect()
    return client if client.connected else None


def _debug_tcp_disconnect() -> None:
    """断开并移除 DebugTcpClient。"""
    DebugTcpClient.remove_from_session(P)


def _debug_add_op(operation: str, detail: str = "", ip: str = "") -> None:
    """记录操作日志到本地 log 并推送到 TCP 调试客户端。"""
    ts = time.strftime("%H:%M:%S", time.localtime())
    log_line = f"[{ts}] {operation}"
    if ip:
        log_line += f"  IP={ip}"
    if detail:
        log_line += f"  {detail}"
    st.session_state[f"{P}_debug_op_log"].append(log_line)

    if st.session_state.get(f"{P}_debug_tcp_enabled"):
        client = DebugTcpClient.from_session(P)
        client.send(operation, detail, ip)


# Module-level lock + buffer for thread-safe local debug server logging
_local_debug_lock = threading.Lock()
_local_debug_buffer: list[str] = []


def _drain_local_debug_buffer() -> list[str]:
    """Drain received messages from the local debug server into session state."""
    with _local_debug_lock:
        batch = list(_local_debug_buffer)
        _local_debug_buffer.clear()
    return batch


def _debug_tcp_start_local_server() -> None:
    """启动本地 TCP 调试服务器，无设备时用于测试调试消息收发。"""
    host = "127.0.0.1"
    port = st.session_state.get(f"{P}_debug_tcp_port", DEBUG_PORT)

    st.session_state[f"{P}_local_debug_server"] = True
    st.session_state[f"{P}_local_debug_logs"].clear()
    _local_debug_buffer.clear()

    def _run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as e:
            logger.error(f"Local debug server bind failed: {e}")
            st.session_state[f"{P}_local_debug_server"] = False
            return
        sock.listen(5)
        sock.settimeout(1.0)
        logger.info(f"Local debug server listening on {host}:{port}")
        while st.session_state.get(f"{P}_local_debug_server", False):
            try:
                conn, addr = sock.accept()
                logger.debug(f"Local debug server accepted {addr}")
                conn.settimeout(0.5)
                buf = b""
                try:
                    while st.session_state.get(f"{P}_local_debug_server", False):
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            msg = line.decode("utf-8").strip()
                            if msg:
                                with _local_debug_lock:
                                    _local_debug_buffer.append(msg)
                except (socket.timeout, OSError, ConnectionError):
                    pass
                finally:
                    conn.close()
            except socket.timeout:
                continue
            except OSError:
                break
        sock.close()
        logger.info("Local debug server stopped")

    thread = threading.Thread(target=_run, daemon=True)
    st.session_state[f"{P}_local_debug_thread"] = thread
    thread.start()

    # 配置调试客户端连接到本地服务器
    st.session_state[f"{P}_debug_tcp_host"] = host
    st.session_state[f"{P}_debug_tcp_port"] = port
    client = DebugTcpClient.from_session(P)
    client.configure(host, port)
    client.connect()


def _debug_tcp_stop_local_server() -> None:
    """停止本地 TCP 调试服务器。"""
    st.session_state[f"{P}_local_debug_server"] = False
    _debug_tcp_disconnect()


def _sidebar_debug_panel() -> None:
    """Sidebar 调试面板: 仿真状态 + 操作日志显示。"""
    with st.container(border=True):
        st.markdown("##### 🐛 调试面板")

        # 仿真模式状态 (各模式独立)
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

        # 操作日志折叠区
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

    # 仿真模式下直接报告可达
    if st.session_state.get(f"{P}_simulate", False):
        _set_feedback(f"🟡 [仿真模式] {ip}:{port} 模拟可达", "success")
        return

    tcp_ok = _tcp_reachable(ip, port)
    ping_ok = _ping_reachable(ip)
    if tcp_ok:
        msg = f"✅ TCP {ip}:{port} 可连通" + ("" if ping_ok else " (ICMP ping 未响应)")
        _set_feedback(msg, "success")
    else:
        detail = "TCP 端口不可达" + ("" if ping_ok else "，且 ICMP ping 未响应")
        _set_feedback(f"❌ {ip}:{port} {detail}", "error")


# =============================================================================
# 连接 / 断开
# =============================================================================

def connect() -> None:
    """连接单个 R50Power 控制器。自动配置 TCP 调试客户端到同一 IP。

    仿真模式 (``{P}_simulate``) 下使用 ``SimulatedR50Controller`` 替代真实设备。
    """
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

        simulate = st.session_state.get(f"{P}_simulate", False)
        if simulate:
            ctrl = SimulatedR50Controller(controller_id=1, ip=ip, port=port)
            ctrl.open()
            feedback_msg = f"🟡 仿真模式 — 已连接到 {ip}:{port} (模拟设备)"
            logger.info(f"SimulatedR50Controller connected: {ip}:{port}")
        else:
            ctrl = R50Controller(controller_id=1, ip=ip, port=port)
            if not ctrl.open():
                raise ConnectionError(f"无法建立 TCP 连接到 {ip}:{port}")
            feedback_msg = f"已连接到 {ip}:{port}"

        st.session_state[f"{P}_controller"] = ctrl
        st.session_state[f"{P}_connected"] = True
        st.session_state[f"{P}_relay_on"] = False
        st.session_state[f"{P}_connection_error"] = ""
        logger.info(f"R50Controller connected: {ip}:{port}")

        # 自动配置 TCP 调试客户端到同一控制器 IP:Port
        debug_msg = ""
        if st.session_state.get(f"{P}_debug_tcp_enabled"):
            client = DebugTcpClient.from_session(P)
            client.configure(ip, port)
            if client.connect():
                debug_msg = "，已连接到本地调试客户端"
        _set_feedback(f"{feedback_msg}{debug_msg}", "success")

        _debug_add_op("connect", f"port={port}", ip)
    except Exception as e:
        st.session_state[f"{P}_connection_error"] = f"连接失败: {e}"
        st.session_state[f"{P}_connected"] = False
        st.session_state[f"{P}_controller"] = None
        _set_feedback(f"连接失败: {e}", "error")
        logger.exception(f"R50Controller connect failed: {e}")


def disconnect() -> None:
    """断开控制器连接 (若继电器仍上电则先自动下电)。"""
    ctrl = st.session_state[f"{P}_controller"]
    st.session_state[f"{P}_sine_running"] = False
    st.session_state[f"{P}_square_running"] = False
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
    ip = st.session_state.get(f"{P}_ip", "")
    st.session_state[f"{P}_controller"] = None
    st.session_state[f"{P}_connected"] = False
    st.session_state[f"{P}_relay_on"] = False
    st.session_state[f"{P}_confirm_disconnect"] = False
    st.session_state[f"{P}_connection_error"] = ""
    logger.info("R50Controller disconnected")
    _set_feedback("已断开连接 (已先下电)", "info")
    _debug_add_op("disconnect", "", ip)


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
        _set_feedback("设备未连接", "error")
        return
    ip = st.session_state.get(f"{P}_ip", "")
    try:
        packet = HEADER + bytes([CMD_RELAY_ON if on else CMD_RELAY_OFF]) + FOOTER
        if ctrl.set_relay(on):
            st.session_state[f"{P}_relay_on"] = on
            _debug_log_packet(f"RELAY {'ON(上电)' if on else 'OFF(下电)'}", packet)
            _debug_add_op("relay_on" if on else "relay_off", "", ip)
            if on:
                _set_feedback("✅ 继电器已上电 (输出接通)", "success")
                logger.info("Relay powered ON")
            else:
                _set_feedback("⏻ 继电器已下电 (输出断开)", "info")
                logger.info("Relay powered OFF")
        else:
            _set_feedback("继电器指令发送失败", "error")
    except Exception as e:
        _set_feedback(f"继电器操作失败: {e}", "error")
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
    ip = st.session_state.get(f"{P}_ip", "")
    _debug_add_op("set_voltage", f"ch={channel} {v:.1f}V", ip)


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
    ip = st.session_state.get(f"{P}_ip", "")
    _debug_add_op("set_voltage", f"all_channels {v:.1f}V", ip)


def _require_relay_on() -> bool:
    """检查继电器是否已上电, 未上电时给出反馈并返回 False。"""
    if not st.session_state[f"{P}_connected"]:
        _set_feedback("设备未连接", "error")
        return False
    if not st.session_state[f"{P}_relay_on"]:
        _set_feedback("⚠️ 请先继电器上电后再下发电压", "error")
        return False
    return True


def _send_channels(voltage: float) -> None:
    """下发电压: 全部单元(50)模式 -> 全部通道; 否则 -> 指定单元(多选)。"""
    if st.session_state[f"{P}_all_mode"]:
        _send_all(voltage)
    else:
        for ch in st.session_state[f"{P}_channels"]:
            _send_single(int(ch), voltage)


def _send_success_feedback(voltage: float) -> None:
    """发送成功后显示反馈 (all_mode / 指定单元 两种格式)。"""
    if st.session_state[f"{P}_all_mode"]:
        _mapped = []
        for _ch in range(SINGLE_CHANNELS):
            _ci = _get_channel_info(_ch)
            if _ci and _ci.needle_id:
                _mapped.append(_channel_label(_ch))
        _extra = f" (映射通道: {', '.join(_mapped[:5])}{'...' if len(_mapped) > 5 else ''})" if _mapped else ""
        _set_feedback(f"已向全部 50 通道下发 {voltage:.1f} V{_extra}", "success")
    else:
        ch_list = st.session_state[f"{P}_channels"]
        _detail_parts = []
        for _ch in ch_list:
            _ci = _get_channel_info(int(_ch))
            _detail_parts.append(_channel_label(int(_ch)) if _ci else str(_ch))
        _detail = ", ".join(_detail_parts)
        _set_feedback(f"已向 {len(ch_list)} 个单元下发 {voltage:.1f} V: {_detail}", "success")


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
        _set_feedback(f"持续下发异常: {e}", "error")


def _sine_loop(amp: float, offset: float, freq: float, dt: float) -> None:
    """正弦电压持续下发线程。使用当前单元选择 (all_mode / channels)。"""
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
            and not st.session_state[f"{P}_square_running"]
        ):
            elapsed = time.time() - t0
            v = offset + amp * np.sin(omega * elapsed)
            _send_channels(v)
            time.sleep(dt)
    except Exception as e:
        st.session_state[f"{P}_sine_running"] = False
        logger.exception(f"正弦下发异常: {e}")
        _set_feedback(f"正弦下发异常: {e}", "error")


def _alt_loop(voltage: float, freq: float, dt: float) -> None:
    """交替下发 0V 和 input 电压的循环线程。"""
    ctrl = st.session_state[f"{P}_controller"]
    if ctrl is None:
        return
    try:
        state = 0  # 0 = sending 0V, 1 = sending input voltage
        while (
            st.session_state[f"{P}_alt_running"]
            and st.session_state[f"{P}_relay_on"]
            and not st.session_state[f"{P}_hold"]
            and not st.session_state[f"{P}_sine_running"]
        ):
            if state == 0:
                _send_all(0.0)
            else:
                _send_all(voltage)
            state = 1 - state
            time.sleep(1.0 / (2.0 * max(freq, 0.01)))
    except Exception as e:
        st.session_state[f"{P}_alt_running"] = False
        logger.exception(f"交替下发异常: {e}")
        _set_feedback(f"交替下发异常: {e}", "error")


def _square_loop(voltage_a: float, voltage_b: float, freq: float, dt: float) -> None:
    """A/B 方波电压持续下发线程。使用当前单元选择 (all_mode / channels)。"""
    ctrl = st.session_state[f"{P}_controller"]
    if ctrl is None:
        return
    try:
        half_period = 1.0 / (2.0 * max(freq, 0.01))
        while (
            st.session_state[f"{P}_square_running"]
            and st.session_state[f"{P}_relay_on"]
            and not st.session_state[f"{P}_sine_running"]
        ):
            _send_channels(voltage_a)
            time.sleep(half_period)
            if not st.session_state[f"{P}_square_running"]:
                break
            _send_channels(voltage_b)
            time.sleep(half_period)
    except Exception as e:
        st.session_state[f"{P}_square_running"] = False
        logger.exception(f"方波下发异常: {e}")
        _set_feedback(f"方波下发异常: {e}", "error")


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
# Group Control: CSV Wiring Groups (1300-5-enriched.csv)
# =============================================================================

def _gc_build_groups(df: pd.DataFrame | None = None) -> dict[str, GroupDef]:
    """Build {group_name: GroupDef} from CSV for group control."""
    if df is None:
        df = _load_csv()
    result: dict[str, GroupDef] = {}
    if df.empty:
        return result
    try:
        for group_name, grp_df in df.groupby("组"):
            channels_by_ip: dict[int, list[ChannelInfo]] = {}
            for _, row in grp_df.iterrows():
                ci = _row_to_channel_info(row)
                channels_by_ip.setdefault(ci.ip_suffix, []).append(ci)
            for ip_s in channels_by_ip:
                channels_by_ip[ip_s].sort(key=lambda ch: ch.payload_position)
            result[group_name] = GroupDef(name=group_name, channels_by_ip=channels_by_ip)
    except Exception as e:
        logger.warning(f"Failed to build group index from CSV: {e}")
    return result


# =============================================================================
# Group Control Session State
# =============================================================================

def _init_gc_state() -> None:
    """初始化分组控制 (Group Control) 的 session_state 变量。"""
    gc = f"{P}_gc"

    groups = _gc_build_groups()
    group_names = sorted(groups.keys())

    st.session_state.setdefault(f"{gc}_groups", groups)
    st.session_state.setdefault(f"{gc}_group_names", group_names)
    st.session_state.setdefault(
        f"{gc}_selected_group", group_names[0] if group_names else None
    )

    st.session_state.setdefault(f"{gc}_controllers", {})
    st.session_state.setdefault(f"{gc}_connected", False)
    st.session_state.setdefault(f"{gc}_relay_on", False)
    st.session_state.setdefault(f"{gc}_connection_error", "")

    st.session_state.setdefault(f"{gc}_voltage", 0.0)
    st.session_state.setdefault(f"{gc}_selected_channels", [])

    st.session_state.setdefault(f"{gc}_feedback", "")
    st.session_state.setdefault(f"{gc}_feedback_type", "")


# Group Control feedback aliases (delegated to generic helpers with prefix="gc")
_gc_show_feedback = lambda: _show_feedback(prefix="gc")  # type: ignore[assignment]
_gc_set_feedback = lambda msg, tp="info": _set_feedback(msg, tp, prefix="gc")  # type: ignore[assignment]


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
    controllers: dict[int, R50Controller] = {}
    connected_count = 0
    total_count = len(group_def.channels_by_ip)
    errors: list[str] = []

    for ip_suffix in sorted(group_def.channels_by_ip.keys()):
        ip = f"192.168.0.{ip_suffix}"
        port = 10000 + ip_suffix
        try:
            if simulate:
                ctrl = SimulatedR50Controller(controller_id=ip_suffix, ip=ip, port=port)
                ctrl.open()
            else:
                ctrl = R50Controller(controller_id=ip_suffix, ip=ip, port=port)
                if not ctrl.open():
                    errors.append(f"{ip}:{port} 打开失败")
                    continue
            controllers[ip_suffix] = ctrl
            connected_count += 1
            logger.info(f"Group control connected: {ip}:{port}")
        except Exception as e:
            errors.append(f"{ip}:{port} {e}")
            logger.exception(f"Group control connect failed for {ip}:{port}: {e}")

    st.session_state[f"{gc}_controllers"] = controllers
    st.session_state[f"{gc}_connected"] = connected_count > 0
    st.session_state[f"{gc}_relay_on"] = False

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
    """断开所选组的所有控制器连接。"""
    gc = f"{P}_gc"
    controllers: dict[int, R50Controller] = st.session_state.get(f"{gc}_controllers", {})

    for ip_suffix, ctrl in controllers.items():
        try:
            if st.session_state.get(f"{gc}_relay_on", False):
                ctrl.set_relay(False)
            ctrl.close()
        except Exception as e:
            logger.warning(f"Group control disconnect warning for ip={ip_suffix}: {e}")

    st.session_state[f"{gc}_controllers"] = {}
    st.session_state[f"{gc}_connected"] = False
    st.session_state[f"{gc}_relay_on"] = False
    st.session_state[f"{gc}_connection_error"] = ""
    _gc_set_feedback("已断开所有控制器 (已先下电)", "info")
    logger.info("Group control disconnected all controllers")
    _debug_add_op("disconnect", "group", "")


# =============================================================================
# Group Control: Relay / Voltage
# =============================================================================

def _gc_set_relay(on: bool) -> None:
    """所有控制器继电器上下电。"""
    gc = f"{P}_gc"
    controllers: dict[int, R50Controller] = st.session_state.get(f"{gc}_controllers", {})

    if not controllers:
        _gc_set_feedback("设备未连接", "error")
        return

    success_count = 0
    error_count = 0
    for ip_suffix, ctrl in controllers.items():
        try:
            if ctrl.set_relay(on):
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
        logger.info(f"Group control relay {'ON' if on else 'OFF'}: {success_count} controllers")
        _debug_add_op("relay_on" if on else "relay_off", f"group, {success_count} ok", "")
    else:
        _gc_set_feedback(
            f"⚠️ 继电器操作: {success_count} 成功, {error_count} 失败",
            "warning",
        )


def _gc_apply_voltage(all_channels: bool = False) -> None:
    """向所选组下发电压。all_channels=True 时忽略 selected_channels 过滤。"""
    gc = f"{P}_gc"
    controllers: dict[int, R50Controller] = st.session_state.get(f"{gc}_controllers", {})
    if not controllers:
        _gc_set_feedback("设备未连接", "error")
        return
    if not st.session_state.get(f"{gc}_relay_on", False):
        _gc_set_feedback("⚠️ 请先继电器上电后再下发电压", "error")
        return

    voltage = float(st.session_state.get(f"{gc}_voltage", 0.0))
    clipped = _clip_voltage(voltage)

    selected_group = st.session_state.get(f"{gc}_selected_group")
    groups = st.session_state.get(f"{gc}_groups", {})
    if not selected_group or selected_group not in groups:
        _gc_set_feedback("未选择有效组别", "error")
        return

    group_def = groups[selected_group]
    selected_channels = st.session_state.get(f"{gc}_selected_channels", [])
    sent_count = 0
    error_count = 0

    for ip_suffix, channel_list in group_def.channels_by_ip.items():
        ctrl = controllers.get(ip_suffix)
        if ctrl is None:
            continue
        for ch_info in channel_list:
            pp = ch_info.payload_position
            if not all_channels and selected_channels and pp not in selected_channels:
                continue
            try:
                ctrl.set_channel_voltage(pp - 1, clipped)
                sent_count += 1
            except Exception as e:
                error_count += 1
                logger.exception(f"Group control voltage failed: ip={ip_suffix} ch={pp}: {e}")

    label = "全部 " if all_channels else ""
    if error_count == 0:
        _gc_set_feedback(
            f"✅ 已向 {selected_group} {label}{sent_count} 个通道下发 {clipped:.1f} V",
            "success",
        )
        _debug_add_op("set_voltage", f"group {selected_group} {label}{sent_count}ch {clipped:.1f}V", "")
    else:
        _gc_set_feedback(
            f"⚠️ 下发完成: {sent_count} 成功, {error_count} 失败",
            "warning",
        )


# =============================================================================
# Group Control: Batch Power On/Off (with Ping Test)
# =============================================================================


def _gc_batch_power_on() -> None:
    """批量上电: 先 ping 测试组内所有控制器, 再继电器上电。"""
    gc = f"{P}_gc"
    controllers: dict[int, R50Controller] = st.session_state.get(
        f"{gc}_controllers", {}
    )
    if not controllers:
        _gc_set_feedback("设备未连接", "error")
        return

    reachable = []
    unreachable = []
    for ip_suffix in sorted(controllers.keys()):
        ip = f"192.168.0.{ip_suffix}"
        if _ping_reachable(ip, timeout=1.0):
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
            if ctrl.set_relay(True):
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
# Group Control: Main Tab Renderer
# =============================================================================

def render_tab_single_group() -> None:
    """单组控制 Tab: 按 wiring map 组别选择控制器并下发电压。"""
    st.title("🧩 单组控制")
    st.caption("按 wiring map 组别同时控制多个控制器")

    gc = f"{P}_gc"

    # 检查连接模式
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
    group_names = st.session_state.get(f"{gc}_group_names", [])

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
            total_channels = group_def.total_channels
            st.caption(
                f"**{selected}** — {len(group_def.channels_by_ip)} 个控制器, "
                f"{total_channels} 个通道"
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

    group_def = groups.get(selected)  # type: ignore[assignment]
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
                nid = ch_info.needle_id
                label = ch_info.physical_label
                desc = f"ch{pp}"
                if nid:
                    desc += f" 针脚#{nid}"
                if label:
                    desc += f" ({label})"
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
# 单单元控制 Tab
# =============================================================================


def _build_all_units() -> list[ChannelInfo]:
    """Build flat list of all physical units from CSV (for 单单元控制 Tab)."""
    df = _load_csv()
    if df.empty:
        return []
    seen: set[tuple[int, int]] = set()
    all_units: list[ChannelInfo] = []
    try:
        for _, row in df.iterrows():
            ci = _row_to_channel_info(row)
            key = (ci.ip_suffix, ci.payload_position)
            if key in seen:
                continue
            seen.add(key)
            all_units.append(ci)
    except Exception as e:
        logger.warning(f"Failed to build all units index: {e}")
        all_units.clear()
        return all_units

    all_units.sort(key=lambda u: (u.group, u.ip_suffix, u.payload_position))
    return all_units


def _channel_info_to_dict(ci: ChannelInfo) -> dict:
    """Convert ChannelInfo to display dict for DataFrame."""
    return {
        "控制器 IP": ci.ip,
        "通道号": ci.payload_position,
        "组别": ci.group,
        "针脚 ID": ci.needle_id,
        "物理标签": ci.physical_label,
        "物理位置": ci.physical_position,
    }


INFO_DISPLAY_COLS = ["控制器 IP", "通道号", "组别", "针脚 ID", "物理标签", "物理位置"]


def _apply_joint(units: list[ChannelInfo], voltage: float) -> None:
    """Joint 模式: 构建 flat 数组, 通过 MicroDM 下发。"""
    clipped = _clip_voltage(voltage)
    dm = st.session_state.get(f"{P}_jc_dm")
    if dm is None:
        st.error("MicroDM 实例不可用")
        return
    pos_to_hw = st.session_state.get(f"{P}_jc_pos_to_hw", {})
    ip_to_ctrl = st.session_state.get(f"{P}_jc_ip_to_controller_idx", {})
    dm_num = st.session_state.get(f"{P}_jc_dm_num", 0)
    if dm_num <= 0:
        st.error("DM flat array 构建失败")
        return
    flat = np.zeros(dm_num, dtype=np.float64)
    success = 0
    for u in units:
        if u.physical_position not in pos_to_hw:
            continue
        ip_suffix, payload_pos = pos_to_hw[u.physical_position]
        ctrl_idx = ip_to_ctrl.get(ip_suffix)
        if ctrl_idx is None:
            continue
        flat_idx = ctrl_idx * 50 + (payload_pos - 1)
        if flat_idx >= dm_num:
            continue
        flat[flat_idx] = clipped
        success += 1
    if success == 0:
        st.warning("没有可下发的单元")
        return
    try:
        dm.send_voltages(flat)
        st.success(f"✅ 已向 {success} 个单元下发 {clipped:.1f} V")
        _debug_add_op("set_voltage", f"single_unit joint {success}ch {clipped:.1f}V", "all")
    except Exception as e:
        st.error(f"下发失败: {e}")
        logger.exception(f"Single unit apply (joint mode) failed: {e}")


def _apply_single(units: list[ChannelInfo], voltage: float) -> tuple[int, int, set[str]]:
    """Single 模式: 通过已连接控制器逐个下发, 返回 (成功数, 失败数, IP 集合)。"""
    ctrl = st.session_state.get(f"{P}_controller")
    if ctrl is None:
        return 0, 0, set()
    clipped = _clip_voltage(voltage)
    success = 0
    errors = 0
    ips: set[str] = set()
    for u in units:
        try:
            ctrl.set_channel_voltage(u.payload_position - 1, clipped)
            ips.add(u.ip)
            success += 1
        except Exception as e:
            errors += 1
            logger.exception(f"Single unit apply (single mode) failed: {e}")
    return success, errors, ips


def _apply_group(units: list[ChannelInfo], voltage: float) -> tuple[int, int, set[str]]:
    """Group 模式: 通过各自控制器逐个下发, 返回 (成功数, 失败数, IP 集合)。"""
    controllers = st.session_state.get(f"{P}_gc_controllers", {})
    if not controllers:
        return 0, 0, set()
    clipped = _clip_voltage(voltage)
    success = 0
    errors = 0
    ips: set[str] = set()
    for u in units:
        ctrl = controllers.get(u.ip_suffix)
        if ctrl is None:
            continue
        try:
            ctrl.set_channel_voltage(u.payload_position - 1, clipped)
            ips.add(u.ip)
            success += 1
        except Exception as e:
            errors += 1
            logger.exception(f"Single unit apply (group mode) failed: {e}")
    return success, errors, ips


def render_tab_single_unit() -> None:
    """单单元控制 Tab: 跨控制器选择个別物理单元并下发电压."""
    st.title("💠 单单元控制")
    st.caption("从 1300-5 映射表中选择单个物理单元并设置电压 (支持跨控制器)")

    # 构建全量单元列表 (cached)
    if "r50c_single_unit_list" not in st.session_state:
        st.session_state["r50c_single_unit_list"] = _build_all_units()
    all_units: list[ChannelInfo] = st.session_state["r50c_single_unit_list"]

    if not all_units:
        st.warning("⚠️ 1300-5-enriched.csv 加载失败或无有效物理单元数据")
        return

    # ---- 分组 ----
    group_names = sorted(set(u.group for u in all_units if u.group))
    selected_group = st.selectbox(
        "按组别筛选", ["全部"] + group_names,
        key="r50c_su_group_filter",
    )

    filtered = all_units
    if selected_group != "全部":
        filtered = [u for u in filtered if u.group == selected_group]

    # ---- IP 筛选 ----
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

    # 搜索
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

    # ---- 选择显示 ----
    st.markdown(f"**匹配 {len(filtered)} 个单元**")
    df_display = pd.DataFrame([_channel_info_to_dict(u) for u in filtered])

    with st.container(border=True):
        selected_indices = st.dataframe(
            df_display[INFO_DISPLAY_COLS],
            use_container_width=True,
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

    # ---- 电压下发 ----
    st.divider()
    st.markdown(f"**已选 {len(selected_units)} 个单元**")
    df_sel = pd.DataFrame([_channel_info_to_dict(u) for u in selected_units])
    st.dataframe(df_sel[INFO_DISPLAY_COLS[:4]], use_container_width=True, hide_index=True)

    voltage = st.number_input(
        "电压 (V)",
        min_value=st.session_state[f"{P}_vmin"],
        max_value=st.session_state[f"{P}_vmax"],
        value=0.0, step=1.0, format="%.1f",
        key="r50c_su_voltage",
    )

    # 检查连接状态并发送
    mode = st.session_state.get(f"{P}_connection_mode", "single")
    jc_connected = st.session_state.get(f"{P}_jc_connected", False)
    single_connected = st.session_state.get(f"{P}_connected", False)
    gc_connected = st.session_state.get(f"{P}_gc_connected", False)

    has_connection = jc_connected or single_connected or gc_connected
    if not has_connection:
        st.warning("⚠️ 请先在侧边栏连接控制器")
        return

    # 检查继电器
    relay_key = {"single": f"{P}_relay_on", "joint": f"{P}_jc_relay_on", "group": f"{P}_gc_relay_on"}
    conn_key = {"single": single_connected, "joint": jc_connected, "group": gc_connected}
    relay_ok = st.session_state.get(relay_key.get(mode, ""), False) if conn_key.get(mode, False) else False
    if not relay_ok:
        st.warning("⚠️ 继电器未上电, 请先在侧边栏上电")
        return

    # 单控制器模式: 限制到当前 IP (仿真模式下跳过)
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
        "⚡ 下发电压到所选单元", type="primary", use_container_width=True,
        key="r50c_su_apply",
    ):
        if mode == "joint" and jc_connected:
            _apply_joint(selected_units, voltage)

        elif mode == "single" and single_connected:
            success, errors, ips = _apply_single(selected_units, voltage)
            if success:
                msg = f"✅ 已向 {success} 个单元下发 {_clip_voltage(voltage):.1f} V"
                msg += f" ({errors} 失败)" if errors else ""
                st.success(msg)
                _debug_add_op("set_voltage", f"single_unit single {success}ch {_clip_voltage(voltage):.1f}V", ", ".join(sorted(ips)))

        elif mode == "group" and gc_connected:
            success, errors, ips = _apply_group(selected_units, voltage)
            if success:
                msg = f"✅ 已向 {success} 个单元下发 {_clip_voltage(voltage):.1f} V"
                msg += f" ({errors} 失败)" if errors else ""
                st.success(msg)
                _debug_add_op("set_voltage", f"single_unit group {success}ch {_clip_voltage(voltage):.1f}V", ", ".join(sorted(ips)))

        # 更新电压跟踪
        if mode == "single" and single_connected:
            clipped = _clip_voltage(voltage)
            for u in selected_units:
                ch = u.payload_position - 1
                if 0 <= ch < SINGLE_CHANNELS:
                    st.session_state[f"{P}_current_voltages"][ch] = clipped
            st.rerun()


# =============================================================================
# Sidebar: 连接配置 (统一入口)
# =============================================================================


def _sidebar_connection_config() -> None:
    """Sidebar 连接配置: 三种连接模式的统一入口。"""
    # ---- 连接模式选择 ----
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

        # ---- 电压安全范围 (通用) ----
        with st.sidebar:
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

            # ---- 调试面板 (TCP 调试客户端 + 操作日志) ----
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
    """Sidebar 联合控制连接配置(批量Ping/连接/上下电)。"""
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
        group_names = st.session_state.get(f"{gc}_group_names", [])
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
                total_channels = group_def.total_channels
                st.caption(
                    f"**{selected}** — {len(group_def.channels_by_ip)} 个控制器, "
                    f"{total_channels} 个通道"
                )
                rows = []
                for ip_suffix in sorted(group_def.channels_by_ip.keys()):
                    for ch_info in group_def.channels_by_ip[ip_suffix]:
                        rows.append(_channel_info_to_dict(ch_info))
                if rows:
                    with st.expander("📋 通道详情", expanded=False):
                        st.dataframe(
                            pd.DataFrame(rows), use_container_width=True, hide_index=True,
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


def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="R50 控制器控制面板",
        page_icon="🔌",
        layout="wide",
    )

    _initialize_state()

    # Sidebar: 统一连接配置 (三种模式)
    _sidebar_connection_config()

    # 主窗口: 四个控制粒度 Tab
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


def render_tab_single_controller() -> None:
    """单控制器控制 Tab: 完整 50 通道电压控制。"""
    st.title("🔌 单控制器控制")
    st.caption("单个 R50Controller (50 通道) 电压控制 | 单元选择与下发方式分离")

    # 检查连接模式
    if st.session_state.get(f"{P}_connection_mode") != "single":
        st.info("💡 当前未在「单控制器」连接模式。请在侧边栏切换到「单控制器」并连接。")
        return
    if not st.session_state.get(f"{P}_connected", False):
        st.info("💡 请先在侧边栏「单控制器」连接模式下连接控制器。")
        return

    _show_feedback()

    # ===== 1. 单元选择 (沿用现有逻辑) =====
    with st.container(border=True):
        st.markdown("##### 单元选择")
        st.caption("选择下发目标单元，所有下发方式共用此选择 (除「全部清 0」强制所有通道外)")

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

            # 显示已选单元的针脚映射信息
            if st.session_state[f"{P}_channels"]:
                _infos = []
                for _ch in st.session_state[f"{P}_channels"]:
                    _ci = _get_channel_info(int(_ch))
                    _infos.append(_channel_label(int(_ch)) if _ci else f"ch{_ch}: 无映射")
                st.caption("针脚映射: " + " ｜ ".join(_infos))

    # ===== 2. 电压下发方式 (四种模式) =====
    with st.container(border=True):
        st.markdown("##### 电压下发方式")
        st.caption("选择下发模式后配置对应参数并执行")

        send_mode = st.radio(
            "选择模式",
            options=["clear", "fixed", "sine", "square"],
            format_func={
                "clear": "🧹 全部清 0",
                "fixed": "⚡ 下发固定电压",
                "sine": "〰️ 持续正弦电压",
                "square": "⬆️⬇️ 持续方波电压",
            }.get,
            key=f"{P}_send_mode",
            horizontal=False,
        )

        # ========== 模式 1: 全部清 0 ==========
        if send_mode == "clear":
            st.caption("将所有 50 个单元电压设置为 0V (不依赖单元选择)")
            if st.button(
                "🧹 全部清 0", type="primary", width='stretch',
                disabled=not st.session_state[f"{P}_connected"],
                key=f"{P}_clear_all_btn",
            ):
                if _require_relay_on():
                    _send_all(0.0)
                    st.session_state[f"{P}_voltage"] = 0.0
                    _set_feedback("✅ 所有单元已清零", "success")
                    st.rerun()

        # ========== 模式 2: 下发固定电压 ==========
        elif send_mode == "fixed":
            voltage = st.number_input(
                "电压 (V)",
                min_value=st.session_state[f"{P}_vmin"],
                max_value=st.session_state[f"{P}_vmax"],
                value=st.session_state[f"{P}_voltage"],
                step=1.0, format="%.1f",
                key=f"{P}_voltage_input",
            )
            st.session_state[f"{P}_voltage"] = float(voltage)

            if st.button(
                "⚡ 下发固定电压", type="primary", width='stretch',
                disabled=not st.session_state[f"{P}_connected"],
                key=f"{P}_fixed_send_btn",
            ):
                if _require_relay_on():
                    if not st.session_state[f"{P}_all_mode"] and not st.session_state[f"{P}_channels"]:
                        _set_feedback("未选择任何指定单元", "warning")
                    else:
                        try:
                            _send_channels(voltage)
                            _send_success_feedback(voltage)
                        except Exception as e:
                            _set_feedback(f"发送失败: {e}", "error")
                    st.rerun()

        # ========== 模式 3: 持续正弦电压 ==========
        elif send_mode == "sine":
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

            if not st.session_state[f"{P}_sine_running"]:
                if st.button(
                    "▶ 开始正弦下发", type="primary", width='stretch',
                    disabled=(
                        not st.session_state[f"{P}_connected"]
                        or st.session_state[f"{P}_square_running"]
                    ),
                    key=f"{P}_sine_start",
                ):
                    if _require_relay_on():
                        st.session_state[f"{P}_square_running"] = False
                        st.session_state[f"{P}_sine_running"] = True
                        threading.Thread(
                            target=_sine_loop,
                            args=(amp, offset, freq, 0.05),
                            daemon=True,
                        ).start()
                        _set_feedback(
                            f"正弦下发中: amp={amp}V, offset={offset}V, f={freq}Hz",
                            "success",
                        )
                        st.rerun()
            else:
                if st.button(
                    "⏹ 停止正弦", type="primary", width='stretch',
                    key=f"{P}_sine_stop",
                ):
                    st.session_state[f"{P}_sine_running"] = False
                    _set_feedback("正弦下发已停止", "info")
                    st.rerun()

        # ========== 模式 4: 持续方波电压 (A/B) ==========
        elif send_mode == "square":
            col_a, col_b, col_f = st.columns(3)
            with col_a:
                sq_a = st.number_input(
                    "电压 A (V)", min_value=st.session_state[f"{P}_vmin"],
                    max_value=st.session_state[f"{P}_vmax"],
                    value=st.session_state[f"{P}_square_voltage_a"], step=1.0, format="%.1f",
                    key=f"{P}_square_a_input",
                )
                st.session_state[f"{P}_square_voltage_a"] = float(sq_a)
            with col_b:
                sq_b = st.number_input(
                    "电压 B (V)", min_value=st.session_state[f"{P}_vmin"],
                    max_value=st.session_state[f"{P}_vmax"],
                    value=st.session_state[f"{P}_square_voltage_b"], step=1.0, format="%.1f",
                    key=f"{P}_square_b_input",
                )
                st.session_state[f"{P}_square_voltage_b"] = float(sq_b)
            with col_f:
                sq_f = st.number_input(
                    "频率 (Hz)", min_value=0.01, max_value=50.0,
                    value=st.session_state[f"{P}_square_freq"], step=0.05, format="%.2f",
                    key=f"{P}_square_freq_input",
                )
                st.session_state[f"{P}_square_freq"] = float(sq_f)

            if not st.session_state[f"{P}_square_running"]:
                if st.button(
                    "▶ 开始方波下发", type="primary", width='stretch',
                    disabled=(
                        not st.session_state[f"{P}_connected"]
                        or st.session_state[f"{P}_sine_running"]
                    ),
                    key=f"{P}_square_start",
                ):
                    if _require_relay_on():
                        st.session_state[f"{P}_sine_running"] = False
                        st.session_state[f"{P}_square_running"] = True
                        threading.Thread(
                            target=_square_loop,
                            args=(sq_a, sq_b, sq_f, 0.01),
                            daemon=True,
                        ).start()
                        _set_feedback(
                            f"方波下发中: A={sq_a}V, B={sq_b}V, f={sq_f}Hz",
                            "success",
                        )
                        st.rerun()
            else:
                if st.button(
                    "⏹ 停止方波", type="primary", width='stretch',
                    key=f"{P}_square_stop",
                ):
                    st.session_state[f"{P}_square_running"] = False
                    _set_feedback("方波下发已停止", "info")
                    st.rerun()

    # ---- 各单元当前电压 ----
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

    # ---- 调试日志 (指令 / 下发包) ----
    if st.session_state[f"{P}_debug"]:
        st.divider()
        st.markdown("##### 调试日志 (指令 / 下发包)")
        log_lines = list(st.session_state[f"{P}_debug_log"])
        st.code("\n".join(log_lines) if log_lines else "(无记录)", language="text")

    if (
        st.session_state[f"{P}_sine_running"]
        or st.session_state[f"{P}_square_running"]
    ):
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


if __name__ == "__main__":
    main()
