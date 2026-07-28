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
import json
import socket
import subprocess
import threading
import time
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
    WiringMap,
)
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT

# wiring_map.json 路径 (单元物理信息对照)
WIRING_MAP_PATH = PROJECT_ROOT / "libs" / "micro_drive1300" / "wiring_map.json"
# (ip_suffix, payload_position) → {"group": str, "needle_id": int, "label": str}
_WIRING_INDEX: dict[tuple[int, int], dict] = {}

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
# Joint Control Constants (36×36 矩阵联合控制)
# =============================================================================

GRID_SIZE = 36  # 36×36 压电陶瓷矩阵
DM_NUM_WIRING = 1296  # wiring_map 中的物理位置数 (36×36)


# =============================================================================
# Wiring Map Index (针脚 ↔ 组别 映射)
# =============================================================================

def _build_wiring_index() -> dict[tuple[int, int], dict]:
    """从 wiring_map.json 构建 (ip_suffix, payload_position) → 针脚信息的索引。

    用于在电压下发时提示用户当前通道对应的组别和针脚号。
    payload_position 在 JSON 中是 1-based, 索引 key 也保持 1-based。
    """
    index: dict[tuple[int, int], dict] = {}
    if not WIRING_MAP_PATH.exists():
        return index
    try:
        with open(WIRING_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_key, group in data.get("groups", {}).items():
            group_name = group.get("name", group_key)
            for ch in group.get("channels", []):
                ip_suffix = ch.get("ip_suffix")
                payload_pos = ch.get("payload_position")
                needle_id = ch.get("needle_id")
                if ip_suffix is None or payload_pos is None:
                    continue
                index[(int(ip_suffix), int(payload_pos))] = {
                    "group": group_name,
                    "needle_id": needle_id,
                    "label": ch.get("physical_label", ""),
                }
    except Exception as e:
        logger.warning(f"Failed to build wiring index: {e}")
    return index


def _get_needle_info(channel: int) -> str:
    """根据当前控制器 IP 和通道号 (0-based) 返回针脚信息字符串。

    返回格式示例: "一组 针脚#277 (3-3-1)" 或 "" (无映射时)。
    """
    ip = st.session_state.get(f"{P}_ip", "").strip()
    if not ip or not _WIRING_INDEX:
        return ""
    try:
        ip_suffix = int(ip.split(".")[-1])
    except (ValueError, IndexError):
        return ""
    payload_pos = channel + 1  # UI channel 0-based → payload_position 1-based
    info = _WIRING_INDEX.get((ip_suffix, payload_pos))
    if info is None:
        return ""
    needle = info["needle_id"]
    label = info["label"]
    group = info["group"]
    return f"{group} 针脚#{needle} ({label})" if needle else f"{group} ({label})"


def _channel_label(ch: int) -> str:
    """格式化通道号, 附加针脚映射信息 (供 multiselect format_func 使用)。"""
    info = _get_needle_info(ch)
    return f"{ch} | {info}" if info else str(ch)


# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化所有 session_state 变量。"""

    # ---- Wiring map 索引 (仅构建一次) ----
    global _WIRING_INDEX
    if not _WIRING_INDEX:
        _WIRING_INDEX = _build_wiring_index()

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
# Joint Control: Wiring Map Index
# =============================================================================

def _jc_build_wiring_index() -> dict[int, tuple[int, int]]:
    """从 wiring_map.json 构建 physical_position → (ip_suffix, payload_position) 映射。

    physical_position (1-1296) 对应 36×36 矩阵位置: row=(pos-1)//36, col=(pos-1)%36
    """
    pos_to_hw: dict[int, tuple[int, int]] = {}
    if not WIRING_MAP_PATH.exists():
        return pos_to_hw
    try:
        wm = WiringMap.from_file(WIRING_MAP_PATH)
        if wm is None:
            return pos_to_hw
        for ch in wm.all_channels:
            if ch.is_valid and ch.physical_position is not None and ch.ip_suffix is not None and ch.payload_position is not None:
                pos_to_hw[ch.physical_position] = (ch.ip_suffix, ch.payload_position)
    except Exception as e:
        logger.warning(f"Failed to build joint control wiring index: {e}")
    return pos_to_hw


def _jc_build_ip_index(wm: WiringMap) -> dict[int, int]:
    """构建 ip_suffix → controller_index 映射 (flat array 顺序)。"""
    ips = wm.unique_ips
    ip_to_idx: dict[int, int] = {}
    for idx, ip_str in enumerate(ips):
        try:
            suffix = int(ip_str.split(".")[-1])
            ip_to_idx[suffix] = idx
        except (ValueError, IndexError):
            pass
    return ip_to_idx


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
    """连接 MicroDM (所有控制器)。"""
    jc = f"{P}_jc"
    try:
        dm = MicroDM(use_wiring_map=True)  # type: ignore[call-arg]
        dm.open()
        st.session_state[f"{jc}_dm"] = dm
        st.session_state[f"{jc}_connected"] = True
        st.session_state[f"{jc}_relay_on"] = False
        st.session_state[f"{jc}_connection_error"] = ""
        st.session_state[f"{jc}_controller_count"] = len(dm._controllers)  # type: ignore

        # 构建索引
        pos_to_hw = _jc_build_wiring_index()
        wm = WiringMap.from_file(WIRING_MAP_PATH)
        if wm is None:
            st.session_state[f"{jc}_connection_error"] = "wiring_map.json 加载失败"
            st.session_state[f"{jc}_connected"] = False
            return
        ip_to_ctrl_idx = _jc_build_ip_index(wm)
        st.session_state[f"{jc}_pos_to_hw"] = pos_to_hw
        st.session_state[f"{jc}_ip_to_controller_idx"] = ip_to_ctrl_idx
        st.session_state[f"{jc}_sorted_ips"] = wm.unique_ips
        st.session_state[f"{jc}_dm_num"] = dm.DM_Num
        st.session_state[f"{jc}_matrix_init"] = True

        # 读取当前电压
        matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl_idx)
        st.session_state[f"{jc}_matrix"] = matrix

        st.session_state[f"{jc}_feedback"] = (
            f"✅ 已连接 MicroDM: {st.session_state[f'{jc}_controller_count']} 个控制器"
        )
        st.session_state[f"{jc}_feedback_type"] = "success"
        logger.info(
            f"MicroDM connected: {st.session_state[f'{jc}_controller_count']} controllers"
        )
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
    if isinstance(_dm, MicroDM):
        try:
            if st.session_state[f"{jc}_relay_on"]:
                _dm.set_relay_state(False)  # type: ignore[attr-defined]
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


# =============================================================================
# Joint Control: Relay / Apply
# =============================================================================

def _jc_set_relay(on: bool) -> None:
    """所有控制器继电器上下电。"""
    jc = f"{P}_jc"
    _dm = st.session_state[f"{jc}_dm"]
    if not isinstance(_dm, MicroDM):
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        _dm.set_relay_state(on)  # type: ignore[attr-defined]
        st.session_state[f"{jc}_relay_on"] = on
        if on:
            st.session_state[f"{jc}_feedback"] = "✅ 所有控制器继电器已上电 (输出接通)"
            st.session_state[f"{jc}_feedback_type"] = "success"
            logger.info("MicroDM relay ON (all controllers)")
        else:
            st.session_state[f"{jc}_feedback"] = "⏻ 所有控制器继电器已下电 (输出断开)"
            st.session_state[f"{jc}_feedback_type"] = "info"
            logger.info("MicroDM relay OFF (all controllers)")
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
            f"✅ 已下发 36×36 矩阵电压 (非零通道: {non_zero}/{DM_NUM_WIRING})"
        )
        st.session_state[f"{jc}_feedback_type"] = "success"
        logger.info(f"MicroDM voltage applied: {non_zero} non-zero channels")
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
    dm: MicroDM | None = st.session_state[f"{jc}_dm"]
    if dm is None:
        return
    pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
    ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
    matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl)
    st.session_state[f"{jc}_matrix"] = matrix
    st.session_state[f"{jc}_feedback"] = "已从硬件刷新电压矩阵"
    st.session_state[f"{jc}_feedback_type"] = "info"


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
    st.image(img, caption="36×36 电压分布 (蓝色低 · 红色高)", use_container_width=True)


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
                use_container_width=True,
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
            st.bar_chart(df_row, height=200, use_container_width=True)
        with tab_c:
            df_col = pd.DataFrame({"列号": list(range(1, GRID_SIZE + 1)), "均值 (V)": col_means}).set_index("列号")
            st.bar_chart(df_col, height=200, use_container_width=True)


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
        st.metric("非零通道", f"{non_zero_count}/{DM_NUM_WIRING}")


def _jc_show_feedback() -> None:
    """显示联合控制反馈信息。"""
    jc = f"{P}_jc"
    message = st.session_state.get(f"{jc}_feedback", "")
    msg_type = st.session_state.get(f"{jc}_feedback_type", "")
    if message:
        if msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
        elif msg_type == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state[f"{jc}_feedback"] = ""
        st.session_state[f"{jc}_feedback_type"] = ""


# =============================================================================
# Joint Control: Main Tab Renderer
# =============================================================================

def render_tab_joint_control() -> None:
    """渲染联合控制 Tab (36×36 矩阵)。"""
    jc = f"{P}_jc"
    matrix: np.ndarray = st.session_state[f"{jc}_matrix"]
    connected = st.session_state[f"{jc}_connected"]
    relay_on = st.session_state[f"{jc}_relay_on"]

    _jc_show_feedback()

    # ===== 连接面板 =====
    with st.container(border=True):
        col_status, col_actions = st.columns([1, 1])
        with col_status:
            st.markdown("##### 连接状态")
            if connected:
                n_ctrl = st.session_state[f"{jc}_controller_count"]
                n_ips = len(st.session_state[f"{jc}_sorted_ips"])
                st.success(f"✅ MicroDM 已连接 ({n_ctrl} 控制器, {n_ips} IP)")
                if relay_on:
                    st.success("⚡ 继电器已上电 (输出接通)")
                else:
                    st.warning("⏻ 继电器已下电 (输出断开)")
            else:
                st.error("❌ MicroDM 未连接")
            if st.session_state[f"{jc}_connection_error"]:
                st.caption(f"错误: {st.session_state[f'{jc}_connection_error']}")

        with col_actions:
            st.markdown("##### 操作")
            if not connected:
                if st.button("🔌 连接 MicroDM", type="primary", use_container_width=True, key=f"{jc}_connect_btn"):
                    with st.spinner("连接所有控制器..."):
                        _jc_connect()
                    st.rerun()
            else:
                col_r1, col_r2, col_disc = st.columns(3)
                with col_r1:
                    if st.button("⚡ 上电", type="primary", use_container_width=True,
                                 disabled=relay_on, key=f"{jc}_relay_on_btn"):
                        _jc_set_relay(True)
                        st.rerun()
                with col_r2:
                    if st.button("⏻ 下电", use_container_width=True,
                                 disabled=not relay_on, key=f"{jc}_relay_off_btn"):
                        _jc_set_relay(False)
                        st.rerun()
                with col_disc:
                    if st.button("⏏ 断开", use_container_width=True, key=f"{jc}_disconnect_btn"):
                        _jc_disconnect()
                        st.rerun()

    if not connected:
        st.info("💡 请先连接 MicroDM 以查看和编辑 36×36 矩阵。")
        return

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
            if st.button("填充全部", use_container_width=True, key=f"{jc}_fill_all_btn"):
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
            if st.button("设置单元", use_container_width=True, key=f"{jc}_set_cell_btn"):
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
                if st.button("填充行", use_container_width=True, key=f"{jc}_fill_row_btn"):
                    _jc_fill_row(int(fill_row), fill_v)
                    st.rerun()
            with col_fc:
                fill_col = st.number_input("目标列", 0, GRID_SIZE - 1, 0, 1, key=f"{jc}_fill_col_input")
                if st.button("填充列", use_container_width=True, key=f"{jc}_fill_col_btn"):
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
            if st.button("填充矩形", use_container_width=True, key=f"{jc}_rect_btn"):
                _jc_fill_rect(int(rx1), int(ry1), int(rx2), int(ry2), rect_v)
                st.rerun()

    # ===== 发送/硬件操作 =====
    st.divider()
    st.markdown("##### 硬件操作")
    col_send, col_reset, col_refresh = st.columns(3)
    with col_send:
        if st.button("⚡ 下发全部电压到硬件", type="primary", use_container_width=True,
                     disabled=not connected, key=f"{jc}_apply_btn"):
            if not relay_on:
                st.warning("⚠️ 请先继电器上电")
            else:
                _jc_apply_matrix()
                st.rerun()
    with col_reset:
        if st.button("🔄 清零矩阵", use_container_width=True,
                     key=f"{jc}_reset_btn"):
            _jc_reset_matrix()
            st.rerun()
    with col_refresh:
        if st.button("📡 从硬件刷新", use_container_width=True,
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
            f"36×36 矩阵共 {DM_NUM_WIRING} 个压电陶瓷单元 · "
            f"wiring_map 已映射 {len(pos_to_hw)} 个物理位置 · "
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
# Group Control: Wiring Map Groups
# =============================================================================

def _gc_build_groups() -> dict[str, dict[int, list[dict]]]:
    """wiring_map.json → {group_name: {ip_suffix: [channel_info]}} for group control."""
    groups: dict[str, dict[int, list[dict]]] = {}
    if not WIRING_MAP_PATH.exists():
        return groups
    try:
        with open(WIRING_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_key, group in data.get("groups", {}).items():
            group_name = group.get("name", group_key)
            channels_by_ip: dict[int, list[dict]] = {}
            for ch in group.get("channels", []):
                ip_suffix = ch.get("ip_suffix")
                payload_pos = ch.get("payload_position")
                if ip_suffix is None or payload_pos is None:
                    continue
                ip_suffix = int(ip_suffix)
                channel_info = {
                    "payload_position": int(payload_pos),
                    "needle_id": ch.get("needle_id"),
                    "physical_label": ch.get("physical_label", ""),
                    "physical_position": ch.get("physical_position"),
                }
                channels_by_ip.setdefault(ip_suffix, []).append(channel_info)
            for ip_suffix in channels_by_ip:
                channels_by_ip[ip_suffix].sort(key=lambda c: c["payload_position"])
            groups[group_name] = channels_by_ip
    except Exception as e:
        logger.warning(f"Failed to build group index: {e}")
    return groups


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


# =============================================================================
# Group Control: Feedback
# =============================================================================

def _gc_show_feedback() -> None:
    """显示分组控制反馈信息。"""
    gc = f"{P}_gc"
    message = st.session_state.get(f"{gc}_feedback", "")
    msg_type = st.session_state.get(f"{gc}_feedback_type", "")
    if message:
        if msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
        elif msg_type == "warning":
            st.warning(message)
        else:
            st.info(message)
        st.session_state[f"{gc}_feedback"] = ""
        st.session_state[f"{gc}_feedback_type"] = ""


def _gc_set_feedback(message: str, msg_type: str = "info") -> None:
    """设置分组控制反馈信息。"""
    gc = f"{P}_gc"
    st.session_state[f"{gc}_feedback"] = message
    st.session_state[f"{gc}_feedback_type"] = msg_type


# =============================================================================
# Group Control: Connect / Disconnect
# =============================================================================

def _gc_connect() -> None:
    """连接所选组的所有控制器。"""
    gc = f"{P}_gc"
    selected = st.session_state.get(f"{gc}_selected_group")
    groups = st.session_state.get(f"{gc}_groups", {})

    if not selected or selected not in groups:
        _gc_set_feedback("未选择有效组别", "error")
        return

    channels_by_ip = groups[selected]
    controllers: dict[int, R50Controller] = {}
    connected_count = 0
    total_count = len(channels_by_ip)
    errors: list[str] = []

    for ip_suffix in sorted(channels_by_ip.keys()):
        ip = f"192.168.0.{ip_suffix}"
        port = 10000 + ip_suffix
        try:
            ctrl = R50Controller(controller_id=ip_suffix, ip=ip, port=port)
            if ctrl.open():
                controllers[ip_suffix] = ctrl
                connected_count += 1
                logger.info(f"Group control connected: {ip}:{port}")
            else:
                errors.append(f"{ip}:{port} 打开失败")
        except Exception as e:
            errors.append(f"{ip}:{port} {e}")
            logger.exception(f"Group control connect failed for {ip}:{port}: {e}")

    st.session_state[f"{gc}_controllers"] = controllers
    st.session_state[f"{gc}_connected"] = connected_count > 0
    st.session_state[f"{gc}_relay_on"] = False

    if connected_count == total_count:
        _gc_set_feedback(
            f"✅ 已连接 {selected} 全部 {connected_count} 个控制器",
            "success",
        )
    elif connected_count > 0:
        error_detail = "; ".join(errors) if errors else ""
        _gc_set_feedback(
            f"⚠️ 已连接 {connected_count}/{total_count} 个控制器"
            + (f" ({error_detail})" if error_detail else ""),
            "warning",
        )
    else:
        error_detail = "; ".join(errors) if errors else "未知错误"
        _gc_set_feedback(f"❌ 连接失败: {error_detail}", "error")


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
    else:
        _gc_set_feedback(
            f"⚠️ 继电器操作: {success_count} 成功, {error_count} 失败",
            "warning",
        )


def _gc_apply_voltage() -> None:
    """向所选通道下发电压。"""
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

    channels_by_ip = groups[selected_group]
    selected_channels = st.session_state.get(f"{gc}_selected_channels", [])

    sent_count = 0
    error_count = 0

    for ip_suffix, channel_list in channels_by_ip.items():
        ctrl = controllers.get(ip_suffix)
        if ctrl is None:
            continue
        for ch_info in channel_list:
            pp = ch_info["payload_position"]
            if selected_channels and pp not in selected_channels:
                continue
            try:
                ctrl.set_channel_voltage(pp - 1, clipped)
                sent_count += 1
            except Exception as e:
                error_count += 1
                logger.exception(
                    f"Group control voltage failed: ip={ip_suffix} ch={pp}: {e}"
                )

    if error_count == 0:
        _gc_set_feedback(
            f"✅ 已向 {sent_count} 个通道下发 {clipped:.1f} V ({selected_group})",
            "success",
        )
    else:
        _gc_set_feedback(
            f"⚠️ 下发完成: {sent_count} 成功, {error_count} 失败",
            "warning",
        )


def _gc_apply_all_voltage() -> None:
    """向组内全部通道统一下发电压。"""
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

    channels_by_ip = groups[selected_group]
    sent_count = 0
    error_count = 0

    for ip_suffix, channel_list in channels_by_ip.items():
        ctrl = controllers.get(ip_suffix)
        if ctrl is None:
            continue
        for ch_info in channel_list:
            pp = ch_info["payload_position"]
            try:
                ctrl.set_channel_voltage(pp - 1, clipped)
                sent_count += 1
            except Exception as e:
                error_count += 1
                logger.exception(
                    f"Group control voltage failed: ip={ip_suffix} ch={pp}: {e}"
                )

    if error_count == 0:
        _gc_set_feedback(
            f"✅ 已向 {selected_group} 全部 {sent_count} 个通道下发 {clipped:.1f} V",
            "success",
        )
    else:
        _gc_set_feedback(
            f"⚠️ 下发完成: {sent_count} 成功, {error_count} 失败",
            "warning",
        )


# =============================================================================
# Group Control: Main Tab Renderer
# =============================================================================

def render_tab_group_control() -> None:
    """渲染分组控制 Tab。"""
    gc = f"{P}_gc"

    _gc_show_feedback()

    groups = st.session_state.get(f"{gc}_groups", {})
    group_names = st.session_state.get(f"{gc}_group_names", [])
    connected = st.session_state.get(f"{gc}_connected", False)
    relay_on = st.session_state.get(f"{gc}_relay_on", False)

    if not group_names:
        st.warning("未找到 wiring_map.json 或无有效组别定义")
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
            key=f"{gc}_group_select",
        )
        st.session_state[f"{gc}_selected_group"] = selected

        if selected and selected in groups:
            channels_by_ip = groups[selected]
            total_channels = sum(len(chs) for chs in channels_by_ip.values())
            st.caption(
                f"**{selected}** — {len(channels_by_ip)} 个控制器, "
                f"{total_channels} 个通道"
            )

            rows = []
            for ip_suffix in sorted(channels_by_ip.keys()):
                for ch_info in channels_by_ip[ip_suffix]:
                    rows.append({
                        "控制器 IP": f"192.168.0.{ip_suffix}",
                        "通道号": ch_info["payload_position"],
                        "针脚 ID": ch_info.get("needle_id", ""),
                        "物理标签": ch_info.get("physical_label", ""),
                    })
            if rows:
                with st.expander("📋 通道详情", expanded=False):
                    st.dataframe(
                        pd.DataFrame(rows), use_container_width=True, hide_index=True,
                    )

    with st.container(border=True):
        col_status, col_actions = st.columns([1, 1])
        with col_status:
            st.markdown("##### 连接状态")
            if connected:
                n_ctrl = len(st.session_state.get(f"{gc}_controllers", {}))
                st.success(f"✅ 已连接 {selected} ({n_ctrl} 个控制器)")
                if relay_on:
                    st.success("⚡ 继电器已上电 (输出接通)")
                else:
                    st.warning("⏻ 继电器已下电 (输出断开)")
            else:
                st.error("❌ 未连接")

        with col_actions:
            st.markdown("##### 操作")
            if not connected:
                if st.button(
                    "🔌 连接组控制器", type="primary", use_container_width=True,
                    key=f"{gc}_connect_btn",
                ):
                    with st.spinner("连接中..."):
                        _gc_connect()
                    st.rerun()
            else:
                col_r1, col_r2, col_disc = st.columns(3)
                with col_r1:
                    if st.button(
                        "⚡ 上电", type="primary", use_container_width=True,
                        disabled=relay_on, key=f"{gc}_relay_on_btn",
                    ):
                        _gc_set_relay(True)
                        st.rerun()
                with col_r2:
                    if st.button(
                        "⏻ 下电", use_container_width=True,
                        disabled=not relay_on, key=f"{gc}_relay_off_btn",
                    ):
                        _gc_set_relay(False)
                        st.rerun()
                with col_disc:
                    if st.button("⏏ 断开", use_container_width=True, key=f"{gc}_disconnect_btn"):
                        _gc_disconnect()
                        st.rerun()

    if not connected:
        st.info("💡 请先连接组控制器以进行控制操作。")
        return

    st.divider()
    st.markdown("##### 电压控制")

    channels_by_ip = groups.get(selected, {})
    all_payload_positions = sorted(
        ch["payload_position"]
        for chs in channels_by_ip.values()
        for ch in chs
    )

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
        for ip_suffix in sorted(channels_by_ip.keys()):
            for ch_info in channels_by_ip[ip_suffix]:
                pp = ch_info["payload_position"]
                nid = ch_info.get("needle_id", "")
                label = ch_info.get("physical_label", "")
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
            "⚡ 下发电压", type="primary", use_container_width=True,
            key=f"{gc}_apply_btn", disabled=not relay_on,
        ):
            _gc_apply_voltage()
            st.rerun()
    with col_apply_all:
        if st.button(
            "⚡ 全部通道下发", type="secondary", use_container_width=True,
            key=f"{gc}_apply_all_btn", disabled=not relay_on,
        ):
            st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()
            _gc_apply_all_voltage()
            st.rerun()
    with col_sel:
        if st.button("全选通道", use_container_width=True, key=f"{gc}_select_all_btn"):
            st.session_state[f"{gc}_selected_channels"] = all_payload_positions.copy()
            st.rerun()
    with col_desel:
        if st.button("清空选择", use_container_width=True, key=f"{gc}_deselect_all_btn"):
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
# Main App
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="R50 控制器控制面板",
        page_icon="🔌",
        layout="wide",
    )

    _initialize_state()

    tab1, tab2, tab3 = st.tabs(["🔌 单控制器", "🔗 联合控制 (36×36)", "🧪 分组控制"])

    with tab1:
        _render_tab_single_controller()

    with tab2:
        render_tab_joint_control()

    with tab3:
        render_tab_group_control()


def _render_tab_single_controller() -> None:
    """渲染单控制器 Tab (原有功能)。"""
    st.title("🔌 R50Power 单控制器控制面板")
    st.caption("单个 R50Controller (单 IP) 控制 | 连通性检测 · 继电器上下电 · 电压下发")

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
            _connected = st.session_state[f"{P}_connected"]
            st.text_input(
                "IP 地址", value=st.session_state[f"{P}_ip"],
                disabled=_connected, key=f"{P}_ip_input",
            )
            st.session_state[f"{P}_ip"] = st.session_state[f"{P}_ip_input"]
            st.number_input(
                "端口", min_value=1, max_value=65535,
                value=st.session_state[f"{P}_port"], step=1,
                disabled=_connected, key=f"{P}_port_input",
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
                    format_func=_channel_label,
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

            # 显示已选单元的针脚映射信息
            if st.session_state[f"{P}_channels"]:
                _infos = []
                for _ch in st.session_state[f"{P}_channels"]:
                    _ni = _get_needle_info(int(_ch))
                    _infos.append(f"ch{_ch}: {_ni}" if _ni else f"ch{_ch}: 无映射")
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
                                # 显示当前 IP 下所有有映射的通道
                                _mapped = []
                                for _ch in range(SINGLE_CHANNELS):
                                    _ni = _get_needle_info(_ch)
                                    if _ni:
                                        _mapped.append(f"ch{_ch}:{_ni}")
                                _extra = f" (映射通道: {', '.join(_mapped[:5])}{'...' if len(_mapped) > 5 else ''})" if _mapped else ""
                                set_feedback(f"已向全部 50 通道下发 {voltage:.1f} V{_extra}", "success")
                            else:
                                ch_list = st.session_state[f"{P}_channels"]
                                _detail_parts = []
                                for _ch in ch_list:
                                    _ni = _get_needle_info(int(_ch))
                                    _detail_parts.append(f"{_ch}({_ni})" if _ni else str(_ch))
                                _detail = ", ".join(_detail_parts)
                                set_feedback(
                                    f"已向 {len(ch_list)} 个单元下发 {voltage:.1f} V: {_detail}",
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
            _sine_ni = _get_needle_info(int(sine_ch))
            if _sine_ni:
                st.caption(f"针脚映射: {_sine_ni}")

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
