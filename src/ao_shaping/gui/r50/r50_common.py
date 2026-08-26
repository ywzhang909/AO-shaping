"""R50 公共模块 — 反馈 / 会话状态初始化 / 通道标签 / 发送循环管理。

被 ``r50_single`` / ``r50_group`` / ``r50_tabs`` / ``r50_sidebar`` 等
上层模块复用。只依赖底层模块, 不依赖任何 UI 渲染模块。

依赖: ``r50_debug`` (DebugTcpClient), ``r50_channel_select`` (常量 /
ChannelSelection / build_groups), ``r50_voltage_send`` (start_loop / stop_loop),
``ao_shaping.utils.network`` (ip_last_octet)。
"""

from __future__ import annotations

import collections
import queue
import threading
from typing import Any

import numpy as np
import streamlit as st

from ao_shaping.gui.r50.r50_channel_select import (
    CFG,
    DEBUG_HOST,
    DEBUG_LOG_MAX,
    DEBUG_PORT,
    HW_VOLTAGE_MAX,
    HW_VOLTAGE_MIN,
    P,
    SINGLE_CHANNELS,
    ChannelInfo,
    ChannelSelection,
    build_groups,
    get_channel_info,
)
from ao_shaping.gui.r50.r50_debug import DebugTcpClient
from ao_shaping.gui.r50.r50_voltage_send import start_loop, stop_loop
from ao_shaping.utils.network import ip_last_octet


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
# 通道标签助手 (单控制器 IP 下)
# =============================================================================

def _current_ip_suffix() -> int | None:
    """当前单控制器 IP 的末段 (int), 解析失败返回 None。"""
    ip = st.session_state.get(f"{P}_ip", "").strip()
    return ip_last_octet(ip)


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
    # NOTE: 不能用 setdefault — 旧版 UI 可能残留 None/错误类型, 会导致 widget 创建失败
    # (st.text_input(value=None) 抛异常, widget key 不注册, 后续读取直接 KeyError)。
    if not isinstance(st.session_state.get(f"{P}_ip"), str) or not st.session_state[f"{P}_ip"].strip():
        st.session_state[f"{P}_ip"] = "192.168.0.101"
    if not isinstance(st.session_state.get(f"{P}_port"), int):
        st.session_state[f"{P}_port"] = CFG.DEFAULT_PORT
    if not isinstance(st.session_state.get(f"{P}_connected"), bool):
        st.session_state[f"{P}_connected"] = False
    st.session_state.setdefault(f"{P}_controller", None)
    if not isinstance(st.session_state.get(f"{P}_connection_error"), str):
        st.session_state[f"{P}_connection_error"] = ""
    if not isinstance(st.session_state.get(f"{P}_simulate"), bool):
        st.session_state[f"{P}_simulate"] = False
    if not isinstance(st.session_state.get(f"{P}_vmin"), (int, float)):
        st.session_state[f"{P}_vmin"] = HW_VOLTAGE_MIN
    if not isinstance(st.session_state.get(f"{P}_vmax"), (int, float)):
        st.session_state[f"{P}_vmax"] = HW_VOLTAGE_MAX
    if not isinstance(st.session_state.get(f"{P}_relay_on"), bool):
        st.session_state[f"{P}_relay_on"] = False
    if not isinstance(st.session_state.get(f"{P}_confirm_disconnect"), bool):
        st.session_state[f"{P}_confirm_disconnect"] = False

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

    # ---- 逐序下发 (sequential per-channel) ----
    st.session_state.setdefault(f"{P}_seq_running", False)
    st.session_state.setdefault(f"{P}_seq_voltage", 20.0)
    st.session_state.setdefault(f"{P}_seq_interval", 1.0)
    st.session_state.setdefault(f"{P}_seq_auto_loop", False)

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
    st.session_state.setdefault(f"{P}_seq_running", False)
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
    st.session_state.setdefault(f"{p}_applied_matrix", None)
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
    st.session_state[f"{P}_seq_running"] = False
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
    params["feedback_q"] = q
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
