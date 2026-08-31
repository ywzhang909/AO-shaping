"""R50 单控制器模块 — 连通性测试 / 连接 / 断开 / 继电器 / 单次下发。

被 ``r50_tabs.render_tab_single_controller`` 与
``r50_sidebar._sidebar_single_connection`` 复用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    voltages_to_payload,
)
from ao_shaping.gui.r50.r50_channel_select import (
    CFG,
    P,
    SINGLE_CHANNELS,
    ChannelSelection,
)
from ao_shaping.gui.r50.r50_common import _loop_stop_all, _set_feedback
from ao_shaping.gui.r50.r50_connection import (
    create_controller,
    ping_reachable,
    power_off_and_close,
    set_relay,
    tcp_reachable,
)
from ao_shaping.gui.r50.r50_command import Packet, reset_unit_states, r50_command
from ao_shaping.gui.r50.r50_debug import _debug_log_packet
from ao_shaping.gui.r50.r50_voltage_send import (
    SendResult,
    apply_single_controller,
    build_bulk_array,
)


def _get_active_controller() -> Any | None:
    """获取当前模式下的活跃控制器。

    - single 模式: 返回 session_state 中的单控制器。
    - joint 模式: 从 MicroDM 中按选中的 IP 返回对应的 R50Controller。
    """
    mode = st.session_state.get(f"{P}_connection_mode", "single")
    if mode == "single":
        return st.session_state.get(f"{P}_controller")
    if mode == "joint":
        dm = st.session_state.get(f"{P}_jc_dm")
        if dm is None:
            return None
        selected_ip = st.session_state.get(f"{P}_jc_selected_ip", "")
        if not selected_ip:
            return None
        target_suffix = int(selected_ip.split(".")[-1])
        for ctrl in getattr(dm, "_controllers", []):
            if hasattr(ctrl, "ip") and ctrl.ip == selected_ip:
                return ctrl
            if hasattr(ctrl, "controller_id") and ctrl.controller_id == target_suffix:
                return ctrl
        return None
    return None


# =============================================================================
# 连通性测试
# =============================================================================


def test_connectivity() -> None:
    """测试与 192.168.0.x 控制器网络的连通性 (Ping + TCP)。"""
    ip = st.session_state.get(f"{P}_ip", "192.168.0.101").strip()
    port = int(st.session_state.get(f"{P}_port", CFG.DEFAULT_PORT))
    controller_num = int(st.session_state.get(f"{P}_controller_num", 1))
    st.write(f"#### 测试目标: 控制器 #{controller_num} ({ip}:{port})")
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
    controller_num = int(st.session_state.get(f"{P}_controller_num", 1))
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
        ctrl = create_controller(
            controller_id=controller_num, ip=ip, port=port, simulate=simulate
        )
    except ConnectionError as exc:
        st.session_state[f"{P}_connection_error"] = (
            f"无法建立 TCP 连接到 {ip}:{port} ({exc})"
        )
        st.session_state[f"{P}_connected"] = False
        logger.error(st.session_state[f"{P}_connection_error"])
        return
    st.session_state[f"{P}_controller"] = ctrl
    st.session_state[f"{P}_connected"] = True
    st.session_state[f"{P}_feedback"] = (
        f"已连接 控制器 #{controller_num} ({ip}:{port}) {'(仿真)' if simulate else ''}"
    )
    st.session_state[f"{P}_feedback_type"] = "success"
    logger.info(f"R50 控制器已连接: #{controller_num} {ip}:{port} simulate={simulate}")


def disconnect() -> None:
    """断开单控制器 (先停循环, 下电继电器, 再关闭连接)。"""
    _loop_stop_all()
    ctrl = st.session_state.get(f"{P}_controller")
    if ctrl is not None:
        try:
            ok = power_off_and_close(ctrl)
            if not ok:
                logger.warning("power_off_and_close returned False")
                st.toast("⚠️ 下电或关闭过程存在异常，请检查设备状态", icon="⚠️")
        except Exception as exc:
            logger.error(f"关闭控制器异常: {exc}")
    st.session_state[f"{P}_controller"] = None
    st.session_state[f"{P}_connected"] = False
    st.session_state[f"{P}_relay_on"] = False
    reset_unit_states()
    st.session_state[f"{P}_confirm_disconnect"] = False
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
        if not on:
            reset_unit_states()
        _set_feedback(f"继电器已{'开启' if on else '关闭'}", "success")
    else:
        _set_feedback("继电器操作失败 (未收到确认)", "error")


# =============================================================================
# 下发保护 / 单次下发
# =============================================================================


def _require_relay_on() -> bool:
    """下发前强制检查继电器状态。"""
    mode = st.session_state.get(f"{P}_connection_mode", "single")
    relay_key = f"{P}_relay_on" if mode == "single" else f"{P}_jc_relay_on"
    if not st.session_state.get(relay_key, False):
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


def _send_channels(voltage: float) -> SendResult:
    """单次下发当前选择 (全选或勾选通道), 更新当前电压表并反馈。

    统一走 :func:`r50_command`: 上电检查 / 发送日志 / 全局单元状态更新。
    """
    ctrl = _get_active_controller()
    if ctrl is None:
        _set_feedback("请先连接控制器", "error")
        return SendResult(fail=1, failed_targets=["未连接"])
    all_mode = st.session_state.get(f"{P}_all_mode", True)
    selection = ChannelSelection(
        all_mode=all_mode,
        channels=list(st.session_state.get(f"{P}_channels", [])),
    )
    if selection.is_empty:
        _set_feedback("未选择任何通道", "warning")
        return SendResult(fail=1, failed_targets=["无选中通道"])
    ip = getattr(ctrl, "ip", "") or st.session_state.get(f"{P}_ip", "")
    try:
        suffix = int(ip.split(".")[-1]) if ip else -1
    except ValueError:
        suffix = -1
    vmin = st.session_state[f"{P}_vmin"]
    vmax = st.session_state[f"{P}_vmax"]
    channels = list(range(SINGLE_CHANNELS)) if all_mode else selection.channels
    units = [(suffix, ch + 1) for ch in channels] if suffix > 0 else []

    def _send() -> tuple[SendResult, list[Packet]]:
        current, result = apply_single_controller(
            ctrl,
            st.session_state[f"{P}_current_voltages"],
            selection,
            voltage,
            vmin,
            vmax,
        )
        st.session_state[f"{P}_current_voltages"] = current
        if all_mode:
            payload = voltages_to_payload(np.asarray([voltage], dtype=np.float64))
            pkt = (
                HEADER
                + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, int(payload[0]), int(payload[1])])
                + FOOTER
            )
            packets = [("CMD_SET_ALL_CHANNEL_VOLTAGE", ip, pkt)]
        else:
            arr = build_bulk_array(
                current,
                selection.normalized(len(current)),
                voltage,
                vmin,
                vmax,
            )
            pkt = (
                HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(arr) + FOOTER
            )
            packets = [("CMD_SET_ALL_VOLTAGE_BY_ARR", ip, pkt)]
        return result, packets

    result = r50_command("single", "SEND_VOLTAGE", units, voltage, vmin, vmax, _send)
    _send_success_feedback(voltage, result)
    return result
