"""R50 分组控制模块 (GC) — 组连接 / 断开 / 继电器 / 下发 / 批量上下电。

被 ``r50_tabs.render_tab_single_group`` 与
``r50_sidebar._sidebar_group_connection`` 复用。
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from loguru import logger

from ao_shaping.gui.r50.r50_channel_select import (
    CFG,
    P,
)
from ao_shaping.gui.r50.r50_common import _set_feedback, _show_feedback
from ao_shaping.gui.r50.r50_connection import (
    create_controller,
    ping_reachable,
    power_off_and_close,
    set_relay,
)
from ao_shaping.gui.r50.r50_debug import _debug_add_op
from ao_shaping.gui.r50.r50_voltage_send import (
    apply_group_controllers,
    clip_voltage,
)


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
