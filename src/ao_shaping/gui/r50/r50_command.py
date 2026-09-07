"""R50 centralized send-command layer + global per-unit voltage state.

所有电压下发指令统一走 :func:`r50_command`, 它集中处理: 上电检查、发送日志
(指令日志 / 操作日志)、全局单元电压状态更新, 以及单元状态显示刷新标记。

全局单元状态 ``r50c_unit_states`` 是一个 dict[(ip_suffix, payload_position)] -> float:
    - ``float("inf")`` 表示该单元「未上电」(未下发过电压, 或继电器已下电);
    - 有限数值表示该单元当前保持的具体电压 (V)。

规范 key 为 (ip_suffix, payload_position), 与联合控制 ``pos_to_hw`` /
分组 ``channels_by_ip`` / 单单元 ``ChannelInfo`` 的映射口径一致。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import streamlit as st

from ao_shaping.gui.r50.r50_channel_select import P
from ao_shaping.gui.r50.r50_common import _set_feedback
from ao_shaping.gui.r50.r50_debug import _debug_add_op, _debug_log_packet
from ao_shaping.gui.r50.r50_voltage_send import SendResult, clip_voltage

# 未上电哨兵
INF = float("inf")

UNIT_STATES_KEY = f"{P}_unit_states"
UNIT_STATE_DIRTY = f"{P}_unit_state_dirty"

# 各模式下继电器上电状态对应的 session key
_RELAY_KEY = {
    "single": f"{P}_relay_on",
    "joint": f"{P}_jc_relay_on",
    "group": f"{P}_gc_relay_on",
}

# send_fn 返回的实际发送包: (cmd_name, ip 标签, bytes)
Packet = tuple[str, str, bytes]


def _unit_states() -> dict[tuple[int, int], float]:
    states = st.session_state.get(UNIT_STATES_KEY)
    if states is None:
        states = {}
        st.session_state[UNIT_STATES_KEY] = states
    return states


def get_unit_voltage(ip_suffix: int, payload_position: int) -> float:
    """某单元当前电压; 未上电返回 ``INF``。"""
    return _unit_states().get((int(ip_suffix), int(payload_position)), INF)


def set_unit_voltage(ip_suffix: int, payload_position: int, voltage: float) -> None:
    """写入某单元电压 (有限值 = 已上电且保持该电压)。"""
    _unit_states()[(int(ip_suffix), int(payload_position))] = float(voltage)


def reset_unit_states() -> None:
    """将全局单元状态清零 —— 等价于全部单元未上电 (inf)。

    在继电器下电 / 断开连接时调用。
    """
    st.session_state[UNIT_STATES_KEY] = {}


def mark_units_powered(
    units: Iterable[tuple[int, int]], voltage: float
) -> None:
    """下发成功后, 将受影响单元记为已上电并持有 voltage。"""
    for suffix, pos in units:
        set_unit_voltage(suffix, pos, voltage)


def refresh_unit_displays() -> None:
    """标记需刷新单元状态显示 (表格 / 联合矩阵), 下次 rerun 时重读全局状态。"""
    st.session_state[UNIT_STATE_DIRTY] = True


def consume_unit_refresh_flag() -> bool:
    """读取并清除刷新标记 (供渲染函数判断是否需重建单元状态显示)。"""
    dirty = bool(st.session_state.get(UNIT_STATE_DIRTY, False))
    st.session_state[UNIT_STATE_DIRTY] = False
    return dirty


def r50_command(
    mode: str,
    command: str,
    units: Iterable[tuple[int, int]],
    voltage: float,
    vmin: float,
    vmax: float,
    send_fn: Callable[[], tuple[SendResult, list[Packet]]],
    state_updater: Callable[[], None] | None = None,
) -> SendResult:
    """所有 R50 电压下发的统一入口。

    Args:
        mode: "single" | "joint" | "group" (用于定位继电器上电状态)。
        command: 指令名 (写入日志)。
        units: 受影响单元 [(ip_suffix, payload_position), ...]。
        voltage: 下发电压 (V)。
        vmin/vmax: 电压裁剪范围。
        send_fn: 执行实际下发, 返回 (SendResult, 已发送数据包列表)。
        state_updater: 下发成功后自定义更新全局单元状态; 缺省时将所有
            ``units`` 记为已上电并持有 ``clip_voltage(voltage)``。联合模式下
            各单元电压不同, 由调用方传入此函数按映射逐格写入。

    Returns:
        下发结果。
    """
    units = list(units)
    relay_key = _RELAY_KEY.get(mode)
    if relay_key is None or not st.session_state.get(relay_key, False):
        _set_feedback("⚠️ 继电器未上电, 无法下发电压 (请先上电)", "warning")
        return SendResult(fail=max(len(units), 1), failed_targets=["继电器未上电"])

    result, packets = send_fn()

    for cmd, ip, pkt in packets:
        _debug_log_packet(cmd, pkt)
        _debug_add_op(cmd, f"to {ip}", ip)

    if result.ok and not result.fail:
        if state_updater is not None:
            state_updater()
        else:
            v = clip_voltage(voltage, vmin, vmax)
            mark_units_powered(units, v)
        refresh_unit_displays()
    return result


def unit_states_frame(
    units: Iterable[tuple[int, int]],
    label: Callable[[int, int], str] | None = None,
) -> list[dict[str, Any]]:
    """把一组单元 (suffix, pos) 转为状态行, 供表格展示。

    label 缺省时用 '192.168.0.{suffix} 通道{pos}'。
    """
    rows: list[dict[str, Any]] = []
    for suffix, pos in units:
        v = get_unit_voltage(suffix, pos)
        display = "未上电" if v == INF else f"{v:.1f} V"
        rows.append(
            {
                "控制器 IP": f"192.168.0.{suffix}",
                "通道号": pos,
                "电压状态": display,
            }
        )
    return rows
