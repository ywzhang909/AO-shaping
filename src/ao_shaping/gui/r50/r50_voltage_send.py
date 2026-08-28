"""R50 voltage send logic: clipping, bulk array build, send with retry/reconnect,
per-mode apply helpers and thread-safe send loops.

Pure logic — no top-level streamlit import (streamlit-dependent UI lives in the
controller main file). Testable with simulated controllers.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from loguru import logger

from ao_shaping.gui.r50.r50_channel_select import (
    CFG,
    SINGLE_CHANNELS,
    ChannelInfo,
    ChannelSelection,
    GroupDef,
)

HW_VOLTAGE_MIN = CFG.HW_VOLTAGE_MIN
HW_VOLTAGE_MAX = CFG.HW_VOLTAGE_MAX


# =============================================================================
# Clipping & Bulk Array
# =============================================================================


def clip_voltage(
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
) -> float:
    """Clip a voltage into the hardware-safe range [vmin, vmax]."""
    return float(np.clip(float(voltage), vmin, vmax))


def build_bulk_array(
    current: np.ndarray,
    selected: list[int],
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
) -> np.ndarray:
    """Copy of `current` with `selected` 0-based channels set to clipped voltage.

    Unselected channels keep their current values, so a bulk send never
    disturbs channels outside the selection.
    """
    v = clip_voltage(voltage, vmin, vmax)
    arr = np.asarray(current, dtype=np.float64).copy()
    for ch in selected:
        idx = int(ch)
        if 0 <= idx < arr.size:
            arr[idx] = v
    return arr


# =============================================================================
# Send with Retry / Reconnect (S3: dead-socket recovery)
# =============================================================================


def _controller_connected(ctrl: Any) -> bool:
    """Check connectivity. Real R50Controller exposes `is_connected` as a
    property; sim controllers expose it as a method. Handles both."""
    attr = getattr(ctrl, "is_connected", None)
    if attr is None:
        return True  # no connectivity surface: nothing to recover
    if callable(attr):
        try:
            return bool(attr())
        except Exception:
            return False
    return bool(attr)


def _recover_controller(ctrl: Any) -> bool:
    """Reopen the controller if it reports disconnected. Returns connected state."""
    try:
        if _controller_connected(ctrl):
            return True
        open_fn = getattr(ctrl, "open", None)
        if callable(open_fn):
            return bool(open_fn())
        return False
    except Exception:
        return False


def _send_with_recovery(
    ctrl: Any,
    send_fn: Callable[[], bool],
    retries: int = 2,
    backoff: float = 0.05,
) -> bool:
    """Call send_fn() up to retries+1 times; on failure try to recover the
    connection before the next attempt. Returns overall success."""
    for attempt in range(retries + 1):
        try:
            if bool(send_fn()):
                return True
        except Exception:
            pass
        if attempt < retries:
            _recover_controller(ctrl)
            time.sleep(backoff * (attempt + 1))
    return False


def send_bulk_with_retry(
    ctrl: Any,
    voltages: list[float],
    retries: int = 2,
    backoff: float = 0.05,
) -> bool:
    """Send a full 50-channel array in ONE packet (CMD 0x09) with reconnect."""
    return _send_with_recovery(ctrl, lambda: ctrl.set_all_voltage_array(voltages), retries, backoff)


def send_all_channels(
    ctrl: Any,
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
) -> bool:
    """Set all 50 channels to one voltage (CMD 0x08) with reconnect."""
    v = clip_voltage(voltage, vmin, vmax)
    return _send_with_recovery(ctrl, lambda: ctrl.set_all_channel_voltage(v))


def send_selection(
    ctrl: Any,
    current: np.ndarray,
    selection: ChannelSelection,
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
) -> tuple[np.ndarray, bool]:
    """Send `voltage` to the selected channels of one controller.

    all_mode -> single 0x08 packet; channel list -> single 0x09 bulk packet.
    Mutates `current` in place on success. Returns (current, ok).
    """
    v = clip_voltage(voltage, vmin, vmax)
    channels = selection.normalized(len(current))
    if selection.all_mode:
        ok = send_all_channels(ctrl, v, vmin, vmax)
        if ok:
            current[:] = v
        return current, ok
    if not channels:
        return current, False
    arr = build_bulk_array(current, channels, v, vmin, vmax)
    ok = _send_with_recovery(ctrl, lambda: ctrl.set_all_voltage_array(arr.tolist()))
    if ok:
        current[:] = arr
    return current, ok


# =============================================================================
# Apply Helpers (per tab)
# =============================================================================


@dataclass
class SendResult:
    """Truthful outcome of a send operation."""

    ok: int = 0
    fail: int = 0
    failed_targets: list[str] = field(default_factory=list)


def _target_label(ctrl: Any) -> str:
    return str(getattr(ctrl, "ip", None) or ctrl)


def apply_single_controller(
    ctrl: Any,
    current: np.ndarray,
    selection: ChannelSelection,
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
) -> tuple[np.ndarray, SendResult]:
    """Single-controller tab: send to the shared selection in one bulk packet."""
    if selection.is_empty:
        return current, SendResult()
    try:
        current, ok = send_selection(ctrl, current, selection, voltage, vmin, vmax)
    except Exception as e:
        logger.warning(f"apply_single_controller failed: {e}")
        return current, SendResult(fail=1, failed_targets=[_target_label(ctrl)])
    if ok:
        return current, SendResult(ok=1)
    return current, SendResult(fail=1, failed_targets=[_target_label(ctrl)])


def apply_group_controllers(
    controllers: dict[int, Any],
    group_def: GroupDef,
    selected_payloads: list[int],
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
    current_map: dict[int, np.ndarray] | None = None,
) -> SendResult:
    """Group tab: deliver to EVERY affected controller in ONE bulk packet each.

    The legacy code sent per-channel `set_channel_voltage`, so a socket drop
    mid-way silently left half the channels unsent (the multi-click bug).
    This path sends exactly one 0x09 array per controller, checks the return
    value, reconnects+retries on failure, and reports failures truthfully.
    """
    if current_map is None:
        current_map = {}
    v = clip_voltage(voltage, vmin, vmax)
    sel = {int(p) for p in selected_payloads}
    result = SendResult()
    for ip_suffix, ch_list in group_def.channels_by_ip.items():
        ctrl = controllers.get(int(ip_suffix))
        if ctrl is None:
            result.fail += 1
            result.failed_targets.append(f"192.168.0.{ip_suffix} (未连接)")
            continue
        cur = current_map.setdefault(int(ip_suffix), np.zeros(SINGLE_CHANNELS, dtype=np.float64))
        arr = np.asarray(cur, dtype=np.float64).copy()
        touched = False
        for ci in ch_list:
            if ci.payload_position in sel:
                arr[ci.payload_position - 1] = v
                touched = True
        if not touched:
            continue  # no selected channel lives on this controller
        if _send_with_recovery(ctrl, lambda: ctrl.set_all_voltage_array(arr.tolist())):
            cur[:] = arr
            result.ok += 1
        else:
            result.fail += 1
            result.failed_targets.append(_target_label(ctrl))
    return result


def apply_units_via_controller(
    ctrl: Any,
    current: np.ndarray,
    units: list[ChannelInfo],
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
) -> SendResult:
    """Single-unit tab (single/group modes): one bulk packet per controller."""
    if not units:
        return SendResult()
    v = clip_voltage(voltage, vmin, vmax)
    arr = np.asarray(current, dtype=np.float64).copy()
    for u in units:
        idx = u.payload_position - 1
        if 0 <= idx < arr.size:
            arr[idx] = v
    if _send_with_recovery(ctrl, lambda: ctrl.set_all_voltage_array(arr.tolist())):
        current[:] = arr
        return SendResult(ok=len(units))
    return SendResult(fail=len(units), failed_targets=[_target_label(ctrl)])


def apply_joint(
    dm: Any,
    current_flat: np.ndarray,
    units: list[ChannelInfo],
    voltage: float,
    vmin: float = HW_VOLTAGE_MIN,
    vmax: float = HW_VOLTAGE_MAX,
    pos_to_hw: dict[int, tuple[int, int]] | None = None,
    ip_to_ctrl: dict[int, int] | None = None,
) -> tuple[np.ndarray, SendResult]:
    """Single-unit tab (joint mode): sparse flat-array send via MicroDM.

    Unselected positions are preserved from `current_flat` (the legacy code
    zeroed everything else on every click).
    """
    if pos_to_hw is None:
        pos_to_hw = {}
    if ip_to_ctrl is None:
        ip_to_ctrl = {}
    cur = np.asarray(current_flat, dtype=np.float64)
    # Size the flat array to match the real controller layout (DM_Num), NOT the
    # 36×36 matrix size. The single-unit tab hands in ``jc_current_flat`` which is
    # 36×36=1296 long, but ``MicroDM.send_voltages`` requires exactly ``(DM_Num,)``
    # = (n_controllers × 50,). Sending 1296 against a smaller DM_Num raises
    # MicroDMVoltageError, so we rebuild at the correct length and seed any
    # overlapping tail from the caller's current state to preserve it.
    if dm is not None:
        dm_num = int(getattr(dm, "DM_Num", 0) or 0)
    else:
        dm_num = 0
    if dm_num <= 0:
        dm_num = len(ip_to_ctrl) * SINGLE_CHANNELS
    if dm_num <= 0:
        dm_num = int(cur.size)
    flat = np.zeros(dm_num, dtype=np.float64)
    n = min(int(cur.size), dm_num)
    flat[:n] = cur[:n]
    if not units:
        return flat, SendResult()
    v = clip_voltage(voltage, vmin, vmax)
    applied = 0
    for u in units:
        hw = pos_to_hw.get(u.physical_position)
        if hw is None:
            continue
        ip_suffix, payload_pos = hw
        ctrl_idx = ip_to_ctrl.get(ip_suffix)
        if ctrl_idx is None:
            continue
        flat_idx = ctrl_idx * SINGLE_CHANNELS + (payload_pos - 1)
        if 0 <= flat_idx < flat.size:
            flat[flat_idx] = v
            applied += 1
    try:
        dm.send_voltages(flat)
        return flat, SendResult(ok=applied)
    except Exception as e:
        logger.warning(f"apply_joint failed: {e}")
        return flat, SendResult(fail=len(units), failed_targets=["joint"])


# =============================================================================
# Send Loops (hold / sine / alt) — thread-safe
# =============================================================================


def hold_tick(ctrl: Any, current: np.ndarray, p: dict) -> None:
    """Hold: resend the fixed voltage every tick."""
    send_selection(ctrl, current, p["selection"], p["voltage"], p["vmin"], p["vmax"])


def sine_tick(ctrl: Any, current: np.ndarray, p: dict) -> None:
    """Sine: v = offset + amp*sin(2*pi*freq*t)."""
    t = time.time() - p["t0"]
    v = p["offset"] + p["amp"] * np.sin(2.0 * np.pi * p["freq"] * t)
    send_selection(ctrl, current, p["selection"], v, p["vmin"], p["vmax"])


def alt_tick(ctrl: Any, current: np.ndarray, p: dict) -> None:
    """Alternate: toggle between `voltage` and 0V every half period."""
    elapsed = time.time() - p["t0"]
    on = int(elapsed * 2.0 * p["freq"]) % 2 == 0
    v = p["voltage"] if on else 0.0
    send_selection(ctrl, current, p["selection"], v, p["vmin"], p["vmax"])


def seq_tick(ctrl: Any, current: np.ndarray, p: dict) -> None:
    """Sequential: for each channel in order, send voltage X then 0V with T interval.

    State tracked in ``p``:
        seq_channels  – ordered list of 0-based channel indices
        seq_index     – current channel position
        seq_phase     – 0 = send voltage X, 1 = send 0V
        seq_last_tick – timestamp of last operation
        seq_interval  – seconds between each operation
        voltage       – the voltage X to apply
        seq_done      – set True when all channels are swept (unless auto-loop)
        seq_auto_loop – bool, if True restart sweep from channel 0 when done
        seq_round     – current sweep round number (incremented on each complete sweep)
    """
    now = time.time()
    if now - p["seq_last_tick"] < p["seq_interval"]:
        return
    p["seq_last_tick"] = now

    idx = p["seq_index"]
    channels: list[int] = p["seq_channels"]
    total = len(channels)

    if idx >= total:
        if p.get("seq_auto_loop"):
            p["seq_round"] = p.get("seq_round", 0) + 1
            p["seq_index"] = 0
            p["seq_phase"] = 0
            try:
                q: queue.Queue = p["feedback_q"]  # type: ignore[assignment]
                q.put(("info", f"逐序下发第 {p['seq_round']} 轮完成，开始下一轮"))
            except Exception:
                pass
        else:
            p["seq_done"] = True
            try:
                q: queue.Queue = p["feedback_q"]  # type: ignore[assignment]
                q.put(("info", "逐序下发已完成"))
            except Exception:
                pass
        return

    ch = channels[idx]
    sel = ChannelSelection(all_mode=False, channels=[ch])

    if p["seq_phase"] == 0:
        send_selection(ctrl, current, sel, p["voltage"], p["vmin"], p["vmax"])
        p["seq_phase"] = 1
    else:
        send_selection(ctrl, current, sel, 0.0, p["vmin"], p["vmax"])
        p["seq_phase"] = 0
        p["seq_index"] = idx + 1
        # Report progress after completing one channel
        try:
            q: queue.Queue = p["feedback_q"]  # type: ignore[assignment]
            q.put(("progress", f"{idx + 1}/{total}"))
        except Exception:
            pass


def run_loop(
    loop_fn: Callable[[Any, np.ndarray, dict], None],
    ctrl: Any,
    current: np.ndarray,
    params: dict,
    stop_event: threading.Event,
    feedback_q: queue.Queue,
) -> None:
    """Generic loop runner. Runs in a daemon thread; never touches session_state.

    Thread safety: `params` is a snapshot taken at start time; feedback goes
    through the queue; the main thread drains it on rerun.
    """
    try:
        while not stop_event.is_set():
            loop_fn(ctrl, current, params)
            time.sleep(float(params.get("dt", 0.05)))
    except Exception as e:
        try:
            feedback_q.put(("error", f"下发异常: {e}"))
        except Exception:
            pass
    finally:
        stop_event.set()


def start_loop(
    loop_fn: Callable[[Any, np.ndarray, dict], None],
    ctrl: Any,
    current: np.ndarray,
    params: dict,
    stop_event: threading.Event,
    feedback_q: queue.Queue,
) -> threading.Thread:
    """Start the loop in a daemon thread; returns the thread handle."""
    thread = threading.Thread(
        target=run_loop,
        args=(loop_fn, ctrl, current, params, stop_event, feedback_q),
        daemon=True,
        name="r50-send-loop",
    )
    thread.start()
    return thread


def stop_loop(stop_event: threading.Event) -> None:
    """Request the loop to stop (event set; thread exits on next tick)."""
    stop_event.set()
