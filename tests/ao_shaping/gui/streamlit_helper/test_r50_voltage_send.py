"""RED tests for r50_voltage_send (pure logic, no streamlit).

Contract covered (Wave 1):
- S1: multi-channel send -> single correct bulk array, sim controller matches
- S2 (BUG regression): group send delivers to ALL controllers in ONE call,
      exactly one 0x09 packet per controller
- S3: dead socket -> auto-reconnect + retry succeeds
- S4: out-of-range voltage clipped to [vmin, vmax]
- S5: failures counted truthfully (SendResult)
- Loop threads: stop via event, error -> feedback queue
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import (
    CFG,
    ChannelInfo,
    ChannelSelection,
    GroupDef,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_connection import (
    SimulatedMicroDM,
    SimulatedR50Controller,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_voltage_send import (
    SendResult,
    alt_tick,
    apply_group_controllers,
    apply_joint,
    apply_single_controller,
    apply_units_via_controller,
    build_bulk_array,
    clip_voltage,
    hold_tick,
    send_all_channels,
    send_bulk_with_retry,
    send_selection,
    sine_tick,
    start_loop,
    stop_loop,
)

VMIN = CFG.HW_VOLTAGE_MIN
VMAX = CFG.HW_VOLTAGE_MAX
N = CFG.SINGLE_CHANNELS


def _ci(ip_suffix: int, pp: int, pos: int, label: str = "1-1-1") -> ChannelInfo:
    return ChannelInfo(
        ip_suffix=ip_suffix,
        payload_position=pp,
        physical_position=pos,
        group="一组",
        needle_id=pp,
        physical_label=label,
    )


def _sim_ctrl(ip_suffix: int = 101) -> SimulatedR50Controller:
    ctrl = SimulatedR50Controller(ip_suffix, f"192.168.0.{ip_suffix}", 10101)
    ctrl.open()
    return ctrl


class CountingCtrl(SimulatedR50Controller):
    """Sim controller that counts which send paths were used."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array_sends = 0
        self.channel_sends = 0

    def set_all_voltage_array(self, voltages):
        self.array_sends += 1
        return super().set_all_voltage_array(voltages)

    def set_channel_voltage(self, channel, voltage):
        self.channel_sends += 1
        return super().set_channel_voltage(channel, voltage)


class FlakyCtrl:
    """Models the real bug: socket dies mid-send (is_connected -> False),
    open() reconnects."""

    def __init__(self):
        self.send_calls = 0
        self.open_calls = 0
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def open(self) -> bool:
        self.open_calls += 1
        self._connected = True
        return True

    def set_all_voltage_array(self, voltages) -> bool:
        self.send_calls += 1
        if self.send_calls == 1:
            self._connected = False  # socket died mid-send
            return False
        return True


class FailingCtrl:
    """Always fails; no connection surface (recovery has nothing to do)."""

    def set_all_voltage_array(self, voltages) -> bool:
        return False

    def set_all_channel_voltage(self, voltage) -> bool:
        return False


# ---------------------------------------------------------------------------
# S4: clipping
# ---------------------------------------------------------------------------

class TestClipVoltage:
    def test_within_range(self):
        assert clip_voltage(5.0, VMIN, VMAX) == 5.0

    def test_above_max(self):
        assert clip_voltage(999.0, VMIN, VMAX) == VMAX

    def test_below_min(self):
        assert clip_voltage(-999.0, VMIN, VMAX) == VMIN

    def test_edge_values(self):
        assert clip_voltage(VMIN, VMIN, VMAX) == VMIN
        assert clip_voltage(VMAX, VMIN, VMAX) == VMAX


# ---------------------------------------------------------------------------
# S1: bulk array construction
# ---------------------------------------------------------------------------

class TestBuildBulkArray:
    def test_selected_set_others_preserved(self):
        current = np.zeros(N)
        current[7] = 2.5
        arr = build_bulk_array(current, [0, 1, 2], 5.0, VMIN, VMAX)
        assert np.all(arr[0:3] == 5.0)
        assert arr[7] == 2.5
        assert np.all(arr[3:7] == 0.0)
        assert np.all(arr[8:] == 0.0)

    def test_voltage_clipped_in_array(self):
        current = np.zeros(N)
        arr = build_bulk_array(current, [0], 999.0, VMIN, VMAX)
        assert arr[0] == VMAX

    def test_out_of_range_channel_ignored(self):
        current = np.zeros(N)
        arr = build_bulk_array(current, [N + 5, -1, 3], 5.0, VMIN, VMAX)
        assert arr[3] == 5.0
        assert arr.sum() == 5.0

    def test_does_not_mutate_current(self):
        current = np.zeros(N)
        build_bulk_array(current, [0], 5.0, VMIN, VMAX)
        assert np.all(current == 0.0)


# ---------------------------------------------------------------------------
# S3: send with retry / reconnect
# ---------------------------------------------------------------------------

class TestSendBulkWithRetry:
    def test_success_first_try(self):
        ctrl = _sim_ctrl()
        assert send_bulk_with_retry(ctrl, [1.0] * N)

    def test_reconnects_after_dead_socket(self):
        ctrl = FlakyCtrl()
        assert send_bulk_with_retry(ctrl, [1.0] * N, retries=2, backoff=0.0)
        assert ctrl.open_calls == 1
        assert ctrl.send_calls == 2

    def test_fails_after_retries_exhausted(self):
        ctrl = FailingCtrl()
        assert not send_bulk_with_retry(ctrl, [1.0] * N, retries=2, backoff=0.0)

    def test_exception_on_send_fails_after_retries(self):
        class RaiseCtrl:
            def set_all_voltage_array(self, voltages) -> bool:
                raise OSError("socket gone")

        assert not send_bulk_with_retry(RaiseCtrl(), [1.0] * N, retries=1, backoff=0.0)


class TestSendAllChannels:
    def test_all_channels(self):
        ctrl = _sim_ctrl()
        assert send_all_channels(ctrl, 7.5, VMIN, VMAX)
        assert np.all(ctrl.readback() == 7.5)

    def test_clipped(self):
        ctrl = _sim_ctrl()
        assert send_all_channels(ctrl, 999.0, VMIN, VMAX)
        assert np.all(ctrl.readback() == VMAX)


# ---------------------------------------------------------------------------
# S1: send_selection (single controller)
# ---------------------------------------------------------------------------

class TestSendSelection:
    def test_channels_bulk(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        sel = ChannelSelection(channels=[0, 1, 2])
        new_current, ok = send_selection(ctrl, current, sel, 5.0, VMIN, VMAX)
        assert ok
        assert np.all(new_current[0:3] == 5.0)
        assert np.all(ctrl.readback()[0:3] == 5.0)
        assert np.all(ctrl.readback()[3:] == 0.0)

    def test_all_mode(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        sel = ChannelSelection(all_mode=True)
        _, ok = send_selection(ctrl, current, sel, 5.0, VMIN, VMAX)
        assert ok
        assert np.all(ctrl.readback() == 5.0)

    def test_empty_selection_noop(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        _, ok = send_selection(ctrl, current, ChannelSelection(), 5.0, VMIN, VMAX)
        assert not ok
        assert np.all(ctrl.readback() == 0.0)

    def test_failure_keeps_current(self):
        ctrl = FailingCtrl()
        current = np.full(N, 3.0)
        new_current, ok = send_selection(
            ctrl, current, ChannelSelection(channels=[0]), 5.0, VMIN, VMAX
        )
        assert not ok
        assert np.all(new_current == 3.0)


class TestApplySingleController:
    def test_happy_path(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        sel = ChannelSelection(channels=[0, 1])
        current, res = apply_single_controller(ctrl, current, sel, 5.0, VMIN, VMAX)
        assert res.ok == 1 and res.fail == 0
        assert np.all(ctrl.readback()[0:2] == 5.0)

    def test_empty_selection(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        current, res = apply_single_controller(ctrl, current, ChannelSelection(), 5.0, VMIN, VMAX)
        assert res.ok == 0 and res.fail == 0

    def test_failure_truthful(self):
        ctrl = FailingCtrl()
        current = np.zeros(N)
        current, res = apply_single_controller(
            ctrl, current, ChannelSelection(channels=[0]), 5.0, VMIN, VMAX
        )
        assert res.ok == 0 and res.fail == 1
        assert res.failed_targets  # target identified


# ---------------------------------------------------------------------------
# S2 (BUG regression): group send — all controllers in ONE call, one packet each
# ---------------------------------------------------------------------------

def _group_def() -> GroupDef:
    return GroupDef(
        name="测试组",
        channels_by_ip={
            101: [_ci(101, 1, 5), _ci(101, 2, 6)],
            102: [_ci(102, 1, 50)],
        },
    )


class TestApplyGroupControllers:
    def test_all_selected_delivered_in_one_call(self):
        """THE regression: one call -> every controller updated, exactly one
        bulk packet per controller (no per-channel sends)."""
        c0 = CountingCtrl(101, "192.168.0.101", 10101)
        c0.open()
        c1 = CountingCtrl(102, "192.168.0.102", 10101)
        c1.open()
        controllers = {101: c0, 102: c1}
        current_map: dict[int, np.ndarray] = {}

        res = apply_group_controllers(
            controllers, _group_def(), [1, 2], 5.0, VMIN, VMAX, current_map
        )
        assert (res.ok, res.fail) == (2, 0)
        # exactly one 0x09 packet per controller, zero per-channel sends
        assert (c0.array_sends, c0.channel_sends) == (1, 0)
        assert (c1.array_sends, c1.channel_sends) == (1, 0)
        # all channels arrived in the single call
        assert np.all(c0.readback()[0:2] == 5.0)
        assert np.all(c0.readback()[2:] == 0.0)
        assert c1.readback()[0] == 5.0
        assert np.all(c1.readback()[1:] == 0.0)

    def test_partial_selection_skips_untouched_controllers(self):
        c0 = CountingCtrl(101, "192.168.0.101", 10101)
        c0.open()
        c1 = CountingCtrl(102, "192.168.0.102", 10101)
        c1.open()
        controllers = {101: c0, 102: c1}

        res = apply_group_controllers(
            controllers, _group_def(), [2], 5.0, VMIN, VMAX, {}
        )
        assert (res.ok, res.fail) == (1, 0)
        assert (c0.array_sends, c1.array_sends) == (1, 0)  # c1 untouched
        assert c0.readback()[1] == 5.0
        assert c0.readback()[0] == 0.0
        assert np.all(c1.readback() == 0.0)

    def test_missing_controller_counted_as_failure(self):
        c0 = _sim_ctrl(101)
        controllers = {101: c0}
        res = apply_group_controllers(
            controllers, _group_def(), [1, 2], 5.0, VMIN, VMAX, {}
        )
        assert (res.ok, res.fail) == (1, 1)
        assert any("192.168.0.102" in t for t in res.failed_targets)

    def test_current_map_preserved_across_calls(self):
        c0 = CountingCtrl(101, "192.168.0.101", 10101)
        c0.open()
        c1 = CountingCtrl(102, "192.168.0.102", 10101)
        c1.open()
        controllers = {101: c0, 102: c1}
        current_map: dict[int, np.ndarray] = {}
        apply_group_controllers(controllers, _group_def(), [1, 2], 5.0, VMIN, VMAX, current_map)
        # second send of a subset must preserve the other channel
        apply_group_controllers(controllers, _group_def(), [1], 3.0, VMIN, VMAX, current_map)
        assert c0.readback()[0] == 3.0
        assert c0.readback()[1] == 5.0  # preserved from current_map


# ---------------------------------------------------------------------------
# single-unit tab helpers
# ---------------------------------------------------------------------------

class TestApplyUnitsViaController:
    def test_bulk_single_packet(self):
        ctrl = CountingCtrl(101, "192.168.0.101", 10101)
        ctrl.open()
        current = np.zeros(N)
        units = [_ci(101, 3, 7), _ci(101, 7, 11)]
        res = apply_units_via_controller(ctrl, current, units, 5.0, VMIN, VMAX)
        assert (res.ok, res.fail) == (2, 0)
        assert (ctrl.array_sends, ctrl.channel_sends) == (1, 0)
        assert ctrl.readback()[2] == 5.0
        assert ctrl.readback()[6] == 5.0
        assert np.all(ctrl.readback()[:2] == 0.0)

    def test_unselected_preserved(self):
        ctrl = _sim_ctrl(101)
        current = np.zeros(N)
        current[10] = 4.0
        units = [_ci(101, 3, 7)]
        apply_units_via_controller(ctrl, current, units, 5.0, VMIN, VMAX)
        assert ctrl.readback()[2] == 5.0
        assert ctrl.readback()[10] == 4.0

    def test_empty_units(self):
        ctrl = _sim_ctrl(101)
        res = apply_units_via_controller(ctrl, np.zeros(N), [], 5.0, VMIN, VMAX)
        assert (res.ok, res.fail) == (0, 0)

    def test_failure_truthful(self):
        ctrl = FailingCtrl()
        units = [_ci(101, 3, 7)]
        res = apply_units_via_controller(ctrl, np.zeros(N), units, 5.0, VMIN, VMAX)
        assert (res.ok, res.fail) == (0, 1)


class TestApplyJoint:
    def test_sparse_send_across_controllers(self):
        dm = SimulatedMicroDM(ips=[101, 102])
        dm.open()
        pos_to_hw = {5: (101, 1), 50: (102, 1)}
        ip_to_ctrl = {101: 0, 102: 1}
        units = [_ci(101, 1, 5), _ci(102, 1, 50)]
        current_flat = np.zeros(100)

        new_flat, res = apply_joint(
            dm, current_flat, units, 5.0, VMIN, VMAX, pos_to_hw, ip_to_ctrl
        )
        assert res.ok == 2
        assert new_flat[0] == 5.0
        assert new_flat[50] == 5.0
        assert dm.readback(101)[0] == 5.0
        assert dm.readback(102)[0] == 5.0
        assert np.all(dm.readback(101)[1:] == 0.0)

    def test_unselected_positions_preserved(self):
        dm = SimulatedMicroDM(ips=[101, 102])
        dm.open()
        pos_to_hw = {5: (101, 1), 50: (102, 1)}
        ip_to_ctrl = {101: 0, 102: 1}
        units = [_ci(101, 1, 5)]
        current_flat = np.zeros(100)
        current_flat[50] = 9.0

        new_flat, res = apply_joint(dm, current_flat, units, 3.0, VMIN, VMAX, pos_to_hw, ip_to_ctrl)
        assert res.ok == 1
        assert new_flat[50] == 9.0  # preserved, not zeroed
        assert dm.readback(102)[0] == 9.0

    def test_unknown_position_skipped(self):
        dm = SimulatedMicroDM(ips=[101])
        dm.open()
        pos_to_hw = {5: (101, 1)}
        ip_to_ctrl = {101: 0}
        units = [_ci(101, 1, 9999)]  # physical position not in mapping
        _, res = apply_joint(dm, np.zeros(50), units, 5.0, VMIN, VMAX, pos_to_hw, ip_to_ctrl)
        assert res.ok == 0


# ---------------------------------------------------------------------------
# Send loops (threads)
# ---------------------------------------------------------------------------

def _loop_params(**overrides) -> dict:
    params = {
        "selection": ChannelSelection(channels=[0]),
        "voltage": 5.0,
        "amp": 5.0,
        "offset": 1.0,
        "freq": 1.0,
        "t0": time.time(),
        "vmin": VMIN,
        "vmax": VMAX,
        "dt": 0.02,
    }
    params.update(overrides)
    return params


class TestSendLoops:
    def test_hold_tick(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        hold_tick(ctrl, current, _loop_params())
        assert ctrl.readback()[0] == 5.0

    def test_sine_tick_at_t0(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        sine_tick(ctrl, current, _loop_params())  # t≈0 -> v ≈ offset
        assert ctrl.readback()[0] == pytest.approx(1.0, abs=0.1)

    def test_alt_tick_on(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        # elapsed ~0.1s with freq=1 -> half-period 0.5s -> ON phase
        alt_tick(ctrl, current, _loop_params(t0=time.time() - 0.1))
        assert ctrl.readback()[0] == 5.0

    def test_alt_tick_off(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        # elapsed ~0.9s -> phase toggled odd number of times -> OFF (0V)
        alt_tick(ctrl, current, _loop_params(t0=time.time() - 0.9))
        assert ctrl.readback()[0] == 0.0

    def test_run_loop_stops_via_event(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)
        stop = threading.Event()
        q: queue.Queue = queue.Queue()
        thread = start_loop(sine_tick, ctrl, current, _loop_params(), stop, q)
        time.sleep(0.12)
        stop_loop(stop)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert q.empty()  # no errors reported
        assert np.any(ctrl.readback() != 0.0)  # loop actually sent values

    def test_loop_error_reported_via_queue(self):
        ctrl = _sim_ctrl()
        current = np.zeros(N)

        def bad_tick(ctrl, current, p):
            raise RuntimeError("boom")

        stop = threading.Event()
        q: queue.Queue = queue.Queue()
        run_loop = start_loop  # alias to keep import list short
        thread = run_loop(bad_tick, ctrl, current, {"dt": 0.01}, stop, q)
        thread.join(timeout=2.0)
        assert stop.is_set()
        assert q.qsize() == 1
        kind, msg = q.get()
        assert kind == "error"
        assert "boom" in msg


class TestSendResult:
    def test_defaults(self):
        res = SendResult()
        assert (res.ok, res.fail) == (0, 0)
        assert res.failed_targets == []
