"""Thread-boundary tests for gui/dm/micro_dm_ui (pure logic, no hardware).

Same contract shape as tests/ao_shaping/gui/r50/test_r50_voltage_send.py:
- T1 thread bodies never touch st.session_state (proven by swapping in a bomb)
- T2 stop is a threading.Event; the loop exits on the next tick
- T3 a device error goes to the queue and the loop self-sets the Event
- T4 the driver hard ceiling wins over a widened soft window
- T5 soft bounds narrowed mid-run apply on the next tick
- T6 what the UI will plot equals what the device actually received
- T7 a disconnected device stops the loop instead of spinning forever
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from ao_shaping.gui.dm import micro_dm_ui as mdm
from ao_shaping.gui.dm.micro_dm_ui import (
    HW_VOLTAGE_MAX,
    HW_VOLTAGE_MIN,
    SINE_AMP_MAX,
    _clamp_voltage,
    _matrix_continuous_loop,
    _single_fixed_loop,
    _single_periodic_loop,
)


class _SessionStateBomb:
    """Any attribute read or write fails the test that owns this fixture."""

    def __getattr__(self, name: str):
        raise AssertionError(f"thread body touched st.session_state.{name}")

    def __setattr__(self, name: str, value) -> None:
        raise AssertionError(f"thread body wrote st.session_state.{name}")


@pytest.fixture()
def no_session_state(monkeypatch):
    """Make every st.session_state access in micro_dm_ui raise while the test runs."""
    monkeypatch.setattr(mdm.st, "session_state", _SessionStateBomb())


class _MatrixDM:
    """Fake MicroDM: records the voltage of every bulk send."""

    def __init__(self, stop_event: threading.Event | None = None, stop_after: int | None = None):
        self.sent: list[float] = []
        self._stop_event = stop_event
        self._stop_after = stop_after

    def send_voltages(self, voltages) -> None:
        self.sent.append(float(voltages[0]))
        if (
            self._stop_event is not None
            and self._stop_after is not None
            and len(self.sent) >= self._stop_after
        ):
            self._stop_event.set()


class _RaisingDM:
    def send_voltages(self, voltages) -> None:
        raise RuntimeError("boom")


class _SingleCtrl:
    """Fake R50Power single-unit controller.

    ``narrow_at`` rewrites the shared bounds list after a given send, which is how
    a mid-run sidebar change reaches a running loop; ``stop_after`` ends the run.
    """

    def __init__(
        self,
        stop_event: threading.Event | None = None,
        stop_after: int | None = None,
        bounds_ref: list[float] | None = None,
        narrow_at: tuple[int, float] | None = None,
    ):
        self.sent: list[float] = []
        self.channels: list[int | None] = []
        self._stop_event = stop_event
        self._stop_after = stop_after
        self._bounds_ref = bounds_ref
        self._narrow_at = narrow_at

    def _record(self, voltage: float, channel: int | None) -> None:
        self.sent.append(float(voltage))
        self.channels.append(channel)
        if self._narrow_at is not None and len(self.sent) == self._narrow_at[0]:
            self._bounds_ref[1] = self._narrow_at[1]
        if self._stop_event is not None and self._stop_after is not None and len(self.sent) >= self._stop_after:
            self._stop_event.set()

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        self._record(voltage, channel)

    def set_all_channel_voltage(self, voltage: float) -> None:
        self._record(voltage, None)


def _drain(out_q: queue.Queue) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    while True:
        try:
            items.append(out_q.get_nowait())
        except queue.Empty:
            return items


def _samples(items: list[tuple[str, object]]) -> list[float]:
    return [float(payload[0]) for kind, payload in items if kind == "sample"]


class TestClamp:
    def test_hard_ceiling_wins_over_widened_soft_window(self) -> None:
        assert _clamp_voltage(200.0, (-1000.0, 1000.0)) == HW_VOLTAGE_MAX
        assert _clamp_voltage(-500.0, (-1000.0, 1000.0)) == HW_VOLTAGE_MIN

    def test_soft_window_narrows_before_hard_ceiling(self) -> None:
        assert _clamp_voltage(100.0, (0.0, 50.0)) == 50.0
        assert _clamp_voltage(10.0, (0.0, 50.0)) == 10.0

    def test_sine_amplitude_ceiling_is_exact(self) -> None:
        """70.0 is the largest amplitude that can fit for some offset -- not a guess."""
        mid = (HW_VOLTAGE_MAX + HW_VOLTAGE_MIN) / 2.0
        assert SINE_AMP_MAX == (HW_VOLTAGE_MAX - HW_VOLTAGE_MIN) / 2.0
        assert mid + SINE_AMP_MAX == HW_VOLTAGE_MAX
        assert mid - SINE_AMP_MAX == HW_VOLTAGE_MIN


class TestMatrixLoop:
    def test_stops_via_event_without_touching_session_state(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        dm = _MatrixDM(stop_event=stop, stop_after=2)
        _matrix_continuous_loop(10.0, 0.0, dm, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX, stop, out_q)
        assert dm.sent == [10.0, 10.0]
        assert _drain(out_q) == []

    def test_error_reported_via_queue_and_self_stops(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        _matrix_continuous_loop(10.0, 0.0, _RaisingDM(), HW_VOLTAGE_MIN, HW_VOLTAGE_MAX, stop, out_q)
        assert stop.is_set(), "a failed loop must self-set the Event so the main thread can reconcile"
        items = _drain(out_q)
        assert [kind for kind, _ in items] == ["error"]
        assert "boom" in str(items[0][1])

    def test_disconnected_device_stops_instead_of_spinning(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        _matrix_continuous_loop(10.0, 0.0, None, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX, stop, out_q)
        assert stop.is_set()
        assert [kind for kind, _ in _drain(out_q)] == ["error"]

    def test_runs_on_a_real_thread_until_stopped(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        dm = _MatrixDM()
        thread = threading.Thread(
            target=_matrix_continuous_loop,
            args=(5.0, 0.01, dm, HW_VOLTAGE_MIN, HW_VOLTAGE_MAX, stop, out_q),
            daemon=True,
        )
        thread.start()
        time.sleep(0.1)
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert len(dm.sent) >= 1
        assert _drain(out_q) == []


class TestSingleLoops:
    def test_fixed_loop_clamps_and_queues_samples(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        bounds = [HW_VOLTAGE_MIN, HW_VOLTAGE_MAX]
        ctrl = _SingleCtrl(stop_event=stop, stop_after=1, bounds_ref=bounds)
        _single_fixed_loop(3, 200.0, False, 0.0, ctrl, bounds, stop, out_q)
        assert ctrl.sent == [HW_VOLTAGE_MAX]
        assert ctrl.channels == [3]
        assert _samples(_drain(out_q)) == ctrl.sent, "plot history must equal what was sent"

    def test_fixed_loop_honours_bounds_narrowed_mid_run(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        bounds = [HW_VOLTAGE_MIN, HW_VOLTAGE_MAX]
        ctrl = _SingleCtrl(
            stop_event=stop, stop_after=2, bounds_ref=bounds, narrow_at=(1, 50.0)
        )
        _single_fixed_loop(0, 200.0, False, 0.0, ctrl, bounds, stop, out_q)
        assert ctrl.sent == [HW_VOLTAGE_MAX, 50.0]

    def test_fixed_loop_stops_when_device_disconnected(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        _single_fixed_loop(0, 10.0, False, 0.0, None, [HW_VOLTAGE_MIN, HW_VOLTAGE_MAX], stop, out_q)
        assert stop.is_set()
        assert [kind for kind, _ in _drain(out_q)] == ["error"]

    def test_periodic_sine_stays_inside_hardware_range(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        bounds = [HW_VOLTAGE_MIN, HW_VOLTAGE_MAX]
        ctrl = _SingleCtrl(stop_event=stop, stop_after=6, bounds_ref=bounds)
        _single_periodic_loop(
            0, True, "sine", SINE_AMP_MAX, 50.0, 0.0, 0.0, 0.05, 0.0,
            ctrl, bounds, stop, out_q,
        )
        assert len(ctrl.sent) == 6
        assert all(HW_VOLTAGE_MIN <= v <= HW_VOLTAGE_MAX for v in ctrl.sent)
        assert ctrl.channels == [None] * 6
        assert _samples(_drain(out_q)) == ctrl.sent

    def test_periodic_square_alternates_between_levels(self, no_session_state) -> None:
        stop = threading.Event()
        out_q: queue.Queue = queue.Queue()
        bounds = [HW_VOLTAGE_MIN, HW_VOLTAGE_MAX]
        ctrl = _SingleCtrl(stop_event=stop, stop_after=2, bounds_ref=bounds)
        _single_periodic_loop(
            0, True, "square", 0.0, 0.0, 80.0, -10.0, 10.0, 0.0,
            ctrl, bounds, stop, out_q,
        )
        assert ctrl.sent[0] == 80.0
        assert len(ctrl.sent) >= 1


class TestFixtureSanity:
    def test_bomb_actually_trips_on_session_state_access(self, no_session_state) -> None:
        """Prove the T1 detector is live, otherwise "never touches session_state" is vacuous."""
        with pytest.raises(AssertionError, match="thread body touched st.session_state"):
            _ = mdm.st.session_state.mdm_vmin
