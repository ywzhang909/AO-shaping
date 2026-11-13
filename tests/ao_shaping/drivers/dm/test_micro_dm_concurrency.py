"""MicroDM multi-controller dispatch: concurrency and failure aggregation.

``MicroDM._apply_voltages`` must *start* a send on every controller before it
awaits any result (otherwise one slow/dead controller serialises the whole
mirror), must await every started send, and must report **all** failed
controllers in one :class:`MicroDMConnectionError` -- including controllers
whose ``send`` raises instead of returning ``False``.

No hardware and no sockets are involved here: the controller objects are
in-memory doubles synchronised with :class:`threading.Barrier`.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ao_shaping.drivers.dm.micro import (
    MAX_CHANNELS,
    MicroDM,
    MicroDMConnectionError,
)

# Barrier timeout: long enough for a loaded CI box, short enough that a
# sequential dispatcher fails this suite in seconds instead of hanging it.
_BARRIER_TIMEOUT_S = 5.0


class _FakeController:
    """Controller double whose ``send`` blocks until every peer has started.

    ``Barrier.wait`` only returns once *all* dispatches have called ``send``,
    so a dispatcher that awaits one controller before starting the next one
    trips ``BrokenBarrierError`` instead of quietly passing the test.
    """

    def __init__(
        self,
        ip: str,
        barrier: threading.Barrier,
        *,
        result: bool = True,
        exc: BaseException | None = None,
        finish_delay_s: float = 0.0,
    ) -> None:
        self.ip = ip
        self.started = threading.Event()
        self.finished = threading.Event()
        self._barrier = barrier
        self._result = result
        self._exc = exc
        self._finish_delay_s = finish_delay_s

    def set_all_voltage_array(self, data: np.ndarray):
        self.started.set()
        # Every peer must reach this line before any of us may continue.
        self._barrier.wait(timeout=_BARRIER_TIMEOUT_S)
        if self._finish_delay_s:
            time.sleep(self._finish_delay_s)
        if self._exc is not None:
            raise self._exc
        self.finished.set()
        return self._result

    def send(self, data: bytes):
        return self.set_all_voltage_array(data)


def _make_dm(n_controllers: int) -> MicroDM:
    ips = [f"192.168.0.{101 + i}" for i in range(n_controllers)]
    return MicroDM(ips=ips, use_wiring_map=False, safety_mode=False)


def _install(dm: MicroDM, controllers: list[_FakeController]) -> None:
    dm._controllers = controllers


class TestConcurrentDispatch:
    """All sends start before any is awaited; all completions are awaited."""

    def test_starts_every_send_before_awaiting_any(self) -> None:
        dm = _make_dm(2)
        barrier = threading.Barrier(2)
        ctrls = [
            _FakeController(f"192.168.0.{101 + i}", barrier) for i in range(2)
        ]
        _install(dm, ctrls)

        target = np.full(MAX_CHANNELS * 2, 7.5)
        result = dm._apply_voltages(target)

        # The barrier would have raised had the second send waited on the
        # first one, so both sends provably overlapped.
        assert all(c.started.is_set() for c in ctrls)
        assert all(c.finished.is_set() for c in ctrls)
        np.testing.assert_allclose(result, target)
        np.testing.assert_allclose(dm._last_voltages, target)

    def test_awaits_completion_of_every_started_send(self) -> None:
        dm = _make_dm(2)
        barrier = threading.Barrier(2)
        # Controller 2 fails fast; controller 1 is slow but successful.
        slow = _FakeController(
            "192.168.0.101", barrier, finish_delay_s=0.1
        )
        failing = _FakeController("192.168.0.102", barrier, result=False)
        _install(dm, [slow, failing])

        with pytest.raises(MicroDMConnectionError):
            dm._apply_voltages(np.full(MAX_CHANNELS * 2, 1.0))

        # The failure must not short-circuit the other controller: its send
        # had already started, and the caller waited for it to finish.
        assert slow.finished.is_set()

    def test_aggregates_false_returns_and_raised_errors_by_ip(self) -> None:
        dm = _make_dm(3)
        barrier = threading.Barrier(3)
        ok = _FakeController("192.168.0.101", barrier)
        refused = _FakeController("192.168.0.102", barrier, result=False)
        raising = _FakeController(
            "192.168.0.103", barrier, exc=RuntimeError("socket died")
        )
        _install(dm, [ok, refused, raising])

        with pytest.raises(MicroDMConnectionError) as excinfo:
            dm._apply_voltages(np.full(MAX_CHANNELS * 3, 5.0))

        message = str(excinfo.value)
        assert refused.ip in message
        assert raising.ip in message
        # A raised send is a failure of that controller, not a reason to stop
        # reporting the others.
        assert ok.ip not in message
        assert ok.finished.is_set()