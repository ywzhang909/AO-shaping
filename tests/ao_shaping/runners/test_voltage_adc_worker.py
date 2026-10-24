"""Offline tests for :func:`_adc_worker`, the ADC acquisition thread.

The worker polls a DAQ in a background thread and pushes ``(timestamp, value)`` for
the runner to drain. The drain loop only counts what it *received*, so a read that
fails and is swallowed leaves a gap in the recorded voltage series with nothing to
attribute it to -- the run still looks clean. These tests pin that a failure is both
survivable and visible.

No hardware: ``adc`` is an injected parameter, so a fake object is enough.
"""

from __future__ import annotations

import queue
import threading
import time

from loguru import logger

from ao_shaping.runners.micro_drive.voltage_runner import _adc_worker


class _FlakyAdc:
    """DAQ stub; ``fail_first`` reads raise, later reads return a value."""

    def __init__(self, fail_first: int | None = None) -> None:
        self.calls = 0
        self._fail_first = fail_first

    def read_mean(self) -> float:
        self.calls += 1
        if self._fail_first is None or self.calls <= self._fail_first:
            raise RuntimeError("DAQ dropped off the bus")
        return 1.5


def _run_worker(adc: _FlakyAdc, window_s: float = 0.02):
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(
        target=_adc_worker, args=(adc, stop, q, time.time()), daemon=True
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        thread.start()
        time.sleep(window_s)
        stop.set()
        thread.join(timeout=2.0)
    finally:
        logger.remove(sink)
    return q, messages, thread


def test_a_failing_read_does_not_end_acquisition():
    """One bad read must not kill the thread: the run would lose every later sample."""
    adc = _FlakyAdc(fail_first=3)

    # Wider than 3 backoffs: this asserts polling continues, not a precise count.
    _q, _messages, thread = _run_worker(adc, window_s=0.08)

    assert not thread.is_alive(), "the worker died instead of retrying"
    assert adc.calls > 3, f"stopped polling after the first failures: {adc.calls} reads"


def test_a_failing_read_is_reported_and_the_total_is_counted():
    """The gaps have to be visible: the drain loop alone cannot reveal them."""
    adc = _FlakyAdc()  # every read fails

    _q, messages, _thread = _run_worker(adc)

    assert adc.calls > 0
    assert any("ADC 读取失败" in m for m in messages), messages
    summary = [m for m in messages if "共" in m and "次读取失败" in m]
    assert len(summary) == 1, messages
    assert f"共 {adc.calls} 次读取失败" in summary[0], (summary[0], adc.calls)


def test_a_hard_failure_does_not_spin_the_thread_hot():
    """A dropped DAQ must not be re-polled millions of times a second.

    Measured before the backoff existed: 40,800 attempts inside a 20 ms window, on a
    thread that shares the process with the voltage-update loop.
    """
    adc = _FlakyAdc()

    _q, _messages, _thread = _run_worker(adc)

    assert adc.calls < 50, f"error path is spinning hot: {adc.calls} attempts in 20 ms"


def test_throttle_lines_are_bounded_but_the_summary_total_is_exact():
    """Per-read lines are rate-limited; the summary count is not allowed to be fuzzy."""
    adc = _FlakyAdc()

    _q, messages, _thread = _run_worker(adc)

    per_read = [m for m in messages if "ADC 读取失败" in m]
    # Lines fire on the 1st failure and then every 100th, whatever the failure count.
    assert len(per_read) == 1 + adc.calls // 100, (per_read, adc.calls)


def test_a_recovered_read_still_queues_samples_and_still_warns():
    """Recovery must not erase the record of the samples that were lost."""
    adc = _FlakyAdc(fail_first=1)

    q, messages, _thread = _run_worker(adc)

    assert not q.empty(), "no sample was ever queued after the read recovered"
    assert any("ADC 读取失败" in m for m in messages), messages
