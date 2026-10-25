"""Thread-boundary tests for gui/slm/slm_calibration_ui (pure logic, no hardware).

Same contract shape as tests/ao_shaping/gui/dm/test_micro_dm_threads.py:
- T1 the calibration worker body never touches st.session_state (proven by swapping in a bomb)
- T2 the result travels back on the queue as ("result", obj) and the worker self-sets the stop Event
- T3 the error path puts ("error", msg) on the queue -- it used to write st.session_state.slm_cal_progress
- T4 the camera handle arrives as an argument: auto-exposure needs a non-None camera
"""

from __future__ import annotations

import json
import queue
import threading

import pytest

from ao_shaping.gui.slm import slm_calibration_ui as scu
from ao_shaping.gui.slm.slm_calibration_ui import _calibration_worker


class _SessionStateBomb:
    """Any attribute read or write fails the test that owns this fixture."""

    def __getattr__(self, name: str):
        raise AssertionError(f"thread body touched st.session_state.{name}")

    def __setattr__(self, name: str, value) -> None:
        raise AssertionError(f"thread body wrote st.session_state.{name}")


@pytest.fixture()
def no_session_state(monkeypatch):
    """Make every st.session_state access in the UI module raise while the test runs."""
    monkeypatch.setattr(scu.st, "session_state", _SessionStateBomb())


class _Calibrator:
    """Fake SantecCalibrator: records which entry point ran and with what kwargs."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result = result
        self._error = error

    def _finish(self, kind: str, kwargs: dict):
        self.calls.append((kind, kwargs))
        if self._error is not None:
            raise self._error
        return self._result

    def calibrate_with_background(self, **kwargs):
        return self._finish("background", kwargs)

    def calibrate_with_auto_exposure(self, **kwargs):
        return self._finish("auto_exposure", kwargs)


def _run(tmp_path, calib, *, camera=None, auto_exposure: bool = True):
    """Invoke _calibration_worker with fakes; returns (stop_event, out_q, progress_file)."""
    stop_event = threading.Event()
    out_q: queue.Queue = queue.Queue()
    progress_file = tmp_path / "progress.json"

    _calibration_worker(
        calib,
        (100, 800),  # grayscale_range
        20,  # step
        3,  # n_samples
        True,  # fine_search
        2,  # fine_step
        50,  # fine_range
        True,  # measure_background
        auto_exposure,
        80,  # target_min
        220,  # target_max
        str(progress_file),
        camera,
        stop_event,
        out_q,
    )
    return stop_event, out_q, progress_file


def _drain(out_q: queue.Queue) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    while True:
        try:
            items.append(out_q.get_nowait())
        except queue.Empty:
            return items


def _read_progress(progress_file) -> dict:
    with open(progress_file) as f:
        return json.load(f)


class TestSuccessPath:
    def test_result_on_queue_event_set_and_json_complete(
        self, tmp_path, no_session_state
    ) -> None:
        result = object()  # deliberately not JSON-serializable: it must ride the queue
        calib = _Calibrator(result=result)

        stop_event, out_q, progress_file = _run(tmp_path, calib, camera=object())

        assert stop_event.is_set(), "the worker must self-set the Event in its finally"
        assert _drain(out_q) == [("result", result)]

        progress = _read_progress(progress_file)
        assert progress["status"] == "complete"
        assert progress["percent"] == 100.0
        assert progress["message"] == "Calibration complete!"


class TestErrorPath:
    def test_error_on_queue_keeps_message_contract(self, tmp_path, no_session_state) -> None:
        calib = _Calibrator(error=RuntimeError("boom"))

        stop_event, out_q, _progress_file = _run(tmp_path, calib, camera=object())

        assert stop_event.is_set()
        assert _drain(out_q) == [("error", "Calibration failed: boom")]


class TestCameraArgument:
    def test_auto_exposure_branch_needs_camera_handle(self, tmp_path, no_session_state) -> None:
        calib = _Calibrator(result=object())
        _run(tmp_path, calib, camera=object(), auto_exposure=True)
        assert [kind for kind, _ in calib.calls] == ["auto_exposure"]
        kwargs = calib.calls[0][1]
        assert kwargs["auto_exposure"] is True
        assert kwargs["target_min"] == 80
        assert kwargs["target_max"] == 220

    def test_auto_exposure_without_camera_falls_back_to_background(
        self, tmp_path, no_session_state
    ) -> None:
        calib = _Calibrator(result=object())
        _run(tmp_path, calib, camera=None, auto_exposure=True)
        assert [kind for kind, _ in calib.calls] == ["background"]

    def test_auto_exposure_flag_off_selects_background(
        self, tmp_path, no_session_state
    ) -> None:
        calib = _Calibrator(result=object())
        _run(tmp_path, calib, camera=object(), auto_exposure=False)
        assert [kind for kind, _ in calib.calls] == ["background"]


class TestFixtureSanity:
    def test_bomb_actually_trips_on_session_state_access(self, no_session_state) -> None:
        """Prove the T1 detector is live, otherwise "never touches session_state" is vacuous."""
        with pytest.raises(AssertionError, match="thread body touched st.session_state"):
            _ = scu.st.session_state.slm_cal_result
        with pytest.raises(AssertionError, match="thread body wrote st.session_state"):
            scu.st.session_state.slm_cal_running = False
