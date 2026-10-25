"""Thread-boundary tests for gui/zernike/zernike_response_matrix_ui (pure logic, no hardware).

Same contract shape as tests/ao_shaping/gui/dm/test_micro_dm_threads.py:
- T1 the calibration thread body never touches st.session_state (proven by swapping in a bomb)
- T2 the result travels back on the queue as ("result", obj) and the thread self-sets the stop Event
- T3 the error path writes status "error" to the progress JSON and puts nothing on the queue
- T4 the per-mode progress callback writes to the file path handed in as a parameter
"""

from __future__ import annotations

import json
import queue
import threading

import numpy as np
import pytest

from ao_shaping.gui.zernike import zernike_response_matrix_ui as zrm
from ao_shaping.gui.zernike.zernike_response_matrix_ui import _run_calibration_thread


class _SessionStateBomb:
    """Any attribute read or write fails the test that owns this fixture."""

    def __getattr__(self, name: str):
        raise AssertionError(f"thread body touched st.session_state.{name}")

    def __setattr__(self, name: str, value) -> None:
        raise AssertionError(f"thread body wrote st.session_state.{name}")


@pytest.fixture()
def no_session_state(monkeypatch):
    """Make every st.session_state access in the UI module raise while the test runs."""
    monkeypatch.setattr(zrm.st, "session_state", _SessionStateBomb())


class _Result:
    """Sentinel calibration result; deliberately not JSON-serializable."""

    def __init__(self) -> None:
        self.matrix = "matrix"


def _run(
    tmp_path,
    monkeypatch,
    *,
    calibrate,
    verbose: bool = False,
):
    """Invoke _run_calibration_thread with fakes; returns (stop_event, out_q, progress_file, saved)."""
    stop_event = threading.Event()
    out_q: queue.Queue = queue.Queue()
    progress_file = tmp_path / "progress.json"
    save_path = tmp_path / "response"
    saved: list[tuple[object, str]] = []

    monkeypatch.setattr(zrm, "calibrate_zernike_response_matrix", calibrate)
    monkeypatch.setattr(
        zrm,
        "save_zernike_response_matrix",
        lambda result, path: saved.append((result, path)),
    )

    _run_calibration_thread(
        None,  # zslm
        None,  # wfs
        3,  # n_max
        0.1,  # magnitude
        1,  # n_cycles
        1,  # n_averages
        0.01,  # wait_time
        False,  # excluded_piston
        False,  # excluded_tip_tilt
        False,  # cancel_tile
        True,  # compute_inverses
        verbose,
        save_path,
        str(progress_file),
        stop_event,
        out_q,
    )
    return stop_event, out_q, progress_file, saved


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
        self, tmp_path, monkeypatch, no_session_state
    ) -> None:
        result = _Result()
        calls: list[dict] = []

        def fake_calibrate(**kwargs):
            calls.append(kwargs)
            return result

        stop_event, out_q, progress_file, saved = _run(
            tmp_path, monkeypatch, calibrate=fake_calibrate
        )

        assert stop_event.is_set(), "the thread must self-set the Event in its finally"
        assert _drain(out_q) == [("result", result)]
        assert saved == [(result, str(progress_file.parent / "response"))]

        progress = _read_progress(progress_file)
        assert progress["status"] == "complete"
        assert progress["percent"] == 100.0
        assert progress["message"] == "Calibration complete!"
        assert progress["result_path"] == str(progress_file.parent / "response")


class TestErrorPath:
    def test_error_status_in_json_no_result_on_queue(
        self, tmp_path, monkeypatch, no_session_state
    ) -> None:
        def fake_calibrate(**kwargs):
            raise RuntimeError("boom")

        stop_event, out_q, progress_file, saved = _run(
            tmp_path, monkeypatch, calibrate=fake_calibrate
        )

        assert stop_event.is_set()
        assert _drain(out_q) == [], "a failed calibration must not put a result"
        assert saved == []

        progress = _read_progress(progress_file)
        assert progress["status"] == "error"
        assert progress["message"] == "Calibration failed: boom"
        assert progress["percent"] == -1


class TestProgressCallback:
    def test_per_mode_progress_written_to_handed_in_file(
        self, tmp_path, monkeypatch, no_session_state
    ) -> None:
        seen: list[dict] = []

        def fake_calibrate(**kwargs):
            callback = kwargs["callback"]
            callback(0, 2, np.array([1.5]), None)
            with open(tmp_path / "progress.json") as f:
                seen.append(json.load(f))
            return _Result()

        stop_event, _out_q, progress_file, _saved = _run(
            tmp_path, monkeypatch, calibrate=fake_calibrate, verbose=True
        )

        mid = seen[0]
        assert "status" not in mid
        assert mid["current_mode"] == 0
        assert mid["total_modes"] == 2
        assert mid["percent"] == 50.0
        assert mid["message"] == "Calibrating mode 1/2"
        assert mid["mean_response"] == [1.5]
        assert mid["variance"] is None
        # final state stays the completion marker
        assert _read_progress(progress_file)["status"] == "complete"

    def test_callback_none_when_not_verbose(
        self, tmp_path, monkeypatch, no_session_state
    ) -> None:
        seen: list[dict] = []

        def fake_calibrate(**kwargs):
            seen.append(kwargs)
            return _Result()

        _run(tmp_path, monkeypatch, calibrate=fake_calibrate, verbose=False)
        assert seen[0]["callback"] is None


class TestFixtureSanity:
    def test_bomb_actually_trips_on_session_state_access(self, no_session_state) -> None:
        """Prove the T1 detector is live, otherwise "never touches session_state" is vacuous."""
        with pytest.raises(AssertionError, match="thread body touched st.session_state"):
            _ = zrm.st.session_state.zrm_calibration_result
        with pytest.raises(AssertionError, match="thread body wrote st.session_state"):
            zrm.st.session_state.zrm_calibration_running = False
