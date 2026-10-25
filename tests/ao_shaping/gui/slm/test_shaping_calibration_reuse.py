"""The GUI run **reuses** the sidebar calibration; it never measures the bench.

`ModelInLoopMethod.auto_calibrate` / `_merge_calibration` and the base
``auto_calibrate`` hook were deleted (2026-10-24): the sidebar 「标定 SLM N 光路」
button (:func:`bench_settings.run_session_calibration` →
``runners.slm.model_in_loop_session.calibrate_session``) is the only measuring
path, and a run consumes its result.  Two consequences are pinned here:

1. a whole ``ModelInLoopMethod.run(...)`` must complete with every measurement
   entry point replaced by a raised ``AssertionError`` — if the run path ever
   calls one again, this file fails instead of silently re-measuring a 实验系统;
2. hardware with no reusable sweep geometry is **refused before a device is
   touched**, with a message naming the button that fixes it.  The alternative
   (running and falling back to the abandoned speckle route, measured
   correlation ≈0.07 against the 0.50 floor) costs minutes and ends in failure.

The three resolution rules of :func:`bench_settings.reusable_geometry_file` live
in ``test_bench_settings.py::TestReusableGeometryFile`` (that module owns the
function); what is checked here is only that the resolved path actually reaches
the run gate.

No hardware, no Streamlit server: ``st.session_state`` is swapped for a plain
dict and the SLM factory is replaced, the same convention
``test_shaping_session_contract.py`` uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.runners.slm.model_in_loop_session import SessionCalibration


class _FakeSLM:
    """Minimal stand-in for an open Santec handle (session-state sync reads it)."""

    def __init__(self) -> None:
        self.is_open = True
        self.close_calls = 0
        self.displayed: list[Any] = []
        self.video_mode = 0
        self.shift_x = 0
        self.shift_y = 0
        self.wavelength = 1064
        self.Panel_Res = (1920, 1200)
        self.Pixel_Size_um = 7.8
        self.Gray_Scale_bits = 10
        self._use_120hz = False
        self._serial_number = "FAKE"
        self.slm_number = 1

    def close(self) -> None:
        self.close_calls += 1
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def load_config(self) -> dict[str, Any]:
        return {}

    def create_phase_from_array(self, phase: Any) -> Any:
        import numpy as np

        return np.asarray(phase, dtype="uint16")

    def display_data(self, phase_gray: Any, *, memory_mode: int) -> None:
        self.displayed.append(phase_gray)

    def get_displayed_phase(self) -> tuple[None, str]:
        return None, "fake"


@pytest.fixture(autouse=True)
def state(monkeypatch):
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(sc, "Santec", lambda slm_number: _FakeSLM())
    monkeypatch.setattr(
        "ao_shaping.gui.slm.multi_slm_controller._stop_toggle", lambda n: None
    )
    return state


def _connect(state: dict[str, Any], slm_num: int = 1) -> _FakeSLM:
    prefix = f"slm{slm_num}"
    slm = _FakeSLM()
    slm.slm_number = slm_num
    state[prefix] = slm
    state[f"{prefix}_connected"] = True
    return slm


def _geometry(tmp_path: Path) -> Path:
    """A measured sweep artefact (region matches the MIL model grid)."""
    path = tmp_path / "bench_geometry.json"
    path.write_text(
        json.dumps({"method": "sweep", "region": 64, "correlation": 0.82}),
        encoding="utf-8",
    )
    return path


def _values(tmp_path: Path, **over: Any) -> dict[str, Any]:
    """A valid hardware MIL input set: widget defaults + the fields validate reads."""
    values = dict(sc.MIL_WIDGET_DEFAULTS)
    values.update(
        w_uniformity=0.4,
        w_efficiency=0.6,
        min_holdout_correlation=0.5,
        cam_type="daheng",
        cam_id=0,
        exposure_time_ms="auto",
        max_exposure_ms=0.4,
        cam_size=250,
        region=64,
        geometry_file=str(_geometry(tmp_path)),
        bench_prefix="slm1",
        slm_number=1,
    )
    values.update(over)
    return values


@pytest.fixture
def no_measurement(monkeypatch):
    """Replace every measuring entry point with a hard failure + a call log."""
    from ao_shaping.runners.slm import model_in_loop_session as session

    measured: list[str] = []

    def _boom(name: str):
        def _stub(*args: Any, **kwargs: Any) -> Any:
            measured.append(name)
            raise AssertionError("run path measured the bench")

        return _stub

    monkeypatch.setattr(session, "calibrate_session", _boom("calibrate_session"))
    # Defensive: the sidebar entry point must not be reachable from a run either.
    monkeypatch.setattr(
        bs, "run_session_calibration", _boom("run_session_calibration")
    )
    return measured


class TestRunNeverMeasures:
    def test_a_full_run_reuses_the_session_calibration(
        self, state, no_measurement, monkeypatch, tmp_path
    ) -> None:
        """A whole run completes while every measuring path raises."""
        _connect(state)
        method = sc.ModelInLoopMethod("slm1", 1)
        params = SimpleNamespace(kind="params")
        config = SimpleNamespace(kind="config")
        executed: list[Any] = []

        # Only the algorithm-specific edges are stubbed — ``validate``, ``run``,
        # ``save`` and the device handover stay real, because the handover is
        # exactly where a run-time calibration would have been invoked.
        monkeypatch.setattr(method, "build_params", lambda values: params)
        monkeypatch.setattr(method, "make_config", lambda p, debug: config)
        monkeypatch.setattr(method, "prepare_config", lambda cfg, values: None)
        monkeypatch.setattr(
            method,
            "execute",
            lambda cfg: executed.append(cfg)
            or sc.ShapingOutcome(panel_phase=None, best_score=0.5, note="stub"),
        )

        outcome = method.run(_values(tmp_path), debug=False)

        assert no_measurement == []
        assert executed == [config]
        assert outcome is not None
        assert outcome.best_score == pytest.approx(0.5)
        # The handle is back in the GUI's hands, so the page still owns a device.
        assert state["slm1"] is not None
        assert state["slm1_connected"] is True
        assert bs.bench_busy() is False


class TestGeometryGate:
    def test_hardware_without_a_reusable_geometry_is_refused(
        self, state, monkeypatch, tmp_path
    ) -> None:
        """Refused before any device call, with the fixing action in the message."""
        errors: list[str] = []
        opened: list[Any] = []
        monkeypatch.setattr(st, "error", lambda text, *a, **k: errors.append(str(text)))
        monkeypatch.setattr(
            sc, "Santec", lambda *a, **k: opened.append((a, k))
        )

        values = _values(tmp_path, geometry_file="", cam_type="daheng")
        message = sc.ModelInLoopMethod("slm1", 1).validate(values)

        assert isinstance(message, str)
        assert "没有可复用的 sweep 几何" in message
        assert "标定 SLM 1 光路" in message
        assert errors == []
        assert opened == []

    def test_sim_needs_no_geometry(self, tmp_path) -> None:
        values = _values(tmp_path, cam_type="sim", geometry_file="")
        assert sc.ModelInLoopMethod("slm1", 1).validate(values) is None

    def test_a_geometry_file_satisfies_the_gate(self, tmp_path) -> None:
        assert sc.ModelInLoopMethod("slm1", 1).validate(_values(tmp_path)) is None

    def test_a_blank_geometry_still_blocks_hardware(self, tmp_path) -> None:
        """Whitespace is the same state as an empty box, not a path."""
        values = _values(tmp_path, geometry_file="   ")
        message = sc.ModelInLoopMethod("slm1", 1).validate(values)
        assert isinstance(message, str) and "没有可复用的 sweep 几何" in message


class TestUncalibratedHint:
    def test_the_hint_is_a_caption_not_an_error(
        self, state, monkeypatch, tmp_path
    ) -> None:
        """Never calibrated → still runnable, but the fallback value is named."""
        captions: list[str] = []
        errors: list[str] = []
        original = st.caption

        def _capture(text=None, *a, **k):
            captions.append(str(text))
            return original(text, *a, **k)

        monkeypatch.setattr(st, "caption", _capture)
        monkeypatch.setattr(st, "error", lambda text, *a, **k: errors.append(str(text)))

        assert bs.detected_state("slm1") == {}
        assert sc.ModelInLoopMethod("slm1", 1).validate(_values(tmp_path)) is None

        assert any("兜底实测值" in text for text in captions)
        assert errors == []

    def test_a_calibrated_session_gets_no_hint(self, state, monkeypatch, tmp_path) -> None:
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        captions: list[str] = []
        monkeypatch.setattr(st, "caption", lambda text=None, *a, **k: captions.append(str(text)))

        assert sc.ModelInLoopMethod("slm1", 1).validate(_values(tmp_path)) is None
        assert captions == []


class TestMergeHandsOverTheReusableGeometry:
    """The run only sees the sidebar's resolution of "which geometry file"."""

    def test_the_reuse_box_path_reaches_the_gate(self, state, tmp_path) -> None:
        geometry = _geometry(tmp_path)
        state["slm1_bench_geometry_source"] = "reuse"
        state["slm1_bench_geometry_file"] = str(geometry)

        values = _values(tmp_path, geometry_file="")
        bs.merge_bench_values(values, "slm1", 1, "model-in-loop")

        assert values["geometry_file"] == str(geometry)
        assert sc.ModelInLoopMethod("slm1", 1).validate(values) is None

    def test_model_in_loop_and_transfer_get_the_reusable_file(self, state, tmp_path) -> None:
        geometry = _geometry(tmp_path)
        state["slm1_bench_geometry_source"] = "reuse"
        state["slm1_bench_geometry_file"] = str(geometry)

        for method_key in ("model-in-loop", "transfer"):
            values: dict[str, Any] = {"geometry_file": "operator-typed.json"}
            bs.merge_bench_values(values, "slm1", 1, method_key)
            assert values["geometry_file"] == str(geometry), method_key

    def test_gs_refine_keeps_its_own_geometry_field(self, state, tmp_path) -> None:
        """gs-refine has no forward model, so it must not inherit one."""
        geometry = _geometry(tmp_path)
        state["slm1_bench_geometry_source"] = "reuse"
        state["slm1_bench_geometry_file"] = str(geometry)

        values: dict[str, Any] = {"geometry_file": "operator-typed.json"}
        bs.merge_bench_values(values, "slm1", 1, "gs-refine")

        assert values["geometry_file"] == "operator-typed.json"


def _calibration(center, geometry) -> SessionCalibration:
    return SessionCalibration(
        pupil_center=center,
        geometry_file=geometry,
        geometry_correlation=0.81,
        evidence_dir="data/debug/slm_bench_calibration/x",
        notes=("fake",),
    )


class TestPublicationPinsTheReuseSource:
    """After the button measures, the run must reuse it — that is the whole point."""

    def test_a_backfilled_geometry_pins_the_reuse_source(self, state, tmp_path) -> None:
        geometry = _geometry(tmp_path)
        resolved = SimpleNamespace(pupil_center=(1000, 700), geometry_file=str(geometry))

        bs.publish_session_calibration(
            "slm1",
            resolved,
            _calibration((1000, 700), geometry),
            source="auto",
            save_dir="data/debug/slm_bench_calibration/x",
        )

        # Queued for the next render (widgets are locked during a callback)…
        pending = bs.pending_state("slm1")
        assert pending["slm1_bench_geometry_file"] == str(geometry)
        assert pending["slm1_bench_geometry_source"] == "reuse"

        # …and after the flush the source is really pinned, so the next run
        # resolves this file without measuring again.
        bs.flush_pending("slm1")
        assert bs.bench_geometry_source("slm1") == "reuse"
        assert bs.reusable_geometry_file("slm1") == str(geometry)

    def test_a_hand_typed_centre_is_still_not_overwritten(self, state, tmp_path) -> None:
        """The three-state invariant survives the removal of run-time calibration."""
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        state["slm1_bench_pupil_center"] = "948,690"
        bs.sync_manual_states("slm1")
        assert bs.manual_fields("slm1") == ("pupil_center",)

        geometry = _geometry(tmp_path)
        resolved = SimpleNamespace(pupil_center=(1000, 700), geometry_file=str(geometry))
        bs.publish_session_calibration(
            "slm1",
            resolved,
            _calibration((1000, 700), geometry),
            source="auto",
            save_dir="data/debug/slm_bench_calibration/x",
        )

        pending = bs.pending_state("slm1")
        assert "slm1_bench_pupil_center" not in pending
        assert "slm1_bench_geometry_file" in pending
        bs.flush_pending("slm1")

        assert state["slm1_bench_pupil_center"] == "948,690"
        # The measurement itself is still recorded, so the delta stays visible.
        assert bs.detected_state("slm1")["pupil_center"] == [1000, 700]
