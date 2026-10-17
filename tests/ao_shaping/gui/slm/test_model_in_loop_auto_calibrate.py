"""The run path must measure the bench itself.

A GUI that can only ask for a number is a GUI that only works on the optical path
those numbers came from.  The CLI already measures every session (panel scan →
sweep geometry) before ``optimize_slm_model_in_loop``; what is pinned here is
that the GUI run button reaches the **same** entry point, and that the rules for
when it may skip match the CLI's:

- ``sim``                 — a digital twin has no panel and no hardware geometry;
- ``reuse`` + a file      — a measured geometry is already at hand;
- ``keep`` + a result     — this session was already calibrated;
- otherwise (``auto``)    — measure, because otherwise the run is on an
  uncalibrated 实验系统 and the phase it produces means nothing.

Two further rules matter as much as the skip list:

1. a hand-entered optical-path value outranks the measured one (the three-state
   contract), so a calibration must never silently overwrite what the operator
   typed;
2. a rejected calibration **aborts** the run instead of falling through to the
   optimizer.

No hardware, no Streamlit server: ``st.session_state`` is swapped for a plain
dict and every device-touching entry point is replaced, the same convention
``test_shaping_session_contract.py`` uses.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.runners.runner_common import (
    SlmModelInLoopParams,
    build_model_in_loop_config,
)
from ao_shaping.runners.slm import model_in_loop_session as session


class _FakeSLM:
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

    def get_displayed_phase(self) -> tuple[None, str]:
        return None, "fake"


@pytest.fixture(autouse=True)
def state(monkeypatch):
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "spinner", lambda *a, **k: nullcontext(), raising=False)
    monkeypatch.setattr(sc, "Santec", lambda slm_number: _FakeSLM())
    monkeypatch.setattr(
        "ao_shaping.gui.slm.multi_slm_controller._stop_toggle", lambda n: None
    )
    return state


def _params(**over) -> SlmModelInLoopParams:
    base: dict[str, Any] = dict(
        n_rounds=1,
        seed=0,
        target_shape="square",
        target_side=60,
        w_uniformity=0.4,
        w_efficiency=0.6,
        probe_count=2,
        probe_spread=4.0,
        step_a_iterations=1,
        step_b_iterations=4,
        holdout_probe_count=1,
        freeform_holdout_count=0,
        min_holdout_correlation=0.5,
        region=32,
        panel_span_px=900.0,
        pupil_center="960,690",
        far_field_padding=4,
        cam_type="daheng",
        cam_id=0,
        exposure_time_ms=0.0,
        max_exposure_ms=0.4,
        cam_size=250,
        camera_pixel_um=2.2,
        slm_number=1,
    )
    base.update(over)
    return SlmModelInLoopParams(**base)


def _config(params: SlmModelInLoopParams, debug_dir: Path):
    return replace(build_model_in_loop_config(params, debug=False), debug_dir=str(debug_dir))


def _method() -> Any:
    return sc.ModelInLoopMethod("slm1", 1)


def _values(**over) -> dict[str, Any]:
    base: dict[str, Any] = {"bench_prefix": "slm1"}
    base.update(over)
    return base


@pytest.fixture
def measured(monkeypatch, tmp_path):
    """Replace the shared preflight; record how it was called."""
    calls: list[dict[str, Any]] = []
    # ``SlmModelInLoopConfig`` validates that the geometry file exists, exactly
    # as it does after a real sweep writes it.
    artefact = tmp_path / "bench_geometry.json"
    artefact.write_text('{"method": "sweep", "correlation": 0.82}', encoding="utf-8")

    def calibrate(params, config, save_dir, *, locate=True, geometry=True):
        calls.append(
            {
                "params": params,
                "config": config,
                "save_dir": Path(save_dir),
                "locate": locate,
                "geometry": geometry,
            }
        )
        return session.SessionCalibration(
            pupil_center=(1040, 620),
            panel_scan_response=0.31,
            panel_scan_max_pair_drift=0.01,
            panel_scan_session_drift=0.02,
            geometry_file=artefact,
            geometry_correlation=0.82,
            evidence_dir=Path(save_dir),
        )

    monkeypatch.setattr(session, "calibrate_session", calibrate)
    return calls


class TestSkipRules:
    def test_sim_never_measures(self, state, measured, tmp_path) -> None:
        params = _params(cam_type="sim")
        assert _method().auto_calibrate(params, _config(params, tmp_path), _values()) is None
        assert measured == []

    def test_reuse_with_a_file_never_measures(self, state, measured, tmp_path) -> None:
        geometry = tmp_path / "bench_geometry.json"
        geometry.write_text("{}", encoding="utf-8")
        state[bs.manual_key("slm1")] = {}
        state[f"slm1_{'bench_geometry_source'}"] = "reuse"
        params = _params(geometry_file=str(geometry))
        assert _method().auto_calibrate(params, _config(params, tmp_path), _values()) is None
        assert measured == []

    def test_keep_with_a_result_never_measures(self, state, measured, tmp_path) -> None:
        state["slm1_bench_geometry_source"] = "keep"
        bs.record_detection("slm1", {"pupil_center": [960, 690], "geometry_file": "x.json"})
        params = _params()
        assert _method().auto_calibrate(params, _config(params, tmp_path), _values()) is None
        assert measured == []

    def test_keep_without_a_result_still_measures(
        self, state, measured, tmp_path
    ) -> None:
        state["slm1_bench_geometry_source"] = "keep"
        params = _params()
        _method().auto_calibrate(params, _config(params, tmp_path), _values())
        assert len(measured) == 1


class TestAutoSource:
    def test_auto_measures_with_both_steps(self, state, measured, tmp_path) -> None:
        params = _params()
        result = _method().auto_calibrate(
            params, _config(params, tmp_path), _values()
        )
        assert result is not None
        assert len(measured) == 1
        call = measured[0]
        assert call["locate"] is True
        assert call["geometry"] is True

    def test_evidence_lands_next_to_the_run(self, state, measured, tmp_path) -> None:
        run_dir = tmp_path / "run"
        params = _params()
        _method().auto_calibrate(params, _config(params, run_dir), _values())
        assert measured[0]["save_dir"] == run_dir

    def test_measured_centre_reaches_the_config_and_params(
        self, state, measured, tmp_path
    ) -> None:
        params = _params()
        resolved_params, resolved_config = _method().auto_calibrate(
            params, _config(params, tmp_path), _values()
        )
        assert resolved_config.pupil_center == (1040, 620)
        assert resolved_params.pupil_center == "1040,620"
        assert resolved_config.geometry_file is not None

    def test_a_missing_debug_dir_never_falls_back_to_the_cwd(
        self, state, measured, tmp_path, monkeypatch
    ) -> None:
        """``str(Path(""))`` is ``"."``, so an unset ``debug_dir`` has to be
        tested on the raw string — otherwise the evidence lands in the current
        working directory instead of a run directory."""
        used: list[str] = []
        monkeypatch.setattr(
            "ao_shaping.utils.io.file.debug_run_dir",
            lambda root, topic: used.append(topic) or (tmp_path / "fallback"),
        )
        params = _params()
        config = replace(_config(params, tmp_path), debug_dir="")
        _method().auto_calibrate(params, config, _values())
        assert measured[0]["save_dir"] == tmp_path / "fallback"
        assert used == ["slm_model_in_loop"]


class TestManualWins:
    def test_a_hand_typed_centre_survives_the_calibration(
        self, state, measured, tmp_path
    ) -> None:
        state[bs.manual_key("slm1")] = {"pupil_center": "948,690"}
        params = _params(pupil_center="948,690")
        resolved_params, resolved_config = _method().auto_calibrate(
            params, _config(params, tmp_path), _values()
        )
        # Still measured...
        assert len(measured) == 1
        # ...but the operator's value is what the run uses.
        assert resolved_params.pupil_center == "948,690"
        assert resolved_config.pupil_center == (948, 690)

    def test_a_hand_typed_geometry_file_survives_too(
        self, state, measured, tmp_path
    ) -> None:
        mine = tmp_path / "mine.json"
        mine.write_text("{}", encoding="utf-8")
        state[bs.manual_key("slm1")] = {"geometry_file": str(mine)}
        params = _params(geometry_file=str(mine))
        _, resolved_config = _method().auto_calibrate(
            params, _config(params, tmp_path), _values()
        )
        assert resolved_config.geometry_file == str(mine)


class TestRejectedCalibrationAborts:
    def test_auto_calibrate_raises(self, state, monkeypatch, tmp_path) -> None:
        def boom(*_a, **_k):
            raise RuntimeError("session geometry failed validation")

        monkeypatch.setattr(session, "calibrate_session", boom)
        params = _params()
        with pytest.raises(RuntimeError, match="已中止本次运行"):
            _method().auto_calibrate(params, _config(params, tmp_path), _values())

    def test_run_never_reaches_the_optimizer(self, state, monkeypatch, tmp_path) -> None:
        """A phase shaped on an uncalibrated 实验系统 is meaningless, so a
        rejected preflight must stop the run — not degrade into shaping blind."""

        def boom(*_a, **_k):
            raise RuntimeError("panel scan rejected")

        monkeypatch.setattr(session, "calibrate_session", boom)
        params = _params()
        config = _config(params, tmp_path)
        method = _method()
        executed: list[Any] = []
        monkeypatch.setattr(method, "validate", lambda values: None)
        monkeypatch.setattr(method, "build_params", lambda values: params)
        monkeypatch.setattr(method, "make_config", lambda params, debug: config)
        monkeypatch.setattr(method, "prepare_config", lambda config, values: None)
        monkeypatch.setattr(method, "execute", lambda config: executed.append(config))

        outcome = method.run(_values(), debug=False)

        assert outcome is None
        assert executed == []