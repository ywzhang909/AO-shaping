"""The shared session-preflight entry point, exercised without hardware.

``calibrate_session`` is the one place that locates the beam on the panel and
measures the session geometry, so both the CLI and the GUI go through it. WHAT is
pinned here
--------------
1. Skipped steps leave their fields ``None`` (and never overwrite the caller's
   values in :func:`apply_session_calibration`) -- a GUI must be able to run the
   geometry step alone.
2. Grid choice follows the CLI's rule: a session that brings no geometry file
   sweeps the whole 5x4 panel with a radius floored at 220 px; otherwise the scan
   is a 3x3 neighbourhood of the caller's centre.
3. The sweep aims at the centre measured a moment earlier, not the original
   guess.
4. Failures are plain exceptions, and unusable arguments are distinguishable
   from a rejected measurement (:class:`SessionPreflightParamError`).
5. ``summary_lines()`` is Chinese and obeys ``docs/terminology_glossary.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig
from ao_shaping.runners.runner_common import SlmModelInLoopParams
from ao_shaping.runners.slm import model_in_loop_session as session
from ao_shaping.runners.slm.model_in_loop_session import (
    SessionPreflightParamError,
    apply_session_calibration,
    calibrate_session,
)
from ao_shaping.tools.slm.panel import panel_locate


def _params(**overrides) -> SlmModelInLoopParams:
    return SlmModelInLoopParams(**overrides)


def _config(**overrides) -> SlmModelInLoopConfig:
    base = dict(cam_type="daheng", region=64, far_field_padding=8, pupil_center=(960, 600))
    base.update(overrides)
    return SlmModelInLoopConfig(**base)


@pytest.fixture
def fake_sweep(monkeypatch):
    """Replace the runbook subprocess; record the stages and write the artefact."""
    calls: list[list[str]] = []

    def run(command, *, cwd, check):
        calls.append(list(command))
        if command[-1] == "calibrate":
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "bench_geometry.json").write_text(
                json.dumps({"method": "sweep", "correlation": 0.82}), encoding="utf-8"
            )

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.fixture
def fake_scan(monkeypatch):
    """Replace the panel-locate probe; record the params it was handed."""
    seen: dict = {}

    def scan(params, *, strict, max_session_drift):
        seen["params"] = params
        seen["strict"] = strict
        seen["session_limit"] = max_session_drift
        return panel_locate.PanelLocateResult((960, 720), -0.1, 0.006, 0.012, 0)

    monkeypatch.setattr(panel_locate, "scan_panel", scan)
    return seen


class TestSkipFlags:
    def test_both_steps_off_leave_every_measurement_none(
        self, tmp_path, fake_scan, fake_sweep
    ):
        calibration = calibrate_session(
            _params(), _config(), tmp_path / "run", locate=False, geometry=False
        )

        assert calibration.pupil_center is None
        assert calibration.panel_scan_response is None
        assert calibration.panel_scan_max_pair_drift is None
        assert calibration.panel_scan_session_drift is None
        assert calibration.geometry_file is None
        assert calibration.geometry_correlation is None
        assert calibration.evidence_dir == tmp_path / "run"
        assert not fake_scan and not fake_sweep, "a skipped step must not touch hardware"
        assert len(calibration.notes) == 2

    def test_skipping_leaves_the_caller_config_untouched(self, tmp_path, fake_scan, fake_sweep):
        params, config = _params(), _config()
        calibration = calibrate_session(
            params, config, tmp_path / "run", locate=False, geometry=False
        )
        new_params, new_config = apply_session_calibration(params, config, calibration)

        assert new_params == params
        assert new_config.geometry_file is None
        assert new_config.pupil_center == (960, 600)


class TestLocate:
    def test_no_geometry_file_sweeps_the_whole_panel(self, tmp_path, fake_scan, fake_sweep):
        calibration = calibrate_session(
            _params(), _config(), tmp_path / "run", locate=True, geometry=False
        )

        scan_params = fake_scan["params"]
        assert (scan_params.grid_xs, scan_params.grid_ys) == (
            "240,600,960,1320,1680", "220,480,720,980",
        )
        assert scan_params.patch_radius == 220, "radius is floored at 220 px"
        assert (scan_params.frames, scan_params.seed) == (2, 0)
        assert scan_params.debug_dir == str(tmp_path / "run" / "panel_scan")
        assert fake_scan["strict"] is True
        assert fake_scan["session_limit"] == 0.05
        assert calibration.pupil_center == (960, 720)
        assert calibration.panel_scan_response == -0.1
        assert calibration.panel_scan_max_pair_drift == 0.006
        assert calibration.panel_scan_session_drift == 0.012
        assert calibration.geometry_file is None
        assert not fake_sweep

    def test_a_larger_radius_wins_over_the_floor(self, tmp_path, fake_scan, fake_sweep):
        calibrate_session(
            _params(panel_scan_radius_px=400), _config(), tmp_path / "run",
            locate=True, geometry=False,
        )
        assert fake_scan["params"].patch_radius == 400

    def test_known_geometry_file_scans_a_3x3_neighbourhood(
        self, tmp_path, fake_scan, fake_sweep
    ):
        params = _params(geometry_file="bench_geometry.json", panel_scan_step_px=120)
        calibrate_session(params, _config(), tmp_path / "run", locate=True, geometry=False)

        scan_params = fake_scan["params"]
        assert (scan_params.grid_xs, scan_params.grid_ys) == (
            "840,960,1080", "480,600,720",
        )
        assert scan_params.patch_radius == 150, "no floor without a whole-panel sweep"

    def test_3x3_grid_off_the_panel_is_a_parameter_error(
        self, tmp_path, fake_scan, fake_sweep
    ):
        params = _params(geometry_file="bench_geometry.json", panel_scan_step_px=120)
        params.pupil_center = "1900,1200"

        with pytest.raises(SessionPreflightParamError, match="3x3 grid"):
            calibrate_session(
                params, _config(), tmp_path / "run", locate=True, geometry=False
            )
        assert not fake_scan, "an unusable grid must not reach the devices"

    def test_sim_camera_is_a_parameter_error_not_a_measurement_rejection(
        self, tmp_path, fake_scan, fake_sweep
    ):
        with pytest.raises(SessionPreflightParamError, match="hardware camera") as err:
            calibrate_session(
                _params(cam_type="sim", geometry_file="bench_geometry.json"),
                _config(cam_type="sim"), tmp_path / "run", locate=True, geometry=False,
            )
        assert err.value.param_hint == "--panel-scan"
        assert not fake_scan

    def test_weak_response_is_plain_runtime_error(self, tmp_path, monkeypatch, fake_sweep):
        monkeypatch.setattr(
            panel_locate, "scan_panel",
            lambda *a, **k: panel_locate.PanelLocateResult((960, 720), 0.01, 0.0, 0.0, 0),
        )
        with pytest.raises(RuntimeError, match="too weak"):
            calibrate_session(
                _params(), _config(), tmp_path / "run", locate=True, geometry=False
            )

    def test_a_rejected_scan_keeps_its_own_value_error(self, tmp_path, monkeypatch, fake_sweep):
        def drift(*_args, **_kwargs):
            raise ValueError("session flat drift 8.0% exceeds 5.0%")

        monkeypatch.setattr(panel_locate, "scan_panel", drift)
        with pytest.raises(ValueError, match="session flat drift") as err:
            calibrate_session(
                _params(), _config(), tmp_path / "run", locate=True, geometry=False
            )
        assert not isinstance(err.value, SessionPreflightParamError)


class TestGeometry:
    def test_sweep_runs_with_the_measured_centre(self, tmp_path, fake_scan, fake_sweep):
        params = _params()
        config = _config(pupil_center=(960, 600))
        calibration = calibrate_session(
            params, config, tmp_path / "run", locate=True, geometry=True
        )

        assert [call[-1] for call in fake_sweep] == ["sweep", "calibrate"]
        assert fake_sweep[0][fake_sweep[0].index("--pupil-center") + 1] == "960,720"
        assert calibration.geometry_file == tmp_path / "run" / "session_geometry" / "bench_geometry.json"
        assert calibration.geometry_correlation == 0.82
        assert all(call[call.index("--exposure-ms") + 1] == "0" for call in fake_sweep)

    def test_locate_off_keeps_the_caller_centre(self, tmp_path, fake_scan, fake_sweep):
        calibration = calibrate_session(
            _params(), _config(pupil_center=(960, 600)), tmp_path / "run",
            locate=False, geometry=True,
        )
        assert fake_sweep[0][fake_sweep[0].index("--pupil-center") + 1] == "960,600"
        assert not fake_scan
        assert calibration.pupil_center is None

    @pytest.mark.parametrize(
        ("exposure_ms", "expected"),
        [(1.5, "1.5"), (0.0, "0")],
    )
    def test_sweep_carries_the_session_exposure(
        self, tmp_path, fake_scan, fake_sweep, exposure_ms, expected
    ):
        """The measured session exposure has to reach the runbook.

        ``--exposure-ms 0`` is "not fixed": the probe then starts from ~0.02 ms,
        which on this Daheng is 20-75x darker than the usable range and leaves
        the calibration looking at a near-black frame.  A caller that measured a
        real exposure (the GUI sidebar does) must therefore see it forwarded, and
        a caller that did not must keep the CLI's existing behaviour.
        """
        calibrate_session(
            _params(exposure_time_ms=exposure_ms),
            _config(exposure_time_ms=exposure_ms),
            tmp_path / "run",
            locate=False,
            geometry=True,
        )
        for call in fake_sweep:
            assert call[call.index("--exposure-ms") + 1] == expected

    def test_weak_correlation_stops_the_session(self, tmp_path, fake_scan, monkeypatch):
        def run(command, *, cwd, check):
            if command[-1] == "calibrate":
                out = Path(command[command.index("--out") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "bench_geometry.json").write_text(
                    json.dumps({"method": "sweep", "correlation": 0.12}), encoding="utf-8"
                )

        monkeypatch.setattr(subprocess, "run", run)
        with pytest.raises(RuntimeError, match="failed validation"):
            calibrate_session(
                _params(), _config(), tmp_path / "run", locate=True, geometry=True
            )

    def test_runbook_failure_surfaces_as_called_process_error(
        self, tmp_path, fake_scan, monkeypatch
    ):
        def run(command, *, cwd, check):
            raise subprocess.CalledProcessError(1, command)

        monkeypatch.setattr(subprocess, "run", run)
        with pytest.raises(subprocess.CalledProcessError):
            calibrate_session(
                _params(), _config(), tmp_path / "run", locate=True, geometry=True
            )


class TestApplySessionCalibration:
    def test_writes_only_the_measured_fields(self, tmp_path):
        # ``replace`` re-runs the config's own validation, so the measured
        # artefact has to exist on disk exactly as the runbook leaves it.
        measured = tmp_path / "g.json"
        measured.write_text(json.dumps({"method": "sweep"}), encoding="utf-8")
        params, config = _params(), _config()
        calibration = session.SessionCalibration(
            pupil_center=(960, 720), geometry_file=measured
        )

        new_params, new_config = apply_session_calibration(params, config, calibration)

        assert new_params.pupil_center == "960,720"
        assert new_config.pupil_center == (960, 720)
        assert new_config.geometry_file == str(measured)

    def test_a_skipped_step_cannot_overwrite_a_supplied_value(self, tmp_path):
        supplied = tmp_path / "supplied.json"
        supplied.write_text(json.dumps({"method": "sweep"}), encoding="utf-8")
        params, config = _params(), _config(geometry_file=str(supplied))
        calibration = session.SessionCalibration(evidence_dir=tmp_path)

        new_params, new_config = apply_session_calibration(params, config, calibration)

        assert new_params is params
        assert new_config.geometry_file == str(supplied)
        assert new_config.pupil_center == (960, 600)

    def test_an_empty_calibration_is_a_no_op(self):
        params, config = _params(), _config()
        assert apply_session_calibration(params, config, session.SessionCalibration()) == (
            params, config,
        )


class TestSummaryLines:
    def test_one_chinese_line_per_measurement(self, tmp_path):
        calibration = session.SessionCalibration(
            pupil_center=(960, 720),
            panel_scan_response=-0.1,
            panel_scan_max_pair_drift=0.006,
            panel_scan_session_drift=0.012,
            geometry_file=tmp_path / "bench_geometry.json",
            geometry_correlation=0.82,
            evidence_dir=tmp_path,
            notes=("已跳过 sweep 几何标定, 沿用给定的几何文件",),
        )

        lines = calibration.summary_lines()

        assert lines == (
            "入射定位: 瞳面中心面板坐标 (960, 720)",
            "入射定位响应: -0.100",
            "配对平场漂移上限: 0.60%",
            "整轮平场漂移: 1.20%",
            f"sweep 几何文件: {tmp_path / 'bench_geometry.json'}",
            "几何相关系数: 0.8200",
            f"证据目录: {tmp_path}",
            "已跳过 sweep 几何标定, 沿用给定的几何文件",
        )

    def test_an_empty_calibration_still_names_the_evidence_dir(self, tmp_path):
        assert session.SessionCalibration(evidence_dir=tmp_path).summary_lines() == (
            f"证据目录: {tmp_path}",
        )

    def test_lines_obey_the_terminology_glossary(self, tmp_path):
        """No forbidden colloquialisms may leak into the GUI caption."""
        calibration = session.SessionCalibration(
            pupil_center=(1, 2), panel_scan_response=0.5,
            panel_scan_max_pair_drift=0.1, panel_scan_session_drift=0.2,
            geometry_file=tmp_path / "g.json", geometry_correlation=0.5,
            evidence_dir=tmp_path,
            notes=calibrate_session(
                _params(), _config(), tmp_path, locate=False, geometry=False
            ).notes,
        )
        text = "\n".join(calibration.summary_lines())

        for banned in ("台架", "实机", "真机", "上机", "瞳孔", "围栅", "跑通",
                       "打散", "灌进", "钻空子", "金标准", "平场泄漏"):
            assert banned not in text
        assert "瞳面" in text
        assert "对比择优" in text

    def test_bake_off_is_spelled_out_on_first_mention(self, tmp_path, fake_scan, fake_sweep):
        lines = calibrate_session(
            _params(), _config(), tmp_path, locate=True, geometry=False
        ).summary_lines()
        assert any("对比择优" in line for line in lines)


def test_measure_session_geometry_is_reexported_under_its_private_name() -> None:
    """The CLI runner keeps exposing the historical private alias."""
    from ao_shaping.runners.slm.model_in_loop_runner import _measure_session_geometry

    assert _measure_session_geometry is session.measure_session_geometry
