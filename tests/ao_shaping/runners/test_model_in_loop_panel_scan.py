"""The optional panel scan gates model-in-loop hardware optimization."""

import json

from click.testing import CliRunner

from ao_shaping.optimizer.wfless import slm_model_in_loop
from ao_shaping.runners.slm.model_in_loop_runner import run
from ao_shaping.tools.slm.panel import panel_locate
from ao_shaping.utils.io import file as io_file


def test_panel_scan_updates_center_before_optimization(monkeypatch, tmp_path):
    seen = {}

    def scan(params, *, strict, max_session_drift):
        seen["grid"] = (params.grid_xs, params.grid_ys)
        seen["strict"] = strict
        seen["session_limit"] = max_session_drift
        return panel_locate.PanelLocateResult((960, 720), -0.1, 0.006, 0.012, 0)

    def optimize(config):
        seen["center"] = config.pupil_center
        raise RuntimeError("optimization stub stopped")

    # A supplied geometry file keeps this to the panel-scan step: without one a
    # hardware run is an auto session and would also sweep the geometry (see
    # test_model_in_loop_session_preflight.py for that path).
    geometry = tmp_path / "bench_geometry.json"
    geometry.write_text(json.dumps({"method": "sweep", "correlation": 0.82}), encoding="utf-8")

    monkeypatch.setattr(panel_locate, "scan_panel", scan)
    monkeypatch.setattr(slm_model_in_loop, "optimize_slm_model_in_loop", optimize)
    monkeypatch.setattr(io_file, "debug_run_dir", lambda *_: tmp_path / "run")

    result = CliRunner().invoke(
        run,
        ["--panel-scan", "--pupil-center", "960,600", "--panel-scan-step-px", "120",
         "--geometry-file", str(geometry)],
    )

    assert result.exit_code == 2, result.output
    assert seen == {
        "grid": ("840,960,1080", "480,600,720"),
        "strict": True,
        "session_limit": 0.05,
        "center": (960, 720),
    }
    saved_config = json.loads((tmp_path / "run" / "run_config.json").read_text(encoding="utf-8"))
    assert saved_config["pupil_center"] == [960, 720]
    assert "设备已释放" in result.output


def test_panel_scan_drift_failure_stops_before_optimizer(monkeypatch, tmp_path):
    def scan(*_args, **_kwargs):
        raise ValueError("session flat drift 8.0% exceeds 5.0%")

    def optimize(_config):
        raise AssertionError("optimizer opened after failed panel scan")

    monkeypatch.setattr(panel_locate, "scan_panel", scan)
    monkeypatch.setattr(slm_model_in_loop, "optimize_slm_model_in_loop", optimize)
    monkeypatch.setattr(io_file, "debug_run_dir", lambda *_: tmp_path / "run")

    result = CliRunner().invoke(run, ["--panel-scan", "--pupil-center", "960,600"])

    assert result.exit_code != 0
    assert "panel scan rejected" in result.output
    assert "session flat drift" in result.output
    saved_config = json.loads((tmp_path / "run" / "run_config.json").read_text(encoding="utf-8"))
    assert saved_config["pupil_center"] == [960, 600]
