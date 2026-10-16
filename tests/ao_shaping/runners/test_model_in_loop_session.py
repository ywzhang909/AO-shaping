"""The live session sweep must feed current geometry to the optimizer."""

import json
import subprocess

import pytest

from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig
from ao_shaping.runners.slm.model_in_loop_runner import _measure_session_geometry


def test_session_sweeps_then_calibrates_current_geometry(tmp_path, monkeypatch) -> None:
    config = SlmModelInLoopConfig(
        cam_type="daheng", slm_number=1, slm_wavelength=1064,
        pupil_center=(1040, 620), region=64, far_field_padding=8,
    )
    calls = []

    def run(command, *, cwd, check):
        calls.append((command, cwd, check))
        if command[-1] == "calibrate":
            output = tmp_path / "session_geometry" / "bench_geometry.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"method": "sweep", "correlation": 0.82}))

    monkeypatch.setattr(subprocess, "run", run)
    result = _measure_session_geometry(config, tmp_path)
    assert result == tmp_path / "session_geometry" / "bench_geometry.json"
    assert [call[0][-1] for call in calls] == ["sweep", "calibrate"]
    command = calls[0][0]
    assert command[command.index("--pupil-center") + 1] == "1040,620"
    assert command[command.index("--far-field-size") + 1] == "512"
    assert command[command.index("--collect-disc") + 1] == "450"
    assert command[command.index("--exposure-ms") + 1] == "0"
    assert all(call[2] for call in calls)


def test_session_rejects_weak_geometry(tmp_path, monkeypatch) -> None:
    config = SlmModelInLoopConfig(region=64, far_field_padding=8)

    def run(command, *, cwd, check):
        if command[-1] == "calibrate":
            output = tmp_path / "session_geometry" / "bench_geometry.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"method": "sweep", "correlation": 0.12}))

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="failed validation"):
        _measure_session_geometry(config, tmp_path)


def test_session_stops_after_sweep_failure(tmp_path, monkeypatch) -> None:
    config = SlmModelInLoopConfig(region=64, far_field_padding=8)
    calls = []

    def run(command, *, cwd, check):
        calls.append(command[-1])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        _measure_session_geometry(config, tmp_path)
    assert calls == ["sweep"]
