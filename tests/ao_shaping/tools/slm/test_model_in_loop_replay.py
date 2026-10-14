"""Offline checks for interleaved model-in-loop phase replay."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm import model_in_loop_replay as replay
from ao_shaping.optimizer.wfless.slm_model_in_loop import _HardwareBench


def _saved_run(path, *, pupil_center=(960, 720)):
    path.mkdir()
    config = {
        "cam_type": "daheng", "cam_id": 0, "cam_size": 8,
        "slm_number": 1, "slm_wavelength": 1064, "wavelength_nm": 1064,
        "region": 4, "panel_span_px": 900.0, "pupil_center": pupil_center,
        "target_side": 4, "w_efficiency": 0.6, "w_uniformity": 0.4,
        "far_field_padding": 3, "max_exposure_ms": 0.4,
        "phase_disturbance_file": None,
    }
    (path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    np.save(path / "best_phase.npy", np.ones((4, 4)))


def test_load_plan_rejects_changed_pupil_before_hardware(tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b", pupil_center=(840, 720))
    with pytest.raises(ValueError, match="pupil_center"):
        replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_load_plan_rejects_different_panel_disturbance(tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    config_path = tmp_path / "b" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["phase_disturbance_file"] = "different.npy"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="phase_disturbance_file"):
        replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_load_plan_rejects_unsafe_fixed_exposure(tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    with pytest.raises(ValueError, match="Daheng limit"):
        replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out", 0.6)


def test_replay_records_sequence_and_restores_flat(monkeypatch, tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    plan = replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    seen = []

    class Bench:
        def __init__(self, _config):
            self.last_exposure_ms = 0.02
            self.last_raw_frame = np.ones((8, 8), dtype=np.uint8)

        def display(self, phase):
            seen.append(float(np.mean(phase)))

        def measure(self):
            return np.ones((8, 8), dtype=float)

        def close(self):
            seen.append("closed")

    monkeypatch.setattr(replay, "_HardwareBench", Bench)
    monkeypatch.setattr(replay, "zero_order_center", lambda *_a, **_k: (4, 4))
    rows = replay.run_replay(plan)
    assert [row["phase"] for row in rows] == list(replay.SEQUENCE)
    assert seen[-2:] == [0.0, "closed"]
    assert len(list((tmp_path / "out").glob("*_raw.npy"))) == 7
    assert json.loads((tmp_path / "out" / "measurements.json").read_text()) == rows


def test_replay_uses_fixed_exposure_only_for_shaped_phases(monkeypatch, tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    plan = replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out", 0.4)
    calls = []

    class Bench:
        last_raw_frame = np.ones((8, 8), dtype=np.uint8)
        last_exposure_ms = 0.02

        def __init__(self, _config):
            pass

        def display(self, _phase):
            pass

        def measure(self):
            calls.append("auto")
            return np.ones((8, 8), dtype=float)

        def measure_fixed_exposure(self, exposure):
            calls.append(exposure)
            return np.ones((8, 8), dtype=float)

        def close(self):
            pass

    monkeypatch.setattr(replay, "_HardwareBench", Bench)
    monkeypatch.setattr(replay, "zero_order_center", lambda *_a, **_k: (4, 4))
    replay.run_replay(plan)
    assert calls == ["auto", 0.4, 0.4, "auto", 0.4, 0.4, "auto", "auto"]
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["fixed_shaped_exposure_ms"] == 0.4


def test_fixed_exposure_capture_checks_readback_and_saturation():
    class Camera:
        exposure = 0.4
        raw_peak = 100

        def reset_exposure_time(self, _ms):
            return self.exposure

        def get_numpy_image(self, n_sample):
            assert n_sample == 1
            return np.full((8, 8), self.raw_peak, dtype=np.uint8)

    bench = object.__new__(_HardwareBench)
    bench._config = SimpleNamespace(
        cam_type="daheng", max_exposure_ms=0.4, n_eval_frames=2,
        settle_max_discard=4, settle_wait_s=0.1, settle_tol=0.02,
        settle_max_wait_s=1.0,
    )
    bench._cam = Camera()
    bench._slm = object()
    bench._pending = np.zeros((8, 8))
    bench._capture = lambda *_a, **_k: np.full((8, 8), 100.0)

    frame = bench.measure_fixed_exposure(0.4)
    assert np.all(frame == 5.0)
    assert bench.last_exposure_ms == 0.4
    bench._cam.exposure = 0.6
    with pytest.raises(RuntimeError, match="readback"):
        bench.measure_fixed_exposure(0.4)
    bench._cam.exposure = 0.4
    bench._cam.raw_peak = 250
    with pytest.raises(RuntimeError, match="saturated"):
        bench.measure_fixed_exposure(0.4)
