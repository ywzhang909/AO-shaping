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
        "target_shape": "rectangle", "target_side": 4,
        "w_efficiency": 0.6, "w_uniformity": 0.4,
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


def test_load_plan_rejects_different_target_shape(tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    config_path = tmp_path / "b" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_shape"] = "square"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="target_shape"):
        replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out")


def test_load_plan_allows_different_optimization_weights(tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    config_path = tmp_path / "b" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(w_efficiency=0.2, w_uniformity=0.8)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    plan = replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    assert plan.config.w_uniformity == 0.4
    assert not (tmp_path / "out").exists()


def test_load_plan_rejects_unsafe_fixed_exposure(tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    with pytest.raises(ValueError, match="Daheng limit"):
        replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out", 0.6)
    with pytest.raises(ValueError, match="Daheng limit"):
        replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out", 0.4, 0.6)


def test_replay_records_sequence_and_restores_flat(monkeypatch, tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    plan = replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out")
    seen = []
    metric_calls = []

    class Bench:
        def __init__(self, _config):
            self.last_exposure_ms = 0.02
            self.last_capture_utc = "2026-10-14T00:00:00.000001+00:00"
            self.last_raw_frame = np.ones((8, 8), dtype=np.uint8)

        def display(self, phase):
            seen.append(float(np.mean(phase)))

        def measure(self):
            return np.ones((8, 8), dtype=float)

        def close(self):
            seen.append("closed")

    monkeypatch.setattr(replay, "_HardwareBench", Bench)
    monkeypatch.setattr(replay, "zero_order_center", lambda *_a, **_k: (4, 4))
    original_metrics_at = replay._metrics_at

    def checked_metrics(*args, **kwargs):
        metric_calls.append(kwargs)
        return original_metrics_at(*args, **kwargs)

    monkeypatch.setattr(replay, "_metrics_at", checked_metrics)
    rows = replay.run_replay(plan)
    assert [row["phase"] for row in rows] == list(replay.SEQUENCE)
    assert all(0.0 <= row["shape_iou"] <= 1.0 for row in rows)
    assert all(row["capture_utc"] == "2026-10-14T00:00:00.000001+00:00" for row in rows)
    assert seen[-2:] == [0.0, "closed"]
    assert len(list((tmp_path / "out").glob("*_raw.npy"))) == 7
    assert metric_calls == [
        {"target_shape": "rectangle", "camera_rotation_quarters": 1}
    ] * len(replay.SEQUENCE)
    assert json.loads((tmp_path / "out" / "measurements.json").read_text()) == rows


def test_replay_uses_fixed_exposure_only_for_shaped_phases(monkeypatch, tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    plan = replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out", 0.4)
    calls = []

    class Bench:
        last_raw_frame = np.ones((8, 8), dtype=np.uint8)
        last_exposure_ms = 0.02
        last_capture_utc = "2026-10-14T00:00:00.000001+00:00"

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


def test_replay_can_hold_flat_exposure_fixed(monkeypatch, tmp_path):
    _saved_run(tmp_path / "a")
    _saved_run(tmp_path / "b")
    plan = replay.load_plan(tmp_path / "a", tmp_path / "b", tmp_path / "out", 0.4, 0.06)
    calls = []

    class Bench:
        last_raw_frame = np.ones((8, 8), dtype=np.uint8)
        last_exposure_ms = 0.06
        last_capture_utc = "2026-10-14T00:00:00.000001+00:00"

        def __init__(self, _config):
            pass

        def display(self, _phase):
            pass

        def measure_fixed_exposure(self, exposure):
            calls.append(exposure)
            self.last_exposure_ms = exposure
            return np.ones((8, 8), dtype=float)

        def close(self):
            pass

    monkeypatch.setattr(replay, "_HardwareBench", Bench)
    monkeypatch.setattr(replay, "zero_order_center", lambda *_a, **_k: (4, 4))
    rows = replay.run_replay(plan)
    assert calls == [0.06, 0.4, 0.4, 0.06, 0.4, 0.4, 0.06, 0.06]
    assert [r["exposure_ms"] for r in rows] == calls[:-1]


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
    # ``__init__`` resolves the session ceiling into ``_ceiling_ms``; this
    # bypassed constructor must set it explicitly (here: the explicit config
    # value, which is what resolution would have returned verbatim).
    bench._ceiling_ms = 0.4
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
