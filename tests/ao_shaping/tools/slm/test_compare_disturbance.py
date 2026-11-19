"""Disturbance geometry and timestamp persistence for paired hardware runs."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.compare_disturbance import _disturbance
from ao_shaping.tools.slm.replay_disturbance import _phase
from ao_shaping.tools.kernels.bench_kernels import phase_to_panel
from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts
from ao_shaping.utils.slm.phase_disturbance import load_panel_disturbance


def test_disturbance_is_reproducible_and_centered(tmp_path):
    coefficients, panel = _disturbance(78, 0.5)
    assert len(coefficients) == 12
    assert panel.shape == (1200, 1920)
    assert np.array_equal(panel, _disturbance(78, 0.5)[1])
    assert not np.any(panel[:270])
    assert not np.any(panel[:, :510])
    aperture = panel[270:1170, 510:1410]
    assert float(np.std(aperture[aperture != 0])) == pytest.approx(0.5, rel=1e-5)
    path = tmp_path / "bias.npy"
    np.save(path, panel)
    assert np.array_equal(load_panel_disturbance(str(path), panel.shape), panel)


def test_off_center_disturbance_and_replay_use_saved_panel_geometry(tmp_path: Path):
    center = (750, 510)
    radius = 200
    _, bias = _disturbance(78, 0.5, center, radius)
    assert not np.any(bias[:center[1] - radius])
    assert not np.any(bias[:, :center[0] - radius])
    aperture = bias[center[1] - radius:center[1] + radius,
                    center[0] - radius:center[0] + radius]
    assert float(np.std(aperture[aperture != 0])) == pytest.approx(0.5, rel=1e-5)

    run = tmp_path / "model"
    run.mkdir()
    grid = np.ones((64, 64), dtype=np.float64)
    np.save(run / "best_phase.npy", grid)
    item = {"directory": "model", "algorithm": "model_in_loop", "condition": "disturbed"}
    panel = _phase(tmp_path, item, bias, center, radius)
    np.testing.assert_allclose(panel, phase_to_panel(grid, radius, center) + bias)


def test_debug_pickle_keeps_per_iteration_utc_times(tmp_path):
    recorder = Recorder()
    recorder.history.append({
        "_epoch": 1, "score": 0.5,
        "timestamp_utc": "2026-10-13T08:00:00.123456+00:00",
        "pos_timestamp_utc": "2026-10-13T07:59:59.123456+00:00",
        "step_a_trace": [{"iteration": 0, "loss": 0.1,
                          "timestamp_utc": "2026-10-13T07:59:58.123456+00:00"}],
    })
    save_recorder_debug_artifacts(
        recorder, str(tmp_path), "test_run", output_dir=tmp_path / "run",
        scalar_keys=("score",),
        text_keys=("timestamp_utc", "pos_timestamp_utc"),
        object_keys=("step_a_trace",), json_payload={},
    )
    with (tmp_path / "run" / "test_run_run.pkl").open("rb") as handle:
        record = pickle.load(handle)[1]
    assert record["timestamp_utc"].endswith("123456+00:00")
    assert record["pos_timestamp_utc"].endswith("123456+00:00")
    assert record["step_a_trace"][0]["timestamp_utc"].endswith("123456+00:00")
