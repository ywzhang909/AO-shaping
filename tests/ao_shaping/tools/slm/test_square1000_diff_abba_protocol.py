"""Offline contract tests for the one-session square-1000 protocol."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "square1000_protocol", ROOT/"scripts/slm/square_1000_diff_abba_20261030.py")
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


@pytest.fixture
def seed_archive(tmp_path):
    """Small on-disk archive exercising the same contract as the live v2 run."""
    seed_run = tmp_path/"seed_run"
    seed_run.mkdir()
    native_shape = (8, 8)
    seed = np.arange(16, dtype=np.float32).reshape(4, 4)/100
    camera_xy = np.array(
        [[[1., 2.], [3., 2.]],
         [[1., 5.], [3., 5.]]],
        dtype=np.float64,
    )
    np.savez_compressed(
        seed_run/"plan.npz",
        **{protocol.SEED_KEY: seed, "camera_xy": camera_xy},
    )
    np.savez_compressed(
        seed_run/"capture_0008.npz",
        raw_frames=np.full((3, *native_shape), 100, dtype=np.uint16),
        model_phase=seed,
        camera_xy=camera_xy,
    )
    (seed_run/"manifest.json").write_text(json.dumps([dict(
        label=protocol.SEED_KEY,
        file="capture_0008.npz",
        exposure_ms=5.46875,
    )]), encoding="utf-8")
    (seed_run/"measurement_contract.json").write_text(json.dumps(dict(
        background_counts=15.2,
        coverage_threshold_counts=2.1987076823376803,
        exposure_ms=5.46875,
        noise_counts=0.4880660265853328,
        full_scale=4095,
        smoothing_sigma_native_px=8,
    )), encoding="utf-8")
    checkpoint = tmp_path/"checkpoint.json"
    checkpoint.write_text(json.dumps(dict(
        defocus_diopter=0.,
        origin=[2., 2.],
        amplitude_center=[4., 4.],
        amplitude_width=[2., 2.],
        residual_noll_radians={},
        gain=1.,
        background=0.,
        sigma_ccd_px=0.,
        ramp_matrix=[[1., 0.], [0., 1.]],
        fft_size=32,
        pixel_samples=1,
        grid_size=4,
    )), encoding="utf-8")
    return SimpleNamespace(
        seed_run=seed_run,
        checkpoint=checkpoint,
        native_shape=native_shape,
        seed=seed,
        camera_xy=camera_xy,
    )


def load_fixture(archive):
    return protocol.load_seed_contract(
        archive.seed_run,
        archive.checkpoint,
        native_shape=archive.native_shape,
    )


def test_defaults_point_to_v2_seed_contract():
    assert protocol.SEED_RUN.name == "square1000_bandlimit_live_20261030_v2"
    assert protocol.SEED_KEY == "F_blur4_full"


def test_load_seed_contract_resolves_label_and_uses_measurement_contract(seed_archive):
    archive = load_fixture(seed_archive)
    assert archive.capture_path.name == "capture_0008.npz"
    assert archive.capture_row["label"] == protocol.SEED_KEY
    assert archive.seed.dtype == np.float32
    np.testing.assert_array_equal(archive.seed, seed_archive.seed)
    np.testing.assert_array_equal(archive.camera_xy, seed_archive.camera_xy)
    assert archive.measurement["background_counts"] == pytest.approx(15.2)
    assert archive.measurement["coverage_threshold_counts"] == pytest.approx(
        2.1987076823376803
    )
    assert protocol.measurement_kwargs(archive.measurement) == dict(
        smoothing_sigma=8., full_scale=4095.,
    )


def test_seed_label_must_be_unique(seed_archive):
    manifest_path = seed_archive.seed_run/"manifest.json"
    row = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
    for rows, match in [
        ([], "exactly one"),
        ([row, row], "exactly one"),
    ]:
        manifest_path.write_text(json.dumps(rows), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_fixture(seed_archive)


def test_seed_capture_filename_must_be_a_basename(seed_archive):
    manifest_path = seed_archive.seed_run/"manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows[0]["file"] = "../capture_0008.npz"
    manifest_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid capture filename"):
        load_fixture(seed_archive)


def test_plan_seed_must_exactly_match_capture_phase(seed_archive):
    capture_path = seed_archive.seed_run/"capture_0008.npz"
    with np.load(capture_path) as saved:
        raw = saved["raw_frames"]
        phase = saved["model_phase"]
        xy = saved["camera_xy"]
    phase[0, 0] += 1e-6
    np.savez_compressed(
        capture_path, raw_frames=raw, model_phase=phase, camera_xy=xy,
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        load_fixture(seed_archive)


def test_capture_camera_grid_must_match_plan(seed_archive):
    capture_path = seed_archive.seed_run/"capture_0008.npz"
    with np.load(capture_path) as saved:
        raw = saved["raw_frames"]
        phase = saved["model_phase"]
        xy = saved["camera_xy"]
    xy[0, 0, 0] += 1
    np.savez_compressed(
        capture_path, raw_frames=raw, model_phase=phase, camera_xy=xy,
    )
    with pytest.raises(ValueError, match="camera_xy does not exactly match"):
        load_fixture(seed_archive)


@pytest.mark.parametrize("bad_grid, message", [
    (np.array([[[1.5, 2.], [3., 2.]], [[1., 5.], [3., 5.]]]), "integer-valued"),
    (np.array([[[1., 2.], [3., 2.], [6., 2.]],
               [[1., 5.], [3., 5.], [6., 5.]]]), "uniform"),
    (np.array([[[1., 2.], [9., 2.]], [[1., 5.], [9., 5.]]]), "outside"),
])
def test_camera_grid_contract_rejects_invalid_native_sampling(
        seed_archive, bad_grid, message):
    plan_path = seed_archive.seed_run/"plan.npz"
    np.savez_compressed(
        plan_path, **{protocol.SEED_KEY: seed_archive.seed, "camera_xy": bad_grid},
    )
    with pytest.raises(ValueError, match=message):
        load_fixture(seed_archive)


@pytest.mark.parametrize("raw_frames, message", [
    (np.full((2, 8, 8), 100, dtype=np.uint16), "at least three"),
    (np.full((3, 7, 8), 100, dtype=np.uint16), "native frame shape"),
    (np.full((3, 8, 8), 100., dtype=np.float32), "integer counts"),
])
def test_capture_frames_are_native_integer_stacks(seed_archive, raw_frames, message):
    capture_path = seed_archive.seed_run/"capture_0008.npz"
    np.savez_compressed(
        capture_path,
        raw_frames=raw_frames,
        model_phase=seed_archive.seed,
        camera_xy=seed_archive.camera_xy,
    )
    with pytest.raises(ValueError, match=message):
        load_fixture(seed_archive)


def record(t, candidate=False, **override):
    m = dict(coverage=.3, iou=.27, boundary_error=1., cv=2.7,
             target_rate=100., local_ee=.75, loss=9., clipped_target=0)
    if candidate:
        m.update(coverage=.33, iou=.30, loss=8.5, cv=2.65, boundary_error=.98)
    m.update(override)
    return dict(label=str(t), monotonic_ns=t, exposure_ms=10., metrics=m)


def test_reproducible_shape_gain_required():
    repeats = [record(t) for t in (-3, -2, -1)]
    rows = [record(0), record(1, True), record(2, True), record(3)]
    assert protocol.accept_abba(rows, repeats)["accepted"]
    rows[2] = record(2)
    assert not protocol.accept_abba(rows, repeats)["accepted"]


@pytest.mark.parametrize("changes", [
    dict(cv=2.), dict(clipped_target=1), dict(target_rate=97.),
    dict(boundary_error=1.1), dict(local_ee=.74), dict(coverage=float("nan")),
])
def test_cv_only_clipping_energy_boundary_and_nonfinite_veto(changes):
    repeats = [record(t) for t in (-3, -2, -1)]
    b = record(1, **changes) if changes == dict(cv=2.) else record(1, True, **changes)
    rows = [record(0), b, record(2, True), record(3)]
    assert not protocol.accept_abba(rows, repeats)["accepted"]


def test_exposure_and_same_phase_drift_veto():
    repeats = [record(t) for t in (-3, -2, -1)]
    rows = [record(0), record(1, True), record(2, True), record(3)]
    rows[2]["exposure_ms"] = 11.
    assert not protocol.accept_abba(rows, repeats)["accepted"]
    rows[2]["exposure_ms"] = 10.
    rows[-1]["metrics"]["target_rate"] = 110.
    assert not protocol.accept_abba(rows, repeats)["accepted"]


def test_repeat_noise_and_baseline_drift_raise_resolution():
    repeats = [record(-3, coverage=.29), record(-2), record(-1, coverage=.31)]
    rows = [record(0), record(1, True), record(2, True), record(3, coverage=.32)]
    result = protocol.accept_abba(rows, repeats)
    assert not result["accepted"]
    assert result["resolution"]["coverage"] >= .059


def test_measured_candidate_has_bounded_raw_increment():
    torch = pytest.importorskip("torch")
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusParameters, FrozenDefocusForward,
    )
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
    params = FrozenDefocusParameters(
        defocus_diopter=0., origin_xy=(0., 0.), amplitude_center_xy=(960., 600.),
        amplitude_width_xy=(500., 400.), residual_noll_radians={},
        gain=1., background=0., sigma_ccd_px=0.,
        ramp_matrix=((8000., 0.), (0., 8000.)), grid_size=16, fft_size=64, pixel_samples=1)
    yy, xx = np.mgrid[-20:21:2, -20:21:2]
    forward = FrozenDefocusForward(params, np.stack([xx, yy], -1), device="cpu")
    seed = np.zeros((16, 16), np.float32)
    target = ((abs(xx) < 14) & (abs(yy) < 14)).astype(np.float32)
    objective = ShapeObjective(target, formation=True)
    with torch.no_grad():
        measured = forward.predict(seed)[0].numpy()
    candidate, info = protocol.candidate_step(forward, objective, seed, measured, .03)
    assert candidate.shape == seed.shape
    assert np.isfinite(candidate).all()
    assert 0 < info["rms_rad"] <= .030001
    assert info["max_abs_rad"] <= .300001
    assert info["optimizer"][0]["mode"] == "anchored"
    assert info["optimizer"][0]["l_partition"] > 0
    weights = forward.illumination_weights()
    assert abs(float((candidate*weights).sum()/weights.sum())) < 1e-6
    with pytest.raises(ValueError):
        protocol.candidate_step(forward, objective, seed, measured, float("nan"))


def test_masks_cover_entire_target_exclude_zero():
    xy = np.array([[[1418., 444.], [1267., 943.], [1917., 943.], [2418., 943.]]])
    m = protocol.masks(xy)
    assert m["box"].tolist() == [[True, False, True, False]]
    assert not m["roi"][0, 1]


def test_formation_target_uses_nonempty_binary_mask_not_sum1():
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective

    yy, xx = np.mgrid[300:1588:4, 1300:2540:4]
    target = protocol.masks(np.stack([xx, yy], -1))["box"].astype(np.float32)
    objective = ShapeObjective(target, formation=True)
    assert set(np.unique(target)) == {0., 1.}
    assert target.sum() > 1
    assert objective.core.any() and objective._boundary_mask.any()
    assert objective.mode == "formation"
    assert objective.weights["shape"] > 0 and objective.weights["energy"] > 0


def test_execute_failure_resets_direct_gray_and_closes_before_release(monkeypatch, tmp_path):
    import ao_shaping.tools.kernels.device_mutex as mutex
    events = []
    lock = SimpleNamespace(release=lambda: events.append("release"))
    monkeypatch.setattr(mutex, "wait_until_idle", lambda *a, **k: SimpleNamespace(
        ok=True, lock=lock))
    monkeypatch.setattr(protocol.time, "sleep", lambda _: None)

    class Camera:
        def __enter__(self):
            events.append("camera_open")
            return self

        def __exit__(self, *exc):
            events.append("camera_close")

        def reset_exposure_time(self, _):
            pass

        def get_numpy_image(self, **_):
            raise RuntimeError("injected capture failure")

    class SLM:
        def __init__(self, **kwargs):
            assert kwargs["video_mode"] == 0

        def __enter__(self):
            events.append("slm_open")
            return self

        def __exit__(self, *exc):
            events.append("slm_close")

        def set_phase_overlay(self, _):
            pass

        def create_phase_from_array(self, phase):
            assert phase.dtype == np.float32
            return np.ones((1200, 1920), np.uint16)

        def display_data(self, gray):
            if not gray.any():
                assert gray.dtype == np.uint16 and gray.shape == (1200, 1920)
                events.append("direct_gray_reset")

    modules = {
        "ao_shaping.tools.slm.transfer.hardware": dict(prepare_camera_sdk=lambda _: None),
        "ao_shaping.drivers.ccd.common": dict(create_camera=lambda *a, **k: Camera(),
                                               get_camera_exposure_ms=lambda _: .5),
        "ao_shaping.drivers.slm.santec": dict(Santec=SLM),
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    args = SimpleNamespace(out=tmp_path, wait_until_idle=0., blocks=1)
    with pytest.raises(RuntimeError, match="injected capture failure"):
        protocol.execute(args, np.zeros((640, 640), np.float32), None, None, None, dict(
            background_counts=15.2,
            coverage_threshold_counts=2.2,
            exposure_ms=5.46875,
            noise_counts=.5,
            full_scale=4095.,
            smoothing_sigma_native_px=8.,
        ))
    assert events == ["camera_open", "slm_open", "direct_gray_reset", "slm_close",
                      "camera_close", "release"]
