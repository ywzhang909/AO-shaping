"""Regression-anchor tests for :mod:`ao_shaping.utils.hardware_utils`.

Pins the EXACT current behavior of the module (read-only — the source is not
modified). All tests are mock-based; no real hardware or native SDK access.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from ao_shaping.utils import hardware_utils
from ao_shaping.utils.hardware_utils import (
    apply_auto_exposure,
    auto_exposure_possible,
    auto_exposure_target_ms,
    call_with_timeout,
    capture_amplitude,
    init_frame_recording,
    open_camera,
    record_frame,
    save_frame_png,
)


# ---------------------------------------------------------------------------
# open_camera
# ---------------------------------------------------------------------------
class TestOpenCamera:
    def test_open_camera_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera type"):
            open_camera("bogus", 0, 1.0)


# ---------------------------------------------------------------------------
# call_with_timeout
# ---------------------------------------------------------------------------
class TestCallWithTimeout:
    def test_fast_function_returns_result(self) -> None:
        assert call_with_timeout(lambda: 42, 1.0, "fast") == 42

    def test_fast_function_returning_none_returns_none(self) -> None:
        assert call_with_timeout(lambda: None, 1.0, "none") is None

    def test_slow_function_raises_timeout(self) -> None:
        gate = threading.Event()

        def slow() -> str:
            gate.wait(10)
            return "never"

        with pytest.raises(TimeoutError):
            call_with_timeout(slow, 0.05, "slow")
        gate.set()  # release the daemon thread

    def test_target_exception_is_reraisd(self) -> None:
        def boom() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            call_with_timeout(boom, 1.0, "boom")


# ---------------------------------------------------------------------------
# Frame recording
# ---------------------------------------------------------------------------
class TestFrameRecording:
    def test_init_creates_frames_dir(self, tmp_path) -> None:
        init_frame_recording(tmp_path)
        assert (tmp_path / "frames").is_dir()

    def test_record_frame_writes_files_and_meta(self, tmp_path) -> None:
        init_frame_recording(tmp_path)
        raw = np.zeros((5, 5), dtype=np.uint16)
        raw[2, 2] = 100
        meta = record_frame(raw, "phase-A", 5.0)

        frames_dir = tmp_path / "frames"
        npy_path = frames_dir / "frame_00001.npy"
        png_path = frames_dir / "frame_00001.png"
        jsonl_path = frames_dir / "frame_meta.jsonl"
        assert npy_path.exists()
        assert png_path.exists()
        assert jsonl_path.exists()

        np.testing.assert_array_equal(np.load(npy_path), raw)

        assert set(meta.keys()) == {
            "frame",
            "phase",
            "exposure_ms",
            "peak",
            "sum",
            "centroid",
            "timestamp",
        }
        assert meta["frame"] == 1
        assert meta["phase"] == "phase-A"
        assert meta["exposure_ms"] == 5.0
        assert meta["peak"] == 100.0
        assert meta["sum"] == 100.0
        assert meta["centroid"] == [2.0, 2.0]
        assert isinstance(meta["timestamp"], str)

        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == meta

    def test_record_frame_include_spot_adds_argmax_key(self, tmp_path) -> None:
        init_frame_recording(tmp_path)
        raw = np.zeros((5, 5), dtype=np.uint16)
        raw[2, 3] = 100
        meta = record_frame(raw, "phase-A", 5.0, include_spot=True)
        assert meta["spot"] == [2, 3]

    def test_record_frame_counter_increments(self, tmp_path) -> None:
        init_frame_recording(tmp_path)
        raw = np.ones((3, 3), dtype=np.uint16)
        meta1 = record_frame(raw, "a", 1.0)
        meta2 = record_frame(raw, "b", 2.0)

        assert meta1["frame"] == 1
        assert meta2["frame"] == 2
        assert (tmp_path / "frames" / "frame_00001.npy").exists()
        assert (tmp_path / "frames" / "frame_00002.npy").exists()

        lines = (
            (tmp_path / "frames" / "frame_meta.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        assert len(lines) == 2
        assert json.loads(lines[1])["frame"] == 2

    def test_init_frame_recording_resets_counter(self, tmp_path) -> None:
        init_frame_recording(tmp_path)
        record_frame(np.ones((2, 2), dtype=np.uint16), "a", 1.0)
        init_frame_recording(tmp_path)  # reset
        meta = record_frame(np.ones((2, 2), dtype=np.uint16), "b", 1.0)
        assert meta["frame"] == 1

    def test_record_frame_without_init_skips_files(self, tmp_path) -> None:
        hardware_utils._frames_dir = None
        hardware_utils._frame_counter = 0
        meta = record_frame(np.ones((2, 2), dtype=np.uint16), "no-init", 1.0)
        assert meta["frame"] == 1
        assert not (tmp_path / "frames").exists()


class TestSaveFramePng:
    def test_creates_png_file(self, tmp_path) -> None:
        path = tmp_path / "frame.png"
        save_frame_png(np.zeros((8, 8), dtype=np.uint16), path, "test title")
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Auto-exposure helpers
# ---------------------------------------------------------------------------
class TestAutoExposureTargetMs:
    def test_in_band_returns_current(self) -> None:
        assert auto_exposure_target_ms(10.0, 180.0) == 10.0

    def test_in_band_lower_edge_inclusive(self) -> None:
        assert auto_exposure_target_ms(10.0, 144.0) == 10.0

    def test_below_low_scales_up(self) -> None:
        assert auto_exposure_target_ms(10.0, 100.0) == pytest.approx(18.0)

    def test_above_high_scales_down(self) -> None:
        assert auto_exposure_target_ms(10.0, 220.0) == pytest.approx(10.0 * 180.0 / 220.0)

    def test_saturation_floor_branch(self) -> None:
        # peak >= sat_floor: scale = max(target/peak, 0.1) -> 0.225
        assert auto_exposure_target_ms(10.0, 800.0) == pytest.approx(2.25)

    def test_max_boost_clamp(self) -> None:
        assert auto_exposure_target_ms(10.0, 1.0) == pytest.approx(40.0)

    def test_min_ms_clamp(self) -> None:
        assert auto_exposure_target_ms(0.001, 1.0) == pytest.approx(0.02)

    def test_max_ms_clamp(self) -> None:
        assert auto_exposure_target_ms(500.0, 1.0) == pytest.approx(1000.0)

    def test_max_cut_clamp(self) -> None:
        # peak < sat_floor but > high: scale = max(target/peak, max_cut) -> 0.25
        assert auto_exposure_target_ms(10.0, 800.0, sat_floor=1000.0) == pytest.approx(2.5)

    def test_custom_target_and_tol(self) -> None:
        assert auto_exposure_target_ms(
            10.0, 50.0, target_brightness=100.0, tol=0.1
        ) == pytest.approx(20.0)

    def test_returns_float(self) -> None:
        assert isinstance(auto_exposure_target_ms(10.0, 100.0), float)


class TestAutoExposurePossible:
    def test_callable_exposure_supported(self) -> None:
        class WithExposure:
            def reset_exposure_time(self, ms: float) -> float:
                return ms

        assert auto_exposure_possible(WithExposure()) is True

    def test_missing_exposure_unsupported(self) -> None:
        class WithoutExposure:
            pass

        assert auto_exposure_possible(WithoutExposure()) is False

    def test_non_callable_exposure_unsupported(self) -> None:
        class NonCallableExposure:
            reset_exposure_time = 5

        assert auto_exposure_possible(NonCallableExposure()) is False


class TestApplyAutoExposure:
    def test_changed_applies_new_exposure(self) -> None:
        class MockCamera:
            def __init__(self) -> None:
                self.calls: list[float] = []

            def reset_exposure_time(self, ms: float) -> float:
                self.calls.append(ms)
                return ms

        cam = MockCamera()
        actual, changed = apply_auto_exposure(
            cam, peak=100.0, current_ms=10.0, target_brightness=180.0, tol=0.2
        )
        assert changed is True
        assert actual == pytest.approx(18.0)
        assert cam.calls == [pytest.approx(18.0)]

    def test_unchanged_skips_camera_call(self) -> None:
        class MockCamera:
            def __init__(self) -> None:
                self.calls: list[float] = []

            def reset_exposure_time(self, ms: float) -> float:
                self.calls.append(ms)
                return ms

        cam = MockCamera()
        actual, changed = apply_auto_exposure(
            cam, peak=180.0, current_ms=10.0, target_brightness=180.0, tol=0.2
        )
        assert changed is False
        assert actual == 10.0
        assert cam.calls == []


# ---------------------------------------------------------------------------
# capture_amplitude
# ---------------------------------------------------------------------------
class MockCaptureCamera:
    """Stub exposing only the attributes ``capture_amplitude`` accesses."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.calls: list[tuple[int, bool]] = []
        self.reset_calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.calls.append((n_sample, skip_first))
        # Simulate the camera's internal averaging over n_sample frames.
        return np.mean(self.frames[:n_sample], axis=0)

    def reset_window(self, center: tuple[int, int], size: tuple[int, int]) -> None:
        self.reset_calls.append((center, size))


class TestCaptureAmplitude:
    def test_normalizes_sqrt_intensity(self) -> None:
        img = np.array([[0, 100], [400, 900]], dtype=np.uint16)
        cam = MockCaptureCamera([img])
        amp = capture_amplitude(cam, n_sample=1)

        expected = np.sqrt(img.astype(np.float32))
        expected = expected / expected.max()
        np.testing.assert_allclose(amp, expected, rtol=1e-5, atol=1e-6)
        assert amp.dtype == np.float32
        assert cam.calls == [(1, True)]

    def test_averages_over_n_sample_frames(self) -> None:
        f1 = np.array([[0, 100], [400, 900]], dtype=np.uint16)
        f2 = np.array([[0, 300], [400, 900]], dtype=np.uint16)
        cam = MockCaptureCamera([f1, f2])
        amp = capture_amplitude(cam, n_sample=2)

        mean = np.mean([f1, f2], axis=0).astype(np.float32)
        expected = np.sqrt(mean)
        expected = expected / expected.max()
        np.testing.assert_allclose(amp, expected, rtol=1e-5, atol=1e-6)
        assert cam.calls == [(2, True)]

    def test_reset_window_called_with_int_tuples(self) -> None:
        img = np.array([[0, 100], [400, 900]], dtype=np.uint16)
        cam = MockCaptureCamera([img])
        capture_amplitude(cam, center=[10.5, 20.5], size=[30, 40], n_sample=1)
        assert cam.reset_calls == [((10, 20), (30, 40))]

    def test_reset_window_skipped_when_center_or_size_missing(self) -> None:
        img = np.array([[0, 100], [400, 900]], dtype=np.uint16)
        cam = MockCaptureCamera([img])
        capture_amplitude(cam, center=(5, 5), n_sample=1)
        assert cam.reset_calls == []

    def test_reset_window_failure_is_swallowed(self) -> None:
        class RaisingResetCamera(MockCaptureCamera):
            def reset_window(self, center: tuple[int, int], size: tuple[int, int]) -> None:
                raise RuntimeError("no window support")

        img = np.array([[0, 100], [400, 900]], dtype=np.uint16)
        cam = RaisingResetCamera([img])
        amp = capture_amplitude(cam, center=(5, 5), size=(10, 10), n_sample=1)

        expected = np.sqrt(img.astype(np.float32))
        expected = expected / expected.max()
        np.testing.assert_allclose(amp, expected, rtol=1e-5, atol=1e-6)

    def test_nan_values_are_zeroed(self) -> None:
        img = np.array([[np.nan, 100.0]], dtype=np.float32)
        cam = MockCaptureCamera([img])
        amp = capture_amplitude(cam)
        np.testing.assert_allclose(amp, np.array([[0.0, 1.0]], dtype=np.float32))

    def test_all_zero_frame_stays_zero(self) -> None:
        img = np.zeros((4, 4), dtype=np.uint16)
        cam = MockCaptureCamera([img])
        amp = capture_amplitude(cam)
        np.testing.assert_array_equal(amp, np.zeros((4, 4), dtype=np.float32))