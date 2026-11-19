"""Offline tests for the reusable multi-exposure dark-frame capture CLI.

No hardware, no SDK: the acquisition loop is driven by a duck-typed fake camera
and storage is exercised against a ``tmp_path`` root.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.drivers.ccd.dark_frames import (
    DARK_DIR_NAME,
    MANIFEST_NAME,
    SCHEMA,
    DarkFrame,
    load_dark_frame,
    load_dark_frames,
    save_dark_capture,
    serial_dir_name,
)
from ao_shaping.tools.ccd.capture_darks import (
    build_plan,
    capture_exposure,
    format_plan,
    main as capture_main,
    parse_exposures,
)


class FakeCamera:
    """Minimal camera-like object: contract exposure API + frame source."""

    def __init__(self, frames: list[np.ndarray], readback_ms: float = 0.043):
        self._frames = list(frames)
        self._i = 0
        self._ms = 0.0
        self.readback_ms = readback_ms
        self.calls: list[tuple[int, bool]] = []
        self.set_calls: list[float] = []

    def set_exposure_ms(self, ms: float) -> float:
        self.set_calls.append(float(ms))
        self._ms = float(ms)
        # Device quantises/clamps the request; readback differs from request.
        return self.readback_ms

    def get_exposure_time_ms(self) -> float:
        return self.readback_ms

    def get_numpy_image(self, n_sample: int, skip_first: bool):
        self.calls.append((n_sample, skip_first))
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return frame


# --------------------------------------------------------------------------- #
# capture_exposure (pure acquisition loop)
# --------------------------------------------------------------------------- #
def test_capture_exposure_mean_std_and_readback():
    f1 = np.array([[0.0, 2.0], [4.0, 6.0]])
    f2 = np.array([[2.0, 4.0], [6.0, 8.0]])
    cam = FakeCamera([f1, f2], readback_ms=0.041)

    mean, std, effective = capture_exposure(cam, requested_ms=0.024, n_frames=2)

    assert mean.dtype == np.float32
    assert std.dtype == np.float32
    assert mean.shape == (2, 2)
    np.testing.assert_allclose(mean, (f1 + f2) / 2)
    np.testing.assert_allclose(std, np.std(np.stack([f1, f2]), axis=0))
    # Effective exposure is the device readback, not the request.
    assert effective == pytest.approx(0.041)
    assert effective != pytest.approx(0.024)
    # One frame per averaged read, with the leading frame discarded.
    assert cam.calls == [(1, True), (1, True)]


def test_capture_exposure_preserves_2d_shape_from_rgb():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    cam = FakeCamera([rgb], readback_ms=0.05)

    mean, std, _ = capture_exposure(cam, requested_ms=0.05, n_frames=1)

    assert mean.shape == (3, 4)
    assert std.shape == (3, 4)
    np.testing.assert_allclose(mean, np.full((3, 4), 20.0))


# --------------------------------------------------------------------------- #
# Storage round-trip through save_dark_capture / load_dark_frame
# --------------------------------------------------------------------------- #
def _save(tmp_path, exposure_ms, value):
    mean = np.full((4, 5), value, dtype=np.float32)
    std = np.full((4, 5), 0.5, dtype=np.float32)
    return save_dark_capture(
        "CAM-01", exposure_ms=exposure_ms, mean=mean, std=std, frames=16,
        requested_exposure_ms=exposure_ms, bit_depth=8, full_scale=255.0,
        camera={"cam_type": "daheng"}, root=tmp_path,
    )


def test_storage_round_trip_and_matching(tmp_path):
    _save(tmp_path, 0.024, 10.0)
    _save(tmp_path, 0.047, 20.0)

    loaded = load_dark_frames("CAM-01", root=tmp_path)
    assert len(loaded) == 2
    assert sorted(loaded.exposures_ms) == pytest.approx([0.024, 0.047])
    assert loaded.directory == tmp_path / serial_dir_name("CAM-01") / DARK_DIR_NAME

    exact = load_dark_frame("CAM-01", 0.024, root=tmp_path)
    assert isinstance(exact, DarkFrame)
    assert exact.mean.shape == (4, 5)
    np.testing.assert_allclose(exact.mean, 10.0)

    # Mismatched exposure: None by default (no implicit nearest).
    assert load_dark_frame("CAM-01", 0.043, root=tmp_path) is None
    # Explicit nearest falls back to the closest level (0.047).
    nearest = load_dark_frame("CAM-01", 0.043, root=tmp_path, allow_nearest=True)
    assert isinstance(nearest, DarkFrame)
    assert nearest.exposure_ms == pytest.approx(0.047)


def test_dark_frame_subtract_shape_and_clip(tmp_path):
    _save(tmp_path, 0.024, 10.0)
    dark = load_dark_frame("CAM-01", 0.024, root=tmp_path)
    assert dark is not None

    frame = np.full((4, 5), 8.0, dtype=np.float32)  # below dark -> negative
    no_clip = dark.subtract(frame)
    assert no_clip.shape == (4, 5)
    np.testing.assert_allclose(no_clip, -2.0)

    clipped = dark.subtract(frame, clip=True)
    np.testing.assert_allclose(clipped, 0.0)

    with pytest.raises(ValueError, match="不一致"):
        dark.subtract(np.zeros((2, 2), dtype=np.float32))


def test_manifest_schema_mismatch_rejected(tmp_path):
    _save(tmp_path, 0.024, 10.0)
    manifest = tmp_path / serial_dir_name("CAM-01") / DARK_DIR_NAME / MANIFEST_NAME
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA
    data["schema"] = "bogus/schema/v9"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_dark_frames("CAM-01", root=tmp_path)


# --------------------------------------------------------------------------- #
# Plan / --no-hw (must not touch the device lock)
# --------------------------------------------------------------------------- #
def test_no_hw_never_acquires_device_lock(monkeypatch, capsys):
    import ao_shaping.tools.kernels.device_mutex as mutex

    def explode(*_args, **_kwargs):
        raise AssertionError("--no-hw 不得触碰 DeviceLock")

    monkeypatch.setattr(mutex, "DeviceLock", explode)

    rc = capture_main(
        ["--no-hw", "--exposures-ms", "0.02,0.04", "--frames", "3",
         "--config-root", str("D:/tmp/root"), "--serial", "CAM-01"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "0.02, 0.04" in out
    assert "darks" in out  # resolved target dir contains the darks subdir
    assert "不打开设备" in out


def test_parse_exposures_and_plan():
    assert parse_exposures("0.024, 0.043,0.047") == (0.024, 0.043, 0.047)
    with pytest.raises(ValueError):
        parse_exposures("0.01, -1")
    with pytest.raises(ValueError):
        parse_exposures("")

    plan = build_plan(
        (0.02, 0.04), 4, cam_type="daheng", cam_id=0,
        serial="ABC/12", config_root="D:/tmp/root",
    )
    assert plan.serial == "ABC/12"  # raw serial kept for display
    assert plan.target_dir is not None
    assert plan.target_dir.name == DARK_DIR_NAME
    assert "ABC_12" in str(plan.target_dir)  # sanitized in the path
    assert "ABC/12" in format_plan(plan)


def test_serial_dir_name_sanitization():
    assert serial_dir_name("ABC/12:34") == "ABC_12_34"
    assert serial_dir_name("...") == "unknown"
