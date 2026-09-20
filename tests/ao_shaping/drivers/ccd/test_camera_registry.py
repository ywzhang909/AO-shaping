"""Offline tests for the CCD camera-type registry and exposure helpers.

No hardware or SDK is required: the registry is exercised through a stub camera
class registered at test time.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.ccd.common import (
    CAMERA_TYPES,
    auto_exposure,
    create_camera,
    get_camera_exposure_ms,
    get_camera_exposure_range,
    list_camera_types,
    register_camera,
    set_camera_exposure_ms,
)


class _StubCamera:
    """Minimal camera exercising the registry contract (constructor never opens)."""

    def __init__(self, cam_id=0, exposure_time_ms=20.0, **kwargs):
        self.cam_id = cam_id
        self.exposure_time_ms = float(exposure_time_ms)
        self.opened = False
        self.kwargs = kwargs
        self.frames: list[int] = []

    def open(self):
        self.opened = True
        return self

    def get_numpy_image(self, n_sample=1, skip_first=True):
        self.frames.append(n_sample)
        # Peak brightness scales linearly with exposure (10x), so a proportional
        # controller must drive exposure toward target/10.
        return np.full((4, 4), self.exposure_time_ms * 10.0, dtype=np.float64)


class _NativeAutoExposureCamera(_StubCamera):
    """Stub exposing a Daheng-style native ``auto_exposure`` method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_target = None

    def auto_exposure(self, target_max=0.5, **kwargs):
        self.native_target = target_max
        return np.full((4, 4), 7, dtype=np.uint8)


def test_builtin_types_registered_and_sorted():
    types = list_camera_types()
    for expected in ("daheng", "miicam", "ffmpeg", "image_folder"):
        assert expected in types
    assert types == sorted(types)


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown camera type"):
        create_camera("definitely-not-a-camera", cam_id=0)


def test_create_camera_returns_unopened_instance_and_filters_kwargs():
    register_camera(
        "stub_cam_filtered",
        _StubCamera,
        accepted_kwargs={"cam_id", "exposure_time_ms"},
    )

    cam = create_camera(
        "STUB_CAM_FILTERED", cam_id=3, exposure_time_ms=5.0, ignored=123
    )

    assert isinstance(cam, _StubCamera)
    assert cam.opened is False
    assert cam.cam_id == 3
    assert cam.exposure_time_ms == 5.0
    assert "ignored" not in cam.kwargs


def test_register_camera_forwards_all_kwargs_when_unrestricted():
    register_camera("stub_cam_all_kwargs", _StubCamera)

    cam = create_camera("stub_cam_all_kwargs", cam_id=1, extra="kept")

    assert cam.opened is False
    assert cam.kwargs == {"extra": "kept"}


def test_register_camera_rejects_bad_target():
    with pytest.raises(TypeError):
        register_camera("stub_cam_bad", 123)  # type: ignore[arg-type]


def test_exposure_helpers_roundtrip_and_default_range():
    cam = _StubCamera(exposure_time_ms=12.5)

    assert get_camera_exposure_ms(cam) == pytest.approx(12.5)

    set_camera_exposure_ms(cam, 3.0)
    assert get_camera_exposure_ms(cam) == pytest.approx(3.0)

    lo, hi = get_camera_exposure_range(cam)
    assert lo == pytest.approx(0.011)
    assert hi == pytest.approx(10_000.0)


def test_exposure_helpers_prefer_reset_exposure_time():
    class _ResetCamera(_StubCamera):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reset_calls: list[float] = []

        def reset_exposure_time(self, time_ms):
            self.reset_calls.append(float(time_ms))
            self.exposure_time_ms = float(time_ms)
            return float(time_ms)

    cam = _ResetCamera(exposure_time_ms=10.0)
    set_camera_exposure_ms(cam, 2.5)

    assert cam.reset_calls == [2.5]
    assert get_camera_exposure_ms(cam) == pytest.approx(2.5)


def test_exposure_range_prefers_backend_properties():
    class _RangedCamera(_StubCamera):
        @property
        def min_exposure_ms(self):
            return 0.2

        @property
        def max_exposure_ms(self):
            return 350.0

    lo, hi = get_camera_exposure_range(_RangedCamera())
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(350.0)


def test_get_exposure_raises_when_unsupported():
    class _NoExposureCamera:
        pass

    with pytest.raises(AttributeError):
        get_camera_exposure_ms(_NoExposureCamera())


def test_auto_exposure_delegates_to_native_path():
    cam = _NativeAutoExposureCamera()

    img = auto_exposure(cam, 40)

    assert cam.native_target == 40
    assert img.dtype == np.uint8
    # Native path must NOT run the generic proportional loop.
    assert cam.frames == []


def test_auto_exposure_generic_loop_converges_and_uses_helper_path():
    cam = _StubCamera(exposure_time_ms=20.0)

    img = auto_exposure(cam, target_max=40.0, tolerance=0.05, max_iterations=6)

    assert np.max(img) == pytest.approx(40.0, rel=0.2)
    assert get_camera_exposure_ms(cam) < 20.0
    assert len(cam.frames) >= 2


def test_camera_types_contains_expected_builtins():
    assert {"daheng", "miicam"}.issubset(CAMERA_TYPES.keys())
