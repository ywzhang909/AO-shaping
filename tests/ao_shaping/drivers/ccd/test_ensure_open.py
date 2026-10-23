"""Offline proof for :meth:`BaseCamera._ensure_open` (R-64 / CCD device guards).

The three CCD drivers used to guard every SDK call with
``assert self.cam, "camera not initialized"``.  That guard validates **external
device state** (did ``initialize()`` actually succeed?), and ``assert`` is
stripped wholesale under ``python -O`` -- after which the driver drives the SDK
with a null handle.  The fix collapses all sites into the inherited
``_ensure_open()`` (the SLM side already had this paradigm in
``Santec._ensure_open``), which raises :class:`CameraError` -- the exception
callers already catch.

These tests pin the contract with no hardware and no vendor SDK: importing
``ao_shaping.drivers.ccd.base`` loads only abc / numpy / loguru (verified by
grep -- the daheng / miicam / WFS SDKs stay behind the package's PEP-562 lazy
attributes), and the fake subclass below implements the abstract methods as
no-ops.
"""

from __future__ import annotations

import pytest

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError


class _NullCamera(BaseCamera):
    """A concrete ``BaseCamera`` with no device behind it.

    ``self.cam`` stays at the ``BaseCamera.__init__`` default (``None``) until a
    test plants a sentinel, which is exactly the state ``_ensure_open`` must
    catch.
    """

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def initialize(self) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def reset_exposure_time(self, time_ms: float) -> float:
        return float(time_ms)

    def reset_window(self, center, size):
        return (0, 0), (0, 0)

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True):
        raise NotImplementedError

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return False

    def get_auto_exposure_state(self) -> dict:
        return {}

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        return False

    @staticmethod
    def get_cam_list():
        return []


def test_ensure_open_raises_camera_error_when_cam_is_none() -> None:
    """Un-initialized device => ``CameraError``, not ``AssertionError``."""
    camera = _NullCamera()
    with pytest.raises(CameraError) as exc_info:
        camera._ensure_open()
    # The whole point of leaving `assert`: under `python -O` an AssertionError
    # would not exist at all, and even when asserts run, a *device* guard must
    # carry the exception type callers already catch.
    assert not issubclass(type(exc_info.value), AssertionError)


def test_ensure_open_passes_when_device_is_present() -> None:
    """A non-``None`` sentinel makes the default probe truthful."""
    camera = _NullCamera()
    camera.cam = object()
    assert camera._ensure_open() is None


def test_ensure_open_preserves_custom_detail() -> None:
    """The ``detail`` argument is the exception message, verbatim."""
    camera = _NullCamera()
    with pytest.raises(CameraError, match="camera not found"):
        camera._ensure_open("camera not found")


def test_device_open_default_tracks_cam() -> None:
    """The inherited ``_device_open`` is exactly ``self.cam is not None``."""
    camera = _NullCamera()
    # The fake does not override the probe, so what follows is the *default*.
    assert _NullCamera._device_open is BaseCamera._device_open
    assert camera._device_open() is False
    camera.cam = object()
    assert camera._device_open() is True
