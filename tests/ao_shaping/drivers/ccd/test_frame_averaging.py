"""Offline checks for camera averaging in raw detector-count units."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.drivers.ccd.common import full_scale


@pytest.mark.parametrize("backend", ["daheng", "miicam"])
@pytest.mark.parametrize("dtype,high", [(np.uint8, 255), (np.uint16, 65535)])
def test_batch_matches_individual_frames(monkeypatch, backend, dtype, high):
    if backend == "daheng":
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera as Camera

        shot_name = "_DahengCamera__take_one_shot"
    else:
        from ao_shaping.drivers.ccd.miicam.driver import MIICamera as Camera

        shot_name = "_MIICamera__take_one_shot"

    frames = [
        np.array([[0, high], [high, 1]], dtype=dtype),
        np.array([[1, high], [high, 2]], dtype=dtype),
        np.array([[0, high], [high, 3]], dtype=dtype),
    ]
    cam = object.__new__(Camera)
    cam._bit_depth = 8 if dtype == np.uint8 else 16
    iterator = iter(frames)
    monkeypatch.setattr(Camera, shot_name, lambda self: next(iterator))
    batch = cam.get_numpy_image(n_sample=3, skip_first=False)

    iterator = iter(frames)
    individual = np.mean(
        [cam.get_numpy_image(n_sample=1, skip_first=False) for _ in frames], axis=0
    )
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch, individual, rtol=0, atol=1e-5)
    assert batch[0, 0] == pytest.approx(1 / 3)
    assert batch[0, 1] == high
    assert full_scale(batch) == high


@pytest.mark.parametrize("backend", ["daheng", "miicam"])
def test_skip_first_discards_exactly_one_frame(monkeypatch, backend):
    if backend == "daheng":
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera as Camera

        shot_name = "_DahengCamera__take_one_shot"
    else:
        from ao_shaping.drivers.ccd.miicam.driver import MIICamera as Camera

        shot_name = "_MIICamera__take_one_shot"
    cam = object.__new__(Camera)
    cam._bit_depth = 8
    frames = iter([np.array([[99]], np.uint8), np.array([[1]], np.uint8), np.array([[2]], np.uint8)])
    monkeypatch.setattr(Camera, shot_name, lambda self: next(frames))
    assert cam.get_numpy_image(n_sample=2, skip_first=True)[0, 0] == 1.5


@pytest.mark.parametrize("backend", ["daheng", "miicam"])
def test_nonpositive_sample_count_rejected(backend):
    if backend == "daheng":
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera as Camera
    else:
        from ao_shaping.drivers.ccd.miicam.driver import MIICamera as Camera
    with pytest.raises(AssertionError, match="Sample count"):
        object.__new__(Camera).get_numpy_image(n_sample=0)


def test_ffmpeg_batch_matches_individual_frames(monkeypatch):
    from ao_shaping.drivers.ccd.ffmpeg.driver import FFmpegCamera

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())

    frames = [
        np.array([[0, 255]], np.uint8),
        np.array([[1, 255]], np.uint8),
        np.array([[0, 255]], np.uint8),
    ]

    class Capture:
        def __init__(self):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def read(self):
            return True, next(self.frames)

    cam = object.__new__(FFmpegCamera)
    cam._cap = Capture()
    cam._frame_count = 0
    cam._is_file = False
    cam.skip_sampling = False
    cam.exposure_time_ms = 0
    batch = cam.get_numpy_image(n_sample=3, skip_first=False)
    cam._cap = Capture()
    individual = np.mean(
        [cam.get_numpy_image(n_sample=1, skip_first=False) for _ in frames], axis=0
    )
    np.testing.assert_allclose(batch, individual, rtol=0, atol=1e-5)
    assert batch[0, 0] == pytest.approx(1 / 3)
    assert full_scale(batch) == 255
