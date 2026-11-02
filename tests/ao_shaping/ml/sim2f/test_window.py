"""Geometry and protocol tests for :mod:`ml.sim2f.window`.

The window is pure bookkeeping on top of the full-frame physics, so the tests
here pin exactly that: which pixels a window exposes, how a crop/paste round trip
behaves for numpy and torch, and that :class:`WindowedBench` reports the window
through ``forward`` while leaving ``full_forward`` whole.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.sim2f.config import Sim2fConfig
from ml.sim2f.optics import Sim2f
from ml.sim2f.window import CameraWindow, WindowedBench, crop_measure


def test_offset_clips_and_rounds() -> None:
    """A window that hangs off the frame is clipped, not wrapped."""
    low = CameraWindow(center=(5, 5), size=(32, 32), frame=(48, 48))
    assert low.offset() == (0, 0)
    high = CameraWindow(center=(44, 44), size=(32, 32), frame=(48, 48))
    assert high.offset() == (16, 16)
    fractional = CameraWindow(center=(10.4, 10.6), size=(8, 8), frame=(48, 48))
    assert fractional.offset() == (6, 7)


def test_window_larger_than_frame_is_rejected() -> None:
    """A window wider than the sensor cannot be cropped."""
    window = CameraWindow(center=(8, 8), size=(64, 64), frame=(48, 48))
    with pytest.raises(ValueError, match="does not fit"):
        window.offset()


def test_non_square_window_rejected_by_bench() -> None:
    """The bench protocol needs a square window; a rectangle is a hard error."""
    bench = Sim2f(Sim2fConfig(grid=64, ccd_px=96), device="cpu")
    window = CameraWindow(center=(48, 48), size=(32, 48), frame=(96, 96))
    with pytest.raises(ValueError, match="square window"):
        WindowedBench(bench, window)


def test_crop_paste_round_trip_numpy() -> None:
    """Cropping then pasting restores the window pixels exactly."""
    frame = np.arange(48 * 48, dtype=np.float64).reshape(48, 48)
    window = CameraWindow(center=(24, 20), size=(16, 16), frame=(48, 48))
    patch = window.crop(frame)
    assert patch.shape == (16, 16)
    restored = window.paste(patch, np.zeros_like(frame))
    x0, y0 = window.offset()
    assert np.array_equal(restored[y0 : y0 + 16, x0 : x0 + 16], patch)
    assert restored.sum() == patch.sum()
    assert np.array_equal(window.crop(restored), patch)


def test_crop_paste_round_trip_torch_leading_dims() -> None:
    """Leading batch/channel dimensions survive crop and paste."""
    frame = torch.arange(2 * 1 * 48 * 48, dtype=torch.float32).reshape(2, 1, 48, 48)
    window = CameraWindow(center=(30, 22), size=(12, 12), frame=(48, 48))
    patch = window.crop(frame)
    assert patch.shape == (2, 1, 12, 12)
    restored = window.paste(patch, torch.zeros_like(frame))
    x0, y0 = window.offset()
    assert torch.equal(restored[..., y0 : y0 + 12, x0 : x0 + 12], patch)
    assert torch.equal(window.crop(restored), patch)


def test_mask_matches_offset() -> None:
    """The mask is exactly the rectangle :meth:`offset` describes."""
    window = CameraWindow(center=(12, 40), size=(8, 8), frame=(48, 48))
    mask = window.mask((48, 48))
    assert mask.shape == (48, 48)
    assert mask.sum() == 64
    x0, y0 = window.offset()
    assert mask[y0 : y0 + 8, x0 : x0 + 8].all()


def test_windowed_bench_forward_shapes() -> None:
    """``forward`` is the window, ``full_forward`` is the sensor."""
    bench = Sim2f(Sim2fConfig(grid=64, ccd_px=96), device="cpu")
    window = CameraWindow(center=(48, 36), size=(32, 32), frame=(96, 96))
    wrapped = WindowedBench(bench, window)
    phase = torch.zeros(1, 64, 64)
    cropped = wrapped.forward(phase)
    full = wrapped.full_forward(phase)
    assert cropped.shape == (1, 1, 32, 32)
    assert full.shape == (1, 1, 96, 96)
    assert wrapped.full_ccd_px == 96
    assert torch.allclose(cropped, window.crop(full))
    assert wrapped.window.frame == (96, 96)


def test_crop_measure_crops_tuple_frame_only() -> None:
    """A ``(frame, extras)`` measurement keeps its extras untouched."""
    window = CameraWindow(center=(16, 16), size=(8, 8), frame=(32, 32))
    frame = np.ones((1, 1, 32, 32), dtype=np.float32)

    def measure(_phase: object) -> tuple[np.ndarray, str]:
        return frame, "meta"

    wrapped = crop_measure(measure, window)
    cropped, extras = wrapped(None)
    assert cropped.shape == (1, 1, 8, 8)
    assert extras == "meta"
    plain = crop_measure(lambda _p: frame, window)
    assert plain(None).shape == (1, 1, 8, 8)
