"""Capture I/O helpers: the in-memory frames and the background writer.

Both exist because the 2026-10-29 campaign measured ~5.1 s between two
captures, of which 1.16 s was ``np.savez_compressed`` and another ~0.15 s was
re-opening the file it had just written. Neither helper may change *what* is
recorded -- only when the bytes reach the disk and whether a consumer has to go
there at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import validate


@pytest.fixture(autouse=True)
def _clear_capture_cache():
    validate._CAPTURE_FRAMES.clear()
    yield
    validate._CAPTURE_FRAMES.clear()


def test_a_remembered_capture_is_served_without_touching_the_disk(tmp_path):
    raw = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)
    validate.remember_capture("capture_0000.npz", raw)

    # No file exists, so reaching for it would raise: the cache must answer.
    assert np.array_equal(validate.capture_frames(tmp_path, "capture_0000.npz"), raw)


def test_an_unseen_capture_falls_back_to_the_npz(tmp_path):
    raw = np.arange(6, dtype=np.uint16).reshape(2, 1, 3)
    np.savez(tmp_path / "capture_0001.npz", raw_frames=raw)

    assert np.array_equal(validate.capture_frames(tmp_path, "capture_0001.npz"), raw)


def test_the_cache_is_bounded_and_evicts_the_oldest(tmp_path):
    for index in range(validate._CAPTURE_FRAMES_MAX + 2):
        validate.remember_capture(f"capture_{index:04d}.npz", np.zeros((1, 1, 1), np.uint16))

    assert len(validate._CAPTURE_FRAMES) == validate._CAPTURE_FRAMES_MAX
    assert "capture_0000.npz" not in validate._CAPTURE_FRAMES
    assert f"capture_{validate._CAPTURE_FRAMES_MAX + 1:04d}.npz" in validate._CAPTURE_FRAMES


def test_the_writer_persists_what_it_is_given(tmp_path):
    writer = validate._CaptureWriter()
    payload = dict(raw_frames=np.ones((2, 3, 4), np.uint16), model_phase=np.zeros((2, 2)))
    writer.submit(tmp_path / "capture_0000.npz", payload)
    writer.close()

    with np.load(tmp_path / "capture_0000.npz") as saved:
        assert np.array_equal(saved["raw_frames"], payload["raw_frames"])
        assert np.array_equal(saved["model_phase"], payload["model_phase"])


def test_a_failed_write_is_raised_by_close_not_swallowed(tmp_path):
    writer = validate._CaptureWriter()
    # A directory where the file should be makes np.savez_compressed fail
    # inside the worker thread; the loop must still learn about it.
    (tmp_path / "capture_0000.npz").mkdir()
    writer.submit(tmp_path / "capture_0000.npz", dict(raw_frames=np.zeros((1, 1, 1))))

    with pytest.raises(OSError):
        writer.close()


def test_writes_stay_in_submission_order(tmp_path):
    writer = validate._CaptureWriter()
    for index in range(3):
        writer.submit(tmp_path / f"capture_{index:04d}.npz",
                      dict(raw_frames=np.full((1, 1, 1), index, np.uint16)))
    writer.close()

    for index in range(3):
        with np.load(tmp_path / f"capture_{index:04d}.npz") as saved:
            assert int(saved["raw_frames"][0, 0, 0]) == index
