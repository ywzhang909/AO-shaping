"""``_records_from_cache`` read a real 1-D ``c_flat.npy`` as if it were rectangular.

The cache writer (:mod:`ml.gsnet_debug.cache`) stores ``c_flat.npy`` as a **1-D
concatenation** of every record's coefficients and puts the boundaries in
``c_offsets.npy``, keeping each record's term count exact -- its own reader slices
``flat[start:stop]``. ``_records_from_cache`` never looked at ``c_offsets`` and indexed
``coeffs[index]`` instead, which on a 1-D array is a *scalar* drawn from whichever record
that offset happens to fall in. Every record therefore got one stray number in slot 0 and
zeros everywhere else, silently, and those numbers are the ground truth used to report the
per-record fit error.

The existing cache test hid this by writing a 2-D ``(3, 28)`` array, a shape no writer
produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.model_in_loop_shaping import _records_from_cache

REGION = 8
N_COEFFS = 4


def _cache(
    tmp_path: Path,
    *,
    c_flat: np.ndarray,
    c_offsets: np.ndarray | None,
    n_frames: int = 3,
    img_shape: tuple[int, int] = (REGION, REGION),
) -> Path:
    directory = tmp_path / "run.gsnet_cache"
    directory.mkdir()
    rng = np.random.default_rng(0)
    np.save(
        directory / "img_flat.npy",
        rng.random((n_frames, img_shape[0] * img_shape[1])),
    )
    np.save(directory / "c_flat.npy", c_flat)
    if c_offsets is not None:
        np.save(directory / "c_offsets.npy", c_offsets)
    (directory / "meta.json").write_text(
        json.dumps({"img_shape": list(img_shape)}), encoding="utf-8"
    )
    return directory


class TestRealCacheLayout:
    def test_concatenated_coefficients_are_sliced_by_offsets(self, tmp_path: Path) -> None:
        rows = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        cache = _cache(
            tmp_path,
            c_flat=np.concatenate(rows),
            c_offsets=np.array([0, 4, 8, 12], dtype=np.int64),
        )

        records = _records_from_cache(cache, region=REGION, n_coeffs=N_COEFFS)

        assert len(records) == 3
        for index, expected in enumerate(rows):
            np.testing.assert_allclose(records[index].coefficients, expected)

    def test_ragged_term_counts_survive_a_fixed_model_size(self, tmp_path: Path) -> None:
        # The writer keeps counts exact, so records may hold fewer terms than n_coeffs.
        rows = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]]
        cache = _cache(
            tmp_path,
            c_flat=np.concatenate(rows),
            c_offsets=np.array([0, 2, 5, 6], dtype=np.int64),
        )

        records = _records_from_cache(cache, region=REGION, n_coeffs=N_COEFFS)

        for index, expected in enumerate(rows):
            actual = records[index].coefficients
            assert actual is not None and actual.shape == (N_COEFFS,)
            np.testing.assert_allclose(actual[: len(expected)], expected)
            np.testing.assert_allclose(actual[len(expected) :], 0.0)

    def test_flat_array_without_offsets_is_rejected_not_guessed(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, c_flat=np.arange(12, dtype=np.float64), c_offsets=None)

        with pytest.raises(ValueError, match="c_offsets"):
            _records_from_cache(cache, region=REGION, n_coeffs=N_COEFFS)

    def test_offsets_that_do_not_span_the_array_are_rejected(self, tmp_path: Path) -> None:
        cache = _cache(
            tmp_path,
            c_flat=np.arange(12, dtype=np.float64),
            c_offsets=np.array([0, 4, 8, 10], dtype=np.int64),
        )

        with pytest.raises(ValueError, match="c_offsets"):
            _records_from_cache(cache, region=REGION, n_coeffs=N_COEFFS)


class TestRectangularLayout:
    def test_two_dimensional_rows_still_load(self, tmp_path: Path) -> None:
        rows = np.arange(12, dtype=np.float64).reshape(3, 4)
        cache = _cache(tmp_path, c_flat=rows, c_offsets=None)

        records = _records_from_cache(cache, region=REGION, n_coeffs=N_COEFFS)

        for index in range(3):
            np.testing.assert_allclose(records[index].coefficients, rows[index])

    def test_row_count_disagreeing_with_the_frames_is_rejected(self, tmp_path: Path) -> None:
        # Two coefficient rows for three frames used to hand the third record None.
        cache = _cache(
            tmp_path,
            c_flat=np.zeros((2, N_COEFFS), dtype=np.float64),
            c_offsets=None,
            n_frames=3,
        )

        with pytest.raises(ValueError, match="row|rows"):
            _records_from_cache(cache, region=REGION, n_coeffs=N_COEFFS)
