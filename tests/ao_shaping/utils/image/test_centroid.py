"""Tests for the shared windowed centroid primitive (issue #199 R6).

``ao_shaping.utils.image.centroid.windowed_centroid`` is the single
implementation behind ``frame_analysis.measure_spot`` and
``displacement._centroid_in_window``. Besides the primitive's own contract,
these tests pin **bitwise equivalence** with the two historical inline
implementations it replaced, so the dedup cannot silently move a spot.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.centroid import windowed_centroid


def _legacy_measure_spot_centroid(sm: np.ndarray, box: int) -> tuple[float, float]:
    """The exact pre-refactor body of ``frame_analysis.measure_spot``."""
    cy, cx = (int(v) for v in np.unravel_index(int(sm.argmax()), sm.shape))
    r = int(box)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, sm.shape[0])
    x0, x1 = max(cx - r, 0), min(cx + r + 1, sm.shape[1])
    patch = np.maximum(sm[y0:y1, x0:x1] - sm[y0:y1, x0:x1].min(), 0.0)
    tot = float(patch.sum())
    if tot <= 0:
        return float(cx), float(cy)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((patch * xx).sum() / tot), float((patch * yy).sum() / tot)


def _legacy_centroid_in_window(image: np.ndarray, win: int) -> tuple[float, float]:
    """The exact pre-refactor body of ``displacement._centroid_in_window``."""
    row, col = np.unravel_index(int(np.argmax(image)), image.shape)
    half = max(2, int(win) // 2)
    row_slice = slice(max(0, row - half), min(image.shape[0], row + half + 1))
    col_slice = slice(max(0, col - half), min(image.shape[1], col + half + 1))
    block = np.clip(
        image[row_slice, col_slice].astype(float)
        - float(image[row_slice, col_slice].min()),
        0,
        None,
    )
    total = float(block.sum())
    if total <= 0:
        return float(col), float(row)
    yy, xx = np.mgrid[: block.shape[0], : block.shape[1]]
    return (
        float((xx * block).sum() / total) + col_slice.start,
        float((yy * block).sum() / total) + row_slice.start,
    )


def _gaussian_spot(size: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


class TestWindowedCentroid:
    def test_centres_a_symmetric_spot(self) -> None:
        frame = _gaussian_spot(64, 32.0, 32.0)
        cx, cy = windowed_centroid(frame, 10)
        assert cx == pytest.approx(32.0, abs=1e-9)
        assert cy == pytest.approx(32.0, abs=1e-9)

    def test_tracks_an_off_centre_spot(self) -> None:
        frame = _gaussian_spot(64, 41.3, 22.7)
        cx, cy = windowed_centroid(frame, 12)
        assert cx == pytest.approx(41.3, abs=0.2)
        assert cy == pytest.approx(22.7, abs=0.2)

    def test_pedestal_does_not_drag_the_centroid(self) -> None:
        frame = _gaussian_spot(64, 30.0, 34.0) + 50.0  # uniform DC pedestal
        cx, cy = windowed_centroid(frame, 10)
        assert cx == pytest.approx(30.0, abs=0.2)
        assert cy == pytest.approx(34.0, abs=0.2)

    def test_flat_frame_falls_back_to_the_peak_pixel(self) -> None:
        frame = np.zeros((32, 32))
        frame[7, 9] = 5.0  # the only positive dynamic range is one pixel
        assert windowed_centroid(frame, 8) == (9.0, 7.0)

    def test_window_is_clipped_at_the_frame_border(self) -> None:
        # A half-width far exceeding the frame must clip, not crash -- and the
        # clipped window's centroid stays pulled toward the border spot (an
        # asymmetric window biases it away from the true centre, which is the
        # documented behaviour both former call sites rely on).
        frame = _gaussian_spot(32, 1.0, 1.0)
        cx, cy = windowed_centroid(frame, 20)
        assert 0.0 <= cx <= 31.0 and 0.0 <= cy <= 31.0
        cx, cy = windowed_centroid(frame, 4)
        assert 0.0 <= cx < 3.0 and 0.0 <= cy < 3.0

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            windowed_centroid(np.zeros(8), 3)


class TestLegacyEquivalence:
    """The two historical call sites must observe bitwise-identical results."""

    @pytest.mark.parametrize("box", [1, 4, 10, 40])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_former_measure_spot_body(self, box: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # A smoothed, despiked frame is what measure_spot feeds the centroid.
        sm = _gaussian_spot(96, 37.0 + rng.normal(), 61.0 + rng.normal(), sigma=4.0)
        sm = sm + 0.05 * rng.random(sm.shape)
        assert windowed_centroid(sm, box) == _legacy_measure_spot_centroid(sm, box)

    @pytest.mark.parametrize("win", [4, 16, 60])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_former_centroid_in_window_body(self, win: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        base = _gaussian_spot(96, 30.0, 66.0, sigma=3.0)
        # A difference frame can be negative on one side: exactly the
        # displacement module's input.
        diff = base - np.roll(base, (3, -5), axis=(0, 1))
        expected = _legacy_centroid_in_window(diff, win)
        got = windowed_centroid(diff, max(2, win // 2))
        # The legacy body summed RELATIVE window coordinates and added the
        # slice offset after the division; the shared primitive sums ABSOLUTE
        # coordinates. Mathematically identical, but the two summation orders
        # differ in the last ULP (~1e-13 px) -- far below the 0.1 px quantum the
        # displacement records round to, hence the tolerance here (the
        # measure_spot body above, which already used absolute coordinates, is
        # pinned bitwise).
        assert got == pytest.approx(expected, abs=1e-9)

    def test_negative_valued_frame_is_handled_like_the_legacy_body(self) -> None:
        diff = np.zeros((24, 24))
        diff[10, 12] = 4.0
        diff[5, 6] = -3.0
        # The legacy signature takes the window SIDE; the shared primitive takes
        # the half-width. ``_centroid_in_window`` maps side -> max(2, side//2).
        expected = _legacy_centroid_in_window(diff, 6)
        got = windowed_centroid(diff, max(2, 6 // 2))
        assert got == pytest.approx(expected, abs=1e-9)
