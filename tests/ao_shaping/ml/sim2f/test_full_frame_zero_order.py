"""0-order location from a native frame (issue #235).

``argmax`` is not a locator: a single hot pixel wins, and a clipped flat-top
spot answers in whole pixels only.  ``box_centroid`` integrates the half-maximum
region around an ``argmax`` seed instead, so the reported 0-order is both
sub-pixel accurate and insensitive to defective pixels.
"""

from __future__ import annotations

import numpy as np

from ml.sim2f.full_frame import box_centroid


def _gaussian_spot(shape, center_row, center_col, sigma, peak=1000.0, background=5.0):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    core = peak * np.exp(-((rows - center_row) ** 2 + (cols - center_col) ** 2) / (2 * sigma**2))
    return core + background


def test_centroid_recovers_a_sub_pixel_centre():
    truth = (70.4, 101.6)
    frame = _gaussian_spot((140, 180), *truth, sigma=3.0)
    seed_row, seed_col = np.unravel_index(int(np.argmax(frame)), frame.shape)
    row, col = box_centroid(frame, row=int(seed_row), col=int(seed_col))
    assert abs(row - truth[0]) < 0.1
    assert abs(col - truth[1]) < 0.1


def test_centroid_ignores_an_isolated_hot_pixel():
    frame = _gaussian_spot((90, 90), 40.0, 52.0, sigma=2.5)
    frame[10, 80] = 60000.0  # defective pixel, far from the 0-order
    seed_row, seed_col = np.unravel_index(int(np.argmax(frame)), frame.shape)
    assert (seed_row, seed_col) == (10, 80)  # argmax alone is fooled
    row, col = box_centroid(frame, row=int(seed_row), col=int(seed_col))
    # The box around the hot pixel holds no real spot, so the seed is returned
    # instead of a meaningless centroid; the caller then re-seeds on the window.
    assert (row, col) == (10.0, 80.0)


def test_centroid_of_a_flat_top_spot_is_its_geometric_centre():
    frame = np.zeros((60, 60), dtype=np.float64)
    frame[25:36, 25:36] = 1000.0  # 11x11 saturated square, centre at 30, 30
    row, col = box_centroid(frame, row=25, col=25)
    assert abs(row - 30.0) < 0.01
    assert abs(col - 30.0) < 0.01


def test_centroid_falls_back_to_the_seed_on_an_empty_frame():
    frame = np.zeros((40, 40), dtype=np.float64)
    assert box_centroid(frame, row=17, col=21) == (17.0, 21.0)


def test_centroid_stays_inside_the_frame_at_an_edge():
    frame = _gaussian_spot((40, 40), 1.0, 38.0, sigma=2.0)
    row, col = box_centroid(frame, row=0, col=39)
    assert 0.0 <= row <= 39.0
    assert 0.0 <= col <= 39.0