"""Offline tests for the SLM-PIB camera/centre/exposure helpers (no hardware).

Covers the three real failure modes found in the Daheng camera path of
``slm-pib``:

* an empty threshold mask (corner holds the brightest pixels) used to make
  ``centroid()`` raise ``ValueError: cannot convert float NaN to integer``;
* a spot near a frame edge used to abort the run inside
  ``DahengCamera.reset_window`` (negative ROI offset assert);
* the auto-exposure option used to be unreachable behind the fixed-exposure
  branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    clamp_center_to_frame,
    resolve_initial_exposure,
    threshold_spot_center,
)


def _spot_frame(size=64, center=(40, 20), peak=200.0, sigma=3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    img = peak * np.exp(
        -(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * sigma**2))
    )
    return img.astype(np.uint8)


class TestThresholdSpotCenter:
    def test_finds_a_normal_spot(self):
        cx, cy = threshold_spot_center(_spot_frame())
        assert abs(cx - 40) <= 2
        assert abs(cy - 20) <= 2

    def test_corner_brightest_does_not_raise(self):
        """Hot pixel in the corner patch made the mask empty -> NaN -> ValueError."""
        img = _spot_frame(peak=120.0)
        img[0:2, 0:2] = 255  # corner patch (size//50 -> 2) holds the brightest pixels

        cx, cy = threshold_spot_center(img)  # must not raise

        assert np.isfinite(cx) and np.isfinite(cy)
        assert abs(cx - 40) <= 3
        assert abs(cy - 20) <= 3

    def test_all_dark_returns_frame_centre(self):
        assert threshold_spot_center(np.zeros((50, 80), dtype=np.uint8)) == (40, 25)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            threshold_spot_center(np.zeros((4, 4, 3), dtype=np.uint8))


class TestClampCenterToFrame:
    def test_near_edge_is_clamped_so_the_roi_fits(self):
        # 250x250 ROI in a 200x200 frame: a valid centre is impossible, but the
        # returned centre must still be inside the frame (driver would assert).
        x, y = clamp_center_to_frame((5, 5), (200, 200), 100)
        assert (x, y) == (50, 50)
        assert 0 <= x < 200 and 0 <= y < 200

    def test_interior_center_is_unchanged(self):
        assert clamp_center_to_frame((125, 125), (512, 512), 200) == (125, 125)

    def test_negative_center_is_clamped_positive(self):
        x, y = clamp_center_to_frame((-30, -30), (512, 512), 200)
        assert x == 100 and y == 100

    def test_window_larger_than_frame_stays_in_frame(self):
        x, y = clamp_center_to_frame((5, 5), (120, 90), 250)
        assert 0 <= x < 90 and 0 <= y < 120


class TestResolveInitialExposure:
    def test_fixed_exposure_wins(self):
        assert resolve_initial_exposure(80.0, 40) == ("fixed", 80.0)

    def test_auto_exposure_when_no_fixed(self):
        assert resolve_initial_exposure(0.0, 40) == ("auto", 40.0)

    def test_keep_current_when_both_zero(self):
        assert resolve_initial_exposure(0.0, 0) == ("keep", 0.0)

    def test_auto_exposure_now_reachable_with_runner_defaults(self):
        # The runner defaults are -t 0.0 / -b 40, which must select auto-exposure.
        assert resolve_initial_exposure(0.0, 40)[0] == "auto"
