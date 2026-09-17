"""Tests for slm_lut_runner spot-location geometry (2f Fourier system).

The optical train is a strict 2f system (SLM at front focal plane, camera at
back focal plane).  The lens Fourier-transforms the SLM field, so both half-
screen gratings' +1 orders land on the SAME camera row as the 0-order, with
x-offsets given by ``_expected_x_offset(period)``.  The 0-order (optical
axis) is the frame global maximum, which need NOT be the frame center.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_lut_runner import _expected_x_offset, _locate_spots


def _make_frame(
    width: int,
    height: int,
    period_ref: int,
    period_test: int,
    cx0: int,
    cy0: int,
    side: int = +1,
    noise: float = 0.5,
) -> np.ndarray:
    """Synthetic 2f calibration frame: three Gaussian spots on one row."""
    frame = np.zeros((height, width), dtype=np.float64)
    x_off_ref = _expected_x_offset(period_ref)
    x_off_test = _expected_x_offset(period_test)
    spots = [
        (cx0, cy0, 180.0),  # 0-order (brightest)
        (cx0 + side * x_off_ref, cy0, 120.0),  # ref +1
        (cx0 + side * x_off_test, cy0, 90.0),  # test +1
    ]
    yy, xx = np.mgrid[0:height, 0:width]
    for sx, sy, amp in spots:
        frame += amp * np.exp(
            -(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 12.0**2))
        )
    if noise:
        rng = np.random.default_rng(0)
        frame += rng.uniform(0, noise, (height, width))
    return frame


class TestLocateSpots2fGeometry:
    def test_all_orders_on_same_row(self):
        """0-order, ref and test +1 all share the optical-axis row."""
        period_ref, period_test = 64, 32
        cx0, cy0 = 1450, 642  # off-center optical axis (as measured)
        frame = _make_frame(2688, 1520, period_ref, period_test, cx0, cy0)
        spots = _locate_spots(frame, period_ref, period_test, spot_window=41)

        x_off_ref = _expected_x_offset(period_ref)
        x_off_test = _expected_x_offset(period_test)

        assert spots["center"] == (cx0, cy0)
        assert spots["ref"] == (cx0 + x_off_ref, cy0)
        assert spots["test"] == (cx0 + x_off_test, cy0)
        # Same row as center (the whole point of the 2f fix)
        assert spots["ref"][1] == spots["center"][1]
        assert spots["test"][1] == spots["center"][1]

    def test_negative_side(self):
        """Both +1 orders on the same (left) side of the 0-order."""
        period_ref, period_test = 64, 32
        cx0, cy0 = 800, 400
        frame = _make_frame(1600, 800, period_ref, period_test, cx0, cy0, side=-1)
        spots = _locate_spots(frame, period_ref, period_test, spot_window=41)

        x_off_ref = _expected_x_offset(period_ref)
        x_off_test = _expected_x_offset(period_test)
        assert spots["center"] == (cx0, cy0)
        assert spots["ref"] == (cx0 - x_off_ref, cy0)
        assert spots["test"] == (cx0 - x_off_test, cy0)

    def test_optical_axis_not_frame_center(self):
        """0-order is the frame global max, not w//2 — the old geometry bug."""
        period_ref, period_test = 64, 32
        width, height = 2688, 1520
        frame = _make_frame(width, height, period_ref, period_test, 1450, 642)
        # Prove frame center ≠ optical axis
        assert (width // 2, height // 2) != (1450, 642)
        spots = _locate_spots(frame, period_ref, period_test, spot_window=41)
        assert spots["center"] == (1450, 642)
        assert spots["ref"][1] == 642

    def test_single_side_determination_consistent(self):
        """Ref side must determine test side — same diffraction order."""
        period_ref, period_test = 96, 40
        cx0, cy0 = 700, 350
        frame = _make_frame(1400, 700, period_ref, period_test, cx0, cy0, side=+1)
        spots = _locate_spots(frame, period_ref, period_test, spot_window=41)
        x_off_test = _expected_x_offset(period_test)
        # test spot on the SAME side as ref (tolerance allows pixel noise)
        assert abs((spots["test"][0] - spots["center"][0]) - x_off_test) <= 2

    def test_raises_when_spots_absent(self):
        """Flat frame (no diffraction) must fail loudly, not guess."""
        frame = np.full((800, 800), 10.0, dtype=np.float64)
        with pytest.raises(SystemExit):
            _locate_spots(frame, 64, 32, spot_window=41)

    def test_raises_when_only_zero_order_present(self):
        """Only a 0-order (panel not modulating) must not fabricate spots."""
        frame = np.zeros((800, 800), dtype=np.float64)
        yy, xx = np.mgrid[0:800, 0:800]
        frame += 200.0 * np.exp(-(((xx - 400) ** 2 + (yy - 400) ** 2) / 288.0))
        with pytest.raises(SystemExit):
            _locate_spots(frame, 64, 32, spot_window=41)