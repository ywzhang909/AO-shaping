"""Metric tests for :mod:`ml.sim2f.extrapolate` on synthetic frames.

The point of the module is to score what a model predicted *outside* a camera
window, so these tests use frames whose answers are known analytically: identical
frames must score zero error, a shifted Gaussian must show that shift in the
centroid error, and a window's energy split must equal the mask sum.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.sim2f.extrapolate import (
    extrapolation_metrics,
    moment_metrics,
    open_loop_frame,
)
from ml.sim2f.window import CameraWindow

_N = 64


def _gaussian(cx: float = 32.0, cy: float = 32.0, sigma: float = 3.0) -> np.ndarray:
    """Unit-peak Gaussian on a ``_N x _N`` grid."""
    yy, xx = np.mgrid[0:_N, 0:_N]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))


def test_identical_frames_score_zero() -> None:
    """A perfect prediction has zero L1, unit NCC and no moment error."""
    frame = _gaussian()
    window = CameraWindow(center=(32, 30), size=(20, 20), frame=(_N, _N))
    metrics = extrapolation_metrics(frame, frame, window)
    assert metrics.l1_all == pytest.approx(0.0, abs=1e-12)
    assert metrics.l1_in == pytest.approx(0.0, abs=1e-12)
    assert metrics.l1_out == pytest.approx(0.0, abs=1e-12)
    assert metrics.ncc_all == pytest.approx(1.0, abs=1e-9)
    assert metrics.ncc_in == pytest.approx(1.0, abs=1e-9)
    assert metrics.ncc_out == pytest.approx(1.0, abs=1e-9)
    assert metrics.centroid_err_px == pytest.approx(0.0, abs=1e-9)
    assert metrics.d86_err_px == pytest.approx(0.0, abs=1e-9)
    assert metrics.ee_out_err == pytest.approx(0.0, abs=1e-12)
    assert metrics.energy_total_ratio == pytest.approx(1.0, abs=1e-12)


def test_shifted_gaussian_reports_the_shift() -> None:
    """A prediction displaced by (3, 0) px shows a 3 px centroid error."""
    truth = _gaussian()
    pred = _gaussian(cx=35.0)
    window = CameraWindow(center=(32, 32), size=(24, 24), frame=(_N, _N))
    metrics = extrapolation_metrics(pred, truth, window)
    assert metrics.centroid_err_px == pytest.approx(3.0, abs=0.2)
    assert metrics.d86_err_px < 1.0
    assert metrics.l1_all > 0.0
    assert metrics.ncc_all < 1.0


def test_energy_split_matches_mask_sum() -> None:
    """``ee_in_true`` is the truth energy the window mask selects."""
    truth = _gaussian(cx=30.0, cy=34.0)
    window = CameraWindow(center=(30, 34), size=(20, 20), frame=(_N, _N))
    mask = window.mask((_N, _N)).astype(bool)
    metrics = extrapolation_metrics(truth, truth, window)
    expected = float(truth[mask].sum() / truth.sum())
    assert metrics.ee_in_true == pytest.approx(expected, abs=1e-12)
    assert metrics.ee_in_pred == pytest.approx(expected, abs=1e-12)
    assert 0.0 < metrics.ee_in_true < 1.0


def test_window_none_reports_outside_as_nan() -> None:
    """With no window the whole frame is measured, so ``*_out`` is undefined."""
    truth = _gaussian()
    metrics = extrapolation_metrics(truth, truth, None)
    assert np.isnan(metrics.l1_out)
    assert np.isnan(metrics.ncc_out)
    assert metrics.l1_all == pytest.approx(0.0, abs=1e-12)


def test_shape_mismatch_raises() -> None:
    """A prediction of the wrong sensor size cannot be scored."""
    with pytest.raises(ValueError, match="disagree"):
        extrapolation_metrics(np.ones((8, 8)), np.ones((9, 9)), None)


def test_moment_metrics_d86_of_a_disc() -> None:
    """D86 of a uniform disc equals ``radius * sqrt(0.866)``."""
    yy, xx = np.mgrid[0:_N, 0:_N]
    disc = ((xx - 32) ** 2 + (yy - 32) ** 2 <= 10.0**2).astype(np.float64)
    moments = moment_metrics(disc)
    assert moments["d86_px"] == pytest.approx(10.0 * np.sqrt(0.866), abs=0.6)
    assert moments["cx"] == pytest.approx(32.0, abs=0.1)
    assert moments["cy"] == pytest.approx(32.0, abs=0.1)


class _FixedBench:
    """A bench whose frame is a constant; enough for the size check."""

    def __init__(self, ccd: int) -> None:
        self.ccd = int(ccd)

    def forward(self, _phase: object, **_kwargs: object) -> np.ndarray:
        return np.ones((1, 1, self.ccd, self.ccd), dtype=np.float32)


def test_open_loop_frame_returns_unit_sum_full_frame() -> None:
    """The open-loop path yields one unit-sum whole-sensor frame."""
    frame = open_loop_frame(_FixedBench(32), None, full_ccd_px=32)
    assert frame.shape == (32, 32)
    assert frame.sum() == pytest.approx(1.0, abs=1e-12)


def test_open_loop_frame_rejects_wrong_size() -> None:
    """A prediction that is not the expected sensor size raises."""
    with pytest.raises(ValueError, match="expected a 48x48"):
        open_loop_frame(_FixedBench(32), None, full_ccd_px=48)
