"""Tests for zernike_matrix_runner snapshot capture.

Covers the single-frame snapshot contract of ``_capture_wfs_full_state``
(previously flagged with FIXME "读取的数据似乎一样"): all three readings
(spot deviation / Zernike coefficients / wavefront) derive from ONE
``take_image()`` acquisition — identical provenance between reads is the
intended frozen-snapshot semantics for closed-loop control, not a fault.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from ao_shaping.runners.slm.zernike_matrix_runner import (
    WfsStateSnapshot,
    _capture_wfs_full_state,
)


def _make_mock_wfs() -> Mock:
    wfs = Mock()
    wfs.take_image.return_value = None
    wfs.get_spot_deviation.return_value = (np.float64(0.5), np.float64(-0.25))
    wfs.get_zernike.return_value = np.arange(10.0)
    wf_2d = np.zeros((8, 8))
    wfs.get_wavefront.return_value = (wf_2d, {"rms": 0.123})
    return wfs


def test_capture_wfs_full_state_single_frame_contract() -> None:
    """One take_image() precedes all reads; every reading comes from that frame."""
    wfs = _make_mock_wfs()

    snap = _capture_wfs_full_state(wfs, cancel_tile=False, zernike_order=10)

    assert isinstance(snap, WfsStateSnapshot)
    # 单帧契约: take_image 恰好一次, 且先于所有读数
    wfs.take_image.assert_called_once_with()
    assert [c[0] for c in wfs.mock_calls] == [
        "take_image",
        "get_spot_deviation",
        "get_zernike",
        "get_wavefront",
    ]


def test_capture_wfs_full_state_passthrough_and_fields() -> None:
    """cancel_tile / zernike_order forwarded; snapshot fields mirror the WFS."""
    wfs = _make_mock_wfs()

    snap = _capture_wfs_full_state(wfs, cancel_tile=True, zernike_order=5)

    wfs.get_spot_deviation.assert_called_once_with(cancel_tile=True)
    wfs.get_zernike.assert_called_once_with(zernike_order=5)
    wfs.get_wavefront.assert_called_once_with(cancel_tile=True)

    assert snap.dev_x == np.float64(0.5)
    assert snap.dev_y == np.float64(-0.25)
    np.testing.assert_array_equal(snap.zernike_coeffs, np.arange(10.0))
    np.testing.assert_array_equal(snap.wavefront, np.zeros((8, 8)))
    assert snap.rms == 0.123


def test_capture_wfs_full_state_rms_nan_when_stats_missing() -> None:
    """rms falls back to NaN when the WFS returns empty stats."""
    wfs = Mock()
    wfs.take_image.return_value = None
    wfs.get_spot_deviation.return_value = (0.0, 0.0)
    wfs.get_zernike.return_value = np.zeros(3)
    wfs.get_wavefront.return_value = (np.zeros((2, 2)), {})

    snap = _capture_wfs_full_state(wfs, cancel_tile=False, zernike_order=3)

    assert np.isnan(snap.rms)
