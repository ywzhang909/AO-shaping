"""判据① v2 质心口径的离线测试：信号掩膜 + 无信号 fail-closed。

2026-11-27 修订背景：旧的无掩膜基线质心被全帧背景噪声主导（噪声权重 61–72%），
输出的是"光斑 + 噪声"的加权混合，不是光斑位置。本文件锁定 v2 的行为契约，
不改 ``tests/ao_shaping/scripts/test_capture_s1_mapping_disambiguation_20261124.py``
里的冻结采集测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.slm.capture_s1_mapping_disambiguation_20261124 import (
    SIGNAL_PEAK_SNR_MIN,
    SIGNAL_SIGMA,
    _centroid_px,
    flat_signal_measurement,
)

ROOT = Path(__file__).resolve().parents[3]
FROZEN_FLAT = (
    ROOT
    / "data/debug/slm_transfer/s1_verify_20261126_retry/raw/00_flat_scale993.npy"
)
A1_REFERENCE_PX = (1105.249, 1227.556)
TOLERANCE_PX = 5.0


def _synthetic_frame(peak: float, spot: tuple[float, float]) -> np.ndarray:
    """Poisson background plus a deterministic Gaussian spot (no hardware)."""
    rng = np.random.default_rng(20261127)
    rows, cols = 200, 300
    frame = rng.poisson(20, size=(rows, cols)).astype(np.float64)
    rr, cc = np.mgrid[0:rows, 0:cols]
    dr = rr - spot[0]
    dc = cc - spot[1]
    frame += peak * np.exp(-(dr * dr + dc * dc) / (2.0 * 3.0 * 3.0))
    return frame


def test_signal_masked_centroid_finds_spot_not_frame_centre() -> None:
    spot = (60.3, 210.7)
    result = flat_signal_measurement(_synthetic_frame(2000.0, spot))
    assert result["has_signal"] is True
    assert result["centroid_is_fallback"] is False
    assert result["signal_sigma"] == SIGNAL_SIGMA
    assert result["signal_peak_snr_min"] == SIGNAL_PEAK_SNR_MIN
    row, col = result["centroid_px"]
    assert abs(row - spot[0]) < 1.0
    assert abs(col - spot[1]) < 1.0
    assert np.hypot(row - 99.5, col - 149.5) > 50.0


def test_signal_weight_is_dominated_by_the_spot_region() -> None:
    spot = (60.3, 210.7)
    frame = _synthetic_frame(2000.0, spot)
    result = flat_signal_measurement(frame)
    threshold = result["signal_threshold_counts"]
    weights = np.where(frame >= threshold, np.maximum(frame - np.median(frame), 0), 0)
    rr, cc = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    near = (np.abs(rr - spot[0]) <= 10) & (np.abs(cc - spot[1]) <= 10)
    assert float(weights[near].sum() / weights.sum()) > 0.99
    assert result["signal_pixels"] < frame.size // 10


def test_blank_frame_is_fail_closed_to_geometric_centre() -> None:
    frame = np.full((200, 300), 20.0)
    result = flat_signal_measurement(frame)
    assert result["has_signal"] is False
    assert result["centroid_is_fallback"] is True
    assert result["centroid_px"] == [99.5, 149.5]
    assert result["masked_centroid_px"] is None
    assert result["sigma_counts"] == 1.0
    assert result["sigma_is_floor"] is True


def test_no_signal_frame_does_not_report_the_noise_cluster_as_measurement() -> None:
    rng = np.random.default_rng(777)
    frame = rng.poisson(20, size=(200, 300)).astype(np.float64)
    result = flat_signal_measurement(frame)
    assert result["has_signal"] is False
    assert result["centroid_is_fallback"] is True
    assert result["centroid_px"] == [99.5, 149.5]


def test_dim_spot_fails_the_peak_snr_gate() -> None:
    frame = _synthetic_frame(200.0, (60.3, 210.7))
    result = flat_signal_measurement(frame)
    assert result["peak_snr"] < SIGNAL_PEAK_SNR_MIN
    assert result["has_signal"] is False
    assert result["centroid_is_fallback"] is True


def test_centroid_wrapper_matches_the_measurement() -> None:
    frame = _synthetic_frame(2000.0, (60.3, 210.7))
    assert _centroid_px(frame) == flat_signal_measurement(frame)["centroid_px"]


@pytest.mark.skipif(not FROZEN_FLAT.exists(), reason="frozen raw frame unavailable")
def test_frozen_11_26_flat_is_within_a1_tolerance_under_v2() -> None:
    result = flat_signal_measurement(np.load(FROZEN_FLAT))
    assert result["has_signal"] is True
    row, col = result["centroid_px"]
    dist = float(np.hypot(row - A1_REFERENCE_PX[0], col - A1_REFERENCE_PX[1]))
    assert dist <= TOLERANCE_PX
