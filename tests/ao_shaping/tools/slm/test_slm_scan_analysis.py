"""Tests for ``ao_shaping.tools.slm.slm_scan_analysis`` (pure numpy/stdlib helpers).

Covers the 7 public symbols extracted verbatim from the Zernike report scripts
and ``slm_shift_calib.py``: ``LINEARITY_AMPS``, ``outlier_mask``,
``clamp_shift``, ``parabolic_min``, ``latest_match``, ``group_raw_scan`` and
``analyze_linearity`` (all four verdict strings + skip conditions).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_scan_analysis import (
    LINEARITY_AMPS,
    analyze_linearity,
    clamp_shift,
    group_raw_scan,
    latest_match,
    outlier_mask,
    parabolic_min,
)


# ---------------------------------------------------------------------------
# outlier_mask
# ---------------------------------------------------------------------------

def test_outlier_mask_keeps_values_below_factor_times_median() -> None:
    arr = np.array([1.0, 2.0, 3.0, 100.0])
    mask = outlier_mask(arr, 3.0)
    assert mask.tolist() == [True, True, True, False]


def test_outlier_mask_factor_one() -> None:
    arr = np.array([1.0, 2.0, 3.0, 100.0])
    mask = outlier_mask(arr, 1.0)
    assert mask.tolist() == [True, True, False, False]


def test_outlier_mask_zero_factor_keeps_nothing() -> None:
    arr = np.array([1.0, 2.0, 3.0, 100.0])
    mask = outlier_mask(arr, 0.0)
    assert mask.tolist() == [False, False, False, False]


def test_outlier_mask_huge_factor_keeps_everything() -> None:
    arr = np.array([1.0, 2.0, 3.0, 100.0])
    mask = outlier_mask(arr, 1e9)
    assert mask.tolist() == [True, True, True, True]


def test_outlier_mask_zero_median_keeps_everything() -> None:
    arr = np.array([0.0, 0.0, 0.0])
    mask = outlier_mask(arr, 3.0)
    assert mask.tolist() == [True, True, True]


def test_outlier_mask_negative_median_keeps_everything() -> None:
    arr = np.array([-1.0, -2.0, -3.0])
    mask = outlier_mask(arr, 3.0)
    assert mask.tolist() == [True, True, True]


def test_outlier_mask_preserves_shape_and_dtype() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 100.0]])
    mask = outlier_mask(arr, 3.0)
    assert mask.shape == (2, 2)
    assert mask.dtype == bool
    assert mask.tolist() == [[True, True], [True, False]]


# ---------------------------------------------------------------------------
# clamp_shift
# ---------------------------------------------------------------------------

def test_clamp_shift_rounds() -> None:
    assert clamp_shift(3.4, 10) == 3
    assert clamp_shift(3.6, 10) == 4
    assert clamp_shift(-3.6, 10) == -4


def test_clamp_shift_clips_to_limit() -> None:
    assert clamp_shift(12.0, 10) == 10
    assert clamp_shift(-12.0, 10) == -10


def test_clamp_shift_zero() -> None:
    assert clamp_shift(0.0, 10) == 0
    assert clamp_shift(0.4, 10) == 0


# ---------------------------------------------------------------------------
# parabolic_min
# ---------------------------------------------------------------------------

def test_parabolic_min_exact_vertex() -> None:
    pts = [(0.0, 10.0), (3.0, 1.0), (6.0, 10.0)]  # y = (x-3)^2 + 1
    assert parabolic_min(pts) == pytest.approx(3.0)


def test_parabolic_min_fewer_than_three_points_returns_none() -> None:
    assert parabolic_min([]) is None
    assert parabolic_min([(0.0, 1.0)]) is None
    assert parabolic_min([(0.0, 1.0), (1.0, 2.0)]) is None


def test_parabolic_min_edge_minimum_returns_edge_x() -> None:
    assert parabolic_min([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]) == 0.0
    assert parabolic_min([(0.0, 3.0), (1.0, 2.0), (2.0, 1.0)]) == 2.0


def test_parabolic_min_degenerate_denominator_returns_middle_x() -> None:
    pts = [(0.0, 5.0), (1e-13, 1.0), (1.0, 5.0)]
    assert parabolic_min(pts) == pytest.approx(1e-13)


def test_parabolic_min_near_zero_curvature_returns_middle_x() -> None:
    pts = [(0.0, 2.0), (1.0, 2.0 - 1e-13), (2.0, 2.0)]
    assert parabolic_min(pts) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# latest_match
# ---------------------------------------------------------------------------

def test_latest_match_returns_lexicographically_last_file() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "scan_1.json").write_text("{}")
        (tmp / "scan_2.json").write_text("{}")
        result = latest_match(str(tmp / "scan_*.json"))
        assert result == tmp / "scan_2.json"
    finally:
        shutil.rmtree(tmp)


def test_latest_match_no_match_returns_none() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        assert latest_match(str(tmp / "none_*.json")) is None
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# group_raw_scan
# ---------------------------------------------------------------------------

def test_group_raw_scan_to_waves_divides_by_0_532() -> None:
    raw = [{"dll_index": 5, "radius": 300, "amp_rad": 2.0, "sign": 1,
            "readout_um": [0.0, 0.532, 1.064]}]
    g = group_raw_scan(raw, to_waves=True)
    assert np.allclose(g[(5, 300.0)][2.0][1], [0.0, 1.0, 2.0])


def test_group_raw_scan_keeps_raw_um_when_not_to_waves() -> None:
    raw = [{"dll_index": 5, "radius": 300, "amp_rad": 2.0, "sign": 1,
            "readout_um": [0.0, 0.532, 1.064]}]
    g = group_raw_scan(raw, to_waves=False)
    assert np.allclose(g[(5, 300.0)][2.0][1], [0.0, 0.532, 1.064])


def test_group_raw_scan_key_types_and_dtype() -> None:
    raw = [{"dll_index": 5, "radius": 300, "amp_rad": 2.0, "sign": 1,
            "readout_um": [0.0, 0.532, 1.064]}]
    g = group_raw_scan(raw, to_waves=False)
    key = next(iter(g))
    assert isinstance(key[0], int)
    assert isinstance(key[1], float)
    amp_key = next(iter(g[key]))
    assert isinstance(amp_key, float)
    sign_key = next(iter(g[key][amp_key]))
    assert isinstance(sign_key, int)
    assert g[key][amp_key][sign_key].dtype == np.float64


# ---------------------------------------------------------------------------
# analyze_linearity
# ---------------------------------------------------------------------------

def _readout(m: int, length: int, diag_val: float, baseline: float) -> np.ndarray:
    """Vector with ``diag_val`` at index ``m`` and ``baseline`` elsewhere."""
    z = np.full(length, baseline, dtype=float)
    z[m] = diag_val
    return z


def _group_with_diag(m: int, length: int, diag_vals: list[float],
                     baseline: float) -> dict:
    """Group for ``amps=(2.0, 5.0, 10.0)`` with the given diag values."""
    amps = (2.0, 5.0, 10.0)
    d: dict[float, dict[int, np.ndarray]] = {}
    for a, dv in zip(amps, diag_vals):
        d[a] = {1: _readout(m, length, dv, baseline)}
        d[-a] = {-1: _readout(m, length, -dv, baseline)}
    return {(m, 300.0): d}


def test_analyze_linearity_proportional() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 1.0], baseline=0.0)
    rows = analyze_linearity(groups)
    assert len(rows) == 1
    row = rows[0]
    assert row["m"] == 0
    assert row["R"] == 300.0
    assert row["diag"] == pytest.approx([0.2, 0.5, 1.0])
    assert row["vec"] == pytest.approx([0.0, 0.0, 0.0])
    assert row["base"] == pytest.approx(0.0)
    assert row["k"] == pytest.approx(0.1)
    assert row["r2"] == pytest.approx(1.0)
    assert row["cv"] == pytest.approx(0.0)
    assert row["ratio_10_2"] == pytest.approx(5.0)
    assert row["snr"] == float("inf")
    assert row["verdict"] == "成比例"


def test_analyze_linearity_proportional_weak_coupling() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 1.0], baseline=0.5)
    rows = analyze_linearity(groups)
    assert len(rows) == 1
    row = rows[0]
    assert row["base"] == pytest.approx(1.5)
    assert row["snr"] == pytest.approx(1.0 / 1.5)
    assert row["verdict"] == "成比例 (弱耦合)"


def test_analyze_linearity_noise_limited() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 0.3], baseline=0.5)
    rows = analyze_linearity(groups)
    assert len(rows) == 1
    row = rows[0]
    assert row["r2"] == pytest.approx(-1.3605, abs=1e-3)
    assert row["cv"] == pytest.approx(0.4304, abs=1e-3)
    assert row["snr"] == pytest.approx(0.2)
    assert row["verdict"] == "噪声受限"


def test_analyze_linearity_not_proportional() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 0.3], baseline=0.001)
    rows = analyze_linearity(groups)
    assert len(rows) == 1
    row = rows[0]
    assert row["base"] == pytest.approx(0.003)
    assert row["snr"] == pytest.approx(100.0)
    assert row["verdict"] == "**不成比例**"


def test_analyze_linearity_skips_missing_sign_pair() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 1.0], baseline=0.0)
    # drop the negative-sign readout for the largest amplitude
    del groups[(0, 300.0)][-10.0]
    rows = analyze_linearity(groups)
    assert rows == []


def test_analyze_linearity_skips_missing_amplitude() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 1.0], baseline=0.0)
    del groups[(0, 300.0)][10.0]
    rows = analyze_linearity(groups)
    assert rows == []


def test_analyze_linearity_skips_fewer_than_three_amplitudes() -> None:
    groups = _group_with_diag(0, 10, [0.2, 0.5, 1.0], baseline=0.0)
    rows = analyze_linearity(groups, amps=(2.0, 5.0))
    assert rows == []


def test_linearity_amps_constant() -> None:
    assert LINEARITY_AMPS == (2.0, 5.0, 10.0)