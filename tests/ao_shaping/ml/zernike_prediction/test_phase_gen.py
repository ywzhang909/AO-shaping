"""Tests for ml.zernike_prediction.phase_gen (T1: capture-exact Zernike phase math + DCT unwrap).

The correctness anchor is ``test_golden_0414_exact_match``: the generated image
must reproduce the stored ``phase.csv`` bit-for-bit (``np.array_equal``), not
approximately. All unwrap tests use amplitudes verified (empirically) to stay
below the pi-per-pixel sampling limit of the DCT Poisson solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.zernike_prediction import (
    COEFF_ORDER_NAMES,
    build_basis_maps,
    coefficients_to_phase_radians,
    coefficients_to_wrapped_gray,
    count_zernike_terms,
    gray_to_wrapped,
    iter_nm_terms,
    load_stored_gray,
    metadata_order_to_noll,
    non_piston_indices,
    unwrap_phase_lsq,
)
from ml.zernike_prediction.phase_gen import zernike_radial

_GOLDEN_DIR = Path("data/slm_dual_spot/20260414_171241/sample_0000")
_TWO_PI = 2.0 * np.pi


def test_radial_known_values() -> None:
    """Hand-computed standard Zernike radial polynomials (Noll normalization)."""
    rs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(zernike_radial(0, 0, rs), np.ones_like(rs), atol=1e-12)
    np.testing.assert_allclose(zernike_radial(2, 0, rs), 2 * rs**2 - 1, atol=1e-12)
    np.testing.assert_allclose(zernike_radial(2, 2, rs), rs**2, atol=1e-12)
    np.testing.assert_allclose(zernike_radial(3, 1, rs), 3 * rs**3 - 2 * rs, atol=1e-12)
    np.testing.assert_allclose(zernike_radial(3, -1, rs), 3 * rs**3 - 2 * rs, atol=1e-12)
    np.testing.assert_allclose(zernike_radial(4, 0, rs), 6 * rs**4 - 6 * rs**2 + 1, atol=1e-12)
    # invalid pairs: |m| > n or (n-|m|) odd
    with pytest.raises(ValueError):
        zernike_radial(2, 1, rs)


def test_golden_0414_exact_match() -> None:
    """Correctness anchor: regenerate the stored phase.csv with 100% exact match."""
    if not (_GOLDEN_DIR / "metadata.json").exists():
        pytest.skip("requires slm_dual_spot data")
    meta = json.loads((_GOLDEN_DIR / "metadata.json").read_text())
    phase_params = meta["phase_params"]
    coeffs = np.asarray(phase_params["coefficients"], dtype=np.float64)
    n_max = int(phase_params["n_max"])
    assert count_zernike_terms(n_max) == 66
    stored = load_stored_gray(_GOLDEN_DIR)
    regen = coefficients_to_wrapped_gray(coeffs, n_max=n_max)  # full 1200x1920
    assert regen.shape == stored.shape == (1200, 1920)
    assert np.array_equal(regen, stored)


def test_generation_self_consistent() -> None:
    """Random coeffs -> wrapped gray -> unwrap -> basis fit -> recover within 1e-2."""
    size = (600, 960)
    n_max = 10
    rng = np.random.default_rng(42)
    coeffs = np.zeros(count_zernike_terms(n_max))
    coeffs[0] = 1.0
    coeffs[1:] = rng.uniform(-0.25, 0.25, 65)
    gray = coefficients_to_wrapped_gray(coeffs, n_max=n_max, height=size[0], width=size[1])
    wrapped = gray_to_wrapped(gray)
    basis, mask_u8 = build_basis_maps(n_max=n_max, height=size[0], width=size[1])
    mask = mask_u8.astype(bool)
    unwrapped = unwrap_phase_lsq(wrapped, mask)
    fit, *_ = np.linalg.lstsq(basis[:, mask].T, unwrapped[mask], rcond=None)
    recovered = fit / _TWO_PI
    max_err = float(np.abs(recovered[1:] - coeffs[1:]).max())
    assert max_err < 1e-2


def test_coeff_count_and_order() -> None:
    """Term counting, metadata order, and human labels."""
    assert count_zernike_terms(10) == 66
    assert count_zernike_terms(2) == 6
    assert count_zernike_terms(0) == 1
    terms = iter_nm_terms(10)
    assert len(terms) == 66
    assert terms[0] == (0, 0)
    assert all((n - abs(m)) % 2 == 0 for n, m in terms)
    assert all(n >= 0 and abs(m) <= n for n, m in terms)
    assert iter_nm_terms(2) == [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
    assert non_piston_indices(10) == list(range(1, 66))
    assert COEFF_ORDER_NAMES(2) == ["n0m0", "n1m-1", "n1m1", "n2m-2", "n2m0", "n2m2"]
    with pytest.raises(ValueError):
        iter_nm_terms(-1)
    with pytest.raises(ValueError):
        count_zernike_terms(-1)


def test_65_prepends_piston() -> None:
    """Passing 65 coefficients is equivalent to 66 with piston 1.0 prepended."""
    rng = np.random.default_rng(7)
    coeffs65 = rng.uniform(-0.3, 0.3, 65)
    img65 = coefficients_to_wrapped_gray(coeffs65, height=300, width=480)
    img66 = coefficients_to_wrapped_gray(
        np.concatenate([[1.0], coeffs65]), height=300, width=480
    )
    assert np.array_equal(img65, img66)
    # invalid lengths rejected
    with pytest.raises(ValueError):
        coefficients_to_wrapped_gray(np.zeros(64), height=100, width=100)


def test_unwrap_roundtrip() -> None:
    """Known smooth phase -> wrap -> unwrap -> recover up to a constant."""
    size = (600, 960)
    n_max = 6
    terms = iter_nm_terms(n_max)
    rng = np.random.default_rng(3)
    coeffs = np.zeros(count_zernike_terms(n_max))
    coeffs[1:] = rng.uniform(-0.25, 0.25, len(terms) - 1)

    # Unwrapped truth, built from the same public math (no final wrap)
    height, width = size
    radius = min(size) // 2
    x = (np.arange(width, dtype=np.float64) - width / 2.0) / radius
    y = (np.arange(height, dtype=np.float64) - height / 2.0) / radius
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask_g = R <= 1.0
    phase_total = np.zeros((height, width), dtype=np.float64)
    for (n, m), amp in zip(terms, coeffs):
        if abs(amp) < 1e-10:
            continue
        rn = np.asarray(zernike_radial(n, m, R), dtype=np.float64)
        z = rn * np.cos(m * Theta) if m >= 0 else rn * np.sin(-m * Theta)
        phase_total += (z * mask_g) * amp * _TWO_PI

    wrapped = coefficients_to_phase_radians(coeffs, n_max=n_max, size=size)
    unwrapped = unwrap_phase_lsq(wrapped, mask_g)
    err = unwrapped[mask_g] - phase_total[mask_g]
    err = err - err.mean()  # unwrap is defined up to a global constant
    assert np.abs(err).max() < 1e-2


def test_metadata_order_to_noll() -> None:
    """Metadata (n,m) order -> standard Noll order (task-fixed convention)."""
    terms2 = iter_nm_terms(2)  # [(0,0),(1,-1),(1,1),(2,-2),(2,0),(2,2)]
    # Standard Noll: j=1 piston, j=2 tip (1,-1), j=3 tilt (1,1), j=4 defocus
    # (2,0), j=5 astig (2,-2), j=6 astig (2,2)
    noll_terms = [(0, 0), (1, -1), (1, 1), (2, 0), (2, -2), (2, 2)]
    for j, (n, m) in enumerate(noll_terms, start=1):
        vec = np.zeros(6)
        vec[terms2.index((n, m))] = 1.0
        converted = metadata_order_to_noll(vec, n_max=2)
        assert converted[j - 1] == 1.0
        assert converted.sum() == 1.0

    # Full n_max=10 cross-check against aotools zernIndex where available.
    # aotools swaps j=2 (1,1) / j=3 (1,-1) vs the standard table, so skip those.
    terms10 = iter_nm_terms(10)
    try:
        from aotools.functions.zernike import zernIndex
    except ImportError:
        pytest.skip("aotools not available for Noll cross-check")
    for j in range(1, 67):
        if j in (2, 3):
            continue
        n, m = zernIndex(j)
        vec = np.zeros(66)
        vec[terms10.index((int(n), int(m)))] = 1.0
        converted = metadata_order_to_noll(vec, n_max=10)
        assert converted[j - 1] == 1.0
        assert converted.sum() == 1.0
    # aotools itself: j=2 -> (1,1), j=3 -> (1,-1) (opposite of standard)
    assert tuple(zernIndex(2)) == (1, 1)
    assert tuple(zernIndex(3)) == (1, -1)
    assert tuple(zernIndex(4)) == (2, 0)
