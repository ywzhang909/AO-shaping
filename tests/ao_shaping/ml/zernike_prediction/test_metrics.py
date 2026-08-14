"""Tests for ml.zernike_prediction.metrics (T2: regression metrics for Zernike prediction).

Conventions under test:
- pred/true are ``(N, 65)`` or ``(N, 66)`` (metadata (n, m) order, piston at col 0).
- All metrics compare **non-piston coefficients only**: the piston (index 0,
  always 1.0 by construction) is skipped whenever a 66-wide array is involved.
- Phase metrics use the circular wrapped-phase difference on a (192, 192) grid
  so they stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.zernike_prediction import (
    COEFF_ORDER_NAMES,
    alignment_ok,
    coefficient_names,
    iter_nm_terms,
    mae,
    metrics_summary,
    mse,
    per_coeff_mae,
    per_order_mae,
    phase_mae,
    phase_rmse,
    r2,
    rmse,
)

_N_TERMS = 65  # non-piston coefficients
_N_FULL = 66  # including piston
_PHASE_SIZE = (192, 192)


def test_mae_rmse_known() -> None:
    """Hand-computed values on a tiny 1-D vector."""
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.zeros(5)
    assert mae(pred, true) == pytest.approx(3.0)  # (1+2+3+4+5)/5
    assert mse(pred, true) == pytest.approx(11.0)  # (1+4+9+16+25)/5
    assert rmse(pred, true) == pytest.approx(np.sqrt(11.0))


def test_piston_alignment_65_vs_66() -> None:
    """65-col pred vs 66-col true must equal 66-col (piston 0) vs 66-col true."""
    rng = np.random.default_rng(7)
    pred_65 = rng.normal(size=(8, _N_TERMS))
    true_66 = np.column_stack([np.ones(8), rng.normal(size=(8, _N_TERMS))])
    pred_66 = np.column_stack([np.zeros(8), pred_65])

    assert alignment_ok(pred_65, true_66)
    assert alignment_ok(pred_66, true_66)

    for metric in (mae, rmse, mse, r2):
        assert metric(pred_65, true_66) == pytest.approx(metric(pred_66, true_66), rel=1e-12)


def test_r2_perfect_and_worst() -> None:
    """R2 = 1 for identical data, ~0 for mean prediction, nan for constant true."""
    true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert r2(true.copy(), true) == pytest.approx(1.0, abs=1e-10)

    # predicting the global mean baseline -> R2 = 0 exactly (SS_res == SS_tot)
    pred_mean = np.full_like(true, true.mean())
    assert r2(pred_mean, true) == pytest.approx(0.0, abs=1e-6)

    const_true = np.full_like(true, 3.0)
    assert np.isnan(r2(np.zeros_like(true), const_true))


def test_per_coeff_and_per_order() -> None:
    """Per-coefficient MAE shape (65,); per-order MAE groups by radial order n."""
    rng = np.random.default_rng(11)
    pred = rng.normal(size=(16, _N_TERMS))
    true = pred + rng.normal(scale=0.1, size=(16, _N_TERMS))

    pcm = per_coeff_mae(pred, true)
    assert pcm.shape == (_N_TERMS,)

    pom = per_order_mae(pred, true)
    assert set(pom.keys()) == set(range(1, 11))

    # verify the n -> indices mapping against iter_nm_terms(10), skipping (0, 0)
    # per_coeff_mae is indexed over the 65 non-piston terms = terms[1:]
    non_piston = iter_nm_terms(10)[1:]
    for n in range(1, 11):
        idx = [j for j, (nn, _m) in enumerate(non_piston) if nn == n]
        assert pom[n] == pytest.approx(float(np.mean(pcm[idx])), rel=1e-12)


def test_phase_metrics_circular() -> None:
    """Circular wrapped-phase metrics: zero for identical, sane for perturbed."""
    rng = np.random.default_rng(3)
    coeffs_a = np.zeros(_N_FULL)
    coeffs_a[0] = 1.0
    idx = rng.choice(np.arange(1, _N_FULL), size=10, replace=False)
    coeffs_a[idx] = rng.uniform(-1.0, 1.0, 10)

    # identical coefficients -> identical wrapped phase -> zero error
    assert phase_mae(coeffs_a, coeffs_a, _PHASE_SIZE) < 1e-6
    assert phase_rmse(coeffs_a, coeffs_a, _PHASE_SIZE) < 1e-6

    # perturb one term by 0.5 waves -> error is positive and bounded by pi
    coeffs_b = coeffs_a.copy()
    coeffs_b[int(idx[0])] += 0.5
    pr = phase_rmse(coeffs_a, coeffs_b, _PHASE_SIZE)
    assert pr > 0.0
    assert pr < np.pi

    # 65-col input is equivalent to 66-col (piston 1.0 prepended internally)
    pr_65 = phase_rmse(coeffs_a[1:], coeffs_b[1:], _PHASE_SIZE)
    assert pr_65 == pytest.approx(pr, rel=1e-6)


def test_metrics_summary_keys() -> None:
    """metrics_summary contains every documented key with correct types/shapes."""
    rng = np.random.default_rng(5)
    pred = rng.normal(size=(12, _N_TERMS))
    true = pred + rng.normal(scale=0.2, size=(12, _N_TERMS))

    summary = metrics_summary(pred, true, phase_size=_PHASE_SIZE)
    assert set(summary.keys()) == {
        "mae",
        "rmse",
        "mse",
        "r2",
        "per_coeff_mae",
        "per_order_mae",
        "phase_mae",
        "phase_rmse",
        "n_samples",
    }
    assert isinstance(summary["mae"], float)
    assert isinstance(summary["rmse"], float)
    assert isinstance(summary["mse"], float)
    assert isinstance(summary["r2"], float)
    assert isinstance(summary["per_coeff_mae"], np.ndarray)
    assert summary["per_coeff_mae"].shape == (_N_TERMS,)
    assert isinstance(summary["per_order_mae"], dict)
    assert set(summary["per_order_mae"].keys()) == set(range(1, 11))
    assert summary["n_samples"] == 12
    assert summary["mae"] == pytest.approx(mae(pred, true), rel=1e-12)


def test_shape_mismatch_raises() -> None:
    """Incompatible shapes: alignment_ok False and metrics raise ValueError."""
    pred = np.zeros((4, _N_TERMS))
    true = np.zeros((4, 60))
    assert not alignment_ok(pred, true)
    with pytest.raises(ValueError, match="align"):
        mae(pred, true)
    with pytest.raises(ValueError, match="align"):
        metrics_summary(pred, true)


def test_coefficient_names() -> None:
    """coefficient_names = COEFF_ORDER_NAMES minus the piston term."""
    names = coefficient_names()
    assert len(names) == _N_TERMS
    assert names == COEFF_ORDER_NAMES(10)[1:]
    assert names[0] == "n1m-1"
    assert names[-1] == "n10m10"
