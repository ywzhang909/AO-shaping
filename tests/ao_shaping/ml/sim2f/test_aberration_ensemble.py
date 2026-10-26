"""Tests for the sim2f-vs-OOPAO aberration-ensemble validation script.

The script itself is a report generator, so what is pinned here is the part that
can be wrong silently:

* the **scoring** (``compare``) -- a known-good estimate must pass, a distorted
  one must fail, and the amplitude-matching mode must remove exactly a global
  scale factor and nothing else;
* the **pupil geometry** -- the OOPAO pupil must span the model's grid
  (``grid * pupil_pitch_m``), not the Zernike disc.  That distinction is a real
  bug this validation found: with the disc diameter every screen is sampled 4/3
  finer and every variance comes out 0.62x too small;
* the two **estimators** -- symmetry, positivity of the diagonal, and the
  agreement band that the 4/3 bug would blow through.

OOPAO-dependent tests skip when the library is unavailable, like
``tests/ao_shaping/drivers/sim/test_oopao_backend.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

validate = pytest.importorskip("scripts.ml.validate_sim2f_aberration_ensemble")


def _samples(covariance: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """``n`` draws from ``N(0, covariance)`` as a ``(n, K)`` matrix."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(covariance + 1e-18 * np.eye(covariance.shape[0]))
    return rng.standard_normal((n, covariance.shape[0])) @ chol.T


def _covariance(k: int = 5, seed: int = 1) -> np.ndarray:
    """A well-conditioned, strongly correlated test covariance."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(k, k)) * 0.5
    factor += np.eye(k) * 1.5
    cov = factor @ factor.T
    return 0.5 * (cov + cov.T)


class TestScoring:
    def test_a_consistent_estimate_passes(self):
        cov = _covariance()
        draws = _samples(cov, 4000)
        estimate = draws.T @ draws / draws.shape[0]
        result = validate.compare(estimate, cov, n_samples=4000)
        assert result["verdict"] == "consistent within sampling noise"
        assert result["rms_over_floor"] < 1.0
        assert result["noise_floor"] == pytest.approx(np.sqrt(2.0 / 4000))
        assert result["amplitude_scale"] == pytest.approx(1.0, rel=0.1)
        # Each entry's error sits at the floor, so the majority must be inside it.
        assert result["fraction_within_floor"] > 0.6

    def test_a_doubled_diagonal_fails_and_is_visible_per_mode(self):
        cov = _covariance()
        estimate = 2.0 * cov
        result = validate.compare(estimate, cov, n_samples=4000)
        assert result["verdict"] == "inconsistent"
        assert result["amplitude_scale"] == pytest.approx(2.0)
        assert np.allclose(result["diagonal_ratio"], 2.0)

    def test_amplitude_matching_removes_only_the_global_scale(self):
        cov = _covariance()
        draws = _samples(cov, 2000)
        estimate = draws.T @ draws / draws.shape[0]
        scaled = estimate * 1e4

        plain = validate.compare(scaled, cov, n_samples=2000)
        matched = validate.compare(scaled, cov, n_samples=2000, match_amplitude=True)

        # `amplitude_scale` is the ratio of mean diagonals, so it carries the
        # Monte-Carlo error of the sample diagonal as well as the injected 1e4.
        assert plain["amplitude_scale"] == pytest.approx(1e4, rel=0.02)
        assert matched["amplitude_scale"] == pytest.approx(1e4, rel=0.02)
        assert plain["verdict"] == "inconsistent"
        # After the scale is divided out the shape is the same estimate again.
        reference_shape = validate.compare(
            estimate, cov, n_samples=2000, match_amplitude=True
        )
        assert matched["entry_error_rms"] == pytest.approx(
            reference_shape["entry_error_rms"], rel=1e-9
        )

    def test_amplitude_matching_does_not_hide_a_shape_change(self):
        cov = _covariance()
        draws = _samples(cov, 3000)
        estimate = draws.T @ draws / draws.shape[0]
        # Compress the ladder: low modes down, high modes up. A global rescale
        # cannot explain this, so the matched score must still fail.
        k = cov.shape[0]
        ramp = np.linspace(0.5, 1.5, k)
        distorted = (ramp[:, None] * estimate) * ramp[None, :]
        result = validate.compare(distorted, cov, n_samples=3000, match_amplitude=True)
        assert result["verdict"] == "inconsistent"
        assert result["diagonal_ratio"][0] < result["diagonal_ratio"][-1]

    def test_strongest_pairs_are_off_diagonal_and_sorted(self):
        cov = _covariance(k=6)
        result = validate.compare(cov, cov, n_samples=100)
        noll = np.arange(4, 10)
        pairs = validate._strongest_pairs(result, noll, count=5)
        assert len(pairs) == 5
        assert all(pair["noll_i"] != pair["noll_j"] for pair in pairs)
        magnitudes = [abs(pair["corr_ref"]) for pair in pairs]
        assert magnitudes == sorted(magnitudes, reverse=True)


class TestPupilGeometry:
    def test_grid_span_is_not_the_disc_diameter(self):
        """The 4/3 trap: the OOPAO pupil must span the grid, not the disc."""
        sim = _bench()
        span = validate._grid_span_m(sim)
        assert span == pytest.approx(sim.grid * float(sim.cfg.pupil_pitch_m))
        assert span != pytest.approx(float(sim.cfg.aperture_m))
        # pupil_radius_px=48 of grid=128 -> span / disc = 128/96 = 4/3
        assert span / float(sim.cfg.aperture_m) == pytest.approx(sim.grid / (2 * sim.cfg.pupil_radius_px))


def _bench():
    """The bench at the committed asset's geometry (skips without OOPAO)."""
    pytest.importorskip("ao_shaping.drivers.sim._oopao_compat")
    ref = validate._reference()
    return validate._bench(ref)


class TestEstimators:
    """The two estimators, against the committed asset."""

    @pytest.fixture(autouse=True)
    def _require_oopao(self):
        pytest.importorskip("ao_shaping.drivers.sim._oopao_compat")

    def test_oopao_ensemble_is_a_covariance_at_the_asset_scale(self):
        ref = validate._reference()
        sim = validate._bench(ref)
        sigma = validate.oopao_ensemble(sim, ref, n_samples=200, seed=0)
        assert sigma.shape == ref["covariance"].shape
        assert np.allclose(sigma, sigma.T, rtol=0, atol=0)
        assert np.all(np.diag(sigma) > 0.0)
        result = validate.compare(sigma, ref["covariance"], n_samples=200)
        # The physical scale is a claim, not a free parameter: with the wrong
        # pupil diameter this median was 0.62 (measured), so a 0.85..1.15 band
        # catches that regression without pinning Monte-Carlo noise.
        assert 0.85 < float(np.median(result["diagonal_ratio"])) < 1.15

    def test_sampler_ensemble_keeps_the_correlation_shape(self):
        ref = validate._reference()
        sim = validate._bench(ref)
        sigma = validate.sampler_ensemble(sim, ref, n_draws=600, seed=0)
        assert sigma.shape == ref["covariance"].shape
        assert np.all(np.diag(sigma) > 0.0)
        result = validate.compare(
            sigma, ref["covariance"], n_samples=600, match_amplitude=True
        )
        # The documented RMS-rescale artefact compresses the ladder, so this is a
        # loose band rather than a pass/fail on the noise floor -- but the shape
        # must survive well enough to beat an i.i.d. draw (which would be ~0 here).
        assert result["offdiag_corr_rms_error"] < 0.35
        ratios = np.asarray(result["diagonal_ratio"])
        assert 0.5 < float(np.median(ratios)) < 1.6
