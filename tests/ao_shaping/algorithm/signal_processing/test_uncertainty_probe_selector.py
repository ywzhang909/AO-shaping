"""Simulation-first tests for :mod:`uncertainty_probe_selector`.

Pure torch/numpy on a synthetic Zernike forward model: no hardware, no SDK
imports, CPU only, deterministic seeds.  The load-bearing test is the reverse
validation in :class:`TestReverseValidation` -- ``maximize`` must climb the
predictive-variance objective while ``minimize`` must fall it, from the same
seeded start.  That is what proves the selector optimises *uncertainty* and not
random noise (issue R-82 acceptance criterion).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from ao_shaping.algorithm.signal_processing.uncertainty_probe_selector import (
    UncertaintyProbeSelector,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)

REGION = 32
ORDERS = 6


def make_optimizer(region: int = REGION, orders: int = ORDERS) -> ZernikeCoefficientOptimizer:
    """A Step-A-shaped optimizer with its source amplitude already set."""
    optimizer = ZernikeCoefficientOptimizer(
        n_orders=orders, region=region, max_iterations=1
    )
    optimizer.set_source_amplitude(ZernikeCoefficientOptimizer.native_amplitude(region))
    return optimizer


class TestConstruction:
    """``__init__`` validates every input and sets up ready-to-step state."""

    def test_valid_construction_sets_state(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=1)
        assert selector.region == REGION
        assert selector.n_coefficients == make_optimizer().n_coefficients
        assert selector.mode == "maximize"
        assert selector.uncertainty_history == []

    def test_current_phase_shape_and_gradable(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=1)
        phase = selector.current_phase()
        assert phase.shape == (REGION, REGION)
        assert np.all(np.isfinite(phase))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "maximise"},          # typo'd mode
            {"n_samples": 0},
            {"ridge": -1.0},
            {"prior_std": 0.0},
            {"spread_floor": 5.0, "spread_cap": 4.0},  # floor > cap
            {"lr": 0.0},
            {"diversity_weight": -0.1},
            {"fd_eps": 0.0},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            UncertaintyProbeSelector(make_optimizer(), **kwargs)


class TestDeterminism:
    """Same seed reproduces the exact candidate-phase trajectory."""

    def test_same_seed_same_trajectory(self) -> None:
        a = UncertaintyProbeSelector(make_optimizer(), seed=7)
        b = UncertaintyProbeSelector(make_optimizer(), seed=7)
        for _ in range(5):
            np.testing.assert_array_equal(a.update(), b.update())

    def test_different_seed_diverges(self) -> None:
        a = UncertaintyProbeSelector(make_optimizer(), seed=7)
        b = UncertaintyProbeSelector(make_optimizer(), seed=8)
        assert not np.array_equal(a.current_phase(), b.current_phase())


class TestReverseValidation:
    """The objective really tracks uncertainty: maximize climbs, minimize falls."""

    def test_maximize_beats_minimize_from_same_start(self) -> None:
        up = UncertaintyProbeSelector(make_optimizer(), seed=3, lr=0.3)
        down = UncertaintyProbeSelector(make_optimizer(), seed=3, lr=0.3, mode="minimize")
        for _ in range(25):
            up.update()
            down.update()
        assert up.current_uncertainty() > down.current_uncertainty()

    def test_maximize_history_is_rising(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=3, lr=0.3)
        for _ in range(25):
            selector.update()
        hist = selector.uncertainty_history
        # Not required to be monotone (constraints re-project), but the net
        # climb over the run must be clearly positive.
        assert hist[-1] > hist[0]

    def test_minimize_history_is_falling(self) -> None:
        selector = UncertaintyProbeSelector(
            make_optimizer(), seed=3, lr=0.3, mode="minimize"
        )
        for _ in range(25):
            selector.update()
        hist = selector.uncertainty_history
        assert hist[-1] < hist[0]


class TestConstraints:
    """Every returned probe stays a physically deliverable phase."""

    def test_spread_cap_respected(self) -> None:
        selector = UncertaintyProbeSelector(
            make_optimizer(), seed=2, spread_cap=2.0, spread_floor=1.0, lr=0.2
        )
        for _ in range(20):
            phase = selector.update()
            assert float(phase.std()) <= 2.0 + 1e-6

    def test_spread_floor_respected(self) -> None:
        selector = UncertaintyProbeSelector(
            make_optimizer(), seed=2, spread_cap=6.0, spread_floor=3.0, lr=0.2
        )
        for _ in range(20):
            phase = selector.update()
            assert float(phase.std()) >= 3.0 - 1e-6

    def test_bandlimit_makes_phase_smooth(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=4, bandlimit_sigma=2.0)
        phase = selector.update()
        # A band-limited field is positively autocorrelated along a row;
        # white noise would sit near zero.
        row = phase[REGION // 2]
        corr = float(np.corrcoef(row[:-1], row[1:])[0, 1])
        assert corr > 0.2

    def test_no_bandlimit_when_sigma_zero(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=4, bandlimit_sigma=0.0)
        phase = selector.update()
        assert np.all(np.isfinite(phase))


class TestCovarianceAndCommit:
    """``update_covariance`` re-derives Σ; ``commit`` registers and reseeds."""

    def test_update_covariance_changes_deltas(self) -> None:
        optimizer = make_optimizer()
        selector = UncertaintyProbeSelector(optimizer, seed=5)
        before = selector._deltas.detach().cpu().numpy().copy()
        probes = [np.random.default_rng(0).normal(0, 3.0, (REGION, REGION))]
        selector.update_covariance(probes)
        after = selector._deltas.detach().cpu().numpy()
        assert not np.allclose(before, after)

    def test_update_covariance_accepts_single_2d_array(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=5)
        probe = np.random.default_rng(0).normal(0, 3.0, (REGION, REGION))
        selector.update_covariance(probe)  # must not raise
        assert np.all(np.isfinite(selector._sigma))

    def test_update_covariance_empty_raises(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=5)
        with pytest.raises(ValueError):
            selector.update_covariance([])

    def test_commit_registers_and_reseeds(self) -> None:
        selector = UncertaintyProbeSelector(make_optimizer(), seed=6)
        first = selector.current_phase().copy()
        selector.update()
        selector.commit()
        assert len(selector._chosen) == 1
        # After commit the candidate is a fresh probe, not the committed one.
        assert not np.array_equal(selector.current_phase(), first)

    def test_diversity_pushes_second_probe_away(self) -> None:
        # With a committed probe and a strong diversity weight, the next
        # candidate must not collapse onto the first one.
        selector = UncertaintyProbeSelector(
            make_optimizer(), seed=6, diversity_weight=5.0, lr=0.3
        )
        for _ in range(10):
            selector.update()
        first = selector.current_phase()
        selector.commit()
        for _ in range(10):
            selector.update()
        second = selector.current_phase()

        def cos(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        assert cos(first, second) < 0.9


class TestPredictiveVariancePublicApi:
    """issue #153: the public ``predictive_variance`` is the single source of
    truth; the private ``_predictive_variance`` kept for backward compatibility
    must delegate to it (identical result, not a second copy of the formula)."""

    def test_public_equals_private(self) -> None:
        import torch

        selector = UncertaintyProbeSelector(make_optimizer(), seed=3)
        probe = selector.current_phase()
        leaf = torch.from_numpy(
            np.ascontiguousarray(probe, dtype=np.float64)
        )
        leaf.requires_grad_(True)
        with torch.no_grad():
            pub = float(selector.predictive_variance(leaf))
            prv = float(selector._predictive_variance(leaf))
        assert pub == prv  # bitwise: same code path, no re-implementation

    def test_public_returns_finite_nonnegative(self) -> None:
        """The score is a mean of squares: finite and >= 0 for a valid probe."""
        import torch

        selector = UncertaintyProbeSelector(make_optimizer(), seed=3)
        leaf = torch.from_numpy(
            np.ascontiguousarray(selector.current_phase(), dtype=np.float64)
        )
        leaf.requires_grad_(True)
        with torch.no_grad():
            u = float(selector.predictive_variance(leaf))
        assert np.isfinite(u)
        assert u >= 0.0
