"""Tests for the Strehl landscape used by the offline benchmark.

Small grid (N=128) keeps the unit tests fast; correctness does not depend on
the grid size.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.strehl_sim_eval import StrehlLandscape


def _make(seed: int = 42, n_grid: int = 128) -> StrehlLandscape:
    return StrehlLandscape(seed=seed, n_grid=n_grid)


class TestStrehlLandscape:
    def test_dim_bounds_init(self):
        landscape = _make()
        assert landscape.dim == 64  # dm_actuators=8 -> 8**2
        assert landscape.bounds == (-1.0, 1.0)
        assert landscape.init_v.shape == (64,)
        assert np.all(landscape.init_v == 0.0)

    def test_score_returns_strehl_triplet(self):
        landscape = _make()
        s = landscape.score(landscape.init_v)
        assert isinstance(s, tuple) and len(s) == 3
        s0, s1, s2 = s
        assert s0 == s1  # first two components both carry the Strehl
        assert s2 == 0.0  # third component unused placeholder
        assert 0.0 < s0 <= 1.0  # Strehl ratio in (0, 1]

    def test_score_deterministic_same_seed(self):
        """Same seed -> identical turbulence screen -> identical Strehl."""
        a = _make(seed=42)
        b = _make(seed=42)
        sa = a.score(a.init_v)
        sb = b.score(b.init_v)
        assert sa == sb

    def test_different_seed_changes_turbulence(self):
        a = _make(seed=1)
        b = _make(seed=2)
        sa = a.score(a.init_v)
        sb = b.score(b.init_v)
        assert not np.isclose(sa[0], sb[0], rtol=1e-6, atol=1e-6)

    def test_voltages_clipped_to_bounds(self):
        """set_dm_voltages clips to [-1, 1]; score must reflect the clip."""
        landscape = _make()
        s_ones = landscape.score(np.ones(64))
        s_two = landscape.score(2.0 * np.ones(64))
        assert s_ones == s_two  # both clip to ones

    def test_score_equals_simulator_strehl(self):
        """Fast path must match the simulator's own observe()['strehl']."""
        landscape = _make()
        rng = np.random.default_rng(7)
        v = rng.uniform(-0.5, 0.5, size=64)
        s = landscape.score(v)[0]
        observed = float(landscape.ao.observe()["strehl"])
        assert s == observed

    def test_score_clips_raw_ratio_to_strehl_definition(self):
        """score() must clip at 1.0 like observe() even when the raw
        peak/ideal_peak ratio exceeds 1 (the 'ideal' reference is the
        Gaussian-pupil far-field, which shaped DM phases can focus tighter
        than; at n_grid=256 raw ratios reach ~3.6)."""
        landscape = _make()
        # Shrinking the ideal normalisation makes the raw ratio >> 1 for the
        # SAME voltages, deterministically reproducing the >1 regime.
        landscape.ao._ideal_peak = 1e-12
        v = np.zeros(64)
        s = float(landscape.score(v)[0])
        observed = float(landscape.ao.observe()["strehl"])
        assert s == 1.0
        assert observed == 1.0

    def test_repeated_score_is_stateless(self):
        """score() must not drift across calls (turbulence stays fixed)."""
        landscape = _make()
        rng = np.random.default_rng(7)
        v = rng.uniform(-1.0, 1.0, size=64)
        first = landscape.score(v)
        second = landscape.score(v)
        assert first == second

    def test_render_shape_and_sensitivity(self):
        landscape = _make()
        img_init = landscape.render(landscape.init_v)
        assert img_init.shape == (128, 128)
        assert img_init.dtype == np.uint16
        rng = np.random.default_rng(7)
        v_jitter = rng.uniform(-0.1, 0.1, size=64)
        img_corrected = landscape.render(v_jitter)
        assert not np.array_equal(img_init, img_corrected)

    def test_wrong_dimension_raises(self):
        landscape = _make()
        with pytest.raises(ValueError, match="64"):
            landscape.score(np.zeros(63))
        with pytest.raises(ValueError, match="64"):
            landscape.render(np.zeros((64, 1)))