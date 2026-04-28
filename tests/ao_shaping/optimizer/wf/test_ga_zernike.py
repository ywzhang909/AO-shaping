"""Tests for ga_zernike optimizer module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestImport:
    """Test that module can be imported."""

    def test_import(self):
        """Test importing the optimizer function."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        assert callable(optimizer_ga)


class TestHelperFunctions:
    """Test helper functions in the module."""

    def test_noll_to_nm(self):
        """Test Noll index to (n, m) conversion with known values."""
        from ao_shaping.optimizer.wf.ga_zernike import noll_to_nm

        # Test known Noll indices
        assert noll_to_nm(1) == (0, 0)   # piston
        assert noll_to_nm(2) == (1, -1)  # tilt x
        assert noll_to_nm(3) == (1, 1)  # tilt y
        assert noll_to_nm(4) == (2, -2)  # oblique astigmatism
        assert noll_to_nm(5) == (2, 0)  # defocus
        assert noll_to_nm(6) == (2, 2)  # oblique astigmatism

    def test_noll_to_nm_invalid(self):
        """Test that invalid Noll indices raise ValueError."""
        from ao_shaping.optimizer.wf.ga_zernike import noll_to_nm

        with pytest.raises(ValueError):
            noll_to_nm(0)  # too small

        with pytest.raises(ValueError):
            noll_to_nm(100)  # too large

    def test_zernike_indices(self):
        """Test Zernike indices generation with n_max=4."""
        from ao_shaping.optimizer.wf.ga_zernike import _zernike_indices

        modes = _zernike_indices(n_max=4)

        # n_max=4 should give 15 modes (including piston)
        assert len(modes) == 15

        # First few should be piston and tilts
        assert modes[0] == (0, 0)
        assert modes[1] == (1, -1)
        assert modes[2] == (1, 1)

    def test_zernike_indices_n_max_2(self):
        """Test Zernike indices with n_max=2."""
        from ao_shaping.optimizer.wf.ga_zernike import _zernike_indices

        modes = _zernike_indices(n_max=2)

        # n_max=2 gives 6 modes
        assert len(modes) == 6


class TestGAParameters:
    """Test GA-specific parameters."""

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        from ao_shaping.optimizer.wf.ga_zernike import (
            ZERNIKE_MIN,
            ZERNIKE_MAX,
            SLM_WAVELENGTH_DEFAULT,
        )

        assert ZERNIKE_MIN == -50.0
        assert ZERNIKE_MAX == 50.0
        assert SLM_WAVELENGTH_DEFAULT == 532


class TestTournamentSelection:
    """Test tournament selection function."""

    def test_tournament_selection(self):
        """Test tournament selection returns valid individual."""
        from ao_shaping.optimizer.wf.ga_zernike import _tournament_selection

        population = np.random.randn(10, 5)
        fitness = np.random.randn(10)

        selected = _tournament_selection(population, fitness, tournament_size=3)

        assert selected.shape == (5,)


class TestBlendCrossover:
    """Test blend crossover function."""

    def test_blend_crossover_shape(self):
        """Test that crossover produces correct shapes."""
        from ao_shaping.optimizer.wf.ga_zernike import _blend_crossover

        parent1 = np.array([1.0, 2.0, 3.0])
        parent2 = np.array([4.0, 5.0, 6.0])

        child1, child2 = _blend_crossover(parent1, parent2)

        assert child1.shape == (3,)
        assert child2.shape == (3,)

    def test_blend_crossover_bounds(self):
        """Test that children are within reasonable bounds."""
        from ao_shaping.optimizer.wf.ga_zernike import (
            _blend_crossover,
            ZERNIKE_MIN,
            ZERNIKE_MAX,
        )

        parent1 = np.array([10.0, 20.0, 30.0])
        parent2 = np.array([15.0, 25.0, 35.0])

        child1, child2 = _blend_crossover(parent1, parent2, alpha=0.5)

        # With alpha=0.5, children should be within expanded range
        assert np.all(child1 >= parent1.min() * 0.5 - 5)


class TestGaussianMutation:
    """Test Gaussian mutation function."""

    def test_gaussian_mutation_shape(self):
        """Test mutation preserves shape."""
        from ao_shaping.optimizer.wf.ga_zernike import _gaussian_mutation

        individual = np.array([1.0, 2.0, 3.0])
        mutated = _gaussian_mutation(individual, mutation_rate=0.5, sigma=1.0)

        assert mutated.shape == (3,)

    def test_gaussian_mutation_bounds(self):
        """Test mutation respects bounds."""
        from ao_shaping.optimizer.wf.ga_zernike import (
            _gaussian_mutation,
            ZERNIKE_MIN,
            ZERNIKE_MAX,
        )

        individual = np.array([0.0, 0.0, 0.0])
        mutated = _gaussian_mutation(
            individual,
            mutation_rate=1.0,
            sigma=10.0,
            bounds=(ZERNIKE_MIN, ZERNIKE_MAX),
        )

        assert np.all(mutated >= ZERNIKE_MIN)
        assert np.all(mutated <= ZERNIKE_MAX)


class TestOptimizerCreation:
    """Test that optimizer can be created (basic smoke test)."""

    def test_optimizer_ga_creation(self):
        """Test that optimizer_ga can be imported and has correct signature."""
        import inspect
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        sig = inspect.signature(optimizer_ga)
        params = list(sig.parameters.keys())

        # Check expected parameters
        assert "n_generations" in params
        assert "population_size" in params
        assert "crossover_prob" in params
        assert "mutation_prob" in params
        assert "elite_count" in params
        assert "n_max" in params

    def test_default_values(self):
        """Test that default parameter values are sensible."""
        import inspect
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        sig = inspect.signature(optimizer_ga)

        # Test some default values
        assert sig.parameters["population_size"].default == 50
        assert sig.parameters["crossover_prob"].default == 0.7
        assert sig.parameters["mutation_prob"].default == 0.15
        assert sig.parameters["elite_count"].default == 2
        assert sig.parameters["n_max"].default == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])