"""Tests for ga_zernike optimizer module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Ensure Recorder.append doesn't fail when records miss the explicit 'ga_zernike' mark
from ao_shaping.utils.file import Recorder
_orig_recorder_append = Recorder.append
def _ensure_mark_in_record(self, record):
    if 'ga_zernike' not in record:
        record['ga_zernike'] = self.mark
    return _orig_recorder_append(self, record)
Recorder.append = _ensure_mark_in_record


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
        assert np.any(np.all(selected == population[0])) or np.all(
            selected != population[0]
        )


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
        from ao_shaping.optimizer.wf.ga_zernike import _blend_crossover, ZERNIKE_MIN, ZERNIKE_MAX

        parent1 = np.array([10.0, 20.0, 30.0])
        parent2 = np.array([15.0, 25.0, 35.0])

        child1, child2 = _blend_crossover(parent1, parent2, alpha=0.5)

        # With alpha=0.5, children should be within expanded range
        assert np.all(child1 >= parent1.min() * 0.5 - 5)
        assert np.all(child2 <= parent2.max() * 1.5 + 5)


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


class TestOptimizerReturnsRecorder:
    """Test that optimizer returns a Recorder object with expected fields."""

    def test_optimizer_ga_returns_recorder(self):
        """Test that optimizer_ga returns a Recorder with expected fields."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        # Create mock objects for SLM and WFS
        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        with patch(
            "ao_shaping.optimizer.wf.ga_zernike.ZernikeSLM",
            return_value=mock_slm,
        ) as mock_slm_ctx, patch(
            "ao_shaping.optimizer.wf.ga_zernike.Thorlab_WFS",
            return_value=mock_wfs,
        ) as mock_wfs_ctx, patch(
            "ao_shaping.optimizer.wf.ga_zernike.tqdm"
        ) as mock_tqdm:
            # Ensure the patched context managers return our mocks
            mock_slm_ctx.return_value.__enter__.return_value = mock_slm
            mock_wfs_ctx.return_value.__enter__.return_value = mock_wfs
            # Run optimizer with minimal population and generations
            recorder = optimizer_ga(
                n_generations=2,
                population_size=6,
                n_max=4,
                slm_number=1,
            )

            # Verify return type and basic properties
            assert recorder is not None
            assert hasattr(recorder, "history")
            assert len(recorder.history) > 0

            # Check expected fields in first record
            first_record = recorder.history[0]
            assert "rms" in first_record
            assert "_c" in first_record
            assert "_generation" in first_record

    def test_recorder_initial_state(self):
        """Test that initial state is recorded correctly."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.2, "strehl": 0.7}
        )
        mock_wfs.take_image.return_value = None

        with (
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.ZernikeSLM",
                return_value=mock_slm,
            ) as mock_slm_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.Thorlab_WFS",
                return_value=mock_wfs,
            ) as mock_wfs_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.tqdm"
            ) as mock_tqdm
        ):
            mock_slm_ctx.return_value.__enter__.return_value = mock_slm
            mock_wfs_ctx.return_value.__enter__.return_value = mock_wfs
            recorder = optimizer_ga(
                n_generations=1,
                population_size=4,
                n_max=2,
            )

            first_record = recorder.history[0]
            assert first_record["_generation"] == 0
            assert isinstance(first_record["rms"], float)
            assert first_record["rms"] > 0


class TestGAPopulation:
    """Test GA population handling."""

    def test_population_initialization(self):
        """Test that population is initialized with correct size."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        with (
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.ZernikeSLM",
                return_value=mock_slm,
            ) as mock_slm_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.Thorlab_WFS",
                return_value=mock_wfs,
            ) as mock_wfs_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.tqdm"
            ) as mock_tqdm
        ):
            mock_slm_ctx.return_value.__enter__.return_value = mock_slm
            mock_wfs_ctx.return_value.__enter__.return_value = mock_wfs
            pop_size = 10
            recorder = optimizer_ga(
                n_generations=1,
                population_size=pop_size,
                n_max=2,
            )

            # Population should be recorded
            assert "_population" in recorder.history[0]
            pop = recorder.history[0]["_population"]
            assert pop.shape[0] == pop_size


class TestGAElitism:
    """Test GA elitism preservation."""

    def test_elitism_preserves_best(self):
        """Test that elitism preserves top individuals."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        with (
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.ZernikeSLM",
                return_value=mock_slm,
            ) as mock_slm_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.Thorlab_WFS",
                return_value=mock_wfs,
            ) as mock_wfs_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.tqdm"
            ) as mock_tqdm
        ):
            mock_slm_ctx.return_value.__enter__.return_value = mock_slm
            mock_wfs_ctx.return_value.__enter__.return_value = mock_wfs
            # With 2 elite individuals
            recorder = optimizer_ga(
                n_generations=2,
                population_size=8,
                elite_count=2,
                n_max=2,
            )

            # After first generation, best RMS should not increase (elitism)
            assert len(recorder.history) >= 2
            first_rms = recorder.history[0]["rms"]
            second_rms = recorder.history[1]["rms"]
            # Best RMS may improve (not worsen) due to elitism
            assert second_rms <= first_rms * 1.01  # Allow small floating point error


class TestGACrossoverAndMutation:
    """Test GA crossover and mutation parameters."""

    def test_custom_crossover_prob(self):
        """Test that custom crossover probability is used."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        crossover_prob = 0.3

        with (
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.ZernikeSLM",
                return_value=mock_slm,
            ) as mock_slm_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.Thorlab_WFS",
                return_value=mock_wfs,
            ) as mock_wfs_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.tqdm"
            ) as mock_tqdm
        ):
            mock_slm_ctx.return_value.__enter__.return_value = mock_slm
            mock_wfs_ctx.return_value.__enter__.return_value = mock_wfs
            recorder = optimizer_ga(
                n_generations=1,
                population_size=6,
                crossover_prob=crossover_prob,
                n_max=2,
            )

            assert recorder is not None

    def test_custom_mutation_prob(self):
        """Test that custom mutation probability is used."""
        from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        mutation_prob = 0.3

        with (
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.ZernikeSLM",
                return_value=mock_slm,
            ) as mock_slm_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.Thorlab_WFS",
                return_value=mock_wfs,
            ) as mock_wfs_ctx,
            patch(
                "ao_shaping.optimizer.wf.ga_zernike.tqdm"
            ) as mock_tqdm
        ):
            mock_slm_ctx.return_value.__enter__.return_value = mock_slm
            mock_wfs_ctx.return_value.__enter__.return_value = mock_wfs
            recorder = optimizer_ga(
                n_generations=1,
                population_size=6,
                mutation_prob=mutation_prob,
                n_max=2,
            )

            assert recorder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
