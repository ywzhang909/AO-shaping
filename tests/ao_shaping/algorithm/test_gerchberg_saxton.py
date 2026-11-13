"""Tests for Gerchberg-Saxton hologram generation algorithm.

These tests use simulation only (no hardware required) to verify:
1. GS algorithm convergence for various preset target shapes
2. Phase pattern validity and properties
3. Reconstruction quality metrics
4. Error convergence behavior

All tests are hardware-independent and use pure NumPy simulation.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

import ao_shaping.algorithm.signal_processing.gerchberg_saxton as gs_module
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (
    gerchberg_saxton,
    angular_spectrum_propagate,
    calculate_reconstruction_error,
    GSResult,
)


# Default test parameters for fast execution (without iterations)
PHYSICAL_PARAMS = {
    "cell_spacing": 8e-6,  # 8 µm pixel size
    "distance": 0.1,  # 10 cm propagation
    "wavelength": 1064e-9,  # 1064 nm YAG laser
}

# Full test params including iterations
TEST_PARAMS = {**PHYSICAL_PARAMS, "iterations": 30}

# Small size for unit tests (faster execution)
SMALL_SIZE = (128, 128)  # (height, width)
# Larger size for integration tests
LARGE_SIZE = (512, 512)


class TestAngularSpectrumPropagation:
    """Test Angular Spectrum Method (ASM) propagation."""

    def test_forward_backward_propagation(self):
        """Test that forward then backward propagation returns approximately original field."""
        # Create a simple field (uniform with circular aperture)
        y, x = np.ogrid[-1:1:128j, -1:1:128j]
        r = np.sqrt(x**2 + y**2)
        amplitude = (r < 0.5).astype(float)
        phase = np.zeros_like(amplitude)
        field = amplitude * np.exp(1j * phase)

        # Forward propagate
        params = TEST_PARAMS.copy()
        propagated = angular_spectrum_propagate(
            field,
            params["cell_spacing"],
            params["distance"],
            params["wavelength"],
        )

        # Backward propagate
        back_propagated = angular_spectrum_propagate(
            propagated,
            params["cell_spacing"],
            -params["distance"],
            params["wavelength"],
        )

        # Check shapes preserved
        assert propagated.shape == field.shape
        assert back_propagated.shape == field.shape

        # Check amplitude is roughly preserved (with some numerical error)
        original_amplitude = np.abs(field)
        back_amplitude = np.abs(back_propagated)

        # Allow for significant numerical error due to sampling limitations
        # but ensure pattern is roughly correct
        correlation = np.corrcoef(
            original_amplitude.flatten(), back_amplitude.flatten()
        )[0, 1]
        assert correlation > 0.7, (
            f"Forward-backward propagation correlation too low: {correlation}"
        )

    def test_propagation_preserves_shape(self):
        """Test that propagation preserves array shape."""
        sizes = [(64, 64), (128, 128), (256, 256)]

        for height, width in sizes:
            field = np.ones((height, width), dtype=complex)
            result = angular_spectrum_propagate(
                field,
                TEST_PARAMS["cell_spacing"],
                TEST_PARAMS["distance"],
                TEST_PARAMS["wavelength"],
            )
            assert result.shape == (height, width), (
                f"Shape mismatch for input {(height, width)}: got {result.shape}"
            )

    def test_uniform_field_propagation(self):
        """Test propagation of uniform field produces expected behavior."""
        size = 128
        field = np.ones((size, size), dtype=complex)

        propagated = angular_spectrum_propagate(
            field,
            TEST_PARAMS["cell_spacing"],
            TEST_PARAMS["distance"],
            TEST_PARAMS["wavelength"],
        )

        # Uniform field should remain roughly uniform in amplitude
        # (energy conservation)
        input_energy = np.sum(np.abs(field) ** 2)
        output_energy = np.sum(np.abs(propagated) ** 2)

        # Energy should be approximately conserved
        assert np.abs(output_energy - input_energy) / input_energy < 0.1, (
            "Energy not conserved in propagation"
        )


class TestGerchbergSaxtonBasic:
    """Test basic GS algorithm functionality."""

    def test_gaussian_target(self):
        """Test GS with Gaussian target shape."""
        height, width = SMALL_SIZE

        # Create Gaussian target
        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r2 = x**2 + y**2
        target_intensity = np.exp(-r2 / (2 * 0.3**2))
        target_amplitude = np.sqrt(target_intensity)

        # Uniform source
        source_amplitude = np.ones((height, width))

        # Run GS
        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        # Verify result structure
        assert isinstance(result, GSResult)
        assert result.phase.shape == (height, width)
        assert result.amplitude.shape == (height, width)
        assert len(result.error_history) == TEST_PARAMS["iterations"]
        assert result.iterations == TEST_PARAMS["iterations"]

        # Verify phase is in reasonable range
        assert np.all(result.phase >= -np.pi)
        assert np.all(result.phase <= np.pi)

        # Verify error decreases overall (not necessarily monotonically)
        assert result.error_history[-1] < result.error_history[0], (
            "Error should decrease from start to end"
        )

    def test_circle_target(self):
        """Test GS with circular target shape."""
        height, width = SMALL_SIZE

        # Create circular target
        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r = np.sqrt(x**2 + y**2)
        target_intensity = (r < 0.5).astype(float)
        target_amplitude = np.sqrt(target_intensity)

        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        assert result.phase.shape == (height, width)
        # Should have reasonable correlation with target
        assert result.error_history[-1] < result.error_history[0]

    def test_square_target(self):
        """Test GS with square target shape."""
        height, width = SMALL_SIZE

        # Create square target
        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        target_intensity = ((np.abs(x) < 0.4) & (np.abs(y) < 0.4)).astype(float)
        target_amplitude = np.sqrt(target_intensity)

        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        assert result.phase.shape == (height, width)
        assert result.error_history[-1] < result.error_history[0]

    def test_annular_target(self):
        """Test GS with annular (ring) target shape."""
        height, width = SMALL_SIZE

        # Create annular target
        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r = np.sqrt(x**2 + y**2)
        target_intensity = ((r > 0.3) & (r < 0.6)).astype(float)
        target_amplitude = np.sqrt(target_intensity)

        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        assert result.phase.shape == (height, width)
        assert result.error_history[-1] < result.error_history[0]

    def test_grid_target(self):
        """Test GS with grid/crosshair target shape."""
        height, width = SMALL_SIZE

        # Create grid target
        target_intensity = np.zeros((height, width))
        # Horizontal lines
        for i in range(3):
            y_pos = int(height * (0.25 + i * 0.25))
            target_intensity[y_pos - 2 : y_pos + 2, :] = 1.0
        # Vertical lines
        for i in range(3):
            x_pos = int(width * (0.25 + i * 0.25))
            target_intensity[:, x_pos - 2 : x_pos + 2] = 1.0

        target_amplitude = np.sqrt(target_intensity)
        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        assert result.phase.shape == (height, width)
        assert result.error_history[-1] < result.error_history[0]


class TestGerchbergSaxtonConvergence:
    """Test GS algorithm convergence properties."""

    def test_error_decreases_over_iterations(self):
        """Test that error generally decreases over iterations."""
        height, width = SMALL_SIZE

        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r = np.sqrt(x**2 + y**2)
        target_amplitude = np.sqrt((r < 0.5).astype(float))
        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=50,
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        # Error at end should be significantly lower than at start
        initial_error = result.error_history[0]
        final_error = result.error_history[-1]

        assert final_error < initial_error, (
            f"Error should decrease: initial={initial_error:.4f}, final={final_error:.4f}"
        )

        # Should have at least 20% improvement
        improvement = (initial_error - final_error) / initial_error
        assert improvement > 0.2, f"Error improvement too small: {improvement:.2%}"

    def test_more_iterations_better_result(self):
        """Test that more iterations generally give better results."""
        height, width = SMALL_SIZE

        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r = np.sqrt(x**2 + y**2)
        target_amplitude = np.sqrt((r < 0.5).astype(float))
        source_amplitude = np.ones((height, width))

        # Run with different iteration counts
        result_10 = gerchberg_saxton(
            source_amplitude, target_amplitude, iterations=10, **PHYSICAL_PARAMS
        )
        result_30 = gerchberg_saxton(
            source_amplitude, target_amplitude, iterations=30, **PHYSICAL_PARAMS
        )
        result_50 = gerchberg_saxton(
            source_amplitude, target_amplitude, iterations=50, **PHYSICAL_PARAMS
        )

        # Generally, more iterations should give lower error
        # (though GS can oscillate, the trend should be downward)
        assert result_30.error_history[-1] <= result_10.error_history[-1] * 1.1, (
            "30 iterations should be at least as good as 10"
        )
        assert result_50.error_history[-1] <= result_30.error_history[-1] * 1.1, (
            "50 iterations should be at least as good as 30"
        )


class TestGerchbergSaxtonEdgeCases:
    """Test edge cases and error handling."""

    def test_non_uniform_source_amplitude(self):
        """Test GS with non-uniform source amplitude (e.g., Gaussian illumination)."""
        height, width = SMALL_SIZE

        # Gaussian source
        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r2 = x**2 + y**2
        source_amplitude = np.exp(-r2 / (2 * 0.7**2))

        # Circle target
        target_amplitude = np.sqrt((np.sqrt(r2) < 0.5).astype(float))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        assert result.phase.shape == (height, width)
        assert len(result.error_history) == TEST_PARAMS["iterations"]

    def test_phase_continuity(self):
        """Test that computed phase is continuous (no abrupt jumps)."""
        height, width = SMALL_SIZE

        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r = np.sqrt(x**2 + y**2)
        target_amplitude = np.sqrt((r < 0.5).astype(float))
        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        phase = result.phase

        # Unwrap phase to check continuity
        # (actual continuity check would require phase unwrapping)
        # For now, just verify no NaN or Inf
        assert np.all(np.isfinite(phase)), "Phase contains non-finite values"

        # Check that phase values are in valid range
        assert np.all(phase >= -np.pi), "Phase below -π"
        assert np.all(phase <= np.pi), "Phase above π"

    def test_invalid_inputs_raise_errors(self):
        """Test that invalid inputs raise appropriate errors."""
        height, width = SMALL_SIZE
        source = np.ones((height, width))
        target = np.ones((height, width))

        # Mismatched shapes
        with pytest.raises(ValueError, match="shapes must match"):
            gerchberg_saxton(
                source_amplitude=source,
                target_amplitude=np.ones((height, width + 10)),
                iterations=10,
                **PHYSICAL_PARAMS,
            )

        # Zero iterations
        with pytest.raises(ValueError, match="Iterations must be >= 1"):
            gerchberg_saxton(
                source_amplitude=source,
                target_amplitude=target,
                iterations=0,
                **PHYSICAL_PARAMS,
            )

        # Negative distance
        with pytest.raises(ValueError, match="Physical parameters must be positive"):
            gerchberg_saxton(
                source_amplitude=source,
                target_amplitude=target,
                iterations=10,
                cell_spacing=PHYSICAL_PARAMS["cell_spacing"],
                distance=-0.1,
                wavelength=PHYSICAL_PARAMS["wavelength"],
            )


class TestReconstructionMetrics:
    """Test reconstruction quality metrics calculation."""

    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        height, width = SMALL_SIZE

        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        r = np.sqrt(x**2 + y**2)
        target_amplitude = np.sqrt((r < 0.5).astype(float))
        source_amplitude = np.ones((height, width))

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=TEST_PARAMS["iterations"],
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        # Calculate metrics
        metrics = calculate_reconstruction_error(
            result.phase,
            source_amplitude,
            target_amplitude,
            TEST_PARAMS["cell_spacing"],
            TEST_PARAMS["distance"],
            TEST_PARAMS["wavelength"],
        )

        # Check all expected metrics are present
        assert "mse" in metrics
        assert "nmse" in metrics
        assert "correlation" in metrics
        assert "efficiency" in metrics

        # Check metric ranges
        assert 0 <= metrics["mse"], "MSE should be non-negative"
        assert 0 <= metrics["nmse"], "NMSE should be non-negative"
        assert -1 <= metrics["correlation"] <= 1, "Correlation should be in [-1, 1]"
        assert 0 <= metrics["efficiency"], "Efficiency should be non-negative"

        # For a reasonable reconstruction, correlation should be positive
        assert metrics["correlation"] > 0, (
            "Correlation should be positive for valid reconstruction"
        )

    def test_perfect_reconstruction_metrics(self):
        """Test metrics with self-consistent input/output."""
        size = 64

        # Create non-uniform target (circular aperture) to avoid NaN in correlation
        y, x = np.ogrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
        r = np.sqrt(x**2 + y**2)
        amplitude = np.sqrt((r < 0.5).astype(float))
        phase = np.zeros((size, size))

        metrics = calculate_reconstruction_error(
            phase,
            amplitude,
            amplitude,
            PHYSICAL_PARAMS["cell_spacing"],
            PHYSICAL_PARAMS["distance"],
            PHYSICAL_PARAMS["wavelength"],
        )

        # Check metrics are valid (not NaN)
        assert not np.isnan(metrics["correlation"]), "Correlation should not be NaN"
        assert not np.isnan(metrics["mse"]), "MSE should not be NaN"

        # For identical source and target, correlation should be high
        assert metrics["correlation"] > 0.5, (
            "Self-consistent case should have positive correlation"
        )


class TestPresetShapesIntegration:
    """Integration tests for all preset target shapes at higher resolution."""

    @pytest.mark.parametrize(
        "shape_name,shape_fn",
        [
            ("gaussian", lambda y, x: np.exp(-(x**2 + y**2) / (2 * 0.3**2))),
            ("circle", lambda y, x: (np.sqrt(x**2 + y**2) < 0.5).astype(float)),
            (
                "square",
                lambda y, x: ((np.abs(x) < 0.4) & (np.abs(y) < 0.4)).astype(float),
            ),
            (
                "annular",
                lambda y, x: (
                    ((np.sqrt(x**2 + y**2)) > 0.3) & ((np.sqrt(x**2 + y**2)) < 0.6)
                ).astype(float),
            ),
        ],
    )
    def test_all_preset_shapes(self, shape_name, shape_fn):
        """Test GS algorithm with each preset shape."""
        height, width = LARGE_SIZE

        # Create coordinate grids
        y_coords = np.linspace(-1, 1, height)
        x_coords = np.linspace(-1, 1, width)
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")

        # Create target
        target_intensity = shape_fn(y_grid, x_grid)
        target_amplitude = np.sqrt(target_intensity)
        source_amplitude = np.ones((height, width))

        # Run GS
        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=20,  # Reduced for faster integration tests
            cell_spacing=TEST_PARAMS["cell_spacing"],
            distance=TEST_PARAMS["distance"],
            wavelength=TEST_PARAMS["wavelength"],
        )

        # Verify results
        assert result.phase.shape == (height, width), (
            f"{shape_name}: Phase shape mismatch"
        )
        assert result.amplitude.shape == (height, width), (
            f"{shape_name}: Amplitude shape mismatch"
        )
        assert len(result.error_history) == 20, (
            f"{shape_name}: Wrong number of iterations"
        )

        # Verify error decreased
        assert result.error_history[-1] < result.error_history[0], (
            f"{shape_name}: Error did not decrease"
        )

        # Verify correlation is reasonable
        metrics = calculate_reconstruction_error(
            result.phase,
            source_amplitude,
            target_amplitude,
            TEST_PARAMS["cell_spacing"],
            TEST_PARAMS["distance"],
            TEST_PARAMS["wavelength"],
        )
        assert metrics["correlation"] > 0.5, (
            f"{shape_name}: Correlation too low: {metrics['correlation']:.3f}"
        )


class TestGerchbergSaxtonEarlyStopping:
    """Test the opt-in patience-based early stopping of the GS loop.

    The rule under test: an iteration counts as a *meaningful improvement* only
    if its MSE is at least ``early_stop_min_delta`` below the best MSE seen so
    far (best starts at ``+inf``, so the first measurement is always a best).
    ``early_stop_patience`` consecutive non-improving iterations stop the loop;
    ``early_stop_patience=0`` (the default) disables the check entirely.

    Reaching ``error_threshold`` still wins, and ``converged`` keeps its old
    meaning ("threshold reached") -- a plateau stop is not convergence.
    """

    # Small grid: these tests exercise the stopping rule, not propagation quality.
    SCRIPT_SIZE = (64, 64)

    @staticmethod
    def _install_mse_script(monkeypatch, values):
        """Feed the GS loop a fixed MSE sequence instead of the real one.

        Deterministic and hardware-free. Returns the list of values the loop
        actually consumed, so tests can assert the exact iteration count.
        """
        consumed: list[float] = []

        def _scripted_mse(amplitude, target_amplitude):
            idx = len(consumed)
            consumed.append(float(values[idx] if idx < len(values) else values[-1]))
            return consumed[-1]

        monkeypatch.setattr(gs_module, "_iteration_mse", _scripted_mse)
        return consumed

    def _run(self, monkeypatch, values, **kwargs):
        """Run GS on a tiny grid with ``values`` as the scripted MSE history."""
        consumed = self._install_mse_script(monkeypatch, values)
        kwargs.setdefault("iterations", 50)
        result = gerchberg_saxton(
            source_amplitude=np.ones(self.SCRIPT_SIZE),
            target_amplitude=np.zeros(self.SCRIPT_SIZE),
            propagation="fft",
            **kwargs,
        )
        return result, consumed

    def test_plateau_stops_after_patience(self, monkeypatch):
        """Two consecutive non-improving iterations stop the loop at patience=2."""
        values = [0.50, 0.40, 0.30, 0.30, 0.30, 0.30, 0.30]
        reported: list[int] = []

        result, consumed = self._run(
            monkeypatch,
            values,
            early_stop_patience=2,
            early_stop_min_delta=0.01,
            progress_callback=lambda i, err: reported.append(i),
        )

        # iters 1-3 improve (0.5 -> 0.4 -> 0.3); iters 4-5 plateau -> stop
        assert result.iterations == 5
        assert result.error_history == values[:5]
        assert consumed == values[:5]
        assert len(reported) == 5, "callbacks must fire for run iterations only"
        # A plateau stop is not convergence
        assert result.converged is False

    def test_first_measurement_counts_as_best(self, monkeypatch):
        """With best MSE initialized to +inf, iteration 1 is always an improvement."""
        result, _ = self._run(
            monkeypatch, [0.4] * 10, early_stop_patience=1
        )

        # iteration 1 = new best, iteration 2 = first stagnation -> stop
        assert result.iterations == 2
        assert result.error_history == [0.4, 0.4]

    def test_disabled_by_default_runs_every_iteration(self, monkeypatch):
        """Default (and explicit) patience 0 keeps the old loop behaviour."""
        plateau = [0.3] * 8

        default_result, default_consumed = self._run(monkeypatch, plateau, iterations=8)
        assert default_result.iterations == 8
        assert default_result.error_history == plateau
        assert len(default_consumed) == 8

        explicit_result, _ = self._run(
            monkeypatch, plateau, iterations=8, early_stop_patience=0
        )
        assert explicit_result.iterations == 8
        assert explicit_result.error_history == default_result.error_history

    def test_min_delta_gates_what_counts_as_improvement(self, monkeypatch):
        """Improvements smaller than the delta count as stagnation."""
        # Strictly decreasing, but only by 1e-3 per step
        creeping = [0.30 - 1e-3 * k for k in range(10)]

        loose_result, _ = self._run(
            monkeypatch, creeping, iterations=10, early_stop_patience=3
        )
        # min_delta=0.0 (default): every step is a real improvement -> no stop
        assert loose_result.iterations == 10

        tight_result, _ = self._run(
            monkeypatch,
            creeping,
            iterations=10,
            early_stop_patience=3,
            early_stop_min_delta=0.01,
        )
        # steps of 1e-3 < 0.01 -> first 3 iterations stagnate -> stop on the 4th
        assert tight_result.iterations == 4
        assert tight_result.converged is False

    def test_threshold_stop_takes_precedence(self, monkeypatch):
        """error_threshold breaks before the patience check on the same iteration."""
        decreasing = [0.5, 0.4, 0.3, 0.2, 0.1]

        result, _ = self._run(
            monkeypatch,
            decreasing,
            error_threshold=0.25,
            early_stop_patience=10,
        )
        assert result.iterations == 4
        assert result.error_history[-1] == 0.2
        assert result.converged is True

        # Even with patience=1 the threshold wins on iteration 1
        immediate, _ = self._run(
            monkeypatch, [0.3] * 5, error_threshold=0.35, early_stop_patience=1
        )
        assert immediate.iterations == 1
        assert immediate.converged is True

    def test_real_plateau_stops_without_monkeypatch(self):
        """A dark target gives a bitwise-identical MSE, so the loop plateaus for real."""
        source = np.ones(self.SCRIPT_SIZE)
        target = np.zeros(self.SCRIPT_SIZE)

        result = gerchberg_saxton(
            source_amplitude=source,
            target_amplitude=target,
            iterations=20,
            propagation="fft",
            early_stop_patience=2,
        )
        assert result.iterations == 3
        assert len(result.error_history) == 3
        assert result.error_history[1] == result.error_history[2]
        assert result.converged is False

        # Same run without patience keeps all 20 iterations (no regression)
        full = gerchberg_saxton(
            source_amplitude=source,
            target_amplitude=target,
            iterations=20,
            propagation="fft",
        )
        assert full.iterations == 20
        assert len(full.error_history) == 20

    def test_threshold_behaviour_unchanged_on_real_run(self):
        """Untouched inputs still converge exactly as before the change."""
        size = 64
        y, x = np.ogrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
        r = np.sqrt(x**2 + y**2)
        source = np.ones((size, size))
        target = (r < 0.5).astype(float)

        plain = gerchberg_saxton(
            source_amplitude=source,
            target_amplitude=target,
            iterations=12,
            **PHYSICAL_PARAMS,
        )
        assert plain.iterations == 12
        assert len(plain.error_history) == 12
        assert plain.converged is False

        converged = gerchberg_saxton(
            source_amplitude=source,
            target_amplitude=target,
            iterations=12,
            error_threshold=float("inf"),
            **PHYSICAL_PARAMS,
        )
        assert converged.iterations == 1
        assert converged.converged is True

    def test_invalid_patience_and_delta_raise(self):
        """Negative patience / delta are rejected like the other parameters."""
        source = np.ones(self.SCRIPT_SIZE)
        target = np.zeros(self.SCRIPT_SIZE)

        with pytest.raises(
            ValueError, match="Early stop patience must be a non-negative integer"
        ):
            gerchberg_saxton(
                source_amplitude=source,
                target_amplitude=target,
                iterations=2,
                early_stop_patience=-1,
            )

        with pytest.raises(ValueError, match="Early stop min delta must be >= 0"):
            gerchberg_saxton(
                source_amplitude=source,
                target_amplitude=target,
                iterations=2,
                early_stop_min_delta=-1e-6,
            )


if __name__ == "__main__":
    # Run tests with pytest if available
    pytest.main([__file__, "-v"])
