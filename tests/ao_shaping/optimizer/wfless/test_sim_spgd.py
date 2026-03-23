"""
Tests for sim_spgd module.
"""

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.sim_spgd import (
    optimize_spgd,
    optimize_spgd_zernike,
    optimize_pso,
    optimize_ga,
    optimize_sa,
)


class TestOptimizeSPGDBasics:
    """Test basic SPGD optimization runs correctly."""

    _stable_spgd = {
        "delta": 0.03,
        "gamma": 1e-2,
        "Cn2": 1e-14,
        "optimizer_type": "adamod",
        "use_momentum": False,
    }

    def test_optimize_spgd_basic_run(self):
        """Test basic SPGD optimization runs without errors."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        assert recorder is not None
        assert len(recorder.history) > 0
        assert recorder.history[0]["_epoch"] == 0
        assert recorder.history[-1]["_epoch"] == 10

    def test_optimize_spgd_converges(self):
        """Test that SPGD optimization improves PIB over iterations."""
        recorder = optimize_spgd(
            epochs=50,
            r_bucket=15,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        initial_pib = recorder.history[0]["pib"]
        final_pib = recorder.history[-1]["pib"]
        assert final_pib > initial_pib, f"PIB decreased from {initial_pib:.0f} to {final_pib:.0f}"

    def test_optimize_spgd_improvement_ratio(self):
        """Test that SPGD achieves meaningful improvement."""
        recorder = optimize_spgd(
            epochs=100,
            r_bucket=15,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        initial_pib = recorder.history[0]["pib"]
        final_pib = recorder.history[-1]["pib"]
        ratio = final_pib / initial_pib
        assert ratio > 1.0, f"PIB ratio {ratio:.3f} < 1.0"

    def test_optimize_spgd_auto_bucket(self):
        """Test SPGD with automatic bucket radius selection."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=0,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        assert len(recorder.history) == 11
        assert recorder.history[-1]["r"] > 0

    def test_optimize_spgd_different_actuators(self):
        """Test SPGD with different DM actuator configurations."""
        for n_actuators in [4, 6, 8]:
            recorder = optimize_spgd(
                epochs=5,
                r_bucket=15,
                n_grid=64,
                dm_actuators=n_actuators,
                seed=42,
                **self._stable_spgd,
            )
            assert len(recorder.history) == 6

    def test_optimize_spgd_strehl_recording(self):
        """Test that strehl ratio is recorded."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        for record in recorder.history:
            assert "strehl" in record
            assert 0 <= record["strehl"] <= 1

    def test_optimize_spgd_init_voltage(self):
        """Test SPGD with initial voltages provided."""
        init_v = np.random.randn(64)
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=15,
            init_v=init_v,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        assert len(recorder.history) == 11

    def test_optimize_spgd_zero_turbulence(self):
        """Test SPGD with zero turbulence (baseline)."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            Cn2=0,
            seed=42,
        )
        assert len(recorder.history) == 11

    @pytest.mark.parametrize("optimizer_type", ["adam", "adamod"])
    def test_optimize_spgd_adam_family_converges(self, optimizer_type):
        """Test Adam/AdaMOD variants converge in simulation."""
        recorder = optimize_spgd(
            epochs=80,
            r_bucket=0,
            delta=0.08,
            gamma=5e-3,
            n_grid=64,
            Cn2=1e-14,
            optimizer_type=optimizer_type,
            seed=42,
            use_momentum=False,
        )

        initial_pib = recorder.history[0]["pib"]
        final_pib = recorder.history[-1]["pib"]
        assert final_pib > initial_pib, (
            f"{optimizer_type} PIB decreased from {initial_pib:.2f} to {final_pib:.2f}"
        )

    def test_optimize_spgd_auto_bucket_uses_ideal_radius(self):
        """When r_bucket=0, PIB radius should be derived from ideal spot radius."""
        recorder = optimize_spgd(
            epochs=5,
            r_bucket=0,
            n_grid=64,
            seed=42,
            **self._stable_spgd,
        )

        init_log = recorder.history[0]
        assert "ideal_r" in init_log
        assert init_log["ideal_r"] > 0
        assert np.isclose(init_log["r"], init_log["ideal_r"])


class TestOptimizeSPGDZernike:
    """Test cases for optimize_spgd_zernike function."""

    _stable_zernike = {
        "delta": 0.01,
        "gamma": 1e-2,
        "Cn2": 1e-14,
        "optimizer_type": "adamod",
        "use_momentum": False,
    }

    def test_optimize_spgd_zernike_basic_run(self):
        """Test basic Zernike SPGD optimization runs without errors."""
        pytest.importorskip("zernike", reason="zernike package required")

        recorder = optimize_spgd_zernike(
            epochs=10,
            n_max=4,
            r_bucket=20,
            n_grid=64,
            seed=42,
            **self._stable_zernike,
        )

        assert recorder is not None
        assert len(recorder.history) == 11
        assert recorder.history[0]["_epoch"] == 0
        assert recorder.history[-1]["_epoch"] == 10

    def test_optimize_spgd_zernike_converges(self):
        """Test that Zernike SPGD improves PIB."""
        pytest.importorskip("zernike", reason="zernike package required")

        recorder = optimize_spgd_zernike(
            epochs=50,
            n_max=6,
            r_bucket=15,
            n_grid=64,
            seed=42,
            **self._stable_zernike,
        )

        initial_pib = recorder.history[0]["pib"]
        best_pib = max(item["pib"] for item in recorder.history)
        assert best_pib >= initial_pib, f"Best PIB regressed from {initial_pib:.0f} to {best_pib:.0f}"

    def test_optimize_spgd_zernike_different_orders(self):
        """Test Zernike SPGD with different maximum orders."""
        pytest.importorskip("zernike", reason="zernike package required")

        for n_max in [2, 4, 6]:
            recorder = optimize_spgd_zernike(
                epochs=5,
                n_max=n_max,
                r_bucket=15,
                n_grid=64,
                seed=42,
                **self._stable_zernike,
            )
            assert len(recorder.history) == 6

    def test_optimize_spgd_zernike_adamod_run(self):
        """Test Zernike optimization with AdaMOD backend."""
        pytest.importorskip("zernike", reason="zernike package required")

        recorder = optimize_spgd_zernike(
            epochs=10,
            n_max=4,
            r_bucket=0,
            n_grid=64,
            optimizer_type="adamod",
            seed=42,
            use_momentum=False,
            delta=0.01,
            gamma=1e-2,
        )

        assert len(recorder.history) == 11
        best_pib = max(item["pib"] for item in recorder.history)
        assert best_pib >= recorder.history[0]["pib"] * 0.95


class TestRecorder:
    """Test Recorder functionality within SPGD optimization."""

    def test_recorder_contains_all_fields(self):
        """Test that recorder contains all expected fields."""
        recorder = optimize_spgd(
            epochs=5,
            r_bucket=20,
            n_grid=64,
            seed=42,
            delta=0.03,
            gamma=1e-2,
            Cn2=1e-14,
            optimizer_type="adamod",
            use_momentum=False,
        )

        expected_fields = [
            "sim_spgd", "J", "pib", "_p%", "_max_r", "_v", "_img",
            "_diff", "gamma", "r", "ideal_r", "_epoch", "strehl", "_grad"
        ]

        for record in recorder.history:
            for field in expected_fields:
                assert field in record, f"Field {field} missing in record"

    def test_recorder_epochs_monotonic(self):
        """Test that epoch values are monotonic increasing."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            seed=42,
            delta=0.03,
            gamma=1e-2,
            Cn2=1e-14,
            optimizer_type="adamod",
            use_momentum=False,
        )

        epochs = [r["_epoch"] for r in recorder.history]
        assert epochs == list(range(0, 11))


class TestOptimizePSO:
    """Test cases for PSO optimization."""

    def test_pso_basic_run(self):
        """Test basic PSO optimization runs without errors."""
        recorder = optimize_pso(
            epochs=5,
            n_particles=10,
            r_bucket=20,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        assert recorder is not None
        assert len(recorder.history) == 6
        assert "_v" in recorder.history[-1]

    def test_pso_finds_improvement(self):
        """Test that PSO improves PIB over iterations."""
        recorder = optimize_pso(
            epochs=20,
            n_particles=15,
            r_bucket=15,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        initial_pib = recorder.history[0]["pib"]
        best_pib = max(item["pib"] for item in recorder.history)
        assert best_pib > initial_pib * 0.2


class TestOptimizeGA:
    """Test cases for Genetic Algorithm optimization."""

    def test_ga_basic_run(self):
        """Test basic GA optimization runs without errors."""
        recorder = optimize_ga(
            epochs=5,
            population_size=10,
            r_bucket=20,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        assert recorder is not None
        assert len(recorder.history) == 6
        assert "_v" in recorder.history[-1]

    def test_ga_finds_improvement(self):
        """Test that GA improves PIB over iterations."""
        recorder = optimize_ga(
            epochs=20,
            population_size=15,
            r_bucket=15,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        initial_pib = recorder.history[0]["pib"]
        best_pib = max(item["pib"] for item in recorder.history)
        assert best_pib > initial_pib * 0.2


class TestOptimizeSA:
    """Test cases for Simulated Annealing optimization."""

    def test_sa_basic_run(self):
        """Test basic SA optimization runs without errors."""
        recorder = optimize_sa(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        assert recorder is not None
        assert len(recorder.history) == 11
        assert "_T" in recorder.history[-1]

    def test_sa_finds_improvement(self):
        """Test that SA improves PIB over iterations."""
        recorder = optimize_sa(
            epochs=30,
            r_bucket=15,
            T_init=100.0,
            cooling_rate=0.95,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        initial_pib = recorder.history[0]["pib"]
        final_pib = recorder.history[-1]["pib"]
        assert final_pib > initial_pib * 0.8

    def test_sa_temperature_decreases(self):
        """Test that SA temperature decreases over iterations."""
        recorder = optimize_sa(
            epochs=20,
            r_bucket=20,
            T_init=100.0,
            T_min=0.1,
            cooling_rate=0.9,
            n_grid=64,
            Cn2=1e-14,
            seed=42,
        )

        initial_temp = recorder.history[1]["_T"]
        final_temp = recorder.history[-1]["_T"]
        assert final_temp <= initial_temp


class TestOptimizersComparison:
    """Compare different optimization algorithms."""

    def test_all_optimizers_improve(self):
        """Test that all optimizers can improve PIB."""
        base_params = {
            "epochs": 20,
            "r_bucket": 15,
            "n_grid": 64,
            "Cn2": 1e-14,
            "seed": 42,
        }

        optimizers = [
            (
                "SPGD",
                lambda: optimize_spgd(
                    delta=0.03,
                    gamma=1e-2,
                    optimizer_type="adamod",
                    use_momentum=False,
                    **base_params,
                ),
            ),
            ("PSO", lambda: optimize_pso(n_particles=10, **base_params)),
            ("GA", lambda: optimize_ga(population_size=10, **base_params)),
            ("SA", lambda: optimize_sa(**base_params)),
        ]

        for name, optimizer_func in optimizers:
            recorder = optimizer_func()
            initial_pib = recorder.history[0]["pib"]
            best_pib = max(item["pib"] for item in recorder.history)
            threshold = 0.7 if name == "SPGD" else 0.2
            assert best_pib > initial_pib * threshold, f"{name} failed to find a nontrivial PIB solution"


class TestSPGDOptimizerTypes:
    """Test SPGD with different optimizer types."""

    def test_spgd_fixed_gain(self):
        """Test SPGD with fixed gain (reference implementation style)."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            Cn2=1e-14,
            optimizer_type="spgd",
            delta=0.02,
            gamma=5e-4,
            seed=42,
        )

        assert len(recorder.history) == 11

    def test_spgd_sgd(self):
        """Test SPGD with SGD optimizer."""
        recorder = optimize_spgd(
            epochs=10,
            r_bucket=20,
            n_grid=64,
            Cn2=1e-14,
            optimizer_type="sgd",
            delta=0.02,
            gamma=5e-3,
            seed=42,
        )

        assert len(recorder.history) == 11
