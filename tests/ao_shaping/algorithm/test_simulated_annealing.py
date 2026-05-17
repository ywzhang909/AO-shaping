import numpy as np

from ao_shaping.algorithm.simulated_annealing import (
    SimulatedAnnealing,
    SAParams,
    TempSchedule,
    minimize_sa,
)


class TestSAParams:
    """Test SAParams dataclass."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = SAParams()
        
        assert params.n_iterations == 1000
        assert params.initial_temp == 100.0
        assert params.final_temp == 0.01
        assert params.schedule == TempSchedule.EXPONENTIAL
        assert params.step_size == 0.5
        assert params.bounds == (-10.0, 10.0)
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = SAParams(
            n_iterations=500,
            initial_temp=200.0,
            final_temp=0.001,
            schedule=TempSchedule.LINEAR,
            step_size=0.3,
            bounds=(-5, 5)
        )
        
        assert params.n_iterations == 500
        assert params.initial_temp == 200.0
        assert params.final_temp == 0.001
        assert params.schedule == TempSchedule.LINEAR
        assert params.step_size == 0.3
        assert params.bounds == (-5, 5)


class TestTempSchedule:
    """Test temperature schedule types."""
    
    def test_schedule_values(self):
        """Test all schedule types are available."""
        assert TempSchedule.LINEAR is not None
        assert TempSchedule.EXPONENTIAL is not None
        assert TempSchedule.LOGARITHMIC is not None
        assert TempSchedule.COSINE is not None


class TestSimulatedAnnealing:
    """Test SimulatedAnnealing class."""
    
    def test_initialization(self):
        """Test SA initialization."""
        sa = SimulatedAnnealing(dim=10)
        
        assert sa.dim == 10
        assert isinstance(sa.params, SAParams)
        assert sa.history is not None
    
    def test_optimize_sphere_function(self):
        """Test SA on simple sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(
            n_iterations=100,
            initial_temp=50.0,
            final_temp=0.01,
            bounds=(-5, 5)
        )
        sa = SimulatedAnnealing(dim=5, params=params)
        
        best_x, best_f = sa.optimize(sphere)
        
        assert best_x.shape == (5,)
        assert best_f >= 0
    
    def test_optimize_with_init(self):
        """Test SA with initial point."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=100, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=3, params=params)
        
        init_x = np.array([1.0, 1.0, 1.0])
        best_x, best_f = sa.optimize(sphere, init_x=init_x)
        
        assert best_x.shape == (3,)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=1000, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        best_x, best_f = sa.optimize(sphere, early_stop_threshold=0.01)
        
        assert len(sa.convergence_history) < 1000
    
    def test_convergence_history(self):
        """Test that convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=50, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        sa.optimize(sphere)
        
        history = sa.convergence_history
        assert len(history) > 0
        assert len(history) <= params.n_iterations + 1
    
    def test_temperature_history(self):
        """Test that temperature history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=30, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        sa.optimize(sphere)
        
        assert len(sa.history.temperature) > 0
        assert len(sa.history.temperature) <= params.n_iterations + 1
    
    def test_temperature_decreases(self):
        """Test that temperature decreases over iterations."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=100, schedule=TempSchedule.LINEAR, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        sa.optimize(sphere)
        
        temps = sa.history.temperature
        assert temps[0] >= temps[-1]


class TestTemperatureSchedules:
    """Test different temperature schedules."""
    
    def test_linear_schedule(self):
        """Test linear schedule."""
        sa = SimulatedAnnealing(dim=2, params=SAParams(
            n_iterations=100,
            initial_temp=100.0,
            final_temp=0.0,
            schedule=TempSchedule.LINEAR,
            bounds=(-5, 5)
        ))
        
        temps = [sa._get_temperature(i) for i in range(101)]
        
        assert temps[0] == 100.0
        assert temps[50] == 50.0
        assert temps[100] == 0.0
    
    def test_exponential_schedule(self):
        """Test exponential schedule."""
        sa = SimulatedAnnealing(dim=2, params=SAParams(
            n_iterations=100,
            initial_temp=100.0,
            final_temp=0.01,
            schedule=TempSchedule.EXPONENTIAL,
            bounds=(-5, 5)
        ))
        
        t0 = sa._get_temperature(0)
        t50 = sa._get_temperature(50)
        t100 = sa._get_temperature(100)
        
        assert t0 >= t50 >= t100
        assert t0 == 100.0
        assert t100 == 0.01
    
    def test_cosine_schedule(self):
        """Test cosine schedule."""
        sa = SimulatedAnnealing(dim=2, params=SAParams(
            n_iterations=100,
            initial_temp=100.0,
            final_temp=0.0,
            schedule=TempSchedule.COSINE,
            bounds=(-5, 5)
        ))
        
        t0 = sa._get_temperature(0)
        t50 = sa._get_temperature(50)
        t100 = sa._get_temperature(100)
        
        assert t0 == 100.0
        assert t50 < 50.0  # Cosine dips below linear
        assert t100 == 0.0


class TestNeighborGeneration:
    """Test neighbor generation."""
    
    def test_neighbor_within_bounds(self):
        """Test that generated neighbors stay within bounds."""
        sa = SimulatedAnnealing(dim=3, params=SAParams(
            bounds=(-2, 2),
            step_size=1.0
        ))
        
        current = np.array([0.0, 0.0, 0.0])
        
        for _ in range(100):
            neighbor = sa._generate_neighbor(current)
            assert np.all(neighbor >= -2 - 1e-6)
            assert np.all(neighbor <= 2 + 1e-6)
    
    def test_neighbor_changes(self):
        """Test that neighbor is different from current."""
        sa = SimulatedAnnealing(dim=3, params=SAParams(step_size=1.0))
        
        current = np.array([0.0, 0.0, 0.0])
        
        neighbor = sa._generate_neighbor(current)
        
        assert not np.allclose(neighbor, current)


class TestAcceptanceProbability:
    """Test acceptance probability."""
    
    def test_accept_better(self):
        """Test that better solutions are always accepted."""
        sa = SimulatedAnnealing(dim=2, params=SAParams(initial_temp=100.0))
        
        prob = sa._acceptance_probability(10.0, 5.0, 50.0)
        
        assert prob == 1.0
    
    def test_reject_worse_with_high_temp(self):
        """Test that worse solutions can be accepted with high temp."""
        sa = SimulatedAnnealing(dim=2, params=SAParams(initial_temp=100.0))
        
        prob = sa._acceptance_probability(5.0, 10.0, 100.0)
        
        assert prob > 0.0
        assert prob < 1.0
    
    def test_reject_worse_with_low_temp(self):
        """Test that worse solutions are rarely accepted with low temp."""
        sa = SimulatedAnnealing(dim=2, params=SAParams(initial_temp=100.0))
        
        prob = sa._acceptance_probability(5.0, 10.0, 0.001)
        
        assert prob < 0.1
    
    def test_zero_temperature(self):
        """Test acceptance with zero temperature."""
        sa = SimulatedAnnealing(dim=2, params=SAParams())
        
        prob = sa._acceptance_probability(5.0, 10.0, 0.0)
        
        assert prob == 0.0


class TestMinimizeSA:
    """Test convenience function."""
    
    def test_minimize_sa_basic(self):
        """Test minimize_sa convenience function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        best_x, best_f = minimize_sa(
            fitness_fn=sphere,
            dim=3,
            n_iterations=100,
            bounds=(-5, 5)
        )
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_minimize_sa_with_early_stop(self):
        """Test minimize_sa with early stopping."""
        def sphere(x):
            return np.sum(x ** 2)
        
        best_x, best_f = minimize_sa(
            fitness_fn=sphere,
            dim=2,
            n_iterations=500,
            early_stop_threshold=0.1
        )
        
        assert best_f < 0.1


class TestSAStability:
    """Test SA algorithm stability."""
    
    def test_deterministic_with_seed(self):
        """Test that SA produces same results with same seed."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=50, bounds=(-5, 5))
        
        sa1 = SimulatedAnnealing(dim=3, params=params, random_state=np.random.default_rng(123))
        sa2 = SimulatedAnnealing(dim=3, params=params, random_state=np.random.default_rng(123))
        
        best_x1, best_f1 = sa1.optimize(sphere)
        best_x2, best_f2 = sa2.optimize(sphere)
        
        np.testing.assert_array_equal(best_x1, best_x2)
        assert best_f1 == best_f2
    
    def test_no_nan_output(self):
        """Test that SA never produces NaN outputs."""
        def rosenbrock(x):
            return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
        
        params = SAParams(n_iterations=50, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=4, params=params)
        
        best_x, best_f = sa.optimize(rosenbrock)
        
        assert not np.isnan(best_x).any()
        assert not np.isnan(best_f)
        assert not np.isinf(best_f)
    
    def test_best_fitness_non_increasing(self):
        """Test that best fitness is non-increasing over time."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = SAParams(n_iterations=100, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        sa.optimize(sphere)
        
        history = sa.history.best_fitness
        for i in range(1, len(history)):
            assert history[i] <= history[i-1]


class TestSACallback:
    """Test SA callback functionality."""
    
    def test_callback_called(self):
        """Test that callback is called each iteration."""
        call_count = [0]
        
        def sphere(x):
            return np.sum(x ** 2)
        
        def callback(iter, pos, fit, temp):
            call_count[0] += 1
        
        params = SAParams(n_iterations=30, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        sa.optimize(sphere, callback=callback)
        
        assert call_count[0] == 30
    
    def test_callback_receives_correct_args(self):
        """Test that callback receives correct arguments."""
        received_args = []
        
        def sphere(x):
            return np.sum(x ** 2)
        
        def callback(iter, pos, fit, temp):
            received_args.append((iter, fit, temp))
        
        params = SAParams(n_iterations=10, bounds=(-5, 5))
        sa = SimulatedAnnealing(dim=2, params=params)
        
        sa.optimize(sphere, callback=callback)
        
        assert len(received_args) == 10
        assert received_args[0][0] == 1  # First iteration
        assert received_args[-1][0] == 10  # Last iteration