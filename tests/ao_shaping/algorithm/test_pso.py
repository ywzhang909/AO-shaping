import numpy as np

from ao_shaping.algorithm.pso import (
    ParticleSwarmOptimizer,
    PSOParams,
    Particle,
    minimize_pso,
)


class TestPSOParams:
    """Test PSOParams dataclass."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = PSOParams()
        
        assert params.n_particles == 30
        assert params.n_iterations == 1000
        assert params.w == 0.729
        assert params.c1 == 1.49
        assert params.c2 == 1.49
        assert params.v_max == 2.0
        assert params.bounds == (-10.0, 10.0)
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = PSOParams(
            n_particles=50,
            n_iterations=500,
            w=0.5,
            c1=1.5,
            c2=2.0,
            bounds=(-5, 5)
        )
        
        assert params.n_particles == 50
        assert params.n_iterations == 500
        assert params.w == 0.5
        assert params.c1 == 1.5
        assert params.c2 == 2.0
        assert params.bounds == (-5, 5)


class TestParticle:
    """Test Particle class."""
    
    def test_particle_creation(self):
        """Test creating a particle."""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        fitness = 10.0
        
        p = Particle(position, velocity, fitness)
        
        np.testing.assert_array_equal(p.position, position)
        np.testing.assert_array_equal(p.velocity, velocity)
        assert p.fitness == fitness
        np.testing.assert_array_equal(p.best_position, position)
        assert p.best_fitness == fitness


class TestParticleSwarmOptimizer:
    """Test ParticleSwarmOptimizer class."""
    
    def test_initialization(self):
        """Test PSO initialization."""
        pso = ParticleSwarmOptimizer(dim=10)
        
        assert pso.dim == 10
        assert isinstance(pso.params, PSOParams)
        assert pso.history is not None
    
    def test_optimize_sphere_function(self):
        """Test PSO on simple sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(
            n_particles=20,
            n_iterations=50,
            bounds=(-5, 5)
        )
        pso = ParticleSwarmOptimizer(dim=5, params=params)
        
        best_x, best_f = pso.optimize(sphere)
        
        assert best_x.shape == (5,)
        assert best_f >= 0
        assert best_f < 50  # Should converge to near 0
    
    def test_optimize_with_init(self):
        """Test PSO with initial point."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(n_particles=20, n_iterations=30, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=3, params=params)
        
        init_x = np.array([1.0, 1.0, 1.0])
        best_x, best_f = pso.optimize(sphere, init_x=init_x)
        
        assert best_x.shape == (3,)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(n_particles=20, n_iterations=1000, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=2, params=params)
        
        best_x, best_f = pso.optimize(sphere, early_stop_threshold=0.01)
        
        assert len(pso.convergence_history) < 1000
    
    def test_convergence_history(self):
        """Test that convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(n_particles=10, n_iterations=20, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=2, params=params)
        
        pso.optimize(sphere)
        
        history = pso.convergence_history
        assert len(history) > 0
        assert len(history) <= params.n_iterations + 1
    
    def test_global_best_tracking(self):
        """Test that global best is properly tracked."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(n_particles=10, n_iterations=20, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=2, params=params)
        
        pso.optimize(sphere)
        
        # Global best should be set
        assert pso.global_best_position is not None
        assert pso.global_best_fitness < float('inf')
    
    def test_particles_initialized(self):
        """Test that particles are initialized correctly."""
        params = PSOParams(n_particles=15, n_iterations=10, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=4, params=params)
        
        pso.optimize(lambda x: np.sum(x**2))
        
        assert len(pso.particles) == params.n_particles
        for p in pso.particles:
            assert p.position.shape == (4,)
            assert p.velocity.shape == (4,)


class TestMinimizePSO:
    """Test convenience function."""
    
    def test_minimize_pso_basic(self):
        """Test minimize_pso convenience function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        best_x, best_f = minimize_pso(
            fitness_fn=sphere,
            dim=3,
            n_particles=20,
            n_iterations=30,
            bounds=(-5, 5)
        )
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_minimize_pso_with_early_stop(self):
        """Test minimize_pso with early stopping."""
        def sphere(x):
            return np.sum(x ** 2)
        
        best_x, best_f = minimize_pso(
            fitness_fn=sphere,
            dim=2,
            n_iterations=500,
            early_stop_threshold=0.05
        )
        
        assert best_f < 0.05


class TestPSOStability:
    """Test PSO algorithm stability."""
    
    def test_deterministic_with_seed(self):
        """Test that PSO produces same results with same seed."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(n_particles=20, n_iterations=30, bounds=(-5, 5))
        
        pso1 = ParticleSwarmOptimizer(dim=3, params=params, random_state=np.random.default_rng(123))
        pso2 = ParticleSwarmOptimizer(dim=3, params=params, random_state=np.random.default_rng(123))
        
        best_x1, best_f1 = pso1.optimize(sphere)
        best_x2, best_f2 = pso2.optimize(sphere)
        
        np.testing.assert_array_almost_equal(best_x1, best_x2)
        assert best_f1 == best_f2
    
    def test_no_nan_output(self):
        """Test that PSO never produces NaN outputs."""
        def rosenbrock(x):
            return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
        
        params = PSOParams(n_particles=30, n_iterations=50, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=4, params=params)
        
        best_x, best_f = pso.optimize(rosenbrock)
        
        assert not np.isnan(best_x).any()
        assert not np.isnan(best_f)
        assert not np.isinf(best_f)
    
    def test_velocity_clamping(self):
        """Test that velocities are clamped to v_max."""
        params = PSOParams(n_particles=10, n_iterations=10, v_max=1.0, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=3, params=params)
        
        pso.optimize(lambda x: np.sum(x**2))
        
        for p in pso.particles:
            assert np.all(np.abs(p.velocity) <= params.v_max + 1e-6)
    
    def test_position_bounds(self):
        """Test that positions stay within bounds."""
        params = PSOParams(n_particles=10, n_iterations=20, bounds=(-3, 3))
        pso = ParticleSwarmOptimizer(dim=2, params=params)
        
        pso.optimize(lambda x: np.sum(x**2))
        
        for p in pso.particles:
            assert np.all(p.position >= params.bounds[0] - 1e-6)
            assert np.all(p.position <= params.bounds[1] + 1e-6)


class TestPSOConvergence:
    """Test PSO convergence properties."""
    
    def test_convergence_to_global_minimum(self):
        """Test that PSO converges to global minimum on sphere."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = PSOParams(
            n_particles=50,
            n_iterations=200,
            bounds=(-10, 10)
        )
        pso = ParticleSwarmOptimizer(dim=5, params=params)
        
        best_x, best_f = pso.optimize(sphere)
        
        assert best_f < 1.0
    
    def test_callback_function(self):
        """Test that callback is called correctly."""
        call_count = [0]
        
        def sphere(x):
            return np.sum(x ** 2)
        
        def callback(iter, pos, fit):
            call_count[0] += 1
        
        params = PSOParams(n_particles=10, n_iterations=20, bounds=(-5, 5))
        pso = ParticleSwarmOptimizer(dim=2, params=params)
        
        pso.optimize(sphere, callback=callback)
        
        assert call_count[0] == 20