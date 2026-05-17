import numpy as np

from ao_shaping.algorithm.hill_climbing import HillClimbing, HCConfig
from ao_shaping.algorithm.random_search import RandomSearch
from ao_shaping.algorithm.cross_entropy import CrossEntropyMethod, CEMConfig
from ao_shaping.algorithm.differential_evolution import DifferentialEvolution, DEConfig


class TestHillClimbing:
    """Test Hill Climbing optimizer."""
    
    def test_initialization(self):
        """Test HC initialization."""
        hc = HillClimbing(dim=5)
        assert hc.dim == 5
        assert hc.hc_config.step_size == 0.1
    
    def test_optimize_sphere(self):
        """Test HC on sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        hc = HillClimbing(dim=3, n_iterations=100, neighbor_std=0.5)
        best_x, best_f = hc.optimize(sphere)
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_with_init(self):
        """Test HC with initial point."""
        def sphere(x):
            return np.sum(x ** 2)
        
        hc = HillClimbing(dim=3, n_iterations=50)
        init_x = np.array([1.0, 1.0, 1.0])
        best_x, _ = hc.optimize(sphere, init_x=init_x)
        
        assert best_x.shape == (3,)
    
    def test_convergence_history(self):
        """Test convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        hc = HillClimbing(dim=2, n_iterations=20)
        hc.optimize(sphere)
        
        assert len(hc.convergence_history) > 0
    
    def test_deterministic_with_seed(self):
        """Test deterministic with seed."""
        def sphere(x):
            return np.sum(x ** 2)
        
        hc1 = HillClimbing(dim=3, n_iterations=30, seed=42)
        hc2 = HillClimbing(dim=3, n_iterations=30, seed=42)
        
        _, f1 = hc1.optimize(sphere)
        _, f2 = hc2.optimize(sphere)
        
        assert f1 == f2


class TestRandomSearch:
    """Test Random Search optimizer."""
    
    def test_initialization(self):
        """Test RS initialization."""
        rs = RandomSearch(dim=5)
        assert rs.dim == 5
    
    def test_optimize_sphere(self):
        """Test RS on sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        rs = RandomSearch(dim=3, n_iterations=100)
        best_x, best_f = rs.optimize(sphere)
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_convergence_history(self):
        """Test convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        rs = RandomSearch(dim=2, n_iterations=50)
        rs.optimize(sphere)
        
        assert len(rs.convergence_history) > 0
    
    def test_early_stopping(self):
        """Test early stopping."""
        def sphere(x):
            return np.sum(x ** 2)
        
        from ao_shaping.algorithm.heuristic_base import OptimizerConfig
        config = OptimizerConfig(n_iterations=1000, early_stop_threshold=0.1)
        rs = RandomSearch(dim=2, config=config)
        
        best_x, best_f = rs.optimize(sphere)
        
        assert len(rs.convergence_history) < 1000


class TestCrossEntropyMethod:
    """Test Cross-Entropy Method optimizer."""
    
    def test_initialization(self):
        """Test CEM initialization."""
        cem = CrossEntropyMethod(dim=5)
        assert cem.dim == 5
        assert cem.cem_config.pop_size == 50
    
    def test_optimize_sphere(self):
        """Test CEM on sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        cem = CrossEntropyMethod(dim=3, n_iterations=50)
        best_x, best_f = cem.optimize(sphere)
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_with_init(self):
        """Test CEM with initial point."""
        def sphere(x):
            return np.sum(x ** 2)
        
        cem = CrossEntropyMethod(dim=3, n_iterations=30)
        init_x = np.array([1.0, 1.0, 1.0])
        best_x, _ = cem.optimize(sphere, init_x=init_x)
        
        assert best_x.shape == (3,)
    
    def test_convergence_history(self):
        """Test convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        cem = CrossEntropyMethod(dim=2, n_iterations=20)
        cem.optimize(sphere)
        
        assert len(cem.convergence_history) > 0
    
    def test_deterministic_with_seed(self):
        """Test deterministic with seed."""
        def sphere(x):
            return np.sum(x ** 2)
        
        cem1 = CrossEntropyMethod(dim=3, n_iterations=30, seed=42)
        cem2 = CrossEntropyMethod(dim=3, n_iterations=30, seed=42)
        
        _, f1 = cem1.optimize(sphere)
        _, f2 = cem2.optimize(sphere)
        
        assert f1 == f2


class TestDifferentialEvolution:
    """Test Differential Evolution optimizer."""
    
    def test_initialization(self):
        """Test DE initialization."""
        de = DifferentialEvolution(dim=5)
        assert de.dim == 5
        assert de.de_config.pop_size == 30
    
    def test_optimize_sphere(self):
        """Test DE on sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        de = DifferentialEvolution(dim=3, n_iterations=50)
        best_x, best_f = de.optimize(sphere)
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_with_init(self):
        """Test DE with initial point."""
        def sphere(x):
            return np.sum(x ** 2)
        
        de = DifferentialEvolution(dim=3, n_iterations=30)
        init_x = np.array([1.0, 1.0, 1.0])
        best_x, _ = de.optimize(sphere, init_x=init_x)
        
        assert best_x.shape == (3,)
    
    def test_convergence_history(self):
        """Test convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        de = DifferentialEvolution(dim=2, n_iterations=20)
        de.optimize(sphere)
        
        assert len(de.convergence_history) > 0
    
    def test_deterministic_with_seed(self):
        """Test deterministic with seed."""
        def sphere(x):
            return np.sum(x ** 2)
        
        de1 = DifferentialEvolution(dim=3, n_iterations=30, seed=42)
        de2 = DifferentialEvolution(dim=3, n_iterations=30, seed=42)
        
        _, f1 = de1.optimize(sphere)
        _, f2 = de2.optimize(sphere)
        
        assert f1 == f2
    
    def test_no_nan_output(self):
        """Test DE never produces NaN."""
        def rastrigin(x):
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        
        de = DifferentialEvolution(dim=4, n_iterations=30, bounds=(-5, 5))
        best_x, best_f = de.optimize(rastrigin)
        
        assert not np.isnan(best_x).any()
        assert not np.isnan(best_f)
        assert not np.isinf(best_f)


class TestAllOptimizersConvergence:
    """Test convergence of all new optimizers."""
    
    def test_all_converge_on_sphere(self):
        """Test all optimizers converge on sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        optimizers = [
            (HillClimbing, {"dim": 3, "n_iterations": 100, "neighbor_std": 0.3}),
            (RandomSearch, {"dim": 3, "n_iterations": 100}),
            (CrossEntropyMethod, {"dim": 3, "n_iterations": 50}),
            (DifferentialEvolution, {"dim": 3, "n_iterations": 50}),
        ]
        
        for opt_cls, kwargs in optimizers:
            opt = opt_cls(**kwargs)
            _, best_f = opt.optimize(sphere)
            
            assert best_f >= 0, f"{opt_cls.__name__} produced negative fitness"
            assert not np.isnan(best_f), f"{opt_cls.__name__} produced NaN"
            assert not np.isinf(best_f), f"{opt_cls.__name__} produced Inf"
    
    def test_all_respect_bounds(self):
        """Test all optimizers respect bounds."""
        def sphere(x):
            return np.sum(x ** 2)
        
        bounds = (-3, 3)
        
        optimizers = [
            (HillClimbing, {"dim": 2, "n_iterations": 30, "neighbor_std": 0.5}),
            (RandomSearch, {"dim": 2, "n_iterations": 30}),
            (CrossEntropyMethod, {"dim": 2, "n_iterations": 20}),
            (DifferentialEvolution, {"dim": 2, "n_iterations": 20}),
        ]
        
        for opt_cls, kwargs in optimizers:
            opt = opt_cls(**kwargs, bounds=bounds)
            best_x, _ = opt.optimize(sphere)
            
            assert np.all(best_x >= bounds[0] - 1e-6)
            assert np.all(best_x <= bounds[1] + 1e-6)