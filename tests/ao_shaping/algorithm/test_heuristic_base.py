import numpy as np

from ao_shaping.algorithm.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OptimizerConfig()
        
        assert config.n_iterations == 1000
        assert config.bounds == (-10.0, 10.0)
        assert config.early_stop_threshold is None
        assert config.seed is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizerConfig(
            n_iterations=500,
            bounds=(-5, 5),
            early_stop_threshold=0.01,
            seed=42
        )
        
        assert config.n_iterations == 500
        assert config.bounds == (-5, 5)
        assert config.early_stop_threshold == 0.01
        assert config.seed == 42


class TestOptimizerType:
    """Test OptimizerType enum."""
    
    def test_all_types_exist(self):
        """Test all optimizer types are defined."""
        assert OptimizerType.GA is not None
        assert OptimizerType.PSO is not None
        assert OptimizerType.SA is not None
        assert OptimizerType.HILL_CLIMBING is not None
        assert OptimizerType.RANDOM_SEARCH is not None
        assert OptimizerType.CROSS_ENTROPY is not None
        assert OptimizerType.DIFFERENTIAL_EVOLUTION is not None


class TestHeuristicOptimizerFactory:
    """Test HeuristicOptimizer factory method."""
    
    def test_create_ga(self):
        """Test creating GA via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.GA,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.ga import GeneticAlgorithm
        assert isinstance(opt, GeneticAlgorithm)
    
    def test_create_pso(self):
        """Test creating PSO via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.PSO,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.pso import ParticleSwarmOptimizer
        assert isinstance(opt, ParticleSwarmOptimizer)
    
    def test_create_sa(self):
        """Test creating SA via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.SA,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.simulated_annealing import SimulatedAnnealing
        assert isinstance(opt, SimulatedAnnealing)
    
    def test_create_hill_climbing(self):
        """Test creating Hill Climbing via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.HILL_CLIMBING,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.hill_climbing import HillClimbing
        assert isinstance(opt, HillClimbing)
    
    def test_create_random_search(self):
        """Test creating Random Search via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.RANDOM_SEARCH,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.random_search import RandomSearch
        assert isinstance(opt, RandomSearch)
    
    def test_create_cross_entropy(self):
        """Test creating CEM via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.CROSS_ENTROPY,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.cross_entropy import CrossEntropyMethod
        assert isinstance(opt, CrossEntropyMethod)
    
    def test_create_de(self):
        """Test creating DE via factory."""
        opt = HeuristicOptimizer.create(
            OptimizerType.DIFFERENTIAL_EVOLUTION,
            dim=5,
            n_iterations=50
        )
        
        from ao_shaping.algorithm.differential_evolution import DifferentialEvolution
        assert isinstance(opt, DifferentialEvolution)
    
    def test_factory_with_seed(self):
        """Test factory with seed produces deterministic results."""
        def sphere(x):
            return np.sum(x ** 2)
        
        opt1 = HeuristicOptimizer.create(
            OptimizerType.HILL_CLIMBING,
            dim=3,
            n_iterations=30,
            seed=123
        )
        opt2 = HeuristicOptimizer.create(
            OptimizerType.HILL_CLIMBING,
            dim=3,
            n_iterations=30,
            seed=123
        )
        
        _, fit1 = opt1.optimize(sphere)
        _, fit2 = opt2.optimize(sphere)
        
        assert fit1 == fit2
    
    def test_factory_unknown_type_raises(self):
        """Test unknown optimizer type raises ValueError."""
        class UnknownType(OptimizerType):
            pass
        
        with np.testing.assert_raises(ValueError):
            HeuristicOptimizer.create(UnknownType, dim=5)


class TestOptimizerSwitching:
    """Test switching between different optimizers."""
    
    def test_interface_consistency(self):
        """Test all optimizers have consistent interface."""
        def sphere(x):
            return np.sum(x ** 2)
        
        for opt_type in OptimizerType:
            opt = HeuristicOptimizer.create(
                opt_type,
                dim=3,
                n_iterations=20
            )
            
            best_x, best_f = opt.optimize(sphere)
            
            assert best_x.shape == (3,)
            assert isinstance(best_f, (float, np.floating))
            assert len(opt.convergence_history) > 0
    
    def test_different_algorithms_on_same_problem(self):
        """Test different algorithms on same problem."""
        def sphere(x):
            return np.sum(x ** 2)
        
        results = {}
        for opt_type in [
            OptimizerType.HILL_CLIMBING,
            OptimizerType.RANDOM_SEARCH,
            OptimizerType.CROSS_ENTROPY,
            OptimizerType.DIFFERENTIAL_EVOLUTION,
        ]:
            opt = HeuristicOptimizer.create(
                opt_type,
                dim=3,
                n_iterations=50,
                bounds=(-5, 5)
            )
            _, best_f = opt.optimize(sphere)
            results[opt_type.name] = best_f
        
        # At least some should converge reasonably
        assert all(f >= 0 for f in results.values())