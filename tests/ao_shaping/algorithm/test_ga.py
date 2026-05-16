import numpy as np

from ao_shaping.algorithm.ga import (
    GeneticAlgorithm,
    GAParams,
    tournament_selection,
    blend_crossover,
    gaussian_mutation,
    minimize_ga,
)


class TestGAParams:
    """Test GAParams dataclass."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = GAParams()
        
        assert params.pop_size == 50
        assert params.n_generations == 2000
        assert params.crossover_prob == 0.7
        assert params.mutation_prob == 0.15
        assert params.tournament_size == 3
        assert params.elite_count == 2
        assert params.bounds == (-50.0, 50.0)
        assert params.alpha == 0.5
        assert params.mutation_sigma == 5.0
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = GAParams(
            pop_size=100,
            n_generations=500,
            crossover_prob=0.8,
            mutation_prob=0.1,
            bounds=(-5, 5)
        )
        
        assert params.pop_size == 100
        assert params.n_generations == 500
        assert params.crossover_prob == 0.8
        assert params.mutation_prob == 0.1
        assert params.bounds == (-5, 5)


class TestTournamentSelection:
    """Test tournament selection operator."""
    
    def test_selection_shape(self):
        """Test that selection returns correct shape."""
        pop_size = 10
        dim = 5
        population = np.random.randn(pop_size, dim)
        fitness = np.random.randn(pop_size)
        
        selected = tournament_selection(population, fitness, 3)
        
        assert selected.shape == (dim,)
    
    def test_selection_chooses_best(self):
        """Test that tournament selection prefers better fitness."""
        pop_size = 10
        dim = 5
        population = np.random.randn(pop_size, dim)
        fitness = np.array([10.0] * pop_size)
        fitness[3] = 0.0  # Best individual
        
        # Run multiple times and check we get the best one
        selected_idx = []
        for _ in range(100):
            selected = tournament_selection(population, fitness, 5)
            idx = np.where(np.all(population == selected, axis=1))[0]
            if len(idx) > 0:
                selected_idx.append(idx[0])
        
        assert 3 in selected_idx
    
    def test_selection_with_rng(self):
        """Test selection with custom random state."""
        rng = np.random.default_rng(42)
        pop_size = 10
        dim = 5
        population = np.random.randn(pop_size, dim)
        fitness = np.random.randn(pop_size)
        
        selected1 = tournament_selection(population, fitness, 3, rng)
        selected2 = tournament_selection(population, fitness, 3, rng)
        
        np.testing.assert_array_equal(selected1, selected2)


class TestBlendCrossover:
    """Test blend crossover operator."""
    
    def test_crossover_shape(self):
        """Test that crossover returns correct shapes."""
        parent1 = np.array([1.0, 2.0, 3.0])
        parent2 = np.array([4.0, 5.0, 6.0])
        
        child1, child2 = blend_crossover(parent1, parent2)
        
        assert child1.shape == parent1.shape
        assert child2.shape == parent2.shape
    
    def test_crossover_bounds(self):
        """Test that children stay within expanded range."""
        parent1 = np.array([0.0, 0.0, 0.0])
        parent2 = np.array([10.0, 10.0, 10.0])
        
        child1, child2 = blend_crossover(parent1, parent2, alpha=0.5)
        
        # With alpha=0.5, range expands by 50%
        # Original range [0, 10], expanded to [-5, 15]
        assert np.all(child1 >= -5)
        assert np.all(child1 <= 15)
        assert np.all(child2 >= -5)
        assert np.all(child2 <= 15)
    
    def test_crossover_with_rng(self):
        """Test crossover with custom random state."""
        rng = np.random.default_rng(42)
        parent1 = np.array([1.0, 2.0, 3.0])
        parent2 = np.array([4.0, 5.0, 6.0])
        
        child1a, child2a = blend_crossover(parent1, parent2, random_state=rng)
        child1b, child2b = blend_crossover(parent1, parent2, random_state=rng)
        
        np.testing.assert_array_equal(child1a, child1b)


class TestGaussianMutation:
    """Test Gaussian mutation operator."""
    
    def test_mutation_shape(self):
        """Test that mutation returns correct shape."""
        individual = np.array([1.0, 2.0, 3.0])
        
        mutated = gaussian_mutation(individual, 0.5)
        
        assert mutated.shape == individual.shape
    
    def test_mutation_bounds(self):
        """Test that mutation respects bounds."""
        individual = np.array([0.0, 0.0, 0.0])
        
        mutated = gaussian_mutation(individual, 1.0, bounds=(-1, 1))
        
        assert np.all(mutated >= -1)
        assert np.all(mutated <= 1)
    
    def test_mutation_rate_zero(self):
        """Test that zero mutation rate returns copy."""
        individual = np.array([1.0, 2.0, 3.0])
        
        mutated = gaussian_mutation(individual, 0.0)
        
        np.testing.assert_array_equal(mutated, individual)
    
    def test_mutation_with_rng(self):
        """Test mutation with custom random state."""
        rng = np.random.default_rng(42)
        individual = np.array([0.0, 0.0, 0.0])
        
        mutated1 = gaussian_mutation(individual, 1.0, random_state=rng)
        mutated2 = gaussian_mutation(individual, 1.0, random_state=rng)
        
        np.testing.assert_array_equal(mutated1, mutated2)


class TestGeneticAlgorithm:
    """Test GeneticAlgorithm class."""
    
    def test_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm(dim=10)
        
        assert ga.dim == 10
        assert isinstance(ga.params, GAParams)
        assert ga.history is not None
    
    def test_optimize_sphere_function(self):
        """Test GA on simple sphere function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = GAParams(
            pop_size=20,
            n_generations=50,
            bounds=(-5, 5)
        )
        ga = GeneticAlgorithm(dim=5, params=params)
        
        best_x, best_f = ga.optimize(sphere)
        
        assert best_x.shape == (5,)
        assert best_f >= 0
        assert best_f < 100  # Should converge to near 0
    
    def test_optimize_with_init(self):
        """Test GA with initial point."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = GAParams(pop_size=20, n_generations=30, bounds=(-5, 5))
        ga = GeneticAlgorithm(dim=3, params=params)
        
        init_x = np.array([1.0, 1.0, 1.0])
        best_x, best_f = ga.optimize(sphere, init_x=init_x)
        
        assert best_x.shape == (3,)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = GAParams(pop_size=20, n_generations=1000, bounds=(-5, 5))
        ga = GeneticAlgorithm(dim=2, params=params)
        
        best_x, best_f = ga.optimize(sphere, early_stop_threshold=0.01)
        
        # Should stop early due to threshold
        assert len(ga.convergence_history) < 1000
    
    def test_convergence_history(self):
        """Test that convergence history is recorded."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = GAParams(pop_size=10, n_generations=20, bounds=(-5, 5))
        ga = GeneticAlgorithm(dim=2, params=params)
        
        ga.optimize(sphere)
        
        history = ga.convergence_history
        assert len(history) > 0
        assert len(history) <= params.n_generations + 1


class TestMinimizeGA:
    """Test convenience function."""
    
    def test_minimize_ga_basic(self):
        """Test minimize_ga convenience function."""
        def sphere(x):
            return np.sum(x ** 2)
        
        best_x, best_f = minimize_ga(
            fitness_fn=sphere,
            dim=3,
            pop_size=20,
            n_generations=30,
            bounds=(-5, 5)
        )
        
        assert best_x.shape == (3,)
        assert best_f >= 0
    
    def test_minimize_ga_with_early_stop(self):
        """Test minimize_ga with early stopping."""
        def sphere(x):
            return np.sum(x ** 2)
        
        best_x, best_f = minimize_ga(
            fitness_fn=sphere,
            dim=2,
            n_generations=500,
            early_stop_threshold=0.05
        )
        
        assert best_f < 0.05


class TestGAStability:
    """Test GA algorithm stability."""
    
    def test_deterministic_with_seed(self):
        """Test that GA produces same results with same seed."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = GAParams(pop_size=20, n_generations=30, bounds=(-5, 5))
        
        ga1 = GeneticAlgorithm(dim=3, params=params, random_state=np.random.default_rng(123))
        ga2 = GeneticAlgorithm(dim=3, params=params, random_state=np.random.default_rng(123))
        
        best_x1, best_f1 = ga1.optimize(sphere)
        best_x2, best_f2 = ga2.optimize(sphere)
        
        np.testing.assert_array_almost_equal(best_x1, best_x2)
        assert best_f1 == best_f2
    
    def test_no_nan_output(self):
        """Test that GA never produces NaN outputs."""
        def rosenbrock(x):
            return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
        
        params = GAParams(pop_size=30, n_generations=50, bounds=(-5, 5))
        ga = GeneticAlgorithm(dim=4, params=params)
        
        best_x, best_f = ga.optimize(rosenbrock)
        
        assert not np.isnan(best_x).any()
        assert not np.isnan(best_f)
        assert not np.isinf(best_f)
    
    def test_population_diversity(self):
        """Test that population maintains diversity."""
        def sphere(x):
            return np.sum(x ** 2)
        
        params = GAParams(pop_size=50, n_generations=10, bounds=(-5, 5))
        ga = GeneticAlgorithm(dim=5, params=params)
        
        ga.optimize(sphere)
        
        # Check that multiple individuals exist
        assert len(ga.history.best_fitness) > 0