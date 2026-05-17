"""Genetic Algorithm optimization module.

This module provides pure Genetic Algorithm operators for general optimization.
Extracts core GA components from ga_zernike.py for reusability.

Key features:
- Tournament selection
- Blend crossover (BLX-alpha)
- Gaussian mutation
- Elitism preservation

Example:
    >>> from ao_shaping.algorithm.ga import GeneticAlgorithm
    >>> import numpy as np
    >>> 
    >>> def objective(x):
    ...     return np.sum(x ** 2)  # Sphere function
    >>> 
    >>> ga = GeneticAlgorithm(
    ...     dim=5,
    ...     pop_size=50,
    ...     n_generations=100,
    ...     bounds=(-10, 10)
    ... )
    >>> best_x, best_f = ga.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class FitnessFunction(Protocol):
    """Protocol for fitness function."""
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate fitness.
        
        Args:
            x: Individual to evaluate.
            
        Returns:
            Fitness value (lower is better for minimization).
        """
        ...


@dataclass
class GAParams:
    """Genetic Algorithm parameters."""
    pop_size: int = 50
    n_generations: int = 2000
    crossover_prob: float = 0.7
    mutation_prob: float = 0.15
    tournament_size: int = 3
    elite_count: int = 2
    bounds: tuple[float, float] = (-50.0, 50.0)
    alpha: float = 0.5  # BLX-alpha expansion factor
    mutation_sigma: float = 5.0  # Gaussian mutation sigma


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Select an individual using tournament selection.
    
    Args:
        population: Array of shape (pop_size, dim) containing the population.
        fitness: Array of shape (pop_size,) containing fitness values (lower is better).
        tournament_size: Number of individuals to compete in the tournament.
        random_state: Random generator for reproducibility.
        
    Returns:
        Selected individual as array of shape (dim,).
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    pop_size = len(population)
    contestants = rng.choice(pop_size, tournament_size, replace=False)
    # For minimization, select the one with minimum fitness
    best_idx = contestants[np.argmin(fitness[contestants])]
    return population[best_idx].copy()


def blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
    random_state: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend crossover (BLX-alpha) for two parents.
    
    Creates two children by interpolating/extrapolating between parents
    within a range expanded by factor alpha.
    
    Args:
        parent1: First parent array of shape (dim,).
        parent2: Second parent array of shape (dim,).
        alpha: Expansion factor for the search range.
        random_state: Random generator for reproducibility.
        
    Returns:
        Tuple of two children arrays.
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    n = len(parent1)
    # Calculate the range between parents
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    I = c_max - c_min
    
    # Expand the range
    lower = c_min - alpha * I
    upper = c_max + alpha * I
    
    # Generate children uniformly in the expanded range
    child1 = lower + rng.random(n) * (upper - lower)
    child2 = lower + rng.random(n) * (upper - lower)
    
    return child1, child2


def gaussian_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    sigma: float = 5.0,
    bounds: tuple[float, float] = (-50.0, 50.0),
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply Gaussian mutation to an individual.
    
    Each gene has a probability of being mutated according to mutation_rate.
    Mutated genes are perturbed by a Gaussian with mean 0 and given sigma.
    Values are clipped to the specified bounds.
    
    Args:
        individual: Individual to mutate, shape (dim,).
        mutation_rate: Probability of mutating each gene.
        sigma: Standard deviation of the Gaussian perturbation.
        bounds: Tuple of (min, max) bounds for clipping.
        random_state: Random generator for reproducibility.
        
    Returns:
        Mutated individual.
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    mutated = individual.copy()
    mask = rng.random(len(individual)) < mutation_rate
    if np.any(mask):
        noise = rng.normal(0, sigma, int(np.sum(mask)))
        mutated[mask] += noise
    return np.clip(mutated, bounds[0], bounds[1])


@dataclass
class GAHistory:
    """History of GA optimization run."""
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    best_individual: np.ndarray | None = None


class GeneticAlgorithm:
    """Genetic Algorithm optimizer for continuous optimization.
    
    Attributes:
        dim: Dimension of the problem.
        params: GA parameters.
        history: Optimization history.
    """
    
    def __init__(
        self,
        dim: int,
        params: GAParams | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """Initialize Genetic Algorithm.
        
        Args:
            dim: Dimension of the optimization problem.
            params: GA parameters. If None, uses default GAParams.
            random_state: Random generator for reproducibility.
        """
        self.dim = dim
        self.params = params if params is not None else GAParams()
        self.rng = random_state if random_state is not None else np.random.default_rng()
        self.history = GAHistory()
        
    def _initialize_population(self, init_x: np.ndarray | None = None) -> np.ndarray:
        """Initialize population.
        
        Args:
            init_x: Initial point to include in population. If None, starts from random.
            
        Returns:
            Initial population array of shape (pop_size, dim).
        """
        pop = np.random.uniform(
            self.params.bounds[0],
            self.params.bounds[1],
            (self.params.pop_size, self.dim)
        )
        # Include initial point if provided
        if init_x is not None:
            pop[0] = init_x.copy()
        return pop
    
    def _evaluate_population(self, pop: np.ndarray, fitness_fn: FitnessFunction) -> np.ndarray:
        """Evaluate fitness for entire population.
        
        Args:
            pop: Population array.
            fitness_fn: Fitness function.
            
        Returns:
            Fitness values array of shape (pop_size,).
        """
        fitness = np.array([fitness_fn(ind) for ind in pop])
        return fitness
    
    def optimize(
        self,
        fitness_fn: FitnessFunction,
        init_x: np.ndarray | None = None,
        early_stop_threshold: float | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run genetic algorithm optimization.
        
        Args:
            fitness_fn: Fitness function to minimize.
            init_x: Initial point to include in population.
            early_stop_threshold: Stop if best fitness below this threshold.
            callback: Optional callback function called after each generation
                      with (generation, best_individual, best_fitness).
        
        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = self._initialize_population(init_x)
        
        # Evaluate initial population
        fitness = self._evaluate_population(population, fitness_fn)
        
        # Find best in initial population
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        # Record history
        self.history.best_fitness.append(best_fitness)
        self.history.mean_fitness.append(np.mean(fitness))
        
        # Main GA loop
        for gen in range(1, self.params.n_generations + 1):
            # Create new population
            new_population = []
            
            # Elitism: preserve top elite_count individuals
            elite_indices = np.argsort(fitness)[:self.params.elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring through selection, crossover, and mutation
            while len(new_population) < self.params.pop_size:
                # Tournament selection for parents
                parent1 = tournament_selection(
                    population, fitness, self.params.tournament_size, self.rng
                )
                parent2 = tournament_selection(
                    population, fitness, self.params.tournament_size, self.rng
                )
                
                # Crossover
                if self.rng.random() < self.params.crossover_prob:
                    child1, child2 = blend_crossover(
                        parent1, parent2, self.params.alpha, self.rng
                    )
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = gaussian_mutation(
                    child1,
                    self.params.mutation_prob,
                    self.params.mutation_sigma,
                    self.params.bounds,
                    self.rng
                )
                child2 = gaussian_mutation(
                    child2,
                    self.params.mutation_prob,
                    self.params.mutation_sigma,
                    self.params.bounds,
                    self.rng
                )
                
                new_population.append(child1)
                if len(new_population) < self.params.pop_size:
                    new_population.append(child2)
            
            # Trim to exact population size
            population = np.array(new_population[:self.params.pop_size])
            
            # Evaluate new population
            fitness = self._evaluate_population(population, fitness_fn)
            
            # Find best individual in this generation
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            current_best_individual = population[current_best_idx].copy()
            
            # Update global best
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
            
            # Record history
            self.history.best_fitness.append(best_fitness)
            self.history.mean_fitness.append(np.mean(fitness))
            
            # Callback
            if callback is not None:
                callback(gen, best_individual.copy(), best_fitness)
            
            # Early stopping
            if early_stop_threshold is not None and best_fitness < early_stop_threshold:
                break
        
        # Store final best
        self.history.best_individual = best_individual.copy()
        
        return best_individual.copy(), best_fitness
    
    @property
    def convergence_history(self) -> list[float]:
        """Return convergence history (best fitness per generation)."""
        return self.history.best_fitness


# Convenience function for simple optimization
def minimize_ga(
    fitness_fn: FitnessFunction,
    dim: int,
    pop_size: int = 50,
    n_generations: int = 1000,
    bounds: tuple[float, float] = (-10.0, 10.0),
    init_x: np.ndarray | None = None,
    early_stop_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convenience function for GA optimization.
    
    Args:
        fitness_fn: Fitness function to minimize.
        dim: Dimension of the problem.
        pop_size: Population size.
        n_generations: Number of generations.
        bounds: Search space bounds (min, max).
        init_x: Initial point.
        early_stop_threshold: Early stopping threshold.
        
    Returns:
        Tuple of (best_solution, best_fitness).
    """
    params = GAParams(
        pop_size=pop_size,
        n_generations=n_generations,
        bounds=bounds,
    )
    ga = GeneticAlgorithm(dim=dim, params=params)
    return ga.optimize(
        fitness_fn=fitness_fn,
        init_x=init_x,
        early_stop_threshold=early_stop_threshold,
    )