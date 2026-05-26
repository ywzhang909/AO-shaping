"""Differential Evolution (DE) optimization algorithm.

Global optimization algorithm that uses vector differences for mutation.

Example:
    >>> from ao_shaping.algorithm.differential_evolution import DifferentialEvolution
    >>> import numpy as np
    >>> 
    >>> def objective(x):
    ...     return np.sum(x ** 2)
    >>> 
    >>> de = DifferentialEvolution(dim=5, n_iterations=100)
    >>> best_x, best_f = de.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.algorithm.heuristic_base import HeuristicOptimizer, OptimizerConfig


@dataclass
class DEConfig:
    """Differential Evolution specific configuration."""
    pop_size: int = 30
    crossover_prob: float = 0.9
    mutation_factor: float = 0.8


class DifferentialEvolution(HeuristicOptimizer):
    """Differential Evolution optimizer.
    
    Uses differential mutation: mutant = best + F * (r1 - r2)
    """
    
    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        de_config: DEConfig | None = None,
        random_state: np.random.Generator | None = None,
        pop_size: int = 30,
        crossover_prob: float = 0.9,
        mutation_factor: float = 0.8,
    ):
        """Initialize DE optimizer."""
        super().__init__(dim, config, random_state)
        
        if de_config is None:
            de_config = DEConfig(
                pop_size=pop_size,
                crossover_prob=crossover_prob,
                mutation_factor=mutation_factor,
            )
        
        self.de_config = de_config
        self._population: np.ndarray | None = None
    
    def optimize(
        self,
        fitness_fn: callable,
        init_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run Differential Evolution optimization."""
        pop_size = max(10, self.de_config.pop_size)
        
        self._population = self.rng.uniform(
            self.config.bounds[0],
            self.config.bounds[1],
            (pop_size, self.dim)
        )
        
        if init_x is not None:
            self._population[0] = init_x.copy()
        
        fitness = np.array([fitness_fn(ind) for ind in self._population])
        
        best_idx = np.argmin(fitness)
        self._best_solution = self._population[best_idx].copy()
        self._best_fitness = fitness[best_idx]
        self._convergence_history = [self._best_fitness]
        
        for _ in range(self.config.n_iterations):
            for i in range(pop_size):
                indices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = self.rng.choice(indices, 3, replace=False)
                
                mutant = self._best_solution + self.de_config.mutation_factor * (
                    self._population[r1] - self._population[r2]
                )
                mutant = np.clip(mutant, self.config.bounds[0], self.config.bounds[1])
                
                trial = self._population[i].copy()
                j_rand = self.rng.integers(0, self.dim)
                
                for j in range(self.dim):
                    if j == j_rand or self.rng.random() < self.de_config.crossover_prob:
                        trial[j] = mutant[j]
                
                trial_fitness = fitness_fn(trial)
                
                if trial_fitness <= fitness[i]:
                    self._population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < self._best_fitness:
                        self._best_solution = trial.copy()
                        self._best_fitness = trial_fitness
            
            self._convergence_history.append(self._best_fitness)
            
            if (self.config.early_stop_threshold is not None and 
                self._best_fitness < self.config.early_stop_threshold):
                break
        
        return self._best_solution.copy(), self._best_fitness