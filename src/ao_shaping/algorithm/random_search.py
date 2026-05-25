"""Random Search optimization algorithm.

Samples random solutions and keeps the best found.

Example:
    >>> from ao_shaping.algorithm.random_search import RandomSearch
    >>> import numpy as np
    >>> 
    >>> def objective(x):
    ...     return np.sum(x ** 2)
    >>> 
    >>> rs = RandomSearch(dim=5, n_iterations=1000)
    >>> best_x, best_f = rs.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.algorithm.heuristic_base import HeuristicOptimizer, OptimizerConfig


class RandomSearch(HeuristicOptimizer):
    """Random Search optimizer.
    
    Pure random search that samples uniformly from the search space.
    """
    
    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """Initialize Random Search optimizer."""
        super().__init__(dim, config, random_state)
    
    def optimize(
        self,
        fitness_fn: callable,
        init_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run Random Search optimization."""
        if init_x is not None:
            self._best_solution = init_x.copy()
            self._best_fitness = fitness_fn(init_x)
        else:
            self._best_solution = self.rng.uniform(
                self.config.bounds[0],
                self.config.bounds[1],
                self.dim
            )
            self._best_fitness = fitness_fn(self._best_solution)
        
        self._convergence_history = [self._best_fitness]
        
        for _ in range(self.config.n_iterations):
            candidate = self.rng.uniform(
                self.config.bounds[0],
                self.config.bounds[1],
                self.dim
            )
            candidate_fitness = fitness_fn(candidate)
            
            if candidate_fitness < self._best_fitness:
                self._best_solution = candidate.copy()
                self._best_fitness = candidate_fitness
            
            self._convergence_history.append(self._best_fitness)
            
            if (self.config.early_stop_threshold is not None and 
                self._best_fitness < self.config.early_stop_threshold):
                break
        
        return self._best_solution.copy(), self._best_fitness