"""Hill Climbing optimization algorithm.

Simple local search optimization that iteratively improves solution by local moves.

Example:
    >>> from ao_shaping.algorithm.hill_climbing import HillClimbing
    >>> import numpy as np
    >>> 
    >>> def objective(x):
    ...     return np.sum(x ** 2)
    >>> 
    >>> hc = HillClimbing(dim=5, n_iterations=1000, step_size=0.1)
    >>> best_x, best_f = hc.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heuristic_base import HeuristicOptimizer, OptimizerConfig


@dataclass
class HCConfig:
    """Hill Climbing specific configuration."""
    step_size: float = 0.1
    neighbor_std: float = 0.1


class HillClimbing(HeuristicOptimizer):
    """Hill Climbing optimizer.
    
    Attributes:
        dim: Dimension of the problem.
        config: Configuration.
        hc_config: Hill Climbing specific config.
    """
    
    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        hc_config: HCConfig | None = None,
        random_state: np.random.Generator | None = None,
        step_size: float = 0.1,
        neighbor_std: float = 0.1,
    ):
        """Initialize Hill Climbing optimizer."""
        super().__init__(dim, config, random_state)
        
        if hc_config is None:
            hc_config = HCConfig(step_size=step_size, neighbor_std=neighbor_std)
        
        self.hc_config = hc_config
    
    def _generate_neighbor(self, current: np.ndarray) -> np.ndarray:
        """Generate neighbor solution."""
        neighbor = current + self.rng.normal(0, self.hc_config.neighbor_std, self.dim)
        return np.clip(neighbor, self.config.bounds[0], self.config.bounds[1])
    
    def optimize(
        self,
        fitness_fn: callable,
        init_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run Hill Climbing optimization."""
        if init_x is not None:
            current = init_x.copy()
        else:
            current = self.rng.uniform(
                self.config.bounds[0],
                self.config.bounds[1],
                self.dim
            )
        
        current_fitness = fitness_fn(current)
        self._best_solution = current.copy()
        self._best_fitness = current_fitness
        self._convergence_history = [current_fitness]
        
        for _ in range(self.config.n_iterations):
            candidate = self._generate_neighbor(current)
            candidate_fitness = fitness_fn(candidate)
            
            if candidate_fitness < current_fitness:
                current = candidate
                current_fitness = candidate_fitness
                
                if current_fitness < self._best_fitness:
                    self._best_solution = current.copy()
                    self._best_fitness = current_fitness
            
            self._convergence_history.append(self._best_fitness)
            
            if (self.config.early_stop_threshold is not None and 
                self._best_fitness < self.config.early_stop_threshold):
                break
        
        return self._best_solution.copy(), self._best_fitness