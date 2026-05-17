"""Cross-Entropy Method (CEM) optimization algorithm.

Population-based optimization that uses sampling from a Gaussian distribution.

Example:
    >>> from ao_shaping.algorithm.cross_entropy import CrossEntropyMethod
    >>> import numpy as np
    >>> 
    >>> def objective(x):
    ...     return np.sum(x ** 2)
    >>> 
    >>> cem = CrossEntropyMethod(dim=5, n_iterations=100)
    >>> best_x, best_f = cem.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .heuristic_base import HeuristicOptimizer, OptimizerConfig


@dataclass
class CEMConfig:
    """Cross-Entropy Method specific configuration."""
    pop_size: int = 50
    elite_fraction: float = 0.2
    initial_std: float = 5.0


class CrossEntropyMethod(HeuristicOptimizer):
    """Cross-Entropy Method optimizer.
    
    Uses Gaussian sampling with parameters updated based on elite samples.
    """
    
    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        cem_config: CEMConfig | None = None,
        random_state: np.random.Generator | None = None,
        pop_size: int = 50,
        elite_fraction: float = 0.2,
        initial_std: float = 5.0,
    ):
        """Initialize CEM optimizer."""
        super().__init__(dim, config, random_state)
        
        if cem_config is None:
            cem_config = CEMConfig(
                pop_size=pop_size,
                elite_fraction=elite_fraction,
                initial_std=initial_std,
            )
        
        self.cem_config = cem_config
        self._mean = np.zeros(dim)
        self._std = np.full(dim, self.cem_config.initial_std)
    
    def optimize(
        self,
        fitness_fn: callable,
        init_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run Cross-Entropy Method optimization."""
        if init_x is not None:
            self._mean = init_x.copy()
        
        n_elite = max(1, int(self.cem_config.pop_size * self.cem_config.elite_fraction))
        
        for _ in range(self.config.n_iterations):
            samples = self.rng.normal(
                self._mean,
                self._std,
                (self.cem_config.pop_size, self.dim)
            )
            samples = np.clip(samples, self.config.bounds[0], self.config.bounds[1])
            
            fitness = np.array([fitness_fn(s) for s in samples])
            
            elite_idx = np.argsort(fitness)[:n_elite]
            elite_samples = samples[elite_idx]
            
            self._mean = np.mean(elite_samples, axis=0)
            self._std = np.std(elite_samples, axis=0) + 1e-6
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < (self._best_fitness if self._best_fitness is not None else float('inf')):
                self._best_solution = samples[best_idx].copy()
                self._best_fitness = fitness[best_idx]
            
            if self._best_fitness is None:
                self._best_solution = samples[best_idx].copy()
                self._best_fitness = fitness[best_idx]
            
            self._convergence_history.append(self._best_fitness)
            
            if (self.config.early_stop_threshold is not None and 
                self._best_fitness < self.config.early_stop_threshold):
                break
        
        return self._best_solution.copy(), self._best_fitness