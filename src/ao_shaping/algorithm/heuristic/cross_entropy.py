"""Cross-Entropy Method (CEM) optimization algorithm.

Population-based optimization that uses sampling from a Gaussian distribution.

Example:
    >>> from ao_shaping.algorithm.heuristic.cross_entropy import CrossEntropyMethod
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
import numpy.typing as npt

from ao_shaping.algorithm.heuristic.gm import NumpyPopulationGM, guided_mutation
from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


@dataclass
class CEMConfig:
    """Cross-Entropy Method specific configuration."""

    pop_size: int = 50
    elite_fraction: float = 0.2
    initial_std: float = 5.0


class CrossEntropyMethod(NumpyPopulationGM, HeuristicOptimizer):
    """Cross-Entropy Method optimizer.

    Uses Gaussian sampling with parameters updated based on elite samples.
    """

    _registry_key = OptimizerType.CROSS_ENTROPY

    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        cem_config: CEMConfig | None = None,
        random_state: np.random.Generator | None = None,
        n_iterations: int = 1000,
        bounds: tuple[float, float] = (-10.0, 10.0),
        seed: int | None = None,
        pop_size: int = 50,
        elite_fraction: float = 0.2,
        initial_std: float = 5.0,
    ):
        """Initialize CEM optimizer."""
        if config is None:
            config = OptimizerConfig(
                n_iterations=n_iterations, bounds=bounds, seed=seed
            )
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
        # Generation state lives on self so @guided_mutation can read it.
        self._population: npt.NDArray[np.float64] = np.empty((0, self.dim))
        self._fitness_vals: npt.NDArray[np.float64] = np.empty(0)
        self._current_iter: int = 0

    @guided_mutation(merge="replace_worst")
    def _sample_generation(self) -> npt.NDArray[np.float64]:
        """Draw one generation from the current Gaussian, clipped into bounds.

        The decorator reads ``self._population`` / ``self._fitness_vals`` from the
        *previous* generation, which is deliberate: CEM has no selection pressure
        inside a generation, so guidance by the previous generation's scores is
        exactly the feedback loop the GM paper describes. On the very first
        generation those are empty, so the decorator declines to inject anything.

        Returns:
            The sampled generation, shape ``(pop_size, dim)``.
        """
        samples = self.rng.normal(
            self._mean, self._std, (self.cem_config.pop_size, self.dim)
        )
        return np.clip(samples, self.config.bounds[0], self.config.bounds[1])

    def optimize(
        self,
        fitness_fn: callable,
        init_x: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Run Cross-Entropy Method optimization."""
        if init_x is not None:
            self._mean = init_x.copy()

        n_elite = max(1, int(self.cem_config.pop_size * self.cem_config.elite_fraction))

        for iteration in range(1, self.config.n_iterations + 1):
            self._current_iter = iteration
            # Sample (the decorator may splice in guided-mutation offspring).
            evolved = self._sample_generation()
            self._population = np.asarray(evolved, dtype=np.float64)

            fitness = np.array([fitness_fn(s) for s in self._population])

            elite_idx = np.argsort(fitness)[:n_elite]
            elite_samples = self._population[elite_idx]

            self._mean = np.mean(elite_samples, axis=0)
            self._std = np.std(elite_samples, axis=0) + 1e-6

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < (
                self._best_fitness if self._best_fitness is not None else float("inf")
            ):
                self._best_solution = self._population[best_idx].copy()
                self._best_fitness = fitness[best_idx]

            if self._best_fitness is None:
                self._best_solution = self._population[best_idx].copy()
                self._best_fitness = fitness[best_idx]

            self._fitness_vals = fitness
            self._convergence_history.append(self._best_fitness)

            if (
                self.config.early_stop_threshold is not None
                and self._best_fitness < self.config.early_stop_threshold
            ):
                break

        return self._best_solution.copy(), self._best_fitness
