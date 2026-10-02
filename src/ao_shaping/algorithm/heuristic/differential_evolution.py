"""Differential Evolution (DE) optimization algorithm.

Global optimization algorithm that uses vector differences for mutation.

Example:
    >>> from ao_shaping.algorithm.heuristic.differential_evolution import DifferentialEvolution
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

from ao_shaping.algorithm.heuristic.gm import NumpyPopulationGM, guided_mutation
from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


@dataclass
class DEConfig:
    """Differential Evolution specific configuration."""

    pop_size: int = 30
    crossover_prob: float = 0.9
    mutation_factor: float = 0.8


class DifferentialEvolution(NumpyPopulationGM, HeuristicOptimizer):
    """Differential Evolution optimizer.

    Uses differential mutation: mutant = best + F * (r1 - r2)
    """

    _registry_key = OptimizerType.DIFFERENTIAL_EVOLUTION

    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        de_config: DEConfig | None = None,
        random_state: np.random.Generator | None = None,
        n_iterations: int = 1000,
        bounds: tuple[float, float] = (-10.0, 10.0),
        seed: int | None = None,
        pop_size: int = 30,
        crossover_prob: float = 0.9,
        mutation_factor: float = 0.8,
    ):
        """Initialize DE optimizer."""
        if config is None:
            config = OptimizerConfig(
                n_iterations=n_iterations, bounds=bounds, seed=seed
            )
        super().__init__(dim, config, random_state)

        if de_config is None:
            de_config = DEConfig(
                pop_size=pop_size,
                crossover_prob=crossover_prob,
                mutation_factor=mutation_factor,
            )

        self.de_config = de_config
        # Generation state lives on self so @guided_mutation can read it.
        self._population: np.ndarray = np.empty((0, self.dim))
        self._fitness_vals: np.ndarray = np.empty(0)
        self._current_iter: int = 0

    @guided_mutation(merge="replace_worst")
    def _evolve_generation(self, fitness_fn: callable) -> np.ndarray:
        """Run one full DE pass with greedy selection.

        Each target is crossed with a mutant built from the current best and two
        random population members; the trial replaces the target when it is no
        worse. The population is updated in place, so with GM disabled the
        returned array *is* ``self._population`` and the caller skips the
        re-scoring step entirely.

        Args:
            fitness_fn: Fitness function to minimize.

        Returns:
            The population after one pass, shape ``(pop_size, dim)``.
        """
        population = self._population
        fitness = self._fitness_vals
        pop_size = population.shape[0]
        low, high = self.config.bounds[0], self.config.bounds[1]

        for i in range(pop_size):
            indices = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = self.rng.choice(indices, 3, replace=False)

            mutant = self._best_solution + self.de_config.mutation_factor * (
                population[r1] - population[r2]
            )
            mutant = np.clip(mutant, low, high)

            trial = population[i].copy()
            j_rand = self.rng.integers(0, self.dim)

            for j in range(self.dim):
                if j == j_rand or self.rng.random() < self.de_config.crossover_prob:
                    trial[j] = mutant[j]

            trial_fitness = fitness_fn(trial)

            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                if trial_fitness < self._best_fitness:
                    self._best_solution = trial.copy()
                    self._best_fitness = trial_fitness

        return population

    def optimize(
        self,
        fitness_fn: callable,
        init_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run Differential Evolution optimization."""
        pop_size = max(10, self.de_config.pop_size)

        self._population = self.rng.uniform(
            self.config.bounds[0], self.config.bounds[1], (pop_size, self.dim)
        )

        if init_x is not None:
            self._population[0] = init_x.copy()

        self._fitness_vals = np.array(
            [fitness_fn(ind) for ind in self._population], dtype=np.float64
        )

        best_idx = np.argmin(self._fitness_vals)
        self._best_solution = self._population[best_idx].copy()
        self._best_fitness = self._fitness_vals[best_idx]
        self._convergence_history = [self._best_fitness]

        for iteration in range(1, self.config.n_iterations + 1):
            self._current_iter = iteration
            evolved = self._evolve_generation(fitness_fn)

            if evolved is not self._population:
                # GM overwrote the worst rows, so their cached scores are stale.
                self._population = np.asarray(evolved, dtype=np.float64)
                self._fitness_vals = np.array(
                    [fitness_fn(ind) for ind in self._population], dtype=np.float64
                )
                gm_best_idx = int(np.argmin(self._fitness_vals))
                if self._fitness_vals[gm_best_idx] < self._best_fitness:
                    self._best_solution = self._population[gm_best_idx].copy()
                    self._best_fitness = float(self._fitness_vals[gm_best_idx])

            self._convergence_history.append(self._best_fitness)

            if (
                self.config.early_stop_threshold is not None
                and self._best_fitness < self.config.early_stop_threshold
            ):
                break

        return self._best_solution.copy(), self._best_fitness
