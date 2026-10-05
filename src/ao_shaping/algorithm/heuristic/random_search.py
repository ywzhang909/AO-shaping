"""随机搜索优化算法。

采样随机解并保留其中最好的。

Example:
    >>> from ao_shaping.algorithm.heuristic.random_search import RandomSearch
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
import numpy.typing as npt

from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


class RandomSearch(HeuristicOptimizer):
    """随机搜索优化器。

    纯随机搜索, 在搜索空间里均匀采样。
    """

    _registry_key = OptimizerType.RANDOM_SEARCH

    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        random_state: np.random.Generator | None = None,
        n_iterations: int = 1000,
        bounds: tuple[float, float] = (-10.0, 10.0),
        seed: int | None = None,
    ):
        """初始化随机搜索优化器。"""
        if config is None:
            config = OptimizerConfig(
                n_iterations=n_iterations, bounds=bounds, seed=seed
            )
        super().__init__(dim, config, random_state)

    def optimize(
        self,
        fitness_fn: callable,
        init_x: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """执行随机搜索优化。"""
        if init_x is not None:
            self._best_solution = init_x.copy()
            self._best_fitness = fitness_fn(init_x)
        else:
            self._best_solution = self.rng.uniform(
                self.config.bounds[0], self.config.bounds[1], self.dim
            )
            self._best_fitness = fitness_fn(self._best_solution)

        self._convergence_history = [self._best_fitness]

        for _ in range(self.config.n_iterations):
            candidate = self.rng.uniform(
                self.config.bounds[0], self.config.bounds[1], self.dim
            )
            candidate_fitness = fitness_fn(candidate)

            if candidate_fitness < self._best_fitness:
                self._best_solution = candidate.copy()
                self._best_fitness = candidate_fitness

            self._convergence_history.append(self._best_fitness)

            if (
                self.config.early_stop_threshold is not None
                and self._best_fitness < self.config.early_stop_threshold
            ):
                break

        return self._best_solution.copy(), self._best_fitness
