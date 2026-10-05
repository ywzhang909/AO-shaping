"""交叉熵方法 (CEM) 优化算法。

基于种群的优化, 从高斯分布中采样。

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
    """交叉熵方法专用配置。"""

    pop_size: int = 50
    elite_fraction: float = 0.2
    initial_std: float = 5.0


class CrossEntropyMethod(NumpyPopulationGM, HeuristicOptimizer):
    """交叉熵方法优化器。

    用高斯采样, 并依据精英样本来更新参数。
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
        """初始化 CEM 优化器。"""
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
        # 每一代的状态放在 self 上, 供 @guided_mutation 读取。
        self._population: npt.NDArray[np.float64] = np.empty((0, self.dim))
        self._fitness_vals: npt.NDArray[np.float64] = np.empty(0)
        self._current_iter: int = 0

    @guided_mutation(merge="replace_worst")
    def _sample_generation(self) -> npt.NDArray[np.float64]:
        """从当前高斯分布采出一代, 并裁剪到边界内。

        该装饰器读取的是*上一*代的 ``self._population`` / ``self._fitness_vals``,
        这是刻意为之: CEM 在一代之内没有选择压力, 所以由上一代的分数来引导,
        恰好就是 GM 论文所描述的那个反馈回路。在最初一代它们是空的, 于是装饰器
        拒绝注入任何东西。

        Returns:
            采样出的一代, 形状 ``(pop_size, dim)``。
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
        """执行交叉熵方法优化。"""
        if init_x is not None:
            self._mean = init_x.copy()

        n_elite = max(1, int(self.cem_config.pop_size * self.cem_config.elite_fraction))

        for iteration in range(1, self.config.n_iterations + 1):
            self._current_iter = iteration
            # 采样 (装饰器可能会把引导式变异的后代拼接进来)。
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
