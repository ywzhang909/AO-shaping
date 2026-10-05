"""遗传算法优化模块。

本模块为通用优化提供纯遗传算法算子。
把 ga_zernike.py 里的核心 GA 组件抽出来以便复用。

主要特性:
- 锦标赛选择
- 混合交叉 (BLX-alpha)
- 高斯变异
- 精英保留

Example:
    >>> from ao_shaping.algorithm.heuristic.ga import GeneticAlgorithm
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
from typing import Any, Callable, Protocol

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.heuristic.gm import NumpyPopulationGM, guided_mutation
from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


class FitnessFunction(Protocol):
    """适应度函数的 Protocol。"""

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        """评估适应度。

        Args:
            x: 待评估的个体。

        Returns:
            适应度值 (做最小化时越小越好)。
        """
        ...


@dataclass
class GAParams:
    """遗传算法参数。"""

    pop_size: int = 50
    n_generations: int = 2000
    crossover_prob: float = 0.7
    mutation_prob: float = 0.15
    tournament_size: int = 3
    elite_count: int = 2
    bounds: tuple[float, float] = (-50.0, 50.0)
    alpha: float = 0.5  # BLX-alpha 扩张系数
    mutation_sigma: float = 5.0  # 高斯变异的 sigma


def tournament_selection(
    population: npt.NDArray[np.float64],
    fitness: npt.NDArray[np.float64],
    tournament_size: int,
    random_state: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """用锦标赛选择挑出一个个体。

    Args:
        population: 形状 (pop_size, dim) 的种群数组。
        fitness: 形状 (pop_size,) 的适应度值数组 (越小越好)。
        tournament_size: 参加锦标赛的个体数。
        random_state: 用于可复现性的随机数生成器。

    Returns:
        选中的个体, 形状 (dim,) 的数组。
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    pop_size = len(population)
    contestants = rng.choice(pop_size, tournament_size, replace=False)
    # 做最小化时, 选适应度最小的那个
    best_idx = contestants[np.argmin(fitness[contestants])]
    return population[best_idx].copy()


def blend_crossover(
    parent1: npt.NDArray[np.float64],
    parent2: npt.NDArray[np.float64],
    alpha: float = 0.5,
    random_state: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """对两个亲本做混合交叉 (BLX-alpha)。

    在按系数 alpha 扩张过的范围内, 通过在两个亲本之间插值/外插生成两个子代。

    Args:
        parent1: 形状 (dim,) 的第一个亲本数组。
        parent2: 形状 (dim,) 的第二个亲本数组。
        alpha: 搜索范围的扩张系数。
        random_state: 用于可复现性的随机数生成器。

    Returns:
        两个子代数组组成的二元组。
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    n = len(parent1)
    # 计算两个亲本之间的范围
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    I = c_max - c_min

    # 扩张该范围
    lower = c_min - alpha * I
    upper = c_max + alpha * I

    # 在扩张后的范围内均匀生成子代
    child1 = lower + rng.random(n) * (upper - lower)
    child2 = lower + rng.random(n) * (upper - lower)

    return child1, child2


def gaussian_mutation(
    individual: npt.NDArray[np.float64],
    mutation_rate: float,
    sigma: float = 5.0,
    bounds: tuple[float, float] = (-50.0, 50.0),
    random_state: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """对一个个体施加高斯变异。

    每个基因都有 mutation_rate 的概率被变异。被变异的基因由一个均值为 0、sigma
    为给定值的高斯分布扰动。取值会被裁剪到指定边界。

    Args:
        individual: 待变异的个体, 形状 (dim,)。
        mutation_rate: 每个基因被变异的概率。
        sigma: 高斯扰动的标准差。
        bounds: 用于裁剪的 (最小, 最大) 边界二元组。
        random_state: 用于可复现性的随机数生成器。

    Returns:
        变异后的个体。
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
    """GA 优化运行的历史。"""

    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    best_individual: npt.NDArray[np.float64] | None = None


class GeneticAlgorithm(NumpyPopulationGM, HeuristicOptimizer):
    """用于连续优化的遗传算法优化器。

    Attributes:
        dim: 问题的维度。
        params: GA 参数。
        history: 优化历史。
    """

    _registry_key = OptimizerType.GA

    def __init__(
        self,
        dim: int,
        params: GAParams | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """初始化遗传算法。

        Args:
            dim: 优化问题的维度。
            params: GA 参数。为 None 时用默认的 GAParams。
            random_state: 用于可复现性的随机数生成器。
        """
        self.params = params if params is not None else GAParams()
        # 从 GAParams 派生 OptimizerConfig (iterations/bounds), 复用基类的 dim/rng 设置。
        config = OptimizerConfig(
            n_iterations=self.params.n_generations,
            bounds=self.params.bounds,
        )
        super().__init__(dim, config, random_state)
        self.history = GAHistory()
        # 每一代的状态放在 self 上, 供 @guided_mutation 读取。
        self._population: npt.NDArray[np.float64] = np.empty((0, self.dim))
        self._fitness_vals: npt.NDArray[np.float64] = np.empty(0)
        self._current_iter: int = 0

    @classmethod
    def _construct(
        cls,
        dim: int,
        config: OptimizerConfig,
        random_state: np.random.Generator | None,
        **kwargs: Any,
    ) -> "GeneticAlgorithm":
        """由通用配置构造一个 GA, 并把它翻译成 GAParams。

        Args:
            dim: 优化问题的维度。
            config: ``HeuristicOptimizer.create`` 构造出的通用配置。
            random_state: 由 ``config.seed`` 派生的生成器, 或 None。
            **kwargs: 只有 ``pop_size`` 会被采纳; GA 不接受其他额外参数。

        Returns:
            构造出的 GeneticAlgorithm。
        """
        params = GAParams(
            pop_size=kwargs.get("pop_size", 30),
            n_generations=config.n_iterations,
            bounds=config.bounds,
        )
        return cls(dim=dim, params=params, random_state=random_state)

    def _initialize_population(self, init_x: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """初始化种群。

        Args:
            init_x: 要纳入种群的初始点。为 None 时从随机点开始。

        Returns:
            形状 (pop_size, dim) 的初始种群数组。
        """
        pop = self.rng.uniform(
            self.params.bounds[0],
            self.params.bounds[1],
            (self.params.pop_size, self.dim),
        )
        # 给了初始点就把它纳入
        if init_x is not None:
            pop[0] = init_x.copy()
        return pop

    def _evaluate_population(
        self, pop: npt.NDArray[np.float64], fitness_fn: FitnessFunction
    ) -> npt.NDArray[np.float64]:
        """评估整个种群的适应度。

        Args:
            pop: 种群数组。
            fitness_fn: 适应度函数。

        Returns:
            形状 (pop_size,) 的适应度值数组。
        """
        fitness = np.array([fitness_fn(ind) for ind in pop])
        return fitness

    @guided_mutation()
    def _evolve_generation(self) -> npt.NDArray[np.float64]:
        """Build the next generation: elitism + tournament + crossover + mutation.

        Returns a population of exactly ``pop_size`` rows. When GM is enabled,
        ``gm_offspring_fraction`` of those slots are left for the guided-mutation
        offspring the decorator appends, so the total size is unchanged.

        Returns:
            The next population, shape ``(pop_size - n_gm, dim)``.
        """
        population = self._population
        fitness = self._fitness_vals
        pop_size = self.params.pop_size
        target = max(self.params.elite_count, pop_size - self._gm_offspring())

        # Elitism: preserve top elite_count individuals.
        new_population = [
            population[idx].copy()
            for idx in np.argsort(fitness)[: self.params.elite_count]
        ]

        # Generate the remaining offspring through selection, crossover, mutation.
        while len(new_population) < target:
            parent1 = tournament_selection(
                population, fitness, self.params.tournament_size, self.rng
            )
            parent2 = tournament_selection(
                population, fitness, self.params.tournament_size, self.rng
            )

            if self.rng.random() < self.params.crossover_prob:
                child1, child2 = blend_crossover(
                    parent1, parent2, self.params.alpha, self.rng
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = gaussian_mutation(
                child1,
                self.params.mutation_prob,
                self.params.mutation_sigma,
                self.params.bounds,
                self.rng,
            )
            child2 = gaussian_mutation(
                child2,
                self.params.mutation_prob,
                self.params.mutation_sigma,
                self.params.bounds,
                self.rng,
            )

            new_population.append(child1)
            if len(new_population) < target:
                new_population.append(child2)

        return np.array(new_population[:pop_size])

    def optimize(
        self,
        fitness_fn: FitnessFunction,
        init_x: npt.NDArray[np.float64] | None = None,
        early_stop_threshold: float | None = None,
        callback: Callable[[int, npt.NDArray[np.float64], float], None] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
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
        self._population = self._initialize_population(init_x)

        # Evaluate initial population
        self._fitness_vals = self._evaluate_population(self._population, fitness_fn)

        # Find best in initial population
        best_idx = np.argmin(self._fitness_vals)
        best_fitness = self._fitness_vals[best_idx]
        best_individual = self._population[best_idx].copy()

        # Record history
        self.history.best_fitness.append(best_fitness)
        self.history.mean_fitness.append(np.mean(self._fitness_vals))

        # Main GA loop
        for gen in range(1, self.params.n_generations + 1):
            self._current_iter = gen - 1

            # Evolve (the decorator may splice in guided-mutation offspring).
            evolved = self._evolve_generation()
            self._population = np.asarray(evolved, dtype=np.float64)[
                : self.params.pop_size
            ]

            # Evaluate new population
            self._fitness_vals = self._evaluate_population(
                self._population, fitness_fn
            )

            # Find best individual in this generation
            current_best_idx = np.argmin(self._fitness_vals)
            current_best_fitness = self._fitness_vals[current_best_idx]
            current_best_individual = self._population[current_best_idx].copy()

            # Update global best
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()

            # Record history
            self.history.best_fitness.append(best_fitness)
            self.history.mean_fitness.append(np.mean(self._fitness_vals))

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
    init_x: npt.NDArray[np.float64] | None = None,
    early_stop_threshold: float | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
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
