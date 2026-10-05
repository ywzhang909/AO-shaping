"""模拟退火 (Simulated Annealing, SA) 优化模块。

用于连续优化的标准 SA 算法。

主要特性:
- 多种温度调度方案
- Metropolis 接受准则
- 邻解生成
- 自适应降温

Example:
    >>> from ao_shaping.algorithm.heuristic.simulated_annealing import SimulatedAnnealing
    >>> import numpy as np
    >>>
    >>> def objective(x):
    ...     return np.sum(x ** 2)  # Sphere function
    >>>
    >>> sa = SimulatedAnnealing(
    ...     dim=5,
    ...     n_iterations=1000,
    ...     bounds=(-10, 10)
    ... )
    >>> best_x, best_f = sa.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


class FitnessFunction(Protocol):
    """适应度函数的 Protocol。"""

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        """评估适应度。"""
        ...


class TempSchedule(Enum):
    """温度调度类型。"""

    LINEAR = auto()
    EXPONENTIAL = auto()
    LOGARITHMIC = auto()
    COSINE = auto()


@dataclass
class SAParams:
    """模拟退火参数。"""

    n_iterations: int = 1000
    initial_temp: float = 100.0
    final_temp: float = 0.01
    schedule: TempSchedule = TempSchedule.EXPONENTIAL
    step_size: float = 0.5  # 生成邻解所用的标准差
    bounds: tuple[float, float] = (-10.0, 10.0)


@dataclass
class SAHistory:
    """SA 优化运行的历史记录。"""

    best_fitness: list[float] = field(default_factory=list)
    current_fitness: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)


class SimulatedAnnealing(HeuristicOptimizer):
    """模拟退火优化器。

    Attributes:
        dim: 问题维度。
        params: SA 参数。
        history: 优化历史。
    """

    _registry_key = OptimizerType.SA

    def __init__(
        self,
        dim: int,
        params: SAParams | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """初始化 SA 优化器。

        Args:
            dim: 优化问题的维度。
            params: SA 参数。None 表示使用默认的 SAParams。
            random_state: 用于复现的随机数生成器。
        """
        self.params = params if params is not None else SAParams()
        # 从 SAParams 派生 OptimizerConfig (iterations/bounds), 复用基类的 dim/rng 设置。
        config = OptimizerConfig(
            n_iterations=self.params.n_iterations,
            bounds=self.params.bounds,
        )
        super().__init__(dim, config, random_state)
        self.history = SAHistory()

    @classmethod
    def _construct(
        cls,
        dim: int,
        config: OptimizerConfig,
        random_state: np.random.Generator | None,
        **kwargs: Any,
    ) -> "SimulatedAnnealing":
        """由通用配置构造 SA, 并把配置转换为 SAParams。

        Args:
            dim: 优化问题的维度。
            config: 由 ``HeuristicOptimizer.create`` 构建的通用配置。
            random_state: 由 ``config.seed`` 导出的生成器, 或 None。
            **kwargs: 仅为与基类钩子保持签名兼容而接受;
                模拟退火不提供任何 ``create()`` 额外参数。

        Returns:
            构造好的 SimulatedAnnealing。
        """
        params = SAParams(
            n_iterations=config.n_iterations,
            bounds=config.bounds,
        )
        return cls(dim=dim, params=params, random_state=random_state)

    def _get_temperature(self, iteration: int) -> float:
        """获取当前迭代的温度。

        Args:
            iteration: 当前迭代轮次。

        Returns:
            当前温度。
        """
        t = iteration / self.params.n_iterations

        if self.params.schedule == TempSchedule.LINEAR:
            return self.params.initial_temp * (1 - t) + self.params.final_temp * t

        elif self.params.schedule == TempSchedule.EXPONENTIAL:
            return (
                self.params.initial_temp
                * (self.params.final_temp / self.params.initial_temp) ** t
            )

        elif self.params.schedule == TempSchedule.LOGARITHMIC:
            if t == 0:
                return self.params.initial_temp
            return self.params.initial_temp / (
                1 + t * (self.params.initial_temp / self.params.final_temp - 1)
            )

        elif self.params.schedule == TempSchedule.COSINE:
            return self.params.final_temp + 0.5 * (
                self.params.initial_temp - self.params.final_temp
            ) * (1 + np.cos(np.pi * t))

        return self.params.initial_temp

    def _generate_neighbor(self, current: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """生成邻解。

        Args:
            current: 当前解。

        Returns:
            邻解。
        """
        neighbor = current + self.rng.normal(0, self.params.step_size, self.dim)
        return np.clip(neighbor, self.params.bounds[0], self.params.bounds[1])

    def _acceptance_probability(
        self,
        current_fitness: float,
        new_fitness: float,
        temperature: float,
    ) -> float:
        """按 Metropolis 准则计算接受概率。

        Args:
            current_fitness: 当前解的适应度。
            new_fitness: 新解的适应度。
            temperature: 当前温度。

        Returns:
            接受概率。
        """
        if new_fitness < current_fitness:
            return 1.0

        if temperature <= 0:
            return 0.0

        delta = new_fitness - current_fitness
        return np.exp(-delta / temperature)

    def optimize(
        self,
        fitness_fn: FitnessFunction,
        init_x: npt.NDArray[np.float64] | None = None,
        early_stop_threshold: float | None = None,
        callback: Callable[[int, npt.NDArray[np.float64], float, float], None] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """运行模拟退火优化。

        Args:
            fitness_fn: 要最小化的适应度函数。
            init_x: 初始点。None 表示从随机点出发。
            early_stop_threshold: 最优适应度低于该阈值时提前停止。
            callback: 可选的回调函数, 每轮迭代后以
                      (iteration, current_position, current_fitness, temperature) 调用。

        Returns:
            (best_solution, best_fitness) 元组。
        """
        if init_x is not None:
            current = init_x.copy()
        else:
            current = self.rng.uniform(
                self.params.bounds[0], self.params.bounds[1], self.dim
            )

        current_fitness = fitness_fn(current)
        best = current.copy()
        best_fitness = current_fitness

        self.history.best_fitness.append(best_fitness)
        self.history.current_fitness.append(current_fitness)
        self.history.temperature.append(self.params.initial_temp)

        for iteration in range(1, self.params.n_iterations + 1):
            temperature = self._get_temperature(iteration)

            candidate = self._generate_neighbor(current)
            candidate_fitness = fitness_fn(candidate)

            if self.rng.random() < self._acceptance_probability(
                current_fitness, candidate_fitness, temperature
            ):
                current = candidate
                current_fitness = candidate_fitness

                if current_fitness < best_fitness:
                    best = current.copy()
                    best_fitness = current_fitness

            self.history.best_fitness.append(best_fitness)
            self.history.current_fitness.append(current_fitness)
            self.history.temperature.append(temperature)

            if callback is not None:
                callback(iteration, current.copy(), current_fitness, temperature)

            if early_stop_threshold is not None and best_fitness < early_stop_threshold:
                break

        return best.copy(), best_fitness

    @property
    def convergence_history(self) -> list[float]:
        """返回收敛历史 (每轮迭代的最优适应度)。"""
        return self.history.best_fitness


def minimize_sa(
    fitness_fn: FitnessFunction,
    dim: int,
    n_iterations: int = 1000,
    initial_temp: float = 100.0,
    final_temp: float = 0.01,
    schedule: TempSchedule = TempSchedule.EXPONENTIAL,
    bounds: tuple[float, float] = (-10.0, 10.0),
    init_x: npt.NDArray[np.float64] | None = None,
    early_stop_threshold: float | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    """SA 优化的便捷函数。

    Args:
        fitness_fn: 要最小化的适应度函数。
        dim: 问题维度。
        n_iterations: 迭代次数。
        initial_temp: 初始温度。
        final_temp: 最终温度。
        schedule: 温度调度方案。
        bounds: 搜索空间边界 (min, max)。
        init_x: 初始点。
        early_stop_threshold: 提前停止阈值。

    Returns:
        (best_solution, best_fitness) 元组。
    """
    params = SAParams(
        n_iterations=n_iterations,
        initial_temp=initial_temp,
        final_temp=final_temp,
        schedule=schedule,
        bounds=bounds,
    )
    sa = SimulatedAnnealing(dim=dim, params=params)
    return sa.optimize(
        fitness_fn=fitness_fn,
        init_x=init_x,
        early_stop_threshold=early_stop_threshold,
    )
