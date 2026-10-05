"""粒子群优化 (Particle Swarm Optimization, PSO) 模块。

用于连续优化的标准 PSO 算法。

主要特性:
- 提供动量的惯性权重
- 认知分量 (个体最优)
- 社会分量 (全局最优)
- 速度限幅
- 位置边界

Example:
    >>> from ao_shaping.algorithm.heuristic.pso import ParticleSwarmOptimizer
    >>> import numpy as np
    >>>
    >>> def objective(x):
    ...     return np.sum(x ** 2)  # Sphere function
    >>>
    >>> pso = ParticleSwarmOptimizer(
    ...     dim=5,
    ...     n_particles=30,
    ...     n_iterations=100,
    ...     bounds=(-10, 10)
    ... )
    >>> best_x, best_f = pso.optimize(objective)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.heuristic.gm import GMOptimizerMixin, guided_mutation
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


@dataclass
class PSOParams:
    """PSO 参数。"""

    n_particles: int = 30
    n_iterations: int = 1000
    w: float = 0.729  # 惯性权重
    c1: float = 1.49  # 认知系数
    c2: float = 1.49  # 社会系数
    v_max: float = 2.0  # 最大速度
    bounds: tuple[float, float] = (-10.0, 10.0)


@dataclass
class PSOHistory:
    """PSO 优化运行的历史记录。"""

    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)


class Particle:
    """PSO 中的单个粒子。"""

    def __init__(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        fitness: float,
    ):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.best_position = position.copy()
        self.best_fitness = fitness


class ParticleSwarmOptimizer(GMOptimizerMixin, HeuristicOptimizer):
    """粒子群优化器。

    Attributes:
        dim: 问题维度。
        params: PSO 参数。
        history: 优化历史。
    """

    _registry_key = OptimizerType.PSO

    def __init__(
        self,
        dim: int,
        params: PSOParams | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """初始化 PSO 优化器。

        Args:
            dim: 优化问题的维度。
            params: PSO 参数。None 表示使用默认的 PSOParams。
            random_state: 用于复现的随机数生成器。
        """
        self.params = params if params is not None else PSOParams()
        # 从 PSOParams 派生 OptimizerConfig (iterations/bounds), 复用基类的 dim/rng 设置。
        config = OptimizerConfig(
            n_iterations=self.params.n_iterations,
            bounds=self.params.bounds,
        )
        super().__init__(dim, config, random_state)
        self.history = PSOHistory()
        self.particles: list[Particle] = []
        self.global_best_position: npt.NDArray[np.float64] | None = None
        self.global_best_fitness: float = float("inf")
        self._current_iter: int = 0

    @classmethod
    def _construct(
        cls,
        dim: int,
        config: OptimizerConfig,
        random_state: np.random.Generator | None,
        **kwargs: Any,
    ) -> "ParticleSwarmOptimizer":
        """由通用配置构造 PSO, 并把配置转换为 PSOParams。

        Args:
            dim: 优化问题的维度。
            config: 由 ``HeuristicOptimizer.create`` 构建的通用配置。
            random_state: 由 ``config.seed`` 导出的生成器, 或 None。
            **kwargs: 只接受 ``n_particles``; PSO 不接受其他额外参数。

        Returns:
            构造好的 ParticleSwarmOptimizer。
        """
        params = PSOParams(
            n_particles=kwargs.get("n_particles", 30),
            n_iterations=config.n_iterations,
            bounds=config.bounds,
        )
        return cls(dim=dim, params=params, random_state=random_state)

    def _reset(self) -> None:
        """重置优化器状态。"""
        self.global_best_position = None
        self.global_best_fitness = float("inf")

    def _initialize_particles(self, init_x: npt.NDArray[np.float64] | None = None) -> list[Particle]:
        """初始化粒子群。

        Args:
            init_x: 要纳入粒子群的初始点。

        Returns:
            粒子列表。
        """
        particles = []

        for i in range(self.params.n_particles):
            if init_x is not None and i == 0:
                position = init_x.copy()
            else:
                position = self.rng.uniform(
                    self.params.bounds[0], self.params.bounds[1], self.dim
                )

            velocity = self.rng.uniform(-self.params.v_max, self.params.v_max, self.dim)

            particles.append(
                Particle(position=position, velocity=velocity, fitness=float("inf"))
            )

        return particles

    def _evaluate_particles(
        self,
        fitness_fn: FitnessFunction,
    ) -> None:
        """评估所有粒子。

        Args:
            fitness_fn: 适应度函数。
        """
        for p in self.particles:
            p.fitness = fitness_fn(p.position)

            if p.fitness < p.best_fitness:
                p.best_position = p.position.copy()
                p.best_fitness = p.fitness

                if p.fitness < self.global_best_fitness:
                    self.global_best_position = p.position.copy()
                    self.global_best_fitness = p.fitness

    def _update_velocities(self) -> None:
        """更新所有粒子的速度。"""
        r1 = self.rng.random((self.params.n_particles, self.dim))
        r2 = self.rng.random((self.params.n_particles, self.dim))

        for i, p in enumerate(self.particles):
            cognitive = self.params.c1 * r1[i] * (p.best_position - p.position)
            social = self.params.c2 * r2[i] * (self.global_best_position - p.position)

            p.velocity = self.params.w * p.velocity + cognitive + social

            p.velocity = np.clip(p.velocity, -self.params.v_max, self.params.v_max)

    @guided_mutation(merge="replace_worst")
    def _advance_positions(self) -> npt.NDArray[np.float64]:
        """让每个粒子前进一步, 并把粒子群以数组形式返回。

        装饰器通过 :class:`GMOptimizerMixin` 契约读取粒子群, 再经
        :meth:`_gm_commit` 提交任何引导变异的子代。返回数组 (而不是 ``None``)
        正是同一个装饰器既能服务 PSO、又能服务以数组为后端的优化器的原因。

        Returns:
            粒子位置, 形状 ``(n_particles, dim)``。
        """
        for p in self.particles:
            p.position = p.position + p.velocity
            p.position = np.clip(
                p.position, self.params.bounds[0], self.params.bounds[1]
            )
        return self._gm_population()

    # ------------------------------------------------------------------
    # GMOptimizerMixin 契约 -- PSO 保存的是 list[Particle], 因此每个钩子
    # 都需要在 (n, dim) 数组与该列表之间来回投影。
    # ------------------------------------------------------------------
    def _gm_population(self) -> npt.NDArray[np.float64]:
        return np.array([p.position for p in self.particles])

    def _gm_fitness(self) -> npt.NDArray[np.float64]:
        return np.array([p.fitness for p in self.particles])

    def _gm_iteration(self) -> int:
        return self._current_iter

    def _gm_offspring(self) -> int:
        if not self.use_gm or self._gm_operator is None:
            return 0
        if not self.particles:
            return 0
        return int(max(1, round(len(self.particles) * self.gm_offspring_fraction)))

    def _gm_bounds(self) -> tuple[float, float]:
        return self.params.bounds

    def _gm_commit(self, population: npt.NDArray[np.float64]) -> None:
        """把合并后的粒子群写回, 并把被移动粒子的速度清零。

        注入的粒子速度为零, 免得粒子群立刻把新候选飞走; 优化器自己移动的粒子
        保留原速度, 从而保住 PSO 的动量。被注入的粒子由紧随其后的
        ``_evaluate_particles`` 调用打分。
        """
        for particle, position in zip(self.particles, population):
            if not np.array_equal(particle.position, position):
                particle.position = np.array(position, dtype=np.float64)
                particle.velocity = np.zeros_like(particle.velocity)

    def optimize(
        self,
        fitness_fn: FitnessFunction,
        init_x: npt.NDArray[np.float64] | None = None,
        early_stop_threshold: float | None = None,
        callback: Callable[[int, npt.NDArray[np.float64], float], None] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """运行 PSO 优化。

        Args:
            fitness_fn: 要最小化的适应度函数。
            init_x: 要纳入粒子群的初始点。
            early_stop_threshold: 最优适应度低于该阈值时提前停止。
            callback: 可选的回调函数, 每轮迭代后以
                      (iteration, best_position, best_fitness) 调用。

        Returns:
            (best_solution, best_fitness) 元组。
        """
        self.particles = self._initialize_particles(init_x)

        self._evaluate_particles(fitness_fn)

        self.history.best_fitness.append(self.global_best_fitness)
        self.history.mean_fitness.append(np.mean([p.fitness for p in self.particles]))

        for iteration in range(1, self.params.n_iterations + 1):
            self._current_iter = iteration
            self._update_velocities()
            self._advance_positions()
            self._evaluate_particles(fitness_fn)

            self.history.best_fitness.append(self.global_best_fitness)
            self.history.mean_fitness.append(
                np.mean([p.fitness for p in self.particles])
            )

            if callback is not None:
                assert self.global_best_position is not None
                callback(
                    iteration,
                    self.global_best_position.copy(),
                    self.global_best_fitness,
                )

            if (
                early_stop_threshold is not None
                and self.global_best_fitness < early_stop_threshold
            ):
                break

        assert self.global_best_position is not None
        return self.global_best_position.copy(), self.global_best_fitness

    @property
    def convergence_history(self) -> list[float]:
        """返回收敛历史 (每轮迭代的最优适应度)。"""
        return self.history.best_fitness


def minimize_pso(
    fitness_fn: FitnessFunction,
    dim: int,
    n_particles: int = 30,
    n_iterations: int = 1000,
    bounds: tuple[float, float] = (-10.0, 10.0),
    init_x: npt.NDArray[np.float64] | None = None,
    early_stop_threshold: float | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    """PSO 优化的便捷函数。

    Args:
        fitness_fn: 要最小化的适应度函数。
        dim: 问题维度。
        n_particles: 粒子数量。
        n_iterations: 迭代次数。
        bounds: 搜索空间边界 (min, max)。
        init_x: 初始点。
        early_stop_threshold: 提前停止阈值。

    Returns:
        (best_solution, best_fitness) 元组。
    """
    params = PSOParams(
        n_particles=n_particles,
        n_iterations=n_iterations,
        bounds=bounds,
    )
    pso = ParticleSwarmOptimizer(dim=dim, params=params)
    return pso.optimize(
        fitness_fn=fitness_fn,
        init_x=init_x,
        early_stop_threshold=early_stop_threshold,
    )
