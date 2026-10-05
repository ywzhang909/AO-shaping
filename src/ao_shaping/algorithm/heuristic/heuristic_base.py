"""启发式优化算法的基类。

为在不同优化器之间切换提供统一接口。

具体优化器**自行注册**; 本模块从不导入它们。因此新增一个算法只会多碰一个新
模块 (它声明 ``_registry_key``), 这里什么也不用动 -- 开闭原则成立, 因为
:meth:`HeuristicOptimizer.create` 只是一次 dict 查找, 而不是 ``if``/``elif``
链。

注册发生在子类定义时, 借助 ``__init_subclass__`` 完成, 所以一个具体优化器在
它的模块被导入之后就已经可用。顶层 ``ao_shaping.algorithm`` 门面会导入每一个
具体优化器, 所以导入 ``ao_shaping.algorithm.heuristic`` 下的任何东西时, 注册表
早已填好。

本模块不做任何容错。``create`` 会把每一个它不拥有的关键字原样转发给具体构造
函数, 所以传入某个算法不支持的选项时, 会由该构造函数抛出 ``TypeError`` --
并指名道姓地指出那个参数 -- 而不会在这里被悄悄丢掉。那些提供超集选项的调用方
本来就应该知道哪些算法接受哪些选项; 种群规模的情形见
``search.POPULATION_ALGORITHMS``。

Example:
    >>> from ao_shaping.algorithm.heuristic.heuristic_base import (
    ...     HeuristicOptimizer,
    ...     OptimizerType,
    ... )
    >>> from ao_shaping.algorithm import GeneticAlgorithm  # registers OptimizerType.GA
    >>>
    >>> opt = HeuristicOptimizer.create(
    ...     OptimizerType.GA,
    ...     dim=5,
    ...     n_iterations=100,
    ... )
    >>> best_x, best_f = opt.optimize(lambda x: float(np.sum(x**2)))
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.base import RegisteredBase


class OptimizerType(Enum):
    """可用的优化器类型。

    每一个成员都需要一个已注册的实现, 因为 :meth:`HeuristicOptimizer.create`
    预期要服务整个枚举。
    """

    GA = auto()
    PSO = auto()
    SA = auto()
    HILL_CLIMBING = auto()
    RANDOM_SEARCH = auto()
    CROSS_ENTROPY = auto()
    DIFFERENTIAL_EVOLUTION = auto()


@dataclass
class OptimizerConfig:
    """优化器通用配置。

    这些就是 :meth:`HeuristicOptimizer.create` 代调用方消费掉的参数; 其余每个
    关键字都转发给具体构造函数。
    """

    n_iterations: int = 1000
    bounds: tuple[float, float] = (-10.0, 10.0)
    early_stop_threshold: float | None = None
    seed: int | None = None


class HeuristicOptimizer(RegisteredBase):
    """启发式优化器的抽象基类。

    所有启发式算法都继承本类, 以获得统一接口。

    子类通过声明 ``_registry_key`` 接入 :meth:`create` 工厂; 声明后即被自动注册::

        class MyAlgorithm(HeuristicOptimizer):
            _registry_key = OptimizerType.MY_ALGORITHM
    """

    #: 本家族的注册表, 以 :class:`OptimizerType` 为键。
    _registry: ClassVar[dict[OptimizerType, type[HeuristicOptimizer]]] = {}

    #: 由 :meth:`create` 消费、而非转发给具体构造函数的关键字参数。取自
    #: :class:`OptimizerConfig`, 所以新增一个配置字段会被自动保留。
    _CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        OptimizerConfig.__dataclass_fields__
    )

    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """初始化优化器。

        Args:
            dim: 优化问题的维度。必须为正。
            config: 通用配置。为 None 时用默认值。
            random_state: 用于可复现性的随机数生成器。为 None 时由 ``config.seed``
                派生, 于是带种子的运行是可复现的。

        Raises:
            ValueError: 若 ``dim`` 不是正数。
        """
        self._validate_dim(dim)
        self.dim = dim
        self.config = config if config is not None else OptimizerConfig()
        self.rng = (
            random_state
            if random_state is not None
            else np.random.default_rng(self.config.seed)
        )
        self._best_solution: npt.NDArray[np.float64] | None = None
        self._best_fitness: float | None = None
        self._convergence_history: list[float] = []

    @abstractmethod
    def optimize(
        self,
        fitness_fn: Callable[[npt.NDArray[np.float64]], float],
        init_x: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """执行优化。

        Args:
            fitness_fn: 待最小化的适应度函数。
            init_x: 初始点。

        Returns:
            (best_solution, best_fitness) 二元组。
        """
        pass

    @property
    def best_solution(self) -> npt.NDArray[np.float64] | None:
        """返回找到的最佳解。"""
        return self._best_solution

    @property
    def best_fitness(self) -> float | None:
        """返回找到的最佳适应度。"""
        return self._best_fitness

    @property
    def convergence_history(self) -> list[float]:
        """返回收敛历史 (每轮迭代的最佳适应度) 的副本。

        返回副本, 是为了让调用方无法改动已记录的历史。
        """
        return self._convergence_history.copy()

    def reset(self) -> None:
        """把实例恢复到运行前的状态, 使其可被复用。

        清空已记录的最佳解 / 适应度与收敛历史, 然后把子类状态的处理委托给
        :meth:`_reset`。
        """
        self._best_solution = None
        self._best_fitness = None
        self._convergence_history.clear()
        self._reset()

    @classmethod
    def _describe_keys(cls) -> list[str]:
        """返回已注册的选择器名字, 供错误消息使用。"""
        return [member.name for member in cls.registered()]

    @classmethod
    def create(
        cls,
        optimizer_type: OptimizerType,
        dim: int,
        **kwargs: Any,
    ) -> HeuristicOptimizer:
        """按类型创建一个优化器, 在注册表里查找。

        Args:
            optimizer_type: 已注册的选择器 (见 :meth:`registered`)。
            dim: 问题的维度。
            **kwargs: ``n_iterations`` / ``bounds`` / ``early_stop_threshold`` /
                ``seed`` 用于配置本次运行, 在这里被消费掉。其余每个关键字都
                转发给具体构造函数, 由它在不接受时抛出 ``TypeError``。

        Returns:
            优化器实例。

        Raises:
            ValueError: 若 ``optimizer_type`` 没有已注册的实现。
            ValueError: 若 ``dim`` 不是正数。
        """
        config = OptimizerConfig(
            n_iterations=kwargs.get("n_iterations", OptimizerConfig.n_iterations),
            bounds=kwargs.get("bounds", OptimizerConfig.bounds),
            early_stop_threshold=kwargs.get(
                "early_stop_threshold", OptimizerConfig.early_stop_threshold
            ),
            seed=kwargs.get("seed", OptimizerConfig.seed),
        )
        # 构造器只接受 np.random.Generator; 由 seed 派生同种子 rng 传给各实现。
        random_state = np.random.default_rng(config.seed)
        forwarded = {
            key: value for key, value in kwargs.items() if key not in cls._CONFIG_KEYS
        }
        return cls._lookup(optimizer_type)._construct(
            dim=dim,
            config=config,
            random_state=random_state,
            **forwarded,
        )

    @classmethod
    def _construct(
        cls,
        dim: int,
        config: OptimizerConfig,
        random_state: np.random.Generator,
        **kwargs: Any,
    ) -> HeuristicOptimizer:
        """由通用配置加上算法特有参数构造实例。

        那些构造函数接收自己专属参数 dataclass (而非 :class:`OptimizerConfig`)
        的优化器会覆写本方法。
        """
        return cls(dim=dim, config=config, random_state=random_state, **kwargs)
