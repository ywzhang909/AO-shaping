"""Base class for heuristic optimization algorithms.

Provides a common interface for switching between different optimizers.

Concrete optimizers **register themselves**; this module never imports them.
Adding an algorithm therefore touches exactly one new module (which declares
``_registry_key``) and nothing here -- the Open/Closed Principle holds because
:meth:`HeuristicOptimizer.create` is a dict lookup, not an ``if``/``elif`` chain.

Registration happens at subclass-definition time via ``__init_subclass__``, so a
concrete optimizer is available as soon as its module has been imported. The
top-level ``ao_shaping.algorithm`` facade imports every concrete optimizer, so
importing anything under ``ao_shaping.algorithm.heuristic`` has already populated
the registry.

This module does no fault tolerance. ``create`` forwards every keyword it does
not own straight to the concrete constructor, so offering an option an algorithm
does not support raises ``TypeError`` from that constructor -- naming the
offending argument -- instead of being silently dropped here. Callers that present
a superset of options are expected to know which algorithms accept which; see
``search.POPULATION_ALGORITHMS`` for the population-size case.

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

from ao_shaping.algorithm.base import RegisteredBase


class OptimizerType(Enum):
    """Available optimizer types.

    Every member needs a registered implementation, because
    :meth:`HeuristicOptimizer.create` is expected to serve the whole enum.
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
    """Common optimizer configuration.

    These are the arguments :meth:`HeuristicOptimizer.create` consumes on the
    caller's behalf; every other keyword is forwarded to the constructor.
    """

    n_iterations: int = 1000
    bounds: tuple[float, float] = (-10.0, 10.0)
    early_stop_threshold: float | None = None
    seed: int | None = None


class HeuristicOptimizer(RegisteredBase):
    """Abstract base class for heuristic optimizers.

    All heuristic algorithms inherit from this class for uniform interface.

    Subclasses opt into the :meth:`create` factory by declaring
    ``_registry_key``; they are then registered automatically::

        class MyAlgorithm(HeuristicOptimizer):
            _registry_key = OptimizerType.MY_ALGORITHM
    """

    #: This family's registry, keyed by :class:`OptimizerType`.
    _registry: ClassVar[dict[OptimizerType, type[HeuristicOptimizer]]] = {}

    #: Keyword arguments consumed by :meth:`create` rather than forwarded to a
    #: concrete constructor. Derived from :class:`OptimizerConfig` so a new
    #: config field is reserved automatically.
    _CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        OptimizerConfig.__dataclass_fields__
    )

    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """Initialize optimizer.

        Args:
            dim: Dimension of the optimization problem. Must be positive.
            config: Common configuration. If None, uses default.
            random_state: Random generator for reproducibility. When None, one
                is derived from ``config.seed`` so a seeded run is reproducible.

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        self._validate_dim(dim)
        self.dim = dim
        self.config = config if config is not None else OptimizerConfig()
        self.rng = (
            random_state
            if random_state is not None
            else np.random.default_rng(self.config.seed)
        )
        self._best_solution: np.ndarray | None = None
        self._best_fitness: float | None = None
        self._convergence_history: list[float] = []

    @abstractmethod
    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        init_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run optimization.

        Args:
            fitness_fn: Fitness function to minimize.
            init_x: Initial point.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        pass

    @property
    def best_solution(self) -> np.ndarray | None:
        """Return best solution found."""
        return self._best_solution

    @property
    def best_fitness(self) -> float | None:
        """Return best fitness found."""
        return self._best_fitness

    @property
    def convergence_history(self) -> list[float]:
        """Return a copy of the convergence history (best fitness per iteration).

        A copy is returned so callers cannot mutate the recorded history.
        """
        return self._convergence_history.copy()

    def reset(self) -> None:
        """Return the instance to its pre-run state so it can be reused.

        Clears the recorded best solution / fitness and the convergence history,
        then delegates to :meth:`_reset` for subclass state.
        """
        self._best_solution = None
        self._best_fitness = None
        self._convergence_history.clear()
        self._reset()

    @classmethod
    def _describe_keys(cls) -> list[str]:
        """Return the registered selector names for error messages."""
        return [member.name for member in cls.registered()]

    @classmethod
    def create(
        cls,
        optimizer_type: OptimizerType,
        dim: int,
        **kwargs: Any,
    ) -> HeuristicOptimizer:
        """Create an optimizer by type, looked up in the registry.

        Args:
            optimizer_type: Registered selector (see :meth:`registered`).
            dim: Dimension of the problem.
            **kwargs: ``n_iterations`` / ``bounds`` / ``early_stop_threshold`` /
                ``seed`` configure the run and are consumed here. Every other
                keyword is forwarded to the concrete constructor, which raises
                ``TypeError`` if it does not accept it.

        Returns:
            Optimizer instance.

        Raises:
            ValueError: If ``optimizer_type`` has no registered implementation.
            ValueError: If ``dim`` is not positive.
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
        """Build an instance from the common config plus algorithm extras.

        Overridden by the optimizers whose constructor takes its own parameter
        dataclass instead of an :class:`OptimizerConfig`.
        """
        return cls(dim=dim, config=config, random_state=random_state, **kwargs)
