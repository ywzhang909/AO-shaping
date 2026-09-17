"""Base class for heuristic optimization algorithms.

Provides a common interface for switching between different optimizers.

Example:
    >>> from ao_shaping.algorithm.heuristic_base import HeuristicOptimizer, OptimizerType
    >>> from ao_shaping.algorithm import GeneticAlgorithm, ParticleSwarmOptimizer
    >>> 
    >>> # Create optimizer from type
    >>> opt = HeuristicOptimizer.create(
    ...     OptimizerType.GA,
    ...     dim=5,
    ...     n_iterations=100
    ... )
    >>> best_x, best_f = opt.optimize(lambda x: np.sum(x**2))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np


class OptimizerType(Enum):
    """Available optimizer types."""
    GA = auto()
    PSO = auto()
    SA = auto()
    HILL_CLIMBING = auto()
    RANDOM_SEARCH = auto()
    CROSS_ENTROPY = auto()
    DIFFERENTIAL_EVOLUTION = auto()


@dataclass
class OptimizerConfig:
    """Common optimizer configuration."""
    n_iterations: int = 1000
    bounds: tuple[float, float] = (-10.0, 10.0)
    early_stop_threshold: float | None = None
    seed: int | None = None


class HeuristicOptimizer(ABC):
    """Abstract base class for heuristic optimizers.
    
    All heuristic algorithms inherit from this class for uniform interface.
    """
    
    def __init__(
        self,
        dim: int,
        config: OptimizerConfig | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """Initialize optimizer.
        
        Args:
            dim: Dimension of the optimization problem.
            config: Common configuration. If None, uses default.
            random_state: Random generator for reproducibility.
        """
        self.dim = dim
        self.config = config if config is not None else OptimizerConfig()
        self.rng = random_state if random_state is not None else (
            np.random.default_rng(self.config.seed) if self.config.seed else np.random.default_rng()
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
        """Return convergence history (best fitness per iteration)."""
        return self._convergence_history
    
    @staticmethod
    def create(
        optimizer_type: OptimizerType,
        dim: int,
        **kwargs,
    ) -> "HeuristicOptimizer":
        """Factory method to create optimizer by type.
        
        Args:
            optimizer_type: Type of optimizer to create.
            dim: Dimension of the problem.
            **kwargs: Additional arguments passed to optimizer.
            
        Returns:
            Optimizer instance.
            
        Raises:
            ValueError: If optimizer type is unknown.
        """
        from ao_shaping.algorithm.ga import GeneticAlgorithm, GAParams
        from ao_shaping.algorithm.pso import ParticleSwarmOptimizer, PSOParams
        from ao_shaping.algorithm.simulated_annealing import SimulatedAnnealing, SAParams, TempSchedule
        from ao_shaping.algorithm.hill_climbing import HillClimbing
        from ao_shaping.algorithm.random_search import RandomSearch
        from ao_shaping.algorithm.cross_entropy import CrossEntropyMethod
        from ao_shaping.algorithm.differential_evolution import DifferentialEvolution
        
        config = OptimizerConfig(
            n_iterations=kwargs.pop('n_iterations', 1000),
            bounds=kwargs.pop('bounds', (-10.0, 10.0)),
            early_stop_threshold=kwargs.pop('early_stop_threshold', None),
            seed=kwargs.pop('seed', None),
        )
        # 构造器只接受 np.random.Generator; 由 seed 派生同种子 rng 传给各分支。
        rng = np.random.default_rng(config.seed) if config.seed is not None else None
        
        if optimizer_type == OptimizerType.GA:
            params = GAParams(
                pop_size=kwargs.pop('pop_size', 30),
                n_generations=kwargs.pop('n_generations', config.n_iterations),
                crossover_prob=kwargs.pop('crossover_prob', 0.7),
                mutation_prob=kwargs.pop('mutation_prob', 0.15),
                tournament_size=kwargs.pop('tournament_size', 3),
                elite_count=kwargs.pop('elite_count', 2),
                bounds=config.bounds,
                alpha=kwargs.pop('alpha', 0.5),
                mutation_sigma=kwargs.pop('mutation_sigma', 5.0),
            )
            return GeneticAlgorithm(dim=dim, params=params, random_state=rng)
        
        elif optimizer_type == OptimizerType.PSO:
            params = PSOParams(
                n_particles=kwargs.pop('n_particles', 30),
                n_iterations=kwargs.pop('n_iterations', config.n_iterations),
                w=kwargs.pop('w', 0.729),
                c1=kwargs.pop('c1', 1.49),
                c2=kwargs.pop('c2', 1.49),
                v_max=kwargs.pop('v_max', 2.0),
                bounds=config.bounds,
            )
            return ParticleSwarmOptimizer(dim=dim, params=params, random_state=rng)
        
        elif optimizer_type == OptimizerType.SA:
            params = SAParams(
                n_iterations=kwargs.pop('n_iterations', config.n_iterations),
                initial_temp=kwargs.pop('initial_temp', 100.0),
                final_temp=kwargs.pop('final_temp', 0.01),
                schedule=kwargs.pop('schedule', TempSchedule.EXPONENTIAL),
                step_size=kwargs.pop('step_size', 0.5),
                bounds=config.bounds,
            )
            return SimulatedAnnealing(dim=dim, params=params, random_state=rng)
        
        elif optimizer_type == OptimizerType.HILL_CLIMBING:
            return HillClimbing(dim=dim, config=config, random_state=rng, **kwargs)
        
        elif optimizer_type == OptimizerType.RANDOM_SEARCH:
            return RandomSearch(dim=dim, config=config, random_state=rng, **kwargs)
        
        elif optimizer_type == OptimizerType.CROSS_ENTROPY:
            return CrossEntropyMethod(dim=dim, config=config, random_state=rng, **kwargs)
        
        elif optimizer_type == OptimizerType.DIFFERENTIAL_EVOLUTION:
            return DifferentialEvolution(dim=dim, config=config, random_state=rng, **kwargs)
        
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")