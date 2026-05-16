"""Simulated Annealing (SA) optimization module.

Standard SA algorithm for continuous optimization.

Key features:
- Multiple temperature schedules
- Metropolis acceptance criterion
- Neighbor generation
- Adaptive cooling

Example:
    >>> from ao_shaping.algorithm.simulated_annealing import SimulatedAnnealing
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
from typing import Callable, Protocol

import numpy as np


class FitnessFunction(Protocol):
    """Protocol for fitness function."""
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate fitness."""
        ...


class TempSchedule(Enum):
    """Temperature schedule types."""
    LINEAR = auto()
    EXPONENTIAL = auto()
    LOGARITHMIC = auto()
    COSINE = auto()


@dataclass
class SAParams:
    """Simulated Annealing parameters."""
    n_iterations: int = 1000
    initial_temp: float = 100.0
    final_temp: float = 0.01
    schedule: TempSchedule = TempSchedule.EXPONENTIAL
    step_size: float = 0.5  # Standard deviation for neighbor generation
    bounds: tuple[float, float] = (-10.0, 10.0)


@dataclass
class SAHistory:
    """History of SA optimization run."""
    best_fitness: list[float] = field(default_factory=list)
    current_fitness: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)


class SimulatedAnnealing:
    """Simulated Annealing optimizer.
    
    Attributes:
        dim: Dimension of the problem.
        params: SA parameters.
        history: Optimization history.
    """
    
    def __init__(
        self,
        dim: int,
        params: SAParams | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """Initialize SA optimizer.
        
        Args:
            dim: Dimension of the optimization problem.
            params: SA parameters. If None, uses default SAParams.
            random_state: Random generator for reproducibility.
        """
        self.dim = dim
        self.params = params if params is not None else SAParams()
        self.rng = random_state if random_state is not None else np.random.default_rng()
        self.history = SAHistory()
        
    def _get_temperature(self, iteration: int) -> float:
        """Get temperature for current iteration.
        
        Args:
            iteration: Current iteration number.
            
        Returns:
            Current temperature.
        """
        t = iteration / self.params.n_iterations
        
        if self.params.schedule == TempSchedule.LINEAR:
            return self.params.initial_temp * (1 - t) + self.params.final_temp * t
        
        elif self.params.schedule == TempSchedule.EXPONENTIAL:
            return self.params.initial_temp * (self.params.final_temp / self.params.initial_temp) ** t
        
        elif self.params.schedule == TempSchedule.LOGARITHMIC:
            if t == 0:
                return self.params.initial_temp
            return self.params.initial_temp / (1 + t * (self.params.initial_temp / self.params.final_temp - 1))
        
        elif self.params.schedule == TempSchedule.COSINE:
            return self.params.final_temp + 0.5 * (self.params.initial_temp - self.params.final_temp) * (1 + np.cos(np.pi * t))
        
        return self.params.initial_temp
    
    def _generate_neighbor(self, current: np.ndarray) -> np.ndarray:
        """Generate neighbor solution.
        
        Args:
            current: Current solution.
            
        Returns:
            Neighbor solution.
        """
        neighbor = current + self.rng.normal(0, self.params.step_size, self.dim)
        return np.clip(neighbor, self.params.bounds[0], self.params.bounds[1])
    
    def _acceptance_probability(
        self,
        current_fitness: float,
        new_fitness: float,
        temperature: float,
    ) -> float:
        """Calculate acceptance probability using Metropolis criterion.
        
        Args:
            current_fitness: Fitness of current solution.
            new_fitness: Fitness of new solution.
            temperature: Current temperature.
            
        Returns:
            Acceptance probability.
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
        init_x: np.ndarray | None = None,
        early_stop_threshold: float | None = None,
        callback: Callable[[int, np.ndarray, float, float], None] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run simulated annealing optimization.
        
        Args:
            fitness_fn: Fitness function to minimize.
            init_x: Initial point. If None, starts from random.
            early_stop_threshold: Stop if best fitness below this threshold.
            callback: Optional callback function called after each iteration
                      with (iteration, current_position, current_fitness, temperature).
        
        Returns:
            Tuple of (best_solution, best_fitness).
        """
        if init_x is not None:
            current = init_x.copy()
        else:
            current = self.rng.uniform(
                self.params.bounds[0],
                self.params.bounds[1],
                self.dim
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
        """Return convergence history (best fitness per iteration)."""
        return self.history.best_fitness


def minimize_sa(
    fitness_fn: FitnessFunction,
    dim: int,
    n_iterations: int = 1000,
    initial_temp: float = 100.0,
    final_temp: float = 0.01,
    schedule: TempSchedule = TempSchedule.EXPONENTIAL,
    bounds: tuple[float, float] = (-10.0, 10.0),
    init_x: np.ndarray | None = None,
    early_stop_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convenience function for SA optimization.
    
    Args:
        fitness_fn: Fitness function to minimize.
        dim: Dimension of the problem.
        n_iterations: Number of iterations.
        initial_temp: Initial temperature.
        final_temp: Final temperature.
        schedule: Temperature schedule.
        bounds: Search space bounds (min, max).
        init_x: Initial point.
        early_stop_threshold: Early stopping threshold.
        
    Returns:
        Tuple of (best_solution, best_fitness).
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