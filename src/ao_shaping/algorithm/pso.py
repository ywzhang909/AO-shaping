"""Particle Swarm Optimization (PSO) module.

Standard PSO algorithm for continuous optimization.

Key features:
- Inertia weight for momentum
- Cognitive component (personal best)
- Social component (global best)
- Velocity clamping
- Position bounds

Example:
    >>> from ao_shaping.algorithm.pso import ParticleSwarmOptimizer
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
from typing import Callable, Protocol

import numpy as np


class FitnessFunction(Protocol):
    """Protocol for fitness function."""
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate fitness."""
        ...


@dataclass
class PSOParams:
    """PSO parameters."""
    n_particles: int = 30
    n_iterations: int = 1000
    w: float = 0.729  # Inertia weight
    c1: float = 1.49  # Cognitive coefficient
    c2: float = 1.49  # Social coefficient
    v_max: float = 2.0  # Maximum velocity
    bounds: tuple[float, float] = (-10.0, 10.0)


@dataclass
class PSOHistory:
    """History of PSO optimization run."""
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)


class Particle:
    """Single particle in PSO."""
    
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        fitness: float,
    ):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.best_position = position.copy()
        self.best_fitness = fitness


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization optimizer.
    
    Attributes:
        dim: Dimension of the problem.
        params: PSO parameters.
        history: Optimization history.
    """
    
    def __init__(
        self,
        dim: int,
        params: PSOParams | None = None,
        random_state: np.random.Generator | None = None,
    ):
        """Initialize PSO optimizer.
        
        Args:
            dim: Dimension of the optimization problem.
            params: PSO parameters. If None, uses default PSOParams.
            random_state: Random generator for reproducibility.
        """
        self.dim = dim
        self.params = params if params is not None else PSOParams()
        self.rng = random_state if random_state is not None else np.random.default_rng()
        self.history = PSOHistory()
        self.particles: list[Particle] = []
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness: float = float('inf')
        
    def _reset(self) -> None:
        """Reset optimizer state."""
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
    def _initialize_particles(self, init_x: np.ndarray | None = None) -> list[Particle]:
        """Initialize particles.
        
        Args:
            init_x: Initial point to include in particles.
            
        Returns:
            List of particles.
        """
        particles = []
        
        for i in range(self.params.n_particles):
            if init_x is not None and i == 0:
                position = init_x.copy()
            else:
                position = self.rng.uniform(
                    self.params.bounds[0],
                    self.params.bounds[1],
                    self.dim
                )
            
            velocity = self.rng.uniform(
                -self.params.v_max,
                self.params.v_max,
                self.dim
            )
            
            particles.append(Particle(
                position=position,
                velocity=velocity,
                fitness=float('inf')
            ))
        
        return particles
    
    def _evaluate_particles(
        self,
        fitness_fn: FitnessFunction,
    ) -> None:
        """Evaluate all particles.
        
        Args:
            fitness_fn: Fitness function.
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
        """Update velocities for all particles."""
        r1 = self.rng.random((self.params.n_particles, self.dim))
        r2 = self.rng.random((self.params.n_particles, self.dim))
        
        for i, p in enumerate(self.particles):
            cognitive = self.params.c1 * r1[i] * (p.best_position - p.position)
            social = self.params.c2 * r2[i] * (self.global_best_position - p.position)
            
            p.velocity = self.params.w * p.velocity + cognitive + social
            
            p.velocity = np.clip(p.velocity, -self.params.v_max, self.params.v_max)
    
    def _update_positions(self) -> None:
        """Update positions for all particles."""
        for p in self.particles:
            p.position = p.position + p.velocity
            p.position = np.clip(p.position, self.params.bounds[0], self.params.bounds[1])
    
    def optimize(
        self,
        fitness_fn: FitnessFunction,
        init_x: np.ndarray | None = None,
        early_stop_threshold: float | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run PSO optimization.
        
        Args:
            fitness_fn: Fitness function to minimize.
            init_x: Initial point to include in particles.
            early_stop_threshold: Stop if best fitness below this threshold.
            callback: Optional callback function called after each iteration
                      with (iteration, best_position, best_fitness).
        
        Returns:
            Tuple of (best_solution, best_fitness).
        """
        self.particles = self._initialize_particles(init_x)
        
        self._evaluate_particles(fitness_fn)
        
        self.history.best_fitness.append(self.global_best_fitness)
        self.history.mean_fitness.append(np.mean([p.fitness for p in self.particles]))
        
        for iteration in range(1, self.params.n_iterations + 1):
            self._update_velocities()
            self._update_positions()
            self._evaluate_particles(fitness_fn)
            
            self.history.best_fitness.append(self.global_best_fitness)
            self.history.mean_fitness.append(np.mean([p.fitness for p in self.particles]))
            
            if callback is not None:
                assert self.global_best_position is not None
                callback(
                    iteration,
                    self.global_best_position.copy(),
                    self.global_best_fitness
                )
            
            if (early_stop_threshold is not None and 
                self.global_best_fitness < early_stop_threshold):
                break
        
        assert self.global_best_position is not None
        return self.global_best_position.copy(), self.global_best_fitness
    
    @property
    def convergence_history(self) -> list[float]:
        """Return convergence history (best fitness per iteration)."""
        return self.history.best_fitness


def minimize_pso(
    fitness_fn: FitnessFunction,
    dim: int,
    n_particles: int = 30,
    n_iterations: int = 1000,
    bounds: tuple[float, float] = (-10.0, 10.0),
    init_x: np.ndarray | None = None,
    early_stop_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convenience function for PSO optimization.
    
    Args:
        fitness_fn: Fitness function to minimize.
        dim: Dimension of the problem.
        n_particles: Number of particles.
        n_iterations: Number of iterations.
        bounds: Search space bounds (min, max).
        init_x: Initial point.
        early_stop_threshold: Early stopping threshold.
        
    Returns:
        Tuple of (best_solution, best_fitness).
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