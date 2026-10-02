"""Particle Swarm Optimization (PSO) module.

Standard PSO algorithm for continuous optimization.

Key features:
- Inertia weight for momentum
- Cognitive component (personal best)
- Social component (global best)
- Velocity clamping
- Position bounds

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

from ao_shaping.algorithm.heuristic.gm import GMOptimizerMixin, guided_mutation
from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


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


class ParticleSwarmOptimizer(GMOptimizerMixin, HeuristicOptimizer):
    """Particle Swarm Optimization optimizer.

    Attributes:
        dim: Dimension of the problem.
        params: PSO parameters.
        history: Optimization history.
    """

    _registry_key = OptimizerType.PSO

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
        self.params = params if params is not None else PSOParams()
        # 从 PSOParams 派生 OptimizerConfig (iterations/bounds), 复用基类的 dim/rng 设置。
        config = OptimizerConfig(
            n_iterations=self.params.n_iterations,
            bounds=self.params.bounds,
        )
        super().__init__(dim, config, random_state)
        self.history = PSOHistory()
        self.particles: list[Particle] = []
        self.global_best_position: np.ndarray | None = None
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
        """Build a PSO from the common config, translating it into PSOParams.

        Args:
            dim: Dimension of the optimization problem.
            config: Common configuration built by ``HeuristicOptimizer.create``.
            random_state: Generator derived from ``config.seed``, or None.
            **kwargs: Only ``n_particles`` is honoured; PSO takes no other extras.

        Returns:
            The constructed ParticleSwarmOptimizer.
        """
        params = PSOParams(
            n_particles=kwargs.get("n_particles", 30),
            n_iterations=config.n_iterations,
            bounds=config.bounds,
        )
        return cls(dim=dim, params=params, random_state=random_state)

    def _reset(self) -> None:
        """Reset optimizer state."""
        self.global_best_position = None
        self.global_best_fitness = float("inf")

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

    @guided_mutation(merge="replace_worst")
    def _advance_positions(self) -> np.ndarray:
        """Advance every particle one step and return the swarm as an array.

        The decorator reads the swarm through the :class:`GMOptimizerMixin`
        contract, then commits any guided-mutation offspring through
        :meth:`_gm_commit`. Returning the array (rather than ``None``) is what
        lets the same decorator serve PSO as it serves the array-backed
        optimizers.

        Returns:
            The swarm positions, shape ``(n_particles, dim)``.
        """
        for p in self.particles:
            p.position = p.position + p.velocity
            p.position = np.clip(
                p.position, self.params.bounds[0], self.params.bounds[1]
            )
        return self._gm_population()

    # ------------------------------------------------------------------
    # GMOptimizerMixin contract -- PSO keeps a list[Particle], so each hook
    # projects to and from an (n, dim) array.
    # ------------------------------------------------------------------
    def _gm_population(self) -> np.ndarray:
        return np.array([p.position for p in self.particles])

    def _gm_fitness(self) -> np.ndarray:
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

    def _gm_commit(self, population: np.ndarray) -> None:
        """Write the merged swarm back, zeroing the velocity of moved particles.

        An injected particle keeps a zero velocity so the swarm does not
        immediately fly the new candidate away; the particles the optimizer moved
        itself keep theirs, so PSO's momentum is preserved. The injected
        particles are scored by the ``_evaluate_particles`` call that follows.
        """
        for particle, position in zip(self.particles, population):
            if not np.array_equal(particle.position, position):
                particle.position = np.array(position, dtype=np.float64)
                particle.velocity = np.zeros_like(particle.velocity)

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
