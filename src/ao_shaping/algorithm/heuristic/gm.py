"""Guided Mutation (GM) for population-based heuristic optimizers.

This module provides three pieces:

* :class:`GMOperator` -- the operator interface, plus two kernels:
  :class:`RankGuidedMutation` and :class:`ValueFrequencyGuidedMutation`.
* :class:`GMOptimizerMixin` -- the generic GM interface a heuristic mixes in to
  gain ``use_gm`` / ``_gm_operator`` / ``apply_gm_if_enabled``.
* :func:`guided_mutation` -- a decorator that wraps one evolution step and splices
  GM offspring into the population right after it runs.

Two independent GM definitions sit behind the single :class:`GMOperator`
interface, because "guided mutation" has **no canonical published definition**:

``RankGuidedMutation``
    A *continuous* scheme. Parents are drawn with linear-rank weights, then a
    shrinking subset of coordinates is perturbed by an annealed Gaussian whose
    magnitude also decays. This is the closest relative in the literature (rank
    based adaptive mutation, e.g. Basak 2021, arXiv:2104.08842, which adapts the
    per-individual mutation *rate*), here extended to a shared step size.
    **The closed form below is this repository's choice, not a cited formula.**

``ValueFrequencyGuidedMutation``
    Follows the discrete operator of Liu Hui et al., *A universal feedback-based
    improvement strategy for wavefront-shaping algorithms*, Acta Photonica
    Sinica 52(6):0629002 (2023), doi:10.3788/gzxb20235206.0629002 -- the paper
    behind the ``add GM algorithm`` TODO in this package. It keeps a value
    frequency table ``G``, picks ``N_G(k)`` guided units, and samples their new
    values proportionally to that frequency. It is adapted here from a discrete
    domain to the continuous one by resampling recorded elite values per
    dimension.

Neither kernel guarantees elitism: ``parent + noise`` can be worse than the
parent. Callers needing monotone non-degradation must retain elites themselves.
GA does so natively; the ``replace_worst`` merge used by DE/CEM/PSO only ever
overwrites the tail of the population.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt


def _check_population(
    population: npt.NDArray[np.float64],
    fitness: npt.NDArray[np.float64],
    ranks: npt.NDArray[np.intp],
) -> None:
    """Validate a kernel's preconditions loudly.

    The kernels are the concrete implementations, so they -- not the mixin --
    are the right place to reject input they cannot honour.

    Raises:
        ValueError: If the population is empty, not 2-D, or its fitness/ranks do
            not line up with it.
    """
    if population.ndim != 2:
        raise ValueError(f"population must be 2-D, got shape {population.shape}")
    rows = population.shape[0]
    if rows == 0:
        raise ValueError("population must contain at least one individual")
    if fitness.shape != (rows,):
        raise ValueError(
            f"fitness must have shape ({rows},), got {fitness.shape}"
        )
    if ranks.shape != (rows,):
        raise ValueError(f"ranks must have shape ({rows},), got {ranks.shape}")


class GMOperator(ABC):
    """Interface for a guided-mutation kernel.

    Implementations turn a scored population into ``n_offspring`` new candidates.
    They must not mutate ``population`` or ``fitness``.
    """

    @abstractmethod
    def __call__(
        self,
        population: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
        ranks: npt.NDArray[np.intp],
        current_iter: int,
        n_offspring: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Generate guided-mutation offspring.

        Args:
            population: Population array of shape ``(n, dim)``.
            fitness: Fitness per individual, shape ``(n,)``; **lower is better**,
                matching the minimising convention of this package.
            ranks: Rank per individual, ``1`` = best.
            current_iter: Zero-based generation index, used for annealing.
            n_offspring: Number of children to produce.
            bounds: Scalar ``(low, high)`` clip applied to every child.
            rng: Random generator (never the global numpy RNG).

        Returns:
            Array of shape ``(n_offspring, dim)``, clipped into ``bounds``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear cross-generation state. No-op for stateless kernels."""


class RankGuidedMutation(GMOperator):
    """Continuous rank-weighted guided mutation (this repo's definition).

    Parents are sampled with linear ranking weights
    ``w_i = (N + 1 - r_i) / (0.5 * N * (N + 1))``, so rank 1 is heaviest and the
    weights sum to 1. Each child perturbs ``N_G(k)`` coordinates drawn without
    replacement, where both the perturbed fraction and the Gaussian magnitude
    decay with the generation::

        fG(k)  = (g0 - g_end) * k ** (-1 / lam_g) + g_end
        N_G(k) = clip(round(dim * fG(k)), 1, dim)
        sigma(k) = sigma0 * (sigma_min / sigma0) ** (k / lam_s)

    Attributes:
        g0: Initial perturbed fraction of coordinates.
        g_end: Asymptotic perturbed fraction.
        lam_g: Decay constant of the perturbed fraction.
        sigma0: Initial perturbation magnitude, as a fraction of the bound span.
        sigma_min: Floor on the perturbation magnitude.
        lam_s: Decay constant of the magnitude.
    """

    def __init__(
        self,
        g0: float = 0.10,
        g_end: float = 0.0025,
        lam_g: float = 250.0,
        sigma0: float = 0.10,
        sigma_min: float = 1e-3,
        lam_s: float = 50.0,
    ) -> None:
        """Initialize the kernel.

        Args:
            g0: Initial perturbed fraction of coordinates, in ``(0, 1]``.
            g_end: Asymptotic perturbed fraction, in ``[0, g0]``.
            lam_g: Decay constant of the perturbed fraction; larger is slower.
            sigma0: Initial magnitude as a fraction of the bound span.
            sigma_min: Magnitude floor as a fraction of the bound span.
            lam_s: Decay constant of the magnitude; larger is slower.

        Raises:
            ValueError: If the schedule parameters are inconsistent.
        """
        if not 0.0 < g_end <= g0 <= 1.0:
            raise ValueError(f"require 0 < g_end <= g0 <= 1, got g_end={g_end}, g0={g0}")
        if not 0.0 < sigma_min <= sigma0:
            raise ValueError(
                f"require 0 < sigma_min <= sigma0, got {sigma_min}, {sigma0}"
            )
        if lam_g <= 0.0 or lam_s <= 0.0:
            raise ValueError(f"lam_g and lam_s must be > 0, got {lam_g}, {lam_s}")
        self.g0 = float(g0)
        self.g_end = float(g_end)
        self.lam_g = float(lam_g)
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.lam_s = float(lam_s)

    def _fraction(self, k: int) -> float:
        """Return the perturbed coordinate fraction at generation ``k``."""
        if k <= 0:
            return self.g0
        return (self.g0 - self.g_end) * k ** (-1.0 / self.lam_g) + self.g_end

    def __call__(
        self,
        population: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
        ranks: npt.NDArray[np.intp],
        current_iter: int,
        n_offspring: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        low, high = float(bounds[0]), float(bounds[1])
        n, dim = population.shape
        _check_population(population, fitness, ranks)
        span = high - low

        # Linear ranking weights: rank 1 (best) gets the largest weight.
        weights = (n + 1.0 - ranks) / (0.5 * n * (n + 1.0))
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(n, 1.0 / n)

        parents = population[rng.choice(n, size=n_offspring, p=weights)].copy()

        k = max(int(current_iter), 0)
        n_guided = int(np.clip(round(dim * self._fraction(k)), 1, dim))
        sigma = self.sigma0 * (self.sigma_min / self.sigma0) ** (k / self.lam_s) * span

        for row in range(n_offspring):
            loci = rng.choice(dim, size=n_guided, replace=False)
            parents[row, loci] += rng.normal(0.0, sigma, size=n_guided)

        return np.clip(parents, low, high)


class ValueFrequencyGuidedMutation(GMOperator):
    """Discrete value-frequency GM, adapted to continuous domains.

    Follows Liu Hui et al., Acta Photonica Sinica 52(6):0629002 (2023): the
    paper keeps a value-frequency table ``G`` and samples new values for the
    chosen guided units proportionally to that frequency.

    Continuous adaptation used here: each *coordinate* is a "unit", and the
    frequency table records, per coordinate, how often each recorded elite value
    was seen. Guided units are drawn by frequency (so coordinates that
    historically produced good individuals get perturbed more often), and their
    new values are resampled from that coordinate's recorded elite values
    (value-frequency sampling) plus a Gaussian jitter that anneals to zero.

    Because it resamples elite values, the first call must be preceded by
    :meth:`observe` (which :meth:`__call__` does automatically when no history
    has been recorded yet, seeding from the incoming population).

    Attributes:
        elite_fraction: Fraction of the population treated as elite.
        n_guided: Guided units per child, or ``None`` to derive it from the
            generation index via ``n_guided0`` and ``lam_g``.
        n_guided0: Initial guided-unit count when ``n_guided`` is None.
        lam_g: Decay constant of the guided-unit count.
        jitter: Gaussian jitter added to a resampled elite value, as a fraction
            of the bound span.
        elite_decay: Probability of forgetting a recorded elite value, keeping
            the table from growing without bound.
    """

    def __init__(
        self,
        elite_fraction: float = 0.2,
        n_guided: int | None = None,
        n_guided0: int = 3,
        lam_g: float = 25.0,
        jitter: float = 0.01,
        elite_decay: float = 0.05,
    ) -> None:
        """Initialize the kernel.

        Args:
            elite_fraction: Fraction of the population recorded as elite, in
                ``(0, 1]``.
            n_guided: Fixed guided-unit count, or None to anneal it.
            n_guided0: Guided-unit count at generation 0 when annealing.
            lam_g: Decay constant of the guided-unit count.
            jitter: Jitter magnitude as a fraction of the bound span.
            elite_decay: Per-call probability of forgetting a recorded value.

        Raises:
            ValueError: If the parameters are out of range.
        """
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError(
                f"elite_fraction must be in (0, 1], got {elite_fraction}"
            )
        if n_guided is not None and n_guided < 1:
            raise ValueError(f"n_guided must be >= 1 or None, got {n_guided}")
        if n_guided0 < 1:
            raise ValueError(f"n_guided0 must be >= 1, got {n_guided0}")
        if lam_g <= 0.0:
            raise ValueError(f"lam_g must be > 0, got {lam_g}")
        if jitter < 0.0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        if not 0.0 <= elite_decay < 1.0:
            raise ValueError(f"elite_decay must be in [0, 1), got {elite_decay}")
        self.elite_fraction = float(elite_fraction)
        self.n_guided = n_guided
        self.n_guided0 = int(n_guided0)
        self.lam_g = float(lam_g)
        self.jitter = float(jitter)
        self.elite_decay = float(elite_decay)
        self._elite_values: dict[int, list[float]] = {}

    def reset(self) -> None:
        """Forget every recorded elite value."""
        self._elite_values.clear()

    def observe(self, population: npt.NDArray[np.float64], fitness: npt.NDArray[np.float64]) -> None:
        """Record this generation's elite individuals into the value table.

        Args:
            population: Population array of shape ``(n, dim)``.
            fitness: Fitness per individual, shape ``(n,)``; lower is better.
        """
        if population.size == 0:
            return
        n_elite = max(1, int(population.shape[0] * self.elite_fraction))
        elite_idx = np.argsort(fitness)[:n_elite]
        for row in elite_idx:
            for coord, value in enumerate(population[row]):
                bucket = self._elite_values.setdefault(coord, [])
                bucket.append(float(value))
                # Bound the table so long runs cannot grow it without limit.
                cap = max(8, n_elite * 8)
                if len(bucket) > cap:
                    del bucket[: len(bucket) - cap]

    def _guided_count(self, dim: int, k: int) -> int:
        """Return the number of guided units at generation ``k``."""
        if self.n_guided is not None:
            return int(np.clip(self.n_guided, 1, dim))
        k = max(int(k), 1)
        scaled = self.n_guided0 * k ** (-1.0 / self.lam_g)
        return int(np.clip(round(scaled), 1, dim))

    def __call__(
        self,
        population: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
        ranks: npt.NDArray[np.intp],
        current_iter: int,
        n_offspring: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        low, high = float(bounds[0]), float(bounds[1])
        span = high - low
        n, dim = population.shape
        _check_population(population, fitness, ranks)
        if n_offspring <= 0:
            return np.empty((0, dim), dtype=np.float64)

        # Rank-weighted parents, same weighting as the continuous kernel so both
        # operators exploit the same selection pressure.
        weights = (n + 1.0 - ranks) / (0.5 * n * (n + 1.0))
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(n, 1.0 / n)

        if not self._elite_values:
            self.observe(population, fitness)
            self.elite_decay = 0.0  # seed only; real forgetting starts next call

        children = population[rng.choice(n, size=n_offspring, p=weights)].copy()

        # Units that already carry recorded elite values are the "guided" ones.
        recorded = [c for c in range(dim) if self._elite_values.get(c)]
        if not recorded:
            return children
        n_guided = min(self._guided_count(dim, current_iter), len(recorded))

        for row in range(n_offspring):
            coords = rng.choice(recorded, size=n_guided, replace=False)
            for coord in coords:
                pool = self._elite_values[coord]
                children[row, coord] = rng.choice(pool)
                if self.jitter > 0.0:
                    children[row, coord] += rng.normal(
                        0.0, self.jitter * span
                    )

        if self.elite_decay > 0.0:
            for coord, pool in self._elite_values.items():
                keep = [v for v in pool if rng.random() >= self.elite_decay]
                if keep:
                    self._elite_values[coord] = keep

        return np.clip(children, low, high)


class GMOptimizerMixin:
    """Guided-mutation contract for a population-based optimizer.

    A concrete optimizer joins the GM family by mixing this in, declaring
    ``_gm_operator`` support, and implementing the five hooks below. The mixin
    holds no state of its own beyond the on/off switch, so it can be combined
    with :class:`HeuristicOptimizer` without affecting the MRO or constructor.

    The five hooks are the whole contract. They are declared
    ``NotImplementedError`` rather than given ``getattr``-based fallbacks,
    because a silently-defaulting hook is how a GM operator ends up mutating the
    wrong population:

    ``_gm_population()``
        The current population as an ``(n, dim)`` array.
    ``_gm_fitness()``
        The matching fitness array, shape ``(n,)``, **lower is better**.
    ``_gm_iteration()``
        The zero-based generation index, used by the kernels' annealing.
    ``_gm_offspring()``
        How many candidates GM may contribute this generation.
    ``_gm_bounds()``
        The scalar ``(low, high)`` every child is clipped into.

    After merging, :meth:`_gm_commit` receives the new population and must write
    it back to the optimizer's own state.

    An optimizer whose population is already an ``(n, dim)`` array should mix in
    :class:`NumpyPopulationGM` instead, which implements all six hooks from three
    attributes.
    """

    #: Master switch. GM costs nothing while this is False. A plain class
    #: attribute (not a ClassVar) so :meth:`enable_gm` shadows it per instance and
    #: one optimizer's setting can never leak into another's.
    use_gm: bool = False

    #: The active kernel. ``None`` means GM is unavailable, whatever ``use_gm``
    #: says -- the two are set together by :meth:`enable_gm`.
    _gm_operator: GMOperator | None = None

    #: Fraction of the population handed to GM. The remaining slots stay with the
    #: algorithm's own variation operators, so the population size never changes.
    gm_offspring_fraction: float = 0.2

    #: Supplied by the host ``HeuristicOptimizer``.
    dim: int
    rng: np.random.Generator

    def enable_gm(self, operator: GMOperator, use_gm: bool = True) -> None:
        """Install a GM kernel and turn GM on (or off).

        Args:
            operator: The kernel to use.
            use_gm: Whether GM is active after this call.
        """
        self._gm_operator = operator
        self.use_gm = use_gm

    def disable_gm(self) -> None:
        """Turn GM off, keeping the installed kernel for a later re-enable."""
        self.use_gm = False

    # ------------------------------------------------------------------
    # Contract to be implemented by the concrete optimizer
    # ------------------------------------------------------------------
    def _gm_population(self) -> npt.NDArray[np.float64]:
        """Return the current population as an ``(n, dim)`` array."""
        raise NotImplementedError

    def _gm_fitness(self) -> npt.NDArray[np.float64]:
        """Return the current fitness array, shape ``(n,)``; lower is better."""
        raise NotImplementedError

    def _gm_iteration(self) -> int:
        """Return the zero-based generation index."""
        raise NotImplementedError

    def _gm_offspring(self) -> int:
        """Return how many candidates GM may contribute this generation."""
        raise NotImplementedError

    def _gm_bounds(self) -> tuple[float, float]:
        """Return the scalar ``(low, high)`` children are clipped into."""
        raise NotImplementedError

    def _gm_commit(self, population: npt.NDArray[np.float64]) -> None:
        """Adopt ``population`` as the optimizer's new population."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared behaviour
    # ------------------------------------------------------------------
    def _ranks(self, fitness: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """Return competition-free ranks, ``1`` for the best (lowest) fitness."""
        return fitness.argsort().argsort() + 1

    def apply_gm_if_enabled(self) -> npt.NDArray[np.float64]:
        """Generate one batch of GM offspring from the optimizer's own state.

        Returns an empty ``(0, dim)`` array when GM is off or no kernel is
        installed, which every merge policy treats as "nothing to add". That is
        the only reason the method is safe to call unconditionally.

        Returns:
            Array of shape ``(n_offspring, dim)``, or ``(0, dim)`` when disabled.
        """
        if not self.use_gm or self._gm_operator is None:
            return np.empty((0, self.dim), dtype=np.float64)
        population = self._gm_population()
        fitness = self._gm_fitness()
        n_offspring = self._gm_offspring()
        if n_offspring <= 0:
            return np.empty((0, self.dim), dtype=np.float64)
        children = self._gm_operator(
            population=population,
            fitness=fitness,
            ranks=self._ranks(fitness),
            current_iter=self._gm_iteration(),
            n_offspring=n_offspring,
            bounds=self._gm_bounds(),
            rng=self.rng,
        )
        return np.asarray(children, dtype=np.float64)


class NumpyPopulationGM(GMOptimizerMixin):
    """:class:`GMOptimizerMixin` for optimizers already holding an ndarray.

    Implements all six hooks from three attributes the concrete optimizer
    maintains as its generation state:

    ``_population``
        The current ``(n, dim)`` population.
    ``_fitness_vals``
        Its scores, shape ``(n,)``.
    ``_current_iter``
        The zero-based generation index.
    """

    _population: npt.NDArray[np.float64]
    _fitness_vals: npt.NDArray[np.float64]
    _current_iter: int
    config: Any  # provides .bounds, supplied by HeuristicOptimizer

    def _gm_population(self) -> npt.NDArray[np.float64]:
        return self._population

    def _gm_fitness(self) -> npt.NDArray[np.float64]:
        return self._fitness_vals

    def _gm_iteration(self) -> int:
        return self._current_iter

    def _gm_offspring(self) -> int:
        if not self.use_gm or self._gm_operator is None:
            return 0
        n = self._gm_population().shape[0]
        if n == 0:
            return 0
        return int(max(1, round(n * self.gm_offspring_fraction)))

    def _gm_bounds(self) -> tuple[float, float]:
        return self.config.bounds

    def _gm_commit(self, population: npt.NDArray[np.float64]) -> None:
        self._population = population


_GMStep = TypeVar("_GMStep", bound=Callable[..., npt.NDArray[np.float64]])


def _merge_grow(
    evolved: npt.NDArray[np.float64], children: npt.NDArray[np.float64], worst: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Append GM offspring after the step's own offspring.

    The population grows by ``len(children)``; the caller is responsible for the
    resulting size. Only correct when the algorithm reserves exactly that many
    slots -- see :meth:`NumpyPopulationGM._gm_offspring`.
    """
    return np.vstack([evolved, children])


def _merge_replace_worst(
    evolved: npt.NDArray[np.float64], children: npt.NDArray[np.float64], worst: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Overwrite the ``worst`` rows with GM offspring, keeping the size fixed.

    Required for any algorithm whose population has a fixed size, where growing
    it would break the operator's own indexing.
    """
    merged = evolved.copy()
    merged[worst] = children
    return merged


def guided_mutation(
    merge: Literal["grow", "replace_worst"] = "grow",
) -> Callable[[_GMStep], _GMStep]:
    """Splice guided-mutation offspring into the population after an evolution step.

    The decorated step must return the new population as an ``(n, dim)`` array
    and its host must satisfy the :class:`GMOptimizerMixin` contract. With GM off
    the wrapper returns exactly what the step returned, having consumed no random
    numbers -- which is what keeps a disabled optimizer bit-identical to a run
    without the decorator.

    Args:
        merge: ``"grow"`` appends the offspring (the step must have reserved their
            slots, so the size ends up unchanged). ``"replace_worst"`` overwrites
            the worst rows instead, for algorithms that tolerate a fixed size.

    Returns:
        A decorator that wraps one evolution step.
    """
    merge_fn = _merge_grow if merge == "grow" else _merge_replace_worst

    def decorator(evolve_step: _GMStep) -> _GMStep:
        def wrapper(self: GMOptimizerMixin, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
            evolved = evolve_step(self, *args, **kwargs)
            if not self.use_gm or self._gm_operator is None:
                return evolved

            fitness = self._gm_fitness()
            children = self.apply_gm_if_enabled()
            if children.size == 0:
                return evolved

            if merge_fn is _merge_replace_worst:
                n_replace = children.shape[0]
                worst = np.argsort(self._ranks(fitness), kind="stable")[-n_replace:]
                merged = merge_fn(evolved, children, worst)
            else:
                merged = merge_fn(evolved, children, worst=np.empty(0, dtype=int))
            self._gm_commit(merged)
            return merged

        wrapper.__name__ = evolve_step.__name__
        wrapper.__doc__ = evolve_step.__doc__
        wrapper.__wrapped__ = evolve_step  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


