"""Shared black-box heuristic-search driver for the hardware optimization loops.

The heuristic optimizers in this package **minimise** their fitness function,
while the hardware objectives differ in direction (PIB is maximised, RMS is
minimised). This driver centralises everything the optimizers would otherwise
re-implement:

* the ``algorithm name -> OptimizerType`` mapping (``"spgd"`` is excluded — that
  is a gradient method handled by the callers' own loops),
* ``maximize`` sign handling (fitness = ±value),
* bounds clipping so hardware never receives an out-of-range vector,
* a per-evaluation callback (recording / live display),
* cooperative early abort (e.g. the user closed the live pygame window).

This is a leaf algorithm-layer module: it imports no hardware and no optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerType,
)

#: Heuristic algorithm names (excluding ``"spgd"``) -> optimizer type.
HEURISTIC_ALGORITHM_MAP: dict[str, OptimizerType] = {
    "ga": OptimizerType.GA,
    "pso": OptimizerType.PSO,
    "sa": OptimizerType.SA,
    "hc": OptimizerType.HILL_CLIMBING,
    "rs": OptimizerType.RANDOM_SEARCH,
    "cem": OptimizerType.CROSS_ENTROPY,
    "de": OptimizerType.DIFFERENTIAL_EVOLUTION,
}

#: Heuristics that accept a population size (PSO names the same knob
#: ``n_particles``).
POPULATION_ALGORITHMS: frozenset[OptimizerType] = frozenset(
    {
        OptimizerType.GA,
        OptimizerType.PSO,
        OptimizerType.CROSS_ENTROPY,
        OptimizerType.DIFFERENTIAL_EVOLUTION,
    }
)


class SearchAborted(RuntimeError):
    """Raised by a ``should_stop`` callback to end a heuristic search early."""


@dataclass
class HeuristicSearchResult:
    """Outcome of :func:`run_heuristic_search`.

    Attributes:
        best_x: Best parameter vector found (clipped to the search bounds).
        best_value: Raw objective value of ``best_x`` (in the caller's direction).
        evaluations: Number of objective evaluations performed.
        history: Raw objective value of every evaluation, in order.
    """

    best_x: np.ndarray
    best_value: float
    evaluations: int
    history: list[float] = field(default_factory=list)


def heuristic_algorithm_choices(include_spgd: bool = True) -> tuple[str, ...]:
    """Return the algorithm names accepted by the optimizers (``spgd`` first)."""
    names = tuple(HEURISTIC_ALGORITHM_MAP)
    return ("spgd", *names) if include_spgd else names


def create_heuristic(
    optimizer_type: OptimizerType,
    dim: int,
    iterations: int,
    bounds: tuple[float, float],
    seed: int | None = None,
    pop_size: int | None = None,
) -> HeuristicOptimizer:
    """Build a heuristic optimizer through the shared factory.

    Args:
        optimizer_type: Heuristic selector (GA/PSO/SA/HC/RS/CEM/DE).
        dim: Problem dimensionality.
        iterations: Generations / iterations (``n_iterations``).
        bounds: ``(low, high)`` search bounds for every parameter.
        seed: Optional random seed for reproducibility.
        pop_size: Optional population size for GA/PSO/CEM/DE (ignored by
            SA/HC/RS, which do not take one).

    Returns:
        A ready-to-run :class:`HeuristicOptimizer` (call ``.optimize(fn, x0)``).
    """
    factory_kwargs: dict = {
        "n_iterations": max(1, int(iterations)),
        "bounds": (float(bounds[0]), float(bounds[1])),
        "seed": seed,
    }
    if pop_size is not None:
        if optimizer_type is OptimizerType.PSO:
            factory_kwargs["n_particles"] = int(pop_size)
        elif optimizer_type in POPULATION_ALGORITHMS:
            factory_kwargs["pop_size"] = int(pop_size)
    return HeuristicOptimizer.create(optimizer_type, dim=dim, **factory_kwargs)


def run_heuristic_search(
    algorithm: str,
    evaluate: Callable[[np.ndarray], float],
    dim: int,
    iterations: int,
    bounds: tuple[float, float],
    x0: np.ndarray | None = None,
    maximize: bool = False,
    seed: int | None = None,
    pop_size: int | None = None,
    on_evaluate: Callable[[np.ndarray, float, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> HeuristicSearchResult:
    """Drive ``evaluate`` with the named heuristic algorithm.

    ``evaluate`` must return the **raw** objective value (in the caller's own
    units/direction); with ``maximize=True`` the driver searches for its maximum
    by handing ``-value`` to the (minimising) heuristic.

    Args:
        algorithm: One of :data:`HEURISTIC_ALGORITHM_MAP` (case-insensitive).
        evaluate: Black-box objective; receives a bounds-clipped vector.
        dim: Problem dimensionality (e.g. number of Zernike terms).
        iterations: Heuristic iterations/generations (callers map their ``epochs``
            here; note population methods perform ``pop_size`` evaluations each).
        bounds: ``(low, high)`` clip/search bounds for every parameter.
        x0: Optional initial vector (included by GA and used as the start by the
            local methods).
        maximize: True to ascend ``evaluate`` (PIB-like), False to descend it
            (RMS-like).
        seed: Optional random seed.
        pop_size: Optional population size (GA/PSO/CEM/DE).
        on_evaluate: Optional ``(x, value, evaluation_index)`` callback after every
            evaluation — use it to append Recorder rows / refresh a live display.
            ``evaluation_index`` is 1-based.
        should_stop: Optional predicate; when it returns True the search aborts and
            the best-so-far result is returned (no exception).

    Returns:
        HeuristicSearchResult with the best vector, its raw value and history.

    Raises:
        ValueError: Unknown ``algorithm``.
    """
    key = str(algorithm).lower()
    if key not in HEURISTIC_ALGORITHM_MAP:
        raise ValueError(
            f"Unknown heuristic algorithm {algorithm!r}. "
            f"Available: {sorted(HEURISTIC_ALGORITHM_MAP)}"
        )

    lo, hi = float(bounds[0]), float(bounds[1])
    optimizer = create_heuristic(
        HEURISTIC_ALGORITHM_MAP[key], dim, iterations, (lo, hi), seed, pop_size
    )

    evaluations = {"n": 0}
    history: list[float] = []
    best_value: list[float | None] = [None]
    best_x: list[np.ndarray | None] = [None]

    def fitness(x) -> float:
        x_clipped = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
        value = float(evaluate(x_clipped))

        evaluations["n"] += 1
        history.append(value)
        if best_value[0] is None or (
            value > best_value[0] if maximize else value < best_value[0]
        ):
            best_value[0] = value
            best_x[0] = x_clipped.copy()

        if on_evaluate is not None:
            on_evaluate(x_clipped, value, evaluations["n"])
        if should_stop is not None and should_stop():
            raise SearchAborted

        return -value if maximize else value

    try:
        optimizer.optimize(fitness, init_x=x0)
    except SearchAborted:
        # Aborted (e.g. the live window was closed): the best-so-far was already
        # recorded inside ``fitness`` before the stop check, so fall through.
        pass

    if best_x[0] is None:
        raise RuntimeError(
            f"heuristic {algorithm!r} performed no objective evaluation"
        )

    return HeuristicSearchResult(
        best_x=best_x[0],
        best_value=float(best_value[0]),  # type: ignore[arg-type]
        evaluations=evaluations["n"],
        history=history,
    )
