"""Abstract base for iterative, class-based beam-shaping optimizers.

Subclasses that own both a convergence loop (a single ``update()`` step plus
an optional ``run()`` that calls it repeatedly) MAY inherit from
:class:`IterativeOptimizer`.  This enforces the project convention:

- ``__init__`` validates all parameters and initializes state (iteration
  counter, history, best-value tracking).
- ``update()`` performs **one** step.  The return type is left to the
  subclass (e.g. a phase map, a loss value, a result dataclass).
- ``run()`` is optional and drives the full loop until a stopping criterion.

Concrete subclasses:

- :class:`PhaseWrapOptimizer` (``phase_wrap.py``)
- :class:`DifferentiableBeamOptimizer` (``differentiable_beam.py``)

This class is intended ONLY for class-based optimizers that manage their own
iteration state.  Function-based optimizers (e.g. GS, diff-shaping) do not
inherit from it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IterativeOptimizer(ABC):
    """Base class for iterative, class-based beam-shaping optimizers.

    Attributes:
        max_iterations: Maximum number of ``update()`` steps for ``run()``.
        convergence_history: Objective values recorded after each step.
        best_value: Best (lowest) objective seen so far.
    """

    def __init__(self, max_iterations: int = 1000):
        """Initialize the iterative optimizer.

        Args:
            max_iterations: Maximum number of update steps for ``run()``.
                Must be a positive integer.

        Raises:
            ValueError: If ``max_iterations`` is not a positive integer.
        """
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                f"max_iterations must be a positive integer, got {max_iterations!r}"
            )
        self.max_iterations: int = max_iterations
        self._iteration: int = 0
        self._convergence_history: list[float] = []
        self._best_value: float = float("inf")

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self) -> Any:
        """Perform one optimization step.

        Subclasses MUST:
        - update internal state (iteration counter, best value, history)
        - call ``self._record(value)`` with the step's objective so that
          bookkeeping stays consistent.

        Returns:
            The step's result (e.g. the new phase map, or the current
            objective value).  The exact type is implementation-defined.
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def run(self) -> Any:
        """Run the full optimization loop.

        Calls :meth:`update` until ``max_iterations`` is reached or
        :attr:`is_converged` becomes ``True``.

        Returns:
            The result of the last ``update()`` call.
        """
        result: Any = None
        for _ in range(self.max_iterations):
            result = self.update()
            if self.is_converged:
                break
        return result

    @property
    def convergence_history(self) -> list[float]:
        """Objective values recorded after each completed step."""
        return list(self._convergence_history)

    @property
    def best_value(self) -> float:
        """Best (lowest) objective value seen so far."""
        return self._best_value

    @property
    def is_converged(self) -> bool:
        """True when the iteration budget is exhausted.

        Subclasses may override this to implement earlier stopping criteria
        (e.g. plateau detection, gradient norm threshold).
        """
        return self._iteration >= self.max_iterations

    # ------------------------------------------------------------------
    # Internal bookkeeping (used by subclasses inside ``update``)
    # ------------------------------------------------------------------

    def _record(self, value: float) -> None:
        """Record a step result and update bookkeeping state.

        Args:
            value: Objective value for this step.
        """
        self._iteration += 1
        self._convergence_history.append(value)
        if value < self._best_value:
            self._best_value = value
