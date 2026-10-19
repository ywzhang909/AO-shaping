"""Optional-numba decorator shim shared by the algorithm and utils layers.

A zero-dependency leaf (stdlib + an optional ``numba`` import) so both layers
can use it without a cycle: ``utils`` must not import ``algorithm``, and
``algorithm`` legitimately sits above ``utils``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import numba
except ImportError:  # pragma: no cover - optional dependency
    numba = None


def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Return ``numba.njit`` if available, otherwise a pass-through decorator."""

    if numba is None:
        def passthrough(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return passthrough

    return numba.njit(*args, **kwargs)
