from __future__ import annotations

from typing import Any
from collections.abc import Callable

try:
    import numba
except ImportError:  # pragma: no cover - optional dependency
    numba = None


def _njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Return numba.njit if available, otherwise a no-op decorator."""

    if numba is None:
        def passthrough(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return passthrough

    return numba.njit(*args, **kwargs)
