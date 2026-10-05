from __future__ import annotations

from typing import Any
from collections.abc import Callable

try:
    import numba
except ImportError:  # pragma: no cover - 可选依赖
    numba = None


def _njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """若 numba 可用则返回 numba.njit, 否则返回一个什么都不做的装饰器。"""

    if numba is None:
        def passthrough(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return passthrough

    return numba.njit(*args, **kwargs)
