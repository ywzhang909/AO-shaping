"""Optional-numba decorator shim shared by the algorithm and utils layers.

A zero-dependency leaf (stdlib + an optional ``numba`` import) so both layers
can use it without a cycle: ``utils`` must not import ``algorithm``, and
``algorithm`` legitimately sits above ``utils``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any

# Redirect numba's JIT cache out of the repository tree. ``@njit(cache=True)`` writes
# ``.nbi``/``.nbc`` next to the source module by default (``<pkg>/__pycache__``),
# which breaks the "import must not touch disk" contract
# (tests/ao_shaping/utils/test_import_makes_no_disk_writes.py) and fails on a
# read-only mount. ``ao_shaping/__init__.py`` already biases NUMBA_CACHE_DIR before
# any submodule import; this belt-and-suspenders copy covers the case where
# ``numba_compat`` is imported directly (it is the package's only numba entry point).
_cachedir = os.environ.get("NUMBA_CACHE_DIR", "")
if not _cachedir or not os.path.isabs(_cachedir):
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), "ao_shaping_numba_cache"
    )

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
