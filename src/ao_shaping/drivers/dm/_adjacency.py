"""Single loader for the DM actuator adjacency matrix.

The matrix used to be read twice, with divergent behaviour:

* ``drivers/dm/nlight/driver.py`` read ``"data/dm_adj.txt"`` in its **class body**,
  so the read happened at *import* time and a wrong working directory made
  ``import ao_shaping`` fail outright with ``FileNotFoundError``.
* ``drivers/sim/dm/simulated_dm.py`` read the same path in ``__init__`` but
  fell back to a synthetic grid when it was missing.

The path is resolved against the installed package rather than the process CWD.
``PATHS.root_dir`` is intentionally CWD-relative because it holds run *output*;
this file is a repository *input*, so it must not move with the caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

#: ``src/ao_shaping/drivers/dm/_adjacency.py`` -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
ADJACENCY_PATH: Path = _REPO_ROOT / "data" / "dm_adj.txt"

_ACTUATOR_COUNT = 64

#: ``np.loadtxt`` yields float64; the synthetic fallback yields int. Both are
#: consumed only via ``== 1`` / boolean indexing, so the cache is typed to the
#: common numeric supertype rather than coercing (which would change values).
_cache: npt.NDArray[np.number] | None = None


def _synthetic_grid(size: int = _ACTUATOR_COUNT) -> npt.NDArray[np.intp]:
    """A 4-neighbour grid, used only when the asset is genuinely unavailable."""
    side = int(np.sqrt(size))
    if side * side != size:
        raise ValueError(f"{size} is not a square grid")
    adj = np.zeros((size, size), dtype=int)
    for i in range(size):
        row, col = divmod(i, side)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < side and 0 <= c < side:
                adj[i, r * side + c] = 1
    return adj


def _cache_clear() -> None:
    """Drop the memoised matrix (used by tests that swap the path)."""
    global _cache
    _cache = None


def load_adjacency() -> npt.NDArray[np.number]:
    """Return the ``(64, 64)`` actuator adjacency matrix, loading it once.

    Resolution order: the packaged asset, then a CWD-relative ``data/`` copy for
    checkouts that keep it elsewhere, then a synthetic 4-neighbour grid. The last
    step means a missing asset degrades the neighbour-safety checks rather than
    making the driver unconstructable.
    """
    global _cache
    if _cache is not None:
        return _cache

    candidates = [ADJACENCY_PATH, Path("data") / "dm_adj.txt"]
    for candidate in candidates:
        try:
            matrix = np.loadtxt(candidate)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        _cache = matrix
        return _cache

    _cache = _synthetic_grid()
    return _cache


class lazy_adjacency:
    """Descriptor deferring the matrix read to first attribute access.

    Assigning the result of :func:`load_adjacency` in a class body would read the
    file during import. A descriptor keeps ``NLight.Units_Adj_Mat`` and
    ``self.Units_Adj_Mat`` working while moving the I/O to first use.
    """

    def __get__(
        self, obj: object, objtype: type | None = None
    ) -> npt.NDArray[np.number]:
        return load_adjacency()


__all__ = ["ADJACENCY_PATH", "lazy_adjacency", "load_adjacency"]
