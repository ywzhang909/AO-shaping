"""Pin cuBLASLt to the copy torch ships, before CuPy can bind the name elsewhere.

Why this exists
---------------
``import ao_shaping`` reaches OOPAO through ``ao_shaping.algorithm`` and
``ao_shaping.drivers.sim``, and OOPAO imports CuPy. On Windows the CUDA runtime
inside CuPy loads ``cublasLt64_13.dll`` **by bare name**, so the loader walks
``PATH`` and takes whichever CUDA toolkit is registered there. Torch ships its
own ``cublasLt64_13.dll`` in ``torch/lib`` and its ``cublas64_13.dll`` binds that
bare name to whatever copy is *already loaded*. Put a toolkit copy on ``PATH``
(a dev bench with CUDA v13.2 installed) and torch ends up calling a cuBLASLt it
was not built against: an access violation (``0xc0000005``) inside
``cublasLt64_13.dll`` on the first sizeable CUDA matmul, with no Python
traceback and no chance to flush anything.

Observed on this bench on 2026-11-03: the transfer-learning GUI session
``data/debug/slm_transfer/20261103_213400`` wrote ``fit/inputs.npz`` at 21:35:54
and then vanished -- ``iteration_0000.npz`` never appeared, the journal stopped at
``fit -> running``, and the Application event log recorded
``python.exe`` faulting in
``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin\\x64\\cublasLt64_13.dll``
(``6.14.11.1340``) while torch 2.14.1+cu130 bundles ``6.14.11.1300``. Reproduced
headless from those recorded inputs: the process died in the first
``fit_loss`` -> ``model.prediction()``, and ``--preload`` before the crash point
made the same call succeed.

How
---
Windows keeps one module per loaded file name for bare-name lookups, so loading
torch's copy **by full path** first makes every later ``cublasLt64_13`` import
resolve to it; CuPy then calls the copy torch was built against. Measured cost:
~50 ms per process, and CuPy keeps working -- its ``linalg.inv`` still agrees
with numpy to 2.97e-07 after the preload.

Why it is safe to do at package-import time
-------------------------------------------
This runs from ``ao_shaping/__init__.py``, the only hook that is guaranteed to
precede every ``ao_shaping.*`` submodule (the import contract there, R-20, keeps
submodules lazy). It stays within that contract: nothing heavy is imported
(``ctypes`` is imported only when a load is actually attempted), torch is located
with ``importlib.util.find_spec`` and never imported, and every guard below
degrades to "do nothing" -- a machine without torch, without a driver, or with a
partial install still imports the package.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import util as _importlib_util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, and this module is imported at startup
    from collections.abc import Callable

__all__ = ["preload_torch_cublaslt"]

_log = logging.getLogger(__name__)

#: torch ships one cuBLASLt per CUDA major: ``cublasLt64_13.dll`` for cu130.
_CUBLASLT_GLOB = "cublasLt64_*.dll"


def _torch_origin() -> str | None:
    """``torch.__file__`` without importing torch; ``None`` when it is not installed."""
    loaded = sys.modules.get("torch")
    if loaded is not None:
        return getattr(loaded, "__file__", None)
    try:
        spec = _importlib_util.find_spec("torch")
    except (ImportError, ValueError):
        # ImportError: a broken install/import hook; ValueError: a half-initialised
        # parent whose __spec__ is None. Either way there is nothing to preload.
        return None
    return None if spec is None else spec.origin


def _pick_cublaslt(lib_dir: Path) -> Path | None:
    """Newest ``cublasLt64_<cuda-major>.dll`` under ``lib_dir``, or ``None``.

    A missing ``lib_dir`` globs to nothing, so callers need no ``is_dir`` check.
    """
    best: tuple[int, Path] | None = None
    for candidate in lib_dir.glob(_CUBLASLT_GLOB):
        suffix = candidate.stem.rpartition("_")[2]
        if not suffix.isdigit():
            continue
        major = int(suffix)
        if best is None or major > best[0]:
            best = (major, candidate)
    return None if best is None else best[1]


def preload_torch_cublaslt(
    *,
    platform: str | None = None,
    loader: Callable[[str], object] | None = None,
    torch_origin: str | None = None,
) -> Path | None:
    """Load torch's bundled cuBLASLt by full path; return it, or ``None`` if skipped.

    ``platform``, ``loader`` and ``torch_origin`` exist so tests can exercise the
    decision and the path resolution without mapping a 456 MB DLL: the production
    call site passes none of them.
    """
    if (sys.platform if platform is None else platform) != "win32":
        return None
    if os.environ.get("AO_SKIP_CUBLAS_PRELOAD"):
        _log.debug("cuBLASLt preload skipped: AO_SKIP_CUBLAS_PRELOAD is set")
        return None

    origin = _torch_origin() if torch_origin is None else torch_origin
    if not origin:
        return None
    bundled = _pick_cublaslt(Path(origin).parent / "lib")
    if bundled is None:
        return None

    if loader is None:
        import ctypes

        loader = ctypes.CDLL
    try:
        loader(str(bundled))
    except OSError as exc:
        # No NVIDIA driver, a 32/64-bit mismatch, or a truncated install: the
        # caller's own imports must still work, so this is a log line, not a raise.
        _log.debug("cuBLASLt preload failed for %s: %s", bundled, exc)
        return None

    _log.debug("preloaded torch's cuBLASLt: %s", bundled)
    return bundled
