"""Pin both bare cuBLAS names to one self-consistent build, before CuPy binds them.

Why this exists
---------------
``import ao_shaping`` reaches OOPAO through ``ao_shaping.algorithm`` and
``ao_shaping.drivers.sim``, and OOPAO imports CuPy. On Windows the CUDA runtime
inside CuPy loads ``cublasLt64_13.dll`` **and** ``cublas64_13.dll`` *by bare name*,
so the loader walks ``PATH`` and binds whichever CUDA toolkit is registered there.
Torch ships its own copies in ``torch/lib``. Which build a bare name gets is
decided by *whoever enters the process first* -- and the failure is not "the PATH
build instead of torch's", it is **the two names landing on two different
builds**: a ``cublas64_13`` from build A calling into a ``cublasLt64_13`` from
build B.

Observed on this bench on 2026-11-03: the transfer-learning GUI session
``data/debug/slm_transfer/20261103_213400`` wrote ``fit/inputs.npz`` at 21:35:54
and then vanished -- ``iteration_0000.npz`` never appeared, the journal stopped at
``fit -> running``, and the Application event log recorded ``python.exe`` faulting
with ``0xC0000005`` inside
``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin\\x64\\cublasLt64_13.dll``
(``6.14.11.1340``) while torch 2.14.1+cu130 bundles ``6.14.11.1300``.

Measured on an RTX 5080 Laptop with torch 2.14.1+cu130 (``6.14.11.1300``), CUDA
toolkit v13.2 first on ``PATH`` (``6.14.11.1340``) and cupy-cuda13x 14.2.0 -- one
process per row, both bare names resolved the same way in each:

    preload                                   cupy matmul  cupy inv/pinv  torch matmul
    nothing (both names -> PATH 1340)         OK           OK             OK
    torch cublasLt only (commit 8014f579)     FAIL         FAIL           FAIL
    torch cublasLt+cublas (1300 pair)         OK           FAIL           OK
    toolkit cublasLt+cublas (1340 pair)       OK           OK             OK

The failures were ``CUBLAS_STATUS_INVALID_VALUE`` on the matmul and
``CUSOLVER_STATUS_INTERNAL_ERROR`` on ``inv``/``pinv``. Note the "nothing" row:
letting both names fall to ``PATH`` is *not* itself broken -- torch merely runs on
a newer cuBLAS than it was built against, which CUDA minor compatibility allows.
What breaks is a **mixed** pair.

Why the toolkit's pair, not torch's
-----------------------------------
``torch/lib`` ships only ``cublas64_13.dll``, ``cublasLt64_13.dll`` and
``cudart64_13.dll``. CuPy additionally needs cuSOLVER / cuFFT / cuRAND, which
exist **only** in a CUDA installation. So pinning torch's pair strands CuPy's
cuSOLVER on a different build than the cuBLAS it is calling (row 3: ``inv`` and
``pinv`` fail, matmul still works). The coherent set is therefore the toolkit's:
pin *both* bare names to it, and torch rides a newer cuBLAS through minor
compatibility -- exactly the ordering that never crashed (row 1). Pinning torch's
1300 pair would also break OOPAO, which really does use device-side linear
algebra (``libs/OOPAO/OOPAO/LiFT.py:321`` calls ``xp.linalg.pinv(...)`` with
``xp`` = cupy).

The one case where the toolkit must *not* win is when its cuBLAS is **older** than
the build torch was compiled against: then the pin falls back to torch's own pair
(which keeps torch correct) and logs a warning, because CuPy's cuSOLVER would then
come from that older toolkit and mix builds.

How
---
Windows keeps one module per loaded file name for bare-name lookups, so loading
the chosen copy **by full path** first makes every later ``cublasLt64_13`` /
``cublas64_13`` import resolve to it. Measured cost of pinning the pair: ~65 ms
per process. ``os.add_dll_directory(torch/lib)`` does *not* redirect CuPy's bare
name (measured: still binds the toolkit build); only a ``PATH`` entry does. A
``.venv`` and ``uv run`` do **not** scrub ``PATH`` either, so the hazard survives
both.

Scope, and what this cannot fix
-------------------------------
This hook only decides the outcome when CuPy enters **before** torch. If torch is
imported first, the bare names are already bound to torch's 1300 pair and CuPy's
cuSOLVER still comes from the toolkit -- the same mixed set as row 3, and the same
``CUSOLVER_STATUS_INTERNAL_ERROR``. The real cure is having a single CUDA minor in
the environment (install the toolkit matching torch's extra, or use the extra
matching the installed toolkit); see ``docs/environment.md``.

Import contract
---------------
Importing this module does nothing: no native library is loaded and nothing is
written to disk (``drivers/AGENTS.md``). It is **not** wired into
``ao_shaping/__init__.py`` -- only the two CuPy entry points call
:func:`pin_cublas`, and only at the moment they are about to import CuPy.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import util as _importlib_util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

__all__ = ["pin_cublas"]

_log = logging.getLogger(__name__)

#: torch ships one cuBLASLt per CUDA major: ``cublasLt64_13.dll`` for cu130.
_CUBLASLT_GLOB = "cublasLt64_*.dll"

#: Process-level latch: the bare names can be decided once, so the ``PATH`` scan,
#: the version probe and the loads all happen at most once per process.
_done = False


def _torch_origin() -> str | None:
    """``torch.__file__`` without importing torch; ``None`` when it is not installed."""
    loaded = sys.modules.get("torch")
    if loaded is not None:
        return getattr(loaded, "__file__", None)
    try:
        spec = _importlib_util.find_spec("torch")
    except (ImportError, ValueError):
        # ImportError: a broken install/import hook; ValueError: a half-initialised
        # parent whose __spec__ is None. Either way there is nothing to pin.
        return None
    return None if spec is None else spec.origin


def _cuda_suffix(lib_dir: Path) -> str | None:
    """``<M>`` of the newest ``cublasLt64_<M>.dll`` under ``lib_dir``, or ``None``.

    The suffix is shared with ``cublas64_<M>.dll`` -- that is what makes the two
    names comparable at all. A missing ``lib_dir`` globs to nothing, so callers
    need no ``is_dir`` check.
    """
    best: tuple[int, str] | None = None
    for candidate in lib_dir.glob(_CUBLASLT_GLOB):
        suffix = candidate.stem.rpartition("_")[2]
        if not suffix.isdigit():
            continue
        major = int(suffix)
        if best is None or major > best[0]:
            best = (major, suffix)
    return None if best is None else best[1]


def _toolkit_dir(cublaslt_name: str, search_path: Sequence[str]) -> Path | None:
    """First ``search_path`` entry holding ``cublaslt_name`` as a file, or ``None``.

    ``PATH`` order matters: the first directory wins, exactly as the Windows loader
    would resolve the bare name.
    """
    for entry in search_path:
        if not entry:
            continue
        directory = Path(entry)
        try:
            if (directory / cublaslt_name).is_file():
                return directory
        except OSError:
            # An unreadable or malformed PATH entry must not stop the scan.
            continue
    return None


def _file_version(path: Path) -> tuple[int, int, int, int] | None:
    """``(major, minor, patch, build)`` from a PE's fixed file info, or ``None``.

    Uses ``version.dll`` directly so nothing is loaded from the DLL being probed.
    Any failure -- no ``version.dll``, not a PE, unreadable file -- is reported as
    ``None``, which callers treat as "this build is not newer".
    """
    import ctypes
    from ctypes import wintypes

    class _FixedFileInfo(ctypes.Structure):
        _fields_ = [
            ("signature", wintypes.DWORD),
            ("struct", wintypes.DWORD),
            ("ms", wintypes.DWORD),
            ("ls", wintypes.DWORD),
        ]

    try:
        version = ctypes.windll.version
        query = str(path)
        size = version.GetFileVersionInfoSizeW(query, None)
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(query, 0, size, buffer):
            return None
        value = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not version.VerQueryValueW(buffer, "\\", ctypes.byref(value), ctypes.byref(length)):
            return None
        if not value.value:
            return None
        info = ctypes.cast(value, ctypes.POINTER(_FixedFileInfo)).contents
    except (AttributeError, OSError, ValueError):
        # AttributeError: no ``windll`` (not Windows). OSError/ValueError: the file
        # is not readable PE data or the query buffer did not survive the call.
        return None

    # The build number lives in the low half of ``ls``; vendors that use the high
    # bit as a flag store it in the remaining 31 bits instead.
    build = (info.ls & 0x7FFFFFFF) if info.ls & 0x80000000 else (info.ls & 0xFFFF)
    return (info.ms >> 16, info.ms & 0xFFFF, info.ls >> 16, build)


def _format_version(version: tuple[int, int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def pin_cublas(
    *,
    platform: str | None = None,
    loader: Callable[[str], object] | None = None,
    torch_origin: str | None = None,
    search_path: Sequence[str] | None = None,
    file_version: Callable[[Path], tuple[int, int, int, int] | None] | None = None,
) -> list[Path] | None:
    """Pin ``cublasLt64_<M>`` and ``cublas64_<M>`` to one coherent build.

    Loads the chosen pair by full path (Lt first, then cublas) so later bare-name
    imports in this process resolve to it. Returns the files loaded, in load
    order, or ``None`` when the hook did nothing.

    The keyword arguments exist so tests can exercise the decision without mapping
    hundreds of megabytes of DLL or probing real file versions: every production
    call site passes none of them.
    """
    global _done

    if (sys.platform if platform is None else platform) != "win32":
        return None
    if os.environ.get("AO_SKIP_CUBLAS_PRELOAD"):
        _log.debug("cuBLAS pinning skipped: AO_SKIP_CUBLAS_PRELOAD is set")
        return None
    if _done:
        _log.debug("cuBLAS pinning already ran in this process")
        return None
    _done = True

    try:
        return _pin(
            loader=loader,
            torch_origin=torch_origin,
            search_path=search_path,
            file_version=file_version,
        )
    except Exception as exc:
        # Never raise and never exit: the caller is in the middle of importing
        # CuPy, and a bench without a driver, a truncated install or an odd PATH
        # must still import the package.
        _log.debug("cuBLAS pinning failed: %s", exc)
        return None


def _pin(
    *,
    loader: Callable[[str], object] | None,
    torch_origin: str | None,
    search_path: Sequence[str] | None,
    file_version: Callable[[Path], tuple[int, int, int, int] | None] | None,
) -> list[Path] | None:
    """The guarded body of :func:`pin_cublas`; may raise, callers swallow."""
    origin = _torch_origin() if torch_origin is None else torch_origin
    if not origin:
        return None
    torch_lib = Path(origin).parent / "lib"
    suffix = _cuda_suffix(torch_lib)
    if suffix is None:
        return None

    cublaslt_name = f"cublasLt64_{suffix}.dll"
    cublas_name = f"cublas64_{suffix}.dll"

    if search_path is None:
        search_path = os.environ.get("PATH", "").split(os.pathsep)
    toolkit_lib = _toolkit_dir(cublaslt_name, search_path)

    source = torch_lib
    if toolkit_lib is not None:
        read_version = _file_version if file_version is None else file_version
        toolkit_version = read_version(toolkit_lib / cublaslt_name)
        torch_version = read_version(torch_lib / cublaslt_name)
        # An unreadable version is treated as "the toolkit is not older": that is
        # the ordering this bench has always run without a crash.
        if (
            toolkit_version is None
            or torch_version is None
            or toolkit_version >= torch_version
        ):
            source = toolkit_lib
            _log.debug("pinning cuBLAS to the CUDA toolkit at %s", toolkit_lib)
        else:
            _log.warning(
                "CUDA toolkit on PATH (%s, cuBLAS %s) is older than the cuBLAS torch "
                "was built against (%s); pinning torch's own pair instead. cupy's "
                "cuSOLVER still comes from that older toolkit, so its linalg may "
                "mix builds -- install the CUDA toolkit matching torch's extra, or "
                "use the extra matching the installed toolkit.",
                toolkit_lib,
                _format_version(toolkit_version),
                _format_version(torch_version),
            )

    if loader is None:
        import ctypes

        loader = ctypes.CDLL

    loaded: list[Path] = []
    # cuBLASLt first: ``cublas64`` binds it on load, so it must already be there.
    for name in (cublaslt_name, cublas_name):
        candidate = source / name
        if not candidate.is_file():
            _log.debug("%s is absent from %s; leaving that name alone", name, source)
            continue
        try:
            loader(str(candidate))
        except OSError as exc:
            # No NVIDIA driver, a bitness mismatch or a truncated install: the
            # caller's own imports must still work, so this is a log line.
            _log.debug("pinning %s failed: %s", candidate, exc)
            continue
        loaded.append(candidate)

    if loaded:
        _log.debug("pinned %d cuBLAS file(s) from %s", len(loaded), source)
    return loaded or None
