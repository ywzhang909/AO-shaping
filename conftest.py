"""Resource guards for the test session.

Why this exists
---------------
The host has 384 cores. Left alone, the numeric stack sizes every thread pool
at that number, and an ordinary ``pytest`` run saturates the machine badly
enough that even ``ps`` and ``pkill`` time out.

**CPU threads.** ``docker run --cpus=16`` is a CFS quota, not a cpuset: it
throttles aggregate CPU time but leaves ``os.cpu_count()`` and
``multiprocessing.cpu_count()`` reporting 384. Libraries that derive a pool size
from those still open hundreds of threads that merely queue. The vendored OOPAO
is the worst offender -- it hands ``multiprocessing.cpu_count()`` straight to
pyfftw's ``threads=`` and to joblib's ``n_jobs``. Measured on this box, a 2048^2
complex128 FFT x3: ``threads=384`` -> 1957 ms, ``threads=64`` -> 310 ms,
``threads=16`` -> 55 ms, and the pool is a high-water mark that never shrinks
within the process. A GPU does not help here; this is pure CPU work.

The env vars baked into ``docker/Dockerfile.test`` are the primary control.
These calls are the defence in depth for a run that lacks them, and the native
pools have to be set explicitly because they captured their sizes at import time.

**GPU memory.** ``docker run --gpus`` cannot cap VRAM. PyTorch's caching
allocator is what holds the repo's tensors, so the ceiling is applied there.

Override with ``AO_CPU_THREADS`` / ``AO_VRAM_LIMIT_MB``.
"""

from __future__ import annotations

import faulthandler
import multiprocessing
import os
import threading
import time

DEFAULT_CPU_THREADS = 16
DEFAULT_VRAM_LIMIT_MB = 4096

_AO_CPU_THREADS = int(os.environ.get("AO_CPU_THREADS", str(DEFAULT_CPU_THREADS)))
_AO_VRAM_LIMIT_MB = int(
    os.environ.get("AO_VRAM_LIMIT_MB", str(DEFAULT_VRAM_LIMIT_MB))
)
WATCH = {"peak": 0, "stop": False}

# Above this the run is already pathological; --pids-limit would have killed the
# container first, so reaching it means the wrapper was bypassed.
THREAD_ALARM = 4000


def _os_thread_count() -> int | None:
    """Live OS thread count, or ``None`` where ``/proc`` does not exist.

    ``/proc/self/status`` is Linux-only. Returning ``None`` rather than raising
    keeps the Windows callers honest instead of letting the read escape as an
    unhandled ``FileNotFoundError`` -- which it did from
    ``pytest_sessionfinish``, aborting the session *after* the tests had already
    run and thereby suppressing pytest's whole summary (counts, failure
    list, exit code). That made an otherwise-green run indistinguishable from a
    broken one on any non-Linux checkout.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return 0


def _monitor() -> None:
    while not WATCH["stop"]:
        count = _os_thread_count()
        if count is not None:
            WATCH["peak"] = max(WATCH["peak"], count)
        time.sleep(5)


def _cap_cpu_threads(threads: int) -> None:
    multiprocessing.cpu_count = lambda: threads  # type: ignore[assignment]
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(threads))

    try:
        import numba

        numba.set_num_threads(threads)
    except Exception:  # pragma: no cover - numba is optional
        pass

    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover - torch is optional
        pass


def _cap_gpu_memory(limit_mb: int) -> None:
    limit_bytes = limit_mb * 1024 * 1024

    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(limit_bytes / total, 0)
    except Exception:  # pragma: no cover - no GPU in some environments
        pass

    try:
        import cupy as cp

        if cp.cuda.is_available():
            cp.cuda.runtime.deviceSetLimit(
                cp.cuda.runtime.cudaLimitMallocHeapSize, limit_bytes
            )
    except Exception:  # pragma: no cover - cupy is optional
        pass


def _gpu_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "GPU unavailable (CPU-only run)"
        props = torch.cuda.get_device_properties(0)
        return (
            f"GPU {props.name}, {props.total_memory / 1024 ** 3:.1f} GiB total, "
            f"allocator capped at {_AO_VRAM_LIMIT_MB} MiB"
        )
    except Exception:  # pragma: no cover - torch is optional
        return "torch unavailable"


def pytest_configure(config) -> None:
    _cap_cpu_threads(_AO_CPU_THREADS)
    _cap_gpu_memory(_AO_VRAM_LIMIT_MB)

    threading.Thread(target=_monitor, daemon=True, name="ao-thread-watch").start()

    dump = os.environ.get("AO_FAULT_HANDLER")
    if dump:
        # Periodic all-thread stack dump: how a thread explosion gets attributed
        # to a creator rather than merely measured.
        stream = open(dump, "a", buffering=1)
        faulthandler.dump_traceback_later(
            int(os.environ.get("AO_FAULT_SECS", "300")), repeat=True, file=stream
        )


def pytest_report_header(config) -> list[str]:
    return [
        f"ao-shaping: CPU thread cap = {_AO_CPU_THREADS}",
        f"ao-shaping: {_gpu_summary()}",
    ]


def pytest_sessionfinish(session, exitstatus) -> None:
    WATCH["stop"] = True
    peak, now = WATCH["peak"], _os_thread_count()
    where = "" if now is not None else " (unavailable: no /proc on this platform)"
    print(
        f"\n[ao-watch] peak OS threads = {peak}, now = {now}{where}, "
        f"AO_CPU_THREADS = {_AO_CPU_THREADS}, "
        f"AO_VRAM_LIMIT_MB = {_AO_VRAM_LIMIT_MB}"
    )
    if peak > THREAD_ALARM:
        print(
            f"[ao-watch] WARNING: peak thread count above {THREAD_ALARM}. Re-run "
            "with AO_FAULT_HANDLER=/tmp/ao-threads.log to capture the stacks "
            "that create them."
        )