"""Resource guards for the test session.

Why this exists
---------------
The host has 384 cores. Left alone, the numeric stack sizes every thread pool
at that number, and an ordinary ``pytest`` run saturates the machine badly
enough that even ``ps`` and ``pkill`` time out.

**Running on the host at all is refused.** Thread and VRAM caps cannot undo
that: they bound how much each pool grows, not how many pools exist, and the
host has no cgroup to fail against. On 2026-10-21 an unconstrained run on this
box froze the machine and it needed a forced power cycle. ``run_pytest.sh`` is
the sanctioned entry point; :func:`_enforce_containerized_run` makes a bare
``pytest`` fail with a pointer to it.

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
    """Live OS thread count, or ``None`` when nothing can supply one.

    ``/proc/self/status`` is Linux-only. Returning ``None`` straight away off
    Linux stopped the crash but also **disarmed the watch**: ``_monitor`` skips
    ``None`` counts, so ``WATCH["peak"]`` stayed at ``0`` on Windows,
    ``THREAD_ALARM`` could never fire, and the teardown line always read
    ``peak = 0``. ``psutil`` reports the same quantity there, so the count is now
    real on every platform and ``None`` means only "no source available".

    Never raises. The read escaping as an unhandled ``FileNotFoundError`` from
    ``pytest_sessionfinish`` aborted the session *after* the tests had already
    run, suppressing pytest's whole summary (counts, failure list, exit code)
    and making an otherwise-green run indistinguishable from a broken one.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass  # no /proc on this platform -- fall through to psutil

    try:
        import psutil

        return int(psutil.Process().num_threads())
    except ImportError:  # pragma: no cover - psutil is an optional extra
        return None


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
    except (ImportError, RuntimeError):  # pragma: no cover - numba optional; set may fail if already started
        pass

    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):  # pragma: no cover - torch optional; interop set raises RuntimeError if already set
        pass


def _cap_gpu_memory(limit_mb: int) -> None:
    limit_bytes = limit_mb * 1024 * 1024

    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(limit_bytes / total, 0)
    except (ImportError, RuntimeError):  # pragma: no cover - no GPU / torch optional
        pass

    try:
        import cupy as cp

        if cp.cuda.is_available():
            cp.cuda.runtime.deviceSetLimit(
                cp.cuda.runtime.cudaLimitMallocHeapSize, limit_bytes
            )
    except (ImportError, RuntimeError):  # pragma: no cover - cupy optional / no GPU
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
    except (ImportError, RuntimeError):  # pragma: no cover - torch optional / no GPU
        return "torch unavailable"


def _in_test_container() -> bool:
    """Whether this process runs inside the sanctioned test container.

    ``run_pytest.sh`` exports ``AO_PYTEST_ROOT``, which is the primary
    signal: it is set on the ``docker run`` command line and nowhere else.
    The cgroup and ``/.dockerenv`` checks are the fallback for a run that
    reached the container by some other route.

    The host is not a container, so a bare ``pytest`` here has to be
    refused -- see :func:`_enforce_containerized_run`.
    """
    if os.environ.get("AO_PYTEST_ROOT"):
        return True
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            cgroup = handle.read()
    except OSError:
        return False
    return "/docker-" in cgroup or "/docker/" in cgroup or "docker/" in cgroup


def _enforce_containerized_run() -> None:
    """Refuse to run the suite on the host; direct the caller to the wrapper.

    The host has 384 cores and two RTX 4090s, and an unconstrained run there
    is what takes the machine down: numeric libraries size their pools at
    384, the vendored OOPAO hands ``multiprocessing.cpu_count()`` to
    pyfftw and joblib, and the first symptom is not a failed test but the
    shell itself timing out. On 2026-10-21 that is exactly what happened --
    the host froze and needed a forced power cycle.

    A host run is therefore refused rather than merely discouraged. The
    escape hatch exists for the cases where a container genuinely cannot
    run: a CI runner (no nested docker), and a developer who needs one
    test file interactively.
    """
    if os.environ.get("AO_ALLOW_HOST_PYTEST") == "1":
        return
    if _in_test_container():
        return
    if os.environ.get("CI") == "true":
        return

    import pytest

    raise pytest.UsageError(
        "Refusing to run the test suite directly on the host.\n"
        "\n"
        "This machine has 384 cores; an unconstrained run saturates it and can\n"
        "freeze the host (it did on 2026-10-21). Use the wrapper, which caps\n"
        "CPU, memory and pids:\n"
        "\n"
        "    ./run_pytest.sh                          # whole suite\n"
        "    ./run_pytest.sh tests/ao_shaping/runners  # a subtree\n"
        "    ./run_pytest.sh -k 'not slow' -- -x      # extra args after --\n"
        "\n"
        "For a single test file during development, or on a CI runner with no\n"
        "docker, set AO_ALLOW_HOST_PYTEST=1 (thread and VRAM caps still apply)."
    )


def pytest_configure(config) -> None:
    _enforce_containerized_run()
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