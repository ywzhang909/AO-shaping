"""Pytest configuration and shared fixtures for AO-Shaping tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Thread budget. MUST stay above every numpy/torch/numba import in this file
# and in every test module: a native pool sizes itself once, at load time, and
# a later `os.environ` write cannot shrink it.
#
# The budget is a *whole-run* figure that fan-out divides, not a per-pool one.
# 16 cores is spent as `16 // workers` threads per native pool, so
#     pytest            -> one process, 16 threads per pool
#     pytest -n 16      -> 16 processes x 1 thread  = 16 cores, exactly
#     pytest -n 4       ->  4 processes x 4 threads = 16 cores, exactly
# which is what "at most 16 cores" has to mean: a flat "16 per pool" would let
# one serial process ask for 16 (BLAS) + 16 (torch intra) + 16 (torch inter).
# Override with AO_TEST_THREADS=<n>, which then wins over every var below.
#
# Each var is load-bearing for a *different* runtime, so a plausible-looking
# subset here is silently useless:
#   * numpy 2.4 ships scipy-openblas (see `np.show_config()`) whose OpenBLAS is
#     a pthread build -- it reads OPENBLAS_NUM_THREADS, so OMP_NUM_THREADS
#     alone caps nothing.
#   * torch's intra-op *default* reads OMP_NUM_THREADS then MKL_NUM_THREADS
#     (aten ParallelCommon.cpp :: intraop_default_num_threads), and on x86-64
#     MKL_NUM_THREADS *overrides* the former -- hence both, at the same value.
#   * MKL_DYNAMIC defaults to TRUE, i.e. MKL is free to ignore MKL_NUM_THREADS.
#   * numba's pool is its own; OMP_NUM_THREADS does not govern it.
#   * CuPy has no CPU thread-pool env var -- its documented list has none, its
#     "threads" are CUDA grid/block, and CUPY_NUM_NVCC_THREADS is build-time.
#     There is nothing to pin for it; do not invent one.
#
# Without AO_TEST_THREADS the vars are `setdefault`ed, so a developer who
# exports a value for one BLAS-bound test keeps it.
#
# Before this, on this machine (32 logical CPUs), a single pytest process asked
# for 24 (OpenBLAS) + 24 (torch intra-op) + 24 (torch inter-op) = 72, and every
# extra worker multiplied it again. The 30x wall-clock swing that contention
# alone produced was measured at
# tests/ao_shaping/optimizer/wfless/test_model_in_loop_shaping.py, which pinned
# torch locally before this policy generalised it to the whole suite. Note that
# measurement also found torch's pool fastest at 1 thread for this suite's
# 64x64 work -- i.e. for small arrays 16 threads buys nothing and costs
# spin-wait, which is why the budget is worth spending on worker count
# (`pytest -n 16`) rather than on pool width.
# ---------------------------------------------------------------------------
#: Cores the whole run may use. 16 sits half-way up this machine's 32 logical
#: CPUs: enough to keep the machine busy, far enough from the top that a pool
#: plus the scheduler plus an IDE cannot tip it into thrashing.
CORE_BUDGET = 16

_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
)


def _xdist_workers() -> int:
    """Parallel pytest workers in this run (1 when xdist is not in use).

    xdist exports ``PYTEST_XDIST_WORKER_COUNT`` to every worker, so the divisor
    is the same number in all of them -- which is what keeps the product
    (workers x threads) equal to the budget instead of drifting per worker.
    """
    raw = os.environ.get("PYTEST_XDIST_WORKER_COUNT") or "1"
    return max(1, int(raw))


def _pool_threads() -> int:
    """Threads allowed per native pool per process under the current fan-out."""
    total = max(1, int(os.environ.get("AO_TEST_THREADS") or CORE_BUDGET))
    return max(1, total // _xdist_workers())


_OVERRIDE = "AO_TEST_THREADS" in os.environ
for _thread_env_var in _THREAD_ENV_VARS:
    if _OVERRIDE:
        os.environ[_thread_env_var] = str(_pool_threads())
    else:
        os.environ.setdefault(_thread_env_var, str(_pool_threads()))
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

# Below the pinning block on purpose -- see above.
import numpy as np


def _pin_torch_threads() -> tuple[int, int] | None:
    """Cap torch's two pools, which the env vars alone do not fully reach.

    Returns the ``(intra_op, inter_op)`` counts actually in force, or None when
    torch is absent -- it is an optional extra, and a torch-less install must
    still be able to run the suite.
    """
    if importlib.util.find_spec("torch") is None:
        return None

    import torch

    # The intra-op default honours OMP_NUM_THREADS/MKL_NUM_THREADS, but if any
    # plugin imported torch and sized the pool before this conftest ran, that
    # default is already spent -- so pin it through the API as well. Inter-op
    # is a coordination pool that is idle outside JIT, so it stays at 1 rather
    # than claiming a share of the budget it will never use.
    torch.set_num_threads(_pool_threads())
    try:
        # One-shot by design: raises once inter-op parallel work has started in
        # this process. Those vars never reached this pool, so this is the only
        # handle on it; a late failure is reported, not hidden.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.get_num_threads(), torch.get_num_interop_threads()


_TORCH_POOLS = _pin_torch_threads()
if _TORCH_POOLS is not None:
    import torch


@pytest.fixture(autouse=True)
def _enforce_thread_budget():
    """Re-assert the process-wide thread budget before every test.

    Tests may legitimately shrink torch's pools for determinism, but the budget
    is a process-wide invariant (asserted by
    ``tests/ao_shaping/test_conventions.py::test_the_thread_budget_is_in_force_in_this_process``).
    Without this, that assertion measures which module happened to run last
    instead of whether the suite honours the budget: two modules that pin 1 and 4
    threads leave the process at 4 and redden a later test that never touched the
    pools. Re-pinning costs two attribute reads and makes a leak local to one test.
    """
    threads = _pool_threads()
    for var in _THREAD_ENV_VARS:
        if os.environ.get(var) != str(threads):
            os.environ[var] = str(threads)
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    if _TORCH_POOLS is not None and torch.get_num_threads() != threads:
        torch.set_num_threads(threads)


def pytest_report_header() -> list[str]:
    """Make an otherwise invisible policy visible at the top of every run."""
    workers = _xdist_workers()
    header = [
        f"thread budget: {CORE_BUDGET} cores -> {workers} worker(s) x "
        f"{_pool_threads()} thread(s) per native pool"
        f"{' [AO_TEST_THREADS override]' if _OVERRIDE else ''}"
    ]
    if _TORCH_POOLS is None:
        header.append("torch: not installed, only the env vars apply")
    else:
        intra, inter = _TORCH_POOLS
        header.append(f"torch pools: intra-op={intra}, inter-op={inter}")
    return header


@pytest.fixture
def dm_n_actuators():
    """Number of DM actuators."""
    return 64


@pytest.fixture
def init_voltages(dm_n_actuators):
    """Initial zero voltages for DM."""
    return [0.0] * dm_n_actuators


@pytest.fixture
def dm_unit_mask_all(dm_n_actuators):
    """DM unit mask with all actuators enabled except disabled ones."""
    mask = np.ones(dm_n_actuators, dtype=bool)
    mask[0] = False
    return mask


@pytest.fixture
def dm_unit_mask_inner():
    """DM unit mask with only inner actuators enabled."""
    mask = np.ones(64, dtype=bool)
    mask[0] = False
    mask[21:] = False
    return mask


@pytest.fixture
def dm_unit_mask_outer():
    """DM unit mask with only outer actuators enabled."""
    mask = np.ones(64, dtype=bool)
    mask[0] = False
    mask[:39] = False
    return mask


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.rand(200, 200).astype(np.float32)


@pytest.fixture
def sample_wavefront():
    """Create a sample wavefront for testing."""
    return np.random.rand(64, 64).astype(np.float32)