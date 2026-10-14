"""Import purity tests (TDD RED step for the import-time network I/O bug).

Pins the contract in ``src/ao_shaping/drivers/AGENTS.md`` §"驱动惰性加载契约
(2026-10-01)": ``import ao_shaping`` and every subpackage must load no native
SDK and must perform no network I/O.

The socket-counting probes run in **subprocesses** on purpose: ``ao_shaping``
is already present in ``sys.modules`` inside the pytest process (the in-process
guard would be vacuous), and a subprocess also gets a pristine interpreter whose
``sys.modules`` is empty before the import under test.

Wall-clock note: this file is slow by design. Each probe subprocess pays a full
re-import of the package, and while the bug is unfixed every one of those
imports additionally blocks on the sequential 1 s-per-host TCP reachability
sweep in ``MicroDM.is_reachable``. Expect roughly 90 s for the whole file.

The probes deliberately call through to the *real* ``socket.socket.connect_ex``
so the measured ``elapsed`` reflects genuine blocking behaviour rather than an
artificially fast stub; only the *count* of connection attempts is what the
assertions are about.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.drivers.dm._registry import list_dm_types
from tests.ao_shaping.drivers._subprocess_env import subprocess_env_with_pythonpath

# Bound generous enough for a slow import on a loaded box; the assertions below
# are about connect counts and elapsed time, not about the timeout.
SUBPROCESS_TIMEOUT_S = 180.0

# Preamble shared by every probe: wrap ``socket.socket.connect_ex`` so each
# attempt is appended to ``connects`` while still calling through to the real
# implementation (keeping the blocking behaviour intact).
_COUNTING_PREAMBLE = """
import socket
import sys

connects = []
_real_connect_ex = socket.socket.connect_ex


def _counting_connect_ex(self, address):
    connects.append(address)
    return _real_connect_ex(self, address)


socket.socket.connect_ex = _counting_connect_ex
"""


def _repo_root() -> Path:
    """Return the repository root derived from this test file's location.

    ``parents[2]``, not ``parents[3]``: this file sits at ``tests/ao_shaping/``, so
    parents[0] is ``ao_shaping``, parents[1] is ``tests`` and parents[2] is the root.
    The off-by-one handed ``/`` to ``PYTHONPATH`` and to ``cwd``, so every child
    interpreter in this file died with ModuleNotFoundError before running a line of
    the snippet under test -- which is why six tests here could report a contract
    violation that was purely the harness. ``_subprocess_env`` records the same
    mistake in two other files.
    """
    return Path(__file__).resolve().parents[2]


def _src_dir() -> Path:
    """Return the ``src`` directory holding the ``ao_shaping`` package."""
    return _repo_root() / "src"


def _run_python(code: str, timeout: float = SUBPROCESS_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with ``src`` on ``PYTHONPATH``.

    pytest's ``pythonpath`` ini option only affects the pytest process itself,
    so the environment must be handed to the child explicitly.

    Args:
        code: Python source to execute with ``-c``.
        timeout: Subprocess timeout in seconds.

    Returns:
        The completed process, with captured stdout/stderr.
    """
    env = subprocess_env_with_pythonpath(("src",), absolute=True, root=str(_repo_root()))
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_repo_root()),
        timeout=timeout,
    )


def _stderr_tail(text: str, limit: int = 2000) -> str:
    """Return the tail of ``text`` for inclusion in failure messages."""
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def _probe_import(import_stmt: str, timeout: float = SUBPROCESS_TIMEOUT_S) -> tuple[float, int]:
    """Import ``import_stmt`` in a subprocess and report connects + elapsed.

    The probe installs the counting preamble *before* the import under test,
    then emits a machine-readable line on stderr::

        PROBE elapsed=<float> connects=<int>

    Args:
        import_stmt: The import statement to exercise, e.g. ``"import ao_shaping"``.
        timeout: Subprocess timeout in seconds.

    Returns:
        Tuple of ``(elapsed_seconds, connect_attempts)``.

    Raises:
        AssertionError: If the subprocess failed or its output is unparseable.
    """
    code = (
        _COUNTING_PREAMBLE
        + f"""
import time

_start = time.perf_counter()
{import_stmt}
_elapsed = time.perf_counter() - _start

sys.stderr.write("PROBE elapsed={{:.6f}} connects={{}}\\n".format(_elapsed, len(connects)))
"""
    )
    proc = _run_python(code, timeout=timeout)

    assert proc.returncode == 0, (
        f"probe subprocess exited with {proc.returncode}\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )

    match = re.search(r"PROBE\s+elapsed=([0-9.]+)\s+connects=(\d+)", proc.stderr)
    assert match is not None, (
        f"could not parse PROBE line from probe stderr\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )
    return float(match.group(1)), int(match.group(2))


@pytest.mark.parametrize(
    "import_stmt",
    [
        "import ao_shaping",
        "import ao_shaping.config",
        "import ao_shaping.runners.runner_common",
    ],
    ids=["pkg", "config", "runner_common"],
)
def test_import_performs_zero_socket_connects(import_stmt: str) -> None:
    """Importing must not open a single TCP connection, and must be quick."""
    elapsed, connects = _probe_import(import_stmt)
    assert connects == 0, (
        f"expected 0 socket.connect_ex calls during import, got {connects} "
        f"(import also took {elapsed:.2f}s)"
    )
    assert elapsed < 5.0, f"import took {elapsed:.3f}s, expected < 5.0s (connects={connects})"


def test_dm_n_actuators_is_64_without_hardware() -> None:
    """``DM_N_ACTUATORS`` must report the 64-channel default with no hardware."""
    code = (
        _COUNTING_PREAMBLE
        + """
import ao_shaping.config

assert ao_shaping.config.DM_N_ACTUATORS == 64, ao_shaping.config.DM_N_ACTUATORS
"""
    )
    proc = _run_python(code)
    assert proc.returncode == 0, (
        f"DM_N_ACTUATORS is not 64 without hardware (rc={proc.returncode})\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )


def test_dm_disabled_actuators_is_single_zero() -> None:
    """``DM_DISABLED_ACTUATORS`` must fall back to ``[0]`` with no hardware."""
    code = (
        _COUNTING_PREAMBLE
        + """
import ao_shaping.config

assert ao_shaping.config.DM_DISABLED_ACTUATORS == [0], ao_shaping.config.DM_DISABLED_ACTUATORS
"""
    )
    proc = _run_python(code)
    assert proc.returncode == 0, (
        f"DM_DISABLED_ACTUATORS is not [0] without hardware (rc={proc.returncode})\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )


def test_list_dm_types_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cheap registry listing must stay network-free.

    In-process on purpose: this pins the *cheap* API as the one that must never
    need a socket, and it guards against someone "optimising" the import-time
    resolution by moving the reachability sweep here instead.

    Asserts a SUBSET, never an exact length: the registry gains entries purely as
    an import side effect (``runner_common`` imports ``asyn_micro_dm``, which
    registers ``asyn_micro``), so the count is 6 in a bare process and 7 once
    ``runner_common`` has been imported. That import-order sensitivity is the very
    hazard documented for the ``--dm-type`` choice list, and pinning a count here
    would have made this test order-dependent.
    """
    import socket

    def _boom(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("list_dm_types() must not open a socket")

    monkeypatch.setattr(socket.socket, "connect_ex", _boom)
    types = list_dm_types()
    always_present = {"hadamard", "micro", "nlight", "sim", "sim_micro", "zernike"}
    assert always_present <= set(types), f"missing DM types: {always_present - set(types)}"
    assert types == sorted(set(types)), f"registry listing must be sorted and unique: {types}"


def test_dm_resolution_is_memoized() -> None:
    """Successive ``DM_N_ACTUATORS`` reads must not re-run the probe round.

    ``_resolve_dm_n_actuators()`` is currently uncached, so every attribute
    access re-runs ``list_reachable_types()`` and its full TCP sweep. The
    second read must add zero *new* connection attempts.
    """
    code = (
        _COUNTING_PREAMBLE
        + """
import ao_shaping.config

_first = ao_shaping.config.DM_N_ACTUATORS
_after_first = len(connects)

_second = ao_shaping.config.DM_N_ACTUATORS
_after_second = len(connects)

sys.stderr.write("MEMO first={} second={}\\n".format(_after_first, _after_second))
assert _after_second == _after_first, "second access performed extra socket probes"
"""
    )
    proc = _run_python(code)
    assert proc.returncode == 0, (
        f"DM_N_ACTUATORS resolution is not memoized (rc={proc.returncode})\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )

#: Modules that ``import ao_shaping`` must not drag in. R-20 as written in
#: ``utils/__init__.py`` names numpy / matplotlib / aotools and "any optional
#: SDK"; pandas and scipy are included because they were measured coming in too,
#: and aotools does not exist in this tree. ``gxipy`` is the native SDK the
#: drivers contract already forbids.
_HEAVY_IMPORT_CONTRACT = (
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "pandas",
    "scipy",
    "gxipy",
)


@pytest.mark.parametrize("import_stmt", ["import ao_shaping", "import ao_shaping.utils"])
def test_import_does_not_drag_heavy_dependencies(import_stmt: str) -> None:
    """R-20: the package root must not import numpy/matplotlib/pandas/scipy.

    This is the half of the contract that ``test_import_performs_zero_socket_connects``
    does not cover. That one counts TCP attempts; this one counts *modules*, and
    it is the reason ``ao_shaping/__init__.py`` now resolves its 46 re-exports
    through PEP 562 instead of importing them eagerly -- Python initialises the
    root package before any submodule, so its eager imports were charged to every
    ``import ao_shaping.<anything>`` in the repo, offline tooling included.

    Subprocess, for the same reason as the socket probes: ``ao_shaping`` is
    already in ``sys.modules`` here, so an in-process check would be vacuous.
    """
    code = (
        "import sys\n"
        f"{import_stmt}\n"
        f"dragged = [m for m in {list(_HEAVY_IMPORT_CONTRACT)!r} if m in sys.modules]\n"
        "sys.stderr.write('DRAGGED ' + repr(dragged) + '\\n')\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, (
        f"{import_stmt} failed (rc={proc.returncode})\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )
    match = re.search(r"DRAGGED (\[.*?\])", proc.stderr)
    assert match is not None, f"probe produced no verdict:\n{_stderr_tail(proc.stderr)}"
    dragged = ast.literal_eval(match.group(1))
    assert not dragged, (
        f"{import_stmt} dragged in {dragged}. The root package must resolve its "
        "public API lazily (PEP 562); an eager `from ao_shaping... import` in "
        "ao_shaping/__init__.py reappears here as a cost paid by every consumer."
    )


def test_every_public_name_still_resolves() -> None:
    """Laziness must not cost a single public name, nor change what it is.

    Guards the regression that laziness invites: a name dropped from ``__all__``
    or left pointing at the wrong module fails only at call time, and only for
    the caller who needed it. Every entry is resolved and its fully-qualified
    origin recorded, so a swap is caught here instead.
    """
    code = (
        "import sys\n"
        "import ao_shaping\n"
        "out = {}\n"
        "for name in ao_shaping.__all__:\n"
        "    try:\n"
        "        value = getattr(ao_shaping, name)\n"
        "        qn = getattr(value, '__qualname__', None)\n"
        "        out[name] = ('None' if value is None else\n"
        "                      f'{type(value).__module__}.{qn}' if qn else\n"
        "                      f'{type(value).__module__}.{type(value).__name__}')\n"
        "    except Exception as exc:\n"
        "        out[name] = f'ERROR:{type(exc).__name__}'\n"
        "sys.stderr.write('API ' + repr(sorted(out.items())) + '\\n')\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, (
        f"resolving __all__ failed (rc={proc.returncode})\n"
        f"--- stderr tail ---\n{_stderr_tail(proc.stderr)}"
    )
    match = re.search(r"API (\[.*?\])\n", proc.stderr, re.S)
    assert match is not None, f"probe produced no verdict:\n{_stderr_tail(proc.stderr)}"
    resolved = dict(ast.literal_eval(match.group(1)))

    assert "__version__" in resolved
    assert len(resolved) == len(set(resolved))
    broken = {k: v for k, v in resolved.items() if v.startswith("ERROR:")}
    assert not broken, f"public names no longer resolve: {broken}"

    # The hardware backends may legitimately be None where no SDK is installed, and
    # legitimately be real classes where one is (sim/mock stand them in). Nothing else
    # may ever be None: that is what a half-applied lazy map looks like.
    soft = {"NlightDM", "ThorlabWFS", "Santec", "FFmpegCamera", "MlaRes"}
    unexpected_none = {
        k for k, v in resolved.items() if v == "None" and k not in soft
    }
    assert not unexpected_none, (
        f"these resolve to None but are not optional-SDK backends: {unexpected_none}. "
        "A hard-fail name whose module failed to import is raising, not degrading."
    )
