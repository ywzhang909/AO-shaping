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

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.drivers.dm._registry import list_dm_types

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
    """Return the repository root derived from this test file's location."""
    return Path(__file__).resolve().parents[3]


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
    env = {**os.environ, "PYTHONPATH": str(_src_dir())}
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
    """
    import socket

    def _boom(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("list_dm_types() must not open a socket")

    monkeypatch.setattr(socket.socket, "connect_ex", _boom)
    assert len(list_dm_types()) == 6


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