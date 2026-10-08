"""Minimal, platform-correct environment for tests that spawn a fresh interpreter.

Why this module exists
----------------------
Several tests spawn ``python -c ...`` to prove a property that only holds in a pristine
interpreter (import-time laziness, CWD-independence, no accidental native-SDK load).
They all want the same thing: a child process with a **minimal** environment, so nothing
in the developer's shell can make an assertion pass by accident.

Three of them used to carry this literal::

    env={"PYTHONPATH": "src:libs", "PATH": "/usr/bin:/bin", "HOME": "/tmp"}

On Windows the child died at interpreter start-up with
``OSError: [WinError 10106] WSAEPROVIDERFAILEDINIT`` -- before executing a single line
of the snippet under test -- so the failure masqueraded as a defect in the code being
tested when it was purely the harness.

Measured attribution (isolating one variable at a time, ``import asyncio`` in a child):

==========================================  ======  ==========================================
env                                          rc      note
==========================================  ======  ==========================================
old literal, no ``SYSTEMROOT``              1       ``WinError 10106`` -- the failure
old literal **+ ``SYSTEMROOT``**            0       fixed; nothing else changed
``SYSTEMROOT``, no ``PYTHONPATH``           0       so ``PYTHONPATH`` was NOT load-bearing
==========================================  ======  ==========================================

So the outage cause is exactly one thing: the missing ``SYSTEMROOT``, which Windows needs
to locate the Winsock catalog. Two secondary defects are fixed alongside it because both
make the tests wrong rather than failing:

* ``"src:libs"`` is colon-separated, and Windows splits ``PYTHONPATH`` on ``;``. The whole
  string read as one directory named ``src:libs`` and did nothing. On a machine where
  ``ao_shaping`` is pip-installed that is invisible -- imports resolve anyway -- which is
  why these tests silently stopped exercising the repository source. Use ``os.pathsep``.
* ``_ROOT`` was computed as ``parents[3]`` in two files, which is ``tests/``, not the repo
  root, so ``cwd=_ROOT`` did not do what the docstring claimed.

Minimality is still the point, so only what the platform genuinely requires is forwarded.
"""

from __future__ import annotations

import os

#: Repo-root-relative entries the suite needs importable in child processes.
REPO_PYTHONPATH_ENTRIES = ("src", "libs")

#: Variables Windows needs before a bare interpreter can start. ``SYSTEMROOT`` locates
#: the Winsock catalog (without it: ``WinError 10106``); ``COMSPEC``/``PATHEXT`` are
#: needed for command invocation; ``TEMP``/``TMP`` for anything touching a temp file.
_WINDOWS_REQUIRED = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP")


def minimal_subprocess_env() -> dict[str, str]:
    """Return a minimal, platform-correct environment for a child interpreter.

    Returns:
        A fresh ``dict`` (never the live ``os.environ``) suitable for
        ``subprocess.run(env=...)``. ``PATH`` is always forwarded because DLL/shared
        library loading depends on it; Windows additionally gets ``SYSTEMROOT`` &
        friends, POSIX gets ``HOME``.

    Note:
        Callers set ``PYTHONPATH`` themselves -- see
        :func:`subprocess_env_with_pythonpath`.
    """
    env = {"PATH": os.environ.get("PATH", "")}
    if os.name == "nt":
        for key in _WINDOWS_REQUIRED:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
    else:
        home = os.environ.get("HOME")
        if home is not None:
            env["HOME"] = home
    return env


def subprocess_env_with_pythonpath(
    entries: tuple[str, ...] = REPO_PYTHONPATH_ENTRIES,
    *,
    absolute: bool = False,
    root: str | None = None,
) -> dict[str, str]:
    """Return :func:`minimal_subprocess_env` plus a correctly-joined ``PYTHONPATH``.

    Args:
        entries: Path entries to expose; defaults to ``("src", "libs")``.
        absolute: Join each entry onto ``root`` instead of leaving it relative.
        root: Base directory for ``absolute``; normally the repo root.

    Returns:
        The env dict, with ``PYTHONPATH`` joined by :data:`os.pathsep` -- ``;`` on
        Windows, ``:`` on POSIX. The literal ``"src:libs"`` is the bug this exists to
        prevent.

    Note:
        Relative entries rely on the child's ``cwd`` being the repo root; pass
        ``absolute=True, root=<repo root>`` when the child may run elsewhere.
    """
    env = minimal_subprocess_env()
    parts = [
        os.path.join(root, entry) if absolute and root else entry for entry in entries
    ]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


__all__ = [
    "REPO_PYTHONPATH_ENTRIES",
    "minimal_subprocess_env",
    "subprocess_env_with_pythonpath",
]