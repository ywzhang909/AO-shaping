"""Contract guard: importing an ``ao_shaping`` subpackage must not write to disk.

README.md / AGENTS.md state the laziness contract as "``import ao_shaping`` 及其子包
不得加载原生 SDK、不得做磁盘 I/O". The SDK half has enforcement (lazy ``_load_sdk``);
the disk half had **none**, and it was being violated: ``DeviceConfigManager.__init__``
ran ``mkdir(parents=True)`` and read ``defaults.json``, and that constructor runs at
import time through five module-level singletons -- ``WFS_CONFIG``
(``drivers/wfs/thorlab_wfs.py:193``), ``SLM_CONFIG`` (``drivers/slm/santec/driver.py:276``),
``CCD_CONFIG`` (``drivers/ccd/daheng/driver.py:46``), ``MICRO_DM_CONFIG``
(``drivers/dm/micro/driver.py:96``), ``NLIGHT_CONFIG`` (``drivers/dm/nlight/driver.py:76``).
Because ``drivers/__init__.py:28`` imports ``ThorlabWFS`` eagerly, *every* import under
``ao_shaping.drivers`` created ``data/wfs_configs`` as a side effect. Measured, not
inferred: importing ``ao_shaping.drivers.sim.slm_shaping_bench`` in a container with the
repo mounted read-only died with ``OSError: [Errno 30] Read-only file system:
'/workspace/data/wfs_configs'`` -- an import that could not be used at all.

The probe runs in a **subprocess** with ``-B``: modules already imported in the pytest
process would silently satisfy a same-process check, and bytecode caching is itself a
write that would surface as a false violation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

#: Tripwire installed before any ao_shaping import. Nested triple quotes are avoided
#: on purpose: they would terminate this raw string and leak the probe into the module.
#: Writes are judged only inside GUARD_ROOTS (the repo and the probe cwd): third-party
#: libraries legitimately create their own caches under HOME -- measured, the first run
#: of this guard tripped on matplotlib's pathlib.Path.mkdir('/root/.config/matplotlib')
#: -- and policing those is neither this contract nor ours to fix.
PROBE = r"""
import builtins
import importlib
import os
import pathlib
import sys


class ImportWrite(Exception):
    pass


ROOTS = [r for r in os.environ.get("GUARD_ROOTS", "").split(os.pathsep) if r]


def _in_scope(path):
    ap = os.path.abspath(os.path.expanduser(str(path)))
    return any(ap == r or ap.startswith(r + os.sep) for r in ROOTS)


_real_path_mkdir = pathlib.Path.mkdir
_real_os_makedirs = os.makedirs
_real_os_mkdir = os.mkdir


def _trip_path_mkdir(self, *args, **kwargs):
    if _in_scope(self):
        raise ImportWrite("pathlib.Path.mkdir(" + repr(str(self)) + ")")
    return _real_path_mkdir(self, *args, **kwargs)


def _trip_os_makedirs(*args, **kwargs):
    if args and _in_scope(args[0]):
        raise ImportWrite("os.makedirs(" + repr(args) + ")")
    return _real_os_makedirs(*args, **kwargs)


def _trip_os_mkdir(*args, **kwargs):
    if args and _in_scope(args[0]):
        raise ImportWrite("os.mkdir(" + repr(args) + ")")
    return _real_os_mkdir(*args, **kwargs)


_real_open = builtins.open


def _guarded_open(file, mode="r", *args, **kwargs):
    if any(ch in str(mode) for ch in "wax+") and _in_scope(file):
        raise ImportWrite("open(" + repr(str(file)) + ", mode=" + repr(mode) + ")")
    return _real_open(file, mode, *args, **kwargs)


pathlib.Path.mkdir = _trip_path_mkdir
os.makedirs = _trip_os_makedirs
os.mkdir = _trip_os_mkdir
builtins.open = _guarded_open

for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except ImportWrite as exc:
        print("VIOLATION " + name + ": " + str(exc))
        raise SystemExit(1)
print("OK")
"""

#: Subpackages that must import without touching the disk. The ``drivers.*`` entries are
#: the ones that actually regressed; the leaf entries keep utils/ config honest.
@pytest.mark.parametrize(
    "package",
    [
        "ao_shaping",
        "ao_shaping.config",
        "ao_shaping.utils",
        "ao_shaping.drivers",
        "ao_shaping.drivers.wfs",
        "ao_shaping.drivers.slm",
        "ao_shaping.drivers.ccd.daheng",
        "ao_shaping.drivers.dm.nlight",
        "ao_shaping.drivers.dm.micro",
        "ao_shaping.drivers.sim.slm_shaping_bench",
    ],
)
def test_importing_subpackage_performs_no_disk_write(package: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-B", "-c", PROBE, package],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(SRC),
            "PYTHONDONTWRITEBYTECODE": "1",
            "GUARD_ROOTS": os.pathsep.join((str(REPO), str(tmp_path))),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    violation = next(
        (line for line in proc.stdout.splitlines() if line.startswith("VIOLATION")),
        None,
    )
    assert violation is None, (
        f"{violation} -- importing {package} touched the disk. The laziness contract "
        "forbids import-time I/O; create directories on the write path instead."
    )
    assert proc.returncode == 0, (
        f"probe failed for {package} (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
