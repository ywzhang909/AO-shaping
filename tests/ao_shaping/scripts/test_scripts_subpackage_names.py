"""No ``scripts/<sub>/`` may shadow an importable module name.

Why this exists
---------------
``scripts/`` is on ``sys.path`` (``pythonpath = ["src", ".", "scripts"]`` in
``pyproject.toml``), so any subfolder that is a package is importable by its bare
name. Grouping the scripts into ``scripts/zernike/`` therefore shadowed the
third-party ``zernike`` distribution, which
``src/ao_shaping/utils/wavefront/zernike_calc.py`` imports as
``from zernike import RZern``. Every consumer of that module -- and therefore 50
parametrised cases in ``test_script_imports_resolve`` -- failed with
``ImportError: cannot import name 'RZern' from 'zernike'``, pointing at a package
that looked completely innocent.

The failure is silent at authoring time: the folder name reads fine, the tests
fail far away, and the traceback names the *third-party* package as the broken
thing. So the check runs here, at the point where a name is chosen.

``ml`` is allowed to overlap because ``src/`` precedes ``scripts/`` on the path,
so ``from ml.zernike import ...`` still resolves to ``src/ml``. That precedence is
load-bearing, so it is asserted rather than assumed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"

#: Group name -> why an overlap is acceptable (empty dict entry = must not shadow).
ALLOWED_OVERLAP = {
    "ml": "src/ precedes scripts/ on pythonpath, so `ml` resolves to src/ml",
}


def _subpackages() -> list[Path]:
    return sorted(
        p
        for p in _SCRIPTS.iterdir()
        if p.is_dir()
        and not p.name.startswith((".", "_"))
        and (p / "__init__.py").is_file()
    )


def _taken_without_scripts(name: str) -> str | None:
    """Resolve ``name`` with ``scripts/`` REMOVED from the path.

    Returns the winning module's ``__file__``, or ``None`` when nothing on the
    path claims the name. A non-``None`` result means ``scripts/<name>/`` would
    steal that name from its real owner.
    """
    out = subprocess.run(
        [sys.executable, "-c", f"import {name} as m; print(m.__file__)"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"},
    )
    if out.returncode != 0:
        return None
    owner = out.stdout.strip()
    if owner in ("", "None"):
        # A namespace package has no __file__, so the probe prints "None".
        # Nothing owns the name in the ordinary sense, so there is no
        # module for scripts/<name>/ to shadow -- and "None" as a STRING is
        # truthy, which used to be reported as a shadowing owner.
        return None
    return owner


def test_scripts_has_subpackages() -> None:
    assert _subpackages(), "no scripts/ subpackages found; this guard is vacuous"


@pytest.mark.parametrize("pkg", _subpackages(), ids=lambda p: p.name)
def test_subpackage_does_not_shadow_an_import(pkg: Path) -> None:
    name = pkg.name
    owner = _taken_without_scripts(name)
    if name in ALLOWED_OVERLAP:
        assert owner is None or str(_ROOT / "src" / name) in owner, (
            f"{name} was exempted because src/ wins the path order, but the "
            f"real owner moved to {owner}; re-check the exemption"
        )
        return
    assert owner is None, (
        f"scripts/{name}/ shadows {owner}, which resolves to that module when "
        f"scripts/ is off the path. Adding scripts/ to pythonpath makes a bare "
        f"`import {name}` land in scripts/ instead, breaking that module "
        "repo-wide -- and the traceback names the innocent third-party package. "
        "Rename the subfolder."
    )
