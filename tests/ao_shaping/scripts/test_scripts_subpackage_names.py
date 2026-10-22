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

#: Name -> (why the only owner is not a real distribution, what that owner
#: must look like). The owner is checked, so the exemption cannot outlive its
#: reason: if the name ever gets claimed by a real distribution the exemption
#: stops applying and the guard reports the shadowing again.
#:
#: ``build`` is the one case. Its only owner is the vendored OOPAO
#: submodule's build artifact directory -- setuptools' scratch output, which
#: the submodule gitignores -- so there is no distribution to break. It sits
#: at ``libs/OOPAO/build`` in a checkout and at ``/opt/OOPAO/build`` in the
#: test image, so the check matches the path tail rather than a fixed prefix.
#: PyPI's own ``build`` IS a real distribution with a real ``__file__``; if
#: it is ever installed, the owner stops being the artifact and the guard
#: fires.
ALLOWED_ARTIFACT_OWNER = {
    "build": (
        "the vendored OOPAO submodule's build artifact directory (setuptools "
        "scratch output, gitignored by the submodule), not an importable "
        "distribution",
        "OOPAO/build",
    ),
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

    Returns the winning module's location, or ``None`` when nothing on the
    path claims the name. A non-``None`` result means ``scripts/<name>/`` would
    steal that name from its real owner.

    A namespace package has no ``__file__``, so the probe reports the
    package's own directory for it. Printing ``__file__`` for one yields the
    string "None", which is truthy and used to be reported as the owner of a
    module that does not exist.
    """
    probe = (
        "import importlib;"
        "m = importlib.import_module({name!r});"
        "f = getattr(m, '__file__', None);"
        "print(f if f else (list(m.__path__)[0] if hasattr(m, '__path__') else ''))"
    ).format(name=name)
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"},
    )
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


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
    if name in ALLOWED_ARTIFACT_OWNER:
        reason, expected = ALLOWED_ARTIFACT_OWNER[name]
        assert owner is None or expected in owner, (
            f"{name} was exempted because {reason}, but the owner is now "
            f"{owner}; re-check the exemption"
        )
        return
    assert owner is None, (
        f"scripts/{name}/ shadows {owner}, which resolves to that module when "
        f"scripts/ is off the path. Adding scripts/ to pythonpath makes a bare "
        f"`import {name}` land in scripts/ instead, breaking that module "
        "repo-wide -- and the traceback names the innocent third-party package. "
        "Rename the subfolder."
    )
