"""Every ``ao_shaping.*`` import in scripts/ must actually resolve.

Deferred (function-local) imports are invisible to the test suite: a script
whose hardware path never runs under CI can carry a ``ModuleNotFoundError`` for
as long as nobody notices. That is not hypothetical -- the extraction of
``slm_zernike_primitives`` renamed the module and three hardware scripts kept
importing the old name; every test still passed.

This resolves each target against the real package tree, so a rename that
misses even one caller fails here instead of on the bench.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"


def _gitignored(path: pathlib.Path) -> bool:
    """True when git refuses to track ``path``, i.e. a clean clone will not have it.

    The sweep below walks the filesystem, so without this it also judges scratch
    files that only exist on one machine. ``scripts/dev/`` is ignored wholesale by
    ``.gitignore`` (``scripts/dev**``), so a stale import in one of those files
    fails the guard *here* while a clean checkout has nothing to fail on -- the
    guard would be green in CI and red on the machine it is meant to protect.
    """
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _ao_imports(path: pathlib.Path) -> set[str]:
    """Module names imported from ``ao_shaping`` in any scope of ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ao_shaping"):
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ao_shaping"):
                    # Full dotted path, not just the root: `import
                    # ao_shaping.a.b.c as x` fails if any segment is renamed.
                    found.add(alias.name)
    return found


def _scripts() -> list[pathlib.Path]:
    return sorted(
        p
        for p in (REPO_ROOT / "scripts").rglob("*.py")
        if "__pycache__" not in str(p) and not _gitignored(p)
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_script_ao_shaping_imports_resolve(script: pathlib.Path) -> None:
    unresolvable = []
    for module in sorted(_ao_imports(script)):
        if importlib.util.find_spec(module) is None:
            unresolvable.append(module)
    assert not unresolvable, (
        f"{script.relative_to(REPO_ROOT)} imports module(s) that do not exist: "
        f"{unresolvable}. Deferred imports are not exercised by the suite, so a "
        f"rename that misses a caller fails here rather than on the bench."
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_script_has_no_dangling_local_import_of_moved_code(script: pathlib.Path) -> None:
    """A moved module leaves its old path behind; catch the literal."""
    source = script.read_text(encoding="utf-8")
    for gone in (
        "ao_shaping.optimizer.wfless.slm_zernike_common",
    ):
        assert gone not in source, (
            f"{script.relative_to(REPO_ROOT)} still references the renamed {gone!r}"
        )