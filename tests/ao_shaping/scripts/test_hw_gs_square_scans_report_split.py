"""The GS square-scan report generator is split into sibling modules; keep it sound.

Why this exists
---------------
``scripts/slm/generate_hw_gs_square_scans_report.py`` was 5880 lines. It was
split by **moving** code into ten sibling modules and re-importing every moved
name at the top of the original file ("抽而不删 + 留 facade"): the original file
keeps the same public surface, so no caller changes.

Three things can silently rot that arrangement, and none of them is a
``SyntaxError``, so nothing else in the suite would notice:

1. a sibling's ``sys.path`` bootstrap breaks — the module then imports fine
   under ``pytest`` (repo root is on ``pythonpath``) but dies under the
   **file-form** launch every registered reproduction command uses. This test
   imports each sibling in a **subprocess with a clean cwd and no inherited
   path**, which is what a file-form launch actually looks like;
2. a name disappears — the facade re-imports a name its sibling no longer
   defines. Caught here by resolving each re-exported name against the module
   it claims to come from;
3. a cycle appears — a sibling imports the facade back. Then whichever of the
   two is imported first gets a half-initialised module, and it only fails on
   the combinations someone happens to hit.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_FACADE = "scripts/slm/generate_hw_gs_square_scans_report.py"

#: The siblings. Ordered leaves-first; the list itself is the contract — a new
#: sibling must be added here or it is not covered.
_SIBLINGS = (
    "scripts.slm._hw_gs_square_scans_model",
    "scripts.slm._hw_gs_square_scans_fmt",
    "scripts.slm._hw_gs_square_scans_io",
    "scripts.slm._hw_gs_square_scans_metrics",
    "scripts.slm._hw_gs_square_scans_analysis",
    "scripts.slm._hw_gs_square_scans_checks",
    "scripts.slm._hw_gs_square_scans_figbase",
    "scripts.slm._hw_gs_square_scans_figures",
    "scripts.slm._hw_gs_square_scans_sections",
    "scripts.slm._hw_gs_square_scans_sections_core",
    "scripts.slm._hw_gs_square_scans_sections_scan",
    "scripts.slm._hw_gs_square_scans_atomic",
    "scripts.slm._hw_gs_square_scans_artifacts",
)


def _local_path(dotted: str) -> Path:
    return _ROOT.joinpath(*dotted.split(".")).with_suffix(".py")


def _top_level_bindings(path: Path) -> set[str]:
    """Names a module binds at module level: defs, constants, and imports.

    Imports count because a sibling that delegates an implementation to a shared
    module and re-exports it still *binds* the name the facade consumes — only
    ``import *`` is skipped, since it binds an unknowable set.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            out.update(a.asname or a.name for a in node.names if a.name != "*")
        elif isinstance(node, ast.Import):
            out.update(a.asname or a.name.split(".")[0] for a in node.names)
    return out


def _imported_from(tree: ast.Module, dotted: str) -> list[str]:
    """Names ``tree`` re-exports from ``dotted`` (the multi-line form)."""
    out: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != dotted:
            continue
        out.extend(a.name for a in node.names)
    return out


def test_facade_runs_as_a_file_from_a_foreign_cwd(tmp_path: Path) -> None:
    """``python scripts/slm/generate_hw_gs_square_scans_report.py --help`` must work.

    The registered reproduction command is **file-form**, and the subprocess
    runs from a cwd outside the repo, so a path that only resolves when the
    cwd happens to be the repo root fails here instead of on the bench.
    """
    proc = subprocess.run(
        [sys.executable, str(_ROOT / _FACADE), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, (
        "the report generator cannot be launched in file form:\n"
        f"{proc.stderr.strip()}"
    )
    for flag in ("--run-root", "--retest-root", "--atomic-root", "--check"):
        assert flag in proc.stdout, f"--help lost {flag} (argparse never ran)"


@pytest.mark.parametrize("dotted", _SIBLINGS)
def test_sibling_runs_as_a_file(dotted: str, tmp_path: Path) -> None:
    """``python scripts/slm/_hw_gs_square_scans_x.py`` must exit 0 on its own.

    Each sibling imports the model as ``scripts.slm.*``, which needs the repo
    root on ``sys.path`` — and a file-form launch does **not** provide it. So
    each one carries its own bootstrap. Two things make this worth testing:

    * the bootstrap is per-file, so deleting one line breaks exactly one
      sibling while the facade keeps working (the facade inserts the root
      itself before importing anything) — nothing else in the suite sees it;
    * that one line is what the sibling must have to satisfy
      ``test_script_file_form_imports``, whose rule A demands it.

    The subprocess runs from a foreign cwd so the bootstrap has to do real
    work; under pytest the repo root is already on ``pythonpath``.
    """
    path = _local_path(dotted)
    assert path.is_file(), f"{dotted} is gone: {path}"
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, (
        f"{dotted} cannot be run in file form from a foreign cwd — its "
        f"sys.path bootstrap is missing or wrong:\n{proc.stderr.strip()}"
    )


def test_facade_re_exports_every_name_it_claims() -> None:
    """No name may vanish across the split.

    A move that forgets to re-export leaves the facade importing a name its
    sibling does not define — an ``ImportError`` at the first run, or, worse,
    a name that resolves to something else entirely.
    """
    tree = ast.parse(_ROOT.joinpath(_FACADE).read_text(encoding="utf-8"))
    missing: list[str] = []
    for dotted in _SIBLINGS:
        path = _local_path(dotted)
        have = _top_level_bindings(path)
        for name in _imported_from(tree, dotted):
            if name not in have:
                missing.append(f"  {dotted} no longer defines {name}")
    assert not missing, (
        "the facade re-exports names its siblings do not define:\n"
        + "\n".join(missing)
    )


def test_no_sibling_imports_the_facade_back() -> None:
    """The split must stay acyclic.

    The facade imports every sibling; a sibling importing the facade back
    makes the pair order-dependent, and the failure lands on whichever import
    happens first — i.e. on the innocent caller, not on the file that broke it.
    """
    facade_dotted = _FACADE[: -len(".py")].replace("/", ".")
    cycles: list[str] = []
    for dotted in _SIBLINGS:
        tree = ast.parse(_local_path(dotted).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == facade_dotted:
                cycles.append(f"  {dotted} imports {facade_dotted}")
            elif isinstance(node, ast.Import) and any(
                a.name == facade_dotted for a in node.names
            ):
                cycles.append(f"  {dotted} imports {facade_dotted}")
    assert not cycles, "the split has a cycle:\n" + "\n".join(cycles)


def test_the_model_stays_a_leaf() -> None:
    """The model is the shared base; it must not depend on anything in scripts/.

    Every other sibling imports it. One import of theirs back would make the
    two mutually dependent — and the model is the one module where that is
    least visible, because it is imported first and therefore looks fine.
    """
    tree = ast.parse(_local_path(_SIBLINGS[0]).read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"scripts", "_common", "ao_shaping", "ml"}:
                bad.append(f"  imports {node.module}")
    assert not bad, "the model is no longer a leaf:\n" + "\n".join(bad)


def test_siblings_are_registered_with_the_guard() -> None:
    """A new sibling that is not listed here escapes every check above."""
    found = sorted(
        p.stem
        for p in (_ROOT / "scripts" / "slm").glob("_hw_gs_square_scans_*.py")
    )
    listed = sorted(d.split(".")[-1] for d in _SIBLINGS)
    assert found == listed, (
        f"scripts/slm/_hw_gs_square_scans_*.py changed ({found}); "
        f"update _SIBLINGS in this guard (it lists {listed})"
    )


def test_module_imports_resolve_in_process() -> None:
    """The in-process import must work too — that is how the facade uses them."""
    for dotted in _SIBLINGS:
        assert importlib.import_module(dotted) is not None
