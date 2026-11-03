"""Gate: no module in ``src/`` may bind a native CUDA library at import time unguarded.

Why this is a gate and not a review comment: the 2026-11-03 transfer-fit crash
(``0xc0000005`` inside ``cublasLt64_13.dll``) came from a *bare-name* binding
split across two CUDA builds, and an import-ordering split is invisible in code
review -- it looks like one harmless ``import cupy`` line. The rule that keeps
the binding deterministic is: whatever runs at import time and pulls in CuPy or
OOPAO must first call ``ao_shaping._cublas_preload.pin_cublas()``.

Scope is ``src/`` only. The package is what every process in this repo imports,
so its import graph is the thing that must stay deterministic; a script that
imports CuPy on its own is a single-build process and not this hazard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

#: Top-level distribution names whose import hands the process a CUDA runtime.
NATIVE_TOP_LEVEL = frozenset({"cupy", "oopao"})

#: The hook that must run before any of them binds a DLL name.
PIN_CALL_NAMES = frozenset({"pin_cublas"})

_EXCLUDED = {"_cublas_preload.py"}


def _import_time_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Statements that execute when the module is imported, in source order.

    Descends through ``try`` / ``if`` / ``with`` wrappers and through **class
    bodies** (an import in a class body runs at import time -- that is how
    ``dm/nlight/driver.py`` used to read a file during ``import ao_shaping``),
    and stops at function bodies, where nothing runs until something calls it.
    ``if TYPE_CHECKING:`` blocks are skipped: they never execute.
    """
    found: list[ast.stmt] = []

    def is_type_checking(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "TYPE_CHECKING"
        if isinstance(node, ast.Constant):
            return node.value is False
        return False

    def walk(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                found.append(node)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for handler in node.handlers:
                    walk(handler.body)
                walk(node.orelse)
                walk(node.finalbody)
            elif isinstance(node, ast.If):
                if not is_type_checking(node.test):
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                walk(node.body)
            elif isinstance(node, ast.ClassDef):
                walk(node.body)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

    walk(tree.body)
    return found


def _top_of_import(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".")[0] if node.names else None
    return (node.module or "").split(".")[0]


def native_binding_lines(tree: ast.Module) -> list[int]:
    """Line numbers of import-time CuPy / OOPAO imports."""
    lines = []
    for node in _import_time_nodes(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        top = _top_of_import(node)
        if top and top.lower() in NATIVE_TOP_LEVEL:
            lines.append(node.lineno)
    return lines


def pin_call_line(tree: ast.Module) -> int | None:
    """First import-time call to the preload hook, or ``None``."""
    for node in _import_time_nodes(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in PIN_CALL_NAMES:
            return node.lineno
    return None


def _violations() -> list[str]:
    out = []
    for root in (SRC / "ao_shaping", SRC / "ml"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.name in _EXCLUDED:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            bindings = native_binding_lines(tree)
            if not bindings:
                continue
            pinned = pin_call_line(tree)
            if pinned is not None and pinned < min(bindings):
                continue
            where = ", ".join(f"line {line}" for line in bindings)
            out.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: import-time CUDA binding "
                f"at {where}, pin_cublas() at "
                f"{'absent' if pinned is None else f'line {pinned} (too late)'}"
            )
    return out


def test_no_unguarded_import_time_cuda_binding_in_src() -> None:
    """Every import-time CuPy/OOPAO import is preceded by ``pin_cublas()``."""
    assert _violations() == []


def test_gate_is_not_vacuous() -> None:
    """The scan really finds the guarded CuPy import it exists to police."""
    guarded = SRC / "ao_shaping" / "utils" / "image" / "spots_calc.py"
    tree = ast.parse(guarded.read_text(encoding="utf-8"), filename=str(guarded))
    assert native_binding_lines(tree), f"{guarded} no longer imports CuPy at import time"
    assert pin_call_line(tree) is not None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("import cupy\n", "cupy", id="bare-import"),
        pytest.param("from cupy import linalg\n", "cupy", id="from-import"),
        pytest.param("try:\n    import cupy\nexcept ImportError:\n    cupy = None\n",
                     "cupy", id="try-guarded-still-import-time"),
        pytest.param("import OOPAO\n", "oopao", id="case-insensitive-top"),
        pytest.param("def f():\n    import cupy\n", None, id="function-level-is-fine"),
        pytest.param("class C:\n    import cupy\n", "cupy", id="class-body-runs-at-import"),
        pytest.param("if TYPE_CHECKING:\n    import cupy\n", None, id="type-checking-only"),
        pytest.param("import numpy\n", None, id="other-import"),
    ],
)
def test_scan_rules(source: str, expected: str | None) -> None:
    """The traversal, on snippets -- so a broken scan fails here, not silently."""
    tree = ast.parse(source)
    found = native_binding_lines(tree)
    assert (expected if found else None) == expected
    assert bool(found) is (expected is not None)


def test_pin_must_precede_not_follow() -> None:
    """A hook call *after* the import does not count; the name is already bound."""
    tree = ast.parse("import cupy\nfrom ao_shaping._cublas_preload import pin_cublas\npin_cublas()\n")
    assert pin_call_line(tree) > min(native_binding_lines(tree))
