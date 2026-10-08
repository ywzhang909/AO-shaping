"""R-56 guard: no module-level rendering import in ``optimizer/`` that can be deferred,
and every scope that uses matplotlib/pygame imports it locally.

Two failure modes are covered, and the second is the one that actually bites:

1. a module-level ``matplotlib`` / ``pygame`` / ``pyplot`` import reappearing in a file
   where every use sits inside a function or an ``if __name__ == "__main__":`` block;
2. an import being removed from module level and NOT re-added to the scopes that use
   it -- that produces a ``NameError`` only when that function is called with the
   plotting path taken, so it survives import-only smoke tests.

``optimizer/rl/lr.py``, ``lr_wfs.py`` and ``rl_wfs.py`` are carved out: they use
matplotlib at module scope with no ``__main__`` block at all, so there is nothing to
defer into. They are tracked by their own issue.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZER_ROOT = REPO_ROOT / "src" / "ao_shaping" / "optimizer"

STACK = ("pygame", "matplotlib", "pyplot")
NAMES = ("plt", "pygame")

#: files whose module-level rendering import cannot be deferred (no __main__ block)
CARVE_OUT = {
    "src/ao_shaping/optimizer/rl/lr.py",
    "src/ao_shaping/optimizer/rl/lr_wfs.py",
    "src/ao_shaping/optimizer/rl/rl_wfs.py",
}


class ScopeWalker(ast.NodeVisitor):
    """Attribute each plt/pygame use to (enclosing function, inside __main__)."""

    def __init__(self) -> None:
        self.scope: list[str | None] = []
        self.in_main = 0
        self.uses: dict[tuple[str | None, bool], list[int]] = {}
        self.local_imports: set[tuple[str | None, bool]] = set()

    def _note_import(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = " ".join(a.name for a in node.names)
            mod = getattr(node, "module", "") or ""
            if any(p in names or p in mod for p in STACK):
                self.local_imports.add((self.scope[-1] if self.scope else None, self.in_main > 0))

    def _note_use(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name) and node.id in NAMES and isinstance(node.ctx, ast.Load):
            key = (self.scope[-1] if self.scope else None, self.in_main > 0)
            self.uses.setdefault(key, []).append(node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        dump = ast.dump(node.test)
        is_main = "__name__" in dump and "__main__" in dump
        if is_main:
            self.in_main += 1
        for stmt in node.body:
            self.visit(stmt)
        if is_main:
            self.in_main -= 1
        for stmt in node.orelse:
            self.visit(stmt)

    def generic_visit(self, node: ast.AST) -> None:
        self._note_import(node)
        self._note_use(node)
        super().generic_visit(node)


def _analyse(path: Path) -> tuple[list[int], dict[tuple[str | None, bool], list[int]], set[tuple[str | None, bool]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mod_level = [
        n.lineno
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and any(p in a.name for a in n.names for p in STACK)
    ]
    w = ScopeWalker()
    w.visit(tree)
    return mod_level, w.uses, w.local_imports


def test_no_module_level_rendering_import_outside_carveout() -> None:
    offenders: dict[str, list[int]] = {}
    for path in sorted(OPTIMIZER_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in CARVE_OUT:
            continue
        mod_level, _, _ = _analyse(path)
        if mod_level:
            offenders[rel] = mod_level
    assert not offenders, (
        f"module-level rendering import(s) reappeared in optimizer/: {offenders}. "
        f"Defer the import into the function(s) that plot."
    )


def test_every_scope_using_rendering_imports_it_locally() -> None:
    """The NameError guard: a use must always be accompanied by a local import."""
    missing: dict[str, list[int]] = {}
    for path in sorted(OPTIMIZER_ROOT.rglob("*.py")):
        mod_level, uses, local_imports = _analyse(path)
        if not uses:
            continue
        for key, lines in uses.items():
            if key not in local_imports:
                # a module-level import would satisfy the scope; that is only legitimate
                # for the carve-out files, which are excluded from the other test
                if mod_level:
                    continue
                missing.setdefault(path.relative_to(REPO_ROOT).as_posix(), []).extend(lines)
    assert not missing, (
        f"these lines use plt/pygame with no import in scope: {missing}. "
        f"Removing the module-level import without re-adding it per scope turns into a "
        f"call-time NameError that import-only smoke tests will not catch."
    )


def test_carve_out_is_still_accurate() -> None:
    """If the three rl scripts get a __main__ guard, drop them from CARVE_OUT."""
    for rel in CARVE_OUT:
        path = REPO_ROOT / rel
        assert path.is_file(), f"carve-out names a missing file: {rel}"
        mod_level, uses, _ = _analyse(path)
        assert mod_level, f"{rel} is carved out but has no module-level rendering import"
        stray = {
            key: lines
            for key, lines in uses.items()
            if key[0] is None and not key[1]
        }
        assert stray, (
            f"{rel} is carved out but no longer has a module-level rendering use -- "
            f"drop it from CARVE_OUT so this guard starts covering it."
        )
