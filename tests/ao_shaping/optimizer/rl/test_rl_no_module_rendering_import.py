"""R-61 guard: the three ``optimizer/rl/`` training modules must stay import-clean.

#81 (R-61) was split off from #72 (R-56) on the claim that ``lr.py`` /
``lr_wfs.py`` / ``rl_wfs.py`` import ``matplotlib.pyplot`` at **module level** and
have **no** ``if __name__ == "__main__":`` guard, so "``import`` 即训练+弹图".

That claim is **false** (verified by AST, 2026-10-14). All three files:

* already gate their demo/training entry behind ``if __name__ == "__main__":``;
* keep every ``matplotlib.pyplot`` import **function-local** (inside the
  ``_on_step`` callback) or inside that guard;
* have **zero** module-body rendering imports and **zero** bare module-scope
  plot calls.

So importing these library modules pulls in no rendering stack and runs no
training loop. The only module-level side effect is the intentional
``os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`` OpenMP/MKL workaround, which is
out of scope for the rendering-import contract and must not be "fixed" away.

This test pins that verified state so the refactor loop cannot re-litigate #81
and a future edit cannot regress it.
"""

from __future__ import annotations

import ast
from pathlib import Path

RENDERING_STACK = ("matplotlib", "pygame", "pyplot")

# parents[4] of tests/ao_shaping/optimizer/rl/<this file> is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]
RL_ROOT = REPO_ROOT / "src" / "ao_shaping" / "optimizer" / "rl"

TARGETS = ("lr.py", "lr_wfs.py", "rl_wfs.py")


def _module_body(tree: ast.Module) -> list[ast.stmt]:
    return list(tree.body)


def _has_main_guard(tree: ast.Module) -> bool:
    for node in _module_body(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                return True
    return False


def _module_level_rendering_imports(tree: ast.Module) -> list[int]:
    hits: list[int] = []
    for node in _module_body(tree):  # module scope only -- a def body is not here
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = " ".join(a.name for a in node.names)
            mod = getattr(node, "module", "") or ""
            if any(pkg in names or pkg in mod for pkg in RENDERING_STACK):
                hits.append(node.lineno)
    return hits


def _module_level_bare_plot_calls(tree: ast.Module) -> list[int]:
    plot_attrs = {"plot", "show", "close", "figure", "imshow", "colorbar", "title"}
    hits: list[int] = []
    for node in _module_body(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
        ):
            fn = node.value.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in plot_attrs:
                hits.append(node.lineno)
    return hits


def test_rl_training_modules_are_import_clean() -> None:
    offenders: dict[str, list[str]] = {}
    for fname in TARGETS:
        path = RL_ROOT / fname
        assert path.is_file(), f"expected RL module missing: {path}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        problems: list[str] = []
        if not _has_main_guard(tree):
            problems.append("missing if __name__ == '__main__' guard")
        mod_imports = _module_level_rendering_imports(tree)
        if mod_imports:
            problems.append(f"module-level rendering import at line(s) {mod_imports}")
        bare_calls = _module_level_bare_plot_calls(tree)
        if bare_calls:
            problems.append(f"bare module-scope plot call at line(s) {bare_calls}")
        if problems:
            offenders[fname] = problems

    assert not offenders, (
        "R-61/#81 regression: optimizer/rl training modules must keep rendering "
        "imports function-local and demo code behind a __main__ guard. Offenders:\n"
        + "\n".join(f"  {f}: {p}" for f, p in offenders.items())
    )
