"""R-55 guard: ``utils/`` must not module-level import a rendering stack.

``utils/`` is the leaf layer (AGENTS.md §9 / README coding rule 9): it may not
import ``algorithm/`` / ``drivers/`` / ``optimizer/`` at module import time, and by
the same argument it must not drag in matplotlib / pygame either -- ``utils`` sits on
the import path of essentially everything, so an import-time failure or cost here is
an import-time failure of ``import ao_shaping`` itself.

One carve-out, on purpose: ``utils/image/display.py`` still imports matplotlib and
pygame at module level. That file is the subject of its own issue (#50 / R-29,
"move display.py into display/"), which is deliberately NOT bundled here -- folding
it in would make this guard fail for a reason that has not been decided yet. When
#50 lands, delete ``_CARVE_OUT`` and this guard starts covering it for free.
"""

from __future__ import annotations

import ast
from pathlib import Path

RENDERING_STACK = ("matplotlib", "pygame", "pyplot")

# Resolved against the repo root, not this file's directory: parents[3] of
# tests/ao_shaping/utils/<this file> is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
UTILS_ROOT = REPO_ROOT / "src" / "ao_shaping" / "utils"

#: module-level rendering imports that are known and still open
_CARVE_OUT = {"src/ao_shaping/utils/image/display.py"}


def _module_level_rendering_imports(path: Path) -> list[int]:
    """Line numbers of matplotlib/pygame imports in the module body (not inside a def)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return []
    hits: list[int] = []
    for node in tree.body:  # module scope only -- a def body is not in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = " ".join(a.name for a in node.names)
            mod = getattr(node, "module", "") or ""
            if any(pkg in names or pkg in mod for pkg in RENDERING_STACK):
                hits.append(node.lineno)
    return hits


def test_utils_has_no_module_level_rendering_import() -> None:
    offenders: dict[str, list[int]] = {}
    for path in sorted(UTILS_ROOT.rglob("*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _CARVE_OUT:
            continue
        hits = _module_level_rendering_imports(path)
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        f"module-level rendering import(s) in the utils leaf layer: {offenders}. "
        f"Move the import inside the function(s) that plot, the way "
        f"utils/io/file.py and utils/image/hardware_utils.py already do."
    )


def test_carve_out_is_still_accurate() -> None:
    """If #50 lands, _CARVE_OUT must shrink -- not silently rot."""
    for rel in _CARVE_OUT:
        path = REPO_ROOT / rel
        assert path.is_file(), f"carve-out names a file that no longer exists: {rel}"
        assert _module_level_rendering_imports(path), (
            f"{rel} is carved out but no longer has a module-level rendering import -- "
            f"drop it from _CARVE_OUT so this guard starts covering it."
        )


def test_file_py_plotting_helpers_still_work() -> None:
    """The deferred imports must actually be present where plt is used.

    A guard that only bans module-level imports would still pass if someone deleted a
    local import and the function failed only at call time, so pin the pairing.
    """
    path = REPO_ROOT / "src/ao_shaping/utils/io/file.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    uses_plt = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(x, ast.Attribute)
            and isinstance(x.value, ast.Name)
            and x.value.id == "plt"
            for x in ast.walk(fn)
        )
    }
    assert uses_plt, "expected file.py to still plot somewhere -- has the code moved?"

    missing: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name not in uses_plt:
            continue
        has_local = any(
            isinstance(i, (ast.Import, ast.ImportFrom))
            and any("matplotlib" in a.name for a in i.names)
            for i in ast.walk(fn)
        )
        if not has_local:
            missing.append(fn.name)
    assert not missing, f"these functions use plt but no longer import it: {missing}"
