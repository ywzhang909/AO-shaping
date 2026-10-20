"""Every script must be importable the way the repo documents it: as a file.

Why this exists
---------------
All 67 reproduction commands registered in ``scripts/_common/provenance.py`` are
**file-form**::

    python scripts/ml/compare_models_cv.py --analyse

A file-form launch puts *the script's own directory* on ``sys.path[0]`` and
nothing else -- the cwd is **not** on ``sys.path``. Two consequences:

* ``scripts.*`` (a package path, e.g. ``scripts.ml.compare_unet_baseline``) is
  reachable only if the script puts the **repo root** on ``sys.path`` itself;
* ``ml.*`` / ``ao_shaping.*`` live under ``src/``, and a script that puts
  ``ROOT/"scripts"`` on the path **shadows** them: ``ml`` then resolves to the
  ``scripts/ml`` folder and ``ml.hwdataset`` & co. vanish.

Both failures are silent until the script is actually run -- the report
generator crashes with ``ModuleNotFoundError`` on its own documented command.

The invariant, asserted statically
----------------------------------
No script is executed here (several have module-level ``subprocess.run`` /
``exec`` calls, and this suite must never touch hardware or install anything).
The bootstrap is *resolved* from the AST instead:

* rule A -- a script importing ``scripts.*`` must put the repo root on
  ``sys.path``;
* rule B -- a script importing ``ml.*`` / ``ao_shaping.*`` must not let
  ``ROOT/"scripts"`` shadow ``ROOT/"src"`` (whichever lands in front wins).

A script whose bootstrap cannot be resolved is **skipped, not failed**: a guard
that cries wolf gets disabled.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"

# Third-party packages that are optional extras and need not be installed.
# Everything else the scripts import is either stdlib, the project's own
# packages (``ao_shaping``/``ml`` from src/, ``scripts`` from the root) or a
# declared dependency.
_OPTIONAL_TOP_LEVEL = frozenset({"stable_baselines3", "tensorboard"})


def _packages_under(rel: tuple[str, ...]) -> set[str]:
    """Top-level package names a directory provides (dirs with __init__.py)."""
    base = _ROOT.joinpath(*rel) if rel else _ROOT
    if not base.is_dir():
        return set()
    out = set()
    for child in base.iterdir():
        if child.is_dir() and (child / "__init__.py").is_file():
            out.add(child.name)
    return out


# Names that exist as a package in BOTH src/ and scripts/ -- whichever
# directory lands in front on sys.path wins. Computed, not hardcoded: today
# that is `ml` only (src/ml vs scripts/ml); `ao_shaping` has no scripts/
# counterpart and is therefore never shadowed.
_SHADOWED_NAMES = frozenset(_packages_under(("src",)) & _packages_under(("scripts",)))


# --------------------------------------------------------------------------
# AST -> "which repo directories does this script put on sys.path?"
# --------------------------------------------------------------------------
# A path is a tuple of parts relative to the repo root: () is the root,
# ("src",) is root/src, ("scripts", "ml") is root/scripts/ml.


def _file_tuple(script: Path) -> tuple[str, ...]:
    return script.relative_to(_ROOT).parts


def _is_file_expr(expr: ast.expr) -> bool:
    """True for ``__file__`` -- the name or the string literal."""
    if isinstance(expr, ast.Name):
        return expr.id == "__file__"
    return isinstance(expr, ast.Constant) and expr.value == "__file__"


def _drop(parts: tuple[str, ...], n: int) -> tuple[str, ...] | None:
    """``parts`` with the last ``n + 1`` elements removed.

    ``Path(f).parents[0]`` is the file's directory and ``parents[n]`` climbs n
    levels above it, so ``parents[N]`` drops ``N + 1`` trailing parts.
    """
    keep = len(parts) - 1 - n
    if keep < 0:
        return None
    return parts[:keep]


class _Resolver:
    """Resolve a bootstrap expression to a repo-relative path tuple."""

    def __init__(self, script: Path) -> None:
        self.script = script
        self.aliases: dict[str, ast.expr] = {}
        self.depth = len(_file_tuple(script)) - 1  # levels from the file to the root

    def bind(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = node.value

    def resolve(self, expr: ast.expr, seen: frozenset[str] = frozenset()) -> tuple[str, ...] | None:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "str":
            if not expr.args:
                return None
            return self.resolve(expr.args[0], seen)
        if isinstance(expr, ast.Name):
            if expr.id in seen:
                return None
            target = self.aliases.get(expr.id)
            if target is None:
                return None
            return self.resolve(target, seen | {expr.id})
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
            left = self.resolve(expr.left, seen)
            right = expr.right
            if left is None or not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
                return None
            return left + (right.value,)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            # Path(__file__).resolve() / .parent / .parents[N]
            if expr.func.attr == "resolve" and isinstance(expr.func.value, ast.Call):
                inner = expr.func.value
                if isinstance(inner.func, ast.Name) and inner.func.id == "Path":
                    if inner.args and _is_file_expr(inner.args[0]):
                        return self.file_tuple
            if expr.func.attr == "parent":
                base = self._base_of_chain(expr.value)
                return None if base is None else _drop(base, 1)
        if isinstance(expr, ast.Subscript):
            # X.parents[N] is a Subscript, not a Call -- missing this silently
            # skipped every script (the guard then passed vacuously).
            holder = expr.value
            if isinstance(holder, ast.Attribute) and holder.attr == "parents":
                idx = expr.slice
                if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                    base = self._base_of_chain(holder.value)
                    return None if base is None else _drop(base, idx.value)
        return None

    @property
    def file_tuple(self) -> tuple[str, ...]:
        return _file_tuple(self.script)

    def _base_of_chain(self, expr: ast.expr) -> tuple[str, ...] | None:
        """The path a ``.parent`` / ``.parents[N]`` chain hangs off."""
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) \
                and expr.func.attr == "resolve" and isinstance(expr.func.value, ast.Call):
            inner = expr.func.value
            if isinstance(inner.func, ast.Name) and inner.func.id == "Path":
                if inner.args and _is_file_expr(inner.args[0]):
                    return self.file_tuple
        return self.resolve(expr)


def _sys_path_inserts(tree: ast.AST, resolver: _Resolver) -> tuple[list[tuple[str, ...]], bool]:
    """Repo-relative dirs the script puts on ``sys.path``, front to back.

    Returns ``(paths, understood)``. ``understood`` is False when some insert
    expression could not be resolved -- the caller then skips the script rather
    than reporting a verdict it cannot back up.
    """
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) \
                    and child.func.attr in ("insert", "append"):
                owner = child.func.value
                if isinstance(owner, ast.Attribute) and owner.attr == "path" \
                        and isinstance(owner.value, ast.Name) and owner.value.id == "sys":
                    if len(child.args) >= 2:
                        calls.append((child.func.attr, resolver.resolve(child.args[1])))
            visit(child)

    visit(tree)
    if any(p is None for _, p in calls):
        return [], False
    front = [p for kind, p in calls if kind == "insert" and p is not None]
    back = [p for kind, p in calls if kind == "append" and p is not None]
    # insert(0, x) puts x in front; the last such call wins the front slot.
    return list(reversed(front)) + back, True


def _imports(tree: ast.AST) -> list[str]:
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _script_files() -> list[Path]:
    return sorted(
        p for p in _SCRIPTS.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "__init__.py"
    )


def _audit(script: Path) -> tuple[str, str | None]:
    """``(status, reason)`` for ``script``: status is ok / skip / fail."""
    text = script.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(text, filename=str(script))
    except SyntaxError:
        return "skip", None  # not this guard's business

    resolver = _Resolver(script)
    resolver.bind(tree)
    paths, understood = _sys_path_inserts(tree, resolver)
    if not understood:
        return "skip", None  # cannot prove anything -- skip rather than cry wolf

    mods = [m for m in _imports(tree) if m.split(".")[0] not in _OPTIONAL_TOP_LEVEL]
    tops = {m.split(".")[0] for m in mods}
    needs_scripts = any(m.startswith("scripts.") for m in mods)
    if not needs_scripts and not (tops & _SHADOWED_NAMES):
        return "ok", None

    root = ()
    src = ("src",)
    scripts_dir = ("scripts",)

    if needs_scripts and root not in paths:
        return "fail", "imports `scripts.*` but never puts the repo root on sys.path"

    # A name that exists in BOTH trees (today: `ml`) is resolved by whichever
    # directory lands in front. `scripts/ml` would shadow `src/ml`, and then
    # `ml.hwdataset` & co. vanish -- so src/ must come first. Only scripts
    # that put scripts/ on the path can cause that; the rest resolve `ml`
    # from the environment (the project is installed editable).
    hazardous = sorted(tops & _SHADOWED_NAMES)
    if hazardous and scripts_dir in paths:
        if src not in paths:
            return "fail", (
                f"imports `{hazardous[0]}.*` and puts ROOT/\"scripts\" on sys.path, but never "
                "ROOT/\"src\" -- the scripts/ directory shadows the real package"
            )
        if paths.index(scripts_dir) < paths.index(src):
            return "fail", (
                f"imports `{hazardous[0]}.*` but puts ROOT/\"scripts\" in front of "
                f"ROOT/\"src\" -- `{hazardous[0]}` then resolves to scripts/{hazardous[0]} "
                f"and `{hazardous[0]}.hwdataset` & co. vanish"
            )
    return "ok", None


def test_scripts_are_importable_in_file_form() -> None:
    """No script may rely on a cwd that a file-form launch does not provide."""
    failures: list[str] = []
    for script in _script_files():
        status, reason = _audit(script)
        if status == "fail":
            rel = script.relative_to(_ROOT).as_posix()
            failures.append(f"  {rel}\n      -> {reason}")
    assert not failures, (
        "these scripts cannot import their own dependencies when launched as "
        "`python scripts/...` (the form every registered reproduction command "
        "uses):\n" + "\n".join(failures)
    )


def test_guard_actually_detects_a_broken_bootstrap(tmp_path: Path) -> None:
    """The guard must fail on a script that reproduces the original defect."""
    broken = tmp_path / "scripts" / "sub" / "broken.py"
    broken.parent.mkdir(parents=True)
    broken.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n"
        "sys.path.insert(0, str(ROOT / 'scripts'))\n"
        "\n"
        "from scripts.sub.other import thing\n",
        encoding="utf-8",
    )
    # Rewrite the audit to work against the temporary tree.
    original = _ROOT
    try:
        globals()["_ROOT"] = tmp_path
        globals()["_SCRIPTS"] = tmp_path / "scripts"
        status, reason = _audit(broken)
    finally:
        globals()["_ROOT"] = original
        globals()["_SCRIPTS"] = original / "scripts"
    assert status == "fail", "guard failed to notice a broken bootstrap"
    assert reason


def test_guard_is_not_vacuous() -> None:
    """The audit must really inspect the scripts, not skip them all.

    A resolver bug once made every script "unresolvable", and the guard passed
    while checking nothing -- so the audited count is asserted here.
    """
    scripts = _script_files()
    assert len(scripts) > 100, "script discovery collapsed"
    statuses = [_audit(s)[0] for s in scripts]
    audited = statuses.count("ok")
    skipped = statuses.count("skip")
    assert audited >= 20, f"only {audited} scripts were actually audited ({skipped} skipped)"
    assert skipped <= len(scripts) // 2, (
        f"{skipped}/{len(scripts)} scripts were skipped -- the guard is checking "
        "almost nothing"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
