"""Every ``parents[N]`` in ``scripts/`` must actually resolve to the repo root.

Why this exists
---------------
``scripts/`` used to be flat, so ``Path(__file__).resolve().parents[1]`` happened
to be the repository root in all 83 of the files that used it. Regrouping
``scripts/`` into domain subfolders moves each file one level deeper, which turns
that expression into ``scripts/`` -- **silently**. No other test covers it:
``test_script_imports_resolve`` checks that ``ao_shaping.*`` imports resolve, and
``test_common_helpers_not_reintroduced`` only pins the bootstrap of 13 known
generators. A script whose data/report/output paths now resolve under
``scripts/`` would still import cleanly and still pass the suite, and would only
fail when someone actually ran it against real data.

So the invariant is asserted directly: for every ``parents[N]`` subscript in
every ``scripts/**/*.py``, ``path.resolve().parents[N]`` must equal the repo root.
All offenders are reported in one message rather than failing on the first.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"


def _read(path: Path) -> str:
    """Tolerate a UTF-8 BOM, which one checked-in script still carries."""
    return path.read_text(encoding="utf-8-sig")


def _anchors(tree: ast.AST) -> dict[str, int]:
    """Map local names to how many levels above the file they sit.

    ``scripts/call_graph/finalize_graph.py`` writes
    ``P = Path(__file__).resolve().parent`` then ``P.parents[1]``. Read
    literally that looks one level short, but ``P`` is already the *parent*, so
    ``P.parents[1]`` is the repo root and is correct. Judging the subscript
    without resolving the alias produces false positives on correct code, and a
    guard that cries wolf gets disabled.

    Returns ``{name: extra_levels}`` where ``extra_levels`` is 0 for the file
    itself and 1 for its parent directory. Unknown expressions are absent.
    """
    anchors: dict[str, int] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Attribute) and func.attr == "resolve"):
            continue
        if not (isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name)):
            continue
        if func.value.func.id != "Path":
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        # Path(__file__).resolve() -> 0 ; ...resolve().parent -> 1
        parent_expr = node.value.func.value
        extra = 1 if isinstance(parent_expr, ast.Attribute) and parent_expr.attr == "parent" else 0
        anchors[target.id] = extra
    return anchors


def _parents_checks(tree: ast.AST) -> list[tuple[int, str]]:
    """Every ``(base_levels, N)`` used in a ``<file-or-alias>.parents[N]``.

    ``base_levels`` is how many levels above this file the subscripted object
    sits (0 = the file, 1 = its directory), so ``base_levels + N`` is the total
    climb and must land on the repo root. Matched structurally on the AST so a
    comment or docstring example cannot be mistaken for real code.
    """
    anchors = _anchors(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr == "parents"):
            continue
        index = node.slice
        if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
            continue
        base = value.value
        if isinstance(base, ast.Name):
            if base.id not in anchors:
                continue
            found.append((anchors[base.id], index.value, base.id))
        elif isinstance(base, ast.Call):
            found.append((0, index.value, "Path(__file__).resolve()"))
    return found


def _script_files() -> list[Path]:
    return sorted(
        p
        for p in _SCRIPTS.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "__init__.py"
    )


def _plant_probe_under(tmp_path: Path) -> tuple[Path, Path]:
    """Recreate ``scripts/_common/_depth_probe_tmp/probe.py`` below a throwaway root.

    The self-checks need a file on disk for the scanner to read, but planting one
    in the checkout cannot run where the tree is mounted read-only, and a crash
    between creating it and removing it would leave a stray ``probe.py`` that
    :func:`_script_files` then reports as a real depth offender. Depth arithmetic
    is relative, so an identical structure under a fake root proves the same thing.
    """
    fake_root = tmp_path / "repo"
    planted = fake_root / "scripts" / "_common" / "_depth_probe_tmp" / "probe.py"
    planted.parent.mkdir(parents=True)
    return fake_root, planted


@pytest.mark.parametrize(
    "script", _script_files(), ids=lambda p: str(p.relative_to(_SCRIPTS))
)
def test_parents_depth_resolves_to_repo_root(script: Path) -> None:
    checks = _parents_checks(ast.parse(_read(script), filename=str(script)))
    if not checks:
        pytest.skip(f"{script.relative_to(_SCRIPTS)} uses no parents[N]")

    own_depth = len(script.relative_to(_ROOT).parts) - 1
    wrong = []
    for base_levels, index, label in sorted(set(checks)):
        total = base_levels + index
        if total < 0:
            wrong.append(f"{label}.parents[{index}] (negative)")
            continue
        if script.resolve().parents[total] != _ROOT:
            wrong.append(
                f"{label}.parents[{index}] resolves to {script.resolve().parents[total]} "
                f"(expected the repo root {_ROOT} -> use total depth {own_depth}, "
                f"i.e. parents[{own_depth - base_levels}])"
            )

    assert not wrong, (
        f"{script.relative_to(_ROOT)} computes a repo root that is not the repo "
        f"root: {wrong}. A script moved into a subfolder keeps its old depth and "
        "silently resolves its data/, report/ and output/ paths under scripts/."
    )


#: A drive letter is never valid for a path the repo itself resolves.
#: ``C:/Windows/Fonts/*.ttf`` is exempt: it is an OS resource, not a repo
#: location, it is probed with ``os.path.exists`` and degrades gracefully, and
#: there is no ``__file__``-relative way to spell it.
_WINDOWS_PATH = re.compile(
    r"""["'][A-Za-z]:[\\/]{1,2}(?!Windows[\\/])[^"']*["']"""
)


def test_no_script_hardcodes_a_machine_absolute_path() -> None:
    """A script must derive its paths from ``__file__``, never from a dev's machine.

    38 literals like ``Path(r"D:\\Projects\\TIFO\\AO-shaping")`` survived in
    ``scripts/ml/`` for months. On any other machine ``Path(r"D:\\...")`` is not
    an absolute path at all -- it is one weird relative directory name -- so the
    script imports cleanly, prints a plausible-looking path, and writes its
    output into ``./D:\\Projects\\...``. Nothing fails; the artefact is simply
    somewhere nobody looks.
    """
    offenders: list[str] = []
    for script in _script_files():
        for lineno, line in enumerate(_read(script).splitlines(), 1):
            if _WINDOWS_PATH.search(line):
                offenders.append(f"{script.relative_to(_ROOT)}:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "these scripts hardcode an absolute path from one developer's machine; "
        "derive it from __file__ instead:\n  " + "\n  ".join(offenders)
    )


def test_windows_path_guard_is_not_vacuous(tmp_path: Path) -> None:
    """The detector must fire on a literal like the ones it exists to reject."""
    _fake_root, planted = _plant_probe_under(tmp_path)
    planted.write_text("X = 1\n", encoding="utf-8")
    assert not _WINDOWS_PATH.search(_read(planted))
    planted.write_text('X = Path(r"D:\\Projects\\repo\\src")\n', encoding="utf-8")
    assert _WINDOWS_PATH.search(_read(planted)), "detector missed a planted literal"


def test_guard_is_not_vacuous(tmp_path: Path) -> None:
    """The scanner must actually see a nested file and actually flag its depth.

    A depth check that silently scans nothing is worse than no check: it goes
    green forever. This plants a deliberately wrong file at depth 3 and asserts
    the same routine rejects it, so the real-tree green above means something.
    """
    fake_root, planted = _plant_probe_under(tmp_path)
    planted.write_text("from pathlib import Path\nX = 1\n", encoding="utf-8")
    assert _parents_checks(ast.parse(_read(planted))) == [], (
        "planted fixture carries no parents[N]; the scanner cannot be proven"
    )
    planted.write_text(
        "from pathlib import Path\nX = Path(__file__).resolve().parents[1]\n",
        encoding="utf-8",
    )
    found = _parents_checks(ast.parse(_read(planted)))
    assert found == [(0, 1, "Path(__file__).resolve()")], (
        f"scanner missed the planted subscript: {found}"
    )

    # probe.py sits at scripts/_common/_depth_probe_tmp/probe.py = 3 directories
    # deep, so a real script here would need parents[3]; parents[1] is the wrong
    # depth this guard exists to catch.
    assert planted.resolve().parents[1] == fake_root / "scripts" / "_common"
    assert planted.resolve().parents[1] != fake_root
    assert planted.resolve().parents[3] == fake_root
