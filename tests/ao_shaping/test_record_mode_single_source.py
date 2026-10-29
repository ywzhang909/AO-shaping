"""Issue #130: the ``"max"`` / ``"min"`` vocabulary must not be spelled out again.

The two single-source modules are pinned by unit tests
(``utils/io/test_record_mode.py`` and ``utils/test_objective_direction.py``);
this file is the tree-wide half of the guard. It walks ``src/`` and ``scripts/``
by AST -- not grep -- for the three ways the duplication used to regrow:

* **a second ``class RecordMode``** -- ``utils/io/file.py`` had one while the
  copy in ``utils/image/target/objective.py`` grew a third; two structurally
  equal enums are still different types, so ``isinstance`` / ``is`` checks and
  the lazy attribute in ``ao_shaping.utils.__init__`` silently disagree;
* **a second definition of the direction vocabulary** -- ``MAXIMIZE_OBJECTIVES``
  / ``MINIMIZE_OBJECTIVES`` / ``objective_direction`` / ``objective_maximizes``
  had four copies that had already drifted (one was missing ``quality``);
* **call sites re-spelling the literals** -- 24 ``Recorder(mode="max")``
  constructions and 6 ``objective_mode == "max"`` comparisons in
  ``slm_zernike_pib.py``.

A plain grep cannot do this job: ``from ... import MAXIMIZE_OBJECTIVES`` is a
*use* the rule must allow, ``MAXIMIZE_OBJECTIVES = (...)` is a *definition* it
must reject, and ``""`` inside a docstring mentions both. Only the AST separates
them. ``scripts/`` is scanned too because the report generators mirror the
optimizer's vocabulary by hand.

Deliberately *not* flagged: ``utils/image/target/objective.py`` keeps a
``SQUARE_OBJECTIVE_CHOICES`` tuple, and ``optimizer/wfless/slm_square_shaping.py``
compares ``mode == "max"`` to pick which half of the CCD frame to search for the
zero-order spot. Those are different questions ("is this objective ascended?"
vs "where is the spot?"), and folding them into one vocabulary is what would
make the code wrong.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"

#: Vendored SDK wrappers and scratch notebooks are third-party code, not the
#: project's vocabulary (same exclusions as ``ruff.toml``).
_EXCLUDED_PARTS = ("_miicam_sdk", "libs", "notebooks")

#: The one module allowed to *define* each name.
_RECORD_MODE_HOME = "src/ao_shaping/utils/io/record_mode.py"
_DIRECTION_HOME = "src/ao_shaping/utils/objective_direction.py"
_DIRECTION_NAMES = frozenset(
    {
        "MAXIMIZE_OBJECTIVES",
        "MINIMIZE_OBJECTIVES",
        "objective_direction",
        "objective_maximizes",
    }
)


def _scanned_files() -> list[Path]:
    files = sorted(SRC.rglob("*.py")) + sorted(SCRIPTS.rglob("*.py"))
    return [
        path
        for path in files
        if not any(part in path.parts for part in _EXCLUDED_PARTS)
    ]


def _read(path: Path) -> str:
    # ``utf-8-sig``: ``ast.parse`` rejects a leading BOM outright
    # (``SyntaxError: invalid non-printable character U+FEFF``), and one file in
    # the tree did carry one. CPython itself tolerates it at import time, so a
    # BOM must not turn this guard into a parse crash.
    return path.read_text(encoding="utf-8-sig")


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _text(node: ast.AST, source: str) -> str:
    """Source text of ``node`` (3.9+ ``end_col_offset`` aware)."""
    segment = ast.get_source_segment(source, node)
    return segment or ""


def _is_mode_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in {"max", "min"}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _record_mode_literal_sites(source: str) -> list[int]:
    """Line numbers of ``Recorder(..., mode="max"|"min")``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "Recorder":
            continue
        for keyword in node.keywords:
            if keyword.arg == "mode" and _is_mode_literal(keyword.value):
                lines.append(keyword.value.lineno)
    return lines


def _objective_string_comparisons(source: str) -> list[int]:
    """Line numbers of ``<objective ...> == "max"|"min"`` comparisons."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        mentions_objective = any(
            "objective" in _text(operand, source) for operand in operands
        )
        if not mentions_objective:
            continue
        if any(_is_mode_literal(operand) for operand in operands):
            lines.append(node.lineno)
    return lines


def _definitions(source: str) -> list[tuple[str, int]]:
    """``(name, lineno)`` for every definition of a guarded name."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "RecordMode":
            found.append((node.name, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _DIRECTION_NAMES:
                found.append((node.name, node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _DIRECTION_NAMES:
                    found.append((target.id, node.lineno))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in _DIRECTION_NAMES:
                found.append((node.target.id, node.lineno))
    return found


def test_no_recorder_call_passes_a_mode_literal() -> None:
    offenders = []
    for path in _scanned_files():
        for lineno in _record_mode_literal_sites(
            _read(path)
        ):
            offenders.append(f"{_rel(path)}:{lineno}")
    assert not offenders, (
        "Recorder(...) must pass RecordMode.MAX / RecordMode.MIN, not the bare "
        "string (issue #130): " + ", ".join(offenders)
    )


def test_no_comparison_against_a_bare_objective_mode_string() -> None:
    offenders = []
    for path in _scanned_files():
        for lineno in _objective_string_comparisons(
            _read(path)
        ):
            offenders.append(f"{_rel(path)}:{lineno}")
    assert not offenders, (
        "Compare the RecordMode member with `.maximizing` instead of `== \"max\"` "
        "(issue #130): " + ", ".join(offenders)
    )


def test_record_mode_is_defined_exactly_once() -> None:
    homes = {}
    for path in _scanned_files():
        for name, lineno in _definitions(_read(path)):
            if name == "RecordMode":
                homes.setdefault(_rel(path), []).append(lineno)
    assert list(homes) == [_RECORD_MODE_HOME], (
        "RecordMode must be defined only in "
        f"{_RECORD_MODE_HOME}; found {homes}"
    )


def test_the_direction_vocabulary_is_defined_exactly_once() -> None:
    homes: dict[str, list[str]] = {}
    for path in _scanned_files():
        for name, lineno in _definitions(_read(path)):
            if name in _DIRECTION_NAMES:
                homes.setdefault(_rel(path), []).append(f"{name}@{lineno}")
    assert list(homes) == [_DIRECTION_HOME], (
        "The objective-direction vocabulary must be defined only in "
        f"{_DIRECTION_HOME}; found {homes}"
    )


@pytest.mark.parametrize(
    "helper,call",
    [
        (_record_mode_literal_sites, 'Recorder(mark="cv", mode="max")'),
        (_record_mode_literal_sites, "Recorder(mode='min')"),
        (
            _objective_string_comparisons,
            'if objective_mode == "max":\n    pass',
        ),
        (
            _objective_string_comparisons,
            'maximize=(objective_direction(objective) != "min")',
        ),
    ],
)
def test_the_guard_can_see_the_shapes_it_was_written_for(helper, call: str) -> None:
    # Non-vacuity: a guard that silently matches nothing would pass forever.
    assert helper(call) == [1]


@pytest.mark.parametrize(
    "snippet",
    [
        'recorder = Recorder(mark="cv", mode=RecordMode.MIN)',
        "if objective_mode.maximizing:\n    pass",
        "from ao_shaping.utils.objective_direction import MAXIMIZE_OBJECTIVES",
        'if mode == "max":\n    pass',  # _locate_square_center: not a direction
    ],
)
def test_the_guard_does_not_fire_on_the_corrected_forms(snippet: str) -> None:
    assert _record_mode_literal_sites(snippet) == []
    assert _objective_string_comparisons(snippet) == []
