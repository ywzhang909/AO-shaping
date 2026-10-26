"""Pin the four ``--cam-id`` env-default sites in runner_common.py.

Item 73's env-name census deferred one seam here: the four ``cam_id``
dataclass defaults read ``Far_Cam_ID`` with a mixed literal repertoire
(``"0"`` str vs ``0`` int) and one read was eager (import time) while the
others defer through a lambda (item 81).  These tests pin the slice with
pure AST -- no import of runner_common, which drags hardware-adjacent
modules.  Contract being pinned:

* exactly four ``os.environ.get("Far_Cam_ID", ...)`` reads in the file;
* every one of them is a dataclass field default named ``cam_id``;
* every one is deferred (wrapped in a ``lambda`` -- eager reads are the
  import-time-valve class recorded in item 78);
* every fallback literal is the string ``"0"`` -- an int fallback is
  invisible at the parse layer (click casts STRING defaults) but diverges
  in env-unset ``vars(config)``-style dumps.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER_COMMON = ROOT / "src/ao_shaping/runners/runner_common.py"


def _sites() -> list[tuple[int, bool, object, str | None]]:
    """[(lineno, deferred, fallback_value, field_name)] for Far_Cam_ID reads."""
    source = RUNNER_COMMON.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_COMMON))
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def chain(node: ast.AST) -> list[ast.AST]:
        out: list[ast.AST] = []
        cur = parents.get(id(node))
        while cur is not None:
            out.append(cur)
            cur = parents.get(id(cur))
        return out

    found: list[tuple[int, bool, object, str | None]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "get"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "Far_Cam_ID" or len(node.args) < 2:
            continue
        fallback = node.args[1]
        assert isinstance(fallback, ast.Constant), (
            f"line {node.lineno}: Far_Cam_ID fallback must be a literal"
        )
        ancestors = chain(node)
        def _assign_target(a: ast.AST) -> str | None:
            if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name):
                return a.target.id
            if isinstance(a, ast.Assign):
                for t in a.targets:
                    if isinstance(t, ast.Name):
                        return t.id
            return None

        field = next(
            (name for a in ancestors if (name := _assign_target(a)) is not None),
            None,
        )
        deferred = any(isinstance(a, ast.Lambda) for a in ancestors)
        found.append((node.lineno, deferred, fallback.value, field))
    return found


def test_source_layout_is_where_this_file_assumes() -> None:
    """A silent path/depth slip would make the other checks vacuously pass."""
    assert RUNNER_COMMON.is_file(), f"source moved: {RUNNER_COMMON} -- fix parents[3] first"


def test_cam_id_env_defaults_are_four_deferred_string_literals() -> None:
    sites = _sites()
    assert len(sites) == 4, f"expected exactly 4 Far_Cam_ID reads, found {sites}"
    for lineno, deferred, fallback, field in sites:
        assert deferred, f"line {lineno}: Far_Cam_ID read is eager (import-time), not lambda"
        assert fallback == "0" and isinstance(fallback, str), (
            f"line {lineno}: fallback must be the string \"0\", got {fallback!r}"
        )
        assert field == "cam_id", f"line {lineno}: enclosing field must be cam_id, got {field}"
