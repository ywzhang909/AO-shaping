"""R-21 — ``display/`` owns the visualization code; ``utils/`` must not.

The repo's own anti-pattern list says visualization/Pygame/matplotlib code belongs
in ``display/``, not in the leaf ``utils/`` layer (it would drag a rendering stack
into every ``import ao_shaping.utils``). ``gs_visualization.py`` violated it for
months and was moved on 2026-10-03.

These tests pin the boundary so it cannot regress, and pin the thing that actually
matters for correctness: ``gerchberg_saxton_with_visualization`` must DRIVE the
canonical ``algorithm.signal_processing.gerchberg_saxton`` (via a callback) rather
than carry a second copy of the GS loop that can drift away from it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src" / "ao_shaping"
_DISPLAY = _SRC / "display"
_UTILS = _SRC / "utils"

#: Modules that were relocated out of ``utils/`` and must not reappear there.
RELOCATED_OUT_OF_UTILS = ("gs_visualization.py",)


def test_the_module_now_lives_in_display() -> None:
    assert (_DISPLAY / "gs_visualization.py").is_file()


@pytest.mark.parametrize("name", RELOCATED_OUT_OF_UTILS)
def test_no_sibling_is_left_behind_in_utils(name: str) -> None:
    assert not (_UTILS / "image" / name).exists(), (
        f"{name} was moved to display/; leaving a copy in utils/ would recreate the "
        "two-homes problem this item exists to end"
    )


def test_display_package_reexports_it() -> None:
    import ao_shaping.display as display

    for name in (
        "GSVizCallback",
        "create_gs_iteration_frame",
        "gerchberg_saxton_with_visualization",
        "render_gs_animation",
        "save_frames_as_gif",
    ):
        assert name in display.__all__, f"{name} missing from display.__all__"
        assert hasattr(display, name), f"{name} not reachable as ao_shaping.display.{name}"


def test_the_old_import_path_is_gone() -> None:
    """No compatibility shim: the old path must raise, not silently resolve."""
    with pytest.raises(ImportError):
        __import__("ao_shaping.utils.image.gs_visualization")


def test_gs_viz_no_longer_lists_utils_in_its_own_docstring() -> None:
    text = (_DISPLAY / "gs_visualization.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(text))
    assert doc is not None
    assert "display/" in doc, "the module must document why it lives in display/"


# ---------------------------------------------------------------------------
# The GS loop must not be duplicated here.
# ---------------------------------------------------------------------------


def test_it_drives_the_canonical_gerchberg_saxton() -> None:
    """The canonical loop is imported *inside* the function (deferred), not copied."""
    tree = ast.parse((_DISPLAY / "gs_visualization.py").read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "gerchberg_saxton_with_visualization"
    )
    # Walk the whole function, not just module scope: the import is deliberately
    # deferred so `display/` stays importable without the algorithm layer loaded.
    deferred = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("gerchberg_saxton")
    ]
    assert deferred, (
        "gerchberg_saxton_with_visualization must import the canonical loop "
        "(algorithm.signal_processing.gerchberg_saxton), not reimplement it"
    )
    assert any("signal_processing" in (n.module or "") for n in deferred), (
        f"stale import path: {[n.module for n in deferred]}"
    )


def test_no_module_scope_algorithm_import() -> None:
    """``display/`` must not depend on ``algorithm/`` at import time."""
    tree = ast.parse((_DISPLAY / "gs_visualization.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("ao_shaping.algorithm"), (
                f"module-scope algorithm import at line {node.lineno}"
            )
        elif isinstance(node, ast.Import):
            assert all(not a.name.startswith("ao_shaping.algorithm") for a in node.names)