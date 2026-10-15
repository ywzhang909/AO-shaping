"""R-81: ``bench_constants`` must stay a zero-``ao_shaping`` leaf.

``TILT_SHIFT_SCALE`` used to be spelled out as four separate literal
assignments (``bench_kernels``, ``runner_common``, ``slm_gs_refine``,
``scripts/slm/generate_bench_probe_report``). A bench change would update one
copy and silently leave the others stale. The single source of truth now
lives in ``ao_shaping.utils.slm.bench_constants``, and every consumer
imports it.

This module must stay a **leaf** -- no ``ao_shaping`` imports, no
non-stdlib dependency, no relative imports -- so that importing it from a
hardware script never drags in an SDK / ``ctypes`` / ``torch``. The guard
walks the AST (a deferred import inside a function would pass a naive
``grep``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import ao_shaping.utils.slm.bench_constants as bench_constants
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)

_MODULE = Path(bench_constants.__file__).resolve()
_REPO_ROOT = _MODULE.parents[4]


def _stdlib_roots() -> frozenset[str]:
    roots = set(sys.stdlib_module_names)
    roots |= {"__future__"}
    return frozenset(roots)


_STDLIB = _stdlib_roots()


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import {node.module!r}"
            roots.add((node.module or "").split(".")[0])
    return roots


def test_leaf_imports_nothing_from_ao_shaping() -> None:
    offenders = sorted(_imported_roots(_MODULE) & {"ao_shaping"})
    assert not offenders, f"{_MODULE.name} must not import {offenders}"


def test_leaf_imports_only_stdlib() -> None:
    unknown = sorted(r for r in _imported_roots(_MODULE) if r not in _STDLIB)
    assert not unknown, f"{_MODULE.name} may only import stdlib; got {unknown}"


def test_no_deferred_ao_shaping_import_can_hide_in_a_function() -> None:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"), filename=str(_MODULE))
    nested = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "ao_shaping"
        )
        or (isinstance(node, ast.Import) and any("ao_shaping" in a.name for a in node.names))
    ]
    assert not nested, f"deferred ao_shaping import at lines {[n.lineno for n in nested]}"


def test_value_is_the_bench_scale() -> None:
    """The constant is the measured 2f focal scale, not a drifted copy."""
    assert float(TILT_SHIFT_SCALE) == 7400.0


def test_panel_geometry_constants_are_the_bench_values() -> None:
    """The R-47 panel constants are the measured bench geometry, not drifted copies."""
    assert (SLM_PANEL_W, SLM_PANEL_H) == (1920, 1200)
    assert float(SLM_PITCH_M) == 8e-6
    assert BEAM_CENTER_PANEL == (960, 600)
    assert BEAM_RADIUS_PANEL == 450
