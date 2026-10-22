"""Leaf guards and behaviour tests for :mod:`ml.shaping.gs_reference`.

The golden-reference phase must stay a **leaf**: importing it may not drag
``torch``, ``ao_shaping.drivers``, ``ao_shaping.optimizer`` or
``ao_shaping.runners`` into the process, and its only ``ao_shaping`` import (the
canonical ``gs_shape``) must sit *inside* the function body so module import
stays cheap.

The static guards mirror ``tests/ao_shaping/utils/test_cli_params_leaf.py``: they
walk the **AST**, not grep, so an import smuggled into a function body is still
seen. An isolation probe then loads the module file *directly* to prove its body
is torch-/ao_shaping-free at runtime.

Repo caveat (intentional, out of this task's scope)
---------------------------------------------------
A *package-qualified* ``import ml.shaping.gs_reference`` also executes
``src/ml/__init__.py``, which eagerly imports ``torch`` (via ``ml.phase`` /
``ml.zernike``). That grandparent behaviour is why the isolation probe loads the
file by path (bypassing the ``ml`` package init) and why the AST guards -- not a
naive ``sys.modules`` check on a package import -- are the primary invariant.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.square_shaping import (
    SquareShapingConfig,
)
from ml.shaping.gs_reference import gs_golden_phase

# tests/ao_shaping/ml/shaping/test_gs_reference.py -> parents[4] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_GOLDEN_PATH = _REPO_ROOT / "src" / "ml" / "shaping" / "gs_reference.py"

#: Roots that must never be imported by the leaf, at module load or deferred.
_FORBIDDEN_ROOTS = (
    "torch",
    "ao_shaping.drivers",
    "ao_shaping.optimizer",
    "ao_shaping.runners",
)

_CANONICAL_GS_MODULE = "ao_shaping.algorithm.signal_processing.square_shaping"


# ---------------------------------------------------------------------------
# AST helpers (mirror test_cli_params_leaf.py's tree walk)
# ---------------------------------------------------------------------------


def _parse() -> ast.Module:
    return ast.parse(_GOLDEN_PATH.read_text(encoding="utf-8-sig"), filename=str(_GOLDEN_PATH))


def _leaf_imports() -> list[tuple[int, str, bool]]:
    """``(lineno, module, inside_function)`` for every import at any depth.

    ``inside_function`` distinguishes a **deferred** import (hidden in a function
    body, invisible to grep on the top level) from one that runs at module load.
    """
    tree = _parse()
    found: list[tuple[int, str, bool]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.fn_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.fn_depth += 1
            self.generic_visit(node)
            self.fn_depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                found.append((node.lineno, alias.name, self.fn_depth > 0))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = "." * node.level + (node.module or "")
            found.append((node.lineno, module, self.fn_depth > 0))

    _Visitor().visit(tree)
    return found


def _matches(module: str, root: str) -> bool:
    return module == root or module.startswith(root + ".")


# ---------------------------------------------------------------------------
# 1. the leaf invariant (static, AST)
# ---------------------------------------------------------------------------


def test_module_load_imports_nothing_forbidden() -> None:
    """Nothing forbidden may execute at import time (module top level)."""
    offenders = [
        (line, module)
        for line, module, in_function in _leaf_imports()
        if not in_function
        and (module.split(".")[0] == "torch" or module.split(".")[0] == "ao_shaping")
    ]
    assert not offenders, (
        f"{_GOLDEN_PATH.name} imports {offenders} at module load; "
        "the leaf may only import numpy at module level."
    )


def test_no_forbidden_import_at_any_depth() -> None:
    """AST walk, not grep: a deferred forbidden import is still caught."""
    hits = [
        (line, module)
        for line, module, _ in _leaf_imports()
        if any(_matches(module, root) for root in _FORBIDDEN_ROOTS)
    ]
    assert not hits, (
        f"{_GOLDEN_PATH.name} must never import {_FORBIDDEN_ROOTS}; found {hits}."
    )


def test_canonical_gs_import_is_deferred_and_correct() -> None:
    """The one ``ao_shaping`` import must be the canonical GS module, deferred."""
    ao_imports = [
        (line, module, in_function)
        for line, module, in_function in _leaf_imports()
        if module.split(".")[0] == "ao_shaping"
    ]
    assert ao_imports, "expected a lazy import of the canonical gs_shape"

    eager = [entry for entry in ao_imports if not entry[2]]
    assert not eager, f"ao_shaping imported at module load: {eager}"

    modules = {module for _, module, _ in ao_imports}
    assert modules == {_CANONICAL_GS_MODULE}, modules

    imported_names = {
        alias.name
        for node in ast.walk(_parse())
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {"gs_shape", "ShapingResult"} <= imported_names


# ---------------------------------------------------------------------------
# 2. the leaf invariant (runtime isolation)
# ---------------------------------------------------------------------------

_ISOLATION_PROBE = """\
import importlib.util, json, sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_gs_ref_leaf_probe", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
forbidden = [
    name for name in (
        "torch",
        "ao_shaping",
        "ao_shaping.drivers",
        "ao_shaping.optimizer",
        "ao_shaping.runners",
    )
    if name in sys.modules
]
print("FORBIDDEN:" + json.dumps(forbidden))
"""


def test_module_body_loads_without_torch_or_ao_shaping() -> None:
    """Loading the file directly must not pull torch / ao_shaping into sys.modules.

    A fresh interpreter is required -- sibling test modules already import torch,
    so an in-process ``sys.modules`` check would be meaningless.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATION_PROBE, str(_GOLDEN_PATH)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"

    payload = next(
        (line for line in proc.stdout.splitlines() if line.startswith("FORBIDDEN:")),
        None,
    )
    assert payload is not None, f"probe produced no verdict:\n{proc.stdout}"
    forbidden = json.loads(payload.split("FORBIDDEN:", 1)[1])
    assert forbidden == [], f"isolated module load pulled in {forbidden}"


# ---------------------------------------------------------------------------
# 3. behaviour
# ---------------------------------------------------------------------------


def _small_cfg() -> SquareShapingConfig:
    """A cheap, valid config: this tests plumbing, not shaping quality."""
    return SquareShapingConfig(n_grid=64, target_side_px=12, far_field_padding=4, seed=0)


class TestGoldenPhase:
    def test_returns_finite_float_phase_of_grid_shape(self) -> None:
        phase = gs_golden_phase(_small_cfg(), n_iters=8, seed=0)
        assert isinstance(phase, np.ndarray)
        assert phase.dtype == np.float64
        assert phase.shape == (64, 64)
        assert np.isfinite(phase).all()

    def test_phase_is_not_wrapped_into_zero_two_pi(self) -> None:
        """Raw radians in the canonical (-pi, pi] convention, not [0, 2pi)."""
        phase = gs_golden_phase(_small_cfg(), n_iters=8, seed=0)
        assert phase.min() < 0.0, "an [0, 2*pi) wrap would have no negative values"
        assert phase.max() <= np.pi + 1e-9

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_n_iters_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="n_iters"):
            gs_golden_phase(_small_cfg(), n_iters=bad, seed=0)

    def test_same_seed_is_deterministic(self) -> None:
        first = gs_golden_phase(_small_cfg(), n_iters=8, seed=123)
        second = gs_golden_phase(_small_cfg(), n_iters=8, seed=123)
        assert np.array_equal(first, second)

    def test_different_seed_changes_the_phase(self) -> None:
        """Proves the seed is actually threaded into the GS initial field."""
        a = gs_golden_phase(_small_cfg(), n_iters=8, seed=1)
        b = gs_golden_phase(_small_cfg(), n_iters=8, seed=2)
        assert a.shape == b.shape
        assert not np.array_equal(a, b)
