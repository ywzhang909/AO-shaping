"""R-49 (#65): ``utils/math`` must stay a numpy-only leaf, single source of truth.

The fitting kernel (``fit_linear_slope``) and the stability-judgement kernels
(``settle_time_s``, ``snr_vs_averages``, ``exposure_monotonicity``) used to live
in ``tools/kernels/bench_kernels.py`` alongside camera/SLM duck-typed code. They
are pure numpy, so they now have a leaf home here; ``bench_kernels`` re-exports
them for backward compatibility.

Two invariants are pinned:

1. **Leaf contract** -- these modules import only the standard library and
   ``numpy`` (no ``ao_shaping.*``, no relative import, at module level or
   deferred). The guard walks the AST so a deferred import inside a function
   cannot hide from a naive grep.
2. **Single source** -- ``bench_kernels`` re-exports the *same* objects, so a
   future edit cannot fork a second copy.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import ao_shaping.tools.kernels.bench_kernels as bench_kernels
from ao_shaping.utils.math import array_shape, fitting, stability

_MODULES = [
    Path(array_shape.__file__).resolve(),
    Path(fitting.__file__).resolve(),
    Path(stability.__file__).resolve(),
]


def _allowed_roots() -> frozenset[str]:
    return frozenset(sys.stdlib_module_names) | {"__future__", "numpy"}


_ALLOWED = _allowed_roots()


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
    for module in _MODULES:
        offenders = sorted(_imported_roots(module) & {"ao_shaping"})
        assert not offenders, f"{module.name} must not import {offenders}"


def test_leaf_imports_only_stdlib_and_numpy() -> None:
    for module in _MODULES:
        unknown = sorted(r for r in _imported_roots(module) if r not in _ALLOWED)
        assert not unknown, f"{module.name} may only import stdlib+numpy; got {unknown}"


def test_no_deferred_ao_shaping_import_can_hide_in_a_function() -> None:
    for module in _MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        nested = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "ao_shaping"
            )
            or (
                isinstance(node, ast.Import)
                and any("ao_shaping" in a.name for a in node.names)
            )
        ]
        assert not nested, (
            f"{module.name}: deferred ao_shaping import at lines "
            f"{[n.lineno for n in nested]}"
        )


def test_bench_kernels_reexports_the_same_objects() -> None:
    assert bench_kernels.fit_linear_slope is fitting.fit_linear_slope
    assert bench_kernels.settle_time_s is stability.settle_time_s
    assert bench_kernels.snr_vs_averages is stability.snr_vs_averages
    assert bench_kernels.exposure_monotonicity is stability.exposure_monotonicity


def test_fit_exact_has_one_implementation_across_consumers() -> None:
    """R-65/#87: ``resample`` and ``phase_embed`` used to carry private copies.

    Imported inside the function on purpose - ``resample`` pulls ``scipy``, which
    must stay out of the leaf modules this file guards.
    """
    from ao_shaping.utils.image import resample
    from ao_shaping.utils.slm import phase_embed

    assert resample.fit_exact is array_shape.fit_exact
    assert phase_embed.fit_exact is array_shape.fit_exact


def test_fit_exact_pads_then_crops_to_the_requested_shape() -> None:
    assert array_shape.fit_exact(np.ones((3, 3)), (5, 5)).shape == (5, 5)
    assert array_shape.fit_exact(np.ones((9, 9)), (4, 4)).shape == (4, 4)
    assert array_shape.fit_exact(np.ones((5, 5)), (5, 5)).shape == (5, 5)


def test_fit_linear_slope_recovers_line_and_flags_degenerate() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    slope, intercept = fitting.fit_linear_slope(x, 2.0 * x + 1.0)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    nan_slope, nan_intercept = fitting.fit_linear_slope(np.array([1.0]), np.array([1.0]))
    assert math.isnan(nan_slope) and math.isnan(nan_intercept)


def test_settle_time_s_and_length_guard() -> None:
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    deltas = [10.0, 5.0, 1.0, 1.05, 0.98]
    assert stability.settle_time_s(deltas, times, frac=0.10, run=2) == 2.0
    try:
        stability.settle_time_s([1.0, 2.0], [0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise ValueError")


def test_snr_vs_averages_scales_and_validates_floor() -> None:
    out = stability.snr_vs_averages([10.0, 10.0, 10.0, 10.0], floor=2.0, ks=[1, 4])
    assert out[1] == pytest.approx(5.0)
    assert out[4] == pytest.approx(10.0)
    try:
        stability.snr_vs_averages([1.0], floor=0.0, ks=[1])
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive floor must raise ValueError")


def test_exposure_monotonicity_verdicts() -> None:
    mono = stability.exposure_monotonicity([1.0, 2.0, 4.0], [10.0, 21.0, 42.0])
    assert mono["verdict"] == "monotonic"
    bad = stability.exposure_monotonicity([1.0, 2.0, 4.0], [10.0, 5.0, 42.0])
    assert bad["verdict"] == "non_monotonic"
