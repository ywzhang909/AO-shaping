"""R-46 (#62): ``utils/wavefront/beam_amplitude`` must stay a numpy-only leaf.

``gaussian_grid`` used to live in ``tools/kernels/bench_kernels.py`` beside
camera/SLM duck-typed code. It is pure numpy, so it now has a leaf home here and
``bench_kernels`` re-exports it -- which removes the last module-scope
``optimizer -> tools`` edge (``slm_model_in_loop`` imports it from here).

Three invariants are pinned:

1. **Leaf contract** -- the module imports only ``numpy`` (no ``ao_shaping.*``,
   no relative import, module-scope or deferred). The guard walks the AST so a
   deferred import inside a function cannot hide from a naive grep.
2. **Single source** -- ``bench_kernels`` re-exports the *same* object, so a
   future edit cannot fork a second copy.
3. **Behaviour** -- the exact numeric output is locked, including the odd-region
   mask boundary. The float aperture radius ``region / 2.0`` (threshold
   ``(region/2)**2``) is what distinguishes this generator from the int-clamped
   near-duplicates in ``model_in_loop_shaping`` (R-51 territory); an accidental
   ``int(region // 2)`` here would silently change the mask and is caught below.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

import ao_shaping.tools.kernels.bench_kernels as bench_kernels
from ao_shaping.utils.wavefront import beam_amplitude
from ao_shaping.utils.wavefront.beam_amplitude import gaussian_grid

_MODULE = Path(beam_amplitude.__file__).resolve()


def _allowed_roots() -> frozenset[str]:
    return frozenset(sys.stdlib_module_names) | {"__future__", "numpy"}


_ALLOWED = _allowed_roots()


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
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


def test_leaf_imports_only_stdlib_and_numpy() -> None:
    unknown = sorted(r for r in _imported_roots(_MODULE) if r not in _ALLOWED)
    assert not unknown, f"{_MODULE.name} may only import stdlib+numpy; got {unknown}"


def test_no_deferred_ao_shaping_import_can_hide_in_a_function() -> None:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8-sig"), filename=str(_MODULE))
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
        f"{_MODULE.name}: deferred ao_shaping import at lines "
        f"{[n.lineno for n in nested]}"
    )


def test_bench_kernels_reexports_the_same_object() -> None:
    assert bench_kernels.gaussian_grid is gaussian_grid


def test_even_region_spot_values() -> None:
    amp = gaussian_grid(16, 4.0)
    assert amp.shape == (16, 16)
    assert amp.dtype == np.float64
    assert amp[8, 8] == pytest.approx(1.0)  # r2 = 0 at the centre
    assert amp[0, 0] == 0.0  # corner outside the inscribed circle
    assert amp[8, 0] == pytest.approx(np.exp(-2.0))  # r2 = 64, waist 4 -> exp(-64/32)
    assert int((amp > 0).sum()) == 195


def test_odd_region_uses_float_aperture_radius() -> None:
    """The mask radius is the float ``region / 2.0`` (thr 56.25), NOT ``region // 2`` (49).

    A huge waist makes the Gaussian ~1 everywhere, so the only structure is the
    aperture mask. Exactly 16 pixels sit in ``(49, 56.25]`` -- kept by the float
    threshold, dropped by an int clamp. Pinning them guards the R-51 merge.
    """
    amp = gaussian_grid(15, 1e9)
    yy, xx = np.mgrid[0:15, 0:15]
    r2 = (xx - 7.5) ** 2 + (yy - 7.5) ** 2
    boundary = (r2 > 49.0) & (r2 <= 56.25)
    assert int(boundary.sum()) == 16
    assert np.all(amp[boundary] == pytest.approx(1.0))  # float version keeps them
    assert int((amp > 0).sum()) == 172


def test_waist_is_clamped_to_avoid_div_by_zero() -> None:
    amp = gaussian_grid(8, 0.0)
    assert amp[4, 4] == pytest.approx(1.0)  # centre, r2 = 0
    assert amp[4, 5] == 0.0  # r2 = 1, waist clamped to 1e-6 -> exp(-5e11) ~ 0
