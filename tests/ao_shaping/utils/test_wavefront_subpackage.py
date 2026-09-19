"""Subpackage anchor tests for :mod:`ao_shaping.utils.wavefront`.

Locks the refactor contract: every module moved into the ``wavefront``
subpackage must stay importable from BOTH its canonical subpackage path and its
legacy top-level path, and the objects returned at each path must be identical.

No side effects; import-only assertions.
"""

from __future__ import annotations

import importlib

import pytest

# module -> representative public symbol to compare across paths
WAVEFRONT_SYMBOLS = {
    "zernike_calc": "ZernikeGenerator",
    "zernike_utils": "generate_zernike_phase",
    "wavefront_calc": "centroid_calculation",
    "wfs_utils": "flatten_slopes",
    "phase_unwrap": "PhaseUnwrapper",
    "hadamard_calc": "HadamardGenerator",
    "matrix_utils": "compute_pinv",
    "vi": "Vi",
}


@pytest.mark.parametrize("mod_name,symbol", WAVEFRONT_SYMBOLS.items())
def test_canonical_and_legacy_paths_identical(mod_name: str, symbol: str) -> None:
    canonical = importlib.import_module(f"ao_shaping.utils.wavefront.{mod_name}")
    legacy = importlib.import_module(f"ao_shaping.utils.{mod_name}")
    assert getattr(canonical, symbol) is getattr(legacy, symbol)