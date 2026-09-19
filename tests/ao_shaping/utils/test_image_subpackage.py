"""Subpackage anchor tests for :mod:`ao_shaping.utils.image`.

Locks the refactor contract: every module moved into the ``image`` subpackage
must stay importable from BOTH its canonical subpackage path and its legacy
top-level path, and the objects returned at each path must be identical.

``hardware_utils`` is special: tests mutate its module-global recording state
via the legacy name (``hardware_utils._frames_dir``), so the legacy path must
resolve to the SAME module object (alias), not a copy.
"""

from __future__ import annotations

import importlib

import pytest

# module -> representative public symbol to compare across paths
IMAGE_SYMBOLS = {
    "spots_calc": "centroid",
    "beam_metrics": "compute_metrics",
    "targets": "crop_resize_to_grid",
    "resample": "resample_to_grid",
    "display": "ImageVoltagesDisplay",
    "gs_visualization": "GSVizCallback",
}


@pytest.mark.parametrize("mod_name,symbol", IMAGE_SYMBOLS.items())
def test_canonical_and_legacy_paths_identical(mod_name: str, symbol: str) -> None:
    canonical = importlib.import_module(f"ao_shaping.utils.image.{mod_name}")
    legacy = importlib.import_module(f"ao_shaping.utils.{mod_name}")
    assert getattr(canonical, symbol) is getattr(legacy, symbol)


def test_hardware_utils_legacy_path_is_alias() -> None:
    """Module-global state mutations must write through to the real module."""
    legacy = importlib.import_module("ao_shaping.utils.hardware_utils")
    canonical = importlib.import_module("ao_shaping.utils.image.hardware_utils")
    assert legacy is canonical