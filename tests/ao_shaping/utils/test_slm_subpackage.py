"""Subpackage anchor tests for :mod:`ao_shaping.utils.slm`.

Locks the refactor contract: every module moved into the ``slm`` subpackage
must stay importable from BOTH its canonical subpackage path and its legacy
top-level path, and the objects returned at each path must be identical.

``slm_utils`` is special: tests reset its module-global slot tracker via the
legacy name (``slm_utils._last_slm_slot``), so the legacy path must resolve to
the SAME module object (alias), not a copy.
"""

from __future__ import annotations

import importlib

import pytest

# module -> representative public symbol to compare across paths
SLM_SYMBOLS = {
    "pattern_helper": "PatternHelper",
    "slm_lut": "load_lut",
}


@pytest.mark.parametrize("mod_name,symbol", SLM_SYMBOLS.items())
def test_canonical_and_legacy_paths_identical(mod_name: str, symbol: str) -> None:
    canonical = importlib.import_module(f"ao_shaping.utils.slm.{mod_name}")
    legacy = importlib.import_module(f"ao_shaping.utils.{mod_name}")
    assert getattr(canonical, symbol) is getattr(legacy, symbol)


def test_slm_utils_legacy_path_is_alias() -> None:
    """Module-global state mutations must write through to the real module."""
    legacy = importlib.import_module("ao_shaping.utils.slm_utils")
    canonical = importlib.import_module("ao_shaping.utils.slm.slm_utils")
    assert legacy is canonical