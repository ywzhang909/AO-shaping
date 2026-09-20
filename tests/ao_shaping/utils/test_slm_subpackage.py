"""Subpackage anchor tests for :mod:`ao_shaping.utils.slm`.

Verifies that modules in the ``slm`` subpackage remain
importable from their canonical subpackage path.
"""

from __future__ import annotations

import importlib


def test_pattern_helper_canonical() -> None:
    mod = importlib.import_module("ao_shaping.utils.wavefront.pattern_helper")
    assert hasattr(mod, "PatternHelper")
def test_slm_lut_canonical() -> None:
    mod = importlib.import_module("ao_shaping.utils.slm.slm_lut")
    assert hasattr(mod, "load_lut")
