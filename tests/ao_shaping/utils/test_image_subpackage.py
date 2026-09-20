"""Subpackage anchor tests for :mod:`ao_shaping.utils.image`.

Verifies that modules moved into the ``image`` subpackage remain
importable from their canonical subpackage path, and that the
kept legacy alias (``hardware_utils``) resolves to the same
module object.
"""

from __future__ import annotations

import importlib

import pytest


def test_hardware_utils_legacy_path_is_alias() -> None:
    """Module-global state mutations must write through to the real module."""
    legacy = importlib.import_module("ao_shaping.utils.hardware_utils")
    canonical = importlib.import_module("ao_shaping.utils.image.hardware_utils")
    assert legacy is canonical
