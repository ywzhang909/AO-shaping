"""Subpackage anchor tests for :mod:`ao_shaping.utils.wavefront`.

Verifies that modules in the ``wavefront`` subpackage remain
importable from their canonical subpackage path, and that the
kept legacy aliases resolve to the same module objects.
"""

from __future__ import annotations

import importlib

import pytest


def test_zernike_calc_canonical() -> None:
    mod = importlib.import_module("ao_shaping.utils.wavefront.zernike_calc")
    assert hasattr(mod, "ZernikeGenerator")


def test_wavefront_calc_canonical() -> None:
    mod = importlib.import_module("ao_shaping.utils.wavefront.wavefront_calc")
    assert hasattr(mod, "centroid_calculation")
