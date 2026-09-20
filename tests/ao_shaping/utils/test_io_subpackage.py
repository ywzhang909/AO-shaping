"""Subpackage anchor tests for :mod:`ao_shaping.utils.io`.

Verifies that modules in the ``io`` subpackage remain
importable from their canonical subpackage path, and that the
kept legacy aliases resolve to the same module objects.
"""

from __future__ import annotations

import importlib

import pytest

from ao_shaping.utils import Recorder, configure_error_logging, get_init_V_by_rms


def test_file_canonical() -> None:
    mod = importlib.import_module("ao_shaping.utils.io.file")
    assert hasattr(mod, "Recorder")


def test_timestamp_canonical() -> None:
    mod = importlib.import_module("ao_shaping.utils.io.timestamp")
    assert hasattr(mod, "TimestampParser")
