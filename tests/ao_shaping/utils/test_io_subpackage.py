"""Subpackage anchor tests for :mod:`ao_shaping.utils.io`.

Locks the refactor contract: every module moved into the ``io`` subpackage must
stay importable from BOTH its canonical subpackage path and its legacy top-level
path, and the objects returned at each path must be identical (same class /
function objects, not copies).

No side effects; import-only assertions.
"""

from __future__ import annotations

import importlib

import pytest

from ao_shaping.utils import Recorder, configure_error_logging, get_init_V_by_rms

# module -> representative public symbol to compare across paths
IO_SYMBOLS = {
    "file": "Recorder",
    "timestamp": "TimestampParser",
    "cli_helpers": "parse_tuple",
    "device_config": "ConfigHandler",
    "network": "ping_reachable",
    "handler": "Register",
}


@pytest.mark.parametrize("mod_name,symbol", IO_SYMBOLS.items())
def test_canonical_and_legacy_paths_identical(mod_name: str, symbol: str) -> None:
    canonical = importlib.import_module(f"ao_shaping.utils.io.{mod_name}")
    legacy = importlib.import_module(f"ao_shaping.utils.{mod_name}")
    assert getattr(canonical, symbol) is getattr(legacy, symbol)


def test_facade_still_exposes_io_names() -> None:
    assert Recorder is importlib.import_module("ao_shaping.utils.io.file").Recorder
    assert (
        get_init_V_by_rms
        is importlib.import_module("ao_shaping.utils.io.file").get_init_V_by_rms
    )
    assert callable(configure_error_logging)