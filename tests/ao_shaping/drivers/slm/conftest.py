"""SLM hardware test gating."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--hardware",
            action="store_true",
            default=False,
            help="Run hardware integration tests (requires connected devices)",
        )
    except ValueError:
        pass  # Already registered by tests/ao_shaping/drivers/hardware/conftest.py


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hardware"):
        return
    skip_hardware = pytest.mark.skip(reason="need --hardware option to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)
