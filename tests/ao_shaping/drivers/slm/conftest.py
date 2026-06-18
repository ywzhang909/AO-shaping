"""SLM hardware test configuration.

Adds --hardware flag gating for SLM hardware integration tests.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run hardware integration tests (requires connected devices)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hardware"):
        return
    skip_hardware = pytest.mark.skip(reason="need --hardware option to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)
