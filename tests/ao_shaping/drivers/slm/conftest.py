"""SLM hardware test gating."""

from __future__ import annotations

from tests.ao_shaping.drivers._hardware_gate import (
    add_hardware_option,
    skip_marked_hardware_tests,
)


def pytest_addoption(parser):
    add_hardware_option(parser)


def pytest_collection_modifyitems(config, items):
    skip_marked_hardware_tests(config, items)
