"""``--hardware`` gate for GUI beam-quality tests.

Delegates to the shared gate so the option/help/skip rule stay in one place,
and so this subtree registers ``--hardware`` even when run on its own::

    pytest tests/ao_shaping/gui/ccd/ --hardware -v
"""

from __future__ import annotations

from tests.ao_shaping.drivers._hardware_gate import (
    add_hardware_option,
    skip_marked_hardware_tests,
)


def pytest_addoption(parser):
    add_hardware_option(parser)


def pytest_collection_modifyitems(config, items):
    skip_marked_hardware_tests(config, items)
