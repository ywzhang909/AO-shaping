"""Shared ``--hardware`` gate for every conftest under ``drivers/``.

Two conftest files used to register the same option independently:
``drivers/hardware/conftest.py`` did it unconditionally and
``drivers/slm/conftest.py`` wrapped it in ``try/except ValueError``. That made the
run depend on **argument order**: pytest registers conftests as it walks the paths
it was given, so ``pytest drivers/slm drivers/hardware`` hit the unguarded copy
second and died at startup with

    ValueError: option names {'--hardware'} already added

while the alphabetical walk from ``tests`` (``hardware`` before ``slm``) happened to
survive. Narrowing the default exclusion from the whole ``drivers`` tree to
``drivers/dm`` put both subtrees in one session, which is what exposed it.

Registering through this helper makes the outcome order-independent: whichever
conftest loads first wins, and the option, its help text and the skip rule exist in
exactly one place.
"""

from __future__ import annotations

from typing import Any

import pytest

HARDWARE_OPTION = "--hardware"
HARDWARE_HELP = "Run hardware integration tests (requires connected devices)"
SKIP_REASON = "need --hardware option to run"


def add_hardware_option(parser: pytest.Parser) -> None:
    """Register ``--hardware`` unless a sibling conftest already did."""
    try:
        parser.addoption(
            HARDWARE_OPTION,
            action="store_true",
            default=False,
            help=HARDWARE_HELP,
        )
    except ValueError:
        pass


def skip_marked_hardware_tests(config: Any, items: list[Any]) -> None:
    """Skip ``@pytest.mark.hardware`` items unless ``--hardware`` was requested."""
    if config.getoption(HARDWARE_OPTION):
        return
    marker = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(marker)
