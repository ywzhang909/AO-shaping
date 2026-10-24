"""Session-wide ``--hardware`` gate for every test under ``drivers/``.

Before this, the gate lived in ``drivers/hardware/conftest.py`` and
``drivers/slm/conftest.py`` only. ``pytest_collection_modifyitems`` is a
*session* hook, so a ``@pytest.mark.hardware`` test was skipped or executed
depending on which unrelated subtree happened to be in the same invocation:
``pytest drivers/dm drivers/slm`` skipped the Micro-DM report tests (slm's
conftest loaded), while ``pytest drivers/dm`` ran them.

The order-dependence was not cosmetic. ``TestMicroDMReport`` is marked
``hardware`` yet nothing enforced that mark under ``dm/``, so its fixture swept
26 unreachable controllers (192.168.0.101-126) to exhaustion -- 260 s of *setup*
per test, 520 s for the two tests, out of a 560 s domain. Enforcing the mark
where the tests actually live makes the skip deterministic and removes the whole
cost, instead of relying on a sibling subtree being collected.
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
