"""Backward-compat shim. Real module: ao_shaping.algorithm.signal_processing.differentiable_shaping

Moved to the signal_processing/ subpackage; this shim keeps the legacy import path working.
"""  # fmt: skip
from ao_shaping.algorithm.signal_processing.differentiable_shaping import *  # noqa: F403,F401
# ``import *`` only re-exports public names; underscore-prefixed names used by
# the legacy import path (test suites) must be re-exported explicitly.
from ao_shaping.algorithm.signal_processing.differentiable_shaping import (  # noqa: F401
    _build_forward,
    _torch,
)
