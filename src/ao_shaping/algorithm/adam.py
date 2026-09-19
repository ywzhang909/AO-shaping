"""Backward-compat shim. Real module: ao_shaping.algorithm.gradient.adam

Moved to the gradient/ subpackage; this shim keeps the legacy import path working.
"""  # fmt: skip
from ao_shaping.algorithm.gradient.adam import *  # noqa: F403,F401
