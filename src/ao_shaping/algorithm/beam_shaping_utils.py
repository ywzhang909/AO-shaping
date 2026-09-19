"""Backward-compat shim. Real module: ao_shaping.algorithm.signal_processing.beam_shaping_utils

Moved to the signal_processing/ subpackage; this shim keeps the legacy import path working.
"""  # fmt: skip
from ao_shaping.algorithm.signal_processing.beam_shaping_utils import *  # noqa: F403
import ao_shaping.algorithm.signal_processing.beam_shaping_utils as _m

__all__ = _m.__all__
