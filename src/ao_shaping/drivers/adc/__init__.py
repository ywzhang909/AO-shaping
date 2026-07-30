"""ADC (Analog-to-Digital Converter) driver package.

Provides NI DAQ-based ADC voltage acquisition with simulated fallback.
"""

from __future__ import annotations

from loguru import logger

from ao_shaping.drivers.adc.driver import NidaqADC
from ao_shaping.drivers.adc.driver import NidaqADCError, NidaqADCNotFoundError

__all__ = [
    "NidaqADC",
    "NidaqADCError",
    "NidaqADCNotFoundError",
]
