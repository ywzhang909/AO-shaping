"""Backward-compat shim for :mod:`ao_shaping.utils.device_config`.

Real module moved to :mod:`ao_shaping.utils.io.device_config`.
"""

from __future__ import annotations

from ao_shaping.utils.io.device_config import *  # noqa: F403,F401
from ao_shaping.utils.io.device_config import (  # noqa: F401
    ConfigHandler,
    DeviceParam,
    param,
)