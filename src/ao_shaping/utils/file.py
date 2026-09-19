"""Backward-compat shim for :mod:`ao_shaping.utils.file`.

Real module moved to :mod:`ao_shaping.utils.io.file`.
"""

from __future__ import annotations

from ao_shaping.utils.io.file import *  # noqa: F403,F401
from ao_shaping.utils.io.file import (  # noqa: F401
    ROOT_DIR,
    DeviceConfigManager,
    Recorder,
    SLMConfigManager,
)