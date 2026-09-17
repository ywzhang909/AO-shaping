"""Backward-compat shim for :mod:`ao_shaping.utils.handler`.

Real module moved to :mod:`ao_shaping.utils.io.handler`.
"""

from __future__ import annotations

from ao_shaping.utils.io.handler import *  # noqa: F403,F401
from ao_shaping.utils.io.handler import Register  # noqa: F401