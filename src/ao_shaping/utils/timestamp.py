"""Backward-compat shim for :mod:`ao_shaping.utils.timestamp`.

Real module moved to :mod:`ao_shaping.utils.io.timestamp`.
"""

from __future__ import annotations

from ao_shaping.utils.io.timestamp import *  # noqa: F403,F401
from ao_shaping.utils.io.timestamp import (  # noqa: F401
    TimestampParser,
    parse_timestamp,
    sort_by_timestamp,
)