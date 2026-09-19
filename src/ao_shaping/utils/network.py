"""Backward-compat shim for :mod:`ao_shaping.utils.network`.

Real module moved to :mod:`ao_shaping.utils.io.network`.
"""

from __future__ import annotations

from ao_shaping.utils.io.network import *  # noqa: F403,F401
from ao_shaping.utils.io.network import (  # noqa: F401
    controller_tcp_port,
    ip_last_octet,
    ping_reachable,
    tcp_reachable,
)