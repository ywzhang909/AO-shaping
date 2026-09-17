"""Backward-compat shim for :mod:`ao_shaping.utils.cli_helpers`.

Real module moved to :mod:`ao_shaping.utils.io.cli_helpers`.
"""

from __future__ import annotations

from ao_shaping.utils.io.cli_helpers import *  # noqa: F403,F401
from ao_shaping.utils.io.cli_helpers import (  # noqa: F401
    create_save_dir,
    get_date_dir_name,
    get_debug_mode,
    get_timestamp_str,
    parse_tuple,
    setup_coredumpy,
)