"""Backward-compat alias for :mod:`ao_shaping.utils.image.hardware_utils`.

Real module moved to :mod:`ao_shaping.utils.image.hardware_utils`.

An alias (not a copy) is REQUIRED: tests reset module-global recording state
via the legacy name (``hardware_utils._frames_dir`` / ``_frame_counter``), so
the legacy path must resolve to the SAME module object.
"""

from __future__ import annotations

import sys

from ao_shaping.utils.image import hardware_utils as _real

sys.modules[__name__] = _real