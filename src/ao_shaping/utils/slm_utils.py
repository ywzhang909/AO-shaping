"""Backward-compat alias for :mod:`ao_shaping.utils.slm.phase_display`.

Real module moved to :mod:`ao_shaping.utils.slm.phase_display`.

An alias (not a copy) is REQUIRED: tests reset module-global persistent state
via the legacy name (``slm_utils._last_slm_slot``), so the legacy path must
resolve to the SAME module object.
"""

from __future__ import annotations

import sys

from ao_shaping.utils.slm import phase_display as _real

sys.modules[__name__] = _real
