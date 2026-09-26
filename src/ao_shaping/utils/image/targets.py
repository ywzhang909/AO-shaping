"""Backward-compatible shim for the split :mod:`ao_shaping.utils.image.target` package.

The implementation moved into :mod:`ao_shaping.utils.image.target`, split by type
(``patterns`` / ``metrics`` / ``square`` / ``ccd`` / ``objective``). This module
re-exports the former flat public surface - including the private weight helpers
that ``slm_zernike_pib`` and the tests import - so existing
``from ao_shaping.utils.image.targets import ...`` imports keep working unchanged.

New code should import from the :mod:`ao_shaping.utils.image.target` package.

See Also:
    :mod:`ao_shaping.utils.image.target`
"""

from __future__ import annotations

from ao_shaping.utils.image.target import *  # noqa: F401,F403
from ao_shaping.utils.image.target import __all__ as __all__  # noqa: F401
