"""Array-rank conventions shared by the image utilities in this package.

``beam_metrics``, ``target.metrics`` and ``exposure_fusion`` each check that an argument is
a bare image plane before computing on it. The number lives here once on purpose: with a
copy per module a shape guard can drift, and one helper then rejects a 3-D frame while its
neighbour silently accepts it.
"""

from __future__ import annotations

__all__ = ["NDIM_SINGLE_CHANNEL", "NDIM_SPATIAL_STACK"]

#: numpy rank of a bare image plane: ``(height, width)``, no channel axis. It counts axes
#: and nothing else -- it is neither a pair-component count (a ``(height, width)`` tuple)
#: nor a minimum side length, and callers must not borrow it for those questions.
NDIM_SINGLE_CHANNEL = 2

#: numpy rank of a spatial stack, i.e. ``(C, H, W)``. The fusion filters in
#: ``exposure_fusion`` take either that rank or a bare image plane, and the rank test is how
#: the two are told apart -- it counts axes, not channels.
NDIM_SPATIAL_STACK = 3
