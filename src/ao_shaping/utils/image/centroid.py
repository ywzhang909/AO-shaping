"""Windowed intensity-weighted centroid -- the single spot-location primitive.

Issue #199 R6: three copies of "coarse-locate the global peak, then take an
intensity-weighted centroid inside a local window" lived in
:mod:`ao_shaping.utils.image.frame_analysis` (:func:`~ao_shaping.utils.image.
frame_analysis.measure_spot`, on a despiked + box-blurred frame) and
:mod:`ao_shaping.utils.image.displacement` (``_centroid_in_window``, on a raw
difference frame). Both now delegate here.

Semantics (identical in both former call sites):

* the window is centred on the frame's **global argmax** (never the geometric
  centre -- on this bench the 0-order sits hundreds of pixels off-centre);
* the window's own minimum is subtracted as the local background, so a
  pedestal cannot drag the centroid;
* the centroid is the intensity-weighted mean of the background-subtracted
  window, returned as ``(x, y)`` floats in pixel coordinates;
* a window with no positive dynamic range falls back to the peak pixel.

This is deliberately **not** a second-moment or FWHM estimator, and it is not
:func:`~ao_shaping.utils.image.frame_analysis.estimate_shift` (phase
correlation) -- those answer different questions; see that function's docstring
for why a centroid is the wrong tool for sub-pixel pattern shift.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def windowed_centroid(
    image: npt.NDArray[np.floating], half: int
) -> tuple[float, float]:
    """Intensity-weighted centroid ``(x, y)`` in a window about the global peak.

    Args:
        image: 2-D intensity-like frame (any float dtype; non-finite values are
            the caller's responsibility -- both former call sites feed
            already-prepared frames).
        half: Window half-width in pixels; the window spans
            ``2 * half + 1`` pixels per axis, clipped to the frame.

    Returns:
        ``(x, y)`` centroid in pixel coordinates. Falls back to the peak pixel
        when the window carries no positive dynamic range.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must be non-empty")

    row, col = np.unravel_index(int(np.argmax(arr)), arr.shape)
    half = int(half)
    y0, y1 = max(row - half, 0), min(row + half + 1, arr.shape[0])
    x0, x1 = max(col - half, 0), min(col + half + 1, arr.shape[1])
    block = arr[y0:y1, x0:x1]
    weighted = np.maximum(block - float(block.min()), 0.0)
    total = float(weighted.sum())
    if total <= 0.0:
        return float(col), float(row)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((weighted * xx).sum() / total)
    cy = float((weighted * yy).sum() / total)
    return cx, cy


__all__ = ["windowed_centroid"]
