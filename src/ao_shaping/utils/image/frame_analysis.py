"""Camera-frame analysis primitives (leaf ``utils/image`` layer).

Sunk out of :mod:`ao_shaping.tools.kernels.bench_kernels` (R-48, #64) so the
frame maths lives in the leaf layer that ``tools/``, ``optimizer/`` and ``runners/``
may all import from, instead of the other way round. ``bench_kernels`` re-exports
every public name here, so existing callers are unaffected.

This module is **pure numpy/scipy**: it takes already-materialised 2D frames and
returns arrays or scalars. It constructs no devices and knows nothing about the
SLM/DM/WFS hardware layers. ``despike_frame`` and ``prepare_frame`` are imported
from :mod:`~ao_shaping.utils.image.beam_metrics` (the single source for those two);
``spot_fwhm`` was previously the private ``model_in_loop_shaping._spot_fwhm``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ao_shaping.utils.image.beam_metrics import despike_frame, prepare_frame


@dataclass
class SpotMeasurement:
    """One frame's spot characterisation, in camera pixels."""

    peak: float
    fwhm_px: float
    centroid_x: float
    centroid_y: float
    hollowness: float

    def as_row(self) -> str:
        """One-line summary for a log line."""
        return (
            f"peak={self.peak:.0f} fwhm={self.fwhm_px:.1f}px "
            f"hollow={self.hollowness:.2f} c=({self.centroid_x:.1f},{self.centroid_y:.1f})"
        )


def _as_2d_float(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {arr.shape}")
    return arr


def finite_clip(frame: np.ndarray) -> np.ndarray:
    """Mask non-finite pixels to 0 and clip negatives, keeping the pedestal.

    Mirrors the preparation used by
    :func:`~ao_shaping.optimizer.wfless.slm_square_shaping.square_peak_to_background_ratio`.
    Correct for peak/background-style ratios, where subtracting the median would
    destroy the very pedestal that gives the denominator meaning.

    Args:
        frame: 2D camera frame.

    Returns:
        New float64 array, non-negative, same shape.
    """
    arr = _as_2d_float(frame)
    return np.clip(np.where(np.isfinite(arr), arr, 0.0), 0.0, None)


def finite_median_subtract(frame: np.ndarray) -> np.ndarray:
    """Mask non-finite pixels, subtract the median, then clip at 0.

    Float64-coercing wrapper around the canonical
    :func:`~ao_shaping.utils.image.beam_metrics.prepare_frame`.
    Required before any ratio whose denominator is a whole-frame
    statistic: symmetric read noise leaves ~half of a raw frame
    negative, which pushes such a ratio above 1 and makes an
    optimizer chase noise.

    The order matters. Clipping first would rectify the noise
    distribution and invent a DC pedestal proportional to the
    pixel count.

    Args:
        frame: 2D camera frame.

    Returns:
        New float64 array, non-negative, same shape.
    """
    return prepare_frame(_as_2d_float(frame))


def smooth_frame(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Box blur ``img`` with a ``k x k`` kernel.

    Fact 1 in the module docstring: without this, ``argmax`` on a dim speckle
    frame locks onto single-pixel noise.
    """
    frame = np.asarray(img, dtype=np.float64)
    k = int(k)
    if k < 2:
        return frame
    pad = k // 2
    p = np.pad(frame, pad, mode="edge")
    out = np.zeros_like(frame)
    for dy in range(k):
        for dx in range(k):
            out += p[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
    return out / float(k * k)


def spot_fwhm(image: np.ndarray) -> float:
    """Spot full-width-at-half-maximum in pixels, averaged over two axes.

    FWHM rather than a second moment on purpose. A speckle pattern has heavy
    tails, and the second moment weights them by ``r**2``, so it is dominated by
    a few far-out pixels: on a 2048-px model grid it read ~70x the value the
    same spot has on a 160-px camera crop, which made the two sides
    incomparable. FWHM is local and bounded, so it measures the same width on
    any grid that contains the half-maximum region.

    Args:
        image: 2D intensity frame.

    Returns:
        The mean horizontal/vertical FWHM in pixels, or ``nan`` for a
        degenerate frame.
    """
    frame = np.asarray(image, dtype=np.float64)
    peak_index = np.unravel_index(int(np.argmax(frame)), frame.shape)
    peak = float(frame[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan")
    cy, cx = int(peak_index[0]), int(peak_index[1])
    widths = []
    for line in (frame[cy, :], frame[:, cx]):
        half = peak / 2.0
        idx = np.flatnonzero(line >= half)
        if idx.size < 2:
            continue
        lo, hi = int(idx[0]), int(idx[-1])
        # Linear interpolation at the two half-maximum crossings.
        left = lo
        if lo > 0:
            y0, y1 = line[lo - 1], line[lo]
            left = lo - 1 + (half - y0) / (y1 - y0) if y1 != y0 else float(lo)
        right = hi
        if hi < line.size - 1:
            y0, y1 = line[hi], line[hi + 1]
            right = hi + (y0 - half) / (y0 - y1) if y0 != y1 else float(hi)
        widths.append(abs(right - left))
    if not widths:
        return float("nan")
    return float(np.mean(widths))


def estimate_shift(
    reference: np.ndarray, moving: np.ndarray, max_shift: int | None = None
) -> tuple[float, float, float]:
    """Sub-pixel pattern shift of ``moving`` relative to ``reference``.

    Returns ``(dy, dx, peak)``: the shift in pixels and the height of the
    correlation peak (1.0 = a perfect match).

    Why not the centroid
    --------------------
    A tilt large enough to give a good signal also deforms the spot: measured on
    this bench at +/-1.5 rad the single lobe split, ``hollowness`` fell to 0.65,
    and individual centroid readings jumped +/-8 px between repeats of the *same*
    phase. The ABBA mean only partly cancelled that, and the fitted tilt slope
    came out 1.72 cam px/rad against 3.3-3.6 from the smaller, still-single-lobed
    range -- a 2x error in the one number the whole calibration depends on.

    Phase correlation sidesteps the shape change entirely: it cross-correlates
    the two patterns, so a spot that widens or splits still peaks at the same
    displacement. It is also sub-pixel, which the centroid is not.
    """
    a = smooth_frame(despike_frame(np.asarray(reference, np.float64), 3), 3)
    b = smooth_frame(despike_frame(np.asarray(moving, np.float64), 3), 3)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fb * np.conj(fa)
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 0)
    corr = np.fft.ifft2(cross).real
    peak_idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak_val = float(corr[peak_idx])
    # Wrap so the shift is reported relative to the correlation origin.
    dy = int(peak_idx[0])
    dx = int(peak_idx[1])
    if dy > corr.shape[0] // 2:
        dy -= corr.shape[0]
    if dx > corr.shape[1] // 2:
        dx -= corr.shape[1]
    if max_shift is not None and (abs(dy) > max_shift or abs(dx) > max_shift):
        return float("nan"), float("nan"), peak_val

    # Parabolic sub-pixel refinement on the correlation peak:
    # offset = 0.5 * (y[-1] - y[+1]) / (y[-1] - 2*y[0] + y[+1]),
    # taken along each axis with the other index held at the peak.
    def _refine(corr: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
        def along(axis: int) -> float:
            n = corr.shape[axis]
            base = iy if axis == 0 else ix

            def at(k: int) -> float:
                r, c = (k % n, ix) if axis == 0 else (iy, k % n)
                return float(corr[r, c])

            y_m, y_0, y_p = at(base - 1), at(base), at(base + 1)
            denom = y_m - 2.0 * y_0 + y_p
            if denom == 0.0:
                return 0.0
            return float(np.clip(0.5 * (y_m - y_p) / denom, -1.0, 1.0))

        return along(0), along(1)

    fy, fx = _refine(corr, dy, dx)
    return float(dy + fy), float(dx + fx), peak_val


def measure_spot(frame: np.ndarray, box: int = 40) -> SpotMeasurement:
    """Peak, FWHM, centroid and hollowness of the spot in one CCD frame.

    The frame is despiked (median) and then blurred (box) before the spot is
    located -- see :func:`despike_frame` and :func:`smooth_frame` for why both are
    needed. ``hollowness`` is the intensity at the centroid divided by the peak:
    ~1 for a single lobe, well below 1 once the focal plane breaks into rings.
    Width cannot make that distinction -- a -4 rad ring measured *narrower* than
    the -2.5 rad lobe on this bench -- so any width threshold picks the wrong
    points.
    """
    img = np.asarray(frame, dtype=np.float64)
    sm = smooth_frame(despike_frame(img, 3), 5)
    fwhm = float(spot_fwhm(sm))
    peak = float(img.max())
    cy, cx = (int(v) for v in np.unravel_index(int(sm.argmax()), sm.shape))
    r = int(box)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, sm.shape[0])
    x0, x1 = max(cx - r, 0), min(cx + r + 1, sm.shape[1])
    patch = np.maximum(sm[y0:y1, x0:x1] - sm[y0:y1, x0:x1].min(), 0.0)
    tot = float(patch.sum())
    if tot <= 0:
        return SpotMeasurement(peak, fwhm, float(cx), float(cy), 0.0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mx = float((patch * xx).sum() / tot)
    my = float((patch * yy).sum() / tot)
    iy = min(max(int(round(my)), 0), sm.shape[0] - 1)
    ix = min(max(int(round(mx)), 0), sm.shape[1] - 1)
    hollow = float(sm[iy, ix] / peak) if peak > 0 else 0.0
    return SpotMeasurement(peak, fwhm, mx, my, hollow)


def core_fraction(
    frame: np.ndarray, cx: float, cy: float, radius: float = 40.0
) -> float:
    """Fraction of frame energy within ``radius`` px of ``(cx, cy)``.

    Monotone in how much of the pupil is scattered: under a flat phase the beam
    focuses there, so this is near its maximum. Note that *total* frame energy
    carries no size information at all (it is conserved -- measured 0.999-1.005
    across aperture radii from 100 to 850 px), and the energy in a box around the
    0-order is not monotone either (a disc both scatters light out of the box and
    redistributes light into it).
    """
    img = np.asarray(frame, dtype=np.float64)
    tot = float(img.sum())
    if tot <= 0:
        return 0.0
    iy, ix = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    m = (iy - cy) ** 2 + (ix - cx) ** 2 <= float(radius) ** 2
    return float(img[m].sum() / tot)


def crop_around_zero_order(frame: np.ndarray, size: int = 512) -> np.ndarray:
    """Crop a fixed-size window centred on the frame's global argmax.

    The optical axis is the frame's brightest point, never the geometric centre:
    on this bench the 0-order sits ~620 px off-centre in x. A geometry solve
    compares the model's *central* far-field window against the stored frame, so
    an uncropped frame would be compared against a misaligned window and the
    speckle correlation would collapse.

    Args:
        frame: 2D far-field frame.
        size: Window side in pixels, clamped to the frame and zero-padded up.

    Returns:
        The cropped ``(size, size)`` window.
    """
    data = np.asarray(frame, dtype=np.float64)
    side = int(min(max(size, 1), data.shape[0], data.shape[1]))
    cy, cx = np.unravel_index(int(np.argmax(data)), data.shape)
    y0, x0 = int(cy) - side // 2, int(cx) - side // 2
    out = np.zeros((side, side), dtype=np.float64)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + side, data.shape[0]), min(x0 + side, data.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = data[sy0:sy1, sx0:sx1]
    return out


def crop_roi(frame: np.ndarray, center: tuple[int, int], half: int) -> np.ndarray:
    """Crop a ``2*half`` square about ``center``, zero-padding off-frame.

    The spot is frequently not at the frame centre on this bench (measured
    0-order near (674, 1026) on a 1944x2592 frame), so ROI maths must always be
    driven by a measured centre rather than ``shape // 2``.

    Args:
        frame: 2D camera frame.
        center: ROI centre as ``(x, y)`` in pixels.
        half: Half-width; the result is ``2*half`` wide.

    Returns:
        ``(2*half, 2*half)`` float array, zero where the request fell outside.
    """
    arr = _as_2d_float(frame)
    half = int(half)
    if half <= 0:
        raise ValueError(f"half must be positive, got {half!r}")
    h, w = arr.shape
    cx, cy = int(center[0]), int(center[1])
    out = np.zeros((2 * half, 2 * half), dtype=np.float64)
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def roi_l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two same-shape frames or crops.

    Used as the drift observable: on a stable bench two consecutive flat reads
    differ only by noise, whereas an unsettled panel or a drifting laser gives
    a much larger value. Peak intensity is *not* a reliable observable here
    (it is not reproducible run to run); a region sum or norm is.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return float(np.linalg.norm(x - y))


def flat_to_flat_floor(
    frames: Sequence[np.ndarray],
) -> tuple[float, list[float]]:
    """Drift floor from consecutive flat reads.

    Args:
        frames: At least two consecutive flat-field frames.

    Returns:
        ``(median, series)`` where ``series`` holds every consecutive
        ``roi_l2`` difference.

    Raises:
        ValueError: If fewer than two frames are given.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames to estimate a drift floor")
    series = [roi_l2(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    return float(np.median(series)), series
