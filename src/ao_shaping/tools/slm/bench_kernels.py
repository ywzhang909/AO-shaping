"""Shared measurement kernels for the SLM bench (merged).

Merged from:
- :mod:`ao_shaping.tools.slm.slm_bench_probe` (442 lines, 19 public symbols)
- :mod:`ao_shaping.tools.slm.slm_bench_metrics` (372 lines, 10 public symbols)

**Pure measurement + panel geometry, no device construction and no CLI.** Every
entry point takes already-opened device *instances* (anything with the
``BaseCamera`` / ``Santec`` duck-type), so the same code serves the Daheng bench,
a MiiCam, a simulation camera, or a fake in a unit test. The CLI wrappers live in
:mod:`ao_shaping.tools.slm` next to this module.

Why this exists
---------------
Three bench facts, each of which cost a measurement, are encoded here so no probe
reinvents them wrong.

1. **Never locate a spot with a raw ``argmax`` on a dim frame.** At peak 22-46
   against a frame mean of 0.26, a single hot pixel wins, and the *reference*
   centroid wandered 60 px between repeats -- which produced a confident "nothing
   moved" verdict about a perfectly healthy panel. :func:`measure_spot` despikes
   and then blurs before locating the spot.
2. **The panel keeps whatever pattern was last displayed.** A "flat" reference
   captured before any write is the previous run's speckle, which made the same
   3 ms setting measure 100 counts in one run and 23 in the next. Always write
   flat *first*; :func:`measure_flat_reference` does that.
3. **A fixed ``memory_number=`` is a firmware no-op.** The Santec firmware treats
   ``display_memory`` on the already-displayed slot as a no-op and the LCOS does
   not refresh, so every frame after the first is stale. Use ``display_data()``
   with no ``memory_number``: it rotates the slot itself and estimates the LCOS
   flip time from the gray change.
4. **The LCOS flip-time estimate under-reports; settle on stability, not time.**
   That estimate is driven by how much the gray map changed, and two different
   phases with similar gray statistics make it report **0.0 ms**. Measured: the
   same ramp displayed twice read fwhm 43.2 px on the first grab and 12.8 px
   three seconds later, with the centroid moving 62 px. A single-shot
   acquisition then records an unsettled frame that still looks plausible -- it
   made a ramp sweep non-monotone and the Zernike tilt slope read **1.63 cam
   px/rad instead of 5.36**, a 3.3x error that survived three runs because
   repeats were masking it. :func:`display_and_average` therefore discards
   frames until two consecutive readings agree.
"""
from __future__ import annotations

from ao_shaping.utils.wavefront.matrix_utils import (
    camera_pixel_um_from_focal_scale,
    focal_length_from_camera_pixel,
)

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import zoom

#: Union of both sources' star-export surfaces (`slm_bench_probe` had no `__all__`,
#: `slm_bench_metrics` declared 9). Keeping the union stops `import *` narrowing.
__all__ = [
    "BEAM_CENTER_PANEL",
    "BEAM_RADIUS_PANEL",
    "SLM_PANEL_H",
    "SLM_PANEL_W",
    "SLM_PITCH_M",
    "SpotMeasurement",
    "TILT_SHIFT_SCALE",
    "build_block_pattern",
    "core_fraction",
    "crop_around_zero_order",
    "crop_roi",
    "despike_frame",
    "display_and_average",
    "estimate_shift",
    "exposure_monotonicity",
    "finite_clip",
    "finite_median_subtract",
    "fit_linear_slope",
    "flat_to_flat_floor",
    "measure_flat_reference",
    "measure_spot",
    "ramp_panel",
    "random_phase",
    "roi_l2",
    "settle_time_s",
    "smooth_frame",
    "snr_vs_averages",
    "tilt_shift_px",
    "zernike_panel",

    "camera_pixel_um_from_focal_scale",
    "focal_length_from_camera_pixel",
    "gaussian_grid",
    "phase_to_panel",
]

# Bench constants measured 2026-09-30 (Santec SLM-200 #1 22030108, 1920x1200,
# 10-bit, 2pi = 993 gray at 1064 nm; Daheng MER2-507, 2592x1944, 2.2 um pixel).
# See docs/slm/model_in_loop_bench_calibration.md.
SLM_PANEL_W = 1920
SLM_PANEL_H = 1200
SLM_PITCH_M = 8e-6
BEAM_CENTER_PANEL = (960, 600)
BEAM_RADIUS_PANEL = 450
#: Empirical focal scale: a 2*pi phase ramp over ``P`` panel px moves the spot by
#: ``TILT_SHIFT_SCALE / P`` camera px. The naive ``f*lambda/d_slm`` is ~17.5x
#: larger on this bench, and the panel/camera axes are swapped 90 deg, so neither
#: may be derived from first principles.
#:
#: Measured 2026-09-30 with ``slm_tilt_probe --periods 120,240,480,960,1920``
#: (3 repeats each, agreeing to 0.2 px). Displacements from flat: 62.1, 30.3,
#: 14.7, 6.9, 3.4 px -- cleanly 1/P, halving each time the period doubles.
#: Theory for a 450 px (3.6 mm) illuminated radius is
#: ``f*lambda/(pi*R*camera_pixel)`` = 5.34 cam px per radian of Zernike tilt,
#: which the same data gives as 4.6-5.3.
TILT_SHIFT_SCALE = 7400.0
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


def despike_frame(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Replace isolated hot/cold pixels with a ``k x k`` median.

    A box blur alone is not enough against a *single* hot pixel: a 5000-count
    defect spread over a 5x5 box still reads 200, which beats a dim spot peaking at
    20. Real sensors have defects, and one is enough to send ``argmax`` -- and
    therefore the centroid and every width derived from it -- to the wrong place.
    A median kills isolated outliers outright while leaving a real spot (which is
    spatially correlated) essentially unchanged.
    """
    frame = np.asarray(img, dtype=np.float64)
    k = int(k)
    if k < 3:
        return frame
    pad = k // 2
    p = np.pad(frame, pad, mode="edge")
    stack = np.stack(
        [p[dy : dy + frame.shape[0], dx : dx + frame.shape[1]] for dy in range(k) for dx in range(k)],
        axis=0,
    )
    return np.median(stack, axis=0)


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
    cross = np.divide(
        cross, magnitude, out=np.zeros_like(cross), where=magnitude > 0
    )
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
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import _spot_fwhm

    img = np.asarray(frame, dtype=np.float64)
    sm = smooth_frame(despike_frame(img, 3), 5)
    fwhm = float(_spot_fwhm(sm))
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


def measure_flat_reference(
    cam,
    slm,
    n_frames: int = 4,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
) -> tuple[np.ndarray, SpotMeasurement]:
    """Write flat phase first, then average ``n_frames`` frames.

    Fact 2 in the module docstring: the panel retains the last displayed pattern,
    so a flat reference must be *displayed* before it is read.

    Returns the averaged frame and its spot measurement.
    """
    frame = display_and_average(cam, slm, np.zeros(panel_shape), n_frames=n_frames)
    return frame, measure_spot(frame)


def display_and_average(
    cam,
    slm,
    phase_rad: np.ndarray,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float | None = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
) -> np.ndarray:
    """Display ``phase_rad`` and return an averaged frame once the spot settles.

    Uses ``display_data()`` with no ``memory_number`` (fact 3 in the module
    docstring) so the slot rotates itself.

    Why the explicit settle, and why it is a *stability* test rather than a
    longer fixed wait: the driver's automatic LCOS flip-time estimate is driven
    by how much the gray map changed, and two different phases with similar gray
    statistics make it report **0.0 ms**. On this bench the panel was still
    relaxing more than a second after such a change: the same ramp displayed
    twice gave fwhm 43.2 px on the first read and 12.8 px three seconds later,
    with the centroid moving 62 px. A single-shot acquisition therefore silently
    records an unsettled frame, and the resulting numbers look plausible -- a
    ramp sweep came out non-monotone purely for this reason.

    So: wait ``wait_time_s``, then keep discarding frames until two consecutive
    measurements of (peak, centroid) agree to ``stable_tol`` relative, capped at
    ``max_wait_s``.
    """
    import time

    gray = slm.create_phase_from_array(phase_rad)
    slm.display_data(gray, wait_time_s)
    for _ in range(int(n_discard)):
        cam.get_numpy_image(n_sample=1)

    def probe() -> tuple[float, float, float]:
        img = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        m = measure_spot(img)
        return m.peak, m.centroid_x, m.centroid_y

    prev = probe()
    deadline = time.time() + float(max_wait_s)
    while time.time() < deadline:
        cur = probe()
        if (
            abs(cur[0] - prev[0]) <= stable_tol * max(prev[0], 1.0)
            and abs(cur[1] - prev[1]) <= stable_tol * 40.0
            and abs(cur[2] - prev[2]) <= stable_tol * 40.0
        ):
            break
        prev = cur

    acc = [
        np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        for _ in range(int(n_frames))
    ]
    return np.mean(acc, axis=0)


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


def zernike_panel(
    coefficients: dict[tuple[int, int], float],
    radius: int,
    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
) -> np.ndarray:
    """Zernike phase over the illuminated disc, in panel coordinates.

    Centred on the measured beam rather than on the panel, and clipped to the
    panel bounds. The centre must be a *panel* coordinate: the camera's 0-order
    position is a different frame entirely (the axes are swapped 90 deg here), and
    deriving one from the other writes the phase where the beam is not.

    Args:
        coefficients: ``{(n, m): amplitude_in_radians}``, unit-RMS Noll
            normalisation, so an amplitude is the RMS phase in radians.
        radius: Aperture radius in panel pixels.
        pupil_center: Beam centre in panel pixels (x, y).
        panel_shape: ``(height, width)`` of the panel.

    Returns:
        A ``panel_shape`` phase in radians, zero outside the disc.
    """
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    height, width = int(panel_shape[0]), int(panel_shape[1])
    out = np.zeros((height, width), dtype=np.float64)
    if not coefficients:
        return out
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    size = 2 * int(radius) + 1
    n_max = max(n for n, _ in coefficients)
    gen = ZernikeGenerator((size, size), radius=int(radius), n_orders=n_max)
    mode = np.asarray(gen.generate_polynomial(coefficients), dtype=np.float64)
    if mode.shape != (size, size):  # the generator may return (W, H)
        mode = mode.T
    x0, y0 = max(cx - int(radius), 0), max(cy - int(radius), 0)
    x1 = min(cx + int(radius) + 1, width)
    y1 = min(cy + int(radius) + 1, height)
    sub = mode[
        y0 - (cy - int(radius)) : y1 - (cy - int(radius)),
        x0 - (cx - int(radius)) : x1 - (cx - int(radius)),
    ]
    finite = np.isfinite(sub)
    patch = np.zeros(sub.shape, dtype=np.float64)
    patch[finite] = sub[finite]
    out[y0:y1, x0:x1] = patch
    return out


def ramp_panel(
    period: int,
    axis: int = 1,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
) -> np.ndarray:
    """Linear phase advancing 2*pi every ``period`` px along ``axis`` (0=y, 1=x).

    A tilt is the cleanest modulation probe because its focal-plane shift does
    not depend on diffraction efficiency. The period must be *large*: the shift
    is ``TILT_SHIFT_SCALE / period`` camera px, so period = 1 would throw the spot
    7.6 mm, far outside the 5.7 mm frame.
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    n = height if axis == 0 else width
    t = (np.arange(n, dtype=np.float64) % int(period)) / float(period)
    phase = 2.0 * np.pi * t
    shape = (height, 1) if axis == 0 else (1, width)
    return np.broadcast_to(phase.reshape(shape), (height, width)).copy()


def tilt_shift_px(period: int, scale: float = TILT_SHIFT_SCALE) -> float:
    """Expected focal shift, in camera px, for a 2*pi ramp over ``period`` px."""
    return float(scale) / float(period)


def fit_linear_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)``; ``(nan, nan)`` if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or not np.isfinite(y).all():
        return float("nan"), float("nan")
    var = float(np.sum((x - x.mean()) ** 2))
    if var <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / var)
    return slope, float(y.mean() - slope * x.mean())


def random_phase(
    shape: tuple[int, int], seed: int = 0
) -> np.ndarray:
    """Uniform random phase in [0, 2*pi) -- scatters the pupil for speckle.

    Only useful where the panel can resolve pixel-scale phase. On this bench it
    measurably does not (see :mod:`ao_shaping.tools.slm.slm_phase_resolution`),
    so a random phase leaves the far field a single tight focus.
    """
    return np.random.default_rng(int(seed)).uniform(
        0.0, 2.0 * np.pi, (int(shape[0]), int(shape[1]))
    )
# ---------------------------------------------------------------------------
# Frame preparation — the two variants are deliberately different
# ---------------------------------------------------------------------------


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

    Mirrors :func:`~ao_shaping.optimizer.wfless.slm_gs_refine._prepare_frame`.
    Required before any ratio whose denominator is a whole-frame statistic:
    symmetric read noise leaves ~half of a raw frame negative, which pushes such
    a ratio above 1 and makes an optimizer chase noise.

    The order matters. Clipping first would rectify the noise distribution and
    invent a DC pedestal proportional to the pixel count.

    Args:
        frame: 2D camera frame.

    Returns:
        New float64 array, non-negative, same shape.
    """
    arr = _as_2d_float(frame)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(clean - float(np.median(clean)), 0.0, None)


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------


def gaussian_grid(region: int, waist_grid: float) -> np.ndarray:
    """Gaussian illumination on the model grid, masked to the inscribed circle.

    Args:
        region: Model grid edge length in pixels.
        waist_grid: Gaussian waist in **model pixels**.

    Returns:
        A ``(region, region)`` amplitude array, zero outside the inscribed
        circle of radius ``region / 2``.
    """
    yy, xx = np.mgrid[0:region, 0:region]
    r2 = (xx - region / 2.0) ** 2 + (yy - region / 2.0) ** 2
    amp = np.exp(-r2 / (2.0 * max(float(waist_grid), 1e-6) ** 2))
    amp[r2 > (region / 2.0) ** 2] = 0.0
    return amp


def phase_to_panel(
    phase_model: np.ndarray, disc_radius: int, pupil_center: tuple[int, int]
) -> np.ndarray:
    """Resize a model-grid phase onto the panel disc, centred on the beam.

    Args:
        phase_model: ``(region, region)`` phase, raw unwrapped radians.
        disc_radius: Half-width of the target window, **panel** pixels.
        pupil_center: Beam centre in **panel** pixels as ``(x, y)``.

    Returns:
        A ``(SLM_PANEL_H, SLM_PANEL_W)`` float64 array holding the phase inside
        the disc and zero elsewhere -- the shape the Santec driver expects.

    Raises:
        ValueError: If ``phase_model`` is not square.

    Note:
        ``pupil_center`` must be measured **on the panel**. The camera's 0-order
        is a different coordinate frame entirely (on this bench the two axes are
        swapped and the scales differ by more than 10x), so deriving one from the
        other silently writes the phase where the beam is not.
    """
    phase_model = np.asarray(phase_model, dtype=np.float64)
    if phase_model.ndim != 2 or phase_model.shape[0] != phase_model.shape[1]:
        raise ValueError(f"phase_model must be square, got {phase_model.shape}")
    r = int(disc_radius)
    region = phase_model.shape[0]
    sub = np.asarray(
        zoom(phase_model, (2 * r / region, 2 * r / region), order=1), dtype=np.float64
    )
    panel = np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64)
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    # Clip the window so an off-centre or oversized disc stays in bounds.
    x0, x1 = max(cx - r, 0), min(cx + r, SLM_PANEL_W)
    y0, y1 = max(cy - r, 0), min(cy + r, SLM_PANEL_H)
    sub = sub[y0 - (cy - r) : y1 - (cy - r), x0 - (cx - r) : x1 - (cx - r)]
    panel[y0:y1, x0:x1] = sub
    return panel


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


def crop_roi(
    frame: np.ndarray, center: tuple[int, int], half: int
) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Settle / drift characterisation
# ---------------------------------------------------------------------------


def settle_time_s(
    deltas: Sequence[float],
    times_s: Sequence[float],
    *,
    frac: float = 0.10,
    run: int = 3,
) -> float | None:
    """First time the settle curve stays within ``frac`` of its final value.

    Replaces "sleep a fixed duration and hope". A fixed wait is invalid on this
    bench because the driver's flip-time estimate under-reports (it reports
    0.0 ms for two phases with similar grey statistics), so the same ramp read
    43.2 px FWHM immediately and 12.8 px three seconds later.

    Args:
        deltas: Observable (e.g. a norm) at each sample.
        times_s: Sample times, seconds.
        frac: Tolerance as a fraction of the final value.
        run: Number of consecutive samples that must all be within tolerance.

    Returns:
        The settle time, or ``None`` if it never settles.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    d = np.asarray(list(deltas), dtype=np.float64)
    t = np.asarray(list(times_s), dtype=np.float64)
    if d.shape != t.shape:
        raise ValueError("deltas and times_s must have the same length")
    if d.size == 0:
        return None
    tol = abs(float(frac)) * abs(float(d[-1]))
    if tol <= 0.0:
        # A zero-valued curve is trivially settled from the first sample.
        return float(t[0]) if abs(float(d[0])) <= 0.0 else None
    for i in range(d.size):
        window = d[i : i + int(run)]
        if window.size < int(run):
            break
        if bool(np.all(np.abs(window - d[-1]) <= tol)):
            return float(t[i])
    return None


def snr_vs_averages(
    signal_norms: Sequence[float],
    floor: float,
    ks: Sequence[int],
) -> dict[int, float]:
    """SNR at K-frame averages against a measured noise floor.

    Useful as a *diagnostic*: if averaging K frames does not improve the SNR,
    the residual is not independent read noise but drift, and averaging longer
    will not help. That distinction decides whether a bench needs a better
    exposure or a better settle protocol.

    Args:
        signal_norms: Observable for each of K identical repeats.
        floor: Single-frame noise floor (same observable, same units).
        ks: Averaging factors to report.

    Returns:
        ``{K: snr}``.

    Raises:
        ValueError: If ``floor`` is not positive.
    """
    f = float(floor)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(f"floor must be finite and positive, got {floor!r}")
    x = np.asarray(list(signal_norms), dtype=np.float64)
    out: dict[int, float] = {}
    for k in ks:
        kk = int(k)
        if kk < 1:
            raise ValueError(f"K must be >= 1, got {k!r}")
        mean = float(x[:kk].mean()) if x.size >= kk else float(x.mean())
        out[kk] = mean / (f / np.sqrt(kk))
    return out


def exposure_monotonicity(
    exposures_ms: Sequence[float],
    peaks: Sequence[float],
    *,
    rel_tol: float = 0.02,
    saturation_level: float | None = None,
) -> dict[str, object]:
    """Check that peak brightness rises monotonically with exposure.

    Monotonicity is the cheap way to prove the camera is not being pushed past
    its linear range and that the laser is not drifting across the bracket.

    Args:
        exposures_ms: Exposure settings, ascending.
        peaks: Measured peak per exposure.
        rel_tol: Allowed relative shortfall on each step.
        saturation_level: Detector full-scale value; flags saturation when a
            peak reaches it.

    Returns:
        Dict with ``verdict``, ``ratios``, ``expected`` and ``saturated``.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    e = np.asarray(list(exposures_ms), dtype=np.float64)
    p = np.asarray(list(peaks), dtype=np.float64)
    if e.shape != p.shape:
        raise ValueError("exposures_ms and peaks must have the same length")
    if e.size < 2:
        return {
            "verdict": "insufficient_data",
            "ratios": [],
            "expected": [],
            "saturated": False,
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = e[1:] / e[:-1]
        ratios = p[1:] / p[:-1]
    ok = ratios >= (expected * (1.0 - float(rel_tol)))
    saturated = False
    if saturation_level is not None:
        saturated = bool(np.any(p >= float(saturation_level)))
    return {
        "verdict": "monotonic" if bool(np.all(ok)) and not saturated else
                   "saturated" if saturated else "non_monotonic",
        "ratios": [float(v) for v in ratios],
        "expected": [float(v) for v in expected],
        "saturated": saturated,
    }


# ---------------------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------------------


def build_block_pattern(
    coeffs: np.ndarray,
    grid: int,
    panel_shape: tuple[int, int],
) -> np.ndarray:
    """Tile a ``grid x grid`` coefficient map onto the panel as blocks.

    Matches the freeform SPGD basis used by
    ``slm_square_shaping._freeform_phase_radians``: coefficients are
    block-replicated, not applied per pixel, so one DOF covers an 80x50 px SLM
    region. A 24x24 grid on a 1200x1920 panel therefore yields 5x8 blocks.

    Args:
        coeffs: ``grid*grid`` coefficients, row-major.
        grid: Grid edge length.
        panel_shape: ``(height, width)`` of the SLM panel.

    Returns:
        ``panel_shape`` float64 phase array (raw radians, unwrapped).

    Raises:
        ValueError: If ``grid`` is not positive or ``coeffs`` is the wrong size.
    """
    g = int(grid)
    if g <= 0:
        raise ValueError(f"grid must be positive, got {grid!r}")
    c = np.asarray(coeffs, dtype=np.float64).ravel()
    if c.size != g * g:
        raise ValueError(
            f"coeffs must hold grid*grid = {g * g} values, got {c.size}"
        )
    ph, pw = int(panel_shape[0]), int(panel_shape[1])
    blocks = c.reshape(g, g)
    bh, bw = ph // g, pw // g
    if bh < 1 or bw < 1:
        raise ValueError(
            f"grid {g} is too coarse for panel {panel_shape!r}"
        )
    tiled = np.kron(blocks, np.ones((bh, bw), dtype=np.float64))
    # np.kron stops at the first g*bh rows / g*bw cols; pad if the panel is
    # not an exact multiple of the grid.
    if tiled.shape != (ph, pw):
        padded = np.zeros((ph, pw), dtype=np.float64)
        padded[: tiled.shape[0], : tiled.shape[1]] = tiled
        return padded
    return tiled


# --- settled single-phase acquisition ------------------------------
# Moved here from slm_zernike_sweep_probe.py (issue #62 / R-46): it is a pure
# settle+averaging kernel over duck-typed cam/slm, not probe policy. The probe
# re-exports it, so the public name is unchanged for its existing callers.
def capture_settled(
    cam: Any,
    slm: Any,
    phase_rad: np.ndarray,
    *,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
) -> np.ndarray:
    """Display one phase and return the averaged far-field frame once settled.

    The single-phase entry point, for callers that build their own acquisition
    order (an ABBA interleave, a repeated push-pull) and only need the settle
    discipline and the averaging from here.

    Args:
        cam: An open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: An open SLM exposing ``create_phase_from_array`` and
            ``display_data(gray, wait_time_s)``.
        phase_rad: Raw unwrapped radians, panel-shaped.
        n_frames: Frames averaged into the returned measurement.
        n_discard: Frames discarded before the stability loop starts.
        wait_time_s: Explicit LCOS settle before measuring.
        stable_tol: Relative agreement required between consecutive readings.
        max_wait_s: Cap on the wait for stability.

    Returns:
        The averaged frame.
    """
    return display_and_average(
        cam, slm, phase_rad, n_frames=n_frames, n_discard=n_discard,
        wait_time_s=wait_time_s, stable_tol=stable_tol, max_wait_s=max_wait_s,
    )
