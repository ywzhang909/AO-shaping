"""Shared measurement kernels for the SLM bench (merged).

Merged from:
- :mod:`slm_bench_probe` (442 lines, 19 public symbols)
- :mod:`slm_bench_metrics` (372 lines, 10 public symbols)

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

from ao_shaping.utils.image.beam_metrics import despike_frame
from ao_shaping.utils.image.frame_analysis import (
    SpotMeasurement,
    core_fraction,
    crop_around_zero_order,
    crop_roi,
    estimate_shift,
    finite_clip,
    finite_median_subtract,
    flat_to_flat_floor,
    measure_spot,
    roi_l2,
    smooth_frame,
)
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE
from ao_shaping.utils.wavefront.matrix_utils import (
    camera_pixel_um_from_focal_scale,
    focal_length_from_camera_pixel,
)

from collections.abc import Sequence
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
# ``TILT_SHIFT_SCALE`` is defined once in ``ao_shaping.utils.slm.bench_constants``
# (imported above) so the bench value cannot drift across modules.


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


def random_phase(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Uniform random phase in [0, 2*pi) -- scatters the pupil for speckle.

    Only useful where the panel can resolve pixel-scale phase. On this bench it
    measurably does not (see :mod:`ao_shaping.tools.slm.panel.phase_resolution`),
    so a random phase leaves the far field a single tight focus.
    """
    return np.random.default_rng(int(seed)).uniform(
        0.0, 2.0 * np.pi, (int(shape[0]), int(shape[1]))
    )


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
        "verdict": "monotonic"
        if bool(np.all(ok)) and not saturated
        else "saturated"
        if saturated
        else "non_monotonic",
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
        raise ValueError(f"coeffs must hold grid*grid = {g * g} values, got {c.size}")
    ph, pw = int(panel_shape[0]), int(panel_shape[1])
    blocks = c.reshape(g, g)
    bh, bw = ph // g, pw // g
    if bh < 1 or bw < 1:
        raise ValueError(f"grid {g} is too coarse for panel {panel_shape!r}")
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
        cam,
        slm,
        phase_rad,
        n_frames=n_frames,
        n_discard=n_discard,
        wait_time_s=wait_time_s,
        stable_tol=stable_tol,
        max_wait_s=max_wait_s,
    )
