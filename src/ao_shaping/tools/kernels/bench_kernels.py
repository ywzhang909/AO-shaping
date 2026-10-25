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
   frames until several consecutive readings agree *on peak, position and
   shape* -- see the "Settle criterion" section below, which is shared by
   every acquisition path in this module.

Which acquisition policy a measurement needs
--------------------------------------------
Two kinds of measurement share this module, and they do **not** need the same
thing from a phase write:

* **Geometry** (where is the spot, how wide is the beam) needs a *stable* frame,
  because it reads position and shape. It keeps the settle criterion
  (:data:`ProbePolicy` ``mode="settle"``, the default everywhere), including
  :func:`wait_for_settle` and its ``stable_tol``/``n_stable``.
* **Energy-ratio probes** (panel locate, beam extent) read the *ratio* of two
  box energies and need a *low-noise* frame, not a stable one. On the 2026-10-24
  calibration run those probes paid **8.1 s per phase write** and still timed
  out: peak varied 1.4% and fwhm 0.8% (well inside ``stable_tol=0.02``) while the
  **centroid** moved 1.2 px against a 0.8 px tolerance -- and the centroid is the
  one quantity neither :mod:`ao_shaping.tools.slm.panel.panel_locate` (box-energy
  ratio) nor :mod:`ao_shaping.tools.slm.panel.beam_extent` (core fraction at a
  frozen centroid) uses. The gate was held shut by an irrelevant term.

  So those probes pass ``ProbePolicy(mode="frames")``: no settle loop at all,
  a fixed budget of ``n_discard + n_frames`` frames combined **per pixel** by
  median, and a ``rel_sigma`` noise verdict in place of a settle verdict. A
  median is the right combiner because the outliers on that run were *corrupted
  frames*, not a settling trend: peak 14 / fwhm 32.5 against peak 72 / fwhm 13.3
  carry the same ``peak * fwhm**2`` (14.8k vs 12.7k), i.e. the energy was
  redistributed within the frame. Rejection fixes that; waiting does not.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

import numpy as np

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
# Backward-compat re-export: fitting + stability kernels moved to utils/math (R-49).
from ao_shaping.utils.math.fitting import fit_linear_slope
from ao_shaping.utils.math.stability import (
    exposure_monotonicity,
    settle_time_s,
    snr_vs_averages,
)
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)
from ao_shaping.utils.slm.panel_geometry import phase_to_panel
from ao_shaping.utils.wavefront.beam_amplitude import gaussian_grid
from ao_shaping.utils.wavefront.matrix_utils import (
    camera_pixel_um_from_focal_scale,
    focal_length_from_camera_pixel,
)

#: Union of both sources' star-export surfaces (`slm_bench_probe` had no `__all__`,
#: `slm_bench_metrics` declared 9). Keeping the union stops `import *` narrowing.
__all__ = [
    "BEAM_CENTER_PANEL",
    "BEAM_RADIUS_PANEL",
    "DEFAULT_MIN_WAIT_S",
    "DEFAULT_N_STABLE",
    "SLM_PANEL_H",
    "SLM_PANEL_W",
    "SLM_PITCH_M",
    "ProbePolicy",
    "SettleReport",
    "SettledAcquisition",
    "SpotMeasurement",
    "TILT_SHIFT_SCALE",
    "acquire_settled",
    "build_block_pattern",
    "capture_settled",
    "capture_settled_report",
    "core_fraction",
    "crop_around_zero_order",
    "crop_roi",
    "despike_frame",
    "display_and_average",
    "estimate_shift",
    "exposure_monotonicity",
    "finite_clip",
    "finite_median_subtract",
    "fresnel_panel",
    "fit_linear_slope",
    "flat_to_flat_floor",
    "measure_flat_reference",
    "measure_flat_reference_report",
    "measure_spot",
    "ramp_panel",
    "random_phase",
    "roi_l2",
    "settle_time_s",
    "smooth_frame",
    "snr_vs_averages",
    "tilt_shift_px",
    "wait_for_settle",
    "zernike_panel",
    "camera_pixel_um_from_focal_scale",
    "focal_length_from_camera_pixel",
    "gaussian_grid",
    "phase_to_panel",
]

# Panel-geometry constants (SLM_PANEL_W/H, SLM_PITCH_M, BEAM_CENTER_PANEL,
# BEAM_RADIUS_PANEL) and phase_to_panel are defined once in the utils/slm leaf
# (R-47) and imported above; re-exported here for existing ``bench_kernels``
# callers.
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


# ---------------------------------------------------------------------------
# Settle criterion -- shared by every acquisition path in this module
# ---------------------------------------------------------------------------
#
# 为什么判据长成这样
# ------------------
# 面板不会瞬时切换: 写入相位后液晶仍在松弛, 而驱动**无法**知道要松弛多久 ——
# 它的翻转时间估算按灰度图变化量推算, 两个灰度统计相近的相位会让它报 **0.0 ms**。
#
# 旧判据 ("连续两次读数的 peak 与质心一致") 在松弛中途**太容易**满足: 质心
# 往往还没动, 而光斑仍在展宽变暗, 于是两帧"一致", 闸门就在松弛中的帧上打开。
# 2026-10-21 本实验系统实测: 同一条平场阶梯在 09:02 / 13:32 / 16:44 读出
# 213 / 33 / 213 counts (10 ms 曝光), 几何自检的平场光斑 FWHM 在两天之间于
# 40.6 px 与 48.1 px 之间跳 —— 每条阶梯**内部严格线性**, 所以既不是曝光没生效,
# 也不是只有第一档被污染, 而是闸门在松弛中途就打开了。
#
# 现在判据要求同时满足:
#
# 1. ``n_stable`` (默认 3) 帧**彼此**一致的连续读数 —— 两帧一致在松弛中途很容易
#    偶然命中, 三帧显著更难。
# 2. 一致性覆盖**形状** (FWHM) 而不只是位置与峰值 —— "峰值已稳但仍在展宽"的
#    中间态正是旧判据漏掉的那一类。
# 3. 写入到第一次探测之间的**最小等待** (``min_wait_s``) —— 闸门在面板还没机会
#    松弛时就打开是没有意义的。
#
# 若在 ``max_wait_s`` 内仍未满足判据, 帧**仍然返回** (既有契约, 调用方依赖它),
# 但结论被明确标记为**未稳定** (:class:`SettleReport`): 沉降失败绝不能与
# "已经稳定" 混为一谈。

#: Consecutive agreeing readings required before a frame counts as settled.
#:
#: Why 3 and not 2: two agreeing readings are easy to hit *by accident* while the
#: panel is still relaxing -- the centroid has often not moved yet. Measured
#: 2026-10-21, a flat ladder read 3.8-6.4x brighter at one time of day than at
#: another, and the flat-spot FWHM flipped between 40.6 px and 48.1 px across two
#: runs of the same geometric self-check. Three mutually consistent readings
#: remove the accidental case at the cost of two extra camera reads.
DEFAULT_N_STABLE = 3

#: Floor on the delay between the phase write and the first stability probe.
#:
#: 0.3 s is the settle time this repo's own flat-phase check uses
#: (``scripts/slm/validate_flat_phase_gray.py --wait-time-s 0.3 --discard-count 3``,
#: documented in ``scripts/README.md``). It is a *floor*, not the whole wait:
#: when the driver-level ``wait_time_s`` already exceeded it, nothing is added.
#: The measured relaxation of this bench reaches past 1 s (fwhm 43.2 px -> 12.8 px
#: over 3 s), so the floor alone is not enough -- it only stops the criterion
#: from opening before the panel has had any chance to relax at all.
DEFAULT_MIN_WAIT_S = 0.3

#: One spot reading: ``(peak, centroid_x, centroid_y, fwhm_px)``.
_SpotReading = tuple[float, float, float, float]

#: Absolute floor (counts) for the relative peak comparison.
_PEAK_FLOOR = 1.0
#: Absolute floor (px) for the relative width comparison.
_WIDTH_FLOOR_PX = 1.0
#: Centroid tolerance per unit of ``stable_tol`` (px).
_CENTROID_TOL_PX = 40.0


@dataclass(frozen=True)
class ProbePolicy:
    """How one phase write becomes one measurement.

    ``mode="settle"`` is the historical path: wait until the settle criterion is
    met, then average ``n_frames`` fresh frames. ``mode="frames"`` is the
    fixed-budget path for energy-ratio probes -- see "Which acquisition policy a
    measurement needs" in the module docstring.
    """

    #: ``"settle"`` (criterion-gated) or ``"frames"`` (fixed budget).
    mode: str = "settle"
    #: Frames kept (``mode="frames"``) -- they are combined per pixel.
    n_frames: int = 7
    #: Frames thrown away after the write, before the kept ones.
    n_discard: int = 2
    #: ``"median"`` (rejects a corrupted frame) or ``"mean"``.
    combine: str = "median"

    def __post_init__(self) -> None:
        if self.mode not in ("settle", "frames"):
            raise ValueError(f'mode must be "settle" or "frames", got {self.mode!r}')
        if self.combine not in ("median", "mean"):
            raise ValueError(
                f'combine must be "median" or "mean", got {self.combine!r}'
            )
        if int(self.n_frames) < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.n_frames!r}")
        if int(self.n_discard) < 0:
            raise ValueError(f"n_discard must be >= 0, got {self.n_discard!r}")

    @property
    def is_fixed_budget(self) -> bool:
        """``True`` when this policy skips the settle criterion."""
        return self.mode == "frames"


@dataclass(frozen=True)
class SettleReport:
    """What the settle criterion concluded for one phase write.

    Attributes:
        settled: ``True`` only when the criterion was met.
        reason: ``"stable"``, ``"timeout"``, or ``"fixed_budget"`` (a
            ``ProbePolicy(mode="frames")`` acquisition, which does not run the
            criterion at all -- that is the chosen policy, not a failure).
        n_probes: Number of stability probes taken (frames mode: every frame
            read for this write, discarded ones included).
        n_stable: Consecutive agreeing readings the criterion demands. ``0`` in
            ``frames`` mode, where no agreement test is applied.
        waited_s: Seconds from the phase write to the end of the criterion.
        peak: Peak of the last reading (``frames`` mode: of the combined frame).
        fwhm_px: FWHM of the last reading (``frames`` mode: of the combined frame).
        centroid_x: Centroid x of the last reading.
        centroid_y: Centroid y of the last reading.
        rel_sigma: Relative MAD-based sigma of the per-frame peak across the
            **kept** frames (``frames`` mode only; ``None`` otherwise, and
            ``None`` for a single kept frame, where a scatter estimate would be
            vacuous). This is the noise verdict that replaces the settle verdict
            for an energy-ratio probe.
    """

    settled: bool
    reason: str
    n_probes: int
    n_stable: int
    waited_s: float
    peak: float
    fwhm_px: float
    centroid_x: float
    centroid_y: float
    rel_sigma: float | None = None

    def as_row(self) -> str:
        """One-line summary for a log line."""
        text = (
            f"settled={self.settled} ({self.reason}) probes={self.n_probes} "
            f"peak={self.peak:.0f} fwhm={self.fwhm_px:.1f}px "
            f"c=({self.centroid_x:.1f},{self.centroid_y:.1f})"
        )
        if self.rel_sigma is not None:
            text += f" rel_sigma={self.rel_sigma:.2%}"
        return text


@dataclass(frozen=True)
class SettledAcquisition:
    """An averaged frame plus the settle verdict that gated it."""

    frame: np.ndarray
    settle: SettleReport

    @property
    def settled(self) -> bool:
        """``True`` when the frame was taken after the criterion was met."""
        return self.settle.settled


def _probe_spot(cam) -> _SpotReading:
    """Read one frame and characterise its spot: ``(peak, cx, cy, fwhm)``.

    A frame that cannot be measured -- a non-finite pixel makes
    :func:`measure_spot` raise -- yields a reading with a ``nan`` width, which
    never counts as "agreeing". An unmeasurable frame must not be mistaken for
    a settled one, and it must not abort the whole acquisition either.
    """
    img = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
    try:
        m = measure_spot(img)
    except (ValueError, FloatingPointError) as exc:
        logger.debug("spot measurement failed on this frame: {}", exc)
        return 0.0, 0.0, 0.0, float("nan")
    return m.peak, m.centroid_x, m.centroid_y, m.fwhm_px


def _readings_agree(a: _SpotReading, b: _SpotReading, stable_tol: float) -> bool:
    """Whether two spot readings agree within ``stable_tol``.

    Peak and width are compared *relatively* (the same shape of criterion, with
    an absolute floor so a near-zero value cannot make the test vacuous); the
    centroid is compared absolutely, in pixels.

    A non-finite width (``nan`` for a degenerate frame) makes the pair
    **disagree**: an unmeasurable shape must never be read as "unchanged".
    """
    peak_a, cx_a, cy_a, width_a = a
    peak_b, cx_b, cy_b, width_b = b
    if not (math.isfinite(peak_a) and math.isfinite(peak_b)):
        return False
    if not (math.isfinite(width_a) and math.isfinite(width_b)):
        return False
    peak_tol = stable_tol * max(abs(peak_a), abs(peak_b), _PEAK_FLOOR)
    width_tol = stable_tol * max(abs(width_a), abs(width_b), _WIDTH_FLOOR_PX)
    centroid_tol = stable_tol * _CENTROID_TOL_PX
    return (
        abs(peak_a - peak_b) <= peak_tol
        and abs(cx_a - cx_b) <= centroid_tol
        and abs(cy_a - cy_b) <= centroid_tol
        and abs(width_a - width_b) <= width_tol
    )


def _window_agrees(window: list[_SpotReading], stable_tol: float) -> bool:
    """Whether every pair in ``window`` agrees within ``stable_tol``."""
    return all(
        _readings_agree(window[i], window[j], stable_tol)
        for i in range(len(window))
        for j in range(i + 1, len(window))
    )


def wait_for_settle(
    cam,
    *,
    n_discard: int = 3,
    min_wait_s: float = DEFAULT_MIN_WAIT_S,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
    n_stable: int = DEFAULT_N_STABLE,
    started_at: float | None = None,
) -> SettleReport:
    """Discard frames until ``n_stable`` consecutive readings agree.

    The single settle criterion of this module -- :func:`display_and_average`,
    :func:`capture_settled` and :func:`acquire_settled` all go through it, so
    there is exactly one definition of "settled" on the bench.

    A reading is ``(peak, centroid_x, centroid_y, fwhm_px)``; two readings must
    agree on **all four** to ``stable_tol`` (peak and width relatively, the
    centroid absolutely) and ``n_stable`` of them must agree *with each other*.

    Args:
        cam: An open camera exposing ``get_numpy_image(n_sample=...)``.
        n_discard: Frames discarded before the first probe.
        min_wait_s: Floor on the delay between the phase write and the first
            probe. ``started_at`` is the timestamp of the write.
        stable_tol: Relative agreement required between readings.
        max_wait_s: Cap on the wait for stability, measured from the first probe.
        n_stable: Consecutive agreeing readings required.
        started_at: Monotonic timestamp of the phase write; defaults to now.

    Returns:
        A :class:`SettleReport`. ``settled`` is ``False`` when the criterion
        could not be met within ``max_wait_s`` -- the caller must not silently
        treat such a frame as a settled measurement.
    """
    n_stable = int(n_stable)
    if n_stable < 1:
        raise ValueError(f"n_stable must be >= 1, got {n_stable!r}")

    t_write = time.monotonic() if started_at is None else float(started_at)

    for _ in range(int(n_discard)):
        cam.get_numpy_image(n_sample=1)

    # Floor on the time between the phase write and the first probe. The
    # driver-level wait usually already covers it; when it does not (an
    # automatic flip-time estimate can be 0.0 ms), this is what stops the
    # criterion from opening on the very first frame after the write.
    remaining = float(min_wait_s) - (time.monotonic() - t_write)
    if remaining > 0.0:
        time.sleep(remaining)

    deadline = time.monotonic() + float(max_wait_s)
    window: list[_SpotReading] = []
    n_probes = 0
    last: _SpotReading = (0.0, 0.0, 0.0, float("nan"))
    settled = False
    while True:
        last = _probe_spot(cam)
        n_probes += 1
        window.append(last)
        if len(window) > n_stable:
            window.pop(0)
        if len(window) >= n_stable and _window_agrees(window, float(stable_tol)):
            settled = True
            break
        if time.monotonic() >= deadline:
            break

    peak, cx, cy, width = last
    return SettleReport(
        settled=settled,
        reason="stable" if settled else "timeout",
        n_probes=n_probes,
        n_stable=n_stable,
        waited_s=time.monotonic() - t_write,
        peak=peak,
        fwhm_px=width,
        centroid_x=cx,
        centroid_y=cy,
    )


def _warn_if_unsettled(settle: SettleReport) -> None:
    """Log a warning when the settle criterion was not met.

    A relaxing frame must not look like a settled one. The numpy-returning
    entry points keep their historical signature, so the verdict is carried by
    this warning -- and by :func:`acquire_settled` for callers that want it
    programmatically.

    ``reason="fixed_budget"`` is **silent**: the caller asked for a fixed frame
    budget instead of the criterion (:class:`ProbePolicy`), so not running the
    criterion is the policy working, not a measurement failure.
    """
    if settle.settled or settle.reason == "fixed_budget":
        return
    logger.warning(
        "settle criterion not met after {:.2f} s ({} probes, n_stable={}): {} "
        "-- the frame is still relaxing and must be treated as provisional",
        settle.waited_s,
        settle.n_probes,
        settle.n_stable,
        settle.as_row(),
    )


def _combine_stacked(stack: np.ndarray, combine: str) -> np.ndarray:
    """Combine a ``(n, h, w)`` frame stack per pixel.

    Median by default: on the bench the deep outliers were *corrupted* frames
    (energy redistributed inside the frame), which a median rejects and a mean
    smears into the answer.
    """
    if combine == "median":
        return np.median(stack, axis=0)
    return np.mean(stack, axis=0)


def _rel_peak_sigma(stack: np.ndarray) -> float | None:
    """Relative MAD-based sigma of the per-frame peak across a frame stack.

    One array op (:meth:`numpy.ndarray.max` over the frame axes). The peak is the
    scalar ``measure_spot`` already reports, so the number means the same thing
    as ``SettleReport.peak``.

    Returns ``None`` for a single frame: the MAD of one sample is not a scatter
    estimate, and reporting ``0.0`` there would read as "noiseless".
    """
    if stack.shape[0] < 2:
        return None
    peaks = stack.max(axis=(1, 2))
    median = float(np.median(peaks))
    if not math.isfinite(median) or median <= 0.0:
        return None
    mad = float(np.median(np.abs(peaks - median)))
    return 1.4826 * mad / median


def _acquire_fixed_budget(
    cam,
    *,
    policy: ProbePolicy,
    min_wait_s: float,
    t_write: float,
) -> tuple[np.ndarray, SettleReport]:
    """Fixed frame budget: discard, grab, combine per pixel, report the noise."""
    # The floor still applies: an automatic flip-time estimate can be 0.0 ms, so
    # reading immediately after the write would record the previous pattern.
    remaining = float(min_wait_s) - (time.monotonic() - float(t_write))
    if remaining > 0.0:
        time.sleep(remaining)

    for _ in range(int(policy.n_discard)):
        cam.get_numpy_image(n_sample=1)

    kept = [
        np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        for _ in range(int(policy.n_frames))
    ]
    stack = np.stack(kept, axis=0)
    frame = _combine_stacked(stack, policy.combine)
    rel_sigma = _rel_peak_sigma(stack)

    try:
        measurement = measure_spot(frame)
        peak, cx, cy, width = (
            measurement.peak,
            measurement.centroid_x,
            measurement.centroid_y,
            measurement.fwhm_px,
        )
    except (ValueError, FloatingPointError) as exc:
        logger.debug("spot measurement failed on the combined frame: {}", exc)
        peak, cx, cy, width = 0.0, 0.0, 0.0, float("nan")

    report = SettleReport(
        settled=False,
        reason="fixed_budget",
        n_probes=int(policy.n_discard) + int(policy.n_frames),
        n_stable=0,
        waited_s=time.monotonic() - float(t_write),
        peak=peak,
        fwhm_px=width,
        centroid_x=cx,
        centroid_y=cy,
        rel_sigma=rel_sigma,
    )
    return frame, report


def acquire_settled(
    cam,
    slm,
    phase_rad: np.ndarray,
    *,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float | None = 0.5,
    min_wait_s: float = DEFAULT_MIN_WAIT_S,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
    n_stable: int = DEFAULT_N_STABLE,
    policy: ProbePolicy | None = None,
) -> SettledAcquisition:
    """Display ``phase_rad``, wait for the panel to settle, average the frames.

    The one acquisition implementation of this module: it writes the phase with
    ``display_data()`` (no ``memory_number``, so the slot rotates itself --
    fact 3 in the module docstring), applies the shared settle criterion, and
    averages ``n_frames`` fresh frames afterwards.

    ``policy=ProbePolicy(mode="frames")`` switches to the fixed-budget path used
    by energy-ratio probes: **no** :func:`wait_for_settle` call, ``n_discard``
    frames thrown away, ``policy.n_frames`` kept and combined per pixel by
    ``policy.combine`` (median), and a ``rel_sigma`` noise verdict instead of a
    settle verdict. ``policy=None`` (default) is the historical behaviour, byte
    for byte.

    Args:
        cam: An open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: An open SLM exposing ``create_phase_from_array`` and
            ``display_data(gray, wait_time_s)``.
        phase_rad: Raw unwrapped radians, panel-shaped.
        n_frames: Frames averaged into the returned measurement.
        n_discard: Frames discarded before the first stability probe.
        wait_time_s: Driver-level wait after the write (``None`` = the driver
            estimates it, which it under-reports -- see the module docstring).
        min_wait_s: Floor on the delay between the write and the first probe.
        stable_tol: Relative agreement required between readings.
        max_wait_s: Cap on the wait for stability.
        n_stable: Consecutive agreeing readings required.
        policy: Acquisition policy; ``None`` means ``mode="settle"`` with the
            keyword arguments above. In ``frames`` mode ``policy.n_frames`` /
            ``policy.n_discard`` / ``policy.combine`` govern instead.

    Returns:
        A :class:`SettledAcquisition`: the averaged frame plus the verdict.
    """
    gray = slm.create_phase_from_array(phase_rad)
    t_write = time.monotonic()
    slm.display_data(gray, wait_time_s)
    if policy is not None and policy.is_fixed_budget:
        frame, settle = _acquire_fixed_budget(
            cam, policy=policy, min_wait_s=min_wait_s, t_write=t_write
        )
        return SettledAcquisition(frame=frame, settle=settle)

    settle = wait_for_settle(
        cam,
        n_discard=n_discard,
        min_wait_s=min_wait_s,
        stable_tol=stable_tol,
        max_wait_s=max_wait_s,
        n_stable=n_stable,
        started_at=t_write,
    )
    acc = [
        np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        for _ in range(int(n_frames))
    ]
    return SettledAcquisition(frame=np.mean(acc, axis=0), settle=settle)


def measure_flat_reference(
    cam,
    slm,
    n_frames: int = 4,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
    *,
    policy: ProbePolicy | None = None,
    wait_time_s: float | None = 0.5,
) -> tuple[np.ndarray, SpotMeasurement]:
    """Write flat phase first, then average ``n_frames`` frames.

    Fact 2 in the module docstring: the panel retains the last displayed pattern,
    so a flat reference must be *displayed* before it is read.

    Returns the averaged frame and its spot measurement.
    """
    frame = display_and_average(
        cam,
        slm,
        np.zeros(panel_shape),
        n_frames=n_frames,
        wait_time_s=wait_time_s,
        policy=policy,
    )
    return frame, measure_spot(frame)


def measure_flat_reference_report(
    cam,
    slm,
    n_frames: int = 4,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
    *,
    policy: ProbePolicy | None = None,
    wait_time_s: float | None = 0.5,
) -> tuple[np.ndarray, SpotMeasurement, SettleReport]:
    """Same flat reference as :func:`measure_flat_reference`, **with** the verdict.

    The flat reference is the frame every later measurement is compared against,
    so "was it settled?" matters most here. :func:`measure_flat_reference` keeps
    its historical two-value return and logs a warning instead.

    Returns:
        The averaged frame, its spot measurement, and the settle verdict.
    """
    result = acquire_settled(
        cam,
        slm,
        np.zeros(panel_shape),
        n_frames=n_frames,
        wait_time_s=wait_time_s,
        policy=policy,
    )
    _warn_if_unsettled(result.settle)
    return result.frame, measure_spot(result.frame), result.settle


def display_and_average(
    cam,
    slm,
    phase_rad: np.ndarray,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float | None = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
    *,
    min_wait_s: float = DEFAULT_MIN_WAIT_S,
    n_stable: int = DEFAULT_N_STABLE,
    policy: ProbePolicy | None = None,
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

    So: write the phase, wait at least ``min_wait_s``, then keep reading until
    ``n_stable`` consecutive readings agree with *each other* on peak, position
    **and width** to ``stable_tol``, capped at ``max_wait_s``.

    Energy-ratio probes pass ``policy=ProbePolicy(mode="frames")``: the settle
    criterion is skipped entirely and a fixed frame budget is combined per pixel
    by median. See the module docstring for why the criterion cannot converge
    for them (the binding term is the centroid, which they never use).

    The verdict is **not** folded into the return value -- this function keeps
    its historical ``np.ndarray`` contract -- but an unsettled outcome is logged
    as a warning, and :func:`acquire_settled` exposes it programmatically.

    Args:
        cam: An open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: An open SLM exposing ``create_phase_from_array`` and
            ``display_data(gray, wait_time_s)``.
        phase_rad: Raw unwrapped radians, panel-shaped.
        n_frames: Frames averaged into the returned measurement.
        n_discard: Frames discarded before the first stability probe.
        wait_time_s: Driver-level wait after the write.
        stable_tol: Relative agreement required between readings.
        max_wait_s: Cap on the wait for stability.
        min_wait_s: Floor on the delay between the write and the first probe.
        n_stable: Consecutive agreeing readings required.
        policy: Acquisition policy (see :func:`acquire_settled`).
    """
    result = acquire_settled(
        cam,
        slm,
        phase_rad,
        n_frames=n_frames,
        n_discard=n_discard,
        wait_time_s=wait_time_s,
        min_wait_s=min_wait_s,
        stable_tol=stable_tol,
        max_wait_s=max_wait_s,
        n_stable=n_stable,
        policy=policy,
    )
    _warn_if_unsettled(result.settle)
    return result.frame


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


def fresnel_panel(
    focal_mm: float,
    radius: int = BEAM_RADIUS_PANEL,
    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
    *,
    wavelength_nm: float = 1064.0,
    pixel_um: float = SLM_PITCH_M * 1e6,
) -> np.ndarray:
    """Quadratic (lens) phase over the illuminated disc, in panel coordinates.

    A plain (flat) phase loses its focus as soon as the camera leaves the 2f
    plane: behind focus the beam is a diverging cone whose blur grows linearly
    with the offset, so the spot gets larger and dimmer. A quadratic phase on the
    panel changes the wavefront curvature and moves the focus back onto the
    detector -- that is the correction this function builds and
    :mod:`ao_shaping.tools.slm.panel.focus_probe` searches for.

    The form is ``pi * r**2 / (lambda * f)``, identical to
    :meth:`PatternHelper.generate_focus`, so the same ``focal_mm`` means the same
    phase everywhere in this repo. **Which sign brings the camera back into focus
    is not assumed here** -- it depends on the sign convention of the panel's
    phase addition and on how far the detector moved, so the probe scans both
    branches and reports what it measured.

    Centred on ``pupil_center`` (the *illuminated* centre) rather than on the
    panel centre: a lens centred anywhere else also steers the beam, because the
    off-centre quadratic carries a linear term. Coordinates are clamped into the
    panel, so a stale centre cannot push the patch out of bounds.

    Args:
        focal_mm: Focal length in mm. ``inf`` returns zeros (a plain phase);
            ``0`` is rejected -- a zero focal length is not a lens and silently
            returning flat would look like a successful correction.
        radius: Aperture radius in panel pixels. Zero outside the disc.
        pupil_center: Beam centre in panel pixels (x, y).
        panel_shape: ``(height, width)`` of the panel.
        wavelength_nm: Illumination wavelength.
        pixel_um: Panel pixel pitch.

    Returns:
        A ``panel_shape`` float64 phase in raw radians, zero outside the disc.

    Raises:
        ValueError: If ``focal_mm`` is zero or non-finite other than ``inf``.
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    out = np.zeros((height, width), dtype=np.float64)
    if np.isinf(focal_mm):
        return out
    focal_mm = float(focal_mm)
    if not np.isfinite(focal_mm):
        raise ValueError(f"focal_mm must be finite or +-inf; got {focal_mm!r}")
    if focal_mm == 0.0:
        raise ValueError(
            "focal_mm = 0 is not a lens; pass inf for a plain phase instead"
        )

    lam_m = float(wavelength_nm) * 1e-9
    pixel_m = float(pixel_um) * 1e-6
    f_m = focal_mm * 1e-3
    r = int(max(int(radius), 1))
    cx = min(max(int(pupil_center[0]), 0), width - 1)
    cy = min(max(int(pupil_center[1]), 0), height - 1)

    size = 2 * r + 1
    yy, xx = np.mgrid[0:size, 0:size]
    rr2 = (xx - r) ** 2 + (yy - r) ** 2
    patch = (np.pi / (lam_m * f_m)) * (rr2 * pixel_m**2)
    patch[rr2 > r * r] = 0.0

    # Clip both the destination and the patch, so a centre near (or past) an edge
    # writes the visible part instead of failing on a shape mismatch.
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r + 1, cy + r + 1
    dx0, dy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, width), min(y1, height)
    if x1c > x0c and y1c > y0c:
        out[y0c:y1c, x0c:x1c] = patch[
            dy0 : dy0 + (y1c - y0c), dx0 : dx0 + (x1c - x0c)
        ]
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


def random_phase(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Uniform random phase in [0, 2*pi) -- scatters the pupil for speckle.

    Only useful where the panel can resolve pixel-scale phase. On this bench it
    measurably does not (see :mod:`ao_shaping.tools.slm.panel.phase_resolution`),
    so a random phase leaves the far field a single tight focus.
    """
    return np.random.default_rng(int(seed)).uniform(
        0.0, 2.0 * np.pi, (int(shape[0]), int(shape[1]))
    )


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
    min_wait_s: float = DEFAULT_MIN_WAIT_S,
    n_stable: int = DEFAULT_N_STABLE,
    policy: ProbePolicy | None = None,
) -> np.ndarray:
    """Display one phase and return the averaged far-field frame once settled.

    The single-phase entry point, for callers that build their own acquisition
    order (an ABBA interleave, a repeated push-pull) and only need the settle
    discipline and the averaging from here.

    It shares the settle criterion with :func:`display_and_average` verbatim --
    both go through :func:`acquire_settled`, so there is only one definition of
    "settled" on this bench.

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
        min_wait_s: Floor on the delay between the write and the first probe.
        n_stable: Consecutive agreeing readings required.
        policy: Acquisition policy; ``None`` runs the settle criterion.

    Returns:
        The averaged frame.
    """
    result = acquire_settled(
        cam,
        slm,
        phase_rad,
        n_frames=n_frames,
        n_discard=n_discard,
        wait_time_s=wait_time_s,
        min_wait_s=min_wait_s,
        stable_tol=stable_tol,
        max_wait_s=max_wait_s,
        n_stable=n_stable,
        policy=policy,
    )
    _warn_if_unsettled(result.settle)
    return result.frame


def capture_settled_report(
    cam: Any,
    slm: Any,
    phase_rad: np.ndarray,
    *,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
    min_wait_s: float = DEFAULT_MIN_WAIT_S,
    n_stable: int = DEFAULT_N_STABLE,
    policy: ProbePolicy | None = None,
) -> SettledAcquisition:
    """Same acquisition as :func:`capture_settled`, but **with** the verdict.

    Use this when the caller has to know whether the frame it got was really
    settled -- a settling failure must never be silently turned into a
    measurement. :func:`capture_settled` keeps its historical ``np.ndarray``
    return type and logs a warning instead.

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
        min_wait_s: Floor on the delay between the write and the first probe.
        n_stable: Consecutive agreeing readings required.
        policy: Acquisition policy; ``None`` runs the settle criterion.

    Returns:
        The averaged frame plus the settle verdict.
    """
    return acquire_settled(
        cam,
        slm,
        phase_rad,
        n_frames=n_frames,
        n_discard=n_discard,
        wait_time_s=wait_time_s,
        min_wait_s=min_wait_s,
        stable_tol=stable_tol,
        max_wait_s=max_wait_s,
        n_stable=n_stable,
        policy=policy,
    )
