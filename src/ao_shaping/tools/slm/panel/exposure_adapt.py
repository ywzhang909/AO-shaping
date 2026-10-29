"""Exposure ladder: pick an exposure that makes a frame comparable, and read it.

Why this is its own module
--------------------------
Two halves of one decision live here. A frame is only comparable to another when
its peak sits inside an accepted band of full scale (below it the reading is read
noise, above it the peak is clipped and stops being linear in exposure), so
"which exposure" and "how do I read a peak at all" cannot be answered separately.
The sweep probes, the fingerprint probe, the transfer capture and the training
corpus collectors all import the ladder, which is why it does not live inside any
one of them (it grew inside :mod:`~ao_shaping.tools.slm.panel.focus_probe` and now
arrives here verbatim, with that module re-exporting the same names).

Nothing here touches a device: ``set_exposure_ms`` and ``read_frame`` are passed
in, so the ladder is exercised with synthetic frames in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ao_shaping.tools.kernels.bench_kernels import despike_frame
from ao_shaping.utils.image.exposure_control import FrameNoiseFloor, frame_noise_floor

#: The exposure band a point must land in to be *comparable*. Below the lower
#: edge the frame is read noise; above the upper edge it may already be clipped
#: (a clipped peak stops being linear in exposure, which is the one thing the
#: ranking depends on).
ACCEPT_LOW = 0.25
ACCEPT_HIGH = 0.80

#: Where the adaptive exposure aims inside that band. The geometric middle of
#: the band, so one multiplicative correction usually lands in it.
TARGET_FRACTION = 0.45

@dataclass(frozen=True)
class ExposureSetting:
    """One point's settled exposure and what it read there."""

    exposure_ms: float
    peak: float
    fraction: float
    status: str
    reads: int


@dataclass(frozen=True)
class ExposureStep:
    """One measured rung of :func:`adapt_exposure`'s ladder, handed to ``on_step``.

    The observer gets the frame's *own* noise floor with the peak, because
    "why is this ladder not converging" is only answerable as a comparison
    between the two: 峰值停在噪声底就是「光路没有光」, 峰值远高于噪声底却仍在
    接受带之外才是「曝光窗口进不去」。``is_last`` says whether another rung is
    still coming, so an observer can end the ladder early instead of waiting for
    a budget that cannot help.
    """

    index: int
    exposure_ms: float
    peak_counts: float
    noise: FrameNoiseFloor
    is_last: bool


def robust_peak(frame: np.ndarray) -> float:
    """Frame peak with isolated hot pixels removed.

    A 3x3 median (``despike_frame``) deletes single-pixel defects while leaving a
    real spot, which is spatially correlated, essentially untouched. Without it a
    hot pixel outranks a dim but genuinely focused spot and the sweep locks onto
    a sensor defect that does not move with the phase at all.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(despike_frame(arr, 3).max())


def frame_full_scale(frame: np.ndarray, override: float | None = None) -> float:
    """Full-scale value for a frame: the dtype maximum, or ``override``.

    Integer frames carry their own maximum. Float frames do not, and this bench's
    consumers all report on the 0-255 detector scale, so 255 is the convention.
    """
    if override is not None and float(override) > 0:
        return float(override)
    dtype = np.asarray(frame).dtype
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 255.0


def adapt_exposure(
    set_exposure_ms: Callable[[float], float],
    read_frame: Callable[[], np.ndarray],
    *,
    start_ms: float,
    floor_ms: float,
    ceiling_ms: float,
    full_scale: float,
    target: float = TARGET_FRACTION,
    accept: tuple[float, float] = (ACCEPT_LOW, ACCEPT_HIGH),
    max_steps: int = 6,
    discard: int = 2,
    ratio_bounds: tuple[float, float] = (0.15, 6.0),
    on_step: Callable[[ExposureStep], None] | None = None,
) -> ExposureSetting:
    """Find an exposure that puts the peak inside ``accept``, then report it.

    Multiplicative correction, because the response is linear in exposure: if the
    peak reads 0.9 of full scale at 0.4 ms, 0.45 needs 0.2 ms, and one step gets
    there. ``discard`` frames are dropped after each change so the reading is not
    taken on the previous exposure's integration.

    ``ratio_bounds`` clamps that one correction factor per step. The default
    ``(0.15, 6.0)`` is deliberately conservative — it is what the bench probes
    have always used. A caller that starts far from the answer (a cold start at
    the camera floor where the first frame reads ~5% of full scale needs a factor
    of ~9) may widen it and cut ``max_steps`` to 2: one exact correction plus one
    verification read. That is only safe because the response *is* linear in
    exposure; a camera with auto-gain would break the premise, not the bounds.

    ``on_step`` is an optional observer called with one :class:`ExposureStep`
    after every rung is measured (peak + that frame's own noise floor + whether
    another rung is still coming). It is how a caller ends an unanswerable
    ladder *early*: 2026-10-28 的光路失效让这个循环空转了 48 s 才报「曝光不收敛」
    (``report/shape_trials/multishape_scale/multishape_scale_wallclock_20261028.md``
    §9), 而峰值停在噪声底这件事在第 2 档就已经看得出来了。A raising observer
    aborts the ladder immediately; an observer that never raises changes nothing.
    The frame's median/MAD is only computed when an observer is registered, so the
    existing callers pay nothing for it.

    Returns the best reachable setting. Its ``status`` says whether it landed in
    the band (``ok``), was still clipped at the floor (``clipped`` — unusable for
    ranking, the peak is saturated), or was still dark at the ceiling (``dim`` —
    nothing measurable there).
    """
    lo, hi = float(accept[0]), float(accept[1])
    low_ratio, high_ratio = (float(ratio_bounds[0]), float(ratio_bounds[1]))
    if low_ratio <= 0 or high_ratio < low_ratio:
        raise ValueError("ratio_bounds must be positive and ordered")
    exposure = min(max(float(start_ms), floor_ms), ceiling_ms)
    peak = 0.0
    fraction = 0.0
    reads = 0

    def report(step: int, exposure: float, peak: float,
               noise: FrameNoiseFloor | None, is_last: bool) -> None:
        if on_step is not None:
            on_step(ExposureStep(index=step, exposure_ms=exposure, peak_counts=peak,
                                 noise=noise, is_last=is_last))

    for step in range(1, int(max_steps) + 1):
        set_exposure_ms(exposure)
        for _ in range(int(discard)):
            read_frame()
            reads += 1
        frame = read_frame()
        reads += 1
        peak = robust_peak(frame)
        fraction = peak / full_scale if full_scale else 0.0
        noise = frame_noise_floor(frame) if on_step is not None else None

        if lo <= fraction <= hi:
            report(step, exposure, peak, noise, True)
            return ExposureSetting(exposure, peak, fraction, "ok", reads)
        ratio = target / fraction if fraction > 1e-9 else 4.0
        ratio = float(min(max(ratio, low_ratio), high_ratio))
        nxt = float(min(max(exposure * ratio, floor_ms), ceiling_ms))
        # Clamped at a rail: more steps cannot improve, so stop and report why.
        if abs(nxt - exposure) <= 1e-6:
            report(step, exposure, peak, noise, True)
            break
        report(step, exposure, peak, noise, step >= int(max_steps))
        exposure = nxt

    status = "clipped" if fraction > hi else "dim"
    return ExposureSetting(exposure, peak, fraction, status, reads)
