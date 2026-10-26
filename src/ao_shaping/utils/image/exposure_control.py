"""Bounded manual exposure control for quantitative CCD measurements.

The module owns the per-session Daheng exposure ceiling.  The ceiling is a
property of the optical path (laser power, attenuation, coupling), not of the
sensor, so it must never be a package constant.  Every caller resolves it
through the single entry point :func:`resolve_daheng_exposure_limit`, which
prefers an explicit value, then the ``AO_DAHENG_MAX_EXPOSURE_MS`` environment
variable, and otherwise probes the CCD image with a doubling exposure ladder
(:func:`probe_exposure_ceiling_ms`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
import os
from typing import Any

import numpy as np

from loguru import logger


DAHENG_EXPOSURE_LIMIT_ENV = "AO_DAHENG_MAX_EXPOSURE_MS"


def configured_daheng_exposure_limit() -> float:
    """Return the optional per-session Daheng exposure ceiling.

    A missing variable or a literal ``0`` means that the caller must resolve
    the ceiling from the connected camera.  This intentionally does not bake a
    laboratory-specific value into the package: the usable range changes with
    attenuation and the optical path.
    """
    raw = os.environ.get(DAHENG_EXPOSURE_LIMIT_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DAHENG_EXPOSURE_LIMIT_ENV} must be a finite number in ms"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{DAHENG_EXPOSURE_LIMIT_ENV} must be a finite non-negative number"
        )
    return value


def resolve_daheng_exposure_limit(
    requested_ms: float = 0.0,
    *,
    camera: Any | None = None,
    probe_start_ms: float = 0.02,
    probe_hard_max_ms: float | None = None,
) -> float:
    """Resolve a Daheng exposure ceiling: explicit config, env, then image probe.

    ``0`` is an auto sentinel at every layer.  A positive explicit value takes
    precedence over ``AO_DAHENG_MAX_EXPOSURE_MS``, which takes precedence over
    probing the connected ``camera`` with a doubling exposure ladder.  When a
    probe runs, its outcome is logged and the returned value is the safe
    ceiling.  The probe leaves the camera at the last ladder step, so the caller
    owns re-arming the working exposure afterwards.

    A probe that lands below ``probe_start_ms`` means the optical path saturates
    even at the ladder floor — no exposure setting can fix that, so it raises an
    actionable error instead of returning an unusable ceiling.  ``camera`` is
    used only through ``reset_exposure_time`` and ``get_numpy_image``, so no
    driver import is required here.

    The ``shoot`` callback must NOT drop a frame of its own: the driver layer
    already consumes the stale frames when the exposure changes (the Daheng
    backend's ``reset_exposure_time`` drops ``EXPOSURE_CHANGE_DISCARD_FRAMES``
    frames), so dropping again in the probe would be a duplicate — one extra
    frame per ladder step for every backend, and no extra safety.
    """
    requested = float(requested_ms)
    if not math.isfinite(requested) or requested < 0:
        raise ValueError("requested Daheng exposure limit must be finite and non-negative")
    limit = requested or configured_daheng_exposure_limit()
    if not limit:
        if camera is None:
            raise ValueError(
                "Daheng exposure ceiling is unavailable; set "
                f"{DAHENG_EXPOSURE_LIMIT_ENV} or pass an open camera to probe it"
            )
        try:
            estimate = probe_exposure_ceiling_ms(
                camera.reset_exposure_time,
                lambda: np.asarray(camera.get_numpy_image(n_sample=1)),
                start_ms=probe_start_ms,
                hard_max_ms=probe_hard_max_ms,
            )
        except ValueError as exc:
            if "saturated at the minimum probe exposure" in str(exc):
                raise RuntimeError(
                    "CCD saturates at minimum exposure; reduce laser power or add attenuation"
                ) from exc
            raise
        limit = min(estimate.ceiling_ms, probe_hard_max_ms) if probe_hard_max_ms else estimate.ceiling_ms
        logger.info(
            "Daheng exposure ceiling probed from the image: {:.4f} ms "
            "(last safe {:.4f} ms, saturated at {})",
            limit,
            estimate.last_safe_ms,
            "never" if estimate.saturated_ms is None else f"{estimate.saturated_ms:.4f} ms",
        )
        if limit < probe_start_ms:
            raise RuntimeError(
                f"probed Daheng exposure ceiling {limit:.4f} ms is below the "
                f"ladder floor {probe_start_ms:.4f} ms: the optical path is "
                "saturated even at minimum exposure — reduce laser power or "
                "add attenuation before probing again"
            )
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(
            "Daheng exposure ceiling is unavailable; set "
            f"{DAHENG_EXPOSURE_LIMIT_ENV} or probe an open camera"
        )
    return limit


@dataclass(frozen=True)
class ExposureCeilingEstimate:
    """Outcome of an image-based safe-exposure-ceiling estimate.

    ``ceiling_ms`` is the value callers should use as the session ceiling: the
    exposure at which the brightest pixel would reach ``saturation_fraction``
    of full scale, discounted by the safety factor.  ``last_safe_ms`` is the
    largest ladder exposure measured *below* saturation (0.0 when every frame
    saturated), and ``saturated_ms`` the smallest one at or above it.

    The remaining fields are diagnostics of *how* the ceiling was reached, so a
    probe can be replayed from its evidence.  ``peak_fractions`` is one entry
    per ladder step, in exposure order; ``estimates_ms`` the per-rung
    extrapolation ``ms * saturation_fraction / peak_fraction`` for every rung
    that carried signal (the old ``min`` of these is what the fit replaces),
    with ``estimate_exposures_ms`` the matching exposures — ``estimates_ms``
    has one entry per *carrying* rung, so zip the two to pair them, while
    ``peak_fractions`` stays one entry per ladder step.  ``fit_ms`` is the
    fitted saturation exposure before clamping and headroom (``None`` when the
    fit was rejected); ``fit_slope`` / ``fit_intercept`` the fitted
    ``peak_counts = a + k * ms`` in counts per ms and counts; ``fit_points``
    how many rungs entered the fit *whether or not it was accepted*.
    ``first_clipped_ms`` is the first step whose frame reached full scale (a
    censored measurement, ``None`` when nothing clipped) and ``bound_ms`` the
    upper bound actually applied to the fit before ``headroom`` (``None`` when
    the ceiling was unbounded).
    """

    ceiling_ms: float
    last_safe_ms: float
    saturated_ms: float | None
    peak_fractions: tuple[float, ...]
    estimates_ms: tuple[float, ...] = ()
    estimate_exposures_ms: tuple[float, ...] = ()
    fit_ms: float | None = None
    fit_slope: float | None = None
    fit_intercept: float | None = None
    fit_points: int = 0
    first_clipped_ms: float | None = None
    bound_ms: float | None = None


#: How far above a frame's own background its peak has to sit before the peak
#: counts as light instead of detector read noise.  See
#: :func:`estimate_exposure_ceiling_ms` for the measurement that fixes it.
NOISE_FLOOR_SIGMAS: float = 4.0

#: Noise scale assumed when a frame's MAD is exactly zero (more than half of
#: the frame sits on the median, so the MAD cannot resolve the spread).  One
#: count is the detector's quantisation step, i.e. the smallest spread a
#: digital frame can express.
_NOISE_FLOOR_FALLBACK_SIGMA: float = 1.0

#: How much of full scale a ladder rung has to reach before its peak can inform
#: the extrapolation.  The gate removes two things a dim rung cannot separate
#: from its own peak: quantisation — a rung at this gate is 25 counts on an
#: 8-bit frame, so one count is 4 % of it — and the noise maximum, which
#: inflates a dim rung's peak.  A DC pedestal is *not* one of them: the fitted
#: intercept absorbs it (the measured probe's intercept is 4 counts, 16 % of a
#: 25-count rung, so it is nowhere near a small correction).  That is exactly
#: why the fit needs its own top-rung gate, see
#: ``_FIT_MIN_TOP_PEAK_FRACTION``.
_FIT_MIN_PEAK_FRACTION: float = 0.10

#: How bright the brightest rung entering the fit has to be.  Without it the fit
#: is a slope no rung constrains, and a pedestal-dominated ladder then
#: extrapolates to a ceiling that does not exist.  Measured counter-example:
#: a flat pedestal ladder peaking 26/27/26/27 counts at 0.02/0.04/0.08/0.16 ms
#: (all four admitted by :func:`_carries_signal`, because a frame whose peak
#: equals its median is signal, and 26/255 = 10.2 % squeaks past the 10 % gate)
#: fits 4.35 counts/ms with a 26.2-count intercept — the whole "signal" is the
#: pedestal — so it predicts a ceiling of 46.8 ms and returns **42.1 ms**, 265x
#: the conservative per-rung minimum of 0.159 ms, on a bench that clips long
#: before that.  A two-rung ladder has no residual at all (30/34 counts at
#: 0.02/0.04 ms fits 0.916 ms against a 0.138 ms minimum), hence the separate
#: three-rung floor.
#:
#: 0.25 keeps the measured 2026-10-24 ladder (top rung 142/255 = 0.557, three
#: qualifying rungs) and rejects both counter-examples (26/27 -> 0.106,
#: 30/34 -> 0.133).  It is calibrated for 8-bit frames: a uint16 frame from a
#: <=10-bit sensor never reaches 10 % of 65535 on any rung, so the fit never
#: engages and the conservative ``min`` path is used.  That is the intended
#: safe behaviour, not a defect to work around here.
_FIT_MIN_TOP_PEAK_FRACTION: float = 0.25


def _carries_signal(image: np.ndarray, peak: float, full_scale: float) -> bool:
    """Whether a frame's brightest pixel is light rather than read noise.

    The noise floor is read off the frame itself: ``median + k * sigma`` with
    ``sigma = 1.4826 * MAD`` (a robust scale the beam cannot inflate, because
    the beam occupies a tiny fraction of the frame).  A frame whose MAD is
    exactly zero cannot resolve its own spread, so one count of quantisation
    is used instead.

    Args:
        image: One raw integer CCD frame.
        peak: The frame's brightest count.
        full_scale: The detector's full-scale count.

    Returns:
        True when the peak is light, False when it is noise.
    """
    if peak <= 0.0:
        return False
    if peak >= full_scale:
        return True  # clipped: the sensor saw at least full scale
    data = np.asarray(image, dtype=np.float64)
    median = float(np.median(data))
    if peak == median:
        # A flat frame carries no noise distribution: there is nothing for the
        # peak to be an outlier of, so its level is the signal — or a DC
        # pedestal.  A pedestal drags the per-rung ``min`` down (the safe
        # direction) but the *fitted* intercept absorbs it, so a pedestal
        # ladder looks like a shallow slope instead of a bright one; that is
        # why the fit additionally requires a bright top rung (see
        # ``_FIT_MIN_TOP_PEAK_FRACTION``).
        return True
    mad = float(np.median(np.abs(data - median)))
    sigma = 1.4826 * mad
    if sigma <= 0.0:
        sigma = _NOISE_FLOOR_FALLBACK_SIGMA
    return peak > median + NOISE_FLOOR_SIGMAS * sigma


def estimate_exposure_ceiling_ms(
    ladder: Sequence[tuple[float, np.ndarray]],
    *,
    saturation_fraction: float = 0.9,
    headroom: float = 0.9,
) -> ExposureCeilingEstimate:
    """Estimate a safe exposure ceiling from ``(exposure_ms, raw frame)`` pairs.

    The sensor response is linear below clipping, so the rungs that carry light
    fix a straight line ``peak_counts = a + k * ms``.  Solving it for
    ``peak = saturation_fraction * full_scale`` gives the exposure at which the
    brightest pixel reaches that level; the result is clamped to a measured
    upper bound (below) and discounted by ``headroom``.  It may exceed the last
    tested step when that step remains dim, allowing the same probe to work
    across optical paths.

    The optical path sets where saturation falls, which is why this is
    estimated from the image instead of stored as a constant.

    Rung selection
    --------------
    Taking the smallest per-rung estimate is *not* the same as taking the most
    conservative sensor fact: it selects whichever rung has the worst
    ``peak / ms`` ratio, and that is the dim rung, whose ratio read noise
    inflates most.  One measured probe (Daheng MER2, experiment system,
    2026-10-24, recorded in
    ``data/debug/bench_settings/exposure_probe/20261024_110551.json``) shows
    it::

        ms          0.02   0.04   0.08   0.16   0.32   0.64
        peak/255    0.0510 0.0824 0.1490 0.2902 0.5569 1.0 (clipped)
        peak counts 13     21     38     74     142    255
        estimate    0.353  0.437  0.483  0.496  0.517  0.576   ms

    The three brightest unsaturated rungs agree on ~0.52 ms, but the 0.02 ms
    rung reports 0.353 ms — its 13 counts are read-noise dominated (true
    signal ~8.5 counts) — so ``min`` returned a ceiling of 0.3178 ms, ~1.5x
    below what the rungs that actually carry light say.

    The ceiling is therefore fitted over the rungs that can inform it: they
    must carry signal, must not be clipped (``peak_fraction <
    saturation_fraction``), and must reach ``_FIT_MIN_PEAK_FRACTION`` of full
    scale.  One probe delivers one frame dtype, so every rung shares a full
    scale (the first qualifying rung's is used).

    A fit needs a slope the data actually constrains, so it is only accepted
    when **at least three** rungs qualify (two points leave no residual to
    reject a bad line) and the **brightest** qualifying rung reaches
    ``_FIT_MIN_TOP_PEAK_FRACTION`` of full scale.  The second gate is what
    stops a flat DC pedestal — whose per-rung levels barely change with
    exposure, so the fitted slope is pedestal noise — from extrapolating to a
    ceiling far above the truth; both gates and their measured
    counter-examples are documented on the constants.  Anything less keeps the
    old per-rung minimum, which is all such a ladder can support.

    The gate is calibrated for 8-bit frames.  A uint16 frame from a <=10-bit
    sensor never reaches 10 % of 65535 on any rung, so the fit never engages
    and the conservative ``min`` path is used — intended safe behaviour, not a
    defect to work around here.

    Clamping
    --------
    The first step at or above ``saturation_fraction`` bounds the ceiling, and
    a step whose frame actually reached full scale bounds it again, more
    tightly: ``min(first_saturated_ms, first_clipped_ms *
    saturation_fraction)``.  The second term is valid because clipping means
    the true peak there is *at least* full scale, so the
    ``saturation_fraction`` level was already reached at
    ``ms * saturation_fraction``; with a non-negative fitted intercept that
    bound is exact for the fitted line, and with a negative intercept it is
    merely too tight — still the safe direction.  Whatever the fit
    extrapolates to, this measured bound is never exceeded.

    Noise floor
    -----------
    A frame only carries signal when its peak stands clearly above the
    detector's read noise; the extrapolation above assumes the peak *is*
    light.  The bottom of a doubling ladder sits at the sensor minimum
    exposure, where the frame is pure noise — measured on the Daheng MER2 in
    the experiment system on 2026-10-21, the 0.02 ms frame peaked at **2
    counts** on a zero background (3.53 M pixels at 0, 1.51 M at 1, three at
    2), i.e. read noise, while the bench itself saturates near 100 ms.
    Extrapolating from that frame gives ``0.02 * 0.9 / (2/255) = 2.295 ms``
    and a ceiling of **2.0655 ms** — the probe was low by ~30x, and the
    hardware flow then refused legal exposures.

    Such frames are therefore excluded from the extrapolation: a frame counts
    as signal only when its peak exceeds ``median + NOISE_FLOOR_SIGMAS *
    sigma`` of its own noise.  ``sigma`` is the robust scale ``1.4826 * MAD``
    (the beam occupies a tiny fraction of the frame, so it cannot inflate
    it); the measured frames have a MAD of exactly 0 (their mass sits on 0 and
    1), so one count of quantisation is used instead — the smallest spread a
    digital frame can express, and the right scale for a *peak* statistic,
    which sits several sigma above the bulk of a multi-megapixel noise field.

    ``NOISE_FLOOR_SIGMAS = 4`` puts the threshold in the middle of the
    measured gap: the noise frame tops out at 2 counts (so the threshold,
    0 + 4 counts, rejects it with a 2-count margin) while the first ladder
    frame that carries real light — 2.56 ms, 7 counts — clears it with a
    3-count margin.  Frames that clip the sensor always carry signal.

    A ladder without a single signal frame still fails, with the same
    actionable error as before.
    """
    pairs: list[tuple[float, np.ndarray]] = []
    for exposure, frame in ladder:
        ms = float(exposure)
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"ladder exposure must be positive, got {exposure!r}")
        image = np.asarray(frame)
        if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
            raise ValueError("ladder frames must be raw integer CCD frames")
        pairs.append((ms, image))
    if not pairs:
        raise ValueError("ladder needs at least one (exposure, frame) pair")
    if not 0 < saturation_fraction < 1:
        raise ValueError(
            f"saturation_fraction must lie in (0, 1), got {saturation_fraction}"
        )
    if not 0 < headroom <= 1:
        raise ValueError(f"headroom must lie in (0, 1], got {headroom}")

    pairs.sort(key=lambda pair: pair[0])
    fractions: list[float] = []
    estimates: list[float] = []
    estimate_exposures: list[float] = []
    fit_ms_values: list[float] = []
    fit_peaks: list[float] = []
    fit_full_scale = 0.0
    last_safe = 0.0
    first_saturated: float | None = None
    first_clipped: float | None = None
    for ms, image in pairs:
        full_scale = float(np.iinfo(image.dtype).max)
        peak = float(np.max(image))
        peak_fraction = peak / full_scale
        fractions.append(peak_fraction)
        # A clipped frame is signal by definition; otherwise the peak has to
        # clear the frame's own noise floor before it can be extrapolated.
        carries = _carries_signal(image, peak, full_scale)
        if carries:
            estimates.append(ms * saturation_fraction / peak_fraction)
            estimate_exposures.append(ms)
        if peak_fraction < saturation_fraction:
            last_safe = ms
        elif first_saturated is None:
            first_saturated = ms
        if peak >= full_scale:
            # Censored observation: the true peak is at least full scale.
            if first_clipped is None:
                first_clipped = ms
        elif (
            carries
            and _FIT_MIN_PEAK_FRACTION <= peak_fraction < saturation_fraction
        ):
            if not fit_peaks:
                fit_full_scale = full_scale
            fit_ms_values.append(ms)
            fit_peaks.append(peak)
    if not estimates:
        raise ValueError("no ladder frame carries signal; cannot estimate a ceiling")
    if last_safe <= 0:
        raise ValueError("CCD is saturated at the minimum probe exposure")

    fit_ms: float | None = None
    fit_slope: float | None = None
    fit_intercept: float | None = None
    # Both guards are required: a two-point line has no residual to reject it,
    # and a ladder whose brightest rung is still dim fits its own pedestal
    # rather than the light (see _FIT_MIN_TOP_PEAK_FRACTION).
    top_peak_fraction = (
        max(fit_peaks) / fit_full_scale if fit_peaks and fit_full_scale > 0 else 0.0
    )
    if len(fit_peaks) >= 3 and top_peak_fraction >= _FIT_MIN_TOP_PEAK_FRACTION:
        x = np.asarray(fit_ms_values, dtype=np.float64)
        y = np.asarray(fit_peaks, dtype=np.float64)
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        sxx = float(np.sum((x - x_mean) ** 2))
        sxy = float(np.sum((x - x_mean) * (y - y_mean)))
        if sxx > 0.0:
            slope = sxy / sxx
            intercept = y_mean - slope * x_mean
            if math.isfinite(slope) and slope > 0.0 and math.isfinite(intercept):
                target = saturation_fraction * fit_full_scale
                candidate = (target - intercept) / slope
                if math.isfinite(candidate) and candidate > 0.0:
                    fit_ms = candidate
                    fit_slope = slope
                    fit_intercept = intercept

    bound_ms: float | None = None
    if fit_ms is None:
        # No usable fit (too dim, pedestal-dominated, or too flat): the
        # per-rung minimum is all this ladder can support, and it is the safe
        # direction.
        ceiling_ms = min(estimates) * headroom
    else:
        # The first saturated step bounds the ceiling, and the first *clipped*
        # step bounds it again: clipping means the true peak there is at least
        # full scale, so the saturation level was already reached at
        # ``ms * saturation_fraction``.
        bound = math.inf if first_saturated is None else first_saturated
        if first_clipped is not None:
            bound = min(bound, first_clipped * saturation_fraction)
        bound_ms = bound if math.isfinite(bound) else None
        ceiling_ms = min(fit_ms, bound) * headroom
    return ExposureCeilingEstimate(
        ceiling_ms=ceiling_ms,
        last_safe_ms=last_safe,
        saturated_ms=first_saturated,
        peak_fractions=tuple(fractions),
        estimates_ms=tuple(estimates),
        estimate_exposures_ms=tuple(estimate_exposures),
        fit_ms=fit_ms,
        fit_slope=fit_slope,
        fit_intercept=fit_intercept,
        fit_points=len(fit_peaks),
        first_clipped_ms=first_clipped,
        bound_ms=bound_ms,
    )


def probe_exposure_ceiling_ms(
    set_exposure_ms: Callable[[float], float],
    shoot: Callable[[], np.ndarray],
    *,
    start_ms: float = 0.02,
    factor: float = 2.0,
    max_steps: int = 8,
    hard_max_ms: float | None = None,
    saturation_fraction: float = 0.9,
    headroom: float = 0.9,
) -> ExposureCeilingEstimate:
    """Shoot a doubling exposure ladder until the frame saturates.

    ``set_exposure_ms`` writes the requested exposure and returns the value
    the camera actually applied (drivers clamp and quantise); ``shoot``
    returns one raw integer frame at the current exposure.  The ladder starts
    at ``start_ms`` (the sensor minimum), doubles each step, and stops at the
    first frame whose peak reaches ``saturation_fraction`` of full scale or at
    ``hard_max_ms`` (the camera's hardware maximum) — further steps would only
    waste light on a sensor that is already clipped.

    Stale frames
    ------------
    Changing the exposure does not change the picture *immediately*: the
    camera's exposure pipeline still delivers one or two frames that were
    exposed with the **previous** setting (measured on the Daheng MER2 on
    2026-10-21: after changing the exposure the next one to two frames still
    showed the old exposure, e.g. ``set 0.64 ms`` answered with the 20 ms
    frame).  A ladder that pairs such a frame with the *new* exposure is off
    by one step, so every extrapolation it takes part in is biased.

    The stale frames are consumed **by the driver layer**: the Daheng
    backend's ``reset_exposure_time`` drops ``EXPOSURE_CHANGE_DISCARD_FRAMES``
    frames whenever the exposure really changes, so the first frame the
    caller gets already carries the new exposure.  This probe therefore does
    *not* discard frames of its own: discarding again would cost an extra
    frame per ladder step on every backend and be a no-op on the ones that
    already do it.

    The noise floor of the bottom steps is a second, independent trap, and it
    is what actually drove the measured ceiling down to 2.0655 ms: the frames
    at the sensor minimum exposure are pure read noise (2 counts on the
    Daheng MER2), and extrapolating from them lands far below the truth.  It
    is handled in :func:`estimate_exposure_ceiling_ms`.

    The ladder's light state is *not* restored: the caller owns the camera and
    must re-arm the working exposure afterwards.

    When ``hard_max_ms`` is given, the returned ceiling is capped at it: a
    ladder that reaches the camera maximum without saturating has no measured
    saturation point, and an exposure above the maximum is meaningless.  The
    ``headroom`` discount still applies to the cap, so a capped ceiling is
    ``hard_max_ms * headroom`` rather than the maximum itself.  The walk is
    unchanged — the ladder already stops at the cap.
    """
    if not math.isfinite(start_ms) or start_ms <= 0:
        raise ValueError(f"start_ms must be positive, got {start_ms}")
    if factor <= 1:
        raise ValueError(f"factor must exceed 1, got {factor}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if hard_max_ms is None:
        cap = math.inf
    else:
        cap = float(hard_max_ms)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError(f"hard_max_ms must be positive, got {hard_max_ms}")

    ladder: list[tuple[float, np.ndarray]] = []
    ms = start_ms
    for _ in range(max_steps):
        step_ms = min(ms, cap)
        actual_ms = float(set_exposure_ms(step_ms))
        frame = np.asarray(shoot())
        if frame.ndim != 2 or not np.issubdtype(frame.dtype, np.integer):
            raise ValueError("probe needs raw integer CCD frames")
        ladder.append((actual_ms, frame))
        full_scale = float(np.iinfo(frame.dtype).max)
        if float(np.max(frame)) / full_scale >= saturation_fraction or actual_ms >= cap:
            break
        ms *= factor
    estimate = estimate_exposure_ceiling_ms(
        ladder, saturation_fraction=saturation_fraction, headroom=headroom
    )
    if hard_max_ms is not None:
        # Nothing measured can say where a bench saturates beyond the camera
        # maximum, and an exposure above it is not settable anyway.  Cap the
        # pre-headroom estimate so the safety discount still applies to the cap.
        cap_ceiling = cap * headroom
        if estimate.ceiling_ms > cap_ceiling:
            estimate = replace(estimate, ceiling_ms=cap_ceiling, bound_ms=cap)
    return estimate


def working_exposure_ms(
    ceiling_ms: float,
    *,
    fraction: float = 0.5,
    floor_ms: float = 0.02,
) -> float:
    """The exposure a run should actually use, given a resolved ceiling.

    Single source of truth for "probe said X, so use Y": the panel-locate
    sweep, the geometry sweep and the GUI sidebar probe must all agree, or the
    evidence a run explains was taken at a different exposure than the run
    used.  ``fraction`` defaults to a half of the ceiling -- the ceiling is
    where the brightest pixel would reach 90% of full scale, discounted by the
    estimator's headroom (0.9), so half of it puts the peak near 40%: enough
    raw counts for the shaping metrics, with the clip margin the report asks
    for.  ``floor_ms`` is the driver's own clamp,
    so a very bright bench still lands on a legal exposure.
    """
    if not math.isfinite(ceiling_ms) or ceiling_ms <= 0:
        raise ValueError(f"ceiling_ms must be finite and positive, got {ceiling_ms!r}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    if not math.isfinite(floor_ms) or floor_ms < 0:
        raise ValueError(f"floor_ms must be finite and non-negative, got {floor_ms!r}")
    return max(float(floor_ms), float(fraction) * float(ceiling_ms))


@dataclass(frozen=True)
class ExposureDecision:
    next_ms: float
    accepted: bool
    peak_fraction: float


def choose_exposure(
    frame: np.ndarray,
    exposure_ms: float,
    *,
    max_ms: float,
    target_fraction: float = 0.55,
    saturation_fraction: float = 0.9,
) -> ExposureDecision:
    """Choose a bounded next exposure from an unprocessed integer CCD frame.

    A dim frame at the hardware exposure limit is accepted; a saturated frame
    at the camera's minimum exposure is rejected by the caller after retrying.
    """
    image = np.asarray(frame)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError("auto exposure needs one raw integer CCD frame")
    if exposure_ms <= 0 or max_ms <= 0 or exposure_ms > max_ms + 1e-9:
        raise ValueError("exposure must be positive and no greater than max_ms")
    if not 0 < target_fraction < saturation_fraction < 1:
        raise ValueError("require 0 < target_fraction < saturation_fraction < 1")
    full_scale = float(np.iinfo(image.dtype).max)
    peak_fraction = float(np.max(image)) / full_scale
    if peak_fraction >= saturation_fraction:
        next_ms = max(0.001, exposure_ms * target_fraction / max(peak_fraction, 1e-9))
        return ExposureDecision(min(next_ms, max_ms), False, peak_fraction)
    if peak_fraction < target_fraction * 0.7 and exposure_ms < max_ms:
        next_ms = min(max_ms, exposure_ms * target_fraction / max(peak_fraction, 0.01))
        return ExposureDecision(next_ms, False, peak_fraction)
    return ExposureDecision(exposure_ms, True, peak_fraction)


def exposure_normalized_frame(
    frame: np.ndarray, exposure_ms: float, *, reference_ms: float = 0.1
) -> np.ndarray:
    """Convert raw counts to counts at a fixed reference exposure.

    Background subtraction remains with the caller, after this linear scaling.
    Camera gain must stay fixed; nonlinear or clipped frames must be rejected.
    """
    if exposure_ms <= 0 or reference_ms <= 0:
        raise ValueError("exposure and reference must be positive")
    return np.asarray(frame, dtype=np.float64) * (reference_ms / exposure_ms)
