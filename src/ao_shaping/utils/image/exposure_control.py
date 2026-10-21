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
from dataclasses import dataclass
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
    """

    ceiling_ms: float
    last_safe_ms: float
    saturated_ms: float | None
    peak_fractions: tuple[float, ...]


#: How far above a frame's own background its peak has to sit before the peak
#: counts as light instead of detector read noise.  See
#: :func:`estimate_exposure_ceiling_ms` for the measurement that fixes it.
NOISE_FLOOR_SIGMAS: float = 4.0

#: Noise scale assumed when a frame's MAD is exactly zero (more than half of
#: the frame sits on the median, so the MAD cannot resolve the spread).  One
#: count is the detector's quantisation step, i.e. the smallest spread a
#: digital frame can express.
_NOISE_FLOOR_FALLBACK_SIGMA: float = 1.0


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
        # peak to be an outlier of, so its level is the signal (or a DC
        # pedestal, which only drags the ceiling down — the safe direction).
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

    The sensor response is linear below clipping, so a frame whose peak is
    ``p`` of full scale at exposure ``ms`` saturates (peak = ``saturation_fraction``)
    at ``ms * saturation_fraction / p``.  The smallest estimate is discounted
    by ``headroom``. It may exceed the last tested step when that step remains
    dim, allowing the same probe to work across optical paths.

    The optical path sets where saturation falls, which is why this is
    estimated from the image instead of stored as a constant.

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
    last_safe = 0.0
    first_saturated: float | None = None
    for ms, image in pairs:
        full_scale = float(np.iinfo(image.dtype).max)
        peak = float(np.max(image))
        peak_fraction = peak / full_scale
        fractions.append(peak_fraction)
        # A clipped frame is signal by definition; otherwise the peak has to
        # clear the frame's own noise floor before it can be extrapolated.
        if _carries_signal(image, peak, full_scale):
            estimates.append(ms * saturation_fraction / peak_fraction)
        if peak_fraction < saturation_fraction:
            last_safe = ms
        elif first_saturated is None:
            first_saturated = ms
    if not estimates:
        raise ValueError("no ladder frame carries signal; cannot estimate a ceiling")
    if last_safe <= 0:
        raise ValueError("CCD is saturated at the minimum probe exposure")
    return ExposureCeilingEstimate(
        ceiling_ms=min(estimates) * headroom,
        last_safe_ms=last_safe,
        saturated_ms=first_saturated,
        peak_fractions=tuple(fractions),
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
    return estimate_exposure_ceiling_ms(
        ladder, saturation_fraction=saturation_fraction, headroom=headroom
    )


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
    where the brightest pixel would reach 90% of full scale, so half of it puts
    the peak near 45%: enough raw counts for the shaping metrics, with the
    clip margin the report asks for.  ``floor_ms`` is the driver's own clamp,
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
