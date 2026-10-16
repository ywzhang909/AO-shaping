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
        estimate = probe_exposure_ceiling_ms(
            camera.reset_exposure_time,
            lambda: np.asarray(camera.get_numpy_image(n_sample=2)),
            start_ms=probe_start_ms,
            hard_max_ms=probe_hard_max_ms,
        )
        limit = estimate.ceiling_ms
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


def estimate_exposure_ceiling_ms(
    ladder: Sequence[tuple[float, np.ndarray]],
    *,
    saturation_fraction: float = 0.9,
    headroom: float = 0.9,
) -> ExposureCeilingEstimate:
    """Estimate a safe exposure ceiling from ``(exposure_ms, raw frame)`` pairs.

    The sensor response is linear below clipping, so a frame whose peak is
    ``p`` of full scale at exposure ``ms`` saturates (peak = ``saturation_fraction``)
    at ``ms * saturation_fraction / p``.  The ceiling is the **minimum** of
    these per-frame estimates: a clipped frame can only pull its estimate
    *down* (its true peak is higher than the clipped one), which is the safe
    direction.  The result is discounted by ``headroom`` so the ceiling stays
    below the estimated saturation point, not on it.

    The optical path sets where saturation falls, which is why this is
    estimated from the image instead of stored as a constant.  Zero-peak
    frames carry no signal and are skipped, but a ladder without a single
    signal frame fails.
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
        peak_fraction = float(np.max(image)) / full_scale
        fractions.append(peak_fraction)
        if peak_fraction > 0:
            estimates.append(ms * saturation_fraction / peak_fraction)
        if peak_fraction < saturation_fraction:
            last_safe = ms
        elif first_saturated is None:
            first_saturated = ms
    if not estimates:
        raise ValueError("no ladder frame carries signal; cannot estimate a ceiling")
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
