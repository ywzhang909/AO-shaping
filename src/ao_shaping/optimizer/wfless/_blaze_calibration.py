"""Measured off-axis carrier selection for SLM shaping sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.ndimage import label

from ao_shaping.utils.wavefront.zero_order_overlay import (
    OverlayParams,
    ZeroOrderOverlay,
    apply_zero_order_overlay,
    predicted_shift_px,
)


@dataclass(frozen=True)
class BlazeChoice:
    params: OverlayParams
    zero_xy: tuple[int, int]
    signal_xy: tuple[int, int]
    separation_px: float
    signal_peak: float
    trials: tuple[dict[str, object], ...] = ()


def select_blaze(
    capture: Callable[[np.ndarray], np.ndarray],
    panel_shape: tuple[int, int],
    *,
    target_side: int,
    window_size: int,
    period_px: int = 0,
    direction: str = "vertical",
    flat_frame: np.ndarray | None = None,
) -> BlazeChoice:
    """Probe pure gratings on the full CCD and choose a visible separated order.

    ``period_px=0`` searches a conservative integer-period ladder. A positive
    period tests only that period and the requested direction. The callback must
    acquire at a checked, unsaturated exposure and must return a full CCD frame.
    No predicted sign is trusted: the signal position is measured in each frame.
    """
    if target_side < 1 or window_size < target_side:
        raise ValueError("target_side must fit in the requested CCD window")
    if period_px < 0 or (0 < period_px < 4):
        raise ValueError("blaze period must be 0 (auto) or at least 4 panel pixels")
    if direction not in ("vertical", "horizontal"):
        raise ValueError("blaze direction must be vertical or horizontal")
    blank = np.zeros(panel_shape, dtype=np.float64)
    flat = np.asarray(capture(blank) if flat_frame is None else flat_frame, dtype=float)
    if flat.ndim != 2 or not np.isfinite(flat).all() or np.max(flat) <= 0:
        raise ValueError("flat CCD frame is empty or invalid")
    zy, zx = np.unravel_index(int(np.argmax(flat)), flat.shape)
    zero_xy = (int(zx), int(zy))
    zero_peak = float(flat[zy, zx])
    # Use the connected half-peak core containing the measured optical axis;
    # an arbitrary geometric camera centre or guessed zero radius is unsafe.
    components, _ = label(flat >= 0.5 * zero_peak)
    core = components == components[zy, zx]
    core_y, core_x = np.nonzero(core)
    zero_radius = max(8.0, float(np.max(np.hypot(core_x - zx, core_y - zy))) + 2.0)
    min_separation = zero_radius + target_side / 2 + 4
    periods = (int(period_px),) if period_px else (192, 160, 128, 112, 96, 80, 64)
    directions = (direction,) if period_px else ("vertical", "horizontal")
    yy, xx = np.indices(flat.shape)
    choices: list[BlazeChoice] = []
    trials: list[dict[str, object]] = []
    for candidate_direction in directions:
        for period in periods:
            params = OverlayParams(
                mode=ZeroOrderOverlay.BLAZE,
                blaze_period_px=period,
                blaze_direction=candidate_direction,
            )
            shift = float(np.hypot(*predicted_shift_px(params)))
            if shift < min_separation * 0.7:
                trials.append({"period_px": period, "direction": candidate_direction,
                               "predicted_shift_px": shift, "accepted": False,
                               "reason": "predicted_overlap"})
                continue
            frame = np.asarray(capture(apply_zero_order_overlay(blank, params)), dtype=float)
            if frame.shape != flat.shape or not np.isfinite(frame).all():
                raise ValueError("blaze probe changed CCD geometry or has invalid pixels")
            # The signed camera mapping is not calibrated. Search both sides of
            # the measured zero along the known rotated axis at the expected range.
            axis_offset = yy - zy if candidate_direction == "vertical" else xx - zx
            cross_offset = xx - zx if candidate_direction == "vertical" else yy - zy
            allowed = (
                (np.abs(axis_offset) >= max(min_separation, shift * 0.6))
                & (np.abs(axis_offset) <= shift * 1.5 + 8)
                & (np.abs(cross_offset) <= max(16, target_side / 2))
            )
            if not allowed.any():
                trials.append({"period_px": period, "direction": candidate_direction,
                               "predicted_shift_px": shift, "accepted": False,
                               "reason": "outside_sensor"})
                continue
            probe = np.where(allowed, frame, -np.inf)
            sy, sx = np.unravel_index(int(np.argmax(probe)), frame.shape)
            peak = float(frame[sy, sx])
            separation = float(np.hypot(sx - zx, sy - zy))
            half = max(target_side / 2 + 4, window_size / 2)
            accepted = not (
                peak < max(5.0, zero_peak * 0.05, 1.5 * float(flat[sy, sx]))
                or separation < min_separation
                or sx - half < 0 or sy - half < 0
                or sx + half >= frame.shape[1] or sy + half >= frame.shape[0]
            )
            trials.append({
                "period_px": period, "direction": candidate_direction,
                "predicted_shift_px": shift, "signal_xy": (int(sx), int(sy)),
                "separation_px": separation, "signal_peak": peak,
                "accepted": accepted,
            })
            if not accepted:
                continue
            choices.append(BlazeChoice(params, zero_xy, (int(sx), int(sy)), separation, peak))
    if not choices:
        raise RuntimeError("no measured blaze order fits the CCD window and clears zero order")
    # Keep candidates with at least half the best measured signal, then prefer
    # the least deflection. A barely detectable order should not win solely
    # because it moved one pixel less than a bright, usable order.
    bright_floor = 0.5 * max(item.signal_peak for item in choices)
    bright_choices = [item for item in choices if item.signal_peak >= bright_floor]
    return replace(
        min(bright_choices, key=lambda item: (item.separation_px, -item.signal_peak)),
        trials=tuple(trials),
    )
