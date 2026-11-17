"""ROI window geometry helpers for the target-shape page.

Verbatim extraction from ``target_shape_helper`` (module-size ratchet
split); ``target_shape_helper`` re-exports the names for existing importers.
"""

from __future__ import annotations

from typing import Any


def _effective_window(
    requested_size: tuple[int, int],
    requested_center: tuple[float, float],
    applied: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Resolve which window geometry to trust after ``reset_window``.

    ``DahengCamera.reset_window`` clamps the request onto the GenICam grid and
    reads Width/Height/OffsetX/OffsetY back from the device, returning the
    EFFECTIVE ``((w, h), (cx, cy))`` — requests outside the grid are silently
    preserved by the SDK, so the request is not necessarily what got applied.
    Backends without readback return ``None``; then the requested geometry is
    kept (previous behavior).
    """
    if applied is None:
        return (int(requested_size[0]), int(requested_size[1])), (
            float(requested_center[0]),
            float(requested_center[1]),
        )
    (w, h), (cx, cy) = applied
    return (int(w), int(h)), (float(cx), float(cy))


def _store_effective_window(
    session_state: Any,
    requested_size: int,
    requested_center: tuple[float, float],
    applied: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Persist the effective ROI window into the session geometry.

    Writes ``ts_roi_window`` (effective ``((w, h), (cx, cy))``) and syncs the
    square ``ts_window_size`` to the effective size so downstream crop/window
    consumers use the geometry that actually describes the frames. Returns the
    effective ``(size, center)`` for status text. ``applied is None`` (backend
    without readback) stores the requested geometry unchanged.
    """
    (eff_w, eff_h), (eff_cx, eff_cy) = _effective_window(
        (requested_size, requested_size), requested_center, applied
    )
    session_state["ts_window_size"] = min(eff_w, eff_h)
    session_state["ts_roi_window"] = ((eff_w, eff_h), (eff_cx, eff_cy))
    return (eff_w, eff_h), (eff_cx, eff_cy)
