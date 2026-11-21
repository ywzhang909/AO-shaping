"""Absolute-intensity strength gate (offline), per the frozen 2026-11-19 spec.

The frozen criterion lives in
``docs/dev/transfer_upstream_qualification_plan_20261119.md`` §2 (a priori,
frozen before any candidate was scored): held-out rows, counts/ms both sides,
whole sampling window minus the zero-order disk, a real dark frame subtracted
from *both* signals, and ``relL2 <= 0.20`` on every independent pass.

This module holds only the numeric primitives; the runner is
``scripts/slm/evaluate_strength_gate_20261119.py``. Nothing here imports a
camera, an SLM or a laser.

Units: both arrays handed to :func:`masked_absolute_rel_l2` are counts per
millisecond (``radiometry.count_rate``), which is what ``fit.build_fit_model``
stores in ``comps.images`` and what ``fit.fit_loss``'s prediction path returns.
Dividing both norms by the same exposure makes a row's ``relL2`` independent of
its exposure.

Zero-order radius: ``zero_radius_px`` is expressed in **window pixels** (one
sampling cell, i.e. ``window_step`` sensor pixels), as the frozen plan's §2.2
declares; the equivalent full-frame radius is ``zero_radius_px * window_step``
and is what the runner records in its evidence JSON.
"""
from __future__ import annotations

import numpy as np

#: Frozen gate threshold (pre-registration 2026-11-19, δ = 0.20). Read-only:
#: the plan §2.4 fixes it before any evaluation; do not tune it after seeing
#: a result.
STRENGTH_GATE_DELTA = 0.20


def match_dark_exposure(darks, exposure_ms, *, rel_tol: float = 0.0) -> float | None:
    """Return the mapping key whose exposure equals ``exposure_ms``, else None.

    ``darks`` is a mapping {float exposure_ms: anything}. Matching is strict by
    default (abs(a-b) <= 1e-6) because the dark must be captured AT the probe
    exposure; ``rel_tol`` only offers a documented relaxation for float32
    round-trip noise. A non-positive/NaN probe exposure returns None.

    The 1e-6 absolute floor is kept even with ``rel_tol > 0`` so that a float32
    round-trip (``0.051f`` -> ``0.050999999046325684``) always matches without
    permitting an approximate jump across exposure levels. When several keys
    fall inside the tolerance the closest one wins, which keeps the result
    deterministic for overlapping entries. The **key** is returned, not the
    value, so callers can look up the dark entry without a second scan.
    """
    tolerance = float(rel_tol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"non-negative finite rel_tol required, got {rel_tol!r}")
    probe = float(exposure_ms)
    if not np.isfinite(probe) or probe <= 0:
        return None

    best_key = None
    best_delta = float("inf")
    for key in darks:
        candidate = float(key)
        if not np.isfinite(candidate) or candidate <= 0:
            continue
        delta = abs(candidate - probe)
        allowed = max(1e-6, tolerance * max(abs(candidate), abs(probe)))
        if delta <= allowed and delta < best_delta:
            best_key, best_delta = key, delta
    return best_key


def window_zero_mask(shape, *, window_step, zero_radius_px, reference_xy=None):
    """Boolean keep-mask on the window-sampled grid (True = keep this cell).

    The window grid is the one ``fit.build_fit_model`` builds: its
    ``np.mgrid[-span:span+1:2]`` gives ``span+1`` samples per axis, read from
    the CCD ``window_step`` sensor pixels apart (``161x161`` at ``span=160``,
    the archived refits' setting). The mask is the whole window minus the
    zero-order disk.

    Parameters
    ----------
    shape
        Window grid shape ``(rows, cols)``.
    window_step
        Sensor pixels per grid sample (``2`` for the recorded campaigns). Used
        to convert the radius to full-frame sensor pixels; the mask itself is
        evaluated in grid cells.
    zero_radius_px
        Exclusion radius in **window pixels** (grid cells). ``0`` disables the
        exclusion and returns an all-True mask, which is the frozen default of
        the 2026-11-19 session (``zero_radius = 0``).
    reference_xy
        Disk centre in window pixels ``(x, y)``. Default: the grid centre,
        ``((cols-1)/2, (rows-1)/2)``, which is ``reference_start`` because the
        window is centred on that spot.

    Returns
    -------
    numpy.ndarray of bool
        Same shape as ``shape``; True where the cell is kept.
    """
    rows, cols = (int(value) for value in shape)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"positive window shape required, got {shape!r}")
    step = float(window_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"positive finite window_step required, got {window_step!r}")
    radius = float(zero_radius_px)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError(f"non-negative finite zero_radius_px required, got {zero_radius_px!r}")
    keep = np.ones((rows, cols), dtype=bool)
    if radius <= 0:
        return keep
    if reference_xy is None:
        centre = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0], dtype=float)
    else:
        centre = np.asarray(reference_xy, dtype=float).reshape(2)
        if not np.isfinite(centre).all():
            raise ValueError(f"finite reference_xy required, got {reference_xy!r}")
    yy, xx = np.mgrid[:rows, :cols]
    # Distance in full-frame sensor pixels, so the radius crosses the grid
    # through window_step exactly as the frozen plan declares.
    distance_full_px = np.hypot(xx - centre[0], yy - centre[1]) * step
    keep &= distance_full_px > radius * step
    return keep


def masked_absolute_rel_l2(pred, meas, mask, *, dc_pred=0.0, dc_meas=0.0) -> dict:
    """Absolute (not shape-only) relL2 inside ``mask``, both sides in counts/ms.

    ``rel_l2 = ||(pred - dc_pred) - (meas - dc_meas)||_mask
                / max(||meas - dc_meas||_mask, 1e-9)``

    This is the plan's §2.4 metric: the numerator is an absolute intensity
    difference, not a scale-normalised shape error, so it fails a model that is
    right in shape but wrong in level. ``dc_pred`` / ``dc_meas`` carry the dark
    subtraction (§2.3): the measured side gets the measured dark rate, the
    prediction side gets the model's electronic bias ``bias_counts/exposure``
    while its optical background term stays.

    An empty mask gives ``rel_l2 = 0.0`` (the literal ``max(..., 1e-9)``
    branch) together with ``n_pixels = 0``; callers must treat ``n_pixels == 0``
    as *not assessable* rather than as a pass.

    Returns
    -------
    dict
        ``{"rel_l2": float, "energy_meas": float, "n_pixels": int}`` where
        ``energy_meas`` is ``||meas - dc_meas||_mask``.
    """
    pred = np.asarray(pred, dtype=float)
    meas = np.asarray(meas, dtype=float)
    keep = np.asarray(mask, dtype=bool)
    if pred.shape != meas.shape:
        raise ValueError(f"pred and meas shapes differ: {pred.shape} vs {meas.shape}")
    if keep.shape != pred.shape:
        raise ValueError(f"mask shape {keep.shape} differs from data shape {pred.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(meas).all():
        raise ValueError("non-finite pred or meas: the gate inputs must be counts/ms")

    residual = (pred - dc_pred) - (meas - dc_meas)
    numerator = float(np.linalg.norm(residual[keep]))
    denominator = float(np.linalg.norm((meas - dc_meas)[keep]))
    return {
        "rel_l2": numerator / max(denominator, 1e-9),
        "energy_meas": denominator,
        "n_pixels": int(keep.sum()),
    }


def gate_rel_l2(pred, meas, mask, *, bias_rate=0.0, dark_rate=0.0) -> dict:
    """The strength gate: masked absolute relL2 with both DC terms already in counts/ms.

    ``pred`` and ``meas`` are counts/ms. ``bias_rate`` is the model's electronic
    offset ``bias_counts / exposure`` (counts/ms) and ``dark_rate`` is the
    measured dark frame at the probe exposure (counts/ms). Both are subtracted
    directly, so **no exposure factor appears here**: a caller holding raw dark
    counts must divide them by the probe exposure to counts/ms *before* calling,
    never multiply ``dark_rate`` by an exposure. The archived refit's wrong-sign
    temptation (``meas - dark_rate * exposure``) is exactly what this pins.

    This is the single expression the runner uses, so the unit dimension cannot
    drift between the core and the CLI.
    """
    return masked_absolute_rel_l2(
        np.asarray(pred, dtype=float) - np.asarray(bias_rate, dtype=float),
        np.asarray(meas, dtype=float) - np.asarray(dark_rate, dtype=float),
        mask,
    )
