"""Objective / metric computations on CCD frames.

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).
"""
from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from loguru import logger

from ao_shaping.utils.image.target.patterns import create_target_shape


TARGET_SHAPE_CHOICES = (
    "circle",
    "square",
    "rectangle",
    "annular",
    "grid",
    "cross",
    "gaussian",
    "pentagon",
)


TargetShape = Literal[
    "gaussian",
    "circle",
    "square",
    "annular",
    "grid",
    "cross",
    "rectangle",
    "pentagon",
]


def target_shape_roi(
    image_shape: tuple[int, int],
    center: tuple[float, float],
    shape: str,
    size: float,
    aspect_ratio: float = 4.0 / 3.0,
) -> np.ndarray:
    """Build a target-shaped ROI centred at ``(x, y)`` in an image."""
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must be (height, width), got {image_shape!r}")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape!r}")
    if not np.isfinite(size) or size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio!r}")

    shape = str(shape).lower()
    if shape not in TARGET_SHAPE_CHOICES:
        raise ValueError(f"Unknown target shape: {shape!r}")
    typed_shape = cast(TargetShape, shape)
    if shape == "rectangle":
        template_h = max(1, int(round(float(size))))
        template_w = max(1, int(round(float(size) * aspect_ratio)))
    else:
        template_h = template_w = max(1, int(round(float(size))))

    # The template MUST fit the frame. A 4:3 rectangle is 33% wider than its short
    # side, so clamping only by ``min(image_shape)`` (the old auto-size clamp) let
    # the box run past the window edge and get silently clipped - the metric then
    # scored a *partial* box while ``energy`` was still divided by the window total.
    # Scale down uniformly to fit instead, and say so.
    _fit_scale = min(1.0, height / template_h, width / template_w)
    if _fit_scale < 1.0:
        logger.warning(
            "target ROI {}x{}px does not fit the {}x{} frame - scaled x{:.3f} to "
            "{}x{}px (increase the camera window / cam_size)",
            template_w,
            template_h,
            width,
            height,
            _fit_scale,
            max(1, int(round(template_w * _fit_scale))),
            max(1, int(round(template_h * _fit_scale))),
        )
        template_h = max(1, int(round(template_h * _fit_scale)))
        template_w = max(1, int(round(template_w * _fit_scale)))

    kwargs: dict[str, Any] = {}
    if shape in {"square", "rectangle"}:
        kwargs["side"] = template_h
        kwargs["aspect_ratio"] = aspect_ratio
    elif shape in {"circle", "annular", "gaussian", "pentagon"}:
        kwargs["radius_ratio"] = 1.0
    if shape == "annular":
        kwargs["outer_radius"] = template_h / 2.0

    template = create_target_shape(typed_shape, (template_h, template_w), **kwargs)
    cx, cy = (float(center[0]), float(center[1]))
    x_start = int(np.floor(cx - template_w / 2.0))
    y_start = int(np.floor(cy - template_h / 2.0))
    x_end = x_start + template_w
    y_end = y_start + template_h

    roi = np.zeros((height, width), dtype=bool)
    dst_y0 = max(0, y_start)
    dst_y1 = min(height, y_end)
    dst_x0 = max(0, x_start)
    dst_x1 = min(width, x_end)
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return roi

    src_y0 = dst_y0 - y_start
    src_y1 = dst_y1 - y_start
    src_x0 = dst_x0 - x_start
    src_x1 = dst_x1 - x_start
    roi[dst_y0:dst_y1, dst_x0:dst_x1] = template[src_y0:src_y1, src_x0:src_x1] > 0
    if shape == "gaussian":
        roi[dst_y0:dst_y1, dst_x0:dst_x1] &= (
            template[src_y0:src_y1, src_x0:src_x1] >= 0.01
        )
    return roi


def spot_waist_sigma(
    img: np.ndarray,
    center: tuple[float, float] | None = None,
    threshold_ratio: float = 0.1,
) -> float:
    """Estimate the flat-field spot **waist** radius ``w0`` (second-moment RMS).

    Only pixels above ``threshold_ratio * max`` contribute, so the broad stray
    halo (which makes the 99%-encircled-radius estimate far too large) does not
    inflate the number. For a Gaussian beam the returned RMS radius equals ``w0``;
    a target box of ``2 * w0`` is the waist diameter.
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"img must be 2D, got {frame.ndim}D")
    peak = float(frame.max())
    if not np.isfinite(peak) or peak <= 0.0:
        return 0.0
    mask = frame >= float(threshold_ratio) * peak
    weights = np.where(mask, np.clip(frame, 0.0, None), 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    ys, xs = np.indices(frame.shape, dtype=np.float64)
    if center is None:
        cx = float((weights * xs).sum() / total)
        cy = float((weights * ys).sum() / total)
    else:
        cx, cy = float(center[0]), float(center[1])
    var_x = float((weights * (xs - cx) ** 2).sum() / total)
    var_y = float((weights * (ys - cy) ** 2).sum() / total)
    return float(np.sqrt((max(var_x, 0.0) + max(var_y, 0.0)) / 2.0))


def roi_energy_loss(reference: float, current: float) -> float:
    """Fractional loss of in-ROI energy relative to a reference (0.0 = no loss).

    ``(reference - current) / reference``. A non-positive / non-finite reference
    means "no protection" and yields ``0.0``. Used by the safety guard: an
    evaluation whose loss exceeds ``max_roi_energy_loss`` is **abandoned** - it is
    scored far worse than any valid state so the search never adopts it.
    """
    ref = float(reference)
    cur = float(current)
    if not np.isfinite(ref) or ref <= 0.0:
        return 0.0
    return float((ref - cur) / ref)


def roi_pib_metric(
    img: np.ndarray,
    center: tuple[float, float],
    target_shape: str = "rectangle",
    target_size: float = 44.0,
    target_aspect_ratio: float = 4.0 / 3.0,
) -> tuple[float, float]:
    """Maximise the brightness inside the **target-shaped** ROI ("ROI PIB").

    Returns ``(score, energy)`` with ``score = energy = sum(I[roi]) / sum(I)`` —
    the fraction of the (in-window) light that lands inside the target rectangle,
    i.e. the bucket ratio evaluated over the target SHAPE instead of a radius
    bucket. Properties:

    * exposure / laser-drift invariant (a pure ratio);
    * unlike a ``-CV``-only objective it cannot be "won" by emptying the target
      box (an empty box scores 0);
    * no uniformity / peak / displacement penalty — this is the pure
      "put as much light as possible into the target" objective.
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"img must be 2D, got {frame.ndim}D")
    height, width = frame.shape
    roi = target_shape_roi(
        (height, width),
        center,
        target_shape,
        target_size,
        target_aspect_ratio,
    )
    total = float(frame.sum())
    if not np.isfinite(total) or total <= 0.0:
        return 0.0, 0.0
    energy = float(frame[roi].sum()) / total
    return float(energy), float(energy)


def rms_pib_terms(
    img: np.ndarray,
    center: tuple[float, float],
    target_shape: str = "rectangle",
    target_size: float | None = None,
    target_aspect_ratio: float = 4 / 3,
) -> tuple[float, float]:
    """Combined PIB + in-ROI RMS objective terms for target-shape shaping.

    Returns ``(pib_term, rms_term)``:

    * ``pib_term`` = fraction of the (in-window) light inside the target ROI
      (the exposure-invariant bucket ratio over the target shape);
    * ``rms_term`` = ``1 - u/(1+u)`` with ``u = std/mean`` over the ROI pixels
      (``1`` = perfectly flat, ``0`` = maximally non-uniform).

    Both terms are in ``[0, 1]``; the combined objective is
    ``J = w_pib * pib_term + w_rms * rms_term`` with adaptively-weighted terms
    (see :func:`_update_dynamic_weights`).
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"img must be 2D, got {frame.ndim}D")
    height, width = frame.shape
    size = float(target_size) if target_size is not None else float(min(height, width))
    roi = target_shape_roi(
        (height, width),
        center,
        target_shape,
        size,
        target_aspect_ratio,
    )
    total = float(frame.sum())
    if not np.isfinite(total) or total <= 0.0:
        return 0.0, 0.0
    roi_vals = frame[roi]
    if roi_vals.size == 0:
        return 0.0, 0.0
    pib_term = float(roi_vals.sum()) / total
    mean = float(roi_vals.mean())
    if mean <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    u = float(roi_vals.std()) / mean
    rms_term = 1.0 - u / (1.0 + u)
    return float(pib_term), float(rms_term)


def rmse_shape_metric(
    img: np.ndarray,
    center: tuple[float, float],
    target_shape: str = "rectangle",
    target_size: float = 44.0,
    target_aspect_ratio: float = 4.0 / 3.0,
) -> tuple[float, float]:
    """Minimisable RMSE between the sum-normalised frame and the target shape.

    Both sides are normalised to unit sum (``sum(img) = sum(target) = 1.0``)
    so the comparison is exposure / laser-drift invariant. The target is the
    uniform-intensity target-shaped ROI (1 inside, 0 outside — from
    ``target_shape_roi``) normalised to unit sum, and the frame is divided by
    its own total. The metric is the pixelwise RMSE of the two normalised maps:

        RMSE = sqrt(mean((img / sum(img) - target / sum(target)) ** 2))

    Minimising it drives the beam to a flat, uniform intensity inside the
    target shape and zero elsewhere — a non-PIB beam-shaping objective.

    Returns ``(rmse, energy)`` with ``energy = sum(I[roi]) / sum(I)`` (the
    in-ROI energy fraction) for logging. A dark/NaN/un-normalisable frame
    returns a strong penalty ``(1e3, 0.0)`` so the search never adopts it.
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"img must be 2D, got {frame.ndim}D")
    height, width = frame.shape
    roi = target_shape_roi(
        (height, width),
        center,
        target_shape,
        target_size,
        target_aspect_ratio,
    )
    total = float(frame.sum())
    if not np.isfinite(total) or total <= 0.0 or not roi.any():
        # Un-normalisable frame (dark / NaN) or an off-frame target: score far
        # worse than any valid state so the search never adopts it.
        return 1e3, 0.0
    roi_total = float(frame[roi].sum())
    energy = roi_total / total
    target_n = roi.astype(np.float64) / float(roi.sum())
    frame_n = frame / total
    rmse = float(np.sqrt(np.mean((frame_n - target_n) ** 2)))
    return rmse, energy


SHAPE_STAGE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "coarse": (0.0, 0.0, 0.0),
    "middle": (2.0, 0.0, 0.0),
    "fine": (3.0, 0.5, 0.5),
}


def shape_stage(progress: float) -> str:
    """Map optimisation progress in ``[0, 1]`` to a coarse/middle/fine stage."""
    p = float(min(max(progress, 0.0), 1.0))
    if p < 0.33:
        return "coarse"
    if p < 0.66:
        return "middle"
    return "fine"


def shape_stage_from_energy(energy: float) -> str:
    """Energy-driven stage for the coarse→fine shaping schedule.

    Keys off the best in-box energy reached so far rather than the iteration
    index, so it works for the gradient loop *and* for the heuristic searches
    (which never see an epoch counter): attract power first (``coarse``),
    then flatten it (``middle``), then shave hot spots / drift (``fine``).
    """
    e = float(min(max(energy, 0.0), 1.0))
    if e < 0.5:
        return "coarse"
    if e < 0.8:
        return "middle"
    return "fine"


def shape_metric(
    img: np.ndarray,
    center: tuple[float, float],
    reference_center: tuple[float, float] | None = None,
    target_shape: str = "rectangle",
    target_size: float = 44.0,
    target_aspect_ratio: float = 4.0 / 3.0,
    w_uniformity: float = 2.0,
    w_peak: float = 0.5,
    w_displacement: float = 0.5,
    stage: str | None = None,
    log_uniformity: bool = False,
) -> tuple[float, float]:
    """Return the dynamic-ROI shaping score and encircled-energy ratio.

    ``score = energy - w_u*u - w_pk*pk - w_d*d`` where every penalty term is
    bounded to ``[0, 1)`` (``u = std/mean`` mapped by ``u/(1+u)``,
    ``pk = max/mean`` mapped by ``(pk-1)/(pk+1)``) so the weights stay
    comparable and cannot swamp the energy term.

    Args:
        stage: ``"coarse"``/``"middle"``/``"fine"`` selects the schedule weights
            from :data:`SHAPE_STAGE_WEIGHTS` (use :func:`shape_stage` on the
            progress fraction). ``None`` uses the explicit ``w_*`` weights.
        log_uniformity: use ``log1p(u)`` instead of ``u/(1+u)`` to keep the
            gradient visible once the ROI is nearly flat.
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"img must be 2D, got {frame.ndim}D")
    height, width = frame.shape
    if reference_center is None:
        reference_center = ((width - 1) / 2.0, (height - 1) / 2.0)

    roi = target_shape_roi(
        (height, width),
        center,
        target_shape,
        target_size,
        target_aspect_ratio,
    )
    roi_values = frame[roi]
    total = float(frame.sum())
    encircled = float(roi_values.sum())
    energy = encircled / total if np.isfinite(total) and total > 0.0 else 0.0
    mean = float(roi_values.mean()) if roi_values.size else 0.0
    if np.isfinite(mean) and mean > np.finfo(np.float64).eps:
        uniformity = float(roi_values.std() / mean)
        peak = float(roi_values.max() / mean)
    else:
        uniformity = 0.0
        peak = 0.0

    dx = float(center[0]) - float(reference_center[0])
    dy = float(center[1]) - float(reference_center[1])
    displacement = float(np.hypot(dx, dy) / max(height, width))

    # Bound each penalty to [0, 1) so a poorly-conditioned term (the ROI std/mean
    # and max/mean of a sparse box can reach 10+) cannot dominate the energy term.
    u_term = (
        float(np.log1p(uniformity))
        if log_uniformity
        else float(uniformity / (1.0 + uniformity))
    )
    pk_term = float((peak - 1.0) / (peak + 1.0)) if peak > 1.0 else 0.0
    d_term = float(np.clip(displacement, 0.0, 1.0))

    if stage is not None:
        if stage not in SHAPE_STAGE_WEIGHTS:
            raise ValueError(
                f"stage must be one of {tuple(SHAPE_STAGE_WEIGHTS)}, got {stage!r}"
            )
        w_u, w_pk, w_d = SHAPE_STAGE_WEIGHTS[stage]
    else:
        w_u, w_pk, w_d = float(w_uniformity), float(w_peak), float(w_displacement)

    score = energy - w_u * u_term - w_pk * pk_term - w_d * d_term
    return float(score), float(energy)
