"""SPGD optimization for PIB using SLM with Zernike coefficient control.

This module replaces DM voltage control with SLM phase modulation via Zernike
polynomial coefficients. The optimization perturbs Zernike coefficients,
displays the resulting phase pattern on the SLM, and measures the PIB metric
from the camera image.

Key differences from pib.py:
- DM (NlightDM) → SLM (Santec)
- Voltage vectors → Zernike coefficient vectors
- dm.send_voltages(v) → slm.display_data(phase_pattern)
- Camera backend is selectable via ``cam_type`` (``create_camera`` registry)
- Two search families: SPGD gradient descent (``algorithm="spgd"``) and
  black-box heuristics (``ga``/``pso``/``sa``/``hc``/``rs``/``cem``/``de``)
- Optional pygame live view of the CCD frame, the sent phase and the Zernike
  coefficient bars (``show=True``)

Example:
    >>> from ao_shaping.optimizer.wfless.slm_zernike_pib import optimize_slm_zernike_pib
    >>> recorder = optimize_slm_zernike_pib(
    ...     center="shape",
    ...     epochs=2000,
    ...     n_max=4,
    ...     delta=0.1,
    ...     cam_id=0,
    ...     slm_number=1,
    ... )
"""

from __future__ import annotations

import inspect
import os
import time
from collections import deque
from contextlib import nullcontext
from typing import Any, Literal, cast

import numpy as np
import tqdm

from ao_shaping.algorithm.goal_functions.target_func import ImageTargetFunc
from ao_shaping.algorithm.gradient.adam import (
    SGD,
    Adam,
    AdaMOD,
    AdamW,
    Base,
    Muno,
    MunoW,
)
from ao_shaping.algorithm.heuristic.search import (
    heuristic_algorithm_choices,
    run_heuristic_search,
)
from ao_shaping.display import SlmZernikeDisplay
from ao_shaping.drivers.ccd.common import (
    auto_exposure as camera_auto_exposure,
)
from ao_shaping.drivers.ccd.common import (
    create_camera,
    get_camera_exposure_ms,
    list_camera_types,
    set_camera_exposure_ms,
)
from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL
from ao_shaping.optimizer.spgd import spgd_gradient
from ao_shaping.optimizer.wfless.slm_square_shaping import _zernike_indices
from ao_shaping.utils import Recorder, logger
from ao_shaping.utils.image.spots_calc import centroid, radius
from ao_shaping.utils.image.targets import create_target_shape
from ao_shaping.utils.io.file import gen_date_dir, gen_date_str
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

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

# adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# camera parameters
CAM_SAMPLE_ITER = 1
TEST_EXPOSURE_TIME_BRIGHTNESS = 220  # auto-exposure target used to locate the spot
IDEAL_SPOT_RADIUS = int(os.environ.get("IDEAL_SPOT_RADIUS", 6))

# Safety caps for learning_schedule: a single SPGD step must not be able to
# traverse the Zernike coefficient clip range (+/-5 in the loop). Uncapped the
# schedule returned lr=6 / delta=5, which saturated the clip every step and
# diverged on hardware (PIB 0.70 -> 0.08, peak brightness 199 -> 15).
SCHEDULE_MAX_LR = 1.0
SCHEDULE_MAX_DELTA = 0.5

# slm parameters
SLM_RESPONSE_TIME_S = 0.3  # Santec SLM-200 response time ~300ms
# On exit, leave the best-found phase on the SLM. The candidate set includes
# the initial (flat when starting from zeros, or a loaded) phase: if the search
# never improved on it, that phase is restored instead. Set False to skip
# touching the SLM on exit.
SLM_APPLY_BEST_ON_EXIT = True

# SLM resolution (from Santec.Panel_Res = (1920, 1200))
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
SLM_RESOLUTION = (SLM_WIDTH, SLM_HEIGHT)

OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "adamod": AdaMOD,
    "sgd": SGD,
    "muno": Muno,
    "munow": MunoW,
}

# Search-family selection: "spgd" runs the SPGD gradient loop below; every other
# name is a black-box heuristic handled by the shared driver in
# ``ao_shaping.algorithm.heuristic.search`` (see ``run_heuristic_search``).
ALGORITHM_CHOICES = heuristic_algorithm_choices()

# Symmetric clip applied to every candidate Zernike vector before it is turned
# into a phase (matches the +/-5 clip the SPGD loop historically used).
ZERNIKE_CLIP = 5.0

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


def gauss_center(
    img: np.ndarray, half_win: int = 48, bg: float | None = None
) -> np.ndarray:
    """Locate the spot centre with a background-subtracted squared centroid.

    The returned coordinates use the project convention ``(x, y)``.
    """
    frame = np.asarray(img, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"img must be 2D, got {frame.ndim}D")
    height, width = frame.shape
    yy, xx = np.mgrid[0:height, 0:width]
    total = float(frame.sum())
    fallback = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0:
        return fallback

    cx = float(np.sum(frame * xx) / total)
    cy = float(np.sum(frame * yy) / total)
    half_win = max(1, int(half_win))
    y0 = max(0, int(cy - half_win))
    y1 = min(height, int(cy + half_win) + 1)
    x0 = max(0, int(cx - half_win))
    x1 = min(width, int(cx + half_win) + 1)
    win = frame[y0:y1, x0:x1].copy()

    if bg is None:
        k = min(5, win.shape[0], win.shape[1])
        corners = np.concatenate(
            (
                win[:k, :k].ravel(),
                win[:k, -k:].ravel(),
                win[-k:, :k].ravel(),
                win[-k:, -k:].ravel(),
            )
        )
        background = float(np.median(corners))
    else:
        background = float(bg)
    if not np.isfinite(background):
        background = 0.0

    win = np.clip(win - background, 0.0, None)
    weights = win**2
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= np.finfo(np.float64).eps:
        return np.array([cx, cy], dtype=np.float64)

    cx = float(np.sum(weights * xx[y0:y1, x0:x1]) / weight_sum)
    cy = float(np.sum(weights * yy[y0:y1, x0:x1]) / weight_sum)
    return np.array([np.clip(cx, 0.0, width - 1), np.clip(cy, 0.0, height - 1)])


def threshold_spot_center(img: np.ndarray) -> tuple[int, int]:
    """Threshold-based 0-order spot centre, robust to degenerate frames.

    Marks every pixel brighter than a corner background estimate and takes that
    mask's centroid. Three guards make it safe on real camera frames:

    * empty mask — the brightest pixels sit *inside* the corner patch itself
      (hot pixel / stray light), so ``img > corner_max`` selects nothing and
      ``scipy.center_of_mass`` would divide by zero → ``centroid()`` would raise
      ``ValueError: cannot convert float NaN to integer``; fall back to a
      relative-threshold centroid;
    * all-dark frame → return the frame centre instead of NaN;
    * non-2D / empty input → ``ValueError``.

    Returns ``(x, y)`` in pixels (project convention).
    """
    frame = np.asarray(img)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"img must be a non-empty 2D array, got shape {frame.shape}")
    height, width = frame.shape
    if float(frame.max()) <= 0.0:
        return (width // 2, height // 2)

    corner_h = max(int(height // 50), 2)
    corner_w = max(int(width // 50), 2)
    corner = frame[:corner_h, :corner_w]
    mask = frame > float(np.max(corner))
    if np.any(mask):
        cx, cy = centroid(mask)
        return (int(cx), int(cy))

    # Degenerate mask (the corner patch itself holds the brightest pixels, e.g. a
    # hot pixel): drop that patch and take a plain centroid. A relative threshold
    # would still be scaled by the hot pixel's value and drag the centre towards
    # the corner.
    fallback = frame.copy()
    fallback[:corner_h, :corner_w] = 0
    if float(fallback.max()) > 0.0:
        cx, cy = centroid(fallback)
        return (int(cx), int(cy))
    return (width // 2, height // 2)


def argmax_anchored_center(
    img: np.ndarray, half_win: int | None = None
) -> tuple[int, int]:
    """0-order spot centre: global-argmax anchor + **local** centroid refinement.

    On the 2f bench the 0-order sits at the frame's global maximum (AGENTS.md), so
    the argmax is a far more reliable anchor than a corner-threshold mask (which
    jumps to hot pixels) or a full-image centroid (which stray light / reflections
    drag off the spot). Refinement is restricted to a window around the anchor, so
    light outside the spot cannot move the centre.

    Returns ``(x, y)`` in pixels. All-dark frames return the frame centre.
    """
    frame = np.asarray(img)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"img must be a non-empty 2D array, got shape {frame.shape}")
    height, width = frame.shape
    if float(frame.max()) <= 0.0:
        return (width // 2, height // 2)

    anchor_y, anchor_x = np.unravel_index(int(np.argmax(frame)), frame.shape)
    win = int(half_win) if half_win else max(int(min(frame.shape) // 20), 8)
    y0, y1 = max(0, anchor_y - win), min(height, anchor_y + win + 1)
    x0, x1 = max(0, anchor_x - win), min(width, anchor_x + win + 1)

    patch = frame[y0:y1, x0:x1].astype(np.float64)
    patch = np.clip(patch - float(np.percentile(patch, 20)), 0.0, None)
    total = float(patch.sum())
    if not np.isfinite(total) or total <= 0.0:
        return (int(anchor_x), int(anchor_y))

    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((patch * xx).sum() / total)
    cy = float((patch * yy).sum() / total)
    return (int(round(cx)), int(round(cy)))


def clamp_center_to_frame(
    center: tuple[int|float, int|float], frame_shape: tuple[int|float, int|float], window: int
) -> tuple[int, int]:
    """Clamp an ROI centre so a ``window x window`` ROI always fits the frame.

    ``DahengCamera.reset_window`` asserts ``x_offset >= 0 and y_offset >= 0``, so
    a spot detected near a frame edge aborted the whole run. Clamping keeps the
    ROI inside the frame (the spot is simply off-centre) instead of crashing.

    Args:
        center: ``(x, y)`` ROI centre in pixels.
        frame_shape: ``(height, width)`` of the captured image.
        window: requested (square) ROI size in pixels.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    # ``half`` is capped at the half-frame so a (misconfigured) window larger
    # than the frame still yields an in-frame centre instead of an out-of-range one.
    half = min(max(0, int(window) // 2), min(height, width) // 2)
    max_x = max(half, width - half - 1)
    max_y = max(half, height - half - 1)
    return (
        int(np.clip(int(center[0]), half, max_x)),
        int(np.clip(int(center[1]), half, max_y)),
    )


def resolve_initial_exposure(
    exposure_time_ms: float, target_max_brightness: float
) -> tuple[str, float]:
    """Decide the initial exposure action (precedence: fixed > auto > keep).

    Returns one of:
        ``("fixed", ms)`` — use a fixed exposure time;
        ``("auto", target)`` — auto-expose to the target peak brightness;
        ``("keep", 0.0)`` — leave the exposure as-is (no auto-adjust).
    """
    if exposure_time_ms > 0:
        return ("fixed", float(exposure_time_ms))
    if target_max_brightness > 0:
        return ("auto", float(target_max_brightness))
    return ("keep", 0.0)


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


# Coarse→fine schedule for the ``shape`` objective: pulling the energy into the# box first, then flattening it, then shaving hot spots / drift, converges better
# than one fixed metric for the whole run. Values are
# (w_uniformity, w_peak, w_displacement); "coarse" is energy-only.
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
    u_term = float(np.log1p(uniformity)) if log_uniformity else float(
        uniformity / (1.0 + uniformity)
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


def _create_optimizer(optimizer_type: str, dim: int, lr: float, **kwargs: Any) -> Base:
    """Create the configured optimizer while filtering unsupported kwargs."""
    optimizer_cls = OPTIMIZER_MAP.get(optimizer_type.lower(), AdaMOD)
    filtered_kwargs = {}
    signature = inspect.signature(optimizer_cls.__init__)
    for key, value in kwargs.items():
        if key in signature.parameters:
            filtered_kwargs[key] = value
    return optimizer_cls(dim, lr=lr, **filtered_kwargs)


ZERNIKE_APERTURE_RADIUS = 300.0

# The camera window must be at least this multiple of the target's LONG side.
# The shaping metric divides the in-box energy by the WINDOW total, so a window
# barely larger than the target cannot see the energy the shaping pushes out of
# the box (the measurement would be invalid). Enforced in
# ``optimize_slm_zernike_pib`` by auto-enlarging the window (with a warning).
CAM_WINDOW_TARGET_MARGIN = 1.5

# Auto target-box size = this multiple of the flat-field spot WAIST (w0).
# 2 * w0 is the waist DIAMETER - a real shaping target. (2 x the 99%-encircled
# radius is dominated by the stray halo and lands ~2x too large, which leaves no
# shaping headroom at all.)
TARGET_BOX_WAIST_FACTOR = 2.0
"""Aperture radius (px) the Zernike phase is defined over.

Must match the illuminated beam radius on the SLM. The panel is 1920x1200, so
PatternHelper's default is half the short side = 600 px; this bench's beam
radius is only ~300 px. With a 600 px aperture only the inner half of the
polynomial lands on the beam, so every mode is nearly CONSTANT across the
illuminated area -- and a constant phase does not change the far field, i.e.
the correction silently does nothing (hardware-verified: flat vs a 2 rad
defocus were indistinguishable until this was matched).
"""


# Memory-slot range used for phase writes (never repeat a slot consecutively).
_SLOT_MIN, _SLOT_MAX = 2, 125
_SLOT_STATE = {"slot": _SLOT_MIN - 1}


def _display(slm, gray) -> int:
    """Display ``gray`` on a **rotating explicit memory slot**.

    ``slm.display_data(gray)`` (default slot handling) produced NO optical effect
    on this bench: verified on BOTH cameras, a 5 rad tilt did not move the spot
    and a uniform 0 vs 900 changed nothing. ``slm.display_data(gray,
    memory_number=<distinct slot>, memory_mode=MEMORY_MODE_INTERNAL)`` -- the
    mechanism ``tools/slm/slm_diagnose.py`` uses -- does refresh the LCOS. Since
    writing the *same* slot twice is a firmware no-op, slots are rotated.
    """
    _SLOT_STATE["slot"] = (
        _SLOT_MIN if _SLOT_STATE["slot"] >= _SLOT_MAX else _SLOT_STATE["slot"] + 1
    )
    slm.display_data(
        gray,
        memory_number=_SLOT_STATE["slot"],
        memory_mode=MEMORY_MODE_INTERNAL,
    )
    return _SLOT_STATE["slot"]


def _zernike_to_phase(
    coeffs: np.ndarray,
    n_max: int,
    pattern_helper: PatternHelper,
    radius: float | None = None,
) -> np.ndarray:
    """Convert a flat Zernike coefficient array to a radian phase pattern.

    Args:
        coeffs: Flat array of Zernike coefficients (amplitudes in wavelengths).
        n_max: Maximum Zernike radial order.
        pattern_helper: PatternHelper instance for phase generation.

    Returns:
        float64 raw radian phase pattern array with shape (SLM_HEIGHT, SLM_WIDTH).
        No mod-2π here — the caller must convert to grayscale via
        ``slm.create_phase_from_array()`` before ``slm.display_data()``
        (2026-09: PatternHelper no longer performs phase→gray; the SLM
        driver applies the 2π wrap on radian→grayscale conversion).
    """
    modes = _zernike_indices(n_max)
    coeffs_dict: dict[tuple[int, int], float] = {}
    for i, (n, m) in enumerate(modes):
        if i < len(coeffs):
            coeffs_dict[(n, m)] = float(coeffs[i])
    return pattern_helper.generate_zernike_polynomial(
        n_max=n_max,
        coefficients=coeffs_dict,
        radius=ZERNIKE_APERTURE_RADIUS if radius is None else radius,
    )


def learning_schedule(
    power_radius: float,
    ideal_r: float = IDEAL_SPOT_RADIUS,
    gradient_history: list[float] | None = None,
    pib_history: list[float] | None = None,
    epoch: int = 0,
) -> tuple[float, float]:
    """Dynamic learning rate scheduler based on power radius and convergence state.

    Args:
        power_radius: Current power radius.
        ideal_r: Ideal spot radius.
        gradient_history: Recent gradient magnitude history for convergence detection.
        pib_history: Recent PIB value history for convergence detection.
        epoch: Current iteration number.

    Returns:
        (lr, delta): Learning rate and perturbation amplitude.
    """
    # Base parameters: segmented by power_radius
    if power_radius <= ideal_r:
        base_lr, base_delta = 1.5, 1
    elif power_radius <= 2 * ideal_r:
        base_lr, base_delta = 2, 2
    elif power_radius <= 3 * ideal_r:
        base_lr, base_delta = 2.5, 3
    elif power_radius <= 4 * ideal_r:
        base_lr, base_delta = 3, 4
    elif power_radius <= 5 * ideal_r:
        base_lr, base_delta = 4.5, 5
    else:
        base_lr, base_delta = 6, 5

    # If no convergence history, return base parameters directly (still capped)
    if gradient_history is None or pib_history is None or len(gradient_history) < 5:
        return (
            min(base_lr, SCHEDULE_MAX_LR),
            min(base_delta, SCHEDULE_MAX_DELTA),
        )

    # Convergence state detection
    recent_grads = (
        list(gradient_history[-10:])
        if len(gradient_history) >= 10
        else gradient_history
    )
    recent_pibs = list(pib_history[-10:]) if len(pib_history) >= 10 else pib_history

    # Gradient trend (lower variance = more stable convergence)
    grad_mean = np.mean(recent_grads)
    grad_std = np.std(recent_grads) if len(recent_grads) > 1 else 0
    grad_cv = grad_std / (grad_mean + 1e-8)  # Coefficient of variation

    # PIB trend
    pib_mean = np.mean(recent_pibs)
    pib_std = np.std(recent_pibs) if len(recent_pibs) > 1 else 0
    pib_trend = (
        (recent_pibs[-1] - recent_pibs[0]) / (len(recent_pibs) + 1e-8)
        if len(recent_pibs) > 1
        else 0
    )

    # Dynamic adjustment factors
    lr_factor = 1.0
    delta_factor = 1.0

    # Case 1: Small gradient variance (stable convergence) → reduce lr and delta
    if grad_cv < 0.1:
        lr_factor = 0.5
        delta_factor = 0.5
    # Case 2: Medium gradient variance (normal fluctuation) → keep
    elif grad_cv < 0.3:
        lr_factor = 0.8
        delta_factor = 0.8
    # Case 3: Large gradient variance (oscillation) → reduce lr significantly
    elif grad_cv > 0.8:
        lr_factor = 0.3
        delta_factor = 1.2  # Increase exploration

    # Case 4: PIB not improving (possible local optimum)
    if abs(pib_trend) < 1e-5 and pib_std < 0.01:
        delta_factor = max(delta_factor, 1.5)
        lr_factor = min(lr_factor, 0.7)
    # Case 5: PIB decreasing (diverging) → reduce lr
    elif pib_trend < -0.001:
        lr_factor = 0.4
        delta_factor = 0.6
    # Case 6: PIB increasing (normal convergence) → keep or fine-tune
    elif pib_trend > 0.001:
        lr_factor = min(lr_factor, 1.1)

    # Early epoch warmup
    if epoch < 20:
        lr_factor *= 1.2
        delta_factor *= 1.1

    # Clamp adjustment range
    lr_factor = np.clip(lr_factor, 0.2, 2.0)
    delta_factor = np.clip(delta_factor, 0.3, 2.5)

    final_lr = min(base_lr * lr_factor, SCHEDULE_MAX_LR)
    final_delta = min(base_delta * delta_factor, SCHEDULE_MAX_DELTA)

    return final_lr, final_delta


def optimize_slm_zernike_pib(
    center,
    epochs,
    n_max: int = 4,
    r_bucket=0,
    # 0.2 rad, NOT 0.1: the measured noise floor of the shaping objective is
    # dJ_noise = 4e-4 and a 0.1 rad perturbation moves J by only 2.8e-4
    # (SNR 0.69 -> the SPGD gradient is noise). 0.2 rad gives SNR 2.77
    # (measured on the bench by scripts/measure_shape_sensitivity.py).
    delta: float = 0.2,
    lr: float = 0,
    exposure_time_ms: float = 80.0,
    shrink_iter: int = 0,
    shrink_ratio: float = 0.9,
    cam_id=0,
    cam_type: str = "daheng",
    show: bool = False,
    init_c=None,
    cam_size=250,
    target_max_brightness=40,
    slm_number: int = 1,
    slm_wavelength: int = 1064,
    optimizer_type: str = "adamod",
    algorithm: str = "spgd",
    pop_size: int | None = None,
    random_seed: int | None = None,
    objective: str = "shape",
    target_shape: str | None = "rectangle",
    target_size: float | None = None,
    target_aspect_ratio: float = 4.0 / 3.0,
    target_center_smooth: int = 3,
    # NOTE: keep False. Scores from different stages are NOT comparable (the same
    # frame scores ~e in "coarse" but e-penalties in "fine"), so a baseline
    # measured in "coarse" becomes unbeatable and every search reports gain=0.
    # Enabling it needs per-stage best tracking + re-scoring the baseline at the
    # final stage - see shape_metric(stage=...) for the mechanism.
    shape_schedule: bool = False,
    max_roi_energy_loss: float = 0.6,
    w_uniformity: float = 2.0,
    w_peak: float = 0.5,
    w_displacement: float = 0.0,
    log_uniformity: bool = False,
    zernike_radius: float = ZERNIKE_APERTURE_RADIUS,
    shift_x: int | None = 0,
    shift_y: int | None = 0,
    **kwargs,
):
    """Optimize PIB (Power in Bucket) using SLM with Zernike coefficient control.

    Zernike coefficients displayed on an SLM are searched to optimise an imaging
    objective measured by a camera. Two search families are available:
    ``algorithm="spgd"`` (Stochastic Parallel Gradient Descent, the default) or a
    black-box heuristic (``ga``/``pso``/``sa``/``hc``/``rs``/``cem``/``de``).

    Args:
        center: Center position for PIB calculation. Can be None (auto-detect),
            "mass" (centroid), "max" (brightest pixel), "shape" (threshold-based),
            or a tuple (x, y).
        epochs: Number of optimization iterations.
        n_max: Maximum Zernike radial order. Controls the number of modes.
        r_bucket: Bucket radius. If 0, auto-adjusted based on power radius.
        delta: Perturbation amplitude for Zernike coefficients.
        lr: Learning rate. If 0, auto-adjusted via learning_schedule.
        exposure_time_ms: Camera exposure time in ms. If 0, auto-exposure.
        shrink_iter: Shrink iteration count. If 0, no shrinking.
        shrink_ratio: Shrink ratio for bucket radius.
        cam_id: Camera device ID (or path/URL for file-based backends).
        cam_type: Camera backend selected through ``create_camera``; one of
            ``daheng`` (default) or ``miicam``. Exposure handling is normalised
            across backends by the ``drivers.ccd.common`` helpers.
        show: When True, open a pygame window showing the CCD frame, the sent SLM
            phase and the Zernike coefficient bars during the search.
        init_c: Initial Zernike coefficients. If None, starts from zeros.
        cam_size: Camera window size.
        target_max_brightness: Target max brightness for auto-exposure.
        slm_number: SLM device number (1-8).
        slm_wavelength: SLM wavelength in nm.
        optimizer_type: Optimizer type for the SPGD gradient stage
            (adam/adamod/sgd/muno); ignored when ``algorithm`` is a heuristic.
        algorithm: Search algorithm: ``"spgd"`` or one of the heuristics
            ``ga``/``pso``/``sa``/``hc``/``rs``/``cem``/``de``.
        pop_size: Population size for population-based heuristics
            (ga/pso/cem/de); ignored otherwise.
        random_seed: Random seed for reproducibility (applies to both SPGD
            perturbation sampling and the heuristic RNG).
        objective: Optimization target: 'pib' (maximize), 'radiu' (minimize radius),
            'avg_radiu' (maximize average radius), or 'shape' (dynamic target ROI).
        target_shape: Target ROI shape. Supplying it selects the ``shape`` objective;
            defaults to ``rectangle`` for that objective.
        target_size: Full target extent in camera pixels (rectangle short side,
            circle diameter, square side). ``None`` derives it from the initial
            99% encircled-energy radius.
        target_aspect_ratio: Width:height ratio for a rectangular target ROI.
        target_center_smooth: Number of recent Gaussian-centre estimates averaged
            for each frame (minimum 1).
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization history recorder.
    """
    delta = abs(delta)
    epochs = int(epochs)
    rng = np.random.default_rng(random_seed)

    algorithm = str(algorithm).lower()
    if algorithm not in ALGORITHM_CHOICES:
        raise ValueError(
            f"algorithm must be one of {ALGORITHM_CHOICES}, got {algorithm!r}"
        )

    objective = str(objective).lower()
    if target_shape is not None:
        target_shape = str(target_shape).lower()
        if target_shape not in TARGET_SHAPE_CHOICES:
            raise ValueError(
                f"target_shape must be one of {TARGET_SHAPE_CHOICES}, got {target_shape!r}"
            )
    if target_shape is not None and objective not in ("pib", "shape", "roi_pib"):
        raise ValueError(
            "target_shape can only be used with objective='pib', 'shape' or 'roi_pib'"
        )
    if target_shape is not None and objective != "roi_pib":
        # Supplying target_shape implies the dynamic-ROI shaping objective, except
        # for roi_pib where the shape only selects which ROI to maximise inside.
        objective = "shape"
    if objective in ("shape", "roi_pib") and target_shape is None:
        target_shape = "rectangle"
    shape_for_metric = cast(TargetShape, target_shape or "rectangle")
    if objective not in ("pib", "radiu", "avg_radiu", "shape", "roi_pib"):
        raise ValueError(
            f"objective must be one of ('pib', 'radiu', 'avg_radiu', 'shape', "
            f"'roi_pib'), got {objective}"
        )
    if target_center_smooth < 1:
        raise ValueError(
            f"target_center_smooth must be at least 1, got {target_center_smooth}"
        )
    if target_size is not None and (not np.isfinite(target_size) or target_size <= 0):
        raise ValueError(f"target_size must be positive, got {target_size!r}")
    if not np.isfinite(target_aspect_ratio) or target_aspect_ratio <= 0:
        raise ValueError(
            f"target_aspect_ratio must be positive, got {target_aspect_ratio!r}"
        )

    # Optimization mode mapping: pib, avg_radiu and shape are maximized; radiu is minimized
    if not 0.0 <= max_roi_energy_loss <= 1.0:
        raise ValueError(
            f"max_roi_energy_loss must be within 0..1 (0 disables the guard), "
            f"got {max_roi_energy_loss!r}"
        )
    for _name, _w in (
        ("w_uniformity", w_uniformity),
        ("w_peak", w_peak),
        ("w_displacement", w_displacement),
    ):
        if not np.isfinite(_w) or _w < 0.0:
            raise ValueError(f"{_name} must be a finite, non-negative weight, got {_w!r}")
    objective_mode = (
        "max" if objective in ("pib", "avg_radiu", "shape", "roi_pib") else "min"
    )
    recorder = Recorder(mark=objective, mode=objective_mode)

    # History for convergence detection
    _gradient_history: list[float] = []
    _pib_history: list[float] = []
    _max_history_len = 50

    # Zernike mode count and index mapping
    nk = calc_n_zernike_terms(n_max)
    zernike_modes = _zernike_indices(n_max)

    # SLM pattern helper
    pattern_helper = PatternHelper(resolution=SLM_RESOLUTION, bits=10)

    if show:
        # PIB (bucket ratio) is bounded 0..1 -> fixed y-axis; other objectives auto-scale.
        _curve_y_range = (0.0, 1.0) if objective in ("pib", "roi_pib") else None
        display_ctx = SlmZernikeDisplay(
            zernike_clip=ZERNIKE_CLIP,
            curve_title=f"{objective} curve",
            curve_y_range=_curve_y_range,
        )
    else:
        display_ctx = nullcontext(None)

    with (
        create_camera(
            cam_type,
            cam_id=cam_id,
            exposure_time_ms=exposure_time_ms,
            skip_sampling=False,
        ) as cam,
        Santec(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
        display_ctx as live_display,
    ):
        # Initialize Zernike coefficients
        if init_c is None or len(init_c) == 0:
            _init_c = np.zeros(nk, dtype=np.float64)
        else:
            _init_c = np.array(init_c, dtype=np.float64)
            if len(_init_c) < nk:
                padded = np.zeros(nk, dtype=np.float64)
                padded[: len(_init_c)] = _init_c
                _init_c = padded
            elif len(_init_c) > nk:
                _init_c = _init_c[:nk]

        # Reset SLM to flat phase
        initial_phase = slm.create_phase_from_array(
            _zernike_to_phase(_init_c, n_max, pattern_helper, zernike_radius)
        )
        _display(slm, initial_phase)
        time.sleep(SLM_RESPONSE_TIME_S)

        # Initial acquisition used to locate the spot. When a fixed exposure was
        # requested we honour it here too: auto-exposing to
        # TEST_EXPOSURE_TIME_BRIGHTNESS can exceed a safe exposure limit (e.g.
        # >3 ms on this bench) before the operator's chosen value is applied.
        if exposure_time_ms > 0:
            set_camera_exposure_ms(cam, exposure_time_ms)
            _img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        else:
            _img = camera_auto_exposure(cam, TEST_EXPOSURE_TIME_BRIGHTNESS)

        def intellij_center(img):
            """Smart centre: argmax-anchored local centroid, refined by the full
            centroid when the spot core is not a hole (flat core)."""
            (h, w) = img.shape
            margin = int(IDEAL_SPOT_RADIUS)
            center = argmax_anchored_center(img)
            (cx, cy) = center
            y0, y1 = max(0, cy - margin), min(h, cy + margin)
            x0, x1 = max(0, cx - margin), min(w, cx + margin)
            if y1 > y0 and x1 > x0 and np.all(
                img[y0:y1, x0:x1] >= np.max(img) * 0.4
            ):
                # Flat (non-hollow) core: refine with a LOCAL centroid. A
                # full-image centroid (as intelligen_center does) is dragged tens
                # of pixels by stray light — measured (1394 vs 958 on this bench).
                center = argmax_anchored_center(img, half_win=max(margin * 6, 32))
            return center

        if center is None:
            center = intellij_center(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == "max":
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == "shape":
                # argmax-anchored local centroid (2f bench: 0-order = frame max)
                center = argmax_anchored_center(_img)
            else:
                raise ValueError(f"known center: {center}")
        else:
            center = center

        # Integer pixel coordinates (centroid() already returns ints; the cast
        # also covers user-supplied float tuples).
        center = (int(round(center[0])), int(round(center[1])))

        logger.info(
            f"Centroid brightness: {_img[center[::-1]]}@{center}, "
            f"Max brightness: {np.max(_img)} @ {get_camera_exposure_ms(cam)}ms"
        )

        img_size = (cam_size, cam_size)
        # Keep the ROI inside the frame: DahengCamera.reset_window rejects
        # negative offsets, so a spot near an edge would abort the run.
        center = clamp_center_to_frame(center, _img.shape, cam_size)
        # Full-frame geometry: needed to re-window (enlargement) and to capture
        # the report's un-windowed before/after frames later on.
        center_full = (float(center[0]), float(center[1]))
        full_frame_shape = tuple(int(v) for v in _img.shape)
        img_size, center = cam.reset_window(center, img_size)
        logger.info(f"reset window center @ {center}")

        # Precedence: fixed exposure (`--exposure_time_ms > 0`) wins over
        # auto-exposure (`--target_max_brightness > 0`); `0/0` keeps the current
        # exposure ("若为0则不自动调整曝光时间", matching the runner help).
        _exposure_mode, _exposure_value = resolve_initial_exposure(
            exposure_time_ms, target_max_brightness
        )
        if _exposure_mode == "fixed":
            set_camera_exposure_ms(cam, _exposure_value)
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        elif _exposure_mode == "auto":
            init_img = camera_auto_exposure(cam, _exposure_value)
        else:
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        logger.debug(
            f"Initial Image Max brightness: {np.max(init_img)} "
            f"@ {get_camera_exposure_ms(cam)}ms"
        )
        img_size = init_img.shape[::-1]

        # Enforce a >= CAM_WINDOW_TARGET_MARGIN x target-long-side window: the
        # metric's energy denominator is the window total, so the out-of-box
        # energy must stay visible. Enlarge (and re-capture) when cam_size is
        # smaller than that, whatever value the caller asked for.
        _est_extent = (
            float(target_size)
            if target_size is not None
            else min(
                max(
                    TARGET_BOX_WAIST_FACTOR * float(spot_waist_sigma(init_img, center)),
                    4.0,
                ),
                float(min(img_size)),
            )
        )
        _long_side = _est_extent * (
            float(target_aspect_ratio) if shape_for_metric == "rectangle" else 1.0
        )
        _required = int(np.ceil(CAM_WINDOW_TARGET_MARGIN * _long_side))
        if _required > int(img_size[0]):
            logger.warning(
                "camera window {}x{} is smaller than {:.1f}x the target long side "
                "({:.0f}px) - enlarging to {}x{}",
                img_size[0],
                img_size[1],
                CAM_WINDOW_TARGET_MARGIN,
                _long_side,
                _required,
                _required,
            )
            _req_center = clamp_center_to_frame(
                center_full, full_frame_shape, _required
            )
            img_size, center = cam.reset_window(_req_center, (_required, _required))
            center_full = (float(_req_center[0]), float(_req_center[1]))
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
            img_size = init_img.shape[::-1]
            logger.info("camera window enlarged to {}x{}", img_size[0], img_size[1])

        reference_center: tuple[float, float] = (
            float(center[0]),
            float(center[1]),
        )
        if target_size is None:
            _w, _h = img_size
            # Box = TARGET_BOX_WAIST_FACTOR x the flat-field spot WAIST (not the
            # 99%-encircled radius, which the stray halo blows up ~2x).
            _waist = float(spot_waist_sigma(init_img, reference_center))
            _want = TARGET_BOX_WAIST_FACTOR * _waist
            logger.info(
                "spot waist w0 = {:.1f}px -> target box (short side) = {:.1f}px",
                _waist,
                _want,
            )
            # Fit by the template's LONG side: a 4:3 rectangle is
            # ``size * aspect_ratio`` wide, so clamping only by ``min(img_size)``
            # let the box exceed the window width and get clipped.
            if shape_for_metric == "rectangle":
                _fit = min(float(_h), float(_w) / float(target_aspect_ratio))
            else:
                _fit = min(float(_h), float(_w))
            if _want > _fit:
                logger.warning(
                    "dynamic target size {:.1f}px ({}x waist) exceeds the "
                    "{}x{} window fit {:.1f}px - clamped; raise cam_size for a "
                    "full-size target",
                    _want,
                    TARGET_BOX_WAIST_FACTOR,
                    _w,
                    _h,
                    _fit,
                )
            target_size = float(min(max(_want, 4.0), _fit))
            logger.info(f"Use dynamic target size @ {target_size:.1f}px")

        target_center_smooth = int(target_center_smooth)
        target_center_history: deque[tuple[float, float]] = deque()
        for _ in range(target_center_smooth):
            target_center_history.append(reference_center)

        if r_bucket <= 0:
            _w, _h = img_size
            r_bucket = ImageTargetFunc(_w, _h, center).radius(init_img, energy=0.99)
            r_bucket = min(r_bucket, cam_size // 2) * shrink_ratio
            _fix_bucket = False
            logger.info(f"Use dynamic radiu @ {r_bucket}")
        else:
            _fix_bucket = True

        if (
            shrink_ratio <= 0
            or np.isclose(shrink_ratio, 1.0)
            or r_bucket <= IDEAL_SPOT_RADIUS
        ):
            update_iter = max(1, epochs)
        else:
            shrink_span = np.log(IDEAL_SPOT_RADIUS / r_bucket) / np.log(shrink_ratio)
            if np.isfinite(shrink_span) and shrink_span > 0:
                update_iter = max(1, int(epochs * 0.8 // shrink_span))
            else:
                update_iter = max(1, epochs)
        _init_r = r_bucket

        target_func = ImageTargetFunc.build_from_init_image(init_img)

        # Objective calculation functions
        def test_pib(img):
            return target_func.pib(img, IDEAL_SPOT_RADIUS)[1]

        to_min = 1
        if objective == "pib":
            # Maximize PIB: negate the SPGD estimate so `_init_c - update`
            # ascends the objective. Same convention as pib.py's intended
            # `to_min = -1`; without this flip the loop minimizes PIB.
            to_min = -1

            def calc_objective(img):
                pib, pib_ratio = target_func.pib(img, r_bucket)
                return pib, pib_ratio
        elif objective == "roi_pib":
            # Maximise the brightness inside the TARGET-SHAPED ROI: the fraction of
            # the light landing in the target rectangle (exposure-invariant ratio,
            # no uniformity/peak/drift penalties).
            to_min = -1

            def calc_objective_roi_pib(img):
                # FIXED target ROI (never tracks the spot) - see calc_objective_shape.
                return roi_pib_metric(
                    img,
                    reference_center,
                    shape_for_metric,
                    target_size,
                    target_aspect_ratio,
                )

            calc_objective = calc_objective_roi_pib
        elif objective == "shape":
            to_min = -1

            _shape_state = {"best_energy": 0.0}

            def calc_objective_shape(img):
                # The target ROI is FIXED at reference_center - it never tracks the
                # measured spot, so a spot that drifts/scatters out of the box is
                # penalised instead of being followed (displacement weight = 0 for
                # the same reason: the energy term already captures the drift).
                stage = (
                    shape_stage_from_energy(_shape_state["best_energy"])
                    if shape_schedule
                    else None
                )
                score, energy = shape_metric(
                    img,
                    reference_center,
                    reference_center,
                    shape_for_metric,
                    target_size,
                    target_aspect_ratio,
                    w_uniformity=w_uniformity,
                    w_peak=w_peak,
                    w_displacement=w_displacement,
                    stage=stage,
                    log_uniformity=log_uniformity,
                )
                # Monotone: the schedule may only get stricter, never laxer.
                _shape_state["best_energy"] = max(_shape_state["best_energy"], energy)
                return score, energy

            calc_objective = calc_objective_shape
        elif objective == "radiu":

            def calc_objective_radiu(img):
                r = target_func.radius(img, energy=0.99)
                return r, 0.0

            calc_objective = calc_objective_radiu
        elif objective == "avg_radiu":
            # Maximize average radius: same sign flip as PIB.
            to_min = -1

            def calc_objective_avg(img):
                return target_func.avg_radius(img, moment=1.0)

            calc_objective = calc_objective_avg

        # --- Safety guard: abandon evaluations that lose too much ROI energy --
        # Reference = the in-ROI energy of the initial (flat/loaded) frame. Any
        # evaluation whose in-ROI energy dropped by more than
        # ``max_roi_energy_loss`` (fraction of that reference) is *abandoned*:
        # it is scored far worse than any valid state, so the search never adopts
        # it and the SLM is never left there. ``0`` disables the guard.
        _raw_calc_objective = calc_objective
        _guard_ref_energy: float | None = None
        _guard_violations = 0
        def _fixed_roi_energy(frame) -> float:
            """Light fraction inside the **FIXED** target ROI - the guard's observable.

            Deliberately NOT the running (spot-tracking) ROI used by the objective:
            that one follows the measured spot centre, so it always retains the same
            energy and could never detect a loss. This ROI stays at
            ``reference_center`` / ``target_size`` / ``target_shape``.
            """
            return roi_pib_metric(
                frame,
                reference_center,
                shape_for_metric,
                target_size,
                target_aspect_ratio,
            )[0]

        if max_roi_energy_loss > 0.0 and objective in ("pib", "shape", "roi_pib"):
            _ref = float(_fixed_roi_energy(init_img))
            _guard_ref_energy = _ref if np.isfinite(_ref) and _ref > 0.0 else None
            if _guard_ref_energy is None:
                logger.warning(
                    "ROI energy guard disabled: initial in-ROI energy is {:.6f}", _ref
                )
            else:
                logger.info(
                    "ROI energy guard armed: reference energy {:.4f}, max loss {:.1%}",
                    _guard_ref_energy,
                    max_roi_energy_loss,
                )

        def calc_objective(img):
            """Objective wrapped with the ROI energy-loss safety guard.

            Returns a strongly penalised score (and the true ratio for logging)
            when the in-ROI energy loss exceeds ``max_roi_energy_loss`` - the
            evaluation is thereby abandoned.
            """
            nonlocal _guard_violations
            j, ratio = _raw_calc_objective(img)
            if _guard_ref_energy is None:
                return j, ratio
            loss = roi_energy_loss(_guard_ref_energy, _fixed_roi_energy(img))
            if loss > max_roi_energy_loss:
                _guard_violations += 1
                if _guard_violations <= 5 or _guard_violations % 50 == 0:
                    logger.warning(
                        "ROI energy loss {:.1%} exceeds the {:.1%} limit - "
                        "evaluation abandoned (#{} violations)",
                        loss,
                        max_roi_energy_loss,
                        _guard_violations,
                    )
                bad = j - 1e3 if objective_mode == "max" else j + 1e3
                return float(bad), float(ratio)
            return j, ratio

        j, pib_ratio = calc_objective(init_img)

        optimizer = _create_optimizer(
            optimizer_type=optimizer_type,
            dim=nk,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            **kwargs,
        )
        if lr == 0:
            optimizer.lr, delta = learning_schedule(
                radius(init_img, center=center, energy=0.8),
                gradient_history=_gradient_history,
                pib_history=_pib_history,
                epoch=0,
            )

        # Track the objective's OWN value. For "pib" that is the exposure-
        # independent bucket ratio (matches the logged column); for the other
        # objectives it is the value the gradient uses -- e.g. the encircle
        # radius, which must be MINIMISED (a `>` comparison would keep the worst).
        best_objective = float(test_pib(init_img)) if objective == "pib" else float(j)
        # Baseline objective of the initial phase (flat when ``init_c`` is
        # empty). Used on exit to decide between the best phase and flat.
        _initial_objective = best_objective
        best_c = _init_c.copy()
        last_best_epoch = 0

        recorder.append(
            {
                "J": j,
                objective: best_objective,
                "_p%": pib_ratio,
                "_max_r": _init_r,
                "_c": _init_c,
                "_img": init_img,
                "_diff": 0,
                "lr": optimizer.lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
                "exp_t": get_camera_exposure_ms(cam),
                "max_brt": np.max(init_img),
                "_grad": np.zeros_like(_init_c),
                "optimizer": optimizer_type,
                f"best_{objective}": best_objective,
            }
        )

        def _log_row(
            *,
            epoch: int,
            coeffs: np.ndarray,
            obj_val: float,
            obj_ratio: float,
            J: float,
            diff: float,
            grad: np.ndarray,
            img: np.ndarray,
            lr_val: float,
            delta_val: float,
            max_brt: float,
        ) -> dict:
            """Append one search step to the recorder (shared by both branches)."""
            row = {
                "J": J,
                "_p%": obj_ratio,
                "_max_r": _init_r,
                objective: obj_val,
                "_diff": diff,
                "lr": lr_val,
                "r": r_bucket,
                "delta": delta_val,
                "_epoch": epoch,
                "_c": coeffs,
                "_img": img,
                "exp_t": get_camera_exposure_ms(cam),
                "max_brt": max_brt,
                "_grad": grad,
                f"best_{objective}": best_objective,
            }
            recorder.append(row)
            return row

        def _apply_best_on_exit() -> None:
            """Leave the SLM at the best-found phase (or flat if never improved)."""
            recorder.energy_loss_violations = _guard_violations
            if not SLM_APPLY_BEST_ON_EXIT:
                return
            improved = (
                best_objective > _initial_objective + 1e-4
                if objective_mode == "max"
                else best_objective < _initial_objective - 1e-4
            )
            if improved:
                best_phase = slm.create_phase_from_array(
                    _zernike_to_phase(best_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, best_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                # Un-windowed (full-frame) before/after for the report: the metric
                # window hides the energy that leaves the box, so the illustrative
                # comparison must be captured on the raw sensor.
                try:
                    # ``reset_window`` keeps the box centred on ``center`` and
                    # rejects negative offsets, so the full sensor cannot be asked
                    # for directly - use the largest box that still fits around the
                    # centre (effectively un-windowed: ~96% of the frame here).
                    _fw, _fh = int(full_frame_shape[1]), int(full_frame_shape[0])
                    _raw_w = 2 * int(min(center_full[0], _fw - center_full[0]))
                    _raw_h = 2 * int(min(center_full[1], _fh - center_full[1]))
                    # Some SDK / pixel-format combinations cap the window (observed
                    # ``Width.range=[4,1680,4]``) and then silently keep the old
                    # window; shrink until the camera really returns a large frame.
                    for _attempt in range(4):
                        try:
                            cam.reset_window(center_full, (_raw_w, _raw_h))
                        except (RuntimeError, ValueError, AssertionError) as exc:
                            logger.warning(
                                "raw view {}x{} rejected by the camera: {}",
                                _raw_w,
                                _raw_h,
                                exc,
                            )
                        _probe = cam.get_numpy_image(CAM_SAMPLE_ITER)
                        logger.info(
                            "raw view {}x{} -> frame {}",
                            _raw_w,
                            _raw_h,
                            _probe.shape,
                        )
                        if min(_probe.shape) >= 400:
                            break
                        _raw_w = max(400, _raw_w // 2)
                        _raw_h = max(400, _raw_h // 2)
                    _display(slm, initial_phase)
                    time.sleep(SLM_RESPONSE_TIME_S)
                    recorder.raw_before_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                    _display(slm, best_phase)
                    time.sleep(SLM_RESPONSE_TIME_S)
                    recorder.raw_after_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                    recorder.raw_frame_shape = full_frame_shape
                except (RuntimeError, ValueError, AssertionError) as exc:
                    logger.warning("raw before/after capture failed: {}", exc)
                logger.info(
                    "SLM left at best {} phase: {:.4f} @ epoch {} (initial {:.4f})",
                    objective,
                    best_objective,
                    last_best_epoch,
                    _initial_objective,
                )
            else:
                slm.set_grayscale(0)
                logger.info(
                    "No {} improvement over the initial phase ({:.4f}); "
                    "SLM left at flat",
                    objective,
                    _initial_objective,
                )

        # ------------------------------------------------------------------
        # Heuristic search branch (shared driver: algorithm/heuristic/search.py).
        # The driver clips candidates to the bounds, handles the maximise/minimise
        # sign and aborts cooperatively when the live window is closed.
        # ------------------------------------------------------------------
        if algorithm != "spgd":
            last_eval: dict = {}

            def _evaluate_candidate(coeffs) -> float:
                """Display one candidate and return its RAW objective value."""
                candidate = np.asarray(coeffs, dtype=np.float64)
                candidate_phase = slm.create_phase_from_array(
                    _zernike_to_phase(candidate, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, candidate_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                if exposure_time_ms == 0 and float(np.max(img)) >= 255:
                    # Saturated: re-auto-expose to the requested target, mirroring
                    # the SPGD loop's guard, so the metric stays on a valid frame.
                    img = camera_auto_exposure(
                        cam, target_max_brightness or TEST_EXPOSURE_TIME_BRIGHTNESS
                    )
                obj, obj_ratio = calc_objective(img)
                obj_val = float(test_pib(img)) if objective == "pib" else float(obj)
                last_eval.update(
                    {
                        "phase": candidate_phase,
                        "img": img,
                        "obj": float(obj),
                        "ratio": float(obj_ratio),
                    }
                )
                return obj_val

            with tqdm.tqdm(
                total=None, desc=f"slm_zernike {algorithm}", dynamic_ncols=True
            ) as bar:

                def _on_evaluate(candidate, value, index) -> None:
                    nonlocal best_objective, best_c, last_best_epoch
                    improved = (
                        value > best_objective + 1e-4
                        if objective_mode == "max"
                        else value < best_objective - 1e-4
                    )
                    if improved:
                        best_objective = float(value)
                        best_c = np.asarray(candidate, dtype=np.float64).copy()
                        last_best_epoch = index

                    img = last_eval["img"]
                    row = _log_row(
                        epoch=index,
                        coeffs=candidate,
                        obj_val=float(value),
                        obj_ratio=last_eval["ratio"],
                        J=last_eval["obj"],
                        diff=0.0,
                        grad=np.zeros_like(candidate),
                        img=img,
                        lr_val=0.0,
                        delta_val=float(delta),
                        max_brt=float(np.max(img)),
                    )
                    if live_display is not None:
                        text = (
                            f"{algorithm} eval {index} | "
                            f"best {objective} {best_objective:.4f} @ {last_best_epoch} | "
                            f"J {row['J']:.3g} | r {r_bucket:.1f}"
                        )
                        live_display.update(
                            img,
                            last_eval["phase"],
                            candidate,
                            center,
                            r_bucket,
                            text,
                            value=float(value),
                            epoch=index,
                        )
                    bar.set_postfix({k: v for k, v in row.items() if k[0] != "_"})
                    bar.update(1)

                result = run_heuristic_search(
                    algorithm,
                    _evaluate_candidate,
                    dim=nk,
                    iterations=epochs,
                    bounds=(-ZERNIKE_CLIP, ZERNIKE_CLIP),
                    x0=_init_c,
                    maximize=(objective_mode == "max"),
                    seed=random_seed,
                    pop_size=pop_size,
                    on_evaluate=_on_evaluate,
                    should_stop=(
                        (lambda: live_display.closed)
                        if live_display is not None
                        else None
                    ),
                )

            if live_display is not None and live_display.closed:
                logger.info(
                    "Display window closed; stopped {} after {} evaluations",
                    algorithm,
                    result.evaluations,
                )
            else:
                logger.info(
                    "{} search finished: best {}={:.4f} over {} evaluations",
                    algorithm,
                    objective,
                    result.best_value,
                    result.evaluations,
                )
            _apply_best_on_exit()
            return recorder

        with tqdm.tqdm(
            total=epochs, desc=f"slm_zernike iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1, epochs + 1):
                # Generate random perturbation (±1 pattern)
                disturb_c = rng.binomial(1, 0.5, (nk,)).astype(float) * 2.0 - 1.0
                disturb_c = disturb_c * delta

                # Positive perturbation
                _pos_c = np.clip(_init_c + disturb_c, -5.0, 5.0)
                pos_phase = slm.create_phase_from_array(
                    _zernike_to_phase(_pos_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, pos_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_obj, pos_obj_ratio = calc_objective(pos_img)

                # Negative perturbation
                _neg_c = np.clip(_init_c - disturb_c, -5.0, 5.0)
                neg_phase = slm.create_phase_from_array(
                    _zernike_to_phase(_neg_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, neg_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_obj, neg_obj_ratio = calc_objective(neg_img)

                # Auto-exposure adjustment if saturated
                max_brightness = max([np.max(pos_img), np.max(neg_img)])
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = camera_auto_exposure(cam, target_max_brightness)
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

                pos_j, neg_j = pos_obj, neg_obj
                # `diff` is kept for logging; the SPGD sign comes from the
                # shared helper (optimizer/spgd.py) so it cannot be
                # hand-inverted again (this site maximised/minimised the wrong
                # way until the to_min fix). Cast to float: bucket sums are
                # unsigned.
                diff = (float(pos_j) - float(neg_j)) * to_min
                gradient = spgd_gradient(
                    pos_j, neg_j, disturb_c, maximize=(to_min == -1)
                )
                update = optimizer.update(gradient)
                _to_update_c = np.clip(_init_c - update, -5.0, 5.0)
                _init_c = _to_update_c

                # Value logged under the objective's own name and used by the
                # Recorder to pick its best row: the bucket ratio for "pib",
                # otherwise the objective the gradient optimises (e.g. radius).
                objective_val = (
                    float(test_pib(pos_img)) if objective == "pib" else float(pos_j)
                )
                objective_ratio = (pos_obj_ratio + neg_obj_ratio) / 2
                J = (pos_j + neg_j) / 2

                # Bucket radius shrink
                if epoch % update_iter == update_iter - 1:
                    _init_r = max(_init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                if (
                    (
                        epoch % update_iter == update_iter - 1
                        or (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1)
                        or objective_ratio >= 0.99
                    )
                    and not _fix_bucket
                    and objective_val > 0
                ):
                    power_radio = radius(pos_img, center=center, energy=0.8)
                    _pr = power_radio * shrink_ratio
                    _r = max(r_bucket * shrink_ratio + 1, IDEAL_SPOT_RADIUS, r_bucket)
                    r_bucket = min(_r, _pr, _init_r)
                    if lr == 0:
                        _grad_mag = float(np.linalg.norm(gradient))
                        _gradient_history.append(_grad_mag)
                        _pib_history.append(float(objective_val))
                        if len(_gradient_history) > _max_history_len:
                            _gradient_history.pop(0)
                            _pib_history.pop(0)
                        optimizer.lr, delta = learning_schedule(
                            power_radius=r_bucket,
                            gradient_history=_gradient_history,
                            pib_history=_pib_history,
                            epoch=epoch,
                        )

                # Track best result in the objective's own direction.
                improved = (
                    objective_val > best_objective + 1e-4
                    if objective_mode == "max"
                    else objective_val < best_objective - 1e-4
                )
                if improved:
                    best_objective = float(objective_val)
                    best_j = float(J)
                    best_objective_ratio = float(objective_ratio)
                    # objective_val / pos_img were measured on the PERTURBED
                    # phase `_pos_c`, so the saved coefficients must be `_pos_c`
                    # too. Saving the clean `_init_c` made `save_best` write a
                    # configuration that was never measured -- re-applying it
                    # did not reproduce the reported metric (hardware-verified).
                    best_c = _pos_c.copy()
                    best_img = pos_img.copy()
                    last_best_epoch = epoch

                log = _log_row(
                    epoch=epoch,
                    coeffs=_init_c,
                    obj_val=objective_val,
                    obj_ratio=objective_ratio,
                    J=J,
                    diff=diff,
                    grad=gradient,
                    img=pos_img,
                    lr_val=optimizer.lr,
                    delta_val=delta,
                    max_brt=float(max_brightness),
                )
                if live_display is not None:
                    text = (
                        f"epoch {epoch} | {objective} {objective_val:.4f} "
                        f"@ {last_best_epoch} | r {r_bucket:.1f} | lr {optimizer.lr:.3f}"
                    )
                    if not live_display.update(
                        pos_img,
                        pos_phase,
                        _pos_c,
                        center,
                        r_bucket,
                        text,
                        value=objective_val,
                        epoch=epoch,
                        total_epochs=epochs,
                    ):
                        logger.info(
                            "Display window closed; stopping SPGD at epoch {}", epoch
                        )
                        break

                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        # On exit, leave the SLM at the best phase found. The initial (flat or
        # loaded) phase is one of the candidates: if the search never improved
        # on it, restore that instead of a worse "best". Shared with the
        # heuristic branch through _apply_best_on_exit.
        _apply_best_on_exit()

        return recorder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SPGD PIB optimization using SLM with Zernike coefficients"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=2000, help="Number of iterations"
    )
    parser.add_argument(
        "-n", "--n_max", type=int, default=4, help="Max Zernike radial order"
    )
    parser.add_argument(
        "-c", "--center", type=str, default="shape", help="Center detection method"
    )
    parser.add_argument(
        "-r", "--r_bucket", type=float, default=0, help="Bucket radius (0=auto)"
    )
    parser.add_argument(
        "-d", "--delta", type=float, default=0.1, help="Perturbation amplitude"
    )
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=auto)")
    parser.add_argument(
        "-t", "--exposure_time_ms", type=float, default=80.0, help="Exposure time (ms)"
    )
    parser.add_argument(
        "--cam_id",
        type=str,
        default="0",
        help="Camera device ID (int) or path/URL for file-based backends",
    )
    parser.add_argument(
        "--cam_type",
        type=str,
        default="daheng",
        choices=list_camera_types(),
        help="Camera backend (registry type)",
    )
    parser.add_argument("--slm_number", type=int, default=1, help="SLM device number")
    parser.add_argument(
        "--slm_wavelength", type=int, default=1064, help="SLM wavelength (nm)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamod", help="Optimizer type"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="spgd",
        choices=list(ALGORITHM_CHOICES),
        help="Search algorithm: spgd (gradient) or a black-box heuristic",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=None,
        help="Population size for ga/pso/cem/de (default: heuristic default)",
    )
    parser.add_argument(
        "--objective", type=str, default="pib", help="Optimization target"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--show", action="store_true", help="Show images during optimization"
    )
    parser.add_argument("--cam_size", type=int, default=250, help="Camera window size")

    args = parser.parse_args()

    cam_id = (
        int(args.cam_id) if args.cam_id.strip().lstrip("+-").isdigit() else args.cam_id
    )

    recorder = optimize_slm_zernike_pib(
        center=args.center,
        epochs=args.epochs,
        n_max=args.n_max,
        r_bucket=args.r_bucket,
        delta=args.delta,
        lr=args.lr,
        exposure_time_ms=args.exposure_time_ms,
        cam_id=cam_id,
        cam_type=args.cam_type,
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        optimizer_type=args.optimizer,
        algorithm=args.algorithm,
        pop_size=args.pop_size,
        objective=args.objective,
        random_seed=args.seed,
        show=args.show,
        cam_size=args.cam_size,
    )

    best_iter, (_, best_val) = recorder.get_best_iter()
    logger.info(
        f"Optimization complete. Best {args.objective}: {best_val:.4f} @ epoch {best_iter.get('_epoch', 'N/A')}"
    )
    save_file = (
        gen_date_dir("data") / f"slm_zernike_{args.objective}_{gen_date_str()}.csv"
    )
    recorder.save_dataframe(save_file)
    logger.info(f"Results saved to: {save_file}")
