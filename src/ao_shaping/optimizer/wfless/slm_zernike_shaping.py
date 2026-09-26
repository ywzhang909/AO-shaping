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
    >>> from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    ...     SlmZernikePibConfig,
    ...     optimize_slm_zernike_pib,
    ... )
    >>> recorder = optimize_slm_zernike_pib(
    ...     SlmZernikePibConfig(center="shape", epochs=2000, delta=0.1)
    ... )
"""

from __future__ import annotations

import inspect
import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

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
    capture_with_exposure,
    create_camera,
    get_camera_exposure_ms,
    list_camera_types,
    resample_on_saturation,
    resolve_initial_exposure,
)
from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL
from ao_shaping.optimizer.spgd import spgd_gradient
from ao_shaping.utils import Recorder, logger
from ao_shaping.utils.image.beam_metrics import (
    clamp_center_to_frame,
    smart_zero_order_center,
    zero_order_center,
)
from ao_shaping.utils.image.hardware_utils import (
    log_center_brightness,
    resolve_spot_center,
)
from ao_shaping.utils.image.spots_calc import radius
from ao_shaping.utils.image.targets import (
    SHAPE_STAGE_WEIGHTS,
    TARGET_SHAPE_CHOICES,
    ShapeScoringParams,
    ShapingObjective,
    ShapingObjectiveParams,
    _resolve_init_weights,
    _update_dynamic_weights,
    create_target_shape,
    rms_pib_terms,
    rmse_shape_metric,
    roi_energy_loss,
    roi_pib_metric,
    shape_metric,
    shape_stage,
    shape_stage_from_energy,
    spot_waist_sigma,
    target_shape_roi,
)
from ao_shaping.utils.io.file import gen_date_dir, gen_date_str
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms
from ao_shaping.utils.wavefront.zernike_calc import noll_indices as _zernike_indices
from ao_shaping.utils.wavefront.zernike_utils import parse_zernike_coefficients

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
SLM_RESPONSE_TIME_S = 0.1
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


# ``resolve_initial_exposure`` and ``clamp_center_to_frame`` are imported from
# ``ao_shaping.drivers.ccd.common`` and ``ao_shaping.utils.image.beam_metrics``
# respectively (re-exported here for backward compatibility with tests and
# scripts that import them from this module). ``_update_dynamic_weights`` and
# ``_resolve_init_weights`` likewise now live in ``utils.image.targets`` next to
# the ``ShapingObjective`` that calls them, and are re-exported here unchanged.


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
    """Convert a flat Noll-order Zernike coefficient array to a
    canonical (n, m)→amplitude dict via `parse_zernike_coefficients`
    (skips |amp| < 1e-15 and n > n_max — both contribute zero phase).
    """
    coeffs_dict: dict[tuple[int, int], float] = parse_zernike_coefficients(
        coeffs, n_max=n_max
    )
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


if TYPE_CHECKING:
    from ao_shaping.runners.runner_common import CameraParamsPib, SlmParamsPib


def _default_camera() -> CameraParamsPib:
    """Create the canonical camera/objective group without a module-level cycle."""
    from ao_shaping.runners.runner_common import CameraParamsPib

    return CameraParamsPib()


def _default_slm() -> SlmParamsPib:
    """Create the canonical SLM group without a module-level cycle."""
    from ao_shaping.runners.runner_common import SlmParamsPib

    return SlmParamsPib()


@dataclass
class SlmZernikePibConfig:
    """Grouped configuration for :func:`optimize_slm_zernike_pib`.

    ``center`` and ``epochs`` are the two run-level switches that used to be
    positional arguments of the optimizer; they now live on the config so the
    optimizer takes a single parameter. The camera/objective fields live on
    :class:`CameraParamsPib` (``camera``) and the SLM/Zernike fields on
    :class:`SlmParamsPib` (``slm``); both default to the canonical runner
    groups (``camera.name == "pib"``, ``target_shape is None``, and
    ``slm.zernike_radius == 0.0`` as the sentinel for the optimizer's 300 px
    aperture). The two factories above defer the runner imports until
    instantiation because ``runners.__init__`` eagerly imports this
    optimizer's runner.

    Groups:
        Run: center / epochs.
        Search: algorithm / pop_size / random_seed / optimizer_type.
        Camera / objective: camera.
        SLM / Zernike: slm.
        Bucket / SPGD step: camera.r_bucket / delta / lr / shrink_iter / shrink_ratio.
        Recording: record_phase.
        Escape hatch: kwargs.
    """

    # --- Run ------------------------------------------------------------------
    center: str | tuple[int, int] | None
    epochs: int
    # --- Search ---------------------------------------------------------------
    algorithm: str = "spgd"
    pop_size: int | None = None
    random_seed: int | None = None
    # SPGD gradient optimizer (adam/adamw/adamod/sgd/muno/munow); ignored when
    # ``algorithm`` is a black-box heuristic.
    optimizer_type: str = "adamod"
    # --- Bucket / SPGD step ----------------------------------------------------
    # 0.2 rad, NOT 0.1: the measured noise floor of the shaping objective is
    # dJ_noise = 4e-4 and a 0.1 rad perturbation moves J by only 2.8e-4
    # (SNR 0.69 -> the SPGD gradient is noise). 0.2 rad gives SNR 2.77
    # (measured on the bench by scripts/measure_shape_sensitivity.py).
    delta: float = 0.2
    lr: float = 0
    shrink_iter: int = 0
    shrink_ratio: float = 0.9
    # --- Hardware / objective groups -------------------------------------------
    camera: CameraParamsPib = field(default_factory=_default_camera)
    slm: SlmParamsPib = field(default_factory=_default_slm)
    show: bool = False
    # --- Recording ---------------------------------------------------------------
    record_phase: bool = False
    # --- Escape hatch -----------------------------------------------------------
    #: Extra keyword arguments forwarded to the optimizer constructor
    #: (``_create_optimizer``); kept out of the typed fields.
    kwargs: dict[str, Any] = field(default_factory=dict)


def optimize_slm_zernike_pib(config: SlmZernikePibConfig):
    """Optimize PIB (Power in Bucket) using SLM with Zernike coefficient control.

    Zernike coefficients displayed on an SLM are searched to optimise an imaging
    objective measured by a camera. Two search families are available:
    ``algorithm="spgd"`` (Stochastic Parallel Gradient Descent, the default) or a
    black-box heuristic (``ga``/``pso``/``sa``/``hc``/``rs``/``cem``/``de``).

    Args:
        config: Grouped configuration for the run. ``center`` and ``epochs``
            are the run-level switches that used to be positional arguments
            (center detection method or fixed (x, y) tuple, and the number of
            optimization iterations). The camera/objective fields live on
            ``config.camera`` (:class:`CameraParamsPib`) and the SLM/Zernike
            fields on ``config.slm`` (:class:`SlmParamsPib`); ``config.kwargs``
            is forwarded to the optimizer constructor (``_create_optimizer``).

    Returns:
        Recorder: Optimization history recorder.
    """
    center = config.center
    epochs = config.epochs
    camera_config = config.camera
    slm_config = config.slm
    algorithm = config.algorithm
    pop_size = config.pop_size
    optimizer_type = config.optimizer_type
    r_bucket = camera_config.r_bucket
    delta = config.delta
    lr = config.lr
    exposure_time_ms = camera_config.exposure_time_ms
    shrink_iter = config.shrink_iter
    shrink_ratio = config.shrink_ratio
    show = config.show
    init_c = slm_config.init_c
    cam_size = camera_config.cam_size
    target_max_brightness = camera_config.target_max_brightness
    n_max = slm_config.n_max
    random_seed = config.random_seed
    objective = camera_config.name
    target_shape = camera_config.target_shape
    target_size = camera_config.target_size
    target_aspect_ratio = camera_config.target_aspect_ratio
    target_center_smooth = camera_config.target_center_smooth
    shape_schedule = camera_config.shape_schedule
    max_roi_energy_loss = camera_config.max_roi_energy_loss
    w_uniformity = camera_config.w_uniformity
    w_peak = camera_config.w_peak
    w_displacement = camera_config.w_displacement
    log_uniformity = camera_config.log_uniformity
    w_ema_decay = camera_config.w_ema_decay
    w_floor = camera_config.w_floor
    w_temperature = camera_config.w_temperature
    w_pib_init = camera_config.w_pib_init
    w_rms_init = camera_config.w_rms_init
    w_ee_init = camera_config.w_ee_init
    record_phase = config.record_phase
    zernike_radius = slm_config.zernike_radius
    if zernike_radius is None or zernike_radius <= 0:
        zernike_radius = ZERNIKE_APERTURE_RADIUS

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
    if target_shape is not None and objective not in (
        "pib",
        "rmse",
        "shape",
        "roi_pib",
        "rms_pib",
    ):
        raise ValueError(
            "target_shape can only be used with objective='pib', 'rmse', "
            "'shape', 'roi_pib' or 'rms_pib'"
        )
    if target_shape is not None and objective not in ("roi_pib", "rms_pib", "rmse"):
        # Supplying target_shape implies the dynamic-ROI shaping objective, except
        # for roi_pib/rms_pib/rmse where the shape only selects the target ROI.
        objective = "shape"
    if objective in ("shape", "roi_pib", "rms_pib", "rmse") and target_shape is None:
        target_shape = "rectangle"
    shape_for_metric = cast(TargetShape, target_shape or "rectangle")
    if objective not in (
        "pib",
        "radiu",
        "avg_radiu",
        "rmse",
        "shape",
        "roi_pib",
        "rms_pib",
    ):
        raise ValueError(
            f"objective must be one of ('pib', 'radiu', 'avg_radiu', 'rmse', "
            f"'shape', 'roi_pib', 'rms_pib'), got {objective}"
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
    if not 0.0 <= w_ema_decay <= 1.0:
        raise ValueError(f"w_ema_decay must be within 0..1, got {w_ema_decay!r}")
    if not 0.0 <= w_floor < 0.5:
        raise ValueError(f"w_floor must be within 0..0.5 (exclusive), got {w_floor!r}")
    if not np.isfinite(w_temperature) or w_temperature <= 0.0:
        raise ValueError(f"w_temperature must be positive, got {w_temperature!r}")
    for _name, _w in (
        ("w_pib_init", w_pib_init),
        ("w_rms_init", w_rms_init),
        ("w_ee_init", w_ee_init),
    ):
        if _w is not None and (not np.isfinite(_w) or _w < 0.0):
            raise ValueError(
                f"{_name} must be a finite, non-negative weight, got {_w!r}"
            )

    # Optimization mode mapping: pib, avg_radiu, shape, roi_pib and rms_pib are
    # maximized; radiu and rmse are minimized.
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
            raise ValueError(
                f"{_name} must be a finite, non-negative weight, got {_w!r}"
            )
    objective_mode = (
        "max"
        if objective in ("pib", "avg_radiu", "shape", "roi_pib", "rms_pib")
        else "min"
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
        _curve_y_range = (
            (0.0, 1.0) if objective in ("pib", "roi_pib", "rms_pib") else None
        )
        # Target shape is only meaningful for the shape/roi_pib/rms_pib/rmse
        # objectives; for pib/radiu/avg_radiu the bucket circle is the correct overlay.
        _display_shape = (
            shape_for_metric
            if objective in ("shape", "roi_pib", "rms_pib", "rmse")
            else None
        )
        display_ctx = SlmZernikeDisplay(
            zernike_clip=ZERNIKE_CLIP,
            curve_title=f"{objective} curve",
            curve_y_range=_curve_y_range,
            target_shape=_display_shape,
            target_aspect_ratio=target_aspect_ratio,
        )
    else:
        display_ctx = nullcontext(None)

    # TODO： 先跑一次离线 gs() 把结果作为闭环初值，通常 3~5 次就能收敛，不用每次从零迭代

    cam_ctx = create_camera(config.camera)
    slm_ctx = Santec.from_params(config.slm)

    with cam_ctx as cam, slm_ctx as slm, display_ctx as live_display:
        assert cam is not None and slm is not None
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
        _img = capture_with_exposure(
            cam,
            exposure_time_ms,
            TEST_EXPOSURE_TIME_BRIGHTNESS,
            n_sample=CAM_SAMPLE_ITER,
        )

        # Resolve the centre spec (None / "mass" / "max" / "shape" / tuple).
        # When centre is None, ``smart_zero_order_center`` auto-detects via
        # argmax + flat-core-aware local centroid; when a string, a fresh frame
        # is re-captured and dispatched by name.
        center = resolve_spot_center(cam, _img, center, n_sample=10)
        log_center_brightness(_img, center, get_camera_exposure_ms(cam))

        img_size = (cam_size, cam_size)
        # Keep the ROI inside the frame: DahengCamera.reset_window rejects
        # negative offsets, so a spot near an edge would abort the run.
        center = clamp_center_to_frame(center, _img.shape, cam_size)
        # Full-frame geometry: needed to re-window (enlargement) and to capture
        # the report's un-windowed before/after frames later on.
        center_full = (float(center[0]), float(center[1]))
        full_frame_shape: tuple[int, int] = (
            int(_img.shape[0]),
            int(_img.shape[1]),
        )
        img_size, center = cam.reset_window(center, img_size)
        logger.info(f"reset window center @ {center}")

        # Precedence: fixed exposure (`--exposure_time_ms > 0`) wins over
        # auto-exposure (`--target_max_brightness > 0`); `0/0` keeps the current
        # exposure ("若为0则不自动调整曝光时间", matching the runner help).
        init_img = capture_with_exposure(
            cam,
            exposure_time_ms,
            target_max_brightness,
            n_sample=CAM_SAMPLE_ITER,
        )
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

        # ``reset_window`` returns the window centre in FULL-FRAME sensor
        # coordinates, but every downstream metric (``rms_pib_terms``, waist,
        # radius) operates on the re-windowed ``init_img``. The 0-order spot is
        # therefore re-located inside the window here - this also absorbs the
        # runner's auto exposure/centre probe, which likewise returns a
        # full-frame centre (un-windowed probe camera). Operator-supplied
        # full-frame centres fix the window position, and the freshly positioned
        # window then has the spot at its argmax, so re-locating stays correct
        # for them too.
        _spot = zero_order_center(init_img)
        reference_center: tuple[float, float] = (float(_spot[0]), float(_spot[1]))
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
            # ``reference_center`` is the window-local spot position (see
            # above); ``center`` here holds the FULL-FRAME window centre
            # returned by ``reset_window`` and would be out of bounds for the
            # windowed ``init_img`` - the 99%-entcircled radius must be
            # anchored on the re-located spot.
            r_bucket = ImageTargetFunc(_w, _h, reference_center).radius(
                init_img, energy=0.99
            )
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

        # --- Objective ----------------------------------------------------------
        # The seven objective families, the adaptive ``rms_pib`` term weighting,
        # the in-ROI energy-loss safety guard and the cross-objective ``m_*``
        # metric panel all live in ``ShapingObjective``
        # (utils/image/targets.py). It owns the objective selection, the FIXED
        # target ROI, the ``rms_pib``/panel energy baselines and the live bucket
        # radius (kept in sync through ``set_bucket`` below), so the search loops
        # only hand it frames and read ``ObjectiveResult.j`` / ``.ratio``.
        shaping = ShapingObjective(
            ShapingObjectiveParams(
                objective=objective,
                mode=objective_mode,
                shape=shape_for_metric,
                size=target_size,
                aspect_ratio=target_aspect_ratio,
                reference_center=reference_center,
                shape_schedule=shape_schedule,
                scoring=ShapeScoringParams(
                    w_uniformity=w_uniformity,
                    w_peak=w_peak,
                    w_displacement=w_displacement,
                    log_uniformity=log_uniformity,
                ),
                max_roi_energy_loss=max_roi_energy_loss,
                ideal_spot_radius=IDEAL_SPOT_RADIUS,
                r_bucket=r_bucket,
                w_ema_decay=w_ema_decay,
                w_floor=w_floor,
                w_temperature=w_temperature,
                w_pib_init=w_pib_init,
                w_rms_init=w_rms_init,
                w_ee_init=w_ee_init,
            ),
            target_func,
            init_img,
        )
        _init_res = shaping(init_img)
        j, pib_ratio = _init_res.j, _init_res.ratio

        optimizer = _create_optimizer(
            optimizer_type=optimizer_type,
            dim=nk,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            **config.kwargs,
        )
        if lr == 0:
            optimizer.lr, delta = learning_schedule(
                radius(init_img, center=center, energy=0.8),
                gradient_history=_gradient_history,
                pib_history=_pib_history,
                epoch=0,
            )

        # Track the objective's OWN value. For "pib" that is the exposure-
        # independent bucket ratio at the FIXED ideal radius (matches the logged
        # column); for the other objectives it is the value the gradient uses --
        # e.g. the encircle radius, which must be MINIMISED (a `>` comparison
        # would keep the worst).
        best_objective = shaping.tracking_value(init_img, _init_res)
        # Baseline objective of the initial phase (flat when ``init_c`` is
        # empty). Used on exit to decide between the best phase and flat.
        _initial_objective = best_objective
        best_c = _init_c.copy()
        last_best_epoch = 0

        _row0 = {
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
        if objective == "rms_pib":
            _w_pib, _w_rms, _w_ee = shaping.weights
            _row0["w_pib"] = float(_w_pib)
            _row0["w_rms"] = float(_w_rms)
            _row0["w_ee"] = float(_w_ee)
            _terms = shaping.terms
            _row0["pib_term"] = float(_terms[1])
            _row0["rms_term"] = float(_terms[2])
            _row0["ee_term"] = float(_terms[3])
        _row0.update(shaping.metric_panel(init_img))
        if record_phase:
            _row0["_phase"] = initial_phase
        recorder.append(_row0)

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
            phase: np.ndarray | None = None,
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
            if record_phase and phase is not None:
                # Display-ready grayscale exactly as sent to the SLM device.
                row["_phase"] = phase
            if objective == "rms_pib":
                _w_pib, _w_rms, _w_ee = shaping.weights
                row["w_pib"] = float(_w_pib)
                row["w_rms"] = float(_w_rms)
                _terms = shaping.terms
                row["pib_term"] = float(_terms[1])
                row["rms_term"] = float(_terms[2])
            row.update(shaping.metric_panel(img))
            recorder.append(row)
            return row

        def _apply_best_on_exit() -> None:
            """Leave the SLM at the best-found phase (or flat if never improved)."""
            setattr(recorder, "energy_loss_violations", shaping.guard_violations)
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
                            cam.reset_window(
                                cast(tuple[int, int], center_full), (_raw_w, _raw_h)
                            )
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
                    setattr(
                        recorder,
                        "raw_before_img",
                        cam.get_numpy_image(CAM_SAMPLE_ITER),
                    )
                    _display(slm, best_phase)
                    time.sleep(SLM_RESPONSE_TIME_S)
                    setattr(
                        recorder,
                        "raw_after_img",
                        cam.get_numpy_image(CAM_SAMPLE_ITER),
                    )
                    setattr(recorder, "raw_frame_shape", full_frame_shape)
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
                    img = resample_on_saturation(
                        img,
                        cam,
                        exposure_time_ms,
                        target_max_brightness or TEST_EXPOSURE_TIME_BRIGHTNESS,
                    )
                res = shaping(img)
                obj, obj_ratio = res.j, res.ratio
                if objective == "rms_pib":
                    # Adapt the PIB/RMS/EE weights on every valid candidate
                    # evaluation (abandoned evaluations are penalised to <= -100
                    # and skipped). The result is passed explicitly so the
                    # adaptation uses THIS candidate's terms.
                    if float(obj) > -100.0:
                        shaping.adapt_weights(res)
                obj_val = shaping.tracking_value(img, res)
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
                        phase=last_eval.get("phase"),
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
                            target_size=target_size if target_size else None,
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
                pos_res = shaping(pos_img)
                pos_obj, pos_obj_ratio = pos_res.j, pos_res.ratio

                # Negative perturbation
                _neg_c = np.clip(_init_c - disturb_c, -5.0, 5.0)
                neg_phase = slm.create_phase_from_array(
                    _zernike_to_phase(_neg_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, neg_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_res = shaping(neg_img)
                neg_obj, neg_obj_ratio = neg_res.j, neg_res.ratio

                # Auto-exposure adjustment if saturated
                max_brightness = max([np.max(pos_img), np.max(neg_img)])
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = resample_on_saturation(
                        pos_img,
                        cam,
                        exposure_time_ms,
                        target_max_brightness,
                        saturation_threshold=float(max_brightness),
                    )
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

                pos_j, neg_j = pos_obj, neg_obj
                # `diff` is kept for logging; the SPGD sign comes from the
                # shared helper (optimizer/spgd.py) so it cannot be
                # hand-inverted again (this site maximised/minimised the wrong
                # way until the objective_mode fix). Cast to float: bucket sums
                # are unsigned.
                _spgd_sign = -1.0 if objective_mode == "max" else 1.0
                diff = (float(pos_j) - float(neg_j)) * _spgd_sign
                gradient = spgd_gradient(
                    pos_j, neg_j, disturb_c, maximize=(objective_mode == "max")
                )
                update = optimizer.update(gradient)
                _to_update_c = np.clip(_init_c - update, -5.0, 5.0)
                _init_c = _to_update_c

                # Value logged under the objective's own name and used by the
                # Recorder to pick its best row: the bucket ratio for "pib",
                # otherwise the objective the gradient optimises (e.g. radius).
                objective_val = shaping.tracking_value(pos_img, pos_res)
                objective_ratio = (pos_obj_ratio + neg_obj_ratio) / 2
                J = (pos_j + neg_j) / 2

                if objective == "rms_pib" and pos_obj > -100.0 and neg_obj > -100.0:
                    # Adapt the PIB/RMS/EE weights from the POSITIVE-perturbation
                    # result (abandoned evaluations are penalised to <= -100 and
                    # skipped). ``pos_res`` is an immutable snapshot, so the
                    # negative evaluation above cannot overwrite the terms - the
                    # adaptation therefore uses the direction the gradient
                    # actually follows. The term that improves J more gets the
                    # higher weight.
                    shaping.adapt_weights(pos_res)

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
                    # The objective reads the live bucket radius; keep it in sync.
                    shaping.set_bucket(r_bucket)
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
                    phase=pos_phase,
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
                        target_size=target_size if target_size else None,
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

    def _build_demo_config(args, cam_id: int | str) -> SlmZernikePibConfig:
        from ao_shaping.runners.runner_common import CameraParamsPib, SlmParamsPib

        return SlmZernikePibConfig(
            center=args.center,
            epochs=args.epochs,
            algorithm=args.algorithm,
            pop_size=args.pop_size,
            optimizer_type=args.optimizer,
            delta=args.delta,
            lr=args.lr,
            random_seed=args.seed,
            show=args.show,
            camera=CameraParamsPib(
                name=args.objective,
                cam_id=cast(int, cam_id),
                cam_type=args.cam_type,
                cam_size=args.cam_size,
                exposure_time_ms=args.exposure_time_ms,
                r_bucket=args.r_bucket,
            ),
            slm=SlmParamsPib(
                n_max=args.n_max,
                slm_number=args.slm_number,
                slm_wavelength=args.slm_wavelength,
            ),
        )

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

    recorder = optimize_slm_zernike_pib(_build_demo_config(args, cam_id))

    best_iter, (_, best_val) = recorder.get_best_iter()
    logger.info(
        f"Optimization complete. Best {args.objective}: {best_val:.4f} @ epoch {best_iter.get('_epoch', 'N/A')}"
    )
    save_file = (
        gen_date_dir("data") / f"slm_zernike_{args.objective}_{gen_date_str()}.csv"
    )
    recorder.save_dataframe(save_file)
    logger.info(f"Results saved to: {save_file}")
