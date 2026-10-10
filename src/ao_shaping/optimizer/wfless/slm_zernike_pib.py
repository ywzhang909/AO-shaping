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
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

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
    full_scale,
    get_camera_exposure_ms,
    is_saturated,
    list_camera_types,
    resample_on_saturation,
    resolve_initial_exposure,
)
from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL, PANEL_RES
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
    # ``TARGET_SHAPE_CHOICES`` is no longer referenced directly in this module
    # (ObjectiveSpec.resolve validates the shape), but it is re-exported here for
    # backward compatibility with existing importers/tests.
    DEFAULT_SHAPE_OBJECTIVES,
    SHAPE_STAGE_WEIGHTS,
    TARGET_SHAPE_CHOICES,
    ObjectiveResult,
    ObjectiveSpec,
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
from ao_shaping.optimizer.constants import create_optimizer

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
SLM_RESPONSE_TIME_S = 0.0
# On exit, leave the best-found phase on the SLM. The candidate set includes
# the initial (flat when starting from zeros, or a loaded) phase: if the search
# never improved on it, that phase is restored instead. Set False to skip
# touching the SLM on exit.
SLM_APPLY_BEST_ON_EXIT = True

# SLM resolution. Sourced from the driver rather than repeated here: the literal
# was a third copy of the panel geometry (the driver has PANEL_RES, exposed as
# Santec.Panel_Res). The names stay because tests import them.
SLM_WIDTH, SLM_HEIGHT = PANEL_RES
SLM_RESOLUTION = tuple(PANEL_RES)


# Search-family selection: "spgd" runs the SPGD gradient loop below; every other
# name is a black-box heuristic handled by the shared driver in
# ``ao_shaping.algorithm.heuristic.search`` (see ``run_heuristic_search``).
ALGORITHM_CHOICES = heuristic_algorithm_choices()

# Symmetric clip applied to every candidate Zernike vector before it is turned
# into a phase (matches the +/-5 clip the SPGD loop historically used).
ZERNIKE_CLIP = 5.0

#: A "best" comparison only counts as an improvement when it beats the
#: incumbent by more than this, so measurement jitter cannot install a new
#: best (and the on-exit decision cannot flip on noise).
IMPROVE_EPS = 1e-4


# ``resolve_initial_exposure`` and ``clamp_center_to_frame`` are imported from
# ``ao_shaping.drivers.ccd.common`` and ``ao_shaping.utils.image.beam_metrics``
# respectively (re-exported here for backward compatibility with tests and
# scripts that import them from this module). ``_update_dynamic_weights`` and
# ``_resolve_init_weights`` likewise now live in ``utils.image.targets`` next to
# the ``ShapingObjective`` that calls them, and are re-exported here unchanged.




ZERNIKE_APERTURE_RADIUS = 300.0
"""Aperture radius (px) the Zernike phase is defined over.

Must match the illuminated beam radius on the SLM. The panel is 1920x1200, so
PatternHelper's default is half the short side = 600 px; this bench's beam
radius is only ~300 px. With a 600 px aperture only the inner half of the
polynomial lands on the beam, so every mode is nearly CONSTANT across the
illuminated area -- and a constant phase does not change the far field, i.e.
the correction silently does nothing (hardware-verified: flat vs a 2 rad
defocus were indistinguishable until this was matched).
"""

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


# Memory-slot range used for phase writes (never repeat a slot consecutively).
_SLOT_MIN, _SLOT_MAX = 2, 125
_SLOT_STATE = {"slot": _SLOT_MIN - 1}


def _spgd_capture_signs(abba: bool) -> tuple[int, ...]:
    """Capture order as +1/-1 signs: ``(1, -1)`` normally, ``(1, -1, -1, 1)``
    for ABBA. The palindrome cancels linear slow drift in the sign-means."""
    return (1, -1, -1, 1) if abba else (1, -1)


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
    ``slm.zernike_radius == 600.0`` = SLM 面板短边的一半 as the default aperture
    (与方形整形/GUI 一致, 基圆完整落在面板内), falling back to the optimizer's
    300 px aperture only when it is 0/None). The two factories above defer the
    runner imports until
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
    # --- Evaluation robustness (2026-09 q3, from the delta<0.001 fold/jitter
    # post-mortem; see docs/slm_pib) -----------------------------------------
    # Per-evaluation frame average passed to ``cam.get_numpy_image`` (the
    # Daheng driver averages ``n_sample`` frames, so sigma_J falls ~1/sqrt(N)).
    # 1 = single frame (legacy behaviour). Applies to the SPGD pair frames.
    n_eval_frames: int = 1
    # Brightness-fold integrity gate: an evaluation frame whose peak OR total
    # sum falls below ``fold_ratio`` of the EMA baseline of recent VALID frames
    # is an environment brightness fold (measured discrete brightness states,
    # corr(J, max_brt) = -0.9996 on the bench) - that epoch's update and
    # best-track are skipped entirely. 0 disables.
    fold_ratio: float = 0.5
    # Measured-peak feasibility gate. This is NOT a safety interlock and NOT a
    # projection onto the feasible set - see the long warning below. It refuses
    # to COMMIT the next update once frames that were ALREADY displayed on the
    # SLM exceeded ``max_peak`` camera counts. 0 disables it, which is the
    # default so existing runs stay bit-identical.
    #
    # Why a gate and not a penalty term: a soft ``w_peak * max(0, peak-cap)**2``
    # term has a crossover failure - a step that gains 0.2 merit will happily
    # sit at 3x the cap because 3 penalty units < 0.2 merit units - and it
    # couples a dimensionless image metric to physical units through a constant
    # nobody can calibrate. A feasibility filter has no such trade.
    #
    # ``max_peak`` is a camera-window count under the CURRENT exposure, so it is
    # only meaningful with a FIXED exposure: with ``exposure_time_ms == 0`` the
    # auto-exposure path lowers exposure when a frame approaches saturation, so
    # counts fall while optical power does not and the cap is silently masked.
    # The runner warns when that combination is requested.
    max_peak: float = 0.0
    # Noise-aware update gate: when |J+ - J-| <= ``noise_gate_k`` * sigma_hat
    # (rolling std of the last ``noise_gate_window`` diffs) the SPGD gradient
    # is measurement noise (measured SNR < 0.1 at delta < 0.001) - the update
    # is zeroed so the coefficients stall honestly instead of random-walking.
    # 0 disables (negative k disables too).
    noise_gate_k: float = 3.0
    noise_gate_window: int = 20
    # ABBA sampling: capture 4 frames per epoch in the order `+ - - +` instead of
    # 2 (`+ -`). A drift that is linear in time contributes the same term to both
    # sign-means of a palindrome, so it cancels out of the SPGD difference. Off
    # by default so the 2-capture path stays byte-identical.
    abba_sampling: bool = False
    # --- Hardware / objective groups -------------------------------------------
    camera: CameraParamsPib = field(default_factory=_default_camera)
    slm: SlmParamsPib = field(default_factory=_default_slm)
    show: bool = False
    # --- Recording ---------------------------------------------------------------
    record_phase: bool = False
    # --- Escape hatch -----------------------------------------------------------
    #: Extra keyword arguments forwarded to the optimizer constructor
    #: (``create_optimizer``); kept out of the typed fields.
    kwargs: dict[str, Any] = field(default_factory=dict)


def _frame_fold_check(
    frame: np.ndarray,
    baseline_peak: float | None,
    baseline_sum: float | None,
    fold_ratio: float,
) -> tuple[bool, float, float]:
    """Brightness-fold integrity check for one evaluation frame.

    Returns ``(is_fold, peak, frame_sum)``. ``is_fold`` is True when the frame
    peak or its total sum falls below ``fold_ratio`` of the rolling baseline
    of VALID frames (armed from the initial frame; see
    :func:`_update_fold_baseline`). A folded frame is an environment event
    (measured discrete brightness states with ``corr(J, max_brt) = -0.9996``
    on the bench), NOT a coefficient effect - the epoch must be skipped before
    it can masquerade as a gradient. ``fold_ratio <= 0`` (or no baseline yet)
    disables the check.
    """
    if fold_ratio <= 0.0 or baseline_peak is None or baseline_sum is None:
        return False, float(np.max(frame)), float(np.asarray(frame, dtype=np.float64).sum())
    peak = float(np.max(frame))
    frame_sum = float(np.asarray(frame, dtype=np.float64).sum())
    is_fold = peak < fold_ratio * baseline_peak or frame_sum < fold_ratio * baseline_sum
    return is_fold, peak, frame_sum


def _update_fold_baseline(
    baseline_peak: float | None,
    baseline_sum: float | None,
    peak: float,
    frame_sum: float,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """EMA-update the fold baseline with one VALID frame (``alpha``: 0.1).

    Only validated (non-folded) frames may update the baseline, so a folded
    measurement can never pull the baseline down and blind the gate.
    """
    if baseline_peak is None or baseline_sum is None:
        return peak, frame_sum
    return (
        alpha * peak + (1.0 - alpha) * baseline_peak,
        alpha * frame_sum + (1.0 - alpha) * baseline_sum,
    )


def _rolling_sigma(diffs: Sequence[float]) -> float:
    """Std of the recent ``diff`` history (0.0 when degenerate/empty)."""
    if len(diffs) < 2:
        return 0.0
    arr = np.asarray(diffs, dtype=np.float64)
    s = float(np.std(arr))
    return s if np.isfinite(s) else 0.0


def _noise_gate(diff: float, sigma_hat: float, k: float) -> bool:
    """True when ``diff`` is indistinguishable from the recent diff noise
    (``|diff| <= k * sigma_hat``) - the SPGD update would be pure measurement
    noise and is zeroed so the coefficients stall honestly instead of
    random-walking. ``k <= 0`` (or an unusable ``sigma_hat``) disables the gate.
    """
    if k <= 0.0 or not np.isfinite(sigma_hat) or sigma_hat <= 0.0:
        return False
    return abs(diff) <= k * sigma_hat


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
            is forwarded to the optimizer constructor (``create_optimizer``).

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
    w_pearson = camera_config.w_pearson
    w_coverage = camera_config.w_coverage
    w_loggrad = camera_config.w_loggrad
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

    # Single authority for the objective/target-shape pairing. The rules used to
    # be re-implemented inline here, which had already drifted from
    # ``objective.py``: it omitted ``rmse_out`` entirely and hard-coded its own
    # (shorter) shape-aware list, so any newly registered objective raised
    # ``ValueError`` here even though ``ObjectiveSpec.resolve`` accepted it.
    spec = ObjectiveSpec.resolve(objective, target_shape)
    objective, target_shape = spec.name, spec.shape
    shape_for_metric = cast(TargetShape, target_shape or "rectangle")
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
        ("w_pearson", w_pearson),
        ("w_coverage", w_coverage),
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
        # Target shape is only meaningful for the objectives that score against a
        # target ROI; for pib/radiu/avg_radiu the bucket circle is the correct
        # overlay. Use the shared tuple: a hand-written copy here had already
        # drifted from the leaf (it was missing ``rmse_out`` and ``pearson``), and
        # the sibling module carried a third, different spelling of the same list.
        _display_shape = (
            shape_for_metric if objective in DEFAULT_SHAPE_OBJECTIVES else None
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
            n_sample=config.n_eval_frames,
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
        logger.info("reset window center @ {}", center)

        # Precedence: fixed exposure (`--exposure_time_ms > 0`) wins over
        # auto-exposure (`--target_max_brightness > 0`); `0/0` keeps the current
        # exposure ("若为0则不自动调整曝光时间", matching the runner help).
        init_img = capture_with_exposure(
            cam,
            exposure_time_ms,
            target_max_brightness,
            n_sample=config.n_eval_frames,
        )
        logger.debug(
            "Initial Image Max brightness: {} @ {}ms",
            np.max(init_img),
            get_camera_exposure_ms(cam),
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
            init_img = cam.get_numpy_image(config.n_eval_frames)
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
            logger.info("Use dynamic target size @ {:.1f}px", target_size)

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
            logger.info("Use dynamic radiu @ {}", r_bucket)
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
                    w_pearson=w_pearson,
                    w_coverage=w_coverage,
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
                w_loggrad=w_loggrad,
            ),
            target_func,
            init_img,
        )
        _init_res = shaping(init_img)
        j, pib_ratio = _init_res.j, _init_res.ratio

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            dim=nk,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            **config.kwargs,
        )
        if lr == 0:
            # ``reference_center`` (window-local spot), NOT ``center``: the
            # latter is the FULL-FRAME window centre ``reset_window`` returned,
            # which is out of bounds for the windowed ``init_img`` and made the
            # 80%-encircled radius meaningless. Same rule as the ``r_bucket``
            # block above and the SPGD branch's ``pos_center``.
            optimizer.lr, delta = learning_schedule(
                radius(init_img, center=reference_center, energy=0.8),
                gradient_history=_gradient_history,
                pib_history=_pib_history,
                epoch=0,
            )

        # Track the objective's OWN value. For "pib" that is the exposure-
        # independent bucket ratio at the FIXED ideal radius (matches the logged
        # column); for the other objectives it is the value the gradient uses --
        # e.g. the encircle radius, which must be MINIMISED (a `>` comparison
        # would keep the worst).
        best_objective = _init_res.tracking
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
            _row0["_full_frame_img"] = _img
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
            gate: str | None = None,
            phase: np.ndarray | None = None,
            loggrad: float = 0.0,
        ) -> dict:
            """Append one search step to the recorder (shared by both branches).

            ``gate`` records the SPGD evaluation verdict for offline stats:
            ``"applied"`` (gradient adopted), ``"fold"`` (brightness fold
            rejected before scoring), ``"noise"`` (diff below the noise gate,
            zeroed update) or ``"peak"`` (measured peak exceeded ``max_peak``,
            update not committed). ``None`` (heuristic branch, or legacy
            records) means "no verdict recorded".
            """
            row = {
                "J": J,
                "_p%": obj_ratio,
                "_max_r": _init_r,
                objective: obj_val,
                "_diff": diff,
                "_gate": gate,
                "lr": lr_val,
                "r": r_bucket,
                "delta": delta_val,
                "_epoch": epoch,
                "_c": coeffs,
                "_img": img,
                "exp_t": get_camera_exposure_ms(cam),
                "max_brt": max_brt,
                "_grad": grad,
                "optimizer": optimizer_type,
                f"best_{objective}": best_objective,
            }
            if loggrad != 0.0 or w_loggrad > 0.0:
                # Structure term. Recorded ONLY when the feature is in use: the
                # default recorder schema is a pinned contract (epoch rows may
                # add exactly "_gate" over the baseline row), so a default run
                # must keep its column set byte-identical.
                row["loggrad"] = float(loggrad)
            if record_phase and phase is not None:
                # Display-ready grayscale exactly as sent to the SLM device.
                row["_phase"] = phase
            if objective == "rms_pib":
                _w_pib, _w_rms, _w_ee = shaping.weights
                row["w_pib"] = float(_w_pib)
                row["w_rms"] = float(_w_rms)
                row["w_ee"] = float(_w_ee)
                _terms = shaping.terms
                row["pib_term"] = float(_terms[1])
                row["rms_term"] = float(_terms[2])
                row["ee_term"] = float(_terms[3])
            row.update(shaping.metric_panel(img))
            recorder.append(row)
            return row

        def _apply_best_on_exit() -> None:
            """Leave the SLM at the best-found phase (or flat if never improved)."""
            # Run-level scalar, deliberately an attribute on the Recorder and NOT a
            # row column: ``Recorder.append`` unions per-epoch record keys, so a
            # once-per-run summary would either never appear or appear on the last
            # row only, and ``Recorder`` has no schema for run-level metadata. The
            # single consumer reads it defensively as
            # ``getattr(rec, "energy_loss_violations", 0)``.
            recorder.energy_loss_violations = shaping.guard_violations
            # R8: this runs from a ``finally``. Two consequences, both
            # handled here rather than at the call site: raising would mask
            # the exception that got us here, and the camera may already be
            # dead (the commonest cause), so every device touch below is
            # guarded and the last resort is a flat phase, never a random one.
            try:
                if not SLM_APPLY_BEST_ON_EXIT:
                    return
                improved = (
                    best_objective > _initial_objective + IMPROVE_EPS
                    if objective_mode == "max"
                    else best_objective < _initial_objective - IMPROVE_EPS
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
                            cam.get_numpy_image(config.n_eval_frames),
                        )
                        _display(slm, best_phase)
                        time.sleep(SLM_RESPONSE_TIME_S)
                        setattr(
                            recorder,
                            "raw_after_img",
                            cam.get_numpy_image(config.n_eval_frames),
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

            except BaseException as _exit_exc:  # noqa: BLE001
                # Never let cleanup raise: a masked exception loses the
                # diagnosis, and an SLM left on a random phase damages the
                # next run. Flat is always safe.
                logger.error(
                    "best-phase restore failed ({}: {}); forcing flat phase",
                    type(_exit_exc).__name__,
                    _exit_exc,
                )
                try:
                    slm.set_grayscale(0)
                    logger.warning("SLM forced to flat (gray 0) on cleanup failure")
                except Exception:  # noqa: BLE001 - last resort, cannot raise
                    logger.error("could not force flat phase either; SLM state unknown")
        # ------------------------------------------------------------------
        # Heuristic search branch (shared driver: algorithm/heuristic/search.py).
        # The driver clips candidates to the bounds, handles the maximise/minimise
        # sign and aborts cooperatively when the live window is closed.
        # ------------------------------------------------------------------
        try:
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
                    img = cam.get_numpy_image(config.n_eval_frames)
                    if exposure_time_ms == 0 and is_saturated(img):
                        # Saturated: re-auto-expose to the requested target, mirroring
                        # the SPGD loop's guard, so the metric stays on a valid frame.
                        img = resample_on_saturation(
                            img,
                            cam,
                            exposure_time_ms,
                            target_max_brightness or TEST_EXPOSURE_TIME_BRIGHTNESS,
                        )
                    # Re-locate the target ROI onto the CURRENT spot (a benign beam
                    # drift must not be scored as a shaping loss) - same rule as the
                    # SPGD loop's per-eval re-centering.
                    shaping.set_reference_center(zero_order_center(img))
                    res = shaping(img)
                    obj, obj_ratio = res.j, res.ratio
                    if objective == "rms_pib":
                        # Adapt the PIB/RMS/EE weights on every valid candidate
                        # evaluation (abandoned evaluations are penalised to <= -100
                        # and skipped). The result is passed explicitly so the
                        # adaptation uses THIS candidate's terms.
                        if float(obj) > -100.0:
                            shaping.adapt_weights(res)
                    obj_val = res.tracking
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
                            value > best_objective + IMPROVE_EPS
                            if objective_mode == "max"
                            else value < best_objective - IMPROVE_EPS
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
                            loggrad=float(last_eval.get("loggrad", 0.0)),
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
                return recorder

            # Evaluation-robustness state (see ``config.n_eval_frames`` /
            # ``fold_ratio`` / ``noise_gate_k``): the fold baseline is armed from
            # the initial (valid) frame; the noise-gate diff history starts empty,
            # so the gate stays off until ``noise_gate_window`` diffs have been
            # collected.
            _fold_baseline_peak: float | None = float(np.max(init_img))
            _fold_baseline_sum: float | None = float(
                np.asarray(init_img, dtype=np.float64).sum()
            )
            _diff_history: deque[float] = deque(maxlen=config.noise_gate_window)
            _n_fold_gated = 0
            _n_noise_gated = 0
            _n_peak_gated = 0

            if config.abba_sampling:
                logger.info(
                    "SPGD ABBA sampling enabled: 4 captures/epoch (+ - - +), "
                    "linear drift cancellation"
                )

            if config.max_peak > 0.0 and exposure_time_ms == 0:
                # A counts cap under auto-exposure is not a power cap: the
                # saturation path lowers exposure, so counts fall while optical
                # power does not.
                logger.warning(
                    "max_peak={} is a camera-count threshold but "
                    "exposure_time_ms=0 enables auto-exposure, which lowers "
                    "exposure near saturation and can mask a real over-power "
                    "event. Fix the exposure to make the cap meaningful.",
                    config.max_peak,
                )

            with tqdm.tqdm(
                total=epochs, desc=f"slm_zernike iter {epochs}", dynamic_ncols=True
            ) as bar:
                for epoch in range(1, epochs + 1):
                    # Generate random perturbation (±1 pattern)
                    disturb_c = rng.binomial(1, 0.5, (nk,)).astype(float) * 2.0 - 1.0
                    disturb_c = disturb_c * delta

                    # Capture frames in the configured sign order: `+ -` by default,
                    # `+ - - +` (ABBA) when ``abba_sampling`` is on. The palindrome
                    # makes a drift that is linear in time carry an identical term
                    # in both sign-means, so it cancels out of the SPGD difference
                    # (see tools/slm/slm_snr_probe.py::abba_signal). Cost: 4
                    # captures/epoch instead of 2.
                    _captures: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
                    for _sign in _spgd_capture_signs(config.abba_sampling):
                        _c = np.clip(_init_c + float(_sign) * disturb_c, -ZERNIKE_CLIP, ZERNIKE_CLIP)
                        _phase = slm.create_phase_from_array(
                            _zernike_to_phase(_c, n_max, pattern_helper, zernike_radius)
                        )
                        _display(slm, _phase)
                        time.sleep(SLM_RESPONSE_TIME_S)
                        _captures.append(
                            (_sign, cam.get_numpy_image(config.n_eval_frames), _c, _phase)
                        )

                    _pos_captures = [c for c in _captures if c[0] > 0]
                    # The logged/ROI/adapt_weights frame stays the FIRST positive
                    # capture, so a recorded row always describes the `+d` phase.
                    # Each capture is `(sign, img, coeffs, phase)`; unpack by
                    # position (a slice like ``[1:]`` would silently transpose the
                    # image with the coefficient vector).
                    _first_pos = _pos_captures[0]
                    pos_img, _pos_c, pos_phase = _first_pos[1], _first_pos[2], _first_pos[3]

                    # Evaluation-robustness gates (2026-09, from the delta<0.001
                    # fold/jitter post-mortem; see docs/slm_pib): reject environment
                    # brightness folds BEFORE scoring so they cannot masquerade as a
                    # coefficient-driven change (measured discrete brightness
                    # states, corr(J, max_brt) = -0.9996), and re-locate the target
                    # ROI onto the CURRENT spot of each frame so a benign beam drift
                    # is not scored as a shaping loss (measured 22-px drift). The
                    # energy guard's ROI rides along; its armed baseline is kept.
                    # Every captured frame is gated (ABBA captures 4).
                    _frame_stats: list[tuple[int, bool, float, float]] = []
                    for _c_sign, _img_c, _, _ in _captures:
                        _is_fold, _pk, _sm = _frame_fold_check(
                            _img_c,
                            _fold_baseline_peak,
                            _fold_baseline_sum,
                            config.fold_ratio,
                        )
                        _frame_stats.append((_c_sign, _is_fold, _pk, _sm))
                    _folded_signs = [s for s, is_fold, _, _ in _frame_stats if is_fold]
                    if _folded_signs:
                        _n_fold_gated += 1
                        logger.warning(
                            "epoch {}: brightness fold (signs={}, pk={:.0f}) - epoch skipped",
                            epoch,
                            _folded_signs,
                            max(pk for _, _, pk, _ in _frame_stats),
                        )
                        # Record the epoch honestly (unchanged coefficients, real
                        # mean J over the frames actually scored) so the fold is
                        # visible offline, then skip search.
                        _fold_j: list[float] = []
                        _fold_ratio: list[float] = []
                        with shaping.shape_batch():
                            for _c_sign, _img_c, _, _ in _captures:
                                _r = shaping(_img_c)
                                _fold_j.append(float(_r.j))
                                _fold_ratio.append(float(_r.ratio))
                        _fold_j_mean = float(np.mean(_fold_j))
                        _log_row(
                            epoch=epoch,
                            coeffs=_init_c,
                            obj_val=_fold_j_mean,
                            obj_ratio=float(np.mean(_fold_ratio)),
                            J=_fold_j_mean,
                            diff=0.0,
                            gate="fold",
                            grad=np.zeros(nk, dtype=np.float64),
                            img=pos_img,
                            phase=pos_phase,
                            lr_val=optimizer.lr,
                            delta_val=delta,
                            max_brt=float(max(pk for _, _, pk, _ in _frame_stats)),
                        )
                        bar.update(1)
                        continue

                    # Measured-peak feasibility gate (see ``max_peak``): mirror the
                    # fold gate, NOT the noise gate. A peak violation says nothing
                    # about gradient validity, so unlike the noise gate this must
                    # NOT call ``optimizer.update`` - zero-updating there decays
                    # momentum, which would corrupt the Adam moments with a
                    # statement we have no evidence for. It also must not append
                    # to ``_diff_history`` (that would inflate ``sigma_hat`` and
                    # make the noise gate over-fire) nor refresh the fold
                    # baseline, nor best-track, nor touch the bucket / lr
                    # schedules. Fully orthogonal: no state changes at all.
                    _peak_observed = float(max(pk for _, _, pk, _ in _frame_stats))
                    if config.max_peak > 0.0 and _peak_observed > config.max_peak:
                        _n_peak_gated += 1
                        logger.warning(
                            "epoch {}: peak {:.0f} exceeds max_peak={:.0f} - "
                            "update NOT committed (step #{})",
                            epoch,
                            _peak_observed,
                            config.max_peak,
                            _n_peak_gated,
                        )
                        # Honest row: unchanged coefficients, real mean J over the
                        # frames actually scored, real peak. ``gate`` records the
                        # reason so the freeze is visible offline.
                        _peak_j: list[float] = []
                        _peak_ratio: list[float] = []
                        with shaping.shape_batch():
                            for _c_sign, _img_c, _, _ in _captures:
                                _r = shaping(_img_c)
                                _peak_j.append(float(_r.j))
                                _peak_ratio.append(float(_r.ratio))
                        _peak_j_mean = float(np.mean(_peak_j))
                        _log_row(
                            epoch=epoch,
                            coeffs=_init_c,
                            obj_val=_peak_j_mean,
                            obj_ratio=float(np.mean(_peak_ratio)),
                            J=_peak_j_mean,
                            diff=0.0,
                            gate="peak",
                            grad=np.zeros(nk, dtype=np.float64),
                            img=pos_img,
                            phase=pos_phase,
                            lr_val=optimizer.lr,
                            delta_val=delta,
                            max_brt=_peak_observed,
                        )
                        bar.update(1)
                        continue

                    # All frames valid: refresh the fold baseline (EMA over valid
                    # frames only: a fold can never pull the baseline down and
                    # blind the gate) using the MEAN over every captured frame,
                    # which reduces to the previous 0.5*(pos+neg) for 2 frames, and
                    # score each frame around its OWN spot.
                    _fold_baseline_peak, _fold_baseline_sum = _update_fold_baseline(
                        _fold_baseline_peak,
                        _fold_baseline_sum,
                        float(np.mean([pk for _, _, pk, _ in _frame_stats])),
                        float(np.mean([sm for _, _, _, sm in _frame_stats])),
                    )
                    _sign_results: dict[int, list[ObjectiveResult]] = {1: [], -1: []}
                    with shaping.shape_batch():
                        for _c_sign, _img_c, _, _ in _captures:
                            shaping.set_reference_center(zero_order_center(_img_c))
                            _sign_results[_c_sign].append(shaping(_img_c))
                    pos_res = _sign_results[1][0]
                    # Sign-means (2 frames reduce to the single value they hold).
                    pos_obj = float(np.mean([r.j for r in _sign_results[1]]))
                    neg_obj = float(np.mean([r.j for r in _sign_results[-1]]))
                    pos_obj_ratio = float(np.mean([r.ratio for r in _sign_results[1]]))
                    neg_obj_ratio = float(np.mean([r.ratio for r in _sign_results[-1]]))
                    pos_center = zero_order_center(pos_img)

                    # Auto-exposure adjustment if saturated (over every capture).
                    # The ceiling comes from the frame dtype, never a literal: a
                    # 16-bit backend would saturate at 65535 and a float frame at
                    # ``DETECTOR_FULL_SCALE``.
                    _sat_level = full_scale(pos_img)
                    max_brightness = max(
                        float(np.max(c[1])) for c in _captures
                    )
                    if max_brightness >= _sat_level and exposure_time_ms == 0:
                        _resample_img = resample_on_saturation(
                            pos_img,
                            cam,
                            exposure_time_ms,
                            target_max_brightness,
                            saturation_threshold=_sat_level,
                        )
                        optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

                    pos_j, neg_j = pos_obj, neg_obj
                    # The recorded panel/metrics describe the POSITIVE frame; keep
                    # the objective's ROI on the positive spot for the row.
                    shaping.set_reference_center(pos_center)

                    # `diff` is kept for logging; the SPGD sign comes from the
                    # shared helper (optimizer/spgd.py) so it cannot be
                    # hand-inverted again (this site maximised/minimised the wrong
                    # way until the objective_mode fix). Cast to float: bucket sums
                    # are unsigned.
                    _spgd_sign = -1.0 if objective_mode == "max" else 1.0
                    diff = (float(pos_j) - float(neg_j)) * _spgd_sign

                    # Noise-aware update gate: with delta < 0.001 the measured diff
                    # is 100% noise (SNR < 0.1; see the ``delta`` config note), so a
                    # diff indistinguishable from the recent diff noise MUST NOT
                    # move the coefficients - it is zeroed and the search stalls
                    # honestly instead of random-walking (h6a: 0/55 modes SNR > 2,
                    # gradient == noise, step/|grad| ratio 435x).
                    sigma_hat = _rolling_sigma(_diff_history)
                    _grad_usable = not _noise_gate(diff, sigma_hat, config.noise_gate_k)
                    _diff_history.append(diff)
                    if not _grad_usable:
                        _n_noise_gated += 1

                    gradient = spgd_gradient(
                        pos_j, neg_j, disturb_c, maximize=(objective_mode == "max")
                    )
                    if _grad_usable:
                        update = optimizer.update(gradient)
                    else:
                        # Keep the optimizer's momentum/history state consistent: a
                        # zeroed update decays momentum toward 0 (forgetting the
                        # noise-driven velocity) without moving the coefficients.
                        update = optimizer.update(np.zeros_like(gradient))
                    _to_update_c = np.clip(_init_c - update, -ZERNIKE_CLIP, ZERNIKE_CLIP)
                    _init_c = _to_update_c

                    # Value logged under the objective's own name and used by the
                    # Recorder to pick its best row: the bucket ratio for "pib",
                    # otherwise the objective the gradient optimises (e.g. radius).
                    objective_val = pos_res.tracking
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
                        power_radio = radius(pos_img, center=pos_center, energy=0.8)
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
                        objective_val > best_objective + IMPROVE_EPS
                        if objective_mode == "max"
                        else objective_val < best_objective - IMPROVE_EPS
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
                        gate="applied" if _grad_usable else "noise",
                        grad=gradient,
                        img=pos_img,
                        phase=pos_phase,
                        lr_val=optimizer.lr,
                        delta_val=delta,
                        max_brt=float(max_brightness),
                        loggrad=float(_sign_results[1][0].loggrad),
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
                            pos_center,
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

            logger.info(
                "SPGD finished: {}/{} epochs updates applied, {} brightness-fold "
                "epochs skipped, {} noise-gated updates (noise_gate_k={}), "
                "{} peak-gated updates (max_peak={})",
                epochs - _n_fold_gated - _n_noise_gated - _n_peak_gated,
                epochs,
                _n_fold_gated,
                _n_noise_gated,
                config.noise_gate_k,
                _n_peak_gated,
                config.max_peak,
            )

            # On exit, leave the SLM at the best phase found. The initial (flat or
            # loaded) phase is one of the candidates: if the search never improved
            # on it, restore that instead of a worse "best". Shared with the
            # heuristic branch through _apply_best_on_exit.

            return recorder
        finally:
            _apply_best_on_exit()
