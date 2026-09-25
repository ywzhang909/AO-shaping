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
import math
import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
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
from ao_shaping.utils.wavefront.zernike_calc import noll_indices as _zernike_indices
from ao_shaping.utils import Recorder, logger
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.spots_calc import centroid, radius
from ao_shaping.utils.image.targets import (
    SHAPE_STAGE_WEIGHTS,
    TARGET_SHAPE_CHOICES,
    create_target_shape,
    rmse_shape_metric,
    rms_pib_terms,
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


def clamp_center_to_frame(
    center: tuple[int | float, int | float],
    frame_shape: tuple[int | float, int | float],
    window: int,
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


def _update_dynamic_weights(
    state: dict,
    *,
    pib: float,
    rms: float,
    j: float,
    ee: float | None = None,
    w_ema_decay: float = 0.9,
    w_floor: float = 0.1,
    w_temperature: float = 8.0,
) -> tuple[float, float] | tuple[float, float, float]:
    """Adaptively re-weight the PIB, RMS and (optional) energy terms.

    The term that improves ``J`` more gets the higher weight ("哪个对J提升大则
    哪个权重大"). Weights are softmax-normalised EMA scores of the *positive*
    contributions of each term to the combined objective:

    * ``c_i = w_i * (term_i - prev_term_i)`` for each participating term;
    * the EMA is updated ONLY on positive contributions (improving steps);
    * if ALL contributions are ``<= 0`` the weights stay unchanged;
    * ``w_i = w_floor + (1 - n*w_floor) * softmax(T*ema_i, ...)`` for the ``n``
      participating terms (n=2 or n=3).

    When ``ee`` is ``None`` (legacy two-term objective) the behaviour is
    byte-identical to the previous ``(w_pib, w_rms)`` pair; when ``ee`` is
    given the energy-conservation term participates and a three-weight tuple
    ``(w_pib, w_rms, w_ee)`` is returned (always summing to 1).
    """
    three_term = ee is not None
    state.setdefault("w_pib", 1.0 / 3 if three_term else 0.5)
    state.setdefault("w_rms", 1.0 / 3 if three_term else 0.5)
    state.setdefault("w_ee", 1.0 / 3 if three_term else 0.0)
    state.setdefault("ema_pib", 0.0)
    state.setdefault("ema_rms", 0.0)
    state.setdefault("ema_ee", 0.0)
    state.setdefault("prev_j", None)
    state.setdefault("prev_pib", None)
    state.setdefault("prev_rms", None)
    state.setdefault("prev_ee", None)
    state.setdefault("prev_set", False)

    if not state["prev_set"]:
        # First call: record the baseline and keep the initial weights.
        state["prev_j"] = float(j)
        state["prev_pib"] = float(pib)
        state["prev_rms"] = float(rms)
        if three_term:
            state["prev_ee"] = float(ee)
        state["prev_set"] = True
        if three_term:
            return (
                float(state["w_pib"]),
                float(state["w_rms"]),
                float(state["w_ee"]),
            )
        return float(state["w_pib"]), float(state["w_rms"])

    w_pib = float(state["w_pib"])
    w_rms = float(state["w_rms"])
    c_pib = w_pib * (float(pib) - float(state["prev_pib"]))
    c_rms = w_rms * (float(rms) - float(state["prev_rms"]))
    if c_pib > 0.0:
        state["ema_pib"] = (
            w_ema_decay * float(state["ema_pib"]) + (1.0 - w_ema_decay) * c_pib
        )
    if c_rms > 0.0:
        state["ema_rms"] = (
            w_ema_decay * float(state["ema_rms"]) + (1.0 - w_ema_decay) * c_rms
        )

    c_ee = 0.0
    if three_term:
        w_ee = float(state["w_ee"])
        c_ee = w_ee * (float(ee) - float(state["prev_ee"]))
        if c_ee > 0.0:
            state["ema_ee"] = (
                w_ema_decay * float(state["ema_ee"])
                + (1.0 - w_ema_decay) * c_ee
            )
        if c_pib <= 0.0 and c_rms <= 0.0 and c_ee <= 0.0:
            # No improving contribution this step: keep the current weights.
            state["prev_j"] = float(j)
            state["prev_pib"] = float(pib)
            state["prev_rms"] = float(rms)
            state["prev_ee"] = float(ee)
            return w_pib, w_rms, w_ee
    elif c_pib <= 0.0 and c_rms <= 0.0:
        # No improving contribution this step: keep the current weights.
        state["prev_j"] = float(j)
        state["prev_pib"] = float(pib)
        state["prev_rms"] = float(rms)
        return w_pib, w_rms

    def _softmax_frac(emas: list[float]) -> list[float]:
        clipped = [float(np.clip(e, -10.0, 10.0)) for e in emas]
        clipped = [0.0 if not np.isfinite(e) else e for e in clipped]
        exps = [math.exp(w_temperature * e) for e in clipped]
        total = sum(exps)
        return [e / total for e in exps]

    if three_term:
        fracs = _softmax_frac(
            [float(state["ema_pib"]), float(state["ema_rms"]), float(state["ema_ee"])]
        )
        w_pib = w_floor + (1.0 - 3.0 * w_floor) * fracs[0]
        w_rms = w_floor + (1.0 - 3.0 * w_floor) * fracs[1]
        w_ee = w_floor + (1.0 - 3.0 * w_floor) * fracs[2]
        state["w_pib"] = float(w_pib)
        state["w_rms"] = float(w_rms)
        state["w_ee"] = float(w_ee)
        state["prev_j"] = float(j)
        state["prev_pib"] = float(pib)
        state["prev_rms"] = float(rms)
        state["prev_ee"] = float(ee)
        return float(w_pib), float(w_rms), float(w_ee)

    fracs = _softmax_frac([float(state["ema_pib"]), float(state["ema_rms"])])
    w_pib = w_floor + (1.0 - 2.0 * w_floor) * fracs[0]
    w_rms = 1.0 - w_pib
    state["w_pib"] = float(w_pib)
    state["w_rms"] = float(w_rms)
    state["prev_j"] = float(j)
    state["prev_pib"] = float(pib)
    state["prev_rms"] = float(rms)
    return float(w_pib), float(w_rms)


def _resolve_init_weights(
    w_pib_init: float | None,
    w_rms_init: float | None,
    w_ee_init: float | None,
) -> tuple[float, float, float]:
    """Resolve the initial PIB/RMS/EE weights of the ``rms_pib`` objective.

    With no weight provided the (1/3, 1/3, 1/3) default is returned. When any
    weight is provided, provided terms are kept exactly and unprovided terms
    share the remaining mass equally, so the triple always sums to 1 (the
    invariant the adaptive update maintains from the first adapting step on).
    When all three are provided they are normalised to sum 1.

    Raises:
        ValueError: if the provided weights sum to more than 1 (with fewer than
            three provided) or to 0 (with all three provided).
    """
    given = (w_pib_init, w_rms_init, w_ee_init)
    if all(w is None for w in given):
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)
    given_sum = float(sum(w for w in given if w is not None))
    n_missing = sum(1 for w in given if w is None)
    if n_missing > 0:
        if given_sum > 1.0 + 1e-9:
            raise ValueError(
                f"initial rms_pib weights must sum to <= 1 when not all are "
                f"provided, got {given_sum!r} "
                f"(w_pib_init={w_pib_init!r}, w_rms_init={w_rms_init!r}, "
                f"w_ee_init={w_ee_init!r})"
            )
        fill = (1.0 - given_sum) / n_missing
        out = tuple(fill if w is None else float(w) for w in given)
    else:
        if given_sum <= 0.0:
            raise ValueError(
                "all initial rms_pib weights are provided but sum to 0: "
                f"(w_pib_init={w_pib_init!r}, w_rms_init={w_rms_init!r}, "
                f"w_ee_init={w_ee_init!r})"
            )
        provided = [w for w in given if w is not None]
        out = tuple(float(w) / given_sum for w in provided)
    return (out[0], out[1], out[2])


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


@dataclass
class SlmZernikePibConfig:
    """Grouped configuration for :func:`optimize_slm_zernike_pib`.

    The 45 keyword arguments of the flat signature are grouped here by role.
    Callers still pass them as keyword arguments (``optimize_slm_zernike_pib(
    ..., delta=0.2, w_uniformity=2.0, ...)``) - the dataclass is an
    internal organisation that keeps the public API 100% compatible.

    Groups:
        Search: algorithm / pop_size / random_seed / objective.
        Zernike: n_max / zernike_radius / init_c.
        Bucket / SPGD step: r_bucket / delta / lr / shrink_iter / shrink_ratio.
        Camera / SLM: cam_id / cam_type / cam_size / exposure_time_ms /
            target_max_brightness / slm_number / slm_wavelength /
            shift_x / shift_y / show.
        Target: target_shape / target_size / target_aspect_ratio /
            target_center_smooth / shape_schedule.
        Objective weights: max_roi_energy_loss / w_uniformity / w_peak /
            w_displacement / log_uniformity / w_ema_decay / w_floor /
            w_temperature / w_pib_init / w_rms_init / w_ee_init.
        Recording: record_phase.
    """

    # --- Search ---------------------------------------------------------------
    algorithm: str = "spgd"
    pop_size: int | None = None
    random_seed: int | None = None
    objective: str = "shape"
    # SPGD gradient optimizer (adam/adamw/adamod/sgd/muno/munow); ignored when
    # ``algorithm`` is a black-box heuristic.
    optimizer_type: str = "adamod"
    # --- Zernike ---------------------------------------------------------------
    n_max: int = 4
    zernike_radius: float = ZERNIKE_APERTURE_RADIUS
    init_c: np.ndarray | list[float] | None = None
    # --- Bucket / SPGD step ----------------------------------------------------
    r_bucket: float = 0
    # 0.2 rad, NOT 0.1: the measured noise floor of the shaping objective is
    # dJ_noise = 4e-4 and a 0.1 rad perturbation moves J by only 2.8e-4
    # (SNR 0.69 -> the SPGD gradient is noise). 0.2 rad gives SNR 2.77
    # (measured on the bench by scripts/measure_shape_sensitivity.py).
    delta: float = 0.2
    lr: float = 0
    shrink_iter: int = 0
    shrink_ratio: float = 0.9
    # --- Camera / SLM ----------------------------------------------------------
    cam_id: int | str = 0
    cam_type: str = "daheng"
    cam_size: int = 250
    exposure_time_ms: float = 80.0
    target_max_brightness: float = 40
    slm_number: int = 1
    slm_wavelength: int = 1064
    shift_x: int | None = 0
    shift_y: int | None = 0
    show: bool = False
    # --- Target -----------------------------------------------------------------
    target_shape: str | None = "rectangle"
    target_size: float | None = None
    target_aspect_ratio: float = 4.0 / 3.0
    target_center_smooth: int = 3
    # NOTE: keep False. Scores from different stages are NOT comparable (the same
    # frame scores ~e in "coarse" but e-penalties in "fine"), so a baseline
    # measured in "coarse" becomes unbeatable and every search reports gain=0.
    # Enabling it needs per-stage best tracking + re-scoring the baseline at the
    # final stage - see shape_metric(stage=...) for the mechanism.
    shape_schedule: bool = False
    # --- Objective weights ------------------------------------------------------
    max_roi_energy_loss: float = 0.6
    w_uniformity: float = 2.0
    w_peak: float = 0.5
    w_displacement: float = 0.0
    log_uniformity: bool = False
    w_ema_decay: float = 0.9
    w_floor: float = 0.1
    w_temperature: float = 8.0
    # Initial weights of the adaptively-weighted 'rms_pib' objective. When any
    # is provided, provided terms are kept exactly and unprovided terms share
    # the remainder equally (all three provided -> normalised to sum 1); with
    # none provided the (1/3, 1/3, 1/3) default is used. The dynamic update
    # maintains the sum-1 invariant from the first adapting step onwards.
    w_pib_init: float | None = None
    w_rms_init: float | None = None
    w_ee_init: float | None = None
    # --- Recording ---------------------------------------------------------------
    record_phase: bool = False
    # --- Escape hatch -----------------------------------------------------------
    #: Extra keyword arguments forwarded to the optimizer constructor
    #: (``_create_optimizer``); kept out of the typed fields.
    kwargs: dict[str, Any] = field(default_factory=dict)


def optimize_slm_zernike_pib(
    center,
    epochs,
    config: SlmZernikePibConfig | None = None,
    algorithm: str = "spgd",
    pop_size: int | None = None,
    n_max: int = 4,
    r_bucket=0,
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
    random_seed: int | None = None,
    objective: str = "shape",
    target_shape: str | None = "rectangle",
    target_size: float | None = None,
    target_aspect_ratio: float = 4.0 / 3.0,
    target_center_smooth: int = 3,
    shape_schedule: bool = False,
    max_roi_energy_loss: float = 0.6,
    w_uniformity: float = 2.0,
    w_peak: float = 0.5,
    w_displacement: float = 0.0,
    log_uniformity: bool = False,
    w_ema_decay: float = 0.9,
    w_floor: float = 0.1,
    w_temperature: float = 8.0,
    w_pib_init: float | None = None,
    w_rms_init: float | None = None,
    w_ee_init: float | None = None,
    record_phase: bool = False,
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
            'avg_radiu' (maximize average radius), 'shape' (dynamic target ROI),
            'rms_pib' (adaptively-weighted PIB + in-ROI RMS for target-shape shaping),
            or 'rmse' (minimize the RMSE between the sum-normalised frame and the
            sum-normalised uniform-intensity target shape - non-PIB beam shaping).
        target_shape: Target ROI shape. Supplying it selects the ``shape`` objective
            (except for ``roi_pib``/``rms_pib``/``rmse`` where it only picks the
            target ROI); defaults to ``rectangle`` for those objectives.
        target_size: Full target extent in camera pixels (rectangle short side,
            circle diameter, square side). ``None`` derives it from the initial
            99% encircled-energy radius.
        target_aspect_ratio: Width:height ratio for a rectangular target ROI.
        target_center_smooth: Number of recent Gaussian-centre estimates averaged
            for each frame (minimum 1).
        w_ema_decay: EMA decay for the adaptive PIB/RMS weight scores of the
            ``rms_pib`` objective (0..1; 0 = raw contribution, no smoothing).
        w_floor: Minimum weight floor for each term of the ``rms_pib`` objective
            (0..0.5 exclusive); weights stay within ``[w_floor, 1-w_floor]``.
        w_temperature: Softmax temperature for the ``rms_pib`` weight update;
            higher values make the weights more decisive.
        record_phase: When True, record the full display-ready SLM phase
            (uint16 grayscale, exactly what was sent to the device) in every
            history row under the ``"_phase"`` key. Memory-heavy: one full
            frame (1200x1920) per row, intended for short debug runs that
            export HDF5 artifacts. Default False keeps rows light.
        config: Optional pre-built :class:`SlmZernikePibConfig`. When supplied,
            its fields are used verbatim and the individual keyword parameters
            below are ignored (they exist only to keep the public keyword API
            and the ``inspect.signature`` contract intact). When ``None``
            (the common path) the keyword parameters are folded into a
            config.
        **kwargs: Additional optimizer parameters, forwarded to
            ``_create_optimizer``.

    Returns:
        Recorder: Optimization history recorder.
    """
    if config is None:
        config = SlmZernikePibConfig(
            algorithm=algorithm,
            pop_size=pop_size,
            optimizer_type=optimizer_type,
            n_max=n_max,
            r_bucket=r_bucket,
            delta=delta,
            lr=lr,
            exposure_time_ms=exposure_time_ms,
            shrink_iter=shrink_iter,
            shrink_ratio=shrink_ratio,
            cam_id=cam_id,
            cam_type=cam_type,
            show=show,
            init_c=init_c,
            cam_size=cam_size,
            target_max_brightness=target_max_brightness,
            slm_number=slm_number,
            slm_wavelength=slm_wavelength,
            random_seed=random_seed,
            objective=objective,
            target_shape=target_shape,
            target_size=target_size,
            target_aspect_ratio=target_aspect_ratio,
            target_center_smooth=target_center_smooth,
            shape_schedule=shape_schedule,
            max_roi_energy_loss=max_roi_energy_loss,
            w_uniformity=w_uniformity,
            w_peak=w_peak,
            w_displacement=w_displacement,
            log_uniformity=log_uniformity,
            w_ema_decay=w_ema_decay,
            w_floor=w_floor,
            w_temperature=w_temperature,
            w_pib_init=w_pib_init,
            w_rms_init=w_rms_init,
            w_ee_init=w_ee_init,
            record_phase=record_phase,
            zernike_radius=zernike_radius,
            shift_x=shift_x,
            shift_y=shift_y,
            kwargs=kwargs,
        )
    else:
        # Extra keyword args (e.g. optimizer beta overrides) are merged into the
        # config's escape-hatch dict.
        config.kwargs.update(kwargs)

    algorithm = config.algorithm
    pop_size = config.pop_size
    optimizer_type = config.optimizer_type
    n_max = config.n_max
    r_bucket = config.r_bucket
    delta = config.delta
    lr = config.lr
    exposure_time_ms = config.exposure_time_ms
    shrink_iter = config.shrink_iter
    shrink_ratio = config.shrink_ratio
    cam_id = config.cam_id
    cam_type = config.cam_type
    show = config.show
    init_c = config.init_c
    cam_size = config.cam_size
    target_max_brightness = config.target_max_brightness
    slm_number = config.slm_number
    slm_wavelength = config.slm_wavelength
    random_seed = config.random_seed
    objective = config.objective
    target_shape = config.target_shape
    target_size = config.target_size
    target_aspect_ratio = config.target_aspect_ratio
    target_center_smooth = config.target_center_smooth
    shape_schedule = config.shape_schedule
    max_roi_energy_loss = config.max_roi_energy_loss
    w_uniformity = config.w_uniformity
    w_peak = config.w_peak
    w_displacement = config.w_displacement
    log_uniformity = config.log_uniformity
    w_ema_decay = config.w_ema_decay
    w_floor = config.w_floor
    w_temperature = config.w_temperature
    w_pib_init = config.w_pib_init
    w_rms_init = config.w_rms_init
    w_ee_init = config.w_ee_init
    record_phase = config.record_phase
    zernike_radius = config.zernike_radius
    shift_x = config.shift_x
    shift_y = config.shift_y

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

        def _smart_center(img):
            """Smart centre: argmax-anchored local centroid, refined by the full
            centroid when the spot core is not a hole (flat core)."""
            (h, w) = img.shape
            margin = int(IDEAL_SPOT_RADIUS)
            center = zero_order_center(img)
            (cx, cy) = center
            y0, y1 = max(0, cy - margin), min(h, cy + margin)
            x0, x1 = max(0, cx - margin), min(w, cx + margin)
            if y1 > y0 and x1 > x0 and np.all(img[y0:y1, x0:x1] >= np.max(img) * 0.4):
                # Flat (non-hollow) core: refine with a LOCAL centroid. A
                # full-image centroid (as intelligen_center does) is dragged tens
                # of pixels by stray light — measured (1394 vs 958 on this bench).
                center = zero_order_center(img, half_win=max(margin * 6, 32))
            return center

        if center is None:
            center = _smart_center(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == "max":
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == "shape":
                # argmax-anchored local centroid (2f bench: 0-order = frame max)
                center = zero_order_center(_img)
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

        # Baseline diagnostic: the ideal-radius (r = IDEAL_SPOT_RADIUS, FIXED)
        # PIB bucket ratio, used to seed ``best_objective`` and logged as
        # ``m_pib7``.
        def ideal_pib_ratio(img):
            return target_func.pib(img, IDEAL_SPOT_RADIUS)[1]

        # --- Objective dispatch ------------------------------------------------
        # Seven objective families, each implemented as a closure returning
        # ``(j, ratio)``. The dispatch replaces the previous 7-branch
        # ``calc_objective`` re-bind + ``to_min`` sign scalar: each branch now
        # owns its own closure, and ``objective_mode`` (already computed above)
        # is the single source of truth for the maximise/minimise direction.
        if objective == "pib":
            # Maximize PIB: negate the SPGD estimate so ``_init_c - update``
            # ascends the objective. Same convention as pib.py.

            def _calc_objective_pib(img):
                pib, pib_ratio = target_func.pib(img, r_bucket)
                return pib, pib_ratio

            calc_objective = _calc_objective_pib
        elif objective == "roi_pib":
            # Maximise the brightness inside the TARGET-SHAPED ROI: the fraction
            # of the light landing in the target rectangle (exposure-invariant
            # ratio, no uniformity/peak/drift penalties).

            def _calc_objective_roi_pib(img):
                # FIXED target ROI (never tracks the spot).
                return roi_pib_metric(
                    img,
                    reference_center,
                    shape_for_metric,
                    target_size,
                    target_aspect_ratio,
                )

            calc_objective = _calc_objective_roi_pib
        elif objective == "rms_pib":
            # Combined PIB + in-ROI RMS + energy-conservation objective with
            # adaptively-weighted terms:
            # J = w_pib(t)*pib_term + w_rms(t)*rms_term + w_ee(t)*ee_term.
            # The weights adapt so the term that improves J more gets the higher
            # weight (see _update_dynamic_weights). The target ROI is FIXED at
            # reference_center (never tracks the spot). ``ee_term`` = fraction
            # of the baseline (flat-phase) window energy still inside the window,
            # so a search that diffracts/scatters light out of the window (or
            # pumps it to a dark halo) is penalised even though pib/rms are
            # exposure-invariant ratios (energy-encircled constraint; see
            # AGENTS.md anti-pattern).
            _rms_pib_state: dict = {}
            _init_w_pib, _init_w_rms, _init_w_ee = _resolve_init_weights(
                w_pib_init, w_rms_init, w_ee_init
            )
            # Baseline window energy from the flat-phase capture: "no energy
            # lost" means sum(img) stays at this level.
            _init_energy = float(np.sum(np.asarray(init_img, dtype=np.float64)))
            last_terms: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

            def _calc_objective_rms_pib(img):
                nonlocal last_terms
                w_pib = float(_rms_pib_state.setdefault("w_pib", _init_w_pib))
                w_rms = float(_rms_pib_state.setdefault("w_rms", _init_w_rms))
                w_ee = float(_rms_pib_state.setdefault("w_ee", _init_w_ee))
                pib_term, rms_term = rms_pib_terms(
                    img,
                    reference_center,
                    shape_for_metric,
                    target_size,
                    target_aspect_ratio,
                )
                _frame = np.asarray(img, dtype=np.float64)
                ee_term = float(
                    np.clip(_frame.sum() / max(_init_energy, np.finfo(np.float64).eps), 0.0, 1.0)
                )
                j = w_pib * pib_term + w_rms * rms_term + w_ee * ee_term
                last_terms = (
                    float(j),
                    float(pib_term),
                    float(rms_term),
                    float(ee_term),
                )
                return float(j), float(pib_term)

            calc_objective = _calc_objective_rms_pib
        elif objective == "shape":
            _shape_state = {"best_energy": 0.0}

            def _calc_objective_shape(img):
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

            calc_objective = _calc_objective_shape
        elif objective == "rmse":
            # Minimise the RMSE between the frame and the uniform-intensity target,
            # both normalised to unit sum (exposure / laser-drift invariant).

            def _calc_objective_rmse(img):
                return rmse_shape_metric(
                    img,
                    reference_center,
                    shape_for_metric,
                    target_size,
                    target_aspect_ratio,
                )

            calc_objective = _calc_objective_rmse
        elif objective == "radiu":
            def _calc_objective_radiu(img):
                r = target_func.radius(img, energy=0.99)
                return r, 0.0

            calc_objective = _calc_objective_radiu
        elif objective == "avg_radiu":
            # Maximize average radius.

            def _calc_objective_avg(img):
                return target_func.avg_radius(img, moment=1.0)

            calc_objective = _calc_objective_avg

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

        if max_roi_energy_loss > 0.0 and objective in (
            "pib",
            "rmse",
            "shape",
            "roi_pib",
            "rms_pib",
        ):
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
        best_objective = float(ideal_pib_ratio(init_img)) if objective == "pib" else float(j)
        # Baseline objective of the initial phase (flat when ``init_c`` is
        # empty). Used on exit to decide between the best phase and flat.
        _initial_objective = best_objective
        best_c = _init_c.copy()
        last_best_epoch = 0

        # Baseline window energy (flat-phase capture) for the energy-conservation
        # panel metric (exposure-demanding: valid when run forces exposure or the
        # auto-exposure settles, i.e. every epoch shares a comparable sum).
        _panel_init_sum = float(np.sum(np.asarray(init_img, dtype=np.float64)))

        def _metric_panel(img: np.ndarray) -> dict[str, float]:
            """Cross-objective metric panel recorded on EVERY epoch.

            Evaluates all shaping objectives on the same frame with the run's
            actual configuration (weights / target shape / bucket radius), so any
            two runs can be compared on any shared ``m_*`` column regardless of
            which objective actually drove the search. ``m_`` prefix avoids
            colliding with the objective's own row key (e.g. ``"shape"``).
            """
            _shape_score, _energy = shape_metric(
                img,
                reference_center,
                reference_center,
                shape_for_metric,
                target_size,
                target_aspect_ratio,
                w_uniformity=w_uniformity,
                w_peak=w_peak,
                w_displacement=w_displacement,
                stage=None,
                log_uniformity=log_uniformity,
            )
            _pib_term, _rms_t = rms_pib_terms(
                img,
                reference_center,
                shape_for_metric,
                target_size,
                target_aspect_ratio,
            )
            _rmse, _ = rmse_shape_metric(
                img,
                reference_center,
                shape_for_metric,
                target_size,
                target_aspect_ratio,
            )
            _roi_score, _roi_energy = roi_pib_metric(
                img,
                reference_center,
                shape_for_metric,
                target_size,
                target_aspect_ratio,
            )
            _frame_sum = float(np.asarray(img, dtype=np.float64).sum())
            _ee = float(
                np.clip(
                    _frame_sum / max(_panel_init_sum, np.finfo(np.float64).eps),
                    0.0,
                    1.0,
                )
            )
            return {
                "m_shape": float(_shape_score),
                "m_energy": float(_energy),
                "m_rmse": float(_rmse),
                "m_roi_pib": float(_roi_score),
                "m_pib": float(target_func.pib(img, r_bucket)[1]),
                "m_pib7": float(ideal_pib_ratio(img)),
                # Equal-weight (1/3 each) rms_pib score: the objective's own
                # adaptive weights vary per epoch, so the fixed-weight value is
                # the cross-run comparable form.
                "m_rms_pib": float((_pib_term + _rms_t + _ee) / 3.0),
                "m_rms_t": float(_rms_t),
                "m_ee": _ee,
                "m_brt": float(np.max(img)),
            }

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
            _row0["w_pib"] = float(_rms_pib_state.get("w_pib", 1.0 / 3))
            _row0["w_rms"] = float(_rms_pib_state.get("w_rms", 1.0 / 3))
            _row0["w_ee"] = float(_rms_pib_state.get("w_ee", 1.0 / 3))
            _row0["pib_term"] = float(last_terms[1])
            _row0["rms_term"] = float(last_terms[2])
            _row0["ee_term"] = float(last_terms[3])
        _row0.update(_metric_panel(init_img))
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
                row["w_pib"] = float(_rms_pib_state.get("w_pib", 0.5))
                row["w_rms"] = float(_rms_pib_state.get("w_rms", 0.5))
                row["pib_term"] = float(last_terms[1])
                row["rms_term"] = float(last_terms[2])
            row.update(_metric_panel(img))
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
                if objective == "rms_pib":
                    # Adapt the PIB/RMS/EE weights on every valid candidate
                    # evaluation (abandoned evaluations are penalised to <= -100
                    # and skipped).
                    if float(obj) > -100.0:
                        _update_dynamic_weights(
                            _rms_pib_state,
                            pib=float(last_terms[1]),
                            rms=float(last_terms[2]),
                            ee=float(last_terms[3]),
                            j=float(last_terms[0]),
                            w_ema_decay=w_ema_decay,
                            w_floor=w_floor,
                            w_temperature=w_temperature,
                        )
                obj_val = float(ideal_pib_ratio(img)) if objective == "pib" else float(obj)
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
                pos_obj, pos_obj_ratio = calc_objective(pos_img)
                if objective == "rms_pib":
                    # Keep the POSITIVE-perturbation terms: the negative eval below
                    # overwrites ``last_terms``, and the weight adaptation must use
                    # the direction the gradient actually follows.
                    _pos_terms = last_terms

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
                objective_val = (
                    float(ideal_pib_ratio(pos_img)) if objective == "pib" else float(pos_j)
                )
                objective_ratio = (pos_obj_ratio + neg_obj_ratio) / 2
                J = (pos_j + neg_j) / 2

                if objective == "rms_pib" and pos_obj > -100.0 and neg_obj > -100.0:
                    # Adapt the PIB/RMS/EE weights from the positive-perturbation
                    # terms (abandoned evaluations are penalised to <= -100 and
                    # skipped). The term that improves J more gets the higher
                    # weight.
                    _update_dynamic_weights(
                        _rms_pib_state,
                        pib=float(_pos_terms[1]),
                        rms=float(_pos_terms[2]),
                        ee=float(_pos_terms[3]),
                        j=float(_pos_terms[0]),
                        w_ema_decay=w_ema_decay,
                        w_floor=w_floor,
                        w_temperature=w_temperature,
                    )

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
        config=SlmZernikePibConfig(
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
        ),
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
