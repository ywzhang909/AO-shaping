"""Shared convergence-diagnostics threshold family for dynamic LR/delta schedules.

The three wfless schedulers (``pib`` / ``slm_zernike_primitives`` /
``slm_square_shaping``) each carried a byte-for-byte copy of the same
convergence-detection core: a coefficient-of-variation gate on the recent
gradient magnitudes, a trend gate on the recent objective value, an early-epoch
warm-up, and a final clamp on the resulting (lr_factor, delta_factor). Left
inline, the two copies could drift silently -- which is exactly the failure this
module closes (issue #173, batch 3: "single-source the grad_cv / trend-eps /
window / epoch family").

What is single-sourced here is the *convergence-diagnostics* family only -- the
thresholds and the factor arithmetic that map recent history to (lr_factor,
delta_factor). The per-call-site **base** (lr, delta) segmentation is NOT part
of this family and stays with each caller: ``pib`` / ``slm_zernike_primitives``
segment by power-radius, ``slm_square_shaping`` segments by CV. The two
legitimate per-call-site differences are parameterised, not hidden:

* ``higher_is_better`` -- whether a *rising* objective trend is convergence
  (PIB/quality, ``True``) or divergence (CV, ``False``). This flips which side
  of the zero-trend band is "diverging" (shrinks lr) versus "converging"
  (caps lr).
* ``stagnant_needs_low_std`` -- the PIB family additionally requires a low
  objective std to call the run stagnant; the CV family does not.

Leaf hygiene: imports numpy only (no ``ao_shaping`` import, no SDK, no disk
I/O), matching the repo's lazy-loading contract.
"""

from __future__ import annotations

import numpy as np

# --- history windows --------------------------------------------------------
#: how many recent samples to fold into the gradient / objective diagnostics
RECENT_WINDOW = 10
#: minimum number of gradient samples before convergence detection engages
MIN_HISTORY = 5

# --- gradient coefficient-of-variation gates -------------------------------
#: grad_cv below this = stable convergence -> shrink lr and delta
GRAD_CV_STABLE = 0.1
#: grad_cv below this = normal fluctuation -> keep
GRAD_CV_MODERATE = 0.3
#: grad_cv above this = oscillating -> cut lr hard, widen delta to explore
GRAD_CV_OSCILLATING = 0.8

# --- objective-trend gates --------------------------------------------------
#: |trend| below this (per-sample objective change) = stagnant
TREND_EPSILON = 1e-5
#: |trend| above this = a genuine move (diverging vs converging)
TREND_STEP = 0.001
#: stagnant only counts when the objective std is below this (PIB family)
OBJ_STD_STAGNANT = 0.01

# --- factor magnitudes ------------------------------------------------------
#: stagnant -> widen exploration by this delta factor, cap lr by this
EXPLORE_DELTA_BUMP = 1.5
EXPLORE_LR_CAP = 0.7
#: diverging -> hard shrink
TREND_DIVERGE_LR = 0.4
TREND_DIVERGE_DELTA = 0.6
#: converging -> allow a mild lr bump up to this cap
TREND_CONVERGE_LR_CAP = 1.1

# --- early-epoch warm-up ----------------------------------------------------
WARMUP_EPOCHS = 20
WARMUP_LR_MULT = 1.2
WARMUP_DELTA_MULT = 1.1

# --- final clamp bounds -----------------------------------------------------
LR_CLAMP_LO = 0.2
LR_CLAMP_HI = 2.0
DELTA_CLAMP_LO = 0.3
DELTA_CLAMP_HI = 2.5


def _coefficients_of_variation(history: list[float]) -> float:
    """Std/mean of ``history``; 0.0 when the mean would divide by zero."""
    vals = (
        list(history[-RECENT_WINDOW:]) if len(history) >= RECENT_WINDOW else history
    )
    mean = float(np.mean(vals))
    std = float(np.std(vals)) if len(vals) > 1 else 0.0
    return std / (mean + 1e-8)


def _objective_trend_std(history: list[float]) -> tuple[float, float]:
    """(per-sample trend, std) of the recent objective-value history."""
    vals = list(history[-RECENT_WINDOW:]) if len(history) >= RECENT_WINDOW else history
    std = float(np.std(vals)) if len(vals) > 1 else 0.0
    trend = (
        (vals[-1] - vals[0]) / (len(vals) + 1e-8) if len(vals) > 1 else 0.0
    )
    return trend, std


def convergence_lr_delta_factors(
    gradient_history: list[float],
    objective_history: list[float],
    epoch: int,
    *,
    higher_is_better: bool = True,
    stagnant_needs_low_std: bool = False,
) -> tuple[float, float]:
    """Map recent (gradient, objective) history to ``(lr_factor, delta_factor)``.

    This is the single shared implementation of the convergence-detection core
    that used to be copy-pasted across ``pib`` / ``slm_zernike_primitives`` /
    ``slm_square_shaping``. Callers keep their own base (lr, delta) segmentation
    and multiply it by the returned factors (applying any hard caps of their
    own).

    Args:
        gradient_history: Recent gradient magnitudes (must already satisfy the
            caller's ``len >= MIN_HISTORY`` gate).
        objective_history: Recent objective values (PIB/quality, or CV).
        epoch: Current iteration (drives the early-epoch warm-up).
        higher_is_better: True -> a rising objective trend is convergence;
            False (CV) -> a rising trend is divergence. Flips the diverging vs
            converging branch.
        stagnant_needs_low_std: True -> the stagnant band additionally requires
            the objective std to be low (the PIB family's behaviour); False ->
            a near-zero trend alone is stagnant (the CV family's behaviour).

    Returns:
        ``(lr_factor, delta_factor)`` clamped to the shared bounds.
    """
    grad_cv = _coefficients_of_variation(gradient_history)
    obj_trend, obj_std = _objective_trend_std(objective_history)

    lr_factor = 1.0
    delta_factor = 1.0

    # gradient coefficient-of-variation gate (identical across all callers)
    if grad_cv < GRAD_CV_STABLE:
        lr_factor = 0.5
        delta_factor = 0.5
    elif grad_cv < GRAD_CV_MODERATE:
        lr_factor = 0.8
        delta_factor = 0.8
    elif grad_cv > GRAD_CV_OSCILLATING:
        lr_factor = 0.3
        delta_factor = 1.2

    # objective-trend gate (direction depends on higher_is_better)
    if higher_is_better:
        stagnant = abs(obj_trend) < TREND_EPSILON and (
            (obj_std < OBJ_STD_STAGNANT) if stagnant_needs_low_std else True
        )
        if stagnant:
            delta_factor = max(delta_factor, EXPLORE_DELTA_BUMP)
            lr_factor = min(lr_factor, EXPLORE_LR_CAP)
        elif obj_trend < -TREND_STEP:
            lr_factor = TREND_DIVERGE_LR
            delta_factor = TREND_DIVERGE_DELTA
        elif obj_trend > TREND_STEP:
            lr_factor = min(lr_factor, TREND_CONVERGE_LR_CAP)
    else:
        stagnant = abs(obj_trend) < TREND_EPSILON
        if stagnant:
            delta_factor = max(delta_factor, EXPLORE_DELTA_BUMP)
            lr_factor = min(lr_factor, EXPLORE_LR_CAP)
        elif obj_trend > TREND_STEP:
            lr_factor = TREND_DIVERGE_LR
            delta_factor = TREND_DIVERGE_DELTA
        elif obj_trend < -TREND_STEP:
            lr_factor = min(lr_factor, TREND_CONVERGE_LR_CAP)

    # early-epoch warm-up
    if epoch < WARMUP_EPOCHS:
        lr_factor *= WARMUP_LR_MULT
        delta_factor *= WARMUP_DELTA_MULT

    lr_factor = float(np.clip(lr_factor, LR_CLAMP_LO, LR_CLAMP_HI))
    delta_factor = float(np.clip(delta_factor, DELTA_CLAMP_LO, DELTA_CLAMP_HI))
    return lr_factor, delta_factor
