"""Model-in-the-loop target shaping on real SLM + CCD hardware.

This is the hardware port of :func:`simulate_iterative_shaping` in
:mod:`ao_shaping.optimizer.wfless.model_in_loop_shaping`, which runs the same
alternating loop against the digital twin. Each round does two things:

* **Step A -- wavefront fit (model-in-the-loop).** A set of strong, *varied*
  random pupil probes is displayed on the SLM and the resulting far-field
  frames are measured. A single :class:`ZernikeCoefficientOptimizer` cycles
  through every probe so all of them inform one shared Zernike aberration
  vector, fitted by minimising loss(measured CCD, predicted far field).
* **Step B -- target synthesis.** The fitted aberration is **frozen** and a
  full-pixel SLM phase is optimised against the configured target. That phase is
  displayed, measured, and becomes the warm start for the next round.

The module is a thin orchestration layer. No Zernike math, no forward model, no
metric and no loss is reimplemented here: everything is delegated to the
canonical helpers so the hardware path cannot drift from the simulated one.

Why Step A probes instead of using the shaped phase
----------------------------------------------------
A flat -- or a smoothly shaped -- pupil focuses to a near-delta whose normalised
intensity barely responds to a smooth low-order aberration, so the aberration
is not identifiable from it. Above roughly 3 rad of probe spread the signal
clears the 16-bit quantisation pedestal by two orders of magnitude and the fit
becomes well posed. Because focal-plane intensity is a *non-convex* function of
pupil phase, one probe still leaves many stationary points and Adam stalls in
whichever one it enters; cycling several independent probes through one Adam
state supplies the diversity that removes them.

Why the coupling guards exist
-----------------------------
Step A produces ``c_t``; Step B produces ``phi_t`` conditioned on ``c_t``; the
next round refits ``c_{t+1}`` through ``phi_t``'s correction. Left alone the two
estimators chase each other -- the aberration absorbs part of the shaping phase
and vice versa -- which on hardware (with drift, LCOS settling error and read
noise) shows up as a slow divergence rather than a crash. Two cheap guards
bound it: a **trust region** on the per-round coefficient change, and a **round
acceptance test** that rejects a round whose Step-A loss worsened or whose
measured composite degraded, damping the learning rate and adding probes
instead. Persistent rejection means the bench is outside the model (pupil
registration, panel tilt, out-of-plane defocus, vignetting), and the run aborts
rather than committing a phase the model merely likes.

Self-calibration (opt-in)
-------------------------
Step A can additionally fit two **self-calibration** quantities inside the
forward model -- the pupil-plane affine ``tau_theta`` (6 scalars) and the
per-mode-family travel scale ``gamma`` (7 scalars), following GRAPHYCS (Cho et
al. 2026) -- plus an optional Fourier-domain term in the loss. They are off by
default, so the default run is bit-for-bit the one it was.

The paper has **no** trust region and **no** per-round acceptance test on those
quantities, which is exactly the failure mode this module refuses: a
calibration parameter that sprints away from its start means the forward model
is being bent to fit the data instead of describing the bench. So the per-round
change of every calibration parameter is bounded
(:func:`calibration_trust_region_check`), and a violation rolls the parameters
back and rejects the round.

Nothing in this module imports a hardware driver at import time; every device
import is deferred into the functions that open it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # annotations only (``from __future__ import annotations``).
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import BenchGeometry

# Device-free (NumPy + stdlib only), so importing it cannot pull in a driver.
from ao_shaping.gui.utils.phase_overlay import (
    PhaseOverlay,
    apply_overlays_to_handle,
)
from ao_shaping.optimizer.wfless.adaptive_trust_region import (
    AdaptiveTrustRegion,
    AdaptiveTrustRegionConfig,
)
from ao_shaping.utils.image.frame_analysis import crop_around_zero_order
from ao_shaping.utils.image.target import DIFFERENTIABLE_SHAPE_CHOICES
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BLAZE_PERIOD_MIN_PX,
    SLM_PITCH_M,
)
from ao_shaping.utils.slm.panel_geometry import phase_to_panel
from ao_shaping.utils.wavefront.beam_amplitude import gaussian_grid
from ao_shaping.utils.wavefront.calibration_terms import (
    PupilAffine,
    TravelScale,
)
from ao_shaping.utils.wavefront.model_error_terms import ModelErrorTerms

#: Reused rather than reimplemented: median-subtract THEN clip. Clipping the raw
#: frame would rectify symmetric read noise into a positive pedestal
#: proportional to the pixel count (documented in ``drivers/sim/AGENTS.md``),
#: which corrupts every ratio metric computed afterwards. Canonical home is
#: the leaf ``utils`` layer, so this import pulls in no driver package.
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.image.exposure_control import (
    choose_exposure,
    exposure_normalized_frame,
    resolve_daheng_exposure_limit,
)
from ao_shaping.utils.wavefront.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.io.recorder_keys import (
    RECORD_EPOCH,
)

__all__ = [
    "AdaptiveTrustRegion",
    "AdaptiveTrustRegionConfig",
    "Bench",
    "ModelInLoopStatus",
    "RoundRecord",
    "SlmModelInLoopConfig",
    "acceptance_verdict",
    "calibration_trust_region_check",
    "optimize_slm_model_in_loop",
    "trust_region_clamp",
]


class ModelInLoopStatus:
    """Terminal states of :func:`optimize_slm_model_in_loop`."""

    COMPLETED = "completed"
    """At least one round was accepted."""

    ABORTED_UNIDENTIFIABLE = "aborted_unidentifiable"
    """Geometry calibration never reached the required correlation."""

    ABORTED_REJECTION_STREAK = "aborted_rejection_streak"
    """Too many consecutive rejected rounds -- the bench is outside the model."""

    BAKE_OFF_REJECTED = "bake_off_rejected"
    """Every shaped phase lost to the flat baseline, so flat was kept."""


# ---------------------------------------------------------------------------
# Coupling guards -- pure functions so they can be tested without hardware
# ---------------------------------------------------------------------------


def trust_region_clamp(
    candidate: np.ndarray,
    reference: np.ndarray,
    max_l2: float,
) -> tuple[np.ndarray, float, bool]:
    """Bound how far the fitted aberration may move in one round.

    Step A and Step B chase each other: without a bound on the coefficient
    change the aberration progressively absorbs the shaping phase. Clamping the
    L2 norm of the step keeps the two estimators coupled but not co-adapting.

    Args:
        candidate: Aberration vector proposed by this round's fit, radians.
        reference: The vector in force before this round, radians.
        max_l2: Maximum permitted L2 norm of ``candidate - reference``. Values
            ``<= 0`` disable the guard, which is only appropriate when the fit
            is externally constrained.

    Returns:
        A tuple of ``(clamped_vector, step_l2, was_clamped)`` where
        ``step_l2`` is the *unclamped* L2 norm, recorded so the history shows
        how hard the guard bit.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {candidate.shape} != reference shape {reference.shape}"
        )
    delta = candidate - reference
    step_l2 = float(np.linalg.norm(delta))
    if max_l2 <= 0.0 or step_l2 <= max_l2:
        return candidate.copy(), step_l2, False
    # Scale the *step*, not the vector: clamping the vector itself would drag
    # the coefficients toward the origin rather than toward the previous fit.
    scaled = reference + delta * (float(max_l2) / step_l2)
    return scaled, step_l2, True


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Outcome of the round-acceptance test.

    Attributes:
        accepted: Whether the round's phase and coefficients may be committed.
        loss_worsened: Step A's loss ended materially above where it started.
        score_degraded: The measured composite fell by more than the tolerance.
        reason: Short machine-readable tag for the history (``"accepted"``,
            ``"loss_worsened"``, ``"score_degraded"``,
            ``"calibration_trust_region"`` or any of them joined by ``"+"``).
    """

    accepted: bool
    loss_worsened: bool
    score_degraded: bool
    reason: str

    def __bool__(self) -> bool:
        """Truthy when the round is accepted."""
        return self.accepted


def acceptance_verdict(
    loss_before: float,
    loss_after: float,
    score_before: float | None,
    score_after: float,
    *,
    loss_delta: float = 1e-4,
    score_eps: float = 1e-3,
    cv_before: float | None = None,
    cv_after: float | None = None,
    max_cv_regression: float = 0.02,
    calibration_violated: bool = False,
) -> AcceptanceVerdict:
    """Decide whether a round may be committed, or must be rolled back.

    A round is rejected when the forward-model fit worsens, the measured
    composite drops, or the target-region CV rises beyond the configured
    tolerance. The CV guard prevents an EE gain from replacing a more uniform
    accepted phase.

    ``score_before=None`` means "no shaped phase has been accepted yet", and the
    score test is then skipped. That case is not a loophole but the intended
    behaviour of the first round: the flat baseline was never optimised for this
    target, so its encircled energy is high by accident (the unshaped focus
    happens to sit inside the box) and comparing a real shaping attempt against
    it rejects every genuine attempt. Round one is the exploration step; whether
    it was worth keeping is decided by the final bake-off against flat.

    Args:
        loss_before: Step-A loss at the start of this round's fit.
        loss_after: Step-A loss at the end of this round's fit.
        score_before: Best committed composite score, or ``None`` on round one.
        score_after: Composite score of the frame measured after Step B.
        loss_delta: Tolerance on ``loss_after - loss_before``. Step A is a
            descent method, so any material increase is a genuine regression.
        score_eps: Tolerance on ``score_before - score_after``.
        cv_before: Best accepted shaped CV, if any.
        cv_after: CV of this round's measured shaped phase.
        max_cv_regression: Allowed absolute CV increase between shaped rounds.
        calibration_violated: ``True`` when the round's **self-calibration**
            parameters (the pupil affine τ_θ or the travel scale γ) moved
            further than the per-round trust region allows. Such a round is
            rejected: a calibration quantity that sprints away from its start
            means the forward model is being bent to fit the data rather than
            describing the bench.

    Returns:
        The :class:`AcceptanceVerdict`.
    """
    if (
        not np.isfinite(loss_before)
        or not np.isfinite(loss_after)
        or not np.isfinite(score_after)
    ):
        # A non-finite residual means the fit or the measurement diverged. That
        # is not "no evidence of harm", it is absence of evidence: refuse it.
        return AcceptanceVerdict(False, True, True, "non_finite")
    loss_worsened = bool(loss_after > loss_before + float(loss_delta))
    score_degraded = bool(
        score_before is not None
        and np.isfinite(score_before)
        and score_after < score_before - float(score_eps)
    )
    cv_degraded = bool(
        cv_before is not None
        and cv_after is not None
        and (not np.isfinite(cv_after) or cv_after > cv_before + max_cv_regression)
    )
    tags = [
        name
        for name, hit in (
            ("loss_worsened", loss_worsened),
            ("score_degraded", score_degraded),
            ("cv_degraded", cv_degraded),
            ("calibration_trust_region", bool(calibration_violated)),
        )
        if hit
    ]
    return AcceptanceVerdict(
        accepted=not tags,
        loss_worsened=loss_worsened,
        score_degraded=score_degraded,
        reason="+".join(tags) if tags else "accepted",
    )


def calibration_trust_region_check(
    affine_after: PupilAffine | None,
    affine_before: PupilAffine | None,
    travel_after: TravelScale | None,
    travel_before: TravelScale | None,
    *,
    theta_step_max: float = 0.02,
    gamma_step_max: float = 0.05,
) -> tuple[bool, float, float]:
    """Check whether the self-calibration parameters moved too far this round.

    The paper (GRAPHYCS) has **no** trust region on its self-calibration
    quantities. That is exactly the failure mode this repository refuses: a
    calibration parameter that sprints away from its start means the forward
    model is being bent to fit the data instead of describing the bench. So the
    per-round change is bounded, and a violation rejects the round (see
    :func:`acceptance_verdict`).

    The bound is **per parameter**, not an L2 norm over the group: the six
    affine components have different units (a scale factor, a normalised
    offset, an angle), so one norm would mix them.

    Args:
        affine_after: The pupil affine τ_θ after this round's fit.
        affine_before: The value in force before this round.
        travel_after: The travel scale γ after this round's fit.
        travel_before: The value in force before this round.
        theta_step_max: Allowed per-parameter change of τ_θ. ``<= 0`` disables
            the guard.
        gamma_step_max: Allowed per-family change of γ. ``<= 0`` disables the
            guard.

    Returns:
        A ``(violated, theta_step, gamma_step)`` tuple. ``violated`` is
        ``True`` when any enabled bound was exceeded.
    """
    violated = False
    theta_step = 0.0
    gamma_step = 0.0
    if affine_after is not None and affine_before is not None:
        theta_step = affine_after.max_abs_step(affine_before)
        if float(theta_step_max) > 0.0 and theta_step > float(theta_step_max):
            violated = True
    if travel_after is not None and travel_before is not None:
        gamma_step = travel_after.max_abs_step(travel_before)
        if float(gamma_step_max) > 0.0 and gamma_step > float(gamma_step_max):
            violated = True
    return violated, theta_step, gamma_step


# ---------------------------------------------------------------------------
# Probe split (train / holdout) resolution

ProbeSplitHook = Callable[[int, int, int, int], tuple[Sequence[int], Sequence[int]]]
"""Signature of an injectable probe-split hook.

Called as ``hook(seed, index, probe_count, holdout_probe_count)`` and expected
to return ``(train_seeds, holdout_seeds)`` for one round. It exists so a
cross-validation harness can *choose* which probes train the forward model and
which are reserved for scoring, without this module knowing anything about
folds. The default is ``None``, which reproduces the historical arithmetic.
"""


def _coerce_seed_list(values: Sequence[int], expected: int, label: str) -> list[int]:
    """Validate and normalise one half of a hook-provided split."""
    try:
        items = list(values)
    except TypeError as exc:
        # A flat pair like ``(1, 2)`` unpacks into two scalars, so the unpack in
        # the caller cannot be the only guard: report the same ValueError the
        # documented contract promises instead of leaking a raw TypeError.
        raise ValueError(
            f"probe split hook must return a sequence of {expected} {label} "
            f"seeds, got {values!r}"
        ) from exc
    if len(items) != expected:
        raise ValueError(
            f"probe split hook returned {len(items)} {label} seeds, expected {expected}"
        )
    out: list[int] = []
    for value in items:
        as_int = int(value)
        # Reject 1.5 rather than silently truncating to 1: a fold that indexes
        # the wrong probe is a wrong *number*, not a rounding nuisance.
        if as_int != value:
            raise ValueError(
                f"probe split hook returned a non-integer {label} seed: {value!r}"
            )
        out.append(as_int)
    return out


def _resolve_probe_seeds(
    seed: int,
    index: int,
    probe_count: int,
    holdout_probe_count: int,
    hook: ProbeSplitHook | None = None,
) -> tuple[list[int], list[int]]:
    """Resolve one round's train and holdout probe seeds.

    With ``hook=None`` this is the historical arithmetic verbatim --
    ``seed + index * 1000 + k`` for the fit probes and the same expression
    offset by ``1_000_000`` for the held-out ones -- so a run with no hook is
    bit-for-bit unchanged.

    A hook must return exactly ``probe_count`` train seeds and
    ``holdout_probe_count`` holdout seeds, and the two sets must be disjoint:
    independence of the holdout is the entire point of the split, and a hook
    that reuses a seed would silently turn the held-out correlation into an
    in-sample one.

    Args:
        seed: Base seed from the config.
        index: Zero-based round index.
        probe_count: Number of probes Step A cycles through.
        holdout_probe_count: Number of reserved scoring probes.
        hook: Optional override; ``None`` selects the legacy arithmetic.

    Returns:
        A ``(train_seeds, holdout_seeds)`` tuple of integer seeds.

    Raises:
        ValueError: If the hook returns the wrong shape, a non-integer seed, or
            seeds that overlap between the two halves.
    """
    if hook is None:
        base = int(seed) + int(index) * 1000
        return (
            [base + k for k in range(int(probe_count))],
            [base + 1_000_000 + k for k in range(int(holdout_probe_count))],
        )

    result = hook(int(seed), int(index), int(probe_count), int(holdout_probe_count))
    try:
        train_values, holdout_values = result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "probe split hook must return a (train_seeds, holdout_seeds) pair, "
            f"got {result!r}"
        ) from exc

    train = _coerce_seed_list(train_values, int(probe_count), "train")
    holdout = _coerce_seed_list(holdout_values, int(holdout_probe_count), "holdout")
    overlap = sorted(set(train) & set(holdout))
    if overlap:
        raise ValueError(
            "probe split hook reused seed(s) across the split, so the holdout "
            f"would not be independent: {overlap}"
        )
    return train, holdout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SlmModelInLoopConfig:
    """Single-parameter config for :func:`optimize_slm_model_in_loop`.

    Devices are opened and closed *inside* the optimizer; reusing a device
    across runs is forbidden because the SDK handles are not re-entrant.

    Derived geometry is resolved once in ``__post_init__`` rather than at each
    use site, so the runner, a library caller and the tests all observe the same
    resolved configuration and no consumer can accidentally act on a raw ``0``
    sentinel.
    """

    # --- control -----------------------------------------------------------
    n_rounds: int = 6
    """Fit -> shape -> re-measure rounds."""

    seed: int = 0
    device: str | None = None
    """Torch device for the forward model; ``None`` auto-detects."""

    dtype: str = "float32"
    """Working precision for the forward model: ``"float32"`` or ``"float64"``.

    Defaults to ``float32``, matching
    :class:`~...zernike_coefficient_optimizer.ZernikeCoefficientOptimizer`, which
    has always defaulted to it; this config used to override it back to
    ``float64`` and was the only layer doing so.

    Measured on an RTX 5080 Laptop (2026-10-19), warm one-round runs at a fixed
    config: at ``region=128`` float64/CUDA 2.39 s vs float32/CUDA 1.64 s (1.46x);
    at ``region=256`` 4.91 s vs 2.91 s (1.69x). ``float64`` on a consumer GPU is
    disproportionately slow, so it gave *no* gain over CPU at 128 (2.39 vs 2.35 s)
    -- the precision choice, not the device, was cancelling the GPU.

    The decisive reason is not speed but that the extra precision is *unreachable*.
    A phase reaches the panel only after quantisation to the device's 10-bit gray
    map (2*pi = 993 levels, i.e. 6.33e-3 rad per level). Measured 2026-10-19 with
    random raw-radian phases, mapping float32 and float64 versions of the same
    array through
    :func:`~ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale` gives
    **byte-identical** gray maps -- zero differing pixels out of 262144 at
    ``region=512``, and at 128 and 256 -- with the phase differing by only
    0.0075%-0.015% of one gray level. Float64's mantissa is discarded by the
    quantiser before it can influence anything on the bench.

    Pass ``"float64"`` explicitly only when comparing against a stored float64
    artefact: it cannot change what the panel displays, but it does change the
    arrays written into debug artefacts.

    Speed, measured 2026-10-19 on an RTX 5080 Laptop with every case warmed and
    then run round-robin five times (median, one-round warm run at a fixed config) --
    interleaving matters, because a CUDA case timed first pays one-off kernel
    compilation and reads several times slower than it really is:

    ==============  ==============  ==============  ==============  ==============
    region          float64/CUDA    float32/CUDA    float64/CPU     float32/CPU
    ==============  ==============  ==============  ==============  ==============
    128             1.71 s          1.71 s          3.16 s          2.36 s
    256             3.40 s          3.05 s          10.64 s         5.96 s
    ==============  ==============  ==============  ==============  ==============

    So the device is the big lever (CUDA is 1.85x-3.13x the CPU at equal precision,
    and ``device=None`` already selects it) while this precision change is worth
    1.00x-1.11x on CUDA but 1.34x-1.78x on CPU. It is kept for two reasons that do
    not depend on that: it removes the one layer that contradicted the algorithm
    layer's own default, and it makes the CPU fallback materially faster without
    costing anything on the GPU.
    """
    warm_start: bool = True
    """Warm-start each Step B solve from the previous round's phase.

    Disabling it restarts Step B from flat every round. The conditioning does
    change slightly between rounds, but the previous phase is a good
    initialisation in the same basin and discarding it costs iterations.
    """

    early_stop_score: float = 0.0
    """Stop once a measured composite reaches this (0 = run every round)."""

    settle_discard: int = 2
    """Consecutive unstable frames tolerated before a measurement is taken anyway."""

    # --- target ------------------------------------------------------------
    target_shape: str = "square"
    """Far-field target: square, rectangle, triangle, star or circle."""

    target_side: int = 60
    """Target bounding width in **camera** pixels.

    The forward model only ever sees its own far-field grid, so this is
    converted with the calibrated ``camera_px_per_model_px``.
    """

    w_uniformity: float = 0.4
    w_efficiency: float = 0.6
    """Weight on encircled energy.

    Must stay non-zero: optimising ``-CV`` alone empties the target box
    (hardware measured encircled energy collapsing to 0.002).
    """

    # --- Step A: probe-based forward-model fit -----------------------------
    probe_count: int = 8
    """Distinct probes cycled through one optimizer state per round.

    More than one is required, not merely helpful: a single probe leaves Adam in
    whichever stationary point of the non-convex intensity fit it entered.
    """

    probe_spread: float = 4.0
    """Probe phase standard deviation in radians.

    Must be well above ~3 rad; a near-flat pupil produces a focus whose
    normalised intensity is almost invariant to a smooth low-order aberration,
    and the fit is then blind regardless of iteration count.
    """

    step_a_iterations: int = 80
    holdout_probe_count: int = 2
    """Independent phase probes never used by Step A's fit."""
    freeform_holdout_count: int = 0
    """Additional independent pixel-wise phase probes before Step B (runner
    option ``--freeform-holdout-count``). Zero keeps the historical run."""

    probe_split_hook: ProbeSplitHook | None = None
    """Override which probes train Step A and which are held out.

    ``None`` (default) keeps the historical seed arithmetic exactly:
    ``seed + index * 1000 + k`` for the fit probes and the same offset by
    ``1_000_000`` for the held-out ones. A hook is called as
    ``hook(seed, index, probe_count, holdout_probe_count)`` and returns
    ``(train_seeds, holdout_seeds)``; it must supply the exact requested
    counts. When freeform holdouts are enabled, ``holdout_probe_count`` passed
    to the hook is the standard count plus ``freeform_holdout_count``. Seeds
    must not overlap the two halves, because a reused seed turns
    the held-out correlation into an in-sample one.
    """

    # --- uncertainty-sampled probes (opt-in; R-82 / R-86) -------------------
    uncertainty_probes: bool = False
    """Choose Step A probes by maximising predictive uncertainty (R-82).

    ``False`` (default) keeps the historical fixed-random probe set byte-for-byte.
    When ``True``, each Step A probe is the phase that propagates the current
    coefficient posterior to the largest far-field variance
    (:class:`UncertaintyProbeSelector`), so probes concentrate on the aberration
    directions the fit is least sure about instead of re-probing well-identified
    ones. The selector is seeded from the round's train seeds, so the run-to-run
    seed invariant is preserved; only the probe *phase* changes, not the seed
    arithmetic.
    """

    uncertainty_on_rejection: bool = False
    """Escalate a rejected round to uncertainty probes (R-86).

    Independent of :attr:`uncertainty_probes`: when a round follows a rejection
    (``rejection_streak > 0``), that round's probes switch to uncertainty
    sampling even if the base run uses fixed-random probes. This replaces the
    blind ``escalate_probe_count`` extra-random-probes with probes aimed at the
    residual uncertainty that caused the rejection.
    """

    uncertainty_probe_samples: int = 6
    """Monte-Carlo coefficient samples ``K`` in the predictive-variance estimate."""

    uncertainty_probe_steps: int = 30
    """Normalised-ascent steps taken per uncertainty probe before it is measured."""

    uncertainty_probe_diversity: float = 0.5
    """Penalty pushing successive probes apart (see the selector)."""

    uncertainty_probe_ridge: float = 1e-3
    """Ridge ``lambda`` regularising the Fisher inverse that forms the posterior."""

    min_holdout_correlation: float = 0.5
    """Require each independent CCD/model Pearson correlation before Step B."""

    holdout_on_sim: bool = False
    """Measure the held-out probes on the simulated backend too.

    ``False`` (default) keeps the historical behaviour: on ``cam_type="sim"`` the
    independent probes are never measured, so ``holdout_correlation`` stays NaN
    and the forward-model mismatch guard cannot fire. That is right for a smoke
    run -- the guard exists to stop *hardware* from committing a phase the model
    merely likes, and there is no such risk in simulation.

    Set it when the held-out score itself is the measurement, which is what the
    K-fold cross-validation harness does: it needs a number per fold, and it
    obtains it by running the real loop rather than a re-implementation of Step A.

    Note this changes *what is measured* on sim (extra
    ``holdout_probe_count`` probe round-trips per round), so it is opt-in. The
    mismatch **abort** stays gated on ``cam_type != "sim"`` even when this is on:
    a simulation that aborts would turn a fold into a missing score, and the
    harness wants the score, not a pass/fail verdict.
    """
    step_a_lr: float = 0.05
    """Adam lr for the shared aberration fit.

    0.05 is the measured sweet spot for the cycling loop; 0.1 leaves a 0.30 rad
    full-vector error and 0.01 leaves 0.94 rad, while 0.2 diverges outright.
    """

    n_orders: int = 10
    frozen_modes: tuple[int, ...] = (1, 2, 3)
    """1-based Noll indices held at zero for the whole fit.

    Piston and tilt are unidentifiable from a far-field *intensity*: leaving
    them free put 56% of the fitted coefficient norm into those degenerate
    directions without improving the fit at all.
    """

    forward_checkpoint: str | None = None
    """Path to a train_amp ``best_coefficients.pt`` used to seed the Step-A fit.

    ``None`` (default) keeps the historical all-zero seed. When set, the stored
    coefficients are loaded, checked against this run's geometry, and used as the
    starting point; a mismatch is refused rather than silently ignored.
    """

    prior_run_dir: str | None = None
    """Previous accepted hardware run used to warm-start a short Step-A refit."""

    assume_unverified_geometry: bool = False
    """Acknowledge that the checkpoint records no ``radius``/``center_crop``/
    ``conserve_energy`` (it cannot), so those were reconstructed from defaults.

    Required to be True for a checkpoint to be accepted -- the loader cannot verify
    them, and the run refuses rather than presenting an unverified reconstruction as
    if it were checked.
    """

    # --- Step B: target synthesis with the aberration frozen ----------------
    step_b_iterations: int = 600
    step_b_lr: float = 0.05

    # --- bench model / calibration -----------------------------------------
    region: int = 64
    """Forward-model pupil grid edge, in model pixels."""

    wavelength_nm: int = 1064
    slm_pixel_um: float = 8.0
    camera_pixel_um: float = 2.2
    """CCD pixel pitch -- the bench model's measurement anchor.

    The measured focal scale fixes only the ratio ``f / p_cam``, so exactly one
    external anchor is unavoidable. This one is a per-camera datasheet constant
    rather than a bench assembly choice.
    """

    panel_span_px: float = 900.0
    """Width in **panel** pixels of the region the model grid covers.

    The model grid samples the pupil at ``panel_span_px / region`` times the SLM
    pitch, not at the SLM pitch itself. Getting this wrong makes the modelled
    aperture ~3.5x smaller than the real beam, and then no aperture candidate can
    correlate with the measurement.
    """

    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL
    """Beam centre in **panel** pixels as ``(x, y)``.

    Must be measured on the panel. The camera's 0-order is a different coordinate
    frame (on this bench the axes are swapped and the scales differ by >10x), so
    deriving one from the other writes the phase where the beam is not.
    """

    far_field_padding: int = 8
    """Zero-padding factor for the analytical far field.

    Also the cost driver: the FFT is quadratic in the padded size, so this is
    squared in both time and memory.
    """
    pupil_oversampling: int = 1
    """Bilinear sub-pixel pupil interpolation before the far-field FFT.

    ``1`` keeps the model on the command grid. Higher factors enlarge the pupil
    by that factor, propagate it at the matching finer far-field sampling, and
    crop back to ``region`` -- so the far-field *field of view* is unchanged and
    only its sampling improves. Costs ``factor**2`` in FFT time and memory.
    """

    focal_length_m: float = 0.0
    """2f lens focal length in metres; 0 derives it from the measured scale.

    The measured tilt-shift scale fixes only the ratio ``f / p_cam``, so one
    external anchor is unavoidable; the camera pixel pitch is used for it
    because that is a datasheet constant rather than a bench assembly choice.
    """

    n_calibration_probes: int = 8
    geometry_file: str | None = None
    """Optional independently measured sweep geometry JSON."""

    error_params: str | None = None
    """Optional calibration JSON holding the forward model's error terms.

    The file may be a twin fit result (``tools/slm/bench_twin_calibrate.py``'s
    ``save_fit`` output) or a plain :class:`ModelErrorTerms` dictionary; both
    are read by
    :func:`~ao_shaping.utils.wavefront.model_error_terms.load_error_terms`.
    The error terms it carries (the SLM's LUT nonlinearity above all) are
    applied by the forward model, so Step A's fit and Step B's synthesis see
    the panel the way it really behaves instead of assuming "command phase =
    actual phase".

    ``None`` (default) keeps the ideal model, whose output is bit-for-bit the
    one this path had before the error terms existed.
    """

    error_terms: ModelErrorTerms | None = None
    """The resolved error terms, with THIS run's optical prescription attached.

    Filled in by ``__post_init__`` from :attr:`error_params`; the geometry
    (wavelength / focal length / far-field pitch) is taken from this run, not
    from the calibration file, because the file was measured on whichever grid
    its own run used.
    """
    min_geometry_correlation: float = 0.5
    """Abort before shaping if the calibrated geometry scores below this.

    The geometry is a *bake-off*: it comes from a model of the bench, so if it is
    wrong the shaped spot gets worse rather than better. Refusing to shape on an
    uncalibrated model costs one calibration pass instead of the run.
    """

    # --- self-calibration (GRAPHYCS-style) -----------------------------------
    fourier_weight: float = 0.0
    """Weight α of the Fourier-domain L1 term in Step A's loss.

    ``0`` (default) = the term is off and Step A's output is bit-for-bit the
    one it had before the term existed. The paper uses 7e-4 (simulation) /
    5e-4 / 1e-3, and its sensitivity scan shows the result is flat over
    1e-5…1e-3.
    """

    loss_kind: str = "mse"
    """Image-domain loss: ``"mse"`` (the historical peak-normalised MSE) or
    ``"l1"`` (the paper's choice).

    The default stays ``"mse"`` on purpose: swapping it changes the
    optimisation's basin, so it must be an explicit opt-in, not a silent
    change. The two are compared on recorded hardware probes by
    ``scripts/slm/model_in_loop_selfcal_bakeoff.py``.
    """

    selfcal_affine: bool = False
    """Enable the pupil-plane affine self-calibration τ_θ (6 scalars).

    Off by default, so the forward model's output stays bit-for-bit identical.
    When on, the affine is fitted *inside* Step A (it is part of the forward
    model, not a term in the loss) and is carried across rounds.
    """

    selfcal_travel: bool = False
    """Enable the per-mode-family travel scale γ (7 scalars).

    Off by default. γ scales the **command** phase only -- never the
    aberration being estimated -- exactly as in the paper.
    """

    selfcal_lr: float = 1e-3
    """Adam learning rate of the self-calibration parameters.

    The paper uses 1e-2 for the Zernike coefficients and 1e-6…1e-3 for the
    affine θ, and gives γ the same rate as θ.
    """

    selfcal_theta_step_max: float = 0.02
    """Per-round trust region on the pupil affine τ_θ (per parameter).

    ``<= 0`` disables the guard. The paper has **no** trust region, which is
    exactly the failure mode this repository refuses: a calibration quantity
    that sprints away from its start means the forward model is being bent to
    fit the data rather than describing the bench.
    """

    selfcal_gamma_step_max: float = 0.05
    """Per-round trust region on the travel scale γ (per family).

    ``<= 0`` disables the guard. Same reasoning as
    :attr:`selfcal_theta_step_max`.
    """

    # --- coupling guards ----------------------------------------------------
    trust_region_c_l2: float = 0.5
    """Fixed per-round L2 cap on the coefficient step, radians.

    ``<= 0`` disables the guard, which is only appropriate when the fit is
    externally constrained. Used verbatim when
    :attr:`adaptive_trust_region` is off.
    """

    adaptive_trust_region: bool = False
    """Let the cap adapt to accept/reject evidence instead of staying fixed.

    **Defaults to False, and the default path is unchanged**: with the flag off
    the loop clamps with :attr:`trust_region_c_l2` on every round, exactly as
    before. With it on, the cap grows after consecutive accepts and shrinks hard on
    any rejection, bounded by ``[0.25x, 4x]`` of the configured radius.

    The clamp itself is still
    :func:`trust_region_clamp`; only the radius becomes a schedule. Keeping the
    single clamp implementation is the point -- a second opinion about where the
    boundary is would be exactly the drift this worktree has been guarding
    against elsewhere.
    """
    """Max L2 norm of the per-round coefficient change (0 disables the guard)."""

    acceptance_loss_delta: float = 1e-4
    acceptance_score_eps: float = 1e-3
    max_cv_regression: float = 0.02
    damp_lr_factor: float = 0.5
    """Step-A lr multiplier applied after a rejected round."""

    max_rejection_streak: int = 3
    """Consecutive rejections tolerated before aborting."""

    escalate_probe_count: int = 2
    max_probe_count: int = 16

    # --- hardware -----------------------------------------------------------
    cam_type: str = "daheng"
    cam_id: int = 0
    exposure_time_ms: float = 0.0
    """0 starts Daheng manual auto exposure at 0.02 ms."""

    max_exposure_ms: float = 0.0
    """Daheng exposure ceiling in ms; the optical path sets it, so it is
    per-session, never a package constant.

    0 = auto: read ``AO_DAHENG_MAX_EXPOSURE_MS``, else probe the CCD image
    with a doubling ladder until the peak saturates.  A positive value is
    used verbatim."""

    cam_size: int = 300
    slm_number: int = 1
    phase_disturbance_file: str | None = None
    """Fixed raw-radian panel bias added to every hardware phase display."""
    blaze_period_px: int = -1
    """-1 disables off-axis carrier; 0 measures candidate periods; >=4 tests one."""
    blaze_direction: str = "vertical"
    slm_wavelength: int = 1064
    """SLM operating wavelength in nm.

    The lab has more than one SLM, and a wrong wavelength reprograms the panel's
    phase table, so this is stated rather than assumed.
    """
    send_overlay: PhaseOverlay | None = field(default=None, repr=False)
    """Send-pipeline overlay (GUI 离焦 / 底相位 T4 叠加层) applied to the SLM
    this run opens (driver slot on a real Santec; no-op on a sim SLM that lacks
    the setters). ``None`` (default) = no overlay, so CLI runs are untouched;
    the GUI entry fills it from the session via ``build_send_overlay``."""

    focus_dir: str | None = None
    """Directory holding the focus calibration evidence (``focus.json``).

    The 2f bench sits permanently **out of focus**; a forward model fitted on
    an uncorrected bench predicts a **broader** spot than the hardware
    produces (measured width ratio 1.328, CV bias -0.3009 over 51/51 records
    -- ``report/shape_trials/experiment_summary_20261022.md`` §2).  When
    ``focus_compensation`` is on, the Fresnel focus phase built from this
    directory is added to **every** displayed pupil phase.

    ``None`` (default) = no focus evidence, so nothing is prepended.
    """

    focus_compensation: bool = False
    """Enable the focus (defocus) compensation preflight stage.

    **Off by default**: the preflight is a new capability and must not change
    the behaviour of an existing run.  With it off the panel phase that
    reaches the SLM is the very same array it was before -- the run stays
    bit-identical.

    With it on and no usable ``focus_dir``, the run **aborts** rather than
    quietly collecting on an uncorrected bench; the evidence file
    (``focus_preflight.json``) is written either way.
    """

    focus_live_probe: bool = False
    """Allow an on-the-spot focus calibration when ``focus_dir`` is unset.

    This **opens the panel and the camera**.  Offline paths never set it.
    """

    n_eval_frames: int = 4
    """Frames averaged per measurement."""

    sim_aberration: dict[int, float] = field(default_factory=dict)
    """**Simulated bench only.** Ground-truth aberration as ``{noll: radians}``.

    Injected by :class:`_SimBench` on every displayed phase. It exists so the
    offline path exercises the thing the loop is actually for: with a *perfect*
    pupil there is no aberration to find, Step A fits pure noise, every round
    produces a different random coefficient vector, and the acceptance guard
    correctly refuses all of them -- a correct but vacuous test. With a known
    aberration present, the fit has real signal and the round-to-round
    coefficient change should shrink as the estimate settles.

    Ignored when ``cam_type`` is a hardware backend.
    """

    settle_wait_s: float = 0.5
    settle_tol: float = 0.02
    settle_max_wait_s: float = 6.0
    settle_max_discard: int = 40

    # --- display ------------------------------------------------------------
    show: bool = False
    """Open a live pygame window during the run.

    Shares the four-panel view the ``slm-pib`` runner uses -- see
    :class:`~ao_shaping.display.SlmModelInLoopDisplay`, which subclasses
    :class:`~ao_shaping.display.SlmZernikeDisplay` and swaps the Zernike
    coefficient bars for the forward model's **predicted** far field. Panels:
    measured CCD, phase on the SLM, predicted far field, per-round metric curve.

    Display is diagnostics only: it can never change what the loop optimises or
    which phase it commits, and a prediction that cannot be produced degrades to
    a blank panel instead of aborting a run that has already spent hardware
    time. Closing the window stops the search after the current round, then
    still runs the final bake-off, so the best *already verified* phase is kept.
    """

    # --- bookkeeping --------------------------------------------------------
    output_subdir: str = "slm_model_in_loop"
    debug_dir: str | None = None
    """Optional preallocated run directory for preflight evidence on failure."""
    callback: Callable[[int, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    step_sink: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    """Per-**MEASUREMENT** hook for the ``--debug`` per-step artefact dump.

    Called as ``step_sink(stage, payload)`` once for every frame this run
    measures, and *before* any ``break``.

    WHY this cannot be folded into ``callback``: ``callback`` fires once per
    round, as the very last statement of the round body, and three ``break``
    statements precede it -- the rejection-streak abort, the early stop and the
    display-closed stop. So it is structurally blind to exactly the rounds a
    debug dump exists for: the one that was rolled back and the one that ended
    the run. It also never sees the probe loops' per-probe arrays at all, because
    the only thing it receives is the round row, and that row is scalars-only by
    contract.

    ``stage`` is a closed vocabulary of exactly four strings, and the runner-side
    consumer switches on them: ``"flat"`` (the flat baseline), ``"calib-probe"``
    (the geometry bake-off probes), ``"stepA-probe"`` (the Step-A probe loop --
    the hot path, ``probe_count`` per round) and ``"stepB-shaped"`` (the Step-B
    shaped display). ``payload`` always carries the same eight keys:

    * ``phase_rad`` -- the **model-grid** phase in raw unwrapped radians, shape
      ``(region, region)``, float64. Never the panel array: the panel is a
      different coordinate frame (on this bench the axes are swapped and the
      scales differ by >10x).
    * ``frame`` -- the *prepared* frame (the output of :func:`prepare_frame`),
      i.e. background-subtracted and non-negative.
    * ``raw_peak`` / ``raw_p999`` / ``raw_mean`` -- ``float`` scalars of the
      **raw** ``bench.measure()`` return, taken *before* :func:`prepare_frame`.
      Median subtraction and clipping destroy the saturation ceiling, and
      detecting an over- or under-exposed frame is one of the main reasons to
      dump a step at all.
    * ``round`` -- ``0`` for the two pre-loop stages, ``index + 1`` in the loop.
    * ``probe_index`` / ``probe_count`` -- the probe ordinal inside its own loop
      and that loop's denominator, so a consumer can print ``3/8``; ``None`` when
      the stage is not a probe stage.

    An exception raised here propagates: a silently swallowed failure would burn
    hardware minutes writing nothing.
    """
    progress_every: int = 1

    def __post_init__(self) -> None:
        """Validate the budget, optical carrier and frozen-mode list."""
        if self.blaze_period_px < -1 or 0 < self.blaze_period_px < BLAZE_PERIOD_MIN_PX:
            raise ValueError(
                f"blaze_period_px must be -1 (off), 0 (auto), or >={BLAZE_PERIOD_MIN_PX}"
            )
        if self.blaze_direction not in ("vertical", "horizontal"):
            raise ValueError("blaze_direction must be vertical or horizontal")
        if self.blaze_period_px >= 0 and self.cam_type == "sim":
            raise ValueError("blaze calibration requires a full-frame hardware CCD")
        if self.focus_compensation and self.cam_type == "sim":
            # The digital twin is in focus by construction (its forward model
            # is a plain FFT), so a Fresnel focus phase would *defocus* it.
            # Accepting the flag as a silent no-op would be worse than saying
            # so.
            raise ValueError(
                "focus_compensation is a hardware-only stage: the simulated "
                "bench is already in focus, so the focus phase would defocus it"
            )
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {self.n_rounds}")
        if self.probe_count < 2:
            # One probe is not a degenerate case, it is a broken configuration:
            # the non-convex fit has no phase diversity to escape local minima.
            raise ValueError(
                f"probe_count must be >= 2 (a single probe cannot identify the "
                f"aberration), got {self.probe_count}"
            )
        if self.holdout_probe_count < 1:
            raise ValueError("holdout_probe_count must be >= 1")
        if self.freeform_holdout_count < 0:
            raise ValueError("freeform_holdout_count must be >= 0")
        if not -1 <= self.min_holdout_correlation <= 1:
            raise ValueError("min_holdout_correlation must lie in [-1, 1]")
        if self.probe_spread < 3.0:
            raise ValueError(
                f"probe_spread must be >= 3.0 rad for the fit to be identifiable "
                f"(a near-flat pupil's focus barely responds to a smooth "
                f"low-order aberration), got {self.probe_spread}"
            )
        if self.step_a_iterations < 0 or self.step_b_iterations < 1:
            raise ValueError(
                "step_a_iterations must be >= 0 and step_b_iterations must be >= 1"
            )
        if not math.isfinite(self.max_cv_regression) or self.max_cv_regression < 0:
            raise ValueError("max_cv_regression must be finite and non-negative")
        if not 0.0 < self.damp_lr_factor < 1.0:
            raise ValueError(
                f"damp_lr_factor must lie in (0, 1), got {self.damp_lr_factor}"
            )
        if self.target_side < 1:
            raise ValueError(f"target_side must be >= 1, got {self.target_side}")
        # Single vocabulary (issue #199 P1): the end-to-end supported set of this
        # chain -- non-square targets are scored through
        # ``differentiable_shaping.create_target_mask``, whose geometry mapping
        # covers exactly DIFFERENTIABLE_SHAPE_CHOICES (a subset view of the
        # canonical TARGET_SHAPE_CHOICES).
        if self.target_shape not in DIFFERENTIABLE_SHAPE_CHOICES:
            raise ValueError(
                f"unsupported target_shape: {self.target_shape}; must be one of "
                f"{DIFFERENTIABLE_SHAPE_CHOICES}"
            )
        if self.region < 2:
            raise ValueError(f"region must be >= 2, got {self.region}")
        if self.n_orders < 1:
            raise ValueError(f"n_orders must be >= 1, got {self.n_orders}")
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )
        if self.pupil_oversampling not in (1, 2, 4):
            raise ValueError("pupil_oversampling must be 1, 2 or 4")
        if not math.isfinite(self.fourier_weight) or self.fourier_weight < 0.0:
            raise ValueError("fourier_weight must be finite and non-negative")
        if self.loss_kind not in ("mse", "l1"):
            raise ValueError(f"loss_kind must be 'mse' or 'l1', got {self.loss_kind!r}")
        if not math.isfinite(self.selfcal_lr) or self.selfcal_lr <= 0.0:
            raise ValueError("selfcal_lr must be a positive finite float")
        if not math.isfinite(self.selfcal_theta_step_max):
            raise ValueError("selfcal_theta_step_max must be finite")
        if not math.isfinite(self.selfcal_gamma_step_max):
            raise ValueError("selfcal_gamma_step_max must be finite")
        if self.panel_span_px <= 0:
            raise ValueError(f"panel_span_px must be > 0, got {self.panel_span_px}")
        if self.cam_type == "daheng" and (
            not math.isfinite(self.max_exposure_ms)
            or self.max_exposure_ms < 0
            or not math.isfinite(self.exposure_time_ms)
            or self.exposure_time_ms < 0
            or (
                self.max_exposure_ms > 0
                and self.exposure_time_ms > self.max_exposure_ms
            )
        ):
            raise ValueError(
                "Daheng exposure_time_ms must be non-negative and may not exceed a "
                "positive max_exposure_ms (0 = auto: AO_DAHENG_MAX_EXPOSURE_MS, "
                "else probed from the CCD image)"
            )
        if self.forward_checkpoint is not None:
            # Fail at config time, before any device opens: a typo in the seed
            # path must not burn a hardware run to be discovered in Stage 2.
            if not Path(self.forward_checkpoint).exists():
                raise ValueError(
                    f"forward_checkpoint does not exist: {self.forward_checkpoint!r}"
                )
        if self.prior_run_dir is not None:
            if self.forward_checkpoint is not None:
                raise ValueError("prior_run_dir and forward_checkpoint are exclusive")
            if not Path(self.prior_run_dir).is_dir():
                raise ValueError(
                    f"prior_run_dir does not exist: {self.prior_run_dir!r}"
                )
        if self.geometry_file is not None and not Path(self.geometry_file).is_file():
            raise ValueError(f"geometry_file does not exist: {self.geometry_file!r}")
        self.frozen_modes = tuple(sorted({int(m) for m in self.frozen_modes}))
        # Resolve the focal length once, here, so the runner, a library caller and
        # the tests can never disagree about what geometry a run actually used.
        if self.focal_length_m <= 0.0:
            from ao_shaping.drivers.slm.factory import DEFAULT_WAVELENGTH_NM
            from ao_shaping.optimizer.wfless.slm_gs_refine import (
                _TILT_SHIFT_SCALE_MEASURED_AT_NM,
                _TILT_SHIFT_SCALE_PX,
                focal_length_from_camera_pixel,
            )

            self._focal_length_pinned = False
            self.focal_length_m = focal_length_from_camera_pixel(
                wavelength_nm=float(self.slm_wavelength or DEFAULT_WAVELENGTH_NM),
                camera_pixel_um=self.camera_pixel_um,
                slm_pixel_um=self.slm_pixel_um or SLM_PITCH_M * 1e6,
                focal_scale_px=_TILT_SHIFT_SCALE_PX,
                scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
            )
        else:
            self._focal_length_pinned = True

        # Resolve the error terms LAST: they need the focal length and the
        # far-field grid, both of which are only known here.
        self.error_terms = self._resolve_error_terms()

    def _resolve_error_terms(self) -> ModelErrorTerms | None:
        """Load the error terms and re-anchor them on this run's optics.

        The calibration file carries the geometry of **its own** run (the
        wavelength, the focal length and the far-field sample spacing). A
        defocus in metres and a tilt in radians are only meaningful against a
        specific prescription, so the geometry is replaced by this run's.

        Returns:
            The resolved error terms, or ``None`` when no file was given.

        Raises:
            ValueError: The file does not exist, cannot be read, or carries
                error terms this run's optics cannot express.
        """
        if self.error_params is None:
            # No file: keep whatever was handed in (normally ``None``, i.e.
            # the ideal model). A manually supplied container survives a
            # ``dataclasses.replace`` of the config.
            return self.error_terms
        source = Path(self.error_params)
        if not source.is_file():
            raise ValueError(f"error_params does not exist: {self.error_params!r}")
        from ao_shaping.utils.wavefront.model_error_terms import load_error_terms

        terms = load_error_terms(source)
        far_field_size = int(self.region) * int(self.far_field_padding)
        pupil_pitch_m = (
            float(self.panel_span_px)
            / float(self.region)
            * float(self.slm_pixel_um)
            * 1e-6
        )
        if pupil_pitch_m <= 0.0:
            raise ValueError(
                "the model grid's pupil pitch is not positive; check "
                "panel_span_px / region / slm_pixel_um"
            )
        far_field_pitch_m = (
            float(self.wavelength_nm)
            * 1e-9
            * float(self.focal_length_m)
            / (far_field_size * pupil_pitch_m)
        )
        terms = terms.with_optics(
            wavelength_m=float(self.wavelength_nm) * 1e-9,
            focal_length_m=float(self.focal_length_m),
            far_field_pitch_m=far_field_pitch_m,
        )
        terms.validate_focal_geometry()
        if self.cam_type == "sim":
            logger.warning(
                "error params are set on the simulated bench, whose far field "
                "comes from the analytical bench model and carries no LUT "
                "nonlinearity; the error terms will therefore make the forward "
                "model disagree with the bench it is compared against"
            )
        logger.info(
            "forward-model error terms from {}: gamma={:.4f} defocus={:.3e} m "
            "tilt=({:.3e}, {:.3e}) rad (far-field pitch {:.4e} m)",
            source,
            float(terms.lut_gamma),
            float(terms.camera_defocus_m),
            float(terms.tilt_x_rad),
            float(terms.tilt_y_rad),
            far_field_pitch_m,
        )
        return terms


@dataclass
class ModelInLoopResult:
    """Everything one run produced.

    A dedicated result type rather than attributes bolted onto the ``Recorder``:
    the extra fields are then declared and type-checked instead of being
    dynamically attached, and a caller cannot silently read one that the run
    never set.

    Attributes:
        recorder: The history recorder. Pass it to ``save_dataframe`` or
            ``save_recorder_debug_artifacts`` to persist the run.
        status: One of the :class:`ModelInLoopStatus` values.
        best_phase: The adopted phase, raw unwrapped radians, model grid. This
            is the flat phase when nothing beat it, so it is always safe to send.
        best_frame: The far-field frame that phase actually produced.
        geometry: The bench geometry used, fitted on hardware and taken by
            construction on the simulated bench.
        target_side_camera: Target bounding width in camera pixels.
        target_side_model: The same side in forward-model pixels.
        flat_score: Composite score of the flat baseline.
        best_score: Composite score of ``best_phase``.
        config: The resolved configuration, for provenance.
    """

    recorder: Any
    status: str
    best_phase: np.ndarray
    best_frame: np.ndarray | None
    geometry: Any
    target_side_camera: int
    target_side_model: int
    flat_score: float
    best_score: float
    config: SlmModelInLoopConfig
    probe_records: list[dict[str, Any]] = field(default_factory=list)
    fitted_coefficients: np.ndarray | None = None
    source_amplitude: np.ndarray | None = None
    blaze_params: Any | None = None
    blaze_choice: Any | None = None


@dataclass
class RoundRecord:
    """Per-round record of one fit -> shape -> re-measure iteration.

    Attributes:
        round_index: Zero-based round index.
        stage: ``"calibration"`` for the pre-loop baseline, else the round index.
        accepted: Whether this round's phase and coefficients were committed.
        reason: Acceptance verdict tag, or the abort reason.
        loss_before: Step-A loss at the start of the fit.
        loss_after: Step-A loss after the fit.
        step_a_iterations: Adam steps Step A actually took.
        step_b_iterations: Adam steps Step B actually took.
        coeff_step_l2: L2 norm of the raw coefficient change this round.
        coeff_step_applied: L2 norm actually applied after the trust region.
        clamp_applied: Whether the trust region bit.
        coeff_norm: L2 norm of the committed coefficient vector, radians.
        score_before: Composite score held before the round.
        score_after: Composite score measured after Step B.
        cv_after: Coefficient of variation inside the frozen target box.
        ee_after: Encircled energy inside the frozen target box.
        uniformity_cv: Alias kept so generic reporters can read one key.
        probe_count: Probes used this round (escalates after rejections).
        step_a_lr: Step-A learning rate used this round (damps after rejections).
        rejection_streak: Consecutive rejections *before* this round.
        phase: The committed phase, raw unwrapped radians, or ``None`` on reject.
        selfcal_affine: The pupil affine τ_θ after this round, if that
            self-calibration is on (empty dict otherwise).
        selfcal_travel: The travel scale γ after this round, if that
            self-calibration is on (empty dict otherwise).
        selfcal_theta_step: Largest per-parameter change of τ_θ this round.
        selfcal_gamma_step: Largest per-family change of γ this round.
        selfcal_violated: Whether the self-calibration trust region was
            exceeded (the round is then rejected).
    """

    round_index: int
    stage: str
    accepted: bool
    reason: str
    loss_before: float
    loss_after: float
    step_a_iterations: int
    step_b_iterations: int
    coeff_step_l2: float
    coeff_step_applied: float
    clamp_applied: bool
    coeff_norm: float
    score_before: float
    score_after: float
    cv_after: float
    ee_after: float
    uniformity_cv: float
    probe_count: int
    step_a_lr: float
    rejection_streak: int
    phase: np.ndarray | None = field(default=None, repr=False)
    selfcal_affine: dict[str, float] = field(default_factory=dict)
    selfcal_travel: dict[str, float] = field(default_factory=dict)
    selfcal_theta_step: float = 0.0
    selfcal_gamma_step: float = 0.0
    selfcal_violated: bool = False


# ---------------------------------------------------------------------------
# Bench abstraction: one loop, two backends
# ---------------------------------------------------------------------------


class Bench(Protocol):
    """What the loop needs from a bench, whether simulated or physical.

    Deliberately narrow: display a model-grid phase, read one raw frame, close.
    Everything else (preprocessing, ROI, metrics, guards) is backend-independent
    and therefore exercised identically by the offline and hardware paths.
    """

    def display(self, phase_model: np.ndarray) -> None:
        """Present ``phase_model`` (raw radians, model grid) to the bench."""
        raise NotImplementedError

    def measure(self) -> np.ndarray:
        """Return one raw far-field frame, exactly as the detector gives it."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the devices. Must be idempotent."""
        raise NotImplementedError


#: Structural stand-in for ``CalibrationRecord``; the concrete class is imported
#: lazily so this module's import graph stays shallow.
CalibrationRecordLike = Any


def _raw_frame_scalars(raw: np.ndarray) -> tuple[float, float, float]:
    """``(peak, p99, mean)`` of a **raw** detector frame.

    Call this on ``bench.measure()``'s return value *before* handing that array to
    :func:`prepare_frame`: median
    subtraction and clipping at zero remove the saturation ceiling, and knowing
    whether a step was over- or under-exposed is one of the reasons to dump it.

    ``p99`` rather than the median because the frame is overwhelmingly background,
    so the median says nothing about the beam; ``peak`` because a clipped top
    pixel is the one symptom that is unambiguous. Returned as plain ``float`` so a
    JSON/CSV consumer is not handed ``numpy.float64``.

    Args:
        raw: One frame exactly as ``bench.measure()`` produced it.

    Returns:
        ``(float(frame.max()), float(percentile(frame, 99)), float(frame.mean()))``.
    """
    return float(raw.max()), float(np.percentile(raw, 99)), float(raw.mean())


def _step_payload(
    *,
    phase_rad: np.ndarray,
    frame: np.ndarray,
    raw_scalars: tuple[float, float, float],
    round_index: int,
    probe_index: int | None,
    probe_count: int | None,
) -> dict[str, Any]:
    """Build the ``step_sink`` payload from one measurement.

    A single constructor rather than four dict literals, so the eight keys cannot
    drift between the four call sites -- the runner-side consumer binds to these
    names exactly.

    ``phase_rad`` and ``frame`` are passed through by reference, not copied: the
    hook is on the hot path (``probe_count`` per round), and none of the four
    sites mutates either array afterwards.

    Args:
        phase_rad: Model-grid phase that produced the frame, raw unwrapped radians.
        frame: Prepared frame (output of :func:`prepare_frame`).
        raw_scalars: Output of :func:`_raw_frame_scalars`, taken before
            preparation.
        round_index: ``0`` for the pre-loop stages, else ``index + 1``.
        probe_index: Ordinal within the owning probe loop, else ``None``.
        probe_count: Denominator for that loop, else ``None``.

    Returns:
        The payload dict with exactly the eight documented keys.
    """
    raw_peak, raw_p999, raw_mean = raw_scalars
    return {
        "phase_rad": phase_rad,
        "frame": frame,
        "raw_peak": raw_peak,
        "raw_p999": raw_p999,
        "raw_mean": raw_mean,
        "round": int(round_index),
        "probe_index": None if probe_index is None else int(probe_index),
        "probe_count": None if probe_count is None else int(probe_count),
    }


def _probe_phase(
    region: int, spread: float, seed: int, *, smooth: bool = False
) -> np.ndarray:
    """Strong, zero-mean Gaussian pupil phase used to interrogate the bench.

    Step A cannot identify a smooth low-order aberration from a near-flat pupil:
    the focus is a near-delta whose normalised intensity barely moves, at the
    same order as the detector's quantisation pedestal. A std of a few radians
    clears that pedestal by orders of magnitude and makes the fit well posed.

    Args:
        region: Model grid edge length.
        spread: Standard deviation in radians.
        seed: Seed for the draw, so a round's probe set is reproducible.

    Returns:
        A ``(region, region)`` float64 array of raw radians.
    """
    phase = np.random.default_rng(int(seed)).normal(
        0.0, 1.0, (int(region), int(region))
    )
    if smooth:
        from scipy.ndimage import gaussian_filter

        # Vary bandwidth across seeds while keeping the phase reproducible.
        sigma = max(2.0, int(region) / (8.0 + int(seed) % 5))
        phase = gaussian_filter(phase, sigma=sigma, mode="reflect")
    phase -= phase.mean()
    return phase * (float(spread) / max(float(phase.std()), 1e-12))


def _structured_probe_phase(region: int, spread: float, seed: int) -> np.ndarray:
    """Independent low-order aberration probe for cross-family validation."""
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    rng = np.random.default_rng(int(seed))
    coefficients = {
        noll: float(value)
        for noll, value in zip(range(4, 12), rng.normal(size=8), strict=True)
    }
    raw_phase = generate_zernike_phase(
        coefficients, (int(region), int(region)), n_max=4
    )
    aperture = np.isfinite(raw_phase)
    phase = np.nan_to_num(raw_phase).astype(np.float64)
    phase[aperture] -= phase[aperture].mean()
    phase *= float(spread) / max(float(phase[aperture].std()), 1e-12)
    return phase


def _holdout_phase(
    region: int, spread: float, seed: int, index: int, standard_count: int
) -> tuple[np.ndarray, str]:
    if index >= standard_count:
        return _probe_phase(region, spread, seed), "freeform"
    if index < 2:
        return _probe_phase(region, spread, seed, smooth=True), "smooth"
    return _structured_probe_phase(region, spread, seed), "zernike"


def _select_uncertainty_probes(
    optimizer: Any,
    config: SlmModelInLoopConfig,
    seeds: list[int],
    *,
    prev_probes: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Pick Step A probes by maximising far-field predictive uncertainty (R-82/R-86).

    Args:
        optimizer: Step A optimizer already carrying the running coefficients and
            source amplitude (as built by :func:`_make_optimizer`).
        config: Loop config; supplies region, probe spread and the selector knobs.
        seeds: This round's train seeds; the first seeds the selector RNG so the
            probe set is reproducible for the round.
        prev_probes: Probe phases measured in the previous round, folded into the
            coefficient posterior via Fisher before selecting. ``None`` on the
            first round, which then uses an isotropic prior.

    Returns:
        One ``(region, region)`` raw-radian probe per seed, mutually diverse.
    """
    from ao_shaping.algorithm.signal_processing.uncertainty_probe_selector import (
        UncertaintyProbeSelector,
    )

    spread = float(config.probe_spread)
    selector = UncertaintyProbeSelector(
        optimizer,
        seed=int(seeds[0]) if seeds else int(config.seed),
        n_samples=int(config.uncertainty_probe_samples),
        ridge=float(config.uncertainty_probe_ridge),
        spread_cap=spread,
        spread_floor=0.75 * spread,
        diversity_weight=float(config.uncertainty_probe_diversity),
        mode="maximize",
    )
    if prev_probes:
        selector.update_covariance(prev_probes)

    probes: list[np.ndarray] = []
    for _ in seeds:
        for _ in range(int(config.uncertainty_probe_steps)):
            selector.update()
        probes.append(selector.current_phase())
        selector.commit()
    return probes


class _SimBench:
    """Numerical stand-in for the 2f bench, used for CI and offline QA.

    The far field comes from the canonical analytical model
     (:func:`~ao_shaping.algorithm.signal_processing.square_shaping._forward_intensity`
     re-exported as ``forward_intensity`` from ``slm_shaping_bench``);
    camera-shaped crop around the peak emulates the detector's ROI, and additive
    read noise plus 16-bit quantisation emulate its noise floor. That last part
    matters: the whole reason Step A needs a strong probe is that the pedestal
    from quantisation is the same order as the signal a weak probe produces.
    """

    def __init__(
        self, config: SlmModelInLoopConfig, *, read_noise: float = 2.0
    ) -> None:
        from ao_shaping.drivers.sim.slm_shaping_bench import (
            ShapingBenchConfig,
            forward_intensity,
        )

        self._forward = forward_intensity
        self._config = config
        self._read_noise = float(read_noise)
        self._rng = np.random.default_rng(int(config.seed) + 977)
        self._phase = np.zeros((config.region, config.region), dtype=np.float64)
        # The model grid spans panel_span_px panel pixels, so its physical
        # aperture is that width times the SLM pitch -- not region * pitch.
        aperture_m = float(config.panel_span_px) * config.slm_pixel_um * 1e-6
        self._cfg = ShapingBenchConfig(
            n_grid=int(config.region),
            aperture_size=aperture_m,
            wavelength=float(config.wavelength_nm) * 1e-9,
            focal_length=float(config.focal_length_m),
            cn2=0.0,
            far_field_padding=int(config.far_field_padding),
            target_side_px=int(config.target_side),
            seed=int(config.seed),
        )
        self._cam_size = int(config.cam_size)
        self._aberration = self._build_aberration(config)

    @staticmethod
    def _build_aberration(config: SlmModelInLoopConfig) -> np.ndarray:
        """Render ``config.sim_aberration`` onto the model grid, or zeros.

        Built from the *same* basis the fit uses
        (:meth:`ZernikeCoefficientOptimizer.generate_basis`), so a perfect fit
        would recover the injected coefficients exactly -- which is what makes
        this a real test of Step A rather than a smoke test.
        """
        if not config.sim_aberration:
            return np.zeros((int(config.region), int(config.region)), dtype=np.float64)
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            ZernikeCoefficientOptimizer,
        )
        from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=int(config.n_orders), region=int(config.region)
        )
        basis = np.asarray(optimizer.generate_basis(), dtype=np.float64)
        n_terms = calc_n_zernike_terms(int(config.n_orders))
        vector = np.zeros(int(n_terms), dtype=np.float64)
        for noll, amplitude in config.sim_aberration.items():
            index = int(noll) - 1
            if not 0 <= index < basis.shape[0]:
                raise ValueError(
                    f"sim_aberration Noll {noll} is outside the fitted basis "
                    f"(1..{basis.shape[0]} for n_orders={config.n_orders})"
                )
            vector[index] = float(amplitude)
        logger.info(
            "sim bench: injecting ground-truth aberration {} (rad) over {} fitted modes",
            dict(config.sim_aberration),
            basis.shape[0],
        )
        return np.tensordot(vector, basis, axes=(0, 0)).astype(np.float64)

    def display(self, phase_model: np.ndarray) -> None:
        # The aberration is part of the bench, not of the command: it is added
        # to whatever phase we ask for, exactly as a real misalignment would be.
        self._phase = np.asarray(phase_model, dtype=np.float64) + self._aberration

    def measure(self) -> np.ndarray:
        far = self._forward(self._phase, self._cfg)
        # Emulate the camera window: a fixed-size crop on the peak, which is what
        # reset_window + argmax anchoring gives on the real bench.
        side = min(self._cam_size, far.shape[0], far.shape[1])
        frame = crop_around_zero_order(far, side)
        noisy = frame + self._rng.normal(0.0, self._read_noise, frame.shape)
        # 16-bit detector: half a count at a full-well peak is the pedestal that
        # a weak probe cannot clear.
        return np.clip(noisy, 0.0, 65535.0)

    def close(self) -> None:
        return None

    def true_geometry(self) -> Any:
        """Geometry implied by how this bench was built.

        The hardware path has to *fit* the geometry, which is the whole reason
        :func:`~ao_shaping.optimizer.wfless.model_in_loop_shaping.calibrate_bench_geometry`
        exists. This bench was constructed with its geometry, so fitting it would
        be theatre -- and on simulated data the correlation solve is meaningless
        anyway. The scale relation is exact: zero-padding refines the far-field
        *sampling* without changing its angular extent, so the padded grid spans
        the same angle as ``region`` model pixels and a full-width crop shows
        ``far_field_padding`` camera pixels per model pixel.
        """
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import BenchGeometry

        region = int(self._config.region)
        padding = int(self._config.far_field_padding)
        return BenchGeometry(
            panel_disc_radius=int(round(self._config.panel_span_px / 2.0)),
            region=region,
            beam_waist_panel_px=float(self._config.panel_span_px) / 2.0,
            far_field_size=region * padding,
            camera_px_per_model_px=float(padding),
            spot_fwhm_camera_px=float("nan"),
            spot_fwhm_model_px=float("nan"),
            correlation=1.0,
            method="sim-construction",
            calibration_notes=(
                "Geometry is known by construction for the simulated bench; no "
                "correlation solve was run."
            ),
        )


class _HardwareBench:
    """Santec SLM + CCD bench.

    Phase placement, slot rotation and settle discipline are all delegated to the
    shared probe kernels, so this class only owns the device lifecycle.
    """

    def __init__(self, config: SlmModelInLoopConfig) -> None:
        from ao_shaping.drivers.ccd.common import create_camera
        from ao_shaping.drivers.slm.santec import Santec
        from ao_shaping.tools.kernels.bench_kernels import capture_settled
        from ao_shaping.utils.slm.phase_disturbance import load_panel_disturbance

        self._capture = capture_settled
        self._config = config
        # The focus (defocus) compensation preflight is assembled **before any
        # device is opened**: a broken focus.json must cost nothing, and an
        # uncompensated bench is exactly the failure this stage removes.
        self._preflight = self._build_preflight(config)
        # create_camera forwards only the keys in the backend's accepted_kwargs,
        # and Daheng's allow-list does NOT include cam_size, so the window has to
        # be set explicitly afterwards.
        self._cam = create_camera(
            config.cam_type,
            cam_id=config.cam_id,
            exposure_time_ms=(
                config.exposure_time_ms or 0.02
                if config.cam_type == "daheng"
                else config.exposure_time_ms
            ),
        )
        self._slm = None
        self._original_slot = None
        self._original_gray = None
        self._ceiling_ms: float | None = None
        try:
            self._cam.open()
            if config.cam_type == "daheng":
                self._cam.reset_exposure_time(0.02)
            self._slm = Santec(
                slm_number=config.slm_number,
                wavelength=config.slm_wavelength or None,
            )
            self._slm.open()
            # Send-pipeline overlay (GUI 离焦 / 底相位): apply to the SLM this
            # run opens (driver slot; no-op for a sim SLM that lacks the
            # setters). Default ``None`` leaves CLI runs untouched.
            if config.send_overlay is not None:
                apply_overlays_to_handle(self._slm, config.send_overlay)
            from ao_shaping.drivers.slm.santec import SantecError

            try:
                self._original_slot = self._slm.get_displayed_memory_number()
            except SantecError as exc:
                if exc.code != 1:
                    raise
            if self._original_slot is None:
                self._original_gray = self._slm.get_current_grayscale()
            panel_shape = phase_to_panel(
                np.zeros((config.region, config.region), dtype=np.float64),
                int(round(config.panel_span_px / 2)),
                config.pupil_center,
            ).shape
            self._disturbance = load_panel_disturbance(
                config.phase_disturbance_file, panel_shape
            )
            self._blaze_choice = None
            self.signal_center_local = None
            if config.cam_type == "daheng":
                flat_panel = phase_to_panel(
                    np.zeros((config.region, config.region), dtype=np.float64),
                    int(round(config.panel_span_px / 2)),
                    config.pupil_center,
                )
                self._capture(
                    self._cam,
                    self._slm,
                    self._preflight.apply_to_panel_phase(
                        flat_panel
                        if self._disturbance is None
                        else flat_panel + self._disturbance
                    ),
                    n_frames=1,
                    n_discard=config.settle_max_discard,
                    wait_time_s=config.settle_wait_s,
                    stable_tol=config.settle_tol,
                    max_wait_s=config.settle_max_wait_s,
                )
                self._ceiling_ms = self._resolve_daheng_ceiling()
                initial_ms = config.exposure_time_ms or min(0.02, self._ceiling_ms)
                actual_ms = self._cam.reset_exposure_time(initial_ms)
                if not 0 < actual_ms <= self._ceiling_ms + 1e-9:
                    raise RuntimeError(
                        f"unsafe Daheng exposure readback: {actual_ms} ms "
                        f"(ceiling {self._ceiling_ms} ms)"
                    )
            if config.blaze_period_px >= 0:
                from ao_shaping.optimizer.wfless._blaze_calibration import select_blaze

                def capture_blaze(phase: np.ndarray) -> np.ndarray:
                    displayed = (
                        phase
                        if self._disturbance is None
                        else phase + self._disturbance
                    )
                    displayed = self._preflight.apply_to_panel_phase(displayed)
                    image = np.asarray(
                        self._capture(
                            self._cam,
                            self._slm,
                            displayed,
                            n_frames=1,
                            n_discard=config.settle_max_discard,
                            wait_time_s=config.settle_wait_s,
                            stable_tol=config.settle_tol,
                            max_wait_s=config.settle_max_wait_s,
                        ),
                        dtype=np.float64,
                    )
                    if config.cam_type == "daheng" and float(np.max(image)) >= 250:
                        raise RuntimeError("blaze CCD probe saturated; reduce exposure")
                    return image

                self._blaze_choice = select_blaze(
                    capture_blaze,
                    panel_shape,
                    target_side=config.target_side,
                    window_size=config.cam_size or config.target_side,
                    period_px=config.blaze_period_px,
                    direction=config.blaze_direction,
                )
                self.signal_center_local = self._blaze_choice.signal_xy
                logger.info("measured blaze: {}", self._blaze_choice)
            if config.cam_size > 0:
                from ao_shaping.utils.image.beam_metrics import zero_order_center

                blank_panel = phase_to_panel(
                    np.zeros((config.region, config.region), dtype=np.float64),
                    int(round(config.panel_span_px / 2)),
                    config.pupil_center,
                )
                full_frame = self._capture(
                    self._cam,
                    self._slm,
                    self._preflight.apply_to_panel_phase(
                        blank_panel
                        if self._disturbance is None
                        else blank_panel + self._disturbance
                    ),
                    n_frames=1,
                    n_discard=config.settle_max_discard,
                    wait_time_s=config.settle_wait_s,
                    stable_tol=config.settle_tol,
                    max_wait_s=config.settle_max_wait_s,
                )
                raw = self._cam.get_numpy_image(n_sample=1)
                if config.cam_type == "daheng":
                    decision = choose_exposure(
                        raw,
                        float(self._cam.exposure_time),
                        max_ms=self._ceiling_ms,
                    )
                    if decision.peak_fraction >= 0.9:
                        raise RuntimeError(
                            "full-frame CCD saturated at initial exposure; "
                            "reduce laser power before calibration"
                        )
                full_clean = prepare_frame(np.asarray(full_frame, dtype=np.float64))
                if float(np.max(full_clean)) < 10.0:
                    raise RuntimeError("full-frame CCD has no detectable zero order")
                full_x, full_y = (
                    self._blaze_choice.signal_xy
                    if self._blaze_choice is not None
                    else zero_order_center(full_clean, refine=False)
                )
                half = config.cam_size // 2
                if (
                    full_x - half < 0
                    or full_y - half < 0
                    or full_x + half >= full_clean.shape[1]
                    or full_y + half >= full_clean.shape[0]
                ):
                    raise RuntimeError("CCD zero order is too close to sensor edge")
                actual_size, actual_center = self._cam.reset_window(
                    center=(full_x, full_y),
                    size=(config.cam_size, config.cam_size),
                )
                if self._blaze_choice is not None:
                    self.signal_center_local = (
                        full_x - actual_center[0] + actual_size[0] // 2,
                        full_y - actual_center[1] + actual_size[1] // 2,
                    )
                logger.info(
                    "CCD ROI centered on full-frame zero order ({}, {})", full_x, full_y
                )
        except Exception:
            if self._slm is not None:
                self._restore_display()
                self._slm.close()
            self._cam.close()
            raise
        self._disc_radius = int(round(config.panel_span_px / 2.0))
        self._pupil_center = config.pupil_center
        self._pending: np.ndarray | None = None
        self.last_exposure_ms: float | None = None
        self.last_capture_utc: str | None = None
        self.last_raw_frame: np.ndarray | None = None
        self._closed = False

    def _build_preflight(self, config: SlmModelInLoopConfig) -> Any:
        """Assemble the focus (defocus) compensation preflight stage.

        The evidence file is written **either way** (``focus_preflight.json``
        in ``config.debug_dir``), so a later reader can tell "compensation was
        off" from "nothing was recorded".

        With the compensation requested but no usable calibration, the run
        aborts: collecting on an uncorrected bench is precisely the failure
        this stage exists to remove, and a silently uncompensated run would
        look like a compensated one in the report.
        """
        from ao_shaping.tools.slm.panel.focus_preflight import (
            FocusPreflightError,
            build_focus_preflight,
            write_preflight_record,
        )
        from ao_shaping.utils.slm.bench_constants import SLM_PANEL_H, SLM_PANEL_W

        try:
            preflight = build_focus_preflight(
                focus_dir=config.focus_dir,
                enabled=bool(config.focus_compensation),
                allow_live_probe=bool(config.focus_live_probe),
                live_probe_kwargs={
                    "slm_number": int(config.slm_number),
                    "cam_type": str(config.cam_type),
                    "cam_id": int(config.cam_id),
                    "wavelength_nm": int(config.slm_wavelength or 1064),
                },
                panel_shape=(SLM_PANEL_H, SLM_PANEL_W),
                pupil_center=tuple(config.pupil_center),
                run_pupil_center=tuple(config.pupil_center),
            )
        except FocusPreflightError as exc:
            raise RuntimeError(f"焦面补偿前置阶段失败: {exc}") from exc
        if config.debug_dir is not None:
            path = write_preflight_record(config.debug_dir, preflight)
            logger.info("focus preflight evidence -> {}", path)
        if preflight.enabled:
            calibration = preflight.calibration
            logger.info(
                "focus compensation ON: f={:.1f} mm, 1/f={} m^-1, centre={}",
                calibration.focal_mm if calibration else float("nan"),
                calibration.diopter_m1 if calibration else None,
                tuple(config.pupil_center),
            )
            for note in preflight.warnings:
                logger.warning("focus compensation: {}", note)
        else:
            logger.info("focus compensation OFF: {}", preflight.disable_reason)
            if preflight.requested:
                raise RuntimeError(
                    "焦面补偿被要求但没有生效: " + preflight.disable_reason
                )
        return preflight

    def _resolve_daheng_ceiling(self) -> float:
        """Resolve the session exposure ceiling: explicit > env > image probe.

        The ceiling is optical-path dependent (laser power, attenuation), so no
        constant is baked in.  When nothing pins it, the CCD image is probed
        with a doubling ladder bounded by the camera's exposure range; whatever
        the SLM happens to display at that moment only makes the estimate
        *brighter*, i.e. the ceiling lower — the safe direction.
        """
        from ao_shaping.drivers.ccd.common import get_camera_exposure_range

        min_ms, max_ms = get_camera_exposure_range(self._cam)
        return resolve_daheng_exposure_limit(
            self._config.max_exposure_ms,
            camera=self._cam,
            probe_start_ms=max(min_ms, 1e-3),
            probe_hard_max_ms=max_ms,
        )

    def display(self, phase_model: np.ndarray) -> None:
        # Raw unwrapped radians on the PANEL: the only mod-2pi wrap in the whole
        # repo lives inside the driver's radian -> grayscale conversion. Stashed
        # rather than written here, because capture_settled owns the write: it
        # is the single place that applies the memory-slot rotation and the
        # settle discipline.
        self._pending = phase_to_panel(
            np.asarray(phase_model, dtype=np.float64),
            self._disc_radius,
            self._pupil_center,
        )
        if self._disturbance is not None:
            self._pending += self._disturbance
        if self._blaze_choice is not None:
            from ao_shaping.utils.wavefront.zero_order_overlay import (
                apply_zero_order_overlay,
            )

            self._pending = apply_zero_order_overlay(
                self._pending, self._blaze_choice.params
            )
        # The focus (defocus) compensation is the LAST thing added: it is the
        # bench's own defocus being undone, so it sits on top of whatever
        # pupil phase (plus the fixed disturbance / blaze carrier) is going
        # out.  With the compensation off this returns the very same array.
        self._pending = self._preflight.apply_to_panel_phase(self._pending)

    def measure(self) -> np.ndarray:
        if self._pending is None:
            raise RuntimeError("measure() called before display()")
        for _ in range(5):
            frame = self._capture(
                self._cam,
                self._slm,
                self._pending,
                n_frames=self._config.n_eval_frames,
                n_discard=self._config.settle_max_discard,
                wait_time_s=self._config.settle_wait_s,
                stable_tol=self._config.settle_tol,
                max_wait_s=self._config.settle_max_wait_s,
            )
            if self._config.cam_type != "daheng":
                self.last_capture_utc = datetime.now(timezone.utc).isoformat(
                    timespec="microseconds"
                )
                return np.asarray(frame, dtype=np.float64)
            actual_ms = float(self._cam.exposure_time)
            self.last_exposure_ms = actual_ms
            if not 0 < actual_ms <= self._ceiling_ms + 1e-9:
                raise RuntimeError(
                    f"unsafe Daheng exposure readback: {actual_ms} ms "
                    f"(ceiling {self._ceiling_ms} ms)"
                )
            raw = self._cam.get_numpy_image(n_sample=1)
            self.last_raw_frame = np.asarray(raw).copy()
            decision = choose_exposure(raw, actual_ms, max_ms=self._ceiling_ms)
            if decision.accepted or (
                decision.peak_fraction < 0.9
                and abs(decision.next_ms - actual_ms) < 0.001
            ):
                logger.info(
                    "CCD exposure {:.4f} ms, peak {:.1%} of full scale",
                    actual_ms,
                    decision.peak_fraction,
                )
                self.last_capture_utc = datetime.now(timezone.utc).isoformat(
                    timespec="microseconds"
                )
                return exposure_normalized_frame(frame, actual_ms, reference_ms=0.02)
            if abs(decision.next_ms - actual_ms) < 0.001:
                break
            next_ms = self._cam.reset_exposure_time(decision.next_ms)
            if not 0 < next_ms <= self._ceiling_ms + 1e-9:
                raise RuntimeError(
                    f"unsafe Daheng exposure readback: {next_ms} ms "
                    f"(ceiling {self._ceiling_ms} ms)"
                )
        raise RuntimeError("CCD exposure could not reach an unsaturated stable range")

    def measure_fixed_exposure(self, exposure_ms: float) -> np.ndarray:
        """Capture a shaped-phase comparison at one bounded Daheng exposure."""
        if self._config.cam_type != "daheng":
            raise ValueError("fixed-exposure comparison requires Daheng CCD")
        if self._pending is None:
            raise RuntimeError("measure_fixed_exposure() called before display()")
        if not 0 < exposure_ms <= self._ceiling_ms + 1e-9:
            raise ValueError(
                f"fixed exposure {exposure_ms} ms exceeds the resolved Daheng "
                f"ceiling {self._ceiling_ms} ms"
            )
        actual_ms = float(self._cam.reset_exposure_time(exposure_ms))
        if not 0 < actual_ms <= self._ceiling_ms + 1e-9:
            raise RuntimeError(
                f"unsafe Daheng exposure readback: {actual_ms} ms "
                f"(ceiling {self._ceiling_ms} ms)"
            )
        frame = self._capture(
            self._cam,
            self._slm,
            self._pending,
            n_frames=self._config.n_eval_frames,
            n_discard=self._config.settle_max_discard,
            wait_time_s=self._config.settle_wait_s,
            stable_tol=self._config.settle_tol,
            max_wait_s=self._config.settle_max_wait_s,
        )
        raw = np.asarray(self._cam.get_numpy_image(n_sample=1))
        self.last_exposure_ms = actual_ms
        self.last_raw_frame = raw.copy()
        self.last_capture_utc = datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        )
        full_scale = float(np.iinfo(raw.dtype).max)
        if max(float(np.max(raw)), float(np.max(frame))) >= 0.9 * full_scale:
            raise RuntimeError("fixed-exposure CCD frame is saturated")
        return exposure_normalized_frame(frame, actual_ms, reference_ms=0.02)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._restore_display()
        for device in (self._slm, self._cam):
            try:
                device.close()
            except Exception as exc:  # teardown must not mask the run
                logger.warning("failed to close {}: {}", type(device).__name__, exc)

    def _restore_display(self) -> None:
        if self._slm is None:
            return
        try:
            if self._original_slot is None:
                if self._original_gray is not None:
                    self._slm.set_grayscale(self._original_gray)
            else:
                self._slm._display_memory(self._original_slot)
        except Exception as exc:  # still close both devices
            logger.warning("failed to restore initial SLM display: {}", exc)


def _open_bench(config: SlmModelInLoopConfig) -> Bench:
    """Open the backend named by ``config.cam_type``."""
    if config.cam_type == "sim":
        return _SimBench(config)
    return _HardwareBench(config)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _metrics_at(
    frame: np.ndarray,
    target_side: int,
    center: tuple[int, int],
    *,
    target_shape: str = "square",
    camera_rotation_quarters: int = 0,
) -> dict[str, float]:
    """Target-region metrics on ``frame`` about a **frozen** ``center``.

    The anchor is passed in rather than re-derived. On a speckle field the global
    maximum hops between near-equal grains under a ~1e-3 perturbation, so an
    ``argmax``-rolled box makes the objective discontinuous and the optimiser
    chases a box that no longer covers the beam.

    Args:
        frame: Preprocessed far-field frame.
        target_side: Target bounding width in the frame's own pixels.
        center: The frozen ``(x, y)`` optical axis.

    Returns:
        The metric dict from
        :func:`~ao_shaping.utils.image.beam_metrics.compute_square_metrics`
        (square path). The non-square path builds the same keys itself; its
        ``uniformity_cv`` / ``encircled_energy`` share the definition with the
        square path via
        :func:`~ao_shaping.utils.image.beam_metrics.masked_cv` /
        :func:`~ao_shaping.utils.image.beam_metrics.masked_energy_fraction`
        (issue #199 R3), keeping only its own mask geometry and degenerate
        policy.
    """
    from ao_shaping.utils.image.beam_metrics import (
        compute_shape_iou,
        compute_square_metrics,
        masked_cv,
        masked_energy_fraction,
    )

    if target_shape == "square":
        from ao_shaping.utils.image.targets import square_mask

        metrics = compute_square_metrics(
            frame, int(target_side), (float(center[0]), float(center[1]))
        )
        target = square_mask(
            np.asarray(frame).shape, center[1], center[0], int(target_side)
        )
        metrics["shape_iou"] = compute_shape_iou(frame, target)
        return metrics
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        create_target_mask,
    )

    image = np.asarray(frame, dtype=np.float64)
    target = create_target_mask(target_shape, image.shape, int(target_side))
    target = np.rot90(target, int(camera_rotation_quarters))
    target = np.roll(
        target,
        (int(center[1]) - image.shape[0] // 2, int(center[0]) - image.shape[1] // 2),
        axis=(0, 1),
    )
    support = target > 0.5
    values = image[support]
    mean = float(np.mean(values)) if values.size else 0.0
    return {
        "uniformity_cv": masked_cv(image, support),
        "encircled_energy": masked_energy_fraction(image, support, nonnegative=True),
        "shape_iou": compute_shape_iou(image, support),
        "intensity_max": float(image.max()),
        "intensity_mean": mean,
    }


def _quality(
    metrics: dict[str, float], w_efficiency: float = 0.6, w_uniformity: float = 0.4
) -> float:
    """Composite score to maximise, from target-region metrics.

    Uses the canonical :func:`composite_from_pib_cv`
    (re-exported via ``ao_shaping.drivers.sim.slm_shaping_bench`` for historical
    reasons, defined in
    :mod:`ao_shaping.algorithm.signal_processing.square_shaping`),
    i.e. `w_efficiency * EE + w_uniformity / (1 + CV)`, so the number here is directly
    comparable with the simulated pipeline and `slm-gs-refine`.

    History worth keeping, because it is why this function exists at all. It was written when
    `utils/image/beam_metrics.compute_quality_score` scored uniformity as
    `exp(-((CV/0.3)**2))`, which underflows to ~6e-5 at the CV ~0.9 this task reaches and to
    exactly 0 at the CV ~11 of an *unshaped* focus. Both scored 0 on uniformity, so that
    composite collapsed to `0.3*aspect + 0.3*EE` and ranked a near-delta focus with CV 11
    **above** a genuinely uniform square -- the bake-off would have discarded the correct
    phase and kept flat. `compute_quality_score` has since been **fixed** to the same
    non-saturating `1/(1+CV)` form, so the two now agree on the uniformity axis and the
    inversion is gone. This function is kept because its weights and its absence of an aspect
    term make it the better-conditioned bake-off objective -- not because the canonical one is
    still broken. See `report/loss_defects/inverse_design_report.md` section 10.
    """
    from ao_shaping.algorithm.signal_processing.square_shaping import (
        composite_from_pib_cv,
    )

    return float(
        composite_from_pib_cv(
            float(metrics.get("encircled_energy", 0.0)),
            float(metrics.get("uniformity_cv", float("inf"))),
            w_pib=float(w_efficiency),
            w_unif=float(w_uniformity),
        )
    )


def _to_model_grid(
    frame: np.ndarray,
    center: tuple[int, int],
    region: int,
    camera_px_per_model_px: float,
    camera_rotation_quarters: int = 0,
) -> np.ndarray:
    """Resample a camera frame onto the forward model's ``region`` grid.

    Step A's loss compares the model's prediction with the measurement, so both
    must live on the same grid. The camera oversamples the far field relative to
    the model, so this crops a ``region * scale`` window on the frozen optical
    axis and reduces it to ``region``.

    When the scale is an integer the reduction is a plain block average, which is
    the right operation: the extra camera pixels are a finer sampling of one
    model cell, not independent measurements, so they must be summed coherently
    rather than treated as extra samples. A non-integer scale (a real bench's
    geometry rarely lands on one) falls back to linear interpolation.

    Args:
        frame: Preprocessed far-field frame, camera pixels.
        center: The frozen ``(x, y)`` optical axis.
        region: Target model grid edge.
        camera_px_per_model_px: Camera pixels per model pixel.

    Returns:
        A ``(region, region)`` float64 array.
    """
    from scipy.ndimage import zoom as _zoom

    scale = float(camera_px_per_model_px)
    if scale <= 0.0:
        raise ValueError(f"camera_px_per_model_px must be > 0, got {scale}")
    span = int(round(region * scale))
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = cx - span // 2, cy - span // 2

    data = np.asarray(frame, dtype=np.float64)
    out = np.zeros((span, span), dtype=np.float64)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + span, data.shape[1]), min(y0 + span, data.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = data[sy0:sy1, sx0:sx1]

    if abs(scale - round(scale)) < 1e-9 and int(round(scale)) > 1:
        block = int(round(scale))
        usable = (span // block) * block
        reduced = (
            out[:usable, :usable]
            .reshape(usable // block, block, usable // block, block)
            .mean(axis=(1, 3))
        )
        if reduced.shape[0] == region:
            return np.ascontiguousarray(np.rot90(reduced, -camera_rotation_quarters))
    resized = _zoom(out, (region / span, region / span), order=1, mode="nearest")
    return np.ascontiguousarray(np.rot90(resized, -camera_rotation_quarters))


def _far_field_cell_scale(
    camera_px_per_cell: float, *, from_far_field: int, to_far_field: int
) -> float:
    """Rescale ``camera_px_per_model_px`` between zero-padding sizes.

    A far-field cell is ``lambda*f/(P*d_model)`` wide, so the camera pixels one
    cell spans grow as the zero-padding size ``P`` shrinks: the scale is
    inversely proportional to ``P``.  Single implementation of that rule --
    both the geometry rescale and the prior-run comparison go through it.
    """
    return float(camera_px_per_cell) * float(from_far_field) / float(to_far_field)


def _geometry_on_model_grid(
    geometry: BenchGeometry, *, region: int, far_field_size: int
) -> BenchGeometry:
    """Re-sample a measured geometry onto the model's far-field grid.

    ``camera_px_per_model_px`` is camera pixels per far-field *cell*, and a cell
    is ``lambda*f/(P*d_model)`` wide, so the scale is inversely proportional to
    the zero-padding size ``P`` (``far_field_size``).  The panel box, the beam
    waist and the spot FWHM are bench properties and do not move: a geometry
    measured at one padding factor is still usable at another -- only the
    far-field sampling changes -- so rejecting it would strand the GUI, whose
    padding default (8) differs from the calibration runbook's (3).

    Args:
        geometry: The geometry as measured (or as stored by a prior run).
        region: Model grid side the caller is about to use.
        far_field_size: Zero-padding size the caller is about to use.

    Returns:
        ``geometry`` unchanged when already on that grid, else a copy with
        ``far_field_size`` and ``camera_px_per_model_px`` rescaled.

    Raises:
        ValueError: If ``geometry.region`` differs -- the pupil grid itself is
            different, which no rescale can repair.
    """
    if int(geometry.region) != int(region):
        raise ValueError(
            f"geometry region {geometry.region} != model region {int(region)}"
        )
    if int(geometry.far_field_size) == int(far_field_size):
        return geometry
    scale = _far_field_cell_scale(
        geometry.camera_px_per_model_px,
        from_far_field=int(geometry.far_field_size),
        to_far_field=int(far_field_size),
    )
    logger.info(
        "rescaling measured geometry far field {} -> {} "
        "(camera_px_per_model_px {:.4f} -> {:.4f})",
        geometry.far_field_size,
        int(far_field_size),
        geometry.camera_px_per_model_px,
        scale,
    )
    return replace(
        geometry,
        far_field_size=int(far_field_size),
        camera_px_per_model_px=scale,
    )


def _calibrate_geometry(
    config: SlmModelInLoopConfig, bench: Bench
) -> tuple[Any, list[CalibrationRecordLike]]:
    """Solve the model-to-camera geometry, or take the sim's known answer.

    Hardware needs this because the illuminated pupil extent, the beam width and
    the far-field sampling are all fixed by the bench rather than by the model
    grid. It is scored by **Pearson correlation of speckle patterns**, not by
    peak-normalised MSE: with a peaked spot the MSE is dominated by tail energy
    and reports a small number for patterns that do not actually correspond.

    An explicit ``config.geometry_file`` (a runbook ``--stage sweep`` artefact)
    short-circuits the solver; a file measured at another far-field padding is
    rescaled onto this run's grid by :func:`_geometry_on_model_grid`.

    Returns:
        The :class:`~ao_shaping.optimizer.wfless.model_in_loop_shaping.BenchGeometry`
        and the records it consumed (empty for the simulated bench).
    """
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        CalibrationRecord,
        calibrate_bench_geometry,
        load_bench_geometry,
    )

    if isinstance(bench, _SimBench):
        return bench.true_geometry(), []
    if config.geometry_file is not None:
        geometry = load_bench_geometry(config.geometry_file)
        if not geometry.is_sweep:
            raise ValueError(
                f"geometry file {config.geometry_file!r} has method "
                f"{geometry.method!r}, not 'sweep'. Speckle geometries are the "
                "abandoned route this run's bake-off replaced; measure one with "
                "scripts/slm/model_in_loop_hw_runbook.py."
            )
        geometry = _geometry_on_model_grid(
            geometry,
            region=int(config.region),
            far_field_size=int(config.region) * int(config.far_field_padding),
        )
        logger.info(
            "using independently measured sweep geometry: {}", config.geometry_file
        )
        return geometry, []

    logger.warning(
        "no geometry file: solving the bench geometry from speckle probes. On "
        "this bench that route is a measured failure (correlation ~0.07 against "
        "the {:.2f} bake-off floor), so the run will most likely abort. Measure "
        "one first: python scripts/slm/model_in_loop_hw_runbook.py --stage all",
        config.min_geometry_correlation,
    )

    records: list[CalibrationRecordLike] = []
    for index in range(int(config.n_calibration_probes)):
        probe = _probe_phase(
            config.region, config.probe_spread, config.seed + 50_000 + index
        )
        bench.display(probe)
        raw_image = bench.measure()
        image = prepare_frame(raw_image)
        if config.step_sink is not None:
            config.step_sink(
                "calib-probe",
                _step_payload(
                    phase_rad=probe,
                    frame=image,
                    raw_scalars=_raw_frame_scalars(raw_image),
                    round_index=0,
                    probe_index=index,
                    probe_count=int(config.n_calibration_probes),
                ),
            )
        _save_preflight_capture(config, f"calibration_{index:02d}", probe, image, bench)
        records.append(
            CalibrationRecord(image=image, phase=probe, label=f"calib{index}")
        )
    logger.info("collected {} calibration probes for the geometry solve", len(records))
    geometry = calibrate_bench_geometry(
        records,
        wavelength=float(config.wavelength_nm) * 1e-9,
        focal_length=float(config.focal_length_m),
        slm_pitch=float(config.slm_pixel_um) * 1e-6,
        camera_pixel=float(config.camera_pixel_um) * 1e-6,
        region=int(config.region),
        far_field_size=int(config.region) * int(config.far_field_padding),
        panel_span_px=float(config.panel_span_px),
    )
    if config.debug_dir is not None:
        import json
        from dataclasses import asdict

        path = Path(config.debug_dir) / "preflight" / "geometry.json"
        path.write_text(json.dumps(asdict(geometry), indent=2), encoding="utf-8")
    return geometry, records


def _save_preflight_capture(
    config: SlmModelInLoopConfig,
    name: str,
    phase: np.ndarray,
    frame: np.ndarray,
    bench: Bench,
) -> None:
    """Persist a hardware capture before geometry acceptance can abort the run."""
    if config.debug_dir is None or config.cam_type == "sim":
        return
    import json

    directory = Path(config.debug_dir) / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "phase": np.asarray(phase, dtype=np.float32),
        "frame": np.asarray(frame, dtype=np.float32),
    }
    raw = getattr(bench, "last_raw_frame", None)
    if raw is not None:
        arrays["raw_frame"] = np.asarray(raw)
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    (directory / f"{name}.json").write_text(
        json.dumps(
            {
                "exposure_ms": getattr(bench, "last_exposure_ms", None),
                "timestamp_utc": getattr(bench, "last_capture_utc", None),
                "raw_frame_note": "separate CCD read after the settled frame",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _predicted_far_field(
    optimizer: Any,
    coefficients: np.ndarray,
    phase: np.ndarray,
    source_amplitude: np.ndarray,
) -> np.ndarray | None:
    """Forward-model far field for the live display, or ``None`` if unavailable.

    This is the "predicted CCD" panel: the same forward model Step A fits
    against, evaluated at ``(coefficients, phase)``. It reuses
    :meth:`ZernikeCoefficientOptimizer.forward_intensity` -- the single source of
    truth for the forward model -- rather than re-deriving a far field, so the
    picture on screen cannot disagree with the loss being optimised.

    Returns ``None`` instead of raising when the prediction cannot be produced.
    The live view is diagnostics: a run on hardware has already spent minutes of
    probe acquisitions, and aborting it because a *display* frame failed would
    be strictly worse than showing a blank panel. Only the two failure modes
    that are actually expected are absorbed -- the forward model's own
    shape/finiteness validation (``ValueError``) and a torch backend failure
    (``RuntimeError``) -- so a genuine bug still surfaces.

    Args:
        optimizer: The Step A :class:`ZernikeCoefficientOptimizer`.
        coefficients: Coefficient vector in radians.
        phase: Raw unwrapped-radian SLM phase, ``(region, region)``.
        source_amplitude: Calibrated illumination on the model grid.

    Returns:
        ``(far_field_size, far_field_size)`` float64 intensity, or ``None``.
    """
    try:
        predicted = optimizer.forward_intensity(coefficients, phase, source_amplitude)
    except (ValueError, RuntimeError) as exc:
        logger.warning("live-view prediction unavailable: {}", exc)
        return None
    out = np.asarray(predicted, dtype=np.float64)
    if out.ndim != 2:
        # ``Image2DFrame`` -> ``to_display_uint8`` raises on anything that is not
        # 2-D. Degrade here instead, so a future change to the forward model's
        # return shape blanks one panel rather than killing a hardware run that
        # has already spent minutes of probe acquisitions.
        logger.warning(
            "live-view prediction is {}-D, not a 2-D far field; panel left blank",
            out.ndim,
        )
        return None
    return out


def _seed_coefficients_from_checkpoint(
    config: SlmModelInLoopConfig,
) -> np.ndarray | None:
    """Build the Step-A coefficient seed from a ``train_amp`` checkpoint.

    ``None`` (``forward_checkpoint`` unset) means "keep the historical all-zero
    seed" and is returned untouched. Otherwise the stored coefficients are loaded,
    checked against this run's geometry, and embedded into the piston-*inclusive*
    vector the loop uses.

    The checkpoint vector is piston-*excluded* (``K = calc_n_zernike_terms(n_max) -
    1`` entries, index 0 = Noll 2 / tip); the loop's vector is piston-*inclusive*
    (``n_coeffs`` entries, index 0 = Noll 1 / piston), so the stored vector is
    placed at offset 1. The frozen modes are then re-zeroed, because the checker
    and the loader both trust the caller to honour them: the optimizer's
    ``__init__`` validates the seed's shape/finiteness but does **not** zero the
    frozen entries in it (it only builds the mask for the update path).

    Returns:
        The piston-inclusive seed, or ``None`` for the zero path.

    Raises:
        ValueError: If the checkpoint's geometry disagrees with this run, or its
            coefficient count cannot be placed into the piston-inclusive vector.
    """
    if config.forward_checkpoint is None:
        return None

    from ml.zernike.amp_checkpoint import check_geometry, load_trained_forward_model

    loaded = load_trained_forward_model(config.forward_checkpoint)
    report = check_geometry(
        loaded,
        region=int(config.region),
        n_orders=int(config.n_orders),
        far_field_size=int(config.region) * int(config.far_field_padding),
        assume_unverified_geometry=bool(config.assume_unverified_geometry),
    )
    if not report.ok:
        # Strict: a geometry mismatch aborts rather than being silently ignored.
        raise ValueError(
            "forward checkpoint "
            f"{config.forward_checkpoint!r} fails the geometry check: "
            + "; ".join(report.errors)
        )
    for note in report.notes:
        logger.info("checkpoint geometry: {}", note)

    n_coeffs = calc_n_zernike_terms(int(config.n_orders))
    if loaded.coefficients.shape[0] != n_coeffs - 1:
        raise ValueError(
            "forward checkpoint coefficient count "
            f"{loaded.coefficients.shape[0]} does not fit a piston-inclusive "
            f"vector of {n_coeffs} entries (expected {n_coeffs - 1})"
        )

    seed = np.zeros(n_coeffs, dtype=np.float64)
    seed[1:] = loaded.coefficients[: n_coeffs - 1]
    # Re-apply the frozen modes: the optimizer keeps them at zero in its update
    # path, but a stale non-zero seed entry would otherwise survive until then.
    for noll in config.frozen_modes:
        if 1 <= int(noll) <= n_coeffs:
            seed[int(noll) - 1] = 0.0

    logger.info(
        "Step A seeded from checkpoint {}: max |coefficient| = {:.6f} rad",
        config.forward_checkpoint,
        float(np.max(np.abs(seed))) if n_coeffs else 0.0,
    )
    return seed


def _make_optimizer(
    config: SlmModelInLoopConfig,
    coefficients: np.ndarray | None,
    source_amplitude: np.ndarray | None = None,
    *,
    affine: PupilAffine | None = None,
    travel: TravelScale | None = None,
) -> Any:
    """Build a :class:`ZernikeCoefficientOptimizer` around ``coefficients``.

    Used both for the first fit and to **re-arm** a tripped convergence plateau.
    Re-arming rebuilds around the current coefficients on purpose:
    :meth:`~...ZernikeCoefficientOptimizer.reset` would restore the values passed
    to ``__init__`` and throw away the whole fit.

    The self-calibration parameters (τ_θ and γ) are passed in explicitly: they
    describe the **bench**, not the fit, so they must survive the per-round
    rebuild of the optimizer. ``None`` means "that self-calibration is off".
    """
    from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
        ZernikeCoefficientOptimizer,
    )

    optimizer = ZernikeCoefficientOptimizer(
        n_orders=int(config.n_orders),
        region=int(config.region),
        initial_coefficients=coefficients,
        lr=float(config.step_a_lr),
        max_iterations=max(1, int(config.step_a_iterations)),
        device=config.device,
        seed=int(config.seed),
        dtype=str(config.dtype),
        far_field_size=int(config.region) * int(config.far_field_padding),
        frozen_modes=tuple(config.frozen_modes),
        pupil_oversampling=int(config.pupil_oversampling),
        error_terms=config.error_terms,
        fourier_weight=float(config.fourier_weight),
        loss_kind=str(config.loss_kind),
        affine_params=affine,
        affine_lr=float(config.selfcal_lr),
        travel_gammas=(None if travel is None else tuple(travel.gammas)),
        travel_lr=float(config.selfcal_lr),
    )
    if source_amplitude is not None:
        optimizer.set_source_amplitude(source_amplitude)
    return optimizer


def _prediction_correlation(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Pearson agreement on an independent phase, insensitive to CCD exposure."""
    a = np.asarray(predicted, dtype=np.float64).ravel().copy()
    b = np.asarray(measured, dtype=np.float64).ravel().copy()
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def _fit_flat_leakage(
    optimizer: Any,
    coefficients: np.ndarray,
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    source_amplitude: np.ndarray,
    flat_grid: np.ndarray,
) -> tuple[float, float]:
    """Fit one nonnegative model gain and one phase-independent flat leakage."""
    from scipy.optimize import nnls

    features = []
    measurements = []
    for measured, phase in frames:
        predicted = optimizer.forward_intensity(coefficients, phase, source_amplitude)
        if predicted.shape != measured.shape or predicted.shape != flat_grid.shape:
            raise ValueError("leakage calibration grids must have identical shapes")
        features.append(np.stack((predicted.ravel(), flat_grid.ravel()), axis=1))
        measurements.append(np.asarray(measured, dtype=np.float64).ravel())
    if not features:
        raise ValueError("leakage calibration needs a probe frame")
    weights, _ = nnls(np.concatenate(features), np.concatenate(measurements))
    return float(weights[0]), float(weights[1])


def _seed_coefficients_from_run(config: SlmModelInLoopConfig) -> np.ndarray | None:
    """Load an accepted run seed, refusing incompatible optical geometry."""
    if config.prior_run_dir is None:
        return None
    import ast
    import json

    from ao_shaping.optimizer.wfless.model_in_loop_shaping import load_bench_geometry

    run_dir = Path(config.prior_run_dir)
    if config.geometry_file is None:
        raise ValueError("prior_run_dir requires an explicit geometry_file")
    prior_geometry = load_bench_geometry(run_dir / "bench_geometry.json")
    current_geometry = load_bench_geometry(config.geometry_file)
    # The pupil grid itself cannot be rescaled: both geometries must sit on this
    # run's model grid.
    if int(prior_geometry.region) != int(config.region) or int(
        current_geometry.region
    ) != int(config.region):
        raise ValueError("prior run geometry mismatch: region")
    if not np.isclose(
        float(prior_geometry.panel_disc_radius),
        float(current_geometry.panel_disc_radius),
        rtol=1e-6,
    ):
        raise ValueError("prior run geometry mismatch: panel_disc_radius")
    # Compare the far-field scale on THIS run's grid: a prior run may have been
    # stored at another padding factor, which is a pure re-sampling difference.
    target_far_field = int(config.region) * int(config.far_field_padding)
    prior_scale = _far_field_cell_scale(
        prior_geometry.camera_px_per_model_px,
        from_far_field=int(prior_geometry.far_field_size),
        to_far_field=target_far_field,
    )
    current_scale = _far_field_cell_scale(
        current_geometry.camera_px_per_model_px,
        from_far_field=int(current_geometry.far_field_size),
        to_far_field=target_far_field,
    )
    if not np.isclose(prior_scale, current_scale, rtol=1e-6):
        raise ValueError("prior run geometry mismatch: camera_px_per_model_px")
    metadata_file = run_dir / "run_config.json"
    if metadata_file.is_file():
        previous = json.loads(metadata_file.read_text(encoding="utf-8"))
    else:
        # Older runs persisted the dataclass repr; parse only literal keyword values.
        result_files = sorted(run_dir.glob("slm_model_in_loop_*.json"))
        if len(result_files) != 1:
            raise ValueError("prior run lacks unambiguous configuration metadata")
        representation = json.loads(result_files[0].read_text(encoding="utf-8"))[
            "config"
        ]
        expression = ast.parse(representation, mode="eval").body
        if (
            not isinstance(expression, ast.Call)
            or not isinstance(expression.func, ast.Name)
            or expression.func.id != "SlmModelInLoopConfig"
        ):
            raise ValueError("prior run configuration is not a model-in-loop record")
        previous = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in expression.keywords
            if keyword.arg
            in {
                "cam_type",
                "cam_id",
                "slm_number",
                "slm_wavelength",
                "panel_span_px",
                "n_orders",
                "region",
                "far_field_padding",
                "pupil_center",
            }
        }
    for key in (
        "cam_type",
        "cam_id",
        "slm_number",
        "slm_wavelength",
        "panel_span_px",
        "n_orders",
        "region",
        "far_field_padding",
    ):
        if previous.get(key) != getattr(config, key):
            raise ValueError(f"prior run configuration mismatch: {key}")
    previous_center = previous.get("pupil_center")
    if previous_center is not None:
        previous_center = tuple(previous_center)
    if previous_center != config.pupil_center:
        logger.warning(
            "prior pupil centre {} -> current {}; remeasure holdouts before shaping",
            previous_center,
            config.pupil_center,
        )
    records = sorted(run_dir.glob("slm_model_in_loop_*.csv"))
    if len(records) != 1:
        raise ValueError("prior run must contain exactly one result CSV")
    import csv

    with records[0].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not any(
        row.get("stage") == "round" and row.get("accepted") == "True" for row in rows
    ):
        raise ValueError("prior run has no accepted shaped round")
    coefficients = np.asarray(
        np.load(run_dir / "fitted_coefficients.npy"), dtype=np.float64
    )
    expected = calc_n_zernike_terms(int(config.n_orders))
    if coefficients.shape != (expected,) or not np.isfinite(coefficients).all():
        raise ValueError(f"prior run coefficients must be {expected} finite values")
    coefficients = coefficients.copy()
    for mode in config.frozen_modes:
        coefficients[mode - 1] = 0.0
    return coefficients


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def optimize_slm_model_in_loop(config: SlmModelInLoopConfig) -> ModelInLoopResult:
    """Shape the far field into a target on real SLM + CCD hardware.

    Alternates two steps every round. **Step A** displays strong, varied random
    pupil probes and refits one shared Zernike aberration of the forward model
    against the measured frames, minimising loss(measured, predicted). **Step
    B** freezes that aberration and optimises a full-pixel SLM phase for a
    configured target, displays it, and measures the result. Each round's phase warm
    starts the next.

    Two guards keep the steps from chasing each other -- a trust region on the
    coefficient change and an acceptance test that rolls back a round whose fit
    or whose measured quality regressed. See the module docstring.

    Args:
        config: The resolved run configuration.

    Returns:
        A :class:`ModelInLoopResult` carrying the history recorder plus the
        adopted phase, the geometry, and the flat/best scores.

    Raises:
        RuntimeError: If the geometry bake-off fails or the rejection streak is
            exhausted -- both mean the bench is outside the model, and shaping
            anyway would produce a worse spot than flat.
    """
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        create_target_mask,
    )
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        StepBConfig,
        _fit_aberration_at_probes,
        _probe_set_loss,
        shape_phase_with_frozen_aberration,
    )
    from ao_shaping.utils.image.beam_metrics import zero_order_center
    from ao_shaping.utils.io.file import Recorder

    recorder = Recorder()
    recorder.iteration_traces = {}
    probe_records: list[dict[str, Any]] = []

    # The loop's coefficient vector is piston-inclusive: index 0 is Noll 1
    # (piston), so its length is the full term count, not the optimizer's
    # piston-excluded count.
    n_coeffs = calc_n_zernike_terms(int(config.n_orders))
    # Build -- and validate -- the checkpoint seed *before any device opens*. The
    # loader needs only the config and the file, so a geometry mismatch can and
    # must fail here: doing it later costs a calibration pass and a flat-baseline
    # measurement on hardware to discover a mistake that is knowable offline.
    # With no checkpoint the seed is None and the historical all-zero start is
    # used verbatim.
    seeded = _seed_coefficients_from_run(config)
    if seeded is None:
        seeded = _seed_coefficients_from_checkpoint(config)
    if seeded is None:
        coefficients = np.zeros(n_coeffs, dtype=np.float64)
    else:
        coefficients = seeded

    bench = _open_bench(config)
    # Declared before the ``try`` so the ``finally`` teardown can always reach it.
    # The window itself can only be opened further down, once the geometry solve
    # has produced the target box that the CCD overlay needs.
    live_display: Any = None
    try:
        # ---------------- Stage 0: flat baseline, frozen ROI ---------------
        flat_phase = np.zeros((config.region, config.region), dtype=np.float64)
        bench.display(flat_phase)
        raw_flat = bench.measure()
        flat_frame = prepare_frame(raw_flat)
        if config.step_sink is not None:
            config.step_sink(
                "flat",
                _step_payload(
                    phase_rad=flat_phase,
                    frame=flat_frame,
                    raw_scalars=_raw_frame_scalars(raw_flat),
                    round_index=0,
                    probe_index=None,
                    probe_count=None,
                ),
            )
        _save_preflight_capture(config, "flat", flat_phase, flat_frame, bench)
        flat_capture_utc = getattr(bench, "last_capture_utc", None)
        if config.cam_type != "sim":
            probe_records.append(
                {
                    "round": 0,
                    "split": "flat",
                    "index": 0,
                    "phase": flat_phase.copy(),
                    "raw_frame": getattr(bench, "last_raw_frame", None),
                    "frame": flat_frame.copy(),
                    "exposure_ms": getattr(bench, "last_exposure_ms", None),
                    "timestamp_utc": getattr(bench, "last_capture_utc", None),
                }
            )
        if not np.any(flat_frame > 0):
            raise RuntimeError(
                "the flat-phase frame is empty after background subtraction; "
                "check the exposure and that the SLM is actually retracting "
                "light onto the camera"
            )
        # Locate the optical axis ONCE. The 0-order is the frame's global
        # maximum, never the geometric centre -- on this bench it sits far
        # off-centre -- and it is frozen for the whole run from here on.
        roi_center = getattr(bench, "signal_center_local", None) or zero_order_center(
            flat_frame, refine=False
        )
        logger.info("frozen ROI centre (x, y) = {}", roi_center)

        # ---------------- Stage 1: geometry, then the bake-off -------------
        geometry, calib_records = _calibrate_geometry(config, bench)
        logger.info(
            "geometry [{}]: disc={} waist={:.1f}px far_field={} cam/model={:.4f} corr={:.4f}",
            geometry.method,
            geometry.panel_disc_radius,
            geometry.beam_waist_panel_px,
            geometry.far_field_size,
            geometry.camera_px_per_model_px,
            geometry.correlation,
        )
        if geometry.correlation < float(config.min_geometry_correlation):
            raise RuntimeError(
                f"bench geometry bake-off failed: speckle correlation "
                f"{geometry.correlation:.4f} < required "
                f"{config.min_geometry_correlation:.4f}. The forward model does "
                f"not describe this bench, so shaping would produce a WORSE spot "
                f"than flat. Check panel_span_px against the panel box the "
                f"probes covered, that the pupil phase actually varied across "
                f"the illuminated area, and for pupil-registration / panel-tilt "
                f"error. Fall back to slm-gs-refine."
            )

        # The correlation floor cannot see a defocus mismatch -- on this bench a
        # geometry fitted on an uncorrected pupil scored 0.831 and passed it, while
        # a usable one scored 0.848. So the geometry also has to vouch for itself.
        #
        # Hardware only: the simulated bench synthesises its geometry and never
        # measures a flat spot, so its ``spot_fwhm_camera_px`` is NaN by
        # construction and there is nothing to check. A non-finite width on real
        # hardware stays a failure, because there the quantity is measured.
        if config.cam_type == "sim":
            logger.debug("skipping geometry self-check on the simulated bench")
        else:
            cam_size_check = int(getattr(config, "cam_size", 0) or 0)
            if cam_size_check <= 0:
                cam_size_check = int(getattr(config, "target_side", 0) or 0)
                if cam_size_check <= 0 and "flat_frame" in locals():
                    cam_size_check = int(min(flat_frame.shape[0], flat_frame.shape[1]))
                if cam_size_check <= 0:
                    cam_size_check = 300
                logger.warning(
                    "cam_size missing/invalid in config ({}); falling back to {} for geometry self-check",
                    getattr(config, "cam_size", None),
                    cam_size_check,
                )
            self_check = geometry.self_check_failures(cam_size_check)
            if self_check:
                raise RuntimeError(
                    "bench geometry failed its self-check: "
                    + "; ".join(self_check)
                    + f". The correlation floor ({geometry.correlation:.4f}) does not "
                    "detect this, so it is reported separately. A calibration run that "
                    "cannot carry the session's phase correction measures an uncorrected "
                    "pupil and produces a focal scale that is wrong by a large factor. "
                    "Re-run the focus/coma probe and pass the correction into every stage."
                )

        # Model-space target: the forward model never sees camera pixels.
        scale = float(geometry.camera_px_per_model_px)
        # The forward model emits its far-field grid, which is the pupil grid
        # resampled by the zero-padding factor: padding refines the far-field
        # SAMPLING without changing its angular extent, so one far-field cell is
        # (region / far_field_size) model pixels. Step A therefore needs the
        # measurement on THAT grid, which is a different span from the target.
        # Hardware calibration measures camera pixels per padded FFT cell.
        # The simulated bench reports camera pixels per unpadded model cell.
        cell_scale = (
            scale * float(config.region) / float(geometry.far_field_size)
            if config.cam_type == "sim"
            else scale
        )
        camera_rotation_quarters = 0 if config.cam_type == "sim" else 1
        logger.info(
            "grid mapping: {:.4f} cam px per model px -> {:.4f} cam px per far-field "
            "cell (far_field_size={})",
            scale,
            cell_scale,
            geometry.far_field_size,
        )
        camera_px_per_target_cell = (
            cell_scale * float(geometry.far_field_size) / float(config.region)
        )
        target_side_model = max(
            3, int(round(config.target_side / camera_px_per_target_cell))
        )
        target_side_model = min(target_side_model, int(config.region) - 2)
        target_camera = max(
            3, int(round(target_side_model * camera_px_per_target_cell))
        )
        logger.info(
            "target side: {} camera px -> {} model px (scale {:.4f} cam/model px)",
            config.target_side,
            target_side_model,
            camera_px_per_target_cell,
        )
        target = create_target_mask(
            config.target_shape,
            (int(config.region), int(config.region)),
            target_side_model,
        )

        # Illumination on the model grid, from the CALIBRATED waist. Omitting it
        # would make Step B optimise for a beam that is not the real one.
        waist_model = max(
            1.0,
            float(geometry.beam_waist_panel_px)
            * float(config.region)
            / float(config.panel_span_px),
        )
        source_amplitude = gaussian_grid(int(config.region), waist_model)
        # --- self-calibration state ----------------------------------------
        # The affine τ_θ and the travel scale γ describe the **bench**, so they
        # are initialised once (at identity / zero) and carried across rounds.
        # ``None`` means the corresponding self-calibration is off.
        calibration_affine: PupilAffine | None = (
            PupilAffine() if config.selfcal_affine else None
        )
        calibration_travel: TravelScale | None = (
            TravelScale() if config.selfcal_travel else None
        )
        flat_grid: np.ndarray | None = None
        if config.cam_type != "sim":
            flat_grid = _to_model_grid(
                flat_frame,
                roi_center,
                int(geometry.far_field_size),
                cell_scale,
                camera_rotation_quarters,
            )
            reference_model = _make_optimizer(
                config,
                coefficients,
                source_amplitude,
                affine=calibration_affine,
                travel=calibration_travel,
            ).forward_intensity(coefficients, flat_phase, source_amplitude)
            measured_peak = float(np.max(flat_grid))
            model_peak = float(np.max(reference_model))
            if measured_peak <= 0 or model_peak <= 0:
                raise RuntimeError(
                    "cannot calibrate model intensity from the flat frame"
                )
            source_amplitude *= np.sqrt(measured_peak / model_peak)
            logger.info(
                "flat-frame intensity anchor: CCD {:.4g}, model {:.4g}, "
                "amplitude scale {:.4g}",
                measured_peak,
                model_peak,
                np.sqrt(measured_peak / model_peak),
            )

        flat_metrics = _metrics_at(
            flat_frame,
            target_camera,
            roi_center,
            target_shape=config.target_shape,
            camera_rotation_quarters=camera_rotation_quarters,
        )
        flat_score = _quality(flat_metrics, config.w_efficiency, config.w_uniformity)
        recorder.history.append(
            {
                "stage": "calibration",
                "round": 0,
                # `_epoch` is the Recorder's required index key: the shared writer
                # `utils.io.file.save_recorder_debug_artifacts` does
                # `data[int(rec["_epoch"])] = item`, so a record without it raises
                # KeyError from inside the pkl path. Every other shaping optimizer
                # stamps it (see `slm_zernike_pib.py`); this one did not, which is why
                # its `--debug` artefacts could never be written.
                RECORD_EPOCH: 0,
                "accepted": True,
                "reason": "baseline",
                "score_before": float(flat_score),
                "score_after": float(flat_score),
                "cv_after": float(flat_metrics.get("uniformity_cv", float("nan"))),
                "ee_after": float(flat_metrics.get("encircled_energy", float("nan"))),
                "uniformity_cv": float(flat_metrics.get("uniformity_cv", float("nan"))),
                "coeff_norm": 0.0,
                "coeff_step_l2": 0.0,
                "clamp_applied": False,
                "probe_count": 0,
                "step_a_lr": float(config.step_a_lr),
                "rejection_streak": 0,
                "geometry_correlation": float(geometry.correlation),
                "n_calibration_probes": len(calib_records),
            }
        )
        if config.cam_type != "sim":
            recorder.history[-1]["timestamp_utc"] = flat_capture_utc
        logger.info(
            "flat baseline: score={:.4f} CV={:.4f} EE={:.4f}",
            flat_score,
            flat_metrics.get("uniformity_cv", float("nan")),
            flat_metrics.get("encircled_energy", float("nan")),
        )

        # ---------------- Optional live view ---------------------------------
        # Opened here rather than at the top because the CCD overlay needs the
        # solved ``target_camera``. The flat baseline is rendered from the frames
        # already measured above, so opening later costs no extra acquisition.
        if config.show:
            from ao_shaping.display import SlmModelInLoopDisplay

            live_display = SlmModelInLoopDisplay(
                curve_title="score curve",
                target_shape=config.target_shape,
            )
            live_display.init_window()
            live_display.update(
                measured=flat_frame,
                phase=flat_phase,
                # No prediction yet: the model has no fitted aberration at this
                # point, so the panel stays blank rather than implying a fit.
                predicted=None,
                center=roi_center,
                r=target_camera / 2.0,
                info=f"flat baseline | score {flat_score:.4f}",
                value=float(flat_score),
                epoch=0,
                total_epochs=int(config.n_rounds),
                target_size=target_camera,
            )

        # ---------------- Stage 2: the alternating loop --------------------
        step_a_lr = float(config.step_a_lr)
        probe_count = int(config.probe_count)
        rejection_streak = 0
        prev_uncertainty_probes: list[np.ndarray] | None = None
        phase = np.zeros((config.region, config.region), dtype=np.float64)
        # None until the first shaped round is accepted: the flat baseline is not
        # a fair reference for the acceptance test (see acceptance_verdict).
        score_before: float | None = None
        best_accepted_cv: float | None = None
        best_score, best_phase, best_frame = flat_score, flat_phase.copy(), flat_frame
        status = ModelInLoopStatus.COMPLETED
        step_b_cfg = StepBConfig(
            iterations=int(config.step_b_iterations),
            lr=float(config.step_b_lr),
            target_shape=config.target_shape,
            target_side=int(target_side_model),
            w_uniformity=float(config.w_uniformity),
            w_efficiency=float(config.w_efficiency),
        )

        # Built before any device work so an unusable schedule fails at config
        # time. ``None`` keeps the fixed-radius default path untouched.
        trust_region: AdaptiveTrustRegion | None = None
        if config.adaptive_trust_region:
            base_radius = float(config.trust_region_c_l2)
            trust_region = AdaptiveTrustRegion(
                AdaptiveTrustRegionConfig(
                    initial_c_l2=max(base_radius, 1e-6),
                    min_c_l2=max(base_radius, 1e-6) * 0.25,
                    max_c_l2=max(base_radius, 1e-6) * 4.0,
                )
            )

        for index in range(int(config.n_rounds)):
            # --- Step A: probe the bench, refit one shared aberration -------
            optimizer = _make_optimizer(
                config,
                coefficients,
                source_amplitude,
                affine=calibration_affine,
                travel=calibration_travel,
            )
            step_a_lr = float(getattr(optimizer, "lr", step_a_lr)) or step_a_lr
            train_seeds, holdout_seeds = _resolve_probe_seeds(
                config.seed,
                index,
                probe_count,
                int(config.holdout_probe_count + config.freeform_holdout_count),
                config.probe_split_hook,
            )
            use_uncertainty = bool(
                config.uncertainty_probes
                or (config.uncertainty_on_rejection and rejection_streak > 0)
            )
            uncertainty_probes = (
                _select_uncertainty_probes(
                    optimizer,
                    config,
                    train_seeds,
                    prev_probes=prev_uncertainty_probes,
                )
                if use_uncertainty
                else None
            )
            frames: list[tuple[np.ndarray, np.ndarray]] = []
            for probe_index, probe_seed in enumerate(train_seeds):
                if uncertainty_probes is not None:
                    probe = uncertainty_probes[probe_index]
                else:
                    probe = _probe_phase(
                        config.region,
                        config.probe_spread,
                        probe_seed,
                        smooth=config.cam_type != "sim",
                    )
                bench.display(probe)
                raw_probe = bench.measure()
                measured = prepare_frame(raw_probe)
                if config.step_sink is not None:
                    config.step_sink(
                        "stepA-probe",
                        _step_payload(
                            phase_rad=probe,
                            frame=measured,
                            raw_scalars=_raw_frame_scalars(raw_probe),
                            round_index=index + 1,
                            probe_index=probe_index,
                            probe_count=probe_count,
                        ),
                    )
                if config.cam_type != "sim":
                    probe_records.append(
                        {
                            "round": index + 1,
                            "split": "train",
                            "index": probe_index,
                            "phase": probe.copy(),
                            "raw_frame": getattr(bench, "last_raw_frame", None),
                            "frame": measured.copy(),
                            "exposure_ms": getattr(bench, "last_exposure_ms", None),
                            "timestamp_utc": getattr(bench, "last_capture_utc", None),
                        }
                    )
                if live_display is not None and not live_display.closed:
                    # The most informative frame of the whole loop: the
                    # prediction here uses the coefficients as they stand
                    # *before* this probe is folded in, so watching the
                    # prediction panel track the measurement panel is watching
                    # Step A's fit improve. ``value`` is left at its default so
                    # the intra-round probes redraw the curve without adding
                    # points -- the curve is per *round*, not per probe.
                    live_display.update(
                        measured=measured,
                        phase=probe,
                        predicted=_predicted_far_field(
                            optimizer, coefficients, probe, source_amplitude
                        ),
                        center=roi_center,
                        r=target_camera / 2.0,
                        info=(
                            f"round {index + 1}/{config.n_rounds} "
                            f"step A probe {probe_index + 1}/{probe_count}"
                        ),
                        target_size=target_camera,
                    )
                # Step A's loss needs the measurement on the model's own grid.
                frames.append(
                    (
                        _to_model_grid(
                            measured,
                            roi_center,
                            int(geometry.far_field_size),
                            cell_scale,
                            camera_rotation_quarters,
                        ),
                        probe,
                    )
                )
            logger.info(
                "round {}/{}: fitting the forward model on {} probes",
                index + 1,
                config.n_rounds,
                probe_count,
            )
            loss_before = _probe_set_loss(optimizer, coefficients, frames)
            step_a_trace: list[dict[str, Any]] = []
            if config.step_a_iterations:
                fitted, _, step_a_iters, optimizer = _fit_aberration_at_probes(
                    optimizer,
                    frames,
                    int(config.step_a_iterations),
                    on_iteration=lambda iteration,
                    loss,
                    _trace=step_a_trace: _trace.append(
                        {
                            "iteration": iteration,
                            "loss": loss,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                                timespec="microseconds"
                            ),
                        }
                    ),
                )
            else:
                fitted, step_a_iters = coefficients.copy(), 0
            # --- Guard 0: trust region on the self-calibration --------------
            # The paper has NO trust region on its self-calibration quantities,
            # which is exactly the failure mode this repository refuses: a
            # calibration parameter that sprints away from its start means the
            # forward model is being bent to fit the data instead of
            # describing the bench. So the step is measured against the round's
            # starting values, and a violation rolls the parameters back and
            # rejects the round (see ``acceptance_verdict``).
            affine_after = optimizer.affine
            travel_after = optimizer.travel
            calibration_violated, theta_step, gamma_step = (
                calibration_trust_region_check(
                    affine_after,
                    calibration_affine,
                    travel_after,
                    calibration_travel,
                    theta_step_max=float(config.selfcal_theta_step_max),
                    gamma_step_max=float(config.selfcal_gamma_step_max),
                )
            )
            if calibration_violated:
                logger.warning(
                    "round {}: self-calibration trust region exceeded (theta step "
                    "{:.4g} > {:.4g}, gamma step {:.4g} > {:.4g}); rolling the "
                    "calibration back and rejecting the round",
                    index + 1,
                    theta_step,
                    float(config.selfcal_theta_step_max),
                    gamma_step,
                    float(config.selfcal_gamma_step_max),
                )
                optimizer.set_calibration(
                    affine=calibration_affine, travel=calibration_travel
                )
                affine_after = optimizer.affine
                travel_after = optimizer.travel
            if config.uncertainty_probes or config.uncertainty_on_rejection:
                prev_uncertainty_probes = [probe for _, probe in frames]
            if config.cam_type != "sim":
                recorder.iteration_traces[index + 1] = {"step_a_trace": step_a_trace}
            # --- Guard 1: trust region on the coefficient change -----------
            # The radius is fixed config.trust_region_c_l2 unless the adaptive
            # schedule is on; the clamp itself is unchanged either way.
            radius_now = (
                float(trust_region.radius)
                if trust_region is not None
                else float(config.trust_region_c_l2)
            )
            clamped, step_l2, clamp_applied = trust_region_clamp(
                fitted, coefficients, radius_now
            )
            loss_after = _probe_set_loss(optimizer, clamped, frames)
            if clamp_applied:
                logger.info(
                    "trust region clipped a {:.3f} rad coefficient step to {:.3f} rad",
                    step_l2,
                    radius_now,
                )

            model_gain, leakage = 1.0, 0.0
            if flat_grid is not None:
                model_gain, leakage = _fit_flat_leakage(
                    optimizer, clamped, frames, source_amplitude, flat_grid
                )
                logger.info(
                    "round {} forward calibration: gain {:.4f}, flat leakage {:.4f}",
                    index + 1,
                    model_gain,
                    leakage,
                )
            del frames

            holdout_correlation = float("nan")
            if config.cam_type != "sim" or config.holdout_on_sim:
                holdout_correlations: list[float] = []
                for holdout_index, holdout_seed in enumerate(holdout_seeds):
                    holdout_phase, holdout_family = _holdout_phase(
                        config.region,
                        config.probe_spread,
                        holdout_seed,
                        holdout_index,
                        int(config.holdout_probe_count),
                    )
                    bench.display(holdout_phase)
                    holdout_frame = prepare_frame(bench.measure())
                    holdout_grid = _to_model_grid(
                        holdout_frame,
                        roi_center,
                        int(geometry.far_field_size),
                        cell_scale,
                        camera_rotation_quarters,
                    )
                    prediction = optimizer.forward_intensity(
                        clamped, holdout_phase, source_amplitude
                    )
                    if flat_grid is not None:
                        prediction = model_gain * prediction + leakage * flat_grid
                    holdout_correlations.append(
                        _prediction_correlation(prediction, holdout_grid)
                    )
                    probe_records.append(
                        {
                            "round": index + 1,
                            "split": "holdout",
                            "index": holdout_index,
                            "family": holdout_family,
                            "phase": holdout_phase.copy(),
                            "raw_frame": getattr(bench, "last_raw_frame", None),
                            "frame": holdout_frame.copy(),
                            "model_grid": holdout_grid.copy(),
                            "prediction": np.asarray(prediction).copy(),
                            "correlation": holdout_correlations[-1],
                            "exposure_ms": getattr(bench, "last_exposure_ms", None),
                            "timestamp_utc": getattr(bench, "last_capture_utc", None),
                        }
                    )
                holdout_correlation = float(np.mean(holdout_correlations))
                logger.info(
                    "round {} independent phase correlation: {:.4f} (each {})",
                    index + 1,
                    holdout_correlation,
                    holdout_correlations,
                )
            if config.cam_type != "sim" and (
                not np.isfinite(holdout_correlation)
                or not all(
                    np.isfinite(value) and value >= config.min_holdout_correlation
                    for value in holdout_correlations
                )
            ):
                recorder.history.append(
                    {
                        "stage": "round",
                        "round": index + 1,
                        RECORD_EPOCH: index + 1,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(
                            timespec="microseconds"
                        ),
                        "accepted": False,
                        "reason": "holdout_forward_mismatch",
                        "holdout_correlation": holdout_correlation,
                        "probe_count": probe_count,
                    }
                )
                status = ModelInLoopStatus.ABORTED_UNIDENTIFIABLE
                break

            # --- Step B: synthesise the square with that aberration frozen --
            warm = phase if config.warm_start else flat_phase
            step_b_trace: list[dict[str, Any]] = []
            shaped, shaping_loss = shape_phase_with_frozen_aberration(
                optimizer,
                clamped,
                target,
                np.array(warm, dtype=np.float64, copy=True),
                step_b_cfg,
                dtype=str(config.dtype),
                device=config.device,
                seed=int(config.seed) + index,
                source_amplitude=source_amplitude,
                model_gain=model_gain,
                background_intensity=(
                    leakage * flat_grid if flat_grid is not None else None
                ),
                on_iteration=lambda iteration, loss, _trace=step_b_trace: _trace.append(
                    {
                        "iteration": iteration,
                        "loss": loss,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(
                            timespec="microseconds"
                        ),
                    }
                ),
            )
            if config.cam_type != "sim":
                recorder.iteration_traces[index + 1]["step_b_trace"] = step_b_trace
            shaped_phase = np.asarray(shaped, dtype=np.float64)
            bench.display(shaped_phase)
            raw_shaped = bench.measure()
            shaped_frame = prepare_frame(raw_shaped)
            if config.step_sink is not None:
                config.step_sink(
                    "stepB-shaped",
                    _step_payload(
                        phase_rad=shaped_phase,
                        frame=shaped_frame,
                        raw_scalars=_raw_frame_scalars(raw_shaped),
                        round_index=index + 1,
                        probe_index=None,
                        probe_count=None,
                    ),
                )
            shaped_prediction_correlation = float("nan")
            if config.cam_type != "sim":
                shaped_grid = _to_model_grid(
                    shaped_frame,
                    roi_center,
                    int(geometry.far_field_size),
                    cell_scale,
                    camera_rotation_quarters,
                )
                shaped_prediction = optimizer.forward_intensity(
                    clamped, shaped, source_amplitude
                )
                if flat_grid is not None:
                    shaped_prediction = (
                        model_gain * shaped_prediction + leakage * flat_grid
                    )
                shaped_prediction_correlation = _prediction_correlation(
                    shaped_prediction, shaped_grid
                )
                probe_records.append(
                    {
                        "round": index + 1,
                        "split": "shaped",
                        "index": 0,
                        "phase": shaped.copy(),
                        "raw_frame": getattr(bench, "last_raw_frame", None),
                        "frame": shaped_frame.copy(),
                        "model_grid": shaped_grid.copy(),
                        "prediction": np.asarray(shaped_prediction).copy(),
                        "correlation": shaped_prediction_correlation,
                        "exposure_ms": getattr(bench, "last_exposure_ms", None),
                        "timestamp_utc": getattr(bench, "last_capture_utc", None),
                    }
                )
                logger.info(
                    "round {} shaped-phase forward correlation: {:.4f}",
                    index + 1,
                    shaped_prediction_correlation,
                )
            shaped_metrics = _metrics_at(
                shaped_frame,
                target_camera,
                roi_center,
                target_shape=config.target_shape,
                camera_rotation_quarters=camera_rotation_quarters,
            )
            score_after = _quality(
                shaped_metrics, config.w_efficiency, config.w_uniformity
            )

            # --- Guard 2: accept or roll the round back ----------------------
            verdict = acceptance_verdict(
                loss_before,
                loss_after,
                score_before,
                score_after,
                loss_delta=float(config.acceptance_loss_delta),
                score_eps=float(config.acceptance_score_eps),
                cv_before=best_accepted_cv,
                cv_after=float(shaped_metrics["uniformity_cv"]),
                max_cv_regression=float(config.max_cv_regression),
                calibration_violated=calibration_violated,
            )
            applied_l2 = float(np.linalg.norm(clamped - coefficients))
            if trust_region is not None:
                # Feed the verdict back, then record both radii: the one this
                # round actually clamped with, and the schedule state after the
                # update. Reporting only the post-update radius would misdescribe
                # the step the round committed.
                trust_region.update(accepted=bool(verdict.accepted))
                row_trust_state = trust_region.state()
                row_trust_state["trust_region_c_l2"] = radius_now
                row_trust_state["trust_region_next_c_l2"] = float(trust_region.radius)
            else:
                row_trust_state = {}
            row = {
                "stage": "round",
                "round": index + 1,
                # See the calibration record above: required by the shared pkl writer.
                RECORD_EPOCH: index + 1,
                "accepted": bool(verdict.accepted),
                "reason": verdict.reason,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "step_a_iterations": int(step_a_iters),
                "step_b_iterations": len(shaping_loss),
                "coeff_step_l2": float(step_l2),
                "coeff_step_applied": applied_l2,
                "clamp_applied": bool(clamp_applied),
                "coeff_norm": float(np.linalg.norm(clamped)),
                "score_before": (
                    float(score_before) if score_before is not None else float("nan")
                ),
                "score_after": float(score_after),
                "cv_after": float(shaped_metrics.get("uniformity_cv", float("nan"))),
                "ee_after": float(shaped_metrics.get("encircled_energy", float("nan"))),
                "shape_iou_after": float(shaped_metrics["shape_iou"]),
                "uniformity_cv": float(
                    shaped_metrics.get("uniformity_cv", float("nan"))
                ),
                "probe_count": int(probe_count),
                "step_a_lr": float(step_a_lr),
                "rejection_streak": int(rejection_streak),
                "shape_loss_initial": float(shaping_loss[0])
                if shaping_loss
                else float("nan"),
                "shape_loss_final": float(shaping_loss[-1])
                if shaping_loss
                else float("nan"),
            }
            if config.cam_type != "sim" or config.holdout_on_sim:
                # Recorded whenever the held-out probes were actually measured,
                # which on sim is only under the explicit opt-in. The remaining
                # fields stay hardware-only: they come from measurements the sim
                # path does not take.
                row["holdout_correlation"] = holdout_correlation
            if affine_after is not None or travel_after is not None:
                # Opt-in only: the default history keeps exactly the keys it
                # had before the self-calibration existed. The per-round step
                # sizes are recorded so a reader can see whether the calibration
                # parameters are running away.
                row["selfcal_affine"] = (
                    {} if affine_after is None else affine_after.to_dict()
                )
                row["selfcal_travel"] = (
                    {} if travel_after is None else travel_after.to_dict()
                )
                row["selfcal_theta_step"] = float(theta_step)
                row["selfcal_gamma_step"] = float(gamma_step)
                row["selfcal_violated"] = bool(calibration_violated)
            if row_trust_state:
                # Opt-in only, so the default history keeps exactly the keys it
                # had before the adaptive schedule existed.
                row.update(row_trust_state)
            if config.cam_type != "sim":
                row["shaped_prediction_correlation"] = shaped_prediction_correlation
                row["model_gain"] = model_gain
                row["flat_leakage"] = leakage
                row["exposure_ms"] = getattr(bench, "last_exposure_ms", None)
            recorder.history.append(row)

            if live_display is not None and not live_display.closed:
                # Prediction under the *clamped* coefficients -- the ones the
                # acceptance test actually judged -- not the raw fit, so the
                # picture matches the verdict in ``reason``.
                live_display.update(
                    measured=shaped_frame,
                    phase=shaped_phase,
                    predicted=_predicted_far_field(
                        optimizer, clamped, shaped_phase, source_amplitude
                    ),
                    center=roi_center,
                    r=target_camera / 2.0,
                    info=(
                        f"round {index + 1}/{config.n_rounds} "
                        f"{'accepted' if verdict.accepted else 'rejected'} "
                        f"({verdict.reason}) | score {score_after:.4f} | "
                        f"loss {loss_before:.5f}->{loss_after:.5f}"
                    ),
                    value=float(score_after),
                    epoch=index + 1,
                    total_epochs=int(config.n_rounds),
                    target_size=target_camera,
                )

            if verdict.accepted:
                coefficients = clamped
                # The self-calibration is part of the model, so it is committed
                # with the round: a rejected round keeps the previous values,
                # exactly like the coefficients.
                if affine_after is not None:
                    calibration_affine = affine_after
                if travel_after is not None:
                    calibration_travel = travel_after
                # `.copy()` is load-bearing: `phase` is the warm start for the next
                # round's Step B, and `np.asarray` on an already-float64 array is a
                # no-op, so without it the committed phase would alias the one the
                # live view and the step sink were handed.
                phase = shaped_phase.copy()
                score_before = score_after
                candidate_cv = float(shaped_metrics["uniformity_cv"])
                best_accepted_cv = (
                    candidate_cv
                    if best_accepted_cv is None
                    else min(best_accepted_cv, candidate_cv)
                )
                rejection_streak = 0
                if score_after > best_score:
                    best_score, best_phase = score_after, phase.copy()
                    best_frame = shaped_frame
                logger.info(
                    "round {}/{} accepted: score {} -> {:.4f} (CV {:.4f}, EE {:.4f})",
                    index + 1,
                    config.n_rounds,
                    "n/a" if score_before is None else f"{score_before:.4f}",
                    score_after,
                    row["cv_after"],
                    row["ee_after"],
                )
            else:
                rejection_streak += 1
                logger.warning(
                    "round {}/{} REJECTED ({}): loss {:.5f} -> {:.5f}, score "
                    "{:.4f} -> {:.4f}; streak {}",
                    index + 1,
                    config.n_rounds,
                    verdict.reason,
                    loss_before,
                    loss_after,
                    row["score_before"],
                    score_after,
                    rejection_streak,
                )
                if rejection_streak >= int(config.max_rejection_streak):
                    status = ModelInLoopStatus.ABORTED_REJECTION_STREAK
                    logger.error(
                        "{} consecutive rounds rejected; the bench is outside the "
                        "forward model. Aborting WITHOUT committing the phase. "
                        "Fall back to slm-gs-refine.",
                        rejection_streak,
                    )
                    break
                # Damp the fit and widen its information base rather than pushing
                # harder on a model that is already fighting us.
                step_a_lr = max(step_a_lr * float(config.damp_lr_factor), 1e-4)
                probe_count = min(
                    probe_count + int(config.escalate_probe_count),
                    int(config.max_probe_count),
                )

            if (
                config.early_stop_score > 0
                and score_before is not None
                and score_before >= config.early_stop_score
            ):
                logger.info(
                    "early stop: score {:.4f} reached the target {:.4f}",
                    score_before,
                    config.early_stop_score,
                )
                break
            if live_display is not None and live_display.closed:
                # Stop searching, but fall through to the final bake-off so the
                # best already-verified phase is still committed -- closing a
                # window is a request to stop looking, not to discard the work.
                logger.info(
                    "live display closed; stopping after round {}/{}",
                    index + 1,
                    config.n_rounds,
                )
                break
            if config.callback is not None:
                config.callback(index, row)

        # ---------------- Stage 3: final bake-off against flat --------------
        if best_score <= flat_score:
            logger.warning(
                "no shaped phase beat the flat baseline (best {:.4f} <= flat "
                "{:.4f}); keeping flat",
                best_score,
                flat_score,
            )
            best_phase, best_frame, best_score = (
                flat_phase.copy(),
                flat_frame,
                flat_score,
            )
            if status == ModelInLoopStatus.COMPLETED:
                status = ModelInLoopStatus.BAKE_OFF_REJECTED
        else:
            logger.info(
                "final bake-off: shaped {:.4f} beats flat {:.4f}",
                best_score,
                flat_score,
            )
        logger.info("model-in-the-loop finished with status={}", status)
        return ModelInLoopResult(
            recorder=recorder,
            status=status,
            best_phase=best_phase,
            best_frame=best_frame,
            geometry=geometry,
            target_side_camera=target_camera,
            target_side_model=target_side_model,
            flat_score=float(flat_score),
            best_score=float(best_score),
            config=config,
            probe_records=probe_records,
            fitted_coefficients=coefficients.copy(),
            source_amplitude=source_amplitude.copy(),
            blaze_params=(
                getattr(bench, "_blaze_choice", None).params
                if getattr(bench, "_blaze_choice", None) is not None
                else None
            ),
            blaze_choice=getattr(bench, "_blaze_choice", None),
        )
    finally:
        if live_display is not None:
            live_display.close()
        bench.close()
