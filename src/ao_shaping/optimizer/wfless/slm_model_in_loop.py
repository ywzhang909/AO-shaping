"""Model-in-the-loop square shaping on real SLM + CCD hardware.

This is the hardware port of :func:`simulate_iterative_shaping` in
:mod:`ao_shaping.optimizer.wfless.model_in_loop_shaping`, which runs the same
alternating loop against the digital twin. Each round does two things:

* **Step A -- wavefront fit (model-in-the-loop).** A set of strong, *varied*
  random pupil probes is displayed on the SLM and the resulting far-field
  frames are measured. A single :class:`ZernikeCoefficientOptimizer` cycles
  through every probe so all of them inform one shared Zernike aberration
  vector, fitted by minimising loss(measured CCD, predicted far field).
* **Step B -- square synthesis.** The fitted aberration is **frozen** and a
  full-pixel SLM phase is optimised against a square target. That phase is
  displayed, measured, and becomes the warm start for the next round.

The module is a thin orchestration layer. No Zernike math, no forward model, no
metric and no loss is reimplemented here: everything is delegated to the
canonical helpers so the hardware path cannot drift from the simulated one.

Call graph
----------
Everything below is a real symbol in a real module; the layer that owns each
one is in brackets. Indented lines are the two steps of one round.

::

    main.py `slm-model-in-loop`                         (click hub)
     └─ runners/slm/model_in_loop_runner.py :: run        [hardware orchestration]
         ├─ _parse_frozen_modes / _parse_point           CLI text -> typed fields
         ├─ SlmModelInLoopConfig                         flat dataclass, runner_common
         ├─ optimize_slm_model_in_loop(config)           ← the only public entry
         │   │
         │   ├─ _open_bench(config)                      ──► Bench (Protocol)
         │   │     ├─ _SimBench                           cam_type=sim (digital twin)
         │   │     └─ _HardwareBench                      daheng/miicam + Santec
         │   │        display(phase) / measure() / close()   device imports deferred
         │   │
         │   ├─ _calibrate_geometry(bench, config)
         │   │     └─ calibrate_bench_geometry(...)       [twin] sim -> true_geometry()
         │   │
         │   ├─ flat baseline: display(flat) -> _metrics_at -> _quality
         │   │
         │   └─ for index in range(n_rounds):
         │       │
         │       │  -- Step A: refit ONE shared aberration --------------------------
         │       ├─ coefficients = np.zeros(n_coeffs)      ← the ONLY seed, hardcoded
         │       ├─ _make_optimizer(config, coefficients)
         │       │     └─ ZernikeCoefficientOptimizer(initial_coefficients=...)
         │       │          zernike_coefficient_optimizer.py  Adam over the Zernike vector
         │       ├─ for probe_index in range(probe_count):
         │       │     ├─ _probe_phase(region, probe_spread, seed)      [twin]
         │       │     ├─ bench.display(probe) -> _prepare_frame(bench.measure())
         │       │     │                      _prepare_frame  <- slm_gs_refine
         │       │     └─ _to_model_grid(measured, roi_center, far_field_size, ...)
         │       ├─ _fit_aberration_at_probes(optimizer, frames, iters, rearm)  [twin]
         │       │     └─ cycles every probe through ONE Adam state, then re-arms
         │       └─ trust_region_clamp(...)      caps |c_{t+1} - c_t|
         │       │
         │       │  -- Step B: synthesise the square, aberration frozen ---------------
         │       ├─ shape_phase_with_frozen_aberration(...)          [twin]
         │       ├─ display(shaped) -> _prepare_frame -> _metrics_at -> _quality
         │       ├─ phase = shaped                     becomes next round's warm start
         │       └─ acceptance_verdict(before, after)  -> accept | reject
         │            reject: damp lr, add probes, count the streak
         │            3 straight rejects -> ModelInLoopStatus.ABORTED_REJECTION_STREAK
         │
         └─ save_recorder_debug_artifacts(recorder, ...)      [utils/io/file]
              data/debug/slm_model_in_loop_<ts>/<ts>/*.pkl   written EVERY run

The `[twin]` rows are the point of this layout.
:mod:`model_in_loop_shaping` runs the *same* alternating loop against the
digital twin (:func:`simulate_iterative_shaping`), and this module imports its
math rather than restating it -- every one of those imports is deferred into the
function that uses it, because ``ao_shaping.drivers`` touches hardware at import
time. So the simulation is not a parallel implementation that can rot; it is the
shared library, and ``--cam_type sim`` swaps only the ``Bench`` implementation.

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

Nothing in this module imports a hardware driver at import time; every device
import is deferred into the functions that open it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from loguru import logger

#: Reused rather than reimplemented: median-subtract THEN clip. Clipping the raw
#: frame would rectify symmetric read noise into a positive pedestal
#: proportional to the pixel count (documented in ``drivers/sim/AGENTS.md``),
#: which corrupts every ratio metric computed afterwards.
from ao_shaping.optimizer.wfless.slm_gs_refine import _prepare_frame
from ao_shaping.tools.slm.bench_kernels import (
    BEAM_CENTER_PANEL,
    SLM_PITCH_M,
    crop_around_zero_order,
    gaussian_grid,
    phase_to_panel,
)

__all__ = [
    "Bench",
    "ModelInLoopStatus",
    "RoundRecord",
    "SlmModelInLoopConfig",
    "acceptance_verdict",
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
            ``"loss_worsened"``, ``"score_degraded"`` or both joined by ``"+"``).
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
) -> AcceptanceVerdict:
    """Decide whether a round may be committed, or must be rolled back.

    A round is rejected when the forward-model fit got *worse* (the aberration
    is wandering, not converging) or when the shape it produced measures worse
    than the best shape already held (the synthesis is being led astray).

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

    Returns:
        The :class:`AcceptanceVerdict`.
    """
    if not np.isfinite(loss_before) or not np.isfinite(loss_after) or not np.isfinite(score_after):
        # A non-finite residual means the fit or the measurement diverged. That
        # is not "no evidence of harm", it is absence of evidence: refuse it.
        return AcceptanceVerdict(False, True, True, "non_finite")
    loss_worsened = bool(loss_after > loss_before + float(loss_delta))
    score_degraded = bool(
        score_before is not None
        and np.isfinite(score_before)
        and score_after < score_before - float(score_eps)
    )
    tags = [
        name
        for name, hit in (("loss_worsened", loss_worsened), ("score_degraded", score_degraded))
        if hit
    ]
    return AcceptanceVerdict(
        accepted=not tags, loss_worsened=loss_worsened, score_degraded=score_degraded,
        reason="+".join(tags) if tags else "accepted",
    )


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

    dtype: str = "float64"
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
    target_side: int = 60
    """Target square side in **camera** pixels.

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

    # --- Step B: square synthesis with the aberration frozen ----------------
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

    focal_length_m: float = 0.0
    """2f lens focal length in metres; 0 derives it from the measured scale.

    The measured tilt-shift scale fixes only the ratio ``f / p_cam``, so one
    external anchor is unavoidable; the camera pixel pitch is used for it
    because that is a datasheet constant rather than a bench assembly choice.
    """

    n_calibration_probes: int = 8
    min_geometry_correlation: float = 0.5
    """Abort before shaping if the calibrated geometry scores below this.

    The geometry is a *bake-off*: it comes from a model of the bench, so if it is
    wrong the shaped spot gets worse rather than better. Refusing to shape on an
    uncalibrated model costs one calibration pass instead of the run.
    """

    # --- coupling guards ----------------------------------------------------
    trust_region_c_l2: float = 0.5
    """Max L2 norm of the per-round coefficient change (0 disables the guard)."""

    acceptance_loss_delta: float = 1e-4
    acceptance_score_eps: float = 1e-3
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
    """0 keeps the device default rather than pinning a value."""

    cam_size: int = 300
    slm_number: int = 1
    slm_wavelength: int = 1064
    """SLM operating wavelength in nm.

    The lab has more than one SLM, and a wrong wavelength reprograms the panel's
    phase table, so this is stated rather than assumed.
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
    callback: Callable[[int, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    progress_every: int = 1

    def __post_init__(self) -> None:
        """Validate the budget and normalise the frozen-mode list."""
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {self.n_rounds}")
        if self.probe_count < 2:
            # One probe is not a degenerate case, it is a broken configuration:
            # the non-convex fit has no phase diversity to escape local minima.
            raise ValueError(
                f"probe_count must be >= 2 (a single probe cannot identify the "
                f"aberration), got {self.probe_count}"
            )
        if self.probe_spread < 3.0:
            raise ValueError(
                f"probe_spread must be >= 3.0 rad for the fit to be identifiable "
                f"(a near-flat pupil's focus barely responds to a smooth "
                f"low-order aberration), got {self.probe_spread}"
            )
        if self.step_a_iterations < 1 or self.step_b_iterations < 1:
            raise ValueError("step_a_iterations and step_b_iterations must be >= 1")
        if not 0.0 < self.damp_lr_factor < 1.0:
            raise ValueError(
                f"damp_lr_factor must lie in (0, 1), got {self.damp_lr_factor}"
            )
        if self.target_side < 1:
            raise ValueError(f"target_side must be >= 1, got {self.target_side}")
        if self.region < 2:
            raise ValueError(f"region must be >= 2, got {self.region}")
        if self.n_orders < 1:
            raise ValueError(f"n_orders must be >= 1, got {self.n_orders}")
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )
        if self.panel_span_px <= 0:
            raise ValueError(f"panel_span_px must be > 0, got {self.panel_span_px}")
        self.frozen_modes = tuple(sorted({int(m) for m in self.frozen_modes}))
        # Resolve the focal length once, here, so the runner, a library caller and
        # the tests can never disagree about what geometry a run actually used.
        if self.focal_length_m <= 0.0:
            from ao_shaping.optimizer.wfless.slm_gs_refine import (
                _DEFAULT_WAVELENGTH_NM,
                _TILT_SHIFT_SCALE_MEASURED_AT_NM,
                _TILT_SHIFT_SCALE_PX,
                focal_length_from_camera_pixel,
            )

            self._focal_length_pinned = False
            self.focal_length_m = focal_length_from_camera_pixel(
                wavelength_nm=float(self.slm_wavelength or _DEFAULT_WAVELENGTH_NM),
                camera_pixel_um=self.camera_pixel_um,
                slm_pixel_um=self.slm_pixel_um or SLM_PITCH_M * 1e6,
                focal_scale_px=_TILT_SHIFT_SCALE_PX,
                scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
            )
        else:
            self._focal_length_pinned = True


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
        target_side_camera: Target square side in camera pixels.
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


def _probe_phase(region: int, spread: float, seed: int) -> np.ndarray:
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
    return np.random.default_rng(int(seed)).normal(
        0.0, float(spread), (int(region), int(region))
    )


class _SimBench:
    """Numerical stand-in for the 2f bench, used for CI and offline QA.

    The far field comes from the canonical analytical model
    (:func:`~ao_shaping.drivers.sim.slm_shaping_bench.forward_intensity`); a
    camera-shaped crop around the peak emulates the detector's ROI, and additive
    read noise plus 16-bit quantisation emulate its noise floor. That last part
    matters: the whole reason Step A needs a strong probe is that the pedestal
    from quantisation is the same order as the signal a weak probe produces.
    """

    def __init__(self, config: SlmModelInLoopConfig, *, read_noise: float = 2.0) -> None:
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
            dict(config.sim_aberration), basis.shape[0],
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
        from ao_shaping.tools.slm.slm_zernike_sweep_probe import capture_settled

        self._capture = capture_settled
        self._config = config
        # create_camera forwards only the keys in the backend's accepted_kwargs,
        # and Daheng's allow-list does NOT include cam_size, so the window has to
        # be set explicitly afterwards.
        self._cam = create_camera(
            config.cam_type,
            cam_id=config.cam_id,
            exposure_time_ms=config.exposure_time_ms,
        )
        self._cam.open()
        if config.cam_size > 0:
            self._cam.reset_window(center=(0, 0), size=(config.cam_size, config.cam_size))
        self._slm = Santec(
            slm_number=config.slm_number,
            wavelength=config.slm_wavelength or None,
        )
        self._slm.open()
        self._disc_radius = int(round(config.panel_span_px / 2.0))
        self._pupil_center = config.pupil_center
        self._pending: np.ndarray | None = None
        self._closed = False

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

    def measure(self) -> np.ndarray:
        if self._pending is None:
            raise RuntimeError("measure() called before display()")
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
        return np.asarray(frame, dtype=np.float64)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for device in (self._slm, self._cam):
            try:
                device.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not mask the run
                logger.warning("failed to close {}: {}", type(device).__name__, exc)


def _open_bench(config: SlmModelInLoopConfig) -> Bench:
    """Open the backend named by ``config.cam_type``."""
    if config.cam_type == "sim":
        return _SimBench(config)
    return _HardwareBench(config)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _metrics_at(frame: np.ndarray, target_side: int, center: tuple[int, int]) -> dict[str, float]:
    """Square metrics on ``frame`` about a **frozen** ``center``.

    The anchor is passed in rather than re-derived. On a speckle field the global
    maximum hops between near-equal grains under a ~1e-3 perturbation, so an
    ``argmax``-rolled box makes the objective discontinuous and the optimiser
    chases a box that no longer covers the beam.

    Args:
        frame: Preprocessed far-field frame.
        target_side: Target square side in the frame's own pixels.
        center: The frozen ``(x, y)`` optical axis.

    Returns:
        The metric dict from
        :func:`~ao_shaping.utils.image.beam_metrics.compute_square_metrics`.
    """
    from ao_shaping.utils.image.beam_metrics import compute_square_metrics

    return compute_square_metrics(frame, int(target_side), (float(center[0]), float(center[1])))


def _quality(
    metrics: dict[str, float], w_efficiency: float = 0.6, w_uniformity: float = 0.4
) -> float:
    """Composite score to maximise, from a square-metrics dict.

    Uses the canonical :func:~ao_shaping.drivers.sim.slm_shaping_bench.composite_from_pib_cv,
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
    from ao_shaping.drivers.sim.slm_shaping_bench import composite_from_pib_cv

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
        reduced = out[:usable, :usable].reshape(
            usable // block, block, usable // block, block
        ).mean(axis=(1, 3))
        if reduced.shape[0] == region:
            return np.ascontiguousarray(reduced)
    return np.ascontiguousarray(
        _zoom(out, (region / span, region / span), order=1, mode="nearest")
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

    Returns:
        The :class:`~ao_shaping.optimizer.wfless.model_in_loop_shaping.BenchGeometry`
        and the records it consumed (empty for the simulated bench).
    """
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        CalibrationRecord,
        calibrate_bench_geometry,
    )

    if isinstance(bench, _SimBench):
        return bench.true_geometry(), []

    records: list[CalibrationRecordLike] = []
    for index in range(int(config.n_calibration_probes)):
        probe = _probe_phase(
            config.region, config.probe_spread, config.seed + 50_000 + index
        )
        bench.display(probe)
        image = _prepare_frame(bench.measure())
        records.append(CalibrationRecord(image=image, phase=probe, label=f"calib{index}"))
    logger.info(
        "collected {} calibration probes for the geometry solve", len(records)
    )
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
    return geometry, records


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
        predicted = optimizer.forward_intensity(
            coefficients, phase, source_amplitude
        )
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


def _make_optimizer(config: SlmModelInLoopConfig, coefficients: np.ndarray | None) -> Any:
    """Build a :class:`ZernikeCoefficientOptimizer` around ``coefficients``.

    Used both for the first fit and to **re-arm** a tripped convergence plateau.
    Re-arming rebuilds around the current coefficients on purpose:
    :meth:`~...ZernikeCoefficientOptimizer.reset` would restore the values passed
    to ``__init__`` and throw away the whole fit.
    """
    from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
        ZernikeCoefficientOptimizer,
    )

    return ZernikeCoefficientOptimizer(
        n_orders=int(config.n_orders),
        region=int(config.region),
        initial_coefficients=coefficients,
        lr=float(config.step_a_lr),
        max_iterations=int(config.step_a_iterations),
        device=config.device,
        seed=int(config.seed),
        dtype=str(config.dtype),
        far_field_size=int(config.region) * int(config.far_field_padding),
        frozen_modes=tuple(config.frozen_modes),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def optimize_slm_model_in_loop(config: SlmModelInLoopConfig) -> ModelInLoopResult:
    """Shape the far field into a uniform square on real SLM + CCD hardware.

    Alternates two steps every round. **Step A** displays strong, varied random
    pupil probes and refits one shared Zernike aberration of the forward model
    against the measured frames, minimising loss(measured, predicted). **Step
    B** freezes that aberration and optimises a full-pixel SLM phase for a
    square target, displays it, and measures the result. Each round's phase warm
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
    from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
        ZernikeCoefficientOptimizer,
    )
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        StepBConfig,
        _fit_aberration_at_probes,
        shape_phase_with_frozen_aberration,
    )
    from ao_shaping.utils.image.beam_metrics import zero_order_center
    from ao_shaping.utils.io.file import Recorder

    recorder = Recorder()

    bench = _open_bench(config)
    # Declared before the ``try`` so the ``finally`` teardown can always reach it.
    # The window itself can only be opened further down, once the geometry solve
    # has produced the target box that the CCD overlay needs.
    live_display: Any = None
    try:
        # ---------------- Stage 0: flat baseline, frozen ROI ---------------
        flat_phase = np.zeros((config.region, config.region), dtype=np.float64)
        bench.display(flat_phase)
        flat_frame = _prepare_frame(bench.measure())
        if not np.any(flat_frame > 0):
            raise RuntimeError(
                "the flat-phase frame is empty after background subtraction; "
                "check the exposure and that the SLM is actually retracting "
                "light onto the camera"
            )
        # Locate the optical axis ONCE. The 0-order is the frame's global
        # maximum, never the geometric centre -- on this bench it sits far
        # off-centre -- and it is frozen for the whole run from here on.
        roi_center = zero_order_center(flat_frame, refine=False)
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

        # Model-space target: the forward model never sees camera pixels.
        scale = float(geometry.camera_px_per_model_px)
        # The forward model emits its far-field grid, which is the pupil grid
        # resampled by the zero-padding factor: padding refines the far-field
        # SAMPLING without changing its angular extent, so one far-field cell is
        # (region / far_field_size) model pixels. Step A therefore needs the
        # measurement on THAT grid, which is a different span from the target.
        cell_scale = scale * float(config.region) / float(geometry.far_field_size)
        logger.info(
            'grid mapping: {:.4f} cam px per model px -> {:.4f} cam px per far-field '
            'cell (far_field_size={})',
            scale, cell_scale, geometry.far_field_size,
        )
        target_side_model = max(3, int(round(config.target_side / scale)))
        target_side_model = min(target_side_model, int(config.region) - 2)
        target_camera = max(3, int(round(target_side_model * scale)))
        logger.info(
            "target side: {} camera px -> {} model px (scale {:.4f} cam/model px)",
            config.target_side, target_side_model, scale,
        )
        target = create_target_mask(
            "square", (int(config.region), int(config.region)), target_side_model
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

        flat_metrics = _metrics_at(flat_frame, target_camera, roi_center)
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
                "_epoch": 0,
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
                target_shape="square",
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
        coefficients = np.zeros(
            int(ZernikeCoefficientOptimizer(
                n_orders=int(config.n_orders), region=int(config.region)
            ).n_coefficients),
            dtype=np.float64,
        )
        step_a_lr = float(config.step_a_lr)
        probe_count = int(config.probe_count)
        rejection_streak = 0
        phase = np.zeros((config.region, config.region), dtype=np.float64)
        # None until the first shaped round is accepted: the flat baseline is not
        # a fair reference for the acceptance test (see acceptance_verdict).
        score_before: float | None = None
        best_score, best_phase, best_frame = flat_score, flat_phase.copy(), flat_frame
        status = ModelInLoopStatus.COMPLETED
        step_b_cfg = StepBConfig(
            iterations=int(config.step_b_iterations),
            lr=float(config.step_b_lr),
            target_shape="square",
            target_side=int(target_side_model),
            w_uniformity=float(config.w_uniformity),
            w_efficiency=float(config.w_efficiency),
        )

        for index in range(int(config.n_rounds)):
            # --- Step A: probe the bench, refit one shared aberration -------
            optimizer = _make_optimizer(config, coefficients)
            step_a_lr = float(getattr(optimizer, "lr", step_a_lr)) or step_a_lr
            frames: list[tuple[np.ndarray, np.ndarray]] = []
            for probe_index in range(probe_count):
                probe = _probe_phase(
                    config.region,
                    config.probe_spread,
                    config.seed + index * 1000 + probe_index,
                )
                bench.display(probe)
                measured = _prepare_frame(bench.measure())
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
                        ),
                        probe,
                    )
                )
            logger.info(
                "round {}/{}: fitting the forward model on {} probes",
                index + 1, config.n_rounds, probe_count,
            )
            fitted, losses, step_a_iters, optimizer = _fit_aberration_at_probes(
                optimizer, frames, int(config.step_a_iterations)
            )
            loss_before = float(losses[0]) if losses else float("nan")
            loss_after = float(losses[-1]) if losses else float("nan")
            del frames

            # --- Guard 1: trust region on the coefficient change -----------
            clamped, step_l2, clamp_applied = trust_region_clamp(
                fitted, coefficients, float(config.trust_region_c_l2)
            )
            if clamp_applied:
                logger.info(
                    "trust region clipped a {:.3f} rad coefficient step to {:.3f} rad",
                    step_l2, float(config.trust_region_c_l2),
                )

            # --- Step B: synthesise the square with that aberration frozen --
            warm = phase if config.warm_start else flat_phase
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
            )
            bench.display(np.asarray(shaped, dtype=np.float64))
            shaped_frame = _prepare_frame(bench.measure())
            shaped_metrics = _metrics_at(shaped_frame, target_camera, roi_center)
            score_after = _quality(shaped_metrics, config.w_efficiency, config.w_uniformity)

            # --- Guard 2: accept or roll the round back ----------------------
            verdict = acceptance_verdict(
                loss_before, loss_after, score_before, score_after,
                loss_delta=float(config.acceptance_loss_delta),
                score_eps=float(config.acceptance_score_eps),
            )
            applied_l2 = float(np.linalg.norm(clamped - coefficients))
            row = {
                "stage": "round",
                "round": index + 1,
                # See the calibration record above: required by the shared pkl writer.
                "_epoch": index + 1,
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
                "uniformity_cv": float(shaped_metrics.get("uniformity_cv", float("nan"))),
                "probe_count": int(probe_count),
                "step_a_lr": float(step_a_lr),
                "rejection_streak": int(rejection_streak),
                "shape_loss_initial": float(shaping_loss[0]) if shaping_loss else float("nan"),
                "shape_loss_final": float(shaping_loss[-1]) if shaping_loss else float("nan"),
            }
            recorder.history.append(row)

            if live_display is not None and not live_display.closed:
                # Prediction under the *clamped* coefficients -- the ones the
                # acceptance test actually judged -- not the raw fit, so the
                # picture matches the verdict in ``reason``.
                _shaped_phase = np.asarray(shaped, dtype=np.float64)
                live_display.update(
                    measured=shaped_frame,
                    phase=_shaped_phase,
                    predicted=_predicted_far_field(
                        optimizer, clamped, _shaped_phase, source_amplitude
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
                phase = np.asarray(shaped, dtype=np.float64)
                score_before = score_after
                rejection_streak = 0
                if score_after > best_score:
                    best_score, best_phase = score_after, phase.copy()
                    best_frame = shaped_frame
                logger.info(
                    "round {}/{} accepted: score {} -> {:.4f} (CV {:.4f}, EE {:.4f})",
                    index + 1, config.n_rounds,
                    "n/a" if score_before is None else f"{score_before:.4f}",
                    score_after, row["cv_after"], row["ee_after"],
                )
            else:
                rejection_streak += 1
                logger.warning(
                    "round {}/{} REJECTED ({}): loss {:.5f} -> {:.5f}, score "
                    "{:.4f} -> {:.4f}; streak {}",
                    index + 1, config.n_rounds, verdict.reason,
                    loss_before, loss_after, row["score_before"], score_after,
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
                    score_before, config.early_stop_score,
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
                best_score, flat_score,
            )
            best_phase, best_frame, best_score = flat_phase.copy(), flat_frame, flat_score
            status = ModelInLoopStatus.BAKE_OFF_REJECTED
        else:
            logger.info(
                "final bake-off: shaped {:.4f} beats flat {:.4f}", best_score, flat_score
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
        )
    finally:
        if live_display is not None:
            live_display.close()
        bench.close()
