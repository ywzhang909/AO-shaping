"""Aberration estimation from camera frames: a probe bank, then a trust-region refit.

What is estimated, and in what units
====================================
Coefficients ``c_j`` of the canonical Zernike basis
(:meth:`ml.sim2f.optics.Sim2f.aberration`, built from
``ao_shaping.utils.wavefront.zernike_calc``), for Noll indices
``2 .. n_modes + 1`` -- tip/tilt included, piston excluded.  They are **raw
unwrapped radians** of pupil phase, the same unit
:func:`ml.sim2f.optics.draw_aberration` draws and :class:`~ml.sim2f.config.Sim2fConfig`
documents.  Nothing here wraps, normalises or re-scales a phase.

Two stages, two jobs
====================
**Probe stage** (:func:`probe`, :func:`build_response`).  Push ``+/- a * Z_j``
onto the panel for every probed mode, measure the camera frame after each push,
and difference the pair.  That difference is a *measured* secant of the camera
frame with respect to coefficient ``j`` -- the response matrix -- so it needs no
derivative of the model and no knowledge of the bench state.  Stacking the
columns gives ``J`` (pixels x modes); ``c_lin = J^+ (I_meas - I_model(0))`` is
the one-step linear estimate.

**Closed-loop stage** (:func:`refine`).  ``argmin_c ||I_model(phi_cmd, c) -
I_meas||``, autograd through :meth:`optics.Sim2f.forward`, under an explicit
per-mode trust region.  ``phi_cmd`` is the phase that was **actually sent** when
``I_meas`` was taken, and it is frozen: see "Why the command is frozen" below.

Both stages score camera frames *after* :func:`ml.sim2f.losses.prepare_frame`
(per-frame median removed, negatives clipped), because the bench carries a
coherent 0-order background, a dark offset and read noise.  A raw ``I.sum()`` is
not the modulated energy.

The fit residual is not the shaping objective
---------------------------------------------
The residual is :func:`_residual`, a **fixed-scale** L1 (or L2) distance between
the predicted and the measured prepared frame, with one nuisance scalar
projected out (the photometric gain: a camera frame's absolute brightness is
exposure x laser power x PRNU, none of which is part of the aberration state).

It is deliberately *not* :func:`ml.sim2f.losses.objective_terms`.  That function
normalises both sides to unit sum, i.e. it divides the prediction by its own
total, so the normaliser depends on ``c`` and the fit has to hold the total
energy fixed while it moves -- a constraint the physics does not impose.  Measured
on the calibrated twin (one drawn state, flat command, 80 Adam steps): the
unit-sum form converges 4x worse (coefficient RMS error 0.09 against 0.024 in
the same budget) and its loss magnitude is ~1e-5, which makes every printed
residual unreadable.  The report's *scoring* still uses ``objective_terms``
everywhere; only the estimator's internal data fidelity uses this residual.

Why the command is frozen
-------------------------
``Sim2f.forward`` applies the requested aberration as an additive term after the
panel stages, and the loop's command is ``phi_shape - sum_j c_j Z_j``.  Writing
the fit as ``argmin_c ||I_model(phi_shape - A c, c) - I_meas||`` therefore makes
the two terms cancel **identically**: with ``phase_blur_px == 0`` and
``lut_gamma == 1`` (both true of the calibrated twin) the propagated phase is
``phi_shape - A c + A c = phi_shape``, the gradient with respect to ``c`` is
exactly zero, and the "refit" is a no-op.  That cancellation *is* the oscillation
the hardware pipeline measured in ``slm_model_in_loop``: a shape term and an
aberration term sharing one basis can trade error back and forth.

So ``phase_cmd`` -- the phase that produced the measured frame -- is an input,
not a function of ``c``.  What moves is the bench aberration alone; the
correction is re-applied by the *caller* after the fit (``phi = phi_shape -
A c_new``), which is also how the loop and the hardware do it.  The trust region
then bounds how far one round may move a mode, so a weakly observed direction
cannot absorb real shaping error.

What this stage cannot do
-------------------------
The aberration state of :class:`~ml.sim2f.config.AberrationConfig` also carries
**illumination placement** (centre jitter, waist scale).  It is not in ``c`` and
is not fitted here, on purpose: an illumination offset is nearly degenerate with
Zernike tip/tilt in the far field, so adding three more free parameters would
create exactly the trade-off the trust region exists to prevent.  The price is
measured and reported: with the default ``illum_jitter_frac = 0.06`` the fit's
coefficient error is dominated by that unmodelled term (RMS error 0.17 rad on a
state whose own high-order RMS is 0.17 rad, i.e. the estimate is worthless),
whereas the same estimator on a state without jitter recovers the coefficients to
0.005-0.03 rad.  ``loop.run_loop`` therefore reports both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ml.sim2f.config import AberrationConfig
from ml.sim2f.losses import EPS, as_frame, prepare_frame

__all__ = [
    "DEFAULT_LADDER_EXPOSURES",
    "DEFAULT_RCOND",
    "DEFAULT_TRUST_RAD",
    "FitConfig",
    "FitResult",
    "Measurement",
    "ResponseMatrix",
    "build_response",
    "noise_ladder",
    "probe",
    "probe_noll",
    "refine",
    "refit_noll",
]

#: Relative singular-value cut of the response matrix.  Modes below it are
#: dropped from the linear step and counted in :attr:`ResponseMatrix.dropped`,
#: because a singular value that small is indistinguishable from rounding; a
#: pseudo-inverse there would amplify measurement noise without recovering
#: signal.  Measured on the calibrated twin: the noise-free bank has
#: ``s_max / s_min = 23 .. 114`` for the probed modes (all kept), and a detector
#: frame whose signal is below the fixed-pattern noise collapses the spectrum to
#: the numerical floor, which is the case the cut is for.
DEFAULT_RCOND = 1e-6

#: Per-mode trust radius of one refinement round, radians.
#:
#: Not a cosmetic constant: the lateral ridge of the residual landscape is flat
#: enough that a first-round estimate can slide along it, so a bound is required,
#: and it has to be *at least* as large as any state :class:`AberrationConfig`
#: draws -- ``rms_rad`` tops out at 0.45 rad over 19 modes, where individual
#: coefficients reach ~1.35 rad.  Measured (worst-case draw, high-order RMS
#: 0.446 rad, ``max|c| = 1.07``): a 0.4 rad radius clipped the fit and left a
#: coefficient RMS error of 0.43 rad; 1.0 -> 0.046; 1.5 -> 0.009; 3.0 -> 0.009.
#: The radius therefore only stops a runaway, it never constrains a physical
#: state.
DEFAULT_TRUST_RAD = 1.5

#: Exposure ladder of :func:`noise_ladder`, milliseconds.  ``None`` would be the
#: detector-less bench, which is the reference point the other rungs are read
#: against, so it is listed explicitly by the caller, not defaulted to.
DEFAULT_LADDER_EXPOSURES: tuple[float, ...] = (0.15, 1.0, 5.0, 20.0)

#: A camera read: ``phase (1, grid, grid) radians -> frame``.  The frame may be
#: counts (a detector bench) or model intensity (a detector-less one); the
#: estimator removes the pedestal and projects out the overall gain, so it does
#: not need to be told which.  Wrapping the truth in this callable is what makes
#: the estimator testable against a stub bench: the bench state lives in the
#: closure, never in the fit.
Measurement = Callable[[Tensor], Tensor]


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FitConfig:
    """One fit's budget and switches.

    The fields that also exist in :class:`~ml.sim2f.config.AberrationConfig`
    (``n_modes``, ``probe_modes``, ``probe_frozen``, ``fit_frozen``,
    ``probe_amplitude_rad``, ``iters``, ``lr``) are **required** here and are
    filled by :meth:`from_aberration`.  That is deliberate: a literal default in
    this class would be a second source of truth for a calibrated number, which
    is the exact bug class this repository has already shipped once
    (``train_amp``'s ``far_field_padding``).  The remaining fields have no
    dataclass counterpart and carry their own documented defaults.

    Attributes:
        n_modes: Coefficient count, Noll ``2 .. n_modes + 1``.
        probe_modes: Noll indices pushed as probes; empty means "every high-order
            coefficient", i.e. Noll ``4 .. n_modes + 1``.
        probe_frozen: Noll indices held at zero by the *probe-stage* fit.  Piston
            (Noll 1) is not in the coefficient vector, so the default freezes
            nothing.
        fit_frozen: Noll indices held at their incoming value by
            :func:`refine`.  Default Noll 1-3: once the target itself decides
            where the energy goes, the spot position no longer identifies tilt.
        probe_amplitude_rad: |amplitude| of each probe push, radians.
        iters: Adam steps of one fit (probe polish or closed-loop round).
        lr: Adam learning rate on the coefficients.
        trust_rad: Per-mode bound on ``|c - c_start|``, radians.
        rcond: Relative singular-value cut of the linear step.
        metric: ``"l1"`` (default) or ``"mse"`` for the residual.
        min_improvement: A probe fit is only accepted if it leaves at most this
            fraction of the residual it started from.  The response matrix can be
            full rank and the fit can still identify nothing: on a frame whose
            signal is below the detector's fixed-pattern noise the polish moves
            along the noise direction (measured at the calibrated 0.15 ms: rank
            19/19, residual 1.012 -> 1.006, coefficient RMS error 0.56 rad on a
            state whose own high-order RMS is 0.17 rad).  A fit that removes less
            than a tenth of the residual is that case, and returning its vector
            would move the real spot while looking like a success.
        init: ``"zero"`` (default) or ``"linear"`` -- where the polish starts.
            Zero is the measured default: the linear step inflates the estimate
            (the speckle response is not linear over a 0.35 rad secant, measured
            2.7x on the calibrated twin), and starting the fit there lands in a
            worse basin (coefficient RMS error 0.49 against 0.024 from zero).
    """

    n_modes: int
    probe_modes: tuple[int, ...]
    probe_frozen: tuple[int, ...]
    fit_frozen: tuple[int, ...]
    probe_amplitude_rad: float
    iters: int
    lr: float
    trust_rad: float = DEFAULT_TRUST_RAD
    rcond: float = DEFAULT_RCOND
    metric: Literal["l1", "mse"] = "l1"
    min_improvement: float = 0.9
    init: Literal["zero", "linear"] = "zero"

    def __post_init__(self) -> None:
        if int(self.n_modes) < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes!r}")
        if not (0.0 < float(self.probe_amplitude_rad) <= math.pi):
            raise ValueError(
                f"probe_amplitude_rad must be in (0, pi], got {self.probe_amplitude_rad!r}"
            )
        if int(self.iters) < 1:
            raise ValueError(f"iters must be >= 1, got {self.iters!r}")
        if not (float(self.lr) > 0.0) or not math.isfinite(float(self.lr)):
            raise ValueError(f"lr must be finite and > 0, got {self.lr!r}")
        if not (float(self.trust_rad) > 0.0) or not math.isfinite(float(self.trust_rad)):
            raise ValueError(f"trust_rad must be finite and > 0, got {self.trust_rad!r}")
        if not (0.0 < float(self.rcond) < 1.0):
            raise ValueError(f"rcond must be in (0, 1), got {self.rcond!r}")
        if not (0.0 < float(self.min_improvement) <= 1.0):
            raise ValueError(
                f"min_improvement must be in (0, 1], got {self.min_improvement!r}"
            )
        if self.metric not in ("l1", "mse"):
            raise ValueError(f"metric must be 'l1' or 'mse', got {self.metric!r}")
        if self.init not in ("zero", "linear"):
            raise ValueError(f"init must be 'zero' or 'linear', got {self.init!r}")

    @classmethod
    def from_aberration(
        cls, cfg: AberrationConfig, **overrides: object
    ) -> "FitConfig":
        """Build from :class:`~ml.sim2f.config.AberrationConfig`.

        Args:
            cfg: The probe / fit contract.
            **overrides: Fields to replace afterwards (e.g. ``trust_rad=0.3`` in
                a unit test).  Unknown names raise, so a typo cannot silently do
                nothing.

        Returns:
            The configuration.

        Raises:
            ValueError: If an override is not a field of this class.
        """
        known = {item for item in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown FitConfig field(s) {unknown}; known: {sorted(known)}")
        base = cls(
            n_modes=int(cfg.n_modes),
            probe_modes=tuple(int(v) for v in cfg.probe_modes),
            probe_frozen=tuple(int(v) for v in cfg.probe_frozen),
            fit_frozen=tuple(int(v) for v in cfg.fit_frozen),
            probe_amplitude_rad=float(cfg.probe_amplitude_rad),
            iters=int(cfg.fit_iters),
            lr=float(cfg.fit_lr),
        )
        return replace(base, **overrides) if overrides else base

    def coefficient_noll(self) -> tuple[int, ...]:
        """Noll indices of the coefficient vector, in row order."""
        return tuple(range(2, int(self.n_modes) + 2))


def probe_noll(cfg: AberrationConfig | FitConfig) -> tuple[int, ...]:
    """Noll indices the probe bank pushes, in matrix-column order.

    ``cfg.probe_modes`` when given, otherwise every high-order coefficient
    (Noll ``4 .. n_modes + 1``) -- :class:`~ml.sim2f.config.AberrationConfig`
    documents that empty means "the first ``n_modes`` modes after 4", and the
    coefficient vector ends at ``n_modes + 1``, so that is the range.  Tip/tilt
    are *not* probed by default: on a flat spot they are directly visible as a
    spot shift, which the frame-based polish reads, and a +/-a push of a mode
    that mostly translates the frame buys nothing the frame does not already
    carry.

    Args:
        cfg: :class:`~ml.sim2f.config.AberrationConfig` or :class:`FitConfig`.

    Returns:
        Sorted Noll indices with ``probe_frozen`` removed.
    """
    n_modes = int(cfg.n_modes)
    explicit = tuple(int(v) for v in getattr(cfg, "probe_modes", ()))
    frozen = set(int(v) for v in getattr(cfg, "probe_frozen", ()))
    if explicit:
        chosen = explicit
    else:
        chosen = tuple(j for j in range(2, n_modes + 2) if j >= 4)
    return tuple(j for j in sorted(set(chosen)) if 2 <= j <= n_modes + 1 and j not in frozen)


def refit_noll(cfg: AberrationConfig | FitConfig) -> tuple[int, ...]:
    """Noll indices :func:`refine` is allowed to move.

    Args:
        cfg: :class:`~ml.sim2f.config.AberrationConfig` or :class:`FitConfig`.

    Returns:
        Sorted Noll indices of the coefficient vector minus ``fit_frozen``.
    """
    n_modes = int(cfg.n_modes)
    frozen = set(int(v) for v in getattr(cfg, "fit_frozen", ()))
    return tuple(j for j in range(2, n_modes + 2) if j not in frozen)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ResponseMatrix:
    """The measured probe response of one bench state.

    Attributes:
        noll: Noll index of each matrix column, in order.
        amplitude_rad: Probe half-amplitude the secant was measured over.
        matrix: ``(ccd_px * ccd_px, len(noll))`` float64 secants, in
            **model-intensity units per radian** (the measured frames were
            rescaled by :attr:`gain`).  Column ``k`` is
            ``dI/dc`` of Noll ``noll[k]``.
        singular: ``(len(noll),)`` singular values, descending.
        rank: Number of singular values above the ``rcond`` cut.
        condition: ``s_max / s_rank`` (``inf`` for a rank-0 matrix).
        gain: Photometric scalar applied to the measurement (model units per
            measurement unit), from the least-squares projection of the model
            reference onto the measured frame.
        linear: ``(n_modes,)`` one-step linear estimate, radians; zero for modes
            not in :attr:`noll`.
        linear_residual: ``||J c_lin - y|| / ||y||`` -- how much of the measured
            frame difference the linear model explains (0 = all of it).
        dropped: Noll indices whose singular value fell below the cut.
        frame: ``(1, 1, ccd_px, ccd_px)`` prepared frame the linear estimate was
            fitted to (the probe command's own measurement).
    """

    noll: tuple[int, ...]
    amplitude_rad: float
    matrix: np.ndarray
    singular: np.ndarray
    rank: int
    condition: float
    gain: float
    linear: np.ndarray
    linear_residual: float
    dropped: tuple[int, ...]
    frame: Tensor = field(repr=False)


@dataclass(frozen=True)
class FitResult:
    """One aberration estimate, with the evidence for it.

    ``ok=False`` is a *documented* failure: :attr:`reason` says which degeneracy
    was hit and :attr:`coeffs` is then the starting point unchanged.  Every caller
    in this package treats it as "no correction" rather than using a partially
    converged vector, because a silently wrong ``c_hat`` moves the spot away from
    the target while looking like a successful fit.

    Attributes:
        ok: Whether the fit is usable.
        reason: Why not (``None`` when ``ok``).
        coeffs: ``(n_modes,)`` float64 radians, Noll ``2 .. n_modes + 1``;
            modes outside :attr:`free_noll` keep their incoming value.
        free_noll: Noll indices the fit was allowed to move.
        residual: Final relative L1 (or L2) residual of the prepared frames.
        residual_start: The same residual at the starting point.
        iterations: Adam steps taken (0 for a failure before the loop).
        history: Residual after each step.
        linear: ``(n_modes,)`` linear (probe) estimate when the probe stage ran.
        linear_residual: Its relative residual, or ``None``.
        rank: Response-matrix rank (0 when no probe stage ran).
        condition: Response-matrix condition number (``inf`` without a probe).
    """

    ok: bool
    reason: str | None
    coeffs: np.ndarray
    free_noll: tuple[int, ...]
    residual: float
    residual_start: float
    iterations: int
    history: tuple[float, ...]
    linear: np.ndarray | None = None
    linear_residual: float | None = None
    rank: int = 0
    condition: float = float("inf")


def _failure(
    reason: str,
    *,
    coeffs: np.ndarray,
    free_noll: Sequence[int],
    residual: float = float("nan"),
    residual_start: float = float("nan"),
    linear: np.ndarray | None = None,
    linear_residual: float | None = None,
    rank: int = 0,
    condition: float = float("inf"),
) -> FitResult:
    """Build a documented failure.

    Args:
        reason: What degeneracy was hit, in one clause.
        coeffs: The (unchanged) starting point.
        free_noll: Noll indices that would have been fitted.
        residual: Residual at the starting point, if it was measured.
        residual_start: Alias kept for symmetry with the success path.
        linear: Linear estimate, if the probe stage got that far.
        linear_residual: Its relative residual.
        rank: Response-matrix rank.
        condition: Response-matrix condition number.

    Returns:
        A ``FitResult`` with ``ok=False`` and no iterations.
    """
    return FitResult(
        ok=False,
        reason=str(reason),
        coeffs=np.asarray(coeffs, dtype=np.float64).copy(),
        free_noll=tuple(int(v) for v in free_noll),
        residual=float(residual),
        residual_start=float(residual_start),
        iterations=0,
        history=(),
        linear=None if linear is None else np.asarray(linear, dtype=np.float64).copy(),
        linear_residual=linear_residual,
        rank=int(rank),
        condition=float(condition),
    )


# ----------------------------------------------------------------------
# Frame helpers
# ----------------------------------------------------------------------
def _prepare(frame: Tensor | np.ndarray) -> Tensor:
    """Normalise any camera frame to a prepared ``(1, 1, ccd_px, ccd_px)``.

    Args:
        frame: Anything :func:`ml.sim2f.losses.as_frame` accepts.

    Returns:
        The prepared frame (per-frame median removed, negatives clipped).
    """
    return prepare_frame(as_frame(frame))


def _frame_on(bench: object, frame: Tensor | np.ndarray | Sequence[float]) -> Tensor:
    """Prepare a camera frame and put it on the bench's device and dtype.

    A measurement may arrive from numpy (a saved frame, another process) or from
    a torch path on a different device; :func:`_residual` needs it beside the
    prediction, so the conversion happens here once.

    Args:
        bench: The bench (supplies ``device`` / ``dtype``).
        frame: Measured frame, any layout :func:`ml.sim2f.losses.as_frame` accepts.

    Returns:
        ``(1, 1, ccd_px, ccd_px)`` prepared frame on the bench's device.
    """
    if not torch.is_tensor(frame) and not isinstance(frame, (tuple, list)):
        frame = torch.as_tensor(np.asarray(frame, dtype=np.float64))
    device = getattr(bench, "device", torch.device("cpu"))
    dtype = getattr(bench, "dtype", torch.float32)
    return _prepare(frame).to(device=device, dtype=dtype)


def _as_phase(bench: object, phase: Tensor | np.ndarray | None) -> Tensor:
    """Normalise a command phase to ``(1, grid, grid)`` on the bench's device.

    Args:
        bench: The bench (supplies ``grid``, ``dtype``, ``device``).
        phase: ``None`` (zero phase), ``(grid, grid)`` or ``(1, grid, grid)``.

    Returns:
        The command phase, radians.

    Raises:
        ValueError: If the map does not match the bench grid.
    """
    grid = int(getattr(bench, "grid"))
    dtype = getattr(bench, "dtype", torch.float32)
    device = getattr(bench, "device", torch.device("cpu"))
    if phase is None:
        return torch.zeros(1, grid, grid, dtype=dtype, device=device)
    tensor = phase if isinstance(phase, Tensor) else torch.as_tensor(phase)
    if tensor.dim() == 2:
        tensor = tensor[None]
    if tensor.dim() != 3 or tensor.shape[0] != 1 or tuple(tensor.shape[-2:]) != (grid, grid):
        raise ValueError(
            f"phase must be (1, {grid}, {grid}) or ({grid}, {grid}), got "
            f"{tuple(tensor.shape)}"
        )
    return tensor.to(device=device, dtype=dtype)


def _model_frame(bench: object, phase: Tensor) -> Tensor:
    """Aberration-free model frame at ``phase``, prepared.

    The detector is switched off explicitly: a fit compares predictions with a
    measurement, and a quantised, noisy prediction would floor the gradient the
    fit needs.

    Args:
        bench: The bench.
        phase: ``(1, grid, grid)`` command, radians.

    Returns:
        ``(1, 1, ccd_px, ccd_px)`` prepared model intensity.
    """
    with torch.no_grad():
        frame = bench.forward(phase, detector=False)  # type: ignore[attr-defined]
    return _prepare(frame)


def _coefficient_vector(bench: object, coeffs: np.ndarray | Tensor | Sequence[float]) -> np.ndarray:
    """Validate a coefficient vector against the bench's basis size.

    Args:
        bench: The bench.
        coeffs: ``(n_modes,)`` values, radians.

    Returns:
        A float64 copy.

    Raises:
        ValueError: If the vector is not 1-D or is empty.
    """
    array = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if array.size < 1:
        raise ValueError("a coefficient vector needs at least one element")
    return array.copy()


def _residual(predicted: Tensor, measured: Tensor, *, metric: str) -> tuple[Tensor, float]:
    """Fixed-scale residual between a prediction and a measurement.

    The overall gain is a nuisance parameter -- a camera frame's absolute
    brightness is exposure x laser power x PRNU, none of which is part of the
    aberration state -- so it is projected out in closed form and treated as a
    constant (its own derivative is dropped, which is the usual profile-likelihood
    treatment and keeps the fit from trading scale against shape).

    Args:
        predicted: ``(1, 1, ccd_px, ccd_px)`` prepared model frame, differentiable.
        measured: ``(1, 1, ccd_px, ccd_px)`` prepared measured frame.
        metric: ``"l1"`` or ``"mse"``.

    Returns:
        ``(loss, gain)``: the scalar to differentiate and the gain that was
        projected out.
    """
    with torch.no_grad():
        gain = (predicted * measured).sum() / predicted.pow(2).sum().clamp_min(EPS)
    difference = predicted * gain - measured
    if metric == "mse":
        denominator = measured.pow(2).sum().clamp_min(EPS)
        return difference.pow(2).sum() / denominator, float(gain)
    denominator = measured.abs().sum().clamp_min(EPS)
    return difference.abs().sum() / denominator, float(gain)


def _unit_coefficient(cfg: FitConfig, noll: int) -> np.ndarray:
    """Unit vector with a 1 at the row of Noll ``noll``.

    Args:
        cfg: Fit configuration (supplies ``n_modes``).
        noll: Noll index, ``2 .. n_modes + 1``.

    Returns:
        ``(n_modes,)`` float64 vector.

    Raises:
        ValueError: If ``noll`` is outside the coefficient vector.
    """
    index = int(noll) - 2
    if not 0 <= index < int(cfg.n_modes):
        raise ValueError(
            f"Noll {noll} is outside the coefficient vector "
            f"2..{int(cfg.n_modes) + 1}"
        )
    vector = np.zeros(int(cfg.n_modes), dtype=np.float64)
    vector[index] = 1.0
    return vector


# ----------------------------------------------------------------------
# Probe stage
# ----------------------------------------------------------------------
def build_response(
    bench: object,
    cfg: AberrationConfig | FitConfig,
    measure: Measurement,
    *,
    phase: Tensor | np.ndarray | None = None,
    measured: Tensor | np.ndarray | None = None,
) -> ResponseMatrix:
    """Measure the probe bank and build the response matrix.

    For every probed Noll ``j`` the panel is pushed by ``+a Z_j`` and ``-a Z_j``
    *on top of* ``phase``, and the two prepared frames are differenced:
    ``J[:, k] = gain * (I(+a) - I(-a)) / (2a)``.  A central difference cancels the
    even part of the response, which is the part that does not identify the sign
    of the coefficient.

    The measured frame is put on the model's scale by one scalar ``gain`` (see
    :func:`_residual`); without it a counts frame and a model-intensity frame
    cannot be subtracted at all, and the least-squares solve would return
    coefficients in the wrong units.

    Args:
        bench: The bench.  Needs ``grid``, ``ccd_px``, ``dtype``, ``device``,
            ``aberration(coeffs)`` and ``forward(phase, aberration=..., detector=...)``.
        cfg: Probe contract (``n_modes``, ``probe_modes``, ``probe_amplitude_rad``).
        measure: Callable that pushes a phase and returns a camera frame.
        phase: Command the probes are added to (default zeros).
        measured: Frame the linear estimate is fitted to.  Defaults to
            ``measure(phase)``, measured once and reused.

    Returns:
        The :class:`ResponseMatrix`.  A dark or degenerate bench yields ``rank=0``
        rather than an exception; :func:`probe` turns that into a failure.
    """
    fit_cfg = cfg if isinstance(cfg, FitConfig) else FitConfig.from_aberration(cfg)
    command = _as_phase(bench, phase)
    model0 = _model_frame(bench, command)
    frame = (
        _frame_on(bench, frame_of(measure(command)))
        if measured is None
        else _frame_on(bench, measured)
    )

    with torch.no_grad():
        denominator = model0.pow(2).sum().clamp_min(EPS)
        gain = float((model0 * frame).sum() / denominator)
    # A frame whose total is zero carries no photometry; fall back to 1 so the
    # matrix is exactly zero rather than NaN and `probe` can report the failure.
    if not math.isfinite(gain) or gain <= 0.0:
        gain = 1.0
    model0_flat = model0.flatten().cpu().numpy().astype(np.float64)
    target = frame.flatten().cpu().numpy().astype(np.float64) * gain - model0_flat

    noll = probe_noll(fit_cfg)
    amplitude = float(fit_cfg.probe_amplitude_rad)
    columns = np.zeros((target.size, len(noll)), dtype=np.float64)
    for column, index in enumerate(noll):
        unit = _unit_coefficient(fit_cfg, index)
        map_unit = bench.aberration(unit[None].astype(np.float32))[0]  # type: ignore[attr-defined]
        if isinstance(map_unit, Tensor):
            map_unit = map_unit.detach()
        plus = _prepare(frame_of(measure(command + amplitude * map_unit)))
        minus = _prepare(frame_of(measure(command - amplitude * map_unit)))
        difference = (plus - minus).flatten().cpu().numpy().astype(np.float64)
        columns[:, column] = difference * (gain / (2.0 * amplitude))

    matrix = columns
    if noll:
        singular = np.linalg.svd(matrix, compute_uv=False)
    else:
        singular = np.zeros(0, dtype=np.float64)
    cut = float(fit_cfg.rcond) * float(singular[0]) if singular.size and singular[0] > 0 else 0.0
    keep = [position for position, value in enumerate(singular) if value > cut] if cut > 0 else []
    rank = len(keep)
    dropped = tuple(int(noll[position]) for position in range(len(noll)) if position not in keep)
    condition = (
        float(singular[0] / singular[keep[-1]]) if rank and singular[keep[-1]] > 0 else float("inf")
    )

    linear = np.zeros(int(fit_cfg.n_modes), dtype=np.float64)
    linear_residual = float("nan")
    if rank:
        # Truncated pseudo-inverse on the retained subspace.  `lstsq` with the
        # same `rcond` produces the identical solution; doing it through the
        # singular vectors we already have avoids a second decomposition.
        left, values, right = np.linalg.svd(matrix, full_matrices=False)
        coefficients = right[:rank].T @ ((left[:, :rank].T @ target) / values[:rank])
        for position, index in enumerate(noll):
            linear[int(index) - 2] = coefficients[position]
        predicted = matrix @ coefficients
        norm = float(np.linalg.norm(target))
        linear_residual = (
            float(np.linalg.norm(predicted - target) / norm) if norm > 0 else float("nan")
        )
    else:
        logger.debug(
            "probe response has rank 0 for {} probed mode(s); the linear step is unusable",
            len(noll),
        )

    return ResponseMatrix(
        noll=tuple(int(v) for v in noll),
        amplitude_rad=amplitude,
        matrix=matrix,
        singular=singular,
        rank=rank,
        condition=condition,
        gain=gain,
        linear=linear,
        linear_residual=linear_residual,
        dropped=dropped,
        frame=frame,
    )


def frame_of(value: object) -> Tensor:
    """Normalise a ``measure`` return value to ``(1, 1, H, W)``.

    ``measure`` may hand back a bare ``(ccd, ccd)`` frame, a batch, or a
    ``(frame, extras)`` tuple; :func:`ml.sim2f.losses.as_frame` is the package's
    single adapter for that, and this thin alias keeps the intent readable at the
    call sites above.

    Args:
        value: A tensor or a tuple whose first element is the frame.

    Returns:
        ``(1, 1, H, W)`` frame.
    """
    return as_frame(value)


def probe(
    bench: object,
    cfg: AberrationConfig,
    *,
    measure: Measurement,
    phase: Tensor | np.ndarray | None = None,
    measured: Tensor | np.ndarray | None = None,
    fit_cfg: FitConfig | None = None,
) -> FitResult:
    """Probe-stage estimate: response matrix, linear step, then a fit polish.

    The linear step alone is *not* the answer on this bench.  The camera frame is
    a speckle field whose intensity is quadratic in the phase over the interval
    the probe spans, so a one-step secant inversion inflates the estimate
    (measured 2.7x on the calibrated twin) and leaves a coefficient RMS error as
    large as the coefficients themselves.  The stage therefore uses the linear
    estimate as *evidence* (it is reported, with the rank and condition number
    that say whether the bank had any authority) and runs the same trust-region
    fit :func:`refine` uses, from ``fit_cfg.init`` (zero by default -- measured
    better than the linear start, see :class:`FitConfig`).

    Args:
        bench: The bench (see :func:`build_response`).
        cfg: Probe/fit contract.
        measure: Camera read.
        phase: Command the measurement and probes were taken at (default zeros).
        measured: Frame to fit; defaults to ``measure(phase)``.
        fit_cfg: Explicit budget; defaults to ``FitConfig.from_aberration(cfg)``.

    Returns:
        The :class:`FitResult`.  ``ok=False`` for a rank-deficient bank, a dark
        frame, or a fit that produced a non-finite residual -- never a silently
        wrong vector.
    """
    resolved = FitConfig.from_aberration(cfg) if fit_cfg is None else fit_cfg
    command = _as_phase(bench, phase)
    response = build_response(
        bench, resolved, measure, phase=command, measured=measured
    )
    free = tuple(
        int(j) for j in range(2, int(resolved.n_modes) + 2) if int(j) not in set(resolved.probe_frozen)
    )
    zeros = np.zeros(int(resolved.n_modes), dtype=np.float64)

    if response.rank == 0:
        return _failure(
            "rank-deficient probe matrix: no probed mode moved the frame above the "
            f"noise floor (rank 0 of {len(response.noll)}, gain={response.gain:.3g})",
            coeffs=zeros,
            free_noll=free,
            linear=response.linear,
            linear_residual=response.linear_residual,
            rank=0,
            condition=response.condition,
        )
    if not free:
        return _failure(
            "no free modes: probe_frozen covers the whole coefficient vector",
            coeffs=zeros,
            free_noll=free,
            linear=response.linear,
            linear_residual=response.linear_residual,
            rank=response.rank,
            condition=response.condition,
        )

    start = response.linear.copy() if resolved.init == "linear" else zeros.copy()
    coeffs, history, residual_start = _polish(
        bench,
        command,
        response.frame,
        start,
        _noll_to_rows(resolved, free),
        resolved,
    )
    if not history or not math.isfinite(history[-1]):
        return _failure(
            "fit residual is not finite: the measured frame or the model degenerate",
            coeffs=start,
            free_noll=free,
            residual_start=residual_start,
            linear=response.linear,
            linear_residual=response.linear_residual,
            rank=response.rank,
            condition=response.condition,
        )
    if not math.isfinite(residual_start) or history[-1] > float(resolved.min_improvement) * residual_start:
        return _failure(
            "the frame is below the estimator's noise floor: the fit removed less "
            f"than {100.0 * (1.0 - float(resolved.min_improvement)):.0f}% of the "
            f"starting residual ({residual_start:.3f} -> {history[-1]:.3f}), so the "
            "returned coefficients would be the noise direction",
            coeffs=start,
            free_noll=free,
            residual=float(history[-1]),
            residual_start=residual_start,
            linear=response.linear,
            linear_residual=response.linear_residual,
            rank=response.rank,
            condition=response.condition,
        )
    logger.debug(
        "probe fit: rank={}/{} cond={:.1f} linear_resid={:.3f} fit {:.3f}->{:.4f} "
        "iter={}",
        response.rank,
        len(response.noll),
        response.condition,
        response.linear_residual,
        residual_start,
        history[-1],
        len(history),
    )
    return FitResult(
        ok=True,
        reason=None,
        coeffs=coeffs,
        free_noll=free,
        residual=float(history[-1]),
        residual_start=float(residual_start),
        iterations=len(history),
        history=tuple(history),
        linear=response.linear,
        linear_residual=response.linear_residual,
        rank=response.rank,
        condition=response.condition,
    )


# ----------------------------------------------------------------------
# Closed-loop stage
# ----------------------------------------------------------------------
def refine(
    bench: object,
    phase_cmd: Tensor | np.ndarray,
    measured: Tensor | np.ndarray,
    c_hat0: np.ndarray | Sequence[float],
    *,
    cfg: AberrationConfig,
    fit_cfg: FitConfig | None = None,
) -> FitResult:
    """Re-fit the aberration against one measured frame, under a trust region.

    ``argmin_c ||I_model(phase_cmd, c) - I_meas||`` with autograd through
    :meth:`optics.Sim2f.forward`.  ``phase_cmd`` is the phase that produced
    ``measured`` and is held fixed (see the module docstring: forming it from
    ``c`` cancels the correction identically).  Only the Noll indices in
    :func:`refit_noll` move; each is bounded to ``c_start +/- trust_rad``.

    Args:
        bench: The bench.
        phase_cmd: Command phase that produced ``measured``, ``(1, grid, grid)``
            radians -- already including the correction being validated.
        measured: The camera frame to fit (counts or model intensity).
        c_hat0: ``(n_modes,)`` starting point, radians (the trust-region centre).
        cfg: Contract supplying ``n_modes`` and ``fit_frozen``.
        fit_cfg: Explicit budget; defaults to ``FitConfig.from_aberration(cfg)``.

    Returns:
        The :class:`FitResult`; ``ok=False`` with a reason when there is nothing
        to fit (all modes frozen) or the frame is empty.

    Raises:
        ValueError: If ``c_hat0`` does not match ``cfg.n_modes``.
    """
    resolved = FitConfig.from_aberration(cfg) if fit_cfg is None else fit_cfg
    start = _coefficient_vector(bench, c_hat0)
    if start.size != int(resolved.n_modes):
        raise ValueError(
            f"c_hat0 has {start.size} coefficient(s), expected {int(resolved.n_modes)} "
            f"from n_modes={int(resolved.n_modes)}"
        )
    command = _as_phase(bench, phase_cmd)
    frame = _frame_on(bench, measured)
    free = refit_noll(resolved)
    if not free:
        return _failure(
            "no free modes: fit_frozen covers the whole coefficient vector",
            coeffs=start,
            free_noll=free,
        )
    with torch.no_grad():
        energy = float(frame.abs().sum())
    if not math.isfinite(energy) or energy <= 0.0:
        return _failure(
            "empty measurement: the prepared frame is all zeros, so it carries no "
            "information about the bench state",
            coeffs=start,
            free_noll=free,
        )

    coeffs, history, residual_start = _polish(
        bench, command, frame, start, _noll_to_rows(resolved, free), resolved
    )
    if not history or not math.isfinite(history[-1]):
        return _failure(
            "fit residual is not finite",
            coeffs=start,
            free_noll=free,
            residual_start=residual_start,
        )
    if history[-1] > residual_start:
        return _failure(
            "the refit increased the residual "
            f"({residual_start:.4f} -> {history[-1]:.4f}): keep the previous estimate",
            coeffs=start,
            free_noll=free,
            residual=float(history[-1]),
            residual_start=residual_start,
        )
    return FitResult(
        ok=True,
        reason=None,
        coeffs=coeffs,
        free_noll=free,
        residual=float(history[-1]),
        residual_start=float(residual_start),
        iterations=len(history),
        history=tuple(history),
    )


def _noll_to_rows(cfg: FitConfig, noll: Iterable[int]) -> tuple[int, ...]:
    """Map Noll indices to 0-based row positions of the coefficient vector.

    Args:
        cfg: Fit configuration.
        noll: Noll indices.

    Returns:
        Row positions, sorted.

    Raises:
        ValueError: If an index is outside the vector.
    """
    rows = []
    for index in noll:
        row = int(index) - 2
        if not 0 <= row < int(cfg.n_modes):
            raise ValueError(
                f"Noll {index} is outside the coefficient vector 2..{int(cfg.n_modes) + 1}"
            )
        rows.append(row)
    return tuple(sorted(rows))


def _polish(
    bench: object,
    phase_cmd: Tensor,
    measured: Tensor,
    start: np.ndarray,
    free_rows: Sequence[int],
    cfg: FitConfig,
) -> tuple[np.ndarray, list[float], float]:
    """Adam on the free coefficients, with a per-mode box and a frozen command.

    Args:
        bench: The bench.
        phase_cmd: Fixed command phase, ``(1, grid, grid)`` radians.
        measured: Prepared measured frame, ``(1, 1, ccd_px, ccd_px)``.
        start: ``(n_modes,)`` starting coefficients, radians.
        free_rows: Row positions allowed to move.
        cfg: Budget, trust radius, metric and learning rate.

    Returns:
        ``(coeffs, history, residual_start)``.  ``history`` is the residual after
        each step; ``residual_start`` is measured before the first step.
    """
    dtype = getattr(bench, "dtype", torch.float32)
    device = getattr(bench, "device", torch.device("cpu"))
    centre = torch.as_tensor(start, dtype=dtype, device=device)
    delta = torch.zeros_like(centre, requires_grad=True)
    free_mask = torch.zeros_like(centre)
    for row in free_rows:
        free_mask[row] = 1.0
    optimizer = torch.optim.Adam([delta], lr=float(cfg.lr))
    radius = float(cfg.trust_rad)

    def model_frame() -> Tensor:
        coefficients = (centre + delta * free_mask)[None]
        aberration = bench.aberration(coefficients)  # type: ignore[attr-defined]
        return _prepare(
            bench.forward(phase_cmd, aberration=aberration, detector=False)  # type: ignore[attr-defined]
        )

    with torch.no_grad():
        loss, _gain = _residual(model_frame(), measured, metric=cfg.metric)
    residual_start = float(loss)

    history: list[float] = []
    for _ in range(int(cfg.iters)):
        optimizer.zero_grad(set_to_none=True)
        loss, _gain = _residual(model_frame(), measured, metric=cfg.metric)
        if not torch.isfinite(loss):
            logger.debug("fit residual became non-finite; stopping at {} step(s)", len(history))
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.data.clamp_(-radius, radius)
            delta.data[free_mask == 0.0] = 0.0
        history.append(float(loss.detach()))

    coeffs = (centre + delta * free_mask).detach().cpu().numpy().astype(np.float64)
    return coeffs, history, residual_start


# ----------------------------------------------------------------------
# Noise ladder
# ----------------------------------------------------------------------
def noise_ladder(
    bench: object,
    cfg: AberrationConfig,
    *,
    coeffs: np.ndarray | Sequence[float],
    exposures: Sequence[float | None] = DEFAULT_LADDER_EXPOSURES,
    read_sigmas: Sequence[float] | None = None,
    seed: int = 0,
    illumination: Tensor | np.ndarray | None = None,
    fit_cfg: FitConfig | None = None,
) -> list[dict[str, float | None]]:
    """Measure the fit's coefficient error against detector noise.

    One rung is one (exposure, read-noise) setting: the frame is measured with the
    true aberration applied, the probe stage runs, and the recovered vector is
    compared with the truth.  This is what makes "error falls as noise falls" a
    measured statement instead of an assertion.

    Both noise axes are physical and independent on this bench: ``exposure_ms``
    scales the signal (and with it the 8-bit quantisation step relative to the
    signal), while ``read_sigma_counts`` / the dark fixed-pattern map set the
    floor that does not grow with exposure.  A read-sigma rung is measured on a
    *rebuilt* bench (``replace(cfg, read_sigma_counts=...)``), so every rung is
    still the one forward model, only with a different detector.

    Args:
        bench: The bench (must expose ``cfg`` for the read-noise rungs).
        cfg: Probe/fit contract.
        coeffs: True coefficients, ``(n_modes,)`` radians.
        exposures: Exposures in milliseconds; ``None`` means the detector is off
            (the noise-free reference rung).
        read_sigmas: Optional read-noise values, counts.  ``None`` keeps the
            bench's own detector.
        seed: Noise seed; each rung gets ``seed * 1000 + rung``.
        illumination: Optional ``(1, 1, grid, grid)`` entrance amplitude; defaults
            to the bench's nominal beam.
        fit_cfg: Explicit budget.

    Returns:
        One dict per rung, in ``(read_sigma, exposure)`` order, with
        ``exposure_ms``, ``read_sigma_counts``, ``peak_counts``, ``signal``,
        ``residual``, ``coeff_rmse`` (over the fitted modes), ``coeff_max``,
        ``cos`` (cosine similarity with the truth) and ``rank``.
        A rung whose fit failed reports ``coeff_rmse=None`` -- "not measured" is
        not the same as "zero error".

    Raises:
        ValueError: If the true vector does not match ``cfg.n_modes``.
    """
    resolved = FitConfig.from_aberration(cfg) if fit_cfg is None else fit_cfg
    truth = _coefficient_vector(bench, coeffs)
    if truth.size != int(resolved.n_modes):
        raise ValueError(
            f"coeffs has {truth.size} element(s), expected {int(resolved.n_modes)}"
        )
    # The truth's own aberration map, built through the bench's basis so the two
    # sides cannot disagree about the support radius or the Noll order.
    aberration = bench.aberration(truth[None].astype(np.float32))  # type: ignore[attr-defined]
    if isinstance(aberration, Tensor):
        aberration = aberration.detach()
    device = getattr(bench, "device", torch.device("cpu"))
    dtype = getattr(bench, "dtype", torch.float32)

    rungs: list[dict[str, float | None]] = []
    for sigma in read_sigmas if read_sigmas is not None else (None,):
        active = bench
        if sigma is not None:
            if not hasattr(bench, "cfg"):
                raise ValueError("read_sigmas needs a bench exposing `cfg` to rebuild")
            from ml.sim2f.optics import Sim2f

            active = Sim2f(
                replace(bench.cfg, read_sigma_counts=float(sigma)),  # type: ignore[attr-defined]
                device=str(device),
                dtype=dtype,
            )
        for exposure in exposures:
            index = len(rungs)
            generator = torch.Generator(device=device).manual_seed(int(seed) * 1000 + index)
            detector = exposure is not None

            def measure(
                phase: Tensor,
                _active: object = active,
                _exposure: float | None = exposure,
                _detector: bool = detector,
                _generator: torch.Generator = generator,
            ) -> Tensor:
                with torch.no_grad():
                    return _active.forward(  # type: ignore[attr-defined]
                        phase,
                        aberration=aberration,
                        illumination=illumination,
                        detector=_detector,
                        exposure_ms=_exposure if _detector else None,
                        generator=_generator if _detector else None,
                    )

            result = probe(active, cfg, measure=measure, fit_cfg=resolved)
            frame = _prepare(measure(_as_phase(active, None)))
            error = result.coeffs - truth
            fitted = np.array([result.coeffs[j - 2] for j in refit_noll(resolved)], dtype=np.float64)
            reference = np.array([truth[j - 2] for j in refit_noll(resolved)], dtype=np.float64)
            norm = float(np.linalg.norm(fitted) * np.linalg.norm(reference))
            rungs.append(
                {
                    "exposure_ms": None if exposure is None else float(exposure),
                    "read_sigma_counts": None if sigma is None else float(sigma),
                    "peak_counts": float(frame.amax()),
                    "signal": float(frame.sum()),
                    "residual": float(result.residual),
                    "residual_start": float(result.residual_start),
                    "coeff_rmse": float(np.sqrt(np.mean(error**2))) if result.ok else None,
                    "coeff_max": float(np.abs(error).max()) if result.ok else None,
                    "cos": float(fitted @ reference / norm) if norm > 0 and result.ok else None,
                    "rank": int(result.rank),
                    "iterations": int(result.iterations),
                }
            )
            logger.info(
                "ladder exposure={} read_sigma={} peak={:.1f} residual {:.3f}->{:.4f} "
                "coeff_rmse={}",
                exposure,
                sigma,
                float(frame.amax()),
                result.residual_start,
                result.residual,
                "n/a" if not result.ok else f"{rungs[-1]['coeff_rmse']:.4f}",
            )
    return rungs
