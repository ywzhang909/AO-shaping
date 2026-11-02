"""Active probing for the probe stage of :mod:`ml.sim2f.fit`.

Why a fixed probe order is a choice, not a necessity
====================================================
:func:`ml.sim2f.fit.probe` pushes a fixed bank (Noll ``4 .. n_modes + 1``, in
Noll order) and spends the same number of camera frames on every mode.  That is
the right thing to do when nothing is known yet, but it is not optimal once
*some* frames exist: a mode whose response the camera can barely see buys almost
no information per frame, while a mode the current residual still excites buys a
lot.  With a hard frame budget -- the real constraint on a bench, where each
probe is one SLM write plus one exposure -- the order of the bank decides how
much aberration is left at the end.

This module answers "which mode next?" with a *posterior*: a weighted
least-squares fit of the stacked probe blocks, with its covariance.  The
covariance says how uncertain each coefficient still is, and the greedy
D-optimality rule picks the probe that shrinks ``trace(cov)`` the most.

What is measured, and in what units
===================================
Identical to :mod:`ml.sim2f.fit`, and it reuses that module's helpers
(``_prepare``, ``_as_phase``, ``_frame_on``, ``_unit_coefficient``) rather than
re-deriving them, so the two stages cannot drift apart in units or frame
normalisation:

* the unknown is the coefficient vector ``c`` of the canonical Zernike basis
  (Noll ``2 .. n_modes + 1``), **raw unwrapped radians**;
* one probe is a pair of pushes ``+/- a Z_j`` on top of the current compensation
  command.  The device frame is measured at both, the *nominal* (aberration-free)
  model frame is predicted at the same **push** phase, and the pair is
  differenced into a residual and a response column.

The residual convention is the load-bearing detail, and it is easy to get
backwards.  A measured frame is brought onto the model's scale by **dividing**
by the photometric scalar ``alpha = <m0, y> / <m0, m0>``::

    rho = y / alpha - m0

Multiplying instead of dividing returns coefficients of order several radians
and a residual that never falls (measured while prototyping this module).  With
the division the residual at the truth is ~0.02 RMS on a clean device, which is
what makes the linearisation adequate here.

Because the command already cancels the running estimate, a probe measures the
*residual* aberration ``d = truth - estimate``, not the truth: ``rho`` is
``f(d + push) - f(push)``.  Each frame is stored together with the estimate in
force when it was taken, and the stacked system is re-linearised at the *running*
estimate, so every block constrains the same increment ``truth - estimate``.  That
matters because the camera response is quadratic in the coefficient: a block
linearised once, far from the minimum, carries a bias that grows with the error
rather than shrinking it.  Solving the stacked weighted LS with the trust region
as a zero-mean prior and adding the clipped increment is therefore a Gauss-Newton
iteration, and :func:`probe_active` runs it to convergence *on the stored frames*
after each probe -- the inner loop costs model forwards, not camera frames.

Intensity is even in every single-mode coefficient at zero phase, so a probe
pair's *central difference* vanishes there; that is why the response is taken at
``+/- a`` (the secant) and not at the command.

Why the response comes from the model, not the frames
====================================================
:func:`ml.sim2f.fit.build_response` differences *measured* frames.  That cannot
answer "what would probing mode 7 buy me?" for a mode that has not been probed,
so this module differences the **windowed model prediction** instead
(``d/dc prep(model.forward(phi + h Z_j))``).  The model is the nominal bench --
an ideal :class:`~ml.sim2f.optics.Sim2f` -- so its response is exactly the design
the posterior needs, and the device's unmodelled errors stay in the residual
where they belong.

The design matrix is used twice, and the second use is why
:class:`PosteriorEstimate` carries ``gram``:

* estimation -- stacked blocks ``J`` and residuals ``rho`` are solved by
  :func:`linear_posterior`;
* selection -- probing mode ``j`` adds the *column* ``a J[:, j]`` to the design,
  which contributes ``(a J^T J[:, j])(a J^T J[:, j])^T / sigma2`` to the
  coefficient-space information.  So the exact trace reduction for a candidate
  follows from ``gram = J^T J`` alone, with no need to re-run the model per
  candidate.

Repeated probes
===============
A budget can be larger than the number of modes worth probing, and then a mode is
probed twice.  That is deliberate: each push is an independent noise realisation
at a new operating point, so a repeat still shrinks the posterior -- just less
than a first visit.  :func:`select_probe` prices a repeat with the same gain
formula (evaluated on the current covariance, so it naturally ranks below a
fresh mode), and the fixed baseline cycles :func:`ml.sim2f.fit.probe_noll` when
its bank is shorter than the budget.  A *candidate list* may not contain a
duplicate: that is a typo, not a request, and it raises.

Adaptive probe amplitude
========================

A probe whose push is small compared with the current uncertainty returns a
frame change buried in noise.  :func:`probe_active` therefore scales the push up
-- never down, and never outside ``[0.5, 1.5] x`` the configured
``probe_amplitude_rad`` -- when the predicted per-mode SNR
``a / confidence(j)`` is below ``snr_target``.  The band is a hard bound because
the linearisation (and the panel's own phase range) degrades with a large push:
the rule may spend at most 50% more phase than the calibrated value.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ml.sim2f.config import AberrationConfig
from ml.sim2f.fit import (
    FitConfig,
    Measurement,
    _as_phase,
    _frame_on,
    _prepare,
    _unit_coefficient,
    frame_of,
    probe_noll,
)

__all__ = [
    "ActiveProbeResult",
    "PosteriorEstimate",
    "StrategyOutcome",
    "compare_probe_strategies",
    "linear_posterior",
    "probe_active",
    "select_probe",
]

#: Floor on the estimated noise scale.  A noiseless design has zero residual, and
#: ``cov = sigma2 * pinv(A)`` would then report zero variance for every mode --
#: which would make :func:`select_probe` indifferent.  The floor keeps the
#: covariance a *relative* statement in that degenerate case.
_SIGMA_FLOOR = 1e-12

#: Smallest denominator accepted by the information update, same reason.
_DENOM_FLOOR = 1e-12

#: Gauss-Newton iterations spent re-solving the *stored* frames after each new
#: probe.  A probe costs one SLM write and one exposure; re-solving costs only
#: model forwards, so the inner loop is free and buys accuracy that a single
#: linearisation cannot: the camera response is quadratic in the coefficients,
#: and a linear solve on the wrong side of the minimum converges only linearly.
_INNER_ITERATIONS = 8

#: Largest per-coefficient inner step that counts as converged, radians.
_INNER_TOL = 1e-4


# ----------------------------------------------------------------------
# Posterior
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PosteriorEstimate:
    """A coefficient estimate and the covariance that says how sure it is.

    Attributes:
        coeffs: ``(n_modes,)`` coefficient estimates, radians, Noll
            ``2 .. n_modes + 1``.
        cov: ``(n_modes, n_modes)`` covariance of ``coeffs``, rad^2.
        sigma2: Estimated noise variance of one residual pixel (in the prepared
            frame's units), after the fit.
        rank: Rank of the (weighted) normal matrix ``J^T W J``.
        condition: Condition number of that normal matrix; ``inf`` when nothing
            has been measured.
        n_measurements: Number of probe pushes behind this estimate (two frames
            per probe).
        gram: ``(n_modes, n_modes)`` matrix ``J^T W J`` of the *unit-amplitude*
            response at the operating point the last probe was taken at.  This
            is what :func:`select_probe` needs to price a candidate probe; it is
            ``None`` only for a prior-only estimate built without a model.
    """

    coeffs: np.ndarray
    cov: np.ndarray
    sigma2: float
    rank: int
    condition: float
    n_measurements: int
    gram: np.ndarray | None = None

    def __post_init__(self) -> None:
        coeffs = np.asarray(self.coeffs, dtype=np.float64).reshape(-1)
        cov = np.asarray(self.cov, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square, got shape {cov.shape}")
        if cov.shape[0] != coeffs.size:
            raise ValueError(
                f"cov is {cov.shape[0]}x{cov.shape[0]} but there are "
                f"{coeffs.size} coefficients"
            )
        object.__setattr__(self, "coeffs", coeffs)
        object.__setattr__(self, "cov", cov)

    @property
    def n_modes(self) -> int:
        """Coefficient count (Noll ``2 .. n_modes + 1``)."""
        return int(self.coeffs.size)

    def confidence(self) -> np.ndarray:
        """Per-mode standard deviation, radians (``sqrt(diag(cov))``)."""
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    def trace(self) -> float:
        """``trace(cov)`` -- the D-optimality objective this module minimises."""
        return float(np.trace(self.cov))


def linear_posterior(
    jacobian: np.ndarray | torch.Tensor,
    residual: np.ndarray | torch.Tensor,
    *,
    weights: np.ndarray | torch.Tensor | None = None,
    prior_coeffs: np.ndarray | torch.Tensor | None = None,
    prior_cov: np.ndarray | torch.Tensor | None = None,
    n_measurements: int = 0,
    rcond: float = 1e-6,
) -> PosteriorEstimate:
    """Weighted least-squares posterior from a stacked design and residual.

    Solves ``min_c (r - J c)^T W (r - J c)``.  Without a prior the answer is the
    plain weighted LS solution and its covariance is the textbook
    ``sigma2 * pinv(J^T W J)``.  With ``prior_cov`` the Gaussian posterior is
    returned instead::

        cov = inv(inv(P) + J^T W J / sigma2)
        c   = cov (inv(P) mu0 + J^T W r / sigma2)

    The prior is what makes the *first* probe selectable: with no measurement the
    normal matrix is zero, ``pinv`` of it is zero, and a zero covariance would
    claim every mode is known exactly.  :func:`probe_active` passes the trust
    region as the prior, which is the documented bound on a coefficient anyway.

    ``sigma2`` is always estimated from the *unregularised* LS residual, so the
    prior cannot shrink the noise estimate it is divided by.

    Args:
        jacobian: ``(n_pixels, n_modes)`` unit-amplitude response of the prepared
            prediction.
        residual: ``(n_pixels,)`` measured-minus-predicted residual, same units.
        weights: Optional ``(n_pixels,)`` relative precisions (inverse variances
            up to a common scale).  ``None`` means uniform.
        prior_coeffs: Optional ``(n_modes,)`` prior mean; zero when omitted.
        prior_cov: Optional ``(n_modes, n_modes)`` prior covariance.
        n_measurements: Probe pushes behind ``jacobian`` / ``residual``.
        rcond: Relative singular-value cut of ``pinv``.

    Returns:
        The :class:`PosteriorEstimate`; its ``gram`` is ``J^T W J``.

    Raises:
        ValueError: If the shapes do not line up, a weight is negative, or the
            prior is not a matching square matrix.
    """
    design = np.asarray(
        jacobian.detach().cpu() if isinstance(jacobian, torch.Tensor) else jacobian,
        dtype=np.float64,
    )
    if design.ndim != 2:
        raise ValueError(
            f"jacobian must be 2-D (pixels, modes), got shape {design.shape}"
        )
    values = np.asarray(
        residual.detach().cpu() if isinstance(residual, torch.Tensor) else residual,
        dtype=np.float64,
    ).reshape(-1)
    if values.size != design.shape[0]:
        raise ValueError(
            f"residual has {values.size} pixels but jacobian has {design.shape[0]} rows"
        )
    n_pixels, n_modes = design.shape

    if weights is None:
        weight = np.ones(n_pixels, dtype=np.float64)
    else:
        weight = np.asarray(
            weights.detach().cpu() if isinstance(weights, torch.Tensor) else weights,
            dtype=np.float64,
        ).reshape(-1)
        if weight.size != n_pixels:
            raise ValueError(
                f"weights has {weight.size} entries but jacobian has {n_pixels} rows"
            )
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("weights must be finite and non-negative")

    normal = design.T @ (weight[:, None] * design)
    moment = design.T @ (weight * values)
    normal_inv = np.linalg.pinv(normal, rcond=rcond, hermitian=True)
    rank = int(np.linalg.matrix_rank(normal))
    condition = float(np.linalg.cond(normal)) if rank > 0 else float("inf")

    coeffs = normal_inv @ moment
    misfit = values - design @ coeffs
    dof = max(n_pixels - rank, 1)
    sigma2 = float(np.sum(weight * misfit**2) / dof)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        sigma2 = _SIGMA_FLOOR

    if prior_cov is None:
        cov = sigma2 * normal_inv
    else:
        prior = np.asarray(
            prior_cov.detach().cpu()
            if isinstance(prior_cov, torch.Tensor)
            else prior_cov,
            dtype=np.float64,
        )
        if prior.shape != (n_modes, n_modes):
            raise ValueError(
                f"prior_cov must be ({n_modes}, {n_modes}), got shape {prior.shape}"
            )
        mean = np.zeros(n_modes, dtype=np.float64)
        if prior_coeffs is not None:
            mean = np.asarray(
                prior_coeffs.detach().cpu()
                if isinstance(prior_coeffs, torch.Tensor)
                else prior_coeffs,
                dtype=np.float64,
            ).reshape(-1)
            if mean.size != n_modes:
                raise ValueError(
                    f"prior_coeffs has {mean.size} entries, expected {n_modes}"
                )
        precision = np.linalg.inv(prior)
        cov = np.linalg.inv(precision + normal / sigma2)
        coeffs = cov @ (precision @ mean + moment / sigma2)

    return PosteriorEstimate(
        coeffs=coeffs,
        cov=cov,
        sigma2=sigma2,
        rank=rank,
        condition=condition,
        n_measurements=int(n_measurements),
        gram=normal,
    )


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
def _response_norms(
    nominal: object,
    fit_cfg: FitConfig,
    modes: Sequence[int],
    amplitude: float,
    *,
    difference_rad: float,
) -> np.ndarray:
    """Per-candidate response strength ``||J[:, j]||`` of a fresh probe.

    The gram carried by a posterior only contains columns of blocks that were
    already taken, and those blocks were taken at *other* modes' pushes.  A mode
    never probed therefore has a zero diagonal there -- even though probing it
    would produce a strong column.  Intensity is even in every single-mode
    coefficient, so this is not a corner case: defocus (Noll 4) has no linear
    response at zero phase and would otherwise stay invisible forever.

    This evaluates, for every candidate, the response column at its own
    ``+/- a Z_j`` push and returns the RMS norm of the two columns.

    Args:
        nominal: Aberration-free bench or windowed bench.
        fit_cfg: Supplies ``n_modes``.
        modes: Candidate Noll indices.
        amplitude: Push amplitude, radians.
        difference_rad: Half-step of the central difference, radians.

    Returns:
        ``(len(modes),)`` float64 response norms.
    """
    zero = _as_phase(nominal, None)
    norms = np.zeros(len(modes), dtype=np.float64)
    for index, mode in enumerate(modes):
        zmap = _zernike_map(nominal, fit_cfg, mode)
        plus, minus = _design_pair(
            nominal,
            zero,
            zmap,
            amplitude,
            fit_cfg=fit_cfg,
            difference_rad=difference_rad,
        )
        column = int(mode) - 2
        norms[index] = float(
            np.sqrt(
                (np.sum(plus[:, column] ** 2) + np.sum(minus[:, column] ** 2)) / 2.0
            )
        )
    return norms


def select_probe(
    posterior: PosteriorEstimate,
    candidates: Sequence[int],
    *,
    amplitude_rad: float,
    response_norm: Sequence[float] | None = None,
) -> int:
    """Greedy D-optimality: the candidate whose probe shrinks ``trace(cov)`` most.

    Probing mode ``j`` with amplitude ``a`` adds a design column whose strength
    is ``||J[:, j]||``; its coefficient-space information is
    ``q_j = a^2 ||J[:, j]||^2 / sigma2``.  Under the marginal-variance (diagonal)
    approximation the trace reduction of that one probe is::

        d trace = var_j^2 q_j / (1 + var_j q_j)

    which is evaluated for every candidate and maximised.  The gain is divided by
    the current ``trace(cov)`` so it is a *fractional* reduction: ``cov`` scales
    with the estimated noise, and on a clean bench raw gains collapse into
    round-off while their ratios stay meaningful.  Ties go to the smallest Noll
    index, so the choice is deterministic for a given posterior.

    Args:
        posterior: Current estimate (supplies ``cov``, ``sigma2`` and ``gram``).
        candidates: Noll indices still worth probing.
        amplitude_rad: Push amplitude the probe would use, radians.
        response_norm: Per-candidate ``||J[:, j]||`` of a *fresh* probe, in the
            order of ``candidates``.  Defaults to ``sqrt(diag(gram))``, which is
            only correct for modes already probed; pass it explicitly when the
            candidate set contains modes whose response was never measured.

    Returns:
        The selected Noll index.

    Raises:
        ValueError: On an empty or duplicated candidate list, a Noll index
            outside the coefficient vector, a non-positive amplitude, a
            mismatched ``response_norm``, or a posterior that carries no
            ``gram``.
    """
    modes = [int(mode) for mode in candidates]
    if not modes:
        raise ValueError("candidates must not be empty")
    if len(set(modes)) != len(modes):
        duplicates = sorted({mode for mode in modes if modes.count(mode) > 1})
        raise ValueError(f"duplicate candidate Noll index/indices {duplicates}")
    amplitude = float(amplitude_rad)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError(
            f"amplitude_rad must be positive and finite, got {amplitude_rad!r}"
        )
    if posterior.gram is None:
        raise ValueError("posterior carries no response gram; cannot price a probe")

    n_modes = posterior.n_modes
    for mode in modes:
        if not 2 <= mode <= n_modes + 1:
            raise ValueError(
                f"Noll {mode} is outside the coefficient vector 2..{n_modes + 1}"
            )

    if response_norm is None:
        norms = np.sqrt(np.clip(np.diag(posterior.gram), 0.0, None))
    else:
        norms = np.asarray(response_norm, dtype=np.float64).reshape(-1)
        if norms.size != len(modes):
            raise ValueError(
                f"response_norm has {norms.size} entries but candidates has {len(modes)}"
            )
        if not np.all(np.isfinite(norms)) or np.any(norms < 0.0):
            raise ValueError("response_norm must be finite and non-negative")

    covariance = np.asarray(posterior.cov, dtype=np.float64)
    sigma2 = float(posterior.sigma2)
    # Normalise the gain by the current trace.  On a low-noise bench ``cov`` is
    # proportional to the estimated ``sigma2``, so raw gains collapse toward zero
    # and their ordering becomes pure round-off; the *fractional* trace reduction
    # is scale-free and keeps the ranking meaningful as the noise vanishes.
    scale = float(np.trace(covariance))
    best_mode = -1
    best_gain = -np.inf
    for index, mode in enumerate(modes):
        variance = float(covariance[mode - 2, mode - 2])
        information = amplitude**2 * float(norms[index]) ** 2 / sigma2
        denominator = 1.0 + variance * information
        gain = (
            variance**2 * information / denominator
            if denominator > _DENOM_FLOOR
            else 0.0
        )
        if scale > _DENOM_FLOOR:
            gain /= scale
        # Ties (equal gain, e.g. equal variances and equal response) go to the
        # smaller Noll index, so the answer does not depend on the order the
        # caller happened to list the candidates in.
        tolerance = 1e-12 * max(1.0, abs(best_gain)) if np.isfinite(best_gain) else 0.0
        if gain > best_gain + tolerance or (
            np.isfinite(best_gain)
            and abs(gain - best_gain) <= tolerance
            and mode < best_mode
        ):
            best_mode, best_gain = mode, gain

    if best_gain > 0.0:
        return best_mode

    # No candidate adds information (a blind design).  Fall back to the mode the
    # posterior is least sure about, which is still a defensible next probe and
    # keeps the choice deterministic.
    confidence = posterior.confidence()
    ranked = sorted(modes, key=lambda mode: (-float(confidence[mode - 2]), mode))
    return int(ranked[0])


# ----------------------------------------------------------------------
# Probe loop
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ActiveProbeResult:
    """Outcome of one :func:`probe_active` run.

    Attributes:
        coeffs: ``(n_modes,)`` accumulated coefficient estimate, radians.
        estimates: One cumulative :class:`PosteriorEstimate` per probe, in probe
            order.  ``estimates[-1]`` is ``final``.
        selected_modes: Noll index pushed at each step, in order.
        final: The last cumulative estimate (the one to use).
        history: Per-step records (mode, amplitude, trace, sigma2, rank, the
            increment, ...) for a report or a figure.
        amplitudes: Push amplitude actually used at each step, radians.
    """

    coeffs: np.ndarray
    estimates: tuple[PosteriorEstimate, ...]
    selected_modes: tuple[int, ...]
    final: PosteriorEstimate
    history: tuple[dict, ...]
    amplitudes: tuple[float, ...]


def _zernike_map(bench: object, fit_cfg: FitConfig, noll: int) -> torch.Tensor:
    """``(grid, grid)`` phase map of one unit-amplitude Noll mode."""
    return bench.aberration(_unit_coefficient(fit_cfg, noll)[None])[0]  # type: ignore[attr-defined]


def _unit_response(
    bench: object,
    phase: torch.Tensor,
    noll: Sequence[int],
    *,
    fit_cfg: FitConfig,
    difference_rad: float,
) -> np.ndarray:
    """Central-difference response of the prepared prediction, per mode.

    One batched forward for all ``2 * len(noll)`` pushes: the bench is a
    differentiable Fourier model, so this is a single FFT batch rather than a
    Python-level loop over modes.

    Args:
        bench: The nominal (aberration-free) bench or windowed bench.
        phase: ``(1, grid, grid)`` phase the response is taken at.
        noll: Noll indices of the columns.
        fit_cfg: Supplies ``n_modes`` for the unit vectors.
        difference_rad: Half-step of the central difference, radians.

    Returns:
        ``(n_pixels, len(noll))`` float64 response, columns at unit amplitude.
    """
    maps = [_zernike_map(bench, fit_cfg, mode) for mode in noll]
    with torch.no_grad():
        plus = torch.stack([phase[0] + difference_rad * zmap for zmap in maps])
        minus = torch.stack([phase[0] - difference_rad * zmap for zmap in maps])
        frames = _prepare(bench.forward(torch.cat([plus, minus]), detector=False))  # type: ignore[attr-defined]
        count = len(maps)
        columns = (frames[:count] - frames[count:]) / (2.0 * difference_rad)
    return columns.reshape(count, -1).T.detach().cpu().numpy().astype(np.float64)


def _prior_posterior(gram: np.ndarray, prior_rad: float) -> PosteriorEstimate:
    """Prior-only estimate: isotropic variance, zero mean, no measurement yet."""
    n_modes = int(gram.shape[0])
    return PosteriorEstimate(
        coeffs=np.zeros(n_modes, dtype=np.float64),
        cov=np.eye(n_modes, dtype=np.float64) * float(prior_rad) ** 2,
        sigma2=1.0,
        rank=0,
        condition=float("inf"),
        n_measurements=0,
        gram=gram,
    )


def _design_pair(
    nominal: object,
    phase: torch.Tensor,
    zmap: torch.Tensor,
    amplitude: float,
    *,
    fit_cfg: FitConfig,
    difference_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Model design blocks at the two push phases of one probe.

    The response is taken at the phase that is *actually* on the panel for that
    push (``command +/- a Z_j``), not at the bare command: the frame is quadratic
    in the phase, so a column measured a few tenths of a radian away from where
    the residual was measured is a stale design.

    Args:
        nominal: Aberration-free bench or windowed bench.
        phase: ``(1, grid, grid)`` compensation command.
        zmap: ``(grid, grid)`` unit map of the probed mode.
        amplitude: Push amplitude, radians.
        fit_cfg: Supplies ``n_modes``.
        difference_rad: Half-step of the central difference.

    Returns:
        ``(design_plus, design_minus)``, each ``(n_pixels, n_modes)``.
    """
    noll = fit_cfg.coefficient_noll()
    maps = [_zernike_map(nominal, fit_cfg, mode) for mode in noll]
    count = len(maps)
    with torch.no_grad():
        bases = [phase[0] + amplitude * zmap, phase[0] - amplitude * zmap]
        pushes = torch.stack(
            [
                base + sign * difference_rad * zmap_mode
                for base in bases
                for zmap_mode in maps
                for sign in (1.0, -1.0)
            ]
        )
        frames = _prepare(nominal.forward(pushes, detector=False))  # type: ignore[attr-defined]
        frames = frames.reshape(2, count, 2, *frames.shape[1:])
        columns = (frames[:, :, 0] - frames[:, :, 1]) / (2.0 * difference_rad)
    array = columns.reshape(2, count, -1).permute(0, 2, 1).detach().cpu().numpy()
    return array[0].astype(np.float64), array[1].astype(np.float64)


def _measure_pair(
    bench: object,
    measure: Measurement,
    command: torch.Tensor,
    zmap: torch.Tensor,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure one ``+/- a Z_j`` pair and return the two prepared frames.

    The raw frames are kept (not differenced) because the estimator is
    re-linearised at the running estimate: the model prediction a frame must be
    compared against depends on the estimate at *fit* time, not at measurement
    time.  Both pushes go out in one batched forward: one exposure each is the
    cost, so batching only removes host-side overhead.

    Returns:
        ``(frame_plus, frame_minus)``, each ``(n_pixels,)`` float64.
    """
    pushes = torch.stack([amplitude * zmap, -amplitude * zmap])
    with torch.no_grad():
        measured = _frame_on(bench, frame_of(measure(command + pushes)))
    prepared = (
        _prepare(measured).reshape(2, -1).detach().cpu().numpy().astype(np.float64)
    )
    return prepared[0], prepared[1]


@dataclass
class _Observation:
    """One measured push, with the compensation command in force at the time."""

    estimate: np.ndarray
    push: torch.Tensor
    frame: np.ndarray


#: Observations per batched model forward while re-linearising the stack.  Each
#: observation contributes ``1 + 2 * n_modes`` phase maps, and the padded FFT is
#: ``grid * far_field_padding`` square, so this bounds the transient tensor.
_STACK_BLOCK = 4


def _stacked_system(
    nominal: object,
    observations: Sequence[_Observation],
    estimate: np.ndarray,
    *,
    fit_cfg: FitConfig,
    difference_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-linearise every stored measurement at ``estimate``.

    For observation ``k`` the panel carried ``command_k + truth + push_k`` with
    ``command_k = -estimate_k``; comparing against the model at
    ``command_k + estimate + push_k`` leaves the Gauss-Newton residual
    ``J (truth - estimate)``, with ``J`` the response at that same phase.  Every
    block therefore constrains the *same* increment, so they stack without the
    quadratic bias a fixed linearisation accumulates.

    The blocks are built in chunks of ``_STACK_BLOCK`` observations and handed to
    the bench as one batched forward per chunk: the whole stack is one FFT batch
    per chunk instead of one call per observation, which is what makes the inner
    Gauss-Newton loop affordable.

    Returns:
        ``(design, residual)``, shapes ``(n_blocks * n_pixels, n_modes)`` and
        ``(n_blocks * n_pixels,)``.
    """
    noll = fit_cfg.coefficient_noll()
    maps = [_zernike_map(nominal, fit_cfg, mode) for mode in noll]
    count = len(maps)
    designs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for start in range(0, len(observations), _STACK_BLOCK):
        chunk = observations[start : start + _STACK_BLOCK]
        panel: list[torch.Tensor] = []
        for observation in chunk:
            offset = estimate - observation.estimate
            base = observation.push
            if np.any(offset):
                base = nominal.aberration(offset[None])[0] + observation.push  # type: ignore[attr-defined]
            panel.append(base)
        stacked = torch.stack(panel)
        with torch.no_grad():
            predicted = _prepare(nominal.forward(stacked, detector=False)).reshape(  # type: ignore[attr-defined]
                len(chunk), -1
            )
            shifted = torch.stack(
                [
                    torch.stack(
                        [p + difference_rad * zmap for zmap in maps]
                        + [p - difference_rad * zmap for zmap in maps]
                    )
                    for p in panel
                ]
            )
            frames = _prepare(  # type: ignore[attr-defined]
                nominal.forward(
                    shifted.reshape(-1, stacked.shape[1], stacked.shape[2]),
                    detector=False,
                )
            ).reshape(len(chunk), 2 * count, -1)
        for index, observation in enumerate(chunk):
            prediction = predicted[index]
            frame = torch.as_tensor(
                observation.frame, device=prediction.device, dtype=prediction.dtype
            )
            denominator = float(prediction.pow(2).sum())
            alpha = (
                float((prediction * frame).sum()) / denominator
                if denominator > 0.0
                else 1.0
            )
            if not np.isfinite(alpha) or alpha <= 0.0:
                alpha = 1.0
            residual = (frame / alpha - prediction).detach().cpu().numpy()
            columns = (frames[index, :count] - frames[index, count:]) / (
                2.0 * difference_rad
            )
            designs.append(columns.T.detach().cpu().numpy().astype(np.float64))
            residuals.append(residual.astype(np.float64))
    return np.concatenate(designs, axis=0), np.concatenate(residuals, axis=0)


def probe_active(
    bench: object,
    cfg: AberrationConfig | FitConfig,
    *,
    measure: Measurement,
    model: object | None = None,
    budget: int,
    candidates: Sequence[int] | None = None,
    command: torch.Tensor | np.ndarray | None = None,
    fit_cfg: FitConfig | None = None,
    order: Sequence[int] | None = None,
    prior_rad: float | None = None,
    amplitude_rad: float | None = None,
    difference_rad: float = 0.05,
    snr_target: float = 2.0,
) -> ActiveProbeResult:
    """Probe with a posterior-driven order (or a given order) for ``budget`` steps.

    Each step:

    1. pick a mode -- ``select_probe`` on the current posterior, or ``order[k]``
       when an order is supplied (that is how the fixed baseline is run through
       the identical code path);
    2. pick the amplitude -- the configured one, raised (never lowered, at most
       1.5x) when ``amplitude / confidence(mode) < snr_target``;
    3. push ``+/- a Z_m`` on top of the current compensation command, measure
       both, and keep the frames with the estimate in force;
    4. re-solve the stacked weighted LS on every stored frame, re-linearised at
       the running estimate, with the trust region as a zero-mean prior; iterate
       that solve to convergence (model forwards are free, camera frames are not)
       and move the estimate by the clipped total increment.

    The compensation command is ``-sum_j c_j Z_j`` for the current estimate, so
    the device sees ``truth - estimate + push`` and the residual keeps pointing
    at what is still uncorrected.

    Args:
        bench: Device bench; supplies ``grid`` / ``dtype`` / ``device`` and is the
            target of ``measure``.  Never differentiated.
        cfg: Probe contract, or a :class:`~ml.sim2f.fit.FitConfig` directly.
        measure: ``phase -> frame`` for the device.
        model: Nominal (aberration-free) bench used for the prediction and the
            response.  Defaults to ``bench``, which is right when the bench *is*
            the ideal model (the unit tests).
        budget: Number of probe pushes.
        candidates: Noll indices eligible for selection.  Defaults to every
            coefficient, Noll ``2 .. n_modes + 1``.
        command: Initial command phase; zeros when omitted.
        fit_cfg: Explicit probe/fit budget.
        order: Fixed probe order.  When given, selection is bypassed -- the
            baseline strategy.  A mode may appear twice in ``order``: a repeat is
            a fresh noise realisation at a new operating point, which is a weak
            but legitimate probe, and it is what makes a budget larger than the
            candidate set possible.
        prior_rad: Prior standard deviation of a coefficient, radians.  Defaults
            to ``fit_cfg.trust_rad``.
        amplitude_rad: Base push amplitude; defaults to
            ``fit_cfg.probe_amplitude_rad``.
        difference_rad: Half-step of the model response difference, radians.
        snr_target: Predicted per-mode SNR below which the push is enlarged.

    Returns:
        The :class:`ActiveProbeResult`.

    Raises:
        ValueError: On a non-positive budget/amplitude/prior/step, a duplicated
            candidate, or an order that is too short or names a mode outside
            ``candidates``.
    """
    fit_cfg = fit_cfg or (
        cfg if isinstance(cfg, FitConfig) else FitConfig.from_aberration(cfg)
    )
    n_modes = int(fit_cfg.n_modes)
    nominal = bench if model is None else model

    budget = int(budget)
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    base_amplitude = float(
        fit_cfg.probe_amplitude_rad if amplitude_rad is None else amplitude_rad
    )
    if not np.isfinite(base_amplitude) or base_amplitude <= 0.0:
        raise ValueError(
            f"amplitude_rad must be positive and finite, got {base_amplitude!r}"
        )
    prior = float(fit_cfg.trust_rad if prior_rad is None else prior_rad)
    if not np.isfinite(prior) or prior <= 0.0:
        raise ValueError(f"prior_rad must be positive and finite, got {prior!r}")
    if not np.isfinite(difference_rad) or difference_rad <= 0.0:
        raise ValueError(
            f"difference_rad must be positive and finite, got {difference_rad!r}"
        )

    pool = tuple(
        int(mode)
        for mode in (fit_cfg.coefficient_noll() if candidates is None else candidates)
    )
    if len(set(pool)) != len(pool):
        raise ValueError("candidates contains a duplicate Noll index")
    for mode in pool:
        if not 2 <= mode <= n_modes + 1:
            raise ValueError(
                f"Noll {mode} is outside the coefficient vector 2..{n_modes + 1}"
            )

    fixed: tuple[int, ...] | None = None
    if order is not None:
        fixed = tuple(int(mode) for mode in order)
        if len(fixed) < budget:
            raise ValueError(f"order has {len(fixed)} modes but budget is {budget}")
        outside = [mode for mode in fixed[:budget] if mode not in pool]
        if outside:
            raise ValueError(f"order names modes outside candidates: {outside}")

    phase = _as_phase(bench, command)
    estimate = np.zeros(n_modes, dtype=np.float64)
    prior_cov = np.eye(n_modes, dtype=np.float64) * prior**2
    # Strength of a *fresh* probe of each candidate.  The posterior gram only
    # knows the columns of blocks already taken, which is zero for a mode whose
    # response is quadratic at the current operating point (defocus, spherical).
    response_norm = (
        _response_norms(
            nominal,
            fit_cfg,
            pool,
            base_amplitude,
            difference_rad=difference_rad,
        )
        if fixed is None
        else None
    )
    observations: list[_Observation] = []
    posterior: PosteriorEstimate | None = None
    selected: list[int] = []
    amplitudes: list[float] = []
    estimates: list[PosteriorEstimate] = []
    history: list[dict] = []

    for step in range(budget):
        if fixed is not None:
            mode = fixed[step]
        else:
            if posterior is None:
                unit = _unit_response(
                    nominal,
                    phase,
                    fit_cfg.coefficient_noll(),
                    fit_cfg=fit_cfg,
                    difference_rad=difference_rad,
                )
                posterior = _prior_posterior(unit.T @ unit, prior)
            mode = select_probe(
                posterior,
                pool,
                amplitude_rad=base_amplitude,
                response_norm=response_norm,
            )

        amplitude = base_amplitude
        if posterior is not None:
            confidence = float(posterior.confidence()[mode - 2])
            snr = amplitude / max(confidence, _DENOM_FLOOR)
            if snr < snr_target:
                grown = amplitude * min(snr_target / max(snr, _DENOM_FLOOR), 1.5)
                amplitude = min(max(grown, 0.5 * base_amplitude), 1.5 * base_amplitude)

        zmap = _zernike_map(nominal, fit_cfg, mode)
        # Measure the pair at the current compensation command and *keep* the
        # frames.  What each frame is compared against is decided at fit time,
        # not at measurement time: the estimator re-linearises every stored
        # frame at the running estimate, so a block taken when the estimate was
        # stale still constrains the same increment.
        frame_plus, frame_minus = _measure_pair(bench, measure, phase, zmap, amplitude)
        observations.append(
            _Observation(
                estimate=estimate.copy(),
                push=amplitude * zmap,
                frame=frame_plus,
            )
        )
        observations.append(
            _Observation(
                estimate=estimate.copy(),
                push=-amplitude * zmap,
                frame=frame_minus,
            )
        )

        design, residual = _stacked_system(
            nominal,
            observations,
            estimate,
            fit_cfg=fit_cfg,
            difference_rad=difference_rad,
        )
        # ``residual`` is ``J (truth - estimate)``, so the unknown of the solve is
        # the *increment*, with the trust region as a zero-mean prior.  A single
        # linearisation only steps to the minimum of a quadratic *model* of a
        # response that is itself quadratic in the coefficient, which converges
        # linearly and wastes probes.  Re-solving the stored frames costs model
        # forwards, not camera frames, so iterate to the fixed point here.
        increment = np.zeros(n_modes, dtype=np.float64)
        for _ in range(_INNER_ITERATIONS):
            increment_posterior = linear_posterior(
                design,
                residual,
                prior_coeffs=np.zeros(n_modes, dtype=np.float64),
                prior_cov=prior_cov,
                n_measurements=2 * (step + 1),
                rcond=float(fit_cfg.rcond),
            )
            delta = np.clip(
                increment_posterior.coeffs,
                -float(fit_cfg.trust_rad),
                float(fit_cfg.trust_rad),
            )
            estimate = estimate + delta
            increment = increment + delta
            # Re-linearise at the new estimate *before* testing convergence, so
            # ``residual`` in the history is the misfit at the reported estimate.
            design, residual = _stacked_system(
                nominal,
                observations,
                estimate,
                fit_cfg=fit_cfg,
                difference_rad=difference_rad,
            )
            if float(np.max(np.abs(delta))) < _INNER_TOL:
                break

        selected.append(mode)
        amplitudes.append(amplitude)
        cumulative = dataclasses.replace(increment_posterior, coeffs=estimate.copy())
        estimates.append(cumulative)
        posterior = cumulative
        history.append(
            {
                "step": step,
                "noll": mode,
                "amplitude_rad": float(amplitude),
                "increment": [float(value) for value in increment],
                "coeffs": [float(value) for value in estimate],
                "trace": cumulative.trace(),
                "max_confidence": float(np.max(cumulative.confidence())),
                "sigma2": float(cumulative.sigma2),
                "rank": int(cumulative.rank),
                "condition": float(cumulative.condition),
                "residual_rms": float(np.sqrt(np.mean(residual**2))),
            }
        )

        phase = _as_phase(bench, nominal.aberration((-estimate)[None]))  # type: ignore[attr-defined]

    return ActiveProbeResult(
        coeffs=estimate,
        estimates=tuple(estimates),
        selected_modes=tuple(selected),
        final=estimates[-1],
        history=tuple(history),
        amplitudes=tuple(amplitudes),
    )


# ----------------------------------------------------------------------
# Strategy comparison
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyOutcome:
    """One strategy at one budget, ready for a table row.

    Attributes:
        strategy: ``"fixed"`` or ``"active"``.
        budget: Probe pushes spent.
        coeffs: ``(n_modes,)`` estimate, radians.
        selected_modes: The modes actually pushed, in order.
        amplitudes: Push amplitude per step, radians.
        trace: ``trace(cov)`` of the final estimate.
        condition: Condition number of the stacked normal matrix.
        rank: Rank of that matrix.
        sigma2: Estimated residual-pixel variance.
        confidence: ``(n_modes,)`` per-mode standard deviation.
    """

    strategy: str
    budget: int
    coeffs: np.ndarray
    selected_modes: tuple[int, ...]
    amplitudes: tuple[float, ...]
    trace: float
    condition: float
    rank: int
    sigma2: float
    confidence: np.ndarray


def _outcome(strategy: str, budget: int, run: ActiveProbeResult) -> StrategyOutcome:
    """Slice one prefix of a longer run into a :class:`StrategyOutcome`."""
    estimate = run.estimates[budget - 1]
    return StrategyOutcome(
        strategy=strategy,
        budget=int(budget),
        coeffs=np.asarray(estimate.coeffs, dtype=np.float64).copy(),
        selected_modes=tuple(run.selected_modes[:budget]),
        amplitudes=tuple(run.amplitudes[:budget]),
        trace=estimate.trace(),
        condition=float(estimate.condition),
        rank=int(estimate.rank),
        sigma2=float(estimate.sigma2),
        confidence=estimate.confidence(),
    )


def compare_probe_strategies(
    bench: object,
    cfg: AberrationConfig | FitConfig,
    *,
    measure: Measurement,
    budgets: Sequence[int],
    model: object | None = None,
    candidates: Sequence[int] | None = None,
    command: torch.Tensor | np.ndarray | None = None,
    fit_cfg: FitConfig | None = None,
    fixed_order: Sequence[int] | None = None,
    prior_rad: float | None = None,
    amplitude_rad: float | None = None,
    difference_rad: float = 0.05,
    snr_target: float = 2.0,
) -> dict[str, tuple[StrategyOutcome, ...]]:
    """Run the fixed baseline and the active selector on one bench state.

    Both strategies run to ``max(budgets)`` **once** and every smaller budget is
    read off as a prefix: an active selection at step ``k`` depends only on the
    measurements before it, never on the total budget, so the prefixes are the
    same runs a shorter budget would have produced -- and the device is touched
    once per step instead of once per (budget, step).

    Args:
        bench: Device bench.
        cfg: Probe contract or :class:`~ml.sim2f.fit.FitConfig`.
        measure: ``phase -> frame`` for the device.
        budgets: Budgets to report, probe pushes.
        model: Nominal bench for the prediction and response.
        candidates: Noll indices eligible for selection.
        command: Initial command phase.
        fit_cfg: Explicit probe/fit budget.
        fixed_order: Baseline order; defaults to :func:`ml.sim2f.fit.probe_noll`.
        prior_rad: Prior standard deviation, radians.
        amplitude_rad: Base push amplitude, radians.
        difference_rad: Half-step of the model response difference.
        snr_target: Predicted per-mode SNR below which the push is enlarged.

    Returns:
        ``{"fixed": (...), "active": (...)}``, one outcome per budget in the
        order given.

    Raises:
        ValueError: On an empty budget list or a non-positive budget.
    """
    requested = [int(value) for value in budgets]
    if not requested:
        raise ValueError("budgets must not be empty")
    if min(requested) < 1:
        raise ValueError(f"budgets must be positive, got {requested}")
    horizon = max(requested)

    resolved = fit_cfg or (
        cfg if isinstance(cfg, FitConfig) else FitConfig.from_aberration(cfg)
    )
    if fixed_order is None:
        bank = tuple(int(mode) for mode in probe_noll(resolved))
        # The default bank is shorter than a large budget, and a budget larger
        # than the bank means a mode is probed again (see the module docstring).
        # Cycling the Noll order is the honest baseline: it spends the extra
        # frames exactly where the fixed strategy would have spent them.
        repeats = -(-horizon // len(bank))
        order = (bank * repeats)[:horizon]
    else:
        order = tuple(int(mode) for mode in fixed_order)
        if len(order) < horizon:
            raise ValueError(
                f"fixed order has {len(order)} modes but the largest budget is {horizon}"
            )

    common = dict(
        measure=measure,
        model=model,
        candidates=candidates,
        command=command,
        fit_cfg=resolved,
        prior_rad=prior_rad,
        amplitude_rad=amplitude_rad,
        difference_rad=difference_rad,
        snr_target=snr_target,
    )
    fixed_run = probe_active(bench, resolved, budget=horizon, order=order, **common)
    active_run = probe_active(bench, resolved, budget=horizon, **common)
    return {
        "fixed": tuple(_outcome("fixed", budget, fixed_run) for budget in requested),
        "active": tuple(_outcome("active", budget, active_run) for budget in requested),
    }
