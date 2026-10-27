"""Piston-free phasor loss, physics objective and shaping metrics.

Two independent legs are needed, and this module is where both are defined so
that the trainer, the evaluator and the teacher all report the *same* numbers:

* :func:`piston_free_phasor_l1` -- supervised distance to the teacher's phase.
  The far field loses the absolute phase, so a whole one-parameter family of
  phases (any global piston) produces the same image.  The loss therefore
  rotates the prediction by the closed-form optimal global angle before taking
  the L1, which removes exactly the unobservable direction instead of asking
  the network to guess it.
* :func:`physics_loss` -- the *teacher's own objective*, reproduced term by
  term: ``L = L_shape + w_ee * (1 - EE)`` on unit-sum, background-subtracted
  intensities.  This is the number the "amortisation gap" is measured with, so
  it may not differ from :mod:`ml.sim2f.teacher` by even a normalisation
  constant.  The composition lives in :func:`objective_terms` and the teacher
  and this module both call it.

The bench this scores is **not ideal**
--------------------------------------
``Sim2f`` is the repository's calibrated 2f twin, so a frame carries a zero-order
background term, camera defocus, PRNU / dark FPN, 8-bit quantisation and
Poisson/read noise.  Two consequences are structural here, not cosmetic:

1. ``I.sum()`` is **not** the modulated energy -- it also contains the
   zero-order background and the dark offset.  *Prepared* frames therefore have
   their median subtracted and their negatives clipped before any ratio is
   formed (:func:`prepare_frame`).  Without that, a symmetric read noise leaves
   roughly half the pixels negative, so the "fraction inside the support" of a
   *prepared* frame can exceed 1 (the repository measured ``PIB = 1.0120`` on a
   raw simulated frame) and the optimiser chases read noise.
2. The clipping must happen **after** the subtraction, never on the raw frame:
   clipping a raw frame rectifies the noise into a pixel-count-sized DC pedestal,
   which is the opposite of removing it.

Every scoring function here takes ``prepare`` so the ablation can compare the
prepared and raw objectives explicitly instead of by accident; ``True`` (the
default) is the one the teacher uses.

Why the phasor form
-------------------
A wrapped angle has a branch cut at ``+/- pi``: two nearly identical phases that
straddle the cut look maximally different, and ``atan2`` puts a discontinuity
there in the *gradient* as well.  ``(cos, sin)`` has no cut, so the network can
move continuously between any two phases.  Every function here that has a
choice accepts the phasor and derives the angle only where it is required.

Shape conventions
-----------------
Phasors are ``(B, 2, H, W)`` (channel 0 real, channel 1 imaginary).  Phase maps
are ``(B, H, W)`` raw unwrapped radians -- nothing in this module wraps a phase;
the only ``mod 2*pi`` in this repository lives in the SLM driver.

``as_frame`` and ``forward_image`` are here rather than in ``model`` because
they are the *call adapter* for the forward model, and the forward model is the
one thing every module of this package routes through.
"""

from __future__ import annotations

import math

import torch
from loguru import logger
from torch import Tensor

from ml.sim2f.config import (
    NDIM_BARE_MAP,
    NDIM_BATCH_MAP,
    NDIM_BATCHED_MAP,
)

__all__ = [
    "EPS",
    "as_frame",
    "forward_image",
    "from_phasor",
    "objective_terms",
    "physics_loss",
    "piston_free_phasor_corr",
    "piston_free_phasor_l1",
    "prepare_frame",
    "shaping_metrics",
    "to_phasor",
    "tv_phasor",
]

#: Denominator floor for every normalisation in this module.  A camera frame
#: whose total is below this is empty, and a metric computed from it is
#: meaningless -- clamping keeps it finite instead of returning NaN.
EPS = 1e-8


# ----------------------------------------------------------------------
# Call adapters
# ----------------------------------------------------------------------
def as_frame(value: object) -> Tensor:
    """Normalise a forward-model output to ``(B, 1, H, W)``.

    The 2f sim is a sibling module of this one and its return convention is
    pinned there; this adapter accepts the shapes a differentiable forward
    model in this repository uses -- ``(H, W)``, ``(B, H, W)`` and
    ``(B, 1, H, W)`` -- plus a ``(image, extras)`` tuple, so a caller never has
    to know which one it got.

    Args:
        value: A tensor, or a tuple/list whose first element is the intensity.

    Returns:
        The intensity frame as ``(B, 1, H, W)``.

    Raises:
        TypeError: If ``value`` is not a tensor (or a tuple starting with one).
        ValueError: If the shape is not one of the accepted ones.
    """
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("as_frame got an empty tuple")
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"as_frame expected a tensor, got {type(value).__name__}")
    frame = value
    if frame.ndim == NDIM_BARE_MAP:
        frame = frame[None, None]
    elif frame.ndim == NDIM_BATCH_MAP:
        # Batch-first by repository convention: (B, H, W).
        frame = frame[:, None]
    elif frame.ndim == NDIM_BATCHED_MAP:
        if frame.shape[1] != 1:
            raise ValueError(
                f"as_frame expected one intensity channel, got shape {tuple(frame.shape)}"
            )
    else:
        raise ValueError(f"as_frame expected 2-4 dims, got shape {tuple(frame.shape)}")
    return frame


def forward_image(sim: torch.nn.Module, phase: Tensor) -> Tensor:
    """Run the 2f forward model on a phase map and return ``(B, 1, ccd, ccd)``.

    This is the **only** call site that invokes ``sim.forward``; the model's
    flat-phase context, the physics loss and the evaluator all route through it
    so they cannot disagree about the input or the output convention.

    The detector is deliberately **not** requested: ``Sim2f.forward`` defaults
    to ``detector=False`` / ``exposure_ms=None``, and that is what keeps the
    gradient path differentiable (an 8-bit quantisation would floor it).  The
    twin's *optical* non-idealities -- the 0-order coherent mix, camera defocus,
    panel phase -- are applied regardless, which is exactly the point: the loss
    sees the calibrated chain, just not its sensor.

    Args:
        sim: The differentiable 2f bench (:class:`ml.sim2f.optics.Sim2f`).
        phase: Raw unwrapped phase in radians, ``(B, grid, grid)``.

    Returns:
        The camera-grid intensity, ``(B, 1, ccd_px, ccd_px)``.
    """
    return as_frame(sim.forward(phase))


# ----------------------------------------------------------------------
# Phase <-> phasor
# ----------------------------------------------------------------------
def to_phasor(phase: Tensor) -> Tensor:
    """Map ``(B, H, W)`` raw radians to a ``(B, 2, H, W)`` unit phasor.

    Args:
        phase: Raw unwrapped phase, ``(B, H, W)``.  A ``(B, 1, H, W)`` map is
            accepted and has its singleton channel squeezed.

    Returns:
        ``(B, 2, H, W)`` with channel 0 ``cos(phase)`` and channel 1
        ``sin(phase)``.

    Raises:
        ValueError: If ``phase`` is not 3-D (or 4-D with one channel).
    """
    if phase.ndim == NDIM_BATCHED_MAP and phase.shape[1] == 1:
        phase = phase[:, 0]
    if phase.ndim != NDIM_BATCH_MAP:
        raise ValueError(f"to_phasor expects (B, H, W), got shape {tuple(phase.shape)}")
    return torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)


def from_phasor(phasor: Tensor) -> Tensor:
    """Map a ``(B, 2, H, W)`` phasor to ``(B, H, W)`` radians with ``atan2``.

    The angle is only formed here, off-network and once, where the branch cut is
    a representation artefact rather than a training signal.

    Args:
        phasor: ``(B, 2, H, W)``; channel 0 is the real part.

    Returns:
        ``(B, H, W)`` radians in ``[-pi, pi]``.

    Raises:
        ValueError: If the quadrature axis is not ``dim 1`` of length 2.
    """
    if phasor.ndim != NDIM_BATCHED_MAP or phasor.shape[1] != 2:
        raise ValueError(
            f"from_phasor expects (B, 2, H, W), got shape {tuple(phasor.shape)}"
        )
    return torch.atan2(phasor[:, 1], phasor[:, 0])


def _mask_weights(mask: Tensor | None, phasor: Tensor) -> tuple[Tensor | None, Tensor]:
    """Coerce an optional pupil mask to the phasor's grid, dtype and device.

    Args:
        mask: ``None``, ``(H, W)``, ``(1, 1, H, W)`` or ``(B, 1, H, W)``.
        phasor: The tensor the mask will be broadcast against.

    Returns:
        ``(weights, per_sample_denominator)`` where ``weights`` is ``None``
        when no mask was given.  The denominator is the mask's area per sample
        times the channel count, i.e. exactly the number of elements a masked
        *mean* must divide by.

    Raises:
        ValueError: If the mask cannot be broadcast to the phasor's grid.
    """
    channels = phasor.shape[1]
    if mask is None:
        area = float(phasor.shape[0] * phasor.shape[2] * phasor.shape[3] * channels)
        return None, torch.full(
            (phasor.shape[0],), area, dtype=phasor.dtype, device=phasor.device
        )
    weights = mask.to(dtype=phasor.dtype, device=phasor.device)
    if weights.ndim == NDIM_BARE_MAP:
        weights = weights[None, None]
    elif weights.ndim == NDIM_BATCH_MAP:
        weights = weights[:, None]
    if weights.ndim != NDIM_BATCHED_MAP or weights.shape[1] != 1:
        raise ValueError(
            f"mask must be (H, W), (1, 1, H, W) or (B, 1, H, W), got {tuple(mask.shape)}"
        )
    if weights.shape[-2:] != phasor.shape[-2:]:
        raise ValueError(
            f"mask grid {tuple(weights.shape[-2:])} != phasor grid "
            f"{tuple(phasor.shape[-2:])}"
        )
    denominator = weights.sum(dim=(1, 2, 3)) * channels
    return weights, denominator


def _optimal_rotation(pred: Tensor, gt: Tensor, mask: Tensor | None) -> Tensor:
    """Closed-form global phase that best aligns ``pred`` with ``gt``.

    ``theta = angle(sum(conj(gt) * pred))``, formed from two real dot products so
    the magnitude never has to be divided by: ``atan2(0, 0)`` is ``0``, which is
    the correct answer for a vanishing overlap.

    Args:
        pred: ``(B, 2, H, W)`` prediction phasor.
        gt: ``(B, 2, H, W)`` reference phasor.
        mask: Optional pupil mask (see :func:`_mask_weights`).

    Returns:
        ``(B,)`` angles in radians.
    """
    weights, _ = _mask_weights(mask, pred)
    real = pred[:, 0] * gt[:, 0] + pred[:, 1] * gt[:, 1]
    imag = gt[:, 0] * pred[:, 1] - gt[:, 1] * pred[:, 0]
    if weights is not None:
        real = real * weights[:, 0]
        imag = imag * weights[:, 0]
    real_sum = real.sum(dim=(1, 2))
    imag_sum = imag.sum(dim=(1, 2))
    magnitude = torch.hypot(real_sum, imag_sum)
    # Guard rather than divide: an empty overlap carries no phase information,
    # so the identity rotation is the right answer there.
    return torch.where(magnitude > EPS, torch.atan2(imag_sum, real_sum), torch.zeros_like(real_sum))


def _rotate(phasor: Tensor, theta: Tensor) -> Tensor:
    """Multiply each sample by ``exp(-1j * theta)``.

    Args:
        phasor: ``(B, 2, H, W)``.
        theta: ``(B,)`` rotation angles in radians.

    Returns:
        ``(B, 2, H, W)`` rotated phasor.
    """
    cos = torch.cos(theta)[:, None, None]
    sin = torch.sin(theta)[:, None, None]
    real = phasor[:, 0] * cos + phasor[:, 1] * sin
    imag = phasor[:, 1] * cos - phasor[:, 0] * sin
    return torch.stack([real, imag], dim=1)


def piston_free_phasor_l1(
    pred: Tensor, gt: Tensor, mask: Tensor | None = None
) -> Tensor:
    """L1 between two phasors after removing the unobservable global phase.

    Args:
        pred: ``(B, 2, H, W)`` predicted unit phasor.
        gt: ``(B, 2, H, W)`` reference unit phasor (the teacher's phase).
        mask: Optional pupil mask; only masked cells enter the mean.  Outside
            the pupil the field is undisturbed, so the teacher's phase there is
            arbitrary and must not be fitted.

    Returns:
        A scalar (masked) mean absolute difference over both channels.

    Note:
        Invariant under ``pred -> pred * exp(1j * phi)`` for any per-sample
        ``phi``: the optimal angle absorbs it and the aligned phasor is
        unchanged.  ``tests/.../test_losses.py`` asserts this at ``rtol=1e-5``.
    """
    theta = _optimal_rotation(pred, gt, mask)
    aligned = _rotate(pred, theta)
    diff = (aligned - gt).abs()
    weights, denominator = _mask_weights(mask, pred)
    if weights is None:
        return diff.mean()
    numerator = (diff * weights).sum(dim=(1, 2, 3))
    return (numerator / denominator.clamp_min(EPS)).mean()


def piston_free_phasor_corr(
    pred: Tensor, gt: Tensor, mask: Tensor | None = None
) -> float:
    """Normalised phasor correlation after global-phase alignment.

    ``1`` means the two phases agree up to a global piston, ``0`` means they are
    uncorrelated.  Reported by the evaluator as ``phasor_corr``: it answers "did
    the network find *a* correct phase", which the far field cannot.

    Args:
        pred: ``(B, 2, H, W)`` prediction phasor.
        gt: ``(B, 2, H, W)`` teacher phasor.
        mask: Optional pupil mask.

    Returns:
        The batch mean correlation, finite by construction.
    """
    theta = _optimal_rotation(pred, gt, mask)
    aligned = _rotate(pred, theta)
    weights, _ = _mask_weights(mask, pred)
    product = (aligned * gt).sum(dim=1)
    norm_pred = aligned.pow(2).sum(dim=1)
    norm_gt = gt.pow(2).sum(dim=1)
    if weights is not None:
        product = product * weights[:, 0]
        norm_pred = norm_pred * weights[:, 0]
        norm_gt = norm_gt * weights[:, 0]
    numerator = product.sum(dim=(1, 2))
    denominator = torch.sqrt(
        norm_pred.sum(dim=(1, 2)).clamp_min(EPS) * norm_gt.sum(dim=(1, 2)).clamp_min(EPS)
    )
    return _num((numerator / denominator).mean())


def tv_phasor(phasor: Tensor) -> Tensor:
    """Isotropic total variation of a phasor, normalised by pixel count.

    Defined as the mean absolute first difference of both channels in both
    directions, so the value is scale-free (``O(1)`` for a phasor field) and
    comparable between grids.  Off by default in :class:`TrainConfig` for the
    reason recorded there: a phase-smoothness penalty was measured harmful for
    phase prediction (paired dR2 ``-0.026`` at ``w=0.003``, 0/3 seeds positive).

    Args:
        phasor: ``(B, 2, H, W)``.

    Returns:
        A scalar TV penalty (``0`` for a spatially constant field).
    """
    horizontal = (phasor[:, :, :, 1:] - phasor[:, :, :, :-1]).abs().mean()
    vertical = (phasor[:, :, 1:, :] - phasor[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


# ----------------------------------------------------------------------
# The teacher's objective
# ----------------------------------------------------------------------
def _ncc(pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample mean-subtracted correlation of two ``(B, 1, H, W)`` frames.

    Args:
        pred: Prediction frame.
        target: Reference frame.

    Returns:
        ``(B,)`` correlations in ``[-1, 1]``, finite by construction.
    """
    left = pred.flatten(1)
    right = target.flatten(1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    numerator = (left * right).sum(dim=1)
    denominator = torch.sqrt(
        left.pow(2).sum(dim=1).clamp_min(EPS) * right.pow(2).sum(dim=1).clamp_min(EPS)
    )
    return numerator / denominator


def _frame_median(frame: Tensor) -> Tensor:
    """Per-frame median with ``torch.median``'s exact convention, deterministically.

    ``torch.median(dim=...)`` on CUDA has **no deterministic implementation**.
    Under :attr:`TrainConfig.cudnn_deterministic` (``warn_only=True``) it emits
    ``UserWarning: median CUDA with indices output does not have a deterministic
    implementation``, so the run keeps going but the bit-identical-rerun promise
    is silently void; with ``warn_only=False`` it raises outright.  That matters
    here more than in a random loss term: the median is the pedestal every ratio
    in this module is measured against, so a stochastic one makes ``l1``/``ee``
    non-reproducible by construction.

    ``kthvalue`` gives the same element without the flagged kernel.  For ``n``
    values per frame the median is the element at 1-based rank ``(n - 1) // 2 +
    1`` of the sorted row: odd ``n`` is the single middle element, even ``n`` is
    the *lower* of the two middle ones -- which is what ``torch.median`` returns
    (the mean of the middles is ``torch.quantile(0.5)``, a different function).
    Verified ``torch.equal`` against ``torch.median(...).values`` on CPU and CUDA,
    including the production ``256 x 256`` and ``512 x 512`` frames.

    Measured cost per call, ``B = 8`` at ``256 x 256`` (this bench): CPU 2.1 ms
    with ``kthvalue`` against 2.0 ms with ``torch.median`` -- i.e. no cost at all
    -- and 0.21 ms vs 0.22 ms on the GPU.  (The sort-based spelling of the same
    element costs 28.6 ms on CPU, which is why it is *not* used: it would add
    ~15 % to a CPU training step to compute a number ``kthvalue`` gets for free.)

    Args:
        frame: ``(B, 1, H, W)`` frame.

    Returns:
        ``(B, 1, 1, 1)`` medians, one per sample.
    """
    flat = frame.flatten(1)
    column = flat.kthvalue((flat.shape[1] - 1) // 2 + 1, dim=1).values
    return column[:, None, None, None]


def prepare_frame(frame: Tensor) -> Tensor:
    """Remove the frame's background pedestal: subtract the median, clip at 0.

    A twin frame is ``signal + zero-order background + dark offset + read
    noise``.  The read noise is symmetric, so roughly half of the pixels are
    below the pedestal; a ratio formed on the raw frame (``p / p.sum()``) then
    measures the pedestal as much as the light, and "fraction of energy inside
    the support" can exceed 1 for a *prepared* frame.

    The order is load-bearing: subtract **then** clip.  Clipping the raw frame
    first rectifies every negative noise excursion into a positive one, which
    *adds* a pedestal proportional to the pixel count -- the exact defect
    recorded in ``sim/AGENTS.md`` for the ``clip(x, 0, None)`` case.

    The median is computed by :func:`_frame_median` (``kthvalue``) rather than
    by ``torch.median``, because the latter has no deterministic CUDA kernel and
    would silently void ``--cudnn-deterministic`` (it raises under
    ``warn_only=False``).  The replacement is bit-identical and cost-neutral:
    measured 2.1 ms vs 2.0 ms per call on CPU and 0.21 ms vs 0.22 ms on the GPU,
    ``B = 8`` at ``256 x 256``.

    Args:
        frame: ``(B, 1, H, W)`` intensity frame (any real dtype).

    Returns:
        The same shape with the per-frame median removed and negatives clipped
        to zero.  Idempotent for a frame that already has no pedestal (an
        analytic target has median 0 and no negatives, so preparing it is an
        identity).
    """
    return (frame - _frame_median(frame)).clamp_min(0.0)


def objective_terms(
    image: Tensor,
    target: Tensor,
    support: Tensor,
    *,
    w_ee: float,
    metric: str = "l1",
    prepare: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The teacher's objective, term by term -- the single implementation.

    ``p = I / I.sum()``, ``t = target / target.sum()``,
    ``L_shape`` = L1 (or MSE) between them, ``EE = sum(p * support)`` and
    ``L = L_shape + w_ee * (1 - EE)``.  Both parts are needed: ``L_shape`` is
    scale-free and happily empties the target, while ``EE`` alone is maximised
    by a delta function.

    ``I`` is the **prepared** frame when ``prepare`` is on (see
    :func:`prepare_frame`), which is what makes ``I.sum()`` a proxy for the
    modulated energy on the calibrated twin rather than a mix of light and
    detector pedestal.

    Both ``L_shape`` variants are **sums over the frame**, not per-pixel means:
    ``p`` and ``t`` are probability distributions, so ``sum|p - t|`` is the
    total-variation distance (0 = identical, 2 = no overlap) and
    ``sum((p-t)^2)`` is its L2 analogue.  A per-pixel mean would instead be
    ``2 / (ccd_px ** 2)``-sized -- 4.9e-4 at ``ccd_px = 64`` and 3.1e-5 at
    ``ccd_px = 256`` -- which would (a) make the objective's scale depend on the
    camera window, so two grids could not be compared, and (b) leave the
    ``w_ee * (1 - EE)`` term up to 5000x larger than the shape term, i.e. turn
    the objective into "concentrate light" with the shape as an invisible
    tie-break.  With sums, ``w_ee = 0.25`` means what it reads as: a quarter of
    the shape error.

    Args:
        image: Predicted camera frame, ``(B, 1, H, W)`` (or anything
            :func:`as_frame` accepts).  A single frame is broadcast over the
            batch, which is what lets the evaluator report the flat-phase
            baseline from one propagation.
        target: Requested distribution, ``(B, 1, H, W)``.
        support: Analytic target support (1 inside), broadcastable to ``target``.
        w_ee: Weight of the encircled-energy term.
        metric: ``"l1"`` or ``"mse"`` for ``L_shape``.
        prepare: Subtract the median and clip at 0 before forming any ratio.
            Exposed so the raw-vs-prepared ablation is explicit; the teacher
            and every reported number use the default.

    Returns:
        ``(loss, l1, ee, ncc)``: the composed scalar plus the three batch-mean
        diagnostics, all differentiable (``l1`` is always the true sum-L1 even
        when ``metric="mse"``, because it is the column the report tabulates).

    Raises:
        ValueError: If ``metric`` is not ``"l1"``/``"mse"``, or the frames'
            grids differ.
    """
    if metric not in ("l1", "mse"):
        raise ValueError(f"metric must be 'l1' or 'mse', got {metric!r}")
    pred = as_frame(image)
    reference = as_frame(target).to(dtype=pred.dtype, device=pred.device)
    if pred.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"image grid {tuple(pred.shape[-2:])} != target grid {tuple(reference.shape[-2:])}"
        )
    if pred.shape[0] == 1 and reference.shape[0] != 1:
        pred = pred.expand_as(reference)
    weights = support.to(dtype=pred.dtype, device=pred.device)
    weights = as_frame(weights) if weights.ndim < NDIM_BATCHED_MAP else weights
    if weights.shape[0] == 1 and reference.shape[0] != 1:
        weights = weights.expand_as(reference)

    if prepare:
        pred = prepare_frame(pred)
        reference = prepare_frame(reference)

    predicted = pred / pred.sum(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)
    wanted = reference / reference.sum(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)
    difference = predicted - wanted
    l1 = difference.abs().flatten(1).sum(dim=1)
    mse = difference.pow(2).flatten(1).sum(dim=1)
    shape = mse if metric == "mse" else l1
    ee = (predicted * weights).sum(dim=(1, 2, 3))
    loss = (shape + float(w_ee) * (1.0 - ee)).mean()
    return loss, l1.mean(), ee.mean(), _ncc(predicted, wanted).mean()


def physics_loss(
    sim: torch.nn.Module,
    phasor: Tensor,
    target: Tensor,
    support: Tensor,
    *,
    w_ee: float,
    metric: str = "l1",
    prepare: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    """Forward the predicted phase on the bench and score it.

    The returned numbers are the teacher's, from the same formula on the same
    normalisation, which is what makes the amortisation gap in the report a
    like-for-like difference rather than a comparison of two objectives.

    Args:
        sim: The differentiable 2f bench (the calibrated twin).  Passed in,
            never built here, so the trainer and the evaluator decide which
            instance and configuration to use -- and so the context channel and
            the physics leg cannot come from two different benches.
        phasor: ``(B, 2, grid, grid)`` predicted unit phasor.
        target: ``(B, 1, ccd_px, ccd_px)`` requested distribution.
        support: Analytic target support, broadcastable to ``target``.
        w_ee: Weight of the encircled-energy term (``TeacherConfig.w_ee``).
        metric: ``"l1"`` or ``"mse"`` (``TeacherConfig.metric``).
        prepare: Background removal (see :func:`objective_terms`).

    Returns:
        ``(loss, terms)`` where ``terms`` holds ``l1``, ``ee``, ``ncc`` and
        ``loss`` as floats -- the teacher's four reported quantities.
    """
    image = forward_image(sim, from_phasor(phasor))
    loss, l1, ee, ncc = objective_terms(
        image, target, support, w_ee=w_ee, metric=metric, prepare=prepare
    )
    return loss, {
        "l1": _num(l1),
        "ee": _num(ee),
        "ncc": _num(ncc),
        "loss": _num(loss),
    }


# ----------------------------------------------------------------------
# Shaping metrics
# ----------------------------------------------------------------------
def _num(value: Tensor | float) -> float:
    """Convert to ``float``, mapping a non-finite result to ``0.0``.

    Every caller guards its denominators, so a non-finite value here is a defect;
    ``0.0`` keeps a broken row out of a table without producing ``nan`` columns
    that silently poison downstream means.

    Args:
        value: Scalar tensor or float.

    Returns:
        A finite Python float.
    """
    number = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(number):
        logger.debug("non-finite metric {} reported as 0.0", number)
        return 0.0
    return number


def shaping_metrics(
    pred: Tensor, target: Tensor, support: Tensor, *, prepare: bool = True
) -> dict[str, float]:
    """Camera-space metrics for one batch of shaped spots.

    All frames are first prepared (background removed, see
    :func:`prepare_frame`) and then normalised to unit sum, so the metrics
    describe the *distribution* the teacher optimises and neither a global
    brightness error nor the detector pedestal is counted six times.
    ``cv_in_support`` is measured **inside the analytic support only** --
    measuring uniformity inside the bounding box would count empty background as
    perfect uniformity, which is the failure this metric exists to catch; on a
    pedestal-carrying frame it would also report a near-zero CV for a completely
    dark box.

    Args:
        pred: Predicted frame, ``(B, 1, ccd_px, ccd_px)``.
        target: Requested frame, ``(B, 1, ccd_px, ccd_px)``.
        support: Analytic support (1 inside), broadcastable to ``target``.
        prepare: Subtract the median and clip at 0 before any ratio.  Kept as a
            switch so the raw-vs-prepared ablation is explicit.

    Returns:
        ``{"l1", "ncc", "ee", "cv_in_support", "iou50", "peak_ratio"}``, batch
        means, every value finite.

    Raises:
        ValueError: If the prediction and target grids differ.
    """
    predicted = as_frame(pred)
    wanted = as_frame(target).to(dtype=predicted.dtype, device=predicted.device)
    if predicted.shape[-2:] != wanted.shape[-2:]:
        raise ValueError(
            f"pred grid {tuple(predicted.shape[-2:])} != target grid "
            f"{tuple(wanted.shape[-2:])}"
        )
    if predicted.shape[0] == 1 and wanted.shape[0] != 1:
        predicted = predicted.expand_as(wanted)
    weights = support.to(dtype=predicted.dtype, device=predicted.device)
    weights = as_frame(weights) if weights.ndim < NDIM_BATCHED_MAP else weights
    if weights.shape[0] == 1 and wanted.shape[0] != 1:
        weights = weights.expand_as(wanted)

    if prepare:
        predicted = prepare_frame(predicted)
        wanted = prepare_frame(wanted)

    p = predicted / predicted.sum(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)
    t = wanted / wanted.sum(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)

    # Sum-L1, exactly the objective's shape term (see `objective_terms`): 0 for
    # an identical distribution, 2 for no overlap.  A per-pixel mean would be
    # ~1e-4 and unreadable in a table.
    l1 = (p - t).abs().flatten(1).sum(dim=1)
    ee = (p * weights).sum(dim=(1, 2, 3))

    # Uniformity strictly inside the support.
    area = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mean_in = (p * weights).sum(dim=(1, 2, 3)) / area
    centred = (p - mean_in[:, None, None, None]).pow(2)
    std_in = torch.sqrt((centred * weights).sum(dim=(1, 2, 3)) / area)
    cv = std_in / mean_in.clamp_min(EPS)

    # 50 % of each frame's own peak -- independent of brightness, so a dim frame
    # is not scored as "empty" against a bright one.
    p_binary = (p >= 0.5 * p.amax(dim=(2, 3), keepdim=True)).to(p.dtype)
    t_binary = (t >= 0.5 * t.amax(dim=(2, 3), keepdim=True)).to(t.dtype)
    intersection = (p_binary * t_binary).sum(dim=(1, 2, 3))
    union = ((p_binary + t_binary) > 0).to(p.dtype).sum(dim=(1, 2, 3))
    iou = intersection / union.clamp_min(1.0)

    peak_ratio = predicted.amax(dim=(2, 3)) / wanted.amax(dim=(2, 3)).clamp_min(EPS)

    return {
        "l1": _num(l1.mean()),
        "ncc": _num(_ncc(p, t).mean()),
        "ee": _num(ee.mean()),
        "cv_in_support": _num(cv.mean()),
        "iou50": _num(iou.mean()),
        "peak_ratio": _num(peak_ratio.mean()),
    }
