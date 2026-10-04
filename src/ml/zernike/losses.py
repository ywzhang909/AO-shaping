"""Differentiable far-field losses for the Zernike forward model.

**A library module — no CLI, no ``__main__``.**

Why this exists
---------------
``train_amp`` currently optimises a plain pixel MSE on a peak-normalised frame
(``train_amp.py:490-493``). That objective is blind to the two things a beam
shaping run actually cares about:

* **where** the light lands (an MSE happily puts a diffuse halo where the target
  is dark, and scores it as "close" because the target is dark there too), and
* **how much** of the frame's light is inside the target at all.

The physical objectives that *are* already implemented in this repo —
:func:`roi_pib_metric`, :func:`rms_pib_terms`, :func:`roi_energy_loss` in
``ao_shaping.utils.image.target.metrics`` — capture both, but they are numpy
functions over a boolean ROI, and they are used as *reported metrics*, not as a
gradient path. This module ports them to torch so they can be optimised
directly, and adds the Poisson negative-log-likelihood used for photon-counted
intensity in the FourierGSNet paper (Yan, Holenderski & Meratnia, *Adv. Photon.
Nexus* 5(2) 026005 (2026), Eq. 28):

    L_MLE = -ln p(I_img | I_hat) = sum_{u,v} ( I_hat - I_img * ln I_hat )

with ``p(I_img | I_hat) ~ Poisson(mean = I_hat)``. Note this is **not** a
distance: ``I_img`` multiplies ``ln I_hat``, so a pixel the target wants dark
(``I_img = 0``) contributes ``I_hat`` alone — stray light is penalised
linearly, which is precisely the failure mode MSE under-weights.

Why a *port* and not a reimplementation
---------------------------------------
The ROI mask is built by calling the existing
:func:`ao_shaping.utils.image.target.metrics.target_shape_roi`, so the mask
used by the loss is by construction the same one the hardware optimiser scores
with. Re-deriving the shape geometry here would be exactly the "second copy
that drifts" failure the repo already suffers from elsewhere (see
``tests/ao_shaping/tools/slm/test_cartographer_slot_rotation.py`` and the
``probe_help_golden`` guard family for the same pattern). ``test_losses.py``
asserts agreement with the numpy originals to float tolerance so any future
divergence is caught.

Differentiability
-----------------
A boolean ROI mask is **constant** with respect to the model parameters, so
``sum(I[mask]) / sum(I)`` and ``1 - u/(1+u)`` are exactly differentiable in
the intensities. Nothing here differentiates through ``argmax``/``centroid`` —
the mask may be located by any means (including ``argmax``) because it is
treated as a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from ao_shaping.utils.image.target.metrics import target_shape_roi

#: Floor for ``log`` and for divisions by an intensity that may be exactly 0.
#: The forward model produces exact zeros on the far-field background (sidelobes),
#: so an unguarded ``log`` yields ``-inf`` and an unguarded ``I_hat`` can be 0.
EPS: float = 1e-8


def roi_mask(
    image_shape: tuple[int, int],
    center: tuple[float, float],
    shape: str = "rectangle",
    size: float | None = None,
    aspect_ratio: float = 4.0 / 3.0,
) -> Tensor:
    """Boolean ROI mask as a cached torch tensor, from the canonical builder.

    Thin wrapper over :func:`target_shape_roi` — the geometry is NOT
    reimplemented here. Cached because the mask is a constant during training.
    """
    return _cached_mask(image_shape, center, shape, size, aspect_ratio)


@lru_cache(maxsize=32)
def _cached_mask(
    image_shape: tuple[int, int],
    center: tuple[float, float],
    shape: str,
    size: float | None,
    aspect_ratio: float,
) -> Tensor:
    h, w = int(image_shape[0]), int(image_shape[1])
    resolved = float(min(h, w)) if size is None else float(size)
    mask = target_shape_roi((h, w), center, shape, resolved, aspect_ratio)
    if not mask.any():
        raise ValueError(
            f"ROI for shape={shape!r} size={resolved} centred at {center} is empty "
            f"in a {h}x{w} frame; the loss would divide by a zero-size region"
        )
    return torch.from_numpy(np.ascontiguousarray(mask))


def shape_gap_term(
    intensity: Tensor,
    reference: Tensor,
    mask: Tensor,
) -> Tensor:
    """Anchored physical term: how far the ROI shape statistics are from the target.

    This exists because :func:`pib_term` and :func:`uniformity_term` are
    **unanchored** -- they read only the prediction. That is correct for *shaping*
    (you genuinely want the brightest, flattest box you can synthesise) but
    **wrong for fitting a forward model**, where the model's job is to predict
    the measured frame rather than to produce an ideal spot. Maximising them on a
    fidelity task has a degenerate optimum: emit a clean square, ignore the data.

    Measured on the real corpus, unanchored ``pib + uniformity`` reaches
    ``shape_sum = 1.496`` against a measured target of ``1.113`` -- 34% *better
    than the physics being predicted* -- while val R2 collapses from +0.78 to
    **-0.86**, worse than predicting a constant. The model had stopped
    predicting and started idealising.

    Anchoring to the reference's own terms removes the degenerate direction: the
    best achievable value is 0, attained when the prediction reproduces the
    measured shape statistics, and overshooting is penalised exactly like
    undershooting.

    Returns ``|shape_sum(pred) - shape_sum(reference)|`` per sample, shape
    ``(B,)``.
    """
    pred_sum = pib_term(intensity, mask) + uniformity_term(intensity, mask)
    ref_sum = pib_term(reference, mask) + uniformity_term(reference, mask)
    return (pred_sum - ref_sum).abs()


@torch.no_grad()
def _shape_gap_reference_scale(reference: Tensor, mask: Tensor) -> Tensor:
    """Mean ``shape_sum`` of the reference -- the natural unit for weighting.

    The physical terms are ``O(1)`` while a peak-normalised MSE is ``O(0.003)``,
    a gap of roughly 450x. That is why ``w_mse=1.0`` alongside ``w_pib=1.0`` is
    *not* a balanced blend: measured on the corpus, the blend's val R2 was -0.28,
    i.e. the fidelity anchor contributed about 0.2% of the gradient. Normalising
    the anchored term by the reference's own magnitude puts it back on the same
    footing as MSE, so ``w_mse=1`` and ``w_shape_gap=1`` are comparable weights.
    """
    return (pib_term(reference, mask) + uniformity_term(reference, mask)).mean()


def _flatten_roi(intensity: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Reduce ``(B,1,H,W)`` intensity to per-sample ``(roi_sum, total_sum)``.

    Both results are ``(B,)``: the channel axis is squeezed so every term in
    this module has the same per-sample shape and can be stacked by
    :func:`composite_loss`.
    """
    if intensity.dim() != 4:
        raise ValueError(f"intensity must be (B,1,H,W), got {tuple(intensity.shape)}")
    if tuple(intensity.shape[-2:]) != tuple(mask.shape):
        raise ValueError(
            f"intensity spatial {tuple(intensity.shape[-2:])} does not match mask "
            f"{tuple(mask.shape)}; the ROI was built for a different frame size"
        )
    m = mask.to(device=intensity.device, dtype=intensity.dtype)
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape)
    roi_sum = (intensity * m).sum(dim=(-2, -1)).squeeze(-1)
    total = intensity.sum(dim=(-2, -1)).squeeze(-1)
    return roi_sum, total


def pib_term(intensity: Tensor, mask: Tensor) -> Tensor:
    """Fraction of in-frame light inside the ROI, per sample, in ``[0, 1]``.

    Torch twin of the ``pib_term`` returned by
    :func:`rms_pib_terms`. Exposure-invariant (a ratio), and cannot be "won"
    by emptying the box because an empty ROI scores 0.
    """
    roi_sum, total = _flatten_roi(intensity, mask)
    return roi_sum / torch.clamp(total, min=EPS)


def uniformity_term(intensity: Tensor, mask: Tensor) -> Tensor:
    """``1 - u/(1+u)`` with ``u = std/mean`` over the ROI, per sample.

    Torch twin of the ``rms_term`` returned by :func:`rms_pib_terms`.
    ``1`` = perfectly flat, ``0`` = maximally non-uniform. Note this is a
    *within-ROI* shape term and carries no energy information — which is why
    :func:`pib_term` must accompany it (optimising uniformity alone has
    previously driven the box empty, EE -> 0.002, measured on hardware).
    """
    if intensity.dim() != 4:
        raise ValueError(f"intensity must be (B,1,H,W), got {tuple(intensity.shape)}")
    m = mask.to(device=intensity.device, dtype=intensity.dtype)
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape)
    roi = intensity * m
    count = torch.clamp(m.sum(), min=1.0)
    mean = roi.sum(dim=(-2, -1)).squeeze(-1) / count
    var = (roi.pow(2).sum(dim=(-2, -1)).squeeze(-1) / count) - mean.pow(2)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    u = std / torch.clamp(mean, min=EPS)
    return 1.0 - u / (1.0 + u)


def roi_energy_loss(current: Tensor, reference: float | Tensor) -> Tensor:
    """Fractional in-ROI energy loss vs a reference; 0 means no loss.

    Torch twin of :func:`roi_energy_loss`. A non-positive or non-finite
    reference means "no protection" and yields 0, matching the numpy original.
    """
    ref = torch.as_tensor(reference, dtype=current.dtype, device=current.device)
    ref = ref.reshape(()) if ref.dim() == 0 else ref
    if not torch.isfinite(ref).all() or bool((ref <= 0).all()):
        return torch.zeros_like(current)
    return (ref - current) / ref


def poisson_nll(pred_counts: Tensor, target_counts: Tensor, eps: float = EPS) -> Tensor:
    """Poisson negative log-likelihood, summed over pixels, per sample.

    FourierGSNet Eq. (28): ``sum (I_hat - I_img * ln I_hat)``. Lower is better.

    Both inputs must be in the **same absolute counting units** — this term is
    not scale-invariant, so it cannot be used on peak-normalised predictions
    (see :class:`LossConfig.normalization`).

    ``pred_counts`` is clamped to ``eps`` before the log because a far field has
    exact zeros on its background.
    """
    if pred_counts.shape != target_counts.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred_counts.shape)} vs target "
            f"{tuple(target_counts.shape)}"
        )
    safe = torch.clamp(pred_counts, min=eps)
    per_pixel = safe - target_counts * torch.log(safe)
    return per_pixel.flatten(1).sum(dim=1)


@dataclass(frozen=True)
class LossConfig:
    """Weights for the composite forward-model loss.

    Attributes:
        w_mse: Plain intensity MSE (the incumbent baseline). Set 0 to ablate.
        w_poisson: Poisson NLL weight. Requires absolute counts; see
            :attr:`normalization`.
        w_pib: Weight on the in-ROI energy fraction (maximised).
        w_uniformity: Weight on the in-ROI flatness term (maximised).
        w_shape_gap: Weight on the **anchored** physical term
            :func:`shape_gap_term`, i.e. the absolute error between the
            prediction's ROI shape statistics and the measured frame's. This is
            the term to use when *fitting* a forward model: ``w_pib`` /
            ``w_uniformity`` are unanchored and their best value is "ignore the
            data and emit an ideal spot" (measured: val R2 +0.78 -> -0.86, with
            ``shape_sum`` 34% above the physics being predicted). Set this
            instead of ``w_pib``/``w_uniformity`` for fidelity tasks.
        shape_gap_relative: Normalise :func:`shape_gap_term` by the reference's
            own mean ``shape_sum``. The physical terms are ``O(1)`` while a
            peak-normalised MSE is ``O(0.003)``, so without this a nominal
            ``w_mse=1.0`` contributes ~0.2% of the gradient and cannot anchor
            anything. Default ``True``.
        normalization: Which forward-model observable the loss expects. ``"none"``
            keeps absolute scale (required for :func:`poisson_nll`);
            ``"peak"`` matches the incumbent setup but makes ``w_poisson``
            meaningless.
    """

    w_mse: float = 1.0
    w_poisson: float = 0.0
    w_pib: float = 0.0
    w_uniformity: float = 0.0
    w_shape_gap: float = 0.0
    shape_gap_relative: bool = True
    normalization: str = "peak"

    def __post_init__(self) -> None:
        if self.normalization not in {"peak", "sum", "none"}:
            raise ValueError(
                f"normalization must be one of ['peak', 'sum', 'none'], "
                f"got {self.normalization!r}"
            )
        if self.w_poisson and self.normalization != "none":
            raise ValueError(
                "Poisson NLL requires absolute counting units, so it needs "
                "normalization='none'. With 'peak' the prediction's gain is "
                "divided out and the term stops being a likelihood. Set "
                "w_poisson=0 or change normalization."
            )


def composite_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    cfg: LossConfig,
) -> dict[str, Tensor]:
    """Evaluate every weighted term and return them individually plus ``total``.

    The terms are returned separately on purpose: this repo has been bitten by
    single-number judgements before (``normalization='sum'`` produced
    PSNR 72 dB / SSIM 0.9996 while R^2 was *worse* than a constant predictor),
    so a composite loss must be inspectable, not a single opaque scalar.

    Note the two physical terms are *negated* into loss form: ``pib_term`` and
    ``uniformity_term`` are both "higher is better" scores in ``[0, 1]``, so
    ``1 - term`` is the loss contribution.
    """
    err = pred - target
    out: dict[str, Tensor] = {}
    per_sample: list[Tensor] = []

    if cfg.w_mse:
        mse = err.pow(2).flatten(1).mean(dim=1)
        out["mse"] = mse
        per_sample.append(cfg.w_mse * mse)

    if cfg.w_poisson:
        nll = poisson_nll(pred, target)
        out["poisson_nll"] = nll
        per_sample.append(cfg.w_poisson * nll)

    if cfg.w_pib:
        pib = pib_term(pred, mask)
        out["pib"] = pib
        per_sample.append(cfg.w_pib * (1.0 - pib))

    if cfg.w_uniformity:
        uni = uniformity_term(pred, mask)
        out["uniformity"] = uni
        per_sample.append(cfg.w_uniformity * (1.0 - uni))

    if cfg.w_shape_gap:
        gap = shape_gap_term(pred, target, mask)
        if cfg.shape_gap_relative:
            gap = gap / _shape_gap_reference_scale(target, mask).clamp(min=EPS)
        out["shape_gap"] = gap
        per_sample.append(cfg.w_shape_gap * gap)

    if not per_sample:
        raise ValueError("LossConfig has no non-zero weight; nothing to optimise")

    total = torch.stack(per_sample, dim=0).sum(dim=0)
    out["total"] = total
    out["_mean_total"] = total.mean()
    return out


__all__ = [
    "EPS",
    "LossConfig",
    "composite_loss",
    "pib_term",
    "poisson_nll",
    "roi_energy_loss",
    "roi_mask",
    "uniformity_term",
]
