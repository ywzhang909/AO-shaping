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


def second_moments(
    intensity: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-sample intensity second moments ``(var_x, var_y, var_r)`` in pixel^2.

    These are the central moments about the **intensity centroid**::

        x_bar   = sum(x * I) / sum(I)
        var_x   = sum((x - x_bar)^2 * I) / sum(I)

    and ``var_r = var_x + var_y``, the radial second moment -- the quantity usually
    meant by a beam's "second moment" and the one that sets the spot size.

    Why it is needed alongside MSE: a pixel-wise MSE is dominated by the bright core,
    so a prediction can match the peak closely while getting the *spread* badly wrong,
    and MSE barely notices. The second moments are a 2-number summary of that spread,
    so they react to core/halo redistribution that MSE is nearly blind to.

    Masked (``mask`` given) the centroid and moments are computed inside the ROI only,
    which is what you want when the spot is measured against a target box.

    Returns ``(B,)`` tensors, matching every other term in this module. Pixels outside
    a supplied mask, and any NaN the basis may carry, contribute nothing.
    """
    if intensity.dim() != 4:
        raise ValueError(f"intensity must be (B,1,H,W), got {tuple(intensity.shape)}")
    weights = intensity
    if mask is not None:
        if tuple(intensity.shape[-2:]) != tuple(mask.shape):
            raise ValueError(
                f"intensity spatial {tuple(intensity.shape[-2:])} does not match mask "
                f"{tuple(mask.shape)}; the ROI was built for a different frame size"
            )
        m = mask.to(device=intensity.device, dtype=intensity.dtype)
        if m.dim() == 2:
            m = m.view(1, 1, *m.shape)
        weights = intensity * m
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    height, width = intensity.shape[-2:]
    ys = torch.arange(height, device=intensity.device, dtype=intensity.dtype)
    xs = torch.arange(width, device=intensity.device, dtype=intensity.dtype)
    grid_y = ys.view(1, 1, height, 1)
    grid_x = xs.view(1, 1, 1, width)

    total = weights.sum(dim=(-2, -1)).squeeze(-1)
    safe = torch.clamp(total, min=EPS)

    mean_y = (weights * grid_y).sum(dim=(-2, -1)).squeeze(-1) / safe
    mean_x = (weights * grid_x).sum(dim=(-2, -1)).squeeze(-1) / safe

    dy = grid_y - mean_y.view(-1, 1, 1, 1)
    dx = grid_x - mean_x.view(-1, 1, 1, 1)
    var_y = (weights * dy.pow(2)).sum(dim=(-2, -1)).squeeze(-1) / safe
    var_x = (weights * dx.pow(2)).sum(dim=(-2, -1)).squeeze(-1) / safe
    return var_x, var_y, var_x + var_y


def ellipse_parameters(
    intensity: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-sample second-moment ellipse ``(cx, cy, var_x, var_y, cov)``.

    The intensity-weighted covariance about the centroid::

        cx, cy = sum(pos * I) / sum(I)
        var_x  = sum((x - cx)^2 * I) / sum(I)
        var_y  = sum((y - cy)^2 * I) / sum(I)
        cov    = sum((x - cx)(y - cy) * I) / sum(I)

    ``cov`` is the term the radial ``var_x + var_y`` cannot see: it encodes elongation and
    its orientation, so two ellipses with identical total spread but different aspect or
    tilt are distinguishable here and invisible there.

    Centroids are in pixel units (0-based at pixel centres).

    Returns five ``(B,)`` tensors, matching every other term in this module.
    """
    if intensity.dim() != 4:
        raise ValueError(f"intensity must be (B,1,H,W), got {tuple(intensity.shape)}")
    device, dtype = intensity.device, intensity.dtype
    height, width = intensity.shape[-2:]

    weights = intensity
    if mask is not None:
        if tuple(intensity.shape[-2:]) != tuple(mask.shape):
            raise ValueError(
                f"intensity spatial {tuple(intensity.shape[-2:])} does not match mask "
                f"{tuple(mask.shape)}; the ROI was built for a different frame size"
            )
        m = mask.to(device=device, dtype=dtype)
        if m.dim() == 2:
            m = m.view(1, 1, *m.shape)
        weights = intensity * m

    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    total = weights.sum(dim=(-2, -1)).clamp_min(EPS)

    ys = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width) + 0.5
    cx = (weights * xs).sum(dim=(-2, -1)) / total
    cy = (weights * ys).sum(dim=(-2, -1)) / total

    # (B,1,1,1), not (B,1,1): xs is (1,1,1,W) and ys is (1,1,H,1), and broadcasting
    # aligns from the RIGHT -- a 3-D view silently yields (1,B,1,W) instead of (B,1,1,W),
    # which leaves var_x with the wrong shape rather than raising.
    dx = xs - cx.reshape(-1, 1, 1, 1)
    dy = ys - cy.reshape(-1, 1, 1, 1)
    var_x = (weights * dx * dx).sum(dim=(-2, -1)) / total
    var_y = (weights * dy * dy).sum(dim=(-2, -1)) / total
    cov = (weights * dx * dy).sum(dim=(-2, -1)) / total
    # weights is (B,1,H,W), so reducing over (-2,-1) leaves (B,1). Squeeze to (B,) to match
    # every other term in this module -- otherwise composite_loss's torch.stack over
    # per_sample sees mixed ranks and fails.
    return cx.reshape(-1), cy.reshape(-1), var_x.reshape(-1), var_y.reshape(-1), cov.reshape(-1)


def ellipse_gap_term(
    prediction: Tensor, reference: Tensor, mask: Tensor | None = None
) -> dict[str, Tensor]:
    """Anchored relative error on each ellipse parameter.

    Every component is divided by the **reference's own scale**, so the term measures "does
    the prediction have the same spot as the measurement" and cannot be won by ignoring the
    data. ``cov`` is divided by the geometric mean of the reference's two spreads instead of
    by itself, because ``cov`` is antisymmetric and may legitimately vanish (a spot aligned
    with the grid) -- a self-normalised term would be singular exactly on the easy cases.

    Returns a dict with the five per-component gaps, the summed ``ellipse`` term that the
    composite loss uses, and ``_mean_ellipse`` for logging.
    """
    p_cx, p_cy, p_vx, p_vy, p_cov = ellipse_parameters(prediction, mask)
    r_cx, r_cy, r_vx, r_vy, r_cov = ellipse_parameters(reference, mask)

    # Pixel scale: the largest spread the reference has, floored so a degenerate frame
    # cannot make the centroid terms explode.
    scale = torch.maximum(r_vx, r_vy).clamp_min(EPS)
    cov_scale = torch.sqrt(r_vx.clamp_min(EPS) * r_vy.clamp_min(EPS))

    gaps = {
        "ellipse_cx": (p_cx - r_cx).abs() / scale,
        "ellipse_cy": (p_cy - r_cy).abs() / scale,
        "ellipse_var_x": (p_vx - r_vx).abs() / r_vx.clamp_min(EPS),
        "ellipse_var_y": (p_vy - r_vy).abs() / r_vy.clamp_min(EPS),
        "ellipse_cov": (p_cov - r_cov).abs() / cov_scale,
    }
    out = dict(gaps)
    out["ellipse"] = sum(gaps.values())
    out["_mean_ellipse"] = out["ellipse"].mean()
    return out


def spot_moment_gap_term(
    prediction: Tensor, target: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Relative error in the radial second moment, ``|var_r(pred) - var_r(target)| / var_r(target)``.

    **Anchored** to the target, unlike :func:`pib_term` / :func:`uniformity_term`: the
    reference is always the measurement. That matters. The unanchored physical terms
    were a category error for *fitting* -- their optimum is "ignore the data and emit
    an ideal spot" -- and the fix was :func:`shape_gap_term`. This term is built the
    same way from the start: a spot-size error is only meaningful against the spot
    size that was actually measured.

    Normalising by the target's own moment makes it scale-free, so it is comparable
    across frames with different brightness (the corpus spans uint8/float32/float64
    with maxima of 255 / 239.6 / 100.9) and comparable to ``w_mse``.
    """
    _, _, pred_r = second_moments(prediction, mask)
    _, _, true_r = second_moments(target, mask)
    return (pred_r - true_r).abs() / torch.clamp(true_r, min=EPS)


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


def speckle_detail_term(
    prediction: Tensor, reference: Tensor, mask: Tensor
) -> dict[str, Tensor]:
    """Two-sided relative error on in-ROI total variation. The speckle loss to use.

    Returns ``tv_ratio`` (the loss) plus ``tv_pred`` / ``tv_ref`` for logging.

    Why this form, measured rather than argued (``scripts/speckle_loss_bench.py``, 48 real
    validation targets, three corruption families and two degenerate probes):

    =================================== ============ ===============
    candidate                            discrimination degenerate-safe
    =================================== ============ ===============
    **tv_ratio (this term)**             **1.8886**   yes
    entropy (absolute)                   0.4030       **no**
    cv (anchored)                        0.2410       yes
    cv (unanchored)                      0.2254       **no**
    entropy (anchored)                  0.1536       yes
    uniformity (incumbent)               0.0481       yes
    mse (incumbent)                      0.0064       yes
    =================================== ============ ===============

    Three things follow.

    **The incumbents are nearly blind to speckle.** MSE separates the corruptions by 0.0064
    and ``uniformity_term`` by 0.0481 -- the latter is a ``std/mean`` proxy, which is exactly
    the second moment that speckle noise barely moves, because multiplicative noise inflates
    the mean and the standard deviation together. If the quantity being shaped is the texture
    of the far field, neither term is measuring it.

    **Anchoring is not optional.** Every unanchored candidate scores **exactly 0** on both
    degenerate probes: empty ROI and all-dark. An unanchored contrast or entropy can be driven
    to its optimum by destroying the signal. That is not hypothetical -- optimising
    ``uniformity`` alone drove encircled energy to 0.002 on hardware. Dividing by the
    reference's own scale is what makes the term unwinnable that way, which is the same
    convention ``ellipse_gap_term`` already uses and for the same reason.

    **Two-sided beats one-sided.** ``CV`` and entropy are both one-directional: they only know
    that *too much* speckle is bad, so blurring monotonically improves them. Total variation
    penalises too little detail as well, which is why it separates ``missing_speckle``
    (0.0557) where ``uniformity`` does not (0.0042, and in the wrong direction).

    The physics: fully developed speckle has exponentially distributed intensity, so its
    normalised histogram is flat and ``CV = 1``; a shaped uniform square is a near-delta with
    ``CV -> 0``. A term on the distribution's *shape* is therefore the right objective, and
    this is the shape term that survived both safety checks.

    Still a within-ROI term: it carries no energy information, so it must accompany
    :func:`pib_term`, exactly as :func:`uniformity_term` does.
    """
    m = mask.to(device=prediction.device, dtype=prediction.dtype)
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape)

    def tv(x: Tensor) -> Tensor:
        r = x * m
        # Horizontal and vertical first differences. Gradient magnitude would be equivalent
        # up to a constant and is not used here, to keep the value comparable to
        # `ml.zernike.metrics.total_variation_ratio`.
        dh = (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().flatten(1).sum(dim=1)
        dw = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().flatten(1).sum(dim=1)
        return dh + dw

    tv_p, tv_r = tv(prediction), tv(reference)
    ratio = (tv_p - tv_r).abs() / torch.clamp(tv_r, min=EPS)
    return {"tv_ratio": ratio, "tv_pred": tv_p, "tv_ref": tv_r}


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
        w_spot_moment: Weight on the **anchored** spot-size term
            :func:`spot_moment_gap_term`, the relative error in the radial second
            moment. MSE is dominated by the bright core, so it barely notices a
            prediction that matches the peak while getting the spread wrong; this
            term is a two-number summary of exactly that. Anchored to the measurement,
            so unlike ``w_pib``/``w_uniformity`` it cannot be won by ignoring the data.
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
    w_spot_moment: float = 0.0
    #: Weight on the anchored ellipse-fit gap -- the five second-moment-ellipse
    #: parameters (centroid x/y, var_x, var_y, covariance). Strictly richer than
    #: ``w_spot_moment``: a 2 px shift scores 0.0898 here and exactly 0.0 on the radial
    #: term, which is blind to both position and orientation.
    w_ellipse: float = 0.0
    #: Weight on the **anchored speckle-detail term** :func:`speckle_detail_term`,
    #: the two-sided relative error in in-ROI total variation. This is the term to
    #: use when the far field's *texture* is the target, which is what square
    #: shaping is: measured over 48 real targets it separates three corruption
    #: families by 1.8886, against 0.0481 for ``w_uniformity`` and 0.0064 for MSE.
    #: Anchored to the measurement and two-sided, so it cannot be won by emptying
    #: the ROI (which drove EE to 0.002 on hardware for the unanchored terms) and
    #: it penalises an over-smoothed prediction as well as an over-noisy one.
    w_speckle: float = 0.0
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

    if cfg.w_speckle:
        speck = speckle_detail_term(pred, target, mask)
        out["speckle"] = speck["tv_ratio"]
        per_sample.append(cfg.w_speckle * speck["tv_ratio"])
        out["speckle_tv_pred"] = speck["tv_pred"]
        out["speckle_tv_ref"] = speck["tv_ref"]
    if cfg.w_ellipse:
        ellipse = ellipse_gap_term(pred, target, mask)
        out["ellipse"] = ellipse["ellipse"]
        per_sample.append(cfg.w_ellipse * ellipse["ellipse"])

    if cfg.w_spot_moment:
        moment = spot_moment_gap_term(pred, target, mask)
        out["spot_moment"] = moment
        per_sample.append(cfg.w_spot_moment * moment)

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
    "second_moments",
    "shape_gap_term",
    "spot_moment_gap_term",
    "ellipse_parameters",
    "ellipse_gap_term",
    "uniformity_term",
]
