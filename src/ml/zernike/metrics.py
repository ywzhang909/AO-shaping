"""Image metrics for the Zernike far-field model.

Two families, because this is both an img2img regression *and* a beam-shaping
measurement:

* **img2img-standard** -- the metrics an image-to-image paper would report:
  ``MSE``, ``RMSE``, ``MAE``, ``NRMSE``, ``PSNR``, ``SSIM``, plus Pearson
  ``correlation`` and the scale-invariant ``efficiency`` overlap already
  implemented canonically in
  :func:`ao_shaping.utils.image.beam_metrics.compute_metrics`.
* **beam-domain** -- what the optics actually cares about: where the spot is
  (``centroid_offset_px``) and how wide it is (``spot_diameter_px`` at a stated
  encircled-energy fraction), plus ``peak_ratio`` (a Strehl-like brightness
  agreement). A model can score a respectable PSNR while putting the spot in the
  wrong place or at the wrong width; these catch that and PSNR cannot.

Deliberately **not** included: ``LPIPS`` / ``FID``. Both need a pretrained backbone, and
while ``torchvision`` is now installed the ``lpips`` weight package is still absent, so
``LearnedPerceptualImagePatchSimilarity`` cannot load weights here (verified, not assumed).
Adding a perceptual metric is therefore a dependency decision, not a code change -- see
:func:`available_perceptual_metrics`.

* **speckle / detail** -- :func:`total_variation_ratio`. SSIM is blind to over-smoothing
(0.98 under a blur that TV registers as -25%), which matters because the strongest arm in the
architecture search won on SSIM while being measurably blurrier. The ratio is reported for
that reason, not as a generic extra; ``scripts/metric_discrimination.py`` is the measurement
that selected it and the four metrics it rejected.

Canonical helpers are reused rather than reimplemented, per ``AGENTS.md``:
``compute_metrics`` and ``measure_spot_diameter_cam`` from
:mod:`ao_shaping.utils.image.beam_metrics`, and ``centroid`` from
:mod:`ao_shaping.utils.image.spots_calc`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "available_perceptual_metrics",
    "batch_image_metrics",
    "per_sample_beam_metrics",
    "psnr",
    "total_variation_ratio",
]

#: Encircled-energy fraction used for the spot-diameter comparison. 0.90 matches
#: the default of the canonical ``measure_spot_diameter_cam``.
DEFAULT_EE_FRACTION: float = 0.90


def available_perceptual_metrics() -> tuple[str, ...]:
    """Report which perceptual metrics can actually run here.

    A perceptual metric needs a pretrained backbone. ``lpips`` and ``torchvision``
    are both absent from this environment, so the honest answer is an empty
    tuple rather than a metric that silently returns a constant.

    Returns:
        Names of the usable perceptual metrics.
    """
    import importlib.util

    usable: list[str] = []
    if importlib.util.find_spec("lpips"):
        usable.append("lpips")
    try:
        import torchmetrics.image as tmi

        if hasattr(tmi, "LearnedPerceptualImagePatchSimilarity"):
            usable.append("torchmetrics-lpips")
    except ImportError:
        pass
    return tuple(usable)


def total_variation_ratio(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean total variation of ``pred`` divided by that of ``target``.

    Reported as a ratio rather than a raw energy so the number is comparable across images
    and across runs: 1.0 means the prediction carries the same amount of spatial detail as
    the truth, below 1.0 means it is smoother than the truth, above 1.0 means it is rougher.

    Why this one earned a place when four other candidates did not
    (``scripts/metric_discrimination.py``, 48 real validation targets):

    ============================ ========= =========
    corruption                    SSIM      TV ratio
    ============================ ========= =========
    Gaussian blur sigma=1.5       0.9765    0.7472
    additive speckle CV=0.6      0.4948    7.8737
    ============================ ========= =========

    SSIM barely moves on blur -- 0.98 reads as "essentially perfect" to anyone scanning a
    number -- while TV drops 25%. That is the concrete form of the over-smoothing blindness
    documented in ``report/loss_defects/inverse_design_report.md``, where the U-Net scored the
    best SSIM of any arm while its ellipse error was ~2x worse. TV is also the only metric in
    that sweep whose response to additive speckle (7.9x) exceeded its response to blur, which
    is what a speckle-aware scorer needs.

    Rejected alongside it, for the record:

    * **MS-SSIM** -- needs a grid larger than 160 px; ours is 64, and 6x the compute is not
      affordable for a metric.
    * ``spectral_angle_mapper`` / ``rase`` -- they FFT along the **channel** axis, i.e. they
      are built for multispectral imagery, so on single-channel far-field frames they are
      degenerate (every image ties on magnitude).
    * A radially-averaged 2-D spectrum correlation, which I expected to be the right spectral
      metric for speckle -- it moved by less than 0.002 under every corruption and earned
      nothing.
    * ``vif`` -- exceeded 1.0 and *rose* under additive noise, i.e. it rewarded the corruption.

    Returns:
        The ratio, or NaN when the target has no measurable variation.
    """
    from torchmetrics.functional.image import total_variation

    p = pred.detach().to(torch.float32)
    t = target.detach().to(torch.float32)
    tv_t = float(total_variation(t, reduction="mean"))
    if tv_t <= 1e-12:
        return float("nan")
    return float(total_variation(p, reduction="mean")) / tv_t


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB.

    Args:
        pred: Prediction, any shape.
        target: Target, same shape.
        data_range: Nominal dynamic range of the data.

    Returns:
        Per-element PSNR tensor.
    """
    mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.ndim)))
    return 10.0 * torch.log10(data_range**2 / mse.clamp_min(1e-20))


def batch_image_metrics(
    pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0
) -> dict[str, float]:
    """Compute the img2img-standard metrics for a whole batch.

    SSIM comes from ``torchmetrics`` (it needs a Gaussian window and reflect
    padding, which is not worth reimplementing); everything else is closed-form.

    Args:
        pred: ``(N, C, H, W)`` prediction.
        target: ``(N, C, H, W)`` target, same shape.
        data_range: Nominal dynamic range, ``1.0`` for peak-normalised data.

    Returns:
        Flat dict of scalars: ``mse``, ``rmse``, ``mae``, ``nrmse``, ``psnr``,
        ``ssim``, and ``r2`` (kept so callers need only this one function).
    """
    pred = pred.detach().to(torch.float32)
    target = target.detach().to(torch.float32)
    error = pred - target
    mse = float(error.pow(2).mean())
    mae = float(error.abs().mean())
    span = float(target.max() - target.min())
    variance = float(torch.var(target))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "nrmse": float(np.sqrt(mse) / span) if span > 0 else float("nan"),
        "psnr": float(psnr(pred, target, data_range).mean()),
        "ssim": _ssim(pred, target, data_range=data_range),
        "r2": 1.0 - mse / variance if variance > 0 else float("nan"),
    }


def _ssim(pred: torch.Tensor, target: torch.Tensor, *, data_range: float) -> float:
    """Mean SSIM over the batch, or ``nan`` when torchmetrics is unavailable."""
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError:  # pragma: no cover - torchmetrics is a hard dep here
        return float("nan")
    value = structural_similarity_index_measure(
        pred.clamp(0.0, data_range), target.clamp(0.0, data_range), data_range=data_range
    )
    # torchmetrics returns a Tensor, or a (value, state_dict) tuple when
    # `return_early_return` is set; take the metric in either case.
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def per_sample_beam_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    ee_fraction: float = DEFAULT_EE_FRACTION,
) -> dict[str, float]:
    """Score one predicted far-field pattern against its measured frame.

    Both arrays are ``(H, W)`` and are compared **scale-invariantly**: the
    SLM+CCD chain has an unknown absolute gain, so every quantity here is
    computed after normalising each image to unit sum, exactly as the canonical
    :func:`~ao_shaping.utils.image.beam_metrics.compute_metrics` does.

    Args:
        pred: ``(H, W)`` predicted observable.
        target: ``(H, W)`` measured frame.
        ee_fraction: Encircled-energy fraction for the spot diameter.

    Returns:
        ``correlation``, ``efficiency`` (canonical), ``centroid_offset_px``,
        ``spot_diameter_pred_px``, ``spot_diameter_target_px``,
        ``spot_diameter_ratio`` and ``peak_ratio``.
    """
    from ao_shaping.utils.image.beam_metrics import (
        compute_metrics,
        measure_spot_diameter_cam,
    )
    from ao_shaping.utils.image.spots_calc import centroid

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    canonical = compute_metrics(p, t)

    p_norm = p / p.sum() if p.sum() > 0 else p
    t_norm = t / t.sum() if t.sum() > 0 else t

    # `centroid` delegates to scipy's center_of_mass, whose type stub is loose
    # enough that it may hand back tensors. Go through numpy, which accepts any
    # array-like, rather than assuming a scalar pair.
    centre_p = np.asarray(centroid(p_norm, return_float=True), dtype=np.float64).ravel()
    centre_t = np.asarray(centroid(t_norm, return_float=True), dtype=np.float64).ravel()
    cx_p, cy_p = float(centre_p[0]), float(centre_p[1])
    cx_t, cy_t = float(centre_t[0]), float(centre_t[1])
    offset = float(np.hypot(cx_p - cx_t, cy_p - cy_t))

    d_pred = float(measure_spot_diameter_cam(p_norm, energy=ee_fraction))
    d_true = float(measure_spot_diameter_cam(t_norm, energy=ee_fraction))
    peak_p = float(p_norm.max())
    peak_t = float(t_norm.max())

    return {
        "correlation": canonical["correlation"],
        "efficiency": canonical["efficiency"],
        "centroid_offset_px": offset,
        "spot_diameter_pred_px": d_pred,
        "spot_diameter_target_px": d_true,
        # 1.0 == the prediction has the same width as the measurement.
        "spot_diameter_ratio": d_pred / d_true if d_true > 0 else float("nan"),
        # 1.0 == the prediction peaks as sharply as the measurement.
        "peak_ratio": peak_p / peak_t if peak_t > 0 else float("nan"),
    }


def roi_shape_terms(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    center: tuple[float, float],
    size: float,
    aspect_ratio: float = 4.0 / 3.0,
    shape: str = "rectangle",
) -> dict[str, float]:
    """ROI shape terms for one ``(H, W)`` predicted/measured pair.

    This closes a real gap in the panel. ``summarise_beam_metrics`` reports
    correlation / efficiency / spot diameter / peak ratio, but **none of those
    say where the light landed inside the target box** -- and those are exactly
    the quantities the shaping objective optimises (``roi_pib_metric``) and the
    physical losses in :mod:`ml.zernike.losses` optimise (``pib_term``,
    ``uniformity_term``). Without them a loss change is unmeasurable: training
    with ``loss="physical"`` could improve or wreck the ROI terms and the logged
    panel would be identical either way.

    The target's own terms are returned alongside so a *relative* change can be
    read off directly; a prediction is not "better" just because its absolute
    PIB is high if the measured frame sits lower.

    Both images are passed through the canonical
    :func:`~ao_shaping.utils.image.target.metrics.rms_pib_terms`, so this cannot
    drift from the metric the hardware optimiser reports.

    Args:
        pred: ``(H, W)`` predicted observable.
        target: ``(H, W)`` measured frame.
        center: ROI centre in pixels, ``(x, y)``.
        size: ROI short side in pixels.
        aspect_ratio: ROI width / height.
        shape: ROI shape name (see ``target_shape_roi``).

    Returns:
        ``pib_term`` / ``uniformity`` / ``shape_sum`` of the prediction, the
        same three for the measured frame, and ``shape_sum_delta``.
    """
    from ao_shaping.utils.image.target.metrics import rms_pib_terms

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    pred_pib, pred_uni = rms_pib_terms(p, center, shape, size, aspect_ratio)
    ref_pib, ref_uni = rms_pib_terms(t, center, shape, size, aspect_ratio)
    return {
        "pib_term": float(pred_pib),
        "uniformity": float(pred_uni),
        "shape_sum": float(pred_pib + pred_uni),
        "target_pib_term": float(ref_pib),
        "target_uniformity": float(ref_uni),
        "target_shape_sum": float(ref_pib + ref_uni),
        "shape_sum_delta": float((pred_pib + pred_uni) - (ref_pib + ref_uni)),
    }


def summarise_beam_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average per-sample beam metrics into one flat dict.

    Args:
        rows: Per-sample dicts from :func:`per_sample_beam_metrics`.

    Returns:
        Mean of every key, or ``nan`` when ``rows`` is empty.
    """
    if not rows:
        return {}
    keys = rows[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out