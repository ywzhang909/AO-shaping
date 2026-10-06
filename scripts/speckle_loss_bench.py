"""Which loss actually fits a speckle field? Measure, don't guess.

Background: the project already has ``uniformity_term`` = ``1 - u/(1+u)`` with ``u = std/mean``.
That is a speckle-contrast proxy, and it is *non-saturating* -- which is already better than
``compute_quality_score``'s ``exp(-(cv/0.3)**2)``. So the honest question is not "is there a
speckle loss" but "is the existing one the right one, or does a different functional form
measure the thing we actually care about?"

The physics that matters: **fully developed speckle has an exponentially distributed
intensity**, so its normalised histogram is flat and ``CV = 1`` exactly. A shaped uniform
square is a near-delta, ``CV -> 0``. So the ideal speckle objective should push the intensity
*distribution* from exponential toward flat-top, which means a loss on the distribution's
shape -- not only its second moment.

Three axes are measured, because a loss that wins on one usually loses on another:

1. **Discrimination** -- does the term separate corrupted predictions from the truth? A term
   that does not move is not worth adding.
2. **Degenerate optimum** -- can it be won by destroying the signal? This is not
   hypothetical: optimising ``uniformity`` alone drove encircled energy to **0.002** on
   hardware. Each candidate is therefore probed with the empty-box and the all-dark
   predictions, which must both score *worse* than a good prediction.
3. **Anchoring** -- is it divided by the reference's own scale? An unanchored term can be
   driven to zero by shrinking the amplitude, which is the same failure in slow motion.

Run::

    python scripts/speckle_loss_bench.py

Writes ``report/loss_defects/speckle_loss_bench.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "report" / "loss_defects"

GRID = 64
N_SAMPLES = 48
EPS = 1e-12
SIDE = 32  # ROI short side in pixels, same as every ROI experiment in this report


# --------------------------------------------------------------------------- candidates


def _roi(intensity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Values inside the ROI as ``(B, N)``, plus the flat ROI."""
    m = mask.to(device=intensity.device, dtype=intensity.dtype)
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape)
    b = intensity.shape[0]
    return intensity.reshape(b, -1), m.reshape(1, -1).expand(b, -1)


def speckle_contrast(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``CV = std/mean`` within the ROI. 0 = flat top, 1 = fully developed speckle."""
    flat, m = _roi(img, mask)
    sel = flat * m
    count = torch.clamp(m.sum(dim=1), min=1.0)
    mean = sel.sum(dim=1) / count
    var = (sel.pow(2).sum(dim=1) / count) - mean.pow(2)
    return torch.sqrt(torch.clamp(var, min=0.0)) / torch.clamp(mean, min=EPS)


def cv_ratio_loss(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Anchored speckle contrast: ``|CV_pred / CV_ref - 1|``."""
    p, r = speckle_contrast(pred, mask), speckle_contrast(ref, mask)
    return (p - r).abs() / torch.clamp(r, min=EPS)


def cv_unanchored_loss(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``CV_pred`` on its own -- the ablation that shows anchoring is necessary."""
    return speckle_contrast(pred, mask)


def entropy_uniformity(img: torch.Tensor, mask: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Negative Shannon entropy of the intensity histogram, in ``[0, log(bins)]``.

    Exponential (speckle) intensity gives a flat histogram -> maximal entropy.
    A flat-top target gives a delta -> zero. So minimising this drives speckle to
    flat-top, which is the shaping objective, and unlike ``CV`` it uses the whole
    distribution rather than only the second moment.
    """
    flat, m = _roi(img, mask)
    sel = (flat * m).clamp_min(0.0)
    # Histogram by bucketising the *normalised* intensity: scale-invariant, so it measures
    # distribution shape and not brightness (brightness is `pib_term`'s job).
    peak = sel.amax(dim=1, keepdim=True).clamp_min(EPS)
    x = sel / peak
    idx = (x * bins).long().clamp(0, bins - 1)
    b = sel.shape[0]
    hist = torch.zeros(b, bins, device=sel.device, dtype=sel.dtype)
    hist.scatter_add_(1, idx, m)
    p = hist / torch.clamp(hist.sum(dim=1, keepdim=True), min=EPS)
    ent = -(p * torch.log(p.clamp_min(EPS))).sum(dim=1)
    return ent


def entropy_ratio_loss(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Anchored histogram entropy: ``|H_pred/H_ref - 1|``."""
    p, r = entropy_uniformity(pred, mask), entropy_uniformity(ref, mask)
    return (p - r).abs() / torch.clamp(r, min=EPS)


def entropy_absolute_loss(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Raw entropy, minimised -- the ablation that shows anchoring is necessary."""
    return entropy_uniformity(pred, mask)


def tv_ratio_loss(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Anchored total variation: ``|TV_pred/TV_ref - 1|`` over the ROI.

    This is the loss-side twin of `ml.zernike.metrics.total_variation_ratio`, which was
    selected as the one metric that detects over-smoothing. Here it measures *detail*
    agreement; note it is a two-sided term, so unlike the one-sided entropy it also
    penalises a prediction that is *rougher* than the truth.
    """
    m = mask.to(device=pred.device, dtype=pred.dtype)
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape)

    def tv(x: torch.Tensor) -> torch.Tensor:
        r = x * m
        dh = (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().flatten(1).sum(dim=1)
        dw = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().flatten(1).sum(dim=1)
        return dh + dw

    p, r = tv(pred), tv(ref)
    return (p - r).abs() / torch.clamp(r, min=EPS)


def mse_loss(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The incumbent, for reference."""
    return (pred - ref).pow(2).flatten(1).mean(dim=1)


def uniformity_existing(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``1 - u/(1+u)`` -- the incumbent speckle proxy, already non-saturating."""
    u = speckle_contrast(pred, mask)
    return 1.0 - u / (1.0 + u)


CANDIDATES = {
    "mse (incumbent)": mse_loss,
    "uniformity (incumbent)": uniformity_existing,
    "cv_ratio": cv_ratio_loss,
    "cv_unanchored": cv_unanchored_loss,
    "entropy_ratio": entropy_ratio_loss,
    "entropy_absolute": entropy_absolute_loss,
    "tv_ratio": tv_ratio_loss,
}


# --------------------------------------------------------------------------- data


def _load_targets() -> torch.Tensor:
    from scripts.compare_unet_baseline import _peak_normalise, _select_records
    from ml.zernike.train_amp import AmpTrainConfig, collect_split
    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=15, grid=GRID, epochs=1, lr=1e-3,
        batch_size=64, max_train=64, max_val=N_SAMPLES, beam_samples=N_SAMPLES,
        use_wandb=False, save_checkpoint=False, seed=0,
    )
    _, val = _select_records(dataset, cfg)
    return _peak_normalise(collect_split(dataset, val, torch.device("cpu"))["target"].clone())


def _mask() -> torch.Tensor:
    from ml.zernike.losses import roi_mask

    c = (GRID / 2.0, GRID / 2.0)
    # Float, not bool: `roi_mask` returns a boolean mask and `1 - mask` on a bool tensor is
    # a TypeError, while the losses under test want a multiplicative weight anyway.
    return roi_mask((GRID, GRID), c, "rectangle", float(SIDE), 1.0).float()


def _gauss_kernel(sigma: float) -> torch.Tensor:
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss_kernel(sigma)
    r = (k.numel() - 1) // 2
    n, c, h, w = img.shape
    f = img.reshape(n * c, 1, h, w)
    f = torch.nn.functional.pad(f, (r, r, 0, 0), mode="reflect")
    f = torch.nn.functional.conv2d(f, k.view(1, 1, 1, -1))
    f = torch.nn.functional.pad(f, (0, 0, r, r), mode="reflect")
    f = torch.nn.functional.conv2d(f, k.view(1, 1, -1, 1))
    return f.reshape(n, c, h, w)


def _speckle(img: torch.Tensor, cv: float, seed: int) -> torch.Tensor:
    """Fully developed speckle: exponential intensity, so CV is set by ``cv`` exactly."""
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / cv**2
    mult = torch.distributions.Exponential(scale).sample(img.shape).to(img.dtype) * scale
    return img * (mult - 1.0 + 1.0 / cv).clamp_min(0.0)


# --------------------------------------------------------------------------- driver


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = _load_targets()
    mask = _mask()

    # Two corruption families, because they break in opposite directions and a loss that
    # catches only one is not a speckle loss.
    cases = {
        "exact": ref,
        "over_smooth": _blur(ref, 1.5),
        "extra_speckle": _speckle(ref, 0.9, seed=3),
        "missing_speckle": _blur(ref, 0.6),
        # Degenerate optima. Both must score WORSE than `exact`; a term that does not is
        # gameable and must not be adopted regardless of how well it separates.
        "empty_roi": ref * (1.0 - mask),
        "all_dark": torch.zeros_like(ref),
    }

    scores: dict[str, dict[str, float]] = {}
    for name, fn in CANDIDATES.items():
        row: dict[str, float] = {}
        for cname, pred in cases.items():
            row[cname] = float(fn(pred, ref, mask).mean())
        scores[name] = row
        print(f"  {name:26s} " + "  ".join(f"{c}={row[c]:.4f}" for c in cases), flush=True)

    verdict: dict[str, dict[str, object]] = {}
    for name, row in scores.items():
        base = row["exact"]
        # Discrimination: mean absolute deviation from the truth across the four real
        # corruptions (the degenerate cases are scored separately, not mixed in).
        real = ["over_smooth", "extra_speckle", "missing_speckle"]
        discrim = float(np.mean([abs(row[c] - base) for c in real]))
        # Degenerate safety: both degenerate cases must be worse than the truth.
        safe = bool(row["empty_roi"] > base and row["all_dark"] > base)
        verdict[name] = {
            "values": row,
            "discrimination": discrim,
            "degenerate_safe": safe,
            "empty_roi_margin": float(row["empty_roi"] - base),
            "all_dark_margin": float(row["all_dark"] - base),
        }

    payload = {
        "config": {"grid": GRID, "n_samples": int(ref.shape[0]), "roi_side": SIDE},
        "cases": list(cases),
        "scores": scores,
        "verdict": verdict,
    }
    (OUT_DIR / "speckle_loss_bench.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'speckle_loss_bench.json'}")

    # Adoption rule, applied explicitly rather than left to taste.
    print("\n--- adoption verdict (must discriminate AND be degenerate-safe) ---")
    for name, v in sorted(verdict.items(), key=lambda kv: -kv[1]["discrimination"]):
        ok = bool(v["degenerate_safe"])
        print(
            f"  {name:26s} discrim={v['discrimination']:.4f} "
            f"degenerate_safe={ok} (empty {v['empty_roi_margin']:+.4f}, "
            f"dark {v['all_dark_margin']:+.4f})"
        )


if __name__ == "__main__":
    main()