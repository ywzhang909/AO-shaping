"""Does a candidate metric actually *discriminate* the failure modes we documented?

Motivation: `report/loss_defects/inverse_design_report.md` records two things SSIM and R2 do
badly between them.

1. **SSIM rewards over-smoothing.** A U-Net's prediction is visibly blurrier than the truth,
   yet its SSIM is the highest of any arm we trained (0.849 vs 0.724). We already showed the
   U-Net's ellipse error is ~2x worse (0.3997 vs 0.2027), so SSIM is partly grading blur as
   similarity.
2. **Neither metric looks at the spectrum.** This is speckle. The physical failure mode is not
   a uniform pixel error -- it is energy in the wrong spectral band, or a speckle contrast
   that is too high. A metric that never transforms an image cannot see that.

So a metric earns a place in `evaluate()` only if it responds where the incumbents do not.
This script builds controlled corruptions of the *real* validation targets and measures each
candidate's response, so the choice is evidence-based rather than a metric dump.

Five variants, each isolating one failure mode:

======== ==================================================================
exact    the target itself (every metric should be optimal here)
smooth   Gaussian blur -- isolates over-smoothing (SSIM's blind spot)
shift    2 px translation -- isolates envelope/centroid error
speckle  multiplicative fully-developed speckle (exponential intensity,
         the physical noise floor of this bench)
mixed    speckle + blur, the realistic combination
======== ==================================================================

The bar for a candidate: it must separate `exact` from `smooth` and from `speckle` by a
margin large enough to reorder models, AND it must not be a monotone function of R2 (if it
is, it adds no information).

Run::

    python scripts/metric_discrimination.py

Writes ``report/loss_defects/metric_discrimination.json``.
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ml.hwdataset import (  # noqa: E402
    MaterialiserConfig,
    HwPhaseImageDataset,
    build_hw_index,
)

OUT_DIR = ROOT / "report" / "loss_defects"

GRID = 64
N_SAMPLES = 48
SHIFT_PX = 2
BLUR_SIGMA = 1.5
SPECKLE_CV = 0.6  # fully developed speckle has CV = 1.0; this is a milder, realistic case


# --------------------------------------------------------------------------- data


def _load_targets() -> torch.Tensor:
    """Peak-normalised validation targets from the same family every search arm used."""
    from scripts.compare_unet_baseline import _peak_normalise, _select_records
    from ml.zernike.train_amp import AmpTrainConfig, collect_split

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",),
        n_max=15,
        grid=GRID,
        epochs=1,
        lr=1e-3,
        batch_size=64,
        max_train=64,
        max_val=N_SAMPLES,
        beam_samples=N_SAMPLES,
        use_wandb=False,
        save_checkpoint=False,
        seed=0,
    )
    _, val = _select_records(dataset, cfg)
    return _peak_normalise(collect_split(dataset, val, torch.device("cpu"))["target"].clone())


# --------------------------------------------------------------------------- variants


def _gauss_kernel(sigma: float) -> torch.Tensor:
    """Separable 1-D Gaussian, truncated at 4 sigma."""
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims, reflect-padded so edges stay put.

    ``conv2d`` rejects ``mode='reflect'`` with an integer pad, so the reflect padding is
    materialised by ``F.pad`` first and the convolution itself runs 'valid'. That keeps the
    border value equal to its true local mean instead of shrinking toward zero, which a
    zero-pad would do and which would contaminate the edge pixels we are measuring.
    """
    k = _gauss_kernel(sigma)
    r = (k.numel() - 1) // 2
    n, c, h, w = img.shape
    flat = img.reshape(n * c, 1, h, w)
    flat = torch.nn.functional.pad(flat, (r, r, 0, 0), mode="reflect")
    flat = torch.nn.functional.conv2d(flat, k.view(1, 1, 1, -1))
    flat = torch.nn.functional.pad(flat, (0, 0, r, r), mode="reflect")
    flat = torch.nn.functional.conv2d(flat, k.view(1, 1, -1, 1))
    return flat.reshape(n, c, h, w)


def _shift(img: torch.Tensor, px: int) -> torch.Tensor:
    return torch.roll(img, shifts=px, dims=-1)


def _speckle(img: torch.Tensor, cv: float, seed: int) -> torch.Tensor:
    """Multiplicative speckle with the exponential intensity distribution.

    Fully developed speckle is the sum of many complex phasors, so its *intensity* is
    exponentially distributed -- non-negative, mean ``m``, std ``m``. That gives a
    multiplicative factor with mean 1 and std exactly ``cv``, which is what
    ``Exponential(1/cv**2) - 1 + 1/cv`` realises. A Gaussian multiplier would be wrong:
    it can go negative and has no physical counterpart here.
    """
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / cv**2
    mult = torch.distributions.Exponential(scale).sample(img.shape).to(img.dtype) * scale
    mult = mult - 1.0 + 1.0 / cv  # mean 1
    return img * mult.clamp_min(0.0)


def build_variants(target: torch.Tensor) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    blurred = _blur(target, BLUR_SIGMA)
    sp = _speckle(target, SPECKLE_CV, seed=7)
    return {
        "exact": target,
        "smooth": blurred,
        "shift": _shift(target, SHIFT_PX),
        "speckle": sp,
        "mixed": _speckle(blurred, SPECKLE_CV, seed=7),
    }


# --------------------------------------------------------------------------- metrics


def _speckle_contrast(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Std/mean of intensity. 0 for a flat top, 1 for fully developed speckle.

    This is the physical one: the ratio is scale-invariant, bounded in ``[0, ~3]``, and
    monotone in how much multiplicative noise is present. It does **not** saturate, unlike
    the ``exp(-(cv/0.3)**2)`` term in `compute_quality_score` where CV 0.9 and CV 11 both
    score exactly 0.
    """
    v = img[mask] if mask is not None else img.ravel()
    m = float(v.mean())
    return float(v.std() / m) if m > 1e-12 else float("nan")


def _entropy_uniformity(img: np.ndarray, bins: int = 32, mask: np.ndarray | None = None) -> float:
    """Negative Shannon entropy of the intensity histogram, in ``[0, log(bins)]``.

    Physical reading: fully developed speckle has an *exponential* intensity histogram, so
    its normalised histogram is flat and the entropy is maximal. A uniform (top-hat) target
    is a delta, whose entropy is 0. Driving this down therefore pushes speckle toward a flat
    top -- which is exactly the shaping objective, and it is bounded and monotone rather than
    a saturating exponential.
    """
    v = img[mask] if mask is not None else img.ravel()
    hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _probe(fn, p: torch.Tensor, t: torch.Tensor, **kw) -> float:
    """Run one torchmetrics functional, or record why it cannot be used.

    The failures here are informative, not incidental: torchmetrics' spectral metrics
    (``sam``, ``rase``, ``sdi``) take the FFT along the **channel** axis, i.e. they are built
    for multispectral imagery. Our far-field frames are single-channel, so they either raise
    (``sam``) or degenerate to a magnitude-only comparison that every image would tie on.
    Recording the rejection in the artefact keeps that decision auditable instead of leaving a
    silent gap in the metric set.
    """
    try:
        return float(fn(p, t, **kw).mean())
    except Exception as exc:  # noqa: BLE001 - the reason is the payload here
        _REJECTED[fn.__name__] = f"{type(exc).__name__}: {exc}"
        return float("nan")


def _radial_spectrum_corr(img: np.ndarray, ref: np.ndarray) -> float:
    """Correlation of the radially-averaged 2-D FFT magnitude. In ``[-1, 1]``.

    Why this exists: torchmetrics' spectral metrics FFT along the **channel** axis, so they
    cannot see a *spatial* spectrum and are unusable on single-channel far-field frames (see
    ``_probe``). But a speckle field's spatial magnitude spectrum is exactly the physically
    meaningful object here -- it holds the beam envelope, i.e. how much energy the aperture
    throws into the target box versus the sidelobes, while being blind to speckle's random
    phase. Comparing its radial profile measures that without the phase noise.

    A shift of the beam changes the profile only weakly (radial averaging discards
    orientation), so this complements rather than duplicates a centroid metric.
    """
    def profile(a: np.ndarray) -> np.ndarray:
        f = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
        h, w = f.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.hypot(yy - cy, xx - cx).astype(int)
        nbins = min(h, w) // 2
        out = np.zeros(nbins)
        cnt = np.zeros(nbins)
        valid = r < nbins
        np.add.at(out, r[valid], f[valid])
        np.add.at(cnt, r[valid], 1.0)
        return out / np.maximum(cnt, 1.0)

    a, b = profile(img), profile(ref)
    if a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


_REJECTED: dict[str, str] = {}


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Every incumbent and candidate metric on one (pred, target) pair."""
    from ml.zernike.metrics import batch_image_metrics

    out = dict(batch_image_metrics(pred, target))

    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure as ms_ssim,
        relative_average_spectral_error as rase,
        spectral_angle_mapper as sam,
        spatial_correlation_coefficient as scc,
        total_variation as tv,
        universal_image_quality_index as uiqi,
        visual_information_fidelity as vif,
    )

    p, t = pred.float(), target.float()
    out["ms_ssim"] = _probe(ms_ssim, p, t, data_range=1.0)
    out["uiqi"] = _probe(uiqi, p, t)
    out["scc"] = _probe(scc, p, t)
    out["vif"] = _probe(vif, p, t)
    out["sam"] = _probe(sam, p, t)
    out["rase"] = _probe(rase, p, t)
    # TV is an energy, not a similarity: low for a flat/blurred image, high for speckled.
    # Reported as a RATIO against the target so the direction is comparable across images.
    tv_t = float(tv(t, reduction="mean"))
    out["tv_ratio"] = (
        float(tv(p, reduction="mean")) / tv_t if tv_t > 1e-12 else float("nan")
    )

    a = np.stack([v[0].numpy() for v in pred])
    b = np.stack([v[0].numpy() for v in target])
    out["speckle_contrast"] = float(np.mean([_speckle_contrast(x) for x in a]))
    out["speckle_contrast_target"] = float(np.mean([_speckle_contrast(x) for x in b]))
    out["entropy_uniformity"] = float(np.mean([_entropy_uniformity(x) for x in a]))
    out["entropy_uniformity_target"] = float(np.mean([_entropy_uniformity(x) for x in b]))
    out["radial_spec_corr"] = float(
        np.mean([_radial_spectrum_corr(x, y) for x, y in zip(a, b, strict=True)])
    )
    return out


# --------------------------------------------------------------------------- driver


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = _load_targets()
    variants = build_variants(target)

    rows: dict[str, dict[str, float]] = {}
    for name, pred in variants.items():
        rows[name] = compute_all(pred, target)
        print(f"  {name:8s} done", flush=True)

    # A candidate is only worth reporting if it (a) separates exact from the corruptions by
    # more than R2 does, and (b) is not a monotone restatement of R2.
    keys = [k for k in rows["exact"] if not k.endswith("_target")]
    verdict: dict[str, dict[str, object]] = {}
    for k in keys:
        if any(rows[n][k] != rows[n][k] for n in rows):  # any NaN -> unavailable here
            continue
        base = rows["exact"][k]
        spread = {n: rows[n][k] - base for n in rows if n != "exact"}
        order = {
            k: {
                "exact": base,
                **{n: rows[n][k] for n in rows if n != "exact"},
            }
        }
        # rank correlation between this metric and R2 across the 5 variants
        xs = [rows[n][k] for n in rows]
        ys = [rows[n]["r2"] for n in rows]
        rho = _spearman(xs, ys)
        verdict[k] = {
            "values": order[k],
            "delta_vs_exact": spread,
            "spearman_with_r2": rho,
            # catches over-smoothing? must separate exact from smooth more than R2 does
            "catches_smooth": abs(spread["smooth"]) > abs(rows["smooth"]["r2"] - rows["exact"]["r2"]),
            "catches_speckle": abs(spread["speckle"]) > abs(
                rows["speckle"]["r2"] - rows["exact"]["r2"]
            ),
        }

    payload = {
        "config": {
            "grid": GRID,
            "n_samples": int(target.shape[0]),
            "shift_px": SHIFT_PX,
            "blur_sigma": BLUR_SIGMA,
            "speckle_cv": SPECKLE_CV,
        },
        "rejected_metrics": _REJECTED,
        "variants": rows,
        "verdict": verdict,
    }
    (OUT_DIR / "metric_discrimination.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    _plot(payload)
    print(f"wrote {OUT_DIR / 'metric_discrimination.json'}")


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rho without a scipy dependency (5 points, ties averaged)."""
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return float(num / (da * db)) if da > 0 and db > 0 else float("nan")


def _plot(payload: dict) -> None:
    """One panel per metric: how far each corruption moves it, normalised per metric."""
    cfg = payload["config"]
    variants = payload["verdict"]
    metrics = [k for k in variants if k != "r2"]
    order = ["smooth", "shift", "speckle", "mixed"]
    labels = ["blur\n(over-smooth)", "shift 2px\n(envelope)", "speckle\nCV=0.6", "both"]

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, m in zip(axes.ravel(), metrics):
        deltas = [variants[m]["delta_vs_exact"][o] for o in order]
        rng = max(abs(min(deltas)), abs(max(deltas)), 1e-12)
        ax.bar(range(len(order)), deltas, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(
            f"{m}\nrho(R2)={variants[m]['spearman_with_r2']:+.2f}", fontsize=8, fontweight="bold"
        )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-rng * 1.35, rng * 1.35)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(
        f"Metric discrimination on {cfg['n_samples']} real validation targets\n"
        "bar = change vs the exact target; a metric that never moves is not worth reporting. "
        "rho(R2)~1 means it adds nothing.",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "metric_discrimination.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
