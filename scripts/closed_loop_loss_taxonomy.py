"""Score the proposed closed-loop loss taxonomy against what this repo already measures.

The proposal lists seven loss families for closed-loop beam shaping under physical error.
Five are already implemented here or already measured *against*, and one of them predicts a
specific failure that has already been observed. This script measures rather than argues, on
three axes the proposal does not state:

1. **Discrimination** -- does the term separate the failure modes this model actually
   exhibits? Measured over three corruption families in *opposite* directions
   (over-smoothed, over-speckled, energy displaced), because a term that only catches one
   direction is a directional prior, not a fidelity measure.
2. **Degenerate safety** -- can it be won by destroying the signal? Every unanchored term
   scores exactly 0 on an empty ROI, and optimising ``uniformity`` alone drove encircled
   energy to 0.002 on hardware. A term that fails this is unusable no matter how well it
   discriminates.
3. **Redundancy and coverage** -- the decisive axis. If a candidate's per-sample values
   correlate with an incumbent term's, it adds a knob, not information. And a candidate is
   only worth its complexity if it moves *more* than the best incumbent on at least one
   corruption. This is the test that killed the ellipse loss (0/3 paired, monotonically
   worse with weight), so it is applied here before anything is adopted.

The phase-domain smoothness term is the one genuinely new axis: every term currently in
``LossConfig`` is image-domain, while the proposal's smoothness constraint acts on the SLM
phase. That distinction matters because the repo has a measured symptom of an
unimplementable phase -- freeform refinement destroyed the GS solution in 9/9 ROIs
(-0.2779) while Zernike refinement improved it in 78/90, and freeform carries 576 degrees of
freedom against the model's 135 Zernike modes.

Run::

    python scripts/closed_loop_loss_taxonomy.py

Writes ``report/loss_defects/closed_loop_loss_taxonomy.json``.
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
SIDE = 32
EPS = 1e-12

#: ROI weight for the weighted-MSE family: 1.0 inside the target, 0.05 in the background.
#: The ratio matters more than the exact value -- it is what "high weight on target, low on
#: background" means numerically.
BG_WEIGHT = 0.05


# --------------------------------------------------------------------------- data


def _load() -> tuple[torch.Tensor, torch.Tensor]:
    """Real validation targets plus the ROI mask, same family as every other bench."""
    from scripts.compare_unet_baseline import _peak_normalise, _select_records
    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
    from ml.zernike.losses import roi_mask
    from ml.zernike.train_amp import AmpTrainConfig, collect_split

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=15, grid=GRID, epochs=1, lr=1e-3,
        batch_size=64, max_train=64, max_val=N_SAMPLES, beam_samples=N_SAMPLES,
        use_wandb=False, save_checkpoint=False, seed=0,
    )
    _, val = _select_records(dataset, cfg)
    target = _peak_normalise(collect_split(dataset, val, torch.device("cpu"))["target"].clone())
    mask = roi_mask((GRID, GRID), (GRID / 2.0, GRID / 2.0), "rectangle", float(SIDE), 1.0).float()
    return target, mask


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
    """Fully developed speckle: exponential intensity, so the multiplier's CV is exactly ``cv``."""
    scale = 1.0 / cv**2
    mult = torch.distributions.Exponential(scale).sample(img.shape).to(img.dtype) * scale
    return img * (mult - 1.0 + 1.0 / cv).clamp_min(0.0)


# --------------------------------------------------------------------------- candidates
# Every candidate returns a per-sample loss tensor, ``(B,)``, lower is better.


def _w(m: torch.Tensor) -> torch.Tensor:
    return m.to(device=m.device, dtype=m.dtype) if m.dtype.is_floating_point else m.float()


def mse_incumbent(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Baseline family 1, unweighted."""
    return (pred - ref).pow(2).flatten(1).mean(dim=1)


def weighted_mse(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Family 1, the proposal's weighted variant: ROI 1.0, background ``BG_WEIGHT``."""
    m = mask if mask.dim() == 4 else mask.view(1, 1, *mask.shape)
    w = m + BG_WEIGHT * (1.0 - m)
    return ((pred - ref).pow(2) * w).flatten(1).sum(dim=1) / w.flatten(1).sum(dim=1).clamp_min(EPS)


def ssim_loss(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Family 1: ``1 - SSIM``. The proposal's claim is robustness to global multiplicative error."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim

    return 1.0 - ssim(pred, ref, data_range=1.0)


def gradient_loss(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Family 2, gradient domain: ``L1`` on the intensity gradient difference.

    The proposal's motivation is that high spatial frequencies are what SLM crosstalk and
    alignment error corrupt, so constraining them should avoid fitting unphysical detail.
    Note this is a *one-sided-in-space but two-sided-in-value* term: it penalises a
    gradient mismatch in either direction, unlike TV on the ROI mean.
    """
    def grad(x: torch.Tensor) -> torch.Tensor:
        dh = x[:, :, 1:, :] - x[:, :, :-1, :]
        dw = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dh.abs().flatten(1).mean(dim=1) + dw.abs().flatten(1).mean(dim=1)

    return (grad(pred) - grad(ref)).abs()


def peak_penalty(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Family 3: exceedance above the brightest pixel the reference corpus ever shows.

    The threshold is *derived from the reference*, not hand-set: the proposal's
    ``I_damage_threshold`` has no defensible value in simulation, and picking one
    arbitrarily would make the measurement unfalsifiable. Using the target's own maximum
    means the term asks "is any prediction hotter than a real measurement ever was", which
    is the safety question the closed loop can actually answer offline.
    """
    thr = ref.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp(pred.amax(dim=(-2, -1)) - thr.squeeze(-1).squeeze(-1), min=0.0)


def energy_conservation(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Family 3: ``|sum(I_pred) - sum(I_ref)|`` -- the proposal's loss-path-drift alarm.

    Included specifically to measure whether it survives this repo's normalisation contract.
    ``image_mode='abs255'`` and the model's ``normalization='peak'`` both remove absolute
    scale, so the total may already be pinned; if so the term is identically ~0 and cannot
    detect anything, which is a result worth having rather than assuming.
    """
    return (pred.flatten(1).sum(dim=1) - ref.flatten(1).sum(dim=1)).abs()


# ---- phase domain: the genuinely new axis ----


def _correction_phase(n_max: int = 15, seed: int = 0) -> torch.Tensor:
    """The model's own correction phase -- differentiable w.r.t. its coefficients."""
    from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

    torch.manual_seed(seed)
    m = ZernikeAmpModel(ZernikeAmpConfig(n_max=n_max, grid=GRID))
    with torch.no_grad():
        m.coefficients.add_(torch.randn_like(m.coefficients) * 0.3)
    return m.correction_phase()


def phase_smoothness(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Family 2, smoothness on the SLM phase itself: ``||grad phi||_1``, in radians.

    Every other term in ``LossConfig`` is image-domain. This one constrains the *command*,
    which is the quantity SLM crosstalk and the spatial bandwidth product actually limit. It
    is a property of the model, not of a sample, so it returns the same value per batch --
    hence the expansion to ``(B,)`` for the common per-sample bookkeeping.
    """
    phi = _correction_phase()
    dh = (phi[1:, :] - phi[:-1, :]).abs().mean()
    dw = (phi[:, 1:] - phi[:, :-1]).abs().mean()
    per = dh + dw
    return per.expand(pred.shape[0])


def phase_modulation_depth(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Peak-to-valley of the correction phase -- how hard the LCOS is being driven.

    The other half of the realizability constraint: TV limits *local* rate of change, this
    limits total excursion. A phase can be smooth and still exceed the panel's 2pi range.
    """
    phi = _correction_phase()
    pv = phi.amax() - phi.amin()
    return pv.expand(pred.shape[0])


INCUMBENTS = {
    "mse": mse_incumbent,
    "uniformity": None,  # filled from the canonical implementation below
    "w_speckle": None,
    "w_ellipse": None,
}
CANDIDATES = {
    "weighted_mse": weighted_mse,
    "ssim_loss": ssim_loss,
    "gradient_loss": gradient_loss,
    "peak_penalty": peak_penalty,
    "energy_conservation": energy_conservation,
    "phase_smoothness": phase_smoothness,
    "phase_modulation_depth": phase_modulation_depth,
}


def _spearman(a: list[float], b: list[float]) -> float:
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


def _flat(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().flatten()]


# --------------------------------------------------------------------------- driver


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref, mask = _load()

    # Four corruptions in three directions. 'displace' moves energy out of the ROI without
    # touching its texture, which is the failure an ROI-shape term cannot see by construction.
    cases = {
        "exact": ref,
        "over_smooth": _blur(ref, 1.5),
        "over_speckle": _speckle(ref, 0.9, seed=3),
        "displace": torch.roll(ref, shifts=6, dims=-1),
    }
    degenerate = {"empty_roi": ref * (1.0 - mask), "all_dark": torch.zeros_like(ref)}

    # Canonical incumbents, straight from the shipped implementation -- not re-derived, so
    # the redundancy axis compares against what training actually uses.
    from ml.zernike.losses import ellipse_gap_term, speckle_detail_term, uniformity_term

    def _uniformity(p: torch.Tensor, r: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return 1.0 - uniformity_term(p, m)

    def _wspeckle(p: torch.Tensor, r: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return speckle_detail_term(p, r, m)["tv_ratio"]

    def _wellipse(p: torch.Tensor, r: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return ellipse_gap_term(p, r, m)["ellipse"]

    inc = {
        "mse": mse_incumbent,
        "uniformity": _uniformity,
        "w_speckle": _wspeckle,
        "w_ellipse": _wellipse,
    }
    every = {**{f"incumbent:{k}": v for k, v in inc.items()}, **CANDIDATES}

    # Per-sample values for the redundancy axis. These must come from a case whose
    # per-sample values actually VARY: scored on `exact` every anchored term returns
    # ~0 for every sample, zero variance, and Spearman is undefined (NaN) for all of
    # them. `over_speckle` is the natural choice -- it is the corruption this model
    # is worst at, so its terms disagree the most, which is exactly what a
    # redundancy check needs to see.
    baseline = {k: _flat(fn(cases["over_speckle"], ref, mask)) for k, fn in every.items()}

    rows: dict[str, dict[str, float]] = {}
    for name, fn in every.items():
        row: dict[str, float] = {}
        for cname, pred in {**cases, **degenerate}.items():
            row[cname] = float(fn(pred, ref, mask).mean())
        rows[name] = row

    real = ["over_smooth", "over_speckle", "displace"]
    # Per-term scale, used to make movement comparable across terms.
    #
    # Two earlier normalisations were both wrong and are worth recording:
    # 1. dividing by the value at the exact case -- every term here is *anchored*, so
    #    that value is 0 by construction and every ratio is inf;
    # 2. comparing raw deltas -- `energy_conservation` lives on an O(100) scale and
    #    `w_speckle` on O(1), so the larger-scale term wins on magnitude alone.
    # The only well-defined per-term scale is its own observed range across all cases,
    # so movement is reported as a fraction of that: "how much of this loss's own dynamic
    # range does this corruption consume". 1.0 means the corruption saturates the term.
    def scale_of(name: str) -> float:
        return max(abs(rows[name][c]) for c in rows[name])

    inc_moves = {}
    for c in real:
        best, arg = 0.0, None
        for k in inc:
            full = f"incumbent:{k}"
            s = scale_of(full)
            rel = abs(rows[full][c] - rows[full]["exact"]) / s if s > 1e-12 else 0.0
            if rel > best:
                best, arg = rel, k
        inc_moves[c] = {"best_relative": best, "argmax": arg}

    verdict: dict[str, dict[str, object]] = {}
    for name in CANDIDATES:
        row = rows[name]
        base = row["exact"]
        is_phase = name.startswith("phase_")

        # A phase-domain term is a property of the MODEL, not of a sample. Every
        # corruption leaves it bit-identical, so on this bench it necessarily shows
        # discrimination 0 and "unsafe" (its value never rises on an empty ROI).
        # Recording that as a verdict would be an artefact of the harness, not a
        # finding, so it is reported as `not_screenable_here` with the reason and
        # excluded from ranking. Judging it needs a training run, not a corruption
        # sweep -- see `phase_realizability_train.py`.
        if is_phase:
            verdict[name] = {
                "values": row,
                "discrimination": 0.0,
                "degenerate_safe": None,
                "screenable_here": False,
                "not_screenable_reason": (
                    "phase-domain term: invariant to the input image by construction, so a "
                    "corruption sweep cannot score it; requires a paired training run"
                ),
                "covers": [],
                "phase_domain": True,
            }
            continue

        s = scale_of(name)
        discrim = (
            float(np.mean([abs(row[c] - base) / s for c in real])) if s > 1e-12 else 0.0
        )
        safe = bool(row["empty_roi"] > base and row["all_dark"] > base)
        # Coverage: on which corruptions does this beat every incumbent, relatively?
        covers = [
            c for c in real
            if s > 1e-12
            and abs(row[c] - base) / s > inc_moves[c]["best_relative"]
        ]
        # Redundancy: highest |Spearman| against any incumbent.
        rho = {k: _spearman(baseline[name], baseline[f"incumbent:{k}"]) for k in inc}
        worst = max((abs(v) for v in rho.values() if v == v), default=float("nan"))
        verdict[name] = {
            "values": row,
            "discrimination": discrim,
            "own_scale": s,
            "degenerate_safe": safe,
            "empty_roi_margin": float(row["empty_roi"] - base),
            "all_dark_margin": float(row["all_dark"] - base),
            "covers": covers,
            "max_abs_spearman_vs_incumbent": float(worst),
            "spearman_vs_incumbent": rho,
            "screenable_here": True,
            "phase_domain": False,
        }

    payload = {
        "config": {"grid": GRID, "n_samples": int(ref.shape[0]), "roi_side": SIDE,
                   "bg_weight": BG_WEIGHT},
        "cases": list(cases) + list(degenerate),
        "incumbent_best_movement": inc_moves,
        "scores": rows,
        "verdict": verdict,
    }
    (OUT_DIR / "closed_loop_loss_taxonomy.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'closed_loop_loss_taxonomy.json'}\n")

    print("incumbent best RELATIVE movement per corruption (the bar to clear):")
    for c, v in inc_moves.items():
        print(f"  {c:14s} {v['best_relative']:.4f}  via {v['argmax']}")
    print()
    hdr = (
        f"{'candidate':24s} {'rel_discrim':>11s} {'safe':>6s} {'rho_max':>8s}  covers"
    )
    print(hdr)
    print("-" * len(hdr))
    ranked = sorted(
        (n for n in verdict if verdict[n]["screenable_here"]),
        key=lambda n: -verdict[n]["discrimination"],
    )
    for name in ranked:
        v = verdict[name]
        print(
            f"{name:24s} {v['discrimination']:11.4f} {str(v['degenerate_safe']):>6s} "
            f"{v['max_abs_spearman_vs_incumbent']:8.2f}  {','.join(v['covers']) or '-'}"
        )
    print()
    for name in sorted(n for n in verdict if not verdict[n]["screenable_here"]):
        print(f"{name:24s} NOT SCREENABLE HERE -- {verdict[name]['not_screenable_reason']}")


if __name__ == "__main__":
    main()