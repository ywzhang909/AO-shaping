"""Freeze the trained predictor and optimise the *input phase* for a uniform 50x50 spot.

This inverts the modelling problem. Training fits ``phase -> spot`` so the phase
can be *predicted*; here the weights are **frozen** and the phase becomes the
variable, so the network is used as a differentiable surrogate of the bench and
the phase is synthesised by gradient descent through it::

    maximise  quality( surrogate(phi) )      subject to  phi on the SLM grid

Why bother, when the obvious alternative is to optimise the phase on the real
bench with SPGD? Because the surrogate is *calibrated*: ``far_field_padding=10``
plus the centre crop is what makes a predicted spot pixel-comparable with the
248 px camera window, and that calibration came from 1010 measured frames. A
50x50 target in surrogate space therefore corresponds to 50x50 camera pixels.
SPGD needs two camera reads per iteration and cannot use gradients; this needs
none at all.

**This optimises in surrogate space and that is the whole caveat.** The physics
model reaches R^2 ~ 0.88 on held-out pickles, so a phase that shapes the
surrogate to CV = 0.1 is not thereby a phase that shapes the bench to CV = 0.1.
Three checks are run here because of that, not as decoration:

1. **A real-hardware reference.** The best *measured* frame in the corpus, scored
   with the same 50x50 box, is the bar the surrogate must clear honestly. A
   synthesised phase that "beats" every real frame is exploiting model error.
2. **A cross-model check.** The phase is re-scored through an independently
   trained U-Net surrogate. Agreement means the solution is a property of the
   optics, not of one model's idiosyncrasies.
3. **Flat phase as the null baseline**, so the improvement is attributable.

Usage::

    python scripts/optimize_uniform_spot_phase.py --target-side 50
    python scripts/optimize_uniform_spot_phase.py --epochs 400 --lr 0.05

Outputs ``logs/uniform_spot_phase/``: the raw-radian phase (``.npy``, directly
sendable), the predicted spot, an optimisation history, and a comparison against
the real-hardware reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from loguru import logger  # noqa: E402

from ml.zernike.models import (  # noqa: E402
    Normalization,
    Observable,
    ZernikeAmpConfig,
    ZernikeAmpModel,
)

# Canonical shaping metrics. These are numpy and NOT differentiable -- that is
# deliberate and is the whole division of labour here: the optimiser minimises a
# smooth torch surrogate of the objective, and every *reported* number comes from
# the repo's canonical metric so it is comparable with the hardware runs.
from ao_shaping.utils.image.beam_metrics import (  # noqa: E402
    compute_quality_score,
    compute_square_metrics,
    measure_spot_diameter_cam,
)


def load_frozen_model(checkpoint: Path, device: torch.device) -> ZernikeAmpModel:
    """Rebuild the physics surrogate from a checkpoint and freeze it.

    The checkpoint stores the calibration explicitly (grid, padding, observable,
    normalisation), so the surrogate is reconstructed with exactly the geometry it
    was trained under rather than whatever the current defaults happen to be.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # The checkpoint stores plain strings; the config wants Literal unions. Narrow
    # explicitly (and let ZernikeAmpModel re-validate the choice) rather than
    # leaving the type open.
    observable = cast("Observable", str(blob["observable"]))
    normalization = cast("Normalization", str(blob["normalization"]))
    config = ZernikeAmpConfig(
        n_max=int(blob["n_max"]),
        grid=int(blob["grid"]),
        observable=observable,
        normalization=normalization,
        far_field_padding=int(blob["far_field_padding"]),
        center_crop=True,
    )
    model = ZernikeAmpModel(config)
    coefficients = blob["coefficients"].reshape(-1)
    if coefficients.numel() != model.K:
        raise SystemExit(
            f"checkpoint has {coefficients.numel()} coefficients but "
            f"n_max={config.n_max} needs K={model.K}"
        )
    with torch.no_grad():
        model.coefficients.copy_(coefficients)
    model.to(device).eval()
    model.requires_grad_(False)  # the weights are NOT the variable here
    return model


def box_slice(grid: int, side: int) -> slice:
    """Row/column slice of a ``side`` box centred in a ``grid`` grid.

    An even grid puts the optical axis between pixels, at ``(grid-1)/2``. For
    ``grid=64, side=50`` that is ``start=(64-50)//2=7``, spanning 7..56, whose
    centre is 31.5 -- the axis. Passing ``centre = start + side//2`` to
    :func:`compute_square_metrics` then reproduces exactly this box, so the
    optimisation target and the scoring region are the same pixels.
    """
    if not 0 < side <= grid:
        raise SystemExit(f"side must be in 1..{grid}, got {side}")
    start = (grid - side) // 2
    return slice(start, start + side)


def box_target(grid: int, side: int, device: torch.device) -> torch.Tensor:
    """Uniform 1-inside / 0-outside target the surrogate is driven towards."""
    target = torch.zeros(1, 1, grid, grid, device=device)
    window = box_slice(grid, side)
    target[:, :, window, window] = 1.0
    return target


def score_spot(intensity: np.ndarray, side: int) -> dict[str, float]:
    """Score a predicted spot with the canonical square metrics."""
    grid = intensity.shape[0]
    start = (grid - side) // 2
    metrics = compute_square_metrics(
        intensity.astype(np.float64),
        target_side=side,
        center=(float(start + side // 2), float(start + side // 2)),
    )
    metrics["quality"] = compute_quality_score(metrics)
    metrics["spot_diameter_px"] = measure_spot_diameter_cam(intensity.astype(np.float64))
    return metrics


def surrogate_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, kind: str
) -> torch.Tensor:
    """Smooth differentiable stand-in for the quality objective.

    Two kinds, because they optimise genuinely different things and the plateau
    turns out to matter:

    ``mse``
        Mean-squared error against the uniform box. Well-conditioned, no
        singularity, and the obvious intent -- fill the box evenly, empty
        everything else. But its optimum is *not* the reported metric's optimum,
        and it plateaus around CV 0.37.

    ``quality``
        Negative of a differentiable stand-in for the canonical score: the
        uniformity term ``exp(-k * CV)`` times the in-box fill, with the
        out-of-box mean charged as a penalty. This tracks
        :func:`compute_quality_score` directly. ``std / mean`` is undefined as
        the in-box mean approaches zero, which is exactly where a cold start
        sits, so the denominator carries an epsilon and the CV is clamped.

    Judging is always done by :func:`score_spot`, never by this value.
    """
    # One phase at a time, so the batch axis is a formality; drop it and index the
    # 2-D field directly (a boolean mask cannot index two axes of a 4-D tensor).
    field = prediction[0, 0]
    if kind == "mse":
        return torch.mean((prediction - target) ** 2)

    box = field[mask]
    outside = field[~mask]
    inside_mean = box.mean()
    inside_std = torch.sqrt(torch.clamp(box.var(unbiased=False), min=0.0))
    cv = inside_std / torch.clamp(inside_mean, min=1e-3)
    # Additive, matching `square_quality_score`'s structure (w_cv / w_ee).
    #
    # The additive form is not cosmetic. A *multiplicative* `uniformity * fill`
    # degenerates at the cold start: CV of an all-zero image is 0 (std is 0), so
    # "perfectly uniform" scores 1 while delivering no light at all, and a dark
    # box ties a hot 4x4 blob. Pairing the uniformity term with an *additive*
    # encircled-energy term is what breaks that tie -- darkness then scores 0 on
    # EE instead of winning on CV.
    uniformity = torch.exp(-2.0 * torch.clamp(cv, max=4.0))
    inside_sum = box.sum()
    outside_sum = outside.sum()
    encircled = inside_sum / torch.clamp(inside_sum + outside_sum, min=1e-8)
    score = 0.4 * uniformity + 0.6 * encircled
    return -score


def predict(model: torch.nn.Module, phase: torch.Tensor) -> torch.Tensor:
    """Forward a raw-radian phase through the frozen surrogate.

    The model's *parameters* carry ``requires_grad=False``, so no graph is built
    for them; the graph that remains runs phase -> spot, which is exactly the
    variable being optimised. Do not wrap this in ``no_grad`` -- that would sever
    the very gradient this whole script exists to compute.
    """
    cos = torch.cos(phase).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(phase).unsqueeze(0).unsqueeze(0)
    return model(cos, sin)


@torch.no_grad()
def predict_detached(model: torch.nn.Module, phase: torch.Tensor) -> torch.Tensor:
    """Gradient-free forward, for scoring a finished phase."""
    return predict(model, phase)


def optimise_phase(
    model: torch.nn.Module,
    grid: int,
    side: int,
    epochs: int,
    lr: float,
    device: torch.device,
    init: torch.Tensor | None,
    loss_kind: str = "mse",
) -> tuple[torch.Tensor, list[dict]]:
    """Gradient-descent the phase against the frozen surrogate."""
    target = box_target(grid, side, device)
    window = box_slice(grid, side)
    mask = torch.zeros(grid, grid, dtype=torch.bool, device=device)
    mask[window, window] = True
    phase = torch.zeros(grid, grid, device=device) if init is None else init.clone().to(device)
    phase.requires_grad_(True)
    optimiser = torch.optim.Adam([phase], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        prediction = predict(model, phase)
        loss = surrogate_loss(prediction, target, mask, loss_kind)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if epoch % max(1, epochs // 40) == 0 or epoch == epochs - 1:
            spot = score_spot(prediction.detach()[0, 0].cpu().numpy(), side)
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach()),
                "cv": spot["uniformity_cv"],
                "encircled_energy": spot["encircled_energy"],
                "quality": spot["quality"],
                "aspect_ratio": spot["aspect_ratio"],
            })
            logger.info(
                "epoch {:>4} loss={:.5f} CV={:.4f} EE={:.4f} quality={:.4f} AR={:.3f}",
                epoch, float(loss.detach()), spot["uniformity_cv"],
                spot["encircled_energy"], spot["quality"], spot["aspect_ratio"],
            )
    logger.info("optimised in {:.1f}s", time.perf_counter() - started)
    return phase.detach(), history


def hardware_reference(
    checkpoint_family: str, side: int, index_cache: str, limit: int | None = None
) -> dict[str, float] | None:
    """Best *measured* frame in the corpus under the same 50x50 box.

    **This is context, not a bar.** The ``slm_zernike_shaping`` corpus optimised
    RMS / PIB objectives and never targeted a square, so its best frame under a
    50x50 uniformity box scores CV ~ 1.3 -- worse than a flat phase. Beating it
    therefore says nothing about surrogate exploitation, and this function's
    verdict line must not claim otherwise. The meaningful reference is the
    repo's own square-shaping simulation (``generate_iterative_zernike_shaping_report.py``):
    Gerchberg-Saxton alone reaches CV ~ 0.41 and the full GS + freeform refinement
    pipeline reaches CV ~ 0.12. Those are the numbers to compare against.
    """
    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index

    try:
        index = build_hw_index(index_cache=index_cache).filter(families=[checkpoint_family])
    except Exception as exc:  # pragma: no cover - corpus may be absent
        logger.warning("no hardware reference available: {}", exc)
        return None
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=64), use_cache=True)
    total = len(dataset) if limit is None else min(limit, len(dataset))
    best: dict[str, float] | None = None
    for i in range(total):
        intensity = dataset[i]["image"][0].numpy().astype(np.float64)
        metrics = score_spot(intensity, side)
        if best is None or metrics["quality"] > best["quality"]:
            best = metrics | {"record": i}
    return best


def cross_check_unet(
    phase: torch.Tensor,
    side: int,
    epochs: int,
    device: torch.device,
    index_cache: str,
    family: str,
) -> dict[str, float] | None:
    """Re-score the synthesised phase through an **independently trained** U-Net.

    This is the check that separates "the optics were solved" from "one model's
    errors were exploited". The physics surrogate and this U-Net share only the
    corpus; their architectures, capacity and error patterns do not. If a phase
    that the physics model calls a clean 50x50 square is *also* called a
    reasonably uniform 50x50 square by a model that never saw it optimise
    anything, that is evidence the phase is a genuine solution.

    A negative result is equally informative: if the U-Net sees no square, the
    physics surrogate's confidence was self-deception.

    Returns ``None`` if the corpus is unavailable (offline), never raises.
    """
    try:
        from compare_unet_baseline import _inputs, build_model, fit, forward_for
        from compare_models_cv import build_folds
        from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
        from ml.zernike.train_amp import AmpTrainConfig, collect_split
    except Exception as exc:  # pragma: no cover
        logger.warning("cross-check unavailable (import): {}", exc)
        return None

    try:
        index = build_hw_index(index_cache=index_cache).filter(families=[family])
    except Exception as exc:
        logger.warning("cross-check unavailable (corpus): {}", exc)
        return None
    records = list(index.records)
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=64), use_cache=True)
    fold = build_folds(records, "file")[0]

    config = AmpTrainConfig(
        families=(family,), n_max=15, grid=64, epochs=epochs, lr=0.01, batch_size=64,
        use_wandb=False, save_checkpoint=False, seed=0, out_dir="logs/cross_check",
    )
    train_t = collect_split(dataset, fold.train, device)
    # The surrogate is trained on TRAIN folds only, so the cross-check never sees
    # a model that was fitted on the frame the physics model was fitted near.
    torch.manual_seed(0)
    model = build_model("unet", config, unet_features=[16, 32, 64, 128, 256]).to(device)
    model, _, _ = fit(model, train_t, config, device, forward_for("unet"))
    model.eval()

    with torch.no_grad():
        stacked = torch.stack([torch.cos(phase), torch.sin(phase)])[None]
        prediction = forward_for("unet")(model, stacked)
    spot = score_spot(prediction[0, 0].cpu().numpy(), side)

    flat = torch.zeros_like(phase)
    with torch.no_grad():
        flat_prediction = forward_for("unet")(model, torch.stack([torch.cos(flat), torch.sin(flat)])[None])
    flat_spot = score_spot(flat_prediction[0, 0].cpu().numpy(), side)
    logger.info(
        "cross-check U-Net (independent): flat CV={:.4f} -> synthesised CV={:.4f} (quality {:.4f})",
        flat_spot["uniformity_cv"], spot["uniformity_cv"], spot["quality"],
    )
    return spot | {
        "flat_cv": flat_spot["uniformity_cv"],
        "flat_quality": flat_spot["quality"],
        "prediction": prediction[0, 0].cpu().numpy().tolist(),
    }


def diagnose_extrapolation(
    phase: torch.Tensor, index_cache: str, family: str, n_max: int = 15
) -> dict[str, float]:
    """How far outside the surrogates' training manifold does this phase sit?

    The corpus phases came from ``n_max``-limited Zernike optimisation, so they are
    band-limited by construction. A freeform synthesised phase is not, and that
    matters for reading the cross-check: if the phase is far off-manifold then a
    disagreement between the two surrogates is *at least partly* an extrapolation
    artefact, and neither number can be trusted as the optical truth.

    Two numbers:

    * ``zernike_energy_fraction`` -- fraction of the phase's energy captured by a
      least-squares fit onto the ``n_max`` Zernike basis the corpus lives on.
      Corpus phases score high; ours does not.
    * ``high_frequency_fraction`` -- spectral power above a quarter of Nyquist.
    """
    from ml.zernike.models import ZernikeBasis

    grid = int(phase.shape[0])
    basis = ZernikeBasis(grid, n_max).as_tensor().numpy().astype(np.float64)
    flat = basis.reshape(basis.shape[0], -1).T  # (grid*grid, K)

    def fraction(candidate: np.ndarray) -> float:
        x = candidate.astype(np.float64).ravel()
        coefficients = np.linalg.lstsq(flat, x, rcond=None)[0]
        return float(np.sum((flat @ coefficients) ** 2) / max(np.sum(x**2), 1e-12))

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(phase.detach().cpu().numpy()))) ** 2
    lo = grid // 4
    window = (slice(lo, grid - lo), slice(lo, grid - lo))
    result = {
        "zernike_energy_fraction": fraction(phase.detach().cpu().numpy()),
        "high_frequency_fraction": float(spectrum[window].sum() / max(spectrum.sum(), 1e-12)),
    }

    try:
        from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index

        index = build_hw_index(index_cache=index_cache).filter(families=[family])
        dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=grid), use_cache=True)
        sample = [i for i in range(0, min(len(dataset), 303), 3)]
        corpus = [
            fraction(np.arctan2(dataset[i]["phase_sin"][0].numpy(), dataset[i]["phase_cos"][0].numpy()))
            for i in sample
        ]
        result["corpus_zernike_fraction_mean"] = float(np.mean(corpus))
        result["corpus_zernike_fraction_max"] = float(np.max(corpus))
        result["corpus_samples"] = len(corpus)
    except Exception as exc:  # pragma: no cover - corpus may be absent offline
        logger.warning("OOD corpus baseline unavailable: {}", exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="logs/zernike_amp_final/best_coefficients.pt")
    parser.add_argument("--target-side", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="logs/uniform_spot_phase")
    parser.add_argument("--index-cache", default="data/hw_index_cache.json")
    parser.add_argument("--family", default="slm_zernike_shaping")
    parser.add_argument("--skip-hardware-reference", action="store_true")
    parser.add_argument("--cross-check-unet", type=int, default=0, metavar="EPOCHS",
                        help="also score the phase through an independently trained U-Net (0 = skip)")
    parser.add_argument(
        "--loss", choices=["mse", "quality"], default="quality",
        help="differentiable objective: mse = fill the box, quality = track compute_quality_score",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(Path(args.checkpoint), device)
    logger.info(
        "frozen surrogate: n_max={} K={} padding={} normalisation={} (weights excluded from optimisation)",
        model.n_max, model.K, model.far_field_padding, model.normalization,
    )

    # Null baseline: no phase at all.
    flat = torch.zeros(args.grid, args.grid, device=device)
    flat_metrics = score_spot(predict_detached(model, flat)[0, 0].cpu().numpy(), args.target_side)
    logger.info(
        "baseline (flat phase): CV={:.4f} EE={:.4f} quality={:.4f}",
        flat_metrics["uniformity_cv"], flat_metrics["encircled_energy"], flat_metrics["quality"],
    )

    phase, history = optimise_phase(
        model, args.grid, args.target_side, args.epochs, args.lr, device,
        init=None, loss_kind=args.loss,
    )
    final = score_spot(predict_detached(model, phase)[0, 0].cpu().numpy(), args.target_side)

    reference = None
    if not args.skip_hardware_reference:
        reference = hardware_reference(args.family, args.target_side, args.index_cache)
        if reference is not None:
            logger.info(
                "best MEASURED frame under the same {}-px box: CV={:.4f} EE={:.4f} quality={:.4f} (record {})",
                args.target_side, reference["uniformity_cv"], reference["encircled_energy"],
                reference["quality"], reference["record"],
            )

    cross = None
    if args.cross_check_unet:
        cross = cross_check_unet(
            phase, args.target_side, args.cross_check_unet, device,
            args.index_cache, args.family,
        )

    ood = diagnose_extrapolation(phase, args.index_cache, args.family)
    logger.info("extrapolation: Zernike-span energy ours={:.3f} vs corpus mean={:.3f}; high-freq ours={:.3f}", ood["zernike_energy_fraction"], ood.get("corpus_zernike_fraction_mean", float('nan')), ood["high_frequency_fraction"])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Raw unwrapped radians -- the SLM driver is the only place mod 2pi happens.
    np.save(out_dir / "phase_rad.npy", phase.cpu().numpy().astype(np.float32))
    np.save(out_dir / "predicted_spot.npy", predict_detached(model, phase)[0, 0].cpu().numpy())

    summary = {
        "target_side": args.target_side,
        "grid": args.grid,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "surrogate": {
            "n_max": model.n_max, "K": model.K,
            "far_field_padding": model.far_field_padding,
            "normalization": model.normalization,
        },
        "baseline_flat": flat_metrics,
        "optimised": final,
        "hardware_reference": reference,
        "extrapolation": ood,
        "cross_check_unet": cross,
        "phase_stats": {
            "absmax_rad": float(phase.abs().max()),
            "p2p_rad": float(phase.max() - phase.min()),
            "std_rad": float(phase.std()),
        },
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("--- verdict ---")
    logger.info("CV      flat {:.4f} -> optimised {:.4f}", flat_metrics["uniformity_cv"], final["uniformity_cv"])
    logger.info("EE      flat {:.4f} -> optimised {:.4f}", flat_metrics["encircled_energy"], final["encircled_energy"])
    logger.info("quality flat {:.4f} -> optimised {:.4f}", flat_metrics["quality"], final["quality"])
    if reference is not None:
        logger.info(
            "for context, best MEASURED frame in this corpus scores quality={:.4f} -- "
            "but it never targeted a square, so this is NOT a valid bar "
            "(see generate_iterative_zernike_shaping_report.py: GS ~0.41 CV, full refinement ~0.12 CV)",
            reference["quality"],
        )
    logger.info("wrote {}", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
