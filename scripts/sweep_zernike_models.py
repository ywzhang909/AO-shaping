"""Persist the model sweep results that the Zernike report is drawn from.

The report's figures must be regenerable from saved artefacts rather than from
numbers typed into prose, so this script re-runs every sweep under the **powered**
protocol (leave-one-pickle-out, 10 grouped folds) and writes one JSON.

Everything here is grouped by source pickle, so the objective-mixture lottery that
made the original single split unusable (see ``compare_models_cv``) cannot recur.

What is swept, and why each axis exists:

* **physics ``(n_max, lr)`` grid** -- ``n_max`` is the only large lever on the
  physics model, and ``lr`` is documented non-additive with it, so they are swept
  *jointly*. A coordinate sweep gets the wrong cell.
* **unet ``(features, epochs, lr)``** -- the U-Net configuration was originally
  chosen on the underpowered single split, so it is re-verified here.
* **final CV at the tuned configs** -- the headline table, with the per-objective
  breakdown retained.

Usage::

    python scripts/sweep_zernike_models.py                 # full, ~30 min on one GPU
    python scripts/sweep_zernike_models.py --quick         # 2 folds, smoke only

Writes ``logs/zernike_amp_sweep.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from loguru import logger  # noqa: E402

from compare_models_cv import build_folds, exact_sign_flip_p, objective_of  # noqa: E402
from compare_unet_baseline import (  # noqa: E402
    _inputs,
    _peak_normalise,
    build_model,
    fit,
    forward_for,
)
from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index  # noqa: E402
from ml.zernike.metrics import batch_image_metrics  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

METRICS = ("mse", "r2", "ssim", "psnr", "nrmse")

# (n_max, lr) -- n_max spans the useful range at the default lr, then lr is probed
# at the two n_max values where it could plausibly interact.
PHYSICS_GRID = [
    (11, 0.01), (15, 0.01), (20, 0.01), (25, 0.01), (30, 0.01),
    (15, 0.02), (20, 0.02), (20, 0.005),
]
UNET_GRID = [
    (50, 0.01, [16, 32, 64, 128, 256]),
    (100, 0.01, [16, 32, 64, 128, 256]),
    (100, 0.005, [16, 32, 64, 128, 256]),
    (50, 0.005, [16, 32, 64, 128, 256]),
    (100, 0.01, [24, 48, 96, 192, 384]),
]
PHYSICS_BEST = (20, 0.02)
UNET_BEST = (50, 0.01, [16, 32, 64, 128, 256])


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = batch_image_metrics(prediction, target)
    return {key: float(values[key]) for key in METRICS}


def _predict(model, stacked: torch.Tensor, forward) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, stacked.shape[0], 256):
            stop = min(start + 256, stacked.shape[0])
            chunks.append(forward(model, stacked[start:stop]))
    return torch.cat(chunks)


def run_config(
    folds, dataset, records, device, model_name: str, *, n_max: int, lr: float,
    epochs: int, features: list[int] | None, seed: int = 0,
) -> dict:
    """Train one configuration on every fold and collect per-fold metrics."""
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=n_max, grid=64, epochs=epochs,
        lr=lr, batch_size=64, use_wandb=False, save_checkpoint=False, seed=seed,
        out_dir="logs/zernike_amp_sweep",
    )
    per_fold: list[dict] = []
    started = time.perf_counter()
    for fold in folds:
        train_t = collect_split(dataset, fold.train, device)
        val_t = collect_split(dataset, fold.val, device)
        torch.manual_seed(seed)
        model = build_model(
            model_name, cfg, residual_width=32, unet_features=features
        ).to(device)
        model, seconds, _ = fit(model, train_t, cfg, device, forward_for(model_name))
        prediction = _predict(model, _inputs(val_t), forward_for(model_name))
        target = _peak_normalise(val_t["target"].clone())
        groups = [objective_of(records[p].path) for p in fold.val]
        by_objective: dict[str, dict[str, float]] = {}
        for name in sorted(set(groups)):
            mask = torch.tensor([g == name for g in groups])
            if bool(mask.any()):
                by_objective[name] = _metrics(prediction[mask], target[mask])
        per_fold.append(
            {
                "fold": fold.index,
                "label": fold.label,
                "pooled": _metrics(prediction, target),
                "by_objective": by_objective,
                "seconds": seconds,
            }
        )
        del model, prediction, target, train_t, val_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        metric: {
            "mean": statistics.fmean(f["pooled"][metric] for f in per_fold),
            "std": statistics.stdev(f["pooled"][metric] for f in per_fold) if len(per_fold) > 1 else 0.0,
            "per_fold": [f["pooled"][metric] for f in per_fold],
        }
        for metric in METRICS
    }
    objective_names = sorted({n for f in per_fold for n in f["by_objective"]})
    summary["by_objective"] = {
        name: {
            metric: statistics.fmean(
                f["by_objective"][name][metric]
                for f in per_fold if name in f["by_objective"]
            )
            for metric in METRICS
        }
        for name in objective_names
    }
    return {
        "model": model_name, "n_max": n_max, "lr": lr, "epochs": epochs,
        "features": features, "folds": per_fold, "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }


def paired(a: dict, b: dict) -> dict:
    """Paired per-fold differences ``a - b`` for every metric.

    The paired spread is what makes these effects resolvable at all: the
    between-fold spread of R^2 is ~0.08 while the paired spread is ~0.01, so an
    unpaired comparison cannot see a change of the size being claimed.
    """
    out = {}
    for metric in METRICS:
        first = a["summary"][metric]["per_fold"]
        second = b["summary"][metric]["per_fold"]
        diff = [x - y for x, y in zip(first, second)]
        sd = statistics.stdev(diff) if len(diff) > 1 else 0.0
        out[metric] = {
            "mean_diff": statistics.fmean(diff),
            "std_diff": sd,
            "cohens_dz": statistics.fmean(diff) / sd if sd > 0 else float("inf"),
            "p_signflip": exact_sign_flip_p(diff),
            "per_fold": diff,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="logs/zernike_amp_sweep.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="2 folds, smoke test only")
    parser.add_argument(
        "--final-only", action="store_true",
        help="reuse the grids already in --out and only redo the tuned final block",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=["slm_zernike_shaping"]
    )
    records = list(index.records)
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=64), use_cache=True)
    folds = build_folds(records, "file")
    if args.quick:
        folds = folds[:2]
    logger.info("sweeping on {} folds ({} train / {} val)", len(folds),
                len(folds[0].train), len(folds[0].val))

    report: dict = {
        "folds": len(folds),
        "protocol": "leave-one-pickle-out (grouped by source file)",
        "physics_grid": [], "unet_grid": [], "final": {},
    }

    if args.final_only:
        if not Path(args.out).exists():
            raise SystemExit(f"--final-only needs an existing {args.out}")
        report = json.loads(Path(args.out).read_text(encoding="utf-8"))
        logger.info("reusing {} grids from {}", len(report["physics_grid"]), args.out)
    else:
        logger.info("--- physics (n_max, lr) joint grid ---")
        for n_max, lr in PHYSICS_GRID:
            entry = run_config(folds, dataset, records, device, "physics",
                               n_max=n_max, lr=lr, epochs=50, features=None)
            report["physics_grid"].append(entry)
            s = entry["summary"]
            logger.info("  n_max={:<3} lr={:<6} r2={:+.4f}+-{:.4f} ssim={:.4f}",
                        n_max, lr, s["r2"]["mean"], s["r2"]["std"], s["ssim"]["mean"])

        logger.info("--- unet (features, epochs, lr) grid ---")
        for epochs, lr, features in UNET_GRID:
            entry = run_config(folds, dataset, records, device, "unet",
                               n_max=15, lr=lr, epochs=epochs, features=features)
            report["unet_grid"].append(entry)
            s = entry["summary"]
            logger.info("  unet{:<3} ep={:<4} lr={:<6} r2={:+.4f}+-{:.4f} ssim={:.4f}",
                        features[0], epochs, lr, s["r2"]["mean"], s["r2"]["std"], s["ssim"]["mean"])

    logger.info("--- final CV at the tuned configurations ---")
    # physics and hybrid share the tuned physics setting: the hybrid *wraps* the
    # physics model, so holding it at the untuned (n_max=15, lr=0.01) would make
    # the comparison unfair in the physics model's favour.
    tuned_physics = run_config(folds, dataset, records, device, "physics",
                               n_max=PHYSICS_BEST[0], lr=PHYSICS_BEST[1],
                               epochs=50, features=None)
    hybrid = run_config(folds, dataset, records, device, "hybrid",
                        n_max=PHYSICS_BEST[0], lr=PHYSICS_BEST[1],
                        epochs=50, features=None)
    unet = run_config(folds, dataset, records, device, "unet",
                      n_max=15, lr=UNET_BEST[1], epochs=UNET_BEST[0],
                      features=UNET_BEST[2])
    report["final"] = {"physics": tuned_physics, "hybrid": hybrid, "unet": unet}
    report["configs"] = {
        "physics": {"n_max": PHYSICS_BEST[0], "lr": PHYSICS_BEST[1], "epochs": 50},
        "hybrid": {"n_max": PHYSICS_BEST[0], "lr": PHYSICS_BEST[1], "epochs": 50,
                   "residual_width": 32},
        "unet": {"epochs": UNET_BEST[0], "lr": UNET_BEST[1], "features": UNET_BEST[2]},
    }
    report["paired"] = {
        f"{a}_minus_{b['model']}": paired(report["final"][a], b)
        for a, b in (("physics", hybrid), ("physics", unet), ("hybrid", unet))
    }
    for name, entry in report["final"].items():
        logger.info("  {:<8} r2={:+.4f}+-{:.4f} ssim={:.4f}+-{:.4f}",
                    name, entry["summary"]["r2"]["mean"], entry["summary"]["r2"]["std"],
                    entry["summary"]["ssim"]["mean"], entry["summary"]["ssim"]["std"])
    for pair, entry in report["paired"].items():
        logger.info("  {:<24} r2 {:+.4f} (p={:.4f})  ssim {:+.4f} (p={:.4f})",
                    pair, entry["r2"]["mean_diff"], entry["r2"]["p_signflip"],
                    entry["ssim"]["mean_diff"], entry["ssim"]["p_signflip"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
