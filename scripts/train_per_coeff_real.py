"""Train a separate single-output regressor for each real-data Zernike coefficient.

Splits the 3 real dual-spot runs at RUN level (leakage-free, matches the
dataset docstring): train=20260414_171241 (333 labeled), val=20260402_164456
(48 labeled), test=20260402_155508 (94 labeled). Each coefficient index
``i`` (metadata (n,m) order, 0 = (1,-1) x-tilt) gets its own resnet18 with a
single regression output, evaluated on the held-out 155508 run.

Usage (parallel halves on two GPUs):
    PYTHONPATH=src .venv/bin/python scripts/train_per_coeff_real.py \
        --start 0 --end 33 --device cuda:0 --epochs 150
    PYTHONPATH=src .venv/bin/python scripts/train_per_coeff_real.py \
        --start 33 --end 65 --device cuda:1 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, "src")

from ml.zernike.models import build_model  # noqa: E402
from ml.zernike_prediction.dataset import create_zernike_loaders  # noqa: E402
from ml.zernike_prediction.metrics import coefficient_names  # noqa: E402
from ml.zernike_prediction.trainer import evaluate_regressor, train_regressor  # noqa: E402

RUNS = ["20260402_155508", "20260402_164456", "20260414_171241"]
N_COEFFS = 65
INPUT_MODE = "combined"  # dual-spot: daheng focus + miicam pupil
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 42
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, required=True, help="First coefficient index (inclusive).")
    p.add_argument("--end", type=int, required=True, help="Last coefficient index (exclusive, <=65).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--data-root", type=str, default="data/slm_dual_spot")
    p.add_argument("--checkpoint-root", type=str, default="data/zernike_pred/checkpoints/real_per_coeff")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.start < args.end <= N_COEFFS):
        raise SystemExit(f"invalid range: [{args.start}, {args.end}) not within [0, {N_COEFFS})")
    names = coefficient_names(10)  # 65 'nXmY' labels in metadata order
    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i in range(args.start, args.end):
        name = names[i]
        out_dir = ckpt_root / f"c{i:02d}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "eval_summary.json"
        if summary_path.exists():
            logger.info("coeff {:>2} {} already evaluated; skipping", i, name)
            results.append(json.loads(summary_path.read_text()))
            continue

        loaders = create_zernike_loaders(
            args.data_root, batch_size=BATCH_SIZE, val_split=0.15, test_split=0.5,
            seed=SEED, input_mode=INPUT_MODE, run_ids=RUNS, split_mode="run",
            num_workers=4, skip_unlabeled=True, target_indices=[i], n_target=1,
        )
        model = build_model("resnet18", in_channels=2, n_coeffs=1, device=args.device)

        logger.info("coeff {:>2} {}: training ({} train / {} val / {} test)", i, name,
                    loaders["split_sizes"]["train"], loaders["split_sizes"]["val"],
                    loaders["split_sizes"]["test"])
        result = train_regressor(
            model, loaders["train"], loaders["val"],
            epochs=args.epochs, lr=LR, weight_decay=WEIGHT_DECAY, device=args.device,
            model_name="resnet18", input_mode=INPUT_MODE, run_id=f"c{i:02d}_{name}",
            use_wandb=False, seed=SEED, checkpoint_dir=str(out_dir),
            early_stop_patience=60,
        )

        best_path = Path(result["best_model_path"])
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
        ev = evaluate_regressor(model, loaders["test"], args.device, out_dir=str(out_dir),
                                prefix="eval", return_arrays=True)
        tm = ev["metrics"]
        # Save raw test predictions + truths for downstream per-coefficient analysis.
        np.save(out_dir / "test_pred.npy", ev["pred"])
        np.save(out_dir / "test_true.npy", ev["true"])

        row = {
            "index": i,
            "name": name,
            "best_val_mae": float(result["best_val_mae"]),
            "best_epoch": int(result["best_epoch"]),
            "test_mae": float(tm["mae"]),
            "test_rmse": float(tm["rmse"]),
            "test_r2": float(tm["r2"]) if np.isfinite(tm["r2"]) else None,
            "n_test": int(tm["n_samples"]),
        }
        summary_path.write_text(json.dumps(row, indent=2))
        results.append(row)
        logger.info("coeff {:>2} {}: test_mae {:.4f} r2 {} (best epoch {})", i, name,
                    row["test_mae"], row["test_r2"], row["best_epoch"])

    out = ckpt_root / f"results_{args.start:02d}_{args.end:02d}.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote {} results to {}", len(results), out)


if __name__ == "__main__":
    main()