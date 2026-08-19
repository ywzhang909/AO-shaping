"""Train a JOINT multi-output Zernike regressor on real dual-spot data.

Extends ``train_per_coeff_real.py`` (which trains 65 separate single-output
models) with a shared-feature joint model — the "low-order subset + joint
regression" mitigation for the small-sample regime:

  - ``--target-indices 0,1,...,9`` keeps the 10 lowest-order coefficients and
    outputs them jointly from one resnet18 (n_out = len(indices)).
  - Split mode selects the evaluation regime:
      * ``run``    : whole-run split (train=0414, val=164456, test=155508) —
                     leakage-free but the 0402 daheng focus field is ~flat
                     (no spot), so this measures cross-run generalization.
      * ``sample`` : 0414-only per-sample split (WITH optional quality
                     filtering) — the in-run learnability regime.

Quality filtering (direction 2): when ``--min-dpeak`` is given, samples whose
daheng focus peak falls below the threshold are dropped from ALL splits before
training (e.g. daheng peak < 200 removes weak/dim-spot outliers; the 0414
p50 peak is 255).

Usage (parallel GPUs):
    PYTHONPATH=src .venv/bin/python scripts/train_joint_real.py \
        --target-indices 0,1,2,3,4,5,6,7,8,9 --split-mode run \
        --out-root data/zernike_pred/checkpoints/real_joint10_run \
        --device cuda:0 --epochs 150
    PYTHONPATH=src .venv/bin/python scripts/train_joint_real.py \
        --target-indices 0,1,2,3,4,5,6,7,8,9 --split-mode sample \
        --min-dpeak 200 --out-root data/zernike_pred/checkpoints/real_joint10_0414filter \
        --device cuda:1 --epochs 150
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
from ml.zernike_prediction.dataset import (
    ZernikeDualDataset,
    _split_indices,
    create_zernike_loaders,
)  # noqa: E402
from ml.zernike_prediction.metrics import coefficient_names  # noqa: E402
from ml.zernike_prediction.trainer import evaluate_regressor, train_regressor  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

RUNS = ["20260402_155508", "20260402_164456", "20260414_171241"]
INPUT_MODE = "combined"  # dual-spot: daheng focus + miicam pupil
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 42
BATCH_SIZE = 32


class _FilteredDataset(ZernikeDualDataset):
    """ZernikeDualDataset that drops samples whose daheng focus peak < min_dpeak.

    Applied BEFORE splitting so train/val/test all share the filtered pool.
    """

    def __init__(
        self,
        *args: object,
        min_dpeak: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if min_dpeak is None:
            return
        import numpy as _np

        keep: list[int] = []
        for i, s in enumerate(self.samples):
            d = _np.load(Path(s["sample_dir"]) / "daheng_frame.npy").astype(_np.float32)
            if float(d.max()) >= min_dpeak:
                keep.append(i)
        if not keep:
            raise RuntimeError(f"min_dpeak={min_dpeak} filtered out every sample")
        self.samples = [self.samples[i] for i in keep]
        self._labels = [self._labels[i] for i in keep]
        logger.info("min_dpeak filter kept {} / {} samples", len(keep), len(self._labels) + len(keep))


def _build_loaders(
    data_root: str,
    *,
    split_mode: str,
    run_ids: list[str] | None,
    n_target: int,
    target_indices: list[int],
    min_dpeak: float | None,
) -> dict[str, object]:
    if split_mode == "run":
        return create_zernike_loaders(
            data_root, batch_size=BATCH_SIZE, val_split=0.15, test_split=0.5,
            seed=SEED, input_mode=INPUT_MODE, run_ids=run_ids, split_mode="run",
            num_workers=4, skip_unlabeled=True, n_target=n_target,
            target_indices=target_indices,
        )
    ds = _FilteredDataset(
        data_root, run_ids=run_ids, input_mode=INPUT_MODE, seed=SEED,
        skip_unlabeled=True, n_target=n_target,
        target_indices=target_indices, min_dpeak=min_dpeak,
    )
    train_idx, val_idx, test_idx = _split_indices(
        len(ds), [s["run_id"] for s in ds.samples], 0.15, 0.15, SEED
    )
    splits = {
        "train": Subset(ds, train_idx),
        "val": Subset(ds, val_idx),
        "test": Subset(ds, test_idx),
    }
    loaders = {
        name: DataLoader(
            d, batch_size=BATCH_SIZE, shuffle=(name == "train"),
            num_workers=4, drop_last=False,
            generator=torch.Generator().manual_seed(SEED) if name == "train" else None,
        )
        for name, d in splits.items()
    }
    return {
        "train": loaders["train"],
        "val": loaders["val"],
        "test": loaders["test"],
        "split_sizes": {name: len(d) for name, d in splits.items()},
        "split_mode": "sample",
        "seed": SEED,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-indices", type=str, required=True,
                   help="Comma-separated positions into the 65 non-piston vector (e.g. 0,1,2,3,4,5,6,7,8,9).")
    p.add_argument("--split-mode", choices=["run", "sample"], required=True)
    p.add_argument("--run-ids", type=str, default=None,
                   help="Comma-separated runs (default: 3 real runs for run mode, 0414 for sample mode).")
    p.add_argument("--min-dpeak", type=float, default=None,
                   help="Drop samples whose daheng focus peak < this value (quality filter).")
    p.add_argument("--out-root", type=str, default="data/zernike_pred/checkpoints/real_joint")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--data-root", type=str, default="data/slm_dual_spot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tidx = [int(x) for x in args.target_indices.split(",") if x.strip()]
    if not tidx:
        raise SystemExit("--target-indices must be non-empty")
    names = coefficient_names(10)  # 65 'nXmY' labels in metadata order
    target_names = [names[i] for i in tidx]

    if args.split_mode == "run":
        run_ids = args.run_ids.split(",") if args.run_ids else RUNS
        out_dir = Path(args.out_root)
    else:
        run_ids = args.run_ids.split(",") if args.run_ids else ["20260414_171241"]
        out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_summary.json"
    if summary_path.exists():
        logger.info("already evaluated: {}; skipping", out_dir)
        print(json.loads(summary_path.read_text()))
        return

    loaders = _build_loaders(
        args.data_root, split_mode=args.split_mode, run_ids=run_ids,
        n_target=len(tidx), target_indices=tidx, min_dpeak=args.min_dpeak,
    )
    model = build_model("resnet18", in_channels=2, n_coeffs=len(tidx), device=args.device)

    logger.info("joint-{} {} (split_mode={}, runs={}): {} / {} / {} samples",
                len(tidx), target_names, args.split_mode, run_ids,
                loaders["split_sizes"]["train"], loaders["split_sizes"]["val"],
                loaders["split_sizes"]["test"])
    result = train_regressor(
        model, loaders["train"], loaders["val"],
        epochs=args.epochs, lr=LR, weight_decay=WEIGHT_DECAY, device=args.device,
        model_name="resnet18", input_mode=INPUT_MODE,
        run_id=f"joint{len(tidx)}-{args.split_mode}-dpeak{args.min_dpeak}",
        use_wandb=False, seed=SEED, checkpoint_dir=str(out_dir),
        early_stop_patience=60,
    )
    best_path = Path(result["best_model_path"])
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
    ev = evaluate_regressor(model, loaders["test"], args.device, out_dir=str(out_dir),
                            prefix="eval", return_arrays=True)
    tm = ev["metrics"]
    np.save(out_dir / "test_pred.npy", ev["pred"])
    np.save(out_dir / "test_true.npy", ev["true"])

    per_coeff = {}
    for j, name in enumerate(target_names):
        per_coeff[name] = {
            "corr": float(np.corrcoef(ev["true"][:, j], ev["pred"][:, j])[0, 1]),
            "mae": float(np.abs(ev["true"][:, j] - ev["pred"][:, j]).mean()),
            "r2": float(1 - ((ev["true"][:, j] - ev["pred"][:, j]) ** 2).sum()
                        / ((ev["true"][:, j] - ev["true"][:, j].mean()) ** 2).sum()),
        }
    row = {
        "target_indices": tidx,
        "target_names": target_names,
        "split_mode": args.split_mode,
        "run_ids": run_ids,
        "min_dpeak": args.min_dpeak,
        "split_sizes": loaders["split_sizes"],
        "best_val_mae": float(result["best_val_mae"]),
        "best_epoch": int(result["best_epoch"]),
        "test_mae": float(tm["mae"]),
        "test_rmse": float(tm["rmse"]),
        "test_r2": float(tm["r2"]) if np.isfinite(tm["r2"]) else None,
        "per_coeff": per_coeff,
        "n_test": int(tm["n_samples"]),
    }
    summary_path.write_text(json.dumps(row, indent=2))
    logger.info("joint-{} {}: test_mae {:.4f} r2 {} (best epoch {})", len(tidx),
                target_names, row["test_mae"], row["test_r2"], row["best_epoch"])


if __name__ == "__main__":
    main()