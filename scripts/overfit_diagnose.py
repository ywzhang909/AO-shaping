"""Diagnose the overfitting, then fix what the diagnosis supports.

"Overfitting" is a claim about the gap between what the model fits and what it
generalises to, so that is what gets measured: train vs validation error across training
set size (a learning curve) and across capacity. Nothing here is asserted from a single
run -- the repo's own noise floor makes a single seed unreadable.

One fact worth stating up front, because it is free and suspicious: the family holds 1010
usable records but ``DEFAULT_MAX_RECORDS`` is **512**. Training on half the data is the
first candidate explanation for any gap, and it costs nothing to test.

Levers tested, in the order the diagnosis supports:
  * max_train      -- use the records that already exist
  * l2_penalty     -- shrink the coefficients directly (the model is linear in them)
  * weight_decay   -- shrink via the optimiser
  * n_max          -- fewer degrees of freedom

Reported as paired differences over seeds against the incumbent, because the between-seed
spread (sd ~0.075 on R2) dwarfs every effect below except the data size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ml.zernike.losses import LossConfig  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, train  # noqa: E402

SEEDS = [0, 1, 2]
EPOCHS = 60

# Learning curve first: is there a gap at all, and does more data close it?
CURVE = [
    ("train=128", dict(max_train=128)),
    ("train=256", dict(max_train=256)),
    ("train=512 (default)", dict(max_train=512)),
    ("train=768", dict(max_train=768)),
]

# Then the regularisation levers, all at the full data size.
LEVERS = [
    ("incumbent n_max=20", dict()),
    ("+ l2=1e-3", dict(l2_penalty=1e-3)),
    ("+ l2=1e-2", dict(l2_penalty=1e-2)),
    ("+ weight_decay=1e-2", dict(weight_decay=1e-2)),
    ("n_max=15 (fewer DOF)", dict(n_max=15)),
]


def run(tag: str, kwargs: dict, seed: int) -> dict:
    cfg = AmpTrainConfig(
        seed=seed, epochs=EPOCHS, n_max=kwargs.get("n_max", 20), loss="mse",
        loss_weights=LossConfig(w_mse=1.0, w_shape_gap=1.0),
        max_train=kwargs.get("max_train", 512),
        l2_penalty=kwargs.get("l2_penalty", 0.0),
        weight_decay=kwargs.get("weight_decay", 0.0),
        num_workers=0, use_wandb=False,
        out_dir=ROOT / "logs" / "overfit" / f"{tag.replace(' ', '_').replace('=', '')}_s{seed}",
    )
    r = train(cfg)
    # Best epoch's own train/val pair -- comparing the minima of two different curves is
    # the classic way to fake a gap that is not there.
    best = min(r.history, key=lambda row: row["val_mse"])
    return {
        "tag": tag, "seed": seed,
        "max_train": cfg.max_train, "n_max": cfg.n_max,
        "l2": cfg.l2_penalty, "weight_decay": cfg.weight_decay,
        "train_mse": float(best["train_mse"]), "val_mse": float(best["val_mse"]),
        "val_r2": float(best["val_r2"]),
        "gap": float(best["val_mse"] - best["train_mse"]),
        "ratio": float(best["val_mse"] / max(best["train_mse"], 1e-12)),
    }


def report(rows: list[dict], tags: list[str]) -> None:
    base = {r["seed"]: r for r in rows if r["tag"] == tags[0]}
    print(f"\n{'config':<24}{'R2':>9}{'train':>10}{'val':>10}{'gap':>9}"
          f"{'val/train':>11}{'dR2':>9}{'pos':>6}")
    for tag in tags:
        sel = [r for r in rows if r["tag"] == tag]
        d = np.array([r["val_r2"] - base[r["seed"]]["val_r2"] for r in sel])
        print(f"{tag:<24}{np.mean([r['val_r2'] for r in sel]):>9.4f}"
              f"{np.mean([r['train_mse'] for r in sel]):>10.5f}"
              f"{np.mean([r['val_mse'] for r in sel]):>10.5f}"
              f"{np.mean([r['gap'] for r in sel]):>+9.5f}"
              f"{np.mean([r['ratio'] for r in sel]):>11.2f}"
              f"{d.mean():>+9.4f}{int((d > 0).sum()):>4}/{len(d)}")


def main() -> None:
    print("=" * 104)
    print("OVERFITTING: learning curve first, then regularisation levers (paired over seeds)")
    print("=" * 104)
    print(f"seeds={SEEDS} epochs={EPOCHS} family=slm_zernike_shaping (1010 usable records)")

    curve_rows = [run(t, k, s) for t, k in CURVE for s in SEEDS]
    print("\n--- learning curve: n_max=20, varying the number of TRAINING records ---")
    report(curve_rows, [t for t, _ in CURVE])

    lever_rows = [run(t, k, s) for t, k in LEVERS for s in SEEDS]
    print("\n--- regularisation levers, all at max_train=512 ---")
    report(lever_rows, [t for t, _ in LEVERS])

    print("\nreading:")
    print("  gap > 0        -> validation error exceeds training error (overfitting)")
    print("  val/train ~1   -> no gap; the model is data-limited, not capacity-limited")
    print("  a lever helps only if dR2 is positive AND consistent (pos == len(seeds))")

    out = ROOT / "report" / "loss_defects" / "overfitting.json"
    out.write_text(json.dumps({"curve": curve_rows, "levers": lever_rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
