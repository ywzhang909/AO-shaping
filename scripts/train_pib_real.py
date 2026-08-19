"""Train a REGRESSOR on real dual-spot daheng focus PIB (power-in-bucket target).

Direction 3: instead of predicting 65 Zernike coefficients (r2 ~ 0 on real
data), regress a single intensity metric — PIB = power inside a bucket of
radius ``--bucket-radius`` centered at the fixed daheng spot position
(748, 477), normalized by total frame power. If the real focus images carry
learnable focusing signal, PIB should be predictable with meaningful r2,
contrasting with the flat r2 ~ -0.16 of coefficient regression.

The dataset loads the same save (daheng + miicam frames, ``combined`` 2-channel
input) but resolves each sample's target to a scalar PIB computed from its own
daheng frame instead of the 65-vector Zernike label.

Split is 0414-only sample-level: the 0402 runs' daheng field is ~flat (no
spot, PIB std ~ 2e-4), so cross-run PIB prediction is structurally impossible;
this experiment isolates in-run learnability of the intensity objective.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/train_pib_real.py \
        --bucket-radius 25 --out-root data/zernike_pred/checkpoints/real_pib_r25 \
        --device cuda:0 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, "src")

from ml.zernike.models import build_model  # noqa: E402
from ml.zernike_prediction.dataset import _split_indices  # noqa: E402
from ml.zernike_prediction.trainer import evaluate_regressor, train_regressor  # noqa: E402

RUN_0414 = "20260414_171241"
SPOT_CENTER = (748, 477)  # avg daheng spot position in the 0414 run
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 42
BATCH_SIZE = 32

_DAHENG_H, _DAHENG_W = 1024, 1280
_MIICAM_H, _MIICAM_W = 1520, 2688


def _resize(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """BILINEAR downscale to ``target_size`` (torchvision fallback path)."""
    from torchvision.transforms import functional as F

    t = torch.from_numpy(np.asarray(frame, dtype=np.float32)).unsqueeze(0)
    t = F.resize(t, target_size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    return t.squeeze(0).numpy()


def _pib_label(frame: np.ndarray, cy: int, cx: int, radius: int) -> float:
    """Power fraction of the fixed-center bucket (radius in native pixels)."""
    f = np.asarray(frame, dtype=np.float32)
    total = float(f.sum()) + 1e-12
    y0, y1 = max(0, cy - radius), min(_DAHENG_H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(_DAHENG_W, cx + radius + 1)
    return float(f[y0:y1, x0:x1].sum() / total)


class PIBDataset(Dataset):
    """Single-scalar-target dataset: ``(combined 2ch image, PIB)`` tuples."""

    def __init__(self, data_root: str | Path, bucket_radius: int = 25) -> None:
        root = Path(data_root) / RUN_0414
        self.samples = sorted(p for p in root.glob("sample_*") if p.is_dir())
        self._labels = [
            _pib_label(np.load(s / "daheng_frame.npy"), *SPOT_CENTER, bucket_radius)
            for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sd = self.samples[idx]
        d = np.load(sd / "daheng_frame.npy").astype(np.float32)
        m = np.load(sd / "miicam_frame.npy").astype(np.float32)
        d = _resize(d, (256, 256)); m = _resize(m, (256, 256))
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        img = torch.from_numpy(np.stack([d, m]).astype(np.float32))
        label = torch.tensor([self._labels[idx]], dtype=torch.float32)
        return img, label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket-radius", type=int, default=25)
    p.add_argument("--out-root", type=str, default="data/zernike_pred/checkpoints/real_pib")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--data-root", type=str, default="data/slm_dual_spot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.bucket_radius < 5:
        raise SystemExit(f"--bucket-radius too small: {args.bucket_radius}")
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_summary.json"
    if summary_path.exists():
        logger.info("already evaluated: {}; skipping", out_dir)
        print(json.loads(summary_path.read_text()))
        return

    ds = PIBDataset(args.data_root, bucket_radius=args.bucket_radius)
    train_idx, val_idx, test_idx = _split_indices(
        len(ds), [RUN_0414] * len(ds), 0.15, 0.15, SEED
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
    sizes = {name: len(d) for name, d in splits.items()}
    logger.info("PIB r={} on 0414: {} / {} / {} samples", args.bucket_radius, *sizes.values())

    model = build_model("resnet18", in_channels=2, n_coeffs=1, device=args.device)
    result = train_regressor(
        model, loaders["train"], loaders["val"],
        epochs=args.epochs, lr=LR, weight_decay=WEIGHT_DECAY, device=args.device,
        model_name="resnet18", input_mode="combined",
        run_id=f"pib-r{args.bucket_radius}-0414s", use_wandb=False, seed=SEED,
        checkpoint_dir=str(out_dir), early_stop_patience=60,
    )
    best_path = Path(result["best_model_path"])
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
    ev = evaluate_regressor(model, loaders["test"], args.device, out_dir=str(out_dir),
                            prefix="eval", return_arrays=True)
    tm = ev["metrics"]
    np.save(out_dir / "test_pred.npy", ev["pred"])
    np.save(out_dir / "test_true.npy", ev["true"])

    corr = float(np.corrcoef(ev["true"].ravel(), ev["pred"].ravel())[0, 1])
    row = {
        "bucket_radius": args.bucket_radius,
        "spot_center": list(SPOT_CENTER),
        "split_sizes": sizes,
        "best_val_mae": float(result["best_val_mae"]),
        "best_epoch": int(result["best_epoch"]),
        "test_mae": float(tm["mae"]),
        "test_rmse": float(tm["rmse"]),
        "test_r2": float(tm["r2"]) if np.isfinite(tm["r2"]) else None,
        "test_corr": corr,
        "n_test": int(tm["n_samples"]),
        "label_mean": float(np.mean(ds._labels)),
        "label_std": float(np.std(ds._labels)),
    }
    summary_path.write_text(json.dumps(row, indent=2))
    logger.info("PIB r={}: test_mae {:.4f} r2 {} corr {:.3f} (best epoch {})",
                args.bucket_radius, row["test_mae"], row["test_r2"], corr, row["best_epoch"])


if __name__ == "__main__":
    main()