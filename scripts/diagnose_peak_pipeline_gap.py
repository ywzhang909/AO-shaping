"""Locate the 0.03 gap between the two 'peak-normalised' target pipelines.

Two code paths are supposed to produce the same target, and do not:

* ``MaterialiserConfig(grid=64, image_mode="peak")``  -> measured r2_const = +0.9097
  (``scripts/verify_pod_ridge_canary.py``)
* ``MaterialiserConfig(grid=64)`` (abs255) followed by a per-sample
  ``x / x.amax()`` -> measured r2_const ~= +0.947
  (``scripts/compare_linear_vs_trained.py``)

Algebraically ``(raw/255) / max(raw/255) == raw / max(raw)``, so they should agree
bit-for-bit. They do not, and until that is explained the absolute value of
``r2_const`` is not comparable across scripts -- which matters because the whole
conclusion "a constant beats every model" rests on that floor.

This script materialises both targets for the *same* records and reports, per
record: max absolute difference, the fraction of differing pixels, and each
pipeline's own ``r2_const`` on one held-out fold.

Usage
-----
    python scripts/diagnose_peak_pipeline_gap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from ml.hwdataset import MaterialiserConfig, build_hw_index
from ml.hwdataset.zernike_dataset import ZERNIKE_COEFF_SOURCES, ZernikeCoeffDataset
from ml.zernike.metrics import constant_baseline_metrics

FAMILY = "slm_zernike_shaping"
EPS = 1e-8


def peak_normalise(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(x.amax(dim=(-2, -1), keepdim=True), min=EPS)


def folds_of(records):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(i, k, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for i, k in enumerate(keys)]


def main() -> int:
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY])
    records = list(index.records)

    ds_peak = ZernikeCoeffDataset(
        index,
        config=MaterialiserConfig(grid=64, image_mode="peak"),
        use_cache=True,
    )
    ds_abs = ZernikeCoeffDataset(
        index,
        config=MaterialiserConfig(grid=64),
        use_cache=True,
    )

    folds = folds_of(records)
    _, held_key, train_pos, val_pos = folds[0]

    def grab(ds, positions):
        return np.stack([ds[p]["image"].reshape(-1).numpy() for p in positions])

    tr_peak, va_peak = grab(ds_peak, train_pos), grab(ds_peak, val_pos)
    tr_abs, va_abs = grab(ds_abs, train_pos), grab(ds_abs, val_pos)

    # Pipeline B: abs255 materialisation, then peak-normalise per sample.
    def pipeline_b(flat: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(flat).reshape(-1, 1, 64, 64)
        return peak_normalise(t).reshape(len(flat), -1).numpy()

    tr_b, va_b = pipeline_b(tr_abs), pipeline_b(va_abs)

    out = {
        "held_out": Path(held_key).name,
        "n_train": len(train_pos),
        "n_val": len(val_pos),
    }

    for split, (a, b) in {"train": (tr_peak, tr_b), "val": (va_peak, va_b)}.items():
        diff = np.abs(a - b)
        out[f"{split}_max_abs_diff"] = float(diff.max())
        out[f"{split}_mean_abs_diff"] = float(diff.mean())
        out[f"{split}_frac_pixels_differing"] = float((diff > 0).mean())
        print("%-5s  max|diff|=%.6g  mean|diff|=%.6g  pixels differing=%.4f%%"
              % (split, diff.max(), diff.mean(), 100.0 * (diff > 0).mean()))

    # Where do they differ? If it is a single hot pixel, `peak` and a per-frame
    # amax cannot agree -- the argmax would have to tie.
    for split, flat in (("val", va_peak),):
        frames = torch.from_numpy(flat).reshape(-1, 1, 64, 64)
        maxima = frames.amax(dim=(-2, -1))
        n_tied_max = int((frames == maxima[:, :, None, None]).sum(dim=(-2, -1)).gt(1).sum())
        out[f"{split}_frames_with_tied_max"] = n_tied_max
        print("%-5s  frames whose max is attained by >1 pixel: %d / %d"
              % (split, n_tied_max, len(flat)))

    # Each pipeline's own constant-baseline R^2 on the same held-out fold.
    for name, (tr, va) in {"peak_mode": (tr_peak, va_peak),
                           "abs255_then_peak": (tr_b, va_b)}.items():
        const = torch.from_numpy(
            np.repeat(tr.mean(axis=0, keepdims=True), len(va), axis=0).astype(np.float32)
        ).reshape(-1, 1, 64, 64)
        target = torch.from_numpy(va.astype(np.float32)).reshape(-1, 1, 64, 64)
        m = constant_baseline_metrics(const, target)
        out[f"{name}_r2_const"] = m["r2_const"]
        out[f"{name}_mse_const"] = m["mse_const"]
        print("%-16s r2_const=%+.4f  mse_const=%.6g  var=%.6g"
              % (name, m["r2_const"], m["mse_const"], m["var"]))

    Path("logs/peak_pipeline_gap.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote logs/peak_pipeline_gap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
