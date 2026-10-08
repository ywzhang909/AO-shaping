"""Does a better target transform raise the model's *skill* over a constant?

``diagnose_r2_baseline.py`` established the problem: on the peak-normalised
far field a **constant** predictor already scores R^2 = +0.910, and the whole
recorded model table lives within +-0.03 of that floor. The honest scoreboard is
``skill = 1 - mse_model / mse_constant`` (0.06 for a 4-component ridge), not the
absolute R^2.

The cause is that the pooled variance of the target is only ~0.014 and is
dominated by a near-invariant central structure, so the coefficient-dependent
speckle detail -- the only informative part -- is a tiny fraction of the signal.
That is a **data-processing** problem, not a capacity problem, so this script
varies the target transform and reports skill, which is invariant to the metric's
floor:

``peak``     per-frame max normalisation (the current default)
``raw255``   absolute detector level / 255 (no per-frame normalisation)
``center``   peak-normalise, then crop to the central 32x32 (drops the invariant
             halo, where the informative speckle lives)
``log``      ``log1p(k*I) / log1p(k)`` after peak normalisation (compresses the
             dominant 0-order, expands the dim halo)
``zscore``   per-frame standardisation (zero mean, unit variance)

Run
---
    python scripts/sweep_target_transform.py
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

from ml.hwdataset import MaterialiserConfig, build_hw_index
from ml.hwdataset.zernike_dataset import ZERNIKE_COEFF_SOURCES, ZernikeCoeffDataset
from ml.zernike.metrics import batch_image_metrics

import torch

FAMILY = "slm_zernike_shaping"
K = 4
TRANSFORMS = ("peak", "raw255", "center", "log", "zscore")


def apply_transform(kind: str, frames: np.ndarray, grid: int) -> np.ndarray:
    """frames: (n, grid*grid) float32 in [0,1]-ish absolute level."""
    x = frames.reshape(-1, grid, grid)
    if kind == "peak":
        m = x.max(axis=(1, 2), keepdims=True)
        return (x / np.maximum(m, 1e-8)).reshape(len(x), -1)
    if kind == "raw255":
        return (x / 255.0).reshape(len(x), -1)
    if kind == "center":
        m = x.max(axis=(1, 2), keepdims=True)
        x = x / np.maximum(m, 1e-8)
        half = grid // 4                       # central 32x32 for grid=64
        c = grid // 2
        return x[:, c - half:c + half, c - half:c + half].reshape(len(x), -1)
    if kind == "log":
        m = x.max(axis=(1, 2), keepdims=True)
        x = x / np.maximum(m, 1e-8)
        k = 50.0
        return (np.log1p(k * x) / np.log1p(k)).reshape(len(x), -1)
    if kind == "zscore":
        mu = x.mean(axis=(1, 2), keepdims=True)
        sd = x.std(axis=(1, 2), keepdims=True)
        return ((x - mu) / np.maximum(sd, 1e-8)).reshape(len(x), -1)
    raise ValueError(kind)


def folds_of(records):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(k, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for k in keys]


def ridge(x, z, alpha=1.0):
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    g = xc @ xc.T
    g.flat[:: g.shape[0] + 1] += alpha
    return xc.T @ np.linalg.solve(g, zc), xm, zm


def metrics(pred, truth):
    """`batch_image_metrics` needs BxCxHxW, and the transforms change the side
    length (`center` yields 32x32), so the spatial shape is inferred from the
    flattened width rather than assumed to be `grid`."""
    side = int(round(pred.shape[1] ** 0.5))
    if side * side != pred.shape[1]:
        raise ValueError(f"width {pred.shape[1]} is not a square grid")
    return batch_image_metrics(
        torch.from_numpy(pred.astype(np.float32)).reshape(-1, 1, side, side),
        torch.from_numpy(truth.astype(np.float32)).reshape(-1, 1, side, side))


def main() -> int:
    grid = 64
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY], sources=list(ZERNIKE_COEFF_SOURCES))
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=grid, image_mode="abs255"), use_cache=True)

    coeffs = {}
    images = {}
    for p in range(len(records)):
        s = dataset[p]
        key = (str(records[p].path), p)
        coeffs[key] = s["coeffs"].reshape(-1).numpy()
        images[key] = s["image"].reshape(-1).numpy()

    folds = folds_of(records)
    out: dict = {"transforms": {}}
    for kind in TRANSFORMS:
        per_fold = []
        for held, train_pos, val_pos in folds:
            keys_tr = [(str(records[p].path), p) for p in train_pos]
            keys_va = [(str(records[p].path), p) for p in val_pos]
            x_tr = np.stack([coeffs[k] for k in keys_tr])
            x_va = np.stack([coeffs[k] for k in keys_va])
            y_tr = apply_transform(kind, np.stack([images[k] for k in keys_tr]), grid)
            y_va = apply_transform(kind, np.stack([images[k] for k in keys_va]), grid)

            centred = y_tr - y_tr.mean(0, keepdims=True)
            kk = min(K, centred.shape[1])
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            basis = vt[:kk]
            y_mean = y_tr.mean(0, keepdims=True)
            w, xm, zm = ridge(x_tr, centred @ basis.T)
            pred = ((x_va - xm) @ w + zm) @ basis + y_mean

            m_ridge = metrics(pred, y_va)
            m_const = metrics(np.repeat(y_mean, len(y_va), axis=0), y_va)
            per_fold.append({
                "fold": len(per_fold),
                "r2_ridge": float(m_ridge["r2"]),
                "r2_const": float(m_const["r2"]),
                "mse_ridge": float(m_ridge["mse"]),
                "mse_const": float(m_const["mse"]),
                "skill": 1.0 - float(m_ridge["mse"]) / float(m_const["mse"]),
            })
        agg = {
            "r2_ridge": float(np.mean([f["r2_ridge"] for f in per_fold])),
            "r2_const": float(np.mean([f["r2_const"] for f in per_fold])),
            "skill": float(np.mean([f["skill"] for f in per_fold])),
            "skill_sd": float(np.std([f["skill"] for f in per_fold], ddof=1)),
            "n_negative_skill": int(sum(1 for f in per_fold if f["skill"] < 0)),
        }
        out["transforms"][kind] = {"agg": agg, "folds": per_fold}
        print("%-8s R2_ridge=%+.4f  R2_const=%+.4f  skill=%+.4f +/- %.4f  "
              "(%d/10 folds negative)"
              % (kind, agg["r2_ridge"], agg["r2_const"], agg["skill"],
                 agg["skill_sd"], agg["n_negative_skill"]), flush=True)

    best = max(out["transforms"], key=lambda t: out["transforms"][t]["agg"]["skill"])
    out["best_transform"] = best
    print("\nbest transform by skill:", best)
    Path("logs/target_transform.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote logs/target_transform.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
