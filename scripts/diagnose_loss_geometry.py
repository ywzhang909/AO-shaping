"""Is plain MSE on a near-zero-background target the wrong objective?

Candidate explanation for the negative-skill result (see
``report/zernike_r2_baseline/report.md`` section 9.1): the peak-normalised target
has mean ~0.014 because it is mostly dark background with a small bright core.
An unweighted MSE is then dominated by that core -- the few bright pixels
contribute most of the gradient -- so the dark background, which is most of the
image and all of the *shape* information, is barely fitted.

If that is true, a loss that equalises per-pixel influence should raise skill
materially at identical model capacity. This script tests it with the closed-form
ridge (so the comparison is exact and free of optimisation noise) on the same
10-fold protocol:

``mse``      plain squared error (what ``fit()`` uses today)
``sqrt_mse`` root-space MSE, i.e. fitting amplitude rather than intensity --
             physically the natural space, since intensity is |E|^2
``log_mse``  MSE on ``log1p(k*I)/log1p(k)`` -- compresses the core, expands the
             background, the standard fix for dynamic-range problems

A skill gain from a *loss change alone*, with no capacity change, would confirm
the objective is a bottleneck rather than the model.

Usage
-----
    python scripts/diagnose_loss_geometry.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
LOSSES = ("mse", "sqrt_mse", "log_mse")
K = 4


def folds_of(records):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(i, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for i, k in enumerate(keys)]


def ridge(x: np.ndarray, z: np.ndarray, alpha: float = 1.0):
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    g = xc @ xc.T
    g.flat[:: g.shape[0] + 1] += alpha
    return xc.T @ np.linalg.solve(g, zc), xm, zm


def to_loss_space(y: np.ndarray, kind: str) -> np.ndarray:
    """Map a target into the space the loss is computed in, and return the inverse."""
    if kind == "mse":
        return y, (lambda z: z)
    if kind == "sqrt_mse":
        return np.sqrt(np.maximum(y, 0.0)), (lambda z: np.maximum(z, 0.0) ** 2)
    if kind == "log_mse":
        c = 50.0
        return np.log1p(c * y) / np.log1p(c), (lambda z: np.expm1(z * np.log1p(c)) / c)
    raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/loss_geometry.json")
    args = ap.parse_args(argv)

    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY])
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=64, image_mode="peak"), use_cache=True)

    def stack(positions):
        c, im = [], []
        for p in positions:
            s = dataset[p]
            c.append(s["coeffs"].reshape(-1).numpy())
            im.append(s["image"].reshape(-1).numpy())
        return np.stack(c), np.stack(im)

    started = time.perf_counter()
    out: dict = {"folds": []}
    for fold_i, train_pos, val_pos in folds_of(records):
        x_tr, y_tr = stack(train_pos)
        x_va, y_va = stack(val_pos)
        row = {"fold": fold_i}
        for kind in LOSSES:
            t_tr, inverse = to_loss_space(y_tr, kind)
            t_va, _ = to_loss_space(y_va, kind)
            centred = t_tr - t_tr.mean(0, keepdims=True)
            k = min(K, centred.shape[1])
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            basis = vt[:k]
            t_mean = t_tr.mean(0, keepdims=True)
            w, xm, zm = ridge(x_tr, centred @ basis.T)
            t_pred = ((x_va - xm) @ w + zm) @ basis + t_mean
            # Score in the ORIGINAL intensity space, so the losses are comparable.
            pred = inverse(t_pred)
            tt = torch.from_numpy(y_va.astype(np.float32)).reshape(-1, 1, 64, 64)
            pt = torch.from_numpy(np.clip(pred, 0.0, None).astype(np.float32)).reshape(-1, 1, 64, 64)
            const = torch.from_numpy(
                np.repeat(y_tr.mean(0, keepdims=True), len(y_va), axis=0).astype(np.float32)
            ).reshape(-1, 1, 64, 64)
            row[kind] = {
                "skill": constant_baseline_metrics(pt, tt)["skill"],
                "mse": constant_baseline_metrics(pt, tt)["mse"],
            }
            row["mse_const"] = constant_baseline_metrics(const, tt)["mse"]
        out["folds"].append(row)
        print("fold %2d  " % fold_i + "  ".join(
            "%s skill=%+.4f" % (kind, row[kind]["skill"]) for kind in LOSSES), flush=True)

    print()
    for kind in LOSSES:
        vals = [r[kind]["skill"] for r in out["folds"]]
        mses = [r[kind]["mse"] for r in out["folds"]]
        print("%-9s mean skill = %+.4f  (min %+.4f, max %+.4f)   mean mse = %.6g"
              % (kind, float(np.mean(vals)), float(np.min(vals)),
                 float(np.max(vals)), float(np.mean(mses))))
    best = max(LOSSES, key=lambda k: float(np.mean([r[k]["skill"] for r in out["folds"]])))
    out["best_loss"] = best
    print("\nbest loss space:", best)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out, "in %.1fs" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
