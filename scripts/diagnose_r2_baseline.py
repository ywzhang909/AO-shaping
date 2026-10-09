"""Decompose the val R^2: how much comes from the model, how much from the baseline?

The POD-ridge canary showed a *constant* train-mean predictor scoring +0.9097 on
the same folds where the ridge scored +0.9163. So the absolute R^2 reported
across this project is mostly a property of the target, not of the model. This
script prints the ingredients so the size of that property is explicit:

* ``var``   -- the pooled variance ``batch_image_metrics`` divides by,
* ``mse_const`` -- mean squared error of the constant train-mean predictor,
* ``r2_const``   -- the resulting R^2,
* ``mse_ridge`` / ``r2_ridge`` -- the model's own contribution on top,
* ``skill``      -- ``1 - mse_ridge / mse_const``: the fraction of the *baseline*
  error the model removes. This is the honest scoreboard, and it is bounded in
  [-inf, 1] instead of being anchored to a metric floor that a constant clears.

Usage
-----
    python scripts/diagnose_r2_baseline.py
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
from ml.zernike.metrics import batch_image_metrics

FAMILY = "slm_zernike_shaping"
K = 4


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


def main() -> int:
    grid = 64
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY], sources=list(ZERNIKE_COEFF_SOURCES))
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=grid, image_mode="peak"), use_cache=True)

    def stack(positions):
        c, im = [], []
        for p in positions:
            s = dataset[p]
            c.append(s["coeffs"].reshape(-1).numpy())
            im.append(s["image"].reshape(-1).numpy())
        return np.stack(c), np.stack(im)

    rows = []
    for i, (held, train_pos, val_pos) in enumerate(folds_of(records)):
        x_tr, y_tr = stack(train_pos)
        x_va, y_va = stack(val_pos)
        centred = y_tr - y_tr.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:K]
        y_mean = y_tr.mean(0, keepdims=True)
        w, xm, zm = ridge(x_tr, centred @ basis.T)
        pred_ridge = ((x_va - xm) @ w + zm) @ basis + y_mean
        pred_const = np.repeat(y_mean, len(y_va), axis=0)

        def tensor(a):
            return torch.from_numpy(a.astype(np.float32)).reshape(-1, 1, grid, grid)

        t = tensor(y_va)
        m_const = batch_image_metrics(tensor(pred_const), t)
        m_ridge = batch_image_metrics(tensor(pred_ridge), t)
        # `batch_image_metrics` divides by the pooled variance; recover it so the
        # two error scales are comparable rather than just their ratio.
        var = float(torch.var(t))
        rows.append({
            "fold": i,
            "var_pooled": var,
            "mse_const": float(m_const["mse"]),
            "mse_ridge": float(m_ridge["mse"]),
            "r2_const": float(m_const["r2"]),
            "r2_ridge": float(m_ridge["r2"]),
            "skill_vs_const": 1.0 - float(m_ridge["mse"]) / float(m_const["mse"]),
        })
        r = rows[-1]
        print("fold %2d  var=%.5f  mse_const=%.5f  mse_ridge=%.5f  "
              "r2_const=%+.4f  r2_ridge=%+.4f  skill=%+.4f"
              % (i, r["var_pooled"], r["mse_const"], r["mse_ridge"],
                 r["r2_const"], r["r2_ridge"], r["skill_vs_const"]), flush=True)

    print()
    for key in ("r2_const", "r2_ridge", "skill_vs_const"):
        vals = [r[key] for r in rows]
        print("%-16s mean over %d folds: %+.4f  (min %+.4f, max %+.4f)"
              % (key, len(vals), np.mean(vals), np.min(vals), np.max(vals)))
    Path("logs/r2_baseline.json").write_text(json.dumps(
        {"k": K, "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote logs/r2_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
