"""Leakage canaries for the POD-ridge result.

``diagnose_pod_ridge.py`` reports val R^2 = +0.916 for a 4-component PCA +
ridge, which beats every recorded model in the project. That is the kind of
result that is normally a bug, so this script runs the two controls that would
expose the usual causes:

``mean``
    Predict the **training-mean image** for every validation sample (all
    regression coefficients zero). Any R^2 materially above 0 means the metric
    or the target plumbing is broken, not the model.

``permuted``
    Ridge onto **randomly permuted** training images (labels destroyed, images
    untouched). This isolates the coefficient->image pathway: if permuted R^2 is
    not near/below 0, the pathway is leaking.

``real``
    The actual model, as a positive control that the two arms bracket it.

A leak-free pipeline gives ``real`` >> 0 >= ``permuted`` and ``mean`` ~= 0.

Usage
-----
    python scripts/verify_pod_ridge_canary.py
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


def folds_of(records, protocol="file"):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(k, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for k in keys]


def ridge(x, z, alpha):
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    g = xc @ xc.T
    g.flat[:: g.shape[0] + 1] += alpha
    return xc.T @ np.linalg.solve(g, zc), xm, zm


def score(pred, truth, grid):
    return float(batch_image_metrics(
        torch.from_numpy(pred.astype(np.float32)).reshape(-1, 1, grid, grid),
        torch.from_numpy(truth.astype(np.float32)).reshape(-1, 1, grid, grid),
    )["r2"])


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

    out = {"k": K, "folds": []}
    for index_i, (held, train_pos, val_pos) in enumerate(folds_of(records)):
        x_tr, y_tr = stack(train_pos)
        x_va, y_va = stack(val_pos)
        centred = y_tr - y_tr.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:K]
        y_mean = y_tr.mean(0, keepdims=True)

        w, xm, zm = ridge(x_tr, centred @ basis.T, 1.0)
        real = score(((x_va - xm) @ w + zm) @ basis + y_mean, y_va, grid)

        mean_r2 = score(np.repeat(y_mean, len(y_va), axis=0), y_va, grid)

        rng = np.random.default_rng(1234 + index_i)
        perm = rng.permutation(len(y_tr))
        w_p, xm_p, zm_p = ridge(x_tr, centred[perm] @ basis.T, 1.0)
        perm_r2 = score(((x_va - xm_p) @ w_p + zm_p) @ basis + y_mean, y_va, grid)

        out["folds"].append({"fold": index_i, "real": real, "mean": mean_r2,
                             "permuted": perm_r2})
        print("fold %2d  real=%+.4f  mean=%+.4f  permuted=%+.4f"
              % (index_i, real, mean_r2, perm_r2), flush=True)

    for arm in ("real", "mean", "permuted"):
        vals = [f[arm] for f in out["folds"]]
        out[arm + "_mean"] = float(np.mean(vals))
        print("\n%-8s mean over %d folds: %+.4f" % (arm, len(vals), np.mean(vals)))
    verdict = (out["real_mean"] > 0.5 and out["mean_mean"] < 0.2
               and out["permuted_mean"] < 0.2)
    out["verdict"] = "PASS" if verdict else "SUSPECT"
    print("\nverdict:", out["verdict"], "(real >> 0 >= mean, permuted)")
    Path("logs/pod_ridge_canary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
