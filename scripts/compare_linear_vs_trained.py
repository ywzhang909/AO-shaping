"""Closed-form linear baseline vs the trained models, on one identical pipeline.

Why
---
``compare_models_cv.py`` reports, on the authoritative 10-fold file protocol,
``r2_const = +0.9396`` against a U-Net at ``+0.8986`` -- i.e. **skill < 0** for
every trained arm (unet -0.99, physics -1.27, hybrid -1.41). A model whose MSE
is twice a single fixed image's MSE cannot be called a forward model.

This script puts a closed-form linear model (K-component POD + ridge) on exactly
that pipeline -- same corpus, same ``str(record.path)`` folds, same
``MaterialiserConfig(grid=64)`` target, same ``_peak_normalise`` the CV harness
applies -- so the two numbers are directly comparable, and re-derives the
constant baseline alongside it.

The point is not to ship a ridge as the production model. It is to establish the
bar the neural arms have to clear, and to show the gap is not a tuning artefact.

Usage
-----
    python scripts/compare_linear_vs_trained.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")):
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
    """The same normalisation ``compare_models_cv`` applies to CV targets."""
    scale = x.amax(dim=(-2, -1), keepdim=True)
    return x / torch.clamp(scale, min=EPS)


def folds_of(records):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(i, k, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for i, k in enumerate(keys)]


def ridge(x: np.ndarray, z: np.ndarray, alpha: float):
    """Dual-form multi-output ridge with an intercept (n << 136 features)."""
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    g = xc @ xc.T
    g.flat[:: g.shape[0] + 1] += alpha
    return xc.T @ np.linalg.solve(g, zc), xm, zm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="logs/linear_vs_trained.json")
    args = ap.parse_args(argv)

    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY])
    records = list(index.records)
    # Same *target* pipeline as compare_models_cv.py: grid=64, default image_mode
    # (abs255), then `_peak_normalise`. ZernikeCoeffDataset is used because it is
    # the only dataset that carries the Zernike coefficients -- the ridge needs
    # them as input, and they are what a deployable model would be handed.
    dataset = ZernikeCoeffDataset(
        index,
        config=MaterialiserConfig(grid=64),
        use_cache=True,
    )

    def stack(positions):
        coeffs, frames = [], []
        for p in positions:
            s = dataset[p]
            coeffs.append(s["coeffs"].reshape(-1).numpy())
            frames.append(s["image"].reshape(1, 64, 64).clone())
        # (N,1,g,g) so `_peak_normalise` normalises per sample, as the CV harness does.
        y = peak_normalise(torch.stack(frames)).reshape(len(positions), -1).numpy()
        return np.stack(coeffs), y

    rows = []
    started = time.perf_counter()
    for index_i, held_key, train_pos, val_pos in folds_of(records):
        x_tr, y_tr = stack(train_pos)
        x_va, y_va = stack(val_pos)

        centred = y_tr - y_tr.mean(0, keepdims=True)
        k = min(args.k, centred.shape[1])
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:k]
        y_mean = y_tr.mean(0, keepdims=True)
        w, xm, zm = ridge(x_tr, centred @ basis.T, args.alpha)
        pred = ((x_va - xm) @ w + zm) @ basis + y_mean

        t = torch.from_numpy(y_va.astype(np.float32)).reshape(-1, 1, 1, 64, 64)
        pred_t = torch.from_numpy(pred.astype(np.float32)).reshape(-1, 1, 1, 64, 64)
        const_t = torch.from_numpy(
            np.repeat(y_mean, len(y_va), axis=0).astype(np.float32)
        ).reshape(-1, 1, 1, 64, 64)

        m_pred = constant_baseline_metrics(pred_t, t)
        m_const = constant_baseline_metrics(const_t, t)
        rows.append({
            "fold": index_i,
            "held_out": Path(held_key).name,
            "skill_ridge": m_pred["skill"],
            "skill_const": m_const["skill"],
            "r2_const": m_const["r2_const"],
            "mse_ridge": m_pred["mse"],
            "mse_const": m_const["mse_const"],
        })
        print("fold %2d  skill_ridge=%+.4f  r2_const=%+.4f  mse_ridge=%.6f "
              "mse_const=%.6f"
              % (index_i, m_pred["skill"], m_const["r2_const"],
                 m_pred["mse"], m_const["mse_const"]), flush=True)

    skill = [r["skill_ridge"] for r in rows]
    print("\nK=%d ridge: mean skill = %+.4f  (min %+.4f, max %+.4f)"
          % (args.k, float(np.mean(skill)), float(np.min(skill)), float(np.max(skill))))
    print("constant baseline r2 = %+.4f"
          % float(np.mean([r["r2_const"] for r in rows])))
    print("\nfor reference, from logs/cv_skill.json (same folds, same target):")
    print("  unet    skill = -0.9938")
    print("  physics skill = -1.2674")
    print("  hybrid  skill = -1.4056")

    Path(args.out).write_text(json.dumps({
        "k": args.k, "rows": rows,
        "mean_skill_ridge": float(np.mean(skill)),
        "mean_r2_const": float(np.mean([r["r2_const"] for r in rows])),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
