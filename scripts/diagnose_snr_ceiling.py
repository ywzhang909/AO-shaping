"""Is the coefficient-driven signal even above the measurement noise?

Everything tried so far -- physics, U-Net, hybrid, a 4-component closed-form ridge --
scores a **negative** ``skill`` against a constant-image baseline (see
``report/zernike_r2_baseline/report.md``). Two structural hypotheses were already
refuted (output head, loss space). The remaining explanation is that the
per-frame, coefficient-driven variation of the far-field frame is *smaller than the
measurement noise*, in which case no model can beat "output one fixed image" and
the ceiling is set by SNR, not by capacity.

This script settles it with a model that has **no capacity, no optimiser and no
loss**: a k-nearest-neighbour predictor in coefficient space.

* Fit on the training split, predict each validation record with the mean image of
  its ``k`` nearest training records in (standardised) coefficient space.
* If skill > 0 for some ``k``, the signal is recoverable and the neural arms are
  failing for fixable reasons -> keep optimising models.
* If skill <= 0 for every ``k``, the signal is below the noise floor -> capacity
  work is provably wasted, and the honest deliverable is the bound itself.

It also prints the **signal-vs-separation curve**: image difference binned by
coefficient distance, using *within-pickle* pairs. Adjacent epochs of one run have
near-identical coefficients, so their image difference estimates the noise floor;
if that floor does not fall below the difference seen for far-apart coefficients,
no signal exists.

Pairing: folds share the corpus, so the k-NN arm is scored per fold exactly like
every other arm, and the k=1 vs k=5 comparison is paired on folds via
``ml.zernike.eval_stats``.

Usage
-----
    python scripts/diagnose_snr_ceiling.py
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
from ml.zernike.eval_stats import (
    cohens_dz,
    min_attainable_pvalue,
    sign_flip_pvalue,
)

FAMILY = "slm_zernike_shaping"
K_LIST = (1, 3, 5, 10)


def folds_of(records):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(i, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for i, k in enumerate(keys)]


def load(dataset, positions):
    coeffs, images = [], []
    for p in positions:
        s = dataset[p]
        coeffs.append(s["coeffs"].reshape(-1).numpy())
        images.append(s["image"].reshape(-1).numpy())
    return np.stack(coeffs).astype(np.float64), np.stack(images).astype(np.float32)


def score(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    t = torch.from_numpy(truth).reshape(-1, 1, 64, 64)
    p = torch.from_numpy(pred).reshape(-1, 1, 64, 64)
    return constant_baseline_metrics(p, t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/snr_ceiling.json")
    args = ap.parse_args(argv)

    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY])
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=64, image_mode="peak"), use_cache=True)

    started = time.perf_counter()
    folds = folds_of(records)
    rows: list[dict] = []
    for fold_i, train_pos, val_pos in folds:
        x_tr, y_tr = load(dataset, train_pos)
        x_va, y_va = load(dataset, val_pos)

        # Standardise using TRAIN statistics only. Noll modes differ in magnitude by
        # orders, so an unstandardised distance would be dominated by a few modes.
        mu = x_tr.mean(0, keepdims=True)
        sd = x_tr.std(0, keepdims=True)
        sd[sd <= 0] = 1.0
        z_tr = (x_tr - mu) / sd
        z_va = (x_va - mu) / sd

        # Squared euclidean distances, (n_val, n_train)
        d2 = ((z_va[:, None, :] - z_tr[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(d2, axis=1)

        const = np.repeat(y_tr.mean(0, keepdims=True), len(y_va), axis=0)
        m_const = score(const, y_va)
        row = {"fold": fold_i, "r2_const": m_const["r2_const"],
               "mse_const": m_const["mse"]}
        for k in K_LIST:
            nn = order[:, :k]
            pred = y_tr[nn].mean(axis=1)          # mean image of the k neighbours
            m = score(pred, y_va)
            row[f"skill_k{k}"] = m["skill"]
            row[f"mse_k{k}"] = m["mse"]
        rows.append(row)
        print("fold %2d  r2_const=%+.4f  " % (fold_i, m_const["r2_const"])
              + "  ".join("skill k=%-2d %+.4f" % (k, row[f"skill_k{k}"]) for k in K_LIST),
              flush=True)

    print("\n=== k-NN (no capacity, no optimiser, no loss) vs constant ===")
    summary: dict[str, float] = {}
    for k in K_LIST:
        vals = [r[f"skill_k{k}"] for r in rows]
        summary[f"skill_k{k}"] = float(np.mean(vals))
        print("  k=%-3d mean skill = %+.4f  (min %+.4f, max %+.4f)"
              % (k, np.mean(vals), np.min(vals), np.max(vals)))
    base = [r["skill_k1"] for r in rows]
    print("\n  r2_const mean = %+.4f"
          % float(np.mean([r["r2_const"] for r in rows])))
    for k in K_LIST[1:]:
        diffs = [r[f"skill_k{k}"] - v for r, v in zip(rows, base)]
        print("  paired k=%d minus k=1: mean=%+.4f p=%.4f (min %.4f) dz=%+.2f"
              % (k, np.mean(diffs), sign_flip_pvalue(diffs),
                 min_attainable_pvalue(len(diffs)), cohens_dz(diffs)))

    # ---- signal-vs-separation curve, within-pickle pairs --------------------
    print("\n=== image difference vs coefficient distance (within-pickle pairs) ===")
    curve = signal_curve(dataset, records)
    for entry in curve["bins"]:
        print("  |dc| in [%s): n=%5d  mean|di|=%.6g  rms|di|=%.6g"
              % (entry["range"], entry["n"], entry["mean_abs"], entry["rms"]))

    verdict = "SIGNAL EXISTS (k-NN beats the constant)" if max(
        summary.values()) > 0 else "NO SIGNAL ABOVE NOISE (k-NN cannot beat the constant)"
    out = {"rows": rows, "summary": summary, "curve": curve,
           "verdict": verdict, "seconds": round(time.perf_counter() - started, 1)}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nVERDICT:", verdict)
    print("wrote", args.out)
    return 0


def signal_curve(dataset, records, edges=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)) -> dict:
    """Bin within-pickle pairs by coefficient distance and report image difference.

    Adjacent epochs of one optimisation run sit at small ``|dc|``, so their image
    difference is essentially the noise floor. The first bin's level *is* that floor.
    """
    pairs_dc: list[float] = []
    pairs_di: list[float] = []
    for key in sorted({str(r.path) for r in records}):
        pos = [i for i, r in enumerate(records) if str(r.path) == key]
        if len(pos) < 2:
            continue
        c, im = load(dataset, pos)
        mu, sd = c.mean(0, keepdims=True), c.std(0, keepdims=True)
        sd[sd <= 0] = 1.0
        z = (c - mu) / sd
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                pairs_dc.append(float(np.sqrt(((z[i] - z[j]) ** 2).sum())))
                pairs_di.append(float(np.sqrt(((im[i] - im[j]) ** 2).mean())))
    dcs = np.asarray(pairs_dc)
    dis = np.asarray(pairs_di)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dcs >= lo) & (dcs < hi)
        if not m.any():
            continue
        bins.append({"range": "[%.1f, %.1f)" % (lo, min(hi, 999.0)),
                     "n": int(m.sum()),
                     "mean_abs": float(dis[m].mean()),
                     "rms": float(np.sqrt((dis[m] ** 2).mean()))})
    return {"n_pairs": int(len(dcs)), "bins": bins}


if __name__ == "__main__":
    raise SystemExit(main())
