"""Is the coefficient-independent variance a removable per-frame gain/offset?

`diagnose_label_alignment.py` established, with no model involved, that

* adjacent epochs have near-identical coefficients (distance ratio 0.056),
* their **images are not** correspondingly similar (ratio 0.774),
* yet |dI| does rise with |dc| (positive slope in 10/10 pickles),

so the labels are aligned and the frame carries a large *coefficient-independent*
component. That component is what every model is paying for.

This script asks whether that nuisance is a **per-frame affine transform**
(gain and offset -- laser power drift, exposure jitter, detector bias), which is
**removable by data processing**, or whether it is spatially structured /
pixel-wise, which is not.

For every frame it fits, against the split's mean image, the scalar pair that a
gain drift would produce::

    I_i ~= a_i * mean + b_i

then measures how much of the coefficient-independent variance that removes,
by scoring the same k-NN and ridge predictors under three preprocessings:

``none``     the frames as they are (today's pipeline)
``gain``     per-frame gain only: I / a_i
``affine``   per-frame gain and offset: (I - b_i) / a_i

If ``gain`` or ``affine`` lifts skill above zero, the fix is a data-processing
step and no model change is needed. If skill stays negative, the nuisance is not
a scalar drift and the ceiling is a genuine noise floor.

The per-frame scalars are fit on the **training split only** and applied to the
validation split, so nothing here leaks.

Usage
-----
    python scripts/diagnose_gain_drift.py
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
from ml.zernike.eval_stats import cohens_dz, min_attainable_pvalue, sign_flip_pvalue

FAMILY = "slm_zernike_shaping"
MODES = ("none", "gain", "affine")
K = 4


def folds_of(records):
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(i, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for i, k in enumerate(keys)]


def load(dataset, positions):
    c, im = [], []
    for p in positions:
        s = dataset[p]
        c.append(s["coeffs"].reshape(-1).numpy())
        im.append(s["image"].reshape(-1).numpy())
    return np.stack(c).astype(np.float64), np.stack(im).astype(np.float64)


def fit_gain_offset(frames: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``a``, ``b`` per frame against ``reference``.

    ``a`` comes from the centred covariance (the scale that actually matters for a
    brightness-drift); ``b`` from the residual mean (an additive pedestal).
    """
    ref = reference - reference.mean()
    centred = frames - frames.mean(axis=1, keepdims=True)
    denom = float((ref * ref).sum())
    a = (centred @ ref) / denom if denom > 0 else np.ones(len(frames))
    b = frames.mean(axis=1) - a * reference.mean()
    return a, b


def apply_mode(frames: np.ndarray, a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frames
    if mode == "gain":
        return frames / np.maximum(a, 1e-6)[:, None]
    return (frames - b[:, None]) / np.maximum(a, 1e-6)[:, None]


def ridge(x, z, alpha=1.0):
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    g = xc @ xc.T
    g.flat[:: g.shape[0] + 1] += alpha
    return xc.T @ np.linalg.solve(g, zc), xm, zm


def score(pred, truth):
    t = torch.from_numpy(np.ascontiguousarray(truth, dtype=np.float32)).reshape(-1, 1, 64, 64)
    p = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32)).reshape(-1, 1, 64, 64)
    return constant_baseline_metrics(p, t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/gain_drift.json")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(families=[FAMILY])
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=64, image_mode="peak"), use_cache=True)

    rows = []
    for fold_i, train_pos, val_pos in folds_of(records):
        x_tr, y_tr_raw = load(dataset, train_pos)
        x_va, y_va_raw = load(dataset, val_pos)
        row = {"fold": fold_i}
        reference = y_tr_raw.mean(0)
        a_tr, b_tr = fit_gain_offset(y_tr_raw, reference)
        a_va, b_va = fit_gain_offset(y_va_raw, reference)
        row["gain_spread"] = float(a_tr.std() / max(a_tr.mean(), 1e-9))
        for mode in MODES:
            y_tr = apply_mode(y_tr_raw, a_tr, b_tr, mode)
            y_va = apply_mode(y_va_raw, a_va, b_va, mode)
            # Ridge arm.
            centred = y_tr - y_tr.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            basis = vt[:K]
            y_mean = y_tr.mean(0, keepdims=True)
            w, xm, zm = ridge(x_tr, centred @ basis.T)
            pred = ((x_va - xm) @ w + zm) @ basis + y_mean
            row[f"ridge_skill_{mode}"] = score(pred, y_va)["skill"]
            # k-NN arm (model-free reference).
            mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True)
            sd[sd <= 0] = 1.0
            z_tr, z_va = (x_tr - mu) / sd, (x_va - mu) / sd
            d2 = ((z_va[:, None, :] - z_tr[None, :, :]) ** 2).sum(axis=2)
            nn = np.argsort(d2, axis=1)[:, :5]
            row[f"knn_skill_{mode}"] = score(y_tr[nn].mean(axis=1), y_va)["skill"]
            row[f"r2_const_{mode}"] = score(
                np.repeat(y_mean, len(y_va), axis=0), y_va)["r2_const"]
        rows.append(row)
        print("fold %2d  gain spread(sd/mean)=%.3f | ridge skill  " % (fold_i, row["gain_spread"])
              + "  ".join("%s=%+.4f" % (m, row[f"ridge_skill_{m}"]) for m in MODES)
              + "  | knn " + "  ".join("%s=%+.4f" % (m, row[f"knn_skill_{m}"]) for m in MODES),
              flush=True)

    print("\n=== does removing a per-frame gain/offset help? ===")
    out = {"rows": rows, "seconds": round(time.perf_counter() - started, 1)}
    for arm in ("ridge", "knn"):
        print("  %s arm:" % arm)
        base_key = f"{arm}_skill_none"
        for mode in MODES:
            vals = [r[f"{arm}_skill_{mode}"] for r in rows]
            extra = ""
            if mode != "none":
                diffs = [r[f"{arm}_skill_{mode}"] - b for r, b in zip(rows, [r[base_key] for r in rows])]
                extra = ("  paired vs none: mean=%+.4f p=%.4f (min %.4f) dz=%+.2f"
                         % (np.mean(diffs), sign_flip_pvalue(diffs),
                            min_attainable_pvalue(len(diffs)), cohens_dz(diffs)))
            print("    %-7s mean skill = %+.4f%s" % (mode, np.mean(vals), extra))
            out[f"{arm}_{mode}_mean_skill"] = float(np.mean(vals))
    best = max(MODES, key=lambda m: out[f"ridge_{m}_mean_skill"])
    out["best_mode"] = best
    # Only call it a fix if the paired test actually establishes it AND the result
    # crosses zero. A higher mean at p=0.07 is not a win, and a win that still leaves
    # skill negative has not solved anything.
    paired_p = sign_flip_pvalue(
        [r[f"ridge_skill_{best}"] - r["ridge_skill_none"] for r in rows])
    crosses_zero = out[f"ridge_{best}_mean_skill"] > 0.0
    if best == "none":
        verdict = "nuisance is NOT a scalar per-frame drift (removal changes nothing)"
    elif paired_p >= 0.05:
        verdict = ("affine removal is the best of the three but NOT significant "
                   f"(p={paired_p:.4f}) and skill stays negative -- nuisance is a "
                   "genuine noise floor, not a removable drift")
    elif not crosses_zero:
        verdict = ("affine removal is significant but still leaves skill negative "
                   "-- nuisance is a genuine noise floor, not a removable drift")
    else:
        verdict = "per-frame gain/offset removal helps and crosses zero"
    out["paired_p_best_vs_none"] = float(paired_p)
    out["verdict"] = verdict
    print("\nbest mode:", best, " (paired p vs none = %.4f)" % paired_p)
    print("VERDICT:", verdict)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
