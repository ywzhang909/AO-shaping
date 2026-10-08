"""Is the Zernike-coefficient -> far-field map even nonlinear?

Diagnostic, not a model. For each of the 10 leave-one-pickle-out folds it

1. builds the canonical PCA (POD) basis of the **training** images only,
2. ridge-regresses the coefficient vector onto the K leading POD coefficients,
3. reconstructs the validation images and scores **val R^2** with the repo's
   canonical metric (``ml.zernike.metrics.batch_image_metrics``).

Why this is the right first experiment
--------------------------------------
Every architecture/training idea in the queue (capacity sweeps, FiLM, a
field-output head, EMA, Charbonnier) assumes the residual after a *linear*
map is the thing worth modelling. This script measures that residual's size
directly:

* if ridge lands near the physics model (+0.873) or the ConvNet (+0.900),
  the map is close to affine and capacity work is chasing noise;
* if ridge is far below both, nonlinearity is real and worth spending on.

It is also the literal "predict a linear operator from the coefficients"
architecture from the operator-learning literature (POD-DeepONet), reduced to
its closed-form special case -- so a good score here is a *result*, not just a
diagnostic.

Protocol
--------
Folds are ``str(record.path)`` leave-one-pickle-out, matching
``compare_models_cv.py`` and ``train_coeff.file_folds`` (the repo has no
sklearn). Scoring uses ``batch_image_metrics`` so the number is comparable with
every other val R^2 in the project. Per-fold deltas against the incumbent
physics number are tested with ``ml.zernike.eval_stats`` -- the single
statistics source.

Usage
-----
    python scripts/diagnose_pod_ridge.py
    python scripts/diagnose_pod_ridge.py --k 8 16 32 64 128 256 --folds 0 1 2
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
from ml.zernike.metrics import batch_image_metrics

FAMILY = "slm_zernike_shaping"


def build_folds(records, protocol: str = "file") -> list[tuple[str, list[int], list[int]]]:
    """Leave-one-pickle-out (or leave-one-objective-out) folds on the path group key."""
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    if protocol == "file":
        keys = sorted(groups)
    else:
        by_objective: dict[str, list[str]] = {}
        for key in groups:
            parts = Path(key).stem.split("_")
            objective = parts[3] if len(parts) > 4 else "other"
            by_objective.setdefault(objective, []).append(key)
        keys = sorted(by_objective)
        merged = {k: [p for key in by_objective[k] for p in groups[key]] for k in keys}
        groups = merged
    folds = []
    for index, held in enumerate(keys):
        val = groups[held]
        train = [p for key, positions in groups.items() if key != held for p in positions]
        folds.append((held, train, val))
    return folds


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Multi-output ridge with an intercept, solved in the dual (n << d regime).

    ``x`` is (n_samples, n_features) and ``y`` is (n_samples, n_targets). The
    corpus has ~900 training rows and 136 features, so the Gram matrix is far
    cheaper than the 136x136 primal and is better conditioned here.
    """
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc @ xc.T
    gram.flat[:: gram.shape[0] + 1] += alpha
    dual = np.linalg.solve(gram, yc)
    weights = xc.T @ dual
    return weights, x_mean, y_mean


def ridge_predict(weights, x_mean, y_mean, x: np.ndarray) -> np.ndarray:
    return (x - x_mean) @ weights + y_mean


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", nargs="*", type=int, default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="subset of fold indices (default: all)")
    ap.add_argument("--protocol", choices=["file", "objective"], default="file")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--image-mode", default="peak")
    ap.add_argument("--input-terms", type=int, default=None)
    ap.add_argument("--out", default="logs/pod_ridge.json")
    args = ap.parse_args(argv)

    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY], sources=list(ZERNIKE_COEFF_SOURCES)
    )
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index,
        config=MaterialiserConfig(grid=args.grid, image_mode=args.image_mode),
        use_cache=True,
    )

    def stack(positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
        coeffs, images = [], []
        for position in positions:
            sample = dataset[position]
            coeffs.append(sample["coeffs"].reshape(-1).numpy())
            images.append(sample["image"].reshape(-1).numpy())
        return np.stack(coeffs), np.stack(images)

    folds = build_folds(records, args.protocol)
    if args.folds:
        folds = [f for f in folds if f[0] in set(args.folds) or folds.index(f) in set(args.folds)]
    print(f"corpus {len(records)} records, {len(folds)} folds, "
          f"grid={args.grid} image_mode={args.image_mode}", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    for fold_index, (held, train_pos, val_pos) in enumerate(folds):
        x_train, y_train = stack(train_pos)
        x_val, y_val = stack(val_pos)
        if args.input_terms:
            x_train = x_train[:, : args.input_terms]
            x_val = x_val[:, : args.input_terms]
        # POD basis from the TRAINING images only -- fitting it on the full corpus
        # would leak validation pixels into the basis.
        centred = y_train - y_train.mean(axis=0, keepdims=True)
        # economical SVD on the (n_train, pixels) matrix
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        for k in args.k:
            if k > vt.shape[0]:
                continue
            basis = vt[:k]                       # (k, pixels)
            z_train = centred @ basis.T           # (n_train, k)
            weights, x_mean, z_mean = ridge_fit(x_train, z_train, args.alpha)
            z_val = ridge_predict(weights, x_mean, z_mean, x_val)
            prediction = z_val @ basis + y_train.mean(axis=0, keepdims=True)

            metrics = batch_image_metrics(
                torch.from_numpy(prediction.astype(np.float32)).reshape(-1, 1, args.grid, args.grid),
                torch.from_numpy(y_val.astype(np.float32)).reshape(-1, 1, args.grid, args.grid),
            )
            rows.append({
                "fold": fold_index, "held_out": Path(held).name, "k": k,
                "r2": float(metrics["r2"]), "mse": float(metrics["mse"]),
                "n_train": len(train_pos), "n_val": len(val_pos),
            })
            print(f"  fold {fold_index:2d} k={k:4d} r2={metrics['r2']:+.4f}", flush=True)

    # ---- summary + paired stats ------------------------------------------------
    from ml.zernike.eval_stats import cohens_dz, min_attainable_pvalue, sign_flip_pvalue

    print(f"\n=== val R^2 by K (mean over {len(folds)} folds) ===")
    summary: dict[int, dict] = {}
    for k in args.k:
        per_fold = [r["r2"] for r in rows if r["k"] == k]
        if not per_fold:
            continue
        summary[k] = {
            "n_folds": len(per_fold),
            "mean_r2": float(np.mean(per_fold)),
            "sd_r2": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
        }
        print(f"  K={k:4d}  R2={summary[k]['mean_r2']:+.4f} +/- {summary[k]['sd_r2']:.4f}")

    # Reference points measured elsewhere in this project on the SAME corpus
    # (10-fold file protocol, val R^2): physics(n_max=15) +0.8727, the tuned
    # physics(n_max=20, lr=0.01) +0.8803, U-Net[16..256] +0.9004, and the
    # ConvNet coeff model +0.8205 (single-fold). Paired deltas against the best
    # physics point are what decide whether a linear map is competitive.
    best_k = max(summary, key=lambda k: summary[k]["mean_r2"]) if summary else None
    paired: dict = {}
    if best_k is not None:
        base = {r["fold"]: r["r2"] for r in rows if r["k"] == best_k}
        for k in args.k:
            other = {r["fold"]: r["r2"] for r in rows if r["k"] == k}
            shared = sorted(set(base) & set(other))
            if not shared or k == best_k:
                continue
            diffs = [other[f] - base[f] for f in shared]
            paired[k] = {
                "n_pairs": len(diffs),
                "mean_diff": float(np.mean(diffs)),
                "p_signflip": sign_flip_pvalue(diffs),
                "cohens_dz": cohens_dz(diffs),
                "min_attainable_p": min_attainable_pvalue(len(diffs)),
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args),
        "corpus_records": len(records),
        "folds": len(folds),
        "rows": rows,
        "summary_by_k": {str(k): v for k, v in summary.items()},
        "paired_vs_best_k": {str(k): v for k, v in paired.items()},
        "best_k": best_k,
        "reference_val_r2_same_protocol": {
            "physics_nmax15": 0.8727,
            "physics_nmax20_lr0.01": 0.8803,
            "unet_16_256": 0.9004,
            "hybrid": "statistically indistinguishable from physics (p>=0.61)",
        },
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")
    print(f"\nbest K={best_k}  R2={summary[best_k]['mean_r2']:+.4f}" if best_k else "\nno rows")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
