# -*- coding: utf-8 -*-
"""Offline multiplicative low-rank spatial-gain candidate on session B (no hardware).

Question
--------
The frozen physical forward on session B (``extend_b_20261118``) leaves a
structured residual.  Existing work corrects it with an ADDITIVE static global
term (M1, ``fit_m3_static_global``).  This script tests a different, cheaper
candidate: a MULTIPLICATIVE per-pixel gain shared by every row,

    pred_corrected = pred * (1 + sum_k g_k B_k(x, y)),

where ``B_k`` are a handful of smooth, low-rank spatial basis images built ONLY
from session B's TRAIN rows, and ``g_k`` are global scalars (one per basis),
also fitted on TRAIN only.  There is no dependence on a row's own phase, no
per-row selection and no test/calibration tuning.

The spatial basis is the leading right singular vectors (pixel-space modes) of the train residual
RATIO field ``(measured - pred) / pred`` (computed where ``pred`` is safely
above its train 5th percentile so the ratio does not explode).  That is the
multiplicative analogue of the residual image; its low-rank structure is exactly
what a shared gain can exploit.  ``K`` is predeclared (default 3, probe 1..4)
and never chosen using test data.

Two proximal-projection constraints are probed, both fitted on TRAIN only:
  * bounded coefficient norm  ``||g||_2 <= coeff_norm_max`` (group sparse-ish);
  * per-pixel positive gain floor  ``1 + B g >= gain_floor`` on every train pixel.
Projection is by Euclidean projection of the least-squares solution onto the
(convex) intersection; the script quantifies whether projecting changes the
train/test metrics and how many OOD rows it flags.

OOD fallback
------------
``pred`` can be tiny in places; the ratio ``measured/pred`` is then unstable and
the multiplicative model is unreliable.  A row is flagged OOD when its fraction
of low-prediction pixels (relative to a TRAIN-only threshold on that fraction)
exceeds the train 95th percentile.  Flagged rows are reported separately and
fall back to the frozen prediction (the projection is never applied to them).
The threshold is derived ONLY from train rows.

Gates
-----
Before touching the candidate, the script gates that its numpy metric blocks
reproduce the stored baseline ``results.json`` (fit_m3_full / static_global,
``origin_session``) to 1e-5, and that the stored npz predictions are already
scalar-recalibrated (one more train-only scalar refit must yield ~1.0).  If a
gate fails the script exits without writing a report.

Comparisons (all reported as all/train/test/per-family metric blocks):
  * ``baseline``     -- frozen fit_m3_full prediction, unchanged;
  * ``static_global``-- existing additive M1 correction (from its npz);
  * ``mult_gain``    -- multiplicative low-rank gain, unprojected;
  * ``mult_gain_proj``-- same, with proximal projection applied.

No fit.py or shared file is modified; no hardware or device lock is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PREDICTIONS = {
    "baseline": "predictions_fit_m3_full_origin_session.npz",
    "static_global": "predictions_fit_m3_static_global_origin_session.npz",
}
SPLIT_GROUPS = ("all", "train", "test")
FAMILIES = ("zernike", "smooth", "bandlimited")
METRICS = ("correlation", "relative_l2", "cv_abs_error")
GATE_TOL = 1e-5


# ------------------------------------------------------------------ helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data/debug/slm_transfer/extend_b_20261118",
        help="session-B dataset directory (absolute path; may live outside this worktree)",
    )
    parser.add_argument(
        "--baseline-results", type=Path, default=None,
        help="eval_transfer/results.json (default: <dataset>/eval_transfer/results.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="report JSON path (default: <worktree>/report/shape_trials/physical_refine/"
             "multiplicative_gain_b_20261119.json)",
    )
    parser.add_argument("--k", type=int, default=3,
                        help="predeclared number of spatial basis functions (default 3)")
    parser.add_argument("--probe-k", action="store_true",
                        help="also evaluate predeclared K=1..4 (still train-only, no test tuning)")
    parser.add_argument("--coeff-norm-max", type=float, default=0.15,
                        help="proximal constraint: max L2 norm of the coefficient vector g "
                             "(default 0.15; unprojected norm is ~0.26, so the bound bites)")
    parser.add_argument("--gain-floor", type=float, default=0.7,
                        help="proximal constraint: per-pixel multiplicative gain lower bound")
    parser.add_argument("--ratio-percentile", type=float, default=5.0,
                        help="prediction percentile below which the residual ratio is masked")
    parser.add_argument("--ood-quantile", type=float, default=0.95,
                        help="train quantile of low-prediction pixel fraction for OOD flag")
    parser.add_argument("--projection-iters", type=int, default=500,
                        help="max Dykstra iterations for the intersection projection")
    return parser.parse_args()


# ------------------------------------------------------------------ metrics

def metric_block(pred: np.ndarray, meas: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Mean of per-row metrics; identical definition to train_residual_corrector.

    ``cv_abs_error`` is computed per row then averaged (matches ``row_metrics``),
    not via the vectorised ``mean_metrics`` shortcut.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"n_rows": 0}
    p = pred[idx].astype(np.float64).reshape(idx.size, -1)
    t = meas[idx].astype(np.float64).reshape(idx.size, -1)
    pc = p - p.mean(axis=1, keepdims=True)
    tc = t - t.mean(axis=1, keepdims=True)
    corr = (pc * tc).sum(axis=1) / np.sqrt((pc * pc).sum(axis=1) * (tc * tc).sum(axis=1)).clip(min=1e-30)
    rel = np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1).clip(min=1e-9)
    cv_p = p.std(axis=1) / np.abs(p.mean(axis=1)).clip(min=1e-9)
    cv_t = t.std(axis=1) / np.abs(t.mean(axis=1)).clip(min=1e-9)
    return {
        "n_rows": int(idx.size),
        "correlation": float(corr.mean()),
        "relative_l2": float(rel.mean()),
        "cv_abs_error": float(np.abs(cv_p - cv_t).mean()),
    }


def grouped_metrics(pred: np.ndarray, meas: np.ndarray,
                    splits: np.ndarray, families: np.ndarray) -> dict[str, Any]:
    """Metrics by split (all/train/test) and per family within each split."""
    out: dict[str, Any] = {}
    for split in SPLIT_GROUPS:
        mask = np.ones(len(splits), bool) if split == "all" else splits == split
        out[split] = {
            "overall": metric_block(pred, meas, mask),
            "by_family": {family: metric_block(pred, meas, mask & (families == family))
                          for family in FAMILIES},
        }
    return out


def scalar_gain(pred: np.ndarray, meas: np.ndarray, mask: np.ndarray) -> float:
    """Least-squares scalar ``g = <p, m> / <p, p>`` over ``mask`` (flattened)."""
    p = pred[mask].astype(np.float64).ravel()
    m = meas[mask].astype(np.float64).ravel()
    denom = float(np.dot(p, p))
    if denom <= 0.0:
        raise SystemExit("degenerate prediction set: <p, p> <= 0")
    return float(np.dot(p, m) / denom)


# ------------------------------------------------------------------ gating

def gate_baseline(results: dict[str, Any], arrays: dict[str, dict[str, np.ndarray]],
                  splits: np.ndarray) -> dict[str, Any]:
    """Reproduce stored ``after`` metrics and the stored scalar gain."""
    report: dict[str, Any] = {"tolerance": GATE_TOL, "models": {}}
    for name, key in (("baseline", "fit_m3_full"),
                      ("static_global", "fit_m3_static_global")):
        arr = arrays[name]
        pred, meas = arr["prediction"], arr["measured"]
        ref = results["models"][key]["variants"]["origin_session"]
        ref_after = ref["metrics"]["after"]["by_split"]["all"]
        mine = metric_block(pred, meas, np.ones(len(splits), bool))
        diffs = {m: abs(mine[m] - ref_after[m]) for m in METRICS}
        # The stored npz ``prediction`` arrays are ALREADY scalar-recalibrated:
        # results.json's recalibration.gain is the factor that produced them
        # from the raw forward.  Consequently refitting one more scalar on the
        # train split must yield ~1.0 (no leftover scale mismatch).  The stored
        # gain is kept as provenance; the gate checks the internal consistency.
        gain_ref = float(ref["recalibration"]["gain"])
        gain_mine = scalar_gain(pred, meas, splits == "train")
        gain_diff = abs(gain_mine - 1.0)
        ok = max(max(diffs.values()), gain_diff) <= GATE_TOL
        report["models"][key] = {
            "recomputed_all": mine,
            "stored_after_all": {m: ref_after[m] for m in METRICS},
            "max_metric_abs_diff": max(diffs.values()),
            "stored_recalibration_gain": gain_ref,
            "refit_additive_scale": gain_mine,
            "refit_scale_diff_from_one": gain_diff,
            "passed": bool(ok),
        }
        report["note"] = (
            "npz prediction arrays are already gain-recalibrated; gate verifies metric "
            "reproduction and that no further train scalar is needed (refit ~= 1.0)"
        )
    report["all_passed"] = all(m["passed"] for m in report["models"].values())
    return report


# ------------------------------------------------------------------ basis

def build_spatial_basis(residual_ratio_train: np.ndarray, k: int,
                        valid: np.ndarray) -> dict[str, Any]:
    """Leading right singular vectors (spatial modes) of the train ratio field.

    ``residual_ratio_train`` has shape (n_train, H, W).  We centre it over the
    train rows, flatten each image and take the top-``k`` right singular vectors
    of the data matrix; those vectors, reshaped to (H, W), are the spatial basis
    images ``B_k``.  They are built from TRAIN rows only, are orthogonal on the
    train pixel set and are ordered by explained variance.  Each basis image is
    normalised to unit standard deviation over the TRAIN stable-pixel mask
    ``valid`` so the fitted coefficients and the norm-ball constraint are on an
    interpretable scale (one coefficient unit ~= one std of spatial modulation).
    """
    n, h, w = residual_ratio_train.shape
    matrix = residual_ratio_train.reshape(n, h * w).astype(np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    # Thin SVD is cheap for n ~ 80 and h*w ~ 26k.
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:k].reshape(k, h, w)
    valid_flat = valid.ravel()
    scales = np.array([float(b.ravel()[valid_flat].std()) for b in basis])
    scales[scales <= 0] = 1.0
    basis = basis / scales[:, None, None]
    energy = singular ** 2
    frac = float(energy[:k].sum() / energy.sum()) if energy.sum() > 0 else 0.0
    return {
        "basis": basis,
        "mean_ratio": mean.reshape(h, w),
        "singular_values": singular[:k].tolist(),
        "explained_variance_fraction": frac,
        "normalisation_std": scales.tolist(),
        "n_train": n,
        "pixel_count": h * w,
    }


def fit_coefficients(pred: np.ndarray, meas: np.ndarray, train_mask: np.ndarray,
                     basis: np.ndarray, ratio_floor: np.ndarray,
                     ratio_ceiling: np.ndarray) -> dict[str, Any]:
    """Least-squares fit of g in ``pred*(1 + sum g_k B_k) ~ measured`` on TRAIN.

    For a fixed pixel weight the model is linear in g:

        measured - pred = pred * (B g)   ->   target = A g

    with ``A(pixel) = pred(pixel) * B(pixel)`` and
    ``target = measured - pred``.  The stable-pixel mask is SHARED across all
    rows and derived from the TRAIN-mean prediction (pixels between the train
    floor and ceiling percentiles), so no per-row/test information enters.
    """
    p = pred[train_mask].astype(np.float64)
    m = meas[train_mask].astype(np.float64)
    mean_train_pred = p.mean(axis=0)
    valid = (mean_train_pred > ratio_floor) & (mean_train_pred < ratio_ceiling)
    # p has shape (n_rows, H, W); boolean mask on last two dims keeps rows.
    pf = p[:, valid].reshape(-1)
    target = (m[:, valid] - p[:, valid]).reshape(-1)
    bf = basis.reshape(basis.shape[0], -1)[:, valid.ravel()].T  # (n_valid, K)
    # For each row: A_row = pf_row[:, None] * bf  ->  (n_valid, K).  Rows differ
    # only by the scalar pred value at each pixel; stack them vertically.
    a = np.vstack([pf_row[:, None] * bf
                   for pf_row in p[:, valid]])
    coef, residuals, rank, _ = np.linalg.lstsq(a, target, rcond=None)
    return {
        "coefficients": coef.astype(np.float64).tolist(),
        "n_valid_pixels": int(valid.sum()),
        "rank": int(rank),
        "fit_residual_norm": float(residuals[0]) if residuals.size else None,
    }


# ------------------------------------------------------------------ projection

def prox_ball(g: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection onto the L2 ball ``||g|| <= radius``."""
    norm = float(np.linalg.norm(g))
    if norm <= radius or norm == 0.0:
        return g.copy()
    return g * (radius / norm)


def prox_halfspaces(g: np.ndarray, basis: np.ndarray, floor: float) -> np.ndarray:
    """Projection onto ``1 + B g >= floor`` for all pixels (cyclic halfspaces)."""
    out = g.copy()
    k = len(out)
    b = basis.reshape(k, -1)
    for _ in range(200):
        violated = 1.0 + b.T @ out < floor
        if not violated.any():
            break
        for j in np.flatnonzero(violated):
            row = b[:, j]
            r2 = float(row @ row)
            if r2 <= 0:
                continue
            slack = 1.0 + float(row @ out) - floor
            if slack < 0.0:
                out = out - (slack / r2) * row
    return out


def project_onto_intersection(g: np.ndarray, basis: np.ndarray,
                              coeff_norm_max: float, gain_floor: float,
                              iters: int) -> dict[str, Any]:
    """Dykstra projection of ``g`` onto norm-ball ∩ gain-floor halfspaces."""
    x = g.copy()
    p_ball = np.zeros_like(g)
    p_half = np.zeros_like(g)
    for _ in range(iters):
        y = prox_ball(x + p_ball, coeff_norm_max)
        p_ball = (x + p_ball) - y
        z = prox_halfspaces(y + p_half, basis, gain_floor)
        p_half = (y + p_half) - z
        if np.linalg.norm(z - x) <= 1e-12:
            x = z
            break
        x = z
    # Final feasibility check (the true projection satisfies both).
    final = prox_ball(x, coeff_norm_max)
    final = prox_halfspaces(final, basis, gain_floor)
    min_gain = float(np.min(1.0 + basis.reshape(len(final), -1).T @ final))
    norm = float(np.linalg.norm(final))
    return {"coefficients": final, "norm": norm, "min_gain": min_gain,
            "norm_constraint": coeff_norm_max, "gain_floor": gain_floor,
            "feasible": bool(norm <= coeff_norm_max + 1e-6 and min_gain >= gain_floor - 1e-6)}


def apply_multiplicative(pred: np.ndarray, basis: np.ndarray, g: np.ndarray,
                         ood_mask: np.ndarray) -> np.ndarray:
    """pred*(1 + sum g_k B_k); OOD rows fall back to the frozen prediction."""
    gain = 1.0 + np.tensordot(g, basis, axes=(0, 0))
    out = pred * gain
    out[ood_mask] = pred[ood_mask]
    return out


# ------------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    eval_dir = dataset / "eval_transfer"
    results_path = (args.baseline_results.resolve() if args.baseline_results
                    else eval_dir / "results.json")
    if not dataset.is_dir() or not results_path.is_file():
        raise SystemExit(f"missing dataset or results: dataset={dataset} results={results_path}")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, filename in PREDICTIONS.items():
        path = eval_dir / filename
        if not path.is_file():
            raise SystemExit(f"missing prediction bundle: {path}")
        with np.load(path, allow_pickle=False) as z:
            arrays[name] = {key: z[key] for key in ("prediction", "measured", "split", "family", "label")}

    base = arrays["baseline"]
    pred = base["prediction"]
    meas = base["measured"]
    splits = base["split"]
    families = base["family"]
    labels = base["label"]
    n_rows, h, w = pred.shape
    train_mask = splits == "train"
    test_mask = splits == "test"

    results = json.loads(results_path.read_text(encoding="utf-8"))
    gates = gate_baseline(results, arrays, splits)
    if not gates["all_passed"]:
        print(json.dumps(gates, indent=2))
        raise SystemExit("baseline reproduction gate FAILED; no candidate evaluated")

    # --- train-only prediction statistics for the ratio mask and OOD fallback
    train_pred = pred[train_mask].astype(np.float64)
    ratio_floor = float(np.percentile(train_pred, args.ratio_percentile))
    ratio_ceiling = float(np.percentile(train_pred, 99.0))
    low_frac_all = np.mean(pred.astype(np.float64) <= ratio_floor, axis=(1, 2))
    ood_threshold = float(np.quantile(low_frac_all[train_mask], args.ood_quantile))
    ood_mask = low_frac_all > ood_threshold

    # Shared TRAIN stable-pixel mask (train mean prediction), used both to
    # normalise the basis and for the coefficient least-squares fit.
    train_mean_pred = train_pred.mean(axis=0)
    valid2d = (train_mean_pred > ratio_floor) & (train_mean_pred < ratio_ceiling)
    train_meas = meas[train_mask].astype(np.float64)
    ratio_field = np.where(valid2d[None, :, :], (train_meas - train_pred) / train_pred.clip(min=1e-9), 0.0)

    ks = list(range(1, 5)) if args.probe_k else [args.k]
    fitted: dict[str, dict[str, Any]] = {}
    basis_arrays: dict[int, np.ndarray] = {}
    for k in ks:
        built = build_spatial_basis(ratio_field, k, valid2d)
        basis_arrays[k] = built["basis"]
        basis_summary = {
            key: val for key, val in built.items()
            if key not in ("basis", "mean_ratio")
        }
        fit = fit_coefficients(pred, meas, train_mask, built["basis"],
                               ratio_floor, ratio_ceiling)
        coef = np.asarray(fit["coefficients"], dtype=np.float64)
        plain = apply_multiplicative(pred, built["basis"], coef, ood_mask)
        proj = project_onto_intersection(coef, built["basis"], args.coeff_norm_max,
                                          args.gain_floor, args.projection_iters)
        corrected = apply_multiplicative(pred, built["basis"], proj["coefficients"], ood_mask)
        fitted[f"K{k}"] = {
            "k": k,
            "basis": basis_summary,
            "fit": fit,
            "coefficients_unprojected": coef.tolist(),
            "projection": {key: val for key, val in proj.items() if key != "coefficients"},
            "coefficients_projected": np.asarray(proj["coefficients"]).tolist(),
            "metrics": {
                "unprojected": grouped_metrics(plain, meas, splits, families),
                "projected": grouped_metrics(corrected, meas, splits, families),
            },
        }

    chosen = fitted[f"K{args.k}"]
    coef = np.asarray(chosen["coefficients_unprojected"], dtype=np.float64)
    proj = chosen["projection"]
    unproj_metrics = chosen["metrics"]["unprojected"]
    proj_metrics = chosen["metrics"]["projected"]

    # --- per-row provenance and OOD flag for every row
    rows = []
    for i in range(n_rows):
        rows.append({
            "index": i,
            "label": str(labels[i]),
            "split": str(splits[i]),
            "family": str(families[i]),
            "ood_flagged": bool(ood_mask[i]),
            "low_pred_fraction": float(low_frac_all[i]),
        })

    # --- proximal-constraint sensitivity (predeclared grid, train-only fit;
    #     no setting is selected on test data -- the primary is args defaults).
    #     Quantifies how projection changes train/test and OOD-row metrics.
    constraint_grid = (
        (0.15, 0.7),   # primary
        (0.263, 0.7),  # norm bound inactive (above unprojected norm)
        (0.15, 0.9),   # tighter gain floor
        (0.10, 0.5),   # tighter norm, looser floor
    )
    chosen_basis = basis_arrays[args.k]  # raw (K, H, W) basis for the chosen K
    constraint_trials: dict[str, Any] = {}
    for cnorm, cfloor in constraint_grid:
        pr = project_onto_intersection(coef, chosen_basis, cnorm, cfloor,
                                       args.projection_iters)
        corr_img = apply_multiplicative(pred, chosen_basis, pr["coefficients"], ood_mask)
        blk = grouped_metrics(corr_img, meas, splits, families)
        constraint_trials[f"norm{cnorm}_floor{cfloor}"] = {
            "coeff_norm_max": cnorm,
            "gain_floor": cfloor,
            "feasible": bool(pr["feasible"]),
            "min_gain": pr["min_gain"],
            "coefficient_norm": pr["norm"],
            "train": blk["train"]["overall"],
            "test": blk["test"]["overall"],
            "ood_rows_fallback": int(ood_mask.sum()),
        }

    # --- comparison blocks
    comparison: dict[str, Any] = {
        "baseline_frozen": grouped_metrics(pred, meas, splits, families),
        "static_global_M1": grouped_metrics(arrays["static_global"]["prediction"], meas,
                                            splits, families),
        "mult_gain_unprojected": unproj_metrics,
        "mult_gain_projected": proj_metrics,
    }

    def delta(a: dict[str, Any], b: dict[str, Any], split: str) -> dict[str, float]:
        return {m: float(a[split]["overall"][m] - b[split]["overall"][m]) for m in METRICS}

    verdict = {
        "primary_k": args.k,
        "delta_vs_frozen": {
            sp: {
                "unprojected": delta(unproj_metrics, comparison["baseline_frozen"], sp),
                "projected": delta(proj_metrics, comparison["baseline_frozen"], sp),
            } for sp in SPLIT_GROUPS
        },
        "delta_vs_M1_static": {
            sp: delta(proj_metrics, comparison["static_global_M1"], sp) for sp in SPLIT_GROUPS
        },
        "projection_effect": {
            sp: delta(proj_metrics, unproj_metrics, sp) for sp in SPLIT_GROUPS
        },
        "projection_feasible": bool(proj["feasible"]),
        "projection_changed_coefficients": bool(
            np.max(np.abs(np.asarray(chosen["coefficients_projected"])
                          - np.asarray(chosen["coefficients_unprojected"]))) > 1e-9),
        "ood_rows": int(ood_mask.sum()),
        "ood_flagged_train": int((ood_mask & train_mask).sum()),
        "ood_flagged_test": int((ood_mask & test_mask).sum()),
        "ood_fallback": "flagged rows use frozen prediction (no multiplicative gain)",
    }

    out_path = (args.out.resolve() if args.out else
                Path(__file__).resolve().parents[2] / "report" / "shape_trials" /
                "physical_refine" / "multiplicative_gain_b_20261119.json")
    report: dict[str, Any] = {
        "format_version": 1,
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve()),
        "question": "offline multiplicative low-rank spatial gain on B vs frozen/M1",
        "inputs": {
            "dataset": str(dataset),
            "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
            "baseline_results": str(results_path),
            "baseline_results_sha256": sha256_file(results_path),
            "prediction_bundles": {name: str(eval_dir / fn) for name, fn in PREDICTIONS.items()},
            "row_counts": {"all": n_rows, "train": int(train_mask.sum()),
                           "test": int(test_mask.sum())},
            "family_counts": {f: int((families == f).sum()) for f in FAMILIES},
        },
        "method": {
            "candidate": "pred * (1 + sum_k g_k B_k(x,y)), global coefficients on all rows",
            "basis": "top-k right singular vectors (spatial modes) of TRAIN residual-ratio field "
                     "(measured-pred)/pred on stable-prediction pixels",
            "fit": "least squares on TRAIN rows/pixels only; no B-test selection or tuning",
            "k": args.k,
            "k_probe": ks,
            "ratio_percentile": args.ratio_percentile,
            "ratio_ceiling_percentile": 99.0,
            "ood_quantile": args.ood_quantile,
            "ood_threshold_low_pred_fraction": ood_threshold,
            "ood_fallback": "frozen prediction for flagged rows; threshold from train only",
            "proximal_constraints": {
                "coeff_norm_max": args.coeff_norm_max,
                "gain_floor": args.gain_floor,
                "projection_iters": args.projection_iters,
                "algorithm": "Dykstra projection onto L2 ball ∩ per-pixel gain-floor halfspaces",
            },
        },
        "gates": gates,
        "comparison": comparison,
        "k_trials": fitted,
        "constraint_trials": constraint_trials,
        "rows": rows,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- console summary
    def show(tag: str, blk: dict[str, Any]) -> str:
        o = blk["test"]["overall"]
        return f"{tag}: test corr={o['correlation']:.4f} rel={o['relative_l2']:.4f} cv={o['cv_abs_error']:.4f}"

    print("gates passed:", gates["all_passed"])
    print("K =", args.k, "explained variance fraction =", round(chosen["basis"]["explained_variance_fraction"], 4))
    print("coefficients unprojected:", np.round(chosen["coefficients_unprojected"], 4).tolist())
    print("coefficients projected  :", np.round(chosen["coefficients_projected"], 4).tolist())
    print("projection feasible:", proj["feasible"], "min_gain:", round(proj["min_gain"], 4),
          "norm:", round(proj["norm"], 4))
    print("OOD flagged rows:", verdict["ood_rows"],
          "(train", verdict["ood_flagged_train"], "/ test", verdict["ood_flagged_test"], ")")
    for tag in comparison:
        print(" ", show(tag, comparison[tag]))
    print("report:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
