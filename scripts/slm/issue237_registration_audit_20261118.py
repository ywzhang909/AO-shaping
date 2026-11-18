"""Offline coordinate/registration audit for issue #237 (no hardware access).

The detector saturation mask is a union made before any candidate translation is
selected.  A registered centre is an empirical model/data alignment only; this
script never labels a clipped hotspot centroid as the optical axis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/debug/slm_transfer/measured_forward_hw_probe_20261118_scan"
DEFAULT_OUTPUT = ROOT / "data/debug/slm_transfer/issue237_registration_audit_20261118"
CROP_HALF = 120
MAX_SEARCH_OFFSET_DETECTOR_PX = 256
SATURATION_THRESHOLD = 250


def _load_hw_module() -> Any:
    path = ROOT / "scripts/slm/measured_forward_hw_probe_20261107.py"
    spec = importlib.util.spec_from_file_location("_issue237_hw_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixed_support_ncc_map(
    image: np.ndarray, template: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return NCC for every legal template placement on a fixed detector mask.

    The mask and its observed moments are identical for every placement.  The
    template contributes zero outside its footprint, so saturation cannot move
    the score support as the candidate translation changes.
    """
    y = np.asarray(image, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if y.shape != m.shape or t.ndim != 2 or y.ndim != 2:
        raise ValueError("image/mask must match and all arrays must be 2-D")
    if t.shape[0] > y.shape[0] or t.shape[1] > y.shape[1]:
        raise ValueError("template must fit in detector")
    n = float(m.sum())
    if n < 2:
        raise ValueError("fixed support has fewer than two pixels")
    masked_y = y * m
    sy = float(masked_y.sum())
    syy = float((masked_y * y).sum())
    centered_y_variance = syy - sy * sy / n
    # Full correlation index (top + template_size - 1) is unambiguous.
    t_flip = t[::-1, ::-1]
    s_yt = fftconvolve(masked_y, t_flip, mode="full")
    s_t = fftconvolve(m.astype(np.float64), t_flip, mode="full")
    s_tt = fftconvolve(m.astype(np.float64), (t * t)[::-1, ::-1], mode="full")
    h, w = t.shape
    valid = (slice(h - 1, y.shape[0]), slice(w - 1, y.shape[1]))
    s_yt = s_yt[valid]
    s_t = s_t[valid]
    s_tt = s_tt[valid]
    numerator = s_yt - sy * s_t / n
    denominator = np.sqrt(
        np.maximum(s_tt - s_t * s_t / n, 0.0) * max(centered_y_variance, 0.0)
    )
    scores = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -1.0),
        where=denominator > 1e-12,
    )
    return scores, (h, w)


def _peak_in_window(
    scores: np.ndarray,
    baseline_top: np.ndarray,
    max_offset: int = MAX_SEARCH_OFFSET_DETECTOR_PX,
    step: int = 1,
    max_top: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    lo = np.maximum(baseline_top - max_offset, 0)
    hi = baseline_top + max_offset + 1
    if max_top is not None:
        hi = np.minimum(hi, np.asarray(max_top) + 1)
    hi = np.minimum(hi, np.asarray(scores.shape))
    window = scores[lo[0] : hi[0] : step, lo[1] : hi[1] : step]
    peak = lo + step * np.asarray(np.unravel_index(np.argmax(window), window.shape))
    return peak, float(scores[tuple(peak)])


def _affine_fit(
    rows: list[tuple[np.ndarray, np.ndarray]], mask: np.ndarray
) -> tuple[float, float]:
    x = np.concatenate([p[mask] for _, p in rows])
    y = np.concatenate([i[mask] for i, _ in rows])
    design = np.column_stack((np.ones_like(x), x))
    intercept, gain = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(gain)


def _metrics(
    image: np.ndarray, prediction: np.ndarray, mask: np.ndarray, intercept: float, gain: float
) -> dict[str, float]:
    y = image[mask]
    p = prediction[mask]
    fitted = intercept + gain * p
    return {
        "ncc": float(np.corrcoef(y, p)[0, 1]),
        "rel_l2_affine_train_fit": float(np.linalg.norm(y - fitted) / np.linalg.norm(y)),
    }


def _place(template: np.ndarray, detector_shape: tuple[int, int], top: np.ndarray) -> np.ndarray:
    out = np.zeros(detector_shape, dtype=np.float64)
    h, w = template.shape
    y, x = map(int, top)
    out[y : y + h, x : x + w] = template
    return out


def _search_train(c_points: list[dict[str, Any]], a_mask: np.ndarray, baseline_top: np.ndarray) -> dict[str, Any]:
    """Two-stage coarse/local train search under one fixed mask variant."""
    train = [p for p in c_points if p["exposure_ms"] in (5.0, 8.0)]
    held = [p for p in c_points if p["exposure_ms"] not in (5.0, 8.0)]
    pooled = np.mean(
        [fixed_support_ncc_map(p["image"], p["prediction"], a_mask)[0] for p in train], axis=0
    )
    max_top = np.asarray(a_mask.shape) - np.asarray(train[0]["prediction"].shape)
    coarse_top, _ = _peak_in_window(pooled, baseline_top, step=8, max_top=max_top)
    lo = np.maximum(coarse_top - 8, 0)
    hi = np.minimum(coarse_top + 9, max_top + 1)
    local = pooled[lo[0] : hi[0], lo[1] : hi[1]]
    best = lo + np.asarray(np.unravel_index(np.argmax(local), local.shape))
    held_pooled = np.mean(
        [fixed_support_ncc_map(p["image"], p["prediction"], a_mask)[0] for p in held], axis=0
    )
    held_best, held_best_ncc = _peak_in_window(held_pooled, baseline_top, max_top=max_top)
    return {
        "best": best,
        "train_ncc": float(pooled[tuple(best)]),
        "held_out_ncc_at_train_shift": float(held_pooled[tuple(best)]),
        "held_out_best_translation_yx_ccd_px": (2 * (held_best - baseline_top)).tolist(),
        "held_out_best_ncc": held_best_ncc,
        "pooled_map": pooled,
    }


def _concentration(image: np.ndarray, placed: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Pixel-level leverage: NCC before/after removing the top 1% |contribution|."""
    y = image[mask]
    p = placed[mask]
    ybar, pbar = y.mean(), p.mean()
    contrib = (y - ybar) * (p - pbar)
    total = np.abs(contrib).sum()
    # Drop 1% of retained TEMPLATE-support pixels, not 1% of the 1.26M
    # detector pixels (which would trivially remove nearly the entire template).
    active_indices = np.flatnonzero(p != 0)
    n_drop = max(1, int(np.ceil(0.01 * active_indices.size)))
    ranked = np.argsort(np.abs(contrib[active_indices]))
    dropped_indices = active_indices[ranked[-n_drop:]]
    keep = np.ones(y.size, dtype=bool)
    keep[dropped_indices] = False

    def _ncc(yv: np.ndarray, pv: np.ndarray) -> float:
        numerator = float(np.sum((yv - yv.mean()) * (pv - pv.mean())))
        denom = float(np.sqrt(np.sum((yv - yv.mean()) ** 2) * np.sum((pv - pv.mean()) ** 2)))
        return numerator / denom if denom > 0 else 0.0

    return {
        "ncc_full": _ncc(y, p),
        "ncc_without_top_1pct_contributors": _ncc(y[keep], p[keep]),
        "top_1pct_contribution_share": float(np.abs(contrib[~keep]).sum() / max(total, 1e-12)),
        "top_1pct_template_pixels": int(n_drop),
    }


def _robustness_c(
    points: list[dict[str, Any]],
    mask: np.ndarray,
    baseline_top: np.ndarray,
    n_null: int = 200,
    seed: int = 237,
) -> dict[str, Any]:
    """Null distribution, mask-halo sensitivity and surviving-energy bookkeeping."""
    c_points = [p for p in points if p["phase"] == "C"]
    reference = _search_train(c_points, mask, baseline_top)
    # 1) Null: same fixed-mask metric at random shifts inside the same window.
    rng = np.random.default_rng(seed)
    max_top = np.asarray(mask.shape) - np.asarray(c_points[0]["prediction"].shape)
    lo = np.maximum(baseline_top - MAX_SEARCH_OFFSET_DETECTOR_PX, 0)
    hi = np.minimum(baseline_top + MAX_SEARCH_OFFSET_DETECTOR_PX + 1, max_top + 1)
    random_tops = np.column_stack(
        (rng.integers(lo[0], hi[0], n_null), rng.integers(lo[1], hi[1], n_null))
    )
    null_values = np.array([float(reference["pooled_map"][tuple(t)]) for t in random_tops])
    peak = reference["train_ncc"]
    null = {
        "seed": seed,
        "n": n_null,
        "window_half_detector_px": MAX_SEARCH_OFFSET_DETECTOR_PX,
        "mean": float(null_values.mean()),
        "std": float(null_values.std()),
        "p95": float(np.percentile(null_values, 95)),
        "p99": float(np.percentile(null_values, 99)),
        "max": float(null_values.max()),
        "peak_train_ncc": peak,
        "z_score": float((peak - null_values.mean()) / max(null_values.std(), 1e-12)),
        "percentile_of_peak": float(np.mean(null_values < peak) * 100.0),
        "peak_above_all_null": bool(peak > null_values.max()),
    }
    # 2) Mask-halo sensitivity: enlarge the saturation exclusion by a halo.
    distance = distance_transform_edt(mask)
    halos: list[dict[str, Any]] = []
    for halo_ccd in (0, 30, 60, 90, 120):
        halo_sampled = halo_ccd / 2
        a_mask = mask & (distance > halo_sampled)
        result = _search_train(c_points, a_mask, baseline_top)
        placed = _place(c_points[0]["prediction"], a_mask.shape, result["best"])
        retained = a_mask & (placed != 0)
        halos.append(
            {
                "halo_ccd_px": halo_ccd,
                "translation_yx_ccd_px": (2 * (result["best"] - baseline_top)).tolist(),
                "train_peak_ncc": result["train_ncc"],
                "held_out_ncc_at_train_shift": result["held_out_ncc_at_train_shift"],
                "mask_retained_fraction": float(a_mask.mean()),
                "prediction_energy_retained_fraction": float(
                    np.sum(placed[retained] ** 2)
                    / max(np.sum(c_points[0]["prediction"] ** 2), 1e-12)
                ),
            }
        )
    # 3) Surviving-energy bookkeeping at the halo-0 winning shift.
    best = reference["best"]
    energy_rows: list[dict[str, Any]] = []
    for p in c_points:
        placed = _place(p["prediction"], mask.shape, best)
        retained = mask & (placed != 0)
        full_energy = float(np.sum(p["prediction"] ** 2))
        energy_rows.append(
            {
                "file": p["file"],
                "exposure_ms": p["exposure_ms"],
                "prediction_energy_retained_fraction": float(
                    np.sum(placed[retained] ** 2) / max(full_energy, 1e-12)
                ),
                "measured_energy_retained_fraction": float(
                    np.sum((p["image"] * mask) ** 2) / max(np.sum(p["image"] ** 2), 1e-12)
                ),
                **_concentration(p["image"], placed, mask),
            }
        )
    return {
        "scope": "C panel only; U is bit-identical to blank and carries no shaped-phase evidence",
        "null": null,
        "mask_halo_sensitivity": halos,
        "energy_at_winning_shift": energy_rows,
    }


def _point(path: Path) -> dict[str, Any]:
    with np.load(path) as archive:
        raw = np.asarray(archive["raw_frames"], dtype=np.float64)
        phase = str(archive["phase"])
        exposure = float(archive["exposure_ms"])
        panel = np.asarray(archive["panel_phase"], dtype=np.float32)
        panel_hash = str(archive["panel_phase_sha256"])
    mean = raw.mean(axis=0)
    hotspot = mean >= 0.5 * float(mean.max())
    ys, xs = np.nonzero(hotspot)
    return {
        "file": path.name,
        "phase": phase,
        "exposure_ms": exposure,
        "raw": raw,
        "mean": mean,
        "panel": panel,
        "panel_hash": panel_hash,
        "hotspot_centroid_yx": [float(ys.mean()), float(xs.mean())],
        "hotspot_pixels": int(hotspot.sum()),
        "clipped_pixels_any_repeat": int(np.any(raw >= SATURATION_THRESHOLD, axis=0).sum()),
    }


def _branch_audit(
    points: list[dict[str, Any]], model: Any, mask: np.ndarray, baseline_top: np.ndarray
) -> dict[str, Any]:
    hw = model["hw"]
    # This scan's U panel hash is exactly the blank hash: it is not a shaped-phase
    # registration sample and must never be pooled into or presented as one.
    usable = [p for p in points if p["phase"] == "C"]
    for point in usable:
        point["prediction"] = np.asarray(hw.predict_phase(model["model"], point["panel"]), dtype=np.float64)
        point["prediction"] = np.maximum(point["prediction"], 0.0)
        point["image"] = point["mean"][model["parity"][0] :: 2, model["parity"][1] :: 2] / point["exposure_ms"]
        point["image"] -= float(np.median(point["image"]))
    out: dict[str, Any] = {}
    for phase in ("U", "C"):
        branch = [p for p in usable if p["phase"] == phase]
        train = [p for p in branch if p["exposure_ms"] in (5.0, 8.0)]
        held = [p for p in branch if p["exposure_ms"] not in (5.0, 8.0)]
        if not train:
            continue
        pooled = np.mean(
            [fixed_support_ncc_map(p["image"], p["prediction"], mask)[0] for p in train], axis=0
        )
        held_pooled = (
            np.mean(
                [fixed_support_ncc_map(p["image"], p["prediction"], mask)[0] for p in held], axis=0
            )
            if held
            else None
        )
        # Stage 1: coarse global search on a 16 CCD px grid; stage 2: local 2 CCD
        # px refinement in a +-16 CCD px box around the coarse winner.
        max_top = np.asarray(mask.shape) - np.asarray(train[0]["prediction"].shape)
        coarse_top, coarse_ncc = _peak_in_window(pooled, baseline_top, step=8, max_top=max_top)
        lo = np.maximum(coarse_top - 8, 0)
        hi = np.minimum(coarse_top + 9, max_top + 1)
        local_window = pooled[lo[0] : hi[0], lo[1] : hi[1]]
        best = lo + np.asarray(np.unravel_index(np.argmax(local_window), local_window.shape))
        best_score = float(pooled[tuple(best)])
        held_best = held_ncc = None
        if held_pooled is not None:
            held_best, held_ncc = _peak_in_window(held_pooled, baseline_top, max_top=max_top)
        baseline = baseline_top.copy()
        held_translation = (
            (2 * (held_best - baseline)).tolist() if held_best is not None else None
        )
        placed_best = {p["file"]: _place(p["prediction"], mask.shape, best) for p in branch}
        placed_base = {p["file"]: _place(p["prediction"], mask.shape, baseline) for p in branch}
        fits = {
            variant: _affine_fit([(p["image"], placed[p["file"]]) for p in train], mask)
            for variant, placed in (("baseline", placed_base), ("recentered", placed_best))
        }
        rows: list[dict[str, Any]] = []
        for p in branch:
            score_map, _ = fixed_support_ncc_map(p["image"], p["prediction"], mask)
            individual, individual_score = _peak_in_window(score_map, baseline_top, max_top=max_top)
            placed = placed_best[p["file"]]
            retained = mask & (placed != 0)
            rows.append(
                {
                    "file": p["file"],
                    "exposure_ms": p["exposure_ms"],
                    "split": "train" if p["exposure_ms"] in (5.0, 8.0) else "held_out",
                    "individual_peak_translation_yx_ccd_px": (2 * (individual - baseline)).tolist(),
                    "individual_peak_ncc": individual_score,
                    "retained_support_pixels": int(retained.sum()),
                    "prediction_energy_retained_fraction": float(
                        np.sum((placed[retained]) ** 2) / max(np.sum(p["prediction"] ** 2), 1e-12)
                    ),
                    "baseline": _metrics(
                        p["image"], placed_base[p["file"]], mask, *fits["baseline"]
                    ),
                    "recentered": _metrics(
                        p["image"], placed, mask, *fits["recentered"]
                    ),
                }
            )
        intercept, gain = fits["recentered"]
        held_peaks = [r["individual_peak_translation_yx_ccd_px"] for r in rows if r["split"] == "held_out"]
        out[phase] = {
            "train_exposures_ms": [5.0, 8.0],
            "held_out_exposures_ms": [12.0, 16.0, 20.0],
            "baseline_top_sampled_yx": baseline.tolist(),
            "recentered_top_sampled_yx": best.tolist(),
            "fitted_translation_yx_ccd_px": (2 * (best - baseline)).tolist(),
            "registered_center_xy_not_optical_axis": (
                np.asarray(model["model"].camera_crop_center) + 2 * (best - baseline)[::-1]
            ).tolist(),
            "pooled_train_peak_ncc": best_score,
            "coarse_grid_peak_translation_yx_ccd_px": (2 * (coarse_top - baseline)).tolist(),
            "coarse_grid_peak_ncc": coarse_ncc,
            "held_out_pooled_peak_translation_yx_ccd_px": held_translation,
            "held_out_pooled_peak_ncc": held_ncc,
            "train_peak_positive": bool(best_score > 0.0),
            "affine_fit_train_intercept": intercept,
            "affine_fit_train_gain": gain,
            "held_out_individual_translation_range_yx_ccd_px": (
                [np.min(held_peaks, axis=0).tolist(), np.max(held_peaks, axis=0).tolist()]
                if held_peaks
                else None
            ),
            "points": rows,
        }
    return out


def run(source: Path, output: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Run the fixed-support audit and write only the new output directory."""
    if output.exists() and not refresh:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if refresh and output.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("--refresh is restricted to this audit's default output")
    output.mkdir(parents=True, exist_ok=refresh)
    hw = _load_hw_module()
    log = hw.RunLog(output / "model_load.log", quiet=True)
    model_side = hw.build_model_side(log)
    log.close()
    points = [_point(path) for path in sorted((source / "frames").glob("scan_*.npz"))]
    if len(points) != 15:
        raise ValueError(f"expected 15 scan points, found {len(points)}")
    raw_union = np.zeros(points[0]["raw"].shape[1:], dtype=bool)
    for p in points:
        raw_union |= np.any(p["raw"] >= SATURATION_THRESHOLD, axis=0)
    parity = np.array([int(model_side.cam_y.min() % 2), int(model_side.cam_x.min() % 2)])
    mask = ~raw_union[parity[0] :: 2, parity[1] :: 2]
    baseline_top = np.array(
        [(int(model_side.cam_y.min()) - parity[0]) // 2, (int(model_side.cam_x.min()) - parity[1]) // 2]
    )
    model = {"hw": hw, "model": model_side, "parity": parity}
    branches = _branch_audit(points, model, mask, baseline_top)
    robustness = _robustness_c(points, mask, baseline_top)
    hashes: dict[str, list[str]] = {}
    for p in points:
        hashes.setdefault(p["panel_hash"], []).append(p["phase"])
    geometry = [
        {
            "file": p["file"],
            "phase": p["phase"],
            "exposure_ms": p["exposure_ms"],
            "hotspot_centroid_yx_not_optical_axis": p["hotspot_centroid_yx"],
            "hotspot_pixels": p["hotspot_pixels"],
            "clipped_pixels_any_repeat": p["clipped_pixels_any_repeat"],
        }
        for p in points
    ]
    payload: dict[str, Any] = {
        "source": str(source),
        "hardware_used": False,
        "method": "Frozen M3 prediction; detector sampled on exact archived parity; fixed union raw>=250 saturation mask across all 15 points; full fixed-support Pearson coarse search within +/-512 CCD px of frozen crop; affine amplitude fit on 5/8 ms train only; no PSF, rotation, clipping-dependent crop, or production-fit edits.",
        "limitations": [
            "All points contain clipped pixels; only the fixed unsaturated support is scored.",
            "The 2 CCD px sampled grid limits translation evidence to 2 px increments.",
            "Registered centres are model/data coordinates, not an independently measured optical axis.",
            "U panel hash equals blank on this capture, so U is not evidence of a distinct shaped U target.",
            "A negative or weak global NCC is explicit inability to validate registration, not a license for further optimizer tweaking.",
        ],
        "frozen_model_center_xy": [float(v) for v in model_side.camera_crop_center],
        "mask": {
            "threshold_counts": SATURATION_THRESHOLD,
            "full_detector_union_excluded": int(raw_union.sum()),
            "sampled_retained": int(mask.sum()),
            "sampled_total": int(mask.size),
            "retained_fraction": float(mask.mean()),
            "parity_yx": parity.tolist(),
        },
        "panel_hashes_to_phases": hashes,
        "phase_hash_warning": any("blank" in phases and "U" in phases for phases in hashes.values()),
        "geometry": geometry,
        "branches": branches,
        "robustness_C": robustness,
        "verdict": "stale crop coordinates exist, magnitude ~(+124,-72) px on the C panel, but a universal optical-axis/registration correction is not validated by this data",
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    return payload


def _report(payload: dict[str, Any]) -> str:
    robust = payload["robustness_C"]
    null = robust["null"]
    lines = [
        "# Issue #237 registration audit (offline)",
        "",
        "No hardware was used. Hotspot centroids below are explicitly not optical-axis measurements.",
        "",
        f"- Fixed union saturation mask: {payload['mask']['sampled_retained']}/{payload['mask']['sampled_total']} sampled pixels retained.",
        "- **The U panel is bit-identical to blank (same recorded SHA-256)**, so this dataset contains no shaped-phase registration evidence for U; only the C panel is evidence, and only at the listed exposures. U is never pooled into a global shift.",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "The random-shift null rejects a chance alignment at halo 0, but the fitted peak moves by tens to hundreds of pixels as the exclusion halo grows. Therefore the halo-0 shift is mask-sensitive wing alignment, not a robust axis correction. The retained support carries only ~5% of the predicted L2 energy (the other ~95% lies inside the clipped core), and the C registered centre (x≈1228.6, y≈1067.2) is ~35 CCD px from the clipped hotspot centroid (y≈1102), so the model core and measured wings do not agree on one centre. All energy fractions below are squared-intensity (L2-energy) fractions. Top-1% removal means 1% of retained template-support pixels, ranked by absolute centred NCC product; the detector mask is unchanged except for those explicitly dropped pixels. Train: C 5/8 ms (5 ms alone peaks at (162,-98) and disagrees); held-out: C 12/16/20 ms.",
        "",
        "## Null distribution (same fixed-mask metric, 200 random shifts)",
        "",
        "| mean | std | p95 | p99 | max | peak | z | percentile | peak above all null |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {null['mean']:.4f} | {null['std']:.4f} | {null['p95']:.4f} | {null['p99']:.4f} | "
        f"{null['max']:.4f} | {null['peak_train_ncc']:.4f} | {null['z_score']:.1f} | "
        f"{null['percentile_of_peak']:.1f}% | {null['peak_above_all_null']} |",
        "",
        "## Mask-halo sensitivity (saturation exclusion enlarged by halo)",
        "",
        "| halo (CCD px) | (dy, dx) CCD px | train NCC | held-out NCC at shift | mask retained | pred energy retained |",
        "|---|---|---|---|---|---|",
    ]
    for row in robust["mask_halo_sensitivity"]:
        lines.append(
            f"| {row['halo_ccd_px']} | {row['translation_yx_ccd_px']} | {row['train_peak_ncc']:.4f} | "
            f"{row['held_out_ncc_at_train_shift']:.4f} | {row['mask_retained_fraction']:.4f} | "
            f"{row['prediction_energy_retained_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Surviving-energy bookkeeping at the winning shift (halo 0)",
        "",
        "| file | exposure ms | pred energy in mask | measured energy in mask | NCC | NCC w/o top 1% | top-1% share |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in robust["energy_at_winning_shift"]:
        lines.append(
            f"| {row['file']} | {row['exposure_ms']:g} | {row['prediction_energy_retained_fraction']:.4f} | "
            f"{row['measured_energy_retained_fraction']:.4f} | {row['ncc_full']:.4f} | "
            f"{row['ncc_without_top_1pct_contributors']:.4f} | {row['top_1pct_contribution_share']:.3f} |"
        )
    for phase, result in payload["branches"].items():
        lines += [
            "",
            f"## {phase} branch",
            f"- Coarse (16 CCD px grid) peak NCC: `{result['coarse_grid_peak_ncc']:.4f}` at `{result['coarse_grid_peak_translation_yx_ccd_px']}`.",
            f"- Pooled train NCC at recenter: `{result['pooled_train_peak_ncc']:.6f}`.",
            f"- Translation (dy, dx) CCD px: `{result['fitted_translation_yx_ccd_px']}`.",
            f"- Registered centre (x, y), not optical axis: `{result['registered_center_xy_not_optical_axis']}`.",
            f"- Held-out pooled peak/NCC: `{result['held_out_pooled_peak_translation_yx_ccd_px']}` / `{result['held_out_pooled_peak_ncc']}`.",
            f"- Held-out individual translation range (dy, dx) CCD px: `{result['held_out_individual_translation_range_yx_ccd_px']}`.",
            "",
            "| C file | exposure ms | baseline NCC | recentered NCC | baseline rel-L2 | recentered rel-L2 |",
            "|---|---|---|---|---|---|",
        ]
        for row in result["points"]:
            lines.append(
                f"| {row['file']} | {row['exposure_ms']:g} | {row['baseline']['ncc']:.4f} | "
                f"{row['recentered']['ncc']:.4f} | {row['baseline']['rel_l2_affine_train_fit']:.4f} | "
                f"{row['recentered']['rel_l2_affine_train_fit']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Refresh only this audit's default output")
    args = parser.parse_args()
    payload = run(args.source, args.output, refresh=args.refresh)
    for phase, branch in payload["branches"].items():
        print(phase, json.dumps({k: branch[k] for k in ("pooled_train_peak_ncc", "fitted_translation_yx_ccd_px", "held_out_individual_translation_range_yx_ccd_px")}))


if __name__ == "__main__":
    main()
