"""Square-1000 frozen-model inherent-defect simulation (2026-10-30).

Answers one question offline, with no device access:

    the frozen ``FrozenDefocusForward`` checkpoint predicts a nearly uniform
    field inside the 1000 px target box, while the measured field is dim,
    highly non-uniform and concentrated near the box boundary.  Which axis of
    the forward chain is structurally responsible?

Method
------
1. Rebuild the measured anchor field from the v7 diagnostic captures (the six
   ``role == "A"`` captures at 1.0 ms): per-frame median subtracted, divided by
   the exposure, averaged over 30 frames.  Units are native ``counts/ms``.
2. Rebuild the predicted field by pushing the executed seed phase through the
   frozen checkpoint.  The prediction is decomposed into its additive
   background (``background + bias_counts/exposure``) and its signal, because
   94 % of the raw predicted box value is that constant.
3. Sweep one axis at a time and score every variant against the measured field
   with the same statistics: box mean/std/cv, the fraction of box pixels above
   10/25/50 % of the box maximum, the edge-frame energy fraction, and the
   Pearson correlation with the measurement -- both at zero registration and
   after a global-shift search, so a pure geometry defect and a structural
   defect are told apart.

Modes
-----
``variants``     one seed (the v7 anchor), one axis at a time, scored against
                 the v7 1.0 ms measurement.  Answers "which axis moves cv at
                 all" and "which axis moves the Pearson".
``seed-sweep``   all eight candidate seeds of the v2 seed run, each predicted
                 from its own phase and compared with its own measurement.
                 Answers "does the model reproduce the bench's seed-to-seed
                 spread, and is its response above the bench noise floor".
``common-mode``  all eight seeds under one frozen parameter at a time, scored on
                 the 24 phase-orthogonal seed pairs.  Answers "which axis owns
                 the seed-independent pattern the bench does not have", which is
                 the defect that makes the model gradient point the wrong way.

Collect-only.  No device access, no writes outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/multishape_nogs_20261030"
CHECKPOINT = DATA / "p20261030_s1000_square_plane_formation/model640/frozen_parameters.json"
SEED_RUN = DATA / "square1000_bandlimit_live_20261030_v2"
SEED_KEY = "F_blur4_full"
DIAG = DATA / "square1000_model_diagnostic_20261030_v7"

# The eight candidate seeds captured back to back in the v2 seed run.  Their
# phases are mutually near-orthogonal across groups (complex correlation
# |r| < 0.003), which is what makes the cross-seed common mode measurable.
SEED_LIST = ["A", "B_blur2", "C_blur4", "D_blur6", "E_blur4_half",
             "F_blur4_full", "G_blur4_alpha125", "H_blur4_alpha15"]

# Target-box definition, identical to square_1000_diff_abba_20261030.py.
BOX_CENTRE = (1917.0, 943.0)
ZERO_CENTRE = (1267.0, 943.0)
ZERO_RADIUS = 120.0


# --------------------------------------------------------------------------- masks
def build_masks(camera_xy: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(camera_xy[..., 0], dtype=np.float64)
    y = np.asarray(camera_xy[..., 1], dtype=np.float64)
    cx, cy = BOX_CENTRE
    box = (np.abs(x - cx) < 500.0) & (np.abs(y - cy) < 500.0)
    core = (np.abs(x - cx) < 480.0) & (np.abs(y - cy) < 480.0)
    inner = (np.abs(x - cx) < 430.0) & (np.abs(y - cy) < 430.0)
    zero = (x - ZERO_CENTRE[0]) ** 2 + (y - ZERO_CENTRE[1]) ** 2 < ZERO_RADIUS**2
    roi = (np.abs(x - cx) < 800.0) & (np.abs(y - cy) < 700.0)
    return {
        "box": box,
        "core": core,
        "inner": inner,
        "edge": core & ~inner,
        "roi": roi & ~zero,
        "zero": zero,
        "background": roi & ~box & ~zero,
    }


# ------------------------------------------------------------------------ measured
def sample_to_grid(frame: np.ndarray, camera_xy: np.ndarray) -> np.ndarray:
    """Nearest native pixel for every ``camera_xy`` sample.

    ``camera_xy`` holds **absolute** native sensor coordinates, so the samples
    are read straight out of the frame; they are not a binning lattice.
    """
    ix = np.clip(np.rint(camera_xy[..., 0]).astype(np.int64), 0, frame.shape[1] - 1)
    iy = np.clip(np.rint(camera_xy[..., 1]).astype(np.int64), 0, frame.shape[0] - 1)
    return frame[iy, ix]


def load_measured(camera_xy: np.ndarray, diag_dir: Path = DIAG) -> tuple[np.ndarray, dict]:
    manifest = json.loads((diag_dir / "manifest.json").read_text())
    acc, n = None, 0
    used = []
    for entry in manifest:
        if entry["role"] != "A" or abs(entry["exposure_ms"] - 1.0) > 1e-9:
            continue
        blob = np.load(diag_dir / entry["file"])
        raw = np.asarray(blob["raw_frames"], dtype=np.float64)
        exposure = float(entry["exposure_ms"])
        for frame in raw:
            signal = sample_to_grid(frame - np.median(frame), camera_xy) / exposure
            acc = signal if acc is None else acc + signal
            n += 1
        used.append(entry["file"])
    if acc is None or n == 0:
        raise SystemExit("no 1.0 ms anchor captures found")
    return acc / n, {"captures": used, "frames": n}


# ---------------------------------------------------------------------- prediction
def load_forward(checkpoint: Path = CHECKPOINT, seed_run: Path = SEED_RUN):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    plan = np.load(seed_run / "plan.npz")
    phase = np.asarray(plan[SEED_KEY], dtype=np.float64)
    camera_xy = np.asarray(plan["camera_xy"], dtype=np.float64)
    params = load_frozen_parameters(checkpoint).effective_parameters()
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    forward.exposure_ms = 1.0
    return forward, phase, camera_xy, plan


def predict(forward, phase: np.ndarray, *, exposure_ms: float = 1.0) -> np.ndarray:
    import torch

    with torch.no_grad():
        out = forward.predict(torch.as_tensor(phase, dtype=torch.float32), exposure_ms=exposure_ms)
    arr = np.asarray(out.detach().cpu().numpy(), dtype=np.float64)
    return arr[0] if arr.ndim == 3 else arr


def additive_floor(forward, exposure_ms: float = 1.0) -> float:
    p = forward.params
    return float(p.background) + float(p.bias_counts) / exposure_ms


# ------------------------------------------------------------------------- metrics
def box_stats(signal: np.ndarray, masks: dict) -> dict:
    box = signal[masks["box"]]
    edge, inner = signal[masks["edge"]], signal[masks["inner"]]
    peak = float(box.max())
    mean = float(box.mean())
    std = float(box.std())
    denom = float(np.abs(edge).sum() + np.abs(inner).sum())
    return {
        "box_min": float(box.min()),
        "box_mean": mean,
        "box_std": std,
        "box_cv": std / mean if mean > 0 else float("nan"),
        "box_max": peak,
        "frac_gt_10pct_max": float((box > 0.10 * peak).mean()),
        "frac_gt_25pct_max": float((box > 0.25 * peak).mean()),
        "frac_gt_50pct_max": float((box > 0.50 * peak).mean()),
        "edge_frame_fraction": float(edge.sum() / denom) if denom > 0 else float("nan"),
        "edge_over_inner": float(edge.mean() / inner.mean()) if inner.mean() != 0 else float("nan"),
    }


def pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    va, vb = a[mask], b[mask]
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = float(np.sqrt((va * va).sum() * (vb * vb).sum()))
    return float((va * vb).sum() / denom) if denom > 0 else float("nan")


def shift_search(pred: np.ndarray, meas: np.ndarray, mask: np.ndarray,
                 *, max_index: int = 60) -> dict:
    """Best Pearson over integer index shifts (the camera lattice is uniform 4 px)."""
    ny, nx = pred.shape
    best = {"pearson": -np.inf, "dy_px": 0, "dx_px": 0}
    for dy in range(-max_index, max_index + 1, 2):
        for dx in range(-max_index, max_index + 1, 2):
            ys = slice(max(dy, 0), ny + min(dy, 0))
            xs = slice(max(dx, 0), nx + min(dx, 0))
            yd = slice(max(-dy, 0), ny + min(-dy, 0))
            xd = slice(max(-dx, 0), nx + min(-dx, 0))
            m = mask[yd, xd]
            if m.sum() < 0.5 * mask.sum():
                continue
            r = pearson(pred[ys, xs], meas[yd, xd], m)
            if r > best["pearson"]:
                best = {"pearson": r, "dy_px": int(dy) * 4, "dx_px": int(dx) * 4}
    return best


# ---------------------------------------------------------------------- amplitudes
def amplitude_variants(forward) -> dict:
    """Candidate illumination amplitudes on the frozen panel grid."""
    import torch

    x, y = forward.panel_x, forward.panel_y
    mask = forward.panel_mask
    p = forward.params
    cx, cy = (float(v) for v in p.amplitude_center_xy)
    wx, wy = (float(v) for v in p.amplitude_width_xy)
    dx = (x - cx) / wx
    dy = (y - cy) / wy
    r2 = dx * dx + dy * dy
    out = {
        "gaussian_frozen": forward.amplitude.detach().clone(),
        "uniform_panel": mask.clone(),
    }
    for k in (0.75, 1.0, 1.25, 1.5):
        out[f"hard_ellipse_x{k:g}"] = (
            (dx * dx / (k * k) + dy * dy / (k * k) <= 1.0).float() * mask)
    for order in (4, 8):
        out[f"supergaussian_p{order}"] = torch.exp(-(r2 ** (order / 2.0))) * mask
    return out


def with_amplitude(forward, amplitude):
    saved = forward.amplitude
    forward.amplitude = amplitude
    return saved


def rebuild_forward(**overrides):
    """New forward with one frozen-parameter field replaced."""
    from dataclasses import replace

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    plan = np.load(SEED_RUN / "plan.npz")
    camera_xy = np.asarray(plan["camera_xy"], dtype=np.float64)
    params = load_frozen_parameters(CHECKPOINT).effective_parameters()
    params = replace(params, **overrides) if overrides else params
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    forward.exposure_ms = 1.0
    return forward, np.asarray(plan[SEED_KEY], dtype=np.float64)


def seed_sweep(forward, plan, masks: dict, out_dir: Path) -> dict:
    """Does the bench respond to the seed at all, and does the model agree?

    The v2 seed run holds eight distinct candidate seeds captured back to back at
    one exposure, plus three repeats of ``A``.  That gives two things the v7
    diagnostic cannot: the bench's *measured* seed-to-seed response, and the
    repeat floor that response must clear.
    """
    manifest = json.loads((SEED_RUN / "manifest.json").read_text())
    floor = additive_floor(forward, 1.0)
    seeds = SEED_LIST
    # capture index -> seed key, from the run's execution sequence
    index_of = {"A": [2, 3, 11], "B_blur2": [4], "C_blur4": [5], "D_blur6": [6],
                "E_blur4_half": [7], "F_blur4_full": [8], "G_blur4_alpha125": [9],
                "H_blur4_alpha15": [10]}

    camera_xy = np.asarray(plan["camera_xy"], dtype=np.float64)

    def measured_field(indices: list[int]) -> tuple[np.ndarray, dict]:
        acc, n, peak, exposure = None, 0, 0.0, None
        for idx in indices:
            entry = manifest[idx]
            raw = np.asarray(np.load(SEED_RUN / entry["file"])["raw_frames"], dtype=np.float64)
            peak = max(peak, float(raw.max()))
            exposure = float(entry["exposure_ms"])
            for frame in raw:
                signal = sample_to_grid(frame - np.median(frame), camera_xy) / exposure
                acc = signal if acc is None else acc + signal
                n += 1
        return acc / n, {"captures": indices, "frames": n, "raw_peak": peak,
                         "exposure_ms": exposure}

    meas: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for seed in seeds:
        meas[seed], meta[seed] = measured_field(index_of[seed])

    pred: dict[str, np.ndarray] = {}
    for seed in seeds:
        pred[seed] = predict(forward, np.asarray(plan[seed], dtype=np.float64)) - floor

    box = masks["box"]
    roi = masks["roi"]
    rows = []
    for seed in seeds:
        rows.append({
            "seed": seed,
            "meas_box_mean": float(meas[seed][box].mean()),
            "meas_box_cv": float(meas[seed][box].std() / meas[seed][box].mean()),
            "pred_box_mean": float(pred[seed][box].mean()),
            "pred_box_cv": float(pred[seed][box].std() / pred[seed][box].mean()),
            "r_meas_vs_pred_box": pearson(meas[seed], pred[seed], box),
            "r_meas_vs_pred_roi": pearson(meas[seed], pred[seed], roi),
            **meta[seed],
        })

    def pair_diff(fields, a, b, mask):
        d = fields[a] - fields[b]
        return float(np.sqrt((d[mask] ** 2).mean()))

    # A-repeat floor: the three A captures are separate exposures of one phase.
    reps = [measured_field([idx])[0] for idx in index_of["A"]]
    noise = float(np.mean([pair_diff({0: reps[0], 1: reps[1]}, 0, 1, box),
                           pair_diff({0: reps[1], 1: reps[2]}, 0, 1, box),
                           pair_diff({0: reps[0], 1: reps[2]}, 0, 1, box)]))

    resp = {b: pair_diff(meas, "A", b, box) for b in seeds if b != "A"}
    pred_resp = {b: pair_diff(pred, "A", b, box) for b in seeds if b != "A"}
    meas_pair = {a: {b: pearson(meas[a], meas[b], box) for b in seeds} for a in seeds}
    pred_pair = {a: {b: pearson(pred[a], pred[b], box) for b in seeds} for a in seeds}

    payload = {
        "exposure_ms": meta["A"]["exposure_ms"],
        "additive_floor": floor,
        "rows": rows,
        "meas_pairwise_box": meas_pair,
        "pred_pairwise_box": pred_pair,
        "repeat_floor_rms_box": noise,
        "meas_seed_response_rms_box": resp,
        "pred_seed_response_rms_box": pred_resp,
    }
    np.savez_compressed(out_dir / "seed_sweep_fields.npz",
                        **{f"meas_{k}": v for k, v in meas.items()},
                        **{f"pred_{k}": v for k, v in pred.items()})
    (out_dir / "seed_sweep.json").write_text(json.dumps(payload, indent=1))

    print("\n=== seed sweep (one exposure, bench vs model) ===")
    print(f"{'seed':<20}{'meas mean':>11}{'meas cv':>9}{'pred mean':>11}{'pred cv':>9}"
          f"{'r(box)':>9}{'r(roi)':>9}{'resp meas':>11}{'resp pred':>11}")
    for row in rows:
        r = resp.get(row["seed"], float("nan"))
        rp = pred_resp.get(row["seed"], float("nan"))
        print(f"{row['seed']:<20}{row['meas_box_mean']:11.4f}{row['meas_box_cv']:9.4f}"
              f"{row['pred_box_mean']:11.5f}{row['pred_box_cv']:9.4f}"
              f"{row['r_meas_vs_pred_box']:+9.4f}{row['r_meas_vs_pred_roi']:+9.4f}"
              f"{r:11.4f}{rp:11.5f}")
    print(f"A-repeat floor (rms, box) = {noise:.4f} counts/ms")
    print("measured seed-to-seed Pearson (box):")
    print("       " + " ".join(f"{s[:6]:>7}" for s in seeds))
    for a in seeds:
        print(f"{a[:6]:<7}" + " ".join(f"{meas_pair[a][b]:+7.3f}" for b in seeds))
    return payload


def common_mode_sweep(forward, plan, masks: dict, out_dir: Path) -> dict:
    """Which forward-chain axis owns the model's spurious cross-seed common mode?

    The eight v2 seeds are mutually near-orthogonal in phase (complex
    correlation |r| < 0.003 across groups), and the bench returns mutually
    near-orthogonal intensity fields for them (cross-seed Pearson |r| < 0.06).
    The frozen model instead predicts fields that agree with *each other* at
    r = 0.54-0.95 -- a seed-independent pattern the bench does not have.  A
    forward chain that only reproduces the bench's seed-independent structure
    cannot supply a usable control gradient, whatever its radiometry.

    This sweep rebuilds the prediction for all eight seeds under one frozen
    parameter at a time and reports, per variant:

    ``pred_common_mode_r``  mean off-diagonal Pearson among the eight predictions
                            (the spurious seed-independent pattern; bench ~0.03)
    ``r_diag_mean``         mean Pearson of pred(seed) vs meas(seed)
    ``r_cross_mean``        mean Pearson of pred(seed) vs meas(other seed)
    ``r_margin``            diagonal minus cross: does the model tell seeds apart?
    """
    fields = np.load(out_dir / "seed_sweep_fields.npz")
    meas = {s: np.asarray(fields["meas_" + s], dtype=np.float64) for s in SEED_LIST}
    floor = additive_floor(forward, 1.0)
    box = masks["box"]

    # Cross-group pairs only: {A,E} and {B,C,D} are near-duplicate phases, where
    # a high cross-seed correlation is *correct*.  The defect metric must be
    # taken over pairs whose phases are genuinely orthogonal.
    def phase_corr(a: str, b: str) -> float:
        za = np.exp(1j * np.asarray(plan[a], dtype=np.float64))
        zb = np.exp(1j * np.asarray(plan[b], dtype=np.float64))
        za, zb = za - za.mean(), zb - zb.mean()
        return float(np.real(np.vdot(za, zb))
                     / np.sqrt(float(np.vdot(za, za).real * np.vdot(zb, zb).real)))

    pairs = [(a, b) for i, a in enumerate(SEED_LIST) for b in SEED_LIST[i + 1:]]
    orth = [(a, b) for a, b in pairs if abs(phase_corr(a, b)) < 0.5]

    def predict_all(fwd) -> dict[str, np.ndarray]:
        return {s: predict(fwd, np.asarray(plan[s], dtype=np.float64)) - floor
                for s in SEED_LIST}

    def score(name: str, axis: str, pred: dict[str, np.ndarray]) -> dict:
        off_all = [pearson(pred[a], pred[b], box) for a, b in pairs]
        off_orth = [pearson(pred[a], pred[b], box) for a, b in orth]
        diag = {s: pearson(pred[s], meas[s], box) for s in SEED_LIST}
        cross = [pearson(pred[a], meas[b], box) for a in SEED_LIST
                 for b in SEED_LIST if a != b]
        cvs = {s: box_stats(pred[s], masks)["box_cv"] for s in SEED_LIST}
        return {
            "variant": name,
            "axis": axis,
            "pred_common_mode_r": float(np.mean(off_orth)),
            "pred_common_mode_r_all": float(np.mean(off_all)),
            "pred_common_mode_r_max": float(np.max(off_orth)),
            "r_diag_mean": float(np.mean(list(diag.values()))),
            "r_diag_by_seed": diag,
            "r_cross_mean": float(np.mean(cross)),
            "r_margin": float(np.mean(list(diag.values())) - np.mean(cross)),
            "cv_min": float(min(cvs.values())),
            "cv_max": float(max(cvs.values())),
            "cv_by_seed": cvs,
            "mean_by_seed": {s: float(pred[s][box].mean()) for s in SEED_LIST},
        }

    bench_orth = [pearson(meas[a], meas[b], box) for a, b in orth]
    bench_all = [pearson(meas[a], meas[b], box) for a, b in pairs]
    bench = {
        "orthogonal_pairs": len(orth),
        "pred_common_mode_r": float(np.mean(bench_orth)),
        "pred_common_mode_r_all": float(np.mean(bench_all)),
        "pred_common_mode_r_max": float(np.max(bench_orth)),
        "cv_min": float(min(box_stats(meas[s], masks)["box_cv"] for s in SEED_LIST)),
        "cv_max": float(max(box_stats(meas[s], masks)["box_cv"] for s in SEED_LIST)),
        "cv_by_seed": {s: box_stats(meas[s], masks)["box_cv"] for s in SEED_LIST},
        "mean_by_seed": {s: float(meas[s][box].mean()) for s in SEED_LIST},
    }

    rows = [score("frozen", "baseline", predict_all(forward))]

    saved = forward.amplitude
    for name, amp in amplitude_variants(forward).items():
        if name == "gaussian_frozen":
            continue
        forward.amplitude = amp
        rows.append(score(name, "amplitude", predict_all(forward)))
    forward.amplitude = saved

    for d in (-0.10, -0.05, +0.05, +0.10):
        f2, _ = rebuild_forward(defocus_diopter=float(forward.params.defocus_diopter) + d)
        rows.append(score(f"defocus_{d:+.2f}", "defocus", predict_all(f2)))

    for size in (1024, 4096):
        f3, _ = rebuild_forward(fft_size=size)
        rows.append(score(f"fft_size_{size}", "fft_size", predict_all(f3)))

    payload = {
        "seeds": SEED_LIST,
        "additive_floor": floor,
        "bench": bench,
        "rows": rows,
    }
    (out_dir / "common_mode.json").write_text(json.dumps(payload, indent=1))

    print("\n=== cross-seed common mode (bench reference) ===")
    print(f"bench: {len(orth)} orthogonal phase pairs, mean r = "
          f"{bench['pred_common_mode_r']:+.4f} (max {bench['pred_common_mode_r_max']:+.4f}); "
          f"all-pair mean {bench['pred_common_mode_r_all']:+.4f}; "
          f"cv range {bench['cv_min']:.4f}-{bench['cv_max']:.4f}")
    print(f"\n{'variant':<22}{'axis':>11}{'common r':>10}{'max':>8}"
          f"{'r diag':>9}{'r cross':>9}{'margin':>9}{'cv min':>9}{'cv max':>9}")
    for row in rows:
        print(f"{row['variant']:<22}{row['axis']:>11}{row['pred_common_mode_r']:+10.4f}"
              f"{row['pred_common_mode_r_max']:+8.4f}{row['r_diag_mean']:+9.4f}"
              f"{row['r_cross_mean']:+9.4f}{row['r_margin']:+9.4f}"
              f"{row['cv_min']:9.4f}{row['cv_max']:9.4f}")
    return payload


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DIAG / "defect_sim")
    parser.add_argument("--no-shift", action="store_true", help="skip the registration sweep")
    parser.add_argument("--mode", choices=("variants", "seed-sweep", "common-mode", "both"),
                        default="both")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    forward, phase, camera_xy, plan = load_forward()
    masks = build_masks(camera_xy)

    if args.mode in ("seed-sweep", "both"):
        seed_sweep(forward, plan, masks, args.out)
        if args.mode == "seed-sweep":
            return

    if args.mode == "common-mode":
        common_mode_sweep(forward, plan, masks, args.out)
        return

    meas, meas_meta = load_measured(camera_xy)
    floor = additive_floor(forward, 1.0)

    print(f"measured frames {meas_meta['frames']} from {len(meas_meta['captures'])} captures")
    print(f"model additive floor (background + bias/exposure) = {floor:.6f} counts/ms")

    rows: list[dict] = []

    def record(name: str, axis: str, pred_raw: np.ndarray) -> None:
        pred = pred_raw - floor
        row = {
            "variant": name,
            "axis": axis,
            "pred_signal_mean_box": float(pred[masks["box"]].mean()),
            "pred_signal_sum_roi": float(pred[masks["roi"]].sum()),
            **{f"pred_{k}": v for k, v in box_stats(pred, masks).items()},
            "pearson_box": pearson(pred, meas, masks["box"]),
            "pearson_roi": pearson(pred, meas, masks["roi"]),
        }
        if not args.no_shift:
            row["shift_box"] = shift_search(pred, meas, masks["box"])
            row["shift_roi"] = shift_search(pred, meas, masks["roi"])
        rows.append(row)
        print(f"{name:<28} box_mean={row['pred_signal_mean_box']:9.5f} "
              f"cv={row['pred_box_cv']:7.4f} r_box={row['pearson_box']:+.4f} "
              f"r_roi={row['pearson_roi']:+.4f}"
              + (f" shift_roi={row['shift_roi']['pearson']:+.4f}"
                 f"@({row['shift_roi']['dx_px']},{row['shift_roi']['dy_px']})"
                 if not args.no_shift else ""))

    meas_row = {"variant": "MEASURED", "axis": "measurement", **box_stats(meas, masks)}
    print(f"{'MEASURED':<28} box_mean={meas_row['box_mean']:9.5f} cv={meas_row['box_cv']:7.4f}")

    record("frozen_baseline", "none", predict(forward, phase))

    # --- axis 1: illumination amplitude -------------------------------------
    variants = amplitude_variants(forward)
    for name, amp in variants.items():
        if name == "gaussian_frozen":
            continue
        saved = with_amplitude(forward, amp)
        try:
            record(name, "amplitude", predict(forward, phase))
        finally:
            forward.amplitude = saved

    # --- axis 2: defocus / axial --------------------------------------------
    for label, kw in (("defocus_zero", {"defocus_diopter": 0.0}),
                      ("defocus_x0.5", {"defocus_diopter": 0.5 * float(forward.params.defocus_diopter)}),
                      ("defocus_x2", {"defocus_diopter": 2.0 * float(forward.params.defocus_diopter)})):
        f2, ph2 = rebuild_forward(**kw)
        record(label, "defocus", predict(f2, ph2))

    # --- axis 3: FFT size ---------------------------------------------------
    for size in (1024, 4096):
        f3, ph3 = rebuild_forward(fft_size=size)
        record(f"fft_size_{size}", "fft_size", predict(f3, ph3))

    # --- axis 4: registration only (shift the frozen prediction) ------------
    if not args.no_shift:
        base = predict(forward, phase) - floor
        best = shift_search(base, meas, masks["roi"])
        rows.append({"variant": "frozen_shift_only", "axis": "registration", **best})

    payload = {
        "checkpoint": str(CHECKPOINT),
        "seed_run": str(SEED_RUN),
        "seed_key": SEED_KEY,
        "additive_floor": floor,
        "measured": meas_row,
        "measured_meta": meas_meta,
        "rows": rows,
    }
    (args.out / "sim_variants.json").write_text(json.dumps(payload, indent=1))

    lines = ["| variant | axis | box mean | box cv | frac>10% | frac>25% | edge/inner | r(box) | r(roi) | shift r(roi) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        shift = row.get("shift_roi")
        lines.append(
            f"| {row['variant']} | {row['axis']} | "
            f"{row.get('pred_signal_mean_box', float('nan')):.5f} | "
            f"{row.get('pred_box_cv', float('nan')):.4f} | "
            f"{row.get('pred_frac_gt_10pct_max', float('nan')):.4f} | "
            f"{row.get('pred_frac_gt_25pct_max', float('nan')):.4f} | "
            f"{row.get('pred_edge_over_inner', float('nan')):.4f} | "
            f"{row.get('pearson_box', float('nan')):+.4f} | "
            f"{row.get('pearson_roi', float('nan')):+.4f} | "
            + (f"{shift['pearson']:+.4f} @({shift['dx_px']},{shift['dy_px']})" if shift else "-")
            + " |")
    lines.append(
        f"| MEASURED | measurement | {meas_row['box_mean']:.5f} | {meas_row['box_cv']:.4f} | "
        f"{meas_row['frac_gt_10pct_max']:.4f} | {meas_row['frac_gt_25pct_max']:.4f} | "
        f"{meas_row['edge_over_inner']:.4f} | - | - | - |")
    (args.out / "sim_variants.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out/'sim_variants.md'}")


if __name__ == "__main__":
    main()
