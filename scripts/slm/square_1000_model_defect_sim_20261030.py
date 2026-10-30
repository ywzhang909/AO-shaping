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
``jacobian``     the local Jacobian actually used for control: the v7 ``RX``/``RY``
                 +-probe pairs (drift-corrected rate maps) against the model's own
                 +-probe predictions, per amplitude variant, as a direction
                 cosine.  Stops if the frozen variant misses the reported values,
                 because every number below it would then be unanchored.
``aberration-fit``
                 the rescue attempt.  Refits the path aberration (Noll 4..15
                 radians) from zeros on the eight seeds -- two illuminations x two
                 held-out folds, near-duplicate seed phases never split -- and
                 then re-scores all three gates on the refit model.
``all``          seed-sweep + jacobian + aberration-fit + variants in one run.

Collect-only.  No device access, no writes outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    """New forward with one frozen-parameter field replaced.

    ``amplitude=`` is *not* a frozen-parameter field (the illumination map is a
    tensor living on the panel grid), so it is applied after construction and
    never reaches ``dataclasses.replace``; every other keyword is a parameter
    field, including ``residual_noll_radians``.
    """
    from dataclasses import replace

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    amplitude = overrides.pop("amplitude", None)
    plan = np.load(SEED_RUN / "plan.npz")
    camera_xy = np.asarray(plan["camera_xy"], dtype=np.float64)
    params = load_frozen_parameters(CHECKPOINT).effective_parameters()
    params = replace(params, **overrides) if overrides else params
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    forward.exposure_ms = 1.0
    if amplitude is not None:
        forward.amplitude = amplitude
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
# ------------------------------------------------------------ jacobian gate (b)
# The v7 session executed A, A, N4+, N4-, A, RX+, RX-, A, RY+, RY-, A, A(0.5x),
# A(1.5x), A.  The six A captures (0, 1, 4, 7, 10, 13) are the drift reference the
# report's analyzer interpolates between; the RX/RY probe pairs sit inside them.
DRIFT_ANCHORS = (0, 1, 4, 7, 10, 13)
#: Probe axis -> (plus capture, minus capture, following anchor capture).
DIAG_PROBE_GROUPS = {"N4": (2, 3, 4), "RX": (5, 6, 7), "RY": (8, 9, 10)}
#: Values reported for the frozen checkpoint, used as a reproduction anchor.
#: RX/RY anchor against report section 10; N4 anchors against section 13.7, which
#: reports the same code path run on the lowest-order probe.
FROZEN_JACOBIAN = {"N4": -0.0092, "RX": 0.270, "RY": 0.444}
FROZEN_JACOBIAN_TOLERANCE = 0.01

# ------------------------------------------------------------ aberration refit (c)
#: Phase-correlation groups of the eight v2 seeds: the phases inside one group are
#: near-duplicates, so a cross-validation split must never cut a group in half.
SEED_PHASE_GROUPS = {"AE": ("A", "E_blur4_half"),
                     "BCD": ("B_blur2", "C_blur4", "D_blur6")}
#: Noll 4..15.  Piston and tilt (Noll 1-3) only slide the far field and are not
#: fitted: they cannot fix its shape, and they would absorb the registration error.
FIT_NOLL_MODES = tuple(range(4, 16))
FIT_ITERATIONS = 300
FIT_LR = 0.05
FIT_TORCH_SEED = 0
#: The 2x2 design: frozen Gaussian illumination first, then hard_ellipse_x1.25.
FIT_AMPLITUDES = (("gaussian_frozen", None), ("hard_ellipse_x1.25", "hard_ellipse_x1.25"))
_GATE_COMMON_MODE_R = 0.10
_GATE_JACOBIAN_COSINE = 0.50
_GATE_R_DIAG = 0.30


def control_grid_size(forward) -> int:
    """Side length of the phase control grid the physical prediction model takes."""
    grid = getattr(forward, "control_grid", None)
    grid = forward.params.grid_size if grid is None else grid
    return int(grid[0]) if isinstance(grid, (tuple, list)) else int(grid)


def resize_panel_to_control(phase: np.ndarray, control_grid) -> np.ndarray:
    """Resample a panel-grid phase onto the model control grid (identity when equal).

    The archived probe phases and the checkpoint control grid agree today, so this
    is a no-op for every variant in this script; it exists so the two stays answerable
    if either side changes.
    """
    size = int(control_grid[0]) if isinstance(control_grid, (tuple, list)) else int(control_grid)
    arr = np.asarray(phase, dtype=np.float64)
    if arr.shape == (size, size):
        return arr
    height, width = arr.shape
    ay = (np.arange(size, dtype=np.float64) + 0.5) * height / size - 0.5
    ax = (np.arange(size, dtype=np.float64) + 0.5) * width / size - 0.5
    y0 = np.floor(ay).astype(np.int64)
    x0 = np.floor(ax).astype(np.int64)
    fy, fx = ay - y0, ax - x0
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y0 + 1, 0, height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x0 + 1, 0, width - 1)
    top = arr[np.ix_(y0c, x0c)] * (1.0 - fx)[None, :] + arr[np.ix_(y0c, x1c)] * fx[None, :]
    bottom = arr[np.ix_(y1c, x0c)] * (1.0 - fx)[None, :] + arr[np.ix_(y1c, x1c)] * fx[None, :]
    return top * (1.0 - fy)[:, None] + bottom * fy[:, None]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Direction cosine of two equal-shape fields (flattened before comparing)."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom > 0 else float("nan")


def diagnostic_v7_rates(diag_dir: Path = DIAG) -> dict[int, np.ndarray]:
    """Per-capture rate maps of the v7 session, read exactly as its analyzer reads them.

    One native pixel per lattice node (no binning, **no** median subtraction),
    divided by that capture's own exposure -> ``counts/ms``.  This is the
    calibration that produced the report's gate-(b) numbers, and it is kept
    deliberately apart from :func:`load_measured`, which median-subtracts every
    frame before averaging.
    """
    manifest = json.loads((diag_dir / "manifest.json").read_text())
    rates: dict[int, np.ndarray] = {}
    for position, row in enumerate(manifest):
        blob = np.load(diag_dir / row["file"])
        raw = np.asarray(blob["raw_frames"], dtype=np.float64)
        xy = np.asarray(blob["camera_xy"])
        frame = raw.mean(axis=0)
        key = int(row.get("index", position))
        rates[key] = (frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                      / float(row["exposure_ms"]))
    return rates


def drift_reference(rates: dict[int, np.ndarray],
                    anchors: tuple[int, ...] = DRIFT_ANCHORS) -> dict[int, np.ndarray]:
    """Linear interpolation of the anchor rate maps across every capture index."""
    out = {}
    for i in range(min(rates), max(rates) + 1):
        left = max(a for a in anchors if a <= i)
        right = min(a for a in anchors if a >= i)
        weight = 0.0 if left == right else (i - left) / (right - left)
        out[i] = (1.0 - weight) * rates[left] + weight * rates[right]
    return out


def jacobian_measured_deltas(rates: dict[int, np.ndarray],
                             drift: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Drift-corrected +-probe half-differences, one field per probe axis."""
    return {axis: 0.5 * ((rates[plus] - drift[plus]) - (rates[minus] - drift[minus]))
            for axis, (plus, minus, _anchor) in DIAG_PROBE_GROUPS.items()}


def load_probe_phases(forward, diag_dir: Path = DIAG) -> dict[str, np.ndarray]:
    """The executed v7 probe phases, resampled onto the control grid.

    The phases are read from the session's own archive rather than rebuilt from the
    probe recipe, so what is scored is what was actually displayed.
    """
    with np.load(diag_dir / "probe_phases.npz") as probes:
        keys = [f"{axis}{sign}" for axis in DIAG_PROBE_GROUPS for sign in "+-"]
        return {key: resize_panel_to_control(np.asarray(probes[key], dtype=np.float64),
                                             control_grid_size(forward))
                for key in keys}


def jacobian_predicted_deltas(forward, probe_phases: dict[str, np.ndarray],
                              *, exposure_ms: float = 1.0) -> dict[str, np.ndarray]:
    """Model counterpart of :func:`jacobian_measured_deltas` for one forward."""
    half = {key: predict(forward, value, exposure_ms=exposure_ms)
            for key, value in probe_phases.items()}
    return {axis: 0.5 * (half[axis + "+"] - half[axis + "-"]) for axis in DIAG_PROBE_GROUPS}


def jacobian_row(variant: str, predicted: dict[str, np.ndarray],
                 measured: dict[str, np.ndarray], masks: dict) -> dict:
    box = masks["box"]
    row: dict = {"variant": variant}
    for axis in DIAG_PROBE_GROUPS:
        pred, meas = predicted[axis], measured[axis]
        row[axis] = {
            "cosine_full": cosine(pred, meas),
            "cosine_box": cosine(pred[box], meas[box]),
            "pred_delta_norm": float(np.linalg.norm(pred)),
            "meas_delta_norm": float(np.linalg.norm(meas)),
        }
    return row


def jacobian_gate(forward, masks: dict, out_dir: Path, diag_dir: Path = DIAG) -> dict:
    """Gate (b): does the local Jacobian of the physical prediction model point the
    way the bench does?

    One measurement (the v7 RX/RY probe pairs, drift-corrected) is compared against
    every amplitude variant.  The frozen variant must reproduce the reported values;
    anything else would mean this offline copy of the model drifts from the one the
    report gates on, and every gate number below it would be unanchored.
    """
    rates = diagnostic_v7_rates(diag_dir)
    drift = drift_reference(rates)
    measured = jacobian_measured_deltas(rates, drift)
    probe_phases = load_probe_phases(forward, diag_dir)
    print(f"\n=== jacobian gate (b): {len(DIAG_PROBE_GROUPS)} probe axes, "
          f"grid {measured['RX'].shape} = {measured['RX'].size} points ===")

    rows = [jacobian_row("frozen_baseline",
                         jacobian_predicted_deltas(forward, probe_phases), measured, masks)]
    for name, amplitude in amplitude_variants(forward).items():
        saved = with_amplitude(forward, amplitude)
        rows.append(jacobian_row(name, jacobian_predicted_deltas(forward, probe_phases),
                                 measured, masks))
        forward.amplitude = saved

    sanity = {}
    for axis in DIAG_PROBE_GROUPS:
        value = rows[0][axis]["cosine_full"]
        expected = FROZEN_JACOBIAN[axis]
        sanity[axis] = {"model_cosine": value, "reported_cosine": expected,
                        "tolerance": FROZEN_JACOBIAN_TOLERANCE,
                        "ok": bool(abs(value - expected) <= FROZEN_JACOBIAN_TOLERANCE)}
    failed = [axis for axis, block in sanity.items() if not block["ok"]]
    if failed:
        raise SystemExit(
            "frozen variant does not reproduce the reported Jacobian cosines for "
            + ", ".join(failed) + ": "
            + ", ".join(f"{axis} model {sanity[axis]['model_cosine']:+.4f} vs reported "
                        f"{sanity[axis]['reported_cosine']:+.4f}" for axis in failed)
            + " -- the offline model and the report's model disagree, stopping here.")

    payload = {
        "checkpoint": str(CHECKPOINT),
        "diagnostic_session": str(diag_dir),
        "probe_axes": {axis: {"plus_index": pair[0], "minus_index": pair[1],
                              "probe_amplitude_rad": None}
                       for axis, pair in DIAG_PROBE_GROUPS.items()},
        "calibration": {
            "name": "diagnostic_v7_rate_map",
            "bin_size": 1,
            "median_subtraction": False,
            "drift_anchors": list(DRIFT_ANCHORS),
            "drift_model": "linear interpolation between the bracketing anchor rate maps",
            "prediction_exposure_ms": 1.0,
            "grid_shape": list(measured["RX"].shape),
            "grid_points": int(measured["RX"].size),
            "note": "separate from load_measured(), which median-subtracts every frame",
        },
        "measured": {axis: {"delta_norm": float(np.linalg.norm(measured[axis])),
                            "delta_rms": float(np.sqrt(np.mean(measured[axis] ** 2)))}
                     for axis in DIAG_PROBE_GROUPS},
        "frozen_sanity": sanity,
        "rows": rows,
    }
    (out_dir / "jacobian.json").write_text(json.dumps(payload, indent=1))

    axes = list(DIAG_PROBE_GROUPS)
    control = [a for a in ("RX", "RY") if a in DIAG_PROBE_GROUPS]
    header = f"{'variant':<22}"
    for axis in axes:
        header += f"{(axis + ' full'):>9}{(axis + ' box'):>9}"
    print(header + "".join(f"|d{a}|".rjust(10) for a in axes))
    for row in rows:
        line = f"{row['variant']:<22}"
        for axis in axes:
            line += f"{row[axis]['cosine_full']:+9.4f}{row[axis]['cosine_box']:+9.4f}"
        print(line + "".join(f"{row[a]['pred_delta_norm']:10.4f}" for a in axes))
    for axis, block in sanity.items():
        print(f"frozen sanity {axis}: {block['model_cosine']:+.5f} vs reported "
              f"{block['reported_cosine']:+.3f} (tol {block['tolerance']}) -> OK")
    print(f"gate (b) needs both control cosines >= {_GATE_JACOBIAN_COSINE}; best now "
          f"{max(min(row[a]['cosine_full'] for a in control) for row in rows):+.4f}"
          f" -> defect_sim/jacobian.json")
    return payload


# ------------------------------------------------------- aberration refit (gate c)
def seed_exposure_ms(seed_run: Path = SEED_RUN) -> float:
    """Exposure shared by the eight seed captures (the run opens with one preview)."""
    manifest = json.loads((seed_run / "manifest.json").read_text())
    counts: dict[float, int] = {}
    for entry in manifest:
        key = float(entry["exposure_ms"])
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


def load_seed_fields(out_dir: Path) -> dict[str, np.ndarray]:
    """Measured native fields of the eight v2 seeds (median-subtracted counts/ms).

    Same source ``common_mode_sweep`` reads, so the two modes score the same numbers.
    """
    fields = np.load(out_dir / "seed_sweep_fields.npz")
    return {seed: np.asarray(fields["meas_" + seed], dtype=np.float64) for seed in SEED_LIST}


def _pearson_torch(a, b):
    """Pearson correlation in float64, differentiable through ``a``."""
    import torch

    va = torch.as_tensor(a, dtype=torch.float64).ravel()
    vb = torch.as_tensor(b, dtype=torch.float64).ravel()
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = torch.sqrt((va * va).sum() * (vb * vb).sum())
    return (va * vb).sum() / denom.clamp_min(1e-300)


def fit_residual_aberration(forward, phases: dict[str, np.ndarray],
                            meas: dict[str, np.ndarray], train_seeds: list[str],
                            exposure_ms: float, box: np.ndarray, *,
                            iterations: int = FIT_ITERATIONS, lr: float = FIT_LR,
                            seed: int = FIT_TORCH_SEED, verbose: bool = True) -> dict:
    """Fit the path aberration (Noll 4..15 radians) on ``-Pearson r_diag``.

    ``residual_values`` is swapped for a leaf tensor so ``predict`` stays
    differentiable; ``FrozenDefocusForward`` reads that attribute directly, so no
    other part of the frozen physical prediction model has to be touched.  Both
    classes' float32 working dtype is kept: torch rejects float64 values against
    the float32 mode basis (``einsum``) and the float32 sample grids
    (``grid_sample``).
    """
    import torch

    torch.manual_seed(seed)
    values = torch.zeros(len(forward.residual_modes),
                         dtype=forward.residual_basis.dtype, requires_grad=True)
    forward.residual_values = values
    optimizer = torch.optim.Adam([values], lr=lr)
    floor = additive_floor(forward, exposure_ms)
    box_mask = torch.as_tensor(box, dtype=torch.bool)
    targets = {seed_key: torch.as_tensor(meas[seed_key][box], dtype=torch.float64)
               for seed_key in train_seeds}
    trace: list[float] = []
    for step in range(iterations):
        optimizer.zero_grad()
        total = 0.0
        for seed_key in train_seeds:
            pred = forward.predict(torch.as_tensor(phases[seed_key], dtype=torch.float32),
                                   exposure_ms=exposure_ms)[0]
            loss = -_pearson_torch((pred - floor)[box_mask], targets[seed_key]) / len(train_seeds)
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        trace.append(total)
        if verbose and (step % 25 == 0 or step == iterations - 1):
            print(f"  iteration {step:3d}  loss {total:+.5f}")
    forward.residual_values = values.detach().clone()
    return {"modes": list(forward.residual_modes),
            "values": np.asarray(values.detach(), dtype=np.float64),
            "parameter_dtype": str(values.dtype),
            "trace": trace, "final_loss": float(trace[-1])}


def r_diag_by_seed(forward, phases: dict[str, np.ndarray], meas: dict[str, np.ndarray],
                   exposure_ms: float, box: np.ndarray) -> dict[str, float]:
    """``r_diag`` per seed with the same calibration ``common_mode_sweep`` uses."""
    floor = additive_floor(forward, exposure_ms)
    out = {}
    for seed_key in SEED_LIST:
        signal = predict(forward, np.asarray(phases[seed_key], dtype=np.float64),
                         exposure_ms=exposure_ms) - floor
        out[seed_key] = pearson(signal, meas[seed_key], box)
    return out


def radiometry_gains(forward, phases: dict[str, np.ndarray], meas: dict[str, np.ndarray],
                     exposure_ms: float, box: np.ndarray) -> dict:
    """Least-squares gain ``<meas, sig>/<sig, sig>`` per seed on the target box."""
    floor = additive_floor(forward, exposure_ms)
    gains = {}
    for seed_key in SEED_LIST:
        signal = (predict(forward, np.asarray(phases[seed_key], dtype=np.float64),
                          exposure_ms=exposure_ms) - floor)[box]
        reference = meas[seed_key][box]
        denom = float((signal * signal).sum())
        gains[seed_key] = float((reference * signal).sum() / denom) if denom > 0 else float("nan")
    values = list(gains.values())
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "max": float(np.max(values)), "by_seed": gains}


def aberration_fit(plan, masks: dict, out_dir: Path) -> dict:
    """Gate (c) rescue attempt: refit the path aberration, then re-evaluate all gates.

    The hypothesis is that the ``r_diag`` ceiling is a *missing* pupil aberration,
    so Noll 4..15 radians are fitted from scratch on the v2 seeds (piston/tilt stay
    out) under two illuminations x two held-out folds.  Near-duplicate seed phases
    are grouped, so no fold ever trains on half a group and validates on the other
    half.  ``common_mode_sweep`` is reused for gate (a) through a scratch output
    directory, which leaves the existing artifacts untouched.
    """
    import torch

    meas = load_seed_fields(out_dir)
    exposure_ms = seed_exposure_ms()
    box = masks["box"]
    base_forward, _, _, _ = load_forward()
    phases = {seed_key: np.asarray(plan[seed_key], dtype=np.float64) for seed_key in SEED_LIST}
    amplitudes = amplitude_variants(base_forward)
    # One measurement for every combo: v7 rate maps and the v7 probe archive.
    rates = diagnostic_v7_rates()
    drift = drift_reference(rates)
    probe_measured = jacobian_measured_deltas(rates, drift)
    probe_phases = load_probe_phases(base_forward)

    fits: list[dict] = []
    for amplitude_name, variant_key in FIT_AMPLITUDES:
        amplitude = None if variant_key is None else amplitudes[variant_key]
        for fold_name, held_out in SEED_PHASE_GROUPS.items():
            train = [seed_key for seed_key in SEED_LIST if seed_key not in held_out]
            forward, _ = rebuild_forward(
                residual_noll_radians={int(j): 0.0 for j in FIT_NOLL_MODES},
                amplitude=amplitude)
            print(f"\n=== aberration fit: amplitude {amplitude_name}, fold {fold_name} "
                  f"({FIT_ITERATIONS} iterations, Adam lr {FIT_LR}) ===")
            print(f"  train {train}\n  held-out {list(held_out)}")
            result = fit_residual_aberration(forward, phases, meas, train, exposure_ms, box)
            diagonal = r_diag_by_seed(forward, phases, meas, exposure_ms, box)
            row = {
                "amplitude": amplitude_name,
                "fold": fold_name,
                "train_seeds": train,
                "held_out_seeds": list(held_out),
                "noll_modes": result["modes"],
                "coefficients": {int(mode): float(value)
                                 for mode, value in zip(result["modes"], result["values"])},
                "coefficients_rad_rms": float(np.sqrt(np.mean(result["values"] ** 2))),
                "fitted_tensor_dtype": result["parameter_dtype"],
                 "exported_coefficient_dtype": str(result["values"].dtype),
                "final_train_loss": result["final_loss"],
                "train_loss_trace": result["trace"],
                "r_diag_by_seed": diagonal,
                "r_diag_in_sample_mean": float(np.mean([diagonal[s] for s in train])),
                "r_diag_held_out_mean": float(np.mean([diagonal[s] for s in held_out])),
            }
            fits.append(row)
            print(f"  coefficients rms {row['coefficients_rad_rms']:.4f} rad; "
                  f"r_diag in-sample {row['r_diag_in_sample_mean']:+.4f}, "
                  f"held-out {row['r_diag_held_out_mean']:+.4f}")

    # Gate re-evaluation per combo.  A single refit model is needed for gates (a)
    # and (b) and for the radiometry gain; the fold-AE coefficients are used there.
    scratch = out_dir / "aberration_fit_common_mode"
    scratch.mkdir(parents=True, exist_ok=True)
    if not (scratch / "seed_sweep_fields.npz").exists():
        shutil.copyfile(out_dir / "seed_sweep_fields.npz", scratch / "seed_sweep_fields.npz")

    gates: list[dict] = []
    for amplitude_name, variant_key in FIT_AMPLITUDES:
        amplitude = None if variant_key is None else amplitudes[variant_key]
        for state in ("frozen_residual", "aberration_fit_fold_AE"):
            if state == "frozen_residual":
                forward, _ = rebuild_forward(amplitude=amplitude)
            else:
                forward, _ = rebuild_forward(
                    residual_noll_radians={int(j): 0.0 for j in FIT_NOLL_MODES},
                    amplitude=amplitude)
                fit_row = next(row for row in fits
                               if row["amplitude"] == amplitude_name and row["fold"] == "AE")
                forward.residual_values = torch.tensor(
                    [float(fit_row["coefficients"][int(mode)]) for mode in forward.residual_modes],
                    dtype=forward.residual_basis.dtype)
            print(f"\n--- combo {amplitude_name} / {state} ---")
            common = common_mode_sweep(forward, plan, masks, scratch)
            baseline = next(row for row in common["rows"] if row["variant"] == "frozen")
            predicted = jacobian_predicted_deltas(forward, probe_phases)
            jacobian = {axis: {"cosine_full": cosine(predicted[axis], probe_measured[axis]),
                               "cosine_box": cosine(predicted[axis][box],
                                                    probe_measured[axis][box]),
                               "pred_delta_norm": float(np.linalg.norm(predicted[axis]))}
                        for axis in DIAG_PROBE_GROUPS}
            diagonal = r_diag_by_seed(forward, phases, meas, exposure_ms, box)
            folds = {}
            for fold_name, held_out in SEED_PHASE_GROUPS.items():
                train = [seed_key for seed_key in SEED_LIST if seed_key not in held_out]
                # Refit combos are scored with the model fitted on *that* fold's train
                # seeds, so its held-out seeds really were excluded from the fit.  The
                # frozen-residual combos fitted nothing, so one model serves all folds.
                source = diagonal if state == "frozen_residual" else next(
                    row["r_diag_by_seed"] for row in fits
                    if row["amplitude"] == amplitude_name and row["fold"] == fold_name)
                folds[fold_name] = {
                    "train_seeds": train,
                    "held_out_seeds": list(held_out),
                    "in_sample_mean": float(np.mean([source[s] for s in train])),
                    "held_out_mean": float(np.mean([source[s] for s in held_out])),
                }
            row = {
                "combo": f"{amplitude_name}__{state}",
                "amplitude": amplitude_name,
                "aberration_state": state,
                "common_mode_r": baseline["pred_common_mode_r"],
                "common_mode_r_max": baseline["pred_common_mode_r_max"],
                "jacobian": jacobian,
                "r_diag_all_seeds_mean": float(np.mean(list(diagonal.values()))),
                "r_diag_by_seed": diagonal,
                "r_diag_folds": folds,
                "radiometry_gain": radiometry_gains(forward, phases, meas, exposure_ms, box),
            }
            row["pass_common_mode"] = bool(row["common_mode_r"] < _GATE_COMMON_MODE_R)
            row["pass_jacobian"] = bool(min(block["cosine_full"] for block in jacobian.values())
                                        >= _GATE_JACOBIAN_COSINE)
            row["pass_r_diag_in_sample"] = bool(
                all(block["in_sample_mean"] >= _GATE_R_DIAG for block in folds.values()))
            row["pass_r_diag_held_out"] = bool(
                all(block["held_out_mean"] >= _GATE_R_DIAG for block in folds.values()))
            row["pass_all_three_gates"] = bool(
                row["pass_common_mode"] and row["pass_jacobian"]
                and row["pass_r_diag_in_sample"] and row["pass_r_diag_held_out"])
            gates.append(row)

    best = next(row for row in fits if row["amplitude"] == FIT_AMPLITUDES[1][0]
                and row["fold"] == "AE")
    coefficients_path = out_dir / "aberration_fit_coeffs.json"
    coefficients_path.write_text(json.dumps(
        {"source": "fold AE, hard_ellipse_x1.25",
         "noll_radians": {str(mode): value for mode, value in best["coefficients"].items()}},
        indent=1))

    payload = {
        "checkpoint": str(CHECKPOINT),
        "seed_run": str(SEED_RUN),
        "exposure_ms": exposure_ms,
        "fit": {
            "noll_modes": list(FIT_NOLL_MODES),
            "iterations": FIT_ITERATIONS,
            "lr": FIT_LR,
            "torch_seed": FIT_TORCH_SEED,
            "fitted_tensor_dtype": best["fitted_tensor_dtype"],
            "exported_coefficient_dtype": best["exported_coefficient_dtype"],
            "dtype_note": ("the fitted leaf tensor carries the forward's own working dtype "
                           "(float32): torch rejects float64 values against the float32 mode "
                           "basis (defocus_phase einsum) and float32 sample grids (grid_sample)"),
            "loss": "mean(-Pearson(pred - floor, measured)) over train seeds, target-box mask",
            "signal_definition": "predict(exposure_ms) - (background + bias/exposure_ms)",
            "held_out_folds": {name: list(group) for name, group in SEED_PHASE_GROUPS.items()},
            "single_model_choice": ("gates (a), (b) and the radiometry gain use the "
                                    "fold-AE fitted coefficients"),
            "note": ("checkpoint residual_noll_radians is replaced by Noll 4..15 zeros and "
                     "refitted from scratch, so the fit is not seeded by the frozen values"),
        },
        "gates_needed": {"common_mode_r_below": _GATE_COMMON_MODE_R,
                         "jacobian_cosine_at_least": _GATE_JACOBIAN_COSINE,
                         "r_diag_at_least": _GATE_R_DIAG},
        "fits": fits,
        "gates": gates,
        "verdict": {
            "any_combo_passes_all_three": bool(any(row["pass_all_three_gates"] for row in gates)),
            "best_combo": min(gates, key=lambda row: -int(row["pass_all_three_gates"]))["combo"],
        },
    }
    (out_dir / "aberration_fit.json").write_text(json.dumps(payload, indent=1))

    print("\n=== aberration refit: gate table (2x2) ===")
    print(f"{'combo':<44}{'cmode r':>9}{'RX':>8}{'RY':>8}{'r_ins':>8}{'r_held':>8}"
          f"{'gain':>9}{'pass':>6}")
    for row in gates:
        ins = min(block["in_sample_mean"] for block in row["r_diag_folds"].values())
        held = min(block["held_out_mean"] for block in row["r_diag_folds"].values())
        print(f"{row['combo']:<44}{row['common_mode_r']:+9.4f}"
              f"{row['jacobian']['RX']['cosine_full']:+8.4f}"
              f"{row['jacobian']['RY']['cosine_full']:+8.4f}"
              f"{ins:+8.4f}{held:+8.4f}{row['radiometry_gain']['mean']:9.3f}"
              f"{'YES' if row['pass_all_three_gates'] else 'no':>6}")
    print(f"artifacts -> {out_dir / 'aberration_fit.json'}, {coefficients_path}")
    return payload


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DIAG / "defect_sim")
    parser.add_argument("--no-shift", action="store_true", help="skip the registration sweep")
    parser.add_argument("--mode",
                        choices=("variants", "seed-sweep", "common-mode", "both",
                                 "jacobian", "aberration-fit", "all"),
                        default="both")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    forward, phase, camera_xy, plan = load_forward()
    masks = build_masks(camera_xy)

    if args.mode in ("seed-sweep", "both", "all"):
        seed_sweep(forward, plan, masks, args.out)
        if args.mode == "seed-sweep":
            return

    if args.mode == "common-mode":
        common_mode_sweep(forward, plan, masks, args.out)
        return

    if args.mode in ("jacobian", "aberration-fit", "all"):
        # Gate (b): the local Jacobian of the physical prediction model against
        # the v7 +-probe pairs.  Runs first because it is also the anchor every
        # later number is read against.
        jacobian_gate(forward, masks, args.out)
        if args.mode == "jacobian":
            return
        # Gate (c) refits the pupil aberration and re-evaluates all three gates.
        # Gate (a) is re-run into a scratch directory, never over ``args.out``,
        # so the artifacts the earlier modes wrote stay byte-identical.
        aberration_fit(plan, masks, args.out)
        if args.mode == "aberration-fit":
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
