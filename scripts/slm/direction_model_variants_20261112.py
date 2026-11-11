"""Consolidated per-mode Jacobian-direction diagnosis across forward-model variants.

Offline only (issue #234 / #235 / #237): this script never opens the SLM, the
camera or the laser.  It replays the archived issue234 single-mode screens
(±0.10 / ±0.05 / ±0.02 rad, Z5–Z11, two passes) and compares, per mode and per
model variant, the signed direction cosine between the model C−U differential
and the measured C−U differential.

Variants (all from ``defocus_forward_20261107_focus_r2_fit/forward_fit.json``):

* ``M0_fit``   -- M0: defocus_diopter = 0 (the "analytic" no-defocus baseline).
* ``M1_fit``   -- M1_defocus: the standard slm_phase fitted checkpoint model.
* ``M3_fit``   -- M3_illumination: defocus + fitted illumination envelope.
* ``M1_bg0``   -- M1_defocus with background forced to 0.
* ``M1_bg10``  -- M1_defocus with background forced to +10 counts/ms.
* ``M3_bg0``   -- M3_illumination with background forced to 0.

Because the direction metric is an affine-invariant Pearson on the C−U
differential, background and gain are *expected* to cancel exactly; the bg
variants exist to demonstrate that numerically rather than assume it.

Guard: the ``M1_fit`` amp 0.05 arm must reproduce the archived
``direction_screen_amp005_M1_defocus_signm1_model.json`` per-mode correlation
exactly (bit-for-bit), proving the measurement口径 is unchanged.

Outputs ``results.json`` under ``data/debug/slm_transfer/direction_variants_20261112/``
and prints a per-variant summary table.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.jacobian_guard import direction_agreement

FIT = Path("data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json")
OUT_DIR = Path("data/debug/slm_transfer/direction_variants_20261112/")
CROP = 128
MODES = list(range(5, 12))
AMPS = [0.02, 0.05, 0.10]
# run dir per amp (the amp 0.10 ladder lives in the "single_mode" directory)
RUN_DIRS = {
    0.02: "data/debug/slm_transfer/issue234_amp02_screen_20261107",
    0.05: "data/debug/slm_transfer/issue234_amp05_screen_20261107",
    0.10: "data/debug/slm_transfer/issue234_single_mode_screen_20261107",
}
# archive guard: amp 0.05 M1_defocus sign -1, origin=model
GUARD = Path(
    "data/debug/slm_transfer/issue234_amp05_screen_20261107/"
    "direction_screen_amp005_M1_defocus_signm1_model.json")

VARIANTS = [
    ("M0_fit", "M0", None),
    ("M1_fit", "M1_defocus", None),
    ("M3_fit", "M3_illumination", None),
    ("M1_bg0", "M1_defocus", 0.0),
    ("M1_bg10", "M1_defocus", 10.0),
    ("M3_bg0", "M3_illumination", 0.0),
]


def load_frame(run, run_pass, row):
    d = np.load(run / run_pass / row["file"])
    raw = d["raw_frames"].astype(np.float64).mean(axis=0)
    rate = raw / float(row.get("exposure_ms") or 1.0)
    return d["model_phase"].astype(np.float32), rate


def crop(img, ctr):
    x0 = int(round(ctr[0] - CROP / 2))
    y0 = int(round(ctr[1] - CROP / 2))
    return img[y0:y0 + CROP, x0:x0 + CROP]


def evaluate(run_dir, amp, model_key, bg_override):
    run = Path(run_dir)
    fit = json.loads(FIT.read_text())
    model = fit["models"][model_key]
    ramp = fit["ramp_matrix"]
    origin = np.asarray(model["origin"], dtype=np.float64)
    bg = model["background"] if bg_override is None else bg_override
    params = FrozenDefocusParameters(
        defocus_diopter=model["defocus_diopter"],
        origin_xy=(float(origin[0]), float(origin[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={},
        gain=model["gain"],
        background=bg,
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=-1.0,
        phase_sign_scope="command",
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack([xx + origin[0] - CROP / 2, yy + origin[1] - CROP / 2], axis=-1)
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    amp_tag = f"{amp:.2f}"
    results = []
    for run_pass in ["pass1", "pass2"]:
        manifest = json.loads((run / run_pass / "manifest.json").read_text())
        for mode in MODES:
            lp = [r for r in manifest if f"_Z{mode}_+{amp_tag}" in r["label"]]
            lm = [r for r in manifest if f"_Z{mode}_-{amp_tag}" in r["label"]]
            if not lp or not lm:
                print(f"{run_pass} Z{mode:2d}  MISSING labels")
                continue
            pm, rm = load_frame(run, run_pass, lm[0])
            pp, rp = load_frame(run, run_pass, lp[0])
            with torch.no_grad():
                pred_m = forward.predict(torch.as_tensor(pm)).cpu().numpy()[0]
                pred_p = forward.predict(torch.as_tensor(pp)).cpu().numpy()[0]
            agr = direction_agreement(pred_m, pred_p, crop(rm, origin), crop(rp, origin))
            results.append({
                "pass": run_pass, "mode": mode, "corr": agr.correlation,
                "verdict": agr.verdict, "rel_l2": agr.relative_l2,
                "predicted_rms": agr.predicted_rms, "measured_rms": agr.measured_rms,
            })
    return results, params


def stats(results):
    corrs = np.asarray([r["corr"] for r in results], dtype=np.float64)
    finite = corrs[np.isfinite(corrs)]
    return {
        "n": int(len(results)),
        "n_finite": int(finite.size),
        "median": float(np.median(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std_ddof1": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "n_ge_0p5": int(np.sum(finite >= 0.5)),
        "n_ge_0p5_over": f"{int(np.sum(finite >= 0.5))}/{int(finite.size)}",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fit = json.loads(FIT.read_text())

    # ---- guard: reproduce the archived amp 0.05 M1_fit per-mode corr ---------
    guard_results, _ = evaluate(RUN_DIRS[0.05], 0.05, "M1_defocus", None)
    archived = json.loads(GUARD.read_text())["results"]
    by_key = {(r["pass"], r["mode"]): r["corr"] for r in archived}
    guard = {"n": len(guard_results), "exact_match": True}
    max_delta = 0.0
    for r in guard_results:
        a = by_key.get((r["pass"], r["mode"]))
        if a is None:
            guard["exact_match"] = False
            guard.setdefault("missing", []).append((r["pass"], r["mode"]))
            continue
        max_delta = max(max_delta, abs(a - r["corr"]))
    guard["max_abs_delta"] = max_delta
    guard["median"] = stats(guard_results)["median"]
    print(f"GUARD: n={guard['n']} median={guard['median']:.6f} "
          f"exact={guard['exact_match']} max_delta={max_delta:.3e}")

    # ---- variant matrix -----------------------------------------------------
    all_out = {"schema": "direction_model_variants/1", "amp": AMPS,
               "variants": [], "guard": guard}
    for vname, model_key, bg_override in VARIANTS:
        for amp in AMPS:
            results, params = evaluate(RUN_DIRS[amp], amp, model_key, bg_override)
            st = stats(results)
            per_mode = {}
            for r in results:
                per_mode.setdefault(r["mode"], []).append(r["corr"])
            per_mode_median = {str(m): float(np.median(v)) for m, v in per_mode.items()}
            entry = {
                "variant": vname, "model": model_key, "amp": amp,
                "background_override": bg_override, "stats": st,
                "per_mode_median": per_mode_median, "results": results,
            }
            all_out["variants"].append(entry)
            print(f"{vname:8s} amp={amp:.2f}  n={st['n']}  median={st['median']:+.4f}  "
                  f"mean={st['mean']:+.4f}  >=0.5={st['n_ge_0p5_over']}")

    (OUT_DIR / "results.json").write_text(json.dumps(all_out, indent=1))
    print("wrote", OUT_DIR / "results.json")


if __name__ == "__main__":
    main()
