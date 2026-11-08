"""Per-mode signed direction check for the issue234 single-mode screens, by scope.

Usage:
  python phase_scope_direction_20261107.py --run <dir> --amp 0.05 \
      [--phase-sign-scope aberration] [--model M1_defocus]

Offline only: reads saved .npz / JSON, no device access, no ``ao_shaping.drivers``
import. Reuses the issue234 measurement口径: the saved model is the stored fit,
the origin comes from the fit (``--origin model``), no registration is applied,
and only the command polarity scope varies between arms.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.defocus_forward_core import PHASE_SIGN_SCOPES
from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.jacobian_guard import direction_agreement

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--amp", type=float, required=True)
p.add_argument("--fit", type=Path, default=Path(
    "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json"))
p.add_argument("--model", default="M1_defocus")
p.add_argument("--modes", type=int, nargs="+", default=list(range(5, 12)))
p.add_argument("--out", type=Path, default=None)
# Issue #235: the panel phase enters the propagation with NEGATIVE sign on this
# bench (measured: 42/42 single-mode screens flip from -0.47 to +0.47 median).
p.add_argument("--phase-sign", type=float, default=-1.0, choices=(-1.0, 1.0))
p.add_argument("--phase-sign-scope", default="command", choices=PHASE_SIGN_SCOPES)
# The 0-order is located by box centroid instead of single-pixel argmax.
p.add_argument("--origin", default="model", choices=("model", "centroid"),
               help="centroid is a diagnostic only: the fitted origin is jointly calibrated with ramp_matrix, and a 13 px override collapses the signal")
args = p.parse_args()

RUN = args.run
FIT = json.loads(args.fit.read_text())
MODEL = FIT["models"][args.model]
RAMP = FIT["ramp_matrix"]
CROP = 128


def _frame_rate(run_pass, file, exposure_ms):
    d = np.load(RUN / run_pass / file)
    raw = d["raw_frames"].astype(np.float64).mean(axis=0)
    return raw / float(exposure_ms or 1.0)


def _measured_zero_order() -> np.ndarray:
    """Median box-centroid 0-order over the pass1 frames (issue #235)."""
    from ml.sim2f.full_frame import box_centroid

    manifest = json.loads((RUN / "pass1/manifest.json").read_text())
    spots = []
    for row in manifest[:8]:
        frame = _frame_rate("pass1", row["file"], row.get("exposure_ms"))
        r, c = np.unravel_index(int(np.argmax(frame)), frame.shape)
        spot = box_centroid(frame, row=int(r), col=int(c))
        spots.append((spot[1], spot[0]))  # manifest origin is (x, y)
    return np.median(np.asarray(spots, dtype=np.float64), axis=0)


ORIGIN = np.asarray(MODEL["origin"], dtype=np.float64)
if args.origin == "centroid":
    try:
        ORIGIN = _measured_zero_order()
    except Exception as exc:
        print(f"WARNING: measured 0-order failed ({exc}); using the model origin")
print(f"origin ({args.origin}) = {ORIGIN[0]:.2f}, {ORIGIN[1]:.2f}   "
      f"phase_sign={args.phase_sign:+.0f}  scope={args.phase_sign_scope}")

params = FrozenDefocusParameters(
    defocus_diopter=MODEL["defocus_diopter"],
    origin_xy=(float(ORIGIN[0]), float(ORIGIN[1])),
    amplitude_center_xy=tuple(MODEL["amplitude_center"]),
    amplitude_width_xy=tuple(MODEL["amplitude_width"]),
    residual_noll_radians={},
    gain=MODEL["gain"],
    background=MODEL["background"],
    sigma_ccd_px=0.0,
    ramp_matrix=((RAMP[0][0], RAMP[0][1]), (RAMP[1][0], RAMP[1][1])),
    phase_sign=args.phase_sign,
    phase_sign_scope=args.phase_sign_scope,
)

ctr = ORIGIN
yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
camera_xy = np.stack([xx + ctr[0] - CROP / 2, yy + ctr[1] - CROP / 2], axis=-1)
forward = FrozenDefocusForward(params, camera_xy, device="cpu")

amp_tag = f"{args.amp:.2f}"


def load(run_pass, file, exposure_ms):
    d = np.load(RUN / run_pass / file)
    raw = d["raw_frames"].astype(np.float64).mean(axis=0)
    rate = raw / float(exposure_ms or 1.0)
    return d["model_phase"].astype(np.float32), rate


def crop(img):
    x0 = int(round(ctr[0] - CROP / 2))
    y0 = int(round(ctr[1] - CROP / 2))
    return img[y0:y0 + CROP, x0:x0 + CROP]


results = []
for run_pass in ["pass1", "pass2"]:
    mpath = RUN / run_pass / "manifest.json"
    if not mpath.is_file():
        continue
    manifest = json.loads(mpath.read_text())
    for mode in args.modes:
        lp = [r for r in manifest if f"_Z{mode}_+{amp_tag}" in r["label"]]
        lm = [r for r in manifest if f"_Z{mode}_-{amp_tag}" in r["label"]]
        if not lp or not lm:
            print(f"{run_pass} Z{mode:2d}  MISSING labels")
            continue
        pm, rm = load(run_pass, lm[0]["file"], lm[0]["exposure_ms"])
        pp, rp = load(run_pass, lp[0]["file"], lp[0]["exposure_ms"])
        with torch.no_grad():
            pred_m = forward.predict(torch.as_tensor(pm)).cpu().numpy()[0]
            pred_p = forward.predict(torch.as_tensor(pp)).cpu().numpy()[0]
        meas_m = crop(rm)
        meas_p = crop(rp)
        agr = direction_agreement(pred_m, pred_p, meas_m, meas_p)
        results.append({"pass": run_pass, "mode": mode, "corr": agr.correlation,
                        "verdict": agr.verdict, "rel_l2": agr.relative_l2,
                        "predicted_rms": agr.predicted_rms, "measured_rms": agr.measured_rms})
        print(f"{run_pass} Z{mode:2d}  corr={agr.correlation:+.3f}  verdict={agr.verdict}  "
              f"rel_l2={agr.relative_l2:.3f}")

corrs = np.asarray([r["corr"] for r in results], dtype=np.float64)
finite = corrs[np.isfinite(corrs)]
stats = {
    "n": int(len(results)),
    "n_finite": int(finite.size),
    "median": float(np.median(finite)) if finite.size else None,
    "mean": float(np.mean(finite)) if finite.size else None,
    "std_ddof1": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
    "n_ge_0p5": int(np.sum(finite >= 0.5)),
    "n_ge_0p5_over": f"{int(np.sum(finite >= 0.5))}/{int(finite.size)}",
    "agree": int(sum(1 for r in results if r["verdict"] == "agree")),
    "agree_weak": int(sum(1 for r in results if r["verdict"] in ("agree", "weak"))),
    "corr_positive": int(np.sum(finite > 0)),
}
print(f"\n[{RUN.name} amp={amp_tag} scope={args.phase_sign_scope}] "
      f"pairs={stats['n']}  median={stats['median']:+.3f}  "
      f"mean={stats['mean']:+.3f}  std={stats['std_ddof1']:.3f}  "
      f">=0.5={stats['n_ge_0p5_over']}")

# Component availability for this screen group: the single-mode screens carry only
# the commanded Zernike, with no contained focus and no separable carrier.
availability = {
    "focus_available": False,
    "focus_contained_in_panel_phase": False,
    "carrier_available": False,
    "note": "single-mode screens: total-only tilt-plane split, no external components",
}

tag = f"sign{'p' if args.phase_sign > 0 else 'm'}1_{args.origin}_{args.phase_sign_scope}"
out = args.out or (RUN / f"direction_screen_amp{amp_tag.replace('.', '')}_{args.model}_{tag}.json")
out.write_text(json.dumps({"run": str(RUN), "amp_rad": args.amp, "model": args.model,
                           "fit_source": str(args.fit),
                           "phase_sign": args.phase_sign,
                           "phase_sign_scope": args.phase_sign_scope,
                           "origin_mode": args.origin,
                           "origin_xy": [float(ORIGIN[0]), float(ORIGIN[1])],
                           "crop": CROP,
                           "components": availability,
                           "stats": stats,
                           "results": results}, indent=2))
print("wrote", out)
