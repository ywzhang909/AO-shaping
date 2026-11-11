"""Attribute the M3_illumination direction-gate pass: illumination envelope vs origin/defocus.

Offline only. Reuses the validated口径 from ``direction_model_variants_20261112.py``
but allows per-parameter overrides to isolate which M3-vs-M1 difference drives the
direction improvement. Runs at amp 0.05 only (the gate's best-conditioned amp).

Variants:
* ``M3_env_on_M1`` -- M1_defocus origin/defocus/gain/bg but M3 amplitude envelope.
* ``M1_env_on_M3`` -- M3 illumination envelope + M3 origin/defocus/gain/bg but the
  amplitude_center/width replaced by M1's (inverse: remove the envelope).
* ``M3_origin_on_M1`` -- M1 envelope but M3 origin (isolate origin shift).
* ``M1_origin_on_M3`` -- M3 envelope but M1 origin (isolate origin shift on M3).

Outputs ``attribution.json`` under the same output directory.
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
OUT = Path("data/debug/slm_transfer/direction_variants_20261112/attribution.json")
RUN = Path("data/debug/slm_transfer/issue234_amp05_screen_20261107")
CROP = 128
AMP = 0.05


def load_frame(run_pass, row):
    d = np.load(RUN / run_pass / row["file"])
    raw = d["raw_frames"].astype(np.float64).mean(axis=0)
    return d["model_phase"].astype(np.float32), raw / float(row.get("exposure_ms") or 1.0)


def crop(img, ctr):
    x0 = int(round(ctr[0] - CROP / 2))
    y0 = int(round(ctr[1] - CROP / 2))
    return img[y0:y0 + CROP, x0:x0 + CROP]


def run(model_key, ov):
    fit = json.loads(FIT.read_text())
    model = dict(fit["models"][model_key])
    for k, v in ov.items():
        model[k] = v
    ramp = fit["ramp_matrix"]
    origin = np.asarray(model["origin"], dtype=np.float64)
    params = FrozenDefocusParameters(
        defocus_diopter=model["defocus_diopter"],
        origin_xy=(float(origin[0]), float(origin[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={},
        gain=model["gain"],
        background=model["background"],
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=-1.0,
        phase_sign_scope="command",
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack([xx + origin[0] - CROP / 2, yy + origin[1] - CROP / 2], axis=-1)
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    amp_tag = f"{AMP:.2f}"
    results = []
    for run_pass in ["pass1", "pass2"]:
        manifest = json.loads((RUN / run_pass / "manifest.json").read_text())
        for mode in range(5, 12):
            lp = [r for r in manifest if f"_Z{mode}_+{amp_tag}" in r["label"]]
            lm = [r for r in manifest if f"_Z{mode}_-{amp_tag}" in r["label"]]
            if not lp or not lm:
                continue
            pm, rm = load_frame(run_pass, lm[0])
            pp, rp = load_frame(run_pass, lp[0])
            with torch.no_grad():
                pred_m = forward.predict(torch.as_tensor(pm)).cpu().numpy()[0]
                pred_p = forward.predict(torch.as_tensor(pp)).cpu().numpy()[0]
            agr = direction_agreement(pred_m, pred_p, crop(rm, origin), crop(rp, origin))
            results.append({"pass": run_pass, "mode": mode, "corr": agr.correlation})
    corrs = np.asarray([r["corr"] for r in results])
    return {
        "n": len(results),
        "median": float(np.median(corrs)),
        "mean": float(np.mean(corrs)),
        "n_ge_0p5": int(np.sum(corrs >= 0.5)),
        "per_mode": {str(m): float(np.median([r["corr"] for r in results if r["mode"] == m]))
                     for m in range(5, 12)},
    }


def main():
    m1 = json.loads(FIT.read_text())["models"]["M1_defocus"]
    m3 = json.loads(FIT.read_text())["models"]["M3_illumination"]
    arms = {
        "M1_baseline": run("M1_defocus", {}),
        "M3_baseline": run("M3_illumination", {}),
        "M3_env_on_M1": run("M1_defocus", {
            "amplitude_center": m3["amplitude_center"],
            "amplitude_width": m3["amplitude_width"],
        }),
        "M1_env_on_M3": run("M3_illumination", {
            "amplitude_center": m1["amplitude_center"],
            "amplitude_width": m1["amplitude_width"],
        }),
        "M3_origin_on_M1": run("M1_defocus", {"origin": m3["origin"]}),
        "M1_origin_on_M3": run("M3_illumination", {"origin": m1["origin"]}),
        "M3_defocus_on_M1": run("M1_defocus", {"defocus_diopter": m3["defocus_diopter"]}),
    }
    out = {"amp": AMP, "arms": arms}
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in arms.items():
        print(f"{k:16s} median={v['median']:+.4f} mean={v['mean']:+.4f} "
              f">=0.5={v['n_ge_0p5']}/14")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
