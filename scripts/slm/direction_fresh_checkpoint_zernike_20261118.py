"""Single-mode (±Zernike) direction gate for a *fresh* checkpoint, offline.

Companion to ``direction_fresh_checkpoint_20261118.py``.  That script replays
the 2026-11-17 square-probe sessions with a 3-pair (N4/RX/RY) tilt subset; this
one replays the 2026-11-07 issue234 single-mode screens (±Z5–Z11 at
0.02/0.05/0.10 rad, two passes), which is the full 14-pair screen the 0.50 gate
was defined on.

Why an epoch-correct centre matters
-----------------------------------
The hardware zero order moved ~179 CCD px between 2026-11-07 (measured spot
~1302,943) and 2026-11-16 (measured spot ~1224,1105): see
``report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261116/fwdmodel_fit.md``.
A checkpoint fitted in one epoch is registered to that epoch's origin, so
evaluating it on the other epoch's frames without overriding the origin measures
the axis move, not the model.  ``--center-source measured`` therefore locates the
crop centre (and the model origin) from the replay run's own reference frame, so
both variants see identical, epoch-correct registration; the only thing that
differs between them is the fitted optical parameters.

Offline only: reads archived sessions, never opens SLM, camera, laser or
DeviceLock.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.jacobian_guard import direction_agreement
from ao_shaping.utils.image.zero_order import (
    ZERO_ORDER_METHODS,
    zero_order_center_by_method,
)

OLD_FIT = Path(
    "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json"
)
CROP = 128
GRID = 640
MODES = (5, 6, 7, 8, 9, 10, 11)
CENTER_METHODS = [key for key, _label in ZERO_ORDER_METHODS]
_AMP = re.compile(r"_Z\d+_([+-])(\d+\.\d+)$")


def measured_centre(run: Path, run_pass: str, method: str) -> tuple[float, float]:
    manifest = json.loads((run / run_pass / "manifest.json").read_text(encoding="utf-8"))
    means = [
        np.load(run / run_pass / row["file"])["raw_frames"].astype(np.float64).mean(axis=0)
        for row in manifest
    ]
    cx, cy = zero_order_center_by_method(np.mean(means, axis=0), method=method)
    return float(cx), float(cy)


def pair_rows(manifest: list[dict], mode: int) -> tuple[dict, dict, str] | None:
    plus = [r for r in manifest if re.search(rf"_Z{mode}_\+", r["label"])]
    minus = [r for r in manifest if re.search(rf"_Z{mode}_-", r["label"])]
    if not plus or not minus:
        return None
    amp = _AMP.search(plus[0]["label"])
    return minus[0], plus[0], amp.group(2) if amp else "?"


def build_forward(
    model: dict, ramp: list[list[float]], centre_xy: tuple[float, float], grid: int
) -> FrozenDefocusForward:
    params = FrozenDefocusParameters(
        defocus_diopter=float(model["defocus_diopter"]),
        origin_xy=(float(centre_xy[0]), float(centre_xy[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={
            int(mode): float(value)
            for mode, value in model.get("residual_noll_radians", {}).items()
        },
        gain=float(model["gain"]),
        background=float(model["background"]),
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=-1.0,
        phase_sign_scope="command",
        grid_size=grid,
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack(
        [xx + centre_xy[0] - CROP / 2, yy + centre_xy[1] - CROP / 2], axis=-1
    )
    return FrozenDefocusForward(params, camera_xy, device="cpu")


def crop(img: np.ndarray, centre_xy: tuple[float, float]) -> np.ndarray:
    x0 = int(round(centre_xy[0] - CROP / 2))
    y0 = int(round(centre_xy[1] - CROP / 2))
    return img[y0 : y0 + CROP, x0 : x0 + CROP]


def run_one(
    run: Path,
    models: dict[str, tuple[dict, list[list[float]]]],
    center_method: str,
    center_source: str,
) -> dict:
    out: dict = {"passes": {}, "centre": {}}
    for run_pass in ("pass1", "pass2"):
        manifest = json.loads(
            (run / run_pass / "manifest.json").read_text(encoding="utf-8")
        )
        rows = {
            mode: pair_rows(manifest, mode)
            for mode in MODES
        }
        rows = {m: r for m, r in rows.items() if r is not None}
        if center_source == "measured":
            centre = measured_centre(run, run_pass, center_method)
        else:
            centre = tuple(models["fresh"][0]["origin"])  # type: ignore[arg-type]
        out["centre"][run_pass] = list(centre)
        first_file = next(iter(rows.values()))[0]["file"]
        grid = int(np.load(run / run_pass / first_file)["model_phase"].shape[0])
        out.setdefault("grid_size", grid)
        fwds = {
            name: build_forward(m, r, centre, grid)
            for name, (m, r) in models.items()
        }
        per_pass: dict = {}
        for mode, (rminus, rplus, amp) in rows.items():
            dm = np.load(run / run_pass / rminus["file"])
            dp = np.load(run / run_pass / rplus["file"])
            phase_m = dm["model_phase"].astype(np.float32)
            phase_p = dp["model_phase"].astype(np.float32)
            meas_m = crop(
                dm["raw_frames"].astype(np.float64).mean(axis=0)
                / float(rminus.get("exposure_ms") or 1.0),
                centre,
            )
            meas_p = crop(
                dp["raw_frames"].astype(np.float64).mean(axis=0)
                / float(rplus.get("exposure_ms") or 1.0),
                centre,
            )
            entry = {"amp": float(amp)}
            for name, fwd in fwds.items():
                with torch.no_grad():
                    pred_m = fwd.predict(torch.as_tensor(phase_m)).cpu().numpy()[0]
                    pred_p = fwd.predict(torch.as_tensor(phase_p)).cpu().numpy()[0]
                agr = direction_agreement(pred_m, pred_p, meas_m, meas_p)
                entry[name] = {
                    "corr": float(agr.correlation),
                    "predicted_rms": float(agr.predicted_rms),
                    "measured_rms": float(agr.measured_rms),
                }
            per_pass[f"Z{mode}"] = entry
        out["passes"][run_pass] = per_pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center-method", default="ellipse", choices=CENTER_METHODS)
    parser.add_argument(
        "--center-source", default="measured", choices=["measured", "checkpoint"]
    )
    args = parser.parse_args()

    ckpt = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    fresh = {
        "defocus_diopter": ckpt["defocus_diopter"],
        "amplitude_center": ckpt["amplitude_center"],
        "amplitude_width": ckpt["amplitude_width"],
        "gain": ckpt["gain"],
        "background": ckpt["background"],
        "residual_noll_radians": ckpt.get("residual_noll_radians", {}),
    }
    old_fit = json.loads(OLD_FIT.read_text(encoding="utf-8"))
    old_ramp = old_fit["ramp_matrix"]
    models = {
        "fresh": (fresh, ckpt["ramp_matrix"]),
        "fresh_oldramp": (fresh, old_ramp),
        "M3_old": (old_fit["models"]["M3_illumination"], old_ramp),
    }

    result = {
        "checkpoint": args.checkpoint.as_posix(),
        "convention": {
            "metric": "jacobian_guard.direction_agreement Pearson",
            "rate": "raw_frames mean / exposure_ms",
            "phase_sign": -1.0,
            "phase_sign_scope": "command",
            "crop": CROP,
            "grid_size": GRID,
            "center_method": args.center_method,
            "center_source": args.center_source,
            "modes": list(MODES),
        },
        "runs": {},
    }
    for run in args.runs:
        result["runs"][run.as_posix()] = run_one(
            run, models, args.center_method, args.center_source
        )

    # pooled per-amplitude summary across all runs/passes/modes
    pooled: dict[str, dict] = {}
    for variant in models:
        by_amp: dict[float, list[float]] = {}
        for run_result in result["runs"].values():
            for per_pass in run_result["passes"].values():
                for entry in per_pass.values():
                    by_amp.setdefault(entry["amp"], []).append(entry[variant]["corr"])
        pooled[variant] = {}
        for amp, corrs in sorted(by_amp.items()):
            arr = np.asarray(corrs, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            pooled[variant][f"{amp:.2f}"] = {
                "n": int(arr.size),
                "n_finite": int(finite.size),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "count_ge_0_50": int((finite >= 0.5).sum()),
            }
    result["pooled"] = pooled

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for variant, amps in pooled.items():
        for amp, s in amps.items():
            print(
                f"POOLED {variant} amp={amp} n={s['n']} finite={s['n_finite']} "
                f"median={s['median']:.3f} >=0.5={s['count_ge_0_50']}"
            )


if __name__ == "__main__":
    main()
