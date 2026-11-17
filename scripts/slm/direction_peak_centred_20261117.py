"""Re-run the M0/M1/M3 direction-gate check with a 128x128 crop centred on the measured peak.

Offline only: reads archived sessions; never touches SLM, camera, laser or DeviceLock.

Difference from ``direction_attribution_20261112.py``: the crop and camera grid are centred on the
measured peak (native row 1105, col 1224) for every model variant, instead of each model's fitted
origin. ``--center-method`` replaces that fixed centre with a per-session one located from the
session's mean frame (``zero_order_center_by_method``); unset keeps the fixed peak.
Metric and convention are unchanged: ``jacobian_guard.direction_agreement`` with Pearson
correlation on rate = raw_mean / exposure_ms, phase_sign=-1, phase_sign_scope="command".
Each pair uses minus frame as the anchor and plus frame as the candidate, as the historical script.
"""

from __future__ import annotations

import argparse
import json
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

FIT = Path("data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json")
PEAK_ROW, PEAK_COL = 1105, 1224  # native CCD, measured peak
CROP = 128
GRID = 640  # captured model_phase grid (the historical 128 px grid cannot take 640x640 phases)
# pair name -> (manifest label minus, manifest label plus)
PAIRS = {
    "N4": ("noll4_minus", "noll4_plus"),
    "RX": ("tilt_x_minus", "tilt_x_plus"),
    "RY": ("tilt_y_minus", "tilt_y_plus"),
}
MODELS = ["M0", "M1_defocus", "M3_illumination"]


def _find(manifest: list[dict], label: str) -> dict:
    for row in manifest:
        if row["label"] == label:
            return row
    raise KeyError(label)


def load_frame(session: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(session / row["file"])
    raw = d["raw_frames"].astype(np.float64).mean(axis=0)
    phase = d["model_phase"].astype(np.float32)
    return phase, raw / float(row.get("exposure_ms") or 1.0)


def crop(img: np.ndarray, centre_xy: tuple[float, float]) -> np.ndarray:
    x0 = int(round(centre_xy[0] - CROP / 2))
    y0 = int(round(centre_xy[1] - CROP / 2))
    return img[y0:y0 + CROP, x0:x0 + CROP]


def session_crop_centre(session: Path, method: str) -> tuple[float, float]:
    """用该 session 全部记录帧的均值帧, 按 ``method`` 定位裁剪中心 ``(x, y)``。"""
    manifest = json.loads((session / "manifest.json").read_text())
    means = [
        np.load(session / row["file"])["raw_frames"].astype(np.float64).mean(axis=0)
        for row in manifest
    ]
    cx, cy = zero_order_center_by_method(np.mean(means, axis=0), method=method)
    return (float(cx), float(cy))


def build_forward(model: dict, ramp, centre_xy: tuple[float, float]) -> FrozenDefocusForward:
    params = FrozenDefocusParameters(
        defocus_diopter=model["defocus_diopter"],
        origin_xy=(float(centre_xy[0]), float(centre_xy[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={},
        gain=model["gain"],
        background=model["background"],
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=-1.0,
        phase_sign_scope="command",
        grid_size=GRID,
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack(
        [xx + centre_xy[0] - CROP / 2, yy + centre_xy[1] - CROP / 2], axis=-1
    )
    return FrozenDefocusForward(params, camera_xy, device="cpu")


def run_session(session: Path, models: dict, ramp, centre_xy: tuple[float, float]) -> dict:
    manifest = json.loads((session / "manifest.json").read_text())
    out: dict = {}
    for name, (lab_minus, lab_plus) in PAIRS.items():
        phase_m, rate_m = load_frame(session, _find(manifest, lab_minus))
        phase_p, rate_p = load_frame(session, _find(manifest, lab_plus))
        meas_m = crop(rate_m, centre_xy)
        meas_p = crop(rate_p, centre_xy)
        per_model = {}
        for mkey, mparams in models.items():
            fwd = build_forward(mparams, ramp, centre_xy)
            with torch.no_grad():
                pred_m = fwd.predict(torch.as_tensor(phase_m)).cpu().numpy()[0]
                pred_p = fwd.predict(torch.as_tensor(phase_p)).cpu().numpy()[0]
            agr = direction_agreement(pred_m, pred_p, meas_m, meas_p)
            per_model[mkey] = {"corr": float(agr.correlation), "verdict": agr.verdict}
        out[name] = per_model
    return out


def summarise(per_pair: dict, mkey: str) -> dict:
    arr = np.asarray([per_pair[p][mkey]["corr"] for p in PAIRS], dtype=np.float64)
    return {
        "per_pair": {p: per_pair[p][mkey]["corr"] for p in PAIRS},
        "median": float(np.nanmedian(arr)),
        "count_ge_0_50": int(np.sum(arr >= 0.50)),
        "n_pairs": len(PAIRS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--amplitude-center-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
        help="override the illumination-envelope centre (panel px) in every model variant; "
             "default keeps each model's own fitted amplitude_center",
    )
    ap.add_argument(
        "--center-method",
        choices=[key for key, _ in ZERO_ORDER_METHODS],
        default=None,
        help="per session, locate the crop centre once from that session's mean frame with this "
        f"algorithm; unset keeps the fixed measured peak (col {PEAK_COL}, row {PEAK_ROW})",
    )
    args = ap.parse_args()

    fit = json.loads(FIT.read_text())
    ramp = fit["ramp_matrix"]
    models = {k: dict(fit["models"][k]) for k in MODELS}
    if args.amplitude_center_xy is not None:
        for m in models.values():
            m["amplitude_center"] = [float(args.amplitude_center_xy[0]), float(args.amplitude_center_xy[1])]
    centre_xy = (float(PEAK_COL), float(PEAK_ROW))

    result: dict = {
        "crop_centre_native_row_col": [PEAK_ROW, PEAK_COL],
        "crop_centre_xy": list(centre_xy),
        "crop": CROP,
        "amplitude_center_override_xy": (
            list(args.amplitude_center_xy) if args.amplitude_center_xy is not None else None
        ),
        "amplitude_center_used_xy": {k: list(v["amplitude_center"]) for k, v in models.items()},
        "center_method": args.center_method,
        "convention": {
            "metric": "jacobian_guard.direction_agreement Pearson",
            "rate": "raw_frames mean / exposure_ms",
            "phase_sign": -1.0,
            "phase_sign_scope": "command",
        },
        "sessions": {},
    }
    for s in args.sessions:
        if args.center_method is None:
            session_centre = centre_xy
        else:
            session_centre = session_crop_centre(s, args.center_method)
        per_pair = run_session(s, models, ramp, session_centre)
        result["sessions"][s.name] = {
            "summary": {m: summarise(per_pair, m) for m in MODELS},
            "raw_per_pair": per_pair,
            "center_method": args.center_method,
            "session_crop_center_xy": list(session_centre),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, sess in result["sessions"].items():
        for m, s in sess["summary"].items():
            print(name, m, "median=%.3f" % s["median"], "count>=0.5=%d/%d" % (s["count_ge_0_50"], s["n_pairs"]))


if __name__ == "__main__":
    main()
