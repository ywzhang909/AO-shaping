"""Direction gate for a *fresh* forward-model checkpoint, replayed offline.

Why this script exists (issue #234 / #235 / #237)
------------------------------------------------
``direction_peak_centred_20261117.py`` hardcodes the three 2026-11-07 model
variants (M0/M1/M3) from ``defocus_forward_20261107_focus_r2_fit/forward_fit.json``.
That checkpoint's fitted origin is ~1306,944 while the hardware zero order moved
to ~1224,1105 (issue #235, ``fwdmodel_fit.md``: stale by ~179 CCD px), so its
direction cosine on *fresh* frames is uninformative.

This script keeps the same metric and convention -- ``direction_agreement``
Pearson on rate = raw_mean / exposure_ms, ``phase_sign=-1``,
``phase_sign_scope="command"``, minus frame anchors and plus frame is the
candidate -- but lets the caller inject **any** ``frozen_parameters.json`` as the
model.  That makes the untested question answerable offline: does the 2026-11-18
fit (origin 1227.6,1102.4, epoch-matched to the fresh 11-17 screens) recover
direction on those screens?

Offline only: reads archived sessions, never opens SLM, camera, laser or
DeviceLock.
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

CENTER_METHODS = [key for key, _label in ZERO_ORDER_METHODS]

OLD_FIT = Path(
    "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json"
)
CROP = 128
# Captured ``model_phase`` arrays are 640x640; the historical 128 px grid cannot
# take them, and 640 is what the 2026-11-17 gate used, so keep it for parity.
GRID = 640
PAIRS = {
    "N4": ("noll4_minus", "noll4_plus"),
    "RX": ("tilt_x_minus", "tilt_x_plus"),
    "RY": ("tilt_y_minus", "tilt_y_plus"),
}


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
    return img[y0 : y0 + CROP, x0 : x0 + CROP]


def session_crop_centre(session: Path, method: str) -> tuple[float, float]:
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    means = [
        np.load(session / row["file"])["raw_frames"].astype(np.float64).mean(axis=0)
        for row in manifest
    ]
    cx, cy = zero_order_center_by_method(np.mean(means, axis=0), method=method)
    return float(cx), float(cy)


def build_forward(
    model: dict,
    ramp: list[list[float]],
    centre_xy: tuple[float, float],
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
        grid_size=GRID,
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack(
        [xx + centre_xy[0] - CROP / 2, yy + centre_xy[1] - CROP / 2], axis=-1
    )
    return FrozenDefocusForward(params, camera_xy, device="cpu")


def run_session(
    session: Path,
    models: dict[str, tuple[dict, list[list[float]]]],
    center_method: str,
) -> dict:
    centre_xy = session_crop_centre(session, center_method)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    out: dict = {"crop_centre_xy": list(centre_xy), "variants": {}}
    for name, (model, ramp) in models.items():
        fwd = build_forward(model, ramp, centre_xy)
        per_pair: dict[str, float] = {}
        for pair, (lab_minus, lab_plus) in PAIRS.items():
            phase_m, rate_m = load_frame(session, _find(manifest, lab_minus))
            phase_p, rate_p = load_frame(session, _find(manifest, lab_plus))
            meas_m = crop(rate_m, centre_xy)
            meas_p = crop(rate_p, centre_xy)
            with torch.no_grad():
                pred_m = fwd.predict(torch.as_tensor(phase_m)).cpu().numpy()[0]
                pred_p = fwd.predict(torch.as_tensor(phase_p)).cpu().numpy()[0]
            agr = direction_agreement(pred_m, pred_p, meas_m, meas_p)
            per_pair[pair] = float(agr.correlation)
        values = np.array(list(per_pair.values()), dtype=np.float64)
        out["variants"][name] = {
            "per_pair": per_pair,
            "median": float(np.median(values)),
            "count_ge_0_50": int((values >= 0.5).sum()),
            "n_pairs": int(values.size),
        }
    return out


def checkpoint_model(path: Path) -> tuple[dict, list[list[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = {
        "defocus_diopter": data["defocus_diopter"],
        "amplitude_center": data["amplitude_center"],
        "amplitude_width": data["amplitude_width"],
        "gain": data["gain"],
        "background": data["background"],
        "residual_noll_radians": data.get("residual_noll_radians", {}),
    }
    return model, data["ramp_matrix"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--center-method", default="ellipse", choices=CENTER_METHODS
    )
    args = parser.parse_args()

    fresh, fresh_ramp = checkpoint_model(args.checkpoint)
    old_fit = json.loads(OLD_FIT.read_text(encoding="utf-8"))
    old_ramp = old_fit["ramp_matrix"]
    old_m3 = old_fit["models"]["M3_illumination"]

    models: dict[str, tuple[dict, list[list[float]]]] = {
        # the checkpoint under test, with its own (11-18) ramp geometry
        "fresh": (fresh, fresh_ramp),
        # same optical parameters, 11-07 ramp: isolates the ramp contribution
        "fresh_oldramp": (fresh, old_ramp),
        # sanity check: must reproduce the published 11-17 M3 numbers (~0.00/0.007)
        "M3_old": (old_m3, old_ramp),
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
            "origin": "crop centre (measured zero order), same for every variant",
        },
        "sessions": {
            s.as_posix(): run_session(s, models, args.center_method)
            for s in args.sessions
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for session, payload in result["sessions"].items():
        for name, summary in payload["variants"].items():
            print(
                f"{Path(session).name} {name} median={summary['median']:.3f} "
                f"count>=0.5={summary['count_ge_0_50']}/{summary['n_pairs']}"
            )


if __name__ == "__main__":
    main()
