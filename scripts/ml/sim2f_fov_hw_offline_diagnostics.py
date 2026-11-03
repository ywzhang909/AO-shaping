"""Offline diagnostics for the 2026-11-03 ring refine run (no hardware).

Reads a completed refine directory and a completed capture directory, then
reports three quantities as one JSON artifact:

1. Window alignment: the integer shift that best aligns the model far field
   with the measured CCD crop (normalised cross-correlation, template match).
2. Effective panel phase: the phase actually displayed on the SLM, recovered
   from the recorded ``panel_gray`` arrays (gray * 2*pi / max_gray, area
   resampled to the model grid), compared with the commanded 128-grid phase.
3. Capture response: peak counts, total energy and pairwise correlations of
   the flat and shaped frames of the capture session.

The recovered effective phase is an approximation (it ignores the wavefront
correction and the LUT, which are already inside ``panel_gray``), so it is a
diagnostic, not a replacement for the device pipeline. Nothing here is
acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import match_template

from ml.sim2f.calibrate import CalibratedBench, load_calibration
from ml.sim2f.losses import objective_terms
from ml.sim2f.optics import Sim2f, Sim2fConfig

ROOT = Path(__file__).resolve().parents[2]
HALF = 128
MARGIN = 60
W_EE = 0.25
METRIC = "l1"
PUPIL_CORNER = (510, 150)
PUPIL_SIDE = 900
MAX_GRAY = 993.0
ROUNDS = ["reference", "round_0001", "round_0002", "round_0003", "round_0004"]
SHAPE_KEYS = [
    "shape_ring_nominal_r0",
    "shape_cross_nominal_r0",
    "shape_frame_nominal_r0",
    "shape_two_spot_nominal_r0",
]


def _score(image: np.ndarray, target: torch.Tensor, support: torch.Tensor) -> dict:
    img = torch.as_tensor(image, dtype=torch.float32)[None, None]
    with torch.no_grad():
        loss, l1, ee, ncc = objective_terms(
            img, target, support, w_ee=W_EE, metric=METRIC, prepare=True
        )
    return {"objective": float(loss), "ee": float(ee), "ncc": float(ncc)}


def _effective_phase(refine: Path, name: str) -> np.ndarray:
    gray = np.load(refine / "panel_gray" / f"{name}.npy").astype(np.float64)
    x0, y0 = PUPIL_CORNER
    square = gray[y0 : y0 + PUPIL_SIDE, x0 : x0 + PUPIL_SIDE] * (2 * np.pi / MAX_GRAY)
    tensor = torch.as_tensor(square, dtype=torch.float32)[None, None]
    grid = F.interpolate(tensor, size=(128, 128), mode="area")
    return grid[0, 0].numpy().astype(np.float32)


def _predict(model: CalibratedBench, phase: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model.forward(torch.as_tensor(phase)[None], detector=False)
    return out[0, 0].cpu().numpy().astype(np.float64)


def _align(refine: Path, name: str, pred: np.ndarray, geom: dict) -> dict:
    cx, cy, ox, oy, full = geom["cx"], geom["cy"], geom["ox"], geom["oy"], geom["full"]
    frame = np.load(refine / "frames" / f"{name}.npy").astype(np.float64)
    square = frame[oy : oy + full, ox : ox + full]
    meas = square[cy - HALF : cy + HALF, cx - HALF : cx + HALF]
    region = pred[
        cy - HALF - MARGIN : cy + HALF + MARGIN, cx - HALF - MARGIN : cx + HALF + MARGIN
    ]
    ncc_map = match_template(region, meas, pad_input=False)
    dy, dx = np.unravel_index(int(np.nanargmax(ncc_map)), ncc_map.shape)
    return {
        "shift_xy": [int(dx - MARGIN), int(dy - MARGIN)],
        "ncc_at_zero": float(ncc_map[MARGIN, MARGIN]),
        "ncc_at_best": float(ncc_map[dy, dx]),
        "measured_crop": meas,
    }


def _refine_section(
    refine: Path,
    model: CalibratedBench,
    geom: dict,
    target: torch.Tensor,
    support: torch.Tensor,
) -> dict:
    rows: dict[str, dict] = {}
    for name in ROUNDS:
        raw = np.load(refine / "phases" / f"{name}.npy").astype(np.float32)
        eff = _effective_phase(refine, name)
        pred_raw = _predict(model, raw)
        pred_eff = _predict(model, eff)
        align_raw = _align(refine, name, pred_raw, geom)
        meas = align_raw.pop("measured_crop")
        cx, cy = geom["cx"], geom["cy"]
        sx, sy = align_raw["shift_xy"]
        eff_aligned = pred_eff[
            cy - HALF + sy : cy + HALF + sy, cx - HALF + sx : cx + HALF + sx
        ]
        rows[name] = {
            "raw_vs_effective_max_abs_rad": float(np.abs(eff - raw).max()),
            "raw_vs_effective_mean_abs_rad": float(np.abs(eff - raw).mean()),
            "effective_abs_gt_pi_fraction": float((np.abs(eff) > np.pi).mean()),
            "alignment_raw_model": align_raw,
            "measured": _score(meas, target, support),
            "model_raw_zero_shift": _score(
                pred_raw[cy - HALF : cy + HALF, cx - HALF : cx + HALF], target, support
            ),
            "model_effective_aligned": _score(eff_aligned, target, support),
            "model_effective_zero_shift": _score(
                pred_eff[cy - HALF : cy + HALF, cx - HALF : cx + HALF], target, support
            ),
        }
    ref = rows["reference"]
    gains = {}
    for name in ROUNDS[1:]:
        r = rows[name]
        gains[name] = {
            "measured": ref["measured"]["objective"] - r["measured"]["objective"],
            "model_raw_zero_shift": ref["model_raw_zero_shift"]["objective"]
            - r["model_raw_zero_shift"]["objective"],
            "model_effective_aligned": ref["model_effective_aligned"]["objective"]
            - r["model_effective_aligned"]["objective"],
            "model_effective_zero_shift": ref["model_effective_zero_shift"]["objective"]
            - r["model_effective_zero_shift"]["objective"],
        }
    return {"rounds": rows, "gains_vs_reference": gains}


def _capture_section(capture: Path) -> dict:
    manifest = json.loads(
        (capture / "capture_manifest.json").read_text(encoding="utf-8")
    )
    items = manifest["items"]

    def load(key: str) -> np.ndarray:
        return np.load(capture / items[key]["frame"]).astype(np.float64)

    flat = load("flat_0000")
    stats = {}
    for key in ["flat_0000", "calib_screen_000", *SHAPE_KEYS]:
        frame = load(key)
        stats[key] = {
            "sum": float(frame.sum()),
            "max": float(frame.max()),
            "p99_9": float(np.percentile(frame, 99.9)),
        }
    shaped = {k: load(k) for k in SHAPE_KEYS}
    pairwise = {}
    keys = list(shaped)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            pairwise[f"{a}|{b}"] = float(
                np.corrcoef(shaped[a].ravel(), shaped[b].ravel())[0, 1]
            )
    return {
        "stats": stats,
        "ncc_vs_flat": {
            k: float(np.corrcoef(flat.ravel(), shaped[k].ravel())[0, 1]) for k in keys
        },
        "pairwise_ncc_shaped": pairwise,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refine", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.refine = args.refine.resolve()
    args.capture = args.capture.resolve()
    args.calibration = args.calibration.resolve()
    args.out = args.out.resolve()
    summary = json.loads(
        (args.refine / "refine_summary.json").read_text(encoding="utf-8")
    )
    cx, cy = (int(v) for v in summary["analysis_window"]["center_xy"])
    ox, oy = (int(v) for v in summary["native_square_offset_xy"])
    full = int(summary["full_model_px"])
    geom = {"cx": cx, "cy": cy, "ox": ox, "oy": oy, "full": full}

    cfg = replace(Sim2fConfig.calibrated(), grid=128, ccd_px=full)
    model = CalibratedBench(
        Sim2f(cfg, device="cpu"), load_calibration(args.calibration)
    )
    target = torch.as_tensor(np.load(args.refine / "target.npy").astype(np.float32))
    support = torch.as_tensor(np.load(args.refine / "support.npy").astype(np.float32))

    result = {
        "refine_dir": str(args.refine.relative_to(ROOT)),
        "capture_dir": str(args.capture.relative_to(ROOT)),
        "window": {
            "center_xy": [cx, cy],
            "native_offset_xy": [ox, oy],
            "full_px": full,
        },
        "objective": {"w_ee": W_EE, "metric": METRIC, "prepare": True},
        "refine": _refine_section(args.refine, model, geom, target, support),
        "capture": _capture_section(args.capture),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
