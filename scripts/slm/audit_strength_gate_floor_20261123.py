"""Audit: what does the transfer strength gate read for a *perfect* predictor?

The strength gate fails a checkpoint when the worst holdout pass exceeds the
pre-registered ``delta = 0.20`` masked absolute relL2.  On 8-bit captures the
metric has a noise floor: an oracle that predicts another frame of the *same*
pattern still scores above zero.  This script measures that floor per holdout
row, so a verdict can be compared against what the acquisition can support.

Reads archived frames only: never opens SLM, camera, laser or DeviceLock.

Usage::

    .venv/Scripts/python.exe scripts/slm/audit_strength_gate_floor_20261123.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slm"))

from ao_shaping.tools.slm.transfer import fit as transfer_fit
from ao_shaping.tools.slm.transfer.strength_gate import (
    gate_rel_l2,
    window_zero_mask,
)

import evaluate_strength_gate_20261119 as gate

DATASET = ROOT / "data/debug/slm_transfer/p3_zernike_20261120"
DARK = ROOT / "data/ccd_configs/ccd/FJB24112232/darks/dark-exp-0.087ms.npz"
SPAN = 160
WINDOW_STEP = 2.0
ZERO_RADIUS_PX = 4.0
DARK_EXPOSURE_MS = 0.087
DELTA_PREREGISTERED = 0.20
HOLDOUT_SPLITS = {"test", "validation"}


def main() -> int:
    rows = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    usable = [r for r in rows if transfer_fit.valid(r)]
    reference = next(r for r in usable if r["label"] == "reference_start")
    centre = transfer_fit.spot(reference)
    yy, xx = np.mgrid[-SPAN : SPAN + 1 : 2, -SPAN : SPAN + 1 : 2]
    xy = np.stack((xx + round(centre[0]), yy + round(centre[1])), axis=-1)
    shape = (int(yy.shape[0]), int(xx.shape[0]))
    mask = window_zero_mask(shape, window_step=WINDOW_STEP, zero_radius_px=ZERO_RADIUS_PX)
    dark_rate, _evidence = gate.load_dark(
        DARK, window_shape=shape, xy=xy, exposure_override_ms=DARK_EXPOSURE_MS
    )

    out_rows: list[dict] = []
    for row in usable:
        if row.get("split") not in HOLDOUT_SPLITS:
            continue
        with np.load(DATASET / str(row["file"])) as payload:
            raw = np.asarray(payload["raw_frames"], dtype=float)
        exposure = float(row["exposure_ms"])
        windows = [
            raw[i][xy[..., 1].astype(int), xy[..., 0].astype(int)] / exposure
            for i in range(raw.shape[0])
        ]
        pairs: list[float] = []
        for a in range(len(windows)):
            for b in range(a + 1, len(windows)):
                scored = gate_rel_l2(
                    windows[a], windows[b], mask, bias_rate=0.0, dark_rate=dark_rate
                )
                pairs.append(float(scored["rel_l2"]))
        energy = gate_rel_l2(
            windows[0], windows[0], mask, bias_rate=0.0, dark_rate=dark_rate
        )
        out_rows.append(
            {
                "label": row["label"],
                "split": row["split"],
                "n_frames": len(windows),
                "oracle_floor_rel_l2": round(float(np.mean(pairs)), 4),
                "oracle_floor_pairs": [round(p, 4) for p in pairs],
                "energy_meas_counts_ms": round(float(energy["energy_meas"]), 2),
                "n_pixels": int(energy["n_pixels"]),
                "peak_counts": float(row.get("peak", 0.0)),
            }
        )

    floors = [r["oracle_floor_rel_l2"] for r in out_rows]
    out = ROOT / "data/debug/slm_transfer/p3_gate_20261120/strength_floor_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kind": "strength_gate_noise_floor_audit",
                "dataset": str(DATASET),
                "delta_preregistered": DELTA_PREREGISTERED,
                "oracle_floor_median": round(float(np.median(floors)), 4),
                "oracle_floor_min": round(float(min(floors)), 4),
                "oracle_floor_max": round(float(max(floors)), 4),
                "rows": out_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in out_rows:
        print(
            "%-28s floor=%.4f  energy=%.1f  peak=%.0f"
            % (
                row["label"],
                row["oracle_floor_rel_l2"],
                row["energy_meas_counts_ms"],
                row["peak_counts"],
            )
        )
    print(
        "oracle floor median %.4f (min %.4f max %.4f) vs preregistered delta %.2f -> %s"
        % (
            float(np.median(floors)),
            float(min(floors)),
            float(max(floors)),
            DELTA_PREREGISTERED,
            "floor ABOVE delta" if float(np.median(floors)) > DELTA_PREREGISTERED else "floor below delta",
        )
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
