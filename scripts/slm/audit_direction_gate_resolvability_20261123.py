"""Audit: how much resolving power does the transfer direction gate actually have?

The 2026-11-20 P3 direction holdout gate failed (pooled 4/14, needs 10/14).
A failed gate is only a *model* verdict if the measurement itself is
reproducible: the best score any predictor could earn on this hardware is the
correlation between the measured differential pattern of pass 1 and of pass 2
for the same mode.  This script computes that ceiling per mode, plus the
metric's own signal check, so a later gate can be judged against what the
data can support.

Reads archived frames only: never opens SLM, camera, laser or DeviceLock.

Usage::

    .venv/Scripts/python.exe scripts/slm/audit_direction_gate_resolvability_20261123.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slm"))

from ao_shaping.tools.slm.transfer.jacobian_guard import (
    DEFAULT_MIN_SIGNAL,
)

import direction_holdout_gate_20261119 as gate

PASSES = {
    "pass1": ROOT / "data/debug/slm_transfer/p3_direction_pass1_20261120",
    "pass2": ROOT / "data/debug/slm_transfer/p3_direction_pass2_20261120",
}
MODES = (5, 6, 7, 8, 9, 10, 11)
CROP = 128


def _manifest(dataset: Path) -> list[dict]:
    return json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))


def _crop(img: np.ndarray, centre_xy: tuple[float, float]) -> np.ndarray:
    x0 = int(round(centre_xy[0] - CROP / 2))
    y0 = int(round(centre_xy[1] - CROP / 2))
    return img[y0 : y0 + CROP, x0 : x0 + CROP]


def _measured_pair(
    dataset: Path, manifest: list[dict], mode: int, centre: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Measured anchor/candidate frames (plus/minus) of one mode, cropped."""
    minus, plus, _amp = gate.pair_rows(manifest, mode)
    dm = np.load(dataset / str(minus["file"]))
    dp = np.load(dataset / str(plus["file"]))
    meas_m = _crop(
        dm["raw_frames"].astype(np.float64).mean(axis=0) / float(minus["exposure_ms"]),
        centre,
    )
    meas_p = _crop(
        dp["raw_frames"].astype(np.float64).mean(axis=0) / float(plus["exposure_ms"]),
        centre,
    )
    return meas_m, meas_p


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = (a - a.mean()).ravel()
    y = (b - b.mean()).ravel()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def main() -> int:
    manifests = {name: _manifest(d) for name, d in PASSES.items()}
    centres = {
        name: gate.measured_centre(d, manifests[name]) for name, d in PASSES.items()
    }
    rows: list[dict] = []
    for mode in MODES:
        diffs: dict[str, np.ndarray] = {}
        anchors: dict[str, np.ndarray] = {}
        for name in PASSES:
            meas_m, meas_p = _measured_pair(
                PASSES[name], manifests[name], mode, centres[name]
            )
            diffs[name] = meas_p - meas_m
            anchors[name] = meas_m
        d1 = diffs["pass1"]
        meas_rms = float(np.sqrt(np.mean(d1**2)))
        reference = float(np.mean(np.abs(anchors["pass1"]))) + 1e-12
        rows.append(
            {
                "mode": mode,
                "reproducibility_ceiling": round(_pearson(diffs["pass1"], diffs["pass2"]), 4),
                "anchor_reproducibility": round(_pearson(anchors["pass1"], anchors["pass2"]), 4),
                "measured_rms_over_reference": round(meas_rms / reference, 6),
                "measured_rms": round(meas_rms, 6),
                "reference_mean_abs": round(reference, 4),
                "insufficient_signal": bool(meas_rms / reference < DEFAULT_MIN_SIGNAL),
                "resolvable": bool(_pearson(diffs["pass1"], diffs["pass2"]) >= 0.5),
            }
        )

    out = ROOT / "data/debug/slm_transfer/p3_gate_20261120/resolvability_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kind": "direction_gate_resolvability_audit",
                "passes": {k: str(v) for k, v in PASSES.items()},
                "centres": {k: list(v) for k, v in centres.items()},
                "min_signal_threshold": DEFAULT_MIN_SIGNAL,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows:
        print(
            "Z%-3d ceiling=%+.4f  anchor=%+.4f  rms/ref=%.4f  resolvable=%s"
            % (
                row["mode"],
                row["reproducibility_ceiling"],
                row["anchor_reproducibility"],
                row["measured_rms_over_reference"],
                row["resolvable"],
            )
        )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
