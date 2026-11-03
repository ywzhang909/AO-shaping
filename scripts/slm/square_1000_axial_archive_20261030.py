"""Offline homogeneous-grid/exposure subset and conditional axial scan.

Remove after the square-1000 report is final and no rerun is required.
References original NPZ files by absolute path; never moves or edits inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ao_shaping.tools.slm.transfer.axial import scan

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/multishape_nogs_20261030"
SESSION = DATA / "p20261030_s1000_square_plane_formation"


def select_rows(capture):
    rows = json.loads((capture/"manifest.json").read_text())
    selected, excluded = [], []
    reference_xy, reference_core = None, None
    for row in rows:
        reason = "different exposure or grid"
        if row["camera_grid"] == [377, 377, 2] and abs(row["exposure_ms"]-.717) < 1e-6:
            with np.load(capture/row["file"]) as saved:
                xy, core = saved["camera_xy"], saved["core"]
                raw = saved["raw_frames"]
                # Reject any clipping in the target, not just a sparse core.
                target = saved["target"] > .5
                samples = raw[:, xy[..., 1].astype(int), xy[..., 0].astype(int)]
                if samples[:, target].max() >= .98*row["full_scale"]:
                    reason = "sampled full target saturated"
                elif reference_xy is not None and (not np.array_equal(reference_xy, xy)
                                                   or not np.array_equal(reference_core, core)):
                    reason = "different coordinates or core mask"
                else:
                    reference_xy, reference_core = xy.copy(), core.copy()
                    record = dict(row, file=str((capture/row["file"]).resolve()),
                                  phase_sha256=hashlib.sha256(saved["model_phase"].tobytes()).hexdigest())
                    selected.append(record)
                    continue
        excluded.append(dict(file=row["file"], reason=reason))
    if len({r["phase_sha256"] for r in selected}) < 3:
        raise ValueError("insufficient distinct unsaturated phases")
    return selected, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected, excluded = select_rows(SESSION/"common_replay")
    args.out.mkdir(parents=True, exist_ok=False)
    subset = args.out/"subset"
    subset.mkdir()
    (subset/"manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (args.out/"subset_audit.json").write_text(json.dumps(dict(selected=len(selected),
        distinct_phases=len({r["phase_sha256"] for r in selected}), excluded=excluded,
        warning="single exposure: background and electronic bias cannot be separated; effective conditional distance only"),
        indent=2), encoding="utf-8")
    scan(SimpleNamespace(checkpoint=SESSION/"model640/frozen_parameters.json",
         capture=subset, out=args.out/"scan", sample_stride=4,
         offsets=[-.2, -.15, -.1, -.05, 0., .05, .1, .15, .2], device="cpu"))


if __name__ == "__main__":
    main()
