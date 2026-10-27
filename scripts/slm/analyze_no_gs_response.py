"""Compare raw and affine-corrected CV on the exact recorded shaping core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    run = root / "local_shape1200"
    records = json.loads((run / "manifest.json").read_text())
    final = next(row for row in records if row["label"] == "square_final")
    with np.load(run / final["file"]) as saved:
        core, xy = saved["core"].copy(), saved["camera_xy"].copy()
    with np.load(root / "response_analysis/square_arrays.npz") as saved:
        if not np.array_equal(xy, saved["camera_xy"]):
            raise ValueError("response and shaping grids differ")
        times = saved["exposures_ms"].copy()
        images = saved["images"].copy()
        intercept = saved["intercept"].copy()
        train = saved["train_mask"].copy()
    corrected = (images-intercept)/times[:, None, None]
    rows = []
    for index, (time, raw, signal) in enumerate(zip(times, images, corrected)):
        rows.append(dict(index=index, exposure_ms=float(time), fit=bool(train[index]),
            raw_core_mean=float(raw[core].mean()), raw_core_cv=float(raw[core].std()/raw[core].mean()),
            corrected_core_cv=float(signal[core].std()/signal[core].mean()),
            corrected_core_rate=float(signal[core].mean()),
            corrected_negative_fraction=float(np.mean(signal[core]<0))))
    payload = dict(rows=rows, core_pixels=int(core.sum()),
        intercept_core_percentiles=np.percentile(intercept[core], [5, 50, 95]).tolist(),
        qualification="Same fixed core as shaping; signed illuminated per-pixel intercept fitted on first exposure visits; subsequent repeats validate only this fixed phase; not a dark calibration or cross-phase validation")
    (out / "response_core.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(times, [r["raw_core_cv"] for r in rows], label="raw CV")
    axes[0].scatter(times, [r["corrected_core_cv"] for r in rows], label="affine corrected CV")
    axes[0].set(xlabel="Exposure (ms)", ylabel="Fixed-core CV")
    axes[0].legend()
    axes[1].scatter(times, [r["raw_core_mean"] for r in rows], label="CCD mean")
    axes[1].set(xlabel="Exposure (ms)", ylabel="Fixed-core mean counts")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "response_core.png", dpi=140)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.out)
