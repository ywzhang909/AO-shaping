"""Render paired no-GS endpoints and audit native versus sampled CCD CV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def analyze(run: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "config.json").read_text())
    rows = json.loads((run / "manifest.json").read_text())
    names = {r["label"]: r for r in rows}
    start = names.get("square_seed_replay", names["square_seed"])
    finish = names["square_final"]
    zero = np.asarray(config["zero_xy"])
    target_center = np.asarray(config["target_xy"])
    images, phases, summary = [], [], []
    for row in (start, finish):
        with np.load(run / row["file"]) as data:
            raw = data["raw_frames"].copy()
            xy = data["camera_xy"].copy()
            phases.append(data["model_phase"].copy())
        xmin, xmax = int(xy[..., 0].min()), int(xy[..., 0].max())
        ymin, ymax = int(xy[..., 1].min()), int(xy[..., 1].max())
        frames = raw[:, ymin:ymax+1, xmin:xmax+1].astype(float)
        yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
        half = config["side"]/2
        box = (abs(xx-target_center[0]) < half) & (abs(yy-target_center[1]) < half)
        margin = config.get("core_margin") or 2*config["window_step"]
        core = ((abs(xx-target_center[0]) < half-margin)
                & (abs(yy-target_center[1]) < half-margin))
        excluded = (xx-zero[0])**2+(yy-zero[1])**2 <= config["zero_radius"]**2
        core_excluded = core & ~excluded
        frame = frames.mean(0)
        def cv(mask, frame_values=frame):
            values = frame_values[mask]
            return float(values.std()/values.mean())
        summary.append(dict(label=row["label"], exposure_ms=row["exposure_ms"],
            sampled=row["metrics"], native_cv=cv(core_excluded), native_cv_with_zero=cv(core),
            native_core_mean_counts=float(frame[core_excluded].mean()),
            native_ee_window=float(frame[box].sum()/frame.sum()),
            excluded_core_pixels=int((core & excluded).sum()),
            per_frame_native_cv=[float(f[core_excluded].std()/f[core_excluded].mean()) for f in frames],
            full_frame_peak=int(raw.max()), full_scale=row["full_scale"]))
        images.append(frame/(row["exposure_ms"]*row.get("counts_scale", 1)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    vmax = max(float(im.max()) for im in images)
    extent = (xmin, xmax, ymax, ymin)
    for index, (image, phase, item) in enumerate(zip(images, phases, summary)):
        axes[index, 0].imshow(np.mod(phase, 2*np.pi), cmap="twilight", vmin=0, vmax=2*np.pi)
        axes[index, 0].set_title(item["label"]+" / phase")
        axes[index, 1].imshow(image, cmap="inferno", vmin=0, vmax=vmax, extent=extent)
        axes[index, 1].add_patch(Rectangle(target_center-half, 2*half, 2*half,
                                        fill=False, edgecolor="cyan", linestyle="--"))
        axes[index, 1].set_title(f"counts/ms; native CV={item['native_cv']:.4f}")
        # Border subtraction is only a profile-location diagnostic, not a dark calibration.
        background = np.median(np.concatenate((image[0], image[-1], image[:, 0], image[:, -1])))
        weights = np.maximum(image-background, 0)
        cx = int(round((weights*np.arange(image.shape[1])[None]).sum()/weights.sum()))
        cy = int(round((weights*np.arange(image.shape[0])[:, None]).sum()/weights.sum()))
        axes[index, 2].plot(np.arange(xmin, xmax+1)-xmin-cx, image[cy], label="X")
        axes[index, 2].plot(np.arange(ymin, ymax+1)-ymin-cy, image[:, cx], label="Y")
        axes[index, 2].set_title("Profiles through background-subtracted centroid")
        axes[index, 2].legend()
    fig.tight_layout()
    fig.savefig(out / "endpoints.png", dpi=140)
    plt.close(fig)
    payload = dict(run=str(run.resolve()), rows=summary,
        qualification="Native square masks rebuilt from frozen geometry; sparse edge rounding can differ; EE denominator is recorded window; no measured dark subtraction")
    (out / "endpoints.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)
