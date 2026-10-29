"""Audit a recorded no-GS capture/fit without opening devices or changing data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.tools.slm.transfer.fit import load_dataset
from ao_shaping.utils.image.physical_model_metrics import compute_forward_metrics


def analyze(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    capture, fit = root / "capture", root / "fit"
    manifest = json.loads((capture / "manifest.json").read_text())
    result = json.loads((fit / "result.json").read_text())
    config = json.loads((capture / "config.json").read_text())
    # The square prediction window has 2-pixel sampling in fit.load_dataset.
    with np.load(fit / "iteration_0000.npz") as data:
        initial = data["prediction"].copy()
    span = initial.shape[-1] - 1
    rows, phases, images, xy, matrix, ramps, train = load_dataset(capture, config["grid"], span)
    with np.load(fit / f"iteration_{result['iteration']:04d}.npz") as data:
        best = data["prediction"].copy()
    center = xy[xy.shape[0]//2, xy.shape[1]//2]
    core = (abs(xy[..., 0]-center[0]) < 100) & (abs(xy[..., 1]-center[1]) < 100)
    entries = []
    for index, row in enumerate(rows):
        item = {"label": row["label"], "split": row["split"], "exposure_ms": row["exposure_ms"]}
        for stage, prediction in (("initial", initial), ("best", best)):
            item[stage] = {}
            for mask_name, mask in (("window", np.ones_like(core)), ("central200", core)):
                item[stage][mask_name] = compute_forward_metrics(
                    prediction[index], images[index]*row["exposure_ms"], mask,
                    exposure_ms=row["exposure_ms"], full_scale=255, camera_xy=xy)
        entries.append(item)
    summary = {}
    for split in ("train", "validation"):
        selected = [e for e in entries if e["split"] == split]
        summary[split] = {stage: {mask: {key: float(np.mean([
            e[stage][mask][key] for e in selected])) for key in (
                "pearson_ncc", "relative_l2_raw", "cv_abs_error", "centroid_error_px")}
            for mask in ("window", "central200")} for stage in ("initial", "best")}
    references = []
    reference_images = []
    for row in (manifest[0], manifest[-1]):
        with np.load(capture / row["file"]) as data:
            raw = data["raw_frames"].astype(float)
        rate = raw.mean(0)/row["exposure_ms"]
        reference_images.append(rate)
        references.append(dict(label=row["label"], exposure_ms=row["exposure_ms"],
            spot_xy=row["spot_xy"], peak_rate=float(rate.max()), sum_rate=float(rate.sum()),
            mean_frame_std_counts=float(raw.std(0).mean())))
    payload = dict(root=str(root.resolve()), ramp_matrix=matrix.tolist(), ramp_samples=ramps,
        best_iteration=result["iteration"], summary=summary, references=references, rows=entries,
        qualification="Sparse CCD step=2; central200 is a diagnostic mask, not a shaped target; per-frame affine is diagnostic only")
    (out / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selected = [i for i, r in enumerate(rows) if r["split"] == "validation"][:3]
    fig, axes = plt.subplots(len(selected), 4, figsize=(15, 3.8*len(selected)), squeeze=False)
    extent = (xy[0, 0, 0], xy[0, -1, 0], xy[-1, 0, 1], xy[0, 0, 1])
    for line, index in enumerate(selected):
        axes[line, 0].imshow(np.mod(phases[index], 2*np.pi), cmap="twilight", vmin=0, vmax=2*np.pi)
        axes[line, 0].set_title(rows[index]["label"]+" / phase")
        vmax = float(max(images[index].max(), best[index].max()))
        for col, im, label in ((1, images[index], "CCD counts/ms"), (2, best[index], "fitted prediction")):
            axes[line, col].imshow(im, cmap="inferno", vmin=0, vmax=vmax, extent=extent)
            axes[line, col].set_title(label)
        weight = np.maximum(images[index], 0)
        cy = int(round((weight*np.indices(weight.shape)[0]).sum()/weight.sum()))
        axes[line, 3].plot(xy[cy, :, 0], images[index, cy], label="CCD")
        axes[line, 3].plot(xy[cy, :, 0], best[index, cy], label="prediction")
        axes[line, 3].set_title("X profile through CCD centroid")
        axes[line, 3].legend()
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, image, row in zip(axes, reference_images, references):
        axis.imshow(image, cmap="inferno", vmin=0, vmax=max(im.max() for im in reference_images))
        axis.set_title(f"{row['label']} / {row['exposure_ms']:.3f} ms / counts/ms")
        axis.set_xlim(center[0]-360, center[0]+360)
        axis.set_ylim(center[1]+360, center[1]-360)
    fig.tight_layout()
    fig.savefig(out / "references.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"summary": summary, "references": references}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.out)
