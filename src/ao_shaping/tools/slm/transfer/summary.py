"""Summarize paired hardware runs without opening any device."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np



def summarize(run: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters,
    )
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics

    config = json.loads((run / "config.json").read_text())
    records = json.loads((run / "manifest.json").read_text())
    from .artifacts import resolve_recorded_path
    checkpoint = resolve_recorded_path(config["checkpoint"], relative_to=run)
    params = load_frozen_parameters(checkpoint)
    shapes = config["shapes"]
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(shapes), 3, figsize=(13, 3.7 * len(shapes)), squeeze=False)
    zoom, zoom_axes = plt.subplots(len(shapes), 2, figsize=(10, 3.4 * len(shapes)), squeeze=False)
    results = []
    for index, shape in enumerate(shapes):
        item = dict(shape=shape)
        selected = [next(r for r in records if r["label"] == f"{shape}_{label}") for label in ("seed", "final")]
        for col, row in enumerate(selected):
            with np.load(run / row["file"]) as payload:
                xy, target = payload["camera_xy"], payload["target"]
                image = payload["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                phase = payload["model_phase"]
            forward = FrozenDefocusForward(params, xy, device="cpu")
            prediction = forward.predict(phase).detach().cpu().numpy()[0]
            objective = ShapeObjective(target)
            model_metrics = image_metrics(prediction, objective)
            corr = float(np.corrcoef(prediction.ravel(), image.ravel())[0, 1])
            label = ("seed", "final")[col]
            item[label] = dict(measured=row["metrics"], predicted=model_metrics,
                               correlation=corr, exposure_ms=row["exposure_ms"], loss=row["loss"])
            ax = axes[index, col]
            ax.imshow(image, origin="upper", cmap="magma", vmin=0, vmax=max(r["metrics"]["target_peak"] for r in selected))
            ax.contour(target, levels=[.5], colors=["cyan"], linewidths=.7)
            ax.set_title(f"{shape} CCD {label}: CV={row['metrics']['cv']:.3f}, EE={row['metrics']['ee_roi']:.3f}")
            ax.set_axis_off()
            zax = zoom_axes[index, col]
            zim = zax.imshow(image, origin="upper", cmap="magma", vmin=0,
                            vmax=max(r["metrics"]["target_peak"] for r in selected))
            zax.contour(target, levels=[.5], colors=["cyan"], linewidths=.8)
            zax.contour(objective.core.astype(float), levels=[.5], colors=["lime"], linewidths=.6)
            yy, xx = np.where(target > .5)
            zax.set_xlim(xx.min()-8, xx.max()+8)
            zax.set_ylim(yy.max()+8, yy.min()-8)
            zax.set_title(f"{shape} {label}: CV={row['metrics']['cv']:.3f}, EE={row['metrics']['ee_roi']:.3f}\nExposure {row['exposure_ms']:.3f} ms")
            zax.set_axis_off()
        zoom.colorbar(zim, ax=list(zoom_axes[index]), shrink=.6, label="CCD counts")
        ax = axes[index, 2]
        ax.imshow(prediction, origin="upper", cmap="magma", vmin=0)
        ax.contour(target, levels=[.5], colors=["cyan"], linewidths=.7)
        ax.set_title(f"Model final: CV={model_metrics['cv']:.3f}, corr={corr:.3f}")
        ax.set_axis_off()
        direction_path = run / f"{shape}_direction.json"
        if direction_path.exists():
            item["direction"] = json.loads(direction_path.read_text())
        results.append(item)
    fig.suptitle("Fixed target masks; CCD seed/final share intensity scale per row; model scale is independent", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / "ccd_model_comparison.png", dpi=160)
    plt.close(fig)
    zoom.suptitle("Measured CCD target details: cyan target, green core; shared scale per row", fontsize=11)
    zoom.savefig(out / "ccd_target_zoom.png", dpi=160, bbox_inches="tight")
    plt.close(zoom)
    metadata = dict(run=str(run), checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    manifest_sha256=hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
                    metric_scope="core CV (2 sample erosion), EE within fixed 441px window; raw counts, no offset subtraction",
                    shapes=results)
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for row in results:
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.out)


if __name__ == "__main__":
    main()
