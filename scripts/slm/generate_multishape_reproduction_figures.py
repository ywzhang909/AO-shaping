"""Render traceable three-shape figures from saved physical experiments only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ("square", "rectangle", "triangle")
COLORS = ("#0072B2", "#E69F00", "#009E73")


def generate(run, fit, calibrated, out, figures, figure_tag):
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    history = pickle.loads((fit / "iterations.pkl").read_bytes())
    selected = json.loads((fit / "result.json").read_text())
    before, after = history[0]["parameters"], selected["parameters"]
    final = json.loads(calibrated.read_text())
    rows = {r["label"]: r for r in manifest}
    numeric = []

    def save(fig, name):
        fig.savefig(figures / (name + "_" + figure_tag + ".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    boxes = [(1.5, 5.7, "SLM phase phi\n128 x 128 raw rad"),
             (5., 5.7, "Gaussian input A\ncenter + width (4)"),
             (9., 5.7, "Static phase\ndefocus (1) + Z5..Z11 (7)"),
             (5., 3.8, "Complex field\nA exp[i(phi + static)]"),
             (9., 3.8, "FFT 1024 x 1024\nIntensity |U|^2"),
             (12., 3.8, "Camera mapping\nfixed ramp M + origin (2)"),
             (12., 1.5, "Pixel integration 3 x 3\nEffective blur, gain, background"),
             (7.5, 1.5, "Frozen forward F(phi)\nCCD anchor + local change"),
             (2., 1.5, "Core CV + energy + shape\nAutograd -> Adam -> projection")]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=.6", facecolor="#E8F3FA", edgecolor=COLORS[0]))
    for start, end in [((1.5, 5.2), (4., 4.1)), ((5., 5.2), (5., 4.3)),
                       ((9., 5.2), (6., 4.1)), ((6.2, 3.8), (7.6, 3.8)),
                       ((10.2, 3.8), (10.8, 3.8)), ((12., 3.25), (12., 2.1)),
                       ((10.4, 1.5), (9.1, 1.5)), ((5.7, 1.5), (3.8, 1.5)),
                       ((1.5, 2.1), (1.5, 5.1))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5))
    ax.set_title("Compact differentiable physical model: 14 Adam-fitted scalars\nStep A fits physical parameters; Step B freezes them and optimizes phase", fontsize=14)
    save(fig, "model_structure")

    groups = [([f"Z{j}" for j in range(5, 12)],
               [[p["residual_noll_radians"][str(j)] for j in range(5, 12)] for p in (before, after, final)], "Static aberration / rad"),
              (["cx", "cy", "wx", "wy"], [p["amplitude_center"] + p["amplitude_width"] for p in (before, after, final)], "Input amplitude / SLM px"),
              (["origin x", "origin y"], [p["origin"] for p in (before, after, final)], "Optical origin / CCD px"),
              (["defocus"], [[p["defocus_diopter"]] for p in (before, after, final)], "Defocus / diopter"),
              (["gain"], [[p["gain"]] for p in (before, after, final)], "Gain / model scale"),
              (["background"], [[p["background"]] for p in (before, after, final)], "Additive background / counts per ms"),
              (["sigma"], [[p["sigma_ccd_px"]] for p in (before, after, final)], "Effective blur sigma / CCD px")]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    stages = ["Before exact-grid fit", f"Selected fit (iter {selected['iteration']})", "After blur/radiometry calibration"]
    for ax, (labels, values, title) in zip(axes.flat, groups):
        x = np.arange(len(labels))
        for k, (stage, vals) in enumerate(zip(stages, values)):
            ax.bar(x + (k-1)*.25, vals, .24, label=stage, color=COLORS[k])
            numeric.extend(dict(group=title, parameter=label, stage=stage, value=float(v)) for label, v in zip(labels, vals))
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.axhline(0, color="gray", lw=.5)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[-1, -1].set_axis_off()
    fig.suptitle("Physical parameters, not neural-network layer weights\nBefore is an existing checkpoint; this is not a cold-start experiment")
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "parameter_bars")
    with (out / "parameter_values.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "parameter", "stage", "value"])
        writer.writeheader()
        writer.writerows(numeric)

    with np.load(fit / "inputs.npz") as payload:
        train = payload["train"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    iterations = [r["iteration"] for r in history]
    axes[0, 0].plot(iterations, [r["loss"] for r in history], color=COLORS[0])
    axes[0, 0].set_title("Training objective (selection uses training only)")
    for mask, label, color in ((train, "train", COLORS[0]), (~train, "validation", COLORS[1])):
        for ax, key in ((axes[0, 1], "correlation"), (axes[1, 0], "cv_abs_error")):
            ax.plot(iterations, [np.mean([m[key] for m, use in zip(r["metrics"], mask) if use]) for r in history], label=label, color=color)
    axes[0, 1].set_title("Mean image correlation")
    axes[1, 0].set_title("Whole-window CV MAE (not target-core CV)")
    for j in range(5, 12):
        axes[1, 1].plot(iterations, [r["parameters"]["residual_noll_radians"][str(j)] for r in history], label=f"Z{j}")
    axes[1, 1].set_title("Static aberration trajectories / rad")
    for ax in axes.flat:
        ax.set_xlabel("Fit iteration")
        ax.grid(alpha=.2)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fit_trajectories")

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    metric_rows = []
    for si, shape in enumerate(SHAPES):
        labels = [shape+"_seed"] + [f"{shape}_{b}_B1" for b in range(4)] + [shape+"_final"]
        captures = []
        for label in labels:
            row = rows[label]
            with np.load(run / row["file"]) as payload:
                xy = payload["camera_xy"]
                frame = payload["raw_frames"].mean(0)
                image = frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                captures.append((image, payload["target"], payload["core"]))
            metric_rows.append(dict(shape=shape, label=label, **row["metrics"], exposure_ms=row["exposure_ms"], timestamp_utc=row["timestamp_utc"]))
        vmax = max(r[0].max() for r in captures)
        yy, xx = np.where(captures[0][1] > .5)
        for col, ((image, target, core), label) in enumerate(zip(captures, labels)):
            ax = axes[si, col]
            im = ax.imshow(image, cmap="magma", vmin=0, vmax=vmax)
            ax.contour(target, [.5], colors=["cyan"], linewidths=.7)
            ax.contour(core, [.5], colors=["lime"], linewidths=.5)
            ax.set_xlim(xx.min()-10, xx.max()+10)
            ax.set_ylim(yy.max()+10, yy.min()-10)
            title = "Start" if col == 0 else "Final" if col == 5 else f"Block {col} B1"
            ax.set_title(f"{shape}: {title}\nCV {rows[label]['metrics']['cv']:.3f}", fontsize=10)
            ax.set_axis_off()
        fig.colorbar(im, ax=list(axes[si]), shrink=.65, label="CCD counts (shared per row)")
    fig.suptitle("Measured-anchor run: fixed exposure and fixed masks per shape\nCyan: target; green: eroded core. Start replays an earlier accepted phase, not the original GS seed.")
    save(fig, "ccd_block_evolution")

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for si, shape in enumerate(SHAPES):
        seed, end = rows[shape+"_seed"], rows[shape+"_final"]
        for col, row in enumerate((seed, end)):
            with np.load(run / row["file"]) as payload:
                phase = payload["model_phase"]
                axes[si, col].imshow(np.mod(phase, 2*np.pi), cmap="twilight", vmin=0, vmax=2*np.pi)
                axes[si, col].set_ylim(79.5, -.5)
            axes[si, col].set_title(f"{shape}: {'start' if col == 0 else 'final'} phase / rad")
        with np.load(run / end["file"]) as payload:
            xy = payload["camera_xy"]
            frame = payload["raw_frames"].mean(0)
            image = frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]
            axes[si, 2].imshow(image, cmap="magma", extent=(xy[..., 0].min(), xy[..., 0].max(), xy[..., 1].max(), xy[..., 1].min()))
            axes[si, 2].plot(*json.loads((run / "config.json").read_text())["zero_xy"], "c+", ms=12)
        axes[si, 2].set_title("Final 441px window; + zero-order reference")
    fig.tight_layout()
    save(fig, "phase_and_zero_order")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for si, shape in enumerate(SHAPES):
        subset = [r for r in manifest if r["label"].startswith(shape+"_")]
        origin = subset[0]["monotonic_ns"]
        for ax, key in zip(axes[si], ("cv", "ee_roi")):
            for suffix, color in (("A1", COLORS[0]), ("B1", COLORS[2]), ("B2", COLORS[2]), ("A2", COLORS[1]), ("seed", "black"), ("final", "black")):
                group = [r for r in subset if r["label"].endswith("_"+suffix)]
                ax.scatter([(r["monotonic_ns"]-origin)/1e9 for r in group], [r["metrics"][key] for r in group], label=suffix, color=color, marker="x" if suffix == "B2" else "o", s=30)
            ax.set_title(f"{shape}: {key}")
            ax.set_xlabel("Host elapsed time / s")
            ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, "paired_metrics")

    # Export original GS seeds separately: exposure differs from the anchor run.
    config = json.loads((run / "config.json").read_text())
    if config.get("seed_run"):
        source = Path(config["seed_run"])
        earlier = json.loads((source / "manifest.json").read_text())
        fig, axes = plt.subplots(3, 2, figsize=(9, 11))
        for si, shape in enumerate(SHAPES):
            for col, suffix in enumerate(("seed", "final")):
                row = next(r for r in earlier if r["label"] == f"{shape}_{suffix}")
                with np.load(source / row["file"]) as payload:
                    xy = payload["camera_xy"]
                    image = payload["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                    target = payload["target"]
                axes[si, col].imshow(image, cmap="magma", vmin=0, vmax=max(next(r for r in earlier if r["label"] == f"{shape}_{s}")["metrics"]["target_peak"] for s in ("seed", "final")))
                axes[si, col].contour(target, [.5], colors=["cyan"], linewidths=.7)
                axes[si, col].set_title(f"{shape}: {'original GS seed' if col == 0 else 'model-mode accepted'}\nCV={row['metrics']['cv']:.3f}; exposure={row['exposure_ms']:.3f} ms")
                axes[si, col].set_axis_off()
        fig.tight_layout()
        save(fig, "original_gs_stage")
    sources = [run / "manifest.json", run / "summary.json", fit / "iterations.pkl", fit / "result.json", calibrated, Path(__file__)]
    audit = dict(sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                 metrics=metric_rows, selected_fit_iteration=selected["iteration"],
                 definition="Saved raw counts; core erosion 2 samples; EE denominator fixed 441px window")
    (out / "figure_data.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "data/debug/validate_transfer_shapes/anchored_20261023_2002")
    p.add_argument("--fit", type=Path, default=ROOT / "data/debug/fit_transfer_multishape/exact_grid_20261023_1935")
    p.add_argument("--calibrated", type=Path, default=ROOT / "data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json")
    p.add_argument("--out", type=Path, default=ROOT / "report/shape_trials/multishape_capability/artifacts/reproduction_20261023")
    p.add_argument("--figures", type=Path, default=ROOT / "report/shape_trials/multishape_capability/figures")
    p.add_argument("--figure-tag", default="20261023")
    a = p.parse_args()
    generate(a.run, a.fit, a.calibrated, a.out, a.figures, a.figure_tag)


if __name__ == "__main__":
    main()
