"""Inspect saturation and render the 1000 px band-limit probe.

Reads the archived native frames only. This script never imports a camera or
SLM driver and can be rerun after the device session has been released.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    ROOT
    / "data/debug/slm_transfer/multishape_nogs_20261030"
    / "square1000_bandlimit_live_20261030_v2"
)
FULL_SCALE = 4095
TARGET_CENTER = (1917.0, 943.0)
ZERO_CENTER = (1267.0, 943.0)


def target_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    x, y = xx.astype(float), yy.astype(float)
    return {
        "box": (abs(x - TARGET_CENTER[0]) < 500) & (abs(y - TARGET_CENTER[1]) < 500),
        "roi": (x >= 1300) & (x < 2540) & (y >= 300) & (y < 1588),
        "zero": (x - ZERO_CENTER[0]) ** 2 + (y - ZERO_CENTER[1]) ** 2 < 120.0**2,
    }


def components(mask: np.ndarray) -> list[dict[str, float | int]]:
    labels, count = ndimage.label(mask)
    result = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        result.append(
            {
                "pixels": int(len(xx)),
                "x_min": int(xx.min()),
                "x_max": int(xx.max()),
                "y_min": int(yy.min()),
                "y_max": int(yy.max()),
                "x_centroid": float(xx.mean()),
                "y_centroid": float(yy.mean()),
            }
        )
    return sorted(result, key=lambda item: int(item["pixels"]), reverse=True)


def render(run: Path, manifest: list[dict], output: Path) -> None:
    labels = ("A", "B_blur2", "C_blur4", "F_blur4_full", "G_blur4_alpha125")
    selected = {}
    for label in labels:
        matches = [row for row in manifest if row["label"] == label]
        if matches:
            selected[label] = matches[-1]
    columns = len(selected)
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(4.2 * columns, 8.2),
        constrained_layout=True,
    )
    if columns == 1:
        axes = axes.reshape(2, 1)
    crop = (slice(300, 1588), slice(1300, 2540))
    for column, (label, row) in enumerate(selected.items()):
        with np.load(run / row["file"]) as saved:
            frame = saved["raw_frames"].mean(0).astype(float)
        background = float(np.median(frame[300:1588, 1300:2540]))
        signal = np.maximum(frame - background, 0.0)
        target = signal[crop]
        scale = max(float(np.percentile(target, 99.9)), 1.0)
        axes[0, column].imshow(
            target,
            origin="lower",
            extent=(1300, 2540, 300, 1588),
            cmap="inferno",
            vmax=scale,
        )
        axes[0, column].add_patch(
            plt.Rectangle(
                (TARGET_CENTER[0] - 500, TARGET_CENTER[1] - 500),
                1000,
                1000,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )
        axes[0, column].set_title(f"{label} | target crop")
        axes[0, column].set_aspect("equal")
        axes[1, column].imshow(
            np.log1p(target / max(float(np.median(target)), 1e-9)),
            origin="lower",
            extent=(1300, 2540, 300, 1588),
            cmap="viridis",
        )
        axes[1, column].set_title("log contrast")
        axes[1, column].set_aspect("equal")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for row in manifest:
        with np.load(args.run / row["file"]) as saved:
            raw = saved["raw_frames"]
        frame = raw.mean(0).astype(float)
        masks = target_masks(frame.shape)
        saturated = raw >= 0.98 * FULL_SCALE
        any_saturated = saturated.any(0)
        background = float(np.median(frame[masks["roi"] & ~masks["box"] & ~masks["zero"]]))
        signal = np.maximum(frame - background, 0.0)
        result = {
            "label": row["label"],
            "file": row["file"],
            "exposure_ms": row["exposure_ms"],
            "background_counts": background,
            "mean_frame_max": float(frame.max()),
            "peak_frame_saturated_pixels": int(saturated.sum(1).max()),
            "saturated_pixels_union": int(any_saturated.sum()),
            "target_saturated_pixels_union": int((any_saturated & masks["box"]).sum()),
            "zero_order_saturated_pixels_union": int((any_saturated & masks["zero"]).sum()),
            "outside_target_saturated_pixels_union": int(
                (any_saturated & ~masks["box"]).sum()
            ),
            "target_peak_counts": float(frame[masks["box"]].max()),
            "target_signal_sum_counts": float(signal[masks["box"]].sum()),
            "saturation_components": components(any_saturated),
        }
        results.append(result)
    (out / "saturation_analysis.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    render(args.run, manifest, out / "bandlimit_comparison.png")
    for result in results:
        print(
            result["label"],
            "sat_union",
            result["saturated_pixels_union"],
            "target",
            result["target_saturated_pixels_union"],
            "zero",
            result["zero_order_saturated_pixels_union"],
            "peak",
            round(result["target_peak_counts"], 1),
        )
        for component in result["saturation_components"][:4]:
            print(" ", component)
    print(out / "saturation_analysis.json")
    print(out / "bandlimit_comparison.png")


if __name__ == "__main__":
    main()
