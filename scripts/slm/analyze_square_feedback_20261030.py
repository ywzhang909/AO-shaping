"""Audit the 2026-10-30 square-shaping feedback captures.

Remove after the accompanying experiment report is final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def _frame_mean(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        raw = np.asarray(data["raw_frames"], dtype=np.float64)
    return raw.mean(axis=0), raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = json.loads((root / "manifest.json").read_text())
    selected = ["feedback_reference", "feedback_01", "feedback_04",
                "feedback_08"]
    by_label = {row["label"]: row for row in records}

    yy, xx = np.mgrid[:1944, :2592]
    centre = np.array([1917.0, 943.0])
    dx = xx - centre[0]
    dy = yy - centre[1]
    box = (np.abs(dx) < 500.0) & (np.abs(dy) < 500.0)
    core = (np.abs(dx) < 480.0) & (np.abs(dy) < 480.0)
    inner = (np.abs(dx) < 430.0) & (np.abs(dy) < 430.0)
    edge = core & ~inner
    roi = (xx >= 1300) & (xx < 2540) & (yy >= 300) & (yy < 1588)
    zero = (xx - 1267.0) ** 2 + (yy - 943.0) ** 2 < 90.0 ** 2
    target_valid = core & ~zero

    rows = []
    frames: dict[str, np.ndarray] = {}
    raws: dict[str, np.ndarray] = {}
    for label in selected:
        record = by_label[label]
        frame, raw = _frame_mean(root / record["file"])
        frames[label] = frame
        raws[label] = raw
        bg = float(record["background"])
        signal = frame - bg
        vals = signal[target_valid]
        noise = np.sqrt(raw[:, target_valid].var(axis=0, ddof=1).mean())
        interior_means = signal[inner]
        edge_means = signal[edge]
        # Marginal profile over a 120 px central strip to avoid isolated stripes.
        hprofile = signal[893:993, 1417:2417].mean(axis=0)
        vprofile = signal[443:1443, 1867:1967].mean(axis=1)
        rows.append({
            "label": label,
            "file": record["file"],
            "exposure_ms": record["exposure_ms"],
            "background": bg,
            "temporal_noise": float(noise),
            "clipped_core_frames": int((raw[:, core] >= 0.98 * 4095.0).sum()),
            "target_sum": float(signal[target_valid].sum()),
            "core_mean": float(vals.mean()),
            "core_std": float(vals.std()),
            "cv": float(vals.std() / max(vals.mean(), 1e-9)),
            "q05": float(np.quantile(vals, 0.05)),
            "q25": float(np.quantile(vals, 0.25)),
            "q50": float(np.quantile(vals, 0.50)),
            "q75": float(np.quantile(vals, 0.75)),
            "q95": float(np.quantile(vals, 0.95)),
            "interior_mean": float(interior_means.mean()),
            "edge_mean": float(edge_means.mean()),
            "hprofile_cv": float(hprofile.std() / max(hprofile.mean(), 1e-9)),
            "vprofile_cv": float(vprofile.std() / max(vprofile.mean(), 1e-9)),
        })

    (root / "feedback_audit.json").write_text(json.dumps(rows, indent=2))

    vmax = float(np.quantile(
        np.concatenate([frames[label][target_valid]
                        for label in selected]), 0.99))
    fig, axes = plt.subplots(3, len(selected), figsize=(16, 12),
                             constrained_layout=True)
    for col, label in enumerate(selected):
        record = by_label[label]
        frame = frames[label]
        signal = np.maximum(frame - record["background"], 0.0)
        crop = signal[300:1588, 1300:2540]
        image = axes[0, col].imshow(crop, origin="lower", cmap="inferno",
                                    vmin=0.0, vmax=vmax)
        axes[0, col].set_title(
            f"{label}\n35 ms, target contrast")
        plt.colorbar(image, ax=axes[0, col], fraction=0.046, pad=0.03)

        logcrop = np.log1p(crop)
        image = axes[1, col].imshow(
            logcrop, origin="lower", cmap="inferno",
            vmin=float(np.quantile(logcrop, 0.05)),
            vmax=float(np.quantile(logcrop, 0.995)))
        axes[1, col].set_title("log1p, includes zero order")
        plt.colorbar(image, ax=axes[1, col], fraction=0.046, pad=0.03)

        hprofile = signal[893:993, 1417:2417].mean(axis=0)
        vprofile = signal[443:1443, 1867:1967].mean(axis=1)
        axes[2, col].plot(np.arange(-500, 500), hprofile / max(hprofile.mean(), 1e-9),
                          label="horizontal")
        axes[2, col].plot(np.arange(-500, 500), vprofile / max(vprofile.mean(), 1e-9),
                          label="vertical", alpha=0.75)
        axes[2, col].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axes[2, col].set_ylim(0.0, 2.5)
        axes[2, col].set_xlabel("position inside 1000 px target (px)")
        axes[2, col].set_ylabel("normalized profile")
        axes[2, col].legend(loc="upper right")
        axes[2, col].grid(alpha=0.2)
    for axis in axes[:2].ravel():
        axis.set_xlabel("sensor x - 1300 (px)")
        axis.set_ylabel("sensor y - 300 (px)")
    fig.suptitle("1000 px square feedback audit: target set, not a success claim")
    fig.savefig(root / "feedback_audit.png", dpi=170)
    plt.close(fig)

    # A compact distribution comparison makes edge pile-up visible without
    # pretending that the clipped zero order permits a whole-sensor efficiency.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for label in selected:
        record = by_label[label]
        signal = np.maximum(frames[label] - record["background"], 0.0)
        vals = signal[target_valid]
        q = np.linspace(0.0, 1.0, 201)
        axes[0].plot(q, np.quantile(vals, q),
                     label=label, linewidth=1.8)
    axes[0].set_xlabel("target quantile")
    axes[0].set_ylabel("background-subtracted count")
    axes[0].set_title("Core intensity quantiles")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    labels = [row["label"] for row in rows]
    values = [row["cv"] for row in rows]
    axes[1].bar(labels, values, color=["#777777", "#e08a1e", "#d04e2c", "#8d2f4f"])
    axes[1].set_ylabel("CV inside core (zero excluded)")
    axes[1].set_title("Lower CV is not enough: inspect shape")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(root / "feedback_distributions.png", dpi=170)
    plt.close(fig)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
