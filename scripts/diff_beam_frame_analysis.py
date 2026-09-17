"""Analyze a diff-beam hardware run frame by frame.

Reads ``config.json`` + ``frames/frame_*.npy`` + ``frames/frame_meta.jsonl``
from a run directory and produces: peak/sum/centroid/0.5x-peak footprint per
frame, a square-uniformity metric (CV inside the peak region), a side-by-side
PNG (all frames) and a ``frame_analysis.json`` summary.

Usage:
    python scripts/diff_beam_frame_analysis.py --run-dir data/diff_beam/bench/gs_square_<ts>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    frames_dir = run_dir / "frames"
    meta_path = frames_dir / "frame_meta.jsonl"

    meta = [
        json.loads(line)
        for line in meta_path.read_text().splitlines()
        if line.strip()
    ]
    frame_files = sorted(frames_dir.glob("frame_*.npy"))
    if len(frame_files) != len(meta):
        logger.warning(f"meta={len(meta)} frames={len(frame_files)} mismatch")
    n = min(len(frame_files), len(meta))

    rows = []
    for i in range(n):
        m = meta[i]
        f = np.load(frame_files[i]).astype("float64")
        peak = float(f.max())
        ssum = float(f.sum())
        mean = float(f.mean())
        bg10 = float(np.percentile(f, 10))

        # Largest 0.5x-peak connected region (footprint of the main lobe).
        mask = f >= 0.5 * peak
        labels, nreg = ndimage.label(mask)
        if nreg > 0:
            sizes = ndimage.sum(mask, labels, range(1, nreg + 1)).astype(int)
            big = int(np.argmax(sizes)) + 1
            bbox = ndimage.find_objects((labels == big).astype(np.intp))[0]
            (sy, sx) = ndimage.find_objects((labels == big).astype(np.intp))[0]
            fp_h = sy.stop - sy.start
            fp_w = sx.stop - sx.start
            fp_area = int(sizes[big - 1])
            region = f[sy, sx]
            # Square uniformity inside the footprint (lower CV = more uniform).
            cv = float(region.std() / region.mean()) if region.mean() > 0 else 1.0
        else:
            fp_h = fp_w = fp_area = 0
            cv = 1.0

        rows.append(
            {
                "frame": i + 1,
                "phase": m.get("phase"),
                "peak": round(peak, 1),
                "sum": round(ssum, 1),
                "mean": round(mean, 3),
                "bg10": round(bg10, 1),
                "footprint_h_px": fp_h,
                "footprint_w_px": fp_w,
                "footprint_area_px": fp_area,
                "uniformity_cv": round(cv, 4),
                # 0 级光斑位置: 优先用 argmax (spot), 回退旧强度质心 (centroid)。
                # 真机验证: 全帧强度质心被杂散光晕拖偏 150~450px (见
                # docs/diff_beam/README.md §2), 只有 argmax 是真实光斑。
                "centroid_y": round((m.get("spot") or m.get("centroid", [0, 0]))[0], 1),
                "centroid_x": round((m.get("spot") or m.get("centroid", [0, 0]))[1], 1),
            }
        )

    result = {
        "run_dir": str(run_dir),
        "config": {
            "algorithm": cfg.get("algorithm"),
            "target_shape": cfg.get("target_shape"),
            "target_size": cfg.get("target_size"),
            "target_px": cfg.get("target_px"),
            "target_brightness": cfg.get("target_brightness"),
            "brightness_info": cfg.get("brightness_info"),
            "exposure_ms": cfg.get("exposure_ms"),
            "iterations": cfg.get("iterations"),
            "frames_recorded": cfg.get("frames_recorded"),
            "final_metrics": cfg.get("final_metrics"),
        },
        "frames": rows,
    }
    (run_dir / "frame_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    return result


def plot(result: dict, run_dir: Path) -> None:
    frames_dir = run_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npy"))
    n = len(result["frames"])
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, r in enumerate(result["frames"]):
        f = np.load(frame_files[i]).astype("float64")
        ax = axes[i]
        im = ax.imshow(f, cmap="inferno")
        ax.set_title(
            f"#{i+1} {r['phase']} peak={r['peak']:.0f} "
            f"fp={r['footprint_w_px']}x{r['footprint_h_px']}",
            fontsize=9,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{run_dir.name} - frames", fontsize=12)
    fig.tight_layout()
    out = run_dir / "frames_overview.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    logger.info(f"Saved overview plot: {out}")


def plot_spot_vs_target(result: dict, run_dir: Path) -> None:
    """Side-by-side: original (flat_initial) CCD spot vs the square target.

    Uses the grid-space ``target_intensity.npy`` when present (brightness-driven
    mode) — the target is the square built from the *current* beam position +
    brightness, so this shows exactly what the algorithm was asked to produce.
    """
    frames_dir = run_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npy"))
    target_path = run_dir / "target_intensity.npy"
    if not frame_files:
        logger.warning("No frames found, skipping spot_vs_target plot")
        return
    if not target_path.exists():
        logger.warning("No target_intensity.npy, skipping spot_vs_target plot")
        return
    spot = np.load(frame_files[0]).astype("float64")
    target = np.load(target_path).astype("float64")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(spot, cmap="inferno")
    axes[0].set_title(
        f"original spot (flat_initial) peak={spot.max():.0f} "
        f"sum={spot.sum():.0f}",
        fontsize=10,
    )
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(target, cmap="inferno")
    axes[1].set_title(
        f"target square (grid {target.shape[0]}x{target.shape[1]})",
        fontsize=10,
    )
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(f"{run_dir.name} - original spot vs target", fontsize=12)
    fig.tight_layout()
    out = run_dir / "spot_vs_target.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    logger.info(f"Saved spot-vs-target plot: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="diff-beam run directory")
    ap.add_argument("--plot", action="store_true", help="render frames_overview.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir() or not (run_dir / "config.json").exists():
        raise SystemExit(f"Not a diff-beam run dir: {run_dir}")

    result = analyze_run(run_dir)
    if args.plot:
        plot(result, run_dir)
        plot_spot_vs_target(result, run_dir)

    print(f"\n=== {run_dir.name} ===")
    print(f"algorithm={result['config']['algorithm']} "
          f"iterations={result['config']['iterations']} "
          f"exposure_ms={result['config']['exposure_ms']}")
    bi = result["config"].get("brightness_info")
    if bi:
        print(f"brightness_mode: avg={bi.get('target_brightness')} "
              f"side={bi.get('side_cam_px')}px "
              f"centroid=({bi.get('centroid', [0,0])[0]:.0f}, "
              f"{bi.get('centroid', [0,0])[1]:.0f}) "
              f"total={bi.get('total_intensity')}")
    print(f"{'#':>2} {'phase':<14} {'peak':>7} {'sum':>9} "
          f"{'fpW':>4} {'fpH':>4} {'CV':>6} {'cy':>7} {'cx':>7}")
    for r in result["frames"]:
        print(f"{r['frame']:>2} {r['phase']:<14} {r['peak']:>7.1f} "
              f"{r['sum']:>9.0f} {r['footprint_w_px']:>4} "
              f"{r['footprint_h_px']:>4} {r['uniformity_cv']:>6.3f} "
              f"{r['centroid_y']:>7.1f} {r['centroid_x']:>7.1f}")
    print(f"analysis written to: {run_dir / 'frame_analysis.json'}")


if __name__ == "__main__":
    main()