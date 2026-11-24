"""离线分析已归档的暗场多曝光数据（不占用任何设备）。

对 ``data/ccd_configs/ccd/<serial>/darks/`` 下 12 档暗场 npz 做逐像素统计：
跨曝光暗电平 / 暗电流斜率、时间噪声地板（std 图中位数）、固定图形噪声
（mean 图空间 std）、位深混存审计。结果写 JSON + 英文标题图件。

这是 ``report/shape_trials/dark_frame_record_20261119.md`` §4.2 留白
「全幅阵列仍归档供逐帧噪声地板分析」的离线补充，不改动 manifest 与驱动代码
（位深隔离缺陷的修复会触发 §0.3 哈希重签连锁，另案处理）。

用法::

    .venv\\Scripts\\python.exe scripts/slm/analyze_dark_frame_exposures_20261124.py \\
        --serial FJB24112232 \\
        --data-dir data/ccd_configs/ccd/FJB24112232/darks \\
        --out-dir data/debug/dark_frame_analysis_20261124 \\
        --fig-dir report/shape_trials/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def load_entries(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读 manifest 并对每档校验 npz 完整性与登记值一致性。"""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = sorted(manifest["entries"], key=lambda e: e["exposure_ms"])
    for e in entries:
        path = data_dir / e["file"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest 登记的暗场文件缺失: {path}")
        with np.load(path) as z:
            if set(z.files) != {"mean", "std"}:
                raise ValueError(f"暗场 npz 键异常: {path} -> {z.files}")
            mean = z["mean"]
            std = z["std"]
        if mean.shape != tuple(e["shape"]) or std.shape != mean.shape:
            raise ValueError(f"暗场形状与 manifest 不一致: {path}")
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise ValueError(f"暗场含 NaN/Inf: {path}")
        # 登记值来自 np.mean(mean)/np.mean(std)，逐档复核
        if not np.isclose(float(mean.mean()), e["mean"], rtol=0, atol=1e-3):
            raise ValueError(f"manifest mean 与数组不符: {path}")
        if not np.isclose(float(std.mean()), e["std"], rtol=0, atol=1e-3):
            raise ValueError(f"manifest std 与数组不符: {path}")
    return manifest, entries


def analyse(entries: list[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    """逐档统计 + 分位深暗电流斜率 / 噪声地板 / FPN。"""
    per_exposure: list[dict[str, Any]] = []
    for e in entries:
        with np.load(data_dir / e["file"]) as z:
            mean = z["mean"]
            std = z["std"]
        med = float(np.median(mean))
        per_exposure.append(
            {
                "file": e["file"],
                "exposure_ms": e["exposure_ms"],
                "bit_depth": e["bit_depth"],
                "full_scale": e["full_scale"],
                "frames": e["frames"],
                "mean_counts": float(mean.mean()),
                "median_counts": med,
                "peak_counts": float(mean.max()),
                "min_counts": float(mean.min()),
                "fpn_spatial_std_counts": float(np.std(mean - med)),
                "read_noise_median_counts": float(np.median(std)),
                "read_noise_mean_counts": float(std.mean()),
                "read_noise_p99_counts": float(np.percentile(std, 99)),
                "read_noise_max_counts": float(std.max()),
            }
        )

    by_depth: dict[str, Any] = {}
    for bd in sorted({e["bit_depth"] for e in entries}):
        sub = [a for a in per_exposure if a["bit_depth"] == bd]
        x = np.array([a["exposure_ms"] for a in sub])
        y = np.array([a["mean_counts"] for a in sub])
        slope, intercept = np.polyfit(x, y, 1)
        short = [a for a in sub if a["exposure_ms"] <= 0.12]
        by_depth[str(bd)] = {
            "n_entries": len(sub),
            "exposures_ms": [float(v) for v in x],
            "bias_counts": float(intercept),
            "bias_over_full_scale": float(intercept / sub[0]["full_scale"]),
            "dark_current_slope_counts_per_ms": float(slope),
            "short_exposure_noise_floor_median_counts": float(
                np.median([a["read_noise_median_counts"] for a in short])
            ),
        }

    return {"per_exposure": per_exposure, "by_bit_depth": by_depth}


def make_figures(result: dict[str, Any], data_dir: Path, fig_dir: Path) -> list[Path]:
    """两张英文标题图：跨曝光统计 + 0.08 ms mean/std 空间图。"""
    fig_dir.mkdir(parents=True, exist_ok=True)
    per = result["per_exposure"]
    made: list[Path] = []

    # 图 1：暗电平 / 噪声地板 vs 曝光，按位深分色
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for bd, color, marker in ((12, "tab:blue", "o"), (8, "tab:orange", "s")):
        sub = [a for a in per if a["bit_depth"] == bd]
        x = [a["exposure_ms"] for a in sub]
        axes[0].plot(x, [a["mean_counts"] for a in sub], marker, color=color, label=f"{bd}-bit")
        axes[1].plot(
            x,
            [a["read_noise_median_counts"] for a in sub],
            marker,
            color=color,
            label=f"{bd}-bit",
        )
    for ax, ylab in ((axes[0], "dark level (counts)"), (axes[1], "read noise, std-map median (counts)")):
        ax.set_xscale("log")
        ax.set_xlabel("exposure (ms)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Archived dark frames: level / noise floor vs exposure", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = fig_dir / "dark_frame_level_noise_vs_exposure.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    made.append(p1)

    # 图 2：0.08 ms 代表档 mean/std 空间图（FPN 与时间噪声的空间分布）
    rep = next(a for a in per if a["bit_depth"] == 12 and a["exposure_ms"] == 0.08)
    with np.load(data_dir / rep["file"]) as z:
        mean, std = z["mean"], z["std"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    im0 = axes[0].imshow(mean, cmap="gray")
    im0.set_clim(np.percentile(mean, 1), np.percentile(mean, 99))
    axes[0].set_title(f"mean map (0.08 ms, 12-bit)\nFPN spatial std={rep['fpn_spatial_std_counts']:.3f} cnt")
    im1 = axes[1].imshow(std, cmap="magma")
    im1.set_clim(0, np.percentile(std, 99))
    axes[1].set_title(f"temporal std map (16 frames)\nmedian={rep['read_noise_median_counts']:.3f} cnt")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Dark-frame spatial structure (laser off)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = fig_dir / "dark_frame_spatial_0p08ms.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    made.append(p2)
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="FJB24112232", help="CCD 序列号")
    parser.add_argument("--data-dir", type=Path, required=True, help="darks 归档目录")
    parser.add_argument("--out-dir", type=Path, required=True, help="JSON 输出目录")
    parser.add_argument("--fig-dir", type=Path, required=True, help="图件输出目录")
    args = parser.parse_args()

    _, entries = load_entries(args.data_dir)
    result = analyse(entries, args.data_dir)
    result["schema"] = "ao_shaping.dark_frame_analysis/v1"
    result["serial"] = args.serial
    result["manifest_entries"] = len(entries)
    result["bit_depth_mixed"] = sorted({e["bit_depth"] for e in entries})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figs = make_figures(result, args.data_dir, args.fig_dir)

    logger.info("暗场分析完成: {} 档, 位深 {}", len(entries), result["bit_depth_mixed"])
    for bd, s in result["by_bit_depth"].items():
        logger.info(
            "{}-bit: bias={:.3f} cnt, dark slope={:.4f} cnt/ms, noise floor={:.4f} cnt",
            bd,
            s["bias_counts"],
            s["dark_current_slope_counts_per_ms"],
            s["short_exposure_noise_floor_median_counts"],
        )
    for p in figs:
        logger.info("图件: {}", p)
    logger.info("JSON: {}", args.out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
