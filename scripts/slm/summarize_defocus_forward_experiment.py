"""Render fixed-grid forward-model comparisons and exposure-response diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

#: CJK 字体 (报告图)。找不到就退回默认字体, 不报错。
CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")


def setup_cjk_font() -> None:
    """给 matplotlib 配中文字体。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = list(CJK_FONTS)
    matplotlib.rcParams["axes.unicode_minus"] = False


def fwhm_1d(profile):
    """一维剖面的半高全宽 (半高处跨过的采样点数)。"""
    p = np.asarray(profile, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    peak = float(p.max())
    if peak <= 0:
        return float("nan")
    above = np.flatnonzero(p >= peak / 2.0)
    if above.size == 0:
        return float("nan")
    return float(above[-1] - above[0] + 1)


def pick_representatives(rows, branches=("U", "C")):
    """每个分支挑一条**探针**记录 (排除 reference), 取相关性最高的那条。"""
    picked = []
    for branch in branches:
        candidates = [
            (index, row)
            for index, row in enumerate(rows)
            if row.get("branch") == branch and row.get("family") != "reference"
        ]
        if not candidates:
            continue
        index, row = max(
            candidates, key=lambda item: item[1].get("corr", float("-inf"))
        )
        picked.append((index, row))
    return picked


def render_spot_figure(
    captures, evaluation, model_name, out_path, *, branches=("U", "C")
):
    """上机整形实验的**测试数据可视化**: 下发的相位 + CCD 远场光斑 + 光斑截面。

    每个分支一行, 四列:

    1. **实际下发给 SLM 的面板总相位** (弧度) —— 送出去的那一张;
    2. **CCD 实测画面** (零级附近裁剪, counts) —— 回来的那一帧;
    3. **模型网格上的 |实测 − 预测|** (各自峰值归一后作差) —— 模型差在哪;
    4. **光斑截面**: x / y 两个方向的实测 vs 预测曲线 (峰值归一)。

    缺了这张图, 读者只能看"预测拼图"里的相关系数, 看不到**送出去的相位**
    与**回来的光斑**长什么样。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(evaluation).read_text(encoding="utf-8"))[model_name]["rows"]
    arrays = np.load(Path(evaluation).parent / f"prediction_{model_name}.npz")
    manifest = {
        item["file"]: item
        for item in json.loads(
            (Path(captures) / "manifest.json").read_text(encoding="utf-8")
        )
    }
    picked = pick_representatives(rows, branches)
    if not picked:
        print("[提示] 没有可画的代表性记录, 跳过光斑图")
        return None

    fig, axes = plt.subplots(len(picked), 4, figsize=(19, 5.2 * len(picked)))
    axes = np.atleast_2d(axes)
    for row_index, (index, record) in enumerate(picked):
        axis_row = axes[row_index]
        name = record["file"]
        sample = np.load(Path(captures) / name, allow_pickle=True)
        phase = np.asarray(sample["panel_phase"], dtype=np.float64)
        frame = np.asarray(sample["frame"], dtype=np.float64)
        meta = manifest.get(name, {})
        spot = meta.get("spot") or {}
        cx = int(round(float(spot.get("centroid_x", frame.shape[1] / 2))))
        cy = int(round(float(spot.get("centroid_y", frame.shape[0] / 2))))

        # 1) 下发的面板总相位
        phase_image = axis_row[0].imshow(phase, cmap="twilight", origin="upper")
        axis_row[0].set_title(
            f"{record['branch']} 分支 · 下发的面板总相位\n{name} ({record['label']}, {record['family']})",
            fontsize=9,
        )
        axis_row[0].set_xlabel("面板 x / 像元")
        axis_row[0].set_ylabel("面板 y / 像元")
        fig.colorbar(phase_image, ax=axis_row[0], label="rad")

        # 2) CCD 实测画面 (零级附近裁剪)
        half = 240
        y0, y1 = max(cy - half, 0), min(cy + half + 1, frame.shape[0])
        x0, x1 = max(cx - half, 0), min(cx + half + 1, frame.shape[1])
        patch = frame[y0:y1, x0:x1]
        frame_image = axis_row[1].imshow(
            patch,
            cmap="inferno",
            origin="upper",
            vmin=0.0,
            vmax=max(float(patch.max()), 1e-9),
        )
        axis_row[1].plot(cx - x0, cy - y0, "o", ms=9, mfc="none", mec="#2f6fb0")
        axis_row[1].set_title(
            f"CCD 实测远场光斑 (零级附近 {x1 - x0}×{y1 - y0} px)\n"
            f"峰值 {float(patch.max()):.1f} counts, 曝光 {meta.get('exposure_ms', 'n/a')} ms",
            fontsize=9,
        )
        axis_row[1].set_xlabel("CCD x / px")
        axis_row[1].set_ylabel("CCD y / px")
        fig.colorbar(frame_image, ax=axis_row[1], label="counts")

        # 3) 模型网格上的 |实测 − 预测|
        measured = np.asarray(arrays["measured"][index], dtype=np.float64)
        predicted = np.asarray(arrays["predicted"][index], dtype=np.float64)
        measured_n = measured / max(float(measured.max()), 1e-12)
        predicted_n = predicted / max(float(predicted.max()), 1e-12)
        diff = np.abs(measured_n - predicted_n)
        diff_image = axis_row[2].imshow(
            diff, cmap="inferno", origin="upper", vmin=0.0, vmax=1.0
        )
        axis_row[2].set_title(
            f"|实测 − 预测| (各自峰值归一)\n相关 r = {record.get('corr', float('nan')):.3f}, "
            f"CV 实测 {record.get('measured_cv', float('nan')):.3f} / "
            f"预测 {record.get('predicted_cv', float('nan')):.3f}",
            fontsize=9,
        )
        axis_row[2].set_xlabel("模型网格 x / 采样")
        axis_row[2].set_ylabel("模型网格 y / 采样")
        fig.colorbar(diff_image, ax=axis_row[2], label="|ΔI| / I_max")

        # 4) 光斑截面 (x / y 两个方向)
        axis = axis_row[3]
        row_i, col_j = np.unravel_index(int(measured.argmax()), measured.shape)
        for cut, style, label in (
            (measured_n[row_i, :], "-", "实测 (x)"),
            (predicted_n[row_i, :], "--", "预测 (x)"),
            (measured_n[:, col_j], "-", "实测 (y)"),
            (predicted_n[:, col_j], "--", "预测 (y)"),
        ):
            axis.plot(np.arange(len(cut)), cut, style, lw=1.4, label=label)
        axis.axhline(0.5, color="#7f8c8d", lw=0.8, ls=":")
        axis.set_title(
            f"光斑截面 (过实测峰值 第 {row_i} 行 / 第 {col_j} 列)\n"
            f"实测 FWHM x {fwhm_1d(measured_n[row_i, :]):.0f} / y {fwhm_1d(measured_n[:, col_j]):.0f} 采样, "
            f"预测 x {fwhm_1d(predicted_n[row_i, :]):.0f} / y {fwhm_1d(predicted_n[:, col_j]):.0f} 采样",
            fontsize=9,
        )
        axis.set_xlabel("模型网格采样")
        axis.set_ylabel("I / I_max")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def summarize_rows(rows):
    summary = {}
    for branch in ("U", "C"):
        a = [
            r
            for r in rows
            if r["branch"] == branch
            and r["family"] != "reference"
            and r.get("split", "test") == "test"
        ]
        corr = [r["corr"] for r in a]
        summary[branch] = dict(
            count=len(a),
            mean_corr=float(np.mean(corr)),
            median_corr=float(np.median(corr)),
            min_corr=float(np.min(corr)),
            family_median_corr={
                f: float(np.median([r["corr"] for r in a if r["family"] == f]))
                for f in ("zernike", "smooth", "bandlimited")
            },
        )
        for metric in ("cv", "ee_roi"):
            summary[branch][metric + "_mae"] = float(
                np.mean(
                    [abs(r["measured_" + metric] - r["predicted_" + metric]) for r in a]
                )
            )
            summary[branch]["measured_" + metric + "_mean"] = float(
                np.mean([r["measured_" + metric] for r in a])
            )
            summary[branch]["predicted_" + metric + "_mean"] = float(
                np.mean([r["predicted_" + metric] for r in a])
            )
        s = summary[branch]
        s["correlation_gate_passed"] = bool(
            s["median_corr"] >= 0.85
            and s["min_corr"] >= 0.65
            and min(s["family_median_corr"].values()) >= 0.8
        )
        s["cv_ee_gate_passed"] = bool(s["cv_mae"] <= 0.05 and s["ee_roi_mae"] <= 0.05)
        s["ready_for_inverse"] = False
        s["gate_note"] = (
            "Center/width errors and sampling convergence also require validation; correlation alone is insufficient."
        )
    return summary


def response_summary(directory):
    rows = json.loads((directory / "manifest.json").read_text())
    results = {}
    for branch in ("U", "C"):
        a = [r for r in rows if r["branch"] == branch]
        exposure = np.array([r["exposure_ms"] for r in a])
        peak = np.array([r["peak"] for r in a])
        counts = np.array([r["roi_sum"] for r in a])
        design = np.stack((exposure, np.ones(len(a))), axis=1)
        slope, offset = np.linalg.lstsq(design, counts, rcond=None)[0]
        residual = counts - design @ np.array([slope, offset])
        results[branch] = dict(
            exposures_ms=exposure.tolist(),
            peaks=peak.tolist(),
            peak_per_ms=(peak / exposure).tolist(),
            integrated_slope=float(slope),
            integrated_offset=float(offset),
            max_affine_relative_residual=float(np.max(np.abs(residual) / counts)),
            repeated_half_level_relative_difference=float(
                abs(counts[3] - counts[0]) / counts[0]
            ),
            stable=all(r["settled"]["settled"] for r in a),
        )
    return results


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", nargs="+", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument(
        "--captures",
        type=Path,
        help="采集目录 (含 manifest.json 与 sample_*.npz)。给了就额外画一张"
        "「下发相位 + CCD 光斑 + 光斑截面」的图",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts, figures = args.output / "artifacts", args.output / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary, all_rows = (
        {"development": {}, "response": response_summary(args.response)},
        [],
    )
    for path in args.fits:
        for name, model in json.loads(path.read_text())["models"].items():
            summary["development"][name] = summarize_rows(model["rows"])
            all_rows += [
                dict(model=name, dataset="development", **r) for r in model["rows"]
            ]
    if args.evaluation:
        summary["locked"] = {}
        for name, model in json.loads(args.evaluation.read_text()).items():
            summary["locked"][name] = summarize_rows(model["rows"])
            all_rows += [dict(model=name, dataset="locked", **r) for r in model["rows"]]
    (artifacts / "physical_forward_summary_20261022.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (artifacts / "physical_forward_rows_20261022.csv").open(
        "w", newline=""
    ) as stream:
        fields = sorted({k for row in all_rows for k in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    path = args.fits[-1]
    model_name, model = list(json.loads(path.read_text())["models"].items())[-1]
    arrays = np.load(path.parent / f"prediction_{model_name}.npz")
    selected = [i for i, r in enumerate(model["rows"]) if r["split"] == "test"]
    fig, axes = plt.subplots(4, 6, figsize=(16, 10))
    for j, i in enumerate(selected[:12]):
        row, col = (j // 6) * 2, j % 6
        for k, key in enumerate(("measured", "predicted")):
            axes[row + k, col].imshow(
                arrays[key][i], cmap="inferno", vmin=0, vmax=arrays["measured"][i].max()
            )
            record = model["rows"][i]
            axes[row + k, col].set_title(
                f"{record['branch']} {record['label']} {key}\nr={record['corr']:.3f}"
            )
            axes[row + k, col].axis("off")
    fig.tight_layout()
    fig.savefig(figures / "physical_forward_predictions_20261022.png", dpi=130)
    plt.close(fig)
    if args.captures and args.evaluation:
        setup_cjk_font()
        render_spot_figure(
            args.captures,
            args.evaluation,
            model_name,
            figures / "physical_forward_spot_profiles_20261022.png",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
