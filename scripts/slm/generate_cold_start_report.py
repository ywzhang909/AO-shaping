"""Render the recorded 2026-10-26 cold-start experiment without devices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.tools.slm.transfer.progress import fit_trace, measurement_trace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/shape_trials/cold_start/cold_start_20261026.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/debug/slm_transfer/cold_start_20261026_1004")
    args = parser.parse_args()
    figures = OUT.parent / "figures"
    artifacts = OUT.parent / "artifacts/20261026"
    figures.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    traces = {}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for name in ("fit", "fit_refined", "fit_position"):
        path = args.root / name
        if not (path / "iterations.pkl").exists():
            continue
        trace = traces[name] = fit_trace(path)
        for axis, key in zip(ax, ("loss", "validation_correlation", "validation_cv_abs_error")):
            axis.plot([r["iteration"] for r in trace], [r[key] for r in trace], label=name)
            axis.set_title(key)
            axis.set_xlabel("parameter update")
            axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "fit_metrics_20261026.png", dpi=160)
    plt.close(fig)
    runs = {}
    lines = ["# 无历史权重与无 GS 的冷启动整形实验（2026-10-26）", "",
             "## 数据与方法", "",
             f"原始实验目录：`{args.root.relative_to(ROOT).as_posix() if args.root.is_absolute() else args.root.as_posix()}`。",
             "未使用历史训练权重或历史目标相位。标称高斯振幅、零像差模型只根据本次相位探针拟合。",
             "平面/随机整形初相位另叠加当前几何的定位载波，不叠加拟合静态像差补偿。所有冷启动配置应记录 `gs_iterations=0`。",
             "训练与验证按相位分组；拟合阶段以训练损失选择检查点。最终以独立 CCD 配对测量验收，模型损失下降不等于物理改善。", "",
             "```mermaid", "flowchart TD", " A[Flat reference and random probes] --> B[Nominal physical model]",
             " B --> C[Fit and monitor independent phase validation]", " C --> D[Flat or random shaping phase plus carrier]",
             " D --> E[Differentiable short blocks]", " E --> F[CCD A B B A metrics]",
             " F -->|Accept| E", " F -->|Reject| G[Reduce step or refit model]", " G --> E",
             " F --> H[Save raw frames phases clocks and release devices]", "```", "",
             "## 模型指标轨迹", "", "![训练与验证指标](figures/fit_metrics_20261026.png)", "",
             "| 拟合阶段 | 初始→最终训练损失 | 初始→最终验证相关性 | 初始→最终验证 CV 绝对误差 |",
             "|---|---:|---:|---:|"]
    for name, trace in traces.items():
        a, b = trace[0], trace[-1]
        lines.append(f"| {name} | {a['loss']:.6f} → {b['loss']:.6f} | {a['validation_correlation']:.6f} → {b['validation_correlation']:.6f} | {a['validation_cv_abs_error']:.6f} → {b['validation_cv_abs_error']:.6f} |")
    for path in sorted(args.root.iterdir()):
        if not (path / "summary.json").exists() or not (path / "manifest.json").exists():
            continue
        data = runs[path.name] = measurement_trace(path)
        config = json.loads((path / "config.json").read_text())
        lines += ["", f"## {path.name}", "", f"模式 `{config['mode']}`；初相位 `{config.get('initial_phase')}`；GS 次数 `{config['gs_iterations']}`。", "",
                  "| 形状 | 初始→最终 CV | 初始→最终 EE_ROI | 初始→最终综合损失 | 接受块 |", "|---|---:|---:|---:|---:|"]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for shape in config["shapes"]:
            seed = next((r for r in data["measurements"] if r["label"] == shape+"_seed_replay"),
                        next(r for r in data["measurements"] if r["label"] == shape+"_seed"))
            final = next((r for r in data["measurements"] if r["label"] == shape+"_final"), None)
            blocks = [r for r in data["decisions"] if r["shape"] == shape]
            if final:
                lines.append(f"| {shape} | {seed['cv']:.6f} → {final['cv']:.6f} | {seed['ee_roi']:.6f} → {final['ee_roi']:.6f} | {seed['loss']:.6f} → {final['loss']:.6f} | {sum(b['accepted'] for b in blocks)}/{len(blocks)} |")
            rows = [r for r in data["measurements"] if r["label"].startswith(shape+"_") and (r["label"].endswith("_A1") or r["label"].endswith("_seed") or r["label"].endswith("_final"))]
            for axis, key in zip(ax, ("cv", "ee_roi", "loss")):
                axis.plot(np.arange(len(rows)), [r[key] for r in rows], "o-", label=shape)
                axis.set_title("Measured "+key)
                axis.set_xlabel("seed / accepted reference / final")
                axis.legend()
        fig.tight_layout()
        filename = f"{path.name}_metrics_20261026.png"
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
        lines += ["", f"![实测参考指标轨迹](figures/{filename})", ""]
        raw_rows = json.loads((path / "manifest.json").read_text())
        ccd, axes = plt.subplots(len(config["shapes"]), 2, figsize=(9, 4*len(config["shapes"])), squeeze=False)
        for index, shape in enumerate(config["shapes"]):
            seed_label = "seed_replay" if any(r["label"] == shape+"_seed_replay" for r in raw_rows) else "seed"
            pair = [next((r for r in raw_rows if r["label"] == shape+"_"+label), None) for label in (seed_label, "final")]
            if any(row is None for row in pair):
                continue
            for col, row in enumerate(pair):
                with np.load(path / row["file"]) as payload:
                    xy, target = payload["camera_xy"], payload["target"]
                    image = payload["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                axes[index, col].imshow(image, cmap="magma", vmin=0, vmax=max(r["metrics"]["target_peak"] for r in pair))
                axes[index, col].contour(target, levels=[.5], colors="cyan", linewidths=.7)
                yy, xx = np.where(target > .5)
                axes[index, col].set_xlim(xx.min()-10, xx.max()+10)
                axes[index, col].set_ylim(yy.max()+10, yy.min()-10)
                axes[index, col].set_title(f"{shape} {('seed','final')[col]} CV={row['metrics']['cv']:.3f} EE={row['metrics']['ee_roi']:.3f}")
                axes[index, col].axis("off")
        ccd.tight_layout()
        ccd_name = f"{path.name}_ccd_20261026.png"
        ccd.savefig(figures / ccd_name, dpi=160)
        plt.close(ccd)
        lines += [f"![CCD目标区域，前后共享强度范围](figures/{ccd_name})", ""]
    (artifacts / "metrics.json").write_text(json.dumps(dict(fits=traces, runs=runs), indent=2), encoding="utf-8")
    lines += ["## 解释范围", "", "核心 CV 使用固定目标掩膜腐蚀后的区域；EE_ROI 分母为固定相机采样窗口，不能等同于全传感器能量效率。",
              "曝光在每个形状的配对测量中保持固定；跨曝光归一化假设响应近似线性，不代表本轮独立完成了相机响应标定。",
              "每次相位更新保存相位、模型指标、主机 UTC 微秒时间与单调纳秒时间；并非相机硬触发时间戳。",
              "候选接受只保证当前配对规则下的测量改善，不保证所有时间点或每个指标单调改善。", ""]
    if (args.root / "interpretation.md").exists():
        lines += [(args.root / "interpretation.md").read_text(encoding="utf-8")]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
