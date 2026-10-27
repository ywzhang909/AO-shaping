"""Recompute the large-square experiment from native raw frames; no devices."""
from __future__ import annotations

import argparse
from loguru import logger
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from ao_shaping.tools.slm.transfer.artifacts import audit_run
from ao_shaping.tools.slm.transfer.fit import geometry
from ao_shaping.tools.slm.transfer.capture import to_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/shape_trials/square1200"


def main():
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import FrozenDefocusForward, load_frozen_parameters
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/debug/slm_transfer/square1200_20261026_1350")
    args = parser.parse_args()
    figures = OUT / "figures"
    artifacts = OUT / "artifacts/20261026"
    figures.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    runs, endpoints, observations = [], [], []
    for directory in args.root.iterdir():
        if not (directory / "summary.json").exists() or not (directory / "manifest.json").exists():
            continue
        rows = json.loads((directory / "manifest.json").read_text())
        if not any(row["label"] == "square_final" for row in rows):
            continue
        cfg = json.loads((directory / "config.json").read_text())
        selected = [next(row for row in rows if row["label"] == label) for label in
                    ("square_seed_replay" if any(r["label"] == "square_seed_replay" for r in rows) else "square_seed", "square_final")]
        stage = dict(name=directory.name, config=cfg, audit=audit_run(directory),
                     measurements=rows, endpoints=[], iterations=0)
        params = load_frozen_parameters(Path(cfg["checkpoint"]))
        with (directory / "iterations.pkl").open("rb") as stream:
            history = pickle.load(stream)
        stage["iterations"] = len(history)
        for row in selected:
            with np.load(directory / row["file"]) as saved:
                image = saved["raw_frames"].mean(0)
                xy, target, core = saved["camera_xy"], saved["target"], saved["core"]
                yy, xx = np.indices(image.shape)
                coordinates = [(yy-xy[0, 0, 1])/cfg["window_step"], (xx-xy[0, 0, 0])/cfg["window_step"]]
                full_core = map_coordinates(core.astype(float), coordinates, order=0, mode="constant") > .5
                values = image[full_core]
                native_cv = float(values.std()/values.mean())
                endpoint = dict(label=row["label"], exposure_ms=row["exposure_ms"],
                    sparse_cv=row["metrics"]["cv"], full_resolution_cv=native_cv,
                    native_mean=float(values.mean()), native_peak=float(values.max()),
                    full_resolution_pixels=int(full_core.sum()), bit_depth=row.get("bit_depth", 8),
                    saturated_core_pixels=int((saved["raw_frames"][:, full_core] >= row.get("full_scale", 255)*250/255).sum()))
                forward = FrozenDefocusForward(params, xy, device="cpu")
                with torch.no_grad():
                    prediction = forward.predict(saved["model_phase"], exposure_ms=row["exposure_ms"])[0].numpy()
                predicted_core = prediction[core]
                measured_sample = image[xy[..., 1].astype(int), xy[..., 0].astype(int)]/(row["exposure_ms"]*row.get("counts_scale", 1.))
                endpoint.update(model_core_cv=float(predicted_core.std()/predicted_core.mean()),
                    model_core_mean_equivalent_counts_per_ms=float(predicted_core.mean()),
                    measured_core_mean_equivalent_counts_per_ms=float(measured_sample[core].mean()),
                    model_core_correlation=float(np.corrcoef(prediction[core], measured_sample[core])[0, 1]))
                stage["endpoints"].append(endpoint)
                if row["label"] == "square_final":
                    radius = np.hypot(xx-cfg["zero_xy"][0], yy-cfg["zero_xy"][1])
                    box = (np.abs(xx-cfg["target_xy"][0]) <= cfg["side"]/2-8) & (np.abs(yy-cfg["target_xy"][1]) <= cfg["side"]/2-8)
                    endpoint["zero_exclusion_sensitivity"] = {str(r): float(image[box & (radius > r)].std()/image[box & (radius > r)].mean()) for r in (0, 20, 40, 60, 80)}
                endpoints.append((directory.name, row["label"], image, cfg, float(values.mean())))
        runs.append(stage)
        for row in rows:
            observations.append(dict(stage=directory.name, label=row["label"], **row["metrics"],
                                     exposure_ms=row["exposure_ms"], timestamp_utc=row["timestamp_utc"], monotonic_ns=row["monotonic_ns"]))
    runs.sort(key=lambda s: s["config"]["timestamp_utc"])
    stage_order = {s["name"]: i for i, s in enumerate(runs)}
    endpoints.sort(key=lambda e: stage_order[e[0]])
    formats = json.loads((args.root / "camera_formats/formats.json").read_text())
    calibration = json.loads((args.root / "capture/manifest.json").read_text())
    matrix, spreads = geometry(calibration)
    scale_check = dict(matrix=matrix.tolist(), per_period_matrices=spreads, ccd_pixel_um=2.2,
                       side_ccd_px=1200, side_mm=2.64)
    if (args.root / "geometry_wide/manifest.json").exists():
        wide = json.loads((args.root / "geometry_wide/manifest.json").read_text())
        matrix_wide, wide_spread = geometry(wide)
        scale_check.update(wide_matrix=matrix_wide.tolist(), wide_per_period=wide_spread,
                           relative_matrix_difference=float(np.linalg.norm(matrix_wide-matrix)/np.linalg.norm(matrix)))
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 4.5*len(runs)), squeeze=False)
    for index, (name, label, image, cfg, mean) in enumerate(endpoints):
        axis = axes[index//2, index % 2]
        axis.imshow(image/max(mean, 1e-9), cmap="inferno", vmin=0, vmax=3,
                    extent=(-.5, image.shape[1]-.5, image.shape[0]-.5, -.5))
        x, y = cfg["target_xy"]
        axis.add_patch(plt.Rectangle((x-600, y-600), 1200, 1200, fill=False, edgecolor="cyan", linewidth=.8))
        axis.add_patch(plt.Circle(cfg["zero_xy"], cfg["zero_radius"], fill=False, edgecolor="lime", linewidth=.8))
        axis.set(xlim=(x-720, x+720), ylim=(y+720, y-720), xlabel="CCD x (native px)", ylabel="CCD y (native px)", title=f"{name}: {label}; I / mean(core)")
    fig.tight_layout()
    fig.savefig(figures / "ccd_square1200_20261026.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for stage in runs:
        rows = stage["measurements"]
        start = rows[0]["monotonic_ns"]
        axes[0].plot([(r["monotonic_ns"]-start)/1e9 for r in rows], [r["metrics"]["cv"] for r in rows], ".-", label=stage["name"])
        axes[1].plot([(r["monotonic_ns"]-start)/1e9 for r in rows], [r["exposure_ms"] for r in rows], ".-", label=stage["name"])
    axes[0].set(xlabel="seconds within stage", ylabel="measured core CV")
    axes[1].set(xlabel="seconds within stage", ylabel="exposure (ms)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figures / "metrics_square1200_20261026.png", dpi=160)
    plt.close(fig)
    replay_path = args.root / "common_replay/manifest.json"
    replay_rows = json.loads(replay_path.read_text()) if replay_path.exists() else []
    if replay_rows:
        for row in replay_rows:
            with np.load(args.root / "common_replay" / row["file"]) as saved:
                image = saved["raw_frames"].mean(0)
                xy, core = saved["camera_xy"], saved["core"]
                yy, xx = np.indices(image.shape)
                step = xy[0, 1, 0]-xy[0, 0, 0]
                mask = map_coordinates(core.astype(float), [(yy-xy[0, 0, 1])/step, (xx-xy[0, 0, 0])/step], order=0, mode="constant") > .5
                values = image[mask]
                row["full_resolution_cv"] = float(values.std()/values.mean())
                row["saturated_core_pixels"] = int((saved["raw_frames"][:, mask] >= 4014).sum())
        initial_values = {row["exposure_ms"]: row["metrics"]["cv"] for row in replay_rows if row["label"].endswith("_seed")}
        for row in replay_rows:
            row["cv_reduction_vs_initial"] = 1-row["metrics"]["cv"]/initial_values[row["exposure_ms"]]
        labels = sorted({r["label"] for r in replay_rows if r["label"].endswith("_final")})
        scores = {name: float(np.mean([r["metrics"]["cv"] for r in replay_rows if r["label"] == name])) for name in labels}
        best = min(scores, key=scores.get)
        source_row = next(r for r in replay_rows if r["label"] == best)
        with np.load(args.root / "common_replay" / source_row["file"]) as saved:
            phase = saved["model_phase"].copy()
        np.save(args.root / "best_model_phase.npy", phase)
        np.save(args.root / "best_panel_phase.npy", to_panel(phase))
        selection = dict(selected=best, source_file=source_row["file"], scores=scores,
                         criterion="lowest mean core CV across 50ms and 100ms exact-phase Mono12 replays", independent_test=False)
        (args.root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for i, exposure in enumerate((50., 100.)):
            for j, label in enumerate((next(r["label"] for r in replay_rows if r["label"].endswith("_seed")), best)):
                row = next(r for r in replay_rows if r["label"] == label and r["exposure_ms"] == exposure)
                with np.load(args.root / "common_replay" / row["file"]) as saved:
                    image = saved["raw_frames"].mean(0)
                axis = axes[i, j]
                axis.imshow(image/row["metrics"]["mean"], cmap="inferno", vmin=0, vmax=3,
                    extent=(-.5, image.shape[1]-.5, image.shape[0]-.5, -.5))
                cfg = runs[0]["config"]
                x, y = cfg["target_xy"]
                axis.add_patch(plt.Rectangle((x-600, y-600), 1200, 1200, fill=False, edgecolor="cyan", linewidth=.8))
                axis.add_patch(plt.Circle(cfg["zero_xy"], 60, fill=False, edgecolor="lime", linewidth=.8))
                axis.set(xlim=(x-720, x+720), ylim=(y+720, y-720), xlabel="CCD x (native px)", ylabel="CCD y (native px)",
                    title=f"{'Initial' if j == 0 else 'Selected final'}, {exposure:.0f} ms; CV={row['full_resolution_cv']:.4f}")
        fig.tight_layout()
        fig.savefig(figures / "common_replay_square1200_20261026.png", dpi=160)
        plt.close(fig)
    release_path = args.root / "common_replay/released.json"
    released = json.loads(release_path.read_text()) if release_path.exists() else None
    counts = dict(phase_updates=sum(s["iterations"] for s in runs),
        shaping_raw_frames=sum(s["audit"]["raw_frames"] for s in runs),
        replay_raw_frames=sum(len(row["frames"]) for row in replay_rows),
        fit_parameter_updates=60)
    result = dict(source=str(args.root.resolve()), stages=runs, scale_check=scale_check, camera_formats=formats,
                  common_replay=replay_rows, released=released, counts=counts)
    (artifacts / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (artifacts / "measurements.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    lines = ["# 1200 CCD 像元正方形的物理预测模型可微匀化（2026-10-26）", "",
        "## 目标、设备与坐标口径", "",
        "Santec SLM #1（22030108）、Daheng MER2-507-23GM NIR（FJB24112232），波长 1064 nm。目标边长 1200 CCD 原生像元，像元尺寸 2.2 µm，对应 CCD 面边长 2.64 mm；完整传感器为 2592×1944。",
        "SLM 面板 1920×1200，控制网格从旧 128 提高至 640（每控制单元 3 面板像元），FFT 2048。优化先用 CCD 4 px 步长，最后用 2 px 步长，视场宽 1640 px；保持核心边缘腐蚀 8 CCD px。指标另以全部原生像元重算，不把采样数组宽度当作实际 CCD 像素宽度。",
        "目标生成器原有固定 512×512 中间栅格会裁剪大目标，实验验证窗口也固定为 440 px；本轮修正为随目标扩展的栅格与显式窗口参数，旧小目标默认行为保留。",
        "图像轴直接显示 CCD 原生像素坐标；原始 FFT 网格不是 CCD 网格，利用当前正负光栅测得的二维矩阵映射两者。像元尺寸只用于单位换算，不再乘入已用 CCD 像素标定的矩阵。", "",
        "当前 ramp_matrix（CCD px / 周期频率）：", "", "```json", json.dumps(scale_check, indent=2), "```", "",
        "## 模型与算法", "",
        "采集当前光路 12 个相位探针及正负光栅、首尾参考，训练/验证各 6 相位。由上轮曝光感知物理参数迁移，60 次更新校正当前光路；保留全部预测和参数。此处不是从零权重训练。",
        "初始整形相位不用 GS：对拟合入射振幅平方的两个边缘分布求累积分布，把各自能量映射到目标均匀区间，积分频率生成原始弧度相位，再减去拟合静态像差。正方形先沿实测光栅轴生成，再按 CCD 固定掩膜可微调整。",
        "每块 CCD 锚定当前接受相位，冻结物理预测模型，反传纯核心 CV² 并限制相位增量。第一段按模型损失接受；Mono12 段显式启用同曝光 A/B 顺序检查，仅当候选实测 CV 降低才接受。没有启用 A–B–B–A，顺序检查不抵消时间漂移。",
        "固定排除测得零级中心半径 60 CCD px；核心边缘再腐蚀 8 px。测光仅用目标核心，允许排除区外饱和，保存完整画面与饱和数。该策略在数值评价中排除零级，并没有物理消除零级。", "",
        "## 实测结果", "",
        "每段比较初始重放与最终相位的相同曝光；不能直接以跨段、跨位深的原始 counts 差值论证优化收益。", "",
        "| 阶段 | 位深 | 同曝光 ms | 4 px 采样 CV 初始→最终 | 全原生像元 CV 初始→最终 | 相位更新 | CCD 接受块 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in runs:
        a, b = stage["endpoints"]
        audit = stage["audit"]
        lines.append(f"| {stage['name']} | {a['bit_depth']} | {a['exposure_ms']:.3f} / {b['exposure_ms']:.3f} | {a['sparse_cv']:.6f} → {b['sparse_cv']:.6f} | {a['full_resolution_cv']:.6f} → {b['full_resolution_cv']:.6f} | {stage['iterations']} | {audit['accepted_blocks']}/{audit['blocks']} ({','.join(audit['acceptance_scopes'])}) |")
    lines += ["", "## 最终同格式、同曝光、同掩膜重放", "",
        "直接读取存档原始相位，不重新生成或做几何迁移。全部使用 Mono12，依次以 50 ms、100 ms 重放，原始帧和时间独立保存。", "",
        "| 相位 | 曝光 ms | CCD 核心 CV（采样 / 全分辨率） | 对初相位 CV 降幅 | 核心原生 counts 均值 |",
        "|---|---:|---:|---:|---:|"]
    for row in replay_rows:
        lines.append(f"| {row['label']} | {row['exposure_ms']:.3f} | {row['metrics']['cv']:.6f} / {row['full_resolution_cv']:.6f} | {100*row['cv_reduction_vs_initial']:.2f}% | {row['metrics']['mean']:.3f} |")
    lines += ["", "![同格式同曝光共同重放](figures/common_replay_square1200_20261026.png)"]
    lines += ["", "重放在单一顺序下进行，包含阶段后的光路漂移与噪声，不能声称反映每次参数更新的独立因果收益。两档曝光共同用于选择候选，不能再称为独立测试集。`selection.json` 保存选择规则和全部候选分数，`best_model_phase.npy` 与 `best_panel_phase.npy` 保存选择相位（原始弧度）。", "",
        "最终共同重放中，同一相位在 50 ms 和 100 ms 的 CV 仍有差异，提示曝光响应/低信号影响尚存；同曝光对照给出了更可靠的相位收益。下一轮应在 Mono12 的目标强度范围内单独标定响应，并对宽视场的加性背景（杂散光）项建模，避免把中心小窗口的常数背景推广到整个 1200 px 方形。"]
    lines += ["", "## 冻结模型的目标核心绝对预测", "",
        "以下为未锚定的物理预测模型预测；模型输出和实测均换成 Mono8 等效 counts/ms。锚定优化用的是 CCD 图像加预测增量，不代表绝对预测已经准确。", "",
        "| 最终相位 | 预测 CV | 实测 CV | 预测均值 | 实测均值 | 核心相关性 |",
        "|---|---:|---:|---:|---:|---:|"]
    for stage in runs:
        b = stage["endpoints"][-1]
        lines.append(f"| {stage['name']} | {b['model_core_cv']:.6f} | {b['sparse_cv']:.6f} | {b['model_core_mean_equivalent_counts_per_ms']:.6f} | {b['measured_core_mean_equivalent_counts_per_ms']:.6f} | {b['model_core_correlation']:.6f} |")
    lines += ["", "当前检查点的常数背景约 0.496 等效 counts/ms，已超过宽视场核心实测总均值约 0.05–0.06。这直接否定了将中心小窗口拟合背景作为整个大方形均匀背景的做法；它会把绝对预测均值抬高约一个数量级、把预测 CV 压低。曝光偏置修正和像素尺度一致并不足以解决这一空间外推问题。",
        "后续应使用本次宽视场、Mono12 和目标相位附近的数据拟合空间变化的加性背景（杂散光）项，并以新的相位验证图案结构及局部响应；本轮没有把锚定优化的成功误写为绝对正向预测已经准确。", ""]
    lines += ["", "![CCD 初始与最终，亮度按各自核心均值归一化](figures/ccd_square1200_20261026.png)", "",
        "![实测 CV 与曝光轨迹](figures/metrics_square1200_20261026.png)", "",
        "零级排除半径敏感性、全部测量指标与原始帧审计见 [results.json](artifacts/20261026/results.json) 和 [measurements.json](artifacts/20261026/measurements.json)。",
        "图像仍有结构，不称为理想平顶光。零级附近存在饱和，原始 EE_ROI 不能作为精确物理效率；本轮优先评价固定排除掩膜下的均匀度。", "",
        "## 12 bit 与曝光", "",
        "实际 SDK 枚举支持 Mono8 与 Mono12；Mono12 返回 uint16 容器，保留 12 bit 原始值，满量程按 4095，不能按 uint16 容器的 65535。驱动原先仅区分 16 与其他位深，12 会回落 Mono8；本轮修正请求格式并开放相机工厂参数。",
        "模型此前按 Mono8 counts/ms 标定。Mono12 锚定图像按原生 counts/16 转成等效单位；这只对齐码值尺度，不代表已完成独立 Mono12 非线性或噪声标定。不同曝光下 CV 仍可能不同，因此最终使用同曝光重放。不能把额外优化后改善归因于位深切换本身。", "",
        "## 数据完整性与失败记录", "",
        f"原始数据根目录：`{args.root.as_posix()}`。每组保留 native raw_frames、面板灰度、原始弧度相位、目标与核心掩膜、CCD 坐标、逐帧曝光、UTC 微秒与单调纳秒时间；每次迭代保存相位、损失、梯度范数与时间。时间戳属于主机调用，不是相机硬触发时间。",
        "首次 transport_cv 在目标测光适配中误传一维数组，触发中值滤波接口错误，随后正常释放设备；该次预搜索原始帧未落盘，只有配置和错误日志，不能计为完整采集。修复后从 transport_cv_r2 重新开始，曝光每次读取即单独落盘，即使后续阶段异常也保留原始帧。",
        "完成段均由 audit_run 从原始帧重算并核对相位、核心掩膜、曝光、时间与接受规则。采样 CV 与全分辨率 CV 分开报告，排除区外饱和单独记录。",
        f"四段共 {counts['phase_updates']} 次相位更新、{counts['shaping_raw_frames']} 张正式整形帧，最终共同重放另有 {counts['replay_raw_frames']} 张原始帧；还保留全部正式探针、几何、格式验证和曝光搜索帧。未将搜索帧与汇总 npz 的重复副本当作新增测量。",
        f"最终设备释放记录：`{released}`。保存最后选择相位不等于继续占用设备；退出前恢复平相位，关闭 SLM 和相机上下文。", "",
        "## 复现", "", "```powershell",
        ".venv\\Scripts\\python.exe -m ao_shaping.tools.slm.transfer.validate `",
        f"  --checkpoint {args.root.as_posix()}/model640/frozen_parameters.json `",
        "  --out data/debug/slm_transfer/square1200_new_session `",
        "  --zero-xy 1224 962 --period 128 --side 1200 --window-span 820 --window-step 4 `",
        "  --shapes square --initial-phase transport --objective cv --mode anchored `",
        "  --blocks 8 --updates 20 --phase-lr 0.025 --phase-rms-cap 0.2 `",
        "  --adaptive-step --adaptive-exposure --zero-radius 60 --meter-target `",
        "  --frames 3 --bit-depth 12 --no-abba --ccd-check --accept-metric cv --execute",
        "```", "",
        "该命令重放本次光路参数，不保证变化后的光路无须迁移。新会话先用 capture 重新测几何，再 fit 适配；高频大目标需检查控制 Nyquist 带宽。不能把1200 CCD像元误写成1200面板像元。", ""]
    (OUT / "square1200_20261026.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(json.dumps([dict(name=s["name"], endpoints=s["endpoints"], audit=s["audit"]["status"]) for s in runs], indent=2))


if __name__ == "__main__":
    main()
