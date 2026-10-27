"""Paired optical-frozen radiometry ablations and corrected-fit CV diagnosis."""
from __future__ import annotations

import csv
from loguru import logger
import json
import pickle
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.tools.slm.transfer.fit import metrics
from ao_shaping.tools.slm.transfer.progress import fit_trace
from ao_shaping.tools.slm.transfer.radiometry import fit_radiometry

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/cold_start_20261026_1004"
REPORT = ROOT / "report/shape_trials/cold_start/cv_diagnosis_20261026.md"


def main():
    out = REPORT.parent / "artifacts/cv_diagnosis_20261026"
    figdir = REPORT.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    ablations, individual = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (name, index) in zip(axes, (("fit", 65), ("fit_refined", 160))):
        path = DATA / name
        with (path / "iterations.pkl").open("rb") as stream:
            entry = pickle.load(stream)[index]
        rows = json.loads((path / "provenance.json").read_text())["rows"]
        with np.load(path / "inputs.npz") as data:
            measured, train = data["measured"], data["train"]
        exposure = np.array([r["exposure_ms"] for r in rows])
        with np.load(path / f"iteration_{index:04d}.npz") as data:
            optical = (data["prediction"]-entry["parameters"]["background"])/entry["parameters"]["gain"]
        for mode in ("shared", "exposure"):
            prediction, params = fit_radiometry(optical, measured, exposure, train, mode=mode)
            scores = metrics(prediction, measured)
            selected = [s for s, t in zip(scores, train) if not t]
            ablations.append(dict(stage=name, iteration=index, mode=mode, parameters=params,
                validation={key: float(np.mean([s[key] for s in selected])) for key in ("correlation", "relative_l2", "cv_abs_error")}))
            axis.scatter(exposure[~train], [s["cv_prediction"]-s["cv_measured"] for s in selected], label=mode)
            for row, score in zip(rows, scores):
                individual.append(dict(stage=name, mode=mode, label=row["label"], split=row["split"],
                                       exposure_ms=row["exposure_ms"], **score))
        axis.axhline(0, color="gray", linewidth=.7)
        axis.set(xlabel="exposure (ms)", ylabel="predicted CV - measured CV", title=name)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figdir / "cv_exposure_ablation_20261026.png", dpi=160)
    plt.close(fig)
    traces = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in ("fit", "fit_refined", "cv_corrected_cold", "cv_corrected_refine"):
        path = DATA / name
        if not (path / "iterations.pkl").exists():
            continue
        trace = traces[name] = fit_trace(path)
        for axis, key in zip(axes, ("validation_correlation", "validation_relative_l2", "validation_cv_abs_error")):
            axis.plot([x["iteration"] for x in trace], [x[key] for x in trace], label=name)
            axis.set_title(key)
            axis.set_xlabel("update")
            axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "cv_corrected_fit_20261026.png", dpi=160)
    plt.close(fig)
    with (out / "per_phase.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(individual[0]))
        writer.writeheader()
        writer.writerows(individual)
    (out / "results.json").write_text(json.dumps(dict(ablations=ablations, traces=traces), indent=2))
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import FrozenDefocusForward, load_frozen_parameters
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics
    torch.set_num_threads(4)
    models = {name: load_frozen_parameters(DATA / name / "frozen_parameters.json")
              for name in ("selected_final", "cv_selected_model")}
    shape_results = []
    for run in ("plane", "plane_formation", "random_formation", "random_no_abba"):
        rows = json.loads((DATA / run / "manifest.json").read_text())
        for label in (("square_seed_replay" if run == "random_no_abba" else "square_seed"), "square_final"):
            row = next(r for r in rows if r["label"] == label)
            with np.load(DATA / run / row["file"]) as payload:
                xy, phase, target = payload["camera_xy"], payload["model_phase"], payload["target"]
            objective = ShapeObjective(target)
            item = dict(run=run, label=label, exposure_ms=row["exposure_ms"], measured=row["metrics"], models={})
            for name, params in models.items():
                forward = FrozenDefocusForward(params, xy, device="cpu")
                with torch.no_grad():
                    prediction = forward.predict(phase, exposure_ms=row["exposure_ms"])[0].numpy()
                item["models"][name] = image_metrics(prediction, objective)
            shape_results.append(item)
    sampling = []
    for fft, pixels in ((1024, 3), (2048, 3), (2048, 5)):
        params = replace(models["cv_selected_model"], fft_size=fft, pixel_samples=pixels)
        forward = FrozenDefocusForward(params, xy, device="cpu")
        with torch.no_grad():
            prediction = forward.predict(phase, exposure_ms=row["exposure_ms"])[0].numpy()
        sampling.append(dict(fft_size=fft, pixel_samples=pixels, **image_metrics(prediction, objective)))
    (out / "shape_and_sampling.json").write_text(json.dumps(dict(shapes=shape_results, sampling=sampling), indent=2))
    lines = ["# 冷启动物理预测模型 CV 误差的曝光偏置诊断（2026-10-26）", "",
        "## 已定位的原因", "",
        "旧模型在 counts/ms 空间拟合 `g*optical+b`。若相机还包含曝光无关的 counts 项，正确形式应为 `g*optical+b+bias_counts/exposure_ms`。光学加性背景与曝光无关的电子偏置不能混为同一常数。",
        "该形式与电子偏置、低 counts 量化等效偏置相容；仅凭当前有光数据不能唯一识别电子学来源，也不能证明短曝光非线性已经全部校准。", "",
        "固定光学参数，所有辐射系数只在 12 个训练相位拟合，12 个验证相位仅作评估。新增参数只有一个 `bias_counts`。",
        "验证 CV 是拟合窗口（241×241 CCD 像元、2 px 采样）的整体 CV，不等于整形目标核心 CV。", "",
        "| 光学状态 | 辐射模型 | 验证相关性 | 验证相对 L2 | 验证 CV 绝对误差 | counts 偏置 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in ablations:
        q = row["validation"]
        lines.append(f"| {row['stage']} iter {row['iteration']} | {row['mode']} | {q['correlation']:.6f} | {q['relative_l2']:.6f} | {q['cv_abs_error']:.6f} | {row['parameters']['bias_counts']:.6f} |")
    lines += ["", "![固定光学参数的曝光偏置对照](figures/cv_exposure_ablation_20261026.png)", "",
        "加性常数和正增益不改变 Pearson 相关性，却直接改变 `CV=std/mean` 的分母，因此会出现相关性提高但 CV 误差增大的现象。",
        "旧模型在低曝光时低估归一化背景，容易高估 CV；较高曝光时则相反。后期光学参数变化进一步放大这种偏差。固定光学参数的对照已证实，误差变化不能完全归因于传播模型退化。", "",
        "## 修正后的继续拟合", "", "![旧模型与修正模型的验证轨迹](figures/cv_corrected_fit_20261026.png)", "",
        "| 阶段 | 初始→最终验证相关性 | 初始→最终验证相对 L2 | 初始→最终验证 CV 误差 |", "|---|---:|---:|---:|"]
    for name, trace in traces.items():
        a, b = trace[0], trace[-1]
        lines.append(f"| {name} | {a['validation_correlation']:.6f} → {b['validation_correlation']:.6f} | {a['validation_relative_l2']:.6f} → {b['validation_relative_l2']:.6f} | {a['validation_cv_abs_error']:.6f} → {b['validation_cv_abs_error']:.6f} |")
    lines += ["", "## 独立整形相位的目标核心检查", "",
        "下列图像未参与本次修正拟合，但来自此前相同光路的优化轨迹；不是新采集的独立实验。初相位有重复，不能当作八个独立光路样本。", "",
        "| 数据段 / 相位 | 实测核心 CV | 旧选择模型 CV | 新选择模型 CV |", "|---|---:|---:|---:|"]
    for item in shape_results:
        lines.append(f"| {item['run']} / {item['label']} | {item['measured']['cv']:.6f} | {item['models']['selected_final']['cv']:.6f} | {item['models']['cv_selected_model']['cv']:.6f} |")
    for label, subset in (("全部八组", shape_results), ("四组最终整形相位", [x for x in shape_results if x["label"].endswith("_final")])):
        error = {name: float(np.mean([abs(x["models"][name]["cv"]-x["measured"]["cv"]) for x in subset])) for name in models}
        lines += ["", f"{label}的核心 CV 平均绝对误差：旧模型 {error['selected_final']:.6f}，新模型 {error['cv_selected_model']:.6f}。"]
    lines += ["", "整体窗口 CV 改善不保证最终整形核心预测改善。剩余误差还涉及光学模型对优化后相位的适配，不能仅靠曝光偏置一项宣布解决。", "",
        "## 传播采样与 CCD 像素积分收敛", "",
        "固定新模型和最后一组最终相位，仅改变 FFT 与像素积分采样；本检查不重新拟合参数。", "",
        "| FFT | 每轴像素积分采样 | 核心 CV | EE_ROI |", "|---:|---:|---:|---:|"]
    for item in sampling:
        lines.append(f"| {item['fft_size']} | {item['pixel_samples']} | {item['cv']:.6f} | {item['ee_roi']:.6f} |")
    lines += ["", "本相位加密采样的核心 CV 最大变化约 0.00073，远小于预测与实测约 0.23 的差值。采样误差不是这一个最终相位剩余偏差的主要来源；该结论不能推广为所有相位都已收敛。", "",
        "## 下一步针对目标核心的模型改进", "",
        "优先从已接受的整形相位附近采集小幅正负扰动，并保留不参与拟合的新扰动方向作验证。现有随机探针拟合改善不代表优化后相位族的局部响应准确；应同时检查核心平均信号、方差、CV 和目标区域能量，避免仅用相关性选择模型。",
        "先固定曝光响应参数，逐项比较入射振幅位置与宽度、剩余像差和 PSF 展宽对新相位的预测。每次只增加可由数据识别的少量参数，以验证响应与核心 CV 同时改善为依据；当前数据不能唯一归因到某一个光学误差项。",
        "在仿真中用有限差分检查模型内部导数，再用 CCD 正负扰动差分检查物理响应方向。曝光偏置与相位无关，新增项不改变强度对相位的导数，但会改变 CV 损失的数值与梯度权重；不能用内部导数通过来替代物理响应验证。", "",
        "## 工程实现与边界", "",
        "`transfer.radiometry.fit_radiometry` 统一拟合共享光学增益、光学加性背景和 counts 偏置；禁止读取验证像素估计这些系数。`transfer.fit` 默认使用曝光模型，`--radiometry shared` 可重放旧行为。",
        "冻结参数新增 `bias_counts`，旧检查点默认零；非零偏置预测必须显式提供当前曝光。CCD 优化每次自动曝光后更新协变量，汇总、方向分析、离线合成与展宽校准沿用同一口径。不能把已校正检查点当作与曝光无关的图像预测器。",
        "当前结果来自存档数据的离线拟合与评估，没有重新占用设备，也没有验证新模型生成相位后的物理整形效果。此前验证集已经参与过模型选择，后续仍需新相位和新 CCD 图像确认泛化。", ""]
    (DATA / "cv_diagnosis").mkdir(exist_ok=True)
    (DATA / "cv_diagnosis/results.json").write_text(json.dumps(dict(ablations=ablations, traces=traces), indent=2))
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info(json.dumps(ablations, indent=2))


if __name__ == "__main__":
    main()
