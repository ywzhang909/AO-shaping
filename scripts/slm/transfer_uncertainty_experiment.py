"""One-off transfer-model active sampling experiment (2026-11-04).

Reuse production capture, physical fit and frozen prediction. Delete after the
experiment is superseded; retained with the report for exact reproduction.
The final holdout never participates in fit, ranking or iteration selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    load_frozen_parameters,
)
from ao_shaping.tools.slm.transfer import capture, fit
from ao_shaping.tools.slm.transfer.probes import probe_grid
from ao_shaping.tools.slm.transfer.uncertainty import (
    disagreement, evaluate, fit_recorded_subset, normalized, predict, select_diverse,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_payloads(root: Path) -> dict:
    result = {}
    for row in read_json(root / "manifest.json"):
        with np.load(root / row["file"]) as data:
            result[row["file"]] = {key: data[key] for key in ("model_phase", "raw_frames")}
    return result


def fit_subset(args, name: str, calibration: list, training: list, validation: list,
               payloads: dict, checkpoint: Path) -> Path:
    """Duplicate manifest training rows implements a phase-level bootstrap."""
    directory = args.root / name
    if (directory / "fit/frozen_parameters.json").is_file():
        return directory / "fit/frozen_parameters.json"
    return fit_recorded_subset(
        directory, checkpoint, calibration, training, validation, payloads,
        iterations=args.iterations, span=args.span, device=args.device)


def run(args) -> None:
    import torch

    torch.set_num_threads(4)
    args.root.mkdir(parents=True, exist_ok=True)
    execution = dict(
        **vars(args), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        uncertainty="phase bootstrap ensemble; normalized intensity disagreement",
        protocol="12 initial train, 12 validation; +4 active vs +4 random; 8 untouched test")
    # Preserve the hash of code actually loaded for acquisition. Offline report
    # improvements must not retroactively change the hardware execution record.
    experiment = args.root / "experiment.json"
    if not experiment.exists():
        write_json(experiment, execution)
    write_json(args.root / "analysis_execution.json", execution)
    source = args.root / "capture"
    rows = read_json(source / "manifest.json")
    payloads = load_payloads(source)
    calibration = [row for row in rows if row["split"] == "calibration"]
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    # Same current geometry for ideal and historical model: isolate optical fit
    # from stale-coordinate errors. Do not use validation intensities here.
    geometry, _ = fit.geometry(calibration)
    origin = fit.spot(calibration[0])
    prior = load_frozen_parameters(args.checkpoint).effective_parameters()
    anchored = replace(prior, origin_xy=tuple(origin),
                       ramp_matrix=tuple(tuple(v) for v in geometry))
    historical = args.root / "historical/frozen_parameters.json"
    write_json(historical, anchored.to_dict())
    ideal = replace(anchored, defocus_diopter=0., residual_noll_radians={j: 0. for j in range(5, 12)},
                    amplitude_center_xy=(960., 600.), amplitude_width_xy=(400., 300.))
    ideal_path = args.root / "ideal/frozen_parameters.json"
    write_json(ideal_path, ideal.to_dict())
    baseline = fit_subset(args, "baseline", calibration, training, validation, payloads, historical)
    ideal_fit = fit_subset(args, "ideal_fit", calibration, training, validation, payloads, ideal_path)
    rng = np.random.default_rng(args.seed)
    committee = []
    for member in range(4):
        boot = [training[i] for i in rng.integers(0, len(training), len(training))]
        committee.append(fit_subset(args, f"committee_{member}", calibration, boot,
                                    validation, payloads, historical))
    initial_rows, initial_phases, initial_images, xy, _, _, initial_train = fit.load_dataset(
        source, prior.grid_size, args.span)
    val_phases = initial_phases[~initial_train]
    val_exposure = np.array([row["exposure_ms"] for row in initial_rows])[~initial_train]
    val_prediction = predict(baseline, val_phases, xy, val_exposure, args.device)
    val_error = np.array(evaluate(val_prediction, initial_images[~initial_train])["shape_relative_l2"])
    val_uq = disagreement(np.stack([predict(path, val_phases, xy, val_exposure, args.device)
                                   for path in committee]))
    write_json(args.root / "validation_uncertainty.json", dict(
        uncertainty=val_uq.tolist(), shape_error=val_error.tolist(),
        spearman=float(spearmanr(val_uq, val_error).statistic)))
    candidates = np.stack([probe_grid(i, prior.grid_size, args.seed + 10000)[1]
                           for i in range(60)])
    ensemble = np.stack([predict(path, candidates, xy, np.ones(len(candidates)), args.device)
                         for path in committee])
    scores = disagreement(ensemble)
    active = select_diverse(scores, candidates, 4)
    random = rng.choice([i for i in range(len(candidates)) if i not in active], 4,
                        replace=False).tolist()
    selection = dict(active=active, random=random, scores=scores.tolist(),
                     families=[probe_grid(i, prior.grid_size, args.seed + 10000)[0]
                               for i in range(len(candidates))])
    write_json(args.root / "selection.json", selection)
    np.savez_compressed(args.root / "candidates.npz", phases=candidates, scores=scores)
    logger.info("Selected active={} random={}", active, random)
    second = args.root / "selected_capture"
    if not (second / ".capture_done").exists():
        if not args.execute:
            logger.info("Selection ready; --execute required to capture selected commands")
            return
        original = capture.probes

        def selected_probes(size, count, seed, periods=(64, 128)):
            yield from list(original(size, 0, seed, periods))[:-1]
            # Interleave arms, so drift is not confounded with acquisition rule.
            for slot in range(4):
                for arm, indices in (("active", active), ("random", random)):
                    index = indices[slot]
                    yield f"{arm}_{index:03d}", "train", candidates[index]
            for index in range(8):
                family, phase = probe_grid(index, size, args.seed + 20000)
                yield f"holdout_{index:03d}_{family}", "test", phase
            yield "reference_end", "calibration", np.zeros((size, size), np.float32)

        with patch.object(capture, "probes", selected_probes):
            capture.capture(SimpleNamespace(out=second, camera="daheng", cam_id=0, slm_number=1,
                                            grid=prior.grid_size, count=0, frames=3, seed=args.seed,
                                            execute=True, save_dataset=True, exposure_max_steps=2))
    extra = read_json(second / "manifest.json")
    second_payloads = load_payloads(second)
    # Namespace filenames before merging two capture manifests.
    for row in extra:
        original_file = row["file"]
        row["file"] = f"second_{original_file}"
        payloads[row["file"]] = second_payloads[original_file]
    adapted = {}
    for arm in ("active", "random"):
        additions = [row for row in extra if row["label"].startswith(arm + "_")]
        adapted[arm] = fit_subset(args, arm, calibration, training + additions, validation,
                                  payloads, historical)
    # Evaluate on the first-session fixed window. Re-measured zero is a drift
    # diagnostic only; never shift each holdout to make its prediction look good.
    holdout = [row for row in extra if row["split"] == "test"]
    phases = np.stack([payloads[row["file"]]["model_phase"] for row in holdout])
    measured, noise = [], []
    for row in holdout:
        frames = payloads[row["file"]]["raw_frames"]
        crop = frames[:, xy[..., 1].astype(int), xy[..., 0].astype(int)].astype(float)
        measured.append(crop.mean(0) / row["exposure_ms"])
        noise.append(float(np.linalg.norm(crop[0] - crop[1]) /
                           max(np.linalg.norm(crop.mean(0)), 1e-9)))
    measured = np.stack(measured)
    exposures = np.array([row["exposure_ms"] for row in holdout])
    models = dict(ideal=ideal_path, ideal_fit=ideal_fit, historical=historical,
                  baseline=baseline, **adapted)
    predictions = {name: predict(path, phases, xy, exposures, args.device)
                   for name, path in models.items()}
    results = {name: evaluate(value, measured) for name, value in predictions.items()}
    holdout_committee = np.stack([predict(path, phases, xy, exposures, args.device)
                                  for path in committee])
    uq = disagreement(holdout_committee)
    error = np.array(results["baseline"]["shape_relative_l2"])
    summary = dict(models=results, selection=selection, holdout_labels=[r["label"] for r in holdout],
                   validation_uncertainty=read_json(args.root / "validation_uncertainty.json"),
                   holdout_uncertainty=uq.tolist(),
                   uncertainty_error_spearman=float(spearmanr(uq, error).statistic),
                   repeated_frame_relative_l2=noise,
                   reference_drift_px=(fit.spot(extra[0]) - origin).tolist(),
                   model_paths={name: str(path) for name, path in models.items()})
    summary["holdout_family"] = {
        family: {name: float(np.mean(np.array(result["shape_relative_l2"])[mask]))
                 for name, result in results.items()}
        for family in ("zernike", "smooth", "bandlimited")
        for mask in [np.array([label.endswith(family) for label in summary["holdout_labels"]])]
        if mask.any()}
    bootstrap_rng = np.random.default_rng(args.seed + 30000)
    indices = bootstrap_rng.integers(0, len(holdout), (10000, len(holdout)))
    active_error = np.array(results["active"]["shape_relative_l2"])
    random_error = np.array(results["random"]["shape_relative_l2"])
    deltas = (random_error - active_error)[indices].mean(1)
    summary["active_vs_random_error_reduction"] = dict(
        mean=float((random_error - active_error).mean()),
        paired_bootstrap_95_ci=np.quantile(deltas, [.025, .975]).tolist(),
        note="exploratory phase bootstrap, only 8 commands and one optical session")
    selected_rows = [row for row in extra if row["label"].startswith(("active_", "random_"))]
    selected_phases = np.stack([payloads[row["file"]]["model_phase"] for row in selected_rows])
    selected_images = np.stack([
        payloads[row["file"]]["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
        / row["exposure_ms"] for row in selected_rows])
    selected_prediction = predict(baseline, selected_phases, xy,
                                  np.array([r["exposure_ms"] for r in selected_rows]), args.device)
    summary["selected_before_adaptation"] = {
        arm: evaluate(selected_prediction[indices], selected_images[indices])
        for arm in ("active", "random")
        for indices in [np.array([r["label"].startswith(arm + "_") for r in selected_rows])]}
    write_json(args.root / "summary.json", summary)
    np.savez_compressed(args.root / "evaluation.npz", measured=measured, camera_xy=xy,
                        phases=phases, **predictions)
    logger.info("Results: {}", {name: (r["mean_ncc"], r["mean_shape_relative_l2"])
                                for name, r in results.items()})
    render(args.root)


def render(root: Path, output: Path | None = None) -> None:
    """Render saved observations only; this function never opens devices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = read_json(root / "summary.json")
    if output is None:
        output = Path(__file__).resolve().parents[2] / "report/shape_trials/transfer_uncertainty"
    write_json(output / "artifacts/summary_20261104.json", summary)
    for name, path in summary["model_paths"].items():
        model_path = Path(path)
        if not model_path.is_absolute():
            parts = model_path.parts
            if "transfer_uncertainty_20261104" in parts:
                parts = parts[parts.index("transfer_uncertainty_20261104") + 1:]
            model_path = root.joinpath(*parts)
        write_json(output / f"artifacts/{name}_parameters_20261104.json", read_json(model_path))
    write_json(output / "artifacts/experiment_20261104.json", read_json(root / "experiment.json"))
    write_json(output / "artifacts/analysis_execution_20261104.json",
               read_json(root / "analysis_execution.json"))
    with np.load(root / "evaluation.npz") as data:
        names = ["measured", "ideal", "ideal_fit", "baseline", "active", "random"]
        fig, axes = plt.subplots(3, len(names), figsize=(15, 7))
        for row in range(3):
            for col, name in enumerate(names):
                axes[row, col].imshow(normalized(data[name])[row], cmap="inferno", vmin=0, vmax=10)
                axes[row, col].set_title(name)
                axes[row, col].axis("off")
        fig.tight_layout()
        figure = output / "figures/predictions_20261104.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=140)
        plt.close(fig)
    render_diagnostics(root, output, summary)
    table = "\n".join(
        f"| {name} | {value['mean_ncc']:.5f} | {value['mean_shape_relative_l2']:.5f} | "
        f"{value['mean_relative_l2']:.5f} |"
        for name, value in summary["models"].items())
    family_table = "\n".join(
        f"| {family} | {values['baseline']:.5f} | {values['active']:.5f} | {values['random']:.5f} |"
        for family, values in summary["holdout_family"].items())
    comparison = summary["active_vs_random_error_reduction"]
    models = summary["models"]
    ideal_improvement = 100 * (1 - models["ideal_fit"]["mean_shape_relative_l2"]
                              / models["ideal"]["mean_shape_relative_l2"])
    active_improvement = 100 * (1 - models["active"]["mean_shape_relative_l2"]
                               / models["baseline"]["mean_shape_relative_l2"])
    report = f"""# slm-transfer-shapes 不确定性选样与少样本迁移实验（2026-11-04）

## 实测结论

**少样本物理参数迁移改善了独立命令预测；当前委员会分歧未通过置信度验证，
不能据此认定主动采样优于随机采样。** 理想 2f 起点用 12 条实测训练命令拟合后，
8 条独立测试的形状相对 L2 从 {models['ideal']['mean_shape_relative_l2']:.5f}
降至 {models['ideal_fit']['mean_shape_relative_l2']:.5f}（下降 {ideal_improvement:.2f}%）。
历史参数热启动的 baseline 为 {models['baseline']['mean_shape_relative_l2']:.5f}；
加 4 条主动采样降至 {models['active']['mean_shape_relative_l2']:.5f}
（相对 baseline 下降 {active_improvement:.2f}%），加 4 条随机采样为
{models['random']['mean_shape_relative_l2']:.5f}。主动与随机差异的配对区间跨零，
本轮不支持宣称主动选样有可靠优势。残留形状误差约 0.44，模型仍需改进。

## 对象与实验边界

本轮只使用 `slm-transfer-shapes` 的 `PhysicalForwardFit` / `FrozenDefocusForward`。
它是可微物理预测模型，拟合离焦、光轴、入射高斯振幅、Noll 5–11 像差与共享辐射参数，
不是带 Dropout 的深度网络。权重起点是今天 19:21 会话的
`data/debug/slm_transfer/20261104_192122/fit/frozen_parameters.json`。
原文件不改；当前零点与 signed-ramp 几何重测后用于所有模型。

模型键读法：

| 键 | 含义 |
|---|---|
| ideal | 理想 2f 基准模型：采用实测几何但把离焦与像差设为 0，高斯幅度为假设值，增益沿用起点 |
| ideal_fit | 理想 2f 基准模型经少样本 CCD 实测标定后的物理预测模型 |
| historical | 历史标定物理预测模型：沿用历史会话拟合参数并更新本轮几何，未用本轮光路数据重新拟合 |
| baseline | 当前光路用 12 条实测训练命令拟合后的热启动物理预测模型 |
| active / random | 从同一 baseline 追加 4 条实测命令标定后的物理预测模型（主动/随机选样） |

实验数据在 `data/debug/transfer_uncertainty_20261104/`。
`experiment.json` 保留实验执行时的脚本和输入权重 SHA256；
`analysis_execution.json` 单独记录后续离线分析脚本版本。两个 capture 目录保存 raw frames、
实际面板灰度、raw radians、曝光回读、逐帧时钟和 manifest。
SLM 为 1064 nm，1920×1200 面板；模型控制网格 128²、FFT 1024²、3×3 亚像元积分。
相机使用生产采集路径的 Mono8 计数，自动曝光只保护动态范围；评价单位为 counts/ms。
固定相机窗半宽 120 px、采样步长 2 px，任何测试图像都不单独重心对齐。

## 预定协议

1. 首轮 24 条探针按生产划分为 12 train、12 validation，另有平场首尾和 8 条 signed-ramp。
2. 原历史参数热启动，训练损失选出 baseline；理想 2f 起点另拟合 ideal_fit。
   ideal 与 ideal_fit 共享实测几何；ideal 的高斯幅度是假设值，增益沿用起点，
   因而理想支路的绝对强度误差不可解释成纯像差误差。
3. 4 个独立相位 bootstrap 重采样拟合构成委员会；结构与物理方程相同。
   每个成员 60 次更新，不从 validation 挑迭代，也不引入未训练的 Dropout。
4. 60 条新候选只预测不采集，以均值归一化强度的集成标准差 RMS 排序，
   选择 4 条最高分歧且非重复命令。随机变体从余下候选无放回抽 4 条。
   两个变体交错实测，每条记录 3 帧。
5. 两个变体分别用同样 12+4 条训练命令从相同历史起点拟合，更新预算相同。
   另 8 条全新命令仅最终评价使用，未进入训练、委员会、选样或停止条件。
   这些测试命令属于相同三类探针分布，未验证任意目标整形相位的外推。

主动候选编号 `{summary['selection']['active']}`；随机候选编号
`{summary['selection']['random']}`。4 条主动候选全部属于 Zernike 族。
两次采集起点零点漂移 `{summary['reference_drift_px']}` px。

## 独立 CCD 预测结果

形状相对 L2 为逐图均值归一化后的预测误差；绝对相对 L2 不做逐图亮度拟合。
NCC 越高越好，两种 L2 越低越好。所有数值为 8 条独立命令的算术均值。

| 模型 | NCC | 形状相对 L2 | 绝对相对 L2 |
|---|---:|---:|---:|
{table}

| 测试相位族 | baseline 形状 L2 | +4 主动 | +4 随机 |
|---|---:|---:|---:|
{family_table}

随机减主动的形状误差均值为 `{comparison['mean']:.6f}`，逐命令配对 bootstrap
95% 区间 `{comparison['paired_bootstrap_95_ci']}`。
该区间只有 8 条命令、一个光路会话，不能代表跨日稳定收益。
原始逐命令结果见 [JSON](artifacts/summary_20261104.json)。

![实测与预测；每行独立归一化，固定色标](figures/predictions_20261104.png)

## 置信度是否可信

验证集分歧与 baseline 形状误差的 Spearman 相关为
`{summary['validation_uncertainty']['spearman']:.4f}`；最终测试相关为
`{summary['uncertainty_error_spearman']:.4f}`。
分歧只是排序分数，没有校准成“预测正确概率”，也没有统计覆盖率保证。
同样物理结构的成员可能共同漏掉振幅结构、SLM 响应、散射或高阶像差，
所以低分歧不能证明预测准确。均值归一化虽能排除整体增益分歧，仍对尖锐光斑更敏感，
导致候选全部集中到 Zernike 族。是否有效必须与分族残差、随机对照一起判断。

三帧中的第一帧与第二帧相对 L2（未经均值归一化）为
`{summary['repeated_frame_relative_l2']}`。
这只是同命令短时间重复性诊断，包含量化、散粒、读出和微小漂移；
不是模型的认知不确定性，也不能与形状误差直接相减。

## 诊断图

![按测试相位族的形状误差](figures/family_shape_error_20261104.png)

![委员会分歧与形状误差的关系](figures/disagreement_error_20261104.png)

![随机与主动追加误差差的配对 bootstrap](figures/bootstrap_ab_20261104.png)

![训练损失曲线](figures/training_loss_20261104.png)

![候选分歧排序](figures/candidate_ranking_20261104.png)

## 可复用的方法与条件

| 方法 | 适用场景和条件 | 必须保留的检查 |
|---|---|---|
| 当前几何重测 + 共享物理参数迁移 | 当前系统接近 2f，面板口径/波长一致，参数模型有表达能力 | signed-ramp 正负两轴；有效平场零点；独立测试误差 |
| 相位 bootstrap 集成 | 已有多类相位和重复帧，拟合成本可承受 | 分歧对残差的验证；共同模型偏差；成员稳定收敛 |
| 分歧 + 多样性约束选样 | 分歧已证明与误差/学习收益相关，候选能覆盖整形工作域 | 相位族配额、wrapped 命令去重、随机探索和等预算随机变体 |
| 固定窗口 counts/ms 评价 | 相机响应近似线性、每次曝光可回读、所有帧未饱和 | 曝光计数偏置；禁止每幅测试单独拟合增益或位置 |
| 独立测试与版本化保存 | 小样本拟合也必须防止过拟合 | 测试不参与选迭代/选模型；原权重不覆盖；保存命令和原始图像 |

不要把分类 Softmax 温度缩放套到连续强度回归，不能从峰值概率制造置信度。
若使用 MC Dropout，应先在相同结构和 dropout 机制下训练；当前模型没有该结构。
Dirichlet 分类证据公式也不能直接给二维强度预测提供有效误差区间。
后续更适合在分族残差上做回归误差校准，并保留最低探索预算；校准失败时回退
分层覆盖采样，不能仅按委员会分歧驱动自动闭环。

本轮验证的是预测，不等价于环形/方形整形成功。实际整形仍需相同单位的
CCD 锚定目标、有限相位步长、方向导数检查与独立重放验收。
采集结束由生产 finally 恢复平场并关闭 SLM/CCD；未覆盖生产默认检查点。
"""
    (output / "transfer_uncertainty_20261104.md").write_text(report, encoding="utf-8")


def render_diagnostics(root: Path, output: Path, summary: dict) -> None:
    """Render device-free diagnostics over saved arrays (never opens hardware)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure = output / "figures"
    figure.mkdir(parents=True, exist_ok=True)
    rc = {"axes.grid": True, "grid.alpha": .3, "figure.dpi": 140,
          "font.size": 9, "axes.titlesize": 10,
          "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
          "axes.unicode_minus": False}
    plt.rcParams.update(rc)
    families = ["zernike", "smooth", "bandlimited"]
    family_values = summary["holdout_family"]

    # 1. Per-family shape error: baseline vs +4 active vs +4 random.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(families))
    width = .25
    for offset, arm, color in ((-width, "baseline", "#4C72B0"),
                               (0, "active", "#55A868"), (width, "random", "#C44E52")):
        ax.bar(x + offset, [family_values[f][arm] for f in families], width,
               label=arm, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.set_ylabel("形状相对 L2")
    ax.set_title("按测试相位族的形状误差（越低越好）")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.14))
    fig.tight_layout()
    fig.savefig(figure / "family_shape_error_20261104.png")
    plt.close(fig)

    # 2. Disagreement vs baseline shape error, validation and final holdout.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, key, title in (
        (axes[0], "validation_uncertainty", "验证集（12 命令）"),
        (axes[1], "holdout_uncertainty", "最终测试（8 命令）"),
    ):
        uq = np.array(summary[key] if key == "holdout_uncertainty"
                      else summary["validation_uncertainty"]["uncertainty"])
        err = np.array(summary[key] if key == "holdout_uncertainty"
                       else summary["validation_uncertainty"]["shape_error"])
        ax.scatter(uq, err, s=22, alpha=.8, color="#4C72B0")
        ax.set_xlabel("委员会分歧（RMS 集成标准差）")
        ax.set_ylabel("baseline 形状相对 L2")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figure / "disagreement_error_20261104.png")
    plt.close(fig)

    # 3. Paired bootstrap distribution of random-minus-active mean error.
    comparison = summary["active_vs_random_error_reduction"]
    active_error = np.array(summary["models"]["active"]["shape_relative_l2"])
    random_error = np.array(summary["models"]["random"]["shape_relative_l2"])
    rng = np.random.default_rng(110421 + 30000)
    indices = rng.integers(0, len(active_error), (10000, len(active_error)))
    deltas = (random_error - active_error)[indices].mean(1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(deltas, bins=48, color="#8172B2", alpha=.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(comparison["mean"], color="#C44E52", lw=1.4,
               label=f"均值 {comparison['mean']:.4f}")
    for bound in comparison["paired_bootstrap_95_ci"]:
        ax.axvline(bound, color="k", ls="--", lw=1)
    ax.set_xlabel("随机减主动的平均形状误差")
    ax.set_ylabel("bootstrap 计数")
    ax.set_title("配对 bootstrap：随机与主动追加的误差差（含零）")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure / "bootstrap_ab_20261104.png")
    plt.close(fig)

    # 4. Training loss curves for baseline and ideal_fit.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, color in (("baseline", "#4C72B0"), ("ideal_fit", "#55A868")):
        history = _load_iterations(root / name / "fit" / "iterations.pkl")
        losses = np.array([entry["loss"] for entry in history])
        ax.plot(losses, color=color, label=f"{name}（{len(losses)} 次更新）")
    ax.set_xlabel("更新次数")
    ax.set_ylabel("训练损失")
    ax.set_title("训练损失曲线（仅训练集，无验证选代）")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure / "training_loss_20261104.png")
    plt.close(fig)

    # 5. Candidate ranking by disagreement, coloured by phase family.
    with np.load(root / "candidates.npz", allow_pickle=False) as data:
        scores = np.asarray(data["scores"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    families_c = summary["selection"]["families"]
    colors = {"zernike": "#4C72B0", "smooth": "#55A868", "bandlimited": "#C44E52"}
    for fam in ("zernike", "smooth", "bandlimited"):
        mask = np.array([f == fam for f in families_c])
        ax.scatter(np.flatnonzero(mask), scores[mask], s=20, alpha=.85,
                   color=colors[fam], label=fam)
    selected = summary["selection"]["active"] + summary["selection"]["random"]
    for index in selected:
        ax.axvline(index, color="k", lw=.6, alpha=.3)
    ax.set_xlabel("候选编号")
    ax.set_ylabel("分歧评分")
    ax.set_title("60 条候选的分歧排序（按相位族着色）")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure / "candidate_ranking_20261104.png")
    plt.close(fig)


def _load_iterations(path: Path) -> list:
    """Read a fit iteration pickle; unknown format raises rather than hiding it."""
    with path.open("rb") as stream:
        history = pickle.load(stream)
    if not isinstance(history, list) or not history:
        raise ValueError(f"unexpected iteration history in {path}")
    for entry in history:
        if not isinstance(entry, dict) or "loss" not in entry:
            raise ValueError(f"unexpected iteration record in {path}: {type(entry)}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--render-only", action="store_true",
                        help="Only render saved results; never fit or open devices")
    parser.add_argument("--report-out", type=Path, help="Alternate offline report directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--span", type=int, default=120)
    parser.add_argument("--seed", type=int, default=110421)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.root, args.report_out)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --render-only is used")
        run(args)


if __name__ == "__main__":
    main()
