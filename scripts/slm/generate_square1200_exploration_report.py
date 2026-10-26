"""Recompute the strict/nonmonotone CCD comparison from saved raw captures."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from ao_shaping.tools.slm.transfer.artifacts import audit_run
from ao_shaping.tools.slm.transfer.capture import to_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/shape_trials/square1200_exploration"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/debug/slm_transfer/square1200_explore_20261026_1435")
    args = parser.parse_args()
    figures, artifacts = OUT/"figures", OUT/"artifacts/20261026"
    figures.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    stages = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, ("strict", "nonmonotone")):
        run = args.root/name
        cfg = json.loads((run/"config.json").read_text())
        rows = json.loads((run/"manifest.json").read_text())
        blocks = json.loads((run/"summary.json").read_text())
        with (run/"iterations.pkl").open("rb") as stream:
            iterations = pickle.load(stream)
        assert len(iterations) == cfg["blocks"]*cfg["updates"]
        assert all(i["end_monotonic_ns"] >= i["monotonic_ns"] and i["timestamp_utc"] for i in iterations)
        seed = next(r for r in rows if r["label"] == "square_seed")
        final = next(r for r in rows if r["label"] == "square_final")
        with np.load(run/seed["file"]) as saved:
            phase = saved["model_phase"].copy()
        if name == "strict":
            seed_phase = phase
        else:
            assert np.array_equal(seed_phase, phase), "comparison must start from identical phases"
        stage = dict(config=cfg, audit=audit_run(run), summary=blocks, iterations=len(iterations),
                     seed_cv=seed["metrics"]["cv"], final_cv=final["metrics"]["cv"],
                     accepted=sum(b["accepted"] for b in blocks),
                     accepted_uphill=sum(b["accepted"] and b["candidate"] > b["baseline"] for b in blocks))
        stages[name] = stage
        x = np.arange(len(blocks))
        ax.plot(x, [b["baseline"] for b in blocks], ".-", label="CCD baseline")
        ax.plot(x, [b["candidate"] for b in blocks], ".-", label="CCD proposal")
        ax.plot(x, np.minimum.accumulate([min(seed["metrics"]["cv"], b["candidate"] if b["accepted"] else np.inf) for b in blocks]),
                "--", color="black", label="Best observed accepted CV")
        uphill = [b for b in blocks if b["accepted"] and b["candidate"] > b["baseline"]]
        ax.scatter([b["block"] for b in uphill], [b["candidate"] for b in uphill],
                   marker="^", color="#D55E00", s=60, label="Accepted uphill")
        ax.set(xlabel="Block (20 phase updates)", ylabel="Core CV", title=name)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(figures/"trajectory_20261026.png", dpi=180)
    plt.close(fig)

    replay = args.root/"common_replay"
    records = json.loads((replay/"manifest.json").read_text())
    comparisons, display = [], []
    for row in records:
        with np.load(replay/row["file"]) as saved:
            raw, xy, core = (saved[k] for k in ("raw_frames", "camera_xy", "core"))
            image = raw.mean(0)
            yy, xx = np.indices(image.shape)
            step = xy[0, 1, 0]-xy[0, 0, 0]
            mask = map_coordinates(core.astype(float), [(yy-xy[0, 0, 1])/step, (xx-xy[0, 0, 0])/step],
                                   order=0, mode="constant") > .5
            values = image[mask]
            frame_cv = [float(f[mask].std()/f[mask].mean()) for f in raw]
            entry = dict(label=row["label"], exposure_ms=row["exposure_ms"],
                         sampled_cv=row["metrics"]["cv"], native_cv=float(values.std()/values.mean()),
                         frame_cv=frame_cv, mean_counts=float(values.mean()),
                         saturated_core_samples=int((raw[:, mask] >= 4095*250/255).sum()),
                         native_core_pixels=int(mask.sum()), file=row["file"])
            comparisons.append(entry)
            if row["exposure_ms"] == 50:
                sample = image[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                display.append((row["label"], sample/values.mean(), entry["native_cv"], xy.copy()))
    fig, axes = plt.subplots(1, len(display), figsize=(4*len(display), 4), constrained_layout=True, squeeze=False)
    for ax, (label, sample, cv, xy) in zip(axes[0], display):
        image = ax.imshow(sample, origin="upper", vmin=0, vmax=2, cmap="inferno",
                          extent=[xy[0, 0, 0], xy[0, -1, 0], xy[-1, 0, 1], xy[0, 0, 1]])
        ax.set(title=f"{label}\nCV={cv:.4f}", xlabel="Native CCD x (px)", ylabel="Native CCD y (px)")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Intensity / core mean (clipped display)", shrink=.75)
    fig.savefig(figures/"common_ccd_20261026.png", dpi=150)
    plt.close(fig)
    release = json.loads((replay/"released.json").read_text())
    scores = {}
    for r in comparisons:
        scores.setdefault(r["label"], []).append(r["native_cv"])
    selected = min(scores, key=lambda label: np.mean(scores[label]))
    selected_row = next(r for r in comparisons if r["label"] == selected)
    with np.load(replay/selected_row["file"]) as saved:
        np.save(args.root/"best_model_phase.npy", saved["model_phase"])
        np.save(args.root/"best_panel_phase.npy", to_panel(saved["model_phase"]))
    selection = dict(label=selected, rule="lowest mean native core CV over 50 and 100 ms replay",
                     values=scores[selected], source_file=selected_row["file"], independent_test=False)
    (args.root/"selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    evidence = dict(root=str(args.root), stages=stages, replay=comparisons, release=release, selection=selection)
    (artifacts/"evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    lines = ["# 1200 CCD 像元方形：严格验收与非单调探索对照（2026-10-26）", "",
        "本轮检验拒绝 CV 恶化是否限制继续匀化。两组从完全相同的已有相位出发，使用同一冻结物理模型和 CCD 锚定可微优化；没有调用 GS，没有重新训练物理模型。", "",
        "## 实验口径", "",
        "Santec SLM #1、1064 nm、Daheng Mono12，满量程 4095。目标边长 1200 原生 CCD 像元，以 2.2 µm/像元计为 2.64 mm。相机窗口半宽 820 px，优化采样间隔 2 px；报告最终比较重新使用完整原生图像计算。核心向内缩进 8 px，排除零级中心 (1224, 962) 周围半径 60 px。零级没有被物理消除，核心外可能饱和，不能把此处结果解释为全光斑均匀度或准确效率。", "",
        "两组均固定 50 ms、每次采集 5 帧、16 块×20 次更新、学习率 0.03、相位步长 RMS 上限 0.3 rad；A–B–B–A 关闭，每块用同曝光顺序 A/B。两组按时间先后运行，未随机交错，因此不是排除漂移的因果试验。", "",
        "探索组对 ΔCV>0 的提议按 exp(−ΔCV/0.03) 接受，候选 CV 不得超过历史最佳的 1.25 倍，每 4 块恢复历史最佳相位，最终也恢复该相位。随机种子 20261026。严格组仅接受 CV 降低；探索温度默认 0，原有默认行为保持。", "",
        "## 实测过程", "", "| 策略 | 初始采样 CV | 最终采样 CV | 接受块数 | 接受恶化块数 |", "|---|---:|---:|---:|---:|"]
    for name, s in stages.items():
        lines.append(f"| {name} | {s['seed_cv']:.6f} | {s['final_cv']:.6f} | {s['accepted']}/16 | {s['accepted_uphill']} |")
    lines += ["", "![块级 CCD 轨迹](figures/trajectory_20261026.png)", "",
        "虚线是历史最低观测值，不能消除测量漂移；恢复同一相位后 CV 仍可能变化。严格组也可继续改善，不能将前一轮停滞直接归因于严格验收。", "",
        "## 相同曝光下精确相位重放", "",
        "除两组最终相位，还重放共同初相位、此前最佳相位及每组历史最低的已接受候选。每项均原样重放所保存相位，未重新叠加模型校正。下表为 5 帧均值图的原生核心 CV；逐帧 CV、均值信号和全部块级决定保存在 evidence.json。", "",
        "| 重放相位 | 曝光/ms | 原生核心 CV | 核心均值/counts | 饱和核心样本 |", "|---|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['label']} | {r['exposure_ms']:g} | {r['native_cv']:.6f} | {r['mean_counts']:.3f} | {r['saturated_core_samples']} |")
    gains = []
    for exposure in (50., 100.):
        previous = next(r for r in comparisons if r["exposure_ms"] == exposure and r["label"] == "mono12_dense_square_final")
        current = next(r for r in comparisons if r["exposure_ms"] == exposure and r["label"] == "nonmonotone_square_final")
        gains.append(f"{exposure:g} ms：{previous['native_cv']:.6f} → {current['native_cv']:.6f}，CV 降低 {100*(1-current['native_cv']/previous['native_cv']):.2f}%")
    lines += ["", "探索组最终相位相对此前最佳相位的同曝光重放结果为："+"；".join(gains)+"。这表示本轮保留相位的均匀度提高，不表示已排除时序漂移并证明策略的独立因果收益。"]
    lines += ["", "![同曝光 CCD 图像](figures/common_ccd_20261026.png)", "",
        "## 结论与后续选择", "",
        f"本次按 50/100 ms 原生核心 CV 的算术平均选择 `{selected}`；该相位已导出为原始弧度 `best_model_phase.npy` 和完整面板 `best_panel_phase.npy`。这两档曝光数据参与选择，不是新的独立测试集。", "",
        "允许暂时恶化在数学上可以跨越目标函数障碍；本实验提供可复现的概率接受机制及独立保留最佳相位，但单条轨迹不能证明跨越了局部极小值。应以上表同曝光重放判断策略收益，并区分初始、探索末端、历史最佳和最终导出相位。不能用运行中的最小 CV 代替最终重放结果。", "",
        "本轮严格组运行中的低 CV 未保持到重放，历史最佳候选也需要重新验收。这可能涉及入射状态漂移或相位加载响应的时间依赖，本轮未独立重测这些因素，不能确认具体来源。两组初始相位完全相同，但初始测得 CV 不同，也是时序混杂的证据；不可把全部重放差异归因于是否允许恶化。", "",
        "现有物理模型的宽区域绝对 CV 预测仍受加性背景（杂散光）项外推误差影响，故本轮使用 CCD 锚定的相对变化；模型梯度、测量漂移和可探索范围均可能造成停滞。后续优先在宽目标区域补充相位扰动数据、分离曝光偏置与空间背景，然后比较多个随机种子和交错重放。探索作为显式可选项，不因一次实验默认开启。", "",
        "## 复现与数据", "", f"原始数据目录：`{args.root.relative_to(ROOT).as_posix()}`。每组 config.json 记录几何、检查点哈希、初相位来源、全部策略参数；manifest.json 与 capture_records.pkl 记录采集；NPZ 保存完整原始帧、相位和掩膜；iterations.pkl 保存每次相位更新及 UTC、monotonic_ns、结束时间戳。时间戳是主机调用时间，不是传感器硬件触发时间。", "",
        "运行同一 validate 命令时，严格组不传探索参数；探索组添加以下参数，其余参数必须相同：", "",
        "```powershell", "--fixed-exposure-ms 50 --no-abba --ccd-check --accept-metric cv `", "--exploration-temperature .03 --exploration-budget .25 --exploration-window 4 --random-seed 20261026", "```", "",
        "固定曝光值只用于本次光路比较，并非跨设备或跨光路的永久限制。实际开始前仍须按当前画面选择未饱和档位。可使用 replay_phases 的 --include-best 同时比较历史最佳候选，避免把保留相位的漂移误认为策略失效。", "",
        "两组原始数据均通过指标重算与验收决策审计；探索组额外重放随机数，并检查最终相位和导出文件等于历史最佳相位。", "",
        f"两组共保存 {sum(s['iterations'] for s in stages.values())} 次相位更新；正式优化采集 {sum(s['audit']['raw_frames'] for s in stages.values())} 帧，统一重放 {sum(len(r['frame_cv']) for r in comparisons)} 帧，此外保留固定曝光检查原始帧。", "",
        f"最终设备释放记录（UTC）：`{release['timestamp_utc']}`，状态 `{release['status']}`。", "",
        "数据摘要：[evidence.json](artifacts/20261026/evidence.json)。前序实验：[1200 像元方形](../square1200/square1200_20261026.md)。", ""]
    (OUT/"exploration_20261026.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(stages={k:{f:v[f] for f in ("seed_cv", "final_cv", "accepted", "accepted_uphill")} for k,v in stages.items()}, replay=comparisons, release=release), indent=2))


if __name__ == "__main__":
    main()
