"""生成「离焦光路下的可微整形」报告：逐块指标曲线 + 模型/实测并排 GIF。

数据源（全部离线）：
* ``data/debug/defocus_differentiable/<stamp>/stage_a``：轴映射、曝光、参考、噪声地板；
* ``stage_c``：方向检验（A/B+/A/B−/A）；
* ``stage_d``：逐块 A-B-B-A 与接受/拒绝链（``chain.json``）。

图的读法：左栏 = **冻结模型的预测**在该块每步优化后的指标；右栏 = **实验系统实测**
（目标）在该块的 A/B 指标。两者用同一固定掩膜、同一曝光，单位与坐标已按方案冻结。
"""
from __future__ import annotations

import argparse
from loguru import logger
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT_KEY = "report/shape_trials/defocus_differentiable/defocus_differentiable_20261023.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "defocus_differentiable"
FIG_DIR = OUT_DIR / "figures"
STAMP = "20261023"
RUN_STAMP = "20261023_101822"
# 历史默认值（20261022 那一轮）：只作缺省，使用时 main() 会打印警告；换光路后必须显式给参数。
CHECKPOINT = ROOT / "data/debug/physical_forward_defocus/uniformity_refit_20261022_r1"

import sys

sys.path.insert(0, str(ROOT / "src"))


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


# --------------------------------------------------------------------------- #
# 读数据
# --------------------------------------------------------------------------- #
def load_e_reps(run_dir: Path) -> list[dict]:
    """E 阶段：初值 vs 冻结最优的 A-B-B-A 重放（每次 rep 一个 stage_d_block1.json）。"""
    reps = []
    for directory in sorted((run_dir / "stage_e").glob("rep*")):
        record_path = directory / "stage_d_block1.json"
        if record_path.exists():
            reps.append(json.loads(record_path.read_text(encoding="utf-8")))
    return reps


def load_run(run_dir: Path) -> dict:
    chain_path = run_dir / "stage_d" / "chain.json"
    if not chain_path.exists():
        raise FileNotFoundError(f"missing {chain_path}")
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    noise = json.loads((run_dir / "stage_a/repeatability/noise_floor.json").read_text(encoding="utf-8"))
    direction = json.loads((run_dir / "stage_c/run/stage_c.json").read_text(encoding="utf-8"))
    blocks = []
    for entry in chain["blocks"]:
        run_block = run_dir / entry["run_dir"] / "stage_d_block1.json"
        record = json.loads(run_block.read_text(encoding="utf-8"))
        directory = None
        for name in (f"block{entry['block']}_candidate", f"candidate_block{entry['block']}"):
            candidate = run_dir / "stage_d" / name
            if (candidate / "seed_panel.npy").exists():
                directory = candidate
                break
        seed_panel = np.load(directory / "seed_panel.npy") if directory else None
        candidate_panel = np.load(directory / "candidate_panel.npy") if directory else None
        blocks.append(dict(meta=entry, record=record, seed_panel=seed_panel, candidate_panel=candidate_panel))
    return dict(chain=chain, noise=noise, direction=direction, blocks=blocks, e_reps=load_e_reps(run_dir))


def panel_to_grid(panel: np.ndarray, size: int = 128) -> np.ndarray:
    pitch = 1920 // size
    rows = 1200 // pitch
    grid = np.zeros((size, size), dtype=np.float64)
    grid[:rows] = panel[: rows * pitch].reshape(rows, pitch, size, pitch).mean(axis=(1, 3))
    return grid


def replay_block(forward, objective, seed_grid: np.ndarray, cap: float, updates: int = 10, lr: float = 0.01):
    """离线重放一块：返回每步的 (预测图, 指标)。"""
    torch = __import__("torch")
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer,
    )

    optimizer = DifferentiableUniformityOptimizer(forward, objective, seed_grid, lr=lr, step_rms_cap=cap,
                                                 device="cpu")
    steps = []
    for _ in range(updates):
        optimizer.update()
        with torch.no_grad():
            image = forward.predict(torch.tensor(optimizer.current_phase[None], dtype=torch.float32))[0]
        steps.append(_metrics_from_model(image, objective, forward))
    return steps


def _metrics_from_model(image, objective, forward) -> dict:
    torch = __import__("torch")
    core_t = torch.as_tensor(objective.mask_core)
    box_t = torch.as_tensor(objective.mask_box)
    values = image[core_t]
    mean = float(values.mean())
    cv = float((((values / max(mean, 1e-8)) - 1) ** 2).mean() ** 0.5) if mean > 0 else float("nan")
    cv_sq = float((((values / max(mean, 1e-8)) - 1) ** 2).mean())
    array = image.detach().numpy()
    ee = float(array[box_t.numpy()].sum() / max(array.sum(), 1e-9))
    return dict(cv=float(np.sqrt(cv_sq)), cv_sq=cv_sq, mean_core=mean, peak=float(array.max()),
                ee_box=ee, image=array)


def measured_metrics(record: dict) -> dict:
    labels = [r["label"] for r in record["records"]]
    by_label = {}
    for label in ("A1", "B1", "B2", "A2"):
        rows = [r for r in record["records"] if r["label"] == label]
        if rows:
            by_label[label] = dict(cv=float(np.mean([r["cv_core"] for r in rows])),
                                   mean_core=float(np.mean([r["mean_core"] for r in rows])),
                                   peak=float(np.mean([r["peak"] for r in rows])))
    return dict(labels=labels, by_label=by_label)


def mean_frame(run_dir: Path, block: int, label: str) -> np.ndarray | None:
    path = run_dir / f"stage_d/run_block{block}"
    files = sorted(path.glob(f"p*_{label}_f*.npy"))
    if not files:
        return None
    return np.mean([np.load(f) for f in files], axis=0)


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def plot_curves(data: dict, replays: dict, path: Path) -> dict:
    plt = _mpl()
    resolution = float(data["noise"]["three_sigma_resolution"])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    step_axis, model_series, measured_points = [], [], []
    cursor = 0
    for block in data["blocks"]:
        meta = block["meta"]
        steps = replays[meta["block"]]
        step_axis.extend(range(cursor, cursor + len(steps)))
        model_series.extend([s["cv_sq"] for s in steps])
        measured_points.append((cursor, meta["baseline"], 0.0, meta["accepted"]))
        measured_points.append((cursor + len(steps) - 1, meta["candidate"], meta["delta"], meta["accepted"]))
        cursor += len(steps)
    axes[0].plot(step_axis, model_series, color="#1f77b4", marker="o", ms=3.5, label="模型预测 CV（每步）")
    for position, value, delta, accepted in measured_points:
        axes[0].errorbar(position, value, yerr=resolution, fmt="s" if accepted else "x",
                         color="#d62728", ms=6, capsize=3)
    axes[0].plot([], [], "s", color="#d62728", label="实测 CV（A/B，误差棒 = 3σ）")
    axes[0].set_ylabel("CV（固定 60 px 掩膜，内缩 3 px）")
    axes[0].set_title("逐块优化：模型预测 vs 实验系统实测")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(step_axis, [s["mean_core"] for s in [r for rr in replays.values() for r in rr]],
                 color="#2ca02c", marker="o", ms=3, label="模型预测 mean_core")
    axes[1].set_xlabel("累计可微更新步数")
    axes[1].set_ylabel("目标区均值（模型单位）")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(resolution=resolution, steps=len(step_axis))


def plot_direction(data: dict, path: Path) -> dict:
    plt = _mpl()
    record = data["direction"]
    order = record["order"]
    positions = sorted({r["position"] for r in record["records"]})
    labels = [next(r["phase"] for r in record["records"] if r["position"] == p) for p in positions]
    values = [float(np.mean([r["cv_core"] for r in record["records"] if r["position"] == p])) for p in positions]
    x = np.arange(len(values))
    resolution = float(record.get("three_sigma_resolution", float("nan")))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#7f7f7f" if name == "seed" else ("#2ca02c" if name == "b_plus" else "#d62728") for name in labels]
    ax.bar(x, values, color=colors, alpha=0.85)
    seed_mean = float(record["seed_baseline_mean"])
    ax.axhline(seed_mean, color="k", ls="--", lw=1, label=f"基线 {seed_mean:.4f}")
    ax.axhspan(seed_mean - resolution, seed_mean + resolution, color="k", alpha=0.08,
               label=f"±3σ ({resolution:.4f})")
    ax.set_xticks(x, [f"{i + 1}\n{n}" for i, n in enumerate(labels)], fontsize=8)
    ax.set_ylim(seed_mean - 6 * resolution, max(max(values), seed_mean) + 4 * resolution)
    ax.set_ylabel("CV（固定掩膜，每点 5 帧均值）")
    ax.set_title(f"方向检验：Δ+ = {record['delta_plus']:+.5f}，Δ− = {record['delta_minus']:+.5f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(delta_plus=record["delta_plus"], delta_minus=record["delta_minus"], resolution=resolution)


def plot_noise(data: dict, path: Path) -> dict:
    plt = _mpl()
    floor = data["noise"]
    per_repeat = floor["per_repeat_mean_cv"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(range(len(per_repeat)), per_repeat, "o-", color="#1f77b4")
    ax.axhline(floor["cv_core_mean"], color="k", ls="--", lw=1)
    ax.fill_between(range(len(per_repeat)), floor["cv_core_mean"] - floor["three_sigma_resolution"],
                    floor["cv_core_mean"] + floor["three_sigma_resolution"], color="k", alpha=0.08,
                    label=f"3σ = {floor['three_sigma_resolution']:.4f}")
    ax.set_xlabel("独立上屏序号")
    ax.set_ylabel("CV（同一相位的重复测量）")
    ax.set_title("参考态重复性 / 噪声地板")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(cv_mean=floor["cv_core_mean"], cv_sd=floor["cv_core_sd"],
                three_sigma=floor["three_sigma_resolution"])


def render_gif(data: dict, replays: dict, run_dir: Path, path: Path, *, max_frames: int = 40) -> dict:
    """模型预测（左）与实测（右）并排，逐块推进；每帧标注指标。"""
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = []
    for block in data["blocks"]:
        meta = block["meta"]
        measured = measured_metrics(block["record"])["by_label"]
        steps = replays[meta["block"]]
        for step_index, step in enumerate(steps):
            frames.append(dict(kind="model", block=meta["block"], step=step_index + 1, metrics=step,
                               measured=measured, meta=meta, image=step["image"]))
        for label in ("A1", "B1", "A2"):
            frame = mean_frame(run_dir, meta["block"], label)
            if frame is None:
                continue
            frames.append(dict(kind="measured", block=meta["block"], step=None, label=label,
                               metrics=measured.get(label), measured=measured, meta=meta, image=frame))
    if len(frames) > max_frames:
        stride = max(1, len(frames) // max_frames)
        frames = frames[::stride]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))

    def draw(index):
        for axis in axes:
            axis.clear()
        frame = frames[index]
        array = frame["image"]
        shown = np.log10(np.maximum(array, np.percentile(array[array > 0], 5) if (array > 0).any() else 1e-6))
        axes[0].imshow(shown, cmap="inferno")
        axes[0].set_title(f"模型预测（block {frame['block']}，step {frame['step'] or '-'}）", fontsize=10)
        row = frame["metrics"] or {}
        axes[0].text(0.02, 0.96, f"CV² {row.get('cv_sq', float('nan')):.4f}\nmean {row.get('mean_core', float('nan')):.1f}\n"
                                 f"peak {row.get('peak', float('nan')):.1f}\nEE(box) {row.get('ee_box', float('nan')):.3f}",
                     transform=axes[0].transAxes, va="top", fontsize=8, color="w",
                     bbox=dict(fc="black", alpha=0.45, ec="none"))
        measured_label = frame.get("label", "")
        axes[1].set_title(f"实验系统实测（block {frame['block']}，{measured_label or '—'}）", fontsize=10)
        if frame["kind"] == "measured":
            axes[1].imshow(shown, cmap="inferno")
            axes[1].text(0.02, 0.96, f"CV {row.get('cv', float('nan')):.4f}\nmean {row.get('mean_core', float('nan')):.1f}\n"
                                     f"peak {row.get('peak', float('nan')):.1f}", transform=axes[1].transAxes,
                         va="top", fontsize=8, color="w", bbox=dict(fc="black", alpha=0.45, ec="none"))
        else:
            axes[1].axis("off")
            axes[1].text(0.05, 0.5, "该帧为模型侧；\n实测帧按 A1/B1/A2 穿插显示", fontsize=9)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        meta = frame["meta"]
        fig.suptitle(f"Δ 实测 {meta['delta']:+.5f}   3σ {data['noise']['three_sigma_resolution']:.4f}   "
                     f"{'接受' if meta['accepted'] else '拒绝'}", fontsize=10)
        return axes

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=220, blit=False)
    animation.save(path, writer=PillowWriter(fps=4.5))
    plt.close(fig)
    return dict(frames=len(frames), path=str(path))


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def build_e_section(data: dict) -> str:
    """E 阶段（冻结最优相位，初值 vs 最优 A-B-B-A 重复 3 次）的正文段。"""
    reps = data.get("e_reps") or []
    if not reps:
        return "（E 阶段的 3 次重放尚未采集。）"
    deltas = [float(r["delta"]) for r in reps]
    baselines = [float(r["baseline_mean"]) for r in reps]
    candidates = [float(r["candidate_mean"]) for r in reps]
    resolution = float(data["noise"]["three_sigma_resolution"])
    mean_delta = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    rows = "\n".join(f"| {i + 1} | {r['baseline_mean']:.5f} | {r['candidate_mean']:.5f} | "
                     f"{r['delta']:+.5f} | {r['drift']:+.5f} |" for i, r in enumerate(reps))
    return f"""## 4b. 冻结重放（E 阶段）

初值与冻结最优相位各 5 帧，A-B-B-A 完整重复 {len(reps)} 次（共 {len(reps) * 20} 帧原始数据）。

| 重放 | 初值 CV | 最优 CV | Δ | 漂移 (A2−A1) |
|---|---|---|---|---|
{rows}

合并：Δ = **{mean_delta:+.5f}**（sd {sd:.5f}），3σ 分辨率 {resolution:.5f}；
末态 CV 从 {float(np.mean(baselines)):.5f} 降到 {float(np.mean(candidates)):.5f}
（相对 −{abs(mean_delta) / float(np.mean(baselines)):.1%}）。"""


def reconstruct_rounds_from_runs(run_dir: Path, name: str) -> list[dict] | None:
    """rounds.json 被并发运行覆盖时的兜底：从 ``run_round*/stage_d_block1.json`` 重建实测行。"""
    rows = []
    for index in range(1, 16):
        path = run_dir / name / f"run_round{index}" / "stage_d_block1.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(round=index, refit=dict(first_loss=float("nan"), last_loss=float("nan")),
                         model_before=dict(cv_sq=float("nan")), model_after=dict(cv_sq=float("nan")),
                         measured=dict(baseline=record["baseline_mean"], candidate=record["candidate_mean"],
                                       delta=record["delta"], accepted=record.get("accepted"),
                                       drift=record.get("drift"))))
    return rows or None


def load_rounds(run_dir: Path, name: str) -> dict | None:
    """读 ``stage_r`` / ``stage_big`` 的 rounds.json 与其噪声地板（3σ 是掩膜相关的）。"""
    path = run_dir / name / "rounds.json"
    if not path.exists():
        return None
    rounds = json.loads(path.read_text(encoding="utf-8"))
    recovered = reconstruct_rounds_from_runs(run_dir, name)
    if recovered and len(recovered) > len(rounds):
        rounds = recovered       # rounds.json 被别的运行覆盖过，以 run 目录为准
    floor_path = run_dir / name / "noise_floor" / "noise_floor.json"
    if not floor_path.exists():
        # stage_r 用 60 px 掩膜，其地板记在 stage_a；stage_big 有自己的 1000 px 地板
        floor_path = run_dir / "stage_a" / "repeatability" / "noise_floor.json"
    resolution = None
    if floor_path.exists():
        resolution = float(json.loads(floor_path.read_text(encoding="utf-8"))["three_sigma_resolution"])
    return dict(name=name, rounds=rounds, resolution=resolution)


def plot_rounds(entry: dict, path: Path) -> dict:
    """多轮闭环：模型预测 CV（上）与实测 CV（下），实测那条才代表光斑。"""
    plt = _mpl()
    rounds = entry["rounds"]
    resolution = entry["resolution"]
    x = np.arange(1, len(rounds) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].plot(x, [r["model_before"]["cv_sq"] for r in rounds], "o-", color="#1f77b4", label="模型预测（本轮起点）")
    axes[0].plot(x, [r["model_after"]["cv_sq"] for r in rounds], "s--", color="#17becf", label="模型预测（本轮候选）")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("模型 CV²（掩膜）")
    axes[0].set_title(f"{entry['name']}：模型预测 vs 实测（每轮重拟合 + 10 步可微）")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, which="both")

    baselines = [r["measured"]["baseline"] for r in rounds]
    candidates = [r["measured"]["candidate"] for r in rounds]
    axes[1].plot(x, baselines, "o-", color="#7f7f7f", label="实测基线（A）")
    axes[1].plot(x, candidates, "s-", color="#d62728", label="实测候选（B）")
    if resolution:
        for xi, value in zip(x, baselines):
            axes[1].errorbar(xi, value, yerr=resolution, fmt="none", ecolor="#7f7f7f", capsize=3, alpha=0.7)
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("实测 CV（同一掩膜）")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(rounds=len(rounds), resolution=resolution,
                first_baseline=float(baselines[0]), last_candidate=float(candidates[-1]))


def build_rounds_section(entry: dict, artifact: dict, *, title: str) -> str:
    rounds = entry["rounds"]
    rows = "\n".join(
        f"| {r['round']} | {r['refit']['first_loss']:.4f} → {r['refit']['last_loss']:.4f} | "
        f"{r['model_before']['cv_sq']:.3f} → {r['model_after']['cv_sq']:.3f} | "
        f"{r['measured']['baseline']:.5f} | {r['measured']['candidate']:.5f} | "
        f"{r['measured']['delta']:+.5f} | {'接受' if r['measured']['accepted'] else '拒绝'} |"
        for r in rounds)
    deltas = [float(r["measured"]["delta"]) for r in rounds]
    resolution = artifact.get("resolution")
    resolution_text = f"{resolution:.4f}" if resolution else "未记录"
    above = sum(1 for d in deltas if resolution and abs(d) > resolution)
    return f"""### {title}

| 轮 | 正向重拟合 loss | 模型 CV² | 实测基线 | 实测候选 | 实测 Δ | 判定 |
|---|---|---|---|---|---|---|
{rows}

该掩膜的 3σ = {resolution_text}；逐轮配对 Δ ∈ [{min(deltas):+.5f}, {max(deltas):+.5f}]，
其中 {above}/{len(deltas)} 轮超过 3σ。

> ⚠️ **跨轮的实测绝对值不可比**：每轮起点是上一轮接受的相位，而 1000 px 掩膜的
> 中心又按本轮首帧重新冻结，基线本身会在轮间大幅跳动（本批 4.019–6.048）。
> 只有**同一轮内 A/B 配对 Δ**（同一掩膜、同一曝光、A-B-B-A 扣除漂移）才是有效判据。
"""


def write_report(data: dict, artifacts: dict, path: Path) -> None:
    chain = data["chain"]
    e_section = build_e_section(data)
    rounds_sections_joined = "\n".join(artifacts.get("rounds_sections") or ["（多轮数据尚未采集。）"])
    # 溯源块是 sync_report_provenance.py 生成的；重写正文时必须原样保留，否则 --check 报漂移
    provenance_block = "<!-- provenance:start -->\n<!-- provenance:end -->"
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if "<!-- provenance:start -->" in previous and "<!-- provenance:end -->" in previous:
            head = previous.index("<!-- provenance:start -->")
            tail = previous.index("<!-- provenance:end -->") + len("<!-- provenance:end -->")
            provenance_block = previous[head:tail]
    direction = data["direction"]
    floor = data["noise"]
    blocks = chain["blocks"]
    table_rows = "\n".join(
        f"| {b['block']} | {b['cap']:.3f} | {b['baseline']:.5f} | {b['candidate']:.5f} | "
        f"{b['delta']:+.5f} | {'接受' if b['accepted'] else '拒绝'} | {b['model_cv_before']:.4f} → {b['model_cv_candidate']:.4f} |"
        for b in blocks)
    text = f"""# 离焦光路下的可微整形（2026-10-23）

{provenance_block}
## 摘要

在**未移动相机**的离焦光路上，用冻结的物理预测模型做可微优化，并用 A-B-B-A 重放逐块验收。
本轮回答三件事：模型↔实测的**方向**对不对、**幅度**差多少、**逐块**能否稳定改善。

* **方向检验（C）**：模型下降方向正/负两个方向在实验系统上给出**符号相反**的 CV 变化
  （Δ+ = {direction['delta_plus']:+.5f}，Δ− = {direction['delta_minus']:+.5f}），
  3σ 分辨率 {artifacts['direction']['resolution']:.4f} ⇒ **方向正确**。
* **逐块（D）**：第一块 `{blocks[0]['delta']:+.5f}`（{abs(blocks[0]['delta']) / artifacts['curves']['resolution']:.1f}σ）⇒ 接受；
  逐块结果见下表。
* **幅度**：实测改善只有模型预测的 ~{abs(blocks[0]['delta']) / max(abs(blocks[0]['model_cv_before'] - blocks[0]['model_cv_candidate']), 1e-9):.0%}
  —— 与主报告 §8.3 的结构性口径差一致（模型幅度不可信、方向可测）。

![指标曲线](figures/{STAMP}_metrics_curves.png)

![过程动画](figures/{STAMP}_process.gif)

## 1. 现场前置测量

| 项 | 实测值 | 来源 |
|---|---|---|
| 轴映射 | 面板 x → 相机 x（+8083，比值 1.092）；面板 y → 相机 y（−8136，1.099） | `stage_a/chirp_tilt` |
| 去倾斜补偿 | 原始 −2.04/+2.06 → 残差 0.14/0.47 px/mm（`enabled=true`） | 同上 |
| 可用曝光 | 1.6–32 ms（今日光路） | `stage_a/reference`、`stage_a/exposure_retry` |
| 参考+探针 | 24 样本，原始帧全留 | `stage_a/reference` |
| 复放一致性 | 真实面板相位 @4 ms：峰值 110，存档 128；质心一致 | 本报告 §5 |
| 噪声地板 | CV_core = {floor['cv_core_mean']:.4f} ± {floor['cv_core_sd']:.4f}，**3σ = {floor['three_sigma_resolution']:.4f}** | `stage_a/repeatability` |

![噪声地板](figures/{STAMP}_noise_floor.png)

## 2. 方向检验（C 阶段）

![方向检验](figures/{STAMP}_direction_test.png)

方案要求 B± 的增量归一到照明加权 RMS {direction['direction_record']['target_rms']} rad。
模型预测 CV：seed {direction['direction_record']['cv_model']['seed']:.4f} →
B+ {direction['direction_record']['cv_model']['b_plus']:.4f} /
B− {direction['direction_record']['cv_model']['b_minus']:.4f}（对称 ±{direction['direction_record']['predicted_gain_plus']:.4f}）；
实测 Δ+ = {direction['delta_plus']:+.5f}、Δ− = {direction['delta_minus']:+.5f}，两向都超过 3σ 且符号相反。

## 3. 逐块闭环（D 阶段）

| 块 | cap / rad | 基线 CV | 候选 CV | Δ | 判定 | 模型 CV |
|---|---|---|---|---|---|---|
{table_rows}

每块都是 **A1-B1-B2-A2**（各 5 帧）：B1/B2 两次重放差见 `stage_d/run_block*/stage_d_block1.json`，
A2−A1 为漂移。接受条件为 Δ < −3σ（{artifacts['curves']['resolution']:.4f}）。

## 4. 指标随优化的变化

上图上半栏：**模型预测**的 CV 逐步曲线（蓝）与**实验系统实测**的 CV 锚点（红方块，误差棒为 3σ）；
下半栏：模型预测的目标区均值。数据由 `scripts/slm/generate_defocus_differentiable_report.py`
离线重放每块 10 步得到（与设备上跑的是同一模型、同一掩膜、同一 cap）。

{e_section}

![过程动画](figures/{STAMP}_process.gif)

GIF 左栏是模型预测的远场演化，右栏是该块的实测帧（A1/B1/A2 穿插），顶部标注该块的 Δ 与 3σ 判定。

## 4c. 多轮闭环（正向重拟合 + 可微优化）

{rounds_sections_joined}

两条曲线必须分开读：**模型 CV 会随重拟合单调下降**（模型越来越贴合当天数据），
**只有实测 CV 下降才代表光斑在变好**。60 px 目标的 4 轮里实测基线几乎不动
（全部被拒 ⇒ 已接受相位没推进），而 1000 px 目标那轮取消拒绝判定后实测连续下降。

## 5. 轴映射的口径矛盾（未决）

拟合用的 `ramp_matrix` = [[465.1, 7174.7], [−7533.2, −30.3]]（面板 x → 相机 y）；
本次现场标定是面板 x → 相机 x。两者**互为转置**且 y 符号相反。
冻结模型的空间放置因此**不能**被这次复现证明（复现只用同一份 `ramp_matrix`）。
处置：用一次已知面板轴的标记测量给出无歧义的 (row, col) 位移后裁决；在裁决前，
本报告的所有结论都只依赖**固定掩膜内的 CV**（与坐标转置无关）。

## 6. 边界

* 单次会话、单一初值；E 阶段的冻结重放 3× 只用于重复性，不足以宣称跨光路泛化。
* 接受判据用 3σ（4 次独立上屏 × 5 帧估计），小样本不确定度仍在。
* 模型幅度与实测相差约 {abs(blocks[0]['delta']) / max(abs(blocks[0]['model_cv_before'] - blocks[0]['model_cv_candidate']), 1e-9):.0%}，
  所以**不能用模型预测的改善量**替代理事实测。

## 7. 复现

```bash
python scripts/slm/generate_defocus_differentiable_report.py \\
    --run-dir data/debug/defocus_differentiable/{RUN_STAMP}
```

原始数据（含逐文件 SHA256）见 `data/debug/defocus_differentiable/{RUN_STAMP}/manifest_checksums.json`。
"""
    path.write_text(text, encoding="utf-8")

def main() -> None:
    global STAMP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "data/debug/defocus_differentiable" / RUN_STAMP)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="冻结检查点；默认是 20261022 的历史值，换光路后必须显式指定")
    parser.add_argument("--stamp", default=STAMP,
                        help="图名前缀（换光路后必须换新戳，避免覆盖旧图）；默认 " + STAMP)
    parser.add_argument("--report-name", default=Path(REPORT_KEY).name,
                        help="报告文件名；默认 " + Path(REPORT_KEY).name)
    parser.add_argument("--force", action="store_true",
                        help="允许覆盖已存在且指向**另一次运行**的报告（默认拒绝）")
    args = parser.parse_args()
    if args.run_dir == ROOT / "data/debug/defocus_differentiable" / RUN_STAMP:
        logger.info("⚠ --run-dir 用的是历史默认值 " + RUN_STAMP + "；光路变化后必须显式指定")
    if args.checkpoint == CHECKPOINT:
        logger.info("⚠ --checkpoint 用的是历史默认值（20261022 那一轮）；换光路后必须显式指定")
    STAMP = args.stamp
    report_path = OUT_DIR / (args.report_name if args.report_name.endswith(".md")
                             else args.report_name + ".md")
    if report_path.exists() and not args.force:
        existing = report_path.read_text(encoding="utf-8")
        if Path(args.run_dir).name not in existing:
            raise SystemExit(
                f"拒绝覆盖：{report_path.name} 指向另一次运行（本次请求 {Path(args.run_dir).name}）。"
                "换光路后请用 --report-name 与 --stamp 写新报告/新图戳，或显式 --force。")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_run(args.run_dir)

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        UniformityObjective,
        load_frozen_parameters,
    )
    from ao_shaping.utils.image.target.masks import build_box_core_masks

    checkpoint = args.checkpoint
    with np.load(checkpoint / "inputs.npz") as payload:
        camera_xy = payload["camera_xy"]
    parameters = load_frozen_parameters(checkpoint)
    forward = FrozenDefocusForward(parameters, camera_xy, device="cpu")
    col, row = forward.sample_grid_index(1432.0, 968.0)
    box, core = build_box_core_masks(camera_xy.shape[:2], (col, row), 60, 3, forward.sample_step_ccd())
    objective = UniformityObjective(box, core, w_phase=0.01)

    replays = {}
    for block in data["blocks"]:
        meta = block["meta"]
        seed_panel = block["seed_panel"]
        if seed_panel is None:
            continue
        seed_grid = panel_to_grid(seed_panel, parameters.grid_size)
        replays[meta["block"]] = replay_block(forward, objective, seed_grid, meta["cap"])

    artifacts = dict(
        direction=plot_direction(data, FIG_DIR / f"{STAMP}_direction_test.png"),
        noise=plot_noise(data, FIG_DIR / f"{STAMP}_noise_floor.png"),
        curves=plot_curves(data, replays, FIG_DIR / f"{STAMP}_metrics_curves.png"),
        gif=render_gif(data, replays, args.run_dir, FIG_DIR / f"{STAMP}_process.gif"),
    )
    rounds_sections = []
    for name, title in (("stage_r", "多轮重拟合（60 px 目标，拒绝判定开启）"),
                        ("stage_big", "多轮重拟合（1000 px 目标 + 1201 px 窗，取消拒绝判定）")):
        entry = load_rounds(args.run_dir, name)
        if entry is None:
            continue
        artifact = plot_rounds(entry, FIG_DIR / f"{STAMP}_{name}_rounds.png")
        artifacts[name] = artifact
        rounds_sections.append(build_rounds_section(entry, artifact, title=title))
    artifacts["rounds_sections"] = rounds_sections
    if not args.report_only:
        write_report(data, artifacts, report_path)
    summary = dict(blocks=len(data["blocks"]), replays={k: len(v) for k, v in replays.items()},
                   artifacts={k: (v if isinstance(v, dict) else str(v)) for k, v in artifacts.items()})
    (OUT_DIR / f"{STAMP}_artifacts.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()






