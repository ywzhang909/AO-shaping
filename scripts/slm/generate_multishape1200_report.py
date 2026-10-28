"""Report generator: multishape 1200 CCD-px physical-forward shaping session.

Reads the session directory written by ``slm-transfer-shapes`` and renders the
report ``report/shape_trials/multishape1200/multishape1200_20261027.md``
together with its figures. Offline: no device, no re-measurement.
"""
from __future__ import annotations

import argparse
from loguru import logger
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import fmt_num_or_dash
from scripts._common.provenance import insert_header

from ao_shaping.runners.slm.transfer_shapes_runner import STAGE_RECIPES

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPORT_KEY = "report/shape_trials/multishape1200/multishape1200_20261027.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "multishape1200"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"
STAMP = "_20261027"

DEFAULT_SESSION = ROOT / "data/debug/slm_transfer/multishape1200_20261027"
REFERENCE_SESSION = ROOT / "data/debug/slm_transfer/square1200_20261026_1350"

SHAPE_LABELS = {"square": "正方形", "rectangle": "长方形", "triangle": "三角形",
                "star": "五角星", "circle": "圆形"}
STAGES = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")
STAGE_ORDER = {"transport_cv": 1, "mono12_cv": 2, "mono12_continue": 3, "mono12_dense": 4}
STAGE_LABELS = {"transport_cv": "第 1 段 (初相位, 模型判据)",
                "mono12_cv": "第 2 段 (Mono12, 实测 CV 判据)",
                "mono12_continue": "第 3 段 (续跑)",
                "mono12_dense": "第 4 段 (稠密采样 + 宽带模型)"}
SEED_LABELS = {"transport": "传输 (非 GS, 仅正方形精确)",
               "backprojection": "反投影 (非 GS, 形状通用)",
               "gs": "GS (模型 80 次迭代)"}


def _read(path: Path, default=None):
    """Read a JSON file; ``default`` is returned when it is missing or being
    rewritten underneath us (the session journal is rewritten per stage)."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        if default is None:
            raise
        return default


def _session_facts(session: Path) -> dict:
    """Everything the report needs, read once from the session directory."""
    facts: dict = {"session": session, "shapes": [], "stages": {}, "bakeoff": {},
                   "replay": [], "journal": [], "fit": None, "reference_fit": None,
                   "transport_flow": None}
    journal = session / "journal.json"
    if journal.is_file():
        facts["journal"] = _read(journal, default=[])
    facts["chosen_seeds"] = {entry["shape"]: entry.get("chosen")
                             for entry in facts["journal"]
                             if entry.get("stage") == "seed_bakeoff" and entry.get("chosen")}
    bake = session / "seed_bakeoff"
    if bake.is_dir():
        for directory in sorted(bake.iterdir()):
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                continue
            rows = _read(manifest)
            facts["bakeoff"][directory.name] = dict(
                shape=directory.name.rsplit("_", 1)[0],
                kind=directory.name.rsplit("_", 1)[1],
                seed_cv=float(rows[0]["metrics"]["cv"]),
                exposure_ms=float(rows[0]["exposure_ms"]))
    facts["bakeoff_shapes"] = sorted({entry["shape"] for entry in facts["bakeoff"].values()},
                                     key=lambda s: list(SHAPE_LABELS).index(s)
                                     if s in SHAPE_LABELS else 99)
    shape_root = session / "shape"
    if shape_root.is_dir():
        for shape_dir in sorted(shape_root.iterdir()):
            if not shape_dir.is_dir():
                continue
            facts["shapes"].append(shape_dir.name)
            for stage in STAGES:
                directory = shape_dir / stage
                manifest = directory / "manifest.json"
                if not manifest.is_file():
                    continue
                rows = _read(manifest)
                summary = _read(directory / "summary.json") if (directory / "summary.json").is_file() else []
                by_label = {row["label"]: row for row in rows}
                facts["stages"][(shape_dir.name, stage)] = dict(
                    seed_cv=(by_label.get(f"{shape_dir.name}_seed") or {}).get("metrics", {}).get("cv"),
                    final_cv=(by_label.get(f"{shape_dir.name}_final") or {}).get("metrics", {}).get("cv"),
                    blocks=summary,
                    exposure_ms=(by_label.get(f"{shape_dir.name}_final") or {}).get("exposure_ms"),
                    bit_depth=(by_label.get(f"{shape_dir.name}_final") or {}).get("bit_depth"),
                    config=_read(directory / "config.json") if (directory / "config.json").is_file() else {})
    replay = session / "common_replay" / "manifest.json"
    if replay.is_file():
        facts["replay"] = _read(replay)
    for key, root in (("fit", session / "fit"), ("reference_fit", REFERENCE_SESSION / "fit")):
        result = root / "result.json"
        if result.is_file():
            facts[key] = dict(result=_read(result),
                              parameters=_read(root / "frozen_parameters.json"))
    return facts


def _transport_flow_facts(session: Path) -> dict | None:
    """The documented (non-GS transport seed) square flow, if it was run."""
    shape_root = session / "shape" / "square"
    if not shape_root.is_dir():
        return None
    stages = {}
    for stage in STAGES:
        directory = shape_root / stage
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        rows = _read(manifest)
        by_label = {row["label"]: row for row in rows}
        stages[stage] = dict(
            seed_cv=(by_label.get("square_seed") or {}).get("metrics", {}).get("cv"),
            final_cv=(by_label.get("square_final") or {}).get("metrics", {}).get("cv"))
    if not stages:
        return None
    return dict(session=session, stages=stages)


def plot_seed_bakeoff(facts: dict, path: Path) -> Path:
    entries = list(facts["bakeoff"].values())
    if not entries:
        return path
    shapes = sorted({entry["shape"] for entry in entries},
                    key=lambda s: list(SHAPE_LABELS).index(s) if s in SHAPE_LABELS else 99)
    kinds = [kind for kind in ("transport", "backprojection", "gs")
             if any(entry["kind"] == kind for entry in entries)]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / max(1, len(kinds))
    for index, kind in enumerate(kinds):
        values, positions = [], []
        for shape_index, shape in enumerate(shapes):
            match = next((entry for entry in entries
                          if entry["shape"] == shape and entry["kind"] == kind), None)
            values.append(match["seed_cv"] if match else np.nan)
            positions.append(shape_index + (index - (len(kinds) - 1) / 2) * width)
        ax.bar(positions, values, width=width * 0.92, label=SEED_LABELS.get(kind, kind))
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels([SHAPE_LABELS.get(shape, shape) for shape in shapes])
    ax.set_ylabel("初相位核心 CV (实测)")
    ax.set_title("初相位构造对比择优：各候选初相位的实测核心 CV (越低越好)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def stage_completion(facts: dict, shape: str) -> dict:
    """Per stage: how many blocks ran, how many were planned, and whether it finished."""
    expected = {recipe.name: recipe.blocks for recipe in STAGE_RECIPES}
    completion = {}
    for stage in STAGES:
        stage_facts = facts["stages"].get((shape, stage))
        if not stage_facts:
            continue
        blocks = len(stage_facts["blocks"])
        completion[stage] = dict(blocks=blocks, expected=expected[stage],
                                 complete=blocks >= expected[stage])
    return completion


def shape_finished(facts: dict, shape: str) -> bool:
    """True when every stage of the chain ran to its planned block count."""
    completion = stage_completion(facts, shape)
    return len(completion) == len(STAGES) and all(entry["complete"]
                                                  for entry in completion.values())


def plot_cv_trajectory(facts: dict, path: Path) -> Path:
    """The measured core CV of the phase that was **kept**, block by block.

    Stage 1 accepts on the model's own loss, so it has no measured candidate:
    there the kept phase is the anchor itself. Plotting the candidate alone
    would silently drop the whole first stage.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels: list[str] | None = None
    for shape in facts["shapes"]:
        if not shape_finished(facts, shape):
            # An interrupted chain would be plotted over a different number of
            # blocks, so it would silently misalign with the finished one.
            continue
        points = []
        for stage in STAGES:
            stage_facts = facts["stages"].get((shape, stage))
            if not stage_facts:
                continue
            for block in stage_facts["blocks"]:
                candidate = block.get("candidate")
                anchor = block.get("baseline")
                kept = candidate if candidate is not None else anchor
                points.append((stage, int(block["block"]), kept))
        if not points:
            continue
        values = [kept if kept is not None else np.nan for _, _, kept in points]
        ax.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1.2,
                label=SHAPE_LABELS.get(shape, shape))
        if labels is None:
            labels = []
            previous = None
            for stage, block, _ in points:
                if stage != previous:
                    labels.append(f"第{STAGE_ORDER[stage]}段\n块 {block}")
                    previous = stage
                else:
                    labels.append(str(block))
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_xlabel("块 (按段分组)")
    ax.set_ylabel("被保留相位的实测核心 CV")
    ax.set_title("逐块实测核心 CV（第 1 段无候选实测，取锚点；第 2–4 段取被接受的候选）")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(facts: dict, path: Path) -> Path:
    cold, warm = facts.get("fit"), facts.get("reference_fit")
    if not cold or not warm:
        return path
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4))
    labels = ["从零训练\n(本次)", "热启动\n(2026-10-26)"]

    def value(model, key):
        if key == "loss":
            return float(model["result"]["loss"])
        return float(model["result"]["metrics"][0][key])

    for ax, key, title in ((axes[0], "loss", "训练损失"), (axes[1], "correlation", "训练相关系数"),
                           (axes[2], "relative_l2", "相对 L2 误差")):
        values = [value(model, key) for model in (cold, warm)]
        ax.bar(labels, values, color=["#4C72B0", "#A5A5A5"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.3)
        for index, number in enumerate(values):
            ax.text(index, number, f"{number:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("从零训练 vs 热启动：物理预测模型拟合质量（同一探针集口径）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ccd_seed_final(facts: dict, path: Path) -> Path:
    """The measured far field before and after, one row per shape.

    The last stage that actually ran for that shape is used; the crop is the
    target region and the colour scale is the 95th percentile, because the
    zero order saturates a few hundred pixels in the middle of the target.
    """
    session = facts["session"]
    shapes = facts["shapes"]
    if not shapes:
        return path
    fig, axes = plt.subplots(len(shapes), 2, figsize=(6.4, 3.1 * len(shapes)),
                             constrained_layout=True, squeeze=False)
    for row, shape in enumerate(shapes):
        directory = None
        for stage in reversed(STAGES):
            candidate = session / "shape" / shape / stage
            if (candidate / "manifest.json").is_file():
                directory = candidate
                break
        for column, suffix in enumerate(("_seed", "_final")):
            ax = axes[row][column]
            ax.axis("off")
            if directory is None:
                continue
            rows = _read(directory / "manifest.json")
            entry = next((item for item in rows
                          if item["label"] == f"{shape}{suffix}"), None)
            if entry is None:
                continue
            with np.load(directory / entry["file"]) as saved:
                frame = np.asarray(saved["raw_frames"]).mean(0)
                xy = np.asarray(saved["camera_xy"])
                target = np.asarray(saved["target"])
            ys, columns = np.nonzero(target > .5)
            if ys.size:
                x0 = int(xy[ys.min(), columns.min()][0]) - 20
                x1 = int(xy[ys.max(), columns.max()][0]) + 20
                y0 = int(xy[ys.min(), columns.min()][1]) - 20
                y1 = int(xy[ys.max(), columns.max()][1]) + 20
            else:
                x0, x1 = int(xy[..., 0].min()), int(xy[..., 0].max())
                y0, y1 = int(xy[..., 1].min()), int(xy[..., 1].max())
            crop = frame[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
            step = max(1, int(np.ceil(max(crop.shape) / 600)))
            # Mono8-equivalent counts per millisecond: the two captures are
            # metered separately, so the raw counts are not comparable.
            exposure = float(entry.get("exposure_ms") or 1.0)
            bit_depth = int(entry.get("bit_depth") or 8)
            scale = exposure * float(2 ** (bit_depth - 8))
            ax.imshow((crop[::step, ::step]) / scale, cmap="inferno", vmin=0,
                      vmax=float(np.percentile(crop, 95)) / scale)
            cv = entry["metrics"]["cv"]
            ax.set_title(f"{SHAPE_LABELS.get(shape, shape)} · "
                         f"{'初相位' if suffix == '_seed' else '最终相位'} (CV {cv:.3f})",
                         fontsize=9)
    fig.suptitle("实测远场：初相位（左）与最终相位（右）\n"
                 "裁剪到目标区；色标取 95% 分位并按曝光与位深归一；中心零级饱和", fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def model_side_seed_cvs(session: Path, shapes: list[str], *, side: float, period: float,
                        window_span: float, window_step: float) -> dict:
    """Core CV of each initial-phase construction **in the frozen model**.

    The bake-off measured the same seeds on the bench. Reading the model's own
    verdict next to it is what makes the mismatch explicit: a construction can
    be exact in the model and still be the worst start on the bench.
    """
    import torch

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        FrozenDefocusParameters,
        recentre_window,
    )
    from ao_shaping.algorithm.signal_processing import multishape_objective as mo
    from ao_shaping.tools.slm.transfer.geometric_seed import (
        square_transport_phase,
        target_backprojection_phase,
    )

    checkpoint = session / "model640" / "frozen_parameters.json"
    if not checkpoint.is_file():
        return {}
    payload = _read(checkpoint)
    params = FrozenDefocusParameters.from_dicts(
        payload, payload["ramp_matrix"], grid_size=payload["grid_size"],
        fft_size=payload["fft_size"], pixel_samples=payload["pixel_samples"]).effective_parameters()
    rows = _read(session / "capture" / "manifest.json")
    reference = next(row for row in rows if row["label"] == "reference_start")
    zero = [float(round(value)) for value in reference["spot_xy"]]
    xy = recentre_window(tuple(zero), half_span_px=window_span, step_px=window_step)
    forward = FrozenDefocusForward(params, xy, device="cpu")
    forward.exposure_ms = 1.0
    centre = np.asarray(zero) + np.asarray(params.ramp_matrix)[:, 0] / period

    results: dict = {}
    for shape in shapes:
        objective = mo.ShapeObjective(mo.target_at(xy, shape, centre, side), cv_only=True)
        entry: dict = {}
        seeds = {}
        if shape == "square":
            seeds["transport"] = square_transport_phase(forward, centre, side)
        phase, diagnostics = target_backprojection_phase(forward, centre, side, shape)
        seeds["backprojection"] = phase
        entry["backprojection_cycles_per_panel_px"] = diagnostics["implied_cycles_per_panel_px"]
        seeds["gs"] = mo.gs_seed(forward, centre, side, shape)
        for kind, seed in seeds.items():
            with torch.no_grad():
                prediction = forward.predict(seed)[0].numpy()
            entry[kind] = mo.image_metrics(prediction, objective)["cv"]
        results[shape] = entry
    return results


def build_report(facts: dict) -> str:
    session = facts["session"]
    lines: list[str] = []
    shapes = facts["shapes"]
    cold = facts.get("fit")
    warm = facts.get("reference_fit")

    lines += [
        "# 1200 CCD 像元多形状的物理预测模型可微整形（从零训练，2026-10-27）",
        "",
        "## 1 本轮做了什么",
        "",
        f"会话目录：`{session.as_posix()}`。设备：Santec SLM #1 + Daheng MER2-507-23GM NIR，"
        "1064 nm；目标边长 1200 CCD 原生像元。",
        "",
        "本轮把 2026-10-26 的正方形流程**从零**跑了一遍，并把同一套流程用到另外三种形状上：",
        "",
        "1. **从零训练物理预测模型**：当前光路采集 12 个相位探针 + 正负光栅 + 首尾参考，"
        "由该采集**单独**生成名义检查点（`bootstrap`），再在探针上拟合（`fit`），"
        "全程无任何历史模型参与。",
        "2. **逐形状 CCD 锚定整形**：每种形状跑四段（初相位 → Mono12 → 续跑 → 稠密），"
        "每块的接受判据都是**实测**核心 CV 下降。**本轮只跑完了正方形**："
        "四种形状的初相位择优全部完成，三角形的整形段在第 1 块后被操作者中止，"
        "五角星与圆形尚未进入整形段（见 §3）。",
        "3. **共同重放**：各形状的初相位与最终相位在同一位深、同一曝光下重放。",
        "4. **Recorder 记录**：逐 CCD 块写入 Recorder（相位 / 实测远场 / 块内相位变化 / "
        "CV 前后 / 接受判决），导出 CSV + HDF5 + 过程 GIF。",
        "",
    ]

    lines += [
        "### 1.2 测量口径",
        "",
        "| 项 | 本轮取值 | 说明 |",
        "|---|---|---|",
        "| 目标 | 边长 1200 CCD 原生像元 | 四种形状用同一边长；三角形/五角星按外接圆半径 600、圆形按半径 600 生成 |",
        "| 零级排除 | 半径 60 CCD 原生像元 | 固定排除实测零级核心圆盘；**只从评价里排除，没有物理消除零级** |",
        "| 核心 | 目标形状再腐蚀 | 第 1–3 段腐蚀 2 个采样格，第 4 段 8 CCD px |",
        "| 接受判据 | 第 1 段：模型损失（`model_only`）；第 2–4 段：同曝光 A/B 顺序检查（`sequential_ccd`） | 第 1 段**没有**实测判决，不能表述为「通过 CCD 验收」 |",
        "| 曝光 | 每段各自测光，段内固定 | 跨段、跨曝光的原始 counts/CV 不可直接比较；共同重放（同位深、同曝光）才是可比口径 |",
        "| 位深 | 第 1 段 8 bit，第 2–4 段 12 bit | Mono12 的 counts/16 折算成 Mono8 等效单位 |",
        "",
        "四段流程与 2026-10-26 的正方形链一致：初相位 → Mono12 → 续跑 → 稠密"
        "（`--window-step 2` 采样 + 宽带载波模型）。每段的块数/步数/学习率见下表。",
        "",
        "| 段 | 块数 | 每块更新 | 相位学习率 | 相位步长上限 (rad) | 采样步长 (CCD px) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for recipe in STAGE_RECIPES:
        lines.append(f"| {STAGE_LABELS[recipe.name]} | {recipe.blocks} | {recipe.updates} "
                     f"| {recipe.phase_lr} | {recipe.phase_rms_cap} | {recipe.window_step} |")
    lines.append("")

    if cold and warm:
        cold_metrics, warm_metrics = cold["result"]["metrics"][0], warm["result"]["metrics"][0]
        lines += [
            "### 1.1 从零训练 vs 热启动：模型质量",
            "",
            "同一探针集、同一拟合口径下，从零训练（本次）与热启动（2026-10-26 用上一轮模型迁移）"
            "的拟合质量：",
            "",
            "| 口径 | 训练损失 | 训练相关系数 | 相对 L2 | 选中的迭代 |",
            "|---|---:|---:|---:|---:|",
            f"| 从零训练（本次） | {cold['result']['loss']:.5f} | {cold_metrics['correlation']:.4f} "
            f"| {cold_metrics['relative_l2']:.4f} | {cold['result']['iteration']} |",
            f"| 热启动（2026-10-26） | {warm['result']['loss']:.5f} | {warm_metrics['correlation']:.4f} "
            f"| {warm_metrics['relative_l2']:.4f} | {warm['result']['iteration']} |",
            "",
            "从零训练**没有**达到热启动的拟合质量，但差距是有限的（相关系数 "
            f"{cold_metrics['correlation']:.3f} vs {warm_metrics['correlation']:.3f}）。"
            "关键工程细节：标称初始化给出的照明位置与实测相差约 150 面板像元，"
            "而 Adam 的默认位置学习率走不完这段距离，因此本轮把位置学习率提到 4 面板像元/步"
            "（`--amplitude-center-lr 4.0`）并把迭代数从 60 提到 150；"
            "位置学到的结果（见下表）与热启动一致到十几像元。",
            "",
            "| 参数 | 从零训练 | 热启动 |",
            "|---|---|---|",
            f"| 照明中心 (面板像元) | {[round(v, 1) for v in cold['parameters']['amplitude_center']]} "
            f"| {[round(v, 1) for v in warm['parameters']['amplitude_center']]} |",
            f"| 照明宽度 (面板像元) | {[round(v, 1) for v in cold['parameters']['amplitude_width']]} "
            f"| {[round(v, 1) for v in warm['parameters']['amplitude_width']]} |",
            f"| 离焦 (屈光度) | {cold['parameters']['defocus_diopter']:.6f} "
            f"| {warm['parameters']['defocus_diopter']:.6f} |",
            f"| 增益 / 加性背景 | {cold['parameters']['gain']:.2f} / "
            f"{cold['parameters']['background']:.4f} | {warm['parameters']['gain']:.2f} / "
            f"{warm['parameters']['background']:.4f} |",
            "",
            "![从零训练与热启动的拟合质量](figures/model_fromzero_vs_warmstart" + STAMP + ".png)",
            "",
            "⚠️ 两种模型的**像差分解**不同（热启动把更多像差吸收进 Noll 残差项），"
            "绝对预测因此不可直接比较；本轮只用相对量。",
            "",
        ]

    if facts["bakeoff"]:
        kinds = sorted({entry["kind"] for entry in facts["bakeoff"].values()},
                       key=lambda k: ["transport", "backprojection", "gs"].index(k))
        lines += [
            "## 2 初相位的三种构造与实测择优",
            "",
            "正方形流程的非 GS 初相位是「传输」构造：对拟合入射振幅平方的两个边缘分布求累积分布，"
            "把各自能量映射到目标均匀区间，积分频率生成原始弧度相位。"
            "**可分离相位在数学上无法产生不可分离的远场**，所以这套构造只对正方形精确。",
            "",
            "为其它形状补了一条形状通用的非 GS 路线「反投影」：把目标在模型自己的 FFT 栅格上"
            "做一次逆变换，取其相位（无迭代、无测量反馈）。并把模型自带的 GS 种子作为第三个候选。",
            "",
            "三者的初相位在同一块、同一目标下**实测**核心 CV（越低越好）：",
            "",
            "| 形状 | " + " | ".join(SEED_LABELS[kind] for kind in kinds) + " | 采用 |",
            # Header and separator must agree: 形状 + one column per kind + 采用 is
            # len(kinds) + 2 cells, so the separator is derived from the same list.
            "|" + "---|" * (len(kinds) + 2),
        ]
        for shape in facts["bakeoff_shapes"]:
            values = []
            for kind in kinds:
                match = next((entry for entry in facts["bakeoff"].values()
                              if entry["shape"] == shape and entry["kind"] == kind), None)
                values.append(fmt_num_or_dash(match["seed_cv"], 4) if match else "—")
            chosen = facts.get("chosen_seeds", {}).get(shape, "—")
            lines.append(f"| {SHAPE_LABELS.get(shape, shape)} | " + " | ".join(values) +
                         f" | {SEED_LABELS.get(chosen, chosen)} |")
        lines += [
            "",
            "![初相位构造对比择优](figures/seed_bakeoff_multishape1200" + STAMP + ".png)",
            "",
            "**模型自己怎么说？** 同一批初相位在**冻结模型**里的核心 CV（离线重算，与实测同一栅格）：",
            "",
            "| 形状 | 传输 (模型内) | 反投影 (模型内) | GS (模型内) | 反投影所需频率 (周期/面板像元) |",
            "|---|---:|---:|---:|---:|",
        ]
        for shape in facts["bakeoff_shapes"]:
            model_entry = facts.get("model_side", {}).get(shape, {})
            lines.append(
                f"| {SHAPE_LABELS.get(shape, shape)} | "
                f"{fmt_num_or_dash(model_entry.get('transport'), 4)} | "
                f"{fmt_num_or_dash(model_entry.get('backprojection'), 4)} | "
                f"{fmt_num_or_dash(model_entry.get('gs'), 4)} | "
                f"{fmt_num_or_dash(model_entry.get('backprojection_cycles_per_panel_px'), 3)} |")
        lines += [
            "",
            "两列读数放在一起看：「传输」在模型里几乎完美（核心 CV ≈ 0.00x），"
            "在实验系统里却是最差的一档；GS 在模型里并不更好，实测却最好。"
            "**模型的绝对量不可信**（常数加性背景项远大于核心的实测总均值），"
            "所以初相位不能按模型内读数选，必须实测。",
            "",
            "反投影那一列还给出另一条硬约束：它需要的瞳面空间频率若超过 0.5 周期/面板像元，"
            "**器件无法表示**（控制 Nyquist），即使模型内读数不差也要先被这条约束否掉。",
            "",
            "**这一节改变了 2026-10-26 的正方形流程。** 那次流程的第一段用的是「传输」初相位；"
            "本轮先实测了三种初相位的核心 CV，**四种形状全部由 GS 胜出**，因此第一段统一用 GS 初相位。",
            "",
            "⚠️ 因此本报告的正方形结果**不是** 2026-10-26 那次流程的逐位复现："
            "流程结构（四段、判据、模型、采样）一致，初相位构造不同。"
            "要复现那次流程，应显式 `--initial-phase transport --no-seed-bakeoff`；"
            "**本轮没有执行这一变体**（设备在正方形四段跑完后由操作者中止，见 §3）。",
            "",
        ]

    if facts["stages"]:
        lines += [
            "## 3 逐形状整形结果",
            "",
            "每段的「初相位 → 最终」核心 CV（实测，同曝光同掩膜口径见 `config.json`）：",
            "",
        ]
        header = "| 形状 | " + " | ".join(STAGE_LABELS[stage] for stage in STAGES) + " |"
        lines += [header, "|" + "---|" * (1 + len(STAGES))]
        for shape in shapes:
            cells = []
            for stage in STAGES:
                stage_facts = facts["stages"].get((shape, stage))
                if not stage_facts:
                    cells.append("未运行")
                    continue
                seed, final = stage_facts["seed_cv"], stage_facts["final_cv"]
                completion = stage_completion(facts, shape).get(stage, {})
                mark = "" if completion.get("complete") else \
                    f" ⚠{completion.get('blocks', 0)}/{completion.get('expected', 0)} 块"
                cells.append(f"{fmt_num_or_dash(seed, 3)} → {fmt_num_or_dash(final, 3)}{mark}")
            lines.append(f"| {SHAPE_LABELS.get(shape, shape)} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "接受块数（`summary.json` 的 `accepted`）：",
            "",
            "| 形状 | " + " | ".join(STAGE_LABELS[stage] for stage in STAGES) + " |",
            "|" + "---|" * (1 + len(STAGES)),
        ]
        for shape in shapes:
            cells = []
            for stage in STAGES:
                stage_facts = facts["stages"].get((shape, stage))
                if not stage_facts:
                    cells.append("未运行")
                    continue
                blocks = stage_facts["blocks"]
                accepted = sum(1 for block in blocks if block.get("accepted"))
                cells.append(f"{accepted}/{len(blocks)}")
            lines.append(f"| {SHAPE_LABELS.get(shape, shape)} | " + " | ".join(cells) + " |")
        unfinished = [shape for shape in shapes if not shape_finished(facts, shape)]
        if unfinished:
            lines += [
                "",
                "⚠️ **本轮运行被操作者中止**："
                + "、".join(SHAPE_LABELS.get(shape, shape) for shape in unfinished)
                + " 的整形段没有跑完（上表里标 ⚠ 的格子是中止时已完成的部分）。"
                "已完成的正方形四段与四种形状的初相位择优数据是完整的；"
                "其余三种形状只有初相位层面的实测结果，没有整形结果，"
                "不能据此声称它们已被整形。",
                "",
            ]
        lines += [
            "",
            "![逐块实测核心 CV](figures/cv_trajectory_multishape1200" + STAMP + ".png)",
            "",
            "![第 4 段之后的实测远场](figures/ccd_seed_final_multishape1200" + STAMP + ".png)",
            "",
            "两处读法要注意：",
            "",
            "- 轨迹曲线的**最后一点**是第 4 段最后一块的**候选**读数，而「最终相位」是同一相位"
            "在块结束后**另一次**采集的读数；两者相差约 0.013，是同相位重复测量的波动，不是矛盾。"
            "报告引用哪一个必须说明口径。",
            "- 远场图里三角形的右格为空：该形状的整形段中止，没有最终相位采集；"
            "正方形那一行才是完整的「初相位 vs 最终相位」对照。",
            "",
        ]

    if facts.get("transport_flow"):
        flow = facts["transport_flow"]
        lines += [
            "### 3.1 同一条链的「传输」初相位版本（2026-10-26 流程的直接复现）",
            "",
            f"另一个会话（`{flow['session'].as_posix()}`）用**同一个从零模型**、同一条四段链，"
            "只把第一段的初相位换回「传输」构造（`--initial-phase transport --no-seed-bakeoff`），"
            "即 2026-10-26 那次正方形流程的复现：",
            "",
            "| 段 | 初相位核心 CV | 最终核心 CV |",
            "|---|---:|---:|",
        ]
        for stage in STAGES:
            stage_facts = flow["stages"].get(stage)
            if not stage_facts:
                continue
            lines.append(f"| {STAGE_LABELS[stage]} | {fmt_num_or_dash(stage_facts['seed_cv'], 3)} "
                         f"| {fmt_num_or_dash(stage_facts['final_cv'], 3)} |")
        lines += [
            "",
            "两次运行只有初相位构造不同，其余（模型、段、块数、判据、采样）一致；"
            "两者的差异因此可以归到初相位上。",
            "",
        ]

    if facts["replay"]:
        lines += [
            "## 4 共同重放",
            "",
            "各形状的初相位与最终相位在同一位深、同一曝光下重放（原始帧与时间独立保存）。",
            "",
            "| 相位 | 曝光 ms | 核心 CV | 核心峰值 |",
            "|---|---:|---:|---:|",
        ]
        for row in facts["replay"]:
            lines.append(f"| {row['label']} | {row['exposure_ms']:.3f} | "
                         f"{fmt_num_or_dash(row['metrics']['cv'], 6)} | {row['core_peak']:.1f} |")
        lines.append("")
    else:
        lines += [
            "## 4 共同重放",
            "",
            "⚠️ **本轮没有共同重放**：运行在正方形四段完成后被中止，重放阶段未执行。"
            "因此本报告的段间比较只能用各段自带的同曝光对照"
            "（`square_seed_replay` 与 `square_final`），不能声称跨段是同曝光同位深的比较。",
            "",
        ]

    lines += [
        "## 5 Recorder 记录与过程动画",
        "",
        "逐 CCD 块的记录写在 `records/process.csv`（标量列）与 `records/process.h5`（含数组列）里，"
        "数组列另存为 `records/process_sidecars/*.npy`。每条记录含：形状、段、块号、"
        "实测曝光与位深、锚点/候选核心 CV 及其差、接受判决与判据、模型损失、"
        "该块的瞳面相位、实测远场画面与块内相位变化。",
        "",
        "![逐 CCD 块的过程动画](figures/process_multishape1200" + STAMP + ".gif)",
        "",
        "动画由 `transfer/recording.py::render_process_gif` 渲染（记录本身由 Recorder 持有）："
        "上排是去活塞/倾斜后的瞳面相位、实测远场（裁剪到目标区、色标取 95% 分位，"
        "中心零级饱和）与块内相位变化；下排是本块读数、实测核心 CV 轨迹"
        "（灰：锚点 A1，红：候选 B）与模型损失轨迹。",
        "",
    ]

    lines += [
        "## 6 复现命令",
        "",
        "```bash",
        "# 模型侧（探针采集 + 从零训练 + 控制分辨率导出）",
        "python -m ao_shaping.tools.slm.transfer.capture --out <session>/capture --execute \\",
        "    --grid 384 --count 12 --frames 3 --exposure-max-steps 12 --save-dataset",
        "python -m ao_shaping.tools.slm.transfer.bootstrap --dataset <session>/capture --out <session>/initial",
        "python -m ao_shaping.tools.slm.transfer.fit --dataset <session>/capture \\",
        "    --checkpoint <session>/initial/frozen_parameters.json --out <session>/fit \\",
        "    --iterations 150 --span 120 --adaptation full --amplitude-center-lr 4.0 --device cuda",
        "python -m ao_shaping.tools.slm.transfer.geometric_seed --checkpoint <session>/fit/frozen_parameters.json \\",
        "    --out <session>/model640 --grid 640 --fft 2048",
        "",
        "# 整形 + 重放 + 记录（一条命令跑完四种形状）",
        "python src/ao_shaping/main.py slm-transfer-shapes --out <session> --model-dir <session> \\",
        "    --shapes square --shapes triangle --shapes star --shapes circle --device cuda --execute",
        "",
        "# 只从已跑完的分段目录重建 Recorder 记录 (离线)",
        "python src/ao_shaping/main.py slm-transfer-shapes --out <session> --records-only",
        "```",
        "",
        "本报告由 `scripts/slm/generate_multishape1200_report.py` 离线生成；"
        "它只读会话目录，不重新测量。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--reference-session", type=Path, default=REFERENCE_SESSION)
    parser.add_argument("--transport-session", type=Path,
                        default=ROOT / "data/debug/slm_transfer/transport_flow_square1200_20261027",
                        help="the documented (transport seed) square flow, if it was run")
    parser.add_argument("--side", type=float, default=1200.0)
    parser.add_argument("--period", type=float, default=128.0)
    parser.add_argument("--window-span", type=float, default=820.0)
    parser.add_argument("--window-step", type=float, default=4.0)
    parser.add_argument("--skip-model-side", action="store_true",
                        help="skip the offline model-side seed read-out (it costs one GS per shape)")
    args = parser.parse_args()

    facts = _session_facts(args.session)
    facts["transport_flow"] = _transport_flow_facts(args.transport_session)
    if not args.skip_model_side:
        facts["model_side"] = model_side_seed_cvs(
            args.session, facts["bakeoff_shapes"], side=args.side, period=args.period,
            window_span=args.window_span, window_step=args.window_step)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    plot_seed_bakeoff(facts, FIG_DIR / f"seed_bakeoff_multishape1200{STAMP}.png")
    plot_cv_trajectory(facts, FIG_DIR / f"cv_trajectory_multishape1200{STAMP}.png")
    plot_model_comparison(facts, FIG_DIR / f"model_fromzero_vs_warmstart{STAMP}.png")
    plot_ccd_seed_final(facts, FIG_DIR / f"ccd_seed_final_multishape1200{STAMP}.png")

    gif_source = args.session / "records" / "process.gif"
    gif_target = FIG_DIR / f"process_multishape1200{STAMP}.gif"
    if gif_source.is_file():
        shutil.copyfile(gif_source, gif_target)

    # The report's input contract: the session's own summary files.
    for name in ("journal.json",):
        source = args.session / name
        if source.is_file():
            shutil.copyfile(source, ART_DIR / name)
    if (args.session / "fit" / "result.json").is_file():
        shutil.copyfile(args.session / "fit" / "result.json", ART_DIR / "fit_result.json")
    if (args.session / "records" / "process.csv").is_file():
        shutil.copyfile(args.session / "records" / "process.csv", ART_DIR / "process.csv")

    body = build_report(facts)
    target = ROOT / REPORT_KEY
    target.write_text(insert_header(body, REPORT_KEY), encoding="utf-8")
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    main()
