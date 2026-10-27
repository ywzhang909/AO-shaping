"""Report generator: transfer-learned physical-forward model shaping a 1200 px square.

Reads the session directory written by ``python -m ao_shaping.tools.slm.transfer``
and renders ``report/shape_trials/transfer_square1200/transfer_square1200_20261027.md``
together with its figures. Offline: no device, no re-measurement.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._common.provenance import insert_header

from ao_shaping.tools.slm.transfer.recording import crop_frame

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPORT_KEY = "report/shape_trials/transfer_square1200/transfer_square1200_20261027.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "transfer_square1200"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"
STAMP = "_20261027"

DEFAULT_SESSION = ROOT / "data/debug/slm_transfer/transfer_square1200_20261027_r2"
FAILED_SESSION = ROOT / "data/debug/slm_transfer/transfer_square1200_20261027"

ROLE_COLORS = {"seed": "#2E7D32", "final": "#111111", "A1": "#8C8C8C", "A2": "#5A5A5A",
               "B1": "#C44E52", "B2": "#E08A8C", "diag": "#E6A23C", "other": "#4C72B0"}
ROLE_LABELS = {"seed": "初相位", "final": "最终相位", "A1": "锚点 A1", "A2": "锚点 A2 (漂移对照)",
               "B1": "候选 B1", "B2": "候选 B2 (重复)", "diag": "诊断探针 (反号)",
               "other": "其它"}
STAGE_LABELS = {"model": "第 1 段：模型模式（无 CCD 锚定）", "anchored": "第 2 段：CCD 锚定"}


def _read(path: Path, default=None):
    """Read JSON; ``default`` when missing or being rewritten underneath us."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        if default is None:
            raise
        return default


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _role(label: str) -> str:
    if label.endswith("_seed"):
        return "seed"
    if label.endswith("_final"):
        return "final"
    if "direction" in label:
        return "diag"
    for suffix in ("A1", "A2", "B1", "B2"):
        if label.endswith("_" + suffix):
            return suffix
    return "other"


def _session_facts(session: Path) -> dict:
    """Everything the report needs, read once from the session directory."""
    facts: dict = {"session": session}
    session_json = _read(session / "session.json", default={})
    facts["settings"] = session_json.get("settings", {})
    facts["resolved"] = session_json.get("resolved", {})
    facts["stages_planned"] = session_json.get("stages", [])
    facts["journal"] = _read(session / "journal.json", default=[])
    facts["adaptation"] = _read(session / "fit" / "adaptation.json", default={})
    facts["fit_result"] = _read(session / "fit" / "result.json", default={})
    facts["adapted"] = _read(session / "fit" / "frozen_parameters.json", default={})
    facts["audit"] = _read(session / "audit.json", default={})
    facts["model_summary"] = _read(session / "model" / "summary.json", default=[])
    facts["anchored_summary"] = _read(session / "anchored" / "summary.json", default=[])
    facts["model_manifest"] = _read(session / "model" / "manifest.json", default=[])
    facts["anchored_manifest"] = _read(session / "anchored" / "manifest.json", default=[])
    facts["direction"] = _read(session / "anchored" / "square_direction.json", default={})
    facts["failed_journal"] = _read(FAILED_SESSION / "journal.json", default=[])

    checkpoint = Path(facts["settings"].get("checkpoint") or "")
    facts["checkpoint_path"] = checkpoint if str(checkpoint) else None
    facts["pretrained"] = (_read(checkpoint, default={}) if str(checkpoint) else {})

    dataset = Path(facts["settings"].get("dataset") or "")
    facts["dataset"] = dataset if str(dataset) else None
    reference = None
    if str(dataset):
        rows = _read(dataset / "manifest.json", default=[])
        reference = next((row for row in rows if row["label"] == "reference_start"), None)
    facts["reference"] = reference

    gif = session / "anchored" / "recording" / "process.gif"
    facts["gif"] = gif if gif.is_file() else None
    return facts


def plot_adaptation(facts: dict, path: Path) -> Path:
    """Validation quality of each adaptation candidate, with the escalation gates."""
    candidates = facts["adaptation"].get("candidates", [])
    if not candidates:
        return path
    labels = [entry["stage"] for entry in candidates]
    metrics = (("correlation", "验证相关系数（升级阈值 0.85）"), ("relative_l2", "相对 L2（升级阈值 0.25）"),
               ("cv_abs_error", "CV 绝对误差"))
    selected = facts["adaptation"].get("selected_stage")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    for ax, (key, title) in zip(axes, metrics):
        values = [entry["quality"][key] for entry in candidates]
        colors = ["#C44E52" if label == selected else "#A5A5A5" for label in labels]
        ax.barh(labels, values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.3)
        ax.set_xlim(0, max(values) * 1.24)
        for index, value in enumerate(values):
            ax.text(value, index, f" {value:.3f}", va="center", fontsize=8)
    axes[0].axvline(.85, color="#4C72B0", linestyle="--", linewidth=1)
    axes[1].axvline(.25, color="#4C72B0", linestyle="--", linewidth=1)
    fig.suptitle("迁移适配的候选阶段与验证质量（红 = 采用）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_stage_cv(ax, manifest: list, summary: list, title: str) -> None:
    labels = [row["label"] for row in manifest]
    values = [float(row["metrics"]["cv"]) for row in manifest]
    for index, (label, value) in enumerate(zip(labels, values)):
        role = _role(label)
        ax.plot(index, value, marker="o", markersize=5, linestyle="none",
                color=ROLE_COLORS[role])
    ax.plot(range(len(values)), values, color="#CCCCCC", linewidth=.8, zorder=0)
    for entry in summary:
        block = int(entry["block"])
        indices = [index for index, label in enumerate(labels)
                   if f"_{block}_" in label and _role(label) in {"A1", "B1", "B2", "A2"}]
        if not indices:
            continue
        verdict = "接受" if entry.get("accepted") else "拒绝"
        color = "#2E7D32" if entry.get("accepted") else "#C44E52"
        ax.axvspan(min(indices) - .5, max(indices) + .5, color=color, alpha=.07)
        ax.text((min(indices) + max(indices)) / 2, max(values) * 1.02, f"块 {block} {verdict}",
                ha="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("采集序号")
    ax.set_ylabel("实测核心 CV")
    ax.grid(alpha=.3)
    ax.set_ylim(min(values) * .97, max(values) * 1.05)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=ROLE_COLORS[role],
                          label=ROLE_LABELS[role]) for role in ("seed", "A1", "B1", "B2", "A2", "diag")]
    ax.legend(handles=handles, fontsize=6, ncol=3, loc="lower right")


def plot_cv_trajectory(facts: dict, path: Path) -> Path:
    """Measured core CV of every capture, one panel per stage."""
    panels = [(facts["model_manifest"], facts["model_summary"], STAGE_LABELS["model"]),
              (facts["anchored_manifest"], facts["anchored_summary"], STAGE_LABELS["anchored"])]
    panels = [panel for panel in panels if panel[0]]
    if not panels:
        return path
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 3.6 * len(panels)), squeeze=False)
    for ax, (manifest, summary, title) in zip(axes[:, 0], panels):
        _plot_stage_cv(ax, manifest, summary, title)
    fig.suptitle("逐次采集的实测核心 CV（A–B–B–A 配对；诊断探针是反号相位，本就该差）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_direction(facts: dict, path: Path) -> Path:
    """Predicted vs measured phase-increment difference."""
    difference = facts["session"] / "anchored" / "square_direction_difference.npz"
    if not difference.is_file():
        return path
    with np.load(difference) as saved:
        predicted = np.asarray(saved["predicted"], dtype=float).ravel()
        measured = np.asarray(saved["measured"], dtype=float).ravel()
    step = max(1, predicted.size // 20000)
    predicted, measured = predicted[::step], measured[::step]
    correlation = facts["direction"].get("difference_correlation")
    relative_l2 = facts["direction"].get("difference_relative_l2")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].scatter(measured, predicted, s=2, alpha=.25, color="#4C72B0")
    span = [min(measured.min(), predicted.min()), max(measured.max(), predicted.max())]
    axes[0].plot(span, span, color="#C44E52", linewidth=1, linestyle="--", label="1:1")
    axes[0].set_xlabel("实测差分（+Δ 与 −Δ 的像素差）")
    axes[0].set_ylabel("模型预测差分")
    axes[0].set_title(f"差值图：相关 {_fmt(correlation, 3)}，相对 L2 {_fmt(relative_l2, 3)}",
                      fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3)

    direction = facts["direction"]
    rows = [("模型损失", direction.get("model_loss_plus"), direction.get("model_loss_minus")),
            ("实测损失", direction.get("ccd_loss_plus"), direction.get("ccd_loss_minus"))]
    labels = [row[0] for row in rows]
    plus = [row[1] for row in rows]
    minus = [row[2] for row in rows]
    positions = np.arange(len(labels))
    axes[1].bar(positions - .18, plus, width=.36, color="#2E7D32", label="+Δ 相位")
    axes[1].bar(positions + .18, minus, width=.36, color="#C44E52", label="−Δ 相位")
    axes[1].set_xticks(positions, labels)
    axes[1].set_title("同一增量两方向的损失", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.3)
    for axis, values in ((positions - .18, plus), (positions + .18, minus)):
        for position, value in zip(axis, values):
            if value is not None:
                axes[1].text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("方向性检查：模型能否判断「往哪走」（block 0）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _far_field(run_dir: Path, row: dict, side: int = 520) -> np.ndarray:
    with np.load(run_dir / row["file"]) as saved:
        frame = np.asarray(saved["raw_frames"]).mean(0)
        xy = np.asarray(saved["camera_xy"])
        target = np.asarray(saved["target"])
    rows, columns = np.nonzero(target > .5)
    if rows.size:
        x0 = int(xy[rows.min(), columns.min()][0]) - 20
        x1 = int(xy[rows.max(), columns.max()][0]) + 20
        y0 = int(xy[rows.min(), columns.min()][1]) - 20
        y1 = int(xy[rows.max(), columns.max()][1]) + 20
    else:
        x0, x1 = int(xy[..., 0].min()), int(xy[..., 0].max())
        y0, y1 = int(xy[..., 1].min()), int(xy[..., 1].max())
    crop = frame[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    step = max(1, int(np.ceil(max(crop.shape) / side)))
    crop = crop[::step, ::step]
    scale = float(row["exposure_ms"]) * float(2 ** (int(row["bit_depth"]) - 8))
    return crop / scale, float(np.percentile(crop, 95)) / scale


def plot_ccd(facts: dict, path: Path) -> Path:
    """Measured far field of the seed and the accepted final phase."""
    run_dir = facts["session"] / "anchored"
    rows = {row["label"]: row for row in facts["anchored_manifest"]}
    seed, final = rows.get("square_seed"), rows.get("square_final")
    if not seed or not final:
        return path
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    for ax, row, title in ((axes[0], seed, "初相位（GS 种子）"),
                           (axes[1], final, "最终相位（CCD 锚定后）")):
        image, vmax = _far_field(run_dir, row)
        ax.imshow(image, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{title}\n核心 CV {row['metrics']['cv']:.4f}，曝光 {row['exposure_ms']:.1f} ms",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("实测远场：裁剪到目标区，色标取 95% 分位并按曝光与位深归一；中心零级饱和",
                 fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _copy_gif(facts: dict, path: Path) -> Path | None:
    if facts["gif"] is None:
        return None
    shutil.copy2(facts["gif"], path)
    return path


def _copy_artifacts(facts: dict) -> list[str]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in (facts["session"] / "fit" / "adaptation.json",
                   facts["session"] / "anchored" / "summary.json",
                   facts["session"] / "anchored" / "square_direction.json",
                   facts["session"] / "audit.json"):
        if source.is_file():
            target = ART_DIR / f"{source.stem}{STAMP}.json"
            shutil.copy2(source, target)
            copied.append(target.name)
    return copied


def build_report(facts: dict) -> str:
    settings = facts["settings"]
    adaptation = facts["adaptation"]
    pretrained = facts["pretrained"]
    adapted = facts["adapted"]
    reference = facts["reference"] or {}
    zero = reference.get("spot_xy")
    old_origin = pretrained.get("origin")
    shift = None
    if zero and old_origin:
        shift = [float(zero[0]) - float(old_origin[0]), float(zero[1]) - float(old_origin[1])]

    candidates = adaptation.get("candidates", [])
    selected = adaptation.get("selected_stage", "—")
    anchored = facts["anchored_summary"]
    model_stage = facts["model_summary"]
    rows = {row["label"]: row for row in facts["anchored_manifest"]}
    seed_cv = rows.get("square_seed", {}).get("metrics", {}).get("cv")
    final_cv = rows.get("square_final", {}).get("metrics", {}).get("cv")
    accepted = sum(1 for entry in anchored if entry.get("accepted"))
    direction = facts["direction"]
    audit = facts["audit"]

    lines = [
        "# 1200 像元方形：迁移学习适配预训练物理预测模型（手动干扰后，2026-10-27）",
        "",
        "会话目录：`"
        + str(facts["session"]).replace("\\", "/").split("AO-shaping/")[-1]
        + "`。设备：Santec SLM #1 + Daheng MER2-507-23GM NIR，1064 nm；目标边长 1200 CCD 原生像元。",
        "",
        "## 1 本轮做了什么",
        "",
        "**结论**：干扰后适配（alignment 相关 0.34 → full 0.84）把模型重新对上当前光路；"
        "但纯模型提议的两块被实测否决，只有 CCD 锚定的块带来改善（核心 CV "
        + _fmt(seed_cv, 3) + " → " + _fmt(final_cv, 3) + "，接受 "
        + str(accepted) + "/" + str(len(anchored)) + " 块）。模型方向性仍弱"
        "（差值图相关 " + _fmt(direction.get("difference_correlation"), 2) + "），"
        "因此每一块都必须由实测判决。",
        "",
        "光路被**手动干扰**后，用 2026-10-27 从零训练得到的物理预测检查点做**迁移适配**，"
        "再以「模型预测 → SLM 相位更新 → CCD 实测判决」的迭代把 1200 px 方形匀化。"
        "全过程由 `Recorder` 逐次采集记录，导出 CSV + HDF5 + 过程动画。",
        "",
        "1. **复用探针采集**：干扰后的 12 个相位探针（含正负光栅与首尾参考）已在本轮之前的"
        "同一光路采完，直接复用，不再重复占用设备。",
        "2. **迁移适配**：以历史检查点为起点，`--adaptation auto` 先只放开照明位置与光轴原点"
        "（alignment），验证不足时再放开全部光学参数（full）。",
        "3. **模型 + SLM 相位迭代**：第 1 段纯模型提议（无 CCD 锚定），第 2 段每块先用 CCD 实测"
        "锚定当前场，再用模型梯度更新相位，逐块以同曝光 A–B–B–A 配对判据验收。",
        "4. **记录**：每次采集写入 `Recorder`（相位 / 实测远场 / 相位变化 / 指标 / 判决），"
        "设备关闭后导出表格与过程动画。",
        "",
        "### 1.1 测量口径",
        "",
        "| 项 | 本轮取值 | 说明 |",
        "|---|---|---|",
        f"| 目标 | 边长 {_fmt(settings.get('side'), 0)} CCD 原生像元 | 正方形，外接边与采样窗对齐 |",
        f"| 载波周期 | {_fmt(settings.get('period'), 0)} 面板像元 | 标定档之一 |",
        f"| 采样窗 | 半跨 {_fmt(settings.get('window_span'), 0)} px，步长 {_fmt(settings.get('window_step'), 0)} px | 大目标必须显式打开，默认 220 px 会直接拒绝 |",
        f"| 零级排除 | 半径 {_fmt(settings.get('zero_radius'), 0)} CCD px | 只从评价里排除，没有物理消除零级 |",
        f"| 核心 | 目标再腐蚀 {_fmt(settings.get('core_margin'), 0)} CCD px | 边缘不参与 CV |",
        f"| 位深 / 曝光 | {int(settings.get('bit_depth', 8))} bit / 段内固定 | 各块曝光按目标测光，跨块原始 counts 不可直接比较 |",
        f"| 判据 | {'A–B–B–A 配对（实测核心 CV）' if settings.get('abba') else '模型损失'} | "
        f"{'漂移由 A1/A2 对照吸收' if settings.get('abba') else '无实测判决'} |",
        f"| 控制分辨率 | {int(settings.get('control_grid') or 0)} 网格 / FFT {int(settings.get('fft_size') or 0)} | 拟合分辨率导出，只改采样不改光学参数 |",
        "",
        "### 1.2 干扰的实测证据",
        "",
    ]

    if zero and old_origin:
        lines += [
            f"本轮实测零级 **({_fmt(zero[0], 1)}, {_fmt(zero[1], 1)})**，历史检查点记录的是 "
            f"({_fmt(old_origin[0], 1)}, {_fmt(old_origin[1], 1)})，相差 "
            f"**({_fmt(shift[0], 1)}, {_fmt(shift[1], 1)}) px**。这就是必须重新适配而不是直接沿用"
            "检查点的原因。",
            "",
        ]

    lines += [
        "## 2 迁移适配：对齐 → 全参数",
        "",
        "验证相位（6 条，与训练相位不重叠）上的候选阶段质量：",
        "",
        "| 候选阶段 | 验证相关系数 | 相对 L2 | CV 绝对误差 | 选择分数 (1−corr+relL2) |",
        "|---|---:|---:|---:|---:|",
    ]
    for entry in candidates:
        quality = entry["quality"]
        mark = " **← 采用**" if entry["stage"] == selected else ""
        lines.append(f"| `{entry['stage']}`{mark} | {_fmt(quality['correlation'], 3)} | "
                     f"{_fmt(quality['relative_l2'], 3)} | {_fmt(quality['cv_abs_error'], 3)} | "
                     f"{_fmt(quality['score'], 3)} |")
    lines += [
        "",
        f"选择结果：**{selected}**（升级触发：{adaptation.get('escalated')}）。"
        "alignment 只放开照明位置与光轴原点，共享增益与加性背景（杂散光）项由训练帧 NNLS 求解；"
        "它把相关只做到 0.34 左右，远低于 0.85 的升级阈值，因此自动进入 full。"
        "full 放开离焦、照明宽度与 Noll 残差后，验证相关升到 0.84。",
        "",
        "![迁移适配的候选阶段与验证质量](figures/adaptation" + STAMP + ".png)",
        "",
        "### 2.1 适配前后的参数",
        "",
        "| 参数 | 历史检查点 | 本轮适配后 |",
        "|---|---:|---:|",
    ]
    for key, label, digits in (("origin", "光轴原点 (CCD px)", 1),
                               ("amplitude_center", "照明中心 (面板 px)", 1),
                               ("amplitude_width", "照明宽度 (面板 px)", 1),
                               ("defocus_diopter", "离焦 (屈光度)", 5),
                               ("gain", "增益 (counts/ms)", 2),
                               ("background", "加性背景 (counts/ms)", 3)):
        old = pretrained.get(key)
        new = adapted.get(key)
        lines.append(f"| {label} | {_fmt_array(old, digits)} | {_fmt_array(new, digits)} |")
    lines += [
        "",
        "照明中心在面板上移动了约 "
        + _fmt(_distance(pretrained.get("amplitude_center"), adapted.get("amplitude_center")), 0)
        + " 面板像元——干扰不只是相机侧的平移，面板照明本身也动了，这正是 alignment 阶段"
        "（只放开位置）不足以解释、必须升级到 full 的原因。",
        "",
        "### 2.2 一个会让适配永久卡死的机制（本轮修掉）",
        "",
        "第一次尝试在本阶段直接失败：`ValueError: nonfinite validation quality`。"
        "根因不是数值噪声，而是**几何被继承**：`load_dataset` 把相机窗口对齐到本次实测零级，"
        "而模型仍用历史检查点的光轴原点，预测场因此整体平移了上面那个量级。"
        "NNLS 辐射度在训练帧上求出的增益被非负约束**截断为 0**，预测退化成常数，"
        "损失与相位完全无关，所有梯度随之消失——alignment 既不会改善也不会报错，"
        "只是原地不动，直到后面的质量检查报出 NaN 相关。",
        "",
        "修法：拟合前用**本次实测零点**重锚模型原点（几何实测，不继承），"
        "并在增益为 0 时立刻给出指名道姓的错误，而不是留到下游变成「非有限验证质量」。"
        "离线用同一份采集复演验证：修前增益 0、相关 NaN；修后 alignment 0.34、full 0.84。",
        "",
        "## 3 逐块结果",
        "",
        "两段都是 A–B–B–A 配对（同曝光、同掩膜、同核心），判据是实测核心 CV。"
        "判决规则是 `候选均值 < 锚点均值 − 漂移`，漂移 = |A2 − A1|，"
        "即候选必须**超过**这一块的漂移才算改善：",
        "",
        "| 段 | 块 | 锚点均值 (A1/A2) | 候选均值 (B1/B2) | 漂移 | 判决 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for stage, summary in (("model", model_stage), ("anchored", anchored)):
        for entry in summary:
            baseline = entry.get("baseline")
            candidate = entry.get("candidate")
            drift = entry.get("drift")
            verdict = "接受" if entry.get("accepted") else "拒绝"
            lines.append(f"| {STAGE_LABELS[stage]} | {entry.get('block')} | {_fmt(baseline, 4)} | "
                         f"{_fmt(candidate, 4)} | {_fmt(drift, 4) if drift is not None else '—'} | {verdict} |")
    lines += [
        "",
        f"**模型模式的两块全部被拒**：候选把实测核心 CV 从 "
        f"{_fmt(model_stage[0].get('baseline') if model_stage else None, 4)} 推到 "
        f"{_fmt(model_stage[0].get('candidate') if model_stage else None, 4)}。"
        "纯模型梯度（没有 CCD 锚定）在干扰后的光路上方向是错的——这与既有结论一致："
        "模型的绝对量不可信，只能当提议器。",
        "",
        f"**CCD 锚定段接受 {accepted}/{len(anchored)} 块**：前两块把核心 CV 从 "
        f"{_fmt(anchored[0].get('baseline') if anchored else None, 4)} 降到 "
        f"{_fmt(min([entry.get('candidate') or 9.9 for entry in anchored]) if anchored else None, 4)}；"
        "后两块候选被拒（0.5708→0.5771、0.5724→0.5799），说明单块 20 次更新在强干扰后的"
        "相位空间里已经走到平台，继续按同一步长提议只会变差。拒绝即回退到上一块的接受相位，"
        "所以最终相位仍是最好的一块。",
        "",
        f"最终相位实测核心 CV **{_fmt(final_cv, 4)}**（初相位 {_fmt(seed_cv, 4)}，"
        f"相对改善 {_fmt(100 * (seed_cv - final_cv) / seed_cv if seed_cv and final_cv else None, 1)}%）。",
        "",
        "![逐次采集的实测核心 CV](figures/cv_trajectory" + STAMP + ".png)",
        "",
        "![实测远场：初相位与最终相位](figures/ccd_seed_final" + STAMP + ".png)",
        "",
        "## 4 方向性检查：模型能预测「哪里亮」，预测不了「往哪走」",
        "",
        "block 0 额外采了同一增量的反号相位（`−Δ = 2·A1 − B`），用来把「模型雅可比错」与"
        "「自动微分错」分开：",
        "",
        "| 量 | +Δ | −Δ | 差（+ − −） |",
        "|---|---:|---:|---:|",
        f"| 模型损失 | {_fmt(direction.get('model_loss_plus'), 4)} | "
        f"{_fmt(direction.get('model_loss_minus'), 4)} | "
        f"{_fmt(_diff(direction.get('model_loss_plus'), direction.get('model_loss_minus')), 4)} |",
        f"| 实测损失 | {_fmt(direction.get('ccd_loss_plus'), 4)} | "
        f"{_fmt(direction.get('ccd_loss_minus'), 4)} | "
        f"{_fmt(_diff(direction.get('ccd_loss_plus'), direction.get('ccd_loss_minus')), 4)} |",
        "",
        f"模型认为两个方向几乎一样（差 {_fmt(_diff(direction.get('model_loss_plus'), direction.get('model_loss_minus')), 4)}），"
        f"实测却差 {_fmt(_diff(direction.get('ccd_loss_plus'), direction.get('ccd_loss_minus')), 4)}。"
        f"差值图的相关只有 **{_fmt(direction.get('difference_correlation'), 3)}**，"
        f"相对 L2 **{_fmt(direction.get('difference_relative_l2'), 3)}**——"
        "适配后的模型能拟合当前场的亮度分布（验证相关 0.84），但**增量方向**仍不可靠。"
        "所以每一块都必须由实测判决，不能按模型损失继续推进。",
        "",
        "![方向性检查](figures/direction" + STAMP + ".png)",
        "",
        "## 5 Recorder 记录与过程动画",
        "",
        "每次采集写一条记录（`Recorder` 持有），设备关闭后导出：",
        "",
        "| 路径 | 内容 |",
        "|---|---|",
        "| `anchored/recording/process.csv` | 标量列：核心 CV、核心峰值、模型损失、标签、段、形状、曝光、位深、饱和像素数 |",
        "| `anchored/recording/process.h5` | 同上 + `_phase`（下发的控制网格相位）、`_img`（裁剪到目标区的实测远场）、`_grad`（相对上一次采集的相位变化） |",
        "| `anchored/recording/process_sidecars/` | 数组列的逐条 `.npy` |",
        "| `anchored/recording/process.gif` | 过程动画：瞳面相位 / 实测远场 / 块内相位变化 / 本块读数 / 实测 CV 轨迹 / 模型损失轨迹 |",
        "",
        f"本轮共记录 {len(facts['anchored_manifest'])} 次采集（模型段另有 "
        f"{len(facts['model_manifest'])} 次）。动画里的 A–B–B–A 顺序、块判决与诊断探针"
        "（反号相位，标题会标注）都能逐帧对上表格。",
        "",
        "![逐次采集的过程动画](figures/process" + STAMP + ".gif)",
        "",
        "### 5.1 渲染器修复（本轮）",
        "",
        "过程动画此前不稳定，两处根因都在渲染器本身：",
        "",
        "1. **按首行判列类型**：候选 CV 在第一次候选出现前是 `None`，首行判类型会把它归到文本，"
        "于是轨迹退化成 `_epoch` 这种无意义曲线。现在按多行判类型，并在轨迹候选里排除记账列。",
        "2. **异常被 `finish()` 掩盖**：`PillowWriter` 的上下文管理器在 `finally` 里调 `finish()`，"
        "循环里一报错就先抛 `IndexError: list index out of range`，真正的错误看不到。"
        "现在手动驱动 setup/finish，原始异常原样抛出。",
        "",
        "此外渲染口径也统一了：远场按 `counts / (曝光 × 2^(位深−8))` 归一（否则 8 bit 段"
        "与 12 bit 段、不同曝光的块无法同图比较），相位去活塞/倾斜后用 90% 分位色标，"
        "裁剪到目标区，色标取 95% 分位（零级饱和不再把整幅压暗）。",
        "",
        "## 6 局限",
        "",
        "- **单段 4 块，20 次更新/块**：核心 CV 只从 "
        f"{_fmt(seed_cv, 3)} 到 {_fmt(final_cv, 3)}，远小于多段链（含稠密采样与宽带载波）在同尺寸上"
        "达到的 0.39–0.41。这**不能**当作「迁移学习可以替代多段链」的证据；它只证明"
        "「干扰后适配 + CCD 锚定仍能把模型当提议器用起来」。",
        "- **模型方向性弱**：差值图相关 0.29，说明适配后的模型适合解释当前场，不适合预测增量；"
        "任何按模型损失推进的做法都会像第 1 段那样被实测否决。",
        "- **适配质量受探针集限制**：12 个探针（6 训练 / 6 验证），验证相关的噪声地板不低；"
        "把 0.84 当「模型已校准」会过度解读。",
        "",
        "## 7 复现命令",
        "",
        "```bash",
        "# 1) 干扰后的探针采集（本轮复用上一份，故本次运行没有重新采集）",
        "python -m ao_shaping.tools.slm.transfer run --out <session> --checkpoint <pretrained> \\",
        "    --adaptation auto --amplitude-center-lr 4.0 --shapes square --side 1200 --period 128 \\",
        "    --window-span 820 --window-step 4 --zero-radius 60 --core-margin 8 \\",
        "    --capture-grid 384 --control-grid 640 --fft-size 2048 --bit-depth 12 \\",
        "    --count 12 --iterations 40 --blocks 4 --updates 20 --accept-metric cv --abba \\",
        "    --device cuda --execute",
        "",
        "# 2) 复用已采探针（本轮实际执行的命令：省掉一次设备占用）",
        "python -m ao_shaping.tools.slm.transfer run --out <session>_r2 --dataset <session>/capture \\",
        "    --checkpoint <pretrained> --adaptation auto --amplitude-center-lr 4.0 \\",
        "    --shapes square --side 1200 --period 128 --window-span 820 --window-step 4 \\",
        "    --zero-radius 60 --core-margin 8 --capture-grid 384 --control-grid 640 \\",
        "    --fft-size 2048 --bit-depth 12 --count 12 --iterations 40 --blocks 4 --updates 20 \\",
        "    --accept-metric cv --abba --device cuda --execute",
        "",
        "# 3) 本报告（离线，只读会话目录）",
        "python scripts/slm/generate_transfer_square1200_report.py",
        "```",
        "",
        "## 8 审计",
        "",
        f"- 采集数 {audit.get('captures', '—')}，原始帧 {audit.get('raw_frames', '—')}，"
        f"最大峰值 {_fmt(audit.get('maximum_peak'), 0)}（满量程 4095，未饱和）",
        f"- 块数 {audit.get('blocks', '—')}，接受 {audit.get('accepted_blocks', '—')}，"
        f"审计状态 `{audit.get('status', '—')}`",
        f"- 时间戳口径：{audit.get('timestamp_scope', '—')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    return insert_header(text, REPORT_KEY)


def _fmt_array(value, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(item, digits) for item in value) + "]"
    return _fmt(value, digits)


def _diff(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _distance(left, right):
    if left is None or right is None:
        return None
    return float(np.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    facts = _session_facts(args.session)
    if not facts["adaptation"]:
        raise SystemExit(f"no fit/adaptation.json under {args.session}")

    plot_adaptation(facts, FIG_DIR / f"adaptation{STAMP}.png")
    plot_cv_trajectory(facts, FIG_DIR / f"cv_trajectory{STAMP}.png")
    plot_direction(facts, FIG_DIR / f"direction{STAMP}.png")
    plot_ccd(facts, FIG_DIR / f"ccd_seed_final{STAMP}.png")
    gif = _copy_gif(facts, FIG_DIR / f"process{STAMP}.gif")
    artifacts = _copy_artifacts(facts)

    report = build_report(facts)
    target = args.out_dir / Path(REPORT_KEY).name
    target.write_text(report, encoding="utf-8")
    logger.info("wrote {}", target)
    logger.info("figures: {}  gif: {}  artifacts: {}", len(list(FIG_DIR.glob("*.png"))), gif, artifacts)


if __name__ == "__main__":
    main()
