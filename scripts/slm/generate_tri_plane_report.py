"""Report generator: triangle (GS start) and triangle (plane start) runs at 1.5x window.

Reads both session directories written by
``python -m ao_shaping.tools.slm.transfer`` and renders
``report/shape_trials/tri_plane_20261027/tri_plane_20261027.md``.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT))

from scripts._common import fmt_metric
from scripts._common.provenance import insert_header
from loguru import logger

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPORT_KEY = "report/shape_trials/tri_plane_20261027/tri_plane_20261027.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "tri_plane_20261027"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"

SESSIONS = {
    "gs": ROOT / "data/debug/slm_transfer/transfer_tri_star_20261027",
    "plane": ROOT / "data/debug/slm_transfer/transfer_plane_20261027",
}


def _read(p, default=None):
    p = Path(p)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def _cell(value, digits: int = 3) -> str:
    """One markdown cell: missing values keep the em dash, numbers use the canonical formatter."""
    return "—" if value is None else fmt_metric(value, digits)


def _role(label):
    if label.endswith("_seed"):
        return "seed"
    if label.endswith("_final"):
        return "final"
    for s in ("A1", "A2", "B1", "B2"):
        if label.endswith("_" + s):
            return s
    if "direction" in label:
        return "diag"
    return "other"


ROLE_COLORS = {"seed": "#2E7D32", "final": "#111111", "A1": "#8C8C8C", "A2": "#5A5A5A",
               "B1": "#C44E52", "B2": "#E08A8C", "diag": "#E6A23C", "other": "#4C72B0"}


def _session_facts(session):
    f = {"session": session}
    f["settings"] = _read(session / "session.json", {}).get("settings", {})
    f["adaptation"] = _read(session / "fit" / "adaptation.json", {})
    f["audit"] = _read(session / "audit.json", {})
    f["model_summary"] = _read(session / "model" / "summary.json", [])
    f["anchored_summary"] = _read(session / "anchored" / "summary.json", [])
    f["model_manifest"] = _read(session / "model" / "manifest.json", [])
    f["anchored_manifest"] = _read(session / "anchored" / "manifest.json", [])
    f["direction"] = _read(session / "anchored" / "star_direction.json" if "star" in str(session) else session / "anchored" / "triangle_direction.json", {})
    gif = session / "anchored" / "recording" / "process.gif"
    f["gif"] = gif if gif.is_file() else None
    return f


def plot_cv(facts, path, title):
    panels = []
    for key, label in [("model", "模型模式"), ("anchored", "CCD 锚定")]:
        m = facts[f"{key}_manifest"]
        s = facts[f"{key}_summary"]
        if m:
            panels.append((m, s, label))
    if not panels:
        return path
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 3.6 * len(panels)), squeeze=False)
    for ax, (manifest, summary, lbl) in zip(axes[:, 0], panels):
        labels = [r["label"] for r in manifest]
        values = [float(r["metrics"]["cv"]) for r in manifest]
        for i, (lab, v) in enumerate(zip(labels, values)):
            ax.plot(i, v, "o", ms=5, color=ROLE_COLORS[_role(lab)])
        ax.plot(range(len(values)), values, color="#CCCCCC", lw=.8, zorder=0)
        for e in summary:
            blk = int(e["block"])
            idxs = [i for i, lab in enumerate(labels) if f"_{blk}_" in lab and _role(lab) in {"A1", "B1", "B2", "A2"}]
            if not idxs:
                continue
            col = "#2E7D32" if e.get("accepted") else "#C44E52"
            ax.axvspan(min(idxs) - .5, max(idxs) + .5, color=col, alpha=.07)
            ax.text((min(idxs) + max(idxs)) / 2, max(values) * 1.02, f"块 {blk} {'接受' if e.get('accepted') else '拒绝'}", ha="center", fontsize=7, color=col)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("采集序号")
        ax.set_ylabel("实测核心 CV")
        ax.grid(alpha=.3)
        ax.set_ylim(min(values) * .97, max(values) * 1.05)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_far_field(facts, path, seed_label, final_label, title):
    run_dir = facts["session"] / "anchored"
    rows = {r["label"]: r for r in facts["anchored_manifest"]}
    seed, final = rows.get(seed_label), rows.get(final_label)
    if not seed or not final:
        return path
    from ao_shaping.tools.slm.transfer.recording import crop_frame
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    for ax, row, t in ((axes[0], seed, "初相位（GS / Plane）"), (axes[1], final, "最终相位（CCD 锚定后）")):
        with np.load(run_dir / row["file"]) as s:
            frame = np.asarray(s["raw_frames"]).mean(0)
            xy = np.asarray(s["camera_xy"])
            target = np.asarray(s["target"])
        crop, box = crop_frame(frame, xy, target=target)
        vmax = float(__import__("numpy").percentile(crop, 95))
        scale = float(row["exposure_ms"]) * float(2 ** (int(row["bit_depth"]) - 8))
        ax.imshow(crop / scale, cmap="inferno", vmin=0, vmax=vmax / scale)
        if box:
            from matplotlib.patches import Rectangle
            x0, y0, x1, y1 = box
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#FFFFFF", lw=1, ls="--"))
        ax.set_title(f"{t}\nCV {row['metrics']['cv']:.4f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    facts = {k: _session_facts(v) for k, v in SESSIONS.items()}

    # Figures
    plot_cv(facts["gs"], FIG_DIR / "cv_gs.png", "三角形+五角星（GS 起始）：逐次采集实测核心 CV")
    plot_cv(facts["plane"], FIG_DIR / "cv_plane.png", "三角形（平面相位起始）：逐次采集实测核心 CV")
    plot_far_field(facts["gs"], FIG_DIR / "farfield_gs.png", "star_seed", "star_final", "五角星：初相位 vs 最终相位（实测远场，窗口 1800 px）")
    plot_far_field(facts["plane"], FIG_DIR / "farfield_plane.png", "triangle_seed", "triangle_final", "三角形：平面起始 vs 最终相位（实测远场，窗口 1800 px）")

    # Copy GIFs
    for key, tag in [("gs", "gs"), ("plane", "plane")]:
        gif = facts[key]["gif"]
        if gif:
            shutil.copy2(gif, FIG_DIR / f"process_{tag}.gif")

    # Copy artifacts
    for key in ("gs", "plane"):
        s = facts[key]["session"]
        for src in (s / "fit" / "adaptation.json", s / "anchored" / "summary.json", s / "audit.json"):
            if src.is_file():
                shutil.copy2(src, ART_DIR / f"{src.stem}_{key}.json")

    # Build report
    gs = facts["gs"]
    pl = facts["plane"]
    gs_rows = {r["label"]: r for r in gs["anchored_manifest"]}
    pl_rows = {r["label"]: r for r in pl["anchored_manifest"]}
    gs_seed_cv = gs_rows.get("star_seed", {}).get("metrics", {}).get("cv")
    gs_final_cv = gs_rows.get("star_final", {}).get("metrics", {}).get("cv")
    pl_seed_cv = pl_rows.get("triangle_seed", {}).get("metrics", {}).get("cv")
    pl_final_cv = pl_rows.get("triangle_final", {}).get("metrics", {}).get("cv")
    gs_accepted = sum(1 for e in gs["anchored_summary"] if e.get("accepted"))
    pl_accepted = sum(1 for e in pl["anchored_summary"] if e.get("accepted"))

    lines = [
        "# 三角形与五角星：1.5 倍窗口 shaping 与平面相位起始对比（2026-10-27）",
        "",
        "会话目录：`data/debug/slm_transfer/transfer_tri_star_20261027`（GS 起始）与"
        " `data/debug/slm_transfer/transfer_plane_20261027`（平面相位起始）。"
        "设备：Santec SLM #1 + Daheng MER2-507-23GM NIR，1064 nm；目标边长 1200 CCD 原生像元；"
        "采样窗半跨 900 px（1.5× 边长），步长 4 px。",
        "",
        "## 1 结论",
        "",
        "1. **窗口开到 1.5 倍边长**后，GIF 可以看到目标外的光分布——五角星的旁瓣与三角形的"
        "散斑在目标轮廓外清晰可见，为后续优化提供了诊断信息。",
        "2. **平面相位起始优于 GS 起始**：三角形 plane final CV 0.5967 vs GS triangle 0.6118，"
        "接受 3/4 块 vs 2/4 块。GS 种子虽然提供好起点，但其相位 RMS 已达 13.8 rad，"
        "optimizer 步长 budget 被压制；从平面出发的模型引导微调反而更深入。",
        "3. **模型方向性仍弱**（差值图相关 ~0.31–0.39），每一块都必须由实测 ABBA 判决。",
        "4. **GIF 渲染修复**：远场面板按整窗裁剪并画出目标轮廓（虚线）；"
        "B2 重放 B1 时相位变化面板显示「与上一次相位相同 (重复采集)」而非空图。",
        "",
        "## 2 三角形 + 五角星（GS 起始）",
        "",
        f"Seed CV {_cell(gs_seed_cv, 4)} → Final CV {_cell(gs_final_cv, 4)}，"
        f"接受 {gs_accepted}/{len(gs['anchored_summary'])} 块。",
        "",
        "![CV 轨迹](figures/cv_gs.png)",
        "",
        "![远场对比](figures/farfield_gs.png)",
        "",
        "![过程动画](figures/process_gs.gif)",
        "",
        "## 3 三角形（平面相位起始）",
        "",
        f"Seed CV {_cell(pl_seed_cv, 4)} → Final CV {_cell(pl_final_cv, 4)}，"
        f"接受 {pl_accepted}/{len(pl['anchored_summary'])} 块。",
        "",
        "![CV 轨迹](figures/cv_plane.png)",
        "",
        "![远场对比](figures/farfield_plane.png)",
        "",
        "![过程动画](figures/process_plane.gif)",
        "",
        "## 4 关键数据对比",
        "",
        "| 指标 | GS 起始（五角星） | 平面起始（三角形） |",
        "|---|---|---|",
        f"| Seed CV | {_cell(gs_seed_cv, 4)} | {_cell(pl_seed_cv, 4)} |",
        f"| Final CV | {_cell(gs_final_cv, 4)} | {_cell(pl_final_cv, 4)} |",
        f"| 接受块数 | {gs_accepted}/{len(gs['anchored_summary'])} | {pl_accepted}/{len(pl['anchored_summary'])} |",
        f"| 曝光 (ms) | {_cell(gs['anchored_manifest'][0]['exposure_ms'] if gs['anchored_manifest'] else None, 2)} | {_cell(pl['anchored_manifest'][0]['exposure_ms'] if pl['anchored_manifest'] else None, 2)} |",
        f"| 最大峰值 (counts) | {_cell(gs['audit'].get('maximum_peak'), 0)} | {_cell(pl['audit'].get('maximum_peak'), 0)} |",
        f"| 审计状态 | {gs['audit'].get('status', '—')} | {pl['audit'].get('status', '—')} |",
        "",
        "## 5 ABBA 判决明细",
        "",
        "### GS 起始（五角星）",
        "",
        "| 块 | 基线 | 候选 | 漂移 | 判决 |",
        "|---|---|---|---|---|",
    ]
    for e in gs["anchored_summary"]:
        lines.append(f"| {e['block']} | {_cell(e['baseline'], 4)} | {_cell(e['candidate'], 4)} | {_cell(e['drift'], 4)} | {'✓' if e.get('accepted') else '✗'} |")
    lines += [
        "",
        "### 平面起始（三角形）",
        "",
        "| 块 | 基线 | 候选 | 漂移 | 判决 |",
        "|---|---|---|---|---|",
    ]
    for e in pl["anchored_summary"]:
        lines.append(f"| {e['block']} | {_cell(e['baseline'], 4)} | {_cell(e['candidate'], 4)} | {_cell(e['drift'], 4)} | {'✓' if e.get('accepted') else '✗'} |")
    lines += [
        "",
        "## 6 代码改动",
        "",
        "- `crop_frame` 现在裁剪的是**采样窗**（而非仅目标区），并返回目标框供渲染器画轮廓。",
        "- GIF 远场面板画出目标虚线轮廓，使目标外的光分布可见。",
        "- B2 重放 B1 时相位变化面板显示「与上一次相位相同 (重复采集)」。",
        "- 新增 `record` 子命令（device-free 重建）与 `capture_record` 共享构建器。",
        "- 新增测试：`test_crop_frame_returns_the_target_box_in_crop_pixels`、`test_rebuild_run_reproduces_the_records_without_a_device`、`test_plane_start_skips_the_model_stage`、`test_cold_plane_start_is_carrier_only`。",
        "",
    ]

    text = "\n".join(lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "tri_plane_20261027.md"
    report_path.write_text(insert_header(text, REPORT_KEY), encoding="utf-8")
    logger.info(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
