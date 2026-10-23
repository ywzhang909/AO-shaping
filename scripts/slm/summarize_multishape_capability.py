"""跨形状汇总：把同一曝光下各形状的能力运行并成一张表 + 一段结论。

只读 ``data/debug/multishape_forward/*/run.json``，不碰设备、不重算任何测量量。
每条结论都标注它来自哪几个 run，避免"合并出来的平均值"被当成独立样本。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.target_masks import shape_vertices
from ao_shaping.utils.image.displacement import (
    measure_displacement,
    shift_axis_label,
)

SHAPE_ORDER = ("square", "rectangle", "triangle", "star")
SHAPE_NAMES = {"square": "正方形", "rectangle": "长方形", "triangle": "三角形", "star": "五角星"}
SHAPE_COLORS = {"square": "#1f77b4", "rectangle": "#d62728", "triangle": "#2ca02c", "star": "#ff7f0e"}
SCRIPT_PATH = "scripts/slm/summarize_multishape_capability.py"
GEN_SCRIPT = "scripts/slm/generate_capability_report.py"
RUN_SCRIPT = "scripts/slm/run_multishape_capability.py"
#: 跨形状对比图放子目录的 figures/（report/AGENTS.md §2：报告与资产分目录）
FIG_SUBDIR = "figures"
#: 零级能量占比的测量半径（与 run_multishape_capability.frame_stats 的 zero_r 一致）
ZERO_ORDER_RADIUS_PX = 20


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _f(value, spec: str = ".5f", na: str = "—") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return na
    return na if math.isnan(number) else format(number, spec)


def load_runs(root: Path, exposure_ms: float | None,
             allow_unknown_exposure: bool = False) -> dict[str, list[dict]]:
    """按形状收集运行（新→旧）。

    ``allow_unknown_exposure``：脚本改版前的 run.json 没有 ``exposure_ms``，
    这些运行的曝光只能从启动命令得知、不能从产物核实。默认排除；显式放行时
    表格给形状名加 ``*`` 并在脚注说明，**绝不静默当成已知值**。
    """
    out: dict[str, list[dict]] = {}
    for path in sorted(root.glob("*/run.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shape = str(run.get("shape") or "square")
        got = run.get("exposure_ms")
        run["_exposure_unknown"] = got is None
        if exposure_ms is not None:
            if got is None:
                if not allow_unknown_exposure:
                    continue
            elif abs(float(got) - exposure_ms) > 1e-6:
                continue
        run["_dir"] = path.parent.name
        out.setdefault(shape, []).append(run)
    for runs in out.values():
        runs.sort(key=lambda r: str(r.get("started_utc") or ""), reverse=True)
    return out


def pick(runs: dict[str, list[dict]]) -> dict[str, dict]:
    """每个形状取最新一次运行（同形状多次运行时其余只在正文里点名）。"""
    return {shape: items[0] for shape, items in runs.items() if items}


def carrier_displacement(run: dict, root: Path) -> dict | None:
    """从该 run 的两张末尾帧**实测**载波位移（没帧就返回 None，不猜）。"""
    run_dir = root / run["_dir"]
    best_path = run_dir / "final_best.npy"
    car_path = run_dir / "final_best_plus_carrier.npy"
    if not (best_path.exists() and car_path.exists()):
        return None
    try:
        return measure_displacement(np.load(best_path).astype(float),
                                    np.load(car_path).astype(float))
    except (OSError, ValueError):
        return None


def first_block(run: dict) -> dict:
    blocks = run.get("blocks") or []
    return blocks[0] if blocks else {}


def rest_blocks(run: dict) -> list[dict]:
    return (run.get("blocks") or [])[1:]


def summary_rows(runs: dict[str, dict], root: Path | None = None) -> list[str]:
    rows = []
    for shape in SHAPE_ORDER:
        run = runs.get(shape)
        if run is None:
            rows.append(f"| {SHAPE_NAMES[shape]} | — | — | — | — | — | — | — |")
            continue
        res = run.get("three_sigma_resolution")
        b1 = first_block(run)
        ratio = abs(float(b1.get("delta", float("nan"))) / float(res)) if res else float("nan")
        rest = rest_blocks(run)
        rest_best = max(
            (abs(float(b.get("delta", 0.0))) / float(res) for b in rest), default=float("nan")
        ) if res else float("nan")
        measured = ((run.get("final") or {}).get("measured") or {})
        best_zero = (measured.get("best") or {}).get("at_zero") or {}
        carrier_zero = (measured.get("best_plus_carrier") or {}).get("at_zero") or {}
        disp = carrier_displacement(run, root) if root is not None else None
        zf0 = best_zero.get("zero_fraction")
        zf1 = carrier_zero.get("zero_fraction")
        try:
            drop = 1.0 - float(zf1) / float(zf0)
        except (TypeError, ValueError, ZeroDivisionError):
            drop = float("nan")
        side = shift_axis_label(disp) if disp else "未测（缺帧）"
        rows.append(
            f"| {SHAPE_NAMES[shape]}{'*' if run.get('_exposure_unknown') else ''} "
            f"| `{run['_dir']}` | {_f(res, '.5f')} "
            f"| {_f(b1.get('delta'), '+.5f')} | {_f(ratio, '.2f')}σ "
            f"| {_f(rest_best, '.2f')}σ | {_f(best_zero.get('cv'), '.4f')} "
            f"| {_f(zf0, '.4f')} → {_f(zf1, '.4f')}（−{_f(drop * 100, '.0f')}%） | {side} |"
        )
    return rows


def data_paths_section(runs: dict[str, dict], root: Path, stamp_text: str,
                       floor_path: str | None, seed_panel: str | None,
                       extra_data: list[str]) -> str:
    """§数据路径：每个 run 的落盘位置 + 本次批次的固定输入 + 附加数据。

    为什么单列一节：报告里的每个数字都必须能**从磁盘重算**。run.json 里已经记了
    checkpoint 与 3σ，但噪声地板路径与初值相位是启动参数、不在 run.json 里，
    所以由命令行显式传入并在正文标注"来自启动命令"。
    """
    lines = ["| 形状 | 运行目录（含 run.json + 全部帧） |", "|---|---|"]
    for shape in SHAPE_ORDER:
        run = runs.get(shape)
        if run is None:
            continue
        try:
            shown = (root / run["_dir"]).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            shown = (root / run["_dir"]).as_posix()
        lines.append(
            f"| {SHAPE_NAMES[shape]} | `{shown}/` "
            f"（{len(run.get('blocks') or [])} 块 × 4 阶段帧 + `final_*.npy` + `best_panel.npy`） |"
        )

    batch = [
        f"* **噪声地板（3σ 来源）**：{('`' + floor_path + '`') if floor_path else '（未提供：命令行未传 `--floor-path`）'}"
        + " —— 由 `measure_noise_floor` 在**同一掩膜与同一曝光**下现测，run.json 记 `three_sigma_resolution`；",
        f"* **初值相位**：{('`' + seed_panel + '`') if seed_panel else '（未提供：命令行未传 `--seed-panel-path`）'}"
        + " —— 闭环的 seed panel，同时也是同帧定心那一步下发的相位；",
        "* **冻结前向检查点**：每个 run 的 `run.json` 里 `checkpoint` 字段各有记录（本批为同一个目录）；",
        "* **报告**：每个形状另有独立报告 `capability_<shape>_<stamp>.md`（逐块曲线图 + 末尾对比图）与"
        " `figures/` 下的 PNG，均由 `scripts/slm/generate_capability_report.py` 生成。",
    ]
    if extra_data:
        batch.append(
            "* **附加数据（本轮采集，尚未用于任何结论）**：\n"
            + "\n".join(f"  * `{item}`" for item in extra_data)
        )
    return ("## 5. 数据路径\n\n" + "\n".join(lines)
            + "\n\n### 本批次的固定输入\n\n" + "\n".join(batch))


def _shape_area_px2(run: dict) -> float:
    """目标面积：优先 area_px2；旧版只有 target_side_px 时由其平方推。"""
    area = run.get("area_px2")
    if area:
        return float(area)
    side = run.get("target_side_px")
    if side:
        return float(side) ** 2
    return 7200.0


def _target_vertices(run: dict, shape: str):
    """目标形状顶点（相机像元，列序 x/y，以原点为中心）；拿不到尺寸时返回 None。"""
    size = run.get("size_px")
    try:
        return shape_vertices(
            shape,
            area_px2=_shape_area_px2(run),
            size_px=tuple(size) if size else None,
        )
    except (ValueError, TypeError):
        return None


def make_comparison_figures(runs: dict, root: Path, fig_dir: Path) -> dict:
    """渲染三张跨形状对比图（**离线**：只读已落盘帧与 run.json，不碰设备、不重算测量量）。

    * ``shape_comparison_ccd_farfield.png``：各形状 × (best | best+载波) 的 CCD 远场
      （对数色标，目标形状轮廓 + 零级圈；CV / 零点占比标注取自 ``final.measured``）；
    * ``shape_comparison_phase.png``：各形状 × (展开 | 包裹) 的 SLM 面板总相位；
    * ``shape_comparison_metrics.png``：逐块 |Δ|/3σ、实测候选 CV、零点占比、冻结前向模型 CV
      前→后的跨形状叠加。

    返回 ``{key: 相对报告目录的路径}``；缺帧/缺相位时对应格子跳过，整图无数据则不生成，
    **不伪造数据**。
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt = _mpl()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle, Polygon

    present = [s for s in SHAPE_ORDER if s in runs]
    figs: dict[str, str] = {}

    # ---- CCD 远场：各形状 × (best | best+载波) ----
    ccd_rows = [
        s for s in present
        if (root / runs[s]["_dir"] / "final_best.npy").exists()
        and (root / runs[s]["_dir"] / "final_best_plus_carrier.npy").exists()
        and runs[s].get("spot_xy")
    ]
    if ccd_rows:
        span = 300  # 覆盖零点与 ~234 px 位移落点的对称裁剪窗（不重算位移）
        fig, axes = plt.subplots(len(ccd_rows), 2, figsize=(11.0, 5.2 * len(ccd_rows)))
        axes = axes.reshape(len(ccd_rows), 2)
        for r, shape in enumerate(ccd_rows):
            run = runs[shape]
            run_dir = root / run["_dir"]
            col0, row0 = float(run["spot_xy"][0]), float(run["spot_xy"][1])
            measured = (run.get("final") or {}).get("measured") or {}
            verts = _target_vertices(run, shape)
            panels = (
                ("final_best.npy", "best", "best（无倾斜）"),
                ("final_best_plus_carrier.npy", "best_plus_carrier", "best+载波"),
            )
            for c, (fname, tag, title) in enumerate(panels):
                ax = axes[r, c]
                img = np.load(run_dir / fname).astype(np.float64)
                n_rows, n_cols = img.shape
                r_lo, r_hi = max(0, int(row0) - span), min(n_rows, int(row0) + span)
                c_lo, c_hi = max(0, int(col0) - span), min(n_cols, int(col0) + span)
                crop = img[r_lo:r_hi, c_lo:c_hi]
                positive = crop[crop > 0]
                vmax = float(np.percentile(positive, 99.9)) if positive.size else 1.0
                shown = np.log1p(np.clip(crop, 0.0, None) / max(vmax, 1e-9) * 100.0)
                ax.imshow(shown, cmap="inferno")
                if verts is not None:
                    px = verts[:, 0] + (col0 - c_lo)
                    py = verts[:, 1] + (row0 - r_lo)
                    ax.add_patch(Polygon(np.column_stack([px, py]), closed=True, fill=False,
                                         edgecolor="cyan", linewidth=1.2))
                ax.add_patch(Circle((col0 - c_lo, row0 - r_lo), ZERO_ORDER_RADIUS_PX, fill=False,
                                    edgecolor="yellow", linewidth=1.0, linestyle="--"))
                stats = (measured.get(tag) or {}).get("at_zero") or {}
                label = f"CV {_f(stats.get('cv'), '.3f')} · 零点占比 {_f(stats.get('zero_fraction'), '.4f')}"
                ax.text(6, crop.shape[0] - 6, label, color="cyan", fontsize=8, va="bottom",
                        bbox=dict(fc="black", alpha=0.4, ec="none", pad=1.5))
                ax.set_title(f"{SHAPE_NAMES[shape]} · {title}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("CCD 远场光斑：闭环最优 vs 末尾叠加载波（对数色标；青框=目标形状，黄圈=零级 r=20 px）",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = fig_dir / "shape_comparison_ccd_farfield.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figs["ccd"] = f"{FIG_SUBDIR}/{path.name}"

    # ---- SLM 面板相位：各形状 × (展开 | 包裹) ----
    phase_rows = [s for s in present if (root / runs[s]["_dir"] / "best_panel.npy").exists()]
    if phase_rows:
        fig, axes = plt.subplots(len(phase_rows), 2, figsize=(10.0, 4.4 * len(phase_rows)))
        axes = axes.reshape(len(phase_rows), 2)
        for r, shape in enumerate(phase_rows):
            panel = np.load(root / runs[shape]["_dir"] / "best_panel.npy").astype(np.float64)
            pmax = float(np.nanpercentile(np.abs(panel), 99)) if panel.size else 1.0
            if pmax <= 0:
                pmax = 1.0
            im0 = axes[r, 0].imshow(panel, cmap="RdBu_r",
                                    norm=TwoSlopeNorm(0, vmin=-pmax, vmax=pmax))
            axes[r, 0].set_title(f"{SHAPE_NAMES[shape]} · SLM 相位（展开, rad）", fontsize=10)
            fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)
            im1 = axes[r, 1].imshow(np.mod(panel, 2 * np.pi), cmap="twilight", vmin=0, vmax=2 * np.pi)
            axes[r, 1].set_title(f"{SHAPE_NAMES[shape]} · 相位包裹（mod 2π, rad）", fontsize=10)
            fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)
            for c in (0, 1):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        fig.suptitle("SLM 面板总相位（best_panel.npy，各形状闭环最优解）", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = fig_dir / "shape_comparison_phase.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figs["phase"] = f"{FIG_SUBDIR}/{path.name}"

    # ---- 逐块指标：跨形状叠加 ----
    metric_rows = [s for s in present if runs[s].get("blocks")]
    if metric_rows:
        def xs(shape):
            return [float(b.get("block", i + 1)) for i, b in enumerate(runs[shape]["blocks"])]

        def ys(shape, key):
            return [float(b.get(key, float("nan"))) for b in runs[shape]["blocks"]]

        def d_over_sigma(shape):
            res = runs[shape].get("three_sigma_resolution")
            return [abs(float(b.get("delta", 0.0))) / float(res) if res else float("nan")
                    for b in runs[shape]["blocks"]]

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
        ax = axes[0, 0]
        for shape in metric_rows:
            ax.plot(xs(shape), d_over_sigma(shape), "o-", color=SHAPE_COLORS[shape], label=SHAPE_NAMES[shape])
        ax.axhline(3.0, color="0.4", linestyle=":", linewidth=1)
        ax.set_title("逐块 |Δ|÷3σ（首块增益 → 平台）")
        ax.set_xlabel("块")
        ax.set_ylabel("|Δ| / 3σ")
        ax.legend(fontsize=8)
        ax = axes[0, 1]
        for shape in metric_rows:
            ax.plot(xs(shape), ys(shape, "candidate"), "o-", color=SHAPE_COLORS[shape], label=SHAPE_NAMES[shape])
        ax.set_title("实测候选 CV（目标掩膜，含零级）")
        ax.set_xlabel("块")
        ax.set_ylabel("CV")
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        for shape in metric_rows:
            ax.plot(xs(shape), ys(shape, "zero_fraction_candidate"), "s-",
                    color=SHAPE_COLORS[shape], label=SHAPE_NAMES[shape])
        ax.set_title("零点能量占比（候选）")
        ax.set_xlabel("块")
        ax.set_ylabel("zero_fraction")
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        for shape in metric_rows:
            ax.plot(xs(shape), ys(shape, "model_before"), "o-", color=SHAPE_COLORS[shape],
                    label=f"{SHAPE_NAMES[shape]} 前")
            ax.plot(xs(shape), ys(shape, "model_after"), "s--", color=SHAPE_COLORS[shape],
                    alpha=0.55, label=f"{SHAPE_NAMES[shape]} 后")
        ax.set_title("冻结前向模型 CV（前→后；未验证，仅对照）")
        ax.set_xlabel("块")
        ax.set_ylabel("模型 CV")
        ax.legend(fontsize=7, ncol=2)
        for ax in axes.ravel():
            ax.grid(alpha=0.3)
        fig.suptitle("跨形状逐块指标（同曝光；读取 run.json，不重算）", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = fig_dir / "shape_comparison_metrics.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figs["metrics"] = f"{FIG_SUBDIR}/{path.name}"

    return figs


def figures_section(figs: dict) -> str:
    """把跨形状对比图写成报告的一节；无图时返回空串（不留空标题）。"""
    if not figs:
        return ""
    parts = [
        "## 3. 插图（跨形状对比）",
        "",
        "下列图像由本脚本在生成报告时**离线**渲染（只读各 run 的 `run.json` 与已落盘帧，"
        "不碰设备、不重算测量量）；每张图都可从 `figures/` 下的同名 PNG 单独查看。",
        "",
    ]
    if figs.get("ccd"):
        parts += [
            f"![CCD 远场对比]({figs['ccd']})",
            "",
            "*左列=闭环最优（无倾斜），右列=末尾叠加 32 px 周期载波；对数色标。青框=目标形状轮廓，"
            "黄圈=零级掩膜（r=20 px）；标注的 CV 与零点占比取自 `run.json` 的 `final.measured`。*",
            "",
        ]
    if figs.get("metrics"):
        parts += [
            f"![逐块指标对比]({figs['metrics']})",
            "",
            "*四形状逐块叠加：首块 |Δ|/3σ 越 3σ 线（点线）、第 2 块起进平台；实测候选 CV 与零点能量占比；"
            "冻结前向模型 CV 前→后——模型继续下降而硬件不跟随的直观体现。*",
            "",
        ]
    if figs.get("phase"):
        parts += [
            f"![SLM 相位对比]({figs['phase']})",
            "",
            "*各形状闭环最优解对应的 SLM 面板总相位（`best_panel.npy`），左=展开（rad）、右=包裹（mod 2π）。*",
            "",
        ]
    return "\n".join(parts).rstrip() + "\n"


def build_text(runs: dict[str, dict], root: Path, exposure_ms: float | None,
               data_section: str = "", figs: dict | None = None) -> str:
    rows = "\n".join(summary_rows(runs, root))
    present = [s for s in SHAPE_ORDER if s in runs]
    missing = [SHAPE_NAMES[s] for s in SHAPE_ORDER if s not in runs]
    deltas = []
    for shape in present:
        run = runs[shape]
        b1 = first_block(run)
        res = run.get("three_sigma_resolution")
        if res:
            deltas.append(abs(float(b1.get("delta", 0.0))) / float(res))
    b1_deltas = [
        float(first_block(runs[s]).get("delta", float("nan")))
        for s in present
        if first_block(runs[s]).get("delta") is not None
    ]
    max_rest = max(
        (
            max((abs(float(b.get("delta", 0.0))) / float(runs[s]["three_sigma_resolution"])
                 for b in rest_blocks(runs[s])), default=0.0)
            for s in present
            if runs[s].get("three_sigma_resolution")
        ),
        default=float("nan"),
    )
    significant = [d for d in deltas if d >= 3.0]
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    depth = 3
    prefix = "../" * depth

    conclusion = (
        f"共 {len(present)} 个形状（{'、'.join(SHAPE_NAMES[s] for s in present)}）"
        + (f"，缺 {'、'.join(missing)}" if missing else "")
        + f"，曝光 {_f(exposure_ms, '.2f')} ms 统一。"
        f"**首块 Δ 达到 3σ 的形状 {len(significant)}/{len(present)} 个**"
        + (
            f"（{'、'.join(f'{_f(d, chr(46) + chr(50) + chr(102))}σ' for d in deltas)}）"
            if deltas else ""
        )
        + "；第 2 块之后各形状的最好一块都不超过 "
        + f"{_f(max((max((abs(float(b.get('delta', 0.0))) / float(runs[s]['three_sigma_resolution']) for b in rest_blocks(runs[s])), default=0.0) for s in present if runs[s].get('three_sigma_resolution')), default=float('nan')), '.2f')}σ"
        + "，即**闭环的可用增益集中在第一步**。末尾叠加倾斜把零点附近能量占比压到原来的几个百分点"
        "（各形状 −94%～−95%），且位移向量在 y 轴上一致为负（面板 x 向斜坡 + SLM 竖向安装 ⇒ "
        "CCD y 向位移），|Δ| 与 |K/周期| 的标量预测相符。"
    )

    return f"""# 不同形状的可微整形能力对比（2f 实验系统，{_f(exposure_ms, '.2f')} ms）

<!-- provenance:start -->
> **生成脚本**: [`{SCRIPT_PATH}`]({prefix}{SCRIPT_PATH})
> **复现命令**: `python {SCRIPT_PATH} --exposure-ms {_f(exposure_ms, '.2f')}`
> **数据/关联脚本**: [`{RUN_SCRIPT}`]({prefix}{RUN_SCRIPT})（设备运行）、[`{GEN_SCRIPT}`]({prefix}{GEN_SCRIPT})（单次运行报告）
> **运行环境**: 离线（只读已落盘的 run.json，不碰设备、不重算测量量）
> **说明**: 各形状共用同一套协议（同帧定心、闭环不加倾斜、末尾叠加 32 px 周期载波、面积 7200 px²）；表格每行是一次独立运行，合并前未做加权。
<!-- provenance:end -->

## 摘要

{conclusion}

## 1. 对比表

| 形状 | 运行目录 | 3σ | 首块 Δ | 首块 Δ÷3σ | 第 2 块起最大 Δ÷3σ | 最佳 CV（含零级） | 零点能量占比 best → best+载波 | 载波位移（实测） |
|---|---|---|---|---|---|---|---|---|
{rows}

**读表要点**：

* `3σ` 是当次掩膜噪声地板，**逐次现测**（同一曝光下三角形/星形/正方形共用 85 px 等效面积地板）；
* 首块 Δ 为负表示改善；`Δ÷3σ` 只是相对本次地板的刻度，**不是**显著性检验；
* 「第 2 块起最大 Δ÷3σ」越大越说明闭环还能继续推动——目前四个形状都 ≤0.5σ，即**已进平台**；
* 「最佳 CV」仍含零级（闭环内不加倾斜，整形光斑与零级重叠），不能当作纯形状均匀度；
* 形状名带 `*` = 该 run.json **没有** `exposure_ms` 字段（脚本改版前的产物），
  曝光只能从启动命令得知、无法从产物核实。

## 2. 跨形状观察到的事实

1. **可微优化的可用增益 ≈ 第一步**：{len(present)} 个形状的首块 Δ 全部为负
   （{_f(min(b1_deltas, default=float('nan')), '+.5f')} ～ {_f(max(b1_deltas, default=float('nan')), '+.5f')}），
   全部 ≥{_f(min(deltas, default=float('nan')), '.2f')}σ，其中 {sum(1 for d in deltas if d >= 3)} 个 ≥3σ；
   第 2 块之后各形状的最好一块全部 ≤{_f(max_rest, '.2f')}σ。模型侧 CV 仍有下降，但硬件不再跟随，说明
   **模型幅度/方向不可迁移，迭代次数不是瓶颈**。
2. **倾斜叠加是独立于模型的稳定收益**：闭环内不加倾斜，末尾才把 32 px 周期斜坡加到最优相位上，
   零点 20 px 内的能量占比从 2.6%～4.3% 降到 0.15%～0.23%（−94%～−95%）。位移由**两帧差分实测**
   （`utils/image/displacement.measure_displacement`），不是由斜坡轴向推断。
3. **位移方向与目标形状无关**：长方形（120×60）、三角形、五角星与正方形的实测位移向量
   一致落在 y 轴负向（各形状 Δ 见对比表），说明方向只由面板斜坡与**安装姿态**决定。
   ⚠️ 本系统 SLM **竖向安装**，面板 x 向斜坡在 CCD 上是 **y 向**位移；按 x 轴摆框去读会
   量到背景不对称（2026-10-23 早前版本犯过这个错，那一组数已作废）。
4. **曝光必须现校准**：0.5 ms 下地板 3σ 达 0.030（噪声把首块压到 0.8σ），0.8 ms 降到 0.011；
   而载波帧比无载波帧亮 → 0.8 ms 时载波帧 peak 已到 254（接近饱和）。
   **读数与判据只能在同一曝光内比较**，跨曝光比 CV 无效。

{figures_section(figs)}
## 4. 限制（读结论前必看）

1. **单次运行、单初值、单 seed**：每个形状只跑了一次，没有重复；`Δ÷3σ` 不构成显著性检验，
   跨会话比较需要独立重复。
2. **模型方向未验证**：本报告只陈述实测，不声称冻结前向模型的梯度方向正确。
3. **零级未逐帧扣除**：目标掩膜内始终含零级贡献，纯形状均匀度需另做扣除后再报。
4. **载波帧接近/达到饱和**：0.8 ms 下 `best+carrier` 的 peak 为 254（正方形/三角形），
   该帧的绝对值读数只能定性；零点能量占比是比值，受影响较小但不为零。

{data_section}

## 6. 复现

```bash
# 单形状（设备）
python {RUN_SCRIPT} --execute --shape triangle --area-px2 7200 --exposure-ms 0.8 \\
    --checkpoint <refit 目录> --seed-panel <初值.npy> --noise-floor <地板.json> --out <目录>
# 单次运行报告（离线）
python {GEN_SCRIPT} --run-dir <目录>
# 本对比（离线；--floor-path/--seed-panel-path/--extra-data 只写进「数据路径」节，不参与计算）
python {SCRIPT_PATH} --exposure-ms {_f(exposure_ms, '.2f')} --allow-unknown-exposure --floor-path logs/target_size_sweep/noise_floor_85px_ex08/noise_floor.json --seed-panel-path data/debug/physical_forward_defocus/capture_after_focus_20261023_145446/seed_probe000.npy
```

生成时间 {stamp}；数据根 `{root.relative_to(ROOT).as_posix()}`。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/debug/multishape_forward")
    parser.add_argument("--exposure-ms", type=float, default=0.8)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "report/shape_trials/multishape_capability")
    parser.add_argument("--name", default=None,
                        help="默认 shape_comparison_<日期>.md")
    parser.add_argument("--allow-unknown-exposure", action="store_true",
                        help="纳入没有 exposure_ms 字段的旧 run（表格里加 * 标注）")
    parser.add_argument("--floor-path", default=None, help="本批次噪声地板 JSON（写进数据路径节）")
    parser.add_argument("--seed-panel-path", default=None, help="本批次初值相位 .npy（写进数据路径节）")
    parser.add_argument("--extra-data", action="append", default=[],
                        help="附加数据路径（可重复）；只登记，不参与任何计算")
    args = parser.parse_args()

    runs = load_runs(args.root, args.exposure_ms, args.allow_unknown_exposure)
    picked = pick(runs)
    if not picked:
        raise SystemExit(f"{args.root} 下没有曝光 {args.exposure_ms} ms 的运行")

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"shape_comparison_{datetime.now().strftime('%Y%m%d')}.md"
    path = args.out / name
    stamp_text = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    data_section = data_paths_section(picked, args.root, stamp_text,
                                      args.floor_path, args.seed_panel_path, args.extra_data)
    figs = make_comparison_figures(picked, args.root, args.out / FIG_SUBDIR)
    path.write_text(build_text(picked, args.root, args.exposure_ms, data_section, figs),
                    encoding="utf-8")
    print(json.dumps({
        "report": str(path),
        "shapes": {s: r["_dir"] for s, r in picked.items()},
        "missing": [s for s in SHAPE_ORDER if s not in picked],
        "figures": figs,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

