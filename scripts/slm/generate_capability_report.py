"""「当前配置下可微整形能力测试」的离线报告生成器（正方形目标 + 末尾叠加倾斜）。

读取 `run_multishape_capability.py` 在 `data/debug/multishape_forward/square_<stamp>/`
落盘的 `run.json` 与原始 CCD 帧（**全程离线，不碰设备**），产出一份独立报告：

* `capability_square_<stamp>.md`：摘要 / 方法与边界 / 闭环逐块表 / 末尾对比表 /
  观察与限制 / 复现命令；
* `<date>_capability_square_<stamp>_blocks.png`：逐块实测与模型 CV 曲线 + ±3σ 带；
* `<stamp>_final_compare.png`：best 与 best+carrier 的 CCD 帧并排（对数色标，
  标注目标框与零级位置）。

容忍不完整运行：`run.json` 缺 `final`（运行未走到末尾）时，末尾对比节降级为
文字说明、对比图不生成；帧文件缺失时对应图跳过并在正文注明，**不伪造数据**。

用法:
    python scripts/slm/generate_capability_report.py \\
        --run-dir data/debug/multishape_forward/square_<stamp>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.target_masks import (
    build_polygon_masks,
    shape_vertices,
)
from ao_shaping.utils.image.displacement import (
    measure_displacement,
    shift_axis_label,
)

DEFAULT_OUT_DIR = "report/shape_trials/multishape_capability"
#: 图统一放子目录的 figures/（report/AGENTS.md §2：报告与资产分目录）
FIG_SUBDIR = "figures"
SCRIPT_PATH = "scripts/slm/generate_capability_report.py"
DATA_SCRIPT_PATH = "scripts/slm/run_multishape_capability.py"
#: 零级能量占比的测量半径（与 run_multishape_capability.frame_stats 的 zero_r 一致）
ZERO_ORDER_RADIUS_PX = 20
#: 目标框相对名义边长的内缩（与 run_multishape_capability.cv_in 的 shrink 一致）
BOX_SHRINK_PX = 3
_STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def extract_stamp(run_dir: Path) -> str:
    """从运行目录名提取时间戳：``square_20261023_181846`` → ``20261023_181846``。"""
    match = _STAMP_RE.search(run_dir.name)
    if match:
        return match.group(1)
    return run_dir.stem or "unknown"


def stamp_date(stamp: str) -> str:
    """戳的日期前缀（图名用）；戳里没有日期时给占位，不猜。"""
    return stamp[:8] if len(stamp) >= 8 and stamp[:8].isdigit() else "00000000"


def load_run_json(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}；运行未开始或 run.json 尚未落盘")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_rel(run_dir: Path) -> str:
    """运行目录的仓库根相对路径（报告正文与复现命令用）；不在仓库内时退回目录名。"""
    try:
        return run_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return run_dir.name


def delta_over_3sigma(delta: float, resolution: float) -> float:
    """Δ÷3σ = 实测 Δ 除以本次掩膜的 3σ 分辨率（带符号）。分辨率缺失/非正 → NaN。"""
    try:
        d = float(delta)
        r = float(resolution)
    except (TypeError, ValueError):
        return float("nan")
    if not (r > 0.0) or math.isnan(d):
        return float("nan")
    return d / r


def _f(value, spec: str = ".5f", na: str = "—") -> str:
    """把可能缺失/NaN 的 JSON 数值格式化成表格文本。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(v) or math.isinf(v):
        return na
    return format(v, spec)


def _fsigned(value, spec: str = ".5f", na: str = "—") -> str:
    """同 _f，但强制带符号（正数 +、负数 − 由 format 处理）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(v) or math.isinf(v):
        return na
    return format(v, "+" + spec)


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
SHAPE_NAMES = {
    "square": "正方形",
    "rectangle": "长方形",
    "triangle": "三角形",
    "star": "五角星",
}


def shape_key(run: dict) -> str:
    """run.json 的形状键；旧版 run（仅有 target_side_px）视为正方形。"""
    return str(run.get("shape") or "square")


def shape_label(run: dict) -> str:
    """中文形状名（报告正文与标题用）。"""
    key = shape_key(run)
    return SHAPE_NAMES.get(key, key)


def shape_size_text(run: dict) -> str:
    """一行尺寸描述：按形状给出最有意义的那个尺度，并写出面积（跨形状可比的口径）。"""
    key = shape_key(run)
    area = run.get("area_px2")
    area_text = f"面积 {float(area):.0f} px²" if area else "面积未记录"
    rotation = run.get("rotation_deg")
    rot_text = f"、旋转 {float(rotation):.1f}°" if rotation else ""
    if key == "square":
        side = run.get("target_side_px")
        if side is None and area:
            side = float(area) ** 0.5
        return f"正方形边长 {_f(side, '.2f')} px（{area_text}{rot_text}）"
    if key == "rectangle":
        size = run.get("size_px") or run.get("size") or []
        if isinstance(size, str):
            size = [float(v) for v in size.replace("x", ",").split(",") if v.strip()]
        if len(size) == 2:
            return (f"长方形 {float(size[0]):.0f}×{float(size[1]):.0f} px"
                    f"（{area_text}{rot_text}）")
        return f"长方形（{area_text}{rot_text}）"
    return f"{shape_label(run)}（{area_text}{rot_text}）"


def plot_blocks(run: dict, path: Path) -> dict:
    """逐块曲线：实测 baseline/candidate（带 ±3σ 带）+ 模型 CV 前后。"""
    blocks = run.get("blocks") or []
    if not blocks:
        return dict(ok=False, reason="run.json 尚无闭环块数据")
    resolution = run.get("three_sigma_resolution")
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    x = [float(b.get("block", i + 1)) for i, b in enumerate(blocks)]
    baseline = [float(b.get("baseline", float("nan"))) for b in blocks]
    candidate = [float(b.get("candidate", float("nan"))) for b in blocks]
    if resolution is not None and not math.isnan(float(resolution)) and float(resolution) > 0:
        res = float(resolution)
        for xi, value in zip(x, baseline):
            ax1.fill_between([xi - 0.3, xi + 0.3], [value - res, value - res],
                             [value + res, value + res], color="0.55", alpha=0.18)
        res_text = f"3σ = {res:.5f}"
    else:
        res_text = "3σ 缺失"
    ax1.plot(x, baseline, "o-", color="#7f7f7f", label="实测基线 (A1/A2)")
    ax1.plot(x, candidate, "s-", color="#d62728", label="实测候选 (B1/B2)")
    for xi, cand, block in zip(x, candidate, blocks):
        ratio = delta_over_3sigma(block.get("delta"), resolution)
        if not math.isnan(ratio):
            ax1.annotate(f"Δ/3σ={ratio:+.1f}", (xi, cand), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8)
    ax1.set_ylabel("实测 CV（目标掩膜）")
    ax1.set_title(f"闭环逐块实测（{len(blocks)} 块；{res_text}）")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    model_before = [float(b.get("model_before", float("nan"))) for b in blocks]
    model_after = [float(b.get("model_after", float("nan"))) for b in blocks]
    ax2.plot(x, model_before, "o-", color="#1f77b4", label="模型 CV（块起点）")
    ax2.plot(x, model_after, "s--", color="#17becf", label="模型 CV（块候选）")
    ax2.set_xlabel("块")
    ax2.set_ylabel("模型 CV（同一掩膜）")
    ax2.set_title("冻结前向模型的预测（未验证，仅对照）")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(ok=True, path=str(path), blocks=len(blocks),
                resolution=float(resolution) if resolution is not None else None)


def plot_final_compare(run: dict, run_dir: Path, path: Path,
                       final_measured: dict | None = None) -> dict:
    """best 与 best+carrier 的 CCD 帧并排（对数色标），标注目标框与零级位置。"""
    best_path = run_dir / "final_best.npy"
    carrier_path = run_dir / "final_best_plus_carrier.npy"
    missing = [p.name for p in (best_path, carrier_path) if not p.exists()]
    spot = run.get("spot_xy")
    if missing or not (spot and len(spot) == 2):
        reason = f"缺少最终帧：{', '.join(missing)}" if missing else "run.json 缺 spot_xy，无法定位目标框"
        return dict(ok=False, reason=reason)
    try:
        img_best = np.load(best_path).astype(np.float64)
        img_carrier = np.load(carrier_path).astype(np.float64)
    except Exception as exc:
        return dict(ok=False, reason=f"最终帧读取失败：{exc}")
    if img_best.ndim != 2 or img_carrier.ndim != 2:
        return dict(ok=False, reason="最终帧不是一维二维图像数组")

    final = run.get("final") or {}
    measured = (final.get("measured") or {}) if final else {}
    try:
        shift = float(final.get("shift_px", 0.0))
    except (TypeError, ValueError):
        shift = 0.0
    axis = run.get("carrier_axis", "x")
    col0, row0 = float(spot[0]), float(spot[1])
    disp = (final_measured or {}).get("displacement") or {}
    d_col = float(disp.get("dx") or 0.0)
    d_row = float(disp.get("dy") or 0.0)
    try:
        side = int(float(run.get("target_side_px") or (float(run.get("area_px2") or 0) ** 0.5)))
    except (TypeError, ValueError):
        side = 0
    half = max(0, side // 2 - BOX_SHRINK_PX)
    n_rows, n_cols = img_best.shape
    span = 170
    # 裁剪窗口必须同时覆盖零点与**实测落点**（不再按斜坡轴向猜）
    r_lo = max(0, int(min(row0, row0 + d_row)) - span)
    r_hi = min(n_rows, int(max(row0, row0 + d_row)) + span)
    c_lo = max(0, int(min(col0, col0 + d_col)) - span)
    c_hi = min(n_cols, int(max(col0, col0 + d_col)) + span)
    if c_hi <= c_lo or r_hi <= r_lo:
        return dict(ok=False, reason="运行目录的帧太小，裁剪窗口为空")

    plt = _mpl()
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    panels = (
        (axes[0], img_best, "best（闭环最优，无倾斜）", "best", [((col0, row0), "at_zero")]),
        (axes[1], img_carrier, "best+carrier（末尾叠加倾斜）", "best_plus_carrier",
         [((col0, row0), "零点"),
          ((col0 + d_col, row0 + d_row), "实测落点")]),
    )
    for ax, img, title, tag, box_centres in panels:
        crop = img[r_lo:r_hi, c_lo:c_hi]
        positive = crop[crop > 0]
        vmax = float(np.percentile(positive, 99.9)) if positive.size else 1.0
        shown = np.log1p(np.clip(crop, 0.0, None) / max(vmax, 1e-9) * 100.0)
        ax.imshow(shown, cmap="inferno")
        stats = measured.get(tag) or {}
        for (cx, cy), name in box_centres:
            x_px, y_px = cx - c_lo - half, cy - r_lo - half
            width = height = 2 * half + 1
            if width <= 0 or height <= 0:
                continue
            if x_px + width <= 0 or x_px >= crop.shape[1] or y_px + height <= 0 or y_px >= crop.shape[0]:
                continue  # 框在裁剪窗外，不画（数据里仍有该点的数字）
            ax.add_patch(Rectangle((x_px, y_px), width, height, fill=False,
                                   edgecolor="cyan", linewidth=1.2))
            row = stats.get(name) or {}
            label = f"{name}: CV {_f(row.get('cv'), '.3f')}, mean {_f(row.get('mean_core'), '.2f')}"
            tx = min(max(x_px, 0), crop.shape[1] - 1)
            ty = min(max(y_px + height + 4, 0), crop.shape[0] - 12)
            ax.text(tx, ty, label, color="cyan", fontsize=7.5,
                    bbox=dict(fc="black", alpha=0.4, ec="none", pad=1.5))
        ax.add_patch(Circle((col0 - c_lo, row0 - r_lo), ZERO_ORDER_RADIUS_PX, fill=False,
                            edgecolor="yellow", linewidth=1.0, linestyle="--"))
        ax.text(crop.shape[1] - 6, crop.shape[0] - 6, "黄圈 = 零级掩膜 (r=20 px)",
                color="yellow", fontsize=8, ha="right", va="bottom")
        ax.set_title(f"{title}（零点 ({int(col0)}, {int(row0)})）", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"末尾对比：倾斜（周期 {_f(run.get('carrier_period_px'), '.0f')} px, 轴 {axis}）"
                 f"把整形光斑与零级分离；shift ≈ {_f(shift, '.1f')} px", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(ok=True, path=str(path), shift_px=shift, axis=axis)


# --------------------------------------------------------------------------- #
# 报告正文
# --------------------------------------------------------------------------- #
def _provenance_block(run_dir: Path, out_dir: Path, report_name: str, label: str) -> str:
    """报告放在仓库内时嵌入溯源块（与 sync_report_provenance 的版式一致）。

    仓库外（如测试临时目录）返回空串，避免生成解析不了的相对链接。
    """
    try:
        depth = len(out_dir.resolve().relative_to(ROOT).parts)
    except ValueError:
        return ""
    prefix = "../" * depth
    run_rel = _run_rel(run_dir)
    lines = [
        f"> **生成脚本**: [`{SCRIPT_PATH}`]({prefix}{SCRIPT_PATH})",
        f"> **复现命令**: `python {SCRIPT_PATH} --run-dir {run_rel}`",
        f"> **数据/关联脚本**: [`{DATA_SCRIPT_PATH}`]({prefix}{DATA_SCRIPT_PATH})",
        "> **运行环境**: 离线（读硬件运行落盘产物）",
        f"> **说明**: 当前配置下可微整形能力测试（{label}目标 {run_dir.name}）的离线报告；"
        "逐块 A-B-B-A 实测 + 末尾 best/best+carrier 对比。模型方向未验证，报告只陈述实测。",
    ]
    return "<!-- provenance:start -->\n" + "\n".join(lines) + "\n<!-- provenance:end -->"


def measure_final(run: dict, run_dir: Path) -> dict | None:
    """从末尾两帧**实测**载波位移，并在「零级位置」与「实测落点」两处重算掩膜读数。

    为什么不沿用 run.json 里的 ``at_plus`` / ``at_minus``：那两个框是沿**面板斜坡方向**
    摆的，而面板方向与 CCD 位移方向之间隔着安装姿态——本系统 SLM 竖向安装，面板 x 向
    斜坡在 CCD 上表现为 **y 向**位移。2026-10-23 曾按 x 摆框，量到的是背景不对称（已作废）。
    """
    best_path = run_dir / "final_best.npy"
    car_path = run_dir / "final_best_plus_carrier.npy"
    if not (best_path.exists() and car_path.exists()):
        return None
    best = np.load(best_path).astype(np.float64)
    car = np.load(car_path).astype(np.float64)
    displacement = measure_displacement(best, car)
    if displacement is None:
        return {"displacement": None, "stats": None}

    spot = run.get("spot_xy")
    if not (spot and len(spot) == 2):
        return {"displacement": displacement, "stats": None}

    size = run.get("size_px") or run.get("size") or []
    if isinstance(size, str):
        size = [float(v) for v in size.replace("x", ",").split(",") if v.strip()]
    vertices = shape_vertices(
        shape_key(run), area_px2=float(run.get("area_px2") or 7200.0),
        size_px=tuple(float(v) for v in size) if len(size) == 2 else None,
        rotation_deg=float(run.get("rotation_deg") or 0.0),
    )
    col, row = int(round(float(spot[0]))), int(round(float(spot[1])))
    box, core = build_polygon_masks(best.shape, (col, row), vertices,
                                    inner_shrink_px=3.0, step_ccd_px=(1.0, 1.0))
    yy, xx = np.mgrid[: best.shape[0], : best.shape[1]]
    zero = (xx - col) ** 2 + (yy - row) ** 2 <= ZERO_ORDER_RADIUS_PX ** 2

    def _stats(frame: np.ndarray, box_m: np.ndarray, core_m: np.ndarray) -> dict:
        values = frame[core_m]
        mean = float(values.mean()) if values.size else float("nan")
        cv = (float(np.sqrt((((values / mean) - 1) ** 2).mean()))
              if mean and mean > 0 else float("nan"))
        total = float(frame.sum())
        return dict(cv=round(cv, 4), mean_core=round(mean, 3), peak=float(frame.max()),
                    zero_fraction=(round(float(frame[zero].sum() / total), 4)
                                   if total else float("nan")))

    d_row = int(round(float(displacement["dy"])))
    d_col = int(round(float(displacement["dx"])))
    moved_box = np.roll(np.roll(box, d_row, axis=0), d_col, axis=1)
    moved_core = np.roll(np.roll(core, d_row, axis=0), d_col, axis=1)
    return {
        "displacement": displacement,
        "stats": {
            "best_at_zero": _stats(best, box, core),
            "carrier_at_zero": _stats(car, box, core),
            "carrier_at_displaced": _stats(car, moved_box, moved_core),
        },
    }


def build_report_text(run: dict, run_dir: Path, out_dir: Path, stamp: str,
                      report_name: str, fig_blocks: dict, fig_final: dict,
                      final_measured: dict | None = None) -> str:
    blocks: list[dict] = run.get("blocks") or []
    resolution = run.get("three_sigma_resolution")
    final = run.get("final")
    measured = (final.get("measured") or {}) if final else {}
    side = run.get("target_side_px")
    label = shape_label(run)
    size_text = shape_size_text(run)
    period = run.get("carrier_period_px")
    axis = run.get("carrier_axis", "x")
    spot = run.get("spot_xy")
    started = run.get("started_utc")
    checkpoint = run.get("checkpoint")
    run_rel = _run_rel(run_dir)

    blocks_png = f"{FIG_SUBDIR}/{stamp_date(stamp)}_capability_{shape_key(run)}_{stamp}_blocks.png"
    final_png = f"{FIG_SUBDIR}/{stamp}_final_compare.png"

    # ---- 摘要里的关键数 ----
    best_cv = best_block = None
    if final and final.get("best_cv") is not None:
        best_cv, best_block = float(final["best_cv"]), final.get("best_block")
    elif blocks:
        best = min(blocks, key=lambda b: float(b.get("candidate", float("inf"))))
        best_cv = float(best.get("candidate", float("nan")))
        best_block = best.get("block")
    best_zero = (measured.get("best") or {}).get("at_zero") or {}
    car_zero = (measured.get("best_plus_carrier") or {}).get("at_zero") or {}
    # 实测口径（优先）：位移从两帧差分量出，读数在实测落点重算
    displacement = (final_measured or {}).get("displacement")
    local = (final_measured or {}).get("stats") or {}
    disp_zero = (local.get("carrier_at_zero") or {})
    disp_moved = (local.get("carrier_at_displaced") or {})
    if local:
        # §3 三列必须同源：都用 measure_final 在同一帧、同一掩膜上算出的读数
        best_zero = local.get("best_at_zero") or best_zero
        car_zero = local.get("carrier_at_zero") or car_zero
    car_plus = {"cv": disp_moved.get("cv"), "mean_core": disp_moved.get("mean_core"),
                "peak": disp_moved.get("peak"),
                "zero_fraction": disp_moved.get("zero_fraction")}
    car_minus: dict = {}

    def _side_text() -> str:
        """位移方向**只**来自两帧实测（`measure_displacement`），不从斜坡轴向推断。"""
        if displacement:
            return f"**{shift_axis_label(displacement)}**"
        return "**无法判定**（两帧缺失或位移不可辨识；面板斜坡轴向 ≠ CCD 位移轴向，不可推断）"

    zf_before = _f(best_zero.get("zero_fraction"), ".4f")
    zf_after = _f(car_zero.get("zero_fraction"), ".4f")
    try:
        zf_ratio = (float(best_zero.get("zero_fraction", float("nan")))
                    / float(car_zero.get("zero_fraction", float("nan"))))
    except (TypeError, ValueError):
        zf_ratio = float("nan")

    if final:
        summary = (
            f"当前配置下（{size_text}、载波周期 {_f(period, '.0f')} px、轴 {axis}、"
            f"同帧定心、闭环不加倾斜）跑完 {len(blocks)} 个闭环块，目标掩膜上**零级重叠处**的最佳实测 CV 为 "
            f"**{_f(best_cv, '.4f')}**（块 {best_block}）；末尾叠加倾斜后整形光斑实测移至{_side_text()}，"
            f"零点 {ZERO_ORDER_RADIUS_PX} px 内能量占比 {zf_before} → {zf_after}"
            + (f"（约 {zf_ratio:.0f}× 下降）" if not math.isnan(zf_ratio) and zf_ratio > 1 else "")
            + f"；落点处目标掩膜 CV = {_f(car_plus.get('cv') if car_plus else float('nan'), '.4f')}，"
              f"但该帧 peak = {_f(disp_moved.get('peak'), '.1f')}（接近饱和）⇒ **不作为质量读数**。"
              "**模型方向尚未验证，本报告只陈述实测。**"
        )
    else:
        summary = (
            f"当前配置下（{size_text}、载波周期 {_f(period, '.0f')} px、轴 {axis}、"
            f"同帧定心、闭环不加倾斜）跑完 {len(blocks)} 个闭环块，目标掩膜上零级重叠处的最佳实测 CV 为 "
            f"**{_f(best_cv, '.4f')}**（块 {best_block}）；末尾对比（best vs best+carrier）尚未执行"
            f"（run.json 无 `final`）。**模型方向尚未验证，本报告只陈述实测。**"
        )

    # ---- §2 逐块表 ----
    if blocks:
        res_ok = resolution is not None and not math.isnan(float(resolution)) and float(resolution) > 0
        rows = []
        for b in blocks:
            ratio = delta_over_3sigma(b.get("delta"), resolution) if res_ok else float("nan")
            rows.append(
                f"| {b.get('block', '—')} "
                f"| {_f(b.get('baseline'))} "
                f"| {_f(b.get('candidate'))} "
                f"| {_fsigned(b.get('delta'))} "
                f"| {_fsigned(ratio, '.2f')} "
                f"| {_fsigned(b.get('drift'))} "
                f"| {_f(b.get('zero_fraction_baseline'), '.4f')} "
                f"| {_f(b.get('zero_fraction_candidate'), '.4f')} "
                f"| {_f(b.get('model_before'), '.4f')} → {_f(b.get('model_after'), '.4f')} |"
            )
        block_table = (
            "| 块 | 基线 CV | 候选 CV | Δ | Δ÷3σ | 漂移 (A2−A1) | 零级占比·基线 | 零级占比·候选 | 模型 CV 前→后 |\n"
            "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
            + "\n\nΔ÷3σ = Δ ÷ 本次等效面积掩膜的 3σ（= "
            + (_f(resolution, '.5f') if res_ok else "缺失")
            + "）；漂移 = A2−A1 的同相位重复差。"
        )
    else:
        block_table = "（run.json 尚无闭环块数据：本次运行还没写出任何一块的 A-B-B-A 记录。）"

    # ---- §3 末尾对比表 ----
    if final:
        shift = _f(final.get("shift_px"), ".1f")
        mag = _f((displacement or {}).get("magnitude"), ".1f")
        table3 = (
            "| 指标 | best @零点 | best+载波 @零点 | best+载波 @**实测落点** |\n"
            "|---|---|---|---|\n"
            f"| CV（目标掩膜） | {_f(best_zero.get('cv'), '.4f')} | {_f(car_zero.get('cv'), '.4f')} "
            f"| {_f(car_plus.get('cv'), '.4f')} |\n"
            f"| mean_core（目标掩膜） | {_f(best_zero.get('mean_core'), '.3f')} "
            f"| {_f(car_zero.get('mean_core'), '.3f')} | {_f(car_plus.get('mean_core'), '.3f')} |\n"
            f"| 零点 {ZERO_ORDER_RADIUS_PX} px 内能量占比 | {_f(best_zero.get('zero_fraction'), '.4f')} "
            f"| {_f(car_zero.get('zero_fraction'), '.4f')} "
            f"| {_f(car_plus.get('zero_fraction'), '.4f')} |\n"
            f"| peak（全帧） | {_f(best_zero.get('peak'), '.1f')} | {_f(car_zero.get('peak'), '.1f')} "
            f"| {_f(disp_moved.get('peak'), '.1f')} |"
        )
        section3 = (
            f"末尾展示两组相位：`best`（闭环最优，无倾斜）与 `best+carrier`（叠加周期 "
            f"{_f(period, '.0f')} px 的**面板 {axis} 向**斜坡；几何预测标量 |K/周期| ≈ {shift} px）。\n\n"
            f"{table3}\n\n"
            f"载波把整形光斑实测移到{_side_text()}（|Δ| ≈ {mag} px）；三列读数的差别就是"
            f"「整形光斑还在零级位置」与「光斑已经离开、零点只剩未调制残光」的差别"
            f"（零点能量占比 {_f(best_zero.get('zero_fraction'), '.4f')} → "
            f"{_f(car_zero.get('zero_fraction'), '.4f')}）。"
        )
        if fig_final.get("ok"):
            section3 += f"\n\n![末尾对比（对数色标）]({final_png})"
        else:
            section3 += (
                f"\n\n> 末尾对比图未生成：{fig_final.get('reason', '未知原因')}（不伪造数据；"
                "表内数字仍取自 run.json）。"
            )
    else:
        section3 = (
            "（末尾对比未执行：run.json 尚无 `final` 节——闭环块可能还没跑完，或运行在末尾展示前中断。"
            "对比表与对比图待运行完整后由本脚本重跑生成。）"
        )

    # ---- §4 观察与限制 ----
    mask_info = run.get("mask") or {}
    frame_area = mask_info.get("frame_box_area")
    frame_core = mask_info.get("frame_core_area")
    area_nominal = run.get("area_px2")
    zero_area = math.pi * ZERO_ORDER_RADIUS_PX ** 2
    if frame_area and area_nominal:
        fill_text = (f"栅格化后的目标掩膜 {int(frame_area)} px²、核心掩膜 {int(frame_core)} px²"
                     f"（相对名义 {float(area_nominal):.0f} px² 的填充率 "
                     f"{float(frame_area) / float(area_nominal):.1%}）")
        zero_share = zero_area / float(frame_area)
    else:
        try:
            side_int = int(float(side))
        except (TypeError, ValueError):
            side_int = 0
        box_side = (2 * (side_int // 2 - BOX_SHRINK_PX) + 1
                    if side_int >= 8 * BOX_SHRINK_PX else max(1, side_int))
        box_area = float(box_side) ** 2
        fill_text = f"测量框边长 {box_side} px（旧版 run 未记录掩膜像元数）"
        zero_share = zero_area / box_area if box_area > 0 else float("nan")
    model_before_1 = blocks[0].get("model_before") if blocks else None
    model_after_n = blocks[-1].get("model_after") if blocks else None
    measured_cv_vals: list[float] = []
    for b in blocks:
        for key in ("baseline", "candidate"):
            try:
                v = float(b.get(key))
            except (TypeError, ValueError):
                continue
            if not math.isnan(v):
                measured_cv_vals.append(v)
    measured_cv_note = f"{sum(measured_cv_vals) / len(measured_cv_vals):.2f}" if measured_cv_vals else "—"
    obs = [
        f"1. **目标掩膜口径**：{fill_text}。零级掩膜（r = {ZERO_ORDER_RADIUS_PX} px 圆盘，"
        f"面积 ≈ {zero_area:.0f} px²）占目标掩膜面积 {_f(zero_share, '.1%')} —— 目标掩膜 CV 是"
        f"“{label} + 零级”的复合指标，不是目标形状单独的均匀度。",
        f"2. **零级扣除的必要性**：本次零级能量占比 ≈ {zf_before}（目标框内），零级峰正对框中心，"
        f"不扣除零级时任何均匀度读数都被零级主导（本次逐块实测 CV ≈ {measured_cv_note} 即源于此）；"
        f"本运行只给出“带零级”与“载波分离后”两种读数，未做逐帧零级扣除，"
        f"方斑单独的 CV 尚不能从本运行给出。",
        f"3. **单次运行不做显著性**：{len(blocks)} 个块同一次会话、同一初值、同一 seed，"
        f"“Δ÷3σ”只是相对本次掩膜噪声地板的参考刻度，不构成显著性检验；"
        f"跨会话/跨光路比较需要独立的重复运行。",
        "4. **闭环内未加倾斜 ⇒ 目标位置与零级重叠**：优化在零级位置上做目标形状，"
        "目标框中心即零级中心，闭环逐块的 CV 因此含零级贡献；倾斜只在末尾展示时叠加，"
        "分离后的读数（§3）才是“方斑离开零级”的状态。",
        f"5. **模型方向未验证**：模型 CV 逐块基本平走（{_f(model_before_1, '.3f')} → "
        f"{_f(model_after_n, '.3f')}），与实测同量级但无方向性信息——"
        "本运行不能证明冻结前向模型的梯度方向正确，摘要的声明以本节为准。",
        "6. **位移方向必须实测，不能由斜坡轴向推断**：本实验系统的 SLM **竖向安装**，"
        "面板 x 向斜坡在 CCD 上给出的是 **y 向**位移（本报告实测 "
        f"{_f((displacement or {}).get('dx'), '+.1f')}, {_f((displacement or {}).get('dy'), '+.1f')}"
        f" px，|Δ| = {_f((displacement or {}).get('magnitude'), '.1f')} px，与 |K/周期| = "
        f"{_f(final.get('shift_px') if final else float('nan'), '.1f')} px 的标量一致）。"
        "⚠️ 2026-10-23 早前版本按 x 轴摆了 `at_plus`/`at_minus` 两个框，量到的是**背景不对称**，"
        "那两组数与由此得出的「＋x 侧」结论已作废；本版改为从两帧差分实测位移后再取读数。",
        f"7. **载波可能伤均匀度（提示，非结论）**：best@零点 CV = {_f(best_zero.get('cv'), '.4f')}，"
        f"载波@实测落点 CV = {_f(disp_moved.get('cv'), '.4f')}。两列口径不同（零点那列含零级峰，"
        "落点那列是纯图案 + 载波量化误差），**不能直接相减归因**。但周期 32 px 的斜坡在面板上"
        "约只有 32 个灰阶/2π，量化偏粗、易生谐波；若下一步要同时满足「与零级分离」和「形状均匀」，"
        "应先换更长周期（P = 64/96 ⇒ 位移约 117/78 px，仍在视场内）再比一次。",
    ]
    observations = "\n\n".join(obs)

    # ---- 图引用 ----
    if fig_blocks.get("ok"):
        blocks_fig_md = f"![逐块曲线（±3σ 带）]({blocks_png})\n"
    else:
        blocks_fig_md = (
            f"> 逐块曲线图未生成：{fig_blocks.get('reason', '未知原因')}（不伪造数据）。\n"
        )

    # ---- 复现 ----
    if spot and len(spot) == 2:
        try:
            spot_text = f"({int(float(spot[0]))}, {int(float(spot[1]))})"
        except (TypeError, ValueError):
            spot_text = "（spot_xy 解析失败）"
    else:
        spot_text = "（run.json 未记录 spot_xy）"
    provenance = _provenance_block(run_dir, out_dir, report_name, label)
    block_lines = []
    if provenance:
        block_lines = [provenance, ""]
    text = f"""# 当前配置下的可微整形能力测试（{label}目标 + 末尾叠加倾斜）

{chr(10).join(block_lines)}## 摘要

{summary}

{blocks_fig_md}
## 1. 方法与边界

* **同帧定心**：先显示初值相位抓一帧，用**该帧**的 argmax 定位零级实测位置
  {spot_text}（col, row），同时用它建模型掩膜与实测掩膜——不再按历史 spot
  定方向后再在设备侧重定中心。
* **闭环不加倾斜**：优化在零级位置上做{label}；倾斜（载波周期 {_f(period, '.0f')} px，
  轴 {axis}）只在末尾展示时叠加，使整形光斑与零级分离（零级由未调制光直接透过，留在原位）。
* **判据**：本次等效面积掩膜的 3σ = {_f(resolution, '.5f')}（run.json 的
  `three_sigma_resolution`，来自当次实测的噪声地板）；Δ÷3σ 只作噪声参考。
* **一次运行的局限**：单一会话、单一初值相位、单一 seed；冻结前向模型检查点
  `{checkpoint if checkpoint else '（未记录）'}`；运行起始 {started if started else '（未记录）'}（UTC）。

## 2. 闭环逐块表

{block_table}

## 3. 末尾对比

{section3}

## 4. 观察与限制

{observations}

## 5. 复现

```bash
# 1) 设备运行（数据生产，需 2f 实验系统；参数见该脚本 docstring）
python scripts/slm/run_multishape_capability.py --execute --wait-until-idle 1800 \\
    --checkpoint <refit 目录> --seed-panel <初值.npy> --noise-floor <噪声地板.json> \\
    --target-side-px {_f(side, '.2f')} --carrier-period-px {_f(period, '.0f')} \\
    --blocks {len(blocks)} --out data/debug/multishape_forward/{run_dir.name}

# 2) 本报告（离线，不碰设备）
python scripts/slm/generate_capability_report.py --run-dir {run_rel}
```

原始帧（`block{{N}}_{{A1|B1|B2|A2}}_*.npy`、`final_best.npy`、`final_best_plus_carrier.npy`、
`best_panel.npy`）与 `run.json` 全部在 `{run_rel}/`；本脚本只读这些文件。
"""
    return text


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="run_multishape_capability.py 的运行目录（含 run.json 与帧）")
    parser.add_argument("--out-dir", type=Path, default=ROOT / DEFAULT_OUT_DIR,
                        help=f"报告输出目录（默认 {DEFAULT_OUT_DIR}，独立于 defocus_differentiable/）")
    parser.add_argument("--stamp", default=None,
                        help="时间戳（默认从 run-dir 名提取，如 20261023_181846）")
    parser.add_argument("--report-name", default=None,
                        help="报告文件名（默认 capability_<shape>_<stamp>.md）")
    parser.add_argument("--force", action="store_true",
                        help="允许覆盖已存在且指向另一次运行的报告（默认拒绝）")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    stamp = args.stamp or extract_stamp(run_dir)
    if not (run_dir / "run.json").exists():
        raise SystemExit(f"未找到 {run_dir / 'run.json'}；运行未开始或 run.json 尚未落盘")
    run = load_run_json(run_dir)
    report_name = args.report_name or f"capability_{shape_key(run)}_{stamp}.md"
    if not report_name.endswith(".md"):
        report_name += ".md"
    out_dir: Path = args.out_dir
    report_path = out_dir / report_name

    if report_path.exists() and not args.force:
        existing = report_path.read_text(encoding="utf-8")
        if run_dir.name not in existing:
            raise SystemExit(
                f"拒绝覆盖：{report_path} 指向另一次运行（本次请求 {run_dir.name}）。"
                f"换新 --report-name/--stamp，或显式 --force。"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / FIG_SUBDIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    blocks_png = fig_dir / f"{stamp_date(stamp)}_capability_{shape_key(run)}_{stamp}_blocks.png"
    final_png = fig_dir / f"{stamp}_final_compare.png"
    final_measured = measure_final(run, run_dir)
    fig_blocks = plot_blocks(run, blocks_png)
    fig_final = plot_final_compare(run, run_dir, final_png, final_measured)

    text = build_report_text(run, run_dir, out_dir, stamp, report_name, fig_blocks, fig_final,
                             final_measured)
    report_path.write_text(text, encoding="utf-8")

    summary = dict(
        report=str(report_path),
        stamp=stamp,
        blocks=len(run.get("blocks") or []),
        final_present=bool(run.get("final")),
        figures={
            "blocks": fig_blocks,
            "final_compare": fig_final,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()


