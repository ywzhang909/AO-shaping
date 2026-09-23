"""Generate the FourierGSNet turbulence-sim matrix report (offline).

Reads a ``scripts/fouriergsnet_sim_train.py`` matrix output directory
(``config.json`` / ``summary.json`` / per-cell ``metrics.csv`` / ``final.json``
/ ``frames/<scenario>/far_%04d.npy`` + ``phase_%04d.npy``) and renders an
illustrated Chinese report to ``docs/fouriergsnet_sim/``:

* ``report.md`` — matrix config header, summary table, per-scenario sections
  (2 animated GIFs + 2 figures + auto interpretation), turbulence impact
  section;
* ``figures/`` — per-scenario metric curves + frame montage, turbulence impact
  grouped bar chart;
* ``gifs/`` — per-scenario phase (mod 2π, twilight) and far-field (inferno)
  evolution animations.

**Fully offline** — reads saved artefacts only, no hardware, no pipeline code.

Usage::

    python scripts/generate_fouriergsnet_sim_report.py
    python scripts/generate_fouriergsnet_sim_report.py --matrix-dir /tmp/fgn_probe512b
    python scripts/generate_fouriergsnet_sim_report.py --matrix-dir data/fouriergsnet_sim/<ts> -o docs/fouriergsnet_sim
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Repo bootstrap: ROOT + src on sys.path, then Agg BEFORE pyplot
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# _frames_to_gif: prefer the repo reference implementation, fall back to a
# verbatim copy (it is a pure module-level function).
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_diff_shaping_report import _frames_to_gif  # noqa: E402

    logger.debug("复用 scripts/generate_diff_shaping_report._frames_to_gif")
except Exception as exc:  # noqa: BLE001 — fallback copy keeps the report self-contained
    logger.warning(
        "无法导入 generate_diff_shaping_report._frames_to_gif ({}), 使用内置副本", exc
    )

    def _frames_to_gif(frames: list[np.ndarray], out_path: Path, cmap: str) -> None:
        """Downscale + quantize 2D arrays into an animated GIF (repo convention)."""
        pil_frames: list[Image.Image] = []
        for arr in frames:
            # Normalise to 0..1
            a = arr.astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            if hi - lo < 1e-12:
                a = np.zeros_like(a)
            else:
                a = (a - lo) / (hi - lo)
            # Apply colormap via matplotlib for a nicer look
            cm = plt.get_cmap(cmap)
            rgba = (cm(a)[:, :, :3] * 255.0).astype(np.uint8)
            img = Image.fromarray(rgba)
            # Downscale to ~128px
            w, h = img.size
            scale = 128.0 / max(w, h)
            if scale < 1.0:
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            pil_frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // 15,
            loop=0,
            optimize=True,
        )

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
DPI = 150
PHASE_CMAP = "twilight"  # cyclic — appropriate for mod 2π phase
FAR_CMAP = "inferno"  # far-field intensity


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: object, nd: int = 4) -> str:
    """Format a metric for markdown: ints/raw sums as ints, ratios as floats."""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return "-"
    if abs(f) >= 10.0:
        return f"{f:.0f}"
    return f"{f:.{nd}f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Matrix discovery (robust to in-progress runs: summary.json may be absent)
# ---------------------------------------------------------------------------
def _resolve_matrix_dir(matrix_dir: str | None) -> Path:
    if matrix_dir:
        p = Path(matrix_dir)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise click.ClickException(f"矩阵目录不存在: {p}")
        return p.resolve()
    base = ROOT / "data" / "fouriergsnet_sim"
    if not base.exists():
        raise click.ClickException(f"默认矩阵根目录不存在: {base}")
    subdirs = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        raise click.ClickException(f"{base} 下无时间戳子目录")
    return subdirs[0].resolve()


def _load_matrix_config(matrix_dir: Path) -> dict:
    cfg: dict = {}
    cfg_path = matrix_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = _load_json(cfg_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("顶层 config.json 解析失败: {}", exc)
    # env noise params live in per-cell config.json (not top-level); fall back
    # to the first available cell config. Also fill shared matrix keys
    # (k_px/steps/device/...) from a cell config for mid-run reports where the
    # top-level config.json is not written yet (train script writes it last).
    for d in sorted(matrix_dir.iterdir()):
        if not d.is_dir() or d.name == "frames" or not (d / "config.json").exists():
            continue
        try:
            cc = _load_json(d / "config.json")
        except Exception:  # noqa: BLE001
            continue
        if "env" not in cfg and "env" in cc:
            cfg["env"] = cc["env"]
        for key in ("k_px", "steps", "device", "replay", "init_gs_iters", "native"):
            if key not in cfg and key in cc:
                cfg[key] = cc[key]
        if "env" in cfg and all(k in cfg for k in ("k_px", "steps", "device")):
            break
    return cfg


def _discover_cells(matrix_dir: Path) -> list[dict]:
    """Cells from summary.json (primary) + any scenario dirs found on disk."""
    cells: list[dict] = []
    summary_path = matrix_dir / "summary.json"
    if summary_path.exists():
        try:
            cells = list(_load_json(summary_path).get("cells", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary.json 解析失败: {}", exc)

    seen = {c.get("scenario") for c in cells}
    for d in sorted(matrix_dir.iterdir()):
        if not d.is_dir() or d.name == "frames" or d.name in seen:
            continue
        final_path = d / "final.json"
        if not final_path.exists():
            continue
        try:
            final = _load_json(final_path)
            cfg = _load_json(d / "config.json") if (d / "config.json").exists() else {}
            parts = d.name.split("__")
            cells.append(
                {
                    "scenario": d.name,
                    "shape": cfg.get("shape", parts[0] if len(parts) >= 1 else d.name),
                    "aberration": cfg.get("aberration", parts[1] if len(parts) >= 2 else ""),
                    "turbulence": cfg.get("turbulence", parts[2] if len(parts) >= 3 else ""),
                    "ok": True,
                    "final_uniformity": final.get("final", {}).get("uniformity"),
                    "final_encircled": final.get("final", {}).get("encircled"),
                    "best_uniformity": final.get("best_uniformity"),
                    "best_encircled": final.get("best_encircled"),
                    "wall_time_s": final.get("wall_time_s"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("扫描 cell {} 失败: {}", d.name, exc)
    return cells


# ---------------------------------------------------------------------------
# Per-scenario figures / GIFs
# ---------------------------------------------------------------------------
def _make_metrics_figure(cell_dir: Path, scenario: str, out_path: Path) -> None:
    """2×2 metric curves from metrics.csv (annotate best uniformity)."""
    csv_path = cell_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少 {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} 为空")
    per_step = df.iloc[:-1] if len(df) > 1 else df  # final row = whole-frame record
    final_row = df.iloc[-1] if len(df) else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # top-left: uniformity vs step
    ax = axes[0, 0]
    ax.plot(df["step"], df["uniformity"], marker="o", ms=3, color="tab:blue")
    best = float(df["uniformity"].max())
    ax.axhline(best, ls="--", color="tab:red", alpha=0.6)
    ax.annotate(
        f"best={best:.4f}",
        xy=(float(df["step"].iloc[-1]), best),
        xytext=(0.6, 0.9),
        textcoords="axes fraction",
        fontsize=9,
        color="tab:red",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("uniformity")
    ax.set_title("均匀度 vs step")
    ax.grid(alpha=0.3)

    # top-right: encircled vs step (per-step 0~1 scale; final row is raw sum)
    ax = axes[0, 1]
    ax.plot(per_step["step"], per_step["encircled"], marker="o", ms=3, color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("encircled (0~1)")
    title = "环围能量 vs step"
    if final_row is not None and not pd.isna(final_row.get("encircled", float("nan"))):
        title += f" (整帧 encircled={float(final_row['encircled']):.0f})"
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # bottom-left: mse / correlation (whole-frame, only in the final row)
    ax = axes[1, 0]
    mse_series = df["mse"].dropna()
    corr_series = df["correlation"].dropna()
    if len(mse_series) >= 2 and len(corr_series) >= 2:
        ax.plot(mse_series.index, mse_series.values, marker="o", ms=3, label="mse")
        ax.set_xlabel("row index")
        ax.set_ylabel("mse", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(corr_series.index, corr_series.values, marker="s", ms=3,
                 color="tab:orange", label="correlation")
        ax2.set_ylabel("correlation", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title("MSE / correlation")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        if final_row is not None and not pd.isna(final_row.get("mse", float("nan"))):
            ax.text(
                0.5, 0.5,
                f"整帧 MSE = {float(final_row['mse']):.3e}\n"
                f"整帧 Pearson 相关 = {float(final_row['correlation']):.4f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12,
            )
            ax.set_title("MSE / correlation (整帧, 仅收尾行)")
        else:
            ax.text(0.5, 0.5, "无整帧 MSE/correlation 数据",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("MSE / correlation")

    # bottom-right: efficiency / inference_ms
    ax = axes[1, 1]
    ax.plot(per_step["step"], per_step["inference_ms"], marker="o", ms=3, color="tab:purple")
    ax.set_xlabel("step")
    ax.set_ylabel("inference_ms")
    title = "推理耗时 vs step"
    if final_row is not None and not pd.isna(final_row.get("efficiency", float("nan"))):
        title += f" (整帧效率={float(final_row['efficiency']):.3f})"
    ax.set_title(title)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{scenario} — 逐步指标")
    fig.tight_layout()
    _savefig(fig, out_path)


def _make_frames_montage(frames_dir: Path, scenario: str, out_path: Path) -> None:
    """2 rows × 3 cols: phase (mod 2π) and far-field at first/mid/last frames."""
    far_files = sorted(frames_dir.glob("far_*.npy"))
    phase_files = sorted(frames_dir.glob("phase_*.npy"))
    if not far_files or not phase_files:
        raise FileNotFoundError(f"{frames_dir} 中无帧文件")
    n = len(far_files)
    idxs = [0, n // 2, n - 1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, i in enumerate(idxs):
        ph = np.load(phase_files[i])
        axes[0, col].imshow(np.mod(ph, 2.0 * np.pi), cmap=PHASE_CMAP, aspect="auto")
        axes[0, col].set_title(f"phase_{i:04d} (mod 2π)")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        far = np.load(far_files[i])
        axes[1, col].imshow(far, cmap=FAR_CMAP, aspect="auto")
        axes[1, col].set_title(f"far_{i:04d}")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
    fig.suptitle(f"{scenario} — 固定点预览 (初值 / 中段 / 末态)")
    fig.tight_layout()
    _savefig(fig, out_path)


def _make_gifs(frames_dir: Path, scenario: str, gif_dir: Path) -> tuple[Path, Path]:
    """Two animated GIFs: shaping-phase evolution + far-field evolution."""
    far_files = sorted(frames_dir.glob("far_*.npy"))
    phase_files = sorted(frames_dir.glob("phase_*.npy"))
    if not far_files or not phase_files:
        raise FileNotFoundError(f"{frames_dir} 中无帧文件")
    # mod 2π ONLY for display — never write back
    phase_frames = [np.mod(np.load(p), 2.0 * np.pi) for p in phase_files]
    far_frames = [np.load(p) for p in far_files]

    phase_gif = gif_dir / f"{scenario}_phase.gif"
    far_gif = gif_dir / f"{scenario}_far.gif"
    _frames_to_gif(phase_frames, phase_gif, PHASE_CMAP)
    _frames_to_gif(far_frames, far_gif, FAR_CMAP)
    return phase_gif, far_gif


def _scenario_interpretation(cell: dict, df: pd.DataFrame) -> str:
    """Auto-generated short interpretation for one scenario."""
    shape = cell.get("shape", "")
    aberration = cell.get("aberration", "")
    turbulence = cell.get("turbulence", "")
    per_step = df.iloc[:-1] if len(df) > 1 else df

    u0 = float(df["uniformity"].iloc[0])
    uf = float(df["uniformity"].iloc[-1])
    ub = float(df["uniformity"].max())
    delta = uf - u0
    pct = (delta / u0 * 100.0) if u0 > 0 else float("nan")

    ee0 = float(per_step["encircled"].iloc[0]) if len(per_step) else float("nan")
    eef = float(per_step["encircled"].iloc[-1]) if len(per_step) else float("nan")

    # uniformity = min/mean inside the target ROI (1 = perfect flat-top);
    # for gaussian targets min/mean ≈ 0 by design, so the metric is
    # meaningless there — fall back to encircled energy for the verdict.
    flat_top = shape != "gaussian" and ub > 0.01

    lines = [
        f"- 目标形状 **{shape}**, 静态像差 **{aberration}**, 湍流 **{turbulence}**。",
        f"- 均匀度: 初值 (adaptive_gs_init 后) **{u0:.3f}** → 最终 **{uf:.3f}** "
        f"(最佳 **{ub:.3f}**, 变化 {delta:+.3f} / "
        + (f"{pct:+.1f}%)。" if u0 > 0 else "—)。"),
        f"- 环围能量 (逐步, 0~1): **{ee0:.3f}** → **{eef:.3f}**。",
    ]
    if not flat_top:
        lines.append(
            "- 均匀度 (min/mean) 为平顶目标指标, 对 gaussian 目标恒≈0; "
            "以环围能量/相关性/效率评估整形效果。"
        )
    elif turbulence == "off":
        lines.append("- 无湍流: 静态像差下闭环精修 GS 初值, 均匀度应波动上升并趋于固定点。")
    else:
        gap = ub - uf
        if gap < 0.02:
            lines.append(
                f"- **{turbulence}** 湍流下闭环跟踪良好: 最终均匀度与最佳值差距仅 {gap:.3f}。"
            )
        else:
            lines.append(
                f"- **{turbulence}** 湍流导致跟踪退化: 最终均匀度低于最佳值 {gap:.3f}。"
            )
    return "\n".join(lines)


def _render_scenario(cell: dict, matrix_dir: Path, fig_dir: Path, gif_dir: Path) -> str:
    """One per-scenario report section (robust: failures degrade gracefully)."""
    scenario = cell["scenario"]
    cell_dir = matrix_dir / scenario
    frames_dir = matrix_dir / "frames" / scenario
    lines = [f"### {scenario}", ""]

    if not cell.get("ok"):
        lines.append(f"> ⚠️ 该 cell 运行失败: {cell.get('error', '未知错误')}")
        lines.append("")
        return "\n".join(lines)

    # metrics figure
    try:
        _make_metrics_figure(cell_dir, scenario, fig_dir / f"{scenario}_metrics.png")
        lines.append(f"![{scenario}_metrics](figures/{scenario}_metrics.png)")
        lines.append("")
    except Exception as exc:  # noqa: BLE001
        logger.warning("{} metrics 图失败: {}", scenario, exc)

    # frames montage (needs frames; missing -> degrade to static-only)
    has_frames = False
    try:
        _make_frames_montage(frames_dir, scenario, fig_dir / f"{scenario}_frames.png")
        has_frames = True
        lines.append(f"![{scenario}_frames](figures/{scenario}_frames.png)")
        lines.append("")
    except Exception as exc:  # noqa: BLE001
        logger.warning("{} 帧蒙太奇失败 (降级为仅静态图): {}", scenario, exc)

    # animated GIFs (only when frames exist)
    if has_frames:
        try:
            phase_gif, far_gif = _make_gifs(frames_dir, scenario, gif_dir)
            logger.info("{} GIF: {} / {}", scenario, phase_gif.name, far_gif.name)
            lines.append(f"![{scenario}_phase](gifs/{scenario}_phase.gif)")
            lines.append("")
            lines.append(f"![{scenario}_far](gifs/{scenario}_far.gif)")
            lines.append("")
        except Exception as exc:  # noqa: BLE001
            logger.warning("{} GIF 生成失败: {}", scenario, exc)

    # interpretation
    try:
        df = pd.read_csv(cell_dir / "metrics.csv")
        lines.append(_scenario_interpretation(cell, df))
        lines.append("")
    except Exception as exc:  # noqa: BLE001
        logger.warning("{} 解读文本失败: {}", scenario, exc)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary table + turbulence impact
# ---------------------------------------------------------------------------
def _build_summary_table(cells: list[dict]) -> str:
    headers = [
        "shape", "aberration", "turbulence",
        "final_uniformity", "best_uniformity",
        "final_encircled", "best_encircled",
        "wall_time_s", "ok",
    ]
    rows: list[list[object]] = []
    for c in sorted(
        cells,
        key=lambda c: (c.get("turbulence", ""), c.get("aberration", ""), c.get("shape", "")),
    ):
        if not c.get("ok"):
            rows.append(
                [c.get("shape", ""), c.get("aberration", ""), c.get("turbulence", ""),
                 "-", "-", "-", "-", "-", "✗"]
            )
            continue
        rows.append(
            [c.get("shape", ""), c.get("aberration", ""), c.get("turbulence", ""),
             _fmt(c.get("final_uniformity")), _fmt(c.get("best_uniformity")),
             _fmt(c.get("final_encircled")), _fmt(c.get("best_encircled")),
             _fmt(c.get("wall_time_s"), nd=1), "✓"]
        )
    return _markdown_table(headers, rows)


def _build_turbulence_impact(cells: list[dict], out_dir: Path) -> str:
    """off vs slow vs fast final uniformity per (shape, aberration)."""
    ok_cells = [c for c in cells if c.get("ok")]
    groups: dict[tuple[str, str], dict[str, float]] = {}
    for c in ok_cells:
        key = (c.get("shape", ""), c.get("aberration", ""))
        groups.setdefault(key, {})[c.get("turbulence", "")] = float(
            c.get("final_uniformity", float("nan"))
        )
    if not groups:
        return ""

    headers = ["shape", "aberration", "off", "slow", "fast"]
    rows: list[list[object]] = []
    for (shape, ab), turb_map in sorted(groups.items()):
        rows.append(
            [shape, ab,
             _fmt(turb_map.get("off")), _fmt(turb_map.get("slow")), _fmt(turb_map.get("fast"))]
        )
    table = _markdown_table(headers, rows)

    # grouped bar chart
    keys = sorted(groups.keys())
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, turb in enumerate(["off", "slow", "fast"]):
        vals = [groups[k].get(turb, float("nan")) for k in keys]
        ax.bar(x + (j - 1) * width, vals, width, label=turb)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{a}" for s, a in keys], fontsize=9)
    ax.set_ylabel("final uniformity")
    ax.set_title("湍流影响: 各 (shape, aberration) 下 off/slow/fast 最终均匀度")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _savefig(fig, out_dir / "figures" / "turbulence_impact.png")

    return table + "\n\n![turbulence_impact](figures/turbulence_impact.png)"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _build_report(
    cfg: dict, matrix_dir: Path, cells: list[dict],
    sections: list[str], turb_section: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = [c for c in cells if c.get("ok")]
    env = cfg.get("env", {})
    calib_noise = env.get("calib_noise", {})

    lines = [
        "# FourierGSNet 湍流仿真矩阵报告",
        "",
        f"**生成时间**: {now}",
        f"**数据来源**: `{matrix_dir}` (矩阵运行时间 {cfg.get('generated_at', '未知')})",
        "",
        "**Fully offline** — 本报告由 `scripts/generate_fouriergsnet_sim_report.py` "
        "离线生成, 仅读取已保存的矩阵产物 (config/summary/metrics/frames), "
        "不打开任何硬件。",
        "",
        "## 1. 矩阵配置",
        "",
    ]
    rows = [
        ("网格 K (k_px)", cfg.get("k_px", "-")),
        ("闭环步数 (steps)", cfg.get("steps", "-")),
        ("基础种子 (base_seed)", cfg.get("base_seed", "-")),
        ("计算设备", cfg.get("device", "-")),
        ("在线 replay", cfg.get("replay", "-")),
        ("自适应 GS 初值迭代", cfg.get("init_gs_iters", "-")),
        ("原生像素节距 (native)", "是 (64×64 方板)" if cfg.get("native") else "否 (1920×1200 各向异性)"),
        ("峰值光子数", env.get("peak_photons", "-")),
        ("读出噪声 (e-)", env.get("read_noise_e", "-")),
        (
            "标定噪声",
            f"ΔK={calib_noise.get('deltaK_fraction', '-')}, "
            f"中心偏移 {calib_noise.get('center_offset_px', '-')}px, "
            f"旋转 {calib_noise.get('rotation_deg', '-')}°",
        ),
        ("场景数", f"{len(cells)} ({len(ok)} 成功)"),
    ]
    lines.append(_markdown_table(["参数", "值"], [[k, str(v)] for k, v in rows]))
    lines.append("")
    lines.append("## 2. 汇总表")
    lines.append("")
    lines.append(_build_summary_table(cells))
    lines.append("")
    lines.append("## 3. 逐场景分析")
    lines.append("")
    for s in sections:
        lines.append(s)
    lines.append("## 4. 湍流影响")
    lines.append("")
    if turb_section:
        lines.append(turb_section)
    else:
        lines.append("> 无足够成功 cell 生成湍流对比。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("本报告由 `scripts/generate_fouriergsnet_sim_report.py` 离线生成。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--matrix-dir",
    default=None,
    help="矩阵输出目录 (默认 data/fouriergsnet_sim/<最新时间戳>)",
)
@click.option(
    "-o", "--output",
    default="docs/fouriergsnet_sim",
    show_default=True,
    help="报告输出目录 (默认 docs/fouriergsnet_sim)",
)
def cli(matrix_dir: str | None, output: str) -> None:
    """FourierGSNet 湍流仿真矩阵离线报告生成器 (无硬件)."""
    np.random.seed(SEED)

    matrix_dir = _resolve_matrix_dir(matrix_dir)
    out_dir = (ROOT / output).resolve()
    fig_dir = out_dir / "figures"
    gif_dir = out_dir / "gifs"
    for d in (out_dir, fig_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = _load_matrix_config(matrix_dir)
    cells = _discover_cells(matrix_dir)
    if not cells:
        raise click.ClickException(f"矩阵目录 {matrix_dir} 无 cell 数据")
    ok_cells = [c for c in cells if c.get("ok")]
    logger.info("发现 {} 个 cell ({} 成功) -> {}", len(cells), len(ok_cells), matrix_dir)

    sections: list[str] = []
    for cell in sorted(
        cells,
        key=lambda c: (c.get("turbulence", ""), c.get("aberration", ""), c.get("shape", "")),
    ):
        scenario = cell["scenario"]
        try:
            sections.append(_render_scenario(cell, matrix_dir, fig_dir, gif_dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cell {} 报告段失败: {}", scenario, exc)
            sections.append(f"### {scenario}\n\n> ⚠️ 该 cell 报告段生成失败: {exc}\n")

    try:
        turb_section = _build_turbulence_impact(ok_cells, out_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("湍流影响段失败: {}", exc)
        turb_section = ""

    report = _build_report(cfg, matrix_dir, cells, sections, turb_section)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("报告已写入 {}", report_path)


if __name__ == "__main__":
    cli()