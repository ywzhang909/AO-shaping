"""Generate the illustrated DM response-matrix report (creation / analysis / detection).

Reads the saved artefacts and writes an illustrated markdown report to
``docs/dm_response_matrix_report/`` (report.md + figures/):

  **Creation**    acquisition metadata of the DM push-pull calibration (mode,
                  voltage, cycles, averages, device config).
  **Analysis**    response-matrix heatmap, variance heatmap, per-actuator
                  response-magnitude bar chart, per-subaperture slope-sensitivity
                  spatial map, singular-value spectrum / condition number.
  **Detection**   weak/dead actuator candidates, high-variance actuators.

This script is **offline** — it never touches hardware. Sources (latest found by
default, override with flags):

  - ``data/dm_response_matrix*.h5``   saved DM response matrix

Usage:
    $env:PYTHONPATH = "src"
    python scripts/generate_dm_response_matrix_report.py
    python scripts/generate_dm_response_matrix_report.py --h5 <path> -o docs/dm_response_matrix_report
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Matplotlib must be configured to Agg BEFORE importing pyplot (repo-wide rule)
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from loguru import logger  # noqa: E402

try:
    from ao_shaping.optimizer.wf.dm_response_matrix import (  # noqa: E402
        load_dm_response_matrix,
    )
except ImportError:
    load_dm_response_matrix = None  # type: ignore[assignment]

import h5py  # noqa: E402

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT = ROOT / "docs" / "dm_response_matrix_report"


# ─────────────────────────── h5 fallback loader ───────────────────────────

def _attr(meta: h5py.Group, name: str, default: Any = None) -> Any:
    """Read an h5 attribute, returning ``default`` when absent."""
    if name not in meta.attrs:
        return default
    return meta.attrs[name]


def _load_h5_fallback(path: Path) -> dict:
    """Direct h5py read when the package import is unavailable."""
    result: dict = {}
    with h5py.File(path, "r") as f:
        result["matrix"] = cast(h5py.Dataset, f["matrix"])[:]
        result["variance_matrix"] = cast(h5py.Dataset, f["variance_matrix"])[:]
        result["subaperture_mask"] = (
            cast(h5py.Dataset, f["subaperture_mask"])[:]
            if "subaperture_mask" in f
            else None
        )
        result["pinv_matrix"] = (
            cast(h5py.Dataset, f["pinv_matrix"])[:]
            if "pinv_matrix" in f
            else None
        )
        result["lstsq_matrix"] = (
            cast(h5py.Dataset, f["lstsq_matrix"])[:]
            if "lstsq_matrix" in f
            else None
        )
        meta = cast(h5py.Group, f["metadata"])
        result["n_actuators"] = int(_attr(meta, "n_actuators"))
        result["disturb_voltage"] = float(_attr(meta, "disturb_voltage"))
        result["n_averages"] = int(_attr(meta, "n_averages"))
        result["n_cycles"] = int(_attr(meta, "n_cycles"))
        result["wait_time"] = float(_attr(meta, "wait_time"))
        result["timestamp"] = str(_attr(meta, "timestamp"))
        result["mean_variance"] = float(_attr(meta, "mean_variance"))
        result["max_variance"] = float(_attr(meta, "max_variance"))
        cn = _attr(meta, "condition_number", -1)
        result["condition_number"] = float(cn) if float(cn) > 0 else None
        result["valid_actuator_indices"] = None
        if "valid_actuator_indices" in meta.attrs:
            try:
                result["valid_actuator_indices"] = json.loads(
                    _attr(meta, "valid_actuator_indices")
                )
            except (json.JSONDecodeError, TypeError):
                pass
        result["device_config"] = None
        if "device_config" in meta.attrs:
            try:
                result["device_config"] = json.loads(
                    _attr(meta, "device_config")
                )
            except (json.JSONDecodeError, TypeError):
                pass
        # Legacy attrs: may not exist
        result["calibration_mode"] = str(
            _attr(meta, "calibration_mode", "sequential")
        )
        ho = _attr(meta, "hadamard_order", None)
        result["hadamard_order"] = int(ho) if ho is not None else None
    return result


def _load_result(path: Path) -> dict:
    """Load the DM response matrix, preferring the package loader."""
    if load_dm_response_matrix is not None:
        try:
            r = load_dm_response_matrix(path)
            # Normalize to a flat dict for uniform downstream access
            return {
                "matrix": r.matrix,
                "variance_matrix": r.variance_matrix,
                "subaperture_mask": r.subaperture_mask,
                "pinv_matrix": r.pinv_matrix,
                "lstsq_matrix": r.lstsq_matrix,
                "n_actuators": r.n_actuators,
                "disturb_voltage": r.disturb_voltage,
                "n_averages": r.n_averages,
                "n_cycles": r.n_cycles,
                "wait_time": r.wait_time,
                "timestamp": r.timestamp,
                "mean_variance": r.mean_variance,
                "max_variance": r.max_variance,
                "condition_number": r.condition_number,
                "valid_actuator_indices": r.valid_actuator_indices,
                "device_config": r.device_config,
                "calibration_mode": r.calibration_mode,
                "hadamard_order": r.hadamard_order,
            }
        except Exception as exc:
            logger.warning("Package loader failed ({}), using h5py fallback", exc)
    return _load_h5_fallback(path)


# ─────────────────────────── figures ───────────────────────────

def fig_matrix_heatmap(matrix: np.ndarray, out_png: Path) -> None:
    """Response matrix heatmap (slope channels x actuators)."""
    n_slopes, n_acts = matrix.shape
    fig, ax = plt.subplots(figsize=(12, 8))
    vmax = float(np.percentile(np.abs(matrix), 99.5)) or 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7, label="response  (slope / V)")
    # Label actuators on x-axis (sample if too many)
    step = max(1, n_acts // 20)
    ax.set_xticks(range(0, n_acts, step))
    ax.set_xticklabels([str(i) for i in range(0, n_acts, step)], fontsize=7)
    ax.set_yticks(range(0, n_slopes, max(1, n_slopes // 20)))
    ax.set_yticklabels(
        [str(i + 1) for i in range(0, n_slopes, max(1, n_slopes // 20))],
        fontsize=7,
    )
    ax.set_xlabel("DM actuator index")
    ax.set_ylabel("WFS slope channel")
    ax.set_title(f"DM response matrix  matrix[slopes={n_slopes}, actuators={n_acts}]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_variance_heatmap(variance: np.ndarray, out_png: Path) -> None:
    """Variance matrix (log scale)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    v = np.log10(np.maximum(variance, 1e-15))
    im = ax.imshow(v, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.7, label="log10 variance")
    n_slopes, n_acts = variance.shape
    step = max(1, n_acts // 20)
    ax.set_xticks(range(0, n_acts, step))
    ax.set_xticklabels([str(i) for i in range(0, n_acts, step)], fontsize=7)
    ax.set_yticks(range(0, n_slopes, max(1, n_slopes // 20)))
    ax.set_yticklabels(
        [str(i + 1) for i in range(0, n_slopes, max(1, n_slopes // 20))],
        fontsize=7,
    )
    ax.set_xlabel("DM actuator index")
    ax.set_ylabel("WFS slope channel")
    ax.set_title("Push-pull repeat variance (log10)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_actuator_magnitude(matrix: np.ndarray, valid_indices: list[int] | None,
                           out_png: Path) -> np.ndarray:
    """Per-actuator response magnitude (Frobenius column norm)."""
    col_norms = np.linalg.norm(matrix, axis=0)
    n_acts = len(col_norms)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = valid_indices if valid_indices else list(range(n_acts))
    x = np.arange(n_acts)
    ax.bar(x, col_norms, color="tab:blue", alpha=0.8)
    # Median line
    med = float(np.median(col_norms))
    ax.axhline(med, color="tab:red", ls="--", lw=1.2, label=f"median = {med:.4f}")
    # Dead threshold at 5% of median
    dead_thresh = 0.05 * med
    ax.axhline(dead_thresh, color="orange", ls=":", lw=1.2,
               label=f"dead threshold (5% median) = {dead_thresh:.4f}")
    dead_mask = col_norms < dead_thresh
    if np.any(dead_mask):
        ax.scatter(x[dead_mask], col_norms[dead_mask], color="red", zorder=5,
                   s=40, label=f"weak ({dead_mask.sum()} actuators)")
    ax.set_xlabel("DM actuator index")
    ax.set_ylabel("||column||2  (slope / V)")
    ax.set_title("Per-actuator response magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return col_norms


def fig_subaperture_spatial(matrix: np.ndarray, mask: np.ndarray | None,
                            out_png: Path) -> bool:
    """Per-subaperture slope-sensitivity spatial map (reshape if subaperture
    grid dimensions are derivable from matrix shape and mask).

    Returns True if the figure was generated.
    """
    n_slopes = matrix.shape[0]
    # Try to derive the spatial grid from subaperture_mask shape.
    # WFS slopes come in pairs (dx, dy) per subaperture.
    # If mask is (nx, ny), then n_subaps ≈ mask.sum(), and
    # n_slopes = 2 * n_subaps.
    if mask is not None and mask.size > 0:
        mx, my = mask.shape
        n_subaps = int(mask.sum())
        if n_subaps > 0 and n_slopes == 2 * n_subaps:
            # Pair up consecutive slopes: for each subaperture, take the mean
            # magnitude of its (dx, dy) slope pair across all actuators.
            spatial = np.zeros((mx, my), dtype=float)
            idx = 0
            for j in range(my):
                for i in range(mx):
                    if mask[i, j] and idx + 1 < n_slopes:
                        spatial[i, j] = np.sqrt(matrix[idx] ** 2 + matrix[idx + 1] ** 2).mean()
                        idx += 2
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(spatial, cmap="hot", aspect="equal", origin="lower")
            fig.colorbar(im, ax=ax, shrink=0.8, label="mean slope magnitude per subaperture")
            ax.set_xlabel("subaperture x")
            ax.set_ylabel("subaperture y")
            ax.set_title(f"Slope sensitivity spatial map  ({mx}×{my}, "
                         f"{n_subaps} valid subapertures)")
            fig.tight_layout()
            fig.savefig(out_png, dpi=110)
            plt.close(fig)
            return True
        else:
            # Mask exists but shape doesn't match paired slopes
            fig, ax = plt.subplots(figsize=(8, 6))
            act_mag = np.sqrt(np.sum(matrix ** 2, axis=1))
            ax.plot(act_mag, "o-", ms=2)
            ax.set_xlabel("slope channel index")
            ax.set_ylabel("‖column‖₂")
            ax.set_title(f"Slope magnitude per channel  (mask {mask.shape}, "
                         f"{n_subaps} subaps, n_slopes={n_slopes} — "
                         f"paired reshape not possible)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_png, dpi=110)
            plt.close(fig)
            return True
    else:
        # No mask: just plot per-slope-channel magnitude
        fig, ax = plt.subplots(figsize=(12, 5))
        act_mag = np.sqrt(np.sum(matrix ** 2, axis=1))
        ax.plot(act_mag, "o-", ms=2)
        ax.set_xlabel("slope channel index")
        ax.set_ylabel("‖row‖₂")
        ax.set_title(f"Slope magnitude per channel  (n_slopes={n_slopes}, "
                     f"no subaperture mask)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        return True


def fig_singular_values(matrix: np.ndarray, out_png: Path) -> tuple[np.ndarray, float]:
    """Singular-value spectrum + condition number. Returns (s, cond)."""
    s = np.linalg.svd(matrix, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title(f"Singular values — cond = {cond:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return s, cond


# ─────────────────────────── markdown ───────────────────────────

def _render_device_config(dc: dict | None) -> list[str]:
    """Render device_config dict as markdown table rows."""
    if not dc:
        return []
    lines: list[str] = []
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    for k in sorted(dc.keys()):
        v = dc[k]
        if isinstance(v, dict):
            lines.append(f"| {k} | (dict, {len(v)} keys) |")
        elif isinstance(v, list) and len(v) > 8:
            lines.append(f"| {k} | list[{len(v)}] (first: {v[:3]}…) |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def write_markdown(out: Path, ctx: dict) -> str:
    """Build the report markdown body and return (does not write file)."""
    r = ctx
    matrix = r["matrix"]
    variance = r["variance_matrix"]
    n_slopes, n_acts = matrix.shape
    mask = r["subaperture_mask"]
    col_norms = ctx["col_norms"]
    sv_result = ctx["sv_result"]
    cond = ctx["cond"]

    md: list[str] = []
    md.append("# DM 响应矩阵报告 (创建 / 分析 / 检测)\n")
    md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**数据源**: `{ctx['h5'].name}`\n")

    # ── §1 创建 ──
    md.append("## 1. 创建 (Creation)\n")
    md.append("响应矩阵由 `runners/dm_matrix_runner` 以**推拉法**标定: 对每个 DM "
              "执行器施加 ±V 扰动电压, 测 WFS 斜率读数增量, "
              "`matrix[slope_ch, actuator] = Δ(WFS_slopes)/Δ(voltage)`。\n")
    md.append("| 项 | 值 |")
    md.append("|---|---|")
    md.append(f"| 标定模式 | **{r['calibration_mode']}** |")
    if r["hadamard_order"] is not None:
        md.append(f"| Hadamard 阶数 | {r['hadamard_order']} |")
    md.append(f"| DM 总执行器数 | {r['n_actuators']} |")
    valid = r.get("valid_actuator_indices")
    md.append(f"| 有效执行器数 | {n_acts}" +
              (f" (indices: {valid[:8]}{'…' if valid and len(valid) > 8 else ''})"
               if valid else " (all)") + " |")
    md.append(f"| 扰动电压 | {r['disturb_voltage']:.4f} V |")
    md.append(f"| 帧平均 / 循环次数 | {r['n_averages']} / {r['n_cycles']} |")
    md.append(f"| 等待时间 | {r['wait_time']:.3f} s |")
    md.append(f"| 时间戳 | {r['timestamp']} |")
    md.append(f"| 矩阵形状 | {matrix.shape} (slopes × actuators) |")
    md.append(f"| 平均方差 | {r['mean_variance']:.3e} |")
    md.append(f"| 最大方差 | {r['max_variance']:.3e} |")
    md.append(f"| 条件数 | {cond:.2f}" if cond is not None else "| 条件数 | — (pinv not stored) |")
    md.append("")

    dc = r.get("device_config")
    if dc:
        md.append("### 1.1 设备参数 (device_config)\n")
        md.extend(_render_device_config(dc))
    md.append("")

    # ── §2 分析 ──
    md.append("## 2. 分析 (Analysis)\n")

    md.append("### 2.1 响应矩阵热图\n")
    md.append("![matrix heatmap](figures/01_matrix_heatmap.png)\n")
    md.append(f"矩阵 shape = {matrix.shape}; 行 = WFS 斜率通道, "
              f"列 = DM 执行器。颜色表示单位电压扰动下的斜率响应。\n")

    md.append("### 2.2 重复性方差\n")
    md.append("![variance heatmap](figures/02_variance_heatmap.png)\n")
    md.append(f"平均方差 **{r['mean_variance']:.3e}**, 最大 **{r['max_variance']:.3e}**。"
              f"数量级 1e-6 ~ 1e-5 表示推拉重复性良好。\n")

    md.append("### 2.3 执行器响应幅度\n")
    md.append("![actuator magnitude](figures/03_actuator_magnitude.png)\n")
    dead_thresh = 0.05 * float(np.median(col_norms))
    n_dead = int(np.sum(col_norms < dead_thresh))
    md.append(f"每列 Frobenius 范数 (所有斜率通道的响应总能量)。"
              f"中位数 = {float(np.median(col_norms)):.4f}。")
    if n_dead > 0:
        md.append(f"**{n_dead} 个执行器**低于 5% 中位数阈值 ({dead_thresh:.4f}), "
                  f"疑似弱/无响应。")
    else:
        md.append("无低于 5% 中位数阈值的执行器。\n")
    md.append("")

    md.append("### 2.4 子孔径斜率灵敏度空间图\n")
    md.append("![subaperture spatial](figures/04_subaperture_spatial.png)\n")
    if mask is not None:
        mx, my = mask.shape
        n_subaps = int(mask.sum())
        if n_slopes == 2 * n_subaps:
            md.append(f"子孔径掩膜 {mx}×{my}, 有效子孔径 {n_subaps}。"
                      f"每子孔径的 (dx,dy) 斜率总响应按网格空间排列。\n")
        else:
            md.append(f"子孔径掩膜 {mask.shape}, {n_subaps} 有效子孔径, "
                      f"但斜率通道数 ({n_slopes}) ≠ 2×子孔径数, "
                      f"无法精确配对到空间网格 — 显示逐通道幅值。\n")
    else:
        md.append(f"无子孔径掩膜, 显示逐斜率通道幅值 (n_slopes={n_slopes})。\n")

    md.append("### 2.5 奇异值谱 / 条件数\n")
    md.append("![singular values](figures/05_singular_values.png)\n")
    if cond is not None:
        md.append(f"条件数 **{cond:.2f}**。"
                  + ("良态, 伪逆稳定。" if cond < 100 else
                     ("病态 — 伪逆可能不稳定, 建议检查弱响应执行器。"
                      if cond < 1e4 else "严重病态 — 逆矩阵不可靠。")))
    else:
        md.append("条件数无法计算 (pinv_matrix 未存储)。\n")
    md.append("")

    # ── §3 检测 ──
    md.append("## 3. 检测 (Detection)\n")

    # 3.1 Weak / dead actuators
    md.append("### 3.1 弱/无响应执行器\n")
    median_norm = float(np.median(col_norms))
    if n_dead > 0:
        dead_idx = np.where(col_norms < dead_thresh)[0].tolist()
        md.append(f"检测到 **{n_dead} 个**疑似弱/无响应执行器 "
                  f"(列范数 < 5% 中位数 = {dead_thresh:.4f}):\n")
        md.append("| 执行器索引 | 列范数 | 占中位数 |")
        md.append("|---|---|---|")
        for idx in dead_idx:
            val = float(col_norms[idx])
            pct = 100 * val / median_norm if median_norm > 0 else 0
            md.append(f"| {idx} | {val:.4f} | {pct:.1f}% |")
        md.append("")
        md.append("> **注意**: 弱执行器可能由以下原因造成: (a) 物理执行器故障或断路, "
                  "(b) 光学遮挡导致子孔径无法感知该执行器的响应, "
                  "(c) 标定期间的瞬态噪声。建议结合多次标定结果交叉验证。\n")
    else:
        md.append("> 未检测到低于 5% 中位数阈值的弱/无响应执行器。\n")

    # 3.2 High-variance actuators
    md.append("### 3.2 高方差执行器\n")
    # Per-actuator variance = mean of column in variance_matrix
    act_var = np.mean(variance, axis=0)  # (n_acts,)
    med_var = float(np.median(act_var))
    high_var_thresh = 10 * med_var  # >10x median variance
    n_high_var = int(np.sum(act_var > high_var_thresh))
    if n_high_var > 0:
        high_idx = np.where(act_var > high_var_thresh)[0].tolist()
        md.append(f"检测到 **{n_high_var} 个**高方差执行器 "
                  f"(列均方差 > 10× 中位数 = {high_var_thresh:.3e}):\n")
        md.append("| 执行器索引 | 列均方差 | 倍数 |")
        md.append("|---|---|---|")
        for idx in high_idx:
            val = float(act_var[idx])
            ratio = val / med_var if med_var > 0 else 0
            md.append(f"| {idx} | {val:.3e} | {ratio:.1f}× |")
        md.append("")
        md.append("> 高方差可能表明: (a) 该执行器的机械响应不稳定 (迟滞), "
                  "(b) 光路干扰, (c) WFS 测量噪声局部集中。"
                  "多次标定取平均可降低影响。\n")
    else:
        md.append("> 未检测到超过 10× 中位数方差的高方差执行器。\n")

    # ── §4 结论 ──
    md.append("## 4. 结论与解读\n")

    md.append("### 4.1 矩阵质量\n")
    md.append(f"- **形状** `{matrix.shape}` = (WFS 斜率通道 {n_slopes}) × "
              f"(DM 执行器 {n_acts})。")
    if r.get("valid_actuator_indices"):
        md.append(f"  有效执行器索引: {r['valid_actuator_indices'][:10]}"
                  f"{'…' if len(r['valid_actuator_indices']) > 10 else ''}。")
    if cond is not None:
        md.append(f"- **条件数 `{cond:.2f}`** — "
                  + ("良态, 伪逆稳定。" if cond < 100
                     else "病态, 伪逆可能不稳定, 建议排除弱执行器后重算。"))
    md.append(f"- **平均重复方差 `{r['mean_variance']:.3e}`** — 推拉循环间的读数散布; "
              f"最大 `{r['max_variance']:.3e}`。"
              + ("数量级 1e-6 ~ 1e-5 表示重复性良好。"
                 if r['mean_variance'] < 1e-4
                 else "方差偏大, 建议增加 n_averages 或 n_cycles。"))
    md.append("")

    md.append("### 4.2 弱/无响应执行器\n")
    if n_dead > 0:
        md.append(f"- **{n_dead}/{n_acts}** 个执行器列范数低于 5% 中位数, "
                  "建议在闭环控制中排除或降低其权重。")
        md.append("- 建议: 结合多次独立标定确认是否为永久性故障。\n")
    else:
        md.append("- 所有执行器均有可测量的响应, 质量良好。\n")

    md.append("### 4.3 标定模式\n")
    if r["calibration_mode"] == "hadamard":
        md.append(f"- **Hadamard 模式** (阶数 {r.get('hadamard_order', '?')}): "
                  "每个测量帧同时扰动多个执行器 (Walsh-Hadamard 模式), "
                  "信噪比高于逐路标定, 但对执行器间非线性/串扰敏感。")
    else:
        md.append("- **逐路模式** (sequential): 每次仅扰动一个执行器, "
                  "抗串扰但信噪比较低。")
    md.append("")

    # ── §5 产物 ──
    md.append("## 5. 产物\n")
    md.append("| 产物 | 路径 |")
    md.append("|---|---|")
    md.append(f"| 响应矩阵 h5 | `{ctx['h5']}` |")
    md.append("| 标定工具 | `src/ao_shaping/runners/dm_matrix_runner.py` |")
    md.append("| 核心函数 | `src/ao_shaping/optimizer/wf/dm_response_matrix.py` |")
    md.append("")

    body = "\n".join(md)
    return body


# ─────────────────────────── main ───────────────────────────

def _latest_matrix() -> Path | None:
    """Find latest DM response matrix .h5 file."""
    cands = [
        Path(p) for p in glob.glob(
            str(ROOT / "data" / "dm_response_matrix*.h5"))
    ]
    cands = [p for p in cands if p.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=None,
                    help="DM response matrix .h5 (default: latest)")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT),
                    help="Output directory (default: docs/dm_response_matrix_report)")
    ap.add_argument("--title", default=None, help="Report title override")
    args = ap.parse_args()

    h5 = Path(args.h5) if args.h5 else _latest_matrix()
    if h5 is None or not h5.exists():
        logger.error("未找到 DM 响应矩阵 h5")
        return 1

    out = Path(args.output)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("[DM 响应矩阵报告] 创建 / 分析 / 检测")
    print("=" * 72)
    print(f"  h5    : {h5}")
    print(f"  output: {out}")

    r = _load_result(h5)
    matrix = r["matrix"]
    print(f"[OK] matrix {matrix.shape}, n_actuators={r['n_actuators']}")

    # Generate figures
    fig_matrix_heatmap(matrix, figs / "01_matrix_heatmap.png")
    fig_variance_heatmap(r["variance_matrix"], figs / "02_variance_heatmap.png")
    col_norms = fig_actuator_magnitude(
        matrix, r.get("valid_actuator_indices"),
        figs / "03_actuator_magnitude.png",
    )
    fig_subaperture_spatial(
        matrix, r["subaperture_mask"],
        figs / "04_subaperture_spatial.png",
    )
    sv, cond = fig_singular_values(matrix, figs / "05_singular_values.png")
    print(f"[OK] SVD done, cond={cond:.3f}, sv[0]={sv[0]:.4f}")

    # Dead/weak actuator check
    median_norm = float(np.median(col_norms))
    dead_thresh = 0.05 * median_norm
    n_dead = int(np.sum(col_norms < dead_thresh))
    if n_dead > 0:
        print(f"[WARN] {n_dead} weak/dead actuator(s) detected "
              f"(threshold={dead_thresh:.4f})")
    else:
        print(f"[OK] no weak actuators (median={median_norm:.4f})")

    # Build context and write markdown
    ctx = {**r, "h5": h5, "col_norms": col_norms, "sv_result": sv, "cond": cond}
    body = write_markdown(out, ctx)
    (out / "report.md").write_text(body, encoding="utf-8")
    print(f"\n[OK] 报告: {out / 'report.md'}")
    print(f"[OK] 图: {figs}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
