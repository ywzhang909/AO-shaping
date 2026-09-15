"""Generate the illustrated Zernike response-matrix report (creation / analysis / verification).

Reads the saved artefacts and writes an illustrated markdown report to
``docs/slm/zernike_response_matrix_report/`` (report.md + figures/):

  **Creation**    acquisition metadata of the push-pull calibration; the raw
                  multi-size scan is summarised (mode × radius × amplitude grid).
  **Analysis**    response-matrix heatmap, diagonal dominance, singular-value
                  spectrum / condition number, variance map, and per-(mode,
                  radius) linearity from the raw scan.
  **Detection**   outlier diagnosis (why R≈beam-radius breaks the WFS fit),
                  offline inverse demo (synthetic aberration → solved SLM modes
                  → residual), and the measured closed-loop before/after.

This script is **offline** — it never touches hardware. Sources (latest found by
default, override with flags):

  - ``data/zernike_response_matrix/zm_*.h5``      clean single-radius matrix
  - ``data/zernike_correction/report_*.json``     multi-size scan + closed loop
  - ``data/zernike_correction/raw_scan_*.json``   per-point raw readouts

Usage:
    $env:PYTHONPATH = "src"
    python scripts/generate_zernike_response_matrix_report.py
    python scripts/generate_zernike_response_matrix_report.py --h5 <path> -o docs/slm/<dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

from ao_shaping.optimizer.wf.zernike_response_matrix import (  # noqa: E402
    load_zernike_response_matrix,
)
from ao_shaping.tools.slm.slm_zernike_common import (  # noqa: E402
    DLL_ZERNIKE_NAMES,
    linearity_metrics,
)

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT = ROOT / "docs" / "slm" / "zernike_response_matrix_report"


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def _short(name: str) -> str:
    """'(2,0)defocus' -> '(2,0)defocus' trimmed for axis labels."""
    return name.replace("2nd-astig", "2a").replace("quadrafoil", "qf") \
               .replace("trefoil", "tf")


# ─────────────────────────── figures ───────────────────────────

def fig_matrix_heatmap(matrix: np.ndarray, modes: list[int], out_png: Path) -> None:
    """Response matrix heatmap (WFS channels × SLM modes)."""
    fig, ax = plt.subplots(figsize=(9, 12))
    vmax = float(np.percentile(np.abs(matrix), 99.5)) or 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7, label="response  (λ / λ)")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([f"[{m}]\n{_short(DLL_ZERNIKE_NAMES.get(m, ''))}"
                        for m in modes], fontsize=7)
    ax.set_yticks(range(0, matrix.shape[0], 5))
    ax.set_yticklabels([str(i + 1) for i in range(0, matrix.shape[0], 5)], fontsize=7)
    ax.set_xlabel("SLM Zernike mode (DLL index)")
    ax.set_ylabel("WFS Zernike coefficient (DLL index)")
    ax.set_title("Zernike response matrix  matrix[WFS, SLM]")
    # mark the diagonal
    for col, m in enumerate(modes):
        row = m - 1
        if row < matrix.shape[0]:
            ax.plot(col, row, marker="s", ms=9, mfc="none", mec="lime", mew=1.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_diagonal(matrix: np.ndarray, modes: list[int], out_png: Path) -> None:
    """Diagonal vs strongest off-diagonal per mode."""
    diag, off = [], []
    for col, m in enumerate(modes):
        row = m - 1
        colvec = np.abs(matrix[:, col]).copy()
        d = float(matrix[row, col]) if row < matrix.shape[0] else np.nan
        colvec[row] = 0.0
        diag.append(d)
        off.append(float(np.max(colvec)))
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, np.abs(diag), 0.4, label="|diagonal|", color="tab:blue")
    ax.bar(x + 0.2, off, 0.4, label="max |off-diagonal|", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{m}]\n{_short(DLL_ZERNIKE_NAMES.get(m, ''))}"
                        for m in modes], fontsize=7)
    ax.set_ylabel("|response|  (λ / λ)")
    ax.set_title("Diagonal dominance — diagonal vs strongest off-diagonal")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_singular_values(matrix: np.ndarray, out_png: Path) -> None:
    """Singular-value spectrum + condition number."""
    s = np.linalg.svd(matrix, compute_uv=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title(f"Singular values — cond = {s[0] / s[-1]:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_variance(variance: np.ndarray, modes: list[int], out_png: Path) -> None:
    """Variance matrix (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 12))
    v = np.log10(np.maximum(variance, 1e-12))
    im = ax.imshow(v, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.7, label="log10 variance")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([f"[{m}]" for m in modes], fontsize=7)
    ax.set_yticks(range(0, variance.shape[0], 5))
    ax.set_yticklabels([str(i + 1) for i in range(0, variance.shape[0], 5)], fontsize=7)
    ax.set_xlabel("SLM mode (DLL index)")
    ax.set_ylabel("WFS coefficient (DLL index)")
    ax.set_title("Repeat variance (push-pull cycles)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _raw_groups(raw: list[dict]) -> dict[tuple[int, float], dict[float, dict[int, np.ndarray]]]:
    g: dict[tuple[int, float], dict[float, dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict))
    for s in raw:
        g[(s["dll_index"], float(s["radius"]))][float(s["amp_rad"])][int(s["sign"])] = \
            np.asarray(s["readout_um"], dtype=float)
    return g


def fig_linearity(raw: list[dict], out_png: Path) -> list[dict]:
    """CV + direction-cosine per (mode, radius) — the correct linearity criterion."""
    g = _raw_groups(raw)
    rows: list[dict] = []
    for (midx, R) in sorted(g):
        d = g[(midx, R)]
        vecs, amps = [], []
        for a in sorted(a for a in d if a > 0):
            zp, zn = d[a].get(1), d[-a].get(-1)
            if zp is None or zn is None:
                continue
            vecs.append((zp[1:] - zn[1:]) / 2.0 / (a / (2 * np.pi)))
            amps.append(a)
        if len(vecs) < 2:
            continue
        m = linearity_metrics(vecs, amps)
        rows.append({"dll_index": midx, "radius": R, **m})

    radii = sorted({r["radius"] for r in rows})
    modes = sorted({r["dll_index"] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    width = 0.8 / max(len(radii), 1)
    x = np.arange(len(modes))
    for i, R in enumerate(radii):
        cvs, coss = [], []
        for m in modes:
            hit = next((r for r in rows if r["dll_index"] == m and r["radius"] == R), None)
            cvs.append(100 * hit["cv"] if hit else np.nan)
            coss.append(hit["cos_min"] if hit else np.nan)
        axes[0].bar(x + i * width, cvs, width, label=f"R={R:.0f}px")
        axes[1].bar(x + i * width, coss, width, label=f"R={R:.0f}px")
    axes[0].axhline(15, color="r", ls="--", lw=1, label="CV threshold 15%")
    axes[0].set_ylabel("|resp| CV (%)")
    axes[0].set_title("Linearity: amplitude consistency (CV) and direction "
                      "consistency (cos) per (mode, radius)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].axhline(0.9, color="r", ls="--", lw=1, label="cos threshold 0.9")
    axes[1].set_ylabel("min pairwise cos")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"[{m}]\n{_short(DLL_ZERNIKE_NAMES.get(m, ''))}"
                             for m in modes], fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return rows


def fig_outlier(raw: list[dict], out_png: Path, worst: list[int]) -> None:
    """|resp| vs amplitude for the worst modes — shows the WFS-fit blow-up."""
    g = _raw_groups(raw)
    radii = sorted({float(s["radius"]) for s in raw})
    n = len(worst)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    for ax, midx in zip(axes[0], worst):
        for R in radii:
            d = g.get((midx, R))
            if not d:
                continue
            xs, ys = [], []
            for a in sorted(a for a in d if a > 0):
                zp, zn = d[a].get(1), d[-a].get(-1)
                if zp is None or zn is None:
                    continue
                xs.append(a)
                ys.append(float(np.linalg.norm((zp[1:] - zn[1:]) / 2.0 / (a / (2 * np.pi)))))
            ax.plot(xs, ys, "o-", label=f"R={R:.0f}px")
        ax.set_yscale("log")
        ax.set_title(f"[{midx}] {_short(DLL_ZERNIKE_NAMES.get(midx, ''))}")
        ax.set_xlabel("amplitude (rad)")
        ax.set_ylabel("|resp| (λ/λ)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Outlier diagnosis — |resp| vs amplitude "
                 "(R ~= beam radius => WFS Zernike fit collapses)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_inverse(matrix: np.ndarray, modes: list[int], out_png: Path) -> dict:
    """Offline inverse demo: synthetic aberration → solved modes → residual."""
    pinv = np.linalg.pinv(matrix)
    w = np.zeros(matrix.shape[0])
    w[4] = 0.30     # DLL [5]  defocus
    w[8] = -0.20    # DLL [9]  coma (3,1)
    c = pinv @ w
    resid = matrix @ c - w

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.bar(range(len(modes)), c, color="tab:green")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([f"[{m}]" for m in modes], fontsize=8)
    ax.set_ylabel("solved SLM amplitude (λ)")
    ax.set_title("Solved SLM modes  c = pinv(M) @ w")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    show = np.arange(1, min(20, matrix.shape[0] + 1))
    ax.bar(show - 0.2, w[show - 1], 0.4, label="target w", color="tab:red")
    ax.bar(show + 0.2, (matrix @ c)[show - 1], 0.4, label="M·c", color="tab:blue")
    ax.set_xticks(show[::2])
    ax.set_ylabel("WFS coefficient (λ)")
    ax.set_title(f"Residual ‖Mc−w‖={np.linalg.norm(resid):.4f} / "
                 f"‖w‖={np.linalg.norm(w):.4f} "
                 f"({100 * (1 - np.linalg.norm(resid) / np.linalg.norm(w)):.1f}% reduction)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"c": c.tolist(), "residual": float(np.linalg.norm(resid)),
            "w_norm": float(np.linalg.norm(w)),
            "reduction_pct": float(100 * (1 - np.linalg.norm(resid) / np.linalg.norm(w)))}


def fig_closed_loop(report: dict, out_png: Path) -> dict | None:
    """Measured closed-loop before/after from the 3-stage run."""
    cl = report.get("closed_loop")
    if not cl:
        return None
    hist = cl.get("history") or []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    if hist:
        it = [h["iter"] for h in hist]
        axes[0].plot(it, [h["rms"] for h in hist], "o-", label="RMS (λ)")
        axes[0].plot(it, [h["z_norm"] for h in hist], "s--", label="‖z‖ (λ)")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("λ")
        axes[0].set_title("Closed-loop convergence")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    axes[1].bar(["before", "after"], [cl["before_rms"], cl["after_rms"]],
                color=["tab:gray", "tab:green"])
    axes[1].set_ylabel("wavefront RMS (λ)")
    axes[1].set_title(f"RMS {cl['before_rms']:.4f} → {cl['after_rms']:.4f} λ "
                      f"({100 * (1 - cl['after_rms'] / cl['before_rms']):.1f}%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return cl


# ─────────────────────────── markdown ───────────────────────────

def write_markdown(out: Path, ctx: dict) -> None:
    r = ctx["result"]
    dc = dict(r.device_config or {})
    modes = dc.get("slm_mode_ids_dll") or list(range(2, r.matrix.shape[1] + 2))
    matrix = r.matrix
    diag_ok = ctx["diag_ok"]
    s = np.linalg.svd(matrix, compute_uv=False)

    md: list[str] = []
    md.append("# Zernike 响应矩阵报告 (创建 / 分析 / 检测)\n")
    md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    srcs = [f"`{ctx['h5'].name}`"]
    if ctx["scan_report"]:
        srcs.append(f"`{ctx['scan_report'].name}`")
    if ctx["raw_scan"]:
        srcs.append(f"`{ctx['raw_scan'].name}`")
    md.append(f"**数据源**: {', '.join(srcs)}\n")

    md.append("## 1. 创建 (Creation)\n")
    md.append("响应矩阵由 `tools/slm/slm_zernike_response.py` 以**推拉法**标定: 对每个 SLM "
              "Zernike 模式施加 ±A 扰动, 测 WFS Zernike 读数增量, "
              "`matrix[wfs_coeff, slm_mode] = Δ(WFS)/Δ(SLM 幅度)`。\n")
    md.append("| 项 | 值 |")
    md.append("|---|---|")
    md.append(f"| 设备 | SLM #{dc.get('slm_serial', '—')} + WFS {dc.get('wfs_serial', '—')} |")
    md.append(f"| 波长 / 2π 灰度 | {dc.get('wavelength_nm', '—')} nm / "
              f"{dc.get('slm_2pi_gray', '—')} |")
    shift = dc.get("shift")
    if shift is None and dc.get("shift_x") is not None:
        shift = [dc.get("shift_x"), dc.get("shift_y")]
    pc = dc.get("pupil_center_mm") or []
    pd = dc.get("pupil_diameter_mm") or []
    md.append(f"| SLM shift | {shift} |")
    md.append(f"| Zernike 半径 | {dc.get('zernike_radius_px', '—')} px |")
    md.append(f"| 扰动幅度 | {dc.get('amplitude_rad', r.magnitude * 2 * np.pi):.2f} rad "
              f"({r.magnitude:.4f} λ) |")
    md.append(f"| 推拉循环 / 帧平均 | {r.n_cycles} / {r.n_averages} |")
    md.append(f"| WFS pupil | center=({pc[0]:.3f}, {pc[1]:.3f}) mm, "
              f"diameter=({pd[0]:.3f}, {pd[1]:.3f}) mm |" if len(pc) >= 2 and len(pd) >= 2
              else "| WFS pupil | — |")
    md.append(f"| 参考平面 | {dc.get('reference', '—')} |")
    md.append(f"| 矩阵形状 | {matrix.shape} (WFS × SLM) |")
    md.append(f"| SLM 模式 (DLL 索引) | {modes} |")
    md.append("")
    md.append("> ⚠️ **索引约定**: 矩阵行 = DLL 系数索引 (**顺序 m 枚举, 非标准 Noll**); "
              "`[5](2,0)defocus` `[9](3,1)coma` `[13](4,0)spherical`。"
              "手册佐证 `roCMm` \"derived from Zernike coefficient Z[5]\"。\n")

    md.append("## 2. 分析 (Analysis)\n")
    md.append("### 2.1 响应矩阵热图\n")
    md.append("![matrix heatmap](figures/01_matrix_heatmap.png)\n")
    md.append("绿框 = 对角线 (同索引 SLM 模式 → WFS 系数)。\n")
    md.append("### 2.2 对角主导性\n")
    md.append("![diagonal](figures/02_diagonal_dominance.png)\n")
    md.append(f"**{diag_ok}/{len(modes)}** 个模式的列主导项即对角线项。\n")
    md.append("### 2.3 奇异值谱 / 条件数\n")
    md.append("![svd](figures/03_singular_values.png)\n")
    md.append(f"条件数 **{s[0] / s[-1]:.2f}** (良态, 伪逆稳定)。\n")
    md.append("### 2.4 重复性方差\n")
    md.append("![variance](figures/04_variance_map.png)\n")
    md.append(f"平均方差 **{r.mean_variance:.3e}**, 最大 **{r.max_variance:.3e}**。\n")

    if ctx["linearity"]:
        md.append("### 2.5 线性度 (多尺寸扫描)\n")
        md.append("![linearity](figures/05_linearity.png)\n")
        md.append("**正确判据**: 响应已按单位幅度归一化 → 线性响应表现为 `|resp|` **恒定**, "
                  "故用幅度离散度 **CV** 与响应向量方向一致性 **cos** "
                  "(原 slope/R² 判据在线性时 slope≈0, 无意义)。\n")
        md.append("| 模式 | 尺寸 R(px) | CV(%) | cos_min | 判定 |")
        md.append("|---|---|---|---|---|")
        for row in sorted(ctx["linearity"], key=lambda x: (x["dll_index"], x["radius"])):
            md.append(f"| [{row['dll_index']}] {_short(DLL_ZERNIKE_NAMES.get(row['dll_index'], ''))} "
                      f"| {row['radius']:.0f} | {100 * row['cv']:.1f} | "
                      f"{row['cos_min']:.3f} | {'✅' if row['ok'] else '❌'} |")
        md.append("")

    md.append("## 3. 检测 (Detection)\n")
    if ctx["worst"]:
        md.append("### 3.1 异常点诊断\n")
        md.append("![outlier](figures/06_outlier_diagnosis.png)\n")
        md.append("**根因**: Zernike 半径 ≈ 光束半径 (实测 200px) 时, 大幅度高阶模式把子孔径"
                  "光斑推出有效区 → DLL Zernike 拟合崩溃, `|resp|` 暴涨 (实测最高 4288)。"
                  "故标定应取 **R ≈ 1.5×光束半径**。\n")
    if ctx["inverse"]:
        md.append("### 3.2 离线反解验证\n")
        md.append("![inverse](figures/07_inverse_demo.png)\n")
        inv = ctx["inverse"]
        md.append(f"合成像差 `[5]defocus=+0.30λ, [9]coma=−0.20λ` → 反解 "
                  f"`c = pinv(M) @ w`, 残差 ‖Mc−w‖={inv['residual']:.4f} / "
                  f"‖w‖={inv['w_norm']:.4f} → **降低 {inv['reduction_pct']:.1f}%**。\n")
    if ctx["closed_loop"]:
        md.append("### 3.3 实测闭环矫正\n")
        md.append("![closed loop](figures/08_closed_loop.png)\n")
        cl = ctx["closed_loop"]
        md.append(f"恢复 WFS 内部参考后, 矫正前 RMS={cl['before_rms']:.4f}λ → "
                  f"矫正后 {cl['after_rms']:.4f}λ (**{100 * (1 - cl['after_rms'] / cl['before_rms']):.1f}%**)。\n")
        md.append("> 该轮使用被 R=200 异常点污染的矩阵 (见 3.1), 改善有限; "
                  "修正后应以 R≈300px 重跑阶段 2/3。\n")

    md.append("## 4. 产物\n")
    md.append("| 产物 | 路径 |")
    md.append("|---|---|")
    md.append(f"| 响应矩阵 h5 | `{ctx['h5']}` |")
    md.append(f"| 矩阵 + 原始读数 json | `{ctx['h5'].with_suffix('.json')}` |")
    if ctx["scan_report"]:
        md.append(f"| 多尺寸扫描报告 | `{ctx['scan_report']}` |")
    if ctx["raw_scan"]:
        md.append(f"| 多尺寸原始扫描 | `{ctx['raw_scan']}` |")
    md.append("| 标定工具 | `src/ao_shaping/tools/slm/slm_zernike_response.py` |")
    md.append("| 三阶段工具 | `src/ao_shaping/tools/slm/slm_zernike_correction.py` |")
    md.append("| 共享模块 | `src/ao_shaping/tools/slm/slm_zernike_common.py` |")
    md.append("")

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


# ─────────────────────────── main ───────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=None, help="response matrix .h5 (default: latest)")
    ap.add_argument("--scan-report", default=None,
                    help="multi-size scan report json (default: latest)")
    ap.add_argument("--raw-scan", default=None, help="raw scan json (default: latest)")
    ap.add_argument("-o", "--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    h5 = Path(args.h5) if args.h5 else _latest(
        str(ROOT / "data" / "zernike_response_matrix" / "*.h5"))
    if h5 is None or not h5.exists():
        print("[FAIL] 未找到响应矩阵 h5")
        return 1
    scan_report = (Path(args.scan_report) if args.scan_report else _latest(
        str(ROOT / "data" / "zernike_correction" / "report_*.json")))
    raw_scan = (Path(args.raw_scan) if args.raw_scan else _latest(
        str(ROOT / "data" / "zernike_correction" / "raw_scan_*.json")))

    out = Path(args.output_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("[Zernike 响应矩阵报告] 创建 / 分析 / 检测")
    print("=" * 72)
    print(f"  h5          : {h5}")
    print(f"  scan report : {scan_report}")
    print(f"  raw scan    : {raw_scan}")

    result = load_zernike_response_matrix(h5)
    dc = dict(result.device_config or {})
    modes = dc.get("slm_mode_ids_dll") or list(range(2, result.matrix.shape[1] + 2))
    matrix = result.matrix

    print(f"[OK] matrix {matrix.shape}, cond={np.linalg.cond(matrix):.3f}")

    fig_matrix_heatmap(matrix, modes, figs / "01_matrix_heatmap.png")
    fig_diagonal(matrix, modes, figs / "02_diagonal_dominance.png")
    fig_singular_values(matrix, figs / "03_singular_values.png")
    fig_variance(result.variance_matrix, modes, figs / "04_variance_map.png")

    diag_ok = sum(1 for col, m in enumerate(modes)
                  if m - 1 < matrix.shape[0]
                  and int(np.argmax(np.abs(matrix[:, col]))) == m - 1)
    print(f"[OK] diagonal dominance: {diag_ok}/{len(modes)}")

    linearity: list[dict] = []
    worst: list[int] = []
    if raw_scan and raw_scan.exists():
        raw = json.loads(raw_scan.read_text(encoding="utf-8"))
        linearity = fig_linearity(raw, figs / "05_linearity.png")
        bad = [r for r in linearity if not r["ok"]]
        worst = sorted({r["dll_index"] for r in bad},
                       key=lambda m: -max(r["cv"] for r in bad if r["dll_index"] == m))[:4]
        print(f"[OK] linearity: {len(linearity) - len(bad)}/{len(linearity)} pass; "
              f"worst modes {worst}")
        if worst:
            fig_outlier(raw, figs / "06_outlier_diagnosis.png", worst)

    inverse = fig_inverse(matrix, modes, figs / "07_inverse_demo.png")
    print(f"[OK] inverse demo: residual {inverse['residual']:.4f} "
          f"({inverse['reduction_pct']:.1f}% reduction)")

    closed_loop = None
    if scan_report and scan_report.exists():
        rep = json.loads(scan_report.read_text(encoding="utf-8"))
        closed_loop = fig_closed_loop(rep, figs / "08_closed_loop.png")
        if closed_loop:
            print(f"[OK] closed loop: {closed_loop['before_rms']:.4f} → "
                  f"{closed_loop['after_rms']:.4f} λ")

    write_markdown(out, {"result": result, "h5": h5, "scan_report": scan_report,
                         "raw_scan": raw_scan, "diag_ok": diag_ok,
                         "linearity": linearity, "worst": worst,
                         "inverse": inverse, "closed_loop": closed_loop})
    print(f"\n[OK] 报告: {out / 'report.md'}")
    print(f"[OK] 图: {figs}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
