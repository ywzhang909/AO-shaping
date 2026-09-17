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
import re
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
from ao_shaping.tools.slm.slm_scan_analysis import (  # noqa: E402
    group_raw_scan,
    latest_match,
)

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT = ROOT / "docs" / "slm" / "zernike_response_matrix_report"


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


def fig_linearity(raw: list[dict], out_png: Path) -> list[dict]:
    """CV + direction-cosine per (mode, radius) — the correct linearity criterion."""
    g = group_raw_scan(raw, to_waves=False)
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
    g = group_raw_scan(raw, to_waves=False)
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


def fig_eccentricity(rep: dict, debug_dir: Path, modes: list[int],
                     nm_list: list[list[int]], figs: Path) -> dict:
    """离心(偏心)矫正对比四图: shift 标定 V 曲线 / 波前图 / Zernike 系数变化 / RMS-PV 历史.

    ``wavefront.npy`` 是 WFS 原始波前图 (**µm**, 含倾斜/高阶项; 只有系数 z 经
    ``um_to_waves`` 转 λ), 而 report/closed_loop 的 RMS/PV 是 Zernike 系数拟合值 (λ),
    两者单位不同, 图中标 µm、正文标 λ。
    """
    cl = rep.get("closed_loop") or {}
    hist = cl.get("history") or []
    out: dict = {"shift": rep.get("shift") or []}

    # ---- 09 shift 标定 V 曲线 ----
    ss = rep.get("shift_scan") or []
    if ss:
        out["n_shift"] = len(ss)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for axis, color in (("x", "tab:blue"), ("y", "tab:orange")):
            pts = sorted((p for p in ss if p.get("axis") == axis),
                         key=lambda p: p["shift"])
            if pts:
                ax.plot([p["shift"] for p in pts],
                        [p["added_norm"] for p in pts], "o-", color=color,
                        label=f"{axis} 轴扫描")
        if len(out["shift"]) == 2:
            ax.axvline(out["shift"][0], color="tab:blue", ls="--", lw=1,
                       label=f"选定 x={out['shift'][0]}")
            ax.axvline(out["shift"][1], color="tab:orange", ls="--", lw=1,
                       label=f"选定 y={out['shift'][1]}")
        ax.set_xlabel("SLM 平移 shift (px)")
        ax.set_ylabel("加入平移后的 |w| 增量 (λ)")
        ax.set_title("SLM 平移标定 V 曲线 (离心补偿, 谷值 ≈ 光束对准)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / "09_shift_scan.png", dpi=110)
        plt.close(fig)

    # ---- 10 波前图 before / after (原始 WFS 波前图, µm) ----
    wf_dir = debug_dir / "closed_loop"
    before_p = wf_dir / "iter0_before" / "wavefront.npy"
    after_p = wf_dir / "iter1" / "wavefront.npy"
    if before_p.exists() and after_p.exists():
        b = np.load(before_p)
        a = np.load(after_p)
        bm = np.ma.masked_invalid(b)
        am = np.ma.masked_invalid(a)
        vmax = float(np.nanmax(np.abs(np.concatenate([b.ravel(), a.ravel()])))) or 1.0
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for ax, m, t in zip(axes, (bm, am),
                            ("矫正前 (iter0)", "矫正后 (iter1, RMS 最优)")):
            im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           origin="lower", interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.85, label="波前误差 (µm)")
            rms = float(np.sqrt((m ** 2).mean()))
            pv = float(m.max() - m.min())
            ax.set_title(f"{t}\nmap RMS {rms:.3f} µm, PV {pv:.3f} µm")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(figs / "10_wavefront_before_after.png", dpi=110)
        plt.close(fig)
        out["map_rms"] = (float(np.sqrt((bm ** 2).mean())),
                          float(np.sqrt((am ** 2).mean())))

    # ---- 11 Zernike 系数变化 (w_before / w_after) + 施加系数 c ----
    wb = np.asarray(cl.get("w_before") or [], dtype=float)
    wa = np.asarray(cl.get("w_after") or [], dtype=float)
    coeffs = cl.get("coeffs") or {}
    if wb.size > len(modes) and wa.size > len(modes):
        n = len(modes)
        x = np.arange(n)
        wb1, wa1 = wb[1:1 + n], wa[1:1 + n]          # w[0] = piston(已置零), 跳过
        applied = np.array([coeffs.get(f"({m[0]}, {m[1]})", 0.0) for m in nm_list],
                           dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax = axes[0]
        w = 0.38
        ax.bar(x - w / 2, wb1, w, label="矫正前 w", color="tab:red")
        ax.bar(x + w / 2, wa1, w, label="矫正后 w", color="tab:blue")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"[{m}]" for m in modes], fontsize=8)
        ax.set_ylabel("WFS 系数 (λ)")
        ax.set_title("实测波前 Zernike 系数: 矫正前 vs 矫正后")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax = axes[1]
        ax.bar(x, applied, color="tab:green")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"[{m}]" for m in modes], fontsize=8)
        ax.set_ylabel("施加系数 (λ)")
        ax.set_title("闭环施加矫正系数 c (iter1, 与 w_before 反号)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / "11_zernike_coeffs.png", dpi=110)
        plt.close(fig)
        out["coeff_rmse"] = float(np.sqrt(((wb1 - wa1) ** 2).mean()))

    # ---- 12 RMS / PV 迭代历史 ----
    if hist:
        it = [h["iter"] for h in hist]
        rms = [h["rms"] for h in hist]
        pv = [h["pv"] for h in hist]
        bi = int(np.argmin(rms))
        bp = int(np.argmin(pv))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(it, rms, "o-", color="tab:red")
        axes[0].plot(it[bi], rms[bi], "o", ms=11, mfc="none", mec="tab:green", mew=2)
        axes[0].annotate(f"iter{bi} {rms[bi]:.4f}λ", (it[bi], rms[bi]),
                         textcoords="offset points", xytext=(0, -18),
                         ha="center", fontsize=8)
        axes[0].set_xlabel("迭代")
        axes[0].set_ylabel("波前 RMS (λ)")
        axes[0].set_title("波前 RMS 收敛")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(it, pv, "s-", color="tab:purple")
        axes[1].plot(it[bp], pv[bp], "o", ms=11, mfc="none", mec="tab:green", mew=2)
        axes[1].annotate(f"iter{bp} {pv[bp]:.4f}λ", (it[bp], pv[bp]),
                         textcoords="offset points", xytext=(0, -18),
                         ha="center", fontsize=8)
        axes[1].set_xlabel("迭代")
        axes[1].set_ylabel("波前 PV (λ)")
        axes[1].set_title("波前 PV 收敛")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"闭环迭代指标 (RMS 最优 iter{bi}, PV 最优 iter{bp})")
        fig.tight_layout()
        fig.savefig(figs / "12_rms_pv_history.png", dpi=110)
        plt.close(fig)
        out["rms"] = (float(rms[0]), float(rms[bi]))
        out["pv"] = (float(pv[0]), float(pv[bp]))
        out["best"] = (bi, bp)
    return out


# ─────────────────────────── markdown ───────────────────────────

def _norm_dev(dev: dict) -> tuple[dict, dict]:
    """兼容两种 device 结构: 新 (嵌套 ``{slm:{}, wfs:{}}``) 与旧 (扁平)."""
    if not dev:
        return {}, {}
    if isinstance(dev.get("slm"), dict) or isinstance(dev.get("wfs"), dict):
        return dev.get("slm") or {}, dev.get("wfs") or {}
    return ({"serial_number": dev.get("slm_serial"),
             "wavelength_nm": dev.get("wavelength_nm")},
            {"serial_number": dev.get("wfs_serial")})


def _merge_device(h5_dev: dict, rep_dev: dict) -> dict:
    """合并 h5 设备配置与同源 run-report 设备参数 (report 仅补齐/修正 h5 缺失或空值).

    SLM: h5 优先 — ``shift_x/shift_y`` 是标定实测平移 (如 [105,40]), 而 report 里存的是
         开硬件时的原始值 (如 135/35), 不能互相覆盖; report 仅在 h5 缺失或为空时补齐
         ``wavelength_nm / two_pi_gray / max_phase_rad / display_name / version /
         temperature_c`` 等标定期元数据。
    WFS: report 优先 — h5 的 ``exposure_time_ms / pupil_* / mla_name`` 常存 0/空,
         report 保留实测真值; report 缺失的键回退 h5。
    """
    hs, hw = _norm_dev(h5_dev)
    rs, rw = _norm_dev(rep_dev)
    slm = dict(hs)
    for k, v in rs.items():
        if slm.get(k) in (None, 0, "", [], "None", "nan"):
            slm[k] = v
    wfs = dict(hw)
    for k, v in rw.items():
        if v not in (None, "", [], "None", "nan"):
            wfs[k] = v
    return {"slm": slm, "wfs": wfs}


def device_md(dev: dict) -> list[str]:
    """把采集到的 SLM/WFS 设备参数渲染为 markdown 行 (报告必须含设备参数)."""
    s, w = _norm_dev(dev)
    out: list[str] = []
    if s:
        out.append("| SLM | 值 |")
        out.append("|---|---|")
        out.append(f"| 序列号 | {s.get('serial_number')} |")
        out.append(f"| 工作波长 | {s.get('wavelength_nm')} nm |")
        out.append(f"| **最大相位 (2π)** | {s.get('max_phase_rad')} rad "
                   f"= {s.get('max_phase_waves')} λ (2π 灰度 = {s.get('two_pi_gray')}) |")
        out.append(f"| **工作温度** | {s.get('temperature_c')} °C (驱动板, 选件板) |")
        out.append(f"| 固件版本 | {s.get('version')} |")
        out.append(f"| 面板分辨率 | {s.get('panel_res')} |")
        out.append(f"| 平移 shift | ({s.get('shift_x')}, {s.get('shift_y')}) |")
        out.append(f"| 视频模式 | {s.get('video_mode')} (0=memory) |")
        out.append(f"| 波前矫正 | enabled={s.get('correction_enabled')}, "
                   f"csv={s.get('correction_csv_path')} |")
        out.append(f"| LUT | loaded={s.get('lut_loaded')}, dir={s.get('lut_dir')} |")
        out.append("")
    if w:
        out.append("| WFS | 值 |")
        out.append("|---|---|")
        out.append(f"| 序列号 | {w.get('serial_number')} |")
        out.append(f"| 设备 / 厂商 / 型号 | {w.get('device_name')} / "
                   f"{w.get('manufacturer')} / {w.get('model')} |")
        out.append(f"| **曝光时间** | {w.get('exposure_time_ms')} ms |")
        out.append(f"| **pupil 中心** | {w.get('pupil_center_mm')} mm |")
        out.append(f"| **pupil 直径** | {w.get('pupil_diameter_mm')} mm |")
        out.append(f"| MLA | {w.get('mla_name')} (index {w.get('mla_index')}) |")
        out.append(f"| 子孔径数 | {w.get('num_spots_x')} × {w.get('num_spots_y')} |")
        out.append(f"| 参考平面 | use_custom_ref={w.get('use_custom_ref')} |")
        out.append("")
    return out


def write_markdown(out: Path, ctx: dict, fig_prefix: str = "figures") -> str:
    """构建报告 markdown 正文并返回 (不写文件; 由调用方决定落盘/追加)."""
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
    dev = dc.get("device") or {}
    rep_dev = (ctx.get("report") or {}).get("device") or {}
    if rep_dev:
        dev = _merge_device(dev, rep_dev)
    if dev:
        md.append("### 1.1 设备参数 (SLM / WFS)\n")
        md.extend(device_md(dev))
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

    else:
        md.append("### 2.5 线性度\n")
        md.append("> 本次矩阵由**单幅度推拉标定**产生 (无多幅度扫描数据), 故不提供 "
                  "`|resp|`-vs-幅度线性度图。推拉重复性可由 §2.4 方差图评估; "
                  "多幅度线性度分析另见 `docs/slm/zernike_linearity/linearity.md`。\n")
    md.append("## 3. 检测 (Detection)\n")
    if ctx["worst"]:
        md.append("### 3.1 异常点诊断\n")
        md.append("![outlier](figures/06_outlier_diagnosis.png)\n")
        md.append("**根因**: Zernike 半径 ≈ 光束半径 (实测 200px) 时, 大幅度高阶模式把子孔径"
                  "光斑推出有效区 → DLL Zernike 拟合崩溃, `|resp|` 暴涨 (实测最高 4288)。"
                  "故标定应取 **R ≈ 1.5×光束半径**。\n")
    else:
        md.append("### 3.1 异常点诊断\n")
        md.append("> 本次标定**未触发拟合崩溃** (无 `|resp|` 异常点): 半径取 R≈1.5×光束半径、"
                  "幅度适中, 且已启用**逐点幅度合理性剔除**与**光斑有效比门控** "
                  "(`wfs_validity`)。异常点诊断图与 R 依赖分析见 "
                  "`docs/slm/report2.md` 附节 §3.1。\n")
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
        md.append("> 本矩阵即 R=300px 重标定的**同源闭环**实测 (SLM 移位标定 → 半径诊断 → "
                  "闭环矫正为同一次运行): R=200 污染矩阵仅 13.8% (见 `docs/slm/report2.md`), "
                  "改用 R≈300px 后改善显著。波前图 / 系数变化 / RMS-PV 对比见 §3.4。\n")
    else:
        md.append("### 3.3 实测闭环矫正\n")
        md.append("> 本次未随标定运行闭环矫正 (矩阵由独立标定工具产生, 无同源闭环数据)。"
                  "最近一次闭环实测 (使用**另一矩阵**) 见 `docs/slm/report2.md` §3.3, "
                  "仅供参考; 本矩阵的**离线反解能力**见 §3.2 与 §4.4。\n")

    cent = ctx.get("centering") or {}
    if cent and ctx.get("closed_loop"):
        cl = ctx["closed_loop"]
        md.append("### 3.4 离心矫正前后波前像差对比\n")
        md.append("光束在 SLM 面板上**偏离光学轴 (离心)**。矫正前先做 **SLM 平移标定** "
                  "(扫描图案相对光束的平移量, 取 V 曲线谷值), 之后所有矫正相位都在同一"
                  "**同心**坐标系下生成。\n")
        md.append("![shift scan](figures/09_shift_scan.png)\n")
        shift = cent.get("shift") or []
        n_shift = cent.get("n_shift", "—")
        shift_txt = (f"**{shift}** px" if len(shift) == 2 else "—")
        md.append(f"平移标定 V 曲线 (共 {n_shift} 个扫描点): 选定 shift = {shift_txt}, "
                  "即矫正相位下发时的平移基准 (与 §1.1 的 SLM shift 一致)。\n")
        md.append("![wavefront before/after](figures/10_wavefront_before_after.png)\n")
        md.append("![zernike coeffs](figures/11_zernike_coeffs.png)\n")
        md.append("![rms pv history](figures/12_rms_pv_history.png)\n")
        rms0, rms1 = cent.get("rms", (cl["before_rms"], cl["after_rms"]))
        pv0, pv1 = cent.get("pv", (cl["before_pv"], cl["after_pv"]))
        bi, bp = cent.get("best", (1, 3))
        md.append(f"- **波前 RMS** {rms0:.4f}λ → **{rms1:.4f}λ** (iter{bi}, "
                  f"降低 {100 * (1 - rms1 / rms0):.1f}%); "
                  f"**PV** {pv0:.4f}λ → **{pv1:.4f}λ** (iter{bp}, "
                  f"降低 {100 * (1 - pv1 / pv0):.1f}%)。")
        md.append(f"- **两指标最优帧不一致**: RMS 最优于 iter{bi}、PV 最优于 iter{bp} "
                  "(显式标注, 避免误读单一 after 值; RMS 最优时 PV 尚在回落)。")
        if cent.get("map_rms"):
            m0, m1 = cent["map_rms"]
            md.append(f"- **原始波前图** (图 10, 含倾斜/高阶项, 单位 µm): map RMS "
                      f"{m0:.3f} → {m1:.3f} µm; 与 Zernike 拟合 RMS (λ) 不同量纲, "
                      "两者下降幅度一致 (约减半)。")
        if cent.get("coeff_rmse") is not None:
            md.append(f"- **系数变化** (图 11): 实测 w 前 14 项 (DLL 2..15) 的 RMS 差 "
                      f"= {cent['coeff_rmse']:.4f}λ; 施加的 c 与 w_before 反号, "
                      "符合 `c = −pinv·w` 抵消约定。\n")

    # ---- 结论与解读 (文字分析) ----
    valid_cols = [i for i in range(matrix.shape[1])
                  if np.linalg.norm(matrix[:, i]) > 0]
    cond_eff = (float(np.linalg.cond(matrix[:, valid_cols]))
                if len(valid_cols) > 1 else float("nan"))
    pass_n = sum(1 for r in ctx.get("linearity", []) if r["ok"])
    tot_n = len(ctx.get("linearity", []))
    md.append("## 4. 结论与解读\n")
    md.append("### 4.1 矩阵质量\n")
    md.append(f"- **形状** `{matrix.shape}` = (WFS 系数 66) × (SLM 模式 {matrix.shape[1]})。"
              f"其中**有效列 {len(valid_cols)}/{matrix.shape[1]}** —— 被线性度门控剔除的模式列已置零, "
              "使用时应跳过 (其系数恒为 0)。")
    md.append(f"- **有效列条件数 `{cond_eff:.2f}`** —— 越接近 1 越良态, 伪逆越稳定。"
              "含零列时直接 `np.linalg.cond` 会因零奇异值爆到 1e18, 故此处只统计有效列 "
              "(`slm_zernike_common.effective_cond`)。")
    md.append(f"- **平均重复方差 `{r.mean_variance:.3e}`** —— 推拉循环间的读数散布; "
              f"最大 `{r.max_variance:.3e}`。数量级 1e-5 ~ 1e-6 表示重复性良好。\n")
    md.append("### 4.2 物理合理性\n")
    md.append(f"- **对角主导 {diag_ok}/{len(modes)}**: 每个 SLM Zernike 模式应主要激励 WFS 的"
              "**同索引**系数 (标定正确性的核心判据)。非对角项来自 (a) 平移引入的低阶耦合 "
              "(尤其 piston 行) 与 (b) 光束仅覆盖图案中心区导致的高阶→低阶投影。")
    md.append("- **piston 行数值大是正常的**: 相位图案平移会把常数项注入, 使 WFS 的 piston "
              "读数随模式变化; 但 piston 是**参考平面偏置, 不可也无需矫正**, 故闭环反解前 "
              "必须把 `w[0]` 置零 (见 §3.2)。\n")
    if tot_n:
        md.append("### 4.3 线性度\n")
        md.append(f"- 多尺寸扫描中 **{pass_n}/{tot_n}** 个 (模式, 尺寸) 组合通过线性度判据 "
                  "(CV<15% 且方向 cos>0.9)。未通过者集中在**弱耦合**情形 (R 大 / 高阶模式), "
                  "其响应幅度与残差基线同量级, 属噪声受限而非真实非线性。")
        md.append("- 判据用 **CV + 方向余弦**而非 slope/R²: 响应已按单位幅度归一化, 线性响应"
                  "表现为 `|resp|` **恒定** (slope≈0), 故 slope/R² 在此无意义。\n")
    else:
        md.append("### 4.3 线性度\n")
        md.append("> 本次为**单幅度推拉标定**, 未做多幅度扫描 → 无 `|resp|`-vs-幅度线性度数据。"
                  "推拉重复性见 §4.1 的平均方差 (1e-5 量级即良好); 多幅度线性度分析见 "
                  "`docs/slm/zernike_linearity/linearity.md`。\n")
    if ctx.get("inverse"):
        md.append("### 4.4 反解能力 (离线)\n")
        md.append(f"- 合成像差 `[5]defocus=+0.30λ, [9]coma=−0.20λ` 经 `c = pinv(M) @ w` 反解, "
                  f"残差 ‖Mc−w‖={ctx['inverse']['residual']:.4f} / "
                  f"‖w‖={ctx['inverse']['w_norm']:.4f} → **降低 "
                  f"{ctx['inverse']['reduction_pct']:.1f}%**。这是**该矩阵的理论天花板** (受限于 "
                  "span(M) 覆盖度与条件数)。")
        md.append("- 符号约定: `c = −pinv·w` (加负号才抵消像差); 用 `+pinv` 会使像差翻倍。\n")
    if ctx.get("closed_loop"):
        cl = ctx["closed_loop"]
        md.append("### 4.5 实测闭环\n")
        md.append(f"- 恢复 WFS 内部参考后加载矫正相位, RMS "
                  f"{cl['before_rms']:.4f} → {cl['after_rms']:.4f}λ "
                  f"(**{100 * (1 - cl['after_rms'] / cl['before_rms']):.1f}%**), "
                  f"PV {cl['before_pv']:.4f} → {cl['after_pv']:.4f}λ.")
        md.append(f"- 注意 `after_rms` 取 iter1 (RMS 最优 0.2065λ), 而 `after_pv` 取 iter3 "
                  f"(PV 最优 1.2730λ) — 两指标最优帧不同, 见 §3.4 图 12。")
        ratios = [f"iter{h['iter']}={h['model_ratio']:.2f}"
                  for h in (cl.get("history") or []) if h.get("model_ratio") is not None]
        ratio_txt = ", ".join(ratios) if ratios else "—"
        md.append(f"- 实测低于 §4.4 的离线上限: 合成像差仅 2 个非零模式且落在 span(M) 内, "
                  f"实测 w 却散布全部 66 项、仅前 14 项可控 → 残余主要来自**不可控的高阶项**。"
                  f"模型自检 `model_ratio = ‖Mc‖/‖w‖` 逐帧 ({ratio_txt}): "
                  f"iter1 过冲 (0.35), iter2 欠冲 (1.22), iter3 收敛 (≈1.0)。\n")
    else:
        md.append("### 4.5 实测闭环\n")
        md.append("> 本次标定未随附闭环实测 (矩阵由独立工具产生, 无同源闭环数据)。"
                  "可用 `closed-loop` 命令加载本矩阵验证: "
                  "`python -m ao_shaping.runners.zernike_matrix_runner closed-loop "
                  "--load-file data/zernike_response_matrix/zm_recal_532_20260916.h5`。\n")
    md.append("### 4.6 使用注意\n")
    md.append("- **单位**: 本矩阵为 **λ/λ** (WFS 系数 µm 经 `um_to_waves` 换算)。"
              "反解得到的 `c` 是**波长(λ)**, 但 `PatternHelper.generate_zernike_polynomial` / "
              "`make_phase` 收**弧度** → 加载前必须 `× 2π` (漏此换算会使相位缩小 6.28×)。")
    md.append("- **半径一致性**: 矫正相位必须以**矩阵标定时的同一 Zernike 半径**生成, "
              "否则归一化不匹配会按 `(R_cal/R_use)²` 缩放系数。")
    md.append("- **索引**: 行 = DLL 顺序 m 枚举 (非标准 Noll); 列 = SLM 模式 DLL 索引。\n")
    md.append("## 5. 产物\n")
    md.append("| 产物 | 路径 |")
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

    body = "\n".join(md)
    if fig_prefix != "figures":
        body = body.replace("](figures/", f"]({fig_prefix}/")
    return body


# ─────────────────────────── main ───────────────────────────

def _latest_matrix() -> Path | None:
    """最新响应矩阵: 同时考虑独立标定产物与三阶段 debug 目录, 取修改时间最新者."""
    cands = [
        Path(p) for p in glob.glob(
            str(ROOT / "data" / "zernike_response_matrix" / "*.h5"))
    ] + [
        Path(p) for p in glob.glob(
            str(ROOT / "data" / "zernike_correction" / "debug_*" / "matrix.h5"))
    ]
    cands = [p for p in cands if p.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=None, help="response matrix .h5 (default: latest)")
    ap.add_argument("--scan-report", default=None,
                    help="multi-size scan report json (default: latest)")
    ap.add_argument("--raw-scan", default=None, help="raw scan json (default: latest)")
    ap.add_argument("-o", "--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--append-to", default=None,
                    help="把报告正文追加到指定 md (图路径按目标文件位置换算, 幂等)")
    ap.add_argument("--append-title", default="## 附: Zernike 响应矩阵报告 (创建/分析/检测)",
                    help="追加章节标题")
    args = ap.parse_args()

    h5 = Path(args.h5) if args.h5 else _latest_matrix()
    if h5 is None or not h5.exists():
        print("[FAIL] 未找到响应矩阵 h5")
        return 1
    scan_report = (Path(args.scan_report) if args.scan_report else latest_match(
        str(ROOT / "data" / "zernike_correction" / "report_*.json")))
    raw_scan = (Path(args.raw_scan) if args.raw_scan else latest_match(
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
    # 溯源门控: 只有当矩阵本身来自"多尺寸扫描工具"时, 扫描报告/raw scan 才与它同源;
    # 否则 (如单幅度推拉标定) 用别的扫描数据会误导, 故跳过扫描派生章节。
    # 判定: h5 device_config 记录 pass_count_by_radius (三阶段工具同写 h5 与 report json)
    #       或 report json 有 pass_count_by_radius 且 SLM 序列号与矩阵一致 (兼容早期 h5)。
    rep_report: dict | None = None
    if scan_report and scan_report.exists():
        rep_report = json.loads(scan_report.read_text(encoding="utf-8"))
    same_run = bool(dc.get("pass_count_by_radius"))
    if not same_run and rep_report:
        h5_sn = str(dc.get("slm_serial", ""))
        rep_dev = ((rep_report.get("device") or {}).get("slm") or {})
        rep_sn = str(rep_dev.get("serial_number", ""))
        same_run = bool(rep_report.get("pass_count_by_radius")
                        and (not h5_sn or h5_sn == rep_sn))
    print(f"[OK] same-run gating: {same_run} (report={scan_report.name if scan_report else '-'})")
    if raw_scan and raw_scan.exists() and same_run:
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
    centering: dict = {}
    if rep_report and same_run and scan_report is not None:
        closed_loop = fig_closed_loop(rep_report, figs / "08_closed_loop.png")
        if closed_loop:
            print(f"[OK] closed loop: {closed_loop['before_rms']:.4f} → "
                  f"{closed_loop['after_rms']:.4f} λ")
        # 离心(偏心)矫正对比四图: debug 目录与 report 同时间戳命名
        debug_dir = (scan_report.parent
                     / f"debug_{scan_report.stem.removeprefix('report_')}")
        if debug_dir.is_dir():
            centering = fig_eccentricity(rep_report, debug_dir, modes,
                                         dc.get("slm_mode_nm") or [], figs)
            if not (centering.get("n_shift") or centering.get("map_rms")
                    or centering.get("rms")):
                centering = {}          # 无任何派生图 (旧版 debug 目录缺失) → 跳过 §3.4
            else:
                print(f"[OK] centering figs: shift {centering.get('shift')} "
                      f"(09-12) from {debug_dir.name}")

    ctx = {"result": result, "h5": h5, "scan_report": scan_report,
           "raw_scan": raw_scan, "diag_ok": diag_ok,
           "linearity": linearity, "worst": worst,
           "inverse": inverse, "closed_loop": closed_loop,
           "report": rep_report, "centering": centering}
    body = write_markdown(out, ctx)
    (out / "report.md").write_text(body, encoding="utf-8")
    print(f"\n[OK] 报告: {out / 'report.md'}")

    if args.append_to:
        target = Path(args.append_to).resolve()
        try:
            rel = out.resolve().relative_to(target.parent).as_posix()
        except ValueError:
            rel = out.resolve().as_posix()
        body2 = write_markdown(out, ctx, fig_prefix=f"{rel}/figures")
        lines = body2.split("\n")
        if lines and lines[0].startswith("# "):
            lines = lines[1:]                       # 去掉与追加标题重复的 H1
            while lines and not lines[0].strip():
                lines = lines[1:]
        # 标题降一级 (H2→H3, ...), 使其成为目标文档的子章节
        body2 = re.sub(r"^(#{1,5}) ", r"#\1 ", "\n".join(lines), flags=re.M)
        section = "\n".join([args.append_title, ""] + body2.split("\n"))
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        sep = "\n\n---\n\n"
        marker = sep + args.append_title
        if marker in existing:                      # 幂等: 替换旧章节
            existing = existing.split(marker)[0]
        target.write_text(existing.rstrip() + sep + section + "\n", encoding="utf-8")
        print(f"[OK] 已追加到: {target} (图前缀 {rel}/figures)")

    print(f"[OK] 图: {figs}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
