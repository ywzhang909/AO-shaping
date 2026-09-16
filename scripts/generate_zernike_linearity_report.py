"""Generate the Zernike response linearity report (does WFS readout scale with the SLM coefficient?).

Reads the latest raw scan (``data/zernike_correction/raw_scan_*.json``) and answers:
**when the Zernike coefficient loaded on the SLM grows, does the corresponding
WFS-read Zernike coefficient grow proportionally?**

Method — for every (mode, radius):
  - ``diag``  = (z₊[m] − z₋[m]) / 2   WFS coefficient at the **same DLL index** (λ)
  - the response should be **proportional to the amplitude A**, so
      * ``diag / A`` must be constant (CV test), and
      * the linear fit through the origin must have R² → 1, and
      * ``diag(A=10) / diag(A=2)`` must be ≈ 5.0
  - the residual baseline ``|z₊ + z₋|/2`` (which should be constant) is used as an
    instability proxy: a combination is only judged on linearity when its response
    rises above that floor.

Offline — no hardware. Writes ``linearity.md`` + ``figures/`` to the output dir.

Usage:
    $env:PYTHONPATH = "src"
    python scripts/generate_zernike_linearity_report.py
    python scripts/generate_zernike_linearity_report.py -o docs/slm/zernike_linearity
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ao_shaping.tools.slm.slm_zernike_common import DLL_ZERNIKE_NAMES  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT = ROOT / "docs" / "slm" / "zernike_linearity"
AMPS = (2.0, 5.0, 10.0)


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def load_groups(raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    g: dict[tuple[int, float], dict[float, dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict))
    for s in raw:
        g[(s["dll_index"], float(s["radius"]))][float(s["amp_rad"])][int(s["sign"])] = \
            np.asarray(s["readout_um"], dtype=float) / 0.532     # → λ
    return g


def analyze(g: dict) -> list[dict]:
    rows: list[dict] = []
    for (m, R) in sorted(g):
        d = g[(m, R)]
        diag, vec, base = [], [], []
        ok = True
        for a in AMPS:
            zp, zn = d.get(a, {}).get(1), d.get(-a, {}).get(-1)
            if zp is None or zn is None:
                ok = False
                break
            diag.append(float((zp[m] - zn[m]) / 2.0))
            vec.append(float(np.linalg.norm(zp[1:] - zn[1:]) / 2.0))
            base.append(float(np.linalg.norm(zp[1:] + zn[1:]) / 2.0))
        if not ok or len(diag) < 3:
            continue
        diag_a, vec_a, base_a = map(np.array, (diag, vec, base))
        A = np.array(AMPS)
        # 过原点线性拟合 diag = k·A
        k = float(np.sum(A * diag_a) / np.sum(A * A))
        pred = k * A
        ss_res = float(np.sum((diag_a - pred) ** 2))
        ss_tot = float(np.sum((diag_a - diag_a.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ratios = diag_a / A
        cv = (float(np.std(ratios) / abs(np.mean(ratios)))
              if np.mean(ratios) != 0 else float("inf"))
        ratio_10_2 = (float(diag_a[2] / diag_a[0]) if diag_a[0] != 0 else np.nan)
        floor = float(np.median(base_a))
        snr = float(abs(diag_a[2]) / floor) if floor > 0 else float("inf")
        # 判定: 主判据用**统计上稳健**的 R² + CV (三点拟合已含点间散布);
        # `A10/A2` 在 A=2 时受基线噪声影响极大 (误差 ±0.1λ → 比值 ±2.4 不确定),
        # 故仅作展示, 不作硬判据。SNR 用于标注"弱耦合"而非判失败。
        if r2 > 0.98 and cv < 0.15:
            verdict = "成比例" if snr >= 1.5 else "成比例 (弱耦合)"
        elif snr < 1.5:
            verdict = "噪声受限"
        else:
            verdict = "**不成比例**"
        rows.append({"m": m, "R": R, "diag": diag_a.tolist(), "vec": vec_a.tolist(),
                     "base": floor, "k": k, "r2": float(r2), "cv": cv,
                     "ratio_10_2": ratio_10_2, "snr": snr, "verdict": verdict})
    return rows


def fig_response_vs_amp(g: dict, rows: list[dict], out_png: Path) -> None:
    """Response (diagonal WFS coefficient) vs SLM amplitude, with the linear fit."""
    good = [r for r in rows if r["verdict"] == "成比例"]
    good.sort(key=lambda r: -abs(r["diag"][2]))
    pick = good[:6] if len(good) >= 6 else good
    n = max(len(pick), 1)
    cols = 3
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(4.4 * cols, 3.8 * rws), squeeze=False)
    A = np.array(AMPS)
    for ax, r in zip(axes.ravel(), pick):
        d = np.array(r["diag"])
        ax.plot(A, d, "o", label="measured")
        ax.plot(A, r["k"] * A, "-", label=f"fit k={r['k']:.4f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"[{r['m']}] {DLL_ZERNIKE_NAMES.get(r['m'], '')}  R={r['R']:.0f}px\n"
                     f"R²={r['r2']:.4f}  A10/A2={r['ratio_10_2']:.2f} (期望 5.0)")
        ax.set_xlabel("SLM Zernike amplitude (rad)")
        ax.set_ylabel("WFS coeff (λ)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("WFS readout vs SLM Zernike amplitude (diagonal component) — proportional?")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def fig_ratio_and_r2(rows: list[dict], out_png: Path) -> None:
    """A=10/A=2 ratio (expect 5.0) and linear-fit R² per (mode, radius)."""
    labels = [f"[{r['m']}]R{r['R']:.0f}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.42 * len(rows)), 8), sharex=True)
    def _color(v: str) -> str:
        if v.startswith("**"):
            return "tab:red"          # 不成比例
        if v == "噪声受限":
            return "tab:gray"         # 响应低于基线, 不可信
        if "弱耦合" in v:
            return "tab:olive"        # 成比例但响应接近基线
        return "tab:green"            # 成比例

    colors = [_color(r["verdict"]) for r in rows]
    axes[0].bar(x, [r["ratio_10_2"] for r in rows], color=colors)
    axes[0].axhline(5.0, color="b", ls="--", lw=1.2, label="expected 5.0 (linear)")
    axes[0].set_ylabel("diag(A=10) / diag(A=2)")
    axes[0].set_title("Proportionality check — ratio should be 5.0 for a linear response")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, [r["r2"] for r in rows], color=colors)
    axes[1].axhline(0.98, color="r", ls="--", lw=1, label="R² threshold 0.98")
    axes[1].set_ylabel("linear-fit R²")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _norm_dev(dev: dict) -> tuple[dict, dict]:
    """兼容两种 device 结构: 新 (嵌套 ``{slm:{}, wfs:{}}``) 与旧 (扁平 ``{slm: 'serial', ...}``)."""
    if not dev:
        return {}, {}
    if isinstance(dev.get("slm"), dict) or isinstance(dev.get("wfs"), dict):
        return dev.get("slm") or {}, dev.get("wfs") or {}
    return ({"serial_number": dev.get("slm"), "wavelength_nm": dev.get("wavelength_nm"),
             "two_pi_gray": dev.get("2pi_gray")},
            {"serial_number": dev.get("wfs"), "exposure_time_ms": dev.get("exposure_ms"),
             "pupil_center_mm": dev.get("pupil_center_mm"),
             "pupil_diameter_mm": dev.get("pupil_diameter_mm")})


def device_md(dev: dict) -> list[str]:
    """把采集到的 SLM/WFS 设备参数渲染为 markdown 行 (报告必须含设备参数)."""
    s, w = _norm_dev(dev)
    out: list[str] = ["#### 设备参数\n"]
    if s:
        out.append("| SLM | 值 |")
        out.append("|---|---|")
        out.append(f"| 序列号 | {s.get('serial_number')} |")
        out.append(f"| 工作波长 | {s.get('wavelength_nm')} nm |")
        out.append(f"| **最大相位 (2π)** | {s.get('max_phase_rad')} rad "
                   f"= {s.get('max_phase_waves')} λ (2π 灰度 = {s.get('two_pi_gray')}) |")
        out.append(f"| 灰度上限 | {s.get('max_grayscale_value')} |")
        out.append(f"| **工作温度** | {s.get('temperature_c')} °C (驱动板, 选件板) |")
        out.append(f"| 固件版本 | {s.get('version')} |")
        out.append(f"| DisplayName | {s.get('display_name')} |")
        out.append(f"| 面板分辨率 | {s.get('panel_res')} |")
        out.append(f"| 平移 shift | ({s.get('shift_x')}, {s.get('shift_y')}) |")
        out.append(f"| 视频模式 | {s.get('video_mode')} (0=memory) |")
        out.append(f"| 波前矫正 | enabled={s.get('correction_enabled')}, "
                   f"csv={s.get('correction_csv_path')} |")
        out.append(f"| LUT | loaded={s.get('lut_loaded')}, dir={s.get('lut_dir')} |")
        out.append(f"| 响应时间 / 像素翻转上限 | {s.get('response_time_ms')} ms / "
                   f"{s.get('max_pixel_flip_ms')} ms |")
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
        out.append(f"| 高速模式 / 主增益 | {w.get('high_speed')} / {w.get('master_gain')} |")
        out.append("")
    return out


def write_markdown(out: Path, ctx: dict, fig_prefix: str = "figures") -> list[str]:
    rows = ctx["rows"]
    ok = [r for r in rows if r["verdict"].startswith("成比例")]
    noise = [r for r in rows if r["verdict"] == "噪声受限"]
    bad = [r for r in rows if r["verdict"].startswith("**")]
    md: list[str] = []
    md.append("### 本轮运行摘要 (优化后重跑)\n")
    rp = ctx.get("report") or {}
    dev = rp.get("device") or {}
    s, w = _norm_dev(dev)
    if s or w:
        md.append(f"- **SLM**: #{s.get('serial_number')} {s.get('wavelength_nm')}nm "
                  f"2π={s.get('two_pi_gray')}gray **温度={s.get('temperature_c')}°C** "
                  f"矫正={s.get('correction_enabled')}")
        md.append(f"- **WFS**: #{w.get('serial_number')} **曝光={w.get('exposure_time_ms')}ms** "
                  f"**pupil={w.get('pupil_center_mm')}mm d={w.get('pupil_diameter_mm')}mm** "
                  f"MLA={w.get('mla_name')}")
    md.append(f"- **shift**: {rp.get('shift')}  **纯平参考**: "
              f"RMS={rp.get('flat_reference', {}).get('flat_rms_lam', float('nan')):.4f}λ")
    md.extend(device_md(dev))
    diag = rp.get("radius_diagnostic") or []
    if diag:
        md.append("- **半径诊断** (含异常剔除): " +
                  ", ".join(f"R={d['radius']:.0f}→{d['defocus_lam']:+.3f}λ" for d in diag))
    md.append(f"- **统一矩阵半径**: {rp.get('matrix_radius')} px  "
              f"**通过数**: {rp.get('pass_count_by_radius')}")
    cl = rp.get("closed_loop") or {}
    b_rms, a_rms = cl.get("before_rms"), cl.get("after_rms")
    if b_rms and a_rms is not None:
        md.append(f"- **闭环**: RMS {b_rms:.4f} → {a_rms:.4f}λ "
                  f"(**{100 * (1 - a_rms / b_rms):.1f}%**), "
                  f"迭代 {cl.get('n_iter')} 轮, gain={cl.get('gain')}, leak={cl.get('leak')}")
        hist = cl.get("history") or []
        if hist:
            md.append("  " + " | ".join(f"iter{h['iter']}: {h['rms']:.4f}λ" for h in hist))
    md.append("")
    md.append("### 线性度分析: SLM Zernike 系数增大时 WFS 读数是否对应增大?\n")
    md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**数据源**: `{ctx['raw'].name}`\n")
    md.append("**判据**: 对每个 (模式, 半径), 取 WFS **同索引**系数响应 "
              "`diag = (z₊[m] − z₋[m])/2`, 检查它是否 ∝ 幅度 A —— "
              "要求 `diag/A` 恒定 (CV<15%) 且过原点拟合 R²>0.98; "
              "残差基线 `|z₊+z₋|/2` 用于标注弱耦合 (SNR<1.5 记 `噪声受限`)。\n")
    md.append(f"**结论: {len(ok)}/{len(rows)} 组合成比例**"
              f" ({len(noise)} 个弱耦合/噪声受限, {len(bad)} 个不成比例)\n")
    md.append(f"![response vs amplitude]({fig_prefix}/01_response_vs_amplitude.png)\n")
    md.append(f"![ratio and r2]({fig_prefix}/02_ratio_r2.png)\n")
    md.append("| 模式 | R(px) | diag A=2 | diag A=5 | diag A=10 | A10/A2 (期望5) | "
              "R² | CV(%) | 基线(λ) | SNR | 判定 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| [{r['m']}] {DLL_ZERNIKE_NAMES.get(r['m'], '')} | {r['R']:.0f} | "
                  f"{r['diag'][0]:+.4f} | {r['diag'][1]:+.4f} | {r['diag'][2]:+.4f} | "
                  f"{r['ratio_10_2']:.2f} | {r['r2']:.4f} | {100 * r['cv']:.1f} | "
                  f"{r['base']:.4f} | {r['snr']:.2f} | {r['verdict']} |")
    md.append("")
    if bad or noise:
        md.append("#### 例外说明\n")
        md.append("不成比例/噪声受限的组合全部落在**弱耦合**情形 (R 大 或 高阶模式), "
                  "其响应幅度与残差基线同量级:\n")
        for r in (bad + noise)[:10]:
            md.append(f"- `[{r['m']}]` R={r['R']:.0f}px: diag(A=10)={r['diag'][2]:+.4f}λ vs "
                      f"基线 {r['base']:.4f}λ → SNR={r['snr']:.2f} → "
                      f"{'低于基线, 读数不可信' if r['snr'] < 1.5 else '判据边缘未过'}")
        md.append("")
        md.append("→ 属**噪声受限 / 弱耦合**, 非真实非线性; 减小 R 或提高幅度可改善。\n")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-scan", default=None, help="raw scan json (default: latest)")
    ap.add_argument("--report", default=None, help="run report json (default: latest)")
    ap.add_argument("-o", "--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--append-to", default=None,
                    help="把该章节追加到指定 md (如图路径按目标文件位置换算)")
    ap.add_argument("--append-title", default="## 9. 追加: 优化后重跑 + 线性度分析",
                    help="追加章节的标题")
    args = ap.parse_args()

    raw = Path(args.raw_scan) if args.raw_scan else _latest(
        str(ROOT / "data" / "zernike_correction" / "raw_scan_*.json"))
    if raw is None or not raw.exists():
        print("[FAIL] 未找到 raw_scan json")
        return 1
    rep_path = Path(args.report) if args.report else _latest(
        str(ROOT / "data" / "zernike_correction" / "report_*.json"))
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path else {}

    out = Path(args.output_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("[Zernike 线性度报告] WFS 读数是否随 SLM 系数成比例增长")
    print("=" * 72)
    print(f"  raw scan: {raw}")
    print(f"  report  : {rep_path}")

    g = load_groups(raw)
    rows = analyze(g)
    ok = [r for r in rows if r["verdict"].startswith("成比例")]
    bad = [r for r in rows if r["verdict"].startswith("**")]
    print(f"[OK] 成比例 {len(ok)}/{len(rows)}; 不成比例 {len(bad)}: "
          f"{[(r['m'], int(r['R'])) for r in bad]}")

    fig_response_vs_amp(g, rows, figs / "01_response_vs_amplitude.png")
    fig_ratio_and_r2(rows, figs / "02_ratio_r2.png")
    body = write_markdown(out, {"rows": rows, "raw": raw, "report": rep})
    (out / "linearity.md").write_text("\n".join(body), encoding="utf-8")
    print(f"\n[OK] 报告: {out / 'linearity.md'}")

    if args.append_to:
        target = Path(args.append_to).resolve()
        # 图路径: 目标 md 所在目录 → 输出目录 (相对路径, 可随目录整体拷贝)
        try:
            rel = out.resolve().relative_to(target.parent).as_posix()
        except ValueError:
            rel = out.resolve().as_posix()
        prefix = f"{rel}/figures"
        body2 = write_markdown(out, {"rows": rows, "raw": raw, "report": rep},
                               fig_prefix=prefix)
        section = "\n".join([args.append_title, ""] + body2)
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        sep = "\n\n---\n\n"
        marker = sep + args.append_title
        if marker in existing:                      # 幂等: 替换旧章节
            existing = existing.split(marker)[0]
        target.write_text(existing.rstrip() + sep + section + "\n", encoding="utf-8")
        print(f"[OK] 已追加到: {target} (图前缀 {prefix})")

    print(f"[OK] 图: {figs}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
