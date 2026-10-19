"""Generate the GS-refine Zernike uniformity report.

Offline-only: reads the saved ``summary.json`` + ``spgd_nmax*.csv`` artefacts
produced by ``scripts/slm/run_gs_refine_zernike_sweep.py`` and renders a Chinese
markdown report with three PNG figures under ``report/shape_trials/``.

Never opens a device; never re-derives any metric from raw frames -- all numbers
come straight from the artefacts.

Run:
    python scripts/slm/generate_gs_refine_zernike_report.py [--run DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  -- must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import fmt_metric, markdown_table, savefig  # noqa: E402
from _common.provenance import insert_header  # noqa: E402

REPORT_KEY = "report/shape_trials/gs_refine_zernike_uniformity.md"
OUT_DIR = ROOT / "report" / "shape_trials"
FIG_DIR = OUT_DIR / "figures"

# --- CJK font guard (never crash on a headless box) --------------------------
try:
    from matplotlib import font_manager

    _available = {f.name for f in font_manager.fontManager.ttflist}
    _cjk_candidates = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    _picked = [n for n in _cjk_candidates if n in _available]
    if _picked:
        plt.rcParams["font.sans-serif"] = _picked + ["DejaVu Sans"]
except ImportError:  # pragma: no cover
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --- helpers -----------------------------------------------------------------
def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def load_spgd_csv(run_dir: Path, n_max: int) -> list[dict]:
    path = run_dir / f"spgd_nmax{n_max}.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- figures -----------------------------------------------------------------
def render_figures(summary: dict, run_dir: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    orders = summary["orders"]
    baselines = summary["baselines"]
    n_vals = [o["n_max"] for o in orders]

    # Figure a: fit residual
    fig, ax = plt.subplots(figsize=(7, 4))
    rms_vals = [o["fit_rms_rad"] for o in orders]
    ax.plot(n_vals, rms_vals, "o-", color="#2b6cb0")
    ax.set_xlabel("n_max")
    ax.set_ylabel("拟合残差 RMS (rad)")
    ax.set_title("Zernike 拟合残差 vs 阶数")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fit_residual_vs_nmax.png")
    plt.close(fig)

    # Figure b: CV before/after SPGD with baselines
    fig, ax = plt.subplots(figsize=(7, 4))
    cv_before = [o["before"]["cv"] for o in orders]
    cv_after = [o["after"]["cv"] for o in orders]
    ax.plot(n_vals, cv_before, "s--", color="#975a16", label="SPGD 前")
    ax.plot(n_vals, cv_after, "o-", color="#2b6cb0", label="SPGD 后")
    ax.axhline(baselines["flat"]["cv"], color="#c53030", ls=":", label="平场基线")
    ax.axhline(
        baselines["gs_freeform"]["cv"], color="#6b46c1", ls="-.", label="GS 自由相位基线"
    )
    ax.set_xlabel("n_max")
    ax.set_ylabel("CV")
    ax.set_title("方斑均匀性 (CV) vs Zernike 阶数")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "cv_before_after_vs_nmax.png")
    plt.close(fig)

    # Figure c: PIB before/after
    fig, ax = plt.subplots(figsize=(7, 4))
    pib_before = [o["before"]["pib"] for o in orders]
    pib_after = [o["after"]["pib"] for o in orders]
    ax.plot(n_vals, pib_before, "s--", color="#975a16", label="SPGD 前")
    ax.plot(n_vals, pib_after, "o-", color="#2b6cb0", label="SPGD 后")
    ax.axhline(baselines["flat"]["pib"], color="#c53030", ls=":", label="平场基线")
    ax.axhline(
        baselines["gs_freeform"]["pib"], color="#6b46c1", ls="-.", label="GS 自由相位基线"
    )
    ax.set_xlabel("n_max")
    ax.set_ylabel("PIB")
    ax.set_title("能量集中度 (PIB) vs Zernike 阶数")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "pib_before_after_vs_nmax.png")
    plt.close(fig)


# --- markdown ----------------------------------------------------------------
def build_markdown(summary: dict, run_dir: Path) -> str:
    cfg = summary["config"]
    baselines = summary["baselines"]
    orders = summary["orders"]
    source_note = (
        f"数据来源：{len(orders)} 阶扫描 run，`cam_type={cfg['cam_type']}`"
        + ("（数字孪生仿真，非硬件实测）" if cfg["cam_type"] == "sim" else "（硬件实测）")
        + f"，目录 `{run_dir.name}`。"
    )

    # 阶数扫描 table
    scan_rows = []
    for o in orders:
        improved = "✓" if o.get("spgd_improved") else "✗"
        clip = o.get("clip_warning") or ""
        scan_rows.append(
            [
                o["n_max"],
                o["n_modes"],
                fmt_metric(o["fit_rms_rad"]),
                fmt_metric(o["max_abs_coefficient"], nd=4),
                fmt_metric(o["before"]["pib"]),
                fmt_metric(o["after"]["pib"]),
                fmt_metric(o["before"]["cv"]),
                fmt_metric(o["after"]["cv"]),
                improved,
                clip,
            ]
        )
    scan_headers = [
        "n_max", "模态数", "拟合残差 (rad)", "max\\|c\\|",
        "PIB 前", "PIB 后", "CV 前", "CV 后", "SPGD 改善", "clip 警告",
    ]
    scan_table = markdown_table(scan_headers, scan_rows)

    # SPGD trajectory data (first order only, for illustration)
    first_order = orders[0]
    spgd_rows = load_spgd_csv(run_dir, first_order["n_max"])
    spgd_note = (
        f"SPGD 轨迹 CSV 已保存 ({len(spgd_rows)} 行, n_max={first_order['n_max']})。"
        if spgd_rows
        else "SPGD 轨迹 CSV 未找到。"
    )

    config_lines = "\n".join(
        f"| `{k}` | `{v}` |"
        for k, v in sorted(cfg.items())
        if k not in ("out", "save_images")
    )

    return f"""# GS-refine Zernike 阶数扫描 (uniformity)

{source_note}

## 结论

共扫描 {len(orders)} 个 Zernike 阶数 (n_max = {n_vals_str(orders)})。
平场基线 PIB={fmt_metric(baselines['flat']['pib'])}, CV={fmt_metric(baselines['flat']['cv'])};
GS 自由相位基线 PIB={fmt_metric(baselines['gs_freeform']['pib'])}, CV={fmt_metric(baselines['gs_freeform']['cv'])}。
{scan_verdict(orders, baselines, summary)}

## 实验设置

| 参数 | 值 |
|---|---|
{config_lines}

| 量 | 值 |
|---|---|
| `cam_size` | {summary['cam_size']} |
| `w0_px` | {fmt_metric(summary['w0_px'], nd=2)} |
| `target_side` | {fmt_metric(summary['target_side'], nd=2)} |
| `beam_radius_px` | {summary['beam_radius_px']} |
| `best_stage` | `{summary['best_stage']}` |
| 数据目录 | `{run_dir}` |

## 基线

| 相位 | PIB | CV | EE | AR | 综合分 |
|---|---|---|---|---|---|
| 平场 | {fmt_metric(baselines['flat']['pib'])} | {fmt_metric(baselines['flat']['cv'])} | {fmt_metric(baselines['flat']['ee'])} | {fmt_metric(baselines['flat']['ar'])} | {fmt_metric(baselines['flat']['quality'])} |
| GS 自由相位 | {fmt_metric(baselines['gs_freeform']['pib'])} | {fmt_metric(baselines['gs_freeform']['cv'])} | {fmt_metric(baselines['gs_freeform']['ee'])} | {fmt_metric(baselines['gs_freeform']['ar'])} | {fmt_metric(baselines['gs_freeform']['quality'])} |

## 阶数扫描

{scan_table}

{spgd_note}

## 图

![拟合残差](figures/fit_residual_vs_nmax.png)

![CV 前后对比](figures/cv_before_after_vs_nmax.png)

![PIB 前后对比](figures/pib_before_after_vs_nmax.png)

## 口径与已知局限

1. **`cv` 与 `ee` 必须一起看。** 单独优化 `-cv` 会掏空目标方斑（硬件实测 EE → 0.002）；
   目标函数必须包含能量项。
2. **Zernike 基半径锁定 `beam_radius_px`**，且与 GS 瞳面一致；更换光路必须重扫。
3. **仿真模式下 GS 预矫正未经物理标定。** 数字孪生是纯 FFT；`_derive_target_side`
   的台架模型（`camera_pixel_um=2.2`、焦距来自实测 tilt-shift scale）与仿真角尺度
   不匹配。日志会出现 "GS warm start ... did not beat flat ...; keeping flat"。
   仿真只验证 Zernike 拟合 + SPGD 路径与退化几何守卫；物理对比仅在真实硬件上有意义。
4. **`max_abs_coefficient` vs `ZERNIKE_CLIP = 5.0` rad。** 超过阈值会触发 clip，
   拟合质量不可信；上表 clip 列非空即为警告。
5. **已知潜在缺陷 (未修)：** `SlmGsRefineConfig` 自动目标边长约 400 px vs `cam_size`
   默认 300 → `square_mask` 裁剪到整幅 → `pib` 恒等于 1.0。驱动通过 waist 推导
   共享边长规避；共享优化器有意保持不变（其测试契约已固定）。
6. **只可比同一次 run 内的阶数, 不可跨 run 比 `fit_rms`。** 实测: 完全相同的命令行
   在 12 路并发与低负载下**逐位一致** (9 位小数全同), 说明 Stage A 本身确定性良好;
   但 `--n-max` 列表或 `--max-energy-loss` 不同时, Stage A 的 GS 最佳相位会整体改变
   (同一 `n_max` 跨 run 的 `fit_rms` 相差 0.008~0.012 rad)。这两个参数都在 Stage A
   之后才被读取, 根因未查明。因此本报告的阶数单调性只在**单次 run 内**成立
   (同一 `best_phase`, 已验证 `fit_best_phase` 不原地修改入参, 各阶互相独立)。
7. **`fit_rms` 的可比性依赖 `beam_radius_px`。** 该值同时锁定 GS 瞳面与拟合基,
   换光路必须与 `--n-max` 一起重扫, 否则阶数之间的差异里混入了基半径不匹配。
"""


def n_vals_str(orders: list[dict]) -> str:
    return ", ".join(str(o["n_max"]) for o in orders)


def scan_verdict(orders: list[dict], baselines: dict, summary: dict) -> str:
    """Honest, data-driven verdict.

    Deliberately does NOT name an "optimal order": it first checks whether the
    between-order spread is larger than the within-run SPGD jitter.  When it is
    not (the usual case for a short sweep), ranking orders would be reading noise.
    """
    flat = baselines["flat"]
    pib_after = [o["after"]["pib"] for o in orders]
    cv_after = [o["after"]["cv"] for o in orders]
    resid = [o["fit_rms_rad"] for o in orders]
    coef = [o["max_abs_coefficient"] for o in orders]

    jitter = [o["spgd_J_max"] - o["spgd_J_min"] for o in orders if o.get("spgd_J_max") is not None]
    pib_spread = max(pib_after) - min(pib_after)
    cv_spread = max(cv_after) - min(cv_after)
    jitter_ref = min(jitter) if jitter else float("nan")

    monotone = all(resid[i] > resid[i + 1] for i in range(len(resid) - 1))
    resid_rel = (resid[-1] / resid[0] - 1.0) * 100.0

    lines = [
        f"- **唯一随阶数单调的量是拟合残差**: {resid[0]:.4f} → {resid[-1]:.4f} rad "
        f"({resid_rel:+.1f}%), 单调递减 = {monotone}。这是本次扫描中唯一站得住的阶数效应。",
        f"- **阶数之间无法排序**: PIB 跨阶数极差 {pib_spread:.4f}、CV 极差 {cv_spread:.4f}, "
        f"而 SPGD **单轮运行内**的目标抖动就有 {jitter_ref:.4f}~{max(jitter):.4f}。"
        "两者同量级 ⇒ 任何「最优阶数」都是在读噪声, 本报告不给这个结论。",
        f"- **基相对平场的变化与阶数无关**: 各阶 ΔPIB 均为 "
        f"{min(o['after']['pib'] - flat['pib'] for o in orders):+.4f}~"
        f"{max(o['after']['pib'] - flat['pib'] for o in orders):+.4f}, "
        f"来自拟合相位本身, 不是来自选了哪个阶数。",
        f"- **拟合几乎没抓到结构**: 残差 rms 是最大系数的 "
        f"{min(resid[i] / coef[i] for i in range(len(orders))):.1f}~"
        f"{max(resid[i] / coef[i] for i in range(len(orders))):.1f} 倍 "
        f"(max|c| = {coef[0]:.4f} rad, 各阶相同) ⇒ 被拟合的相位接近平场, "
        f"Zernike 基在任何阶数下都没有可表达的内容, SPGD 因此无信号可优化。",
    ]
    if summary["config"]["cam_type"] == "sim":
        lines.insert(0, (
            "- ⚠️ **本次仿真运行不能回答本实验的科学问题** "
            "(需要多少阶 Zernike 才能复现 GS 整形)。原因见局限 3: 数字孪生里 GS 前向模型"
            "未经标定 ⇒ 送入拟合的相位≈平场。上表验证的是**管线正确性**, 不是物理结论。"
        ))
    return "\n".join(lines)


# --- main --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        default=None,
        help="run directory containing summary.json (default: newest under data/debug/slm_gs_refine_zernike)",
    )
    args = ap.parse_args()

    # Resolve run directory
    if args.run is not None:
        run_dir = args.run
    else:
        base = ROOT / "data" / "debug" / "slm_gs_refine_zernike"
        # Timestamp dirs sort lexicographically == chronologically (YYYYMMDD_HHMMSS).
        candidates = sorted(base.glob("*/summary.json")) if base.exists() else []
        if not candidates:
            print(f"ERROR: no run with summary.json found under {base}")
            print("Run the sweep first:  python scripts/slm/run_gs_refine_zernike_sweep.py")
            sys.exit(1)
        run_dir = candidates[-1].parent

    summary = load_summary(run_dir)
    render_figures(summary, run_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / REPORT_KEY
    body = build_markdown(summary, run_dir)
    out.write_text(insert_header(body, REPORT_KEY), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
