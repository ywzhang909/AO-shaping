"""Offline report generator for the ``slm-pib`` rms_pib hardware matrix.

Reads the debug artifacts written by ``slm_pib_runner --debug`` (the HDF5
``/scalars`` columns plus the per-run summary PNG sketch) for every
``data/debug/slm_pib_rms_pib_*`` run and produces:

* ``docs/slm_pib_rms_pib_hw/figures/matrix_best_curves.png`` — per-algorithm
  ``best_rms_pib`` evolution (the tracked historical best, which stays finite
  even when the energy guard rejects an epoch's evaluation);
* ``docs/slm_pib_rms_pib_hw/figures/summary_bars.png`` — final best rms_pib
  per algorithm, sorted descending;
* ``docs/slm_pib_rms_pib_hw/figures/run_<algo>_<stamp>.png`` — copies of the
  runner's summary sketch (initial/best spot + target box, objective curve,
  best Zernike coefficients);
* ``docs/slm_pib_rms_pib_hw/report.md`` — the markdown report with a
  comparison table (best rms_pib / guard-rejected rows / dynamic weights /
  exposure) and per-run sections.

**Fully offline**: it never opens hardware, it only reads the saved HDF5
scalars and copies the saved PNG sketches (per repo convention, report
generation lives in ``scripts/`` and is offline — see ``scripts/README.md``).

The energy guard: when an evaluation perturbs the far field so roughly that
the ROI energy drops by more than ``max_roi_energy_loss`` (default 0.6), the
optimizer abandons that sample and writes the sentinel ``J = rms_pib = -999.x``
into the record. ``best_rms_pib`` is never touched by the guard, which is why
the curves in this report use it.

Usage:
    python scripts/generate_slm_pib_rms_pib_report.py
    python scripts/generate_slm_pib_rms_pib_report.py --debug-root data/debug --max-runs 5
    python scripts/generate_slm_pib_rms_pib_report.py -o docs/slm_pib_rms_pib_hw
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT = ROOT / "docs" / "slm_pib_rms_pib_hw"
# Sentinel written by the energy guard (see module docstring).
GUARD_FLOOR = -100.0
# Epoch curves only count the epochs actually collected for a run's algorithm.
ALGO_LABEL: dict[str, str] = {
    "spgd": "SPGD",
    "ga": "GA",
    "pso": "PSO",
    "de": "DE",
    "sa": "SA",
    "hc": "HC",
    "rs": "RS",
    "cem": "CEM",
}


# --- artifact loading --------------------------------------------------------


@dataclass
class Run:
    dir: Path
    stamp: str
    algo: str
    epochs: int  # requested epochs (sidecar), informational only
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    summary_png: Path | None = None
    data_rows: int = 0  # actual record count
    guard_count: int = 0  # records where J <= GUARD_FLOOR


def find_runs(debug_root: Path) -> list[Path]:
    """Every ``slm_pib_rms_pib_*/<stamp>`` artifact dir, newest first."""
    dirs = sorted(
        (p for p in debug_root.glob("slm_pib_rms_pib_*/*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return dirs


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def load_run(art_dir: Path) -> Run | None:
    """Load one run's HDF5 scalar columns + sidecar JSON.

    Returns None when the artifact is incomplete (no h5 / no complete scalar
    columns), so a half-written run does not break the whole report.
    """
    stamp = art_dir.name
    h5 = next(iter(art_dir.glob("*.h5")), None)
    if h5 is None:
        return None
    # sidecar JSON: {"algorithm": ..., "epochs": ...}
    algo, req_epochs = "unknown", 0
    jf = next(iter(art_dir.glob("*.json")), None)
    if jf is not None:
        try:
            payload = json.loads(jf.read_text())
        except json.JSONDecodeError:
            payload = {}
        algo = str(payload.get("algorithm", algo))
        req_epochs = int(payload.get("epochs", 0) or 0)

    scalar_cols: dict[str, np.ndarray] = {}
    try:
        with h5py.File(h5, "r") as f:
            sc = f["/scalars"]
            for key in sc.keys():
                arr = np.asarray(sc[key][:])
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                scalar_cols[key] = arr
    except (OSError, KeyError, ValueError) as exc:
        print(f"  [warn] cannot read h5 {h5}: {exc}")
        return None

    J = scalar_cols.get("J")
    if J is None or J.size == 0:
        return None
    guard_count = int(np.sum(np.asarray(J, dtype=float) <= GUARD_FLOOR))
    run = Run(
        dir=art_dir,
        stamp=stamp,
        algo=algo,
        epochs=req_epochs,
        scalars=scalar_cols,
        data_rows=int(J.size),
        guard_count=guard_count,
    )
    png = next(iter(art_dir.glob("*.png")), None)
    if png is not None:
        run.summary_png = png
    return run


def sc(run: Run, key: str) -> np.ndarray:
    """Scalar column as float array (empty when absent)."""
    v = run.scalars.get(key)
    if v is None:
        return np.array([])
    return np.asarray(v, dtype=float)


def best_epoch(run: Run, key: str = "best_rms_pib") -> int | None:
    """Epoch index (into the record array) of the max of ``key``."""
    v = sc(run, key)
    if v.size == 0:
        return None
    return int(np.nanargmax(v)) if not np.all(np.isnan(v)) else None


def best_value(run: Run, key: str = "best_rms_pib") -> float:
    v = sc(run, key)
    if v.size == 0:
        return float("nan")
    return float(np.nanmax(v))


def last_good(run: Run, key: str = "rms_pib") -> float:
    """Last accepted (non-guard) value of ``key``, NaN when none."""
    v = sc(run, key)
    if v.size == 0:
        return float("nan")
    good = v[v > GUARD_FLOOR]
    if good.size == 0:
        return float("nan")
    return float(good[-1])


# --- figure rendering --------------------------------------------------------


def plot_matrix_curves(runs: list[Run], fig_dir: Path) -> Path:
    """Overlaid ``best_rms_pib`` evolution per algorithm (always finite)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    order = sorted(runs, key=lambda r: best_value(r), reverse=True)
    for run in order:
        b = sc(run, "best_rms_pib")
        if b.size == 0:
            continue
        x = np.arange(b.size)
        label = ALGO_LABEL.get(run.algo, run.algo)
        ax.plot(x, b, "-", lw=1.6, ms=3, label=f"{label} ({best_value(run):.3f})")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_title("best_rms_pib 演进 (历史最优, 不受能量守卫哨兵影响)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("best rms_pib (桶内功率比)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out = fig_dir / "matrix_best_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_summary_bars(runs: list[Run], fig_dir: Path) -> Path:
    """Final best rms_pib per algorithm, sorted descending."""
    items = sorted(
        ((ALGO_LABEL.get(r.algo, r.algo), best_value(r)) for r in runs if sc(r, "best_rms_pib").size),
        key=lambda t: t[1],
    )
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * max(len(items), 1) + 1.6))
    labels = [t[0] for t in items]
    vals = [t[1] for t in items]
    ax.barh(labels, vals, color="tab:blue", alpha=0.85)
    for y, v in enumerate(vals):
        ax.text(v + 0.004, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18 + 0.02 if vals else 1.0)
    ax.set_xlabel("best rms_pib")
    ax.set_title("各算法历史最优 rms_pib")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "summary_bars.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --- markdown ----------------------------------------------------------------


def rel(p: Path, out_dir: Path) -> str:
    try:
        return str(p.relative_to(out_dir)).replace("\\", "/")
    except ValueError:
        return str(p)


def _fmt(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if v == v else "—"


def build_markdown(runs: list[Run], figs: dict[str, Path], out_dir: Path, out_md: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# slm-pib rms_pib 硬件矩阵报告 (target shape 整形, 能量约束)")
    lines.append("")
    lines.append(f"**生成时间**: {now}")
    lines.append("")
    lines.append("**Fully offline** — 本报告由 `scripts/generate_slm_pib_rms_pib_report.py` 离线生成, "
                 "仅读取 `slm_pib_runner --debug` 保存的 HDF5 `/scalars` 列与汇总图, 不打开任何硬件。")
    lines.append("")
    lines.append("## 1. 运行说明")
    lines.append("")
    lines.append("本报告覆盖 `data/debug/slm_pib_rms_pib_*/<stamp>` 中最新的 "
                 f"{len(runs)} 次硬件运行 (`slm-pib heuristic --objective rms_pib`, "
                 "Santec SLM-200 + Daheng CCD)。目标为方形目标框 (`--target-shape rectangle`, "
                 "`--target-size 64`, `--target-aspect-ratio 1.0`), 优化目标 `rms_pib` = "
                 "目标框内桶中功率与总功率之比, 并叠加 3 项动态权重目标: "
                 "`w_pib`·pib_term + `w_rms`·rms_term + `w_ee`·ee_term (权重随过程自适应)。")
    lines.append("")
    lines.append("> **注意**: 本次矩阵以当时的默认 `--n-max 4`(15 个 Zernike 模式)运行。"
                 "用户实测 **Zernike 阶数选到 10(66 个模式)整形效果明显更好**, "
                 "`slm_pib_runner` / `slm_zernike_pib` 的默认值已改为 10。后续硬件验证可按新默认复跑。")
    lines.append("")
    lines.append("### 能量守卫 (哨兵)")
    lines.append("")
    lines.append("当一次评估的扰动使 ROI 能量损失超过 `max_roi_energy_loss`(默认 0.6)时, 优化器放弃该样本, "
                 "记录写入哨兵值 `J = rms_pib = -999.x`(见 `slm_zernike_pib.py` 守卫段)。"
                 "`best_rms_pib` 是逐代跟踪的历史最优, 不受守卫影响, 因此下文的曲线与最优值均取自它。"
                 "DE/SA 的大扰动相位常触发守卫(见下表 `guard` 列), 其 `J` 曲线的 -999 悬崖在 runner Final "
                 "日志与汇总图曲线中已被剔除(> -100 过滤)。")
    lines.append("")
    lines.append("## 2. 算法对比")
    lines.append("")
    lines.append("| 算法 | epochs | 守卫行数 | best rms_pib | 最优代 | 末次有效 J | w_ee | w_pib | w_rms | ee_term | exp_t/brt |")
    lines.append("|------|--------|----------|--------------|--------|-----------|------|-------|-------|---------|-----------|")
    for r in runs:
        be = best_epoch(r, "best_rms_pib")
        algo = ALGO_LABEL.get(r.algo, r.algo)
        lines.append(
            "| {} | {} | {}/{} | {:.4f} | {} | {} | {} | {} | {} | {} | {}ms/{} |".format(
                algo,
                r.epochs,
                r.guard_count,
                r.data_rows,
                best_value(r),
                be,
                _fmt(last_good(r, "J")),
                _fmt(sc(r, "w_ee")[be], 3) if be is not None and sc(r, "w_ee").size else "—",
                _fmt(sc(r, "w_pib")[be], 3) if be is not None and sc(r, "w_pib").size else "—",
                _fmt(sc(r, "w_rms")[be], 3) if be is not None and sc(r, "w_rms").size else "—",
                _fmt(sc(r, "ee_term")[be], 3) if be is not None and sc(r, "ee_term").size else "—",
                _fmt(sc(r, "exp_t")[be], 1) if be is not None and sc(r, "exp_t").size else "—",
                int(sc(r, "max_brt")[be]) if be is not None and sc(r, "max_brt").size else "—",
            )
        )
    lines.append("")
    lines.append("![best_rms_pib 演进]({})".format(rel(figs["matrix_curves"], out_dir)))
    lines.append("")
    lines.append("![汇总]({})".format(rel(figs["summary_bars"], out_dir)))
    lines.append("")
    lines.append("## 3. 逐次运行")
    lines.append("")
    for i, r in enumerate(runs, 1):
        be = best_epoch(r, "best_rms_pib")
        algo = ALGO_LABEL.get(r.algo, r.algo)
        lines.append(f"### 3.{i} {algo} (`{r.stamp}`, 请求 {r.epochs} epochs, 实际 {r.data_rows} 行)")
        lines.append("")
        lines.append(f"- 历史最优 best_rms_pib = **{best_value(r):.4f}** (第 {be} 代)" if be is not None
                     else f"- 历史最优 best_rms_pib = n/a")
        lines.append(f"- 末次有效 J = {_fmt(last_good(r, 'J'))}")
        if be is not None and sc(r, "w_ee").size:
            lines.append(f"- 最优代权重: w_ee={_fmt(sc(r, 'w_ee')[be], 3)}, "
                         f"w_pib={_fmt(sc(r, 'w_pib')[be], 3)}, "
                         f"w_rms={_fmt(sc(r, 'w_rms')[be], 3)}; "
                         f"ee_term={_fmt(sc(r, 'ee_term')[be], 3)}")
        if r.summary_png is not None and figs.get("run_pngs", {}).get(r.stamp):
            lines.append("")
            lines.append("![{} 汇总图]({})".format(algo, rel(figs["run_pngs"][r.stamp], out_dir)))
        lines.append("")
    lines.append("## 4. 结论")
    lines.append("")
    best_run = max(runs, key=best_value) if runs else None
    if best_run is not None:
        lines.append(f"- 本矩阵中 **{ALGO_LABEL.get(best_run.algo, best_run.algo)}** 历史最优 rms_pib 最高 "
                     f"({best_value(best_run):.4f})。")
    for r in runs:
        if r.guard_count / max(r.data_rows, 1) > 0.5:
            lines.append(f"- {ALGO_LABEL.get(r.algo, r.algo)} 的评估 {r.guard_count}/{r.data_rows} 行被能量守卫拒绝 — "
                         "大扰动相位反复击穿 ROI 能量约束, 该算法的 '最优' 更像一次幸运的早期样本, 结论不可靠; "
                         "若需稳态解应调小扰动幅度或约束其余弦方向。")
    lines.append("- 守护值说明: 曲线用 `best_rms_pib`(逐代历史最优)绘制, 天然规避哨兵 -999 对曲线的压扁。")
    lines.append("")
    lines.append("## 5. 复现")
    lines.append("")
    lines.append("```powershell")
    lines.append('$env:PYTHONPATH = "src;libs"')
    lines.append("python scripts/generate_slm_pib_rms_pib_report.py")
    lines.append("```")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


# --- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--debug-root", type=Path, default=ROOT / "data" / "debug",
                    help="root dir containing slm_pib_rms_pib_* artifact dirs")
    ap.add_argument("--max-runs", type=int, default=5,
                    help="how many runs (newest first) to render [default 5 = the latest matrix]")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                    help="output dir for figures + report.md")
    args = ap.parse_args()

    art_dirs = find_runs(args.debug_root)
    if not art_dirs:
        print(f"no slm_pib_rms_pib_* artifact dirs under {args.debug_root}")
        raise SystemExit(1)
    art_dirs = art_dirs[: args.max_runs]

    runs: list[Run] = []
    for d in art_dirs:
        print(f"loading {d.name} ...")
        run = load_run(d)
        if run is None:
            continue
        runs.append(run)
    if not runs:
        print("no loadable runs")
        raise SystemExit(1)

    out_dir = args.output
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figs: dict[str, object] = {}
    figs["matrix_curves"] = plot_matrix_curves(runs, fig_dir)
    figs["summary_bars"] = plot_summary_bars(runs, fig_dir)
    run_pngs: dict[str, Path] = {}
    for r in runs:
        if r.summary_png is None:
            continue
        dest = fig_dir / f"run_{r.algo}_{r.stamp}.png"
        shutil.copyfile(r.summary_png, dest)
        run_pngs[r.stamp] = dest
    figs["run_pngs"] = run_pngs

    out_md = out_dir / "report.md"
    build_markdown(runs, figs, out_dir, out_md)
    print(f"report written: {out_md}")
    for r in runs:
        print(f"  {r.algo:6s} {r.stamp}  best={best_value(r):.4f}  "
              f"guard={r.guard_count}/{r.data_rows}")


if __name__ == "__main__":
    main()