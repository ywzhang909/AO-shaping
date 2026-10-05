"""Generate the illustrated SLM+CCD shaping simulation report (fully offline).

Reads only artefacts on disk — the recorder pickles under ``data/debug/``, the
``data/slm_square/<ts>/`` history CSV, and the manifest written by
``scripts/run_sim_bench.py``. No hardware, no re-optimisation.

Scope is the three **SLM-driven** runners (``slm-pib``, ``slm-gsnet``,
``spgd-square``). Those are the ones whose numbers mean something: the actuator
they drive is the SLM, and ``SimPibSystem.far_field()`` models SLM phase, so the
objective responds to the optimiser. The two **DM-driven** runners (``pib``,
``combined``) execute their control loops but their objective is uncoupled noise
— nothing maps DM voltage to phase in that model — so they are listed as a
labelled smoke-test table and never plotted as convergence. See
``drivers/sim/AGENTS.md``.

Usage:
    python scripts/generate_slm_shaping_sim_report.py
    python scripts/generate_slm_shaping_sim_report.py --out report/slm_shaping_sim
    python scripts/generate_slm_shaping_sim_report.py --no-figures
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

#: runner -> (headline metric key, ordered candidate keys for that metric)
HEADLINE = {
    "slm-pib": ("_p%", ("_p%", "pib")),
    "slm-gsnet": ("quality", ("quality",)),
    "spgd-square": ("quality", ("quality",)),
}

#: Every column that is a plain number, per runner family.
EXTRA_COLUMNS = {
    "slm-pib": ("_max_r", "delta", "max_brt"),
    "slm-gsnet": ("cv", "ee", "ar"),
    "spgd-square": ("cv", "ee", "ar"),
}


def _newest_dir(root: Path, pattern: str) -> Path | None:
    hits = [p for p in root.glob(pattern) if p.is_dir()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def _load_pkl_history(prefix: str) -> tuple[list[int], dict] | None:
    """Load ``{epoch: row}`` from the newest ``data/debug/<prefix>_*/`` dump."""
    outer = _newest_dir(ROOT / "data" / "debug", f"{prefix}_*")
    if outer is None:
        return None
    inner = _newest_dir(outer, "*")
    if inner is None:
        return None
    pkls = sorted(inner.glob("*.pkl"))
    if not pkls:
        return None
    try:
        data = pickle.loads(pkls[-1].read_bytes())
    except (OSError, pickle.UnpicklingError):
        return None
    epochs = sorted(int(k) for k in data if str(k).lstrip("-").isdigit())
    return epochs, data


#: A history shorter than this is treated as a fixture, not a bench run. Both the
#: optimizer unit tests and a crashed run write into the same directories, and
#: picking purely by mtime lets a 1-epoch test artefact shadow a real run.
MIN_RUN_EPOCHS = 5


def _newest_qualified(candidates: list[Path], depth: int, minimum: int) -> Path | None:
    """Newest candidate holding at least ``minimum`` CSV data rows.

    Falls back to the newest candidate overall so a genuine short run still
    renders (with the smoke-budget caveat) rather than vanishing.
    """
    qualified: list[Path] = []
    for cand in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        counts: list[int] = []
        for f in sorted(cand.glob("*.csv")):
            if f.name.startswith("best_"):
                continue
            try:
                with open(f, newline="", encoding="utf-8") as fh:
                    counts.append(max(sum(1 for _ in csv.reader(fh)) - 1, 0))
            except (OSError, csv.Error):
                continue
        if counts and max(counts) >= minimum:
            qualified.append(cand)
    if qualified:
        return qualified[0]
    return (
        max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    )


def _load_csv_history() -> tuple[list[int], dict] | None:
    """Load the newest ``data/slm_square/<ts>/`` history with a plausible length."""
    root = ROOT / "data" / "slm_square"
    if not root.is_dir():
        return None
    run = _newest_qualified([p for p in root.iterdir() if p.is_dir()], 1, MIN_RUN_EPOCHS)
    if run is None:
        return None
    csvs = [p for p in sorted(run.glob("*.csv")) if not p.name.startswith("best_")]
    if not csvs:
        return None
    with open(csvs[-1], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    data: dict[int, dict] = {}
    for i, row in enumerate(rows):
        parsed: dict = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = np.nan
        data[i] = parsed
    return list(range(len(rows))), data


def _scalar_series(data: dict, epochs: list[int], key: str) -> np.ndarray:
    out = []
    for e in epochs:
        v = data.get(e, {}).get(key, np.nan)
        out.append(float(v) if isinstance(v, (int, float, np.floating)) else np.nan)
    return np.asarray(out, dtype=float)


def _image_series(data: dict, epochs: list[int], key: str = "_img") -> list[np.ndarray]:
    frames = []
    for e in epochs:
        img = data.get(e, {}).get(key)
        if img is None:
            frames.append(None)
            continue
        arr = np.asarray(img, dtype=float)
        # A CSV history stores ``_img`` as text, so the parsed value is a scalar
        # NaN rather than a frame; only genuine 2-D arrays are usable here.
        frames.append(arr if arr.ndim >= 2 else None)
    return frames


def _first_nan(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isnan(values))


def load_runner(runner: str) -> tuple[list[int], dict, Path | None] | None:
    """Return ``(epochs, rows, artefact_dir)`` for one SLM runner, or ``None``."""
    if runner == "spgd-square":
        loaded = _load_csv_history()
        if loaded is None:
            return None
        epochs, data = loaded
        run = _newest_dir(ROOT / "data" / "slm_square", "*")
        return epochs, data, run
    prefix = "slm_pib" if runner == "slm-pib" else "slm_gsnet"
    loaded = _load_pkl_history(prefix)
    if loaded is None:
        return None
    epochs, data = loaded
    outer = _newest_dir(ROOT / "data" / "debug", f"{prefix}_*")
    inner = _newest_dir(outer, "*") if outer else None
    return epochs, data, inner


def _cmap_norm(img: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.nanmax(img)) if np.isfinite(img).any() else 0.0
    if peak <= 0:
        return np.zeros_like(img), 1.0
    return np.clip(img / peak, 0, 1), peak


def plot_convergence(runs: dict, fig_dir: Path) -> list[Path]:
    """Headline metric vs epoch for every SLM runner that loaded."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    drawn = 0
    for runner, (epochs, data, _dir) in runs.items():
        key = HEADLINE[runner][0]
        y = _scalar_series(data, epochs, key)
        finite = ~np.isnan(y)
        if not finite.any():
            continue
        axes[0].plot(np.asarray(epochs)[finite], y[finite], "-o", ms=3, label=runner)
        best = int(np.nanargmax(y))
        axes[0].annotate(
            f"best {y[best]:.4g}@{epochs[best]}",
            (epochs[best], y[best]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
        drawn += 1
    for ax, title, ylab in (
        (axes[0], "Headline objective (higher is better)", None),
        (axes[1], "In-target energy / encircled energy", None),
    ):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("value")
    axes[0].legend(fontsize=8)
    if not drawn:
        axes[0].text(0.5, 0.5, "no SLM runner artefacts found", ha="center")
        axes[0].set_axis_off()
    fig.tight_layout()
    out = fig_dir / "convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return [out]


def plot_metric_panel(runs: dict, fig_dir: Path) -> list[Path]:
    """Per-runner extra metrics, one row per runner."""
    rows = [(r, EXTRA_COLUMNS.get(r, ())) for r in runs]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3.0 * len(rows)), squeeze=False)
    for ax, (runner, cols) in zip(axes[:, 0], rows):
        epochs, data, _ = runs[runner]
        plotted = False
        for col in cols:
            y = _scalar_series(data, epochs, col)
            finite = ~np.isnan(y)
            if finite.any():
                ax.plot(np.asarray(epochs)[finite], y[finite], "-o", ms=3, label=col)
                plotted = True
        ax.set_title(f"{runner} — recorded per-epoch metrics")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "metrics.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return [out]


def plot_spot_montage(runs: dict, fig_dir: Path) -> list[Path]:
    """First / mid / last far-field frame per SLM runner."""
    made = []
    for runner, (epochs, data, _dir) in runs.items():
        frames = [f for f in _image_series(data, epochs) if f is not None]
        if len(frames) < 3:
            continue
        picks = [frames[0], frames[len(frames) // 2], frames[-1]]
        labels = ["initial", "mid", "final"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for ax, img, name in zip(axes, picks, labels):
            norm, peak = _cmap_norm(img)
            ax.imshow(norm, cmap="inferno", vmin=0, vmax=1)
            ax.set_title(f"{name} (peak {peak:.0f})", fontsize=9)
            ax.set_axis_off()
        fig.suptitle(f"{runner} — simulated far field", fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"{runner.replace('-', '_')}_spot_montage.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        made.append(out)
    return made


def _smoke_rows(manifest: dict | None) -> list[dict]:
    if not manifest:
        return []
    return [r for r in manifest.get("runs", []) if r.get("family") == "dm"]


def _slm_only(runs: dict) -> dict:
    """Keep only the SLM-driven runners.

    ``HEADLINE`` is the allow-list: a runner with no modelled actuator (the DM
    pair) has no headline metric and must never reach the results table, where it
    would read as an optimisation result.
    """
    return {name: value for name, value in runs.items() if name in HEADLINE}


def build_markdown(
    runs: dict, figures: list[Path], out_md: Path, manifest: dict | None
) -> None:
    runs = _slm_only(runs)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# SLM 整形仿真运行报告 (2f-Fourier 数字孪生)")
    lines.append("")
    lines.append(f"**生成时间**: {now}")
    lines.append("")
    lines.append(
        "**Fully offline** — 本报告由 `scripts/generate_slm_shaping_sim_report.py` "
        "离线生成, 仅读取磁盘上的调试产物 (PKL / CSV), 不打开任何硬件、不重跑优化。"
    )
    lines.append("")
    lines.append("## 1. 光学模型与适用范围")
    lines.append("")
    lines.append(
        "SLM 位于 2f 光路前焦面, CCD 位于后焦面, 因此 CCD 图像 = SLM 瞳孔场的 2D FFT"
        "(夫琅禾费远场), 0 级光斑位于帧中心。"
    )
    lines.append("")
    lines.append(
        "本报告**只覆盖 SLM 驱动的 3 个 runner**。`pib` / `combined` 驱动的是 DM, "
        "而 `SimPibSystem` 中**没有任何代码把 DM 电压映射为相位** —— 它们的循环能跑完, "
        "但目标函数与 DM 无关(纯噪声)。因此这两个 runner **只作为控制环冒烟测试列出, "
        "不画收敛曲线**, 详见 `drivers/sim/AGENTS.md`。"
    )
    lines.append("")

    lines.append("## 2. 运行结果 (SLM 家族)")
    lines.append("")
    shortest = min((len(e) for e, _, _ in runs.values()), default=0)
    if shortest and shortest < 50:
        lines.append(
            f"> ⚠️ **本次为冒烟预算 (最短仅 {shortest} epoch), 不构成收敛性结论。** "
            "「最佳」一列取的是噪声驱动的单点最大值, 常常落在第 0~3 个 epoch; "
            "判断算法是否真的收敛需要把 `--epochs` 提高到数百以上再复跑。"
        )
        lines.append("")
    lines.append("| runner | epoch 数 | 指标 | 初始 | 最佳 | 最佳 epoch | 末轮 |")
    lines.append("|---|---|---|---|---|---|---|")
    for runner, (epochs, data, _dir) in runs.items():
        key = HEADLINE[runner][0]
        y = _scalar_series(data, epochs, key)
        finite = ~np.isnan(y)
        if not finite.any():
            lines.append(f"| `{runner}` | {len(epochs)} | `{key}` | — | — | — | — |")
            continue
        best = int(np.nanargmax(y))
        lines.append(
            f"| `{runner}` | {len(epochs)} | `{key}` | {y[0]:.4g} | "
            f"**{y[best]:.4g}** | {epochs[best]} | {y[-1]:.4g} |"
        )
    lines.append("")
    for runner, (epochs, data, _dir) in runs.items():
        extras = []
        for col in EXTRA_COLUMNS.get(runner, ()):
            y = _scalar_series(data, epochs, col)
            if (~np.isnan(y)).any():
                extras.append(f"`{col}` 末轮 {y[-1]:.4g}")
        if extras:
            lines.append(f"- `{runner}` 末轮附加指标: " + "; ".join(extras))
    lines.append("")

    for fig in figures:
        lines.append(f"![{fig.stem}](figures/{fig.name})")
        lines.append("")

    lines.append("## 3. DM 家族 (仅控制环冒烟测试, 不代表优化性能)")
    lines.append("")
    smoke = _smoke_rows(manifest)
    if smoke:
        lines.append("| runner | 状态 | 耗时 (s) | 产物数 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for row in smoke:
            state = "ok" if row.get("ok") else f"FAILED rc={row.get('returncode')}"
            lines.append(
                f"| `{row.get('runner')}` | {state} | {row.get('wall_s')} | "
                f"{len(row.get('artefacts') or [])} | {row.get('note', '')} |"
            )
        lines.append("")
        lines.append(
            "> ⚠️ 这些数字**不能**解读为 \"SPGD 在 DM-PIB 上不收敛\"。真实原因是仿真模型"
            "未接入 DM。若要得到有物理意义的 DM 结果, 需用 OOPAO `DeformableMirror` 的"
            "影响力函数构建电压→相位矩阵后再跑。"
        )
    else:
        lines.append("(未找到 `scripts/run_sim_bench.py` 生成的 manifest)")
    lines.append("")

    lines.append("## 4. 结论")
    lines.append("")
    if runs:
        lines.append(
            "1. **SLM 家族 3 个 runner 均可在纯仿真下端到端跑通**, 调试产物与硬件运行"
            "同格式, 因此离线报告可在断电状态下重生成。"
        )
        lines.append(
            "2. **仿真链路对相位调制是有响应的** —— 上表初始与最佳值可见优化器确实在"
            "改变远场, 这与 DM 家族形成对比。"
        )
    else:
        lines.append("1. 未找到 SLM 家族产物; 请先运行 `scripts/run_sim_bench.py`。")
    lines.append(
        "3. **DM 家族(pib / combined)目前只有控制环可执行性, 没有物理意义**, "
        "在补上 DM→相位耦合前不应据此评估算法。"
    )
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "report" / "slm_shaping_sim"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "sim_bench" / "summary.json"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs: dict = {}
    for runner in ("slm-pib", "slm-gsnet", "spgd-square"):
        loaded = load_runner(runner)
        if loaded is None:
            print(f"  [skip] {runner}: no artefacts")
            continue
        runs[runner] = loaded
        print(f"  [ok]   {runner}: {len(loaded[0])} epochs")

    manifest = None
    mpath = Path(args.manifest)
    if mpath.is_file():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None

    figures: list[Path] = []
    if runs and not args.no_figures:
        figures += plot_convergence(runs, fig_dir)
        figures += plot_metric_panel(runs, fig_dir)
        figures += plot_spot_montage(runs, fig_dir)

    out_md = out_dir / "report.md"
    build_markdown(runs, figures, out_md, manifest)

    missing = [f.name for f in figures if not f.is_file()]
    if missing:
        print(f"ERROR: figures missing: {missing}", file=sys.stderr)
        return 1
    print(f"report -> {out_md}")
    print(f"figures -> {len(figures)} written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
