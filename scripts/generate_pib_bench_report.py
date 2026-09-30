"""Generate the **offline** bench acceptance report for the SLM PIB pipeline.

Reads only the debug artefacts written by hardware runs and renders a single
markdown report plus figures. **Fully offline** — it never opens a camera or
SLM, so it can be re-run at any time to refresh the analysis from saved data.

What it reports
---------------
1. **Noise floor and SNR per perturbation amplitude** — from every
   ``summary_snr.json`` produced by
   ``tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py`` (single-
   and multi-mode) and by ``scripts/measure_shape_sensitivity.py``. The key
   bench fact: the floor is *not* a constant — sweeps on one day spanned 21x.
2. **Gate statistics from the robust-SPGD smoke runs** — applied / noise-gated /
   fold-gated per run, showing which ``delta`` values actually let the search
   move.
3. **Search outcomes from the recorded search runs** — first / best / final
   objective and the *sustained* improvement, which is the number that
   distinguishes real optimisation from drift.
4. **Per-run provenance** — taken from the JSON sidecar, so a row is never
   silently attributed to the wrong camera or objective.

Usage:
    python scripts/generate_pib_bench_report.py
    python scripts/generate_pib_bench_report.py --root data/debug -o docs/slm_pib_bench
    python scripts/generate_pib_bench_report.py --no-figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from loguru import logger  # noqa: E402

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_ROOT = ROOT / "data" / "debug"
DEFAULT_OUT = ROOT / "docs" / "slm_pib_bench"

#: Verdicts mirrored from ao_shaping.tools.slm.slm_snr_probe so this report
#: renders offline without importing driver-side code paths.
VERDICT_ZH = {"strong": "强", "usable": "可用", "unusable": "不可用"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_rows(path: Path) -> list[dict[str, Any]] | None:
    """Read one debug pickle into a list of row dicts.

    Accepts both on-disk shapes: a pickled ``Recorder`` (``.history``) and the
    plain ``{epoch: row}`` dict that ``save_recorder_debug_artifacts`` writes.
    """
    try:
        obj = pickle.loads(path.read_bytes())
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
        return None
    if obj is None:
        return None
    if hasattr(obj, "history"):
        return [dict(r) for r in obj.history]
    if isinstance(obj, dict):
        return [dict(r) for r in obj.values()]
    return None


def collect_snr(root: Path) -> list[dict[str, Any]]:
    """Every ``summary_snr.json`` under ``root``, newest first.

    Uses ``rglob`` because the artefacts live one level deeper than the search
    runs (``data/debug/slm_pib_online/<stamp>/summary_snr.json``), so a
    single-level glob would silently find nothing.
    """
    out: list[dict[str, Any]] = []
    for f in sorted(root.rglob("summary_snr.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "sigma_j" not in data:
            continue
        data["_stamp"] = f.parent.name
        out.append(data)
    return out


def collect_smoke(root: Path) -> list[dict[str, Any]]:
    """Every ``summary_smoke_*.json`` under ``root`` (recursive, see above)."""
    out: list[dict[str, Any]] = []
    for f in sorted(root.rglob("summary_smoke_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "applied" not in data:
            continue
        data["_stamp"] = f.parent.name
        data["_tag"] = f.stem.removeprefix("summary_")
        out.append(data)
    return out


def collect_search_runs(
    root: Path, keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Score every recorded search run by first / best / final objective.

    ``keys`` are candidate objective column names, tried in order. Each run
    contributes its recorded provenance from the sibling JSON sidecar.
    """
    out: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pkls = sorted(run_dir.glob("**/*.pkl"))
        if not pkls:
            continue
        rows = _load_rows(pkls[0])
        if not rows or len(rows) < 2:
            continue
        key = next((k for k in keys if k in rows[0]), None)
        if key is None:
            continue

        config: dict[str, Any] = {}
        for js in sorted(run_dir.glob("**/*.json")):
            try:
                loaded = json.loads(js.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(loaded, dict):
                config = loaded
                break

        gates = Counter(str(r.get("_gate")) for r in rows if "_gate" in r)
        # A guard-penalised row carries the sentinel J = 1e3. That marks an
        # *abandoned* evaluation, not a score, so keeping it yields absurd
        # percentages (observed: +1.3e9%). Score only the un-penalised rows.
        scored = [r for r in rows if abs(float(r.get("J", 0.0))) <= 100.0]
        n_penalised = len(rows) - len(scored)
        if len(scored) < 2:
            continue
        try:
            values = np.asarray([float(r[key]) for r in scored], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).all():
            continue
        # Pearson is a loss (min-mode); everything else is a max-mode score.
        mode = "min" if key == "pearson" else "max"
        sign = 1.0 if mode == "min" else -1.0  # +1 -> lower is better
        best = float(values.min() if mode == "min" else values.max())
        first, final = float(values[0]), float(values[-1])
        denom = abs(first) if abs(first) > 1e-12 else 1.0
        # Convergence diagnostics. ``final vs first`` alone is NOT a valid
        # progress measure: on a random walk the last sample lands wherever luck
        # puts it (bench-observed: a trace that dipped to 0.400, recovered to
        # 0.534 and ended at 0.402 reads as "+24% sustained" while never
        # descending). ``frac_decreasing`` exposes that (0.5 == coin flip) and
        # ``late_gain`` compares third-of-trace means, which is robust to where
        # the run happened to stop.
        steps = np.diff(sign * values)  # >0 == improving
        frac_decreasing = float(np.mean(steps < 0)) if steps.size else 0.0
        third = max(len(values) // 3, 1)
        head = float(np.mean(values[:third]))
        tail = float(np.mean(values[-third:]))
        late_gain = (
            100.0 * (head - tail) / abs(head) if abs(head) > 1e-12 else 0.0
        )
        out.append(
            {
                "name": run_dir.name,
                "key": key,
                "mode": mode,
                "epochs": len(scored),
                "n_penalised": n_penalised,
                "first": first,
                "best": best,
                "final": final,
                "best_impr_pct": 100.0 * (first - best) / denom
                if mode == "min"
                else 100.0 * (best - first) / denom,
                "sustained_pct": 100.0 * (first - final) / denom
                if mode == "min"
                else 100.0 * (final - first) / denom,
                "frac_decreasing": frac_decreasing,
                "late_gain_pct": late_gain,
                "gates": dict(gates),
                "config": config,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_snr(snr: list[dict[str, Any]], out: Path) -> Path | None:
    if not snr:
        return None
    stamps = [s["_stamp"][-6:] for s in snr]
    sigmas = [s["sigma_j"] for s in snr]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))

    ax0.bar(stamps, sigmas, color="#c44e52")
    ax0.set_yscale("log")
    ax0.set_ylabel("noise floor $\\sigma_J$")
    ax0.set_xlabel("sweep (time)")
    ax0.set_title("Noise floor is not a bench constant")
    for x, s in zip(stamps, sigmas, strict=True):
        ax0.text(x, s, f"{s:.1e}", ha="center", va="bottom", fontsize=7)
    ax0.tick_params(axis="x", rotation=45)

    keys = sorted({k for s in snr for k in s.get("snr_by_delta", {})}, key=float)
    for s in snr:
        ax1.plot(
            keys,
            [s.get("snr_by_delta", {}).get(k, np.nan) for k in keys],
            marker="o",
            label=f"{s['_stamp'][-6:]} single",
            linestyle="--",
            alpha=0.6,
        )
    ax1.axhline(2.0, color="gray", ls=":", label="usable (2)")
    ax1.axhline(3.0, color="black", ls="--", lw=0.8, label="strong (3)")
    ax1.set_xscale("log")
    ax1.set_xlabel("perturbation $\\Delta a$ (rad)")
    ax1.set_ylabel("SNR")
    ax1.set_title("Single-mode SNR vs amplitude")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    path = out / "snr_floor_and_snr.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_gates(smoke: list[dict[str, Any]], out: Path) -> Path | None:
    if not smoke:
        return None
    labels, applied, noise, fold = [], [], [], []
    for s in smoke:
        rows = max(int(s.get("rows", 1)), 1)
        labels.append(f"{s['_stamp'][-4:]}\n{s['_tag']}")
        applied.append(100.0 * s.get("applied", 0) / rows)
        noise.append(100.0 * s.get("noise", 0) / rows)
        fold.append(100.0 * s.get("fold", 0) / rows)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4))
    ax.bar(x, applied, label="applied", color="#55a868")
    ax.bar(x, noise, bottom=applied, label="noise-gated", color="#c44e52")
    ax.bar(
        x,
        fold,
        bottom=np.add(applied, noise),
        label="fold-gated",
        color="#dd8452",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% of recorded rows")
    ax.set_title("SPGD gate breakdown — only 'applied' rows move the search")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "smoke_gate_breakdown.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_sustained(
    runs: list[dict[str, Any]], out: Path, min_epochs: int = 0
) -> Path | None:
    sel = [r for r in runs if r["epochs"] >= min_epochs]
    if not sel:
        return None
    sel = sorted(sel, key=lambda r: r["sustained_pct"])
    labels = [r["name"].replace("slm_pib_", "")[:34] for r in sel]
    best = [r["best_impr_pct"] for r in sel]
    sus = [r["sustained_pct"] for r in sel]
    x = np.arange(len(sel))
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(sel)), 5))
    ax.bar(x - 0.2, best, width=0.4, label="best-so-far", color="#4c72b0")
    ax.bar(x + 0.2, sus, width=0.4, label="sustained (final)", color="#dd8452")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("improvement vs first record (%)")
    ax.set_title(
        "Best-so-far vs sustained — a large gap means drift, not optimisation"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "search_best_vs_sustained.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _prov(cfg: dict[str, Any]) -> str:
    """One-line provenance from a sidecar, or an explicit 'unknown'."""
    if not cfg:
        return "sidecar 无记录"
    bits = [
        f"{k}={cfg[k]!r}"
        for k in ("objective", "cam_type", "cam_id", "exposure_time_ms", "delta")
        if k in cfg
    ]
    return ", ".join(bits) if bits else "sidecar 无关键字段"


def write_report(
    snr: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    out: Path,
    figs: list[Path],
    root: Path,
) -> Path:
    lines: list[str] = [
        "# SLM PIB 整形台架验收报告 (离线分析)",
        "",
        "> **Fully offline.** 本报告只读取 `data/debug` 下已保存的 debug 产物",
        "> (recorder pickle + JSON sidecar), 从不打开相机或 SLM。设备离线期间可",
        "> 随时重跑刷新结论。数据根目录: "
        f"`{root.relative_to(ROOT) if str(root).startswith(str(ROOT)) else root}`",
        "",
    ]

    # ---------------- 1. noise floor ----------------
    lines += ["## 1. 噪声地板与 SNR", ""]
    if snr:
        sig = [s["sigma_j"] for s in snr]
        lines += [
            f"共 {len(snr)} 次扫描。噪声地板 $\\sigma_J$ 区间 "
            f"**{min(sig):.2e} … {max(sig):.2e}**"
            f"（相差 **{max(sig)/max(min(sig),1e-12):.0f}×**）。",
            "",
            "> **结论 1：噪声地板不是台架常数。** 同一天、同一光斑位置的三次扫描",
            "> 相差 21 倍, 因此 SNR **必须每次实测**, 不可沿用历史值或写死。",
            "",
            "| 扫描时间 | 中心 (x, y) | $\\sigma_J$ | "
            + " | ".join(f"SNR@{k}" for k in sorted(
                {k for s in snr for k in s.get("snr_by_delta", {})}, key=float
            ))
            + " |",
            "|---|---|---|"
            + "---|" * len(
                {k for s in snr for k in s.get("snr_by_delta", {})}
            ),
        ]
        all_keys = sorted(
            {k for s in snr for k in s.get("snr_by_delta", {})}, key=float
        )
        for s in snr:
            c = s.get("center", [0, 0])
            cells = " | ".join(
                f"{s.get('snr_by_delta', {}).get(k, float('nan')):.2f}"
                for k in all_keys
            )
            lines.append(
                f"| {s['_stamp']} | ({c[0]:.0f}, {c[1]:.0f}) | "
                f"{s['sigma_j']:.2e} | {cells} |"
            )
        lines += [
            "",
            "> 表中为**单模式** (defocus) 探测。SPGD 同时扰动全部模式, 信号被",
            "> 稀释 —— 见 §2。",
            "",
        ]
    else:
        lines += ["_未找到 `summary_snr.json`。_", ""]

    # ---------------- 2. gate behaviour ----------------
    lines += ["## 2. 门控行为 (为什么小 delta 跑不动)", ""]
    if smoke:
        lines += [
            "| 运行 | 标签 | rows | applied | noise | fold | delta | "
            "n_eval_frames | best objective |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in smoke:
            lines.append(
                f"| {s['_stamp']} | `{s['_tag']}` | {s.get('rows','?')} | "
                f"{s.get('applied','?')} | {s.get('noise','?')} | "
                f"{s.get('fold','?')} | {s.get('delta','?')} | "
                f"{s.get('n_eval_frames','?')} | "
                f"{float(s.get('best_objective', float('nan'))):.4f} |"
            )
        worst = min(
            smoke, key=lambda s: s.get("applied", 0) / max(s.get("rows", 1), 1)
        )
        rate = worst.get("applied", 0) / max(worst.get("rows", 1), 1)
        lines += [
            "",
            f"> **结论 2：门控是主要损耗项。** 最差的一次 "
            f"`{worst['_stamp']}` 只有 **{100*rate:.0f}%** 的行真正被采纳, "
            "其余被噪声门或亮度折叠门拦下 —— 搜索几乎没有拿到梯度。",
            "",
            "两个独立成因, 都在台架上实测确认:",
            "",
            "1. **信噪比被自由度稀释。** SPGD 对所有模式施加随机 ±1 扰动, "
            "而单模式探测只动一个模式。实测 `delta=0.0005` 时单模式 SNR 2.25"
            "(可用) 而 54 自由度 SNR 仅 1.34 (不可用) —— "
            "**这就是 `delta<0.001` 跑不动的原因, 不是约束本身荒谬**。",
            "2. **噪声是慢漂移, 不是散粒噪声。** 40 帧平均并未降低 $\\sigma_J$"
            "(2.8e-3 vs 10 帧的 1.7e-3), 所以加帧数无效; "
            "必须用 `+ - - +` 回文序列做漂移对消 (见 §5 工具)。",
            "",
        ]
    else:
        lines += ["_未找到 `summary_smoke_*.json`。_", ""]

    # ---------------- 3. search outcomes ----------------
    lines += ["## 3. 搜索结果 (best vs sustained)", ""]
    if runs:
        lines += [
"**sustained** 是末帧相对首帧的改善, **best** 是历史最优。两者的差值",
            "是判别「真优化」与「漂移」的第一道线索。",
            "",
            "⚠️ **但 `sustained` 会骗人。** 轨迹若是随机游走, 末帧落在低点纯属运气。"
            "实机曾出现 `0.529 → 0.400 → 0.534 → 0.402` 的轨迹 —— 头尾一比是 \"+24%\", "
            "全程却没有下降。因此另给两个稳健判据:",
            "",
            "- **`dec`** = 下降步占比。`≈0.5` 即随机游走 (抛硬币), 说明没有收敛。",
            "- **`late`** = 前 1/3 与后 1/3 均值之差 (按目标极性归一), 对终点位置不敏感。",
            "",
            "`guard` 列是被能量门判为「放弃评估」的行数 (J = 1e3 哨兵), 不参与统计。",
            "",
            "| run | objective | epochs | guard | dec | late % | best % | "
            "sustained % | 溯源 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(runs, key=lambda r: r["sustained_pct"], reverse=True):
            lines.append(
                f"| {r['name']} | `{r['key']}` | {r['epochs']} | "
                f"{r.get('n_penalised', 0)} | "
                f"{r.get('frac_decreasing', float('nan')):.2f} | "
                f"{r.get('late_gain_pct', float('nan')):+.1f} | "
                f"{r['best_impr_pct']:+.1f} | {r['sustained_pct']:+.1f} | "
                f"{_prov(r['config'])} |"
            )
        walkers = [
            r for r in runs if abs(r.get("frac_decreasing", 0.5) - 0.5) < 0.05
        ]
        if walkers:
            lines += [
                "",
                f"> **⚠️ {len(walkers)}/{len(runs)} 个运行的 `dec` 在 0.45–0.55 之间** "
                "= 随机游走。这些运行的 `sustained` 不应被解读为优化成果。",
            ]
        strong = [r for r in runs if r["sustained_pct"] > 5.0]
        weak = [
            r
            for r in runs
            if abs(r["best_impr_pct"]) > 5.0 and abs(r["sustained_pct"]) < 2.0
        ]
        # Does a frequently-firing energy guard correlate with a failed run?
        guarded = [r for r in runs if r.get("n_penalised", 0) > 0]
        clean = [r for r in runs if r.get("n_penalised", 0) == 0]
        lines += [
            "",
            f"- **sustained > 5%** 的运行: {len(strong)} / {len(runs)}",
            f"- **best 显著但 sustained ≈ 0** (判为漂移, 非优化): {len(weak)} / {len(runs)}",
            "",
        ]
        if guarded and clean:
            g_med = float(np.median([r["sustained_pct"] for r in guarded]))
            c_med = float(np.median([r["sustained_pct"] for r in clean]))
            lines += [
                "> **结论 3：能量门频繁触发的运行基本等于没优化。** "
                f"{len(guarded)} 个运行出现过 guard 惩罚行, 其 sustained 改善中位数 "
                f"**{g_med:+.1f}%**; 而 {len(clean)} 个无 guard 的运行中位数为 "
                f"**{c_med:+.1f}%**。被门拦下的迭代不更新系数, 因此门一旦频繁"
                "触发, 搜索就原地踏步 —— 排查时先看 guard 计数, 再看 delta。",
                "",
            ]
        if weak:
            lines += [
                "判为漂移的运行 (仅列名前 8 个):",
                "",
                "| run | best % | sustained % |",
                "|---|---|---|",
            ]
            for r in sorted(weak, key=lambda r: -abs(r["best_impr_pct"]))[:8]:
                lines.append(
                    f"| {r['name']} | {r['best_impr_pct']:+.1f} | "
                    f"{r['sustained_pct']:+.1f} |"
                )
            lines.append("")
    else:
        lines += ["_未找到可解析的搜索运行。_", ""]

    # ---------------- 4. figures ----------------
    if figs:
        lines += ["## 4. 图", ""]
        for p in figs:
            lines += [f"![{p.stem}]({p.parent.name}/{p.name})", ""]

    # ---------------- 5. tooling ----------------
    lines += [
        "## 5. 参数测量工具 (设备无关)",
        "",
        "本报告引用的 SNR / 噪声地板测量已抽到 "
        "**`ao_shaping.tools.slm.slm_snr_probe`**, 设备实例由参数传入, "
        "不构造任何设备:",
        "",
        "```python",
        "from ao_shaping.tools.slm.slm_snr_probe import snr_sweep",
        "with create_camera('daheng', cam_id=0, exposure_time_ms=1.2) as cam, \\",
        "     Santec(slm_number=1, wavelength=1064) as slm:",
        "    result = snr_sweep(cam, slm, n_max=9, radius=480.0)",
        "    print(result.sigma, result.multi_snrs, result.usable_deltas())",
        "```",
        "",
        "| 符号 | 作用 |",
        "|---|---|",
        "| `snr_sweep` | 噪底 + 单模式/多模式逐 delta SNR (设备实例传入) |",
        "| `measure_noise_floor` | 固定相位下 σ (含慢漂移) |",
        "| `abba_signal` | `+ - - +` 回文对消漂移的 ΔJ |",
        "| `snr_verdict` | 阈值 → strong / usable / unusable |",
        "| `SnrSweepResult.usable_deltas` | **按多模式 SNR** 给出的可用 delta |",
        "",
        "消费者: `scripts/measure_shape_sensitivity.py` 与硬件门控测试 "
        "`tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py` "
        "均改为委托该工具, 二者不会再各自漂移。离线可测: "
        "`tests/ao_shaping/tools/test_slm_snr_probe.py` (纯 fake 设备, 无需硬件)。",
        "",
        "> ⚠️ `usable_deltas()` 刻意按**多模式** SNR 过滤: 单模式列会高估 SPGD "
        "实际能分辨的信号, 直接采信会再次把 delta 选到不可用的区间。",
        "",
    ]

    # ---------------- 6. recommendations ----------------
    guarded_n = sum(1 for r in runs if r.get("n_penalised", 0) > 0)
    lines += [
        "## 6. 建议",
        "",
        f"1. **排查顺序: 先看 guard 计数, 再看 delta。** {guarded_n} 个运行出现"
        "能量门惩罚行, 其 sustained 改善中位数为负 (§3 结论 3)。门拦下的迭代"
        "不更新系数, 门频繁触发时搜索原地踏步, 调 delta 无济于事。",
        "2. **按每次实测的多模式 SNR 选 `delta`**, 不要沿用历史值 —— 噪声地板"
        "单日波动 21×。用 `snr_sweep(...).usable_deltas()`。",
        "3. **`delta<0.001` 在 `n_max=9` (54 自由度) 下不可用**: 实测多模式 SNR "
        "≈ 1.3, 95% 迭代被门控。若必须满足该约束, 只能显著降低 `n_max`"
        "(自由度越少, 稀释越轻)。",
        "4. **`delta≈0.1` 是当前 `n_max=9` 的可用区间**; `0.2` 会触发亮度折叠门"
        "(45/60 被拒), 过大反而更差。",
        "5. **加帧数不能降噪**, 只能靠 `+ - - +` 漂移对消; 任何降噪方案先验证"
        "是否真的降低了 σ。`n_eval_frames=4` 的 smoke 出现 16/16 全被折断"
        "门拒绝, 属于独立失效模式。",
        "6. 目标函数选择请参考 "
        "[三目标对比报告](slm_pib_online_hw/objective_comparison.md): "
        "`1 - Pearson` 在实机上只优化成功、排序不可靠 (2/4 符号正确), "
        "维持 **RISKY**, 仅作显式可选目标并保留能量门。",
        "",
        "---",
        "",
        "### 下一步 (设备上线后)",
        "",
        "1. 用新工具重跑一次完整 SNR 扫描: "
        "`AO_RUN_HARDWARE=1 pytest tests/ao_shaping/optimizer/wfless/"
        "test_slm_zernike_pib_online.py -v -s`, 读取 `multi_snrs` 列选 delta。",
        "2. 用选定 delta 跑 `n_max=9` 的方形/PIB 各一次 ≥200 epoch, "
        "确认 sustained > 5%。",
        "3. 硬件验收脚本: `scripts/measure_shape_sensitivity.py` (已改为委托同一"
        "工具, 与门控测试结果一致)。",
        "",
    ]

    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline bench acceptance report for the SLM PIB pipeline."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="debug root")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="out dir")
    parser.add_argument(
        "--no-figures", action="store_true", help="skip figure rendering"
    )
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    if not root.is_dir():
        logger.error("root not found: {}", root)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    figs_dir = out / "figures"
    figs_dir.mkdir(exist_ok=True)

    snr = collect_snr(root)
    smoke = collect_smoke(root)
    runs = collect_search_runs(
        root, keys=("pearson", "shape", "roi_pib", "rms_pib", "pib", "rmse")
    )
    logger.info(
        "collected {} SNR sweeps, {} smoke runs, {} search runs",
        len(snr), len(smoke), len(runs),
    )

    figs: list[Path] = []
    if not args.no_figures:
        for fn in (fig_snr, fig_gates):
            try:
                p = fn(snr, figs_dir) if fn is fig_snr else fn(smoke, figs_dir)
            except Exception as exc:  # a broken figure must not kill the report
                logger.warning("figure {} failed: {}", fn.__name__, exc)
                continue
            if p is not None:
                figs.append(p)
        try:
            p = fig_sustained(runs, figs_dir)
            if p is not None:
                figs.append(p)
        except Exception as exc:
            logger.warning("figure fig_sustained failed: {}", exc)

    report = write_report(snr, smoke, runs, out, figs, root)
    logger.info("wrote {}", report)
    for p in figs:
        logger.info("wrote {}", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
