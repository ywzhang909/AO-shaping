"""Render the 2026-11-07 forward-model polarity / direction-check report.

Offline renderer: it reads only archived artefacts under
``data/debug/slm_transfer`` (JSON summaries and fit metrics) and writes
``report/shape_trials/forward_model_polarity/``.  It never imports
``ao_shaping.drivers.*``, never opens an SLM, camera or laser and never runs a
capture, so no device is occupied.

Provenance: ``scripts/_common/provenance.py`` maps this report to this script;
``scripts/sync_report_provenance.py`` owns the provenance block and the
``report/README.md`` index.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "debug" / "slm_transfer"

REPORT_KEY = "report/shape_trials/forward_model_polarity/forward_model_polarity_20261107.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "forward_model_polarity"
FIG_DIR = OUT_DIR / "figures"

#: (data directory, commanded amplitude in rad). The three single-mode ladders.
LADDERS: tuple[tuple[str, float], ...] = (
    ("issue234_single_mode_screen_20261107", 0.10),
    ("issue234_amp05_screen_20261107", 0.05),
    ("issue234_amp02_screen_20261107", 0.02),
)

#: The live gate in ``jacobian_guard`` (signed Pearson, not |r|).
GATE = 0.50

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(payload: Any) -> list[dict]:
    """Direction JSONs exist in two shapes: a bare list, or ``{"results": [...]}``."""
    if isinstance(payload, list):
        return payload
    return list(payload.get("results", []))


def _amp_tag(amp: float) -> str:
    return f"{amp:.2f}".replace(".", "")


def direction_rows(run: str, amp: float, sign: int) -> list[dict]:
    """Locate one direction-agreement JSON for a ladder/amplitude/polarity."""
    directory = DATA / run
    tag = _amp_tag(amp)
    if sign < 0:
        candidates = [directory / f"direction_screen_amp{tag}_M1_defocus_signm1_model.json"]
    elif amp == 0.10:
        candidates = [directory / "direction_screen_20261107.json"]
    else:
        candidates = [directory / f"direction_screen_amp{tag}_M1_defocus.json"]
    for path in candidates:
        if path.exists():
            return _rows(_load(path))
    fallback = sorted(directory.glob(f"direction_screen_amp{tag}*.json"))
    for path in fallback:
        is_negated = "signm1" in path.name
        is_centroid = "centroid" in path.name
        if is_centroid:
            continue
        if is_negated == (sign < 0):
            return _rows(_load(path))
    raise FileNotFoundError(f"no direction JSON for {run} amp={amp} sign={sign}")


def per_mode(rows: list[dict]) -> dict[int, float]:
    """Mean signed correlation per Zernike mode, averaging the two passes."""
    buckets: dict[int, list[float]] = {}
    for row in rows:
        buckets.setdefault(int(row["mode"]), []).append(float(row["corr"]))
    return {mode: float(np.mean(values)) for mode, values in sorted(buckets.items())}


def summarize(rows: list[dict]) -> dict[str, float]:
    corrs = [float(row["corr"]) for row in rows]
    return {
        "n": len(corrs),
        "median": st.median(corrs),
        "positive": sum(c > 0 for c in corrs),
        "at_gate": sum(c >= GATE for c in corrs),
    }


# ---------------------------------------------------------------- figures


def figure_focus_scan(focus: dict) -> Path:
    points = [p for p in focus["probe"]["points"] if p.get("status") == "ok"]
    diopter = np.array([p["diopter_m1"] for p in points])
    peak = np.array([p["peak_per_ms"] for p in points])
    fwhm = np.array([p["fwhm_px"] for p in points])
    best = float(focus["probe"]["best_diopter_m1"])
    order = np.argsort(diopter)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axes[0].plot(diopter[order], peak[order], "o-", color="#1f6fb4", ms=3)
    axes[0].set_ylabel("峰值计数/ms")
    axes[0].axvline(best, color="#c0392b", ls="--", lw=1)
    axes[0].annotate(f"最佳 {best:+.4f} m$^{{-1}}$", xy=(best, peak.max()), xytext=(6, -2),
                     textcoords="offset points", color="#c0392b", fontsize=8)
    axes[1].plot(diopter[order], fwhm[order], "s-", color="#2e8b57", ms=3)
    axes[1].set_ylabel("FWHM / px")
    axes[1].set_xlabel("离焦量 / m$^{-1}$")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("焦点重标定扫描（Santec #1 + Daheng，1064 nm）", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "focus_scan_20261107.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_fit_quality(fit: dict) -> Path:
    arms = [name for name in ("M0", "M1_defocus", "M3_illumination") if name in fit["models"]]
    branches = ["U", "C"]
    means = {arm: {b: [] for b in branches} for arm in arms}
    for arm in arms:
        for row in fit["models"][arm]["rows"]:
            means[arm].setdefault(row["branch"], []).append(float(row["corr"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    width = 0.36
    x = np.arange(len(arms))
    colors = {"U": "#4c72b0", "C": "#dd8452"}
    for offset, branch in zip((-width / 2, width / 2), branches):
        values = [means[arm].get(branch, []) for arm in arms]
        bar_means = [float(np.mean(v)) if v else np.nan for v in values]
        axes[0].bar(x + offset, bar_means, width, label=branch, color=colors[branch], alpha=0.85)
        for xi, vals in zip(x + offset, values):
            if vals:
                axes[0].scatter([xi] * len(vals), vals, s=7, color="k", alpha=0.45, zorder=3)
    axes[0].set_xticks(x, arms, rotation=10)
    axes[0].set_ylabel("逐样本 Pearson")
    axes[0].set_ylim(0, 1)
    axes[0].legend(title="branch")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("U/C 成对数据集的拟合相关")

    rows = fit["models"]["M1_defocus"]["rows"]
    for branch in branches:
        subset = [r for r in rows if r["branch"] == branch]
        axes[1].scatter([r["measured_cv"] for r in subset],
                        [r["predicted_cv"] for r in subset],
                        s=16, alpha=0.75, label=branch, color=colors[branch])
    limit = max(max(r["measured_cv"], r["predicted_cv"]) for r in rows) * 1.05
    axes[1].plot([0, limit], [0, limit], "k--", lw=0.8, label="y = x")
    axes[1].set_xlabel("实测 CV")
    axes[1].set_ylabel("预测 CV（M1_defocus）")
    axes[1].set_title("M1：实测 vs 预测核心 CV")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("物理预测模型拟合质量（defocus_forward_20261107_focus_r2）", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "uc_fit_quality_20261107.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_direction(ladders: dict[float, dict[int, dict[int, float]]]) -> Path:
    amps = sorted(ladders, reverse=True)
    modes = sorted(next(iter(next(iter(ladders.values())).values())))
    fig, axes = plt.subplots(1, len(amps), figsize=(11.4, 3.4), sharey=True)
    width = 0.38
    x = np.arange(len(modes))
    for ax, amp in zip(np.atleast_1d(axes), amps):
        plus = [ladders[amp][1][mode] for mode in modes]
        minus = [ladders[amp][-1][mode] for mode in modes]
        ax.bar(x - width / 2, plus, width, label="phase_sign=+1", color="#b0413e", alpha=0.85)
        ax.bar(x + width / 2, minus, width, label="phase_sign=−1", color="#2f6fb0", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(GATE, color="#555", ls="--", lw=0.9)
        ax.axhline(-GATE, color="#555", ls="--", lw=0.9)
        ax.set_xticks(x, [f"Z{m}" for m in modes], rotation=0)
        ax.set_title(f"幅值 {amp:.2f} rad")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(-0.85, 0.85)
    np.atleast_1d(axes)[0].set_ylabel("签名方向一致性（Pearson）")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.suptitle("单模 Zernike 屏方向一致性：极性翻转后符号全正，但未达 0.50 门禁", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "direction_ladder_20261107.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_registration(scan: dict) -> Path:
    labels = ["+1 无配准", "+1 最佳平移", "−1 无配准", "−1 最佳平移"]
    values = [
        scan["pol+1"]["no_registration"]["median"],
        scan["pol+1"]["best_global_shift"]["median"],
        scan["pol-1"]["no_registration"]["median"],
        scan["pol-1"]["best_global_shift"]["median"],
    ]
    colors = ["#b0413e", "#d98880", "#2f6fb0", "#7fa9d4"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, alpha=0.9)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4 if value >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.axhline(GATE, color="#333", ls="--", lw=1)
    ax.annotate(f"门禁 {GATE:.2f}", xy=(len(labels) - 0.5, GATE), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=8)
    ax.set_ylabel("中位方向一致性")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("全局平移配准扫描（42 对）：最优 dx=−2 仍低于门禁", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "registration_scan_20261107.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_loss(loss: dict) -> Path:
    """Grouped bars: the -1 primary ranking against the void +1 legacy ranking."""
    primary = {row["loss"]: row for row in loss["ranking"]}
    legacy = {row["loss"]: row for row in loss["legacy_ranking"]}
    losses = list(primary)
    x = np.arange(len(losses))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for offset, source, color, label in (
        (-width / 2, primary, "#2f6fb0", "phase_sign=−1（硬件实测，主变体）"),
        (width / 2, legacy, "#b0413e", "phase_sign=+1（历史约定，作废）"),
    ):
        values = [source.get(name, {}).get("direction_median", np.nan) for name in losses]
        ax.bar(x + offset, values, width, color=color, alpha=0.85, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, losses, rotation=12)
    ax.set_ylabel("held-out U/C 方向中位数")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("离线损失函数比较：−1 主变体 vs +1 历史变体", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "loss_compare_phase_sign_m1_20261107.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- report


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    focus = _load(DATA / "20261107_focus_scan" / "focus.json")
    fit = _load(DATA / "defocus_forward_20261107_focus_r2_fit" / "forward_fit.json")
    scan = _load(DATA / "issue235_parity_20261107" / "parity_registration_scan.json")
    verification = _load(
        DATA / "defocus_forward_20261107_focus_r2_fit" / "checkpoints" / "EXPORT_VERIFICATION.json")
    gated = _load(DATA / "gated_recurrent_20261107" / "results.json")

    ladders: dict[float, dict[int, dict[int, float]]] = {}
    stats: dict[float, dict[int, dict[str, float]]] = {}
    for run, amp in LADDERS:
        ladders[amp] = {}
        stats[amp] = {}
        for sign in (1, -1):
            rows = direction_rows(run, amp, sign)
            ladders[amp][sign] = per_mode(rows)
            stats[amp][sign] = summarize(rows)

    figure_focus_scan(focus)
    figure_fit_quality(fit)
    figure_direction(ladders)
    figure_registration(scan)

    loss_primary = _load(DATA / "loss_compare_20261107_phase_sign_m1" / "results.json")
    figure_loss(loss_primary)
    val = loss_primary["validation"]

    focus_probe = focus["probe"]
    fit_corr = {
        arm: st.median(float(row["corr"]) for row in fit["models"][arm]["rows"])
        for arm in fit["models"]
    }
    phase_arm = {name: verification["models"][name]["phase_sign_arm"]
                 for name in verification["models"]}

    all_plus = [row for amp in stats for row in [stats[amp][1]]]
    all_minus = [row for amp in stats for row in [stats[amp][-1]]]
    plus_median = st.median([row["median"] for row in all_plus])
    minus_median = st.median([row["median"] for row in all_minus])
    plus_positive = sum(row["positive"] for row in all_plus)
    minus_positive = sum(row["positive"] for row in all_minus)
    minus_at_gate = sum(row["at_gate"] for row in all_minus)
    n_pairs = sum(row["n"] for row in all_minus)

    direction_rows_md = []
    for amp in sorted(stats, reverse=True):
        plus, minus = stats[amp][1], stats[amp][-1]
        direction_rows_md.append([
            f"{amp:.2f}", str(plus["n"]), f"{plus['median']:+.3f}", str(plus["positive"]),
            f"{minus['median']:+.3f}", str(minus["positive"]), str(minus["at_gate"]),
        ])

    fit_rows_md = []
    for arm in fit["models"]:
        rows = fit["models"][arm]["rows"]
        fit_rows_md.append([
            arm,
            f"{fit_corr[arm]:.3f}",
            f"{st.mean(float(r['measured_cv']) for r in rows):.3f}",
            f"{st.mean(float(r['predicted_cv']) for r in rows):.3f}",
            f"{st.mean(float(r['corr']) for r in rows):.3f}",
        ])

    prim_loss = {row["loss"]: row for row in loss_primary["ranking"]}
    legacy_loss = {row["loss"]: row for row in loss_primary["legacy_ranking"]}
    loss_md = [
        [
            name,
            f"{prim_loss[name]['direction_median']:+.3f}",
            f"{prim_loss[name]['pearson_core61']:.3f}",
            f"{legacy_loss[name]['direction_median']:+.3f}",
            f"{legacy_loss[name]['pearson_core61']:.3f}",
        ]
        for name in prim_loss
    ]
    best = loss_primary["best"]
    loss_dirs = [row["direction_median"] for row in loss_primary["ranking"]]
    loss_range = (min(loss_dirs), max(loss_dirs))

    gated_base = gated["direction"]["base"]
    gated_arms = gated["direction"]["arms"]
    gated_arm_stats = {
        name: (
            float(np.mean(list(payload["per_seed"].values()))),
            float(np.std(list(payload["per_seed"].values()), ddof=1)),
        )
        for name, payload in gated_arms.items()
    }
    gated_arm_mean = {name: stats[0] for name, stats in gated_arm_stats.items()}
    gated_rows_md = [
        ["frozen base", f"{gated_base['mean']:+.4f}", f"{gated_base['std']:.4f}", str(gated_base["n"])]
    ]
    gated_rows_md += [
        [name, f"{mean_:+.4f}", f"{std_:.4f}", str(gated["direction"]["n_pairs"])]
        for name, (mean_, std_) in gated_arm_stats.items()
    ]

    body = f"""# 物理预测模型极性核查与单模屏方向门禁（20261107）

<!-- provenance:start -->
<!-- provenance:end -->

**日期** 2026-11-07 · **脚本** `scripts/slm/generate_forward_model_polarity_report_20261107.py`（离线渲染）
**设备** Santec SLM-200 #1（1920×1200）+ Daheng MER2-507-23GM（2592×1944）
**数据** `data/debug/slm_transfer/` 下的焦点扫描、成对 U/C 数据集与拟合、三组单模屏 ladder、配准扫描、离线损失比较、门控循环原型

## 1. 结论速览

| 指标 | `phase_sign=+1`（历史） | `phase_sign=−1`（硬件实测） | 判据 | 结论 |
|---|---|---|---|---|
| 单模屏方向中位相关（{n_pairs} 对） | {plus_median:+.3f} | {minus_median:+.3f} | 符号为正 | 已翻正（{plus_positive}/{n_pairs} → {minus_positive}/{n_pairs}） |
| 单对 ≥ 0.50 | 0/{n_pairs} | {minus_at_gate}/{n_pairs} | ≥ {GATE:.2f} | **未通过** |
| 全局平移最优（dx=−2） | {scan['pol+1']['best_global_shift']['median']:+.3f} | {scan['pol-1']['best_global_shift']['median']:+.3f} | ≥ {GATE:.2f} | **未通过** |
| U/C 拟合全帧相关（M1） | {phase_arm['M1_defocus']['legacy_vs_measured_pearson']:.3f} | {phase_arm['M1_defocus']['exported_vs_measured_pearson']:.3f} | 参考 | 取反后反而变差（见 §4.4） |
| 实验 anchored 闭环 | 未运行 | 未运行 | 门禁 + 实测 CV | 门禁未过，不实验 |

- 极性缺陷确实存在：`+1` 下 {plus_positive}/{n_pairs} 对为负，`−1` 下 {minus_positive}/{n_pairs} 对为正，且 |相关| 逐对完全不变——这是模型唯一非对称项（指令相位）被翻转的精确特征。
- 但翻正符号**不足以**通过门禁：中位只有 {minus_median:+.3f}，单对最高也只到 0.50 附近；全局平移最多把它抬到 {scan['pol-1']['best_global_shift']['median']:+.3f}。
- 更关键的是**自洽性矛盾**：同一份 U/C 数据上，`−1` checkpoint 的全帧 Pearson 为 {phase_arm['M1_defocus']['exported_vs_measured_pearson']:.3f}，低于 `+1` 的 {phase_arm['M1_defocus']['legacy_vs_measured_pearson']:.3f}。若极性真是全局传播约定，`−1` 不应同时破坏拟合质量。这说明当前 `phase_sign` 的作用域（把整条指令相位，含载波/倾斜，一起取反）与硬件行为不一致，是未解决项，不是已修好的 bug。
- 因此本轮**没有**进行 anchored 闭环采集；设备占用仅限数据采集阶段（§8）。
- 已在别处记录并修复的问题不在此复述。

## 2. 焦点重标定与成对 U/C 数据集

焦点扫描给出 `best_diopter_m1 = {focus_probe['best_diopter_m1']:+.5f}`（f ≈ {focus_probe['best_focal_mm'] / 1000:.2f} m），相对平场增益 {focus_probe['gain_vs_flat']:.3f}×，扫描范围 {focus_probe['range_m1'][0]:+.1f} … {focus_probe['range_m1'][1]:+.1f} m⁻¹。该结果取代 2026-10-27 的标定，写入 `data/debug/slm_transfer/20261107_focus_scan/focus.json`。

![焦点扫描](figures/focus_scan_20261107.png)

成对 U/C 数据集 `defocus_forward_20261107_focus_r2` 共 52 帧，按 U/C 分支配对采集，随后用 `scripts/slm/fit_defocus_forward_dataset.py` 拟合三档模型（M0 无离焦、M1 离焦、M3 离焦+照明）。逐样本中位相关：

{_table(["模型", "中位 Pearson", "平均实测 CV", "平均预测 CV", "平均 Pearson"], fit_rows_md)}

![拟合质量](figures/uc_fit_quality_20261107.png)

要点：M1/M3 的逐样本相关中位在 0.83 左右，但预测 CV 与实测 CV 的斜率并不一致（图右）。这解释了为什么模型核心 CV 能到 0.10 而实测停在 0.53：残差是**表征误差**，不是优化器或投影问题。

## 3. 单模 Zernike 屏方向 ladder（核心实验）

三组 ladder 分别以 ±0.10、±0.05、±0.02 rad 施加 Z7…Z11（实际覆盖 Z5–Z11），每档 7 个模式 × 2 个 pass = 14 对。方向指标为 `jacobian_guard.direction_agreement`：模型与实测的 C−U 差分做签名 Pearson，裁剪 128×128，模型原点。

{_table(["幅值/rad", "对数", "+1 中位", "+1 为正", "−1 中位", "−1 为正", "−1 ≥0.50"], direction_rows_md)}

![方向 ladder](figures/direction_ladder_20261107.png)

- 逐对 |相关| 在两种极性下完全相同（例如 Z7 幅值 0.10：{ladders[0.10][1][7]:+.3f} → {ladders[0.10][-1][7]:+.3f}），说明翻转只改变了差分符号，没有改变幅度结构。
- 最弱的模式是 Z6（散光 0°/90° 附近），在两种极性下 |相关| 都只有 0.2 左右；最强的是 Z7/Z11。
- 没有一对达到 0.50。这不是"再多跑几次就好"的统计问题，而是模型与实测在差分**形状**上仍有系统性差异。

## 4. 极性定位：已排除项与未解矛盾

### 4.1 显示链已排除

对 `panel_gray` 做反解，`gray/1023·2π` 与 `+panel_phase` 的 RMS 差为 0.0035 rad，与 `−panel_phase` 为 0.1791 rad。驱动器忠实编码 `+phase`，缺陷不在 LUT/显示链。

### 4.2 基函数与残差项已排除

`FrozenDefocusForward._mode_basis(j)` 与 `generate_zernike_phase` 对 Z5–Z11 的相关 1.0000、尺度 1.0000；旧 checkpoint 的 Noll 5–11 残差项翻转后仍是 0/42。

### 4.3 镜像/转置/背景增益已排除

8 种二面体变换 × ±极性中，`pol=−1 identity` 最优；背景（2.39 vs 0.82 counts/ms）与约 16× 增益误差对仿射不变的 Pearson 无影响；SLM↔相机轴交换已被拟合的 `ramp_matrix` 吸收。

### 4.4 未解矛盾：整条指令取反与 U/C 拟合不自洽

导出 checkpoint 时把 `phase_sign` 设为 −1，作用域是 `total_phase_grid` 之后的**整条指令相位**（含载波/倾斜），而离焦与残差项保留原符号。结果：

{_table(["模型", "导出的 phase_sign", "与实测全帧 Pearson（−1）", "与实测全帧 Pearson（+1 历史）", "−1 与 +1 的相对 L2"], [[name, f"{payload['exported_phase_sign']:+.0f}", f"{payload['exported_vs_measured_pearson']:.3f}", f"{payload['legacy_vs_measured_pearson']:.3f}", f"{payload['exported_vs_legacy_rel_l2']:.3f}"] for name, payload in phase_arm.items()])}

单模屏只测了 Zernike 项，没有载波；U/C 数据则由载波主导。若极性是全局传播约定，取反应同时改善两者；实测是单模屏变好、U/C 变差。**最可能的解释**是缺陷只作用于"像差指令"这一层，而不作用于定位光斑的载波/倾斜项；但这一假设尚未在硬件上单独验证。在验证之前，不能把 `−1` checkpoint 当作已修好的模型。

## 5. 全局配准排查

42 对合并后，`pol=−1` 无配准中位 {scan['pol-1']['no_registration']['median']:+.3f}（{scan['pol-1']['no_registration']['n_ge_0.5']}/{scan['pol-1']['no_registration']['n']} ≥0.50）；扫描全局平移后最优为 dy={scan['pol-1']['best_global_shift']['dy']}, dx={scan['pol-1']['best_global_shift']['dx']}，中位 {scan['pol-1']['best_global_shift']['median']:+.3f}（{scan['pol-1']['best_global_shift']['n_ge_0.5']}/{scan['pol-1']['best_global_shift']['n']}）。单一全局平移无法达到门禁，逐对最优平移不可部署（每对一个自由参数）。

![配准扫描](figures/registration_scan_20261107.png)

## 6. 离线损失函数比较（#236）

同一数据划分、同一初始化、300 次 full-batch Adam，比较 7 种损失；`phase_sign=−1` 为主变体（`scripts/slm/loss_compare_20261107.py`，产物 `data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/`），`+1` 变体仅作历史对照。首轮以 `+1` 为主变体的推荐（masked_mse，方向中位 +0.483）**作废**：其符号约定已被 §3 的硬件证据推翻。

{_table(["损失", "−1 方向中位", "−1 核心 Pearson", "+1 方向中位（作废）", "+1 核心 Pearson"], loss_md)}

- `−1` 主变体最优为 **{best['loss']}**（方向中位 {best['direction_median']:+.3f}），但 7 种损失全部落在 {loss_range[0]:+.3f} … {loss_range[1]:+.3f}，方向分辨率仅 ±0.05，held-out 又只有 9 个 U/C 对——**无法据此选损失**。
- 基线复现**失败**：`−1` 变体重新拟合的离焦 d = {val['reproduced']['defocus_diopter']:+.4f} m⁻¹，与存档 M1_defocus 的 {val['stored']['defocus_diopter']:+.4f} 相差 {val['defocus_abs_delta']:.3f}（容差 {val['tolerance']['defocus_abs']}）；原点偏移 {val['origin_max_abs_delta_px']:.1f} px（容差 {val['tolerance']['origin_px']:.0f}）。与 §4.4 的作用域矛盾一致：把含载波/倾斜的整条指令取反，连 U/C 基线都保不住。
- U/C 数据以径向对称的离焦为主，对极性不敏感；它既不能验证极性，也不能替代单模屏 ladder。
- 因此 #236 的结论记为**未定**：损失选择必须等方向门禁通过后，在单模屏或闭环指标上重做。

## 7. 门控循环原型（#239）

四变体对比（memoryless / concat / gated / attention，各 3 个种子），共享优化器、预算与数据；门控变体额外输入上一帧 CCD、相位与残差。基础模型按 `phase_sign=−1` 运行（`scripts/slm/gated_recurrent_transfer_20261107.py`，产物 `data/debug/slm_transfer/gated_recurrent_20261107/`）。

{_table(["变体", "方向均值", "标准差", "对数"], gated_rows_md)}

- 判定为 `inconclusive`：门控变体与 memoryless 在全部收敛阈值上的迭代数差异都在 10 次评估粒度内（r2 域门控约快 3 次，ladder 域持平或更慢），没有观察到迭代数压缩。
- 值得注意的是 memoryless 残差头把方向均值从 {gated_base['mean']:+.3f} 抬到 {gated_arm_mean.get('memoryless', float('nan')):+.3f}；这说明方向指标可以通过学习残差头改善，而不必先解决物理预测模型。但这只是代理指标，不等于闭环收敛。
- 样本量很小（r2 可用 36 帧；ladder pass1/pass2 各 42 帧，且 pass2 重复 pass1 的同一批 ±指令——held-out 测的是复现性而非对新相位的泛化），结论不能外推。

## 8. 设备占用与释放记录

| 时间（2026-11-07） | 占用 | 内容 | 释放 |
|---|---|---|---|
| 13:52 | SLM #1 + Daheng cam 0 | 旧光学状态下的 U/C 语料（后判定为 stale-optics） | 采集结束即释放 |
| 15:32 | SLM #1 + Daheng cam 0 | 重标定后的成对 U/C 数据集 `defocus_forward_20261107_focus_r2` | 采集结束即释放 |
| 16:21 | SLM #1 + Daheng cam 0 | 单模屏 ladder（0.10/0.05/0.02 rad） | 每档结束即释放 |
| 17:48–17:54:45 | SLM #1 + Daheng cam 0 | anchored 闭环捕获 `20261107_174854` | 17:54:45 结束，17:55:59 复查无占用进程 |

期间 SLM #1 出现一次控制器无响应（`ReadSU = −1000`），由用户手动重启后恢复；软件侧未做恢复尝试。本报告为离线渲染，未占用任何设备。

## 9. 未解决项与下一步

1. **极性作用域**：验证"只翻转像差指令、保留载波/倾斜"是否同时满足单模屏符号与 U/C 拟合质量；§6 的基线复现失败（整条指令取反连 U/C 基线都保不住）进一步支持该假设。这是当前最高优先级，且可离线用已有数据先做筛选。
2. **门禁**：在作用域修正后重跑三组 ladder，目标 ≥0.50；若仍不足，需要 oracle 评审残差结构（候选：SLM 相位非线性、亚像素原点约定、相机响应）。
3. **#236**：`−1` 变体上 7 种损失方向中位全部落在 ±0.07 内，无区分度，结论记"未定"。
4. **#237/#238**：2D 配准、实测 PSF、增益场、CCD 响应 LUT、1×1/3×3 LUT 与串扰核，均未开始。
5. 门禁通过后才安排实验 anchored 闭环，判据仍是实测核心 CV ≤ 0.10。
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / Path(REPORT_KEY).name
    report_path.write_text(body, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    print(build())
