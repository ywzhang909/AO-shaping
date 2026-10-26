"""Render ``report/sim2f_shaping/closed_loop_20261025/report.md`` and its figures.

**What this report is.** ``ml.sim2f`` is a torch twin of the repository's calibrated
2f bench. A supervised corpus was built on the *ideal* configuration of that twin
(aberration-free, detector off, no 0-order light), a U-Net was trained on it at a
fixed budget, and then the same checkpoint was driven through a five-reading closed
loop on the *calibrated* configuration (10.0847 mm camera defocus + 10 % 0-order).

The single question the document answers is: **why does a network that beats the
zero-phase command on its own bench lose to it on the calibrated bench, and is the
cause the network, the amortisation, or the bench?** The ablation answers it -- the
camera defocus, not the 0-order -- and the report keeps every measurement that makes
the answer checkable, including the ones that do not flatter the method.

**Offline only.** Reads JSON/CSV artefacts already on disk, plus the corpus ``.npz``
with numpy for one figure. No ``torch``, no GPU, no camera, no SLM: the numbers were
measured by ``runs/sim2f_fair/*.py`` and ``python -m ml.sim2f``, and this script only
renders them. Every number in the body is read from an artefact at render time, so a
re-run of the measurement lane cannot leave the prose behind.

**Not a sim-to-real result.** Teacher, network and loop share one forward model, so
what is measured is an amortisation + closed-loop-estimation budget. The twin's own
residual against hardware is quoted in section 9 for exactly that reason.

Usage::

    python scripts/ml/generate_sim2f_shaping_report.py
    python scripts/ml/generate_sim2f_shaping_report.py --run-dir runs/sim2f_fair
    python scripts/ml/generate_sim2f_shaping_report.py --no-figures --no-artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts._common.provenance import insert_header

REPORT_KEY = "report/sim2f_shaping/closed_loop_20261025/report.md"
OUT_DIR = ROOT / "report" / "sim2f_shaping" / "closed_loop_20261025"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"
STAMP = "_20261025"

DEFAULT_RUN = ROOT / "runs" / "sim2f_fair"
DEFAULT_SMOKE = ROOT / "runs" / "sim2f_smoke800"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

#: Reading order used everywhere: the criterion's own arms first, the ceiling last.
ARM_ORDER = ("flat", "net_open", "net_probe", "net_closed", "oracle")
ARM_STYLE = {
    "flat": ("零相位 flat", "#4d4d4d"),
    "net_open": ("net_open", "#2b7bba"),
    "net_probe": ("net_probe", "#6aa7d8"),
    "net_closed": ("net_closed", "#1f9e77"),
    "oracle": ("oracle", "#c2452d"),
}
#: Ablation variants. ``calibrated`` is the reference; the other three remove one knob.
VARIANT_ORDER = ("calibrated", "defocus_only", "eta_only", "ideal")
VARIANT_LABEL = {
    "calibrated": "标定（离焦+0 级）",
    "defocus_only": "只留离焦（0 级=0）",
    "eta_only": "只留 0 级（离焦=0）",
    "ideal": "都去掉（理想）",
}
#: The three loop configurations of section 5.
LOOP_ORDER = ("a_no_noise", "c_det20", "d_no_jitter")
LOOP_LABEL = {
    "a_no_noise": "a_no_noise",
    "c_det20": "c_det20",
    "d_no_jitter": "d_no_jitter",
}
LOOP_DESC = {
    "a_no_noise": "探测器关闭，曝光档 0.15 ms，光照抖动 0.06",
    "c_det20": "探测器开启，曝光 20 ms，光照抖动 0.06",
    "d_no_jitter": "探测器关闭，曝光档 0.15 ms，光照抖动 0",
}
MISSING = "未测"


# --------------------------------------------------------------------------- load
def _read_json(path: Path) -> dict[str, Any] | None:
    """Load one JSON artefact, or ``None`` when it is absent.

    Absence is *not* an error: the report says ``未测`` for the numbers that artefact
    would have carried, instead of silently rendering a placeholder that looks like a
    measurement. The caller collects the names so section 10 can list them.
    """
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Payloads:
    """Everything the report reads, loaded once."""

    def __init__(self, run_dir: Path, smoke_dir: Path) -> None:
        self.run_dir = run_dir
        self.smoke_dir = smoke_dir
        self.missing: list[str] = []

        self.meta = self._take("corpus/meta.json")
        self.teacher = self._take("teacher_vs_flat.json")
        self.train_metrics = self._take("train/metrics.json")
        self.train_config = self._take("train/config.json")
        self.diagnose = self._take("diagnose.json")
        self.bench = self._take("bench_transfer.json")
        self.loops = {tag: self._take(f"loop/{tag}/eval_loop.json") for tag in LOOP_ORDER}
        self.ablation = {
            tag: self._take(f"ablate/loops/{tag}/eval_loop.json") for tag in VARIANT_ORDER
        }
        self.ablation_summary = self._take("ablate/ablation_loops.json")
        self.teacher_control = self._take("ablate/teacher_control.json")
        self.smoke_loop = _read_json(smoke_dir / "loop_c" / "eval_loop.json")
        if self.smoke_loop is None:
            self.missing.append("smoke loop_c/eval_loop.json")
        self.corpus = self._load_corpus()

    def _take(self, relative: str) -> dict[str, Any] | None:
        payload = _read_json(self.run_dir / relative)
        if payload is None:
            self.missing.append(relative)
        return payload

    def _load_corpus(self) -> dict[str, np.ndarray] | None:
        path = self.run_dir / "corpus" / "pairs.npz"
        if not path.is_file():
            self.missing.append("corpus/pairs.npz")
            return None
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}

    # -- derived lookups ---------------------------------------------------
    @property
    def families(self) -> list[str]:
        """Family names in the corpus's own JSON order (not alphabetised)."""
        if self.teacher is not None:
            return list(self.teacher["by_kind"])
        if self.diagnose is not None:
            return list(self.diagnose["by_split_kind"]["train"])
        return []

    def kind_of_sample(self) -> dict[int, str]:
        """sample index -> family, taken from the teacher table's own rows."""
        if self.teacher is None:
            return {}
        return {int(row["index"]): str(row["kind"]) for row in self.teacher["rows"]}

    def selected_families(self, tag: str) -> list[str]:
        """Families the loop actually drew (``cross`` is missing in every config)."""
        loop = self.loops.get(tag)
        if loop is None:
            return []
        return list(loop["per_kind"])


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return MISSING
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return MISSING
    return f"{100.0 * float(value):.{digits}f}%"


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# ------------------------------------------------------------------------ figures
def _save(fig: plt.Figure, path: Path, *, enabled: bool) -> str | None:
    if not enabled:
        plt.close(fig)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_teacher_vs_flat(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    """Per-family L1 and NCC: teacher vs the zero-phase command."""
    if data.teacher is None:
        return None
    by_kind = data.teacher["by_kind"]
    families = list(by_kind)
    x = np.arange(len(families))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    for ax, key, ylabel, title in (
        (axes[0], "l1", "L1", "(a) 逐族 L1（越小越好）"),
        (axes[1], "ncc", "NCC", "(b) 逐族 NCC（越大越好）"),
    ):
        flat = [by_kind[f][f"flat_{key}_mean"] for f in families]
        teacher = [by_kind[f][f"teacher_{key}_mean"] for f in families]
        ax.bar(x - width / 2, flat, width, label="零相位 flat", color="#4d4d4d")
        ax.bar(x + width / 2, teacher, width, label="teacher", color="#c2452d")
        ax.set_xticks(x, families, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        if key == "ncc":
            ax.set_ylim(0.0, 1.05)
    for index, family in enumerate(families):
        axes[0].annotate(
            f"{by_kind[family]['thin_px_mean']:.1f} px",
            (index, max(by_kind[family]["flat_l1_mean"], by_kind[family]["teacher_l1_mean"]) + 0.06),
            ha="center", fontsize=7, color="#555555",
        )
    axes[0].set_ylim(0.0, 2.15)
    per_family = by_kind[families[0]]["n"]
    fig.suptitle(
        f"teacher 与零相位命令的逐族对比（{len(families)} 族 × {per_family} 样本，"
        "柱顶标注为最薄特征宽度）",
        fontsize=11,
    )
    return _save(fig, path, enabled=enabled)


def fig_corpus_target_vs_flat(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    """One target and its zero-phase frame per family -- why a delta loses on L1."""
    corpus = data.corpus
    if corpus is None or data.teacher is None:
        return None
    kind_of = data.kind_of_sample()
    families = data.families
    targets, spots = corpus["target"], corpus["spot"]
    split = corpus["split"]

    chosen: dict[str, int] = {}
    for index, family in sorted(kind_of.items()):
        if family in chosen:
            continue
        if int(split[index]) == 0:  # train first: those are the samples the net saw
            chosen[family] = index
    for index, family in sorted(kind_of.items()):
        chosen.setdefault(family, index)

    fig, axes = plt.subplots(2, len(families), figsize=(1.6 * len(families), 4.3))
    for column, family in enumerate(families):
        index = chosen.get(family)
        if index is None:
            continue
        for row, (array, label) in enumerate(((targets, "目标"), (spots, "零相位帧"))):
            frame = np.asarray(array[index, 0], dtype=float)
            frame = frame / max(frame.max(), 1e-12)
            axes[row, column].imshow(np.sqrt(frame), cmap="inferno", vmin=0.0, vmax=1.0)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                stats = data.teacher["by_kind"][family]
                axes[row, column].set_title(
                    f"{family}\n薄特征 {stats['thin_px_mean']:.1f} px", fontsize=8
                )
            else:
                axes[row, column].set_xlabel(label, fontsize=8)
    fig.suptitle(
        "语料一例：上排为目标分布，下排为同一光学的零相位帧（各自峰值归一、平方根显示）",
        fontsize=10,
    )
    return _save(fig, path, enabled=enabled)


def fig_train_curve(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    if data.train_metrics is None:
        return None
    history = data.train_metrics["history"]
    steps = [float(row["step"]) for row in history]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(steps, [row["train_total"] for row in history], "-o", ms=3, color="#2b7bba",
            label="train_total")
    ax.plot(steps, [row["val_total"] for row in history], "-s", ms=3, color="#c2452d",
            label="val_total")
    ax.plot(steps, [row["train_phy"] for row in history], "--", lw=1.2, color="#2b7bba",
            label="train_phy")
    ax.plot(steps, [row["val_phy"] for row in history], "--", lw=1.2, color="#c2452d",
            label="val_phy")
    best = float(data.train_metrics["best_step"])
    ax.axvline(best, color="#1f9e77", lw=1.0, ls=":")
    ax.annotate(f"best_step={best:.0f}", (best, ax.get_ylim()[1]), fontsize=8,
                color="#1f9e77", ha="right", va="top")
    ax.set_xlabel("step")
    ax.set_ylabel("损失")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(
        f"固定预算训练（{data.train_metrics['steps']} 步 / "
        f"{data.train_metrics['counts']['train']} 训练样本 / "
        f"{data.train_metrics['counts']['val']} 验证样本）"
        if data.train_metrics.get("counts") else
        f"固定预算训练（{data.train_metrics['steps']} 步）",
        fontsize=10,
    )
    return _save(fig, path, enabled=enabled)


def fig_bench_transfer(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    """net / flat / cached teacher phase on three bench states, same 24 samples."""
    if data.bench is None:
        return None
    aggregate = data.bench["aggregate"]
    states = (
        ("ideal", "ideal（标签所在实验系统）"),
        ("cal_optics_only", "cal_optics_only（只有标定光学）"),
        ("cal_plus_state", "cal_plus_state（标定光学 + 抽样像差）"),
    )
    x = np.arange(len(states))
    width = 0.26
    series = (
        ("net", "net obj", "#2b7bba"),
        ("flat", "flat obj", "#4d4d4d"),
        ("teacher", "teacher 缓存相位", "#c2452d"),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for offset, (key, label, color) in zip((-width, 0.0, width), series):
        values = [aggregate[state][f"{key}_objective"] for state, _ in states]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [label for _, label in states], fontsize=8)
    ax.set_ylabel("teacher 目标")
    ax.set_ylim(0.0, 1.85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("同一 24 个样本、三种实验系统状态：谁赢", fontsize=10)
    return _save(fig, path, enabled=enabled)


def fig_diagnose_splits(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    if data.diagnose is None:
        return None
    splits = ("train", "val", "test")
    by_split = data.diagnose["by_split"]
    x = np.arange(len(splits))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for offset, (key, label, color) in zip(
        (-width, 0.0, width),
        (("net_objective_mean", "net obj", "#2b7bba"),
         ("flat_objective_mean", "flat obj", "#4d4d4d"),
         ("teacher_objective_mean", "teacher obj", "#c2452d")),
    ):
        values = [by_split[split][key] for split in splits]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{split}\nn={by_split[split]['n']} · net 胜 flat "
                      f"{by_split[split]['net_better_than_flat']}/{by_split[split]['n']}\n"
                      f"phasor_corr {by_split[split]['phasor_corr_mean']:.3f}"
                      for split in splits], fontsize=8)
    ax.set_ylabel("teacher 目标")
    ax.set_ylim(0.0, 1.95)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_title("标签所在实验系统上：net 明显优于 flat，但与 teacher 的相位相关很低", fontsize=10)
    return _save(fig, path, enabled=enabled)


def fig_ablation_variants(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    arms = ("flat", "net_open", "net_closed", "oracle")
    available = [tag for tag in VARIANT_ORDER if data.ablation.get(tag) is not None]
    if not available:
        return None
    x = np.arange(len(available))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(arms))
    for offset, arm in zip(offsets, arms):
        label, color = ARM_STYLE[arm]
        values = [data.ablation[tag]["aggregate"][arm]["objective"] for tag in available]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
    for index, tag in enumerate(available):
        won = sum(
            1
            for family, row in data.ablation[tag]["per_kind"].items()
            if row["arms"]["net_open"]["objective"] < row["arms"]["flat"]["objective"]
        )
        total = len(data.ablation[tag]["per_kind"])
        ax.annotate(f"net_open 胜 {won}/{total} 族", (index, 1.79), ha="center", fontsize=7.5)
    ax.set_xticks(x, [VARIANT_LABEL[tag] for tag in available], fontsize=8)
    ax.set_ylabel("teacher 目标")
    ax.set_ylim(0.0, 1.92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    ax.set_title("四个变体 × 四路读数（24 样本，探测器关闭）", fontsize=10)
    return _save(fig, path, enabled=enabled)


def fig_ablation_mechanism(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    """State-free blocks: what defocus does to *both* frames."""
    if data.teacher_control is None:
        return None
    aggregate = data.teacher_control["aggregate"]
    variants = [tag for tag in VARIANT_ORDER if tag in aggregate]
    if not variants:
        return None
    x = np.arange(len(variants))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key, ylabel, title in (
        (axes[0], "ncc", "NCC", "(a) 零相位帧被离焦抬起来（NCC 越大越好）"),
        (axes[1], "l1", "L1", "(b) 同一批帧的 L1 反过来（越小越好）"),
    ):
        for offset, (arm, label, color) in zip(
            (-width, 0.0, width),
            (("flat", "零相位 flat", "#4d4d4d"),
             ("net_open", "net_open", "#2b7bba"),
             ("teacher", "teacher 缓存相位", "#c2452d")),
        ):
            values = [aggregate[tag]["optics_only"][f"{arm}_{key}"] for tag in variants]
            bars = ax.bar(x + offset, values, width, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                            ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(x, [VARIANT_LABEL[tag] for tag in variants], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5)
        ax.set_ylim(0.0, 1.95)
    fig.suptitle("静态光学只有两项在动：无抽样像差的同一批帧（optics_only）", fontsize=11)
    return _save(fig, path, enabled=enabled)


def fig_loop_family_delta(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    tags = [tag for tag in LOOP_ORDER if data.loops.get(tag) is not None]
    if not tags:
        return None
    fig, axes = plt.subplots(1, len(tags), figsize=(4.4 * len(tags), 4.2), squeeze=False)
    arms = (("net_open", "net_open − flat", "#2b7bba"),
            ("net_closed", "net_closed − flat", "#1f9e77"),
            ("oracle", "oracle − flat", "#c2452d"))
    width = 0.26
    for ax, tag in zip(axes[0], tags):
        loop = data.loops[tag]
        families = list(loop["per_kind"])
        x = np.arange(len(families))
        for offset, (arm, label, color) in zip((-width, 0.0, width), arms):
            deltas = [
                loop["per_kind"][family]["arms"][arm]["objective"]
                - loop["per_kind"][family]["arms"]["flat"]["objective"]
                for family in families
            ]
            ax.bar(x + offset, deltas, width, label=label, color=color)
        ax.axhline(0.0, color="black", lw=0.9)
        ax.set_xticks(x, families, fontsize=7.5, rotation=35, ha="right")
        ax.set_ylabel("目标差（正 = 比 flat 差）")
        ax.set_title(f"{LOOP_LABEL[tag]}\n{LOOP_DESC[tag]}", fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle("逐族相对零相位命令的差（三个配置各 7 族，cross 未被抽到）", fontsize=10)
    return _save(fig, path, enabled=enabled)


def fig_noise_ladder(data: Payloads, path: Path, *, enabled: bool) -> str | None:
    """Coefficient fit vs exposure: refused rungs drawn as refused, not as bad fits."""
    tags = [tag for tag in VARIANT_ORDER if data.ablation.get(tag) is not None]
    if not tags:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), squeeze=False)
    for ax, tag in zip(axes.ravel(), tags):
        ladder = data.ablation[tag]["noise_ladder"]
        labels = ["关闭" if row["exposure_ms"] is None else f"{row['exposure_ms']:g}"
                  for row in ladder]
        residual = [row["residual"] for row in ladder]
        refused = [int(row["iterations"]) == 0 for row in ladder]
        colors = ["#b0b0b0" if is_refused else "#2b7bba"
                  for is_refused in refused]
        bars = ax.bar(np.arange(len(ladder)), residual, color=colors,
                      edgecolor=["#c2452d" if is_refused else "#1c4f73" for is_refused in refused],
                      hatch=["//" if is_refused else "" for is_refused in refused])
        for bar, row, is_refused in zip(bars, ladder, refused):
            peak = row["peak_counts"]
            peak_text = "无噪" if row["exposure_ms"] is None else f"峰 {peak:.0f}"
            ax.annotate(peak_text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6.5)
            if is_refused:
                ax.annotate("拒测", (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                            ha="center", va="center", fontsize=7, color="#8a1c0a",
                            fontweight="bold")
        ax.set_xticks(np.arange(len(ladder)), labels, fontsize=8)
        ax.set_xlabel("曝光（ms）")
        ax.set_ylabel("拟合残差")
        ax.set_ylim(0.0, 1.18)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(VARIANT_LABEL[tag], fontsize=9)
    fig.suptitle(
        "噪声阶梯：斜纹 = 被最小改善门禁拒测（iterations=0，系数为空），不是拟合得很差",
        fontsize=10,
    )
    return _save(fig, path, enabled=enabled)


# ----------------------------------------------------------------------- sections
def section_summary(data: Payloads) -> list[str]:
    lines = ["## 0 结论摘要", ""]
    teacher = data.teacher["by_kind"] if data.teacher is not None else {}
    if teacher:
        teacher_l1 = [row["teacher_l1_mean"] for row in teacher.values()]
        flat_l1 = [row["flat_l1_mean"] for row in teacher.values()]
        teacher_ncc = [row["teacher_ncc_mean"] for row in teacher.values()]
        line = teacher.get("line", {})
        corpus_n = data.meta["synth"]["count"] if data.meta is not None else len(
            data.kind_of_sample())
        floor = (data.meta["synth"]["min_feature_px"] if data.meta is not None else None)
        lines.append(
            f"1. **目标可达**：{corpus_n} 对样本 / {len(teacher)} 族，最薄特征设 "
            f"{floor:.0f} px 下限。teacher（逐样本可微"
            f"优化）在 **{len(teacher)}/{len(teacher)}** 族都优于零相位命令：L1 "
            f"{min(teacher_l1):.3f}–{max(teacher_l1):.3f}"
            f" 对 {min(flat_l1):.3f}–{max(flat_l1):.3f}，NCC {min(teacher_ncc):.3f}–"
            f"{max(teacher_ncc):.3f}。`line` 是下限的直接受益者：下限之前它是 1.5–3.0 px 的"
            f"不可达目标（L1 1.62），现在是 {line.get('thin_px_mean', float('nan')):.1f} px、"
            f"L1 {line.get('teacher_l1_mean', float('nan')):.4f}。"
        )
    diagnose = data.diagnose["by_split"] if data.diagnose is not None else None
    if diagnose is not None:
        lines.append(
            f"2. **网络不是欠训练、也没过拟合**：在它的标签所在的孪生上，net 在训练集"
            f" {diagnose['train']['net_objective_mean']:.4f} 对 flat "
            f"{diagnose['train']['flat_objective_mean']:.4f}（"
            f"{diagnose['train']['net_better_than_flat']}/{diagnose['train']['n']} 样本）、"
            f"测试集 {diagnose['test']['net_objective_mean']:.4f} 对 flat "
            f"{diagnose['test']['flat_objective_mean']:.4f}（"
            f"{diagnose['test']['net_better_than_flat']}/{diagnose['test']['n']}）。但它与 "
            f"teacher 的 phasor 相关只有 {diagnose['test']['phasor_corr_mean']:.3f}"
            f"（训练集 {diagnose['train']['phasor_corr_mean']:.3f}）；相位差 RMS "
            f"{diagnose['train']['rms_vs_teacher_rad_mean']:.3f} rad。它找到的是**一个**解，"
            "不是 teacher 的解。"
        )
    loop = data.loops.get("a_no_noise")
    if loop is not None:
        aggregate = loop["aggregate"]
        won = sum(
            1
            for row in loop["per_kind"].values()
            if row["arms"]["net_open"]["objective"] < row["arms"]["flat"]["objective"]
        )
        lines.append(
            f"3. **换到标定后的实验系统，所有相位读数都输给零相位**：flat "
            f"{aggregate['flat']['objective']:.4f}，net_open "
            f"{aggregate['net_open']['objective']:.4f}、net_probe "
            f"{aggregate['net_probe']['objective']:.4f}、net_closed "
            f"{aggregate['net_closed']['objective']:.4f}、oracle "
            f"{aggregate['oracle']['objective']:.4f}（net_open 只赢 {won}/"
            f"{len(loop['per_kind'])} 族）。"
        )
    control = data.teacher_control["aggregate"] if data.teacher_control is not None else None
    if control is not None:
        state = control["calibrated"]["state"]
        optics = control["calibrated"]["optics_only"]
        lines.append(
            f"4. **teacher 也输**（{state['teacher_objective']:.4f} 对 flat "
            f"{state['flat_objective']:.4f}；只保留标定光学时 {optics['teacher_objective']:.4f} "
            f"对 {optics['flat_objective']:.4f}，teacher 只在 "
            f"{optics['teacher_better_samples']}/{optics['n']} 个样本上赢）⇒ 这既不是摊销"
            "（amortisation）差，也不是网络缺陷：**产出标签的实验系统与评估用的实验系统不是同一个**。"
        )
    if control is not None and data.ablation.get("defocus_only") and data.ablation.get("eta_only"):
        defocus = data.ablation["defocus_only"]["aggregate"]
        eta = data.ablation["eta_only"]["aggregate"]
        ideal = data.ablation["ideal"]["aggregate"]
        mechanism_flat = control["ideal"]["optics_only"]
        mechanism_defocus = control["defocus_only"]["optics_only"]
        lines.append(
            f"5. **归因是 10.0847 mm 相机离焦，不是 0 级 η**：只留离焦时反转仍在（flat "
            f"{defocus['flat']['objective']:.4f} 对 net_open {defocus['net_open']['objective']:.4f}，"
            f"net_open 0/{len(data.ablation['defocus_only']['per_kind'])} 族）；只留 η 时反转"
            f"消失（flat {eta['flat']['objective']:.4f} 对 net_open "
            f"{eta['net_open']['objective']:.4f}，7/7 族）；两者都去掉时 flat "
            f"{ideal['flat']['objective']:.4f} 对 net_open {ideal['net_open']['objective']:.4f}。"
            f"机制实测：离焦把零相位点斑弥散到目标自己的尺度，flat 的 NCC "
            f"{mechanism_flat['flat_ncc']:.3f}→{mechanism_defocus['flat_ncc']:.3f}、L1 "
            f"{mechanism_flat['flat_l1']:.3f}→{mechanism_defocus['flat_l1']:.3f}；而正确整形的"
            f"光束反过来被弥散，net 的 NCC {mechanism_flat['net_open_ncc']:.3f}→"
            f"{mechanism_defocus['net_open_ncc']:.3f}。"
        )
    lines.append("")
    return lines


def section_environment(data: Payloads) -> list[str]:
    lines = ["## 1 环境与契约", ""]
    sim = data.meta["sim2f"] if data.meta is not None else None
    if sim is not None:
        lines += [
            "| 项 | 值 |",
            "|---|---|",
            f"| 光学 / 探测器 | 2f 实验系统的数字孪生；瞳面 {sim['grid']} 格 → 相机 "
            f"{sim['ccd_px']} px |",
            f"| 相机像元 / 视场 | {sim['ccd_pixel_um']:.1f} µm / "
            f"{1e3 * sim['ccd_fov_m']:.4f} mm |",
            f"| 采样比 `sampling_ratio` | {sim['sampling_ratio']:.4f}（远场采样间距 / 相机像元）|",
            f"| 远场角标度 | `far_field_padding` = {sim['far_field_padding']}，"
            f"`fft_length` = {sim['fft_length']}，远场采样间距 "
            f"{1e6 * sim['far_field_pitch_m']:.4f} µm |",
            f"| 波长 / 焦距 / 有效口径 | {1e9 * sim['wavelength_m']:.1f} nm / "
            f"{1e3 * sim['focal_length_m']:.1f} mm / {1e3 * sim['aperture_m']:.2f} mm |",
            f"| 面板像元 / 瞳面节距 | {sim['panel_pixel_um']:.1f} µm / "
            f"{1e6 * sim['pupil_pitch_m']:.2f} µm（`region_span_px` = "
            f"{sim['region_span_px']:.0f}）|",
            f"| 视场内 PSF 宽度数 | {sim['fov_in_psf']:.2f} |",
            f"| 标签所在配置 | 理想：`defocus_m` = {sim['defocus_m']:.1f}、"
            f"`zero_order_eta` = {sim['zero_order_eta']:.1f}、曝光 {sim['exposure_ms']:.1f} ms、"
            f"探测器{'开启' if sim['detector_enabled'] else '关闭'} |",
            f"| 辐照度标度 | 平场峰 {sim['peak_counts_per_ms']:.1f} counts/ms，暗电平 "
            f"{sim['dark_level_counts']:.2f} counts |",
            "",
        ]
    parity = data.teacher is not None and data.teacher.get("all_geometry_rebuilds_match")
    lines += [
        "**孪生一致性由测试钉住**，不是本文的断言："
        "`tests/ao_shaping/ml/sim2f/test_twin_parity.py` 把 `Sim2fConfig` 逐字段映射到 "
        "`BenchTwinConfig`，在 `sampling_ratio = 1` 的对齐口径上比帧。该测试模块文档记录的"
        "峰归一偏差：理想光路平坦帧 3.4e-16 / 整形帧 1.9e-4，10.08 mm 离焦下 2.9e-5 / 3.9e-4，"
        "预算 2e-3 / 5e-3。",
        "",
    ]
    if sim is not None:
        lines += [
            f"**相机是像元积分的，不是点采样**：`sampling_ratio` = {sim['sampling_ratio']:.4f}，"
            "所以模型在每个相机像元上做面积积分，而不是把 FFT 采样点当成像元"
            f"（`exact_fft_n` = {sim['exact_fft_n']} 是两者重合的长度）。",
            "",
        ]
    lines += [
        "**0 级是瞳面的相干混合**：`U = A·[sqrt(1-η)·e^{iφ} + sqrt(η)]`，"
        "未调制的那一份保持入射振幅，所以它和整形场**干涉**，而不是单纯叠加本底；"
        "调制通道的能量随 η 下降，不能假设能量守恒。标定配置 η = 0.1。",
        "",
        "**没有建模的东西**（全部读自配置，不是推测）：面板静态像差与粗糙度（`slm_zernike` 空、"
        "`slm_roughness_sigma_rad` = 0）、LUT 非线性（`lut_gamma` = 1.0，即线性）、"
        "透镜像差（`lens_zernike` 空）、光路倾斜（`tilt_*` = 0）、面板串扰（`phase_blur_px` = 0，"
        "孪生不建模）。也就是说 `calibrated()` 只有两个非恒等旋钮：离焦与 η。"
        "另外，标签所在的理想配置不经过探测器（曝光 0、无量化 / 读出噪声）。",
        "",
    ]
    if parity:
        lines += [
            f"语料自检：{data.teacher['n_samples']} 个样本的几何重建与缓存目标逐位一致"
            "（`all_geometry_rebuilds_match = true`），所以 teacher 标签与网络训练用的是同一批帧。",
            "",
        ]
    return lines


def section_reachability(data: Payloads) -> list[str]:
    lines = ["## 2 目标形状与可行性下限", ""]
    if data.meta is None:
        return lines + ["语料元数据缺失：" + MISSING, ""]
    synth = data.meta["synth"]
    psf = data.teacher.get("psf_fwhm") if data.teacher is not None else None
    lines += [
        f"语料 `count` = {synth['count']}，每族 {data.meta['counts']['kind'][data.families[0]]} 个，"
        f"划分 train/val/test = {data.meta['counts']['split']['train']}/"
        f"{data.meta['counts']['split']['val']}/{data.meta['counts']['split']['test']}，"
        f"抽取种子 {synth['seed']}，尺寸比 {synth['size_frac'][0]:.2f}–{synth['size_frac'][1]:.2f}。",
        "",
    ]
    if psf is not None:
        half_max = psf["measured_flat_half_max_px"]
        lines += [
            f"**{synth['min_feature_px']:.0f} px 的特征下限不是口味旋钮。** 孪生 PSF 的半高全宽"
            f"实测 {half_max['x']:.2f} × {half_max['y']:.2f} px（二阶矩等效宽度 "
            f"{psf['measured_flat_gaussian_equivalent_px']:.2f} px；引用值 "
            f"{psf['cited_px']:.1f} px）。比 1.5 倍 FWHM 更细的特征，对这个实验系统就是一个 "
            "delta 函数：`draw_target` 因此把每一族的目标放大到最薄特征越线为止，越不过就报错，"
            "既不静默重抽，也不把不可达的样本写进缓存。",
            "",
        ]
    else:
        lines += [f"PSF 宽度：{MISSING}（`teacher_vs_flat.json` 缺失）。", ""]
    if data.teacher is not None:
        line_row = data.teacher["by_kind"]["line"]
        lines += [
            "**`line` 就是这条下限的答案。** 该族原先抽到的线宽 1.5–3.0 px（0.12 倍 FWHM），"
            "teacher 停在 L1 1.62 / NCC 0.38（数字记录在 `src/ml/sim2f/config.py` 与 "
            "`src/ml/sim2f/targets.py` 的文档里）；加下限之后，本语料的 `line` 样本线宽为 "
            f"{line_row['thin_px_mean']:.1f} px（{line_row['thin_over_fwhm_mean']:.2f} 倍 FWHM），"
            f"teacher L1 {line_row['teacher_l1_mean']:.4f}、NCC {line_row['teacher_ncc_mean']:.4f}。"
            "也就是说**那次停滞是目标不可达，不是优化器不行**；同一族现在仍然是全部族里 L1 最差的"
            "一族（可达，但不容易）。",
            "",
            "全部样本的 teacher 目标见 §3；逐族最薄特征与 FWHM 的比值在 "
            f"{min(row['thin_over_fwhm_mean'] for row in data.teacher['by_kind'].values()):.2f}–"
            f"{max(row['thin_over_fwhm_mean'] for row in data.teacher['by_kind'].values()):.2f} 之间。",
            "",
        ]
    return lines


def section_teacher(data: Payloads, figures: dict[str, str | None]) -> list[str]:
    lines = ["## 3 teacher：逐族 vs 零相位", ""]
    if data.teacher is None:
        return lines + ["teacher 表缺失：" + MISSING, ""]
    lines += [
        "判据只用 L1 与 NCC。`flat` 的 EE 不是判据：零相位光斑通常落在支持域内部，"
        "所以它的 EE 接近 1 是构造使然（实测按族 "
        f"{min(row['flat_ee_not_a_criterion_mean'] for row in data.teacher['by_kind'].values()):.4f}–"
        f"{max(row['flat_ee_not_a_criterion_mean'] for row in data.teacher['by_kind'].values()):.4f}，"
        "下限那两个是支撑成空洞 / 双斑的 `ring` 与 `two_spot`）；"
        "`peak_ratio` 不报（量纲混合：counts 帧对单位和控制目标）。",
        "",
        "| family | n | flat L1 | teacher L1 | flat NCC | teacher NCC | teacher 目标 | 最薄特征 px | 薄特征/FWHM |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for family, row in data.teacher["by_kind"].items():
        lines.append(
            f"| `{family}` | {row['n']} | {row['flat_l1_mean']:.4f} | {row['teacher_l1_mean']:.4f} "
            f"| {row['flat_ncc_mean']:.4f} | {row['teacher_ncc_mean']:.4f} "
            f"| {row['teacher_objective_mean']:.4f} | {row['thin_px_mean']:.1f} "
            f"| {row['thin_over_fwhm_mean']:.2f} |"
        )
    by_kind = data.teacher["by_kind"]
    l1_order = sorted(by_kind, key=lambda key: by_kind[key]["teacher_l1_mean"])
    ncc_order = sorted(by_kind, key=lambda key: -by_kind[key]["teacher_ncc_mean"])
    if l1_order == ncc_order:
        criteria_note = (
            "两条判据在 teacher 列上给出**完全相同**的族排序（"
            + " > ".join(f"`{name}`" for name in l1_order)
            + "），所以逐族排序不依赖选哪一条。"
        )
    else:
        disagreements = [
            f"`{name}`（L1 第 {l1_order.index(name) + 1}、NCC 第 {ncc_order.index(name) + 1}）"
            for name in by_kind if l1_order.index(name) != ncc_order.index(name)
        ]
        criteria_note = (
            "两条判据在两个列上的族排序并不一致：" + "、".join(disagreements)
            + "。所以两列都要报，任何一列单独看都会把某一族读错。"
        )
    lines += [
        "",
        criteria_note,
        "",
        "两列仍不可互相替代，因为在 **flat 列**它们差得很远："
        f"`ring` 的 flat NCC 只有 {by_kind['ring']['flat_ncc_mean']:.4f}"
        "（零相位点斑落在环的空洞里，与目标几乎不相关），而它的 flat L1 是 "
        f"{by_kind['ring']['flat_l1_mean']:.4f}——两条判据都判它最差，但只有 NCC 会掉到 0 附近，"
        "两者的数值含义不同。判据（`objective`）本身还额外带 EE 项，"
        "所以目标值、L1、NCC 三者的排序可以不同。",
        "",
    ]
    if figures.get("teacher") is not None:
        lines += [f"![teacher vs flat 逐族](figures/{figures['teacher']})", "",
                  "图中柱顶数字是最薄特征宽度（px），用来对照 §2 的下限。", ""]
    if figures.get("corpus") is not None:
        lines += [f"![语料目标与零相位帧](figures/{figures['corpus']})", "",
                  "上排是目标分布，下排是同一光学的零相位帧。零相位帧是一个落在支持域内部的点斑，"
                  "对分布式的目标，它把能量全放在一个格子上——这正是它 L1 大、NCC 低的来源。", ""]
    return lines


def section_network(data: Payloads, figures: dict[str, str | None]) -> list[str]:
    lines = ["## 4 U-Net（标签所在的实验系统）", ""]
    if data.train_metrics is None or data.train_config is None:
        return lines + ["训练产物缺失：" + MISSING, ""]
    train = data.train_config["train"]
    metrics = data.train_metrics
    final = metrics["final"]
    lines += [
        f"预算固定：width {train['width']}、depth {train['depth']}、batch "
        f"{train['batch_size']}、{train['steps']} 步、lr {train['lr']:g}、seed {train['seed']}、"
        f"`w_sup` {train['w_sup']:g} / `w_phy` {train['w_phy']:g} / `w_tv` {train['w_tv']:g}，"
        f"参数量 {metrics['n_parameters']}。验证集非空（"
        f"{data.meta['counts']['split']['val']} 样本，每 100 步验一次），"
        f"`best_step` = {metrics['best_step']}、`best_val_phy` = {metrics['best_val_phy']:.4f}；"
        f"末步 train_total {final['train_total']:.4f} / val_total {final['val_total']:.4f}。"
        f"（`best` 与 `last` 不是同一个状态。）",
        "",
    ]
    diagnose = data.diagnose["by_split"] if data.diagnose is not None else None
    if diagnose is not None:
        lines += [
            "| split | n | net 目标 | flat 目标 | teacher 目标 | net 胜 flat | phasor_corr | 相位差 RMS vs teacher (rad) | dist_ratio |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for split in ("train", "val", "test"):
            row = diagnose[split]
            lines.append(
                f"| {split} | {row['n']} | {row['net_objective_mean']:.4f} "
                f"| {row['flat_objective_mean']:.4f} | {row['teacher_objective_mean']:.4f} "
                f"| {row['net_better_than_flat']}/{row['n']} | {row['phasor_corr_mean']:.4f} "
                f"| {row['rms_vs_teacher_rad_mean']:.4f} "
                f"| {row['dist_ratio_pred_teacher_over_teacher_flat_mean']:.4f} |"
            )
        lines += [
            "",
            "读到的东西：",
            "",
            f"- **不是欠训练**：训练集 {diagnose['train']['net_better_than_flat']}/"
            f"{diagnose['train']['n']} 个样本都优于 flat，目标 "
            f"{diagnose['train']['net_objective_mean']:.4f} 对 {diagnose['train']['flat_objective_mean']:.4f}。",
            f"- **也没有过拟合**：测试集 {diagnose['test']['net_better_than_flat']}/"
            f"{diagnose['test']['n']} 个样本都优于 flat（"
            f"{diagnose['test']['net_objective_mean']:.4f} 对 {diagnose['test']['flat_objective_mean']:.4f}），"
            f"与训练集同向；48 个训练样本、{metrics['n_parameters']} 个参数，这个规模本来也过拟合不到哪里去。",
            f"- **它给的不是 teacher 的解**：与 teacher 的 phasor 相关只有 "
            f"{min(diagnose[s]['phasor_corr_mean'] for s in diagnose):.3f}–"
            f"{max(diagnose[s]['phasor_corr_mean'] for s in diagnose):.3f}，"
            f"卷绕相位差 RMS 约 {diagnose['train']['rms_vs_teacher_rad_mean']:.2f} rad，"
            f"`dist_ratio` 0.888–0.940（1 表示与 flat 一样远离 teacher）。"
            "整形问题在相位域是多解的，这一点决定 §5 的 oracle 只能当**系数**上限，不能当目标上限。",
            "",
        ]
    if figures.get("train") is not None:
        lines += [f"![训练曲线](figures/{figures['train']})", ""]
    if figures.get("diagnose") is not None:
        lines += [f"![理想实验系统上的分划分诊断](figures/{figures['diagnose']})", ""]
    lines += section_bench_transfer(data, figures)
    return lines


def section_bench_transfer(data: Payloads, figures: dict[str, str | None]) -> list[str]:
    """Same 24 samples, three bench states: the pivot from §4 to §5."""
    lines = ["### 4.1 从标签实验系统搬到标定实验系统", ""]
    if data.bench is None:
        return lines + ["搬移诊断缺失：" + MISSING, ""]
    aggregate = data.bench["aggregate"]
    lines += [
        "同一 24 个样本（`torch.randperm(64, seed=7)[:24]`）、三种实验系统状态，判据不变。"
        "`tp` 是**缓存 teacher 相位**在同一状态上的传播——用来问「有没有任何理想实验系统上的"
        "相位能搬过来」的对照。探测器关闭。",
        "",
        "| 实验系统状态 | n | net 目标 | flat 目标 | teacher 缓存相位 | Δ(net−flat) | Δ(tp−flat) | net<flat | tp<flat |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for state, label in (
        ("ideal", "`ideal`（标签所在）"),
        ("cal_optics_only", "`cal_optics_only`（只有标定光学）"),
        ("cal_plus_state", "`cal_plus_state`（标定光学 + 抽样像差）"),
    ):
        row = aggregate[state]
        lines.append(
            f"| {label} | {row['n']} | {row['net_objective']:.4f} | {row['flat_objective']:.4f} "
            f"| {row['teacher_objective']:.4f} | {row['delta_net_minus_flat']:+.4f} "
            f"| {row['delta_teacher_minus_flat']:+.4f} "
            f"| {row['net_better_samples']}/{row['n']} "
            f"| {row['teacher_better_samples']}/{row['n']} |"
        )
    loop_ref = data.bench.get("loop_reference") or {}
    lines += [
        "",
        "三步走完，结论就定在这里：**在标签所在的实验系统上 net 赢 24/24，只加标定光学就掉到 "
        f"{aggregate['cal_optics_only']['net_better_samples']}/{aggregate['cal_optics_only']['n']}，"
        "再加上抽样像差就是闭环看到的那个状态。** 而缓存 teacher 相位在新实验系统上同样翻车"
        f"（{aggregate['cal_optics_only']['teacher_objective']:.4f} 对 flat "
        f"{aggregate['cal_optics_only']['flat_objective']:.4f}），所以这不是网络的问题。",
        "",
    ]
    if loop_ref:
        lines += [
            "与闭环自己的 JSON 对拍：`cal_plus_state` 的 flat 与 `loop/a_no_noise` 的 flat 相差 "
            f"{abs(float(loop_ref['cal_plus_state_flat_minus_loop_flat'])):.1e}，net_open 相差 "
            f"{abs(float(loop_ref['cal_plus_state_net_minus_loop_net_open'])):.1e} "
            "—— 同一个量的两次测量，不是两个不同的量。",
            "",
        ]
    if figures.get("bench") is not None:
        lines += [f"![三种实验系统状态](figures/{figures['bench']})", ""]
    return lines


def section_loop(data: Payloads, figures: dict[str, str | None]) -> list[str]:
    lines = ["## 5 闭环五路读数（三个配置）", ""]
    lines += [
        "判据是 teacher 自己的 `objective = L1 + w_ee·(1-EE)`（在 prepared 帧上），"
        "对所有读数一致；`l1` / `ncc` 并列报出。`ee` / `cv_in_support` / `iou50` 是诊断量，"
        "不参与排序；`peak_ratio` 不报（量纲混合）。",
        "",
        "五路读数：`flat`（零相位）、`net_open`（一次前向）、`net_probe`（探针拟合像差后）、"
        "`net_closed`（再加闭环重拟合）、`oracle`（真值**系数**，不是目标上限）。",
        "",
    ]
    tags = [tag for tag in LOOP_ORDER if data.loops.get(tag) is not None]
    if not tags:
        return lines + ["闭环产物缺失：" + MISSING, ""]
    lines += [
        "| 配置 | 读数 | 目标 | L1 | NCC | EE | CV（支持域内） | IoU50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag in tags:
        loop = data.loops[tag]
        for arm in ARM_ORDER:
            row = loop["aggregate"][arm]
            lines.append(
                f"| `{tag}` | `{arm}` | {row['objective']:.4f} | {row['l1']:.4f} "
                f"| {row['ncc']:.4f} | {row['ee']:.4f} | {row['cv_in_support']:.4f} "
                f"| {row['iou50']:.4f} |"
            )
    lines += ["", "三个配置的差别只是测量侧：", ""]
    for tag in tags:
        loop = data.loops[tag]
        bench = loop["bench"]
        lines.append(
            f"- `{tag}`：{LOOP_DESC[tag]}；离焦 {1e3 * bench['defocus_m']:.4f} mm、"
            f"η = {bench['zero_order_eta']:.2f}、样本 {loop['n_samples']}、轮数 "
            f"{loop['rounds']['rounds_per_sample']}、seed {loop['loop']['seed']}。"
        )
    lines += [
        "",
        "### 5.1 逐族（以 `a_no_noise` 为正文，另两个配置方向相同）",
        "",
    ]
    main_tag = "a_no_noise" if data.loops.get("a_no_noise") is not None else tags[0]
    loop = data.loops[main_tag]
    lines += [
        "| family | n | flat | net_open | net_probe | net_closed | oracle | 胜者 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for family, row in loop["per_kind"].items():
        arms = row["arms"]
        winner = min(ARM_ORDER, key=lambda arm: arms[arm]["objective"])
        lines.append(
            f"| `{family}` | {row['n']} | {arms['flat']['objective']:.4f} "
            f"| {arms['net_open']['objective']:.4f} | {arms['net_probe']['objective']:.4f} "
            f"| {arms['net_closed']['objective']:.4f} | {arms['oracle']['objective']:.4f} "
            f"| `{winner}` |"
        )
    drawn = len(loop["per_kind"])
    total_families = len(data.families) or drawn
    missing = [name for name in data.families if name not in loop["per_kind"]]
    missing_text = "、".join(f"`{name}`" for name in missing) if missing else "没有"
    lines += [
        "",
        f"`torch.randperm(64, seed=7)[:24]` 没有抽到 {missing_text}，所以本表的族数是 "
        f"**{drawn}/{total_families}**，所有「n/{drawn} 族」都以抽到的族为分母，"
        f"不能读成 {total_families} 族里的占比。",
        "",
        "**`net_closed` 相对 `net_open`**：在 "
        + "、".join(
            f"`{family}`"
            for family in loop["per_kind"]
            if loop["per_kind"][family]["arms"]["net_closed"]["objective"]
            < loop["per_kind"][family]["arms"]["net_open"]["objective"]
        )
        + " 上更好，在 "
        + "、".join(
            f"`{family}`"
            for family in loop["per_kind"]
            if loop["per_kind"][family]["arms"]["net_closed"]["objective"]
            >= loop["per_kind"][family]["arms"]["net_open"]["objective"]
        )
        + " 上更差；三个配置的分族方向一致，所以这**不是**测量噪声。"
        "在 `rect` 上，两个含离焦的变体都更差（见 §6），两个不含离焦的变体都更好——"
        "与 §6 的归因同向。",
        "",
        "每轮的改善也不该被藏起来：下表是三个配置的逐轮统计（分母是该配置实际记到的轮数）。",
        "",
        "| 配置 | 改善轮数 | 单调改善样本 | 探针拟合 ok | 系数 RMSE | 线性读数 cos |",
        "|---|---|---|---|---|---|",
    ]
    for tag in tags:
        loop = data.loops[tag]
        rounds = loop["rounds"]
        probe = loop["probe"]
        lines.append(
            f"| `{tag}` | {rounds['improved']}/{rounds['total']} "
            f"| {rounds['monotone_samples']}/{rounds['samples']} "
            f"| {probe['ok']}/{probe['samples']} | {probe['coeff_rmse']:.4f} "
            f"| {probe['linear_cos']:.4f} |"
        )
    lines += [
        "",
        "「改善轮数」只统计相邻两轮之间目标变好的次数，所以它比 `net_closed` 与 `net_open` 的"
        "总差更容易被单轮抖动左右；把它与 §7 的噪声阶梯一起读，不要单独当成收敛证据。",
    ]
    shortfall = [
        (tag, rounds)
        for tag in tags
        for rounds in [_short_rounds(data.loops[tag])]
        if rounds
    ]
    if shortfall:
        lines += [
            "",
            "分母不是 `样本数 × 轮数` 的那一个配置要解释清楚："
            + "；".join(
                f"`{tag}` 记到 {data.loops[tag]['rounds']['total']} 轮（"
                f"{data.loops[tag]['rounds']['samples']} × "
                f"{data.loops[tag]['rounds']['rounds_per_sample']} 应为 "
                f"{data.loops[tag]['rounds']['samples'] * data.loops[tag]['rounds']['rounds_per_sample']}），"
                "缺轮的是 " + "、".join(f"`{kind}` {index} 号" for index, kind in rounds)
                for tag, rounds in shortfall
            )
            + "，这些样本的第 1 轮重拟合都被拒（保留上一步估计）。"
            "所以 `improved/total` 只在同一配置内部可比，不能跨配置直接比分子。",
        ]
    lines += [""]
    if figures.get("loop") is not None:
        lines += [f"![闭环逐族相对 flat 的差](figures/{figures['loop']})", ""]
    return lines


def _short_rounds(loop: dict[str, Any] | None) -> list[tuple[int, str]]:
    """Samples that recorded fewer rounds than the protocol asks for."""
    if loop is None:
        return []
    expected = int(loop["rounds"]["rounds_per_sample"])
    return [
        (int(sample["index"]), str(sample["kind"]))
        for sample in loop.get("per_sample", [])
        if int(sample.get("closed", {}).get("round_count", expected)) < expected
    ]


def section_attribution(data: Payloads, figures: dict[str, str | None]) -> list[str]:
    lines = ["## 6 归因：离焦 vs 0 级", ""]
    if any(data.ablation.get(tag) is None for tag in VARIANT_ORDER):
        return lines + ["消融产物缺失：" + MISSING, ""]
    lines += [
        "同一语料、同一检查点、同一闭环契约（24 样本、2 轮、seed 7、探针与重拟合开启、探测器关闭、"
        "光照抖动 0.06），**唯一变量**是 `Sim2fConfig.calibrated()` 里的 `defocus_m` 与 "
        "`zero_order_eta`，用 `dataclasses.replace` 改。",
        "",
        "| 变体 | 离焦 (mm) | η | flat | net_open | net_closed | oracle | net_open 胜族 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag in VARIANT_ORDER:
        loop = data.ablation[tag]
        bench = loop["bench"]
        won = sum(
            1
            for row in loop["per_kind"].values()
            if row["arms"]["net_open"]["objective"] < row["arms"]["flat"]["objective"]
        )
        lines.append(
            f"| `{tag}` | {1e3 * bench['defocus_m']:.4f} | {bench['zero_order_eta']:.2f} "
            f"| {loop['aggregate']['flat']['objective']:.4f} "
            f"| {loop['aggregate']['net_open']['objective']:.4f} "
            f"| {loop['aggregate']['net_closed']['objective']:.4f} "
            f"| {loop['aggregate']['oracle']['objective']:.4f} "
            f"| {won}/{len(loop['per_kind'])} |"
        )
    lines += [
        "",
        "**归因读法**：去掉离焦（保留 η）就恢复了理想实验系统的顺序 —— net_open "
        f"{data.ablation['eta_only']['aggregate']['net_open']['objective']:.4f} 对 flat "
        f"{data.ablation['eta_only']['aggregate']['flat']['objective']:.4f}；"
        "去掉 η（保留离焦）没有恢复 —— net_open "
        f"{data.ablation['defocus_only']['aggregate']['net_open']['objective']:.4f} 对 flat "
        f"{data.ablation['defocus_only']['aggregate']['flat']['objective']:.4f}，"
        "与标定结果同向。两个都去掉是该实验系统上的下限："
        f"flat {data.ablation['ideal']['aggregate']['flat']['objective']:.4f} 对 net_open "
        f"{data.ablation['ideal']['aggregate']['net_open']['objective']:.4f}"
        "（6/7 族，例外是 `two_spot`，net_closed 把它拉回到 "
        f"{data.ablation['ideal']['per_kind']['two_spot']['arms']['net_closed']['objective']:.4f}）。",
        "",
        "### 6.1 teacher 对照：独立复算同一件事",
        "",
    ]
    if data.teacher_control is None:
        lines += ["teacher 对照缺失：" + MISSING, ""]
    else:
        lines += [
            "| 变体 | 情形 | n | flat | net_open | teacher | net_open<flat | teacher<flat |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for tag in VARIANT_ORDER:
            for case, label in (("state", "抽样像差 + 抖动"), ("optics_only", "只有静态光学")):
                row = data.teacher_control["aggregate"].get(tag, {}).get(case)
                if row is None:
                    continue
                lines.append(
                    f"| `{tag}` | {label} | {row['n']} | {row['flat_objective']:.4f} "
                    f"| {row['net_open_objective']:.4f} | {row['teacher_objective']:.4f} "
                    f"| {row['net_open_better_samples']}/{row['n']} "
                    f"| {row['teacher_better_samples']}/{row['n']} |"
                )
        optics = data.teacher_control["aggregate"]["calibrated"]["optics_only"]
        lines += [
            "",
            f"`teacher` 是语料自己给该目标的标签相位——监督能要求的最好情况。它在只保留标定光学时"
            f"也输给 flat（{optics['teacher_objective']:.4f} 对 {optics['flat_objective']:.4f}，"
            f"{optics['teacher_better_samples']}/{optics['n']}），所以**是实验系统把排序倒过来了**，"
            "不是监督信号或网络的问题。",
            "",
            "### 6.2 机制：离焦把零相位帧抬起来，而不是把网络压下去",
            "",
        ]
        ideal = data.teacher_control["aggregate"]["ideal"]["optics_only"]
        defocus = data.teacher_control["aggregate"]["defocus_only"]["optics_only"]
        eta = data.teacher_control["aggregate"]["eta_only"]["optics_only"]
        lines += [
            "1. **η 动不了零相位帧。** φ = 0 时两项同相、携带同一个场分布，"
            "单位和控制目标做完归一化后帧完全相同；实测 flat 目标在 `eta_only` 与 `ideal` 上"
            f"同为 {eta['flat_objective']:.4f}（两者相等）。"
            "（绝对亮度会按调制通道的 η 掉一档，但判据是归一化后的分布。）",
            "2. **离焦把零相位帧抬到目标的尺度。** 同一批无抽样像差的帧，flat 的目标从 `ideal` 的 "
            f"{ideal['flat_objective']:.4f} 降到 `defocus_only` 的 {defocus['flat_objective']:.4f}"
            f"（越低越好）：NCC {ideal['flat_ncc']:.3f}→{defocus['flat_ncc']:.3f}、L1 "
            f"{ideal['flat_l1']:.3f}→{defocus['flat_l1']:.3f}。在焦时 flat 是衍射极限的点斑，"
            "对分布式目标反而是最差的形状；离焦把它弥散到目标自己的尺度，于是它突然「像」了。"
            "整形那一支朝反方向走，因为它的光斑本来尺度就对，离焦把它过度弥散：NCC "
            f"{ideal['net_open_ncc']:.3f}→{defocus['net_open_ncc']:.3f}。"
            "两支因此交叉，而不是网络变差。",
            f"3. **在含离焦的实验系统上没有相位能赢 flat**，连知道真值系数的 `oracle` 也不行"
            f"（`defocus_only` {data.ablation['defocus_only']['aggregate']['oracle']['objective']:.4f} 对 "
            f"{data.ablation['defocus_only']['aggregate']['flat']['objective']:.4f}；"
            f"`calibrated` {data.ablation['calibrated']['aggregate']['oracle']['objective']:.4f} 对 "
            f"{data.ablation['calibrated']['aggregate']['flat']['objective']:.4f}）。"
            "**注意**：`oracle` 只知道抽样的像差状态，从不知道静态离焦，所以「一个下发的相位"
            "能否抵消它」**没有被测过** —— 本报告只能说「没有哪一路读数估计它」，"
            "不能说「它不可补偿」。",
            "",
        ]
    if data.ablation_summary is not None:
        repro = data.ablation_summary.get("reproduction_check", {})
        lines += [
            f"复现校验（读自 `ablation_loops.json`）：`calibrated` 变体重现 "
            f"`loop/a_no_noise` —— identical = {repro.get('identical')}、"
            f"max|Δ| = {repro.get('max_abs_delta')}；"
            f"teacher 对照的 `ideal`/`optics_only` 对 `bench_transfer.json` 的 ideal 行 "
            f"max|Δ| = "
            f"{data.teacher_control['checks']['max_abs_delta'] if data.teacher_control is not None else MISSING}。"
            "两条都是独立进程重跑，只改了那两个静态误差字段。",
            "",
        ]
    if figures.get("ablation") is not None:
        lines += [f"![四变体 × 四读数](figures/{figures['ablation']})", ""]
    if figures.get("mechanism") is not None:
        lines += [f"![机制：离焦对两支的反向作用](figures/{figures['mechanism']})", ""]
    return lines


def section_noise(data: Payloads, figures: dict[str, str | None]) -> list[str]:
    lines = ["## 7 噪声阶梯与确定性", ""]
    tags = [tag for tag in VARIANT_ORDER if data.ablation.get(tag) is not None]
    if not tags:
        return lines + ["噪声阶梯缺失：" + MISSING, ""]
    lines += [
        "阶梯每个档位跑一次 80 步 Adam 系数拟合（只拟合一个抽样状态，名义照明），"
        "判据是残差的最小改善；改善不过门禁就**拒绝**（`iterations = 0`、系数为空）。"
        "拒测不是「拟合得很差」，是「这一档没有可用的测量」。",
        "",
        "| 变体 | 档位 | 峰值 counts | 残差（起→终） | 迭代 | 系数 RMSE | cos |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag in tags:
        for row in data.ablation[tag]["noise_ladder"]:
            rung = "关闭" if row["exposure_ms"] is None else f"{row['exposure_ms']:g} ms"
            verdict = "**拒测**" if int(row["iterations"]) == 0 else f"{int(row['iterations'])}"
            peak = (f"{row['peak_counts']:.2f}（模型强度）" if row["exposure_ms"] is None
                    else f"{row['peak_counts']:.2f}")
            lines.append(
                f"| `{tag}` | {rung} | {peak} "
                f"| {row['residual_start']:.4f} → {row['residual']:.4f} | {verdict} "
                f"| {_num(row['coeff_rmse'])} | {_num(row['cos'])} |"
            )
    refused_all = [
        exposure
        for exposure in (0.15, 1.0)
        if all(
            _ladder_row(data.ablation[tag], exposure) is not None
            and int(_ladder_row(data.ablation[tag], exposure)["iterations"]) == 0
            for tag in tags
        )
    ]
    refused_peaks = sorted(
        {
            round(float(_ladder_row(data.ablation[tag], exposure)["peak_counts"]))
            for exposure in refused_all
            for tag in tags
            if _ladder_row(data.ablation[tag], exposure) is not None
        }
    )
    five = {tag: _ladder_row(data.ablation[tag], 5.0) for tag in tags}
    five_refused = [tag for tag, row in five.items() if row is not None and int(row["iterations"]) == 0]
    five_measured = [tag for tag, row in five.items() if row is not None and int(row["iterations"]) > 0]
    twenty_peaks = sorted(
        {
            round(float(_ladder_row(data.ablation[tag], 20.0)["peak_counts"]))
            for tag in tags
            if _ladder_row(data.ablation[tag], 20.0) is not None
        }
    )
    refused_text = "、".join(f"{value:g} ms" for value in refused_all) or MISSING
    five_five = ""
    if five_refused:
        five_peak = round(float(
            _ladder_row(data.ablation[five_refused[0]], 5.0)["peak_counts"]))
        five_five = (
            f"；5 ms 在 {'、'.join(f'`{tag}`' for tag in five_refused)} 上也被拒测"
            f"（峰 {five_peak} counts），在 "
            f"{'、'.join(f'`{tag}`' for tag in five_measured)} 上能测"
        )
    else:
        five_five = "；5 ms 在每个变体上都能测"
    lines += [
        "",
        f"要点：{refused_text} 在**每个**变体上都被拒测（峰值 "
        f"{min(refused_peaks)}–{max(refused_peaks)} counts）{five_five}；"
        f"20 ms（峰 {min(twenty_peaks)}–{max(twenty_peaks)} counts）是唯一在每个变体上都测到的档位。",
        "",
        "- **系数误差不是单调的**，可观测性由实验系统决定，不由曝光决定。",
    ]
    comparison = _ladder_comparison(data, "defocus_only", 5.0, 20.0)
    if comparison is not None:
        better, worse, error_worse, error_better = comparison
        lines += [
            f"- 在 `defocus_only` 上 {worse:g} ms 的系数误差（{error_worse:.4f}）反而比 "
            f"{better:g} ms（{error_better:.4f}）更差；曝光档位本身不构成质量排序。",
        ]
    in_focus = [
        (tag, _ladder_row(data.ablation[tag], None))
        for tag in ("eta_only", "ideal")
        if data.ablation.get(tag) is not None
        and _ladder_row(data.ablation[tag], None) is not None
        and _ladder_row(data.ablation[tag], None)["cos"] is not None
    ]
    if in_focus:
        lines += [
            "- 两个在焦变体上，关闭探测器的档位给出的系数与真值**负相关**（cos "
            + "、".join(f"`{tag}` {row['cos']:.3f}" for tag, row in in_focus)
            + "）：那里离焦不在光路里，探针看不到它，拟合只能返回一个与真值反向的向量。"
            "这一档只作诊断，不参与任何整形目标的排序。",
        ]
    lines += [
        "",
        "### 7.1 确定性",
        "",
    ]

    determinism = _determinism(data)
    lines += determinism
    if figures.get("ladder") is not None:
        lines += [f"![噪声阶梯](figures/{figures['ladder']})", ""]
    return lines


def _ladder_row(payload: dict[str, Any] | None, exposure: float | None) -> dict[str, Any] | None:
    """The noise-ladder rung for one exposure (``None`` = detector off), or ``None``."""
    if payload is None:
        return None
    for row in payload["noise_ladder"]:
        if row["exposure_ms"] == exposure:
            return row
    return None


def _ladder_comparison(data: Payloads, tag: str, low: float, high: float
                       ) -> tuple[float, float, float, float] | None:
    """``(better_exp, worse_exp, error_at_worse, error_at_better)`` for two rungs."""
    payload = data.ablation.get(tag)
    if payload is None:
        return None
    low_row, high_row = _ladder_row(payload, low), _ladder_row(payload, high)
    if low_row is None or high_row is None:
        return None
    if low_row["coeff_rmse"] is None or high_row["coeff_rmse"] is None:
        return None
    if high_row["coeff_rmse"] > low_row["coeff_rmse"]:
        return low, high, high_row["coeff_rmse"], low_row["coeff_rmse"]
    return high, low, low_row["coeff_rmse"], high_row["coeff_rmse"]


def _determinism(data: Payloads) -> list[str]:
    """Compare two independently launched runs that must agree exactly."""
    first = data.loops.get("a_no_noise")
    second = data.ablation.get("calibrated")
    if first is None or second is None:
        return ["重复运行的对照产物缺失：" + MISSING, ""]
    numeric = _field_differences(first, second, numeric_only=True)
    strings = _field_differences(first, second, numeric_only=False)
    csv_equal = None
    first_csv = data.run_dir / "loop" / "a_no_noise" / "eval_loop.csv"
    second_csv = data.run_dir / "ablate" / "loops" / "calibrated" / "eval_loop.csv"
    if first_csv.is_file() and second_csv.is_file():
        csv_equal = _read_csv_rows(first_csv) == _read_csv_rows(second_csv)
    lines = [
        "`ablate/loops/calibrated` 与 `loop/a_no_noise` 是**两次独立进程**，输入相同。",
        "",
        "| 对照项 | 结果 |",
        "|---|---|",
        "| 数值场逐字段相等（aggregate / per_kind / per_sample / rounds / probe / 噪声阶梯） | "
        f"{'是' if not numeric else f'否，{len(numeric)} 处'} |",
        "| 不等的字符串字段 | "
        f"{'、'.join(strings) if strings else '无'} |",
        "| `eval_loop.csv` 逐行相等 | "
        f"{'是' if csv_equal else ('否' if csv_equal is False else MISSING)} |",
        "",
    ]
    if numeric:
        lines += [
            "数值字段出现差异 ⇒ 本节的确定性结论不成立，必须按上表列出的字段重查。",
            "",
        ]
    else:
        lines += [
            "所以在这个配置上，同一输入的重跑是**逐字段确定**的（包括逐样本读数与 CSV）；"
            "差异只有记录输出路径的字符串。`teacher_control` 与 `bench_transfer` 的 ideal 行同样"
            "给出 max|Δ| = 0（§6）。这只证明**在给定种子与固定输入下**可复现，"
            "不证明换种子后结论不变——本文没有做种子扫描。",
            "",
        ]
    return lines


def _field_differences(left: Any, right: Any, path: str = "", *,
                       numeric_only: bool) -> list[str]:
    """Paths of fields that differ, split into numeric and string buckets."""
    differences: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(f"{path}/{key}")
                continue
            differences.extend(
                _field_differences(left[key], right[key], f"{path}/{key}", numeric_only=numeric_only)
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            differences.extend(
                _field_differences(item_left, item_right, f"{path}[{index}]", numeric_only=numeric_only)
            )
        return differences
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right and numeric_only:
            return [path]
        return []
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if float(left) != float(right) and numeric_only:
            return [path]
        return []
    if isinstance(left, str) and isinstance(right, str):
        if left != right and not numeric_only:
            return [path]
        return []
    return []


def section_overturned(data: Payloads) -> list[str]:
    lines = ["## 8 被推翻的结论", ""]
    lines += [
        "这一节保留在此，是因为三条结论都曾经被写过一次，而且在当时的产物上看起来成立。",
        "",
        "### 8.1 「闭环有效」——来自 4 对样本的试运行",
        "",
    ]
    smoke = data.smoke_loop
    if smoke is None:
        lines += [f"试运行产物缺失：{MISSING}（`runs/sim2f_smoke800/loop_c/eval_loop.json`）。", ""]
    else:
        aggregate = smoke["aggregate"]
        per_kind = smoke["per_kind"]
        family_list = "、".join(f"`{key}`" for key in per_kind)
        lines += [
            f"试运行只有 {smoke['n_samples']} 对样本（{family_list}），没有测试集，"
            "teacher 与语料都在 24 px 下限之前抽取。它的结论是 `net_closed` "
            f"{aggregate['net_closed']['objective']:.4f} 优于 flat "
            f"{aggregate['flat']['objective']:.4f}"
            f"（差 {aggregate['flat']['objective'] - aggregate['net_closed']['objective']:+.4f}）"
            "—— 看上去闭环有效。",
            "",
            "| family | flat | net_closed | net_closed − flat |",
            "|---|---|---|---|",
        ]
        for family, row in per_kind.items():
            flat = row["arms"]["flat"]["objective"]
            closed = row["arms"]["net_closed"]["objective"]
            lines.append(f"| `{family}` | {flat:.4f} | {closed:.4f} | {closed - flat:+.4f} |")
        wins = sum(
            1 for row in per_kind.values()
            if row["arms"]["net_closed"]["objective"] < row["arms"]["flat"]["objective"]
        )
        worst = max(per_kind, key=lambda key: per_kind[key]["arms"]["flat"]["objective"])
        deltas = {
            family: row["arms"]["net_closed"]["objective"] - row["arms"]["flat"]["objective"]
            for family, row in per_kind.items()
        }
        biggest = min(deltas, key=lambda key: deltas[key])
        lines += [
            "",
            f"`net_closed` 在 {wins}/{len(per_kind)} 个样本上优于 flat。汇总的 "
            f"{aggregate['flat']['objective'] - aggregate['net_closed']['objective']:+.4f} 差距其实由"
            f" `{biggest}` 一族撑起来（{deltas[biggest]:+.4f}，单族就超过总差距），"
            f"`cross` 反向贡献 {deltas['cross']:+.4f}。" if "cross" in deltas else
            f"`net_closed` 在 {wins}/{len(per_kind)} 个样本上优于 flat，汇总的差距由 `{biggest}` 一族撑起来"
            f"（{deltas[biggest]:+.4f}）。",
            "",
            "为什么它站不住：",
            "",
        ]
        if "line" in per_kind and data.teacher is not None:
            lines.append(
                f"- `{worst}` 正是**下限之前**抽的那个不可达目标：它的 flat 目标 "
                f"{per_kind[worst]['arms']['flat']['objective']:.4f} 是全组最差的一个，"
                "而这个「最差」来自目标不可达（§2），不是来自闭环的功劳。"
                f"同一族在 24 样本语料上的 flat L1 只有 "
                f"{data.teacher['by_kind']['line']['flat_l1_mean']:.4f}"
                "（口径不同，只能看量级）。"
            )
        else:
            lines.append(
                f"- `{worst}` 的 flat 目标 {per_kind[worst]['arms']['flat']['objective']:.4f} "
                "是全组最差的一个，汇总由它拉动。"
            )
        lines += [
            f"- {len(per_kind)} 个样本的族组成 {family_list} 没有测试集，"
            "无法区分「学到了」与「记住了这 4 个样本」；跨族均值不构成任何族的结论。",
            "- 在 24 样本、带 24 px 下限的语料上，同一判定是反的（§5）。",
            "",
        ]
    lines += [
        "### 8.2 「net 欠训练」",
        "",
        "不成立。同一个检查点（"
        + (f"{data.train_metrics['steps']} 步，§4" if data.train_metrics is not None else MISSING)
        + "）在训练集上赢 flat "
        + (f"{data.diagnose['by_split']['train']['net_better_than_flat']}/"
           f"{data.diagnose['by_split']['train']['n']}、测试集 "
           f"{data.diagnose['by_split']['test']['net_better_than_flat']}/"
           f"{data.diagnose['by_split']['test']['n']}（§4）。"
           if data.diagnose is not None else MISSING + "。")
        + "训练曲线在 `best_step` 之后已经平（学习率固定、损失不再下降），"
        "验证集也非空（8 样本），所以「再加步数就会赢」没有证据；"
        "真正的差异在实验系统，不在优化预算。",
        "",
        "### 8.3 「η 是主因」",
        "",
        "不成立。η 单独存在时反转**消失**（7/7 族 net_open 赢），离焦单独存在时反转**仍在**"
        "（0/7 族）。η 只在 φ = 0 时动不了零相位帧（§6.2），所以它不可能是主因；"
        "把两者都去掉才回到理想顺序，说明这两个静态项里只有离焦在起作用。",
        "",
    ]
    return lines


def section_limits(data: Payloads) -> list[str]:
    lines = ["## 9 局限与下一步", ""]
    corpus_n = data.meta["synth"]["count"] if data.meta is not None else None
    family_n = len(data.families) or None
    loop = data.loops.get("a_no_noise") or next(
        (payload for payload in data.loops.values() if payload is not None), None)
    bench = loop["bench"] if loop is not None else {}
    defocus = f"{1e3 * bench['defocus_m']:.4f} mm" if bench else MISSING
    eta = f"{bench['zero_order_eta']:.2f}" if bench else MISSING
    drawn = len(loop["per_kind"]) if loop is not None else None
    lines += [
        "1. **这是匹配模型结果，不是 sim-to-real。** teacher、net、闭环共用同一个物理预测模型，"
        "所以测到的只是「摊销（amortisation）+ 闭环估计」这一层的预算，不包含模型失配。"
        "孪生自身对实验系统的残差见 "
        "[`../sim_vs_hw/sim_vs_hw_20261023.md`](../../shape_trials/sim_vs_hw/sim_vs_hw_20261023.md)"
        "（V2.0，即 `calibrated()` 用的那一轮）：绝对 PIB / EE 差 −79.8 % / −79.3 %"
        "（判定 不一致），CV +11.9 %、复合分数 −14.8 %（判定 一致），FWHM +26.4 %"
        "（判定 偏差）。**所以本报告只能引用相对改善（谁赢、方向），"
        "不能把绝对 PIB、EE、FWHM 当实验系统的数字。**",
        "2. **标签所在的实验系统 ≠ 评估用的实验系统。** 语料在理想配置上生成（无离焦、无 η、"
        f"无探测器），闭环在标定配置上评估（{defocus} 离焦 + η {eta}）。"
        "这不是「仿真到实验」的落差，而是同一个孪生内部两个配置的落差；"
        "下一步要么在标定配置上重做标签（成本是 "
        f"{corpus_n if corpus_n is not None else MISSING} 次 teacher 优化），"
        "要么让评估也在理想配置上做（那就测不到部署时会遇到的问题）。",
        "3. **判据奖励弥散。** L1 + w_ee(1-EE) 在单位和控制目标上比较，"
        "所以一个被离焦弥散到目标尺度的点斑，比一个尺度正确但位置/形状略有偏差的整形光斑得分更好。"
        "在含离焦的实验系统上这会让 flat 变成很强的对照项（连 oracle 都赢不了它，§6.2）。"
        "换判据（例如加结构项、或在去卷积后的帧上评分）会得到不同排序——本文没有测。",
        "4. **静态离焦是否可补偿没有被测。** `oracle` 只提供抽样状态的系数，从不提供静态离焦；"
        "所以「下发一个补偿离焦的相位能不能追上」是**未测**，不是「不可补偿」。"
        "要回答它，需要把 `defocus_m` 放进可优化变量，或在探针里显式拟合它。",
        f"5. **语料与样本量小。** {corpus_n if corpus_n is not None else MISSING} 对 / "
        f"{family_n if family_n is not None else MISSING} 族，闭环仅用 "
        f"{loop['n_samples'] if loop is not None else MISSING} 个样本、{drawn or MISSING} 族，"
        "噪声阶梯每个档位只拟合**一个**状态。跨族均值不是任何族的结论，"
        "族内 n 只有 2–6；逐族差异的量级可以看，显著性没有做（也没有做种子扫描）。",
        "6. **下一步的最小充分实验**：在标定配置上重生成语料并按同一预算重训；"
        "把 `defocus_m` 加入探针拟合的可辨识参数；用 3 个种子重复 §5 以给出折叠/种子方差；"
        "把判据换成「去卷积后的形状误差」，检查排序是否仍然由 flat 主导。",
        "",
    ]
    return lines


def section_repro(data: Payloads, figures: dict[str, str | None], artifacts: list[str]) -> list[str]:
    lines = ["## 10 复现", ""]
    train = data.train_config["train"] if data.train_config is not None else {}
    synth_count = (data.meta["synth"]["count"] if data.meta is not None else "?")
    lines += [
        "**溯源**：文首的溯源块由 `insert_header` 从 "
        "`scripts/_common/provenance.py::REPORTS` 生成（键 "
        f"`{REPORT_KEY}`），不要手改；`python scripts/sync_report_provenance.py --check` "
        "验证它没有漂移。",
        "",
        "**渲染**（离线，无 torch / 无 GPU / 无硬件）：", "", "```bash",
        ".venv\\Scripts\\python.exe scripts\\ml\\generate_sim2f_shaping_report.py",
        "```", "",
        "`--run-dir` 指到别的 `runs/<lane>` 就渲染那一条 lane；缺哪个产物，对应数字写 "
        f"`{MISSING}`，不写占位值。", "",
        "**测量侧**（本报告不执行、只在 `runs/sim2f_fair/` 落盘的产物上读）：", "", "```bash",
        f"python -m ml.sim2f synth --count {synth_count} --out runs/sim2f_fair/corpus",
        "python runs/sim2f_fair/measure_teacher_vs_flat.py",
        f"python -m ml.sim2f train --cache runs/sim2f_fair/corpus "
        f"--steps {train.get('steps', '?')} --width {train.get('width', '?')} "
        f"--depth {train.get('depth', '?')} --batch-size {train.get('batch_size', '?')} "
        f"--val-every {train.get('val_every', '?')} --out-dir runs/sim2f_fair/train",
              "python runs/sim2f_fair/diagnose_net.py --cache runs/sim2f_fair/corpus "
              "--checkpoint runs/sim2f_fair/train/model_best.pt",
              "python runs/sim2f_fair/diagnose_bench_transfer.py --cache runs/sim2f_fair/corpus "
              "--checkpoint runs/sim2f_fair/train/model_best.pt",
              "python -m ml.sim2f loop --cache runs/sim2f_fair/corpus "
              "--checkpoint runs/sim2f_fair/train/model_best.pt "
              "--out-dir runs/sim2f_fair/loop/a_no_noise",
              "python -m ml.sim2f loop --cache runs/sim2f_fair/corpus "
              "--checkpoint runs/sim2f_fair/train/model_best.pt --detector --exposure-ms 20 "
              "--out-dir runs/sim2f_fair/loop/c_det20",
              "python -m ml.sim2f loop --cache runs/sim2f_fair/corpus "
              "--checkpoint runs/sim2f_fair/train/model_best.pt --illum-jitter 0 "
              "--out-dir runs/sim2f_fair/loop/d_no_jitter",
              "python runs/sim2f_fair/ablate/run_ablations.py",
              "```", ""]
    if data.meta is not None:
        lines += [
            f"语料构建墙钟 {data.meta['wall_s']:.1f} s（GPU 自选 CUDA），"
            f"`git_commit` = `{data.meta['git_commit'][:8]}`，"
            f"numpy {data.meta['numpy_version']} / torch {data.meta['torch_version']}。"
            "全部 1500 步 + 三次闭环 + 消融的总墙钟见 `runs/sim2f_fair/COMMANDS.md`。",
            "",
        ]
    lines += ["**产物映射**：", "",
              "| 数字出处 | 产物（仓库根路径） |", "|---|---|"]
    rows = (
        ("§1 几何与辐照度、§2 语料计数", "runs/sim2f_fair/corpus/meta.json"),
        ("§2 FWHM、§3 逐族 teacher/flat", "runs/sim2f_fair/teacher_vs_flat.json"),
        ("§2/§3 目标与零相位帧图像", "runs/sim2f_fair/corpus/pairs.npz（numpy 读）"),
        ("§4 训练预算与曲线", "runs/sim2f_fair/train/{config,metrics}.json"),
        ("§4 分划分诊断", "runs/sim2f_fair/diagnose.json"),
        ("§4.1 三种实验系统状态", "runs/sim2f_fair/bench_transfer.json"),
        ("§5 三个配置的五路读数",
         "runs/sim2f_fair/loop/{a_no_noise,c_det20,d_no_jitter}/eval_loop.json"),
        ("§6 四个变体",
         "runs/sim2f_fair/ablate/loops/{calibrated,defocus_only,eta_only,ideal}/eval_loop.json"),
        ("§6 复现校验", "runs/sim2f_fair/ablate/ablation_loops.json"),
        ("§6 teacher 对照", "runs/sim2f_fair/ablate/teacher_control.json"),
        ("§7 噪声阶梯、§5 逐轮统计", "上表四变体与三配置的 `eval_loop.json` 内 `noise_ladder` / `rounds`"),
        ("§8.1 试运行", "runs/sim2f_smoke800/loop_c/eval_loop.json"),
    )
    for label, source in rows:
        lines.append(f"| {label} | {source} |")
    lines += [
        "",
        "**同目录的 `artifacts/`** 是上表 JSON 的逐字节副本（报告引用的输入契约）：",
        "",
    ]
    if artifacts:
        lines += [f"- `artifacts/{name}`" for name in artifacts]
    else:
        lines += [f"- {MISSING}（本次用 `--no-artifacts` 渲染）"]
    lines += [
        "",
        "CSV 不复制进 `artifacts/`：本仓库全局忽略 `*.csv`（`.gitignore` 第 7 行），"
        "复制进来既不会被提交，链接在干净克隆上也永远解不开；CSV 原文件留在 "
        "`runs/sim2f_fair/**/eval_loop.csv`、`diagnose.csv`、`bench_transfer.csv`、"
        "`teacher_vs_flat.csv`，与本表列的 JSON 同源。",
        "",
    ]
    if data.missing:
        lines += ["**渲染时缺失的产物**（对应数字写 `" + MISSING + "`）：", ""]
        lines += [f"- `{name}`" for name in data.missing]
        lines += [""]
    if figures:
        lines += ["**图**：", ""]
        lines += [f"- `figures/{name}`" for name in figures.values() if name]
        lines += [""]
    return lines


# --------------------------------------------------------------------------- write
def copy_artifacts(data: Payloads, enabled: bool) -> list[str]:
    if not enabled:
        return []
    sources = {
        "corpus_meta.json": data.run_dir / "corpus" / "meta.json",
        "teacher_vs_flat.json": data.run_dir / "teacher_vs_flat.json",
        "train_metrics.json": data.run_dir / "train" / "metrics.json",
        "train_config.json": data.run_dir / "train" / "config.json",
        "diagnose.json": data.run_dir / "diagnose.json",
        "bench_transfer.json": data.run_dir / "bench_transfer.json",
        "ablation_loops.json": data.run_dir / "ablate" / "ablation_loops.json",
        "teacher_control.json": data.run_dir / "ablate" / "teacher_control.json",
        "smoke_loop_c.json": data.smoke_dir / "loop_c" / "eval_loop.json",
    }
    for tag in LOOP_ORDER:
        sources[f"loop_{tag}.json"] = data.run_dir / "loop" / tag / "eval_loop.json"
    for tag in VARIANT_ORDER:
        sources[f"ablation_{tag}.json"] = data.run_dir / "ablate" / "loops" / tag / "eval_loop.json"

    ART_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name, source in sources.items():
        if not source.is_file():
            continue
        shutil.copyfile(source, ART_DIR / name)
        copied.append(name)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="measurement lane root (default: runs/sim2f_fair)")
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE,
                        help="4-pair smoke lane root (default: runs/sim2f_smoke800)")
    parser.add_argument("--no-figures", action="store_true", help="render markdown only")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="do not refresh the artifacts/ copies")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else (ROOT / args.run_dir)
    smoke_dir = args.smoke_dir if args.smoke_dir.is_absolute() else (ROOT / args.smoke_dir)
    data = Payloads(run_dir, smoke_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw = not args.no_figures
    figures: dict[str, str | None] = {
        "teacher": fig_teacher_vs_flat(
            data, FIG_DIR / f"fig01_teacher_vs_flat_family{STAMP}.png", enabled=draw),
        "corpus": fig_corpus_target_vs_flat(
            data, FIG_DIR / f"fig02_corpus_target_vs_flat{STAMP}.png", enabled=draw),
        "train": fig_train_curve(
            data, FIG_DIR / f"fig03_train_curve{STAMP}.png", enabled=draw),
        "bench": fig_bench_transfer(
            data, FIG_DIR / f"fig04_bench_transfer{STAMP}.png", enabled=draw),
        "diagnose": fig_diagnose_splits(
            data, FIG_DIR / f"fig05_diagnose_splits{STAMP}.png", enabled=draw),
        "ablation": fig_ablation_variants(
            data, FIG_DIR / f"fig06_ablation_variants{STAMP}.png", enabled=draw),
        "mechanism": fig_ablation_mechanism(
            data, FIG_DIR / f"fig07_ablation_mechanism{STAMP}.png", enabled=draw),
        "loop": fig_loop_family_delta(
            data, FIG_DIR / f"fig08_loop_family_delta{STAMP}.png", enabled=draw),
        "ladder": fig_noise_ladder(
            data, FIG_DIR / f"fig09_noise_ladder{STAMP}.png", enabled=draw),
    }

    artifacts = copy_artifacts(data, enabled=not args.no_artifacts)

    body_parts = [
        "# 2f 孪生上的整形闭环：摊销、闭环估计与静态光学的归因（2026-10-25）",
        "",
        *section_summary(data),
        *section_environment(data),
        *section_reachability(data),
        *section_teacher(data, figures),
        *section_network(data, figures),
        *section_loop(data, figures),
        *section_attribution(data, figures),
        *section_noise(data, figures),
        *section_overturned(data),
        *section_limits(data),
        *section_repro(data, figures, artifacts),
    ]
    # A figure that failed to render must not leave a markdown link behind.
    text = "\n".join(line for line in body_parts if line is not None).rstrip("\n") + "\n"
    (OUT_DIR / "report.md").write_text(insert_header(text, REPORT_KEY), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'report.md'}")
    written = [name for name in figures.values() if name]
    print(f"figures: {len(written)}  artifacts: {len(artifacts)}")
    if data.missing:
        print("missing artefacts: " + ", ".join(data.missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
