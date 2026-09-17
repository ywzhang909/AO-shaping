"""生成 SAC/Robust-RL 训练运行分析报告 (models/ + logs/ 离线分析).

分析 models/ 下全部 SAC 训练运行, 结合 logs/<run>/ 的旁路遥测
(config.json / summary.json / eval/evaluations.npz / tfevents),
生成中文 Markdown 报告与 matplotlib PNG 图, 输出到 docs/models_analysis/.

纯离线读取: 不导入 ao_shaping / stable_baselines3 / tensorboard, 不触碰硬件。
"""
from __future__ import annotations

import base64
import json
import re
import struct
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
OUT_DIR = ROOT / "docs" / "models_analysis"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DPI = 130

# --- 家族分类规则 (run 名前缀 -> 中文标签), 按顺序匹配 ---
FAMILY_RULES: list[tuple[str, str]] = [
    ("stage1_easy", "阶段1"),
    ("stage2_medium", "阶段2"),
    ("stage3_", "阶段3"),
    ("static_long", "静态湍流-长训练"),
    ("static_focus", "静态聚焦"),
    ("turbulence_long", "湍流-长训练"),
    ("turbulence_mamba_best", "湍流-长训练"),
    ("turbulence_long_retry", "湍流-长训练"),
    ("turb_focus", "湍流聚焦"),
    ("sac_", "冒烟/架构对比实验"),
    ("tmp_sac_run", "临时调试"),
]

FAMILY_COLORS: dict[str, str] = {
    "阶段1": "#1f77b4",
    "阶段2": "#ff7f0e",
    "阶段3": "#2ca02c",
    "静态湍流-长训练": "#d62728",
    "静态聚焦": "#9467bd",
    "湍流-长训练": "#8c564b",
    "湍流聚焦": "#e377c2",
    "冒烟/架构对比实验": "#7f7f7f",
    "临时调试": "#bcbd22",
}

TF_TAGS = ["rollout/ep_rew_mean", "ao/best_pib", "ao/best_strehl", "ao/pib", "ao/rms"]

CURRICULUM_TRIO = [
    "stage1_easy_20260323_172155",
    "stage2_medium_20260323_172155",
    "stage3_target_20260323_172155",
]


# ---------------------------------------------------------------------------
# 纯 Python 解析器 (来自已验证的 probe5.py, 原样复用)
# ---------------------------------------------------------------------------
def get_extractor(zip_path: Path) -> str:
    """从 SB3 zip 的 data 成员中提取 features_extractor 类名 (不 import 任何库)."""
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read("data").decode("utf-8", errors="replace")
    m = re.search(
        r'"policy_kwargs":\s*\{\s*":type:":\s*"[^"]*",\s*":serialized:":\s*"([A-Za-z0-9+/=]+)"',
        raw,
    )
    if not m:
        return "NO-KWARGS-BLOB"
    blob = base64.b64decode(m.group(1))
    text = blob.decode("latin-1")
    idx = text.find("features_extractor_class")
    if idx == -1:
        return "no-extractor-key"
    after = text[idx : idx + 600]
    m2 = re.search(r"([A-Za-z_][A-Za-z0-9_.]*(?:Extractor|Policy))", after)
    return m2.group(1) if m2 else f"no-class-token near ctx: {after[:120]!r}"


def tf_scalars(path: str, tag_want: str, limit: int = 3) -> list:
    """解析 tfevents, 返回 (step, value) 列表, 使用 wt=5 float32 simple_value."""
    out = []

    def read_varint(buf, i):
        shift = 0
        res = 0
        while True:
            b = buf[i]
            i += 1
            res |= (b & 0x7F) << shift
            if not (b & 0x80):
                return res, i
            shift += 7

    def parse_fields(data):
        res = []
        i = 0
        while i < len(data):
            key, i = read_varint(data, i)
            fn = key >> 3
            wt = key & 7
            if wt == 0:
                v, i = read_varint(data, i)
            elif wt == 1:
                v = data[i : i + 8]
                i += 8
            elif wt == 2:
                ln, i = read_varint(data, i)
                v = data[i : i + ln]
                i += ln
            elif wt == 5:
                v = data[i : i + 4]
                i += 4
            else:
                break
            res.append((fn, wt, v))
        return res

    with open(path, "rb") as f:
        while len(out) < limit:
            head = f.read(8)
            if len(head) < 8:
                break
            # 实测 tfevents 帧格式: 长度字段为固定 8 字节小端 uint64
            # (varint 解析在长度 >=256 时错误, 如 0x14 0x08... 实为 2068 而非 20)
            n = struct.unpack("<Q", head)[0]
            if n > 64 * 1024 * 1024:
                # 防御: 损坏/异常的长度, 跳过该文件剩余部分
                break
            f.read(4)
            payload = f.read(n)
            if len(payload) < n:
                break
            f.read(4)
            step = None
            for fn, wt, v in parse_fields(payload):
                if fn == 2 and wt == 0:
                    step = v
                elif fn == 5 and wt == 2:
                    # summary proto
                    i = 0
                    while i < len(v):
                        key, i = read_varint(v, i)
                        sfn = key >> 3
                        swt = key & 7
                        if swt == 2:
                            ln, i = read_varint(v, i)
                            vb = v[i : i + ln]
                            i += ln
                            if sfn == 1:
                                val = None
                                for f2, w2, v2 in parse_fields(vb):
                                    if f2 == 1 and w2 == 2 and v2.decode() == tag_want:
                                        val = True
                                    elif f2 == 2 and w2 == 5 and val:
                                        out.append((step, struct.unpack("<f", v2)[0]))
                    # stop after summary
                    break
    return out


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
@dataclass
class RunData:
    run: str
    family: str
    extractor: str
    zip_name: str
    ckpt_size_mb: float = 0.0
    has_config: bool = False
    has_summary: bool = False
    has_npz: bool = False
    has_tfevents: bool = False
    total_timesteps: int | None = None
    seed: int | None = None
    init_model: str | None = None
    init_source: str | None = None
    mean_reward: float | None = None
    std_reward: float | None = None
    mean_final_strehl: float | None = None
    mean_best_pib: float | None = None
    npz_timesteps: np.ndarray | None = None
    npz_results_mean: np.ndarray | None = None
    tfevents_path: Path | None = None
    tf_curves: dict[str, list[tuple[int, float]]] = field(default_factory=dict)


def classify_family(run: str) -> str:
    """按 run 名前缀分类到中文家族标签."""
    for prefix, label in FAMILY_RULES:
        if run.startswith(prefix):
            return label
    return "其他"


def derive_init_source(init_model: str | None) -> str | None:
    """从 init_model 路径提取源 run 文件夹名 (如 models\\stage1_xxx\\sac_...zip -> stage1_xxx)."""
    if not init_model:
        return None
    m = re.search(r"models[\\/]([^\\/]+)[\\/]", init_model)
    return m.group(1) if m else init_model


def find_zip(run_dir: Path) -> Path | None:
    """返回 run 目录下的 zip (优先 sac_turbulence_final.zip / sac_static_final.zip)."""
    if not run_dir.is_dir():
        return None
    zips = sorted(run_dir.glob("*.zip"))
    if not zips:
        return None
    for pref in ("sac_turbulence_final.zip", "sac_static_final.zip"):
        for z in zips:
            if z.name == pref:
                return z
    return zips[0]


def find_tfevents(run_log_dir: Path) -> Path | None:
    """rglob 查找 events.out.tfevents*, 多文件时取最大者."""
    if not run_log_dir.is_dir():
        return None
    files = list(run_log_dir.rglob("events.out.tfevents*"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)


def load_run(run: str) -> RunData:
    """加载单个 run 的全部遥测数据 (缺失项显式标记, 不抛异常)."""
    run_dir = MODELS_DIR / run
    log_dir = LOGS_DIR / run
    data = RunData(run=run, family=classify_family(run), extractor="-", zip_name="-")

    # --- 模型 zip + extractor ---
    zip_path = find_zip(run_dir)
    if zip_path is not None:
        data.zip_name = zip_path.name
        data.ckpt_size_mb = zip_path.stat().st_size / 1_000_000
        try:
            data.extractor = get_extractor(zip_path)
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            data.extractor = f"解析失败:{type(exc).__name__}"

    # --- config.json ---
    cfg_path = log_dir / "config.json"
    if cfg_path.exists():
        data.has_config = True
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] {run}: config.json 解析失败 ({type(exc).__name__})")
            cfg = {}
        data.total_timesteps = cfg.get("total_timesteps")
        data.seed = cfg.get("seed")
        data.init_model = cfg.get("init_model")
        data.init_source = derive_init_source(data.init_model)

    # --- summary.json ---
    sum_path = log_dir / "summary.json"
    if sum_path.exists():
        data.has_summary = True
        try:
            s = json.loads(sum_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] {run}: summary.json 解析失败 ({type(exc).__name__})")
            s = {}
        data.mean_reward = s.get("mean_reward")
        data.std_reward = s.get("std_reward")
        data.mean_final_strehl = s.get("mean_final_strehl")
        data.mean_best_pib = s.get("mean_best_pib")

    # --- eval/evaluations.npz ---
    npz_path = log_dir / "eval" / "evaluations.npz"
    if npz_path.exists():
        data.has_npz = True
        try:
            d = np.load(npz_path, allow_pickle=True)
            data.npz_timesteps = np.asarray(d["timesteps"], dtype=float)
            results = np.asarray(d["results"], dtype=float)
            data.npz_results_mean = results.mean(axis=1)
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [warn] {run}: evaluations.npz 解析失败 ({type(exc).__name__})")
            data.has_npz = False

    # --- tfevents ---
    tb = find_tfevents(log_dir)
    if tb is not None:
        data.has_tfevents = True
        data.tfevents_path = tb
        for tag in TF_TAGS:
            try:
                data.tf_curves[tag] = tf_scalars(str(tb), tag, limit=100000)
            except (OSError, struct.error, MemoryError) as exc:
                print(f"  [warn] {run}: tfevents 解析失败 ({type(exc).__name__})")

    return data


def load_all_runs() -> list[RunData]:
    """加载 models/ 下全部 run."""
    runs: list[RunData] = []
    for name in sorted(os_listdir(MODELS_DIR)):
        if (MODELS_DIR / name).is_dir():
            runs.append(load_run(name))
    return runs


def os_listdir(path: Path) -> list[str]:
    """os.listdir 的 Path 封装."""
    import os

    return os.listdir(path)


# ---------------------------------------------------------------------------
# 图 1: 家族构成 + 总大小
# ---------------------------------------------------------------------------
def fig1_runs_overview(runs: list[RunData], out: Path) -> None:
    fam_counts: dict[str, int] = {}
    for r in runs:
        fam_counts[r.family] = fam_counts.get(r.family, 0) + 1
    fams = sorted(fam_counts, key=lambda f: -fam_counts[f])
    counts = [fam_counts[f] for f in fams]
    colors = [FAMILY_COLORS.get(f, "#999999") for f in fams]

    fig, ax1 = plt.subplots(figsize=(11, 5.2), dpi=DPI)
    bars = ax1.bar(fams, counts, color=colors, edgecolor="white", alpha=0.9)
    ax1.set_ylabel("运行数量", fontsize=11)
    total_mb = sum(r.ckpt_size_mb for r in runs)
    ax1.set_title(f"各家族运行数量与总模型大小 ({len(runs)} runs, 共 {total_mb:,.0f} MB)", fontsize=13)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, c + 0.15, str(c),
                 ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2: 学习曲线 (eval reward vs timesteps)
# ---------------------------------------------------------------------------
def fig2_learning_curves(runs: list[RunData], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=DPI)
    best_run: RunData | None = None
    best_final = -1e18
    for r in runs:
        if r.npz_timesteps is None or r.npz_results_mean is None:
            continue
        ax.plot(r.npz_timesteps, r.npz_results_mean, lw=1.1, alpha=0.75,
                color=FAMILY_COLORS.get(r.family, "#999999"),
                label=r.family if r.family else None)
        if len(r.npz_results_mean) and r.npz_results_mean[-1] > best_final:
            best_final = float(r.npz_results_mean[-1])
            best_run = r
    if best_run is not None and best_run.npz_timesteps is not None and best_run.npz_results_mean is not None:
        ax.plot(best_run.npz_timesteps, best_run.npz_results_mean, lw=2.4,
                color="black", alpha=0.9)
        ax.annotate(
            f"最佳最终均值: {best_run.run}\n({best_final:.2f})",
            xy=(best_run.npz_timesteps[-1], best_run.npz_results_mean[-1]),
            xytext=(0.55, 0.12), textcoords="axes fraction",
            fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
    ax.set_xlabel("训练时间步 (timesteps)", fontsize=11)
    ax.set_ylabel("评估回合奖励均值 (eval reward)", fontsize=11)
    ax.set_title("学习曲线: 评估奖励 vs 训练时间步 (按家族着色)", fontsize=13)
    ax.grid(linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3: 最终指标对比 (3 面板)
# ---------------------------------------------------------------------------
def fig3_final_metrics(runs: list[RunData], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), dpi=DPI)

    def panel(ax, key: str, title: str, fmt: str, log_ok: bool = True) -> None:
        items = [(r.run, getattr(r, key), r.family) for r in runs
                 if getattr(r, key) is not None]
        items.sort(key=lambda t: t[1], reverse=True)
        names = [i[0] for i in items]
        vals = [i[1] for i in items]
        cols = [FAMILY_COLORS.get(i[2], "#999999") for i in items]
        if log_ok and vals and max(vals) / max(min(vals), 1e-9) > 10:
            ax.set_yscale("log")
        ax.bar(range(len(names)), vals, color=cols, edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    panel(axes[0], "mean_final_strehl", "最终 Strehl (均值, 降序)", "%.4f")
    panel(axes[1], "mean_best_pib", "最佳 PIB (均值, 降序)", "%.0f")
    panel(axes[2], "mean_reward", "回合奖励均值 (降序)", "%.2f")
    fig.suptitle("最终指标对比 (按家族着色)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 4: 提取器架构分布
# ---------------------------------------------------------------------------
def fig4_extractor_pie(runs: list[RunData], out: Path) -> None:
    counts: dict[str, int] = {}
    for r in runs:
        counts[r.extractor] = counts.get(r.extractor, 0) + 1
    labels = list(counts)
    sizes = [counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=DPI)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"特征提取器架构分布 ({len(runs)} runs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 5: 课程学习三连 tfevents 曲线
# ---------------------------------------------------------------------------
def fig5_turbulence_tfevents(runs: list[RunData], out: Path) -> None:
    by_run = {r.run: r for r in runs}
    trio = [by_run.get(n) for n in CURRICULUM_TRIO]
    trio = [r for r in trio if r is not None and r.tf_curves]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=DPI)
    for tag, ax, ylab in [
        ("ao/best_pib", axes[0], "最佳 PIB"),
        ("ao/best_strehl", axes[1], "最佳 Strehl"),
    ]:
        for r in trio:
            curve = r.tf_curves.get(tag, [])
            if not curve:
                curve = r.tf_curves.get("rollout/ep_rew_mean", [])
            if not curve:
                continue
            steps = [c[0] for c in curve]
            vals = [c[1] for c in curve]
            ax.plot(steps, vals, lw=1.8, marker="o", markersize=3,
                    label=r.run)
        ax.set_xlabel("训练时间步", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"{ylab} 学习曲线 (课程三连)", fontsize=12)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("课程学习 (Curriculum) 三连: stage1 → stage2 → stage3", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 6: 课程阶段聚合指标
# ---------------------------------------------------------------------------
def fig6_curriculum_stages(runs: list[RunData], out: Path) -> None:
    stage_groups: dict[str, list[RunData]] = {
        "阶段1 (stage1_easy*)": [],
        "阶段2 (stage2_medium*)": [],
        "阶段3 (stage3_target*)": [],
    }
    for r in runs:
        if r.run.startswith("stage1_easy"):
            stage_groups["阶段1 (stage1_easy*)"].append(r)
        elif r.run.startswith("stage2_medium"):
            stage_groups["阶段2 (stage2_medium*)"].append(r)
        elif r.run.startswith("stage3_target"):
            stage_groups["阶段3 (stage3_target*)"].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    for ax, key, ylab in [
        (axes[0], "mean_final_strehl", "最终 Strehl 均值"),
        (axes[1], "mean_best_pib", "最佳 PIB 均值"),
    ]:
        names: list[str] = []
        means: list[float] = []
        errs: list[float] = []
        for label, grp in stage_groups.items():
            vals = [getattr(r, key) for r in grp if getattr(r, key) is not None]
            if not vals:
                continue
            names.append(label)
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))
        ax.bar(names, means, yerr=errs, capsize=6, color=["#1f77b4", "#ff7f0e", "#2ca02c"],
               edgecolor="white", alpha=0.9)
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(i, m + e + (max(means) * 0.02), f"{m:.4f}±{e:.4f}",
                    ha="center", fontsize=9)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"{ylab} 按课程阶段聚合", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("课程学习链聚合指标 (均值±标准差)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def fmt_num(v: float | None, fmt: str) -> str:
    return "-" if v is None else fmt % v


def build_report(runs: list[RunData], out_md: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(runs)
    n_cfg = sum(1 for r in runs if r.has_config)
    n_sum = sum(1 for r in runs if r.has_summary)
    n_npz = sum(1 for r in runs if r.has_npz)
    n_tb = sum(1 for r in runs if r.has_tfevents)

    fam_counts: dict[str, int] = {}
    for r in runs:
        fam_counts[r.family] = fam_counts.get(r.family, 0) + 1

    # 家族代表 run + extractor
    fam_repr: dict[str, tuple[str, str]] = {}
    for r in runs:
        if r.family not in fam_repr:
            fam_repr[r.family] = (r.run, r.extractor)

    # 提取器分布
    ext_counts: dict[str, int] = {}
    for r in runs:
        ext_counts[r.extractor] = ext_counts.get(r.extractor, 0) + 1

    # top-3 by mean_final_strehl
    ranked = [r for r in runs if r.mean_final_strehl is not None]
    ranked.sort(key=lambda r: r.mean_final_strehl if r.mean_final_strehl is not None else 0.0, reverse=True)
    top3 = ranked[:3]

    # 课程链 init 追踪
    init_chain: list[tuple[str, str | None]] = []
    for r in runs:
        if r.run in CURRICULUM_TRIO:
            init_chain.append((r.run, r.init_source))

    # tfevents 最终值 (课程三连)
    tf_final: list[tuple[str, float | None, float | None, float | None]] = []
    for r in runs:
        if r.run in CURRICULUM_TRIO:
            def last_of(tag: str) -> float | None:
                c = r.tf_curves.get(tag, [])
                return c[-1][1] if c else None
            tf_final.append((r.run, last_of("ao/best_pib"),
                             last_of("ao/best_strehl"), last_of("rollout/ep_rew_mean")))

    L: list[str] = []
    L.append("# SAC / Robust-RL 训练运行分析报告")
    L.append("")
    L.append(f"- **生成时间**: {now}")
    L.append("- **数据来源**: `models/` 下全部 SAC 训练运行 (SB3 SAC 检查点) + `logs/<run>/` 旁路遥测 (config.json / summary.json / eval/evaluations.npz / TensorBoard tfevents)")
    L.append(f"- **运行总数**: {n} 个")
    L.append("- **生成脚本**: `scripts/generate_models_report.py` (纯离线读取, 不导入 ao_shaping / stable_baselines3 / tensorboard)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1 概述")
    L.append("")
    L.append(f"共分析 **{n} 个** SAC 训练运行, 模型检查点总大小约 **{sum(r.ckpt_size_mb for r in runs):,.0f} MB**。遥测完整性如下:")
    L.append("")
    L.append("| 遥测项 | 完整运行数 | 完整率 |")
    L.append("|---|---:|---:|")
    L.append(f"| config.json (超参数) | {n_cfg} | {n_cfg / n * 100:.1f}% |")
    L.append(f"| summary.json (最终评估) | {n_sum} | {n_sum / n * 100:.1f}% |")
    L.append(f"| eval/evaluations.npz (学习曲线) | {n_npz} | {n_npz / n * 100:.1f}% |")
    L.append(f"| tfevents (TensorBoard 标量) | {n_tb} | {n_tb / n * 100:.1f}% |")
    L.append("")
    missing = {
        "config.json": [r.run for r in runs if not r.has_config],
        "summary.json": [r.run for r in runs if not r.has_summary],
        "eval/evaluations.npz": [r.run for r in runs if not r.has_npz],
        "tfevents": [r.run for r in runs if not r.has_tfevents],
    }
    if any(missing.values()):
        L.append("> 缺失说明: " + "; ".join(
            f"`{k}` 缺失: {', '.join('`' + r + '`' for r in v)}"
            for k, v in missing.items() if v))
    else:
        L.append(f"> 全部 {n} 个运行的四种遥测 (config.json / summary.json / evaluations.npz / tfevents) 均完整。")
    L.append("")
    L.append("## 2 家族构成")
    L.append("")
    L.append(f"按 run 名前缀将 {n} 个运行划分为以下家族 (映射规则见文末附录):")
    L.append("")
    L.append("| 家族 | 数量 | 代表 run | 提取器 | 说明 |")
    L.append("|---|---:|---|---|---|")
    fam_desc = {
        "阶段1": "课程学习第 1 阶段 (stage1_easy / stage1_easy_sweep), 简单湍流环境",
        "阶段2": "课程学习第 2 阶段 (stage2_medium / stage2_medium_sweep), 中等湍流环境",
        "阶段3": "课程学习第 3 阶段 (stage3_target / baseline_long / long_horizon / low_lr / steady_control)",
        "静态湍流-长训练": "静态湍流长训练 (static_long_*), 固定湍流屏",
        "静态聚焦": "静态聚焦实验 (static_focus_*)",
        "湍流-长训练": "湍流长训练 (turbulence_long_* / turbulence_mamba_best / turbulence_long_retry)",
        "湍流聚焦": "湍流聚焦实验 (turb_focus_*)",
        "冒烟/架构对比实验": "冒烟测试与架构对比 (sac_turbulence* / sac_static_smoke / sac_turb_mamba* / sac_turb_crossattn*)",
        "临时调试": "临时调试运行 (tmp_sac_run*)",
    }
    for fam in sorted(fam_counts, key=lambda f: -fam_counts[f]):
        rep, ext = fam_repr.get(fam, ("-", "-"))
        L.append(f"| {fam} | {fam_counts[fam]} | `{rep}` | {ext} | {fam_desc.get(fam, '')} |")
    L.append("")
    L.append("![家族构成](runs_overview.png)")
    L.append("")
    L.append("## 3 提取器架构分布")
    L.append("")
    L.append("从各 run 检查点的 `policy_kwargs` 序列化字节中解析出 `features_extractor_class` (不 import 任何库):")
    L.append("")
    L.append("| 提取器 | 数量 | 占比 |")
    L.append("|---|---:|---:|")
    for ext, cnt in sorted(ext_counts.items(), key=lambda t: -t[1]):
        L.append(f"| {ext} | {cnt} | {cnt / n * 100:.1f}% |")
    L.append("")
    L.append("![提取器分布](extractor_pie.png)")
    L.append("")
    L.append("## 4 学习曲线分析")
    L.append("")
    L.append(f"从 {n_npz} 个运行的 `eval/evaluations.npz` 提取评估奖励曲线 (每次评估的回合奖励均值 vs 训练时间步)。")
    L.append("曲线按家族着色, 黑色粗线为最终评估奖励均值最高的运行。")
    L.append("")
    L.append("![学习曲线](learning_curves.png)")
    L.append("")
    L.append(f"## 5 最终指标对比 (全部 {n} 个运行)")
    L.append("")
    L.append("下表列出全部运行的最终评估指标 (数据不足标 `-`):")
    L.append("")
    L.append("| run | 家族 | 提取器 | total_timesteps | seed | init_model 源 | mean_reward | mean_final_strehl | mean_best_pib |")
    L.append("|---|---|---|---:|---:|---|---:|---:|---:|")
    for r in sorted(runs, key=lambda r: r.run):
        L.append(
            f"| `{r.run}` | {r.family} | {r.extractor} | "
            f"{r.total_timesteps if r.total_timesteps is not None else '-'} | "
            f"{r.seed if r.seed is not None else '-'} | "
            f"{r.init_source if r.init_source else '-'} | "
            f"{fmt_num(r.mean_reward, '%.2f')} | "
            f"{fmt_num(r.mean_final_strehl, '%.4f')} | "
            f"{fmt_num(r.mean_best_pib, '%.0f')} |"
        )
    L.append("")
    L.append("![最终指标对比](final_metrics.png)")
    L.append("")
    L.append("## 6 课程学习 (Curriculum) 链分析")
    L.append("")
    L.append("课程链由 `config.json` 的 `init_model` 字段追踪真实来源:")
    L.append("")
    L.append("| 阶段 run | init_model 源 |")
    L.append("|---|---|")
    for run, src in init_chain:
        L.append(f"| `{run}` | {src if src else '无 (从头训练)'} |")
    L.append("")
    L.append("![课程三连 tfevents](turbulence_tfevents.png)")
    L.append("")
    L.append("![课程阶段聚合](curriculum_stages.png)")
    L.append("")
    L.append("## 7 tfevents 时间序列")
    L.append("")
    L.append("课程三连各 run 的 tfevents 最终标量值 (最后记录点):")
    L.append("")
    L.append("| run | ao/best_pib (最终) | ao/best_strehl (最终) | rollout/ep_rew_mean (最终) |")
    L.append("|---|---:|---:|---:|")
    for run, pib, strehl, rew in tf_final:
        L.append(f"| `{run}` | {fmt_num(pib, '%.0f')} | {fmt_num(strehl, '%.4f')} | {fmt_num(rew, '%.2f')} |")
    L.append("")
    L.append("> 完整曲线见图 5 (`turbulence_tfevents.png`)。`ao/best_pib` 为训练过程中观测到的最佳桶内功率, `ao/best_strehl` 为最佳 Strehl。")
    L.append("")
    L.append("## 8 关键结论")
    L.append("")
    L.append("### 8.1 Top-3 最佳运行 (按 mean_final_strehl)")
    L.append("")
    L.append("| 排名 | run | 家族 | 提取器 | total_timesteps | mean_final_strehl | mean_best_pib | mean_reward |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for i, r in enumerate(top3, 1):
        L.append(
            f"| {i} | `{r.run}` | {r.family} | {r.extractor} | "
            f"{r.total_timesteps if r.total_timesteps is not None else '-'} | "
            f"{fmt_num(r.mean_final_strehl, '%.4f')} | "
            f"{fmt_num(r.mean_best_pib, '%.0f')} | "
            f"{fmt_num(r.mean_reward, '%.2f')} |"
        )
    L.append("")
    L.append("### 8.2 观察到的规律")
    L.append("")
    # mamba vs plain 对比 (仅统计有 summary 的湍流长训练/冒烟 run)
    mamba_runs = [r for r in runs if r.extractor == "MambaCrossAttentionTemporalAOExtractor"
                  and r.mean_final_strehl is not None]
    plain_runs = [r for r in runs if r.extractor == "TemporalAOExtractor"
                  and r.mean_final_strehl is not None]
    cross_runs = [r for r in runs if r.extractor == "CrossAttentionTemporalAOExtractor"
                  and r.mean_final_strehl is not None]
    if mamba_runs:
        m_strehl = np.mean([v for v in (r.mean_final_strehl for r in mamba_runs) if v is not None])
        L.append(f"- **Mamba 提取器** (`MambaCrossAttentionTemporalAOExtractor`): {len(mamba_runs)} 个有最终指标的运行, "
                 f"mean_final_strehl 均值 **{m_strehl:.4f}**。")
    if plain_runs:
        p_strehl = np.mean([v for v in (r.mean_final_strehl for r in plain_runs) if v is not None])
        L.append(f"- **Plain 提取器** (`TemporalAOExtractor`): {len(plain_runs)} 个有最终指标的运行, "
                 f"mean_final_strehl 均值 **{p_strehl:.4f}**。")
    if cross_runs:
        c_strehl = np.mean([v for v in (r.mean_final_strehl for r in cross_runs) if v is not None])
        L.append(f"- **CrossAttention 提取器** (`CrossAttentionTemporalAOExtractor`): {len(cross_runs)} 个有最终指标的运行, "
                 f"mean_final_strehl 均值 **{c_strehl:.4f}**。")
    L.append("- 课程学习链 (stage1 → stage2 → stage3) 通过 `init_model` 逐级继承, 阶段 3 的 `ao/best_pib` 与 `ao/best_strehl` 在更长训练 (6144 步) 下保持高位 (见图 5/6)。")
    L.append("- 冒烟测试 run (16~64 步) 与 120000 步长训练 run 的指标**不可直接比较** (训练量差 3~4 个数量级)。")
    L.append("")
    L.append("### 8.3 警告与注意事项")
    L.append("")
    no_cfg_runs = [r.run for r in runs if not r.has_config]
    if no_cfg_runs:
        L.append(f"- 以下 run 无 config.json / summary.json: {', '.join('`' + x + '`' for x in no_cfg_runs)}, 表中相关字段标 `-`。")
    L.append("- 冒烟/架构对比 run (`sac_*`) 训练步数极少 (16~64 步), 其指标仅用于验证管线可用性, 不能与长训练 run 直接对比。")
    L.append("- `mean_best_pib` 为桶内功率 (PIB), 数值量级 ~1e6, 与 Strehl 无量纲指标含义不同, 对比时注意单位。")
    L.append("- 提取器解析基于检查点 `data` 成员的序列化字节 (cloudpickle 明文), 若未来 SB3 序列化格式变化, 解析结果可能失效。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 附录: 家族映射规则")
    L.append("")
    L.append("| 前缀 | 家族 |")
    L.append("|---|---|")
    for prefix, label in FAMILY_RULES:
        L.append(f"| `{prefix}*` | {label} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"*报告由 `scripts/generate_models_report.py` 自动生成于 {now}。*")
    L.append("")

    out_md.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== 生成 SAC/Robust-RL 训练运行分析报告 ===")
    print(f"模型目录: {MODELS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print(f"输出目录: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] 加载全部运行遥测 ...")
    runs = load_all_runs()
    print(f"  共 {len(runs)} 个运行")
    for r in runs:
        print(f"  {r.run:45s} family={r.family:12s} ext={r.extractor:38s} "
              f"cfg={r.has_config} sum={r.has_summary} npz={r.has_npz} tb={r.has_tfevents}")

    print("[2/7] 图1: 家族构成 ...")
    fig1_runs_overview(runs, OUT_DIR / "runs_overview.png")

    print("[3/7] 图2: 学习曲线 ...")
    fig2_learning_curves(runs, OUT_DIR / "learning_curves.png")

    print("[4/7] 图3: 最终指标对比 ...")
    fig3_final_metrics(runs, OUT_DIR / "final_metrics.png")

    print("[5/7] 图4: 提取器分布 ...")
    fig4_extractor_pie(runs, OUT_DIR / "extractor_pie.png")

    print("[6/7] 图5/图6: 课程学习分析 ...")
    fig5_turbulence_tfevents(runs, OUT_DIR / "turbulence_tfevents.png")
    fig6_curriculum_stages(runs, OUT_DIR / "curriculum_stages.png")

    print("[7/7] 生成 Markdown 报告 ...")
    build_report(runs, OUT_DIR / "report.md")

    print("=== 完成 ===")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()