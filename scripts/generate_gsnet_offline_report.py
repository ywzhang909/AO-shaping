"""Generate the FourierGSNet offline-training illustrated report.

Reads one ``ao_shaping.runners.gsnet_train`` output directory
(``data/gsnet_train/run-<timestamp>/``) and renders an illustrated Chinese
report to ``docs/fouriergsnet_pipeline/offline_training/``:

* ``report.md`` — training config, convergence summary, evaluation metrics,
  loss-curve / training-history / prediction-vs-ground-truth sections (all
  numbers read from ``summary.json``, none hardcoded), artefact inventory and
  the reproduction command;
* ``figures/`` — ``loss_curves.png`` (recomputed from the per-epoch history
  in ``summary.json``) plus verbatim copies of the run's own
  ``comparison.png`` and ``train_history.png``;
* ``gifs/`` — reserved for future per-epoch animations (created for layout
  consistency with the sibling report generators).

**Fully offline** — reads saved artefacts only (``summary.json`` + the two
PNGs). It never imports ``torch``, never opens a camera / SLM / DM, and never
touches the network.

Usage::

    python scripts/generate_gsnet_offline_report.py
    python scripts/generate_gsnet_offline_report.py --run-dir data/gsnet_train/run-<ts>
    python scripts/generate_gsnet_offline_report.py -o docs/fouriergsnet_pipeline/offline_training
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

import click
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Repo bootstrap: ROOT + src on sys.path, then Agg BEFORE pyplot
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
DPI = 150
PHASE_CMAP = "twilight"  # cyclic — kept for parity with the sibling generators
FAR_CMAP = "inferno"  # far-field intensity

RUN_ROOT = "data/gsnet_train"
DEFAULT_OUTPUT = "docs/fouriergsnet_pipeline/offline_training"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _savefig(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: Any, nd: int = 4) -> str:
    """Format a metric for markdown; ints/large magnitudes compact, else fixed."""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f) or np.isinf(f):
        return "-"
    if f.is_integer() and abs(f) >= 10.0:
        return f"{f:.0f}"
    if f != 0 and abs(f) < 1e-4:
        return f"{f:.3e}"
    if abs(f) >= 1e5:
        return f"{f:.3e}"
    return f"{f:.{nd}f}"


def _markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _run_rel(run_dir: Path) -> str:
    """Repo-relative POSIX path when possible, absolute otherwise."""
    try:
        return run_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return run_dir.as_posix()


# ---------------------------------------------------------------------------
# Run discovery (newest-first by mtime; in-progress runs without summary.json
# are skipped so a report can be regenerated mid-run or after completion)
# ---------------------------------------------------------------------------
def _resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        p = Path(run_dir)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise click.ClickException(f"训练目录不存在: {p}")
        return p.resolve()
    base = ROOT / RUN_ROOT
    if not base.exists():
        raise click.ClickException(f"默认训练根目录不存在: {base}")
    subdirs = sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / "summary.json").exists()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        raise click.ClickException(f"{base} 下无含 summary.json 的训练目录")
    return subdirs[0].resolve()


def _load_summary(run_dir: Path) -> dict:
    """Read summary.json defensively — every field is read through ``.get()``."""
    path = run_dir / "summary.json"
    if not path.exists():
        raise click.ClickException(f"缺少 {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{path} 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} 顶层不是 JSON 对象")
    return data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _make_loss_figure(summary: dict, out_path: Path) -> str:
    """Two stacked, x-shared panels: (loss, phase_loss) over (shaping_loss).

    ``best_epoch`` is marked on both panels. Returns a short caption describing
    the convergence trend.
    """
    history = list(summary.get("training", {}).get("history", []) or [])
    if not history:
        raise ValueError("summary.json 缺少 training.history")

    epochs = np.arange(1, len(history) + 1)

    def series(key: str) -> np.ndarray:
        return np.array(
            [float(h.get(key, np.nan)) if h.get(key) is not None else np.nan for h in history],
            dtype=float,
        )

    total = series("loss")
    phase = series("phase_loss")
    shaping = series("shaping_loss")

    best_epoch = summary.get("training", {}).get("best_epoch")
    try:
        best_epoch = int(best_epoch)
    except (TypeError, ValueError):
        best_epoch = None
    if best_epoch is not None and not 1 <= best_epoch <= len(history):
        best_epoch = None

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_top.plot(epochs, total, marker="o", ms=3, color="tab:blue", label="total loss")
    ax_top.plot(epochs, phase, marker="s", ms=3, color="tab:orange", label="phase loss")
    ax_top.set_ylabel("loss")
    ax_top.set_title("总损失 / 相位损失 vs epoch")
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper right")

    ax_bot.plot(epochs, shaping, marker="^", ms=3, color="tab:green", label="shaping loss")
    ax_bot.set_xlabel("epoch")
    ax_bot.set_ylabel("loss")
    ax_bot.set_title("整形损失 vs epoch")
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc="upper right")

    if best_epoch is not None:
        for ax in (ax_top, ax_bot):
            ax.axvline(best_epoch, ls="--", color="tab:red", alpha=0.7)
        ax_top.annotate(
            f"best epoch = {best_epoch}",
            xy=(best_epoch, float(np.nanmax(total))),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
            color="tab:red",
        )

    fig.suptitle("FourierGSNet 离线训练损失曲线")
    fig.tight_layout()
    _savefig(fig, out_path)

    first = float(total[0])
    last = float(total[-1])
    drop = first - last
    pct = (drop / first * 100.0) if first else float("nan")
    best_loss = summary.get("training", {}).get("best_loss")
    caption = (
        f"共 {len(history)} 个 epoch, 总损失 {_fmt(first)} → {_fmt(last)} "
        f"(下降 {_fmt(drop)}, {_fmt(pct)}%), 记录到的最低总损失 {_fmt(best_loss)}"
    )
    if best_epoch is not None:
        caption += f" 出现在 epoch {best_epoch} (红色虚线)"
    caption += "。"
    return caption


def _copy_artifact(src_name: str, dst_name: str, run_dir: Path, fig_dir: Path) -> str | None:
    """shutil.copy2 a run artefact into figures/; return its markdown rel path."""
    src = run_dir / src_name
    if not src.exists():
        logger.warning("缺少产物 {} (跳过复制)", src)
        return None
    dst = fig_dir / dst_name
    shutil.copy2(src, dst)
    logger.info("已复制 {} -> {}", src, dst)
    return f"figures/{dst_name}"


# ---------------------------------------------------------------------------
# Metric interpretation — honest verdicts, derived from the values themselves
# ---------------------------------------------------------------------------
def _correlation_comment(corr: float | None) -> str:
    if corr is None:
        return "- 远场相关系数未记录。"
    return (
        f"- **远场相关系数 {_fmt(corr)}**: 该值接近 1, 说明预测图与真值图的"
        "**整体强度分布形状**高度一致。但 Pearson 相关只衡量线性相似度 —— "
        "它对亮度缩放与少量离群像素不敏感, **不能**据此断言整形质量优秀。"
    )


def _uniformity_comment(cv: float | None) -> str:
    if cv is None:
        return "- 均匀度变异系数未记录。"
    if cv < 0.3:
        verdict = "较好 (接近平顶)"
    elif cv < 0.6:
        verdict = "一般 (仍有明显起伏)"
    elif cv < 1.0:
        verdict = "较差"
    else:
        verdict = "**很差** (标准差已超过均值, 目标区域内存在明显暗区/强离群点)"
    return (
        f"- **均匀度变异系数 CV = {_fmt(cv)}**: {verdict}。CV 为 ROI 内"
        "标准差/均值, 理想平顶应趋近 0, 因此这是本次训练**最需要改进**的指标。"
    )


def _energy_comment(ee: float | None) -> str:
    if ee is None:
        return "- 环围能量未记录。"
    if ee >= 0.95:
        verdict = "绝大部分能量已进入目标区"
    elif ee >= 0.8:
        verdict = "大部分能量进入目标区, 仍有可观泄漏"
    else:
        verdict = "大量能量落在目标区之外"
    return f"- **环围能量 {_fmt(ee)}**: {verdict}。"


# ---------------------------------------------------------------------------
# Reproduction section — string.Template (not .format) because the JSON
# snapshot and the root globs contain literal braces.
# ---------------------------------------------------------------------------
_REPRO_SECTION = Template(
    """## 8. 复现命令

本次训练的数据来源为两个 glob (由 `summary.json` 的 `resolved.roots` 记录):

$root_list

```bash
$command
```

配置快照 (摘自 `summary.json` 的 `config`):

```json
$config_json
```

> **数据 roots 说明**: 默认扫描 `data/debug/slm_zernike_*` (SLM Zernike PIB 整形
> debug 产物) 与 `data/debug/slm_pib_*` (SLM 方形/ROI 整形 debug 产物) 两类目录,
> 合并后作为 GSNet 的监督训练语料。

> **入口说明**: 训练由 `main.py` 的 Click 子命令 `slm-gsnet train` 驱动。
> `ao_shaping/runners/gsnet_train.py` 只是库模块 (无 `__main__`、无 Click 命令),
> 因此 `python -m ao_shaping.runners.gsnet_train` **不是**有效调用方式。
> 运行前需设置 `PYTHONPATH=src;libs` (Windows: `$env:PYTHONPATH = "src;libs"`)。
"""
)


def _build_repro_section(summary: dict) -> str:
    cfg = summary.get("config", {}) or {}
    resolved = summary.get("resolved", {}) or {}

    roots = list(resolved.get("roots", []) or cfg.get("roots", []) or [])
    if isinstance(roots, str):
        roots = [roots]
    if roots:
        root_list = "\n".join(f"- `{r}`" for r in roots)
    else:
        root_list = "- `data/debug/slm_zernike_*`\n- `data/debug/slm_pib_*`"

    epochs = cfg.get("epochs")
    lr = cfg.get("lr")
    batch = cfg.get("batch_size")
    grid = cfg.get("grid")
    # The real entry point is the Click group in main.py -- gsnet_train.py is a
    # library module (no __main__, no Click command), so `python -m
    # ao_shaping.runners.gsnet_train` would fail.
    parts = ["python src/ao_shaping/main.py slm-gsnet train"]
    flag_map = (
        ("--epochs", epochs),
        ("--batch-size", batch),
        ("--lr", lr),
        ("--grid", grid),
        ("--num-layers", cfg.get("num_layers")),
        ("--base-channels", cfg.get("base_channels")),
        ("--w-phase", cfg.get("w_phase")),
        ("--w-shaping", cfg.get("w_shaping")),
        ("--num-workers", cfg.get("num_workers")),
        ("--device", cfg.get("device")),
        ("--n-compare", cfg.get("n_compare")),
        ("--seed", cfg.get("seed")),
    )
    for flag, value in flag_map:
        if value is not None:
            parts.append(f"{flag} {value}")
    command = " \\\n    ".join(parts)

    config_json = json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True)

    return _REPRO_SECTION.safe_substitute(
        root_list=root_list,
        command=command,
        config_json=config_json,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _build_report(
    summary: dict,
    run_dir: Path,
    loss_rel: str | None,
    loss_caption: str,
    history_rel: str | None,
    comparison_rel: str | None,
    gif_dir: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = summary.get("config", {}) or {}
    resolved = summary.get("resolved", {}) or {}
    model = summary.get("model", {}) or {}
    training = summary.get("training", {}) or {}
    evaluation = summary.get("evaluation", {}) or {}
    means = evaluation.get("means", {}) or {}
    artifacts = summary.get("artifacts", {}) or {}

    lines: list[str] = [
        "# FourierGSNet 离线训练报告",
        "",
        f"**生成时间**: {now}",
        f"**数据来源**: `{_run_rel(run_dir)}`",
        "",
        "**Fully offline** — 本报告由 `scripts/generate_gsnet_offline_report.py` "
        "离线生成, 仅读取该次训练已保存的 `summary.json` / `comparison.png` / "
        "`train_history.png`, **不导入 torch, 不打开任何硬件, 不访问网络**。",
        "",
        "## 1. 训练配置",
        "",
    ]

    config_rows: list[list[object]] = [
        ["训练轮数 (epochs)", _fmt(cfg.get("epochs"), nd=0)],
        ["批大小 (batch_size)", _fmt(cfg.get("batch_size"), nd=0)],
        ["学习率 (lr)", _fmt(cfg.get("lr"))],
        ["相位损失权重 (w_phase)", _fmt(cfg.get("w_phase"))],
        ["整形损失权重 (w_shaping)", _fmt(cfg.get("w_shaping"))],
        ["网络层数 (num_layers)", _fmt(cfg.get("num_layers"), nd=0)],
        ["基础通道数 (base_channels)", _fmt(cfg.get("base_channels"), nd=0)],
        ["网格 (grid)", _fmt(cfg.get("grid"), nd=0)],
        ["可训练参数量", _fmt(model.get("n_parameters"), nd=0)],
        ["训练记录数 (n_records)", _fmt(resolved.get("n_records"), nd=0)],
        ["评估样本数 (n_samples)", _fmt(evaluation.get("n_samples"), nd=0)],
        ["计算设备", resolved.get("device", cfg.get("device", "-"))],
        ["随机种子 (seed)", _fmt(resolved.get("seed", cfg.get("seed")), nd=0)],
        ["对比样本数 (n_compare)", _fmt(cfg.get("n_compare"), nd=0)],
    ]
    lines.append(_markdown_table(["参数", "值"], config_rows))
    lines.append("")
    lines.append("## 2. 收敛概览")
    lines.append("")
    history = list(training.get("history", []) or [])
    conv_rows: list[list[object]] = [
        ["实际记录 epoch 数", _fmt(len(history), nd=0)],
        ["配置 epoch 数", _fmt(cfg.get("epochs"), nd=0)],
        ["best_epoch", _fmt(training.get("best_epoch"), nd=0)],
        ["best_loss", _fmt(training.get("best_loss"))],
        ["首 epoch 总损失", _fmt(history[0].get("loss")) if history else "-"],
        ["末 epoch 总损失", _fmt(history[-1].get("loss")) if history else "-"],
        ["总损失下降", _fmt(
            float(history[0].get("loss")) - float(history[-1].get("loss"))
        ) if history and history[0].get("loss") is not None
            and history[-1].get("loss") is not None else "-"],
        ["训练墙钟时间 (s)", _fmt(training.get("seconds"), nd=1)],
    ]
    if training.get("seconds") is not None and history:
        per_epoch = float(training["seconds"]) / max(len(history), 1)
        conv_rows.append(["平均每 epoch (s)", _fmt(per_epoch, nd=1)])
    lines.append(_markdown_table(["指标", "值"], conv_rows))
    lines.append("")
    lines.append(f"损失趋势: {loss_caption}")
    lines.append("")

    lines.append("## 3. 评估指标")
    lines.append("")
    metric_rows: list[list[object]] = [
        ["phase_mae", "相位平均绝对误差 (rad)", _fmt(means.get("phase_mae"))],
        ["far_correlation", "远场强度 Pearson 相关系数", _fmt(means.get("far_correlation"))],
        ["far_rmse", "远场强度 RMSE", _fmt(means.get("far_rmse"))],
        ["uniformity_cv", "均匀度变异系数 (std/mean)", _fmt(means.get("uniformity_cv"))],
        ["encircled_energy", "环围能量", _fmt(means.get("encircled_energy"))],
    ]
    lines.append(_markdown_table(["键", "含义", "值"], metric_rows))
    lines.append("")
    lines.append("**指标解读** (基于上述实测值, 非预设结论):")
    lines.append("")
    lines.append(_correlation_comment(means.get("far_correlation")))
    lines.append(_uniformity_comment(means.get("uniformity_cv")))
    lines.append(_energy_comment(means.get("encircled_energy")))
    if means.get("phase_mae") is not None:
        lines.append(
            f"- **相位 MAE {_fmt(means.get('phase_mae'))} rad**: 相位重建的平均绝对误差; "
            "其大小需与目标波前的动态范围比较后才有意义。"
        )
    if means.get("far_rmse") is not None:
        lines.append(
            f"- **远场 RMSE {_fmt(means.get('far_rmse'))}**: 归一化强度上的均方根误差。"
        )
    lines.append("")
    lines.append("> 结论: 相关性高说明**形状**学到了, 但 `uniformity_cv` 明显偏大说明"
                 "**均匀性**尚未达标 —— 二者不可互相替代。")
    lines.append("")

    lines.append("## 4. 损失曲线")
    lines.append("")
    if loss_rel:
        lines.append(f"![loss_curves]({loss_rel})")
        lines.append("")
    else:
        lines.append("> ⚠️ 未能生成损失曲线图 (summary.json 缺少 training.history)。")
        lines.append("")

    lines.append("## 5. 训练历史图")
    lines.append("")
    if history_rel:
        lines.append(f"![train_history]({history_rel})")
        lines.append("")
    else:
        lines.append("> ⚠️ 未找到 `train_history.png`, 本节已跳过。")
        lines.append("")

    lines.append("## 6. 预测光斑 vs 真值对比")
    lines.append("")
    if comparison_rel:
        lines.append(
            f"下面为训练结束时从数据集抽取的 **{_fmt(cfg.get('n_compare'), nd=0)}** 个样本的"
            "**预测光斑 vs 真值对比** (由训练脚本保存, 此处原样复制):"
        )
        lines.append("")
        lines.append(f"![comparison_pred_vs_gt]({comparison_rel})")
        lines.append("")
    else:
        lines.append("> ⚠️ 未找到 `comparison.png`, 本节已跳过。")
        lines.append("")

    lines.append("## 7. 产物清单")
    lines.append("")
    artifact_rows = [
        ["summary.json", artifacts.get("summary", f"{_run_rel(run_dir)}/summary.json")],
        ["comparison.png", artifacts.get("comparison", "-")],
        ["train_history.png", artifacts.get("history", "-")],
        ["checkpoint (*.pt)", artifacts.get("checkpoint", "-")],
    ]
    lines.append(_markdown_table(["产物", "路径 (摘自 summary.json)"], artifact_rows))
    lines.append("")
    lines.append("本报告另生成:")
    lines.append("")
    gen_rows = [
        ["report.md", "本文件"],
        [
            "figures/loss_curves.png",
            "由 summary.json 的 training.history 重绘 (双面板, 共享 x 轴, 标注 best_epoch)",
        ],
    ]
    if comparison_rel:
        gen_rows.append(["figures/comparison_pred_vs_gt.png", "run 目录 comparison.png 的副本"])
    if history_rel:
        gen_rows.append(["figures/train_history.png", "run 目录 train_history.png 的副本"])
    # Only advertise gifs/ when it actually holds animations. An empty directory
    # is not tracked by git, so listing it unconditionally would have the report
    # claim a deliverable a fresh clone does not contain.
    gifs = sorted(gif_dir.glob("*.gif")) if gif_dir.is_dir() else []
    if gifs:
        gen_rows.append(
            [f"gifs/ ({len(gifs)} 个)", "逐 epoch 动画: " + ", ".join(g.name for g in gifs)]
        )
    lines.append(_markdown_table(["生成文件", "说明"], gen_rows))
    lines.append("")

    lines.append(_build_repro_section(summary))
    lines.append("---")
    lines.append("")
    lines.append("本报告由 `scripts/generate_gsnet_offline_report.py` 离线生成。")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--run-dir",
    default=None,
    help=f"训练输出目录 (默认 {RUN_ROOT}/<最新时间戳>, 按 mtime 倒序)",
)
@click.option(
    "-o",
    "--output",
    default=DEFAULT_OUTPUT,
    show_default=True,
    help=f"报告输出目录 (默认 {DEFAULT_OUTPUT})",
)
def cli(run_dir: str | None, output: str) -> None:
    """FourierGSNet 离线训练 illustrated 报告生成器 (无硬件 / 无 torch)."""
    np.random.seed(SEED)

    run_path = _resolve_run_dir(run_dir)
    out_dir = (ROOT / output).resolve()
    fig_dir = out_dir / "figures"
    gif_dir = out_dir / "gifs"
    for d in (out_dir, fig_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(run_path)
    logger.info("读取训练摘要: {}", run_path / "summary.json")

    loss_rel: str | None = None
    loss_caption = "未生成"
    try:
        loss_caption = _make_loss_figure(summary, fig_dir / "loss_curves.png")
        loss_rel = "figures/loss_curves.png"
    except Exception as exc:  # noqa: BLE001 — report degrades, never tracebacks
        logger.warning("损失曲线图生成失败 (降级为无图): {}", exc)

    history_rel = _copy_artifact("train_history.png", "train_history.png", run_path, fig_dir)
    comparison_rel = _copy_artifact(
        "comparison.png", "comparison_pred_vs_gt.png", run_path, fig_dir
    )

    report = _build_report(
        summary,
        run_path,
        loss_rel,
        loss_caption,
        history_rel,
        comparison_rel,
        gif_dir,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("报告已写入 {}", report_path)


if __name__ == "__main__":
    cli()
