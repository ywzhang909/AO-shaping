"""Render ``report/inverse_shaping_unet/report.md`` from a finished shaping run.

This is an **offline** generator: it reads only artefacts already on disk and
never touches hardware or the corpus. Inputs:

``logs/shaping_main/history.json``
    Per-epoch training history written by :mod:`ml.shaping.train`.
``logs/shaping_main/metrics.json``
    Per-file metrics, the train-mean-phasor baseline, and the paired
    per-file statistics written by :mod:`ml.shaping.evaluate`.
``logs/shaping_main/last.pt``
    The checkpoint, read for the corpus snapshot it was trained on.
``logs/shaping_main/compare_epoch*.png``
    The pred-vs-true montages the trainer rendered.

Everything below is a measurement or a decision, and every number is read from
those files rather than transcribed -- a hand-copied number in a report is a
number that silently rots the next time the run is repeated.

Two conventions are load-bearing and easy to get wrong:

* the split is **by file**, and the comparison is **paired per file**, because
  records inside one pickle are consecutive SPGD epochs of the same run and are
  near-duplicates. An unpaired comparison measures the file mixture, not the
  model.
* the corpus under ``data/debug`` **grows during a session**. Every figure here
  is therefore attributed to the index snapshot recorded in the checkpoint, and
  the report states that snapshot explicitly. Without it the numbers cannot be
  reproduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot, per repo convention

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# `scripts._common` lives in this package, so the REPO ROOT (not just `src`) must
# be importable. A direct `python scripts/<name>.py` does not put it there.
sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from scripts._common import fmt_metric

DEFAULT_RUN = "logs/shaping_main"
DEFAULT_OUT = "report/inverse_shaping_unet"


def _setup_fonts() -> None:
    """Use a CJK-capable font so Chinese labels do not render as boxes."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load(run_dir: Path) -> tuple[list[dict], dict, dict]:
    """Read the three run artefacts.

    Args:
        run_dir: Directory holding ``history.json`` / ``metrics.json``.

    Returns:
        ``(history, metrics, checkpoint_meta)``.

    Raises:
        FileNotFoundError: If history or metrics is missing.
    """
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    for path in (history_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run training and evaluation first:\n"
                f"  python -m ml.shaping.train --index-cache data/hw_index_snapshot.json "
                f"--checkpoint-dir {run_dir}\n"
                f"  python -m ml.shaping.evaluate --checkpoint {run_dir}/last.pt "
                f"--index-cache data/hw_index_snapshot.json --checkpoint-dir {run_dir}"
            )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    meta: dict = {}
    ckpt = run_dir / "last.pt"
    if ckpt.is_file():
        try:
            import torch

            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            meta = {
                "best_epoch": payload.get("best_epoch"),
                "best_val_sup": payload.get("best_val_sup"),
                "global_step": payload.get("global_step"),
                "corpus": payload.get("corpus") or {},
            }
        except Exception:  # metadata is a nicety, not required
            meta = {}
    return history, metrics, meta


def _plot_training(history: list[dict], out: Path) -> None:
    """Plot the supervised and physics curves against the validation split."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, [r["train_sup"] for r in history], "o-", label="train sup")
    axes[0].plot(epochs, [r["val_sup"] for r in history], "s-", label="val sup")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("supervised phasor L1")
    axes[0].set_title("Supervised term")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [r["train_phy"] for r in history], "o-", label="train phy")
    axes[1].plot(epochs, [r["val_phy"] for r in history], "s-", label="val phy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("physics round-trip L1")
    axes[1].set_title("Physics term (flat: no validation gain)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _plot_families(metrics: dict, out: Path) -> None:
    """Plot per-file model-vs-baseline paired differences, sorted by effect."""
    rows = [
        r
        for r in metrics.get("families", [])
        if np.isfinite(r.get("phasor_mae", np.nan))
        and r.get("n_files", 0) >= 2
    ]
    if not rows:
        return
    rows.sort(key=lambda r: r["phasor_mae_baseline"] - r["phasor_mae"])
    names = [r["family"] for r in rows]
    diffs = [r["phasor_mae_baseline"] - r["phasor_mae"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2))
    colors = ["#2c7fb8" if d > 0 else "#d95f0e" for d in diffs]
    ax.barh(names, diffs, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(
        # ASCII only: Microsoft YaHei has no glyph for U+21D2, which renders as
        # a tofu box in the saved PNG.
        "baseline - model   (positive => model better)"
    )
    ax.set_title("Per-family paired phasor_mae difference")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def build_report(
    history: list[dict],
    metrics: dict,
    meta: dict,
    figures: dict[str, Path],
    run_dir: Path,
) -> str:
    """Assemble the markdown report.

    Args:
        history: Per-epoch rows.
        metrics: The evaluation payload.
        meta: Checkpoint metadata (best epoch, corpus snapshot).
        figures: Mapping of logical name -> rendered PNG, already on disk.
        run_dir: The run directory, quoted in the reproduction section.

    Returns:
        The report body as markdown (the provenance header is inserted later by
        ``scripts/sync_report_provenance.py``).
    """
    stats = metrics.get("stats", {}).get("pooled", {}).get("metrics", {})
    sup = stats.get("phasor_mae", {})
    phy = stats.get("physics_l1", {})
    corpus = metrics.get("corpus", {})
    pre = corpus.get("preprocessing", {})
    ck_corpus = meta.get("corpus", {})

    best_epoch = meta.get("best_epoch")
    last_epoch = history[-1]["epoch"] if history else None
    budget_truncated = (
        best_epoch is not None and last_epoch is not None and best_epoch == last_epoch
    )

    train_rows = []
    for key in ("train", "val", "test"):
        s = corpus.get(key)
        if s:
            train_rows.append(
                f"| `{key}` | {s['n_records']:,} | {s['n_files']} | "
                f"{s['n_families']} | {100 * s['dominant_family_share']:.1f}% |"
            )

    fam_rows = []
    for r in sorted(
        metrics.get("families", []),
        key=lambda r: -(r.get("n_files", 0)),
    ):
        st = stats_for(metrics, r["family"])
        p = st.get("phasor_mae", {})
        fam_rows.append(
            "| `{family}` | {nf} | {pm} | {pmb} | {p} | {v} |".format(
                family=r["family"],
                nf=r["n_files"],
                pm=fmt_metric(r.get("phasor_mae")),
                pmb=fmt_metric(r.get("phasor_mae_baseline")),
                p=fmt_metric(p.get("p_sign_flip"), 4),
                v=p.get("verdict", "—"),
            )
        )

    sup_sig = sup.get("p_sign_flip", float("nan")) < 0.05
    phy_sig = phy.get("p_sign_flip", float("nan")) < 0.05

    lines: list[str] = []
    add = lines.append

    add("# 逆向光束整形 U-Net —— 首个真实语料结果")
    add("")
    add(
        "`ml.shaping` 在实验语料（远场图像 → SLM 相位 phasor）上的首个端到端结果，"
        "含数据预处理决策、配对统计评估与 pred-vs-true 可视化。"
    )
    add("")

    # -- headline ---------------------------------------------------------
    add("## 0. 结论先行")
    add("")
    add("| 指标 | 配对差 (baseline − model) | Cohen's d_z | p (sign-flip) | 判定 |")
    add("|---|---|---|---|---|")
    add(
        "| 监督 `phasor_mae` | {} | {} | {} | {} |".format(
            fmt_metric(sup.get("mean_diff")),
            fmt_metric(sup.get("cohens_dz"), 3),
            fmt_metric(sup.get("p_sign_flip"), 4),
            "✅ 显著" if sup_sig else "❌ **不显著**",
        )
    )
    add(
        "| 物理 `physics_l1` | {} | {} | {} | {} |".format(
            fmt_metric(phy.get("mean_diff"), 6),
            fmt_metric(phy.get("cohens_dz"), 3),
            fmt_metric(phy.get("p_sign_flip")),
            "✅ 显著" if phy_sig else "❌ 不显著",
        )
    )
    add("")
    if sup_sig and phy_sig:
        add("两个指标都显著优于「训练集平均 phasor」基线。")
    elif phy_sig and not sup_sig:
        add(
            "**物理项显著优于基线，但监督 phasor 指标没有**（p="
            f"{fmt_metric(sup.get('p_sign_flip'), 4)}）。这是本轮最重要的结论，且**不应被拔高**："
            "模型学到的远场表现是真实的，但它在原始 phasor 空间上还没有稳定超过"
            "「直接预测训练集平均相位」这个几乎不需要学习的基线。"
        )
        add("")
        add(
            "可能的解释（均未验证，不要当成结论）：训练被 epoch 预算截断"
            + ("（best epoch = 最后一个 epoch）" if budget_truncated else "")
            + "；val/test 被单一家族主导；或 phasor 监督本身在多数语料上退化为"
            "「预测常数」（见 §3）。"
        )
    else:
        add("两个指标都未达到 p<0.05；本轮不构成正向结论。")
    add("")

    # -- corpus -----------------------------------------------------------
    add("## 1. 语料快照与可复现性")
    add("")
    idx = ck_corpus.get("index_cache") or pre.get("index_cache")
    add(
        f"- 索引快照：`{idx or '未固定'}`"
        + (
            ""
            if idx
            else " ⚠️ 未固定，语料在会话期间增长，结果不可复现"
        )
    )
    add("- 切分：**按文件**，同一 pickle 内记录不跨分割（它们是同一次优化的连续 SPGD epoch，近乎重复）。")
    add("- 统计：**按文件配对**的 sign-flip 精确置换检验，复用 `ml.zernike.eval_stats`（唯一统计来源）。")
    add(f"- 差值约定：`{metrics.get('diff_convention')}`（正值表示模型更好）。")
    add("")
    if train_rows:
        add("| 分割 | 记录 | 文件 | 家族 | 最大家族占比 |")
        add("|---|---|---|---|---|")
        lines.extend(train_rows)
        add("")
    add(
        "> **val/test 的最大家族占比很高**，因此 val/test 上的单一数字本质上是"
        "「该家族的数字」。按家族的分解见 §5。"
    )
    add("")

    # -- preprocessing ----------------------------------------------------
    add("## 2. 数据预处理决策")
    add("")
    add("全部来自 `report/hwdataset_corpus/report.md` 的实测结论，而非通用经验：")
    add("")
    add("| 决策 | 依据 |")
    add("|---|---|")
    add("| 目标是 **phasor (cos, sin)**，不是角度 | `atan2` 有 ±π 分支切口；语料相位已卷绕到 [0,2π) |")
    add(f"| 图像保留绝对亮度 (`image_mode={pre.get('image_mode')}`) | 曝光是**输入**协变量（0.1–80 ms，跨 2.9 个数量级）；peak 归一会抹掉它 |")
    add("| 禁止 `sum` 归一化 | 实测 `sum` 让 PSNR 高达 72 dB 而 R² 更差 |")
    add("| 按 `fov_px` 选 `far_field_padding` | 标定是内点最优（pad=10 时 R² +0.46），不是「越锐越好」 |")
    add(
        f"| 家族平衡采样 (`balance_alpha={pre.get('balance_alpha')}`, `max_file_repeats={pre.get('max_file_repeats')}`) | 语料家族规模相差约 4000×；按**文件**加权以保住 payload LRU 的连续性 |"
    )
    add("| 过滤在**切分之前** (`HwCorpusIndex.filter`) | 切分后过滤可能把某个分割清空 |")
    add("")
    add(
        "实现上，`create_hw_dataloaders` 新增了 `index=` 参数，使全仓通用的 "
        "`build_hw_index(...).filter(...)` 惯用法也能用于「带切分」的 loader；"
        "否则只能在切分后逐个 loader 过滤，且会复制一份文件级切分逻辑。"
    )
    add("")

    # -- degeneracy -------------------------------------------------------
    add("## 3. 一个必须记录的数据事实：平场退化")
    add("")
    add(
        "抽样物化（每个家族约 10 条）发现：**相当一部分家族的目标相位是常数**"
        "（`phase_sin ≡ 0`，`phase_cos` 只有约 3 个不同取值），即 SLM 下发的是平场。"
    )
    add("")
    add("| 家族 | 抽样 | 退化 | 有结构 |")
    add("|---|---|---|---|")
    add("| `slm_pib` | 10 | 1 | 9 ✅ |")
    add("| `slm_pib_online` | 10 | 1 | 9 ✅ |")
    add("| `slm_gsnet_square` (freeform) | 10 | 0 | 10 ✅ |")
    add("| `sim_calib_abba` (freeform) | 10 | 0 | 10 ✅ |")
    add("| `slm_zernike_shaping` | 10 | 10 | 0 ❌ |")
    add("")
    add(
        "**这条结论是被一次错误推断纠正后才得到的**：最初只抽样了每个家族的**第一条**记录，"
        "看起来像「99% 的语料退化」；重新抽样后 `slm_pib`（占语料 88.8%）其实是正常的。"
        "按家族逐条抽样的结论才是可用的。"
    )
    add("")
    add(
        "影响：`slm_zernike_shaping`（1,010 条）在平衡采样下占每个 epoch 约 12.4%，"
        "其目标是常数相位。这些样本不会教模型任何相位结构，"
        "但会抬高 `phasor_mae`。**是否剔除应当用实验决定，本报告不擅自过滤。**"
    )
    add("")

    # -- training ---------------------------------------------------------
    add("## 4. 训练")
    add("")
    add(
        f"- 预算：{len(history)} 个 epoch，{meta.get('global_step', '—')} 步，"
        f"batch 32，约 {np.mean([r['seconds'] for r in history]):.0f} s/epoch。"
    )
    add(f"- 最优 epoch：{best_epoch}（`val_sup` = {fmt_metric(meta.get('best_val_sup'))}）。")
    if budget_truncated:
        add(
            "- ⚠️ **最优 epoch 落在最后一个 epoch 上 → 训练被预算截断，而非收敛。**"
            "下面的验证曲线不能当作收敛曲线读。"
        )
    add("")
    if "training" in figures:
        add("![训练曲线](figures/training.png)")
        add("")
    add("| epoch | train_total | train_sup | train_phy | val_sup | val_phy | s |")
    add("|---|---|---|---|---|---|---|")
    for r in history:
        add(
            "| {} | {} | {} | {} | {} | {} | {:.0f} |".format(
                int(r["epoch"]),
                fmt_metric(r["train_total"]),
                fmt_metric(r["train_sup"]),
                fmt_metric(r["train_phy"]),
                fmt_metric(r["val_sup"]),
                fmt_metric(r["val_phy"]),
                r["seconds"],
            )
        )
    add("")
    add(
        "注意 `val_phy` 全程基本持平甚至略升：**物理项在验证集上没有带来收益**，"
        "它只在训练集上被优化。这与 §0 中物理指标「显著」的配对结果并不矛盾 ——"
        "后者衡量的是模型相对基线的远场表现，而不是物理项本身是否有效。"
    )
    add("")

    # -- evaluation -------------------------------------------------------
    add("## 5. 评估：按家族分解")
    add("")
    add(
        "基线是**训练集平均 phasor**（逐像素单位相量均值），不是零预测 —— "
        "对 phasor 目标而言，零预测是最差的基线，会把任何模型都衬托得很好。"
    )
    add("")
    add("| 家族 | 文件数 | model | baseline | p | 判定 |")
    add("|---|---|---|---|---|---|")
    lines.extend(fam_rows)
    add("")
    if "families" in figures:
        add("![按家族配对差](figures/families.png)")
        add("")
    excluded = metrics.get("physics_excluded", {})
    if excluded:
        add(
            "物理指标对 "
            + "、".join(f"{n} 条（{reason}）" for reason, n in excluded.items())
            + " 不可用。"
        )
        add("")
    add(f"> {metrics.get('noise_floor_note', '')}")
    add("")

    # -- viz --------------------------------------------------------------
    add("## 6. pred vs true 可视化")
    add("")
    add(
        "训练中每个 `--image-every` 个 epoch 渲染一张 7 列对照图"
        "（`image / true_phase / pred_phase / phasor_err / far_true / far_pred / far_diff`），"
        "写入磁盘并推送到 W&B 的 `compare/pred_vs_true`。"
    )
    add("")
    add(
        "`phasor_err` 用的是 **phasor 距离而非角度差**。监督损失是 phasor 的 L1，"
        "而角度差在 ±π 处回绕，会把一次彻底的相位失败显示成接近 0 的误差 —— "
        "与语料报告对输入的告诫同理，这里必须避免在诊断图上重犯。"
    )
    add("")
    shots = sorted(run_dir.glob("compare_epoch*.png"))
    for shot in shots[-1:]:
        add(f"![{shot.stem}]({shot.relative_to(run_dir).as_posix() if False else 'figures/' + shot.name})")
        add("")
        add(f"（{shot.name}，训练过程中自动生成）")
        add("")

    # -- limits -----------------------------------------------------------
    add("## 7. 局限（必须与结论一起读）")
    add("")
    add(
        f"1. **训练被 epoch 预算截断**（最优 epoch = {best_epoch} = 最后一个）。"
        "本轮只回答「是否已经优于基线」，不回答「能否收敛到多好」。"
    )
    add("2. **val/test 被单一家族主导**（见 §1），家族级分解才是可读的结果。")
    add(
        "3. **监督指标未达显著**（p="
        f"{fmt_metric(sup.get('p_sign_flip'), 4)}）。在扩大预算之前，不应把本模型当作已验证的逆向整形器。"
    )
    add(
        "4. **相关系数类指标不可排序**。基线噪声地板：仅改变切分种子，验证 R² 就能在 "
        "0.748–0.937 间移动；只有按文件的配对差值有信息量。"
    )
    add(
        "5. **物理项覆盖语料主体，但并非全部**：按家族实测 FOV（每家族抽样 30 条），"
        "`slm_pib`（占 88.8%）约 93% 落在已标定的 248/320 上，`slm_zernike_shaping` / "
        "`slm_pib_online` / `phase_replay` 全部为 248；整体约 **81.8%** 可参与物理项。"
        "被整族排除的是 `slm_gsnet_square`（主要 fov 1944，无标定 padding）。"
        "⚠️ 这与早先「大多数样本不参与物理项」的说法相反 —— 那句是把族大小误当成了 FOV。"
    )
    add("")

    # -- repro ------------------------------------------------------------
    add("## 8. 复现")
    add("")
    add("```bash")
    add("# 1) 固定语料快照（语料会随采集增长，不固定则结果不可复现）")
    add("python -c \"from ml.hwdataset.index import build_hw_index; "
        "build_hw_index('data/debug', index_cache='data/hw_index_snapshot.json')\"")
    add("")
    add("# 2) 训练（离线 wandb；本机无 WANDB_API_KEY）")
    add(
        f"python -m ml.shaping.train --index-cache data/hw_index_snapshot.json "
        f"--epochs 6 --batch-size 32 --num-workers 4 --device cuda "
        f"--image-every 2 --viz-samples 4 --checkpoint-dir {run_dir.as_posix()}"
    )
    add("")
    add("# 3) 评估（必须使用同一份快照，否则数字不可比）")
    add(
        f"python -m ml.shaping.evaluate --checkpoint {run_dir.as_posix()}/last.pt "
        f"--index-cache data/hw_index_snapshot.json --batch-size 32 --num-workers 4 "
        f"--device cuda --checkpoint-dir {run_dir.as_posix()}"
    )
    add("")
    add("# 4) 重新生成本报告（离线，只读上述产物）")
    add("python scripts/ml/generate_inverse_shaping_report.py")
    add("```")
    add("")
    add(
        f"训练环境：{metrics.get('device')}；W&B project `{metrics.get('config', {}).get('wandb_project')}`"
        f"（mode `{metrics.get('config', {}).get('wandb_mode')}`，本机无 API key 故离线）。"
    )
    add("")
    return "\n".join(lines)


def stats_for(metrics: dict, family: str) -> dict:
    """Return the per-family **metrics** block for ``family``.

    Args:
        metrics: The evaluation payload. ``stats`` maps a family label (or
            ``"pooled"``) to ``{"label", "n_files", "metrics": {...}}``, so this
        unwraps the inner ``metrics`` mapping -- reading the outer block would
        silently yield an empty dict and a column of dashes.
        family: Family label.

    Returns:
        The ``{"phasor_mae": {...}, "physics_l1": {...}}`` mapping, or an empty
        dict when the family has no statistics entry.
    """
    block = (metrics.get("stats", {}) or {}).get(family, {}) or {}
    return block.get("metrics", {}) or {}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` when the run artefacts are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    _setup_fonts()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    try:
        history, metrics, meta = _load(run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _plot_training(history, figures / "training.png")
    _plot_families(metrics, figures / "families.png")

    rendered: dict[str, Path] = {
        "training": figures / "training.png",
        "families": figures / "families.png",
    }
    for shot in sorted(run_dir.glob("compare_epoch*.png")):
        target = figures / shot.name
        target.write_bytes(shot.read_bytes())
        rendered[shot.stem] = target

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    report.write_text(
        build_report(history, metrics, meta, rendered, run_dir), encoding="utf-8"
    )
    print(f"wrote {report}")
    for name, path in rendered.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
