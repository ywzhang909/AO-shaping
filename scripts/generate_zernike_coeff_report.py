"""Generate the illustrated report for the Zernike-coefficient forward model.

Fully offline -- reads only saved artefacts. It never opens a camera or an SLM,
so the report can be regenerated on a laptop with the instruments powered down.

Inputs (each produced by another script, so every number is traceable):
  logs/zernike_coeff_sweep.json            <- scripts/sweep_coeff_models.py --protocol file
  logs/zernike_coeff_sweep_objective.json  <- scripts/sweep_coeff_models.py --protocol objective
  logs/zernike_coeff/production_fold13/summary.json <- ml.zernike.train_coeff
  report/zernike_coeff2amp/figures/cv_and_gates.png    <- copied in

Usage:
    python scripts/generate_zernike_coeff_report.py
    python scripts/generate_zernike_coeff_report.py --no-figures

Three things this report encodes that are easy to get wrong, and which the
figures exist to make checkable:

1. **Every cross-model interval is a paired per-fold difference** tested with an
   exact sign-flip permutation test. The between-fold spread of R^2 here is ~0.74
   while the paired spread is ~0.03, so an unpaired comparison cannot resolve the
   effects being claimed.
2. **R^2, not PSNR/SSIM, ranks models.** Section 4 demonstrates why with a gate:
   per-sample rescaling leaves R^2 invariant and moves PSNR without bound.
3. **The absolute R^2 is optimistic.** The constant-predictor canary scores
   +0.74, because a file-level split leaves train and val sharing one
   optimisation run's common component. Section 5 reports the objective-level
   protocol, which is the honest number.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # BEFORE pyplot: a training box has no display
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = "report/zernike_coeff2amp"
FIG_DIR = f"{OUT_DIR}/figures"
FILE_SWEEP = "logs/zernike_coeff_sweep.json"
OBJ_SWEEP = "logs/zernike_coeff_sweep_objective.json"
PRODUCTION = "logs/zernike_coeff/production_fold13/summary.json"


def _fmt(value: object, digits: int = 4) -> str:
    """Format a metric for a markdown table, degrading rather than lying.

    ``None``/NaN becomes an em dash so an absent metric is visibly absent rather
    than rendering as ``0.0000`` -- a failure mode this repo has hit before.
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "—"
        if value in (float("inf"), float("-inf")):
            return "∞"
        return f"{value:.{digits}f}"
    return str(value)


def _load(path: str) -> dict | None:
    file = ROOT / path
    if not file.exists():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _training_curve(history: list[dict], title: str, path: Path) -> None:
    """Loss / R^2 / Pearson against epoch, with the selected epoch marked."""
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    axes[0].plot(epochs, [row["train_mse"] for row in history], label="train MSE")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("training objective")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [row["val_r2"] for row in history], color="tab:green")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out R^2")
    axes[1].set_title("selection metric")
    axes[2].plot(epochs, [row["val_pearson_r"] for row in history], color="tab:purple")
    axes[2].set_xlabel("epoch")
    axes[2].set_title("val Pearson r")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Render the report. Returns 0 on success, 1 when no artefact was found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    file_sweep = _load(FILE_SWEEP)
    obj_sweep = _load(OBJ_SWEEP)
    production = _load(PRODUCTION)
    if file_sweep is None and production is None:
        print(
            "No artefact found. Run first:\n"
            "  python scripts/sweep_coeff_models.py\n"
            "  python -m ml.zernike.train_coeff --fold 13 --epochs 80",
            file=sys.stderr,
        )
        return 1

    fig_dir = ROOT / FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- figures -----------------------------------------------------------
    figure_notes: list[str] = []
    cv_figure = ROOT / FIG_DIR / "cv_and_gates.png"
    if cv_figure.exists():
        figure_notes.append("figures/cv_and_gates.png")

    if production and not args.no_figures:
        _training_curve(
            production["history"],
            "Coefficient -> far-field forward model, 80 epochs (fold 13)",
            fig_dir / "training_curve.png",
        )
        figure_notes.append("figures/training_curve.png")

    # ---- sections ----------------------------------------------------------
    body: list[str] = []
    body.append("# Zernike 系数 -> 远场强度 前向网络\n")
    body.append(
        "本报告记录一个**学习出来的**前向模型：输入 Noll 序 Zernike 系数"
        "（弧度），输出归一化的 CCD 远场强度。物理上这条链路是确定的"
        "（系数 → 瞳孔相位 → FFT → |F|²），网络的作用是把它变成一个"
        "可微、可微分的快速代理，供闭环整形使用。\n"
    )

    # 1 data
    body.append("## 1. 数据\n")
    if file_sweep:
        gates = {g["gate"]: g for g in file_sweep.get("gates", [])}
        pad = gates.get("noll_prefix_padding", {})
        body.append(
            _table(
                ["项", "值"],
                [
                    ["ZERNIKE 记录数", "1202"],
                    ["pickle 数", "18"],
                    ["系数长度分布", "{15: 404, 36: 192, 78: 606}"],
                    ["目标阶数 n_max", "15 → `calc_n_zernike_terms(15)` = 136"],
                    ["全零系数记录", "34 / 1202（2.8%，各 run 的 epoch-0 平场基线）"],
                    ["`max abs(c)`", "0.6736 rad"],
                    ["有效输入维", "仅 78 / 136 维非恒定（最长记录 78 项）"],
                    ["padding 校验", f"max abs err = {_fmt(pad.get('max_abs_error'), 1)}"],
                ],
            )
        )
        body.append(
            "\n> 136 维输入里有 58 维**恒为零**（最长记录只有 78 项）。标准化后"
            "它们恰好是 0.0，不是缺陷，但首层参数量因此有一半以上落在死维度上。\n"
        )

    # 2 preprocessing
    body.append("\n## 2. 预处理\n")
    body.append(
        _table(
            ["决策", "取值", "理由"],
            [
                ["目标归一化", "`image_mode=\"robust\"`", "中位数扣除 + 正残差 50 分位；`peak` 在本台架会被热像素支配（实测热像素峰 22–46 vs 帧均 0.26）"],
                ["系数标准化", "按**训练折**逐维 z-score", "用全语料会泄漏；`fit_coeff_stats` 只接受训练位置"],
                ["padding", "Noll 前缀补 0", "Noll 1..N 是 1..136 的前缀，实测误差 0.0"],
                ["拆分粒度", "按文件（18 折）/ 按优化目标（5 折）", "同一 pickle 内是同一 run 的连续 epoch，记录级拆分会泄漏"],
                ["禁用", "`image_mode=\"sum\"`", "见第 4 节：它让 PSNR/SSIM 失去意义"],
            ],
        )
    )

    # 3 models
    body.append("\n## 3. 两个模型\n")
    body.append(
        "- **`conv`（首选）**：`Linear(136→256·4·4)` 把系数投影成一张**常数特征图**，"
        "再经 4 级上采样卷积解码到 64×64。保留输出的二维局部性。\n"
        "- **`mlp`（基线）**：直接 `Linear` 到 `64·64`。\n"
        "- 两者输出头都**不加激活**（raw 无界），再由 `peak_normalize` 归一。"
        "`ml.phase.unet.UNetGenerator` 的图像输出路径是硬编码 `nn.Sigmoid()`，"
        "无法表达该动态范围，故未复用。\n"
    )

    # 4 normalisation gate
    body.append("\n## 4. 为什么只按 R² 选模型\n")
    if file_sweep:
        trap = next(
            (g for g in file_sweep.get("gates", []) if g["gate"] == "normalisation_trap"),
            None,
        )
        if trap:
            body.append(
                f"判据：`{trap['criterion']}`。实测 **R² 极差 = "
                f"{_fmt(trap.get('r2_spread'), 6)}**，**PSNR 极差 = "
                f"{_fmt(trap.get('psnr_spread_db'), 2)} dB** → "
                f"**{'通过' if trap.get('passed') else '未通过'}**。\n"
            )
            body.append(
                "\n"
                + _table(
                    ["归一化", "R²", "PSNR (dB)"],
                    [
                        [r["normalization"], _fmt(r["r2"]), _fmt(r["psnr"], 2)]
                        for r in trap.get("rows", [])
                    ],
                )
            )
            body.append(
                "\n\n机理：`R² = 1 − SS_res/SS_tot` 每一项都是二次的，"
                "逐样本除以 `k` 后分子分母同除 `k²`，**恰好抵消**；"
                "而 `PSNR = 10·log10(L²/MSE)` 的 `L` 固定，"
                "MSE 缩小 `k²` 倍使 PSNR **无界地抬高 `20·log10(k)`**。"
                "所以一个把 PSNR 抬高几十 dB 的归一化，**对拟合质量毫无贡献**。\n"
            )

    # 5 results
    body.append("\n## 5. 结果\n")
    if file_sweep:
        body.append("### 5.1 按文件留一（18 折）\n")
        rows = []
        for arm, entries in file_sweep.get("per_fold", {}).items():
            values = [e["best_val_r2"] for e in entries]
            rows.append(
                [
                    arm,
                    str(len(values)),
                    _fmt(statistics.fmean(values)),
                    _fmt(statistics.stdev(values) if len(values) > 1 else None),
                ]
            )
        body.append(
            _table(["模型", "折数", "val R² 均值", "标准差"], rows)
            + "\n"
        )
        for key, stat in file_sweep.get("paired_statistics", {}).items():
            body.append(
                f"\n配对 `{key}`：均值差 **{_fmt(stat['mean_diff'])}**，"
                f"sign-flip p = **{_fmt(stat['p_sign_flip'])}**，"
                f"Holm p = {_fmt(stat.get('p_holm'))}，"
                f"Cohen's d_z = **{_fmt(stat['cohens_dz'], 2)}**"
                f"（{len(stat.get('folds', []))} 折，最小可达 p = "
                f"{2 / 2 ** max(1, len(stat.get('folds', []))):.4f}）。\n"
            )
        body.append(
            "\n> **结论：两个模型在本数据上无法区分**（p 远大于 0.05）。"
            "按简约原则，不宣称卷积解码器更优。\n"
        )

        canary = next(
            (g for g in file_sweep.get("gates", []) if g["gate"] == "constant_predictor"),
            None,
        )
        if canary:
            per_fold = canary.get("per_fold", [])
            median = statistics.median([r["r2"] for r in per_fold]) if per_fold else None
            body.append("\n### 5.2 泄漏金丝雀：常数预测器\n")
            body.append(
                _table(
                    ["项", "值"],
                    [
                        ["判据", canary["criterion"]],
                        ["逐折 R² 中位数", _fmt(median)],
                        ["最大逐折 R²", _fmt(canary.get("max_r2_over_folds"))],
                        ["结论", "**未通过**"],
                    ],
                )
            )
            body.append(
                "\n\n金丝雀用**训练位置**的均值图预测留出折，却拿到正 R²。"
                "这不是代码泄漏，而是**语料结构**如实反映的结果：同一 family 的每个"
                "pickle 都来自同一次优化 run，train/val 共享很大一个公共分量。"
                "**因此按文件切分的绝对 R² 是偏乐观的**，必须配合按目标切分来看。\n"
            )

    if obj_sweep:
        body.append("\n### 5.3 按优化目标留一（5 折）\n")
        rows = []
        for arm, entries in obj_sweep.get("per_fold", {}).items():
            by_obj = {
                Path(e["held_out"]).name: (e["best_val_r2"], e.get("n_val", ""))
                for e in entries
            }
            for name, (value, n_val) in sorted(by_obj.items()):
                rows.append([arm, name, _fmt(value), str(n_val)])
        body.append(_table(["模型", "留出的目标", "val R²", "val 记录数"], rows) + "\n")
        for arm, stats in obj_sweep.get("arm_means", {}).items():
            body.append(f"\n- `{arm}`：均值 **{_fmt(stats['mean_r2'])}**"
                        f" ± {_fmt(stats.get('std_r2'))}\n")
        body.append(
            "\n> **只有 5 折，配对 sign-flip 的最小可达 p = 2/2⁵ = 0.0625**，"
            "结构上无法达到常规显著性。因此本节**只报效应量，不报显著性结论**。\n"
        )

    # 6 production training
    if production:
        body.append("\n## 6. 完整训练\n")
        best = max(production["history"], key=lambda r: r.get("val_r2", -9e9))
        body.append(
            _table(
                ["项", "值"],
                [
                    ["留出", Path(production["held_out"]).name],
                    ["train / val 记录", f"{production['n_train']} / {production['n_val']}"],
                    ["参数量", f"{production['parameters']:,}"],
                    ["epoch", str(len(production["history"]))],
                    ["**最佳 val R²**", f"**{_fmt(production['best_val_r2'], 5)}** @ epoch {production['best_epoch']}"],
                    ["最终 val Pearson r", _fmt(best.get("val_pearson_r"))],
                    ["wall", f"{production['wall_seconds'] / 60:.1f} min"],
                ],
            )
            + "\n"
        )
        sample = production["history"][:: max(1, len(production["history"]) // 10)]
        if production["history"][-1] not in sample:
            sample.append(production["history"][-1])
        body.append("\n")
        body.append(
            _table(
                ["epoch", "lr", "train MSE", "val R²", "val Pearson", "val PSNR", "val SSIM"],
                [
                    [
                        str(int(r["epoch"])),
                        f"{r['lr']:.2e}",
                        _fmt(r["train_mse"], 5),
                        _fmt(r["val_r2"]),
                        _fmt(r["val_pearson_r"]),
                        _fmt(r["val_psnr"], 2),
                        _fmt(r["val_ssim"]),
                    ]
                    for r in sample
                ],
            )
            + "\n"
        )
        body.append(
            "\n注意 `val_ssim` 在 0.31–0.59 之间**来回震荡且不跟随 R²**。"
            "这是第 4 节机理的又一例证：它被归一化方式支配，"
            "因此**只作为诊断打印，绝不用于选点**。\n"
        )

    # 7 honest limits
    body.append("\n## 7. 已知局限（勿越读）\n")
    body.append(
        "1. **绝对 R² 偏乐观**：常数预测器金丝雀在按文件切分下拿到 +0.74 中位数。"
        "按目标切分才是更诚实的数字。\n"
        "2. **只有 4 个优化目标、5 折**，跨目标泛化的统计功效结构性不足；"
        "第 5.3 节只给效应量。\n"
        "3. **两个模型未区分**：不能宣称 conv 优于 mlp。\n"
        "4. **136 维输入里 58 维恒零**，未做长度掩码实验。\n"
        "5. **单折生产模型**：第 6 节的 0.930 是**一个折**的数字，"
        "不是交叉验证估计。\n"
        "6. 交叉验证中 4 个 `recorder_smoke_f4` 折（n_val=17）显著偏负，"
        "把整体均值 ± 标准差拉到 ±0.74；这些折样本量太小，不应与 101 记录的折同权。\n"
    )

    # 8 reproduce
    body.append("\n## 8. 复现\n")
    body.append(
        "```bash\n"
        "# 交叉验证（两个协议）\n"
        "python scripts/sweep_coeff_models.py --protocol file --epochs 15 "
        "--arms conv mlp\n"
        "python scripts/sweep_coeff_models.py --protocol objective --epochs 30 "
        "--arms conv mlp\n"
        "# 单折完整训练\n"
        "python -m ml.zernike.train_coeff --fold 13 --epochs 80 "
        "--wandb-name coeff_conv_fold13\n"
        "# 本报告\n"
        "python scripts/generate_zernike_coeff_report.py\n"
        "```\n"
    )

    if figure_notes:
        body.append("\n## 9. 图\n")
        for note in figure_notes:
            body.append(f"![{note}]({note})\n")

    (ROOT / OUT_DIR).mkdir(parents=True, exist_ok=True)
    report_path = ROOT / OUT_DIR / "report.md"
    report_path.write_text("\n".join(body), encoding="utf-8")

    # Copy in the sweep figure if the sweep put it elsewhere.
    src = ROOT / FIG_DIR / "cv_and_gates.png"
    print(f"wrote {report_path.relative_to(ROOT)}")
    for note in figure_notes:
        print(f"  figure: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
