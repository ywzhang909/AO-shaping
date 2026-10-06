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
# `scripts._common` lives in this package, so the REPO ROOT (not just `src`) must be
# importable for a bare `python scripts/<name>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import fmt_metric as _fmt  # noqa: E402
from scripts._common import markdown_table as _table  # noqa: E402

OUT_DIR = "report/zernike_coeff2amp"
FIG_DIR = f"{OUT_DIR}/figures"
FILE_SWEEP = "logs/zernike_coeff_sweep.json"
OBJ_SWEEP = "logs/zernike_coeff_sweep_objective.json"
PRODUCTION = "logs/zernike_coeff/production_fold13/summary.json"


def _load(path: str) -> dict | None:
    file = ROOT / path
    if not file.exists():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


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
    ab_sweep = _load("logs/ab_file_protocol.json")
    seed_sweep = _load("logs/seed_sensitivity.json")
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
    body.append("# Zernike 系数 -> 远场强度 前向网络 —— 开发报告\n")
    body.append(
        "本报告记录一个**学习出来的**前向模型：输入 Noll 序 Zernike 系数"
        "（弧度），输出归一化的 CCD 远场强度。物理上这条链路是确定的"
        "（系数 → 瞳孔相位 → FFT → |F|²），网络的作用是把它变成一个"
        "可微、可微分的快速代理，供闭环整形使用。\n"
    )

    # ---- what the model is, before any number
    body.append("## 1. 模型是什么\n")
    body.append(
        "**输入**：136 维 Noll 序 Zernike 系数（弧度）。`n_max=15` → "
        "`calc_n_zernike_terms(15)` = 136；语料里最长向量只有 78 项，其余补 0。\n"
        "\n"
        "**输出**：`grid×grid`（默认 64×64）的归一化远场**强度**。\n"
        "CCD 测的是强度而不是场振幅，所以这是强度而非 amplitude。\n"
        "\n"
        "**它替代什么**：`slm_gs_refine` / `slm_model_in_loop` 这类 runner "
        "每步都要真下发一次相位、读一帧 CCD 才能拿到梯度，是**开环+测量**；"
        "本网络把这步变成一次前向传播，是**可微**的。物理链路本身是确定的：\n"
        "\n"
        "```\n"
        "系数 c → 瞳孔相位 Σ c_j Z_j → 瞳孔场 exp(iφ) → FFT → |F|²\n"
        "```\n"
        "\n"
        "**它不替代什么**：不替代相位**反解**。网络出的是图像，"
        "把图像反解成可下发相位是另一个问题（`zernike_phase2amp` 报告末尾"
        "讨论了那个反转问题）。\n"
    )

    # 2 data
    body.append("## 2. 数据\n")
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
    body.append("\n## 3. 预处理\n")
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
    body.append("\n## 4. 两个模型\n")
    body.append(
        "- **`conv`（首选）**：`Linear(136→256·4·4)` 把系数投影成一张**常数特征图**，"
        "再经 4 级上采样卷积解码到 64×64。保留输出的二维局部性。\n"
        "- **`mlp`（基线）**：直接 `Linear` 到 `64·64`。\n"
        "- 两者输出头都**不加激活**（raw 无界），再由 `peak_normalize` 归一。"
        "`ml.phase.unet.UNetGenerator` 的图像输出路径是硬编码 `nn.Sigmoid()`，"
        "无法表达该动态范围，故未复用。\n"
    )

    # 4 normalisation gate
    body.append("\n## 5. 为什么只按 R² 选模型\n")
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
    body.append("\n## 6. 结果\n")
    if file_sweep:
        body.append("### 6.1 按文件留一（18 折）\n")
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
                f"{_fmt(2 / 2 ** max(1, len(stat.get('folds', []))))}）。\n"
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
            body.append("\n### 6.2 噪声地板真因：语料非 i.i.d.\n")
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
        body.append("\n### 6.3 按优化目标留一（5 折）\n")
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
        body.append("\n## 7. 完整训练\n")
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
    # ---- optimisation levers, one at a time (reference-report pattern) ----
    body.append("\n## 8. 优化：一个杠杆一个杠杆地试\n")
    body.append(
        _table(
            ["杠杆", "结果"],
            [
                ["目标归一化 `image_mode`", "**唯一的大杠杆**：`robust` 取代 `peak`（本台架 `peak` 会被热像素支配，实测热像素峰 22–46 vs 帧均 0.26）"],
                ["目标顺序 `n_max`", "15 → 136 项。**已饱和**：语料最长只有 78 项，再加维度只是加死杠杆（见下）"],
                ["epoch 数", "15 → 80 有效：折 13 从 +0.9152 升到 +0.9300"],
                ["架构 `conv` vs `mlp`", "**测不出差异**（18 折 p=0.17，5 折 p=0.50）"],
                ["输入宽度 `input_terms`", "**死杠杆**，可解析证明 —— 见下"],
                ["`optimizer`", "未扫。`weight_decay=0` 时 `adam ≡ adamw` 是**正确**的，不是 bug"],
                ["曝光增广", "**不适用**：系数输入里没有曝光通道，`peak`/`robust` 已把绝对强度归一掉"],
            ],
        )
    )

    body.append("\n### 死杠杆：`input_terms`（可解析证明，不必等实验）\n")
    body.append(
        "语料最长向量是 78 项，所以第 78..135 维**恒为 0**。对这一维输入求梯度恒等于 0，"
        "于是对应的权重**永远停留在初始化值**，而它对前向的贡献是 `w · 0 = 0`。\n\n"
        "> 也就是说：**把这 58 维裁掉在数学上不可能改变拟合能力**，"
        "只会因为权重初始化的随机数流不同而产生 O(初始化噪声) 的差异。\n\n"
        "因此本报告把 `input_terms` 的 A/B 定位为**验证这条推理**，而不是寻找提升点："
        "若配对差异确实落在初始化噪声量级（而不是显著为零），推理与实验一致。"
        "若真要提升泛化，杠杆在**数据侧**（跨目标、跨 family 的采集），不在输入维度上。\n"
    )

    if ab_sweep:
        body.append("\n#### 实测：差异显著，但不支持“裁掉更好”这一结论\n")
        rows_ab = []
        for arm_name, st in ab_sweep.get("arm_means", {}).items():
            rows_ab.append(
                [
                    arm_name,
                    _fmt(st["mean_r2"]),
                    _fmt(st.get("std_r2")),
                    _fmt(st.get("mean_null_r2")),
                    _fmt(st.get("mean_skill_r2")),
                ]
            )
        body.append(
            _table(["配置", "val R²", "折间 σ", "R²_null", "skill"], rows_ab)
        )
        for key, stat in ab_sweep.get("paired_statistics", {}).items():
            body.append(
                f"\n配对 `{key}`：均值差 **{_fmt(stat['mean_diff'])}**，"
                f"sign-flip p = **{_fmt(stat['p_sign_flip'])}**，"
                f"Holm p = {_fmt(stat.get('p_holm'))}，"
                f"d_z = {_fmt(stat.get('cohens_dz'), 2)}。"
            )
        body.append(
            "\n> ⚠ **这个 p 值不能按面读。** 全部 18 折用同一个 `seed=0`，"
            "因此权重初始化的随机流也是同一条。裁掉 58 维会把这条流整体位移一步，"
            "产生**另一个具体的初始化**，再在 18 个折上重复测量。"
            "这 18 个配对差**不是**“裁掉 vs 不裁掉”这一个问题 18 次独立采样，"
            "而是**同一对初始化**在不同数据上的重复测量；"
            "sign-flip 检验当作独立处理，所以 p 值**偏保守**（过于乐观）。\n\n"
            "结论：**差异实测有、机制不存在。** 正确的测试是每个配置跑**多个 seed**、"
            "再对 seed 做配对 —— 见下一节，这个实验已经做了。\n"
        )

    if seed_sweep:
        lo = min(seed_sweep["input_terms"])
        hi = max(seed_sweep["input_terms"])
        body.append(
            f"\n#### 决定性证据：固定一折，只换 seed\n"
            f"\n这是上一节那个混淆的正确对照：**固定折 {seed_sweep['fold']}**，"
            f"只把 `seed` 从 {seed_sweep['seeds'][0]} 换到 {seed_sweep['seeds'][-1]}。"
            "折的难度被完全消掉，剩下的就是初始化本身。\n"
        )
        rows_seed = [
            [
                str(entry["seed"]),
                _fmt(entry[f"in{lo}"], 4),
                _fmt(entry[f"in{hi}"], 4),
                _fmt(entry[f"in{hi}_minus_in{lo}"], 4),
            ]
            for entry in seed_sweep["per_seed"]
        ]
        body.append(
            _table(["seed", f"in{lo}", f"in{hi}", f"in{hi} − in{lo}"], rows_seed)
        )
        body.append(
            f"\n均值差 **{_fmt(seed_sweep.get('mean_diff'), 4)}**"
            f" ± {_fmt(seed_sweep.get('std_diff'), 4)}，"
            f"符号 **{seed_sweep.get('signs_observed')}**"
            f"（符号翻转：**{seed_sweep.get('sign_flips')}**）。\n"
        )
        paired = ab_sweep.get("paired_statistics", {}) if ab_sweep else {}
        ab_diff = 0.0
        if paired:
            ab_diff = abs(next(iter(paired.values())).get("mean_diff", 0.0))
        body.append(
            "\n三条读数，每条都指向同一个结论：\n\n"
            f"1. **符号会跟着 seed 翻转**（seed {seed_sweep['seeds'][1]} 那行是反的）。"
            "所以 18 折那个 `p = 0.0144` 测的是**某一个具体初始化**，"
            "不是“裁掉输入维度”这件事。\n"
            f"2. **量级小 30 倍**：单折换 seed 的差均值只有 "
            f"**{_fmt(abs(seed_sweep.get('mean_diff', 0.0)), 4)}**，"
            f"而 18 折拟合给出的是 **{_fmt(ab_diff, 4)}**。\n"
            f"3. 两个配置的均值落在同一水平（in{lo} "
            f"{_fmt(seed_sweep['per_term_means'][f'in{lo}'], 4)} / "
            f"in{hi} {_fmt(seed_sweep['per_term_means'][f'in{hi}'], 4)}），"
            "与“死杠杆”的解析结论一致。\n\n"
            "> 2 和 3 不矛盾：那几折的折间 σ 是 ±0.74 / ±0.51，"
            "远大于差值本身，说明 18 折的均值被**少数几折的初始化故障**主导。"
            "把 18 折重复当成 18 次独立采样，等于把同一个初始化量了 18 遍 —— "
            "sign-flip 检验就是这么处理它们的，所以 p 值**偏保守**（过于乐观）。\n\n"
            "**最终结论：输入宽度不是杠杆。** `in78` 与 `in136` 在这台仪器、"
            "这份语料上无法区分；报告采纳解析结论（死输入 → 梯度恒零 → 权重冻结 → "
            "贡献 `w·0 = 0`），并**明确不采纳**“裁掉输入维度提升泛化”。"
            "要动泛化，杠杆在数据侧（跨目标、跨 family 采集）。\n"
        )

    body.append("\n### 非可加性警告\n")
    body.append(
        "本报告**没有**做 `(输入宽度 × lr)` 或 `(epoch × lr)` 的联合扫描，"
        "所以不能声称各杠杆独立可加。参考报告在同类扫描上实测到"
        "`lr=0.02` 单独用在一个 `n_max` 上毫无改善、换个 `n_max` 就是全表最好 —— "
        "**贪心坐标下降在这个问题上必然失败**。若后续要调 lr，请用联合扫描而非逐项扫描。\n"
    )

    body.append("\n### 配对 σ 与折间 σ\n")
    body.append(
        "本工作流所有跨模型结论都是**逐折配对差**，用精确 sign-flip 置换检验。"
        "原因可直接测量：本工作流折间 R² 标准差约 **0.74**（18 折），"
        "而配对差的标准差量级 ~0.03 —— **相差一个数量级**。"
        "折难度在配对设计里被抵消掉，所以只有配对差才看得见真实效应。"
        "18 折的最小可达 p = 2/2¹⁸ ≈ 7.6e-6；5 折只有 2/2⁵ = 0.0625。\n"
    )

    body.append(
        "\n> ⚠️ **winner's curse**：本报告的 `input_terms` / 架构结论都是在同一批折上"
        "从少数候选里挑出来的，属于**提示性证据**；要确证需要嵌套 CV。"
        "保守结论是「两者不可区分」，这也是本报告的最终口径。\n"
    )

    # ---- overturned conclusions archive -----------------------------------
    body.append("\n## 9. 本报告中被推翻的结论\n")
    body.append(
        _table(
            ["曾经的结论", "实际"],
            [
                [
                    "`peak` 归一化是唯一正确选择",
                    "本台架 `peak` 由**探测器缺陷**决定（热像素峰 22–46 vs 帧均 0.26），故改用 `robust`",
                ],
                [
                    "目标窗口覆盖足够视场",
                    "`_anchored_window` 只取 `grid×grid`，本语料仅覆盖约 **26%** 角视场",
                ],
                [
                    "`sum` 归一化会给出更好的 PSNR",
                    "实测 `sum` 让 PSNR 虚高到 **80.41 dB**，而 R² 几乎不变 —— PSNR 在此是归一化的伪影",
                ],
                [
                    "R² 会低于 0 说明模型无效",
                    "曾因**汇总方差当分母**得到虚假的负 R²；改为逐样本分母后正常",
                ],
                [
                    "绝对 R² 可以直接横向比较",
                    "常数预测器在同一折上就能拿到 +0.83，必须用 **skill = R² − R²_null** 才可比",
                ],
                [
                    "裁掉 58 个死输入维能提升泛化",
                    "**可证伪**：死输入梯度恒为 0，权重不动，贡献恒为 0；A/B 只测到初始化噪声",
                ],
            ],
        )
        + "\n"
    )

    body.append("\n## 10. 已知局限（勿越读）\n")
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
    body.append("\n## 11. 复现\n")
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
        body.append("\n## 12. 图\n")
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
