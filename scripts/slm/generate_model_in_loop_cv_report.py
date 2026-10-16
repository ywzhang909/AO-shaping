"""Generate the K-fold cross-validation report for the model-in-loop holdout.

**The question.** ``slm-model-in-loop`` splits its probes by a *fixed seed offset*
-- train probes get ``seed + index*1000 + k`` and the held-out ones the same
expression offset by ``1_000_000``. That is one partition, not cross-validation,
so every model change in this repo has so far been scored against whichever
probes happened to land in that fixed reserve. If the score moves when the reserve
changes, the fixed partition was never a property of the model.

**The pairing.** Fold ``k`` uses base seed ``seed + k``, so each fold has its own
optimiser initialisation and its own training probes. Within a fold the two arms
share *everything* -- initialisation, training probes, probe spread, schedule --
and differ **only** in which probes are reserved for scoring:

* ``legacy``: the fixed-offset reserve the production loop would have used;
* ``rotated``: a reserve drawn from a probe pool by the fold partition.

Holding the training probes fixed is the whole point. Rotating them too would
confound "a different reserve" with "a different training set", and the question
being asked is only about the reserve.

Pairing is on the fold index, which here *is* the seed index (base seed =
``seed + k``), so the comparison is paired along the axis that actually varies --
not on a fixed axis with folds counted as independent replicates, which is the
anti-pattern ``ml/zernike/cv_harness.py`` exists to prevent. The structural floor
``2 / 2**K`` is printed beside every p-value.

**How it is measured, device-free.** Each fold runs the **real** optimiser
(:func:`optimize_slm_model_in_loop`) on the simulated backend, with the
``probe_split_hook`` this worktree added supplying that fold's reserve. The loop
is not re-implemented and Step A is not re-fit here, so the number is the one
production would produce. Two opt-in flags make it observable on sim, both
default-off and documented in the config:

* ``holdout_on_sim=True`` -- measure the held-out probes on the sim backend
  (without it the sim path never measures them and the score is NaN);
* ``probe_split_hook=<fold>`` -- replace the fixed offset with this fold's reserve.

**Fully offline.** Simulated bench only; no camera, no SLM.

Run:
    python scripts/slm/generate_model_in_loop_cv_report.py
    python scripts/slm/generate_model_in_loop_cv_report.py --folds 8 --quick
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._common import fmt_metric, markdown_table, savefig  # noqa: E402
from scripts._common.provenance import insert_header  # noqa: E402

from ao_shaping.optimizer.wfless.slm_model_in_loop import (  # noqa: E402
    SlmModelInLoopConfig,
    optimize_slm_model_in_loop,
)
from ml.zernike.cv_harness import (  # noqa: E402
    build_folds,
    min_attainable_pvalue_folds,
    paired_fold_report,
)

REPORT_KEY = "report/shape_trials/model_in_loop_cv.md"
OUT_DIR = ROOT / "report" / "shape_trials"
FIG_DIR = OUT_DIR / "figures"

#: Base for the probe-seed pool a folded reserve is drawn from. Far from the
#: ``seed + index*1000`` and ``seed + 1_000_000 + index*1000`` schedules so a
#: pooled reserve can never collide with the legacy training probes, which the
#: split resolver rejects as an overlap.
POOL_BASE = 5_000_000

_LEGACY = "固定划分 (legacy)"
_ROTATED = "轮换划分 (rotated)"


@dataclass(frozen=True)
class FoldRun:
    """One fold's outcome.

    Attributes:
        fold: Zero-based fold index.
        base_seed: Run seed for this fold (``seed + fold``), so each fold has its
            own initialisation; shared by both arms.
        holdout_seeds: The folded reserve, i.e. the only difference between arms.
            The **training** probes are not stored: they are delegated to the
            legacy rule, so they grow mid-run whenever the acceptance guard
            rejects a round, and a single stored list would misreport them.
        legacy: Held-out correlation under the fixed production offset.
        rotated: Held-out correlation under this fold's reserve.
        status_legacy / status_rotated: Run status strings, so a fold that
            aborted cannot be silently scored as a normal one.
    """

    fold: int
    base_seed: int
    holdout_seeds: list[int]
    legacy: float
    rotated: float
    status_legacy: str
    status_rotated: str


def _pool(pool_size: int) -> list[int]:
    """The deterministic probe-seed pool the folded reserves are drawn from."""
    return [POOL_BASE + i for i in range(pool_size)]


def _legacy_hook():
    """A hook reproducing the production fixed seed offset exactly.

    Takes ``seed``/``index`` from the loop, so pointing the run at base seed
    ``seed + k`` moves the whole legacy schedule -- which is what makes the legacy
    arm a varying series rather than one constant repeated per fold.

    The counts the loop passes are used, not anything captured here: the
    acceptance guard **adds probes** when it rejects a round, so the number
    requested grows mid-run and a hook answering from a closure would
    under-supply from the second round on.
    """

    def _hook(seed, index, probe_count, holdout_probe_count):  # noqa: ANN001, ANN202
        base = int(seed) + int(index) * 1000
        return (
            [base + n for n in range(int(probe_count))],
            [base + 1_000_000 + n for n in range(int(holdout_probe_count))],
        )

    return _hook


def _fold_hook(legacy_hook, holdout_seeds: list[int]):
    """A hook that keeps the legacy training probes and pins only the reserve.

    The train list is **delegated to the legacy rule** rather than pinned, and
    that is load-bearing: when the acceptance guard rejects a round it *adds
    probes*, so the loop asks the hook for more train seeds on a later round. A
    fixed-size list raises there. Delegating keeps the "arms differ only in the
    reserve" invariant exact even when the run adds probes, because the legacy
    arm grows its train set the same way from the same base seed.

    ``holdout_seeds`` is constant across rounds on purpose: rotating it per round
    would turn one independent measurement into ``n_rounds`` of them and quietly
    stop testing independence.
    """

    def _hook(seed, index, probe_count, holdout_probe_count):  # noqa: ANN001, ANN202
        train, _legacy_reserve = legacy_hook(
            seed, index, probe_count, holdout_probe_count
        )
        if len(holdout_seeds) != holdout_probe_count:
            raise ValueError(
                f"fold supplies {len(holdout_seeds)} holdout seeds but the loop "
                f"asked for {holdout_probe_count}"
            )
        return list(train), list(holdout_seeds)

    return _hook


def _run(config_kwargs: dict, base_seed: int, hook) -> tuple[float, str]:
    """Run the real optimiser and read back the final held-out correlation."""
    config = SlmModelInLoopConfig(
        **{**config_kwargs, "seed": base_seed, "probe_split_hook": hook}
    )
    result = optimize_slm_model_in_loop(config)
    rows = [
        row
        for row in result.recorder.history
        if row.get("stage") == "round" and "holdout_correlation" in row
    ]
    if not rows:
        return float("nan"), str(result.status)
    return float(rows[-1]["holdout_correlation"]), str(result.status)


def run_cv(
    *,
    folds: int,
    pool_size: int,
    seed: int,
    config_kwargs: dict,
) -> list[FoldRun]:
    """Run every fold's legacy and rotated arms.

    Both arms within a fold share ``base_seed`` and the training probe seeds, and
    differ only in the reserve. That isolation is the whole measurement.
    """
    partition = build_folds(pool_size, folds, seed=seed)
    holdout_count = int(config_kwargs["holdout_probe_count"])
    pool = _pool(pool_size)
    legacy_hook = _legacy_hook()

    results: list[FoldRun] = []
    for fold_index, (_train_idx, test_idx) in enumerate(partition):
        base_seed = int(seed) + fold_index
        # The training probes stay exactly what the legacy rule produces at this
        # base seed -- including the extra probes a rejection adds mid-run -- so
        # only the reserve comes from the fold.
        holdout_seeds = [pool[i] for i in test_idx[:holdout_count]]

        legacy_score, legacy_status = _run(config_kwargs, base_seed, legacy_hook)
        rotated_score, rotated_status = _run(
            config_kwargs, base_seed, _fold_hook(legacy_hook, holdout_seeds)
        )
        results.append(
            FoldRun(
                fold=fold_index,
                base_seed=base_seed,
                holdout_seeds=holdout_seeds,
                legacy=legacy_score,
                rotated=rotated_score,
                status_legacy=legacy_status,
                status_rotated=rotated_status,
            )
        )
        logger.info(
            "fold {}/{} (seed {}): legacy {:.4f} ({}) rotated {:.4f} ({})",
            fold_index + 1,
            len(partition),
            base_seed,
            legacy_score,
            legacy_status,
            rotated_score,
            rotated_status,
        )
    return results


def _figure(runs: list[FoldRun], out_path: Path) -> Path:
    """Per-fold legacy vs rotated held-out score, plus their difference."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    folds = [r.fold + 1 for r in runs]
    legacy = [r.legacy for r in runs]
    rotated = [r.rotated for r in runs]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9, 6), height_ratios=[2, 1], sharex=True
    )

    width = 0.38
    ax_top.bar(
        [f - width / 2 for f in folds],
        legacy,
        width,
        label="固定划分 (legacy)",
        color="#4C78A8",
    )
    ax_top.bar(
        [f + width / 2 for f in folds],
        rotated,
        width,
        label="轮换划分 (rotated)",
        color="#F58518",
    )
    ax_top.axhline(
        float(np.mean(legacy)),
        color="#4C78A8",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="legacy 均值",
    )
    ax_top.set_ylabel("留出探针相关系数")
    ax_top.set_title("model-in-loop 留出分数随留出探针划分的变化 (sim)")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(axis="y", alpha=0.3)

    deltas = [r.rotated - r.legacy for r in runs]
    colors = ["#54A24B" if d >= 0 else "#E45756" for d in deltas]
    ax_bottom.bar(folds, deltas, color=colors)
    ax_bottom.axhline(0.0, color="black", linewidth=0.8)
    ax_bottom.set_xlabel("fold")
    ax_bottom.set_ylabel("Δ (rotated − legacy)")
    ax_bottom.grid(axis="y", alpha=0.3)

    savefig(fig, out_path)
    return out_path


def _body(
    runs: list[FoldRun],
    partition: list[tuple[list[int], list[int]]],
    pool_size: int,
    fold_seed: int,
    base_seed: int,
    config_kwargs: dict,
) -> str:
    """Render the report body (the provenance header is stamped at write time)."""
    stats = paired_fold_report(
        {_LEGACY: [r.legacy for r in runs], _ROTATED: [r.rotated for r in runs]},
        partition,
        paired_on="fold",
    )
    comparison = stats["comparison"]
    floor = stats["min_attainable_pvalue"]

    legacy = np.asarray([r.legacy for r in runs], dtype=np.float64)
    rotated = np.asarray([r.rotated for r in runs], dtype=np.float64)
    deltas = rotated - legacy
    last_round = max(int(config_kwargs["n_rounds"]) - 1, 0)
    lines: list[str] = []

    lines.append("# model-in-loop 留出探针的 K 折交叉验证")
    lines.append("")
    lines.append(
        "**问题**：`slm-model-in-loop` 的探针划分是一个**固定种子偏移**"
        "（留出 = `seed + 1_000_000 + index*1000 + k`），不是交叉验证。"
        "本报告量化的是：**换一个留出探针集合，留出分数会动多少**。"
    )
    lines.append("")
    lines.append(
        f"**设置**：{len(partition)} 折，折内两臂共享 base seed "
        f"`{base_seed} + k`（故各有独立初始化）与**完全相同的训练探针**，"
        f"唯一差别是留出集合；折划分种子 {fold_seed}，探针池 {pool_size} 个，"
        "后端 `sim`（无设备）。"
    )
    lines.append("")

    lines.append("## 1. 逐折结果")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "fold",
                "base seed",
                f"留出探针 legacy (第 {last_round + 1} 轮)",
                "留出探针 rotated (固定)",
                "legacy 分数",
                "rotated 分数",
                "Δ",
            ],
            [
                [
                    str(r.fold + 1),
                    str(r.base_seed),
                    f"{(r.base_seed + last_round * 1000 + 1_000_000):,}–"
                    f"{(r.base_seed + last_round * 1000 + 1_000_000 + len(r.holdout_seeds) - 1):,}",
                    ", ".join(f"{s:,}" for s in r.holdout_seeds),
                    fmt_metric(r.legacy),
                    fmt_metric(r.rotated),
                    fmt_metric(r.rotated - r.legacy),
                ]
                for r in runs
            ],
        )
    )
    lines.append("")
    lines.append(
        f"分数取**最后一轮**的留出相关性。两臂的留出集合并非同一形状：legacy 的留出探针"
        f"随轮次前移（`base + index*1000 + 1_000_000 + k`），所以上表给出的是被评分的"
        f"那一轮（第 {last_round + 1} 轮）的集合；rotated 的留出集合在整轮内固定，"
        "否则一次独立测量会变成若干次，留出就不再独立。"
    )
    lines.append("")
    lines.append(
        "两臂运行状态："
        + "；".join(
            f"fold {r.fold + 1} legacy=`{r.status_legacy}` / rotated=`{r.status_rotated}`"
            for r in runs
        )
        + "。（`holdout_on_sim` 只**测量**不**判定**：仿真不会因留出相关性偏低而中止，"
        "所以每折都有分数而不是通过/失败。）"
    )
    lines.append("")

    lines.append("## 2. 配对统计")
    lines.append("")
    lines.append(
        markdown_table(
            ["指标", "值"],
            [
                [
                    "固定划分 均值 ± sd",
                    f"{fmt_metric(stats[_LEGACY]['mean'])} ± {fmt_metric(stats[_LEGACY]['sd'])}",
                ],
                [
                    "轮换划分 均值 ± sd",
                    f"{fmt_metric(stats[_ROTATED]['mean'])} ± {fmt_metric(stats[_ROTATED]['sd'])}",
                ],
                ["Δ 均值 (rotated − legacy)", fmt_metric(comparison["delta_mean"])],
                [
                    "Δ 范围",
                    f"{fmt_metric(float(np.min(deltas)))} … {fmt_metric(float(np.max(deltas)))}",
                ],
                ["Cohen's d_z", fmt_metric(comparison["cohens_dz"])],
                ["sign-flip p", fmt_metric(comparison["pvalue"])],
                ["最小可达 p = 2/2^K", fmt_metric(floor)],
                ["p 是否已触底", "是" if comparison["pvalue_is_at_floor"] else "否"],
            ],
        )
    )
    delta_sd = (
        float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    )
    ratio = (
        abs(float(comparison["delta_mean"])) / delta_sd
        if np.isfinite(delta_sd) and delta_sd > 0
        else float("inf")
    )
    lines.append("")
    lines.append(
        f"Δ 的折间 sd 是 **{fmt_metric(delta_sd)}**，Δ 的均值是 "
        f"{fmt_metric(comparison['delta_mean'])}，"
        f"均值/sd = **{fmt_metric(ratio)}**。"
    )
    if comparison["pvalue_is_at_floor"] or ratio < 1.0:
        lines.append(
            "均值小于折间抖动：换留出集合带来的扰动与两臂之间的系统差**不可区分**。"
            "固定划分给出的那一个数不能当作模型属性引用 —— 它只是这一个留出集合的抽样结果，"
            f"同协议下换一个留出集合可以看到 ±{fmt_metric(delta_sd)} 量级的移动。"
        )
    else:
        lines.append(
            "均值与折间抖动同量级但仍偏小：没有观察到两臂的系统性差异"
            f"（d_z = {fmt_metric(comparison['cohens_dz'])}，"
            f"p = {fmt_metric(comparison['pvalue'])}，最小可达 {fmt_metric(floor)}），"
            f"但折间抖动 ±{fmt_metric(delta_sd)} 说明单个留出集合的读数不宜单独引用。"
        )
    lines.append("")

    lines.append("## 3. 怎么读这张表（本报告不宣称什么）")
    lines.append("")
    lines.append(
        f"1. **本折数在结构上无法给出 p < 0.05 的结论。** sign-flip 精确置换检验的"
        f"最小可达 p 是 `2/2^K` = {fmt_metric(floor)}（{len(partition)} 折）。"
        "触底时只报效应量，不报显著性。"
    )
    lines.append(
        "2. **配对轴是被变动的那个轴。** 折 k 的 base seed 是 `seed + k`，"
        "所以配对是沿 seed/划分轴做的；没有把 K 折当成 K 次独立初始化采样。"
    )
    lines.append(
        "3. **两臂共享训练探针**，这是刻意的：只旋转留出集合才能把"
        "「留出集合不同」与「训练集合不同」分开。"
    )
    lines.append(
        "4. **这不是硬件结论。** 全部数字来自 `sim` 后端；硬件的读出噪声、"
        "漂移与 LCOS settling 都不在模型内。硬件的折间 spread 只可能更大。"
    )
    lines.append(
        "5. **GS 是对照而非基线。** `report/shape_trials/README.md` 记录了 GS 在"
        "实机上输给平场相位；本报告不把任何一臂称作基线。"
    )
    lines.append("")

    lines.append("## 4. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/slm/generate_model_in_loop_cv_report.py")
    lines.append("```")
    lines.append("")
    lines.append(
        f"本报告由 `--folds {len(partition)} --pool-size {pool_size} "
        f"--fold-seed {fold_seed}` 生成；统计量全部来自 "
        "`ml/zernike/cv_harness.py` 与 `ml/zernike/eval_stats.py`，脚本内不复算。"
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="model-in-loop K-fold CV report")
    parser.add_argument("--folds", type=int, default=5, help="K folds")
    parser.add_argument("--pool-size", type=int, default=16, help="probe-seed pool size")
    parser.add_argument("--seed", type=int, default=0, help="base run seed (fold k uses seed+k)")
    parser.add_argument(
        "--fold-seed", type=int, default=20261014, help="seed of the partition itself"
    )
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--holdout-probe-count", type=int, default=2)
    parser.add_argument("--step-a-iterations", type=int, default=20)
    parser.add_argument("--step-b-iterations", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="smoke: 2 folds, 1 round")
    args = parser.parse_args()
    if args.quick:
        args.folds = min(args.folds, 2)
        args.n_rounds = 1
    if args.pool_size < args.holdout_probe_count:
        parser.error(
            "--pool-size must be >= --holdout-probe-count, else a fold cannot "
            "fill its reserve"
        )
    if args.folds < 2:
        parser.error("--folds must be >= 2")
    if args.folds > args.pool_size:
        parser.error("--folds cannot exceed --pool-size")
    return args


def main() -> int:
    """Run the K-fold study and write the report."""
    args = _parse_args()
    config_kwargs = {
        "cam_type": "sim",
        "region": 32,
        "cam_size": 128,
        "far_field_padding": 4,
        "n_rounds": args.n_rounds,
        "probe_count": args.probe_count,
        "holdout_probe_count": args.holdout_probe_count,
        "step_a_iterations": args.step_a_iterations,
        "step_b_iterations": args.step_b_iterations,
        "target_side": 64,
        "n_orders": 4,
        "dtype": "float32",
        "device": "cpu",
        "holdout_on_sim": True,
    }

    logger.info(
        "model-in-loop CV: {} folds over a pool of {} (base seed {})",
        args.folds,
        args.pool_size,
        args.seed,
    )
    partition = build_folds(args.pool_size, args.folds, seed=args.fold_seed)
    runs = run_cv(
        folds=args.folds,
        pool_size=args.pool_size,
        seed=args.fold_seed,
        config_kwargs={**config_kwargs, "seed": args.seed},
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_name = "model_in_loop_cv.png"
    _figure(runs, FIG_DIR / figure_name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = _body(
        runs, partition, args.pool_size, args.fold_seed, args.seed, config_kwargs
    )
    body = body.replace(
        "## 1. 逐折结果",
        f"![留出分数随留出集合的变化](figures/{figure_name})\n\n## 1. 逐折结果",
        1,
    )
    (OUT_DIR / "model_in_loop_cv.md").write_text(
        insert_header(body, REPORT_KEY), encoding="utf-8"
    )

    rotated = np.asarray([r.rotated for r in runs], dtype=np.float64)
    logger.info(
        "wrote {} (+ figures/{}) -- rotated held-out mean {} sd {} (floor p={})",
        REPORT_KEY,
        figure_name,
        fmt_metric(float(np.mean(rotated))),
        fmt_metric(float(np.std(rotated, ddof=1))) if len(rotated) > 1 else "-",
        fmt_metric(min_attainable_pvalue_folds(len(runs))),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())