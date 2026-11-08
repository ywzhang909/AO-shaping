"""Render the 2026-11-07 offline loss-compare report (issue #236).

Offline renderer: reads the archived ``results.json`` produced by
``scripts/slm/loss_compare_20261107.py`` under
``data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1`` and writes
``report/shape_trials/loss_compare_20261107/loss_compare_20261107.md``.
It never imports ``ao_shaping.drivers.*``, never opens an SLM, camera or
laser and never runs a capture, so no device is occupied.

Conclusion policy (issue #236): the loss choice stays UNDECIDED.  The
``+1`` primary arm was invalidated by the #235 hardware evidence
(``phase_sign`` polarity), and the ``-1`` re-run shows no discrimination
between the seven candidate losses on the 9 held-out U/C pairs, with the
baseline reproduction FAILED.

Provenance: ``scripts/_common/provenance.py`` maps this report to this
script; ``scripts/sync_report_provenance.py`` owns the provenance block
and the ``report/README.md`` index.  This renderer therefore writes the
report body WITHOUT a provenance block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "debug" / "slm_transfer"
RESULTS_PATH = (
    DATA_DIR
    / "loss_compare_20261107_phase_sign_m1"
    / "results.json"
)
OUT_DIR = ROOT / "report" / "shape_trials" / "loss_compare_20261107"
REPORT_PATH = OUT_DIR / "loss_compare_20261107.md"

# Relative links from report/shape_trials/loss_compare_20261107/.
REL_GENERATOR = "../../../scripts/slm/generate_loss_compare_report_20261107.py"
REL_LOSS_COMPARE_SCRIPT = "../../../scripts/slm/loss_compare_20261107.py"
REL_RESULTS = "../../../data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json"
REL_REFERENCE_FIT = (
    "../../../data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json"
)
REL_SIBLING_REPORT = "../forward_model_polarity/forward_model_polarity_20261107.md"
ISSUE_236_URL = "https://github.com/ywzhang909/AO-shaping/issues/236"
ISSUE_235_URL = "https://github.com/ywzhang909/AO-shaping/issues/235"
ISSUE_236 = "issue #236"

#: Column order is the pre-registered ranking order of the archived JSON:
#: direction median desc, core-61 Pearson desc, FWHM error asc.
METRIC_HEADER = "| rank | loss | sign_mode | 方向中位（符号 Pearson） | 核心61 Pearson | FWHM 误差（px） |"


def load_results() -> dict[str, Any]:
    """Read the archived loss-compare results."""
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def signed(value: float, digits: int = 3) -> str:
    """Format with an explicit sign, e.g. ``+0.070`` / ``-0.043``."""
    return f"{value:+.{digits}f}"


def plain(value: float, digits: int = 3) -> str:
    """Format without a sign, e.g. ``0.589``."""
    return f"{value:.{digits}f}"


def ranking_table(rows: list[dict[str, Any]]) -> str:
    """Render one sign arm's ranking as a markdown table."""
    lines = [METRIC_HEADER, "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `{row['loss']}` | {row['sign_mode']} "
            f"| {signed(float(row['direction_median']))} "
            f"| {plain(float(row['pearson_core61']))} "
            f"| {plain(float(row['fwhm_error']), 2)} |"
        )
    return "\n".join(lines)


def range_text(rows: list[dict[str, Any]]) -> str:
    """The spread of direction medians in one arm, e.g. ``-0.043…+0.070``."""
    values = [float(row["direction_median"]) for row in rows]
    low, high = min(values), max(values)
    return f"{signed(low)}…{signed(high)}"


def render_report(data: dict[str, Any]) -> str:
    """Build the whole markdown report body (no provenance block)."""
    dataset = data["dataset"]
    config = data["config"]
    validation = data["validation"]
    stored = validation["stored"]
    reproduced = validation["reproduced"]
    primary_rows = data["ranking"]
    legacy_rows = data["legacy_ranking"]
    best = data["best"]
    composite_weights = data.get("preregistered", {}).get("composite_weights", {})
    weights_text = " / ".join(
        f"{name} {plain(float(w), 1)}" for name, w in composite_weights.items()
    )
    direction_range = range_text(primary_rows)
    origin_stored = stored["origin"]
    origin_reproduced = reproduced["origin"]
    n_pairs = int(dataset["pairs"])

    lines = [
        f"# 离线损失函数比较（20261107，[{ISSUE_236}]({ISSUE_236_URL})）：方向中位 + 核心61 Pearson",
        "",
        f"**日期** 2026-11-07 · **脚本** [`scripts/slm/generate_loss_compare_report_20261107.py`]({REL_GENERATOR})"
        "（离线渲染，未占用设备）",
        f"**数据** [`results.json`]({REL_RESULTS})（{data['generated_utc']} 生成，"
        f"`device_occupied=false`）· 数据集 `defocus_forward_20261107_focus_r2`："
        f"{dataset['manifest_rows']} 行清单 / {dataset['sample_count']} 样本"
        f"（{dataset['train']} 训练 / {dataset['test']} 测试）/ {n_pairs} 个 held-out U/C 对 / "
        f"{dataset['crop']} 裁剪 / 曝光归一",
        "",
        "## 1. 背景与首轮作废",
        "",
        f"- [{ISSUE_236}]({ISSUE_236_URL}) 为 `fit_loss` 重构做候选损失 bake-off：7 种损失"
        f"（pearson / mse / log_mse / rel_l2 / masked_mse / ee_uniformity / composite）在冻结前向、"
        f"只优化 (d, origin, log_gain) 三参数（{config['optimizer']}，{config['iterations']} 步）下，"
        f"于同一 U/C 回放数据上离线比较（[`loss_compare_20261107.py`]({REL_LOSS_COMPARE_SCRIPT})）。",
        f"- `composite` 为预登记权重组合（{weights_text}）；初始 d={plain(float(config['initial_defocus']), 4)}、"
        f"origin=({plain(float(config['initial_origin'][0]), 2)}, {plain(float(config['initial_origin'][1]), 2)})、"
        "log_gain=0。",
        f"- **首轮作废**：首轮曾以 `+1` 极性为主臂、据下表历史对照推荐 `masked_mse`"
        f"（方向中位 {signed(float(legacy_rows[0]['direction_median']))}）。"
        f"[#235]({ISSUE_235_URL}) 的单模屏配对证据（42/42 对在 `phase_sign=-1` 下方向相关全部翻正）"
        "推翻 `+1` 极性，首轮推荐随之作废；本报告为 `−1` 主臂重跑结果。",
        "",
        "## 2. −1 主臂排序（sign_mode=negated）",
        "",
        "口径（存档 JSON `ranking`，预登记排序：方向中位 ↓、核心61 ↑、FWHM 误差 ↑；仅 full_batch 变体，"
        "多 seed 平均）：",
        "",
        "- **方向中位（符号 Pearson）**：每个 held-out U/C 对取 C−U 差分帧、去均值后全窗"
        f"（{dataset['crop']}）Pearson，{n_pairs} 对取中位数，再对 seed 取均值；",
        "- **核心61 Pearson**：测试帧核心 61×61 窗 Pearson 对测试帧取均值；",
        "- **FWHM 误差（px）**：|预测面积 FWHM − 实测面积 FWHM| 对测试帧取均值。",
        "",
        ranking_table(primary_rows),
        "",
        f"- **无区分度**：7 种损失的方向中位全部落在 {direction_range}，"
        f"最优 `{best['loss']}` 仅 {signed(float(best['direction_median']))}"
        f"（核心61 {plain(float(best['pearson_core61']))}）；held-out 仅 {n_pairs} 个 U/C 对，"
        "既有数据不足以在损失之间做选择。",
        "",
        "## 3. +1 历史对照（sign_mode=current，legacy，仅对照口径）",
        "",
        "该极性已被 #235 硬件证据推翻（见"
        f"[forward_model_polarity_20261107.md]({REL_SIBLING_REPORT})"
        "），下表数字不构成推荐依据；其高方向中位正是在被推翻的极性下取得的。",
        "",
        ranking_table(legacy_rows),
        "",
        "## 4. 基线复现 FAILED",
        "",
        "判据：以存档参考拟合"
        f"[`forward_fit.json`]({REL_REFERENCE_FIT})"
        " 为基线，同数据同口径重拟合（−1 臂），|Δd| ≤ "
        f"{plain(float(validation['tolerance']['defocus_abs']), 3)} 且 origin 偏移 ≤ "
        f"{plain(float(validation['tolerance']['origin_px']), 0)} px。",
        "",
        "| 量 | 存档基线 | −1 臂重拟合 |",
        "|---|---|---|",
        f"| defocus d（D） | {signed(float(stored['defocus_diopter']), 4)} "
        f"| {signed(float(reproduced['defocus_diopter']), 4)} |",
        f"| origin（px） | ({plain(float(origin_stored[0]), 2)}, {plain(float(origin_stored[1]), 2)}) "
        f"| ({plain(float(origin_reproduced[0]), 2)}, {plain(float(origin_reproduced[1]), 2)}) |",
        f"| mean corr | {plain(float(stored['mean_corr_all_rows']), 3)}（存档全行） "
        f"| {plain(float(reproduced['mean_corr_test_full']), 3)}（测试全窗） |",
        "",
        f"- Δd = {plain(float(validation['defocus_abs_delta']), 3)}"
        f"（容差 {plain(float(validation['tolerance']['defocus_abs']), 3)}）、"
        f"origin 最大偏移 {plain(float(validation['origin_max_abs_delta_px']), 1)} px"
        f"（容差 {plain(float(validation['tolerance']['origin_px']), 0)} px）→ `passed=false`。",
        "- 旁证：当前 `phase_sign` 作用域为整条指令相位（含载波/倾斜项），整体取反会破坏这些非对称项"
        f"——与 [#235]({ISSUE_235_URL}) 报告记录的自洽性矛盾一致（`−1` checkpoint 在同一 U/C 数据上"
        "全帧 Pearson 反而更低）。基线不可复现是损失比较结论不可靠的独立旁证。",
        "",
        "## 5. 结论：未定",
        "",
        f"- 7 种损失无区分度（−1 主臂方向中位 {direction_range}，held-out 仅 {n_pairs} 对）"
        "＋基线复现失败 → **无法据此选定损失函数**；",
        f"- [{ISSUE_236}]({ISSUE_236_URL}) 的消融实现应等条件具备后重跑，重做条件：",
        f"  1. `phase_sign` 语义修复并通过单模屏方向门禁（单对 ≥ 0.50，[#235]({ISSUE_235_URL}) 门禁口径）；",
        "  2. 在方向门禁通过后的数据上，以单模屏 / 闭环指标为主口径重做本比较"
        f"（U/C 回放 {n_pairs} 个 held-out 对不足以支撑 7 路比较）。",
        "",
        "## 6. 复现命令与产物",
        "",
        "- 数据生成（已存档，勿重跑）：",
        "  ```",
        "  python scripts/slm/loss_compare_20261107.py \\",
        "      --output data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1",
        "  ```",
        "- 报告渲染（离线，未占用设备）：",
        "  ```",
        "  python scripts/slm/generate_loss_compare_report_20261107.py",
        "  ```",
        f"- 产物：[`results.json`]({REL_RESULTS})"
        f"（存档于 `data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/`）、"
        "本报告、数据集目录 `data/debug/slm_transfer/defocus_forward_20261107_focus_r2`、"
        f"参考拟合 [`forward_fit.json`]({REL_REFERENCE_FIT})。",
        "- 全程离线（`device_occupied=false`），未占用 SLM / 相机 / 激光。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load_results()
    report = render_report(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    n_primary = len(data["ranking"])
    n_legacy = len(data["legacy_ranking"])
    print(f"report written: {REPORT_PATH}")
    print(f"ranking rows: primary={n_primary} legacy={n_legacy}")
    print(f"validation passed: {data['validation']['passed']}")


if __name__ == "__main__":
    main()
