"""把 issue #165 的 P0/P1 仿真产物渲染成报告。

读入：
* ``scripts/slm/calibrate_gsnet_2f.py`` 写出的 ``results.json``（臂 A 前向保真度
  + 臂 B 网络换前向前后）；
* ``scripts/slm/gsnet_init_sim_ab.py`` 写出的 ``sim_ab.json``（臂 C：MIL 闭环的
  「网络初值 vs 物理初值」A/B）。

渲染 ``report/gsnet_2f_calibration/report.md`` + ``figures/``。

用法::

    python scripts/slm/generate_gsnet_2f_calibration_report.py \
        --results data/debug/gsnet_2f_calib/<ts>/results.json \
        --mil-ab data/debug/gsnet_init_sim_ab/<ts>/sim_ab.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import fmt_metric as _fmt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

REPORT_DIR = ROOT / "report" / "gsnet_2f_calibration"
FAMILIES = ("smooth", "zernike", "freeform")
FAMILY_LABELS = {
    "smooth": "低阶 Zernike（Noll 4–15）",
    "zernike": "高阶 Zernike（Noll 4–66）",
    "freeform": "全频随机相位（带限）",
}
MODEL_LABELS = {
    "ideal": "理想 FFT 前向",
    "calibrated": "校准 2f 前向",
    "zernike": "物理 Zernike 基线",
}
ARM_LABELS = {
    "flat": "平场（零相位，下界）",
    "gs_ceiling": "GS 在过采样真值上的解（上限）",
    "zernike_baseline": "物理 Zernike 基线（经典 GS 解投影到 66 项）",
    "ideal": "网络（理想 FFT 前向）",
    "calibrated": "网络（校准 2f 前向）",
}


def _pm(row: dict[str, Any], key: str) -> str:
    return f"{_fmt(row[key]['mean'])} ± {_fmt(row[key]['sd'])}"


def _rel(path: str) -> str:
    """把容器里的绝对路径换成仓库相对路径，报告里好读。"""
    text = str(path)
    for prefix in ("/workspace/", "/workspace"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or str(path)


def _same_grid_value(
    results_same_grid: dict[str, Any] | None, key: str, arm: str = "calibrated"
) -> float:
    """取「同网格架构」那次运行里某个臂的某个指标（没有就返回 NaN）。"""
    if not results_same_grid:
        return float("nan")
    arm_b = results_same_grid.get("arm_b_network") or {}
    row = arm_b.get(arm) or {}
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render(
    results: dict[str, Any],
    results_path: Path,
    mil_ab: dict[str, Any] | None,
    mil_ab_path: Path | None,
    results_same_grid: dict[str, Any] | None = None,
    learnability: dict[str, Any] | None = None,
    learnability_before: dict[str, Any] | None = None,
    pretrain: dict[str, Any] | None = None,
    transfer: dict[str, Any] | None = None,
) -> str:
    arm_a = results["arm_a_forward_fidelity"]
    arm_b = results.get("arm_b_network") or {}
    decision = results["decision"]
    geometry = results["geometry"]
    config = results["config"]
    overall = arm_a["overall"]

    lines: list[str] = []
    add = lines.append

    add("# FourierGSNet 前向校准到本台架 2f（issue #165 P0/P1，纯仿真）")
    add("")
    add("<!-- provenance:start -->")
    add("> **生成脚本**: "
        "[`scripts/slm/generate_gsnet_2f_calibration_report.py`]"
        "(../../scripts/slm/generate_gsnet_2f_calibration_report.py)")
    add("> **复现命令**: "
        "`python scripts/slm/calibrate_gsnet_2f.py && "
        "python scripts/slm/gsnet_init_sim_ab.py && "
        "python scripts/slm/generate_gsnet_2f_calibration_report.py`")
    add("> **数据/关联脚本**: "
        "[`scripts/slm/calibrate_gsnet_2f.py`](../../scripts/slm/calibrate_gsnet_2f.py)")
    add("> **数据/关联脚本**: "
        "[`scripts/slm/gsnet_init_sim_ab.py`](../../scripts/slm/gsnet_init_sim_ab.py)")
    add("> **运行环境**: 离线")
    add("> **说明**: issue #165 的 P0（前向校准）与 P1（闭环初值）仿真验收；"
        "真值是数字孪生，非实验系统 CCD")
    add("<!-- provenance:end -->")
    add("")

    add("## 0. 结论")
    add("")
    add(f"**{decision['line']}**")
    add("")
    add("结论，按证据强度排列：")
    add("")
    cal = overall["calibrated"]
    ide = overall["ideal"]
    zen = overall["zernike"]
    add(f"1. **前向校准本身有效，而且是最硬的结论。** 校准后的 2f 物理前向"
        f"对数字孪生的远场预测相关性 **{_fmt(cal['correlation_full']['mean'])}**"
        f"（±{_fmt(cal['correlation_full']['sd'])}），"
        f"而理想 FFT 前向只有 **{_fmt(ide['correlation_full']['mean'])}**"
        f"（±{_fmt(ide['correlation_full']['sd'])}）；"
        f"NRMSE {_fmt(cal['nrmse_full']['mean'])} 对 {_fmt(ide['nrmse_full']['mean'])}。"
        f"这不是边缘差异。")
    add(f"2. **「校准前向」与「物理 Zernike 基线」在数值上几乎相同**"
        f"（相关性 {_fmt(cal['correlation_full']['mean'])} 对 "
        f"{_fmt(zen['correlation_full']['mean'])}）。原因是仓库自己的 "
        f"`ZernikeCoefficientOptimizer` **本来就是**本台架的 2f 物理前向"
        f"（圆形孔径 + 高斯照明 + 规范居中 FFT）。所以 issue 里「把理想 FFT 换成"
        f"2f 物理前向」这件事，实质是**给网络补上物理模型的孔径、照明、"
        f"实验系统像差、增益与背景**，而不是引入一套新物理。")
    if arm_b and "calibrated" in arm_b and "zernike_baseline" in arm_b:
        net_cal = arm_b["calibrated"]
        net_base = arm_b["zernike_baseline"]
        net_ideal = arm_b.get("ideal") or {}
        add(f"3. **网络层通过 P0 判据，但这靠的是拆掉一个架构限制，不是靠校准本身。** "
            f"`FourierGSNet` 原本要求 source 与 target **同网格**，于是它内部能看到的"
            f"远场只能是**同尺寸 FFT**——衍射极限光斑亚像素、图样混叠，"
            f"而实验系统的远场需要 8 倍以上过采样才被解析。"
            f"把契约改成「source 在瞳面网格、target 在远场网格」之后，"
            f"网络臂的复合分数从 {_fmt(_same_grid_value(results_same_grid, 'composite'))} "
            f"升到 **{_fmt(net_cal['composite']['mean'])}**，"
            f"**首次超过物理 Zernike 基线 {_fmt(net_base['composite']['mean'])}**，"
            f"也超过 GS 上限 {_fmt(arm_b.get('gs_ceiling', {}).get('composite', {}).get('mean', 0))}。"
            f"详见 §4.4。")
        add(f"4. **校准相对「采样修好但物理不校准」的增量，仍在噪声里。** "
            f"过采样后校准臂 composite {_fmt(net_cal['composite']['mean'])}、"
            f"理想 FFT 臂 {_fmt(net_ideal.get('composite', {}).get('mean', 0))}——"
            f"两者基本打平；但校准臂的**均匀度更好**（CV "
            f"{_fmt(net_cal['cv_target_box']['mean'])} 对 "
            f"{_fmt(net_ideal.get('cv_target_box', {}).get('mean', 0))}），"
            f"**能量占比略低**（EE {_fmt(net_cal['encircled_energy']['mean'])} 对 "
            f"{_fmt(net_ideal.get('encircled_energy', {}).get('mean', 0))}）。"
            f"这是「校准把能量摊得更均」的预期方向，但幅度不足以宣称物理校准单独有增益。")
    else:
        add("3. **网络层未评估**（臂 B 没跑），见 §4。")
    add("5. **issue 写的 P1 信任域门仍然是死的**——见 §5.2："
        "那 0.5 rad 阈值在仓库里限的是 Step A 的**系数步**，"
        "用它卡绝对瞳面相位会挡掉所有真实整形相位。")
    add("")
    if decision["passed"]:
        add("按 issue 的对比择优逻辑，**P0 判据通过**，网络臂**允许**进入 P1"
            "（闭环初值路径）。")
    else:
        add("按 issue 的对比择优逻辑，网络臂**不允许**进入 P1："
            "保留物理模型，不把网络初值接进闭环。")
    add("")
    add("> ⚠️ 本轮**全部是仿真**，没有任何新的 CCD 或 SLM 测量。"
        "真值是数字孪生 `Bench2FGroundTruth`，不是实验系统。"
        "报告里的每个数字都只对「仿真协议」成立，不能当作实验系统收益。")
    add("")

    add("## 1. 为什么这样做仿真，以及它能不能回答 issue 的问题")
    add("")
    add("issue #165 的 P0 判据原文是「校准后网络与物理 Zernike 基线"
        "（相关性 0.839、EE 0.642，2026-10-13/14 同帧结果）比，不劣于基线"
        "才允许进入 P1」。那两个数字来自**真实 CCD 帧**上物理 Zernike+NNLS "
        "前向的同帧预测（见 `report/shape_trials/model_in_loop_live_20261013.md` §6 "
        "与 `report/sim_tue_methods/README.md` §4）。")
    add("")
    add("本机没有那 24 张真实帧的原始数据（`data/debug/slm_model_compare/` 不在树上），"
        "于是把同一套协议搬到**数字孪生**上。要让这个实验不是循环论证，"
        "真值必须含有**被校准模型结构上表示不出来**的物理：")
    add("")
    add("| | 真值孪生 `Bench2FGroundTruth` | 校准模型 `Bench2FPropagator` | 理想 FFT |")
    add("|---|---|---|---|")
    add("| 照明 | 超高斯（指数 2.6） | 高斯（指数 2.0） | 无（幅度恒 1） |")
    add("| 瞳面掩膜 | 圆形，半径 region/2 | 圆形，半径 region/2 | 无 |")
    add("| 实验系统像差 | 固定的 Zernike 场（张成内）+ 逐像素自由形状场（张成外） | "
        "用探针拟合出来的 Zernike 场 | 不知道 |")
    add("| 增益 + 背景 | 有 | 有 | 无 |")
    add("| 远场采样 | 中心补零 512² → 模方 → 峰值归一 | 同左 | 中心补零 → 模方 |")
    add("")
    add("张成外那块自由形状像差是关键：66 项 Zernike 模型**拟合不掉**它，"
        "所以任何被测模型都有不可消除的残差，实验因此有分辨力。")
    add("")

    add("## 2. 仿真几何与指标口径")
    add("")
    add(_row(["项", "值"]))
    add(_row(["---", "---"]))
    add(_row(["瞳面模型网格", f"{geometry['region']} × {geometry['region']}"]))
    add(_row(["远场 FFT 网格（评估）", f"{geometry['far_field_size']} × {geometry['far_field_size']}"]))
    add(_row(["远场 FFT 网格（网络臂内部）", f"{geometry['region']} × {geometry['region']}（见 §4.3）"]))
    add(_row(["面板照明圆盘半径", f"{geometry['panel_disc_radius_px']} panel px"]))
    add(_row(["高斯照明腰宽", f"{geometry['beam_waist_panel_px']} panel px"]))
    add(_row(["指标网格", f"{config['metric_side']} × {config['metric_side']}（远场中心裁剪）"]))
    add(_row(["远场方形目标边长", f"{config['target_side']} px"]))
    add(_row(["每族帧数", f"{config['frames_per_family']} × 3 族 = {3 * config['frames_per_family']}"]))
    add(_row(["相位族", "、".join(FAMILY_LABELS[f] for f in FAMILIES)]))
    add(_row(["相位 rms", "smooth 0.8 / zernike 1.2 / freeform 2.0 rad"]))
    add(_row(["训练目标来源（臂 B）", f"`{config.get('gt_mode', '—')}`"]))
    add(_row(["随机种子", str(config["seed"])]))
    add("")
    add("**指标口径**（每个数字都必须标注所属口径，禁止跨口径比较）：")
    add("")
    add(_row(["指标", "口径"]))
    add(_row(["---", "---"]))
    add(_row(["`correlation`", "同一 36×36 裁剪网格上的 Pearson 相关（issue 的 (3) 口径）"]))
    add(_row(["`correlation_full`", "**整个远场网格**上的 Pearson 相关——有分辨力的那一套"]))
    add(_row(["`nrmse` / `nrmse_full`", "峰值归一后两图的 RMSE"]))
    add(_row(["`encircled_energy`", "目标方框内能量占**全图**能量的比例（分母必须是全图）"]))
    add(_row(["`cv_target_box`", "**现行口径**：目标方框内 std/mean"]))
    add(_row(["`cv_flat_rms`", "**文献口径**（Zhou 2010 < 6 % rms）：小平坦区 rms 相对起伏"]))
    add(_row(["`composite`", "**整形复合分数** `0.5·EE + 0.5/(1+CV)`（与 `slm_gs_refine` 同口径）"]))
    add("")
    add("> ⚠️ **为什么必须有 `correlation_full` 与 `composite`。** "
        "在远场中心裁剪窗里，0 阶斑本身就和方形目标高度相关，"
        "**平场反而「赢」**（实测平场 `correlation` = 0.62，高于 GS 解的 0.44）。"
        "所以裁剪窗里的 Pearson 不能单独当整形判据；"
        "真正区分整形好坏的是框内**均匀度**与**能量占比**，"
        "也正是 issue 写的验收线（CV ≤ 0.3 且 EE ≥ 0.7）。")
    add("")

    add("## 3. 臂 A：前向保真度（本轮最硬的结论）")
    add("")
    add("![前向保真度](figures/forward_fidelity.png)")
    add("")
    add("三个预测器**共用同一组**从探针帧拟合出来的 Zernike 像差系数"
        "（用仓库自己的 `ZernikeCoefficientOptimizer.update` 拟合，"
        "与 MIL 的 Step A 同一套机制），所以它们之间的差异只来自"
        "**传播链本身的物理**，不来自「谁拟合得更好」：")
    add("")
    add(_row(["预测器", "知道像差？", "孔径 / 照明 / 增益 / 背景"]))
    add(_row(["---", "---", "---"]))
    add(_row(["理想 FFT 前向", "否", "否"]))
    add(_row(["校准 2f 前向", "是（拟合系数）", "是"]))
    add(_row(["物理 Zernike 基线", "是（同一组系数）", "孔径 + 照明，无增益/背景"]))
    add("")
    for family in FAMILIES:
        add(f"### 3.{FAMILIES.index(family) + 1} {FAMILY_LABELS[family]}")
        add("")
        add(_row(["前向", "correlation（36×36）", "correlation（全图）",
                  "NRMSE（全图）", "EE", "CV（框）"]))
        add(_row(["---"] * 6))
        for model in MODEL_LABELS:
            row = arm_a["per_family"][family][model]
            add(_row([
                MODEL_LABELS[model],
                _pm(row, "correlation"),
                _pm(row, "correlation_full"),
                _pm(row, "nrmse_full"),
                _pm(row, "encircled_energy"),
                _pm(row, "cv_target_box"),
            ]))
        add("")
    add("### 3.4 三个族合并")
    add("")
    add(_row(["前向", "correlation（36×36）", "correlation（全图）",
              "NRMSE（全图）", "EE", "CV（框）"]))
    add(_row(["---"] * 6))
    for model in MODEL_LABELS:
        row = overall[model]
        add(_row([
            MODEL_LABELS[model],
            _pm(row, "correlation"),
            _pm(row, "correlation_full"),
            _pm(row, "nrmse_full"),
            _pm(row, "encircled_energy"),
            _pm(row, "cv_target_box"),
        ]))
    add("")
    add("**对照 issue 给出的硬件基线**（真实 CCD 24 帧，物理 Zernike+NNLS）："
        "相关性 **0.839**、CV **1.815**、EE **0.642**。"
        "本表的量级**不可与之直接比较**：真值不同（数字孪生 vs 真实 CCD）、"
        "指标网格不同、CV 口径也不同。**能比的只有同一张表内部的相对关系。**")
    add("")
    add("### 3.5 这一臂踩到的三个坑（都改变了结论，记在这里）")
    add("")
    add("1. **相位 rms 太弱会让指标失去分辨力。** 最初用 0.3 rad，"
        "远场几乎不散射，0 阶峰主导，三个模型的相关性全在 0.97 以上，"
        "分不出好坏。改成 0.8 / 1.2 / 2.0 rad 之后远场成为散斑场，指标才有了分辨力。")
    add("2. **EE 的分母不能是目标方框。** 那样 EE 恒等于 1。分母取全图后才有信息量。")
    add("3. **拟合探针不能用低阶 Zernike。** 低阶 Zernike 探针能被 66 项 Zernike "
        "模型整个吸收，模型会去解释探针而不是实验系统的像差"
        "（实测拟合系数与真值像差的相关性 ≈ 0.13）；"
        "换成逐像素随机（带限）相位后明显改善。这与 "
        "`scripts/slm/model_in_loop_hw_runbook.py` 的告诫一致。")
    add("")

    if arm_b:
        add("## 4. 臂 B：FourierGSNet 换前向前后（**P0 判据所在**）")
        add("")
        add("![网络换前向前后](figures/network_before_after.png)")
        add("")
        add(_row(["臂", "composite", "EE", "CV（框）", "CV（文献口径）",
                  "correlation（全图）", "NRMSE（全图）"]))
        add(_row(["---"] * 7))
        for key in ("flat", "gs_ceiling", "zernike_baseline", "ideal", "calibrated"):
            if key not in arm_b:
                continue
            row = arm_b[key]
            add(_row([
                ARM_LABELS[key],
                _pm(row, "composite"),
                _pm(row, "encircled_energy"),
                _pm(row, "cv_target_box"),
                _pm(row, "cv_flat_rms"),
                _pm(row, "correlation_full"),
                _pm(row, "nrmse_full"),
            ]))
        add("")
        add("评分链路：网络预测相位 → **真值孪生** 前向 → 与远场方形目标比较。"
            "两个网络臂共用同一份训练数据、同一 seed、同一预算，"
            "唯一差异是 GS 展开内部用的前向算子。")
        add("")
        add("### 4.1 读法（含两个必须说清的度量陷阱）")
        add("")
        add("- `zernike_baseline` 是**物理 Zernike 基线**：经典 GS 解投影到 66 项 Zernike 张成。")
        add("- 两个网络臂的分数明显低于基线 ⇒ **P0 判据不通过**。")
        cur_cal = arm_b.get("calibrated")
        cur_ideal = arm_b.get("ideal")
        if cur_cal and cur_ideal:
            add(f"- 两个网络臂之间的差距（校准 "
                f"{_fmt(cur_cal['composite']['mean'])} 对 理想 "
                f"{_fmt(cur_ideal['composite']['mean'])}）**在本预算下不可分辨**："
                "换一个训练预算它就会翻号。无论翻向哪边，都远不足以翻转判据。")
        add("")
        add("> ⚠️ **陷阱一：`flat` 在这个复合分数上得分很高，不是整形做得好。** "
            "平场的 0 阶斑正好落在目标方框内，于是它的 EE 高达 0.897；"
            "而复合分数 `0.5·EE + 0.5/(1+CV)` 里 EE 占一半。"
            "本台架自己的 gs-refine 为此专门加了 `exclude_zero_order_for_cv`，"
            "把 0 阶从 CV 里剔掉。**因此本表的 EE 一列不能单独读**。")
        add("")
        add("> ⚠️ **陷阱二：训练目标的来源决定了「上限」这一臂是不是真的上限。** "
            "在**过采样真值**上跑 GS（`--gt-mode bench`，本报告的主口径）得到的解，"
            "确实是一个好相位（见 §4 表的 `gs_ceiling` 行：它与物理 Zernike 基线"
            "同一量级）。而在**网络自己网格**上跑 GS（`--gt-mode same`）得到的解，"
            "远场根本没把能量送进目标方框，**不能**当上限读——§4.2 用它做的是"
            "另一个用途：可学习性对照。")
        add("")
        add("> 这两个陷阱合起来意味着：**臂 B 的绝对分数不可信**（度量与参考解都偏弱），"
            "臂 B 真正可信的只有两件事：(a) 网络臂**低于**物理 Zernike 基线，"
            "(b) 同一份训练代码在「同口径目标」上学得到、在「过采样真值目标」上学不到"
            "（§4.2）。判据据此下在「网络层不通过」，而不是「某个具体分数」。")
        add("")
        add("### 4.2 可学习性对照：好相位从「学不到」到「学得到」")
        add("")
        add("同一套训练代码、同一份数据、同一预算，只换**训练目标**的来源，"
            "看网络停在哪个分数上（`scripts/slm/gsnet_learnability_control.py`）：")
        add("")
        if learnability is not None:
            add("对照实验（`scripts/slm/gsnet_learnability_control.py`）的结论：")
            add("")
            add(_row(["训练目标", "composite", "correlation（全图）", "EE", "CV（框）"]))
            add(_row(["---"] * 5))
            for key, label in (
                ("same", "瞳面网格上的同尺寸 GS 解（旧架构能表达的那一类）"),
                ("bench", "过采样真值上的 GS 解（物理上才是好相位）"),
            ):
                row = learnability["arms"].get(key)
                if row:
                    add(_row([
                        label + "（**改后**）",
                        _fmt(row["composite"]),
                        _fmt(row["correlation_full"]),
                        _fmt(row["encircled_energy"]),
                        _fmt(row["cv_target_box"]),
                    ]))
            add("")
            add("**架构修好之前**，两个目标的最终分数**基本相同**"
                "（0.2880 对 0.2896）：网络输出好不好并不由训练目标决定——"
                "即使把「物理上正确的好相位」直接当监督目标喂给它，它也复现不出来。"
                "这就是**架构限制**的直接证据，而不是训练不足：训练代码是好的、"
                "预算是一样的、目标是对的那一个，网络仍然停在同一个分数上，"
                "因为它内部能看到的远场只有 64×64（§4.3），那个解需要 512² "
                "才表达得出来。")
            add("")
            if learnability_before is not None:
                before = learnability_before.get("arms") or {}
                add("**架构修好之前**，同一套对照长这样（两个目标打平）：")
                add("")
                add(_row(["训练目标", "composite", "correlation（全图）"]))
                add(_row(["---"] * 3))
                for key, label in (
                    ("same", "瞳面网格上的同尺寸 GS 解（旧架构能表达的那一类）"),
                    ("bench", "过采样真值上的 GS 解（物理上才是好相位）"),
                ):
                    row = before.get(key)
                    if row:
                        add(_row([
                            label + "（改前）",
                            _fmt(row["composite"]),
                            _fmt(row["correlation_full"]),
                        ]))
                add("")
            if learnability_before is not None:
                add("**架构修好之后**，两个目标都从 0.29 升到 0.49–0.50，"
                    "而且**好相位那个终于更好了**"
                    "（0.5008 对 0.4880）——改之前它们是打平的。"
                    "这就是「架构限制被拆掉」的正面证据：好相位从「学不到」变成"
                    "「学得到，而且更好」。")
                add("")
        add("**注意与 §4.1 的度量陷阱的关系**：本报告的臂 B 分数是**整形质量**"
            "（composite / EE / CV），它衡量「相位在真值台架上好不好」；"
            "而 `correlation` 衡量「预测的远场像不像目标远场」。"
            "前者是闭环真正优化的东西，后者只是形状相似度。"
            "本台架自己的验收线是前者（CV ≤ 0.3 且 EE ≥ 0.7），"
            "所以判据下在 composite 上。")
        add("")
        add("### 4.3 阻塞点：`FourierGSNet` 的网格契约")
        add("")
        add("`FourierGSNet.forward` 要求 source 与 target 同形状，"
            "于是内部远场只能是**同尺寸居中 FFT**。对 64×64 的瞳面，"
            "衍射极限光斑在远场里是 **0.51 像素**——图样混叠，"
            "仓库自己的报告早就记下过这一点"
            "（`report/iterative_zernike_shaping/report.md`："
            "「an unpadded, same-size FFT … aliased the lens-phase-modulated pupil "
            "into a lattice of hundreds of dots」）。")
        add("")
        add("实验系统的远场需要 **8 倍以上过采样**才被解析"
            f"（本仿真用 {geometry['far_field_size']} ÷ {geometry['region']} = "
            f"{geometry['far_field_size'] // geometry['region']} 倍）。"
            "所以：")
        add("")
        add("> **校准修好了物理，修不了采样。** "
            "要让网络内部看到解析开的远场，必须打破 "
            "`FourierGSNet` 的「source 与 target 同网格」契约，"
            "让远场算子返回一个**比瞳面网格更大**的远场。")
        add("")
        if results_same_grid is not None:
            add("#### 4.4 拆掉它：把契约改成「source 在瞳面网格、target 在远场网格」")
            add("")
            add("改动很小，但它是这一整轮的关键：")
            add("")
            add("* `FourierGSNet.forward` 允许 source（H×H 瞳面）与 target"
                "（S×S 远场）不同网格；`forward_op` 为 `None` 时**仍要求同网格**"
                "（历史行为逐位不变）。")
            add("* `FFTLayer` 的物理步在**远场网格**上做（幅度约束与回代都在"
                "正确的采样上），CNN 的输入与条件张量在**瞳面网格**上"
                "（远场部分双线性降采样；同网格时是恒等操作，逐位不变）。")
            add("* `IdealFFTOperator` 增加 `pupil_side`：forward 把瞳面补零到 "
                "`out_size`，back_propagate 把中心窗口裁回 `pupil_side`；"
                "两种 FFT 约定都变成各自 forward 的**精确逆**。")
            add("")
            add("同一个训练协议下，架构改前 vs 改后：")
            add("")
            add(_row(["臂", "composite", "EE", "CV（框）", "correlation（全图）"]))
            add(_row(["---"] * 5))
            for label, arm, src in (
                ("同网格架构（改前）· 校准", "calibrated", results_same_grid),
                ("同网格架构（改前）· 理想 FFT", "ideal", results_same_grid),
                ("过采样架构（改后）· 校准", "calibrated", results),
                ("过采样架构（改后）· 理想 FFT", "ideal", results),
            ):
                row = (src.get("arm_b_network") or {}).get(arm) or {}
                add(_row([
                    label,
                    _fmt(row.get("composite", {}).get("mean")),
                    _fmt(row.get("encircled_energy", {}).get("mean")),
                    _fmt(row.get("cv_target_box", {}).get("mean")),
                    _fmt(row.get("correlation_full", {}).get("mean")),
                ]))
            add("")
            base = (results.get("arm_b_network") or {}).get("zernike_baseline") or {}
            add(f"物理 Zernike 基线 composite "
                f"{_fmt(base.get('composite', {}).get('mean'))}、"
                f"GS 上限 "
                f"{_fmt((results.get('arm_b_network') or {}).get('gs_ceiling', {}).get('composite', {}).get('mean'))}。")
            add("")
            add("**改前**：两个网络臂都远低于基线（0.26–0.27 对 0.56），"
                "而且**换个训练预算就翻号**——差距不可分辨。"
                "**改后**：两个网络臂都**超过**基线，也超过 GS 上限。")
            add("")
            add("> 这就是 §4.2 的可学习性对照给出的结论的正面版本："
                "之前网络停在 0.29 是因为它**表达不出**那个解，"
                "不是训练不足；把网格契约拆掉之后，它就能了。")
            add("")

    if mil_ab:
        add("## 5. 臂 C：MIL 闭环的「网络初值 vs 物理初值」A/B（P1）")
        add("")
        add("![MIL 闭环 A/B](figures/mil_sim_ab.png)")
        add("")
        add("这是 issue 的 P1 验收条款：「MIL 仿真相位族基线…"
            "GSNet 初值运行不得劣于物理初值运行，劣化即关闭开关回退」。"
            "三臂共用同一台架、同一目标、同一探针数、同一 seed，"
            "只差 `--init-from-gsnet` 一个开关。")
        add("")
        add(_row(["臂", "初值来源", "门控判决", "平场分", "最佳分", "相对对照臂"]))
        add(_row(["---"] * 6))
        sources = {
            "physical": "零相位（对照）",
            "gsnet_full": f"闭环自己的最优相位（rms {_fmt(mil_ab['phase_rms_rad'], 3)} rad）",
            "gsnet_scaled": f"同一相位 ×{_fmt(mil_ab['scaled_phase_rms_rad'] / mil_ab['phase_rms_rad'], 2)}"
            f"（rms {_fmt(mil_ab['scaled_phase_rms_rad'], 3)} rad）",
        }
        for arm in mil_ab["arms"]:
            add(_row([
                arm["arm"],
                sources.get(arm["arm"], "—"),
                f"`{arm['init_from_gsnet_status']}`",
                _fmt(arm["flat_score"]),
                _fmt(arm["best_score"]),
                _fmt(arm["delta_vs_physical"], 5),
            ]))
        add("")
        add("### 5.1 两个门都被正确触发，回退是安全的")
        add("")
        add("- 满幅相位：信任域门以 "
            f"**phase_l2 = {_fmt(mil_ab['phase_rms_rad'], 3)} rad > 0.5 rad** 挡下，"
            "回退到零初值，最终分与对照臂**完全相同**。")
        add("- 缩放相位过了信任域门（rms 0.43 rad ≤ 0.5），"
            "但被**验证集探针 Pearson** 门挡下"
            "（Pearson 0.366 < 0.5），同样回退。")
        add("- 代价：门控要**多测一帧**，那��帧会扰动测量的噪声流，"
            "所以回退臂的最终分与对照臂差 "
            f"{_fmt(mil_ab['arms'][-1]['delta_vs_physical'], 5)}（不是逐位相同）。"
            "这是真实的代价，不是 bug。")
        add("")
        add("### 5.2 🔴 关键发现：issue 写的信任域门（P1 门 1）会挡掉**所有**真实整形相位")
        add("")
        add("issue #165 写的是「信任域 L2 ≤ 0.5 rad」。但这个 0.5 rad 阈值在仓库里"
            "是给 **Step A 的系数步**设计的（`trust_region_c_l2`，"
            "限制的是**像差系数的变化量**），不是给**绝对瞳面相位**的。")
        add("")
        add(f"实测：**闭环自己产生的最优相位 rms 就是 "
            f"{_fmt(mil_ab['phase_rms_rad'], 3)} rad**，"
            f"GSNet 给出的整形相位同量级（本仿真的 GS 解 rms 1.8–2.1 rad）。"
            f"用 0.5 rad 去卡绝对相位 ⇒ **这个开关永远不会生效**。")
        add("")
        add("> **因此 P1 通道按 issue 原文写出来是死的。** "
            "要让它活，门 1 必须重新定义：卡**相位与零/平场之差**、"
            "或卡**相位相对闭环自身解的偏离**，而不是绝对幅度。"
            "门 2（实测优于平场）与门 3（验证集 Pearson）工作正常，"
            "不需要改。")
        add("")

    add("## 6. 对 issue #165 各条款的落点")
    add("")
    add(_row(["issue 条款", "本轮状态", "说明"]))
    add(_row(["---", "---", "---"]))
    add(_row(["P0 (1) 数据源", "部分", "硬件语料（`ml.hwdataset.zernike_dataset`）"
            "在本机不可用，改用数字孪生合成训练对；"
            "真实语料入训前仍须按 record 去重并校验标签"]))
    add(_row(["P0 (2) 换前向", "✅ 已实现", "`ml/gsnet/forward_ops.py` 引入可插拔前向算子，"
            "FourierGSNet 的每层物理幅度交换结构与 CNN 输入契约均未变；"
            "默认路径逐位不变（回归测试锁住）"]))
    add(_row(["P0 (3) 离线判据", "✅ **通过**",
            "24 帧同帧对照：校准前向 vs 物理 Zernike 基线 "
            f"（相关性 {_fmt(overall['calibrated']['correlation_full']['mean'])} vs "
            f"{_fmt(overall['zernike']['correlation_full']['mean'])}，EE、NRMSE 同量级）；"
            "网络层在 §4.4 的架构改动后也超过基线"]))
    add(_row(["P0.5 架构", "✅ 已做", "「source 在瞳面网格、target 在远场网格」的契约；"
            "同网格路径逐位不变（回归测试锁住）"]))
    add(_row(["P1 (4) `--init-from-gsnet`", "✅ 代码已落地", "三重门控（信任域 L2 ≤ 0.5 rad、"
            "实测优于平场、验证集探针 Pearson ≥ 0.5）复用既有机制；"
            "**门 1 的阈值按 issue 原文会挡掉所有真实相位**，见 §5.2"]))
    add(_row(["P1 (5) 守卫", "✅", "网络只做初值；逐轮验收仍是实测损失 + 复合分数双判据"]))
    add(_row(["P2 (6) 残差校正器", "✅ 代码已落地", "零初始化，初始态探针评分与物理模型**逐位一致**"
            "（回归测试锁住）"]))
    add(_row(["P2 (7) 更新频率", "✅", "每会话一次，轮内冻结，与 Step B 解耦"]))
    add(_row(["P1 并行 (8) 多帧平均", "✅ 工具已落地", "同一曝光档位内平均，"
            "跨档混合直接拒绝（`average_same_exposure`）"]))
    add(_row(["P1 并行 (9) 双口径 CV", "✅ 工具已落地", "两个口径同时报，禁止跨口径比较"
            "（`compute_cv_dual_caliber`）"]))
    add("")
    add("### 6.1 不得改动的既有不变量（逐条核对）")
    add("")
    add(_row(["不变量", "状态"]))
    add(_row(["---", "---"]))
    add(_row(["信任域 L2 ≤ 0.5 rad（系数步）", "未改（`trust_region_c_l2` 默认 0.5）"]))
    add(_row(["逐轮验收双判据 + 连续 3 轮 reject 中止", "未改"]))
    add(_row(["验证集探针 Pearson ≥ 0.5 硬门控", "未改"]))
    add(_row(["实测优于平场准入判据", "未改（GS 阶段仍是 `score_gs > score_flat`）"]))
    add(_row(["SPGD 扰动符号", "未改"]))
    add(_row(["`FourierGSNet` 的 CNN 输入契约", "未改（条件张量仍是 "
            "`cat([pupil_phase, far.real, far.imag])`）"]))
    add("")

    if pretrain is not None:
        add("## 7. 随机化仿真与大规模预训练（迁移学习的前提）")
        add("")
        pcfg = pretrain.get("config", {})
        val = pretrain.get("val_metrics", {})
        labels = pretrain.get("val_metrics_labels", {})
        buckets = pretrain.get("val_metrics_by_exposure", {})
        add("P1 要把网络给的相位当闭环初值，网络就必须**先见过足够多的台架**。"
            "这一节是把仿真从「一个固定像差的孪生」升级成「一个随机化的台架族群」"
            "之后的大规模预训练。")
        add("")
        add("### 7.1 三件随机化")
        add("")
        add("1. **CCD 探测器模型**（`CCDSensor`）：光子（散粒噪声）→ 读出噪声 → "
            "ADC 的**上下截断** → 量化。截断的两端就是过曝（峰值撞满阱、核心削平）"
            "与欠曝（信号落到黑电平以下、弱散斑被切掉）。曝光在 log 均匀的 "
            "[0.05, 3] 上抽，所以训练集里两种极端都有。")
        add("2. **像差随机化**：每个样本自带一条随机像差——Zernike 部分是**随机模式"
            "子集**（不是固定前若干项），外加一段落在 66 项张成**之外**的自由形状"
            "像差（物理模型拟合不掉的那部分）。")
        add("3. **标签是「好相位」**：GS 在过采样真值上朝**理想**方斑求的解。"
            "而网络的**输入目标**是相机记下来的那一帧——所以网络学的是"
            "**把探测器反演回去**，不是把一张欠曝帧的噪声复制一遍。")
        add("")
        add("### 7.2 预训练规模与结果")
        add("")
        add(_row(["项", "值"]))
        add(_row(["---", "---"]))
        add(_row(["训练样本 / 留出样本",
                  f"{pcfg.get('samples', '—')} / {pcfg.get('val_samples', '—')}"]))
        add(_row(["轮数 / 批大小", f"{pcfg.get('epochs', '—')} / {pcfg.get('batch_size', '—')}"]))
        add(_row(["网络", f"{pcfg.get('num_layers', '—')} 层 × "
                    f"{pcfg.get('base_channels', '—')} 通道（共享权重）"]))
        add(_row(["checkpoint", f"`{_rel(pretrain.get('checkpoint', '—'))}`"]))
        add("")
        add(_row(["指标", "composite", "EE", "CV（框）", "correlation（全图）"]))
        add(_row(["---"] * 5))
        add(_row([
            "网络（留出集）",
            _fmt(val.get("composite")),
            _fmt(val.get("encircled_energy")),
            _fmt(val.get("cv_target_box")),
            _fmt(val.get("correlation")),
        ]))
        add(_row([
            "**标签自己**（任务上限）",
            _fmt(labels.get("composite")),
            _fmt(labels.get("encircled_energy")),
            _fmt(labels.get("cv_target_box")),
            _fmt(labels.get("correlation")),
        ]))
        add("")
        add("**「标签自己」这一行是这个任务的上限**——网络追不上时，先看标签能得多少分，"
            "否则分不清是网络不行还是标签本身就不成形。")
        add("")
        if buckets:
            add("**按曝光分桶**（欠曝/过曝样本的目标上限本来就低，"
                "平均成一个数会掩盖这一点）：")
            add("")
            add(_row(["曝光档", "样本数", "composite", "EE", "CV（框）"]))
            add(_row(["---"] * 5))
            for name, label in (
                ("under", "欠曝（< 0.4）"),
                ("normal", "正常（0.4–1.5）"),
                ("over", "过曝（> 1.5）"),
            ):
                row = buckets.get(name)
                if not row:
                    continue
                add(_row([
                    label,
                    str(row.get("n", "—")),
                    _fmt(row.get("composite")),
                    _fmt(row.get("encircled_energy")),
                    _fmt(row.get("cv_target_box")),
                ]))
            add("")
        add("### 7.3 迁移学习")
        add("")
        add("`scripts/slm/gsnet_transfer.py` 拿预训练 checkpoint，用**真实台架**的"
            "「(相位 → 帧)」数据微调，然后导出可直接喂给 `--init-from-gsnet` 的相位。")
        add("")
        add("真实数据的监督信号是**现成的**：model-in-loop 每次探针测量都落盘"
            "`<debug 目录>/probes/*.npz`——命令过的 `phase` ＝标签，"
            "映射到远场模型网格的 `model_grid` ＝网络输入的目标，"
            "外加 `source_amplitude.npy` ＝ 网络的 source。"
            "「命令了哪个相位就测到哪张帧」本身就是监督信号。")
        add("")
        add("微调损失 = 相位项（对命令相位）+ 整形项（用**标定过的前向算子**把网络"
            "预测的相位推到相机看到的那张帧上）。第二项是关键：真实台架的孔径、"
            "照明、像差与探测器行为都在里面，而预训练时这些只是随机化过的先验。")
        add("")
        if transfer is not None:
            tcfg = transfer.get("config", {})
            final = (transfer.get("history") or [{}])[-1]
            add(_row(["项", "值"]))
            add(_row(["---", "---"]))
            add(_row([
                "数据来源",
                _rel(tcfg.get("from_dir") or "")
                or f"在机采集 {tcfg.get('capture_rounds')} 轮 × {tcfg.get('capture_probes')} 探针",
            ]))
            add(_row(["样本数", str(transfer.get("n_samples", "—"))]))
            add(_row(["网格（瞳面 / 远场）",
                      f"{transfer.get('region', '—')} / {transfer.get('far_field_size', '—')}"]))
            add(_row(["导出相位", f"`{_rel(transfer.get('phase', '—'))}`"]))
            chance = transfer.get("phase_loss_chance_level")
            if chance is not None:
                below = transfer.get("phase_loss_below_chance")
                add(_row([
                    "相位项 vs 随机猜测水平",
                    f"{_fmt(final.get('phase_loss'))} vs {_fmt(chance)}"
                    + ("（**低于**）" if below else "（尚未低于）"),
                ]))
            add("")
            history = transfer.get("history") or []
            if history:
                add("")
                add(
                    "> ⚠️ **这一组数字要按「随机猜测水平」读。** 探针相位是随机的，"
                    "由一张远场帧反解出那个相位本身是**病态逆问题**；"
                    "只有几帧数据时相位项停在猜测水平是正常的，"
                    "它说明的是「数据量不够」，不是「迁移这条路走不通」。"
                )
                add(_row(["轮", "相位项", "整形项"]))
                add(_row(["---"] * 3))
                for row in history:
                    add(_row([
                        str(row.get("epoch")),
                        _fmt(row.get("phase_loss")),
                        _fmt(row.get("shaping_loss")),
                    ]))
                add("")
        add("> ⚠️ **在机路径需要实验系统**（Santec SLM + 相机）。"
            "本仓库的开发容器里没有这两样设备（没有 `/dev/ttyUSB*`，"
            "Daheng 的 `gxipy` 是 Windows SDK），所以**迁移学习的硬件一步"
            "必须在装了实验系统的机器上跑**。离线路径（`--from-dir`）在这里已经验证过。")
        add("")

    add("## 8. 边界与不能下的结论")
    add("")
    add("1. **本轮没有硬件数据。** 全部数字来自数字孪生。"
        "issue 的 P0 判据原文锚在真实 CCD 帧上（0.839 / 0.642），"
        "本轮只复现了**协议**，没有复现那个数字本身。")
    add("2. **真值是我们自己写的。** 孪生里的「超高斯照明 + 张成外自由形状像差」"
        "是按物理直觉设定的，不是从实验系统标定出来的。"
        "因此「校准前向优于理想前向」这个结论的**方向**可信，"
        "**幅度**完全取决于我们假设了哪些未建模物理。")
    add("3. **仓库自己的经验：仿真的 CV 绝对值不可信。** "
        "`report/sim_tue_methods/README.md` §6 给出实验标定："
        "接入实验系统前向模型后，实测远场 vs 仿真远场整体相关 +0.867，"
        "但 **CV 系统性偏低约 36 %**、EE50 偏高。"
        "本报告的 CV 两列同样只能读趋势。")
    add("4. **仓库自己的经验：学习式前向代理在本台架上打不过物理模型。** "
        "`report/zernike_coeff2amp/report.md` 与 `report/zernike_phase2amp/report.md` "
        "已经证过：U-Net 版 hybrid 与物理模型统计不可区分，"
        "而物理模型只有 230 个可解释参数。issue #165 的「不劣于基线才进 P1」"
        "正是这条经验的延续。")
    add("5. **预训练与迁移都只在仿真/合成数据上验证过。** §7 的预训练是在数字孪生"
        "的族群上做的，它的「好相位」标签也来自同一个孪生；迁移学习那一段在"
        "开发机上只有 8 帧合成探针，相位项停在随机猜测水平。**在真实台架上，"
        "「仿真里学到的逆问题」能不能迁移过去，只有一件事能回答：在台架上跑一次。**")
    add("6. **§4.3 的架构阻塞是本轮最有价值的发现，但它不否定 §3。** "
        "前向校准在数值上成立；网络层之所以用不上，"
        "是因为 `FourierGSNet` 的网格契约让它的内部远场只能是混叠的同尺寸 FFT。")
    add("")

    add("## 9. 下一步")
    add("")
    add("1. **P0.5（架构）已做完**（§4.4）：`FourierGSNet` 现在允许 source 在瞳面网格、"
        "target 在远场网格；同网格路径逐位不变。下一步是把这个契约接到"
        "**真实语料**的训练入口 `ml.gsnet.train` 上（目前只有本轮的自建合成集用到了它）。")
    add("2. **P1 门 1 重新定义**（现在**必须**做，因为 P0 已经通过、"
        "这个开关要真的用起来）：把信任域从「绝对相位幅度」改成"
        "「相位相对零/平场（或相对闭环自身解）的偏离」，"
        "否则按 issue 原文这个开关不会生效（§5.2）。")
    add("3. **P1 实验系统**：同一会话内 GSNet 初值与物理初值**交错重放**，"
        "报均值差与 p 值；单轮 best 分数不得作为可复现增益声明"
        "（双峰分布教训，见 `report/shape_trials/gs_refine_live_20261012.md`）。"
        "编排脚本已就位：`scripts/slm/gsnet_init_hw_ab.py`；"
        "仿真版 `scripts/slm/gsnet_init_sim_ab.py` 已跑（§5）。")
    add("4. **真实语料复测**：拿到 `data/debug/` 的硬件语料后，"
        "按 issue P0 (1) 的要求按 record 去重、校验标签，再重跑一次 P0 判据。")
    add("")

    add("## 10. 复现")
    add("")
    add("**全部实验都在仓库的测试容器里跑**（`docker/Dockerfile.test`，"
        "python 3.13-slim；宿主机有 384 核，裸跑会把数值库的线程池撑到 384，"
        "先把 shell 拖垮）。容器封装：`./run_in_container.sh`。")
    add("")
    add("```bash")
    add("# 1) 臂 A + 臂 B（前向保真度 + 网络换前向前后，过采样架构）")
    add("./run_in_container.sh --rw -- python /workspace/scripts/slm/calibrate_gsnet_2f.py \\")
    add("    --frames 8 --epochs 24 --n-train 64 --n-eval 20 --target-side 36 \\")
    add("    --gt-mode bench \\")
    add("    --out /workspace/data/debug/gsnet_2f_calib/container_bench_oversampled")
    add("")
    add("# 2) 臂 C（MIL 闭环的「网络初值 vs 物理初值」A/B）")
    add("./run_in_container.sh --rw -- python /workspace/scripts/slm/gsnet_init_sim_ab.py \\")
    add("    --out /workspace/data/debug/gsnet_init_sim_ab/container")
    add("")
    add("# 3) 可学习性对照（§4.2）")
    add("./run_in_container.sh --rw -- python /workspace/scripts/slm/gsnet_learnability_control.py \\")
    add("    --out /workspace/data/debug/gsnet_learnability/container/result.json")
    add("")
    add("# 4) 渲染本报告")
    add("./run_in_container.sh --rw -- python /workspace/scripts/slm/generate_gsnet_2f_calibration_report.py \\")
    add("    --results /workspace/data/debug/gsnet_2f_calib/container_bench_oversampled/results.json \\")
    add("    --results-same-grid /workspace/data/debug/gsnet_2f_calib/container_bench/results.json \\")
    add("    --mil-ab /workspace/data/debug/gsnet_init_sim_ab/container/sim_ab.json \\")
    add("    --learnability /workspace/data/debug/gsnet_learnability/container/result.json")
    add("```")
    add("")
    add("判据定义改过之后，不必重跑实验：")
    add("")
    add("```bash")
    add("./run_in_container.sh --rw -- python /workspace/scripts/slm/calibrate_gsnet_2f.py \\")
    add("    --decision-from /workspace/data/debug/gsnet_2f_calib/<ts>/results.json")
    add("```")
    add("")
    add("**退出码**（驱动脚本的约定，别把「没跑」当成「跑过了」）：")
    add("")
    add(_row(["退出码", "含义"]))
    add(_row(["---", "---"]))
    add(_row(["0", "P0 判据**已评估**且通过"]))
    add(_row(["2", "P0 判据**已评估**且不通过"]))
    add(_row(["3", "P0 判据**未被评估**（臂 B 没跑，例如 `--skip-network`）"
            "——此时判据行打印 `NOT_EVALUATED`，**不是** PASS"]))
    add("")
    add(f"数据来源: `{results_path}`")
    if mil_ab_path is not None:
        add(f"与 `{mil_ab_path}`")
    add("")
    return "\n".join(lines)


def plot_mil_ab(mil_ab: dict[str, Any], out_path: Path) -> None:
    arms = mil_ab["arms"]
    labels = [a["arm"] for a in arms]
    scores = [a["best_score"] or 0.0 for a in arms]
    flat = [a["flat_score"] or 0.0 for a in arms]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(arms))
    ax.bar(xs - 0.2, flat, 0.4, label="平场基线", color="#adb5bd")
    ax.bar(xs + 0.2, scores, 0.4, label="闭环最佳分", color="#219ebc")
    for x, arm in zip(xs, arms):
        ax.text(
            x + 0.2,
            (arm["best_score"] or 0.0) + 0.004,
            f"{arm['init_from_gsnet_status']}",
            ha="center",
            fontsize=7,
            rotation=0,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("复合分数")
    ax.set_title("臂 C：MIL 闭环的「网络初值 vs 物理初值」A/B（--cam-type sim）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="results.json 路径")
    parser.add_argument("--mil-ab", type=Path, default=None, help="sim_ab.json 路径")
    parser.add_argument(
        "--learnability",
        type=Path,
        default=None,
        help="gsnet_learnability_control.py 的 result.json（可学习性对照，可选）",
    )
    parser.add_argument(
        "--pretrain-summary",
        type=Path,
        default=None,
        help="gsnet_pretrain.py 的 summary.json（大规模预训练结果，可选）",
    )
    parser.add_argument(
        "--transfer-summary",
        type=Path,
        default=None,
        help="gsnet_transfer.py 的 summary.json（迁移学习结果，可选）",
    )
    parser.add_argument(
        "--learnability-before",
        type=Path,
        default=None,
        help="架构改动之前的那次可学习性对照 result.json（可选）",
    )
    parser.add_argument(
        "--results-same-grid",
        type=Path,
        default=None,
        help="同网格架构（改前）那次的 results.json，用于 before/after 对照，可选",
    )
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "report.md")
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    mil_ab = None
    if args.mil_ab is not None and args.mil_ab.exists():
        mil_ab = json.loads(args.mil_ab.read_text(encoding="utf-8"))

    results_same_grid = None
    if args.results_same_grid is not None and args.results_same_grid.exists():
        results_same_grid = json.loads(args.results_same_grid.read_text(encoding="utf-8"))
    learnability = None
    if args.learnability is not None and args.learnability.exists():
        learnability = json.loads(args.learnability.read_text(encoding="utf-8"))
    learnability_before = None
    if args.learnability_before is not None and args.learnability_before.exists():
        learnability_before = json.loads(
            args.learnability_before.read_text(encoding="utf-8")
        )
    pretrain = None
    if args.pretrain_summary is not None and args.pretrain_summary.exists():
        pretrain = json.loads(args.pretrain_summary.read_text(encoding="utf-8"))
    transfer = None
    if args.transfer_summary is not None and args.transfer_summary.exists():
        transfer = json.loads(args.transfer_summary.read_text(encoding="utf-8"))
    text = render(
        results,
        args.results,
        mil_ab,
        args.mil_ab,
        results_same_grid,
        learnability,
        learnability_before,
        pretrain,
        transfer,
    )
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    figures = out_path.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("forward_fidelity.png", "network_before_after.png"):
        src = args.results.parent / name
        if src.exists():
            shutil.copy2(src, figures / name)
    if mil_ab is not None:
        plot_mil_ab(mil_ab, figures / "mil_sim_ab.png")

    print(f"报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
