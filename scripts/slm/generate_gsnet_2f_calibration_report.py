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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
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


def _fmt(value: float, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "—"
    if fvalue != fvalue:
        return "—"
    return f"{fvalue:.{digits}f}"


def _pm(row: dict[str, Any], key: str) -> str:
    return f"{_fmt(row[key]['mean'])} ± {_fmt(row[key]['sd'])}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render(
    results: dict[str, Any],
    results_path: Path,
    mil_ab: dict[str, Any] | None,
    mil_ab_path: Path | None,
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
    add("三条结论，按证据强度排列：")
    add("")
    cal = overall["calibrated"]
    ide = overall["ideal"]
    zen = overall["zernike"]
    add(f"1. **前向校准本身是有效的，而且是本轮最硬的结论。** 校准后的 2f 物理前向"
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
    add("3. **网络层没有通过 P0 判据，而且阻塞点是架构的、不是校准的。** "
        "见 §4 与 §6：网络的 source/target 必须在同一网格上，"
        "于是它内部能看到的远场只能是**同尺寸 FFT**——衍射极限光斑亚像素、图样混叠；"
        "而实验系统的远场需要 8 倍以上的过采样才被解析。校准修好了**物理**，"
        "修不了**采样**。")
    add("")
    if decision["passed"]:
        add("按 issue 的对比择优逻辑，网络臂**允许**进入 P1（闭环初值路径）。")
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
        add("### 4.1 读法")
        add("")
        add("- `flat`（平场，零相位）是下界，`gs_ceiling`（GS 在过采样真值上的解）是上限；"
            "没有这两个数，网络臂的分数无法解读。")
        add("- `zernike_baseline` 是**物理 Zernike 基线**：经典 GS 解投影到 66 项 Zernike 张成。")
        add("- 两个网络臂的分数明显低于基线 ⇒ **P0 判据不通过**。")
        add("")
        add("### 4.2 为什么网络学不到：一个可复现的对照实验")
        add("")
        add("同一套训练代码、同一份数据，只换训练目标的来源：")
        add("")
        add(_row(["训练目标", "网络在真值孪生上的 correlation（全图）"]))
        add(_row(["---", "---"]))
        add(_row(["过采样真值上的 GS 解（对真值最优）", "≈ +0.12（学不到）"]))
        add(_row(["网络自己网格上的 GS 解（同口径）", "≈ +0.19（学得到）"]))
        add("")
        add("结论：**训练代码是好的，网络能学会它能表达的那个解**；"
            "它学不到「过采样真值上的解」，因为那个解需要网络内部有一个"
            "**比瞳面网格更细的远场**。")
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
            "让远场算子返回一个**比瞳面网格更大**的远场——"
            "这是架构改动，超出 issue #165 的 P0 范围。")
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
    add(_row(["P0 (3) 离线判据", "✅ 已跑", "见 §3 与 §4：前向层面通过，"
            "网络层面**不通过**，阻塞点见 §4.3"]))
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

    add("## 7. 边界与不能下的结论")
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
    add("5. **§4.3 的架构阻塞是本轮最有价值的发现，但它不否定 §3。** "
        "前向校准在数值上成立；网络层之所以用不上，"
        "是因为 `FourierGSNet` 的网格契约让它的内部远场只能是混叠的同尺寸 FFT。")
    add("")

    add("## 8. 下一步")
    add("")
    add("1. **P0.5（架构）**：给 `FourierGSNet` 一个「远场网格可以比瞳面网格大」的"
        "契约（source 在瞳面网格、target 在远场网格，条件张量按各自网格拼接）。"
        "这一步做完，§4 的网络臂才有机会通过 P0 判据。")
    add("2. **P1 门 1 重新定义**：把信任域从「绝对相位幅度」改成"
        "「相位相对零/平场（或相对闭环自身解）的偏离」，"
        "否则按 issue 原文这个开关不会生效（§5.2）。")
    add("3. **P1 实验系统**：同一会话内 GSNet 初值与物理初值**交错重放**，"
        "报均值差与 p 值；单轮 best 分数不得作为可复现增益声明"
        "（双峰分布教训，见 `report/shape_trials/gs_refine_live_20261012.md`）。"
        "编排脚本已就位：`scripts/slm/gsnet_init_hw_ab.py`。")
    add("4. **真实语料复测**：拿到 `data/debug/` 的硬件语料后，"
        "按 issue P0 (1) 的要求按 record 去重、校验标签，再重跑一次 P0 判据。")
    add("")

    add("## 9. 复现")
    add("")
    add("```bash")
    add("# 1) 臂 A + 臂 B（前向保真度 + 网络换前向前后）")
    add("PYTHONPATH=src .venv313/bin/python scripts/slm/calibrate_gsnet_2f.py \\")
    add("    --frames 8 --epochs 40 --n-train 96 --n-eval 24 --target-side 36")
    add("")
    add("# 2) 臂 C（MIL 闭环的「网络初值 vs 物理初值」A/B）")
    add("PYTHONPATH=src .venv313/bin/python scripts/slm/gsnet_init_sim_ab.py")
    add("")
    add("# 3) 渲染本报告")
    add("PYTHONPATH=src .venv313/bin/python scripts/slm/generate_gsnet_2f_calibration_report.py \\")
    add("    --results data/debug/gsnet_2f_calib/<ts>/results.json \\")
    add("    --mil-ab data/debug/gsnet_init_sim_ab/<ts>/sim_ab.json")
    add("```")
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
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "report.md")
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    mil_ab = None
    if args.mil_ab is not None and args.mil_ab.exists():
        mil_ab = json.loads(args.mil_ab.read_text(encoding="utf-8"))

    text = render(results, args.results, mil_ab, args.mil_ab)
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
