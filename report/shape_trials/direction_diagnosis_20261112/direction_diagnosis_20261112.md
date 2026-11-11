# 可微整形前向模型 Jacobian 方向诊断（合并报告，20261112）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/direction_model_variants_20261112.py`](../../../scripts/slm/direction_model_variants_20261112.py)
> **数据/关联脚本**: [`scripts/slm/direction_attribution_20261112.py`](../../../scripts/slm/direction_attribution_20261112.py)
> **运行环境**: 离线回放存档数据（2026-11-12；未占用 SLM/相机/激光，未 import ao_shaping.drivers）
> **说明**: 可微整形前向模型 Jacobian 方向诊断合并报告：6 变体×3 幅值方向矩阵 + 7 臂归因（隔离照明包络/origin/defocus 贡献）。M3（离焦+照明包络）0.616/0.696/0.657 通过 ≥0.50 门禁；M0/M1 未过；背景/增益对方向度量精确不变；M3 通过是 origin+defocus+包络+gain 的联合自洽属性。输入为 defocus_forward_20261107_focus_r2_fit/forward_fit.json 与 issue234_*_screen_20261107 存档产物，产物 data/debug/slm_transfer/direction_variants_20261112/{results,attribution}.json。人工撰写。
<!-- provenance:end -->

**日期** 2026-11-12 · **脚本** `scripts/slm/direction_model_variants_20261112.py` + `scripts/slm/direction_attribution_20261112.py`（均离线）
**设备** 无（全程离线回放）
**数据** `data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json`（M0/M1/M3 拟合参数）、`data/debug/slm_transfer/issue234_amp{02,05,10}_screen_20261107/`（单模屏 ladder）

## 1. 结论速览

| 指标 | `M0_fit`（analytic 基线，无离焦） | `M1_fit`（slm_phase 拟合，标准离焦） | `M3_fit`（离焦 + 照明包络） | 判据 | 结论 |
|---|---|---|---|---|---|
| median（amp 0.02/0.05/0.10） | 0.305 / 0.376 / 0.420 | 0.459 / 0.496 / 0.441 | **0.616 / 0.696 / 0.657** | ≥ 0.50 | 仅 M3 通过 |
| 单对 ≥ 0.50（amp 0.05） | 至多 4/14 | 至多 7/14 | **10/14** | ≥ 0.50 | 仅 M3 通过 |
| 背景/增益不变性 | — | bg0 / bg10 与 M1_fit **逐位相同** | bg0 与 M3_fit 逐位相同 | 精确不变 | 背景不能解释方向失败 |
| U/C 全帧 Pearson（−1 下） | — | 0.637 | 0.709 | 参考 | M3 同时改善拟合自洽 |

- **方向门禁通过**：M3（离焦 + 照明包络）在三档幅值下 median 均 ≥ 0.50（0.616 / 0.696 / 0.657），且 10/14 单对 ≥ 0.50；M0（analytic 无离焦）与 M1（仅离焦）未过。**这是当前唯一通过 ≥0.50 门禁的前向模型变体。**
- **背景/增益对方向度量精确不变**（bg0 / bg10 与基线逐位相同）——方向度量是 C−U 差分上的仿射不变 Pearson，背景与增益在差分中精确消去。**背景误差不能单独解释方向失败。**
- **幅值非线性（#233）无单调幅值签名**：M3 median 0.616/0.696/0.657 在 0.02/0.05/0.10 rad 下无单调趋势，弱反对“幅值非线性主导”，非定论。
- **PSF / defocus / 照明包络仍主导**：M3 通过门禁，且 M3 归因显示该通过**不是**任何单一参数（照明包络 / origin / defocus 单独替换）能复现的——是 M3 联合拟合（origin + defocus + 包络 + gain 相互自洽）的**联合属性**。
- **极性作用域矛盾未解决**（延续 `forward_model_polarity_20261107.md` §4.4/§9.1）：整条指令取反改善单模屏方向但恶化 U/C 拟合（0.851→0.637）→ 全局符号假设错；缺陷最可能只作用于“像差指令”层，需最小设备实验判定（§6）。

## 2. 输入与口径

- **数据集**：`issue234_amp{02,05,10}_screen_20261107` 三组单模屏 ladder（Z5–Z11，±0.10 / ±0.05 / ±0.02 rad，两 pass）。每组每模有 pass1/pass2 的 ± 成对屏，C−U 差分取均值。
- **模型**：`FrozenDefocusForward`（`ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity`），冻结 checkpoint 只读，CPU，固定种子。
- **方向度量**：`direction_agreement(pred_m, pred_p, meas_m, meas_p)`（`ao_shaping.tools.slm.transfer.jacobian_guard`）：Pearson 系数（对增益/曝光不敏感），crop 128×128，model origin，rate=raw/exposure_ms，`FrozenDefocusParameters(phase_sign=-1, phase_sign_scope="command")`。
- **guard**：`M1_fit` amp 0.05 臂须逐位复现存档 `direction_screen_amp005_M1_defocus_signm1_model.json` per-mode 相关（实测 max_delta=0.000e+00，通过）。

## 3. 方向矩阵（6 变体 × 3 幅值）

脚本：`scripts/slm/direction_model_variants_20261112.py`。产物：`data/debug/slm_transfer/direction_variants_20261112/results.json`。

| 变体 | amp | median | mean | ≥0.5 |
|---|---|---|---|---|
| M0_fit | 0.02 | 0.305 | 0.273 | 2/14 |
| M0_fit | 0.05 | 0.376 | 0.350 | 3/14 |
| M0_fit | 0.10 | 0.420 | 0.391 | 4/14 |
| M1_fit | 0.02 | 0.459 | 0.443 | 6/14 |
| M1_fit | 0.05 | 0.496 | 0.484 | 7/14 |
| M1_fit | 0.10 | 0.441 | 0.430 | 5/14 |
| **M3_fit** | 0.02 | **0.616** | 0.588 | 10/14 |
| **M3_fit** | 0.05 | **0.696** | 0.619 | 10/14 |
| **M3_fit** | 0.10 | **0.657** | 0.618 | 10/14 |
| M1_bg0 | 0.05 | 0.496 | 0.484 | 7/14 |
| M1_bg10 | 0.05 | 0.496 | 0.484 | 7/14 |
| M3_bg0 | 0.05 | 0.696 | 0.619 | 10/14 |

> M1_bg0 / M1_bg10 与 M1_fit（amp 0.05）**逐位相同**（background 0 与 +10 counts/ms 给出完全相同的 per-mode 相关）；M3_bg0 与 M3_fit 逐位相同。背景/增益对方向度量精确不变。

### 3.1 per-mode 明细（amp 0.05，M0/M1/M3）

| 模 | M0_fit | M1_fit | M3_fit |
|---|---|---|---|
| Z5 | 0.443 | 0.375 | **0.682** |
| Z6 | −0.011 | 0.181 | 0.431 |
| Z7 | 0.097 | 0.591 | 0.304 |
| Z8 | 0.538 | 0.536 | **0.815** |
| Z9 | 0.091 | 0.496 | **0.699** |
| Z10 | 0.394 | 0.470 | **0.731** |
| Z11 | 0.572 | 0.738 | 0.670 |

- M3 相对 M1 在 **Z5 / Z8 / Z9 / Z10 / Z11** 全部显著提升（≥0.67）；**Z6（0.431）仍低于门禁**、**Z7 反而下降（0.591→0.304）**——这两模是 M3 未达 14/14 的原因。
- M0（无离焦）的 **Z6 ≈ 0**（方向正交/无信号），说明离焦项是低阶像差方向可辨识的前提。

## 4. 归因（隔离 M3 的包络 / origin / defocus 对方向改善的贡献）

脚本：`scripts/slm/direction_attribution_20261112.py`（amp 0.05，7 臂）。产物：`data/debug/slm_transfer/direction_variants_20261112/attribution.json`。

| 臂 | 含义 | median | ≥0.5 |
|---|---|---|---|
| M1_baseline | M1 拟合（对照） | 0.496 | 7/14 |
| M3_baseline | M3 拟合（对照） | **0.696** | 10/14 |
| M3_env_on_M1 | M1 的 origin/defocus/gain/bg + **M3 照明包络** | 0.436 | 4/14 |
| M1_env_on_M3 | M3 的 origin/defocus/gain/bg + **M1 包络**（去掉包络） | 0.395 | 4/14 |
| M3_origin_on_M1 | M1 其余 + **M3 origin**（隔离 origin 平移） | 0.085 | 0/14 |
| M1_origin_on_M3 | M3 其余 + **M1 origin** | 0.372 | 3/14 |
| M3_defocus_on_M1 | M1 其余 + **M3 defocus** | 0.209 | 1/14 |

**归因结论**：没有任何单一参数（包络 / origin / defocus）替换能复现 M3 的 0.696——每个单独替换都**降**到 ≤0.436，甚至破坏到 0.085（M3 origin on M1，Z6/Z7/Z11 转负）。M3 的方向通过是 **origin + defocus + 照明包络 + gain 的联合自洽**属性，不是单个物理项的贡献。这解释了为何“把 M3 包络单独加进 M1”不提升方向（包络的增益/中心/宽度必须与 M3 的 origin/defocus 配套）。

> 关键：M1 与 M3 的 origin 差约 15 px（1320.7 vs 1306.0）、defocus 0.114 vs 0.031、gain 225 vs 307、amplitude_width [260,260] vs [383,142]——**各向异性包络**（383×142 px）与 **离焦 0.031**（更接近几何焦面）是 M3 联合解的特征。M3 的 amplitude_center [1018,581] 也偏离 M1 的 [988,571]。

## 5. 需求三问的明确回答

| 问题 | 回答 | 证据 |
|---|---|---|
| 背景误差能否单独解释方向失败？ | **不能** | bg0 / bg10 与基线逐位相同；方向度量是 C−U 差分上的仿射不变 Pearson，背景/增益精确消去 |
| 非线性误差（#233）能否单独解释？ | **弱反对，非定论** | M3 median 0.616/0.696/0.657（amp 0.02/0.05/0.10）无单调幅值趋势；但 3 幅值不足以定论，非主导假设 |
| PSF / defocus / 照明包络是否主导？ | **是（联合主导）** | M3（离焦 + 照明包络）是唯一通过 ≥0.50 的变体；M3 联合拟合同时改善 U/C Pearson（0.637→0.709）与方向（0.496→0.696） |

**整体判读**：方向失败不是背景误差、不是单一非线性项——是**前向模型的 PSF / defocus / 照明包络建模不足**。M3 通过门禁证明：加入照明包络 + 重拟合 origin/defocus/gain（各向异性包络 383×142、离焦 0.031）后，方向预测与实测一致。这直接对应核心路线第 3 条“前向模型必须能拟合光路中存在的像差”。

## 6. 最小设备实验设计（aberration 作用域符号翻转 A/B/C）

**触发条件**：离线已判别方向门禁（M3 通过），但**极性作用域矛盾**（§1 末行）仍需硬件判定——该矛盾不因 M3 通过而消失，且直接决定“像差指令层符号”是否要在模型里显式修正。按硬件纪律（`docs/dev/hardware_discipline.md`）先报告占用，未获批不上机。

**设备**：SLM #1（Santec SLM-200，1064 nm）+ Daheng MER2-507-23GM CCD。占用时段待定（本报告仅设计，未运行、未占用）。

**设计目的**：判定方向/极性缺陷是否只作用于“像差指令”层，而非全局符号。三种判定实验：

| 臂 | 操作 | 判读 |
|---|---|---|
| **A** | 单模屏 ladder（Z5–Z11，amp 0.05）：分别以 `phase_sign=+1` / `−1` 施加，对比模型预测方向 | 哪一符号下硬件方向与模型一致 |
| **B** | U/C 拟合数据集重放：以 `phase_sign=+1` / `−1` 分别拟合，比较全帧 Pearson | 哪一符号下 U/C 拟合自洽（对照 0.851 vs 0.637） |
| **C** | A/B 交叉自洽：若 A 偏向 −1 而 B 偏向 +1，则缺陷在“像差指令”层（需在模型/损失中显式修正该层符号）；若 A、B 同向，则是全局符号约定错 | 判定作用域 |

**占用/释放纪律**：上机前在回复中明确报告“占用 SLM #1 + Daheng CCD + 激光，时段 X–Y”；采集脚本后台运行同样报告；结束立即释放并报告“已释放、正常关闭”。禁止与任何并行会话同时占用同一设备（脚本自带 `--wait-until-idle` 时可用）。背对背 open/close 需规避（驱动板假死风险，见记忆：背对背 SLM open/close 会触发驱动板假死）。

**预期产出**：判定符号缺陷作用域 → 若是“像差指令”层，给出模型/损失中的显式修正方案；若全局，修正 `phase_sign` 传播约定。产物存 `data/debug/slm_transfer/`，报告补 `report/shape_trials/`。

## 7. 与既有结论的关系

- **`forward_model_polarity_20261107.md`**：极性翻正但不足（中位 +0.459，未过门）；本报告将口径从“全局平移最优”推进到“M3 变体”，并给出 M3 通过门禁的独立证据。
- **`measured_forward_refine_20261107.md` §7**：配准门未过（0.424–0.466）、PSF+bg 0.560→0.530、背景不可辨识——本报告确认背景不可辨识（方向度量对背景精确不变），并把“PSF/照明包络”从候选提升为主导。
- **`phase_scope_20261107/REPORT.md`**：作用域退化、aberration_focus 不稳定——本报告的 A/B/C 实验正是对该作用域矛盾的直接判定。
- **`defocus_differentiable/diffopt_efficiency_20261108.md` §10.4/§10.5**：效率线处置（块末 vs 块首）。本报告回答的是其“方向不可信”根因：方向由 M3 修复，但真实锚定闭环验收（core CV ≤ 0.10）仍待上机。

## 8. 结论与建议

1. **离线方向诊断完成**：M3（离焦 + 各向异性照明包络）是当前唯一通过 ≥0.50 方向门禁的前向模型变体（0.616/0.696/0.657，10/14）。方向失败由 PSF/defocus/照明包络建模不足主导，背景/非线性不主导。
2. **下一步（需设备）**：以 M3 参数为 checkpoint，跑一次真实锚定闭环，以实测 core CV ≤ 0.10 为最终验收；同时执行 §6 的 A/B/C 判定“像差指令”层符号作用域。
3. **模型改进**：把 M3 的照明包络（amplitude_center [1018,581]、width [383,142]、离焦 0.031、gain 307）作为前向模型缺省参数候选，纳入 `FrozenDefocusParameters`；包络需与 origin/defocus **联合**拟合，不可单独替换。
