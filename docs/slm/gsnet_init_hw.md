# GSNet 初值闭环（`slm-model-in-loop --init-from-gsnet`）实验系统手册

> **状态：2026-10-21 仿真验收已跑完，实验系统尚未运行。**
> 仿真结论见 [`report/gsnet_2f_calibration/report.md`](../../report/gsnet_2f_calibration/report.md)。
> **上机前请先读 §3 的两条修正**——仿真证明照 issue 原文跑，这个开关不会生效。

## 1. 这条路径做什么

`slm-model-in-loop` 的 Step B（逐像素相位优化）默认从**零相位**起步。
`--init-from-gsnet` 允许它改从**网络预测的瞳面相位**起步，网络只提供
**初值**，硬件目标函数、逐轮验收、探针门控一律不变。

三条硬门（任一不过即回退到零相位，判决写进首轮历史行的
`init_from_gsnet_status` 与 `ModelInLoopResult`）：

| 门 | 判据 | 依据 |
|---|---|---|
| 1 | 信任域 L2 ≤ `trust_region_c_l2` | 见 §3.1 —— **这一条按 issue 原文会挡掉所有真实相位** |
| 2 | **实测**复合分数优于平场 | 与 gs-refine 阶段 1 的 `score_gs > score_flat` 同一条规则；模型自己觉得好不算数 |
| 3 | 验证集探针 Pearson ≥ `min_holdout_correlation`（0.5） | 复用闭环自己的硬门控 |

## 2. 怎么跑

### 2.1 离线：导出网络相位

```bash
python scripts/slm/export_gsnet_phase.py \
    --checkpoint data/gsnet_train/run-<ts>/fourier_gsnet_best.pt \
    --source  <source.npy> --target <target.npy> \
    --out data/debug/gsnet_init/<ts>/phase.npy
```

`source.npy` / `target.npy` 是 `(region, region)` 的二维数组：source 是瞳面
照明强度（0–1），target 是远场目标强度（0–1）。`region` 必须与闭环那次
运行的 `--region` 一致（相位数组的形状要对得上）。

`--forward calibrated` 会改用本台架 2f 物理前向（见
`ao_shaping.utils.wavefront.bench2f`），可用 `--aberration` 带上拟合出来的
实验系统像差。**注意：checkpoint 是用理想 FFT 训练的，换前向只是改推理时
的物理步，不改权重。**

### 2.2 实验系统：交错重放 A/B

```bash
# 先看计划（不打开设备）
python scripts/slm/gsnet_init_hw_ab.py --plan --gsnet-phase <phase.npy>

# 正式跑：每臂 3 次，交错
python scripts/slm/gsnet_init_hw_ab.py --gsnet-phase <phase.npy> --pairs 3
```

协议要点：

* **同一会话内交错**：`gsnet` 臂与 `physical` 臂交替，第 i 对的顺序按 i 的
  奇偶翻转，抵消「先跑的那一臂占了更好的光功率时段」。
* 每次 MIL 运行前都过**光功率门**（`scripts/slm/diagnose_light_level.py`）。
* 曝光按会话现定（`--exposure-time-ms auto`），**不要写死**：本台架 8 分钟
  内功率可漂 21 %。
* 统计走 `ml/zernike/eval_stats.paired_comparison`（精确 sign-flip p 值 +
  `d_z` + 结构性的 `min_attainable_pvalue`）。
* **单轮 best 分数不得作为可复现增益声明**——本台架分数分布呈双峰
  （见 `report/shape_trials/gs_refine_live_20261012.md`）。

产物：`data/debug/gsnet_init_hw_ab/<ts>/{plan.json,campaign.json,runs.csv}`。

## 3. 上机前必须先决定的两件事

仿真把这两条查出来了，直接照 issue 原文上机会白跑。

### 3.1 门 1 的阈值按 issue 原文是死的

issue #165 写的是「信任域 L2 ≤ 0.5 rad」。但仓库里的 `trust_region_c_l2`
限的是 **Step A 的像差系数变化量**，不是**绝对瞳面相位**。

仿真实测（`report/gsnet_2f_calibration/report.md` §5.2）：

* 闭环自己产生的最优相位 **rms = 1.71 rad**；
* GSNet 给出的整形相位同量级（本仿真的 GS 解 rms 1.8–2.1 rad）；
* 用 0.5 rad 去卡绝对相位 ⇒ **这个开关永远不会生效**。

可选的改法（选一种，并写进 issue）：

1. 门 1 改成卡**相位与零/平场之差**的 L2；
2. 或卡**相位相对闭环自身解的偏离**；
3. 或给网络初值单开一个阈值（例如 `init_from_gsnet_trust_region_c_l2`），
   与 Step A 的系数步阈值解耦。

门 2 与门 3 工作正常，**不需要改**。

### 3.2 网络在仿真里没跑赢物理基线，原因在架构

仿真的结论是：前向校准在**前向层面**有效（远场预测相关性 0.985 对 0.699），
但**网络层面**没通过 P0 判据。阻塞点不是校准，是
`FourierGSNet` 的网格契约：source 与 target 必须同网格，于是网络内部的远场
只能是**同尺寸 FFT**（衍射极限光斑亚像素、图样混叠），而实验系统的远场需要
8 倍以上过采样才被解析。

所以实验系统这一跑要回答的是：

> **网络只当「初值」，闭环自己会用过采样正确的物理模型继续优化——**
> **在真实光路上，一个被限制在低分辨率的初值，能不能帮闭环跳出局部最优？**

这是与「网络相位本身好不好」不同的问题，也是这个开关唯一有希望成立的用法。
**报告里必须按这个口径写，不能把网络的最终分数说成网络的贡献。**

## 4. 判据

* 每一臂的 `init_from_gsnet_status` 都要出现在报告里；
  `adopted` 的比例本身就是结果（门 1 / 门 2 / 门 3 各挡掉多少）。
* 配对均值差 + 精确 sign-flip p 值，**不是**单轮 best。
* `n_pairs` 小的时候报效应量即可（`min_attainable_pvalue(n) = 2 / 2**n`，
  4 对到不了 p < 0.05）。

## 5. 相关文件

| 文件 | 作用 |
|---|---|
| `scripts/slm/export_gsnet_phase.py` | 离线导出网络相位（`.npy`） |
| `scripts/slm/gsnet_init_hw_ab.py` | 实验系统交错重放 A/B 编排 |
| `scripts/slm/gsnet_init_sim_ab.py` | 同样的 A/B 的**仿真**版本（先跑它） |
| `scripts/slm/calibrate_gsnet_2f.py` | P0 前向保真度 + 网络换前向前后 |
| `src/ao_shaping/optimizer/wfless/slm_model_in_loop.py` | `--init-from-gsnet` 的实现与三重门控 |
| `report/gsnet_2f_calibration/report.md` | 仿真验收报告（先读它） |
