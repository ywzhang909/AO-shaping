# GSNet 初值闭环（`slm-model-in-loop --init-from-gsnet`）实验系统手册

> **状态：2026-10-21 仿真验收已跑完，P0 判据通过，实验系统尚未运行。**
> 仿真结论见 [`report/gsnet_2f_calibration/report.md`](../../report/gsnet_2f_calibration/report.md)。
> **实验前请先读 §3**——仿真查出一条按 issue 原文跑会让这个开关永远不生效的门。

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

`--forward calibrated` 会改用本实验系统 2f 物理前向（见
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
* 曝光按会话现定（`--exposure-time-ms auto`），**不要写死**：本实验系统 8 分钟
  内功率可漂 21 %。
* 统计走 `ml/zernike/eval_stats.paired_comparison`（精确 sign-flip p 值 +
  `d_z` + 结构性的 `min_attainable_pvalue`）。
* **单轮 best 分数不得作为可复现增益声明**——本实验系统分数分布呈双峰
  （见 `report/shape_trials/gs_refine_live_20261012.md`）。

产物：`data/debug/gsnet_init_hw_ab/<ts>/{plan.json,campaign.json,runs.csv}`。

## 3. 实验前必须先决定的两件事

仿真把这两条查出来了，直接照 issue 原文实验会白跑。

### 3.1 门 1 的阈值按 issue 原文是死的（**已改**）

issue #165 写的是「信任域 L2 ≤ 0.5 rad」。但仓库里的 `trust_region_c_l2`
限的是 **Step A 的像差系数变化量**，不是**绝对瞳面相位**。

代码里门 1 判的量是**瞳面相位的逐像素归一化 L2**（= 相位的 rms，弧度）：

```python
gsnet_phase_l2 = float(np.linalg.norm(gsnet_phase) / math.sqrt(gsnet_phase.size))
```

而阈值 0.5 rad 是给**系数空间**的步长定的。两个量不在同一个空间里，
放在一起比没有意义。仿真实测：

* 闭环自己产生的最优相位 **rms = 1.678 rad**
  （`data/debug/gsnet_init_sim_ab/twin/sim_ab.json` 的 `phase_rms_rad`）；
* GSNet 给出的整形相位同量级（本仿真的 GS 解 rms 1.8–2.1 rad）；
* 用 0.5 rad 去卡绝对相位 ⇒ **这个开关永远不会生效**。

> 后果不只是「阈值太严」：**已记录的 A/B 里 `gsnet_full` 臂正是被门 1 挡下的**
> （`init_from_gsnet_status = "trust_region_rejected"`），
> 也就是说 issue 写的 P1 验收条款（「GSNet 初值运行不得劣于物理初值运行」）
> **从来没有真正跑过** —— 门 2、门 3 根本没机会表态。

**已改**：给网络初值**单开一个阈值**，与 Step A 的系数步阈值解耦：

| 参数 | 卡的是什么 | 默认 | 依据 |
|---|---|---|---|
| `trust_region_c_l2` | Step A 每轮的**系数步** | 0.5 | 不动，语义不变 |
| `init_from_gsnet_trust_region_c_l2` | GSNet 瞳面相位的 **rms（弧度）** | **2.5** | 实测尺度：闭环自己的最优相位 1.678 rad、GSNet 整形相位 1.8–2.1 rad ⇒ 2.5 验证集余量，同时仍挡掉明显越界的相位；`0.0` = 关闭这道门 |

⚠️ **门 1 只是预筛，不是质量门。** 均匀随机相位的 rms 是
√(π²/3) = **1.81 rad**，与真实整形相位（1.68 rad）几乎一样 ——
**幅度本身分不出「好相位」和「随机相位」**。真正表态的是门 2（实测对比择优：
必须优于平场）与门 3（验证集 Pearson）。报告里不要把门 1 写成质量判据。

### 3.2 架构阻塞已经拆掉，P0 判据已通过

上一版手册写的是「网络在仿真里没跑赢物理基线，原因是架构」。**这一条已经
处理完了**：`FourierGSNet` 现在允许 source 在瞳面网格、target 在远场网格
（见 `src/ml/gsnet/model.py` 与 `src/ml/gsnet/forward_ops.py`），网络内部因此
能看到**被解析开**的远场。

**仿真环境也已更新**为仓库规范的 2f 数字孪生，并加上了 **SLM 灰度→振幅耦合**
（见 `report/gsnet_2f_calibration/report.md` §1.1）：LCOS 的实际振幅依赖驱动
灰度，周期 = 2π 灰度（设备在 1064 nm 的 2π = **993** 灰度）。孪生与 torch
镜像在锁步（实测相关系数 0.999999）。

最新结果（`report/gsnet_2f_calibration/report.md`，改进仿真下重跑）：

**臂 A（前向保真度，同一组 24 帧）**

| 前向 | corr | EE | CV（框） | NRMSE |
|---|---|---|---|---|
| **校准 2f 前向（孪生链）** | **0.9752** | 0.5021 | 1.1455 | **0.0482** |
| 物理 Zernike 基线 | 0.7318 | 0.4561 | 1.1356 | 0.1322 |
| 理想 FFT 前向 | 0.6302 | 0.5135 | 1.1225 | 0.1573 |

**臂 B（网络臂，同一批样本）**

| 臂 | composite | EE | CV（框） | corr（全图） |
|---|---|---|---|---|
| **网络（校准前向）** | **0.3101** | 0.0841 | 0.8863 | 0.6632 |
| 经典 GS 解（同一物理下） | 0.2766 | 0.0643 | 1.0604 | 0.5659 |
| 平场 | 0.2161 | 0.0410 | 1.5558 | 0.3570 |

**P0 判据通过**（前向与网络两个判据都过）：
`forward[issue(3)]: corr 0.8226 vs 0.7074 | EE 0.5021 vs 0.4561 | NRMSE 0.0142 vs 0.0194 -> OK`。

要如实说清楚的三点：

* **「GS 解」不是严格上限**，只是**经典算法达到的水平**（网络直接优化整形
  目标，GS 优化幅度匹配，所以网络可以超过它）；
* **加了耦合之后判据余量变大了**：校准前向几乎没变（0.8226），而两个简化
  模型都变差（理想 FFT 0.7336→0.6309、Zernike 基线 0.7449→0.7074）——
  真值越贴近真实器件，能表达完整物理的前向与简化模型的差距越大；
* **迁移学习默认配置会伤害网络**（见
  [`report/gsnet_aberration_transfer/report.md`](../../report/gsnet_aberration_transfer/report.md)）：
  6 情形 × 6 冻结 × 24 探针，默认配置 6 个情形里 5 个变差。要用网络初值，
  必须同时做三件事：**算子换成拟合出来的那台孪生 + 只留整形项 + 验证集
  对比择优门**。

所以实验系统这一跑要回答的是：

> **在真实光路上，一个「初值」能不能帮闭环跳出局部最优？**

这是与「网络相位本身好不好」不同的问题。**报告里必须按这个口径写，
不能把闭环的最终分数说成网络的贡献。**

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
