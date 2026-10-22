## 实验系统实测指导（准备就绪，issue #165）

仿真侧已全部收口，本条给出**实验系统实测**的操作手册与已量到的结论。
完整手册在仓库里：[`docs/slm/gsnet_hw_session.md`](https://github.com/ywzhang909/AO-shaping/blob/feat/165-gsnet-2f-calib-sim/docs/slm/gsnet_hw_session.md)（分支 `feat/165-gsnet-2f-calib-sim`）。

### 一、仿真里量到的三条结论（先看这个，它决定了实测怎么做）

**1. 默认的迁移学习配置会把网络弄坏。** 6 种光路相差 × 6 种冻结策略 × 24 帧探针：

| 变体 | 开关 | 平均 Δ（迁移后 − 迁移前） | 变好情形 | 最差单情形 |
|---|---|---|---|---|
| A | 默认（相位项 + 整形项，理想前向） | **−0.0225** | 1/6 | −0.0663（**掉到平场以下**） |
| B | 只留整形项（`--w-phase 0`） | −0.0502 | 0/6 | −0.0811 |
| C | 算子换成拟合孪生 | −0.0169 | 2/6 | −0.0837 |
| D | 拟合孪生 + 只留整形项 | −0.0260 | 2/6 | −0.0705 |
| **E** | **拟合孪生 + 只留整形项 + 验证集门** | **−0.0002** | **3/6** | **−0.0325** |

两条机制，都不是玄学：

* **相位项是往下的拽。** 它的监督目标是「命令过的探针相位」＝经典 GS 解，而
  **网络 6/6 情形都已优于它**（panel-coma 0.3216 对 0.3113）⇒ 把网络往那儿拉就是拉低它。
* **整形项在默认配置下追的是做不到的目标。** 整形项拿**理想**前向去匹配**带像差、
  带耦合**的实测帧 ⇒ 目标物理上不可达，梯度近似噪声（所以 B 单独去掉相位项**更糟**）。

**2. 可冻结的参数**（变体 E 跨 6 情形平均）：

| 策略 | 可训练参数 | 平均 composite | 平均 Δ | 变好 |
|---|---|---|---|---|
| `full` | 414,145 | 0.3096 | **−0.0002** | 3/6 |
| `frozen` | 0 | 0.3097 | ±0.0000 | 0/6 |
| **`freeze-decoder`** | 336,097 | **0.3288** | **+0.0191** | 4/6 |
| `freeze-outconv` | 414,120 | 0.3251 | +0.0154 | 3/6 |
| `freeze-films` | 406,153 | 0.3180 | +0.0083 | 3/6 |
| **`freeze-encoder`** | **86,065（模型的 21%）** | 0.3137 | +0.0040 | 4/6 |

* 要**质量** → `--freeze freeze-decoder`；要**少学参数** → `--freeze freeze-encoder`。
* **不要用全参数微调**：几帧探针上它是净负的。

**3. 仿真环境已向 SLM 特性靠拢。** 孪生新增了**灰度→振幅耦合**
（`I(g) = 1 + c·sin(2π·grey/levels + ψ)`，`η = sqrt(max(I,0))` 乘在瞳面场振幅上）：
LCOS 的实际振幅依赖驱动灰度、周期 = 2π 灰度（1064 nm 实测 2π = **993** 灰度）。
`c = 0` 时孪生**逐位不变**；torch 镜像与孪生实测相关系数 **0.999999**。
同时 2π 灰度数从 1023（10 位满量程）改成设备真实的 **993**。

> 仓库自己早就记下这条：`report/loss_defects/inverse_design_report/inverse_design_report.md:246`
> 明确写「面板串扰/有限填充因子、LCOS 灰度-振幅耦合、像差本身的大偏离」三条
> **已实测但仿真没建模**。本轮实现了其中对整形影响最大的一条；
> **串扰/填充因子不做**（瞳面采样 56.25 µm ≫ 8 µm 像元，结构上不可分辨，
> 远场带边效应 <2%）。

### 二、门 1 的阈值已经改好 —— P1 条款此前**从未真正跑过**

issue 原文的「信任域 L2 ≤ 0.5 rad」卡的是 **Step A 的 Zernike 系数步**，而代码里
门 1 判的量是**瞳面相位的 rms（弧度）**。两者不在同一个空间。

* 闭环自己产生的最优相位 **rms = 1.678 rad**（实测）⇒ 0.5 rad 把**所有**真实整形
  相位都挡下了。
* 后果：已记录的 A/B 里 `gsnet_full` 臂正是被 `trust_region_rejected` 挡下的
  ⇒ **P1 验收条款（「GSNet 初值运行不得劣于物理初值运行」）从来没被执行过**。

**已改**：给网络初值单开一个阈值 `--init-from-gsnet-trust-region-c-l2`（默认
**2.5 rad**，由实测尺度定：闭环自己的最优相位 1.678 rad、网络整形相位 1.8–2.1 rad ⇒
预留余量），Step A 的系数步阈值不动。

改完第一次真跑（仿真 A/B）：

| 臂 | 判决 | 最终分 | 相对对照臂 |
|---|---|---|---|
| physical | `disabled` | 0.6765 | 0 |
| gsnet_full | **`holdout_correlation_rejected`** | 0.6768 | +0.0003 |
| gsnet_scaled | `holdout_correlation_rejected` | 0.6768 | +0.0003 |

⇒ **门 3 拒绝、闭环安全回退、最终分与对照臂持平**，正是「劣化即回退」的期望行为。

⚠️ 顺带说清：**门 1 只是预筛，不是质量门**。均匀随机相位的 rms 是
√(π²/3) = **1.81 rad**，与真实整形相位（1.68 rad）几乎一样 ⇒ **幅度本身分不出
好相位和随机相位**；真正表态的是门 2（实测对比择优）与门 3（验证集 Pearson）。

### 三、实测会话的六个阶段

| 阶段 | 做什么 | 工具 | 判据 |
|---|---|---|---|
| 0 | 表征实验系统 | `docs/slm/pre_run_characterization.md` 的三个探针 | 漂移/噪声类型/稠密相位可分辨性已知 |
| 1 | **测灰度→振幅耦合** | `scripts/slm/validate_flat_phase_gray.py --cam-type daheng --scan --csv …` → `scripts/slm/fit_slm_amplitude_coupling.py` | 拟合周期应 ≈ 993；`c` 与 `ψ` |
| 2 | 反演这台设备的孪生 | `scripts/slm/fit_bench_twin.py --from-dir … --coupling-config …` | 拟合损失与拟合前后残差 |
| 3 | 用拟合前向导出相位 | `scripts/slm/export_gsnet_phase.py --twin-config … --forward calibrated` | — |
| 4 | 迁移学习 | `scripts/slm/gsnet_transfer.py --twin-config … --freeze freeze-decoder --gate` | `phase_loss` < π²/3 ≈ 3.290；`gate` 判决 |
| 5 | 闭环 A/B（**交错**、每臂 ≥3 次） | `main.py slm-model-in-loop [--init-from-gsnet …]` | 网络臂 ≥ 物理臂；劣化必须能自动回退 |

阶段 1 是**新增**的：耦合的 `c` 至今**没有被测过**（仓库里只有判据
`rel_spread > 0.15` ⇒ `c ≳ 0.075`，没有曲线）。不测就实验，网络会去利用一个
**假定**出来的耦合。

⚠️ **阶段 2 还有一个容易漏的坑：焦面标度口径。** 孪生的远场标度只通过焦距进入，
所以「用哪个焦距」就是「用哪种标度口径」。上游实测这台实验系统的焦面标度是
**2.614022**（相机像元/模型采样点）⇒ **等效焦距 311.32 mm**；按标称焦距
122.41 mm 算出来的解析标度只有 1.027778，**差 2.54 倍**。拟合脚本会**自动**从
`<debug 目录>/preflight/geometry.json` 读实测标度；读不到就退回标称焦距并
**明确警告** —— 那种孪生与实验系统标度对不上，拟合系数不可信。

⚠️ 阶段 1 用的是**稠密**扫描（`--gray-step 25`）。仓库里 `slm-diagnose` 的调制
检查只采 12 个点（100 灰度步长），**会把周期里的峰谷漏掉** ⇒ 它的
`rel_spread` 只是**下界**，不足以定量拟合 `c`。

### 四、已就绪的代码（分支 `feat/165-gsnet-2f-calib-sim`，已合并上游最新）

上游在本轮期间重构了 `slm_gs_refine.py`（chunk-resume）并修正了**焦面标度口径**；
冲突的解法是把 `init_from_gsnet` 特性**移植到上游的新结构上**（相对上游的 diff
只有该特性，上游的重构完整保留）。

* 仿真：`drivers/sim/bench_twin.py`（耦合，2π 灰度 993）+ `ml/gsnet/forward_ops.py`（torch 镜像）
* 冻结：`ml/gsnet/freezing.py`（四组：encoder / decoder / films / outconv；注意
  `model.train()` 会把冻结组的 BatchNorm running stats 放回训练态，所以有 `prepare_for_training`）
* 迁移：`scripts/slm/gsnet_transfer.py --freeze / --twin-config / --gate`
* 实测链路：`scripts/slm/fit_slm_amplitude_coupling.py`（灰度响应 → `c`, `ψ`）、
  `scripts/slm/fit_bench_twin.py`（探针对 → 拟合孪生，**`fit_twin` 此前没有
  任何脚本调用过**）、`scripts/slm/validate_flat_phase_gray.py`（新增
  `--cam-type daheng` / `--metric bucket` / `--csv`）、`scripts/slm/export_gsnet_phase.py`
* 报告：`report/gsnet_2f_calibration/report.md`（P0 判据，改进仿真下**余量变大**）、
  `report/gsnet_aberration_transfer/report.md`（迁移 × 冻结 × 像差）

### 五、验证状态

容器内 **9 failed / 5609 passed / 91 skipped**；这 9 个失败在干净的
`origin/banckend` 上复跑**同样红**（2 个 oopao + 6 个 `model_in_loop_hw_ab`
+ 1 个 `_fmt` 共享助手）⇒ **零新增失败**。

### 六、还需要在实验系统上回答的

1. **耦合的实测 `c` 与 `ψ`**（阶段 1）—— 决定孪生那条物理是不是真的。
2. **拟合前向在真实帧上的保真度**（阶段 2 之后）—— 仿真里校准前向 corr 0.9752、
   Zernike 基线 0.7318、理想 FFT 0.6302；实验系统上这组数还**没有**。
3. **闭环 A/B 的真实判决**（阶段 5）—— 这是 P1 条款的第一次真实执行。

> ⚠️ 报告里**不要**把闭环的最终分数说成网络的贡献。这一轮要回答的是
> 「**在真实光路上，一个初值能不能帮闭环跳出局部最优**」，不是「网络相位本身好不好」。
