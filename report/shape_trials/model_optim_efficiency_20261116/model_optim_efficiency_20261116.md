# 物理预测模型闭环整形：优化效果与效率（model_optim_efficiency 20261116）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/shape_transition_timing_20261106.py`](../../../scripts/slm/shape_transition_timing_20261106.py)
> **数据/关联脚本**: [`scripts/slm/absolute_fidelity_probe_20261116.py`](../../../scripts/slm/absolute_fidelity_probe_20261116.py)
> **运行环境**: 离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-16 两次实验锚定闭环后离线汇总；报告本体不占用设备）
> **说明**: 物理预测模型（FrozenDefocusForward / M3，grid 128 / FFT 512）驱动可微闭环的优化效果与效率汇总：模型侧 model_cv 3.89→0.66（随机起步规避焦斑盆地）、方向 cosine 0.696 过 ≥0.50 门禁；硬件侧实测核心 CV 停在 6.08/5.01 未达 ≤0.10，与 model_cv 的分叉约 7.5× **已判为采集伪影**（锚定轮 8-bit 量化地板 + 探针过曝 argmax 定心失效，§5.1），并非绝对强度表征误差；12-bit 重采 corr(model_cv_core, measured_cv_core_window)=0.95（§5.2），剩余差距为 checkpoint gain 量纲不可迁移（§5.3）。效率：M3 闭环 ~0.30 s/iter（settle+capture 0.18 / 计算 0.11），比 model640 ~0.92 s/iter 快约 3×；历史预算证 LCOS settle 占会话 99%、优化器 <0.05%。绝对强度保真度已于 2026-11-16 实验补采（探针 absolute_fidelity_probe_20261116.py，commit 807d456a；12-bit 满量程契约修复 96e285ee，§5）。输入为两次锚定运行 timing/device_log/iterations、direction_variants_20261112、wall_clock_budget_20261111、shape_transition_timing_20261107。人工撰写。
<!-- provenance:end -->

**日期** 2026-11-16 · **脚本** `scripts/slm/shape_transition_timing_20261106.py`（实测闭环）+ 离线汇总 · **设备** Santec SLM-200 #1（1920×1200）+ Daheng MER2-507-23GM（2592×1944，0.5 ms，5 帧平均）
**Checkpoint** `defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`（grid 128 / FFT 512 / phase_sign −1.0 / 离焦+照明包络）

> 本报告聚焦**物理预测模型（FrozenDefocusForward / M3）驱动的可微闭环**在实验系统上的**优化效果**与**效率**两条主线，汇总 2026-11-16 两次实验锚定闭环、2026-11-16 绝对强度保真度实验探针、2026-11-12 离线方向诊断、以及 2026-11-07 / 2026-11-11 历史计时数据。绝对强度保真度已实验补采（探针 `scripts/slm/absolute_fidelity_probe_20261116.py`，commit `807d456a`；满量程契约修复 commit `96e285ee`），见 §5。

## 1. 结论速览

- **优化效果（模型侧）**：M3 在离线方向诊断中通过 cosine ≥0.50 门禁（0.696 中位，10/14 达标），是 6 个变体中唯一稳定通过者；随机起步下模型核心 CV 从 3.89（零相位）降到 0.66，证明**起步盆地已规避、模型侧优化方向有效**。
- **优化效果（硬件侧）**：两次实验锚定闭环实测核心 CV 分别停在 6.08（零相位）/ 5.01（随机起步），均**未达 ≤0.10 验收**；模型核心 CV 从 3.80 降到 0.66 而实测几乎不动，方向对齐是必要条件而非充分条件（详见 `report/shape_trials/anchored_m3_20261116/anchored_m3_20261116.md`）。**归因修正**：早前"绝对强度表征误差主导、模型 vs 实测 CV 分叉 7.5×"的判断**已作废**——该 7.5× 是采集伪影（锚定轮 8-bit 量化地板把实测 CV 压在 ~5，§5.1），并非模型表征缺陷；重采 12-bit 后 `corr(model_cv_core, measured_cv_core_window)=0.95`，真正的剩余差距是 checkpoint gain 的**量纲不可迁移**（§5.3）。
- **效率**：M3（grid 128）闭环单步墙钟约 **0.30 s/iter**，其中硬件 settle+capture 约 0.18 s/iter、模型计算约 0.11 s/iter；相比 2026-11-07 的 model640（grid 640）锚定闭环 0.92 s/iter **快约 3×**，主要来自更小的控制网格与 FFT。两次运行均因停滞提前结束（60/120 步，max 300），故墙钟反映的是**未收敛早停**成本，非收敛成本。

## 2. 优化效果

### 2.1 模型侧收敛（model_cv 轨迹）

| 运行 | 起步 | model_cv（iter10 → 末） | measured_cv（全帧） | model/measured | accepted |
|---|---|---|---|---|---|
| A `anchored_m3_20261116` | 零相位 | 3.801 → 3.888（iter60） | 6.05–6.14（min 6.077） | ~0.64× | 否 |
| B `anchored_m3_rs15_20261116` | 随机 RMS1.5 | 0.544 → 0.664（iter120） | 5.00–5.18（min 5.006） | ~0.13× | 否 |

- 随机起步使 model_cv 从 3.89 降到 0.66（**模型认为接近均匀**），证实零相位焦斑盆地已被规避；但 measured_cv 仅从 6.08 降到 5.01（**硬件几乎无改善**）。
- 两次均 `stop_reason=stall_5_blocks_no_0.005_improve`：模型侧与实测侧同时停滞。**注意本表口径**：两次锚定均为 8-bit（实测 CV 地板 ~5，见 §5.1），`measured_cv` 已被量化地板截断，"硬件几乎无改善"是**采集伪影**而非硬件确无改善——真实改善被 8-bit 量化掩埋，须以 12-bit 重采（§5 探针）方可判读。

> **本表 model/measured 比值（0.64× / 0.13×）及 §1 的 7.5× 分叉均为 8-bit 地板伪影，已按 §5.1 作废。** 模型侧收敛轨迹（3.89→0.66）本身成立。

### 2.2 方向保真度（离线诊断，2026-11-12）

`data/debug/slm_transfer/direction_variants_20261112/{results,attribution}.json`，预测 vs 实测梯度方向 cosine，n=14，幅值 [0.02, 0.05, 0.10] rad，`guard.exact_match=true`：

| 变体 | 中位 cosine（amp 0.05） | n≥0.50 |
|---|---|---|
| M0_fit | 0.376 | 4/14 |
| M1_fit / M1_bg0 / M1_bg10 | 0.496 | 7/14 |
| **M3_fit / M3_bg0** | **0.696** | **10/14** |

- 交叉归因（中位）：`M3_origin_on_M1` 0.085（0/14≥0.5）、`M1_origin_on_M3` 0.372、`M3_defocus_on_M1` 0.209 → M3 通过是 origin+defocus+照明包络+gain 的**联合自洽**结果。
- 背景/增益对方向度量**精确不变**（bg0/bg10 与基线逐位相同）→ 背景误差不能单独解释方向失败。
- **判读边界**：cosine 度量梯度方向对齐，不度量绝对强度精度。M3 过门禁但实测 CV 未达标，正是"方向对、幅度错"的假阳性，须以绝对强度保真度补充门禁（§5）。

## 3. 效率

### 3.1 单次闭环墙钟分解（2026-11-16，M3 grid 128）

由 `device_log.json` 时间戳差分（display→capture ≈ settle+capture；capture→display ≈ 模型计算）：

| 指标 | A 零相位 | B 随机 RMS1.5 |
|---|---|---|
| 迭代数（早停/上限） | 60 / 300 | 120 / 300 |
| 形状墙钟 | 18.204 s | 35.485 s |
| s/iter | 0.3034 | 0.2957 |
| display 事件 | 12 | 24 |
| capture 事件 | 12（6 anchor + 6） | 24（12 anchor + 12） |
| settle+capture（display→capture） | 均 0.915 s，0.183 s/iter | 均 0.905 s，0.181 s/iter |
| 模型计算（capture→display） | 均 0.609 s，0.112 s/iter | 均 0.574 s，0.110 s/iter |

硬件成本常数（两次相同，`plan.json`）：`settle_s` 0.4、`frames` 5、`exposure_ms` 0.5（单次曝光 2.5 ms，可忽略）、`lr` 0.01、`phase_rms_cap` 0.06、`updates_per_block`/`check_every` 10/10。

> 注：`plan.json` 记录的 `control_grid 640 / fft_size 2048` 是脚本常量元数据；实际物理预测模型按 checkpoint 的 `grid_size 128 / fft_size 512` 构建（`FrozenDefocusForward(params, …)`），效率以实际网格为准。

### 3.2 跨运行/跨 checkpoint 效率对比

| 运行 | checkpoint（grid） | s/iter | 步数/墙钟 | 备注 |
|---|---|---|---|---|
| 20261116 A/B 锚定 | M3（128） | ~0.30 | 60/18.2 s、120/35.5 s | 本报告；早停未收敛 |
| 20261107 锚定 preanchored | model640（640） | 0.922 | 60/55.3 s | measured 0.671 平 |
| 20261107 锚定 raw | model640（640） | 0.917 | 60/55.0 s | measured 0.704 平 |
| 20261107 模型基线（随机 RMS1.5） | model640（640） | 0.499 | 300/149.6 s | model_cv 5.47→0.099，measured 1.247→0.529 |

- M3（grid 128）锚定闭环 ~0.30 s/iter，比 model640（grid 640）~0.92 s/iter **快约 3×**，主因是控制网格与 FFT 更小（128/512 vs 640/2048）。
- **不可直接横比的项**：checkpoint、网格、常数不同；20261116 两次均早停（60/120 步），20261107 模型基线跑满 300 步。跨运行墙钟仅作量级参考。

### 3.3 历史墙钟预算（2026-11-11，multishape1200，离线 RTX 5080 bench）

`report/shape_trials/multishape1200/wall_clock_budget_20261111.md`（repro1200_20261109 会话）：

| 项 | 值 | 占比 |
|---|---|---|
| 会话总时长 / 帧数 | 2808.2 s / 594 帧 | — |
| 优化器迭代（shape 阶段） | 71 次，GPU 12.583→7.880 ms/iter | 0.032%→0.020% |
| LCOS settle 等待（帧间 gap） | 1212.0 s | 99.0%（acquire+gap 1224.6 s） |
| 模型/seed 构建（stage_other） | ~677.0 s | 35.0% |
| transport_cv | ~127–134 s/shape ×3 | — |
| 曝光调优 / 相机读出 | 71.3 s / 12.6–13.5 s（~23 ms/帧） | 2.5% / 0.5–1.0% |

- **核心效率事实**：优化器计算占会话 < 0.05%，**LCOS settle 等待占 ~99%**。GPU 提速（12.58→7.88 ms/iter）全会话仅省 0.33 s（0.012%）。任何"换更大模型/加更多迭代"的提速收益都被 settle 墙淹没——效率优化的杠杆在**减少 settle 次数/缩短 settle**，不在算力。

## 4. 优化效果与效率的联合判读

1. **效果瓶颈是绝对强度口径，不是效率或算力**：模型侧 0.30 s/iter 已足够快，model_cv 也能降到 0.66；但实测 CV 停在 5.0。早前把这一差距归因于"FrozenDefocusForward 绝对强度表征误差、模型 vs 实测 CV 分叉 7.5×"**已作废**（§5.1：锚定轮 7.5× 是 8-bit 量化地板，另有探针过曝帧 argmax 定心失效的独立伪影）；重采后结构相关性达 0.95，剩余的是 checkpoint gain 的量纲不可迁移（§5.3），须在统一曝光/基座口径下重标定。
2. **效率杠杆在 settle，不在 GPU**：历史预算证明 settle 占 99%、优化器占 <0.05%。M3 把 s/iter 从 0.92 降到 0.30（grid 640→128）是有效提速，但进一步提速须动 settle/采集结构（如减少 anchor 重采、合并 display-capture）。
3. **早停使墙钟低估收敛成本**：两次均停滞早停，"秒/形状"反映非收敛态；要得到真实收敛效率，须先解决表征误差让闭环能收敛，再计时。

## 5. 绝对强度保真度实验数据（2026-11-16，SLM #1 + 大恒 CCD）

探针 `scripts/slm/absolute_fidelity_probe_20261116.py`（commit `807d456a`，满量程契约修复 commit `96e285ee`）。固定 12 种相位（zero、随机 RMS0.5/1.5 各 2 seed、两次锚定 best grid、x/y tilt 1/3 rad、Noll-4 离焦 1 rad），逐模式记录 `model_cv_core`、`measured_cv_core_window`、`measured_cv_core_full`、`mean_intensity_ratio_core`，输出到 `data/debug/slm_transfer/absolute_fidelity_12bit_fixed_20261116/`（summary.json / plan.json / device_log.json / frames/ / model/ / phase/）。

**采集口径**：`--auto-exposure --bit-depth 12`，生效曝光 **0.034 ms**（auto-exposure 将峰值压到目标 180 灰，未触上限；对比旧 12-bit 运行的 1000 ms 上限过曝）。同帧定心 `spot_xy=[1224, 1105]`、定心帧峰值 2383/4095（58%，未饱和）。

### 5.1 先修正上一节的 7.5×：它是采集伪影，不是表征误差

§1/§2.1 的"模型 vs 实测 CV 分叉约 7.5×"来自两次**锚定闭环**的实测 CV 地板，不是 FrozenDefocusForward 的表征缺陷。需要区分两个独立的采集伪影：

- **锚定轮（8-bit，固定 0.5 ms）——7.5× 的来源**：两次锚定运行 `exposure_ms=0.5`、位深为当时默认（8-bit），且定心正确（`spot_xy=[1224,1093]`，未触发 1000 ms 过曝 bug）。但 8-bit 下零级饱和、整形光被量化掩埋，**实测 CV 地板恒在 ~5**（6.077 / 5.006）；而模型侧 `model_cv` 是模型自身尺度上的真值（0.664）。5.006/0.664 ≈ **7.5×** 是"实测地板 / 模型真值"，不是模型错。
- **探针旧 12-bit 轮（argmax 定心失效）——model_cv 塌缩的来源**：12-bit 下曝光被满量程 bug 推到 1000 ms，帧全帧饱和；`np.argmax` 返回**首个**饱和像素（饱和块顶边 `[1220, 606]`），而非光斑中心 `[1224, 1093]`，采样窗与核心掩膜落到暗区，模型核心均值恰好等于 checkpoint 的 `background=2.8847`、`model_cv_core` 塌缩到 ~0，`corr(model_cv, measured_cv)` 出现 −0.51 的**伪相关**。
- 两者修复（`96e285ee`：12-bit 满量程契约 + 定心饱和守卫）后重采：`model_cv_core` 恢复为 0.40–4.17 的物理区间，`corr(model_cv_core, measured_cv_core_window)` 由 −0.51 转为 **+0.95**。故 §2.1 的 7.5× 分叉**作废**；`model_cv 3.89→0.66` 的优化有效性结论保留（模型侧收敛本身成立），但"瓶颈是绝对强度表征误差"的归因须按 5.3 重述。

> 两伪影的共性：**都是采集环节的伪影**（位深/曝光），不是物理预测模型的表征缺陷。这也正是 §5.3 要把"绝对强度比"与"结构相关"分开判读的原因。

### 5.2 逐模式指标（12-bit，0.034 ms）

| pattern | model_mean_core | model_cv_core | measured_mean_core | measured_cv_core | ratio_core |
|---|---|---|---|---|---|
| zero | 3.571 | 3.698 | 20.186 | 3.185 | 5.653 |
| rand_rms0p5_seed0 | 3.483 | 2.920 | 19.334 | 2.651 | 5.551 |
| rand_rms0p5_seed1 | 3.483 | 2.964 | 19.253 | 2.562 | 5.527 |
| rand_rms1p5_seed0 | 3.204 | 0.405 | 18.518 | 1.673 | 5.780 |
| rand_rms1p5_seed1 | 3.213 | 0.496 | 18.590 | 1.674 | 5.786 |
| anchored_m3_best | 3.571 | 3.801 | 19.858 | 2.956 | 5.561 |
| anchored_m3_rs15_best | 3.325 | 0.648 | 18.532 | 1.662 | 5.573 |
| tilt_x_1rad | 3.570 | 3.693 | 19.818 | 2.827 | 5.551 |
| tilt_x_3rad | 3.568 | 3.688 | 19.550 | 2.560 | 5.479 |
| tilt_y_1rad | 3.571 | 3.696 | 19.686 | 2.778 | 5.513 |
| tilt_y_3rad | 3.570 | 3.693 | 19.661 | 2.663 | 5.507 |
| defocus_z4_1rad | 3.572 | 4.167 | 19.719 | 2.762 | 5.521 |

### 5.3 判读：方向保真、绝对幅度不保真（量纲不可迁移）

**(a) 归一化结构（CV）高度相关，但由"两簇分离"主导，簇内排序弱。**
`corr(model_cv_core, measured_cv_core_window)` = **0.9497**，与全帧口径 `measured_cv_core_full` 相关 0.9496（窗口 vs 全帧自身相关 1.00，无裁剪口径问题）。但这是**聚类分离**驱动的：模型把图案切成"均匀簇"（model_cv < 1）与"光斑簇"（≥1）两组，实测也恰好分成对应两组，跨簇相关很高；**簇内**排序显著弱化——

- `spot-like` 簇（9 个，model_cv 2.9–4.2）：簇内相关 **0.41**；
- `uniform` 簇（3 个：rand_rms1p5 两 seed + anchored_m3_rs15_best）：簇内相关 **−0.89**（n=3，模型侧 CV 差异本就小，排序不可依赖）。

即：模型能区分"接近均匀"与"强光斑"两类，但**绝对均匀度数值错向且幅度不稳**：uniform 簇模型 CV 0.40–0.65，实测 1.66（模型高估均匀 ~3×）；spot-like 簇模型 CV ~3.7，实测 ~2.6–3.2（模型高估光斑弥散 ~1.2–1.4×）。方向 cosine 门禁（§2.2）通过不等于幅度对齐，这一结论在本批数据上成立。

**(b) 绝对强度比 ~5.5，且主要由基座口径差主导。**
`mean_intensity_ratio_core` 中位 **5.55**（区间 5.48–5.79，极窄 → 说明比值是系统性的乘性口径差，而非逐模式随机误差）。但拆开看：实测全帧基座（中位数）恒为 **15.0 counts**，而 checkpoint `background=2.8847`，基座比 15.0/2.88 = **5.2**，几乎解释掉整个 ratio_core。剥掉基座后，整形光信号比 `measured_sig/model_sig` = 6.6–11.0（中位 7.1），比值反而**变宽**——说明"整光信号"的增益口径差距真实存在且比基座差更显著。

**(c) 根因是量纲不可迁移，不是模型错。**
模型输出 `sampled·gain + background`（`gain=306.6547`，`background=2.8847`，`bias_counts=0`）。`sampled` 在窗内近似归一，模型核心均值仅 3.2–3.6 counts，而实测核心均值 ~18.5–20.2 counts。checkpoint 的 gain 是在**某一离线标定曝光/增益**下拟合的计数口径，`bias_counts=0` 使模型输出**与 CCD 曝光完全无关**——而实测强度随曝光线性缩放。因此"绝对强度比"**不是一个不变量**，其数值随采集曝光改变（0.034 ms 下为 ~5.5，旧 1000 ms 过曝下为 ~700–864）。把 gain 量纲当作可跨曝光迁移的绝对量，是 §2.1 "7.5× 表征误差"误判的直接来源。

**(d) 对 #234 的门禁建议。** 绝对强度门禁**不能**直接比 raw 计数比，须先在**同一曝光与同一基座口径**下标定（或改用与曝光无关的口径：如归一化到光斑峰值/能量守恒）。可设的最小门禁为：**`corr(model_cv_core, measured_cv_core_window) ≥ 0.8**（本批 0.95 已过，作方向+幅度联合的必要条件），并附加**归一化幅度差**（如剥基座后逐模式信号比的中位偏离 ≤ 2×）作为充分性补强。raw `ratio_core ≈ 5.5` 只登记、不设门禁，因量纲不可迁移。

**(e) 采集质量说明。** 0.034 ms 曝光下整形光已离开零级光斑的动态范围中心：零级斑心峰值 2265 counts、斑心 21×21 均值 1031，而 400×400 核心均值仅 20（核心区 ~95% 面积是 15 基座），整形光信号落在基座之上数 counts。若要更精确的绝对幅度标定，应把曝光压向零级不饱和同时整形光验证集余量（当前 target_max=180 已偏保守，可试 target_max 60–120 观察信噪比），或改在零级外采样整形光区域。本批数据满足"方向/结构判读"用途，绝对幅度仅作量级登记。

## 6. 数据缺口（影响效率横比完整性）

- **无 hardware-free 计时基线**：同一模型/优化器在 SLM+相机关闭下的墙钟未落盘，锚定运行中模型步成本与设备成本只能由 device_log 差分近似，无法精确分离。
- **device_log 无段标记**：无 `slm_write`/`settle`/`read`/`reduce` 细分（与 `wall_clock_budget_20261111.md` §7 同一限制）；settle vs capture vs compute 仅靠 display↔capture 时间差推导，读出开销未隔离。
- **锚定运行无 per-iter 模型步 ms**：12.58→7.88 ms/iter 来自独立离线 GPU bench，非闭环内联计时。
- **跨运行 checkpoint/常数不一致**：20261107（model640，0.92 s/iter）vs 20261116（M3，0.30 s/iter），墙钟非 apples-to-apples。
- **两次锚定均早停**（60/120 of 300），秒/形状为非收敛成本。

## 7. 数据与产物

- 零相位运行：`data/debug/slm_transfer/anchored_m3_20261116/`（timing.jsonl / device_log.json / iterations/0_square.jsonl / plan.json / phase/）
- 随机起步运行：`data/debug/slm_transfer/anchored_m3_rs15_20261116/`（同结构）
- 方向诊断：`data/debug/slm_transfer/direction_variants_20261112/{results,attribution}.json`；报告 `report/shape_trials/direction_diagnosis_20261112/direction_diagnosis_20261112.md`
- 历史墙钟预算：`report/shape_trials/multishape1200/wall_clock_budget_20261111.md`
- 历史锚定计时：`report/shape_trials/shape_transition/shape_transition_timing_20261107.md`
- 验收失败报告：`report/shape_trials/anchored_m3_20261116/anchored_m3_20261116.md`（commit `8a276ce8`）
- 绝对强度探针：`scripts/slm/absolute_fidelity_probe_20261116.py`（commit `807d456a`；满量程契约修复 commit `96e285ee`）
- 绝对强度保真度实验数据：`data/debug/slm_transfer/absolute_fidelity_12bit_fixed_20261116/`（summary.json / plan.json / device_log.json / frames/ / model/ / phase/；2026-11-16，SLM #1 + 大恒 CCD，12-bit 0.034 ms，`spot_xy=[1224,1105]`）
- M3 checkpoint：`data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`
