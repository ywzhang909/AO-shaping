# 物理模型闭环整形：优化效果与效率（model_optim_efficiency 20261116）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/shape_transition_timing_20261106.py`](../../../scripts/slm/shape_transition_timing_20261106.py)
> **数据/关联脚本**: [`scripts/slm/absolute_fidelity_probe_20261116.py`](../../../scripts/slm/absolute_fidelity_probe_20261116.py)
> **运行环境**: 离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-16 两次实机锚定闭环后离线汇总；报告本体不占用设备）
> **说明**: 物理前向模型（FrozenDefocusForward / M3，grid 128 / FFT 512）驱动可微闭环的优化效果与效率汇总：模型侧 model_cv 3.89→0.66（随机起步规避焦斑盆地）、方向 cosine 0.696 过 ≥0.50 门禁；硬件侧实测核心 CV 停在 6.08/5.01 未达 ≤0.10，model_cv 与 measured_cv_window 同掩膜分叉约 7.5×（绝对强度表征误差）。效率：M3 闭环 ~0.30 s/iter（settle+capture 0.18 / 计算 0.11），比 model640 ~0.92 s/iter 快约 3×；历史预算证 LCOS settle 占会话 99%、优化器 <0.05%。绝对强度保真度实机数据待 SLM #1 上线补采（探针 absolute_fidelity_probe_20261116.py，commit 807d456a）。输入为两次锚定运行 timing/device_log/iterations、direction_variants_20261112、wall_clock_budget_20261111、shape_transition_timing_20261107。人工撰写。
<!-- provenance:end -->

**日期** 2026-11-16 · **脚本** `scripts/slm/shape_transition_timing_20261106.py`（实机闭环）+ 离线汇总 · **设备** Santec SLM-200 #1（1920×1200）+ Daheng MER2-507-23GM（2592×1944，0.5 ms，5 帧平均）
**Checkpoint** `defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`（grid 128 / FFT 512 / phase_sign −1.0 / 离焦+照明包络）

> 本报告聚焦**物理前向模型（FrozenDefocusForward / M3）驱动的可微闭环**在实验系统上的**优化效果**与**效率**两条主线，汇总 2026-11-16 两次实机锚定闭环、2026-11-12 离线方向诊断、以及 2026-11-07 / 2026-11-11 历史计时数据。绝对强度保真度的实机数据采集（探针 `scripts/slm/absolute_fidelity_probe_20261116.py`，commit `807d456a`）因 SLM #1 当前未连接而**待下次设备上线补采**，§5 预留其位置。

## 1. 结论速览

- **优化效果（模型侧）**：M3 在离线方向诊断中通过 cosine ≥0.50 门禁（0.696 中位，10/14 达标），是 6 个变体中唯一稳定通过者；随机起步下模型核心 CV 从 3.89（零相位）降到 0.66，证明**起步盆地已规避、模型侧优化方向有效**。
- **优化效果（硬件侧）**：两次实机锚定闭环实测核心 CV 分别停在 6.08（零相位）/ 5.01（随机起步），均**未达 ≤0.10 验收**；`model_cv` 与 `measured_cv_window` 同掩膜同网格分叉约 **7.5×**——方向对齐是必要条件而非充分条件，瓶颈是 FrozenDefocusForward 的**绝对强度表征误差**（详见 `report/shape_trials/anchored_m3_20261116/anchored_m3_20261116.md`）。
- **效率**：M3（grid 128）闭环单步墙钟约 **0.30 s/iter**，其中硬件 settle+capture 约 0.18 s/iter、模型计算约 0.11 s/iter；相比 2026-11-07 的 model640（grid 640）锚定闭环 0.92 s/iter **快约 3×**，主要来自更小的控制网格与 FFT。两次运行均因停滞提前结束（60/120 步，max 300），故墙钟反映的是**未收敛早停**成本，非收敛成本。

## 2. 优化效果

### 2.1 模型侧收敛（model_cv 轨迹）

| 运行 | 起步 | model_cv（iter10 → 末） | measured_cv（全帧） | model/measured | accepted |
|---|---|---|---|---|---|
| A `anchored_m3_20261116` | 零相位 | 3.801 → 3.888（iter60） | 6.05–6.14（min 6.077） | ~0.64× | 否 |
| B `anchored_m3_rs15_20261116` | 随机 RMS1.5 | 0.544 → 0.664（iter120） | 5.00–5.18（min 5.006） | ~0.13× | 否 |

- 随机起步使 model_cv 从 3.89 降到 0.66（**模型认为接近均匀**），证实零相位焦斑盆地已被规避；但 measured_cv 仅从 6.08 降到 5.01（**硬件几乎无改善**）。
- 两次均 `stop_reason=stall_5_blocks_no_0.005_improve`：模型侧与实测侧同时停滞，是表征误差主导（而非优化器/学习率）的特征签名。

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

> 注：`plan.json` 记录的 `control_grid 640 / fft_size 2048` 是脚本常量元数据；实际前向模型按 checkpoint 的 `grid_size 128 / fft_size 512` 构建（`FrozenDefocusForward(params, …)`），效率以实际网格为准。

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

1. **效果瓶颈是表征误差，不是效率或算力**：模型侧 0.30 s/iter 已足够快，model_cv 也能降到 0.66；但实测 CV 停在 5.0，7.5× 差距来自 FrozenDefocusForward 绝对强度表征，而非迭代数/学习率/墙钟。
2. **效率杠杆在 settle，不在 GPU**：历史预算证明 settle 占 99%、优化器占 <0.05%。M3 把 s/iter 从 0.92 降到 0.30（grid 640→128）是有效提速，但进一步提速须动 settle/采集结构（如减少 anchor 重采、合并 display-capture）。
3. **早停使墙钟低估收敛成本**：两次均停滞早停，"秒/形状"反映非收敛态；要得到真实收敛效率，须先解决表征误差让闭环能收敛，再计时。

## 5. 待补：绝对强度保真度实机数据（SLM #1 上线后）

探针 `scripts/slm/absolute_fidelity_probe_20261116.py`（commit `807d456a`，dry-run 验证 exit 0 / ruff 通过 / 未开设备）已就绪：固定 12 种相位（zero、随机 RMS0.5/1.5 各 2 seed、两次锚定 best grid、x/y tilt 1/3 rad、Noll-4 离焦 1 rad），逐模式记录 `model_cv_core`、`measured_cv_core_window`、`measured_cv_core_full`、`mean_intensity_ratio_core/window`，输出到 `data/debug/slm_transfer/absolute_fidelity_20261116/`。

- **当前阻塞**：2026-11-16 检查 SLM #1 `probe_slm(1)` 返回 `connected: false`（未连接），大恒 CCD 在线（SDK 枚举 1 台）。探针需 SLM+CCD 同时在线，故**待 SLM #1 下次上线补采**。
- 补采后将在此节填入：模型 vs 实测强度相关系数/直方图、逐模式强度比，量化"方向对但幅度错"的绝对强度误差，并据此为 #234 提出**绝对强度门禁**（如模型 vs 实测强度相关系数 ≥0.8）作为方向 cosine 门禁的补充。

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
- 绝对强度探针：`scripts/slm/absolute_fidelity_probe_20261116.py`（commit `807d456a`，待实机）
- M3 checkpoint：`data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`
