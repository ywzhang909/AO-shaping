# 无模型自适应光学综述（Selim 等 2026）× 本项目可借鉴点

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 人工调研
> **说明**: 无模型 AO 综述 (Selim 等 2026, J. Optics) × 本项目 model-free 栈 (SPGD/GS/RL/ML/正向模型) 逐条映射 + 6 条可借鉴清单 (自适应增益 / TIE / Cn² 自适应超参 / scintillation index / hybrid 调度 / RL 动态基准); 与 zotero_objectives 与 beam_shaping 文献调研互补, 不重复目标函数公式
<!-- provenance:end -->

> **来源**: 本地 Zotero 库条目 `T2H7VM3S`（副本 `5SCDP36J`），
> *Model-free adaptive optics for free space optical communications: A comprehensive survey*，
> Hebat Allah O. Selim, Rania M. Abdallah, Moustafa H. Aly, Islam E. Shaalan，
> *Journal of Optics* (2026-01-15)，DOI `10.1007/s12596-025-03011-z`。
> 全文 2.27 MB PDF，正文 + 三张汇总表（Table 1/2/4）已读取。
>
> **定位**: 这是一篇 **survey**（教程型综述），不是提出新算法。它的价值在于把"硬件 / ML / model-free"
> 三条线的**权衡**摊开，并明确给出"没有任何单一范式最优，hybrid 最稳健"的结论。
> 本文据此判断本项目现有栈处在什么位置、还缺什么。

---

## 1. 论文在说什么（浓缩）

**三大支柱 (three pillars)**：

1. **硬件为中心** (hardware-centric) — DM 致动器密度、Zernike 阶数、控制带宽。
   高密度 DM 提升校正保真但推高成本/复杂度。
   - 关键数据：173 元 DM 使耦合效率 (CE) 提升 **35×**；349 元 DM 在强湍流下 mixing efficiency 0.999、BER 10⁻¹¹。
   - 拟合误差 σ²_fit ≈ 0.28·(D/r0)^(5/3)·N^(-0.866)；CE = SI·CE_ideal，SI ≈ e^(−σ²φ) (Marechal)。
   - Zernike 阶数需求：D/r0≈2 → 仅 tip/tilt；D/r0≈10 → ≥9 阶；D/r0≈17 → ≥35 阶。

2. **ML 驱动** (ML-driven) — 用 PSF/图像预测波前，用 RL 做无传感控制。
   - CNN PSF→波前：RMSE 0.1263 waves @ D/r0=20。
   - ZOLA / PHASENET：RMSE 从 GS 的 0.124 μm 降到 **0.012 μm**，推理 <1 ms。
   - **无传感 RL**（无需 Shack-Hartmann）：静态 Strehl >55%、动态 >30%。
   - BP-ANN 无传感：约 100–500 Hz，但强湍流下需重训或退化为 hybrid SPGD-BP。

3. **Model-free 校正算法** — SPGD、Gerchberg–Saxton、pre-distortion，不依赖 WFS。
   - **SPGD**（Hu 等 [43]）：u_{k+1} = u_k + γ·ΔJ/δu_i，**自适应增益 γ = γ0/J + C**。
   - **AdamSPGD**（Zhang 等 [44]）：比裸 SPGD 少 ~50% 迭代，更强抗湍流波动。
   - **GS**（Li [45]）：弱湍流 RMS −28%、Strehl 0.51→0.63；**强湍流下经典 AO 反而更好**（RMS 0.3 vs 0.47 rad）。
   - **Pre-distortion**（DLR/Alphasat [46]）：卫星-地链双向跟踪，卫星端 scintillation 降 35%、指向稳定 2 μrad RMS。
   - **TIE (Transport of Intensity)**：无侵入相位测量，替代传统 WFS [Dorrer 48]。

**核心结论 (Table 4 综合)**：

| | 传统 AO | ML 增强 | Model-free (SPGD/GS) | 新兴 (PIC/量子) |
|---|---|---|---|---|
| 需要 WFS | 是 | 可选（ML 可绕过） | **否（无传感）** | 最小/集成 |
| 延迟 | 受 DM/FSM 机械限制 | 降（靠模型速度） | 低但需多次迭代 | 极低（ps–ns） |
| 可扩展性 | 受硬件尺寸/元数限制 | 高（边缘/联邦） | **中（自由度越大收敛越慢）** | 极高（芯片级） |
| 强湍流鲁棒性 | 弱（D/r0>10 退化） | 强（需宽条件训练） | **高（盲优化避开了模型失配）** | 有前景 |

> **一句话**：传统 AO 在强湍流下掉链子；ML 快但吃训练数据；**model-free 在不可预测湍流下最鲁棒，
> 代价是收敛慢 + 自由度大时扩展性差**；PIC/量子重定义可扩展性。没有单一最优，**hybrid 最稳**。

---

## 2. 论文方法 × 本项目现状（逐条映射）

> 本项目 model-free 栈 = `optimizer/wfless/`（SPGD/PIB/方形/GS-refine/model-in-loop）
> + `optimizer/rl/`（SAC、LR-WFS）+ `ml/`（U-Net+GAN、Zernike 前向模型、FourierGSNet、hwdataset）
> + `algorithm/signal_processing/`（GS、DifferentiableBeamOptimizer）+ 数字孪生 `drivers/sim/`。

| # | 论文方法 | 本项目对应 | 状态 |
|---|---|---|---|
| 1 | SPGD 盲优化（Hu [43]） | `optimizer/spgd.py::spgd_gradient` + `slm_zernike_pib`/`slm_square_shaping`/`gs_refine` 的 SPGD 循环 | ✅ 已覆盖 |
| 2 | AdamSPGD（Zhang [44]，少 ~50% 迭代） | SPGD + `algorithm/gradient/adam.py`（默认 `optimizer_type="adamod"`，另有 adam/adamw/muno/munow/muon/adamns/sgd） | ✅ 已覆盖（Adam 族 + AdaMOD） |
| 3 | GS 相位恢复（Li [45]） | `algorithm/signal_processing/gerchberg_saxton.py` + GS warm-start（`gs_refine` Stage 1） | ✅ 已覆盖 |
| 4 | Pre-distortion（DLR [46]） | `slm_model_in_loop.py`（正向模型闭环 + 目标相位合成，trust region + 逐轮验收） | ✅ 已覆盖（且更工程化） |
| 5 | 无传感 RL（Strehl 静态 >55%） | `optimizer/rl/`（SAC、LR-WFS） | ✅ 已覆盖 |
| 6 | ML PSF→波前（CNN RMSE 0.1263 waves） | `ml/phase/`（U-Net+GAN 相位预测）+ `ml/zernike/`（前向模型，10 折 CV，U-Net 胜） | ✅ 已覆盖（且做了配对统计验证） |
| 7 | ZOLA/PHASENET（ML 波前重建，RMSE 0.012 μm） | `optimizer/wf/zernike_response_matrix.py` + WFS 闭环 | ✅ 已覆盖（WFS 响应矩阵路线） |
| 8 | 自适应增益 γ = γ0/J + C | SPGD 用 `delta`（实测噪声地板设 0.2 rad）+ `lr` + `shrink_iter/shrink_ratio` 衰减 | ⚠️ **部分借鉴**（见 §3.1） |
| 9 | **TIE 无侵入相位测量** | 无（本项目 WFS = Thorlabs Shack-Hartmann） | 🆕 **可借鉴**（见 §3.2） |
| 10 | **Cn² 估计 / 湍流参数化**（variance-based 最佳） | 数字孪生有 OU 时变湍流，但**闭环没有 Cn² 反馈去调 SPGD 超参** | 🆕 **可借鉴**（见 §3.3） |
| 11 | nPIB/SIB/FOM 归一化目标（DiComo 2025） | PIB 已有；nPIB/FOM 见 `zotero_objectives` §4 | 🔗 见 `zotero_objectives`（不重复） |
| 12 | 高密度 DM（173/349 元） | SLM 1920×1200 = **~230 万像素**，远超任何 DM | ✅ 本项目已处于高 DOF 区 |
| 13 | PIC 片上相位调制 | 不适用（本项目用 SLM，非硅光） | ❌ 不适用 |
| 14 | 量子 AO（QKD/纠缠） | 不适用（本项目无量子链路） | ❌ 不适用 |
| 15 | 标准化验证框架（BER/CE + scintillation + 信息论） | 报告有配对统计；**scintillation index σ²_I（时域强度方差）未作为验证指标** | 🆕 **可借鉴**（见 §3.4） |

---

## 3. 可借鉴清单（按优先级）

> 优先级 = (对本项目 model-free 闭环的实际收益) × (落地成本)。
> 每条给出落点文件与验收口径，全部**离线可验证**（数字孪生 `--cam_type sim`）先行，硬件后验。

### 3.1 ⭐ 自适应 SPGD 增益（论文 §"SPGD" 的 γ = γ0/J + C）

- **论文**：SPGD 增益 `γ` 随目标函数值 `J` 自适应，`γ = γ0/J + C`，`J` 越小（越远）增益越大，加速早期收敛。
- **本项目现状**：`spgd_gradient(pos, neg, delta, maximize)` 给出方向 + 幅度 `(J_pos−J_neg)·δ`，
  幅度由 `algorithm/gradient/adam.py` 的 `lr` 缩放；`slm_zernike_pib` 另用 `shrink_iter`/`shrink_ratio`
  做**时间轴**学习率衰减。两者都是"按迭代步数"调度，**没有按目标函数值 `J` 调度**。
- **借鉴**：在 `slm_zernike_pib` 的 SPGD 循环里，把固定 `lr`/`shrink` 换成（或叠加）按当前
  `J`（有效目标值）的自适应增益：`lr_t = base · (γ0 / max(J_t, ε) + C)`。
  论文说 AdamSPGD 比裸 SPGD 少 ~50% 迭代；叠加 `J` 自适应后，**早期大 `J` 时大步、后期小 `J` 时小步**，
  与现有 `shrink_ratio` 的正则衰减正交，可能进一步缩短收敛。
- **落点**：`optimizer/wfless/slm_zernike_pib.py` 的 SPGD 主循环；新增 `SlmZernikePibConfig`
  字段 `adaptive_gain: bool` / `gamma0` / `gain_c`（默认关闭，保持现有行为可复现）。
- **验收**：`--cam_type sim` 数字孪生（`drivers/sim/slm_shaping_bench.py`），固定 seed，
  对比 `lr=固定` vs `lr=J 自适应` 的"达到 best J 的 80% 所需迭代数"（与论文 [43] 的口径一致）。
  目标：迭代数下降，且 best J 不劣化。

### 3.2 ⭐ TIE（Transport of Intensity）作为无 WFS 相位测量通道

- **论文**：Dorrer 的 TIE 提供**无侵入**相位测量，替代传统 WFS [48]。
- **本项目现状**：本项目 WFS 是 Thorlabs Shack-Hartmann（`drivers/wfs/`），需光路分光 + 校准
  （`zernike_response_matrix`）。在**无 WFS** 的纯 SLM+CCD 整形链里，目前只有盲 SPGD。
- **借鉴**：TIE 只需 **3 个离焦面**的强度图即可解相位（`∂I/∂z` 方程）。本项目已有 **近/远双相机**
  （`.env`: `Near_Cam_ID` / `Far_Cam_ID`），天然具备多面采样。把 TIE 实现为一个
  `PhaseEstimator`，与 SPGD 的"盲"路径并列，作为**有模型但无 WFS** 的中间档：
  - 比纯 SPGD 收敛快（有相位梯度先验）；
  - 比 WFS 省一套分光硬件。
- **落点**：`src/ao_shaping/algorithm/signal_processing/` 新增 `tie.py`（纯 numpy，仿真先行）；
  在 `slm_zernike_pib` 加 `--wfs-mode {blind, tie, shh}` 开关。
- **验收**：数字孪生里注入已知 Zernike 像差，比较 TIE 重建相位 vs 真值的 RMSE vs 纯 SPGD 的收敛。
  注意 TIE 对**低阶**像差最稳，高阶需要多面 + 正则化（这是已知限制，论文也点到）。

### 3.3 Cn² 估计 → 闭环自适应超参（论文 §"Emerging trends": variance-based 参数化最佳）

- **论文**：Cn²（折射率结构参数，10⁻¹⁷~10⁻¹³）决定湍流强度；variance-based 参数化比 flux/gradient 更准。
  强湍流需更多 Zernike 阶（D/r0≈17 → 35 阶）。
- **本项目现状**：数字孪生 `fouriergsnet_env.py` 有 OU 时变湍流（`configure_turbulence(sigma, tau, dt, nolls)`），
  但 `sigma` 是**设定值**；实机闭环**没有在线估计湍流强度**，SPGD 的 `delta`/`lr`/Zernike 阶数全靠手调。
- **借鉴**：在 CCD 帧序列上在线估计一个 **scintillation 代理量**（强度时间方差 / 峰度），
  映射到 Cn² 代理，据此**自动调度**：
  - 湍流弱 → 低阶 Zernike（省 DOF、快）；
  - 湍流强 → 升高阶 + 加大 `delta`（信号更强于噪声地板）。
- **落点**：`optimizer/wfless/slm_zernike_pib.py`（或抽到 `utils/` 的 `turbulence_estimator`）。
  `SlmZernikePibConfig` 加 `auto_turbulence: bool`。
- **验收**：数字孪生 `configure_turbulence` 扫 `sigma`（弱→强），验证自动 `n_max`/`delta` 选择
  比固定值在**全 sigma 区间**的平均 best J 更优。这与论文 Table 2 "control bandwidth limits overall
  performance" 的痛点直接对应。

### 3.4 把 scintillation index σ²_I 纳入验证指标（论文 §"Verification"）

- **论文**：验证 AO 不仅看静态 PSF（Strehl/PIB/CV），还看**时域** scintillation index σ²_I
  （接收强度方差）—— 这是 FSO 通信里"链路可靠性"的核心。
- **本项目现状**：报告用 PIB/CV/EE/AR/Strehl + 配对统计（`eval_stats.py`），都是**单帧空间**指标。
  没有时域 σ²_I。
- **借鉴**：在 `bench_stability`（`data/debug/bench_stability` 已有 42 条平场漂移记录，
  见 `hwdataset` 排除说明）或新增稳定性探针里，沿时间轴算接收光斑强度的
  `σ²_I = (⟨I²⟩−⟨I⟩²)/⟨I⟩²`，作为"AO 是否真稳"的补充指标。
- **落点**：`scripts/` 新增 `measure_scintillation_index.py`（读已存帧，离线）。
- **验收**：对同一台架，比较"开 AO" vs "关 AO"的 σ²_I 下降幅度（论文期望数量级：scintillation 降 87%@35 阶）。

### 3.5 ⭐ 用 GS warm-start + 模型闭环的 "hybrid" 作为默认推荐（论文核心结论的落地）

- **论文**：明确 GS 在弱湍流好、强湍流差；SPGD 在强湍流鲁棒但慢；**hybrid 最稳**。
- **本项目现状**：`slm_gs_refine` 已经做了 **GS warm-start + bake-off**（GS 赢了才采用）+ SPGD 细化；
  `slm_model_in_loop` 做了正向模型 + SPGD。但两者是**分离的命令**，没有统一的"按湍流强度自动选 hybrid 配方"。
- **借鉴**：把 §3.1 + §3.3 的信号接进来，让一个**调度器**根据在线 Cn² 代理自动选择：
  - 弱 → GS 主导（快）；
  - 中 → GS warm-start + SPGD；
  - 强 → 纯 SPGD（鲁棒）。
  这正是论文 "hybrid frameworks ... represent the most resilient pathway" 的可执行版本。
- **落点**：新增 `optimizer/wfless/hybrid_scheduler.py`，组合 `gs_refine`/`model_in_loop`/`spgd` 三个已有模块
  （**不重写** GS 循环，遵守 anti-pattern "Re-implementing the GS loop"）。
- **验收**：数字孪生扫 `sigma`，验证自动调度在**全区间**的 best J 不低于"人工选的最优单方法"。

### 3.6 无传感 RL 的"动态湍流"基准（论文：RL 静态 55% / 动态 30%）

- **论文**：无传感 RL 在**动态**湍流下 Strehl 只有 30%（静态 55%）—— 动态场景是 RL 的短板。
- **本项目现状**：`optimizer/rl/`（SAC）已有，数字孪生有 OU 时变湍流。
- **借鉴**：给 SAC 加一组**动态湍流**基准（`configure_turbulence` 开启），报告里同时给出静态/动态 Strehl，
  对齐论文的口径（"static vs dynamic"），判断本项目的 SAC 在动态下是否也掉到 ~30%。
  这是**验证**项目（确认 RL 的动态短板是否真实存在），不是新算法。
- **落点**：`optimizer/rl/` 训练脚本加 `--dynamic` 开关，复用 `fouriergsnet_env.configure_turbulence`。
- **验收**：输出 static vs dynamic Strehl 对比表（对齐论文 55% vs 30% 的口径）。

---

## 4. 已覆盖（论文提及但本项目已有，勿重复造）

- **SPGD / AdamSPGD**：本项目 `spgd_gradient` + `algorithm/gradient/adam.py`（默认 AdaMOD）。
  论文的 AdamSPGD（少 50% 迭代）≈ 本项目"SPGD + Adam 族"，**已落地**，且本项目多了
  噪声地板实测（`delta=0.2 rad` 来自 `measure_shape_sensitivity.py`）—— 比论文更工程化。
- **GS warm-start**：`slm_gs_refine` Stage 1 + bake-off。论文的"GS 弱湍流好"被本项目的
  "GS 赢才采用" 直接消解，**无风险**。
- **Pre-distortion / model-in-loop**：`slm_model_in_loop.py` 已是"模型在环"，比 DLR 的静态
  pre-distortion 更动态（每轮重拟合正向模型）。
- **ML 波前/相位预测**：`ml/phase`（U-Net+GAN）+ `ml/zernike`（前向模型）+ `ml/hwdataset`
  （硬件相位→相机图 DataLoader）。论文的 CNN PSF→波前（RMSE 0.1263 waves）是**单篇**结果；
  本项目做了 **10 折分组 CV + 配对 sign-flip 检验**（`compare_models_cv.py`），统计上更严格。
  **本项目领先**，无需借鉴算法，可借鉴的是论文 Table 4 的"ML 吃训练数据"这一定性风险
  —— 已在 `hwdataset` 的"折间难度差"分析里量化过。
- **无传感 RL**：`optimizer/rl`（SAC）。见 §3.6（补动态基准）。

---

## 5. 不适用（论文有，本项目没有对应硬件/场景）

- **PIC 片上相位调制** [27-33]：本项目用 Santec SLM（1920×1200 LCOS），不是硅光芯片。
  PIC 的"ps–ns 校正速度"对本项目无意义。
- **量子 AO**（QKD / 纠缠 / EB-radar）[29-34, 46-48]：本项目无量子链路。
  论文里 AO 对 QBER 的改善（9.6%→5.2%）与本项目的 SLM 整形目标不同轴。
- **单模光纤耦合**（M-SPGD, Yang 2020）：本项目是自由空间远场整形，无 SMF 耦合场景。

---

## 6. 与既有报告的关系（防重复）

| 报告 | 覆盖 | 与本文边界 |
|---|---|---|
| `report/zotero_objectives/README.md` | 目标函数/评价函数目录（PIB/CV/EE/MR/FOM/SSIM…） | 本文 **不复述公式**；§3.1/§3.3 引用的 nPIB/FOM 直接指向该文 §4 |
| `report/beam_shaping/papers/literature_survey.md` | 整形方法 25 篇（GS 族/可微/RL/目标函数/偏振） | 本文聚焦 **model-free AO 综述**（FSO/湍流导向），方法分类不重复 |
| `report/beam_shaping/papers/beam_shaping_papers.md` | 文献方法在同一光学模型上的仿真对比 | 本文 §3.1/§3.5 的验收**复用**该文同一套数字孪生基准 |
| `report/slm_pib_bench/` | SPGD delta 扫描 / 噪声地板 / 门控 | 本文 §3.1 的 `delta` 与该文 `delta_scan*.md` 同一旋钮，结论应一致 |

---

## 7. 建议的执行顺序

按"离线可验证 → 硬件后验"，每步不依赖硬件：

1. **§3.1 自适应增益**（纯算法，`slm_zernike_pib`，sim 验证迭代数）— 最小改动，最快收益。
2. **§3.4 scintillation index**（纯离线脚本，读已存帧）— 验证口径补全，无硬件风险。
3. **§3.6 RL 动态基准**（`optimizer/rl` + OU 湍流，sim 验证）— 确认 RL 动态短板。
4. **§3.2 TIE**（新增 `tie.py`，sim 注入已知像差验证 RMSE）— 新测量通道，中等工作量。
5. **§3.3 Cn² 自适应超参**（在线 scintillation 代理 → 调 `n_max`/`delta`，sim 扫 sigma）— 依赖 §3.4。
6. **§3.5 hybrid 调度器**（组合已有 GS/SPGD/model-in-loop，**不重写 GS 循环**）— 依赖 §3.1+§3.3。

> 全部 6 条都是**加法**（新增开关/模块），不改现有默认行为，保持可复现。
> 验收统一走数字孪生 `--cam_type sim`（`patch_sim_*_shaping`），硬件后验另列。
