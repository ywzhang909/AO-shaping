# 当前 epoch fresh 单模屏离线双门评估（2026-11-18）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/direction_fresh_checkpoint_zernike_20261118.py`](../../../scripts/slm/direction_fresh_checkpoint_zernike_20261118.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/fit.py`](../../../src/ao_shaping/tools/slm/transfer/fit.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py`](../../../src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py)
> **运行环境**: 离线（仅回放归档帧与 checkpoint；未占用 SLM/CCD/激光，未启动任何采集进程）
> **说明**: 用当前 epoch（11/18 采集）fresh 单模 Zernike 屏对 11/18 fit 与各 hybrid 变体做方向门 + 零步绝对强度门回放。方向门 7 对中 11/18 fit 中位 0.6856、4/7 达 0.50，未达既有 10/14 标准（本轮仅单 pass 7 对，物理不可达）；强度门 in-sample 零步统计 11/18 fit relL2 0.355、CV 差 0.310 最优。结论：不存在同时过双门的 checkpoint，不进入硬件闭环验收。输入 data/debug/slm_transfer/current_epoch_zernike_direction_20261118、输出 data/debug/offline_refit_20261118_freshep。人工撰写。
<!-- provenance:end -->

## 结论

**没有 checkpoint 同时通过方向门与绝对强度门；本轮不进入物理设备闭环验收。** SLM #1、Daheng CCD、激光全程未被本轮分析占用（设备采集由主会话完成并已释放），未执行任何设备动作，无设备释放/关闭流程可执行。

1. **方向门（epoch-matched，7 对 ±Z5..Z11，amp=0.05）**：11/18 fit 中位数 0.6856，≥0.50 计数 4/7（Z5 0.877、Z8 0.843、Z10 0.791、Z11 0.686 过门；Z6 0.362、Z9 0.347、Z7 0.026 未过）。既有 10/14 标准需要 2 passes × 7 modes = 14 对，本轮数据只有单 pass 7 对，**物理上不可达**；按 10/14 同比例（71.4%）折算为 5/7，仅诊断变体 `hybrid320_fresh_all`（11/18 光学块、无残差项）达到 5/7，但其 Z7=0.306、Z11=0.222 两档低于 0.50，且它不是带残差项的完整 checkpoint。
2. **绝对强度门（零步回放，span=120）**：本批 15 个 zernike 行全部 split=train，无独立 held-out，因此本轮数字是 **in-sample** 零步统计，只能与 11/18 基线（12 条 held-out）作同指标跨 epoch 对照，不能宣称"通过"。11/18 fit 在 fresh 帧上 CV_pred 4.169 / CV_meas 4.479 / CV 绝对差 0.310 / relL2 0.355 / corr 0.931，为三者中最好；两个 refit 辐射 hybrid 的 relL2 恶化到 0.697–0.707、CV 绝对差 1.34–1.39。
3. **诊断信号**：11/18 残差 Noll 项在 fresh 屏上抬高中位数（0.6856 vs 无残差 0.5732），但压低 Z6/Z7/Z9（0.362/0.026/0.347 vs 无残差 0.573/0.306/0.611）；方向信息仍主要挂在 defocus/envelope 光学块上。

## 数据与完整性

- 数据根：`data/debug/slm_transfer/current_epoch_zernike_direction_20261118/`（manifest sha256 `CFAF9EB0D6786CE0E3B3BCDA6EB0FA12A03CCA1C43E071B522756DCDD2F2F0DB`，24 行，扁平结构、无 pass1/pass2 子目录——与 11/07 旧 run 的唯一结构差异）。
- 标定 9 行：`reference_start` + 8 个 ramp（ramp_x/y_64/128 ±1）。
- Zernike 方向屏：14 帧 amp=0.05（`ztrain_000..013`，7 对完整 ±Z5..Z11）+ 1 帧 amp=0.10（`ztrain_014`，Z5 +0.10，缺配对）。
- 每 npz：`raw_frames (2,1944,2592) uint8`、`frame`、`model_phase (128,128)`、`panel_phase (1200,1920)`、`panel_gray`、`exposure_probe_frames`；曝光 0.043 ms；peak 135–167 无饱和；light_check ok。
- **amp=0.10 弃用原因**：采集末段写第 25 个相位内存槽时 Santec 驱动报 `-10032` USB 错误，amp=0.10 只完成 ztrain_014 一帧、无配对；方向门 `pair_rows` 天然不选入该帧。该帧本身采集完整（peak 167），强度门同时报告含/不含它的两套统计（差异 <0.01，无实质影响）。
- 更早的 `data/debug/offline_refit_20261118_geoinit/`（含废弃计划的 `cand_geoinit.json`）已被本目录 `offline_refit_20261118_freshep` 取代，未参与本轮计算。

## 1. 方向门（正式 epoch-matched 方向筛查）

驱动：`data/debug/offline_refit_20261118_freshep/eval_direction_gate_20261118.py`（sha256 `7FC97DF5D388C8714AE54DDA3D7B661AC72BC2F3157044B266652946DDC2D675`）；结果：同目录 `direction_gate_results.json`；日志：`direction_gate_run.log`。

口径逐字复用 `scripts/slm/direction_fresh_checkpoint_zernike_20261118.py`（sha256 `BF1517E6D9B9B7707972361094A5C55BA5C969E09B839EDF585DFC1EB2A60DAF`）：指标 `jacobian_guard.direction_agreement`（Pearson）、`phase_sign=-1.0`、`phase_sign_scope="command"`（FrozenDefocusParameters 默认 +1 不依赖）、rate = raw_frames 均值 / exposure_ms、实测椭圆中心（全 24 行均值帧，method=ellipse）作 crop 中心与模型 origin、128 px crop、grid=128（本批帧 model_phase 边长）。

中心：全 24 行 (1226.0, 1106.0)；排除 ztrain_014 后 23 行 (1226.0, 1106.0) —— 中心对该帧不敏感，`11_18_fit` 的 23 行敏感性门与主门逐位相同（median 0.6856，4/7）。与 11/18 fit 的 origin (1225.86, 1104.06) 一致，epoch 注册正确。

| 候选 | 参数构成 | median | min | max | ≥0.50 |
|---|---|---:|---:|---:|---:|
| **11_18_fit** | 11/18 defocus/envelope + 11/18 残差 | 0.6856 | 0.0257 | 0.8770 | **4/7** |
| hybrid_test_candidate | 11/07 几何 + 11/18 残差 + refit 辐射（gain 1947.71） | 0.1598 | −0.0552 | 0.6812 | 2/7 |
| hybrid_center_fresh_resid | 11/07 defocus/width + 11/18 center + 11/18 残差 + refit 辐射（gain 2063.51） | 0.5196 | 0.2457 | 0.6361 | 4/7 |
| hybrid320_oldgeom_resid | 11/07 几何 + 11/18 残差（11/18 gain/bg） | 0.1598 | −0.0552 | 0.6812 | 2/7 |
| hybrid320_center_fresh | 11/07 defocus/width + 11/18 center，无残差 | 0.4759 | 0.0711 | 0.8228 | 3/7 |
| hybrid320_old_all | 11/07 几何，无残差 | 0.3454 | −0.0964 | 0.7873 | 3/7 |
| hybrid320_fresh_all | 11/18 defocus/envelope，无残差 | 0.5732 | 0.2224 | 0.6965 | **5/7** |
| M3_old_sanity（锚点） | 11/07 M3 + 11/07 ramp（错 epoch 几何，仅作诊断） | 0.3529 | −0.1184 | 0.7780 | 3/7 |

`11_18_fit` 逐档（Z5..Z11）：0.8770 / 0.3618 / 0.0257 / 0.8429 / 0.3465 / 0.7912 / 0.6856。
`hybrid320_fresh_all` 逐档：0.6538 / 0.5732 / 0.3056 / 0.6965 / 0.6109 / 0.5700 / 0.2224。

判读：

- **10/14 标准本轮不可达**：本轮 7 对（单 pass），10/14 的定义是 14 对中 10 对过 0.50；按同比例折算需 5/7。
- `11_18_fit` 4/7，未过。Z7 方向一致度 0.0257（verdict contradict）是显著短板；Z6/Z9 也低于 0.50。
- `hybrid320_fresh_all` 达到 5/7，但只是诊断变体（无 11/18 残差项），且 Z7 0.306、Z11 0.222 低于 0.50，不是完整可部署 checkpoint，不能据此宣称通过。
- `hybrid320_oldgeom_resid` 与 `hybrid_test_candidate` 数值完全一致（方向 Pearson 对 gain/bg 辐射缩放不敏感，两者几何与残差相同）。
- M3_old_sanity（错 epoch 几何）仅 3/7，说明 crop/origin 注册没有被错误地"对齐"到能虚高结果的程度；11/18 fit 明显优于它，epoch 注册生效。

## 2. 绝对强度门（零步回放）

驱动：`data/debug/offline_refit_20261118_freshep/eval_intensity_zero_step_20261118.py`（sha256 `ECD8DE2917F0937D8743828F8C32913AAFDFB01BDDE73138E2FBA7287B5602C6`）；代码路径与 11/18 基线相同：`ao_shaping.tools.slm.transfer.fit.build_fit_model + fit_loss`（`span=120`、`radiometry=exposure`、`adaptation=full`、`device=cuda`、`train_all=True`、iteration=0）。`train_all=True` 是必要的：本批 zernike 行全部 split=train，否则 `load_dataset` 抛 `ValueError: independent training and validation phases required`。结果：`intensity_<name>/result.json` + `intensity_summary.json`；日志 `intensity_run.log`。

**诚实口径声明**：本批无 held-out 行，以下数字是 in-sample 零步预测统计，只能与 11/18 基线（12 条 held-out，CV_pred 3.291 / CV_meas 3.832 / CV 绝对差 0.842 / relL2 0.529 / corr 0.844）作同指标跨 epoch 对照，**不是** held-out 意义上的"通过"。

| 候选 | n | mean CV_pred | mean CV_meas | mean CV 绝对差 | mean relL2 | mean corr |
|---|---:|---:|---:|---:|---:|---:|
| 11/18 fit（本批 in-sample） | 15 | 4.1690 | 4.4788 | 0.3102 | 0.3551 | 0.9308 |
| 11/18 fit（本批 in-sample，排除 ztrain_014） | 14 | 4.1678 | 4.4705 | 0.3033 | 0.3510 | 0.9325 |
| hybrid_test_candidate（in-sample） | 15 | 3.0902 | 4.4788 | 1.3886 | 0.7069 | 0.6898 |
| hybrid_center_fresh_resid（in-sample） | 15 | 3.1346 | 4.4788 | 1.3442 | 0.6971 | 0.6999 |
| （对照）11/18 fit held-out，11/18 采集集 | 12 | 3.291 | 3.832 | 0.842 | 0.529 | 0.844 |
| （对照）hybrid_oldgeom_resid held-out，11/18 采集集 | 12 | 2.206 | 3.832 | 1.626 | 0.735 | 0.665 |

- 11/18 fit 零步 relL2 0.355 与 CV 绝对差 0.310 在数值上好于其历史 held-out 的 0.529/0.842，但 in-sample 天然偏乐观，不能解读为"模型变好了"。
- 两个 refit 辐射 hybrid 在本批上 relL2 ≈0.70、CV 绝对差 ≈1.35，与它们在 11/18 集上 held-out 的表现（0.735/1.626）一致地差：refit 辐射没有修复绝对强度保真度。
- 逐行数据见各 `intensity_<name>/result.json`（含零步 radiometry：11_18_fit gain 1093.80、background 23.44）。

## 3. 双门结论

| 候选 | 方向门（7 对；10/14 折算需 ≥5/7） | 强度门（零步 in-sample，仅对照） | 双门 |
|---|---|---|---|
| 11_18_fit | 4/7，median 0.6856 —— 未过 | relL2 0.355、CV 差 0.310（in-sample） | **未过** |
| hybrid320_fresh_all | 5/7，median 0.5732（诊断变体，无残差项，Z7/Z11 低于 0.50） | 未跑强度门 | **未过**（非完整 checkpoint） |
| hybrid_test_candidate / hybrid320_oldgeom_resid | 2/7，median 0.1598 | relL2 0.707 | **未过** |
| hybrid_center_fresh_resid | 4/7，median 0.5196 | relL2 0.697 | **未过** |
| 其余 320 变体 / M3_old_sanity | ≤3/7 | — | **未过** |

不存在同时过双门的 checkpoint。

## 4. 失败机制归因：主导响应轴符号错配（本轮新增，2026-11-18 晚）

在 §1–§3 给出"没有 checkpoint 过双门"之后，本轮进一步用**离线消融 + 五个外生参数扫描 + 响应空间秩诊断**定位根因，结论是：方向门失败不是某个自由参数没调好，而是 11/18 fit 的**主导响应轴在 Z6/Z7/Z9 三档与实测反号**。全部脚本只读归档 npz，未占用设备。

### 4.1 排除项（按"失败集合移动、通过数封顶"读法）

| 检验 | 驱动脚本（sha256 前 8） | 结果 | 排除/保留 |
|---|---|---|---|
| 14 参数块消融 + 7 残差留一 | `ablate_direction_failure_20261118.py`（`dd00775c`） | 21 变体无一超过 5/7；`v01_no_residual` 5/7 但 Z11 .222；`v07_m3_origin`/`v08_m3_optical` 把相关打到 ~0 | 残差与光学块单块替换均无解 |
| 噪声预算 | `noise_budget_direction_20261118.py`（`6d37d7f8`） | 逐模式 SNR 2.26–5.04，噪声相关上限 0.914–0.981，全部远高于实测相关 | **排除噪声主导** |
| 差分形态 | 内联分析（见当日 notes） | Z5/Z7/Z8/Z9/Z10 实测差分 68–84% 能量由质心平移解释；Z6 仅 9%、Z11 0% | 定位到"平移 vs 细结构"两类模式 |
| 模型×实测 7×7 交叉匹配 | `cross_mode_match_20261118.py`（`e972edef`） | 对角线 6/7 行占优；无 sina/cos 整体互换；仅弱 Z7↔Z9（+0.50） | **排除索引错位** |
| 相位全局标尺 s 扫描 | `scale_sweep_20261118.py`（`6f7000cf`） | Pearson 对标尺完全不敏感：fit11 恒 4/7、零残差恒 5/7，s=0.5→24 相关变化 <0.02 | **排除幅值/灵敏度标尺** |
| 相机窗口 128→512 px | `window_sweep_20261118.py`（`0317794a`） | 128 是最好档（fit11 4/7、零残差 5/7），加宽单调恶化到 3/7、med 0.40；实测跨模式共线度 0.455→0.289 | **排除"核心窗导致共线"**（128 恰是 fit span） |
| 相机模糊 sigma_ccd_px 0→8 | `sigma_sweep_20261118.py`（`ca1ceec4`） | fit11 恒 4/7；零残差恒 5/7（Z7 .306→.428、Z11 .222→.280，但未过 0.50） | 细结构衰减有改善，不足以过门 |
| 离焦 defocus -0.009→0.18 | `defocus_sweep_20261118.py`（`daabdaa9`） | 11 档内通过数封顶 4/7（fit11）/5/7（零残差），只是失败集合换人；Z7 模型预测平移恒 ~0.10 px（实测 1.04 px），直到 d=0.18 才 0.30 px | **排除离焦标尺**（已验证 M3 的 +0.0309 也救不回 Z7） |

差分域重拟合 `diff_domain_refit_20261118.py`（`72e33fac`）把 s 与残差一起放开时，Adam 把 s 推到 17–40（相位多次回绕）、验证集相关塌到 ~0，是共线退化而非改进，未采用。选择性残差（只删 Z6/Z9 或 Z6/Z7/Z9）**在 fit 管线里根本不可表示**——`PhysicalForwardFit` 把 `residual_modes` 硬编码为 `range(5,12)`，删项后 `build_fit_model` 抛 `KeyError`，故该路线仅诊断、无强度门数字。

### 4.2 根因：主导轴符号错配（可证伪核对）

响应空间秩诊断 `response_rank_20261118.py`（`b551ca2e`）：把 7 个模式的 ± 差分各归一化后求 7×7 Gram 谱。

| 对象 | 参与比 | rank(>0.1) | PC1 能量占比 | 前 3 PC 特征谱 |
|---|---:|---:|---:|---|
| 实测差分 | 2.52 | 5 | **0.583** | 4.08 / 1.32 / 0.93 |
| 模型 fit11 | 3.12 | 4 | 0.460 | 3.22 / 1.78 / 1.38 |
| 模型零残差 | 2.33 | 3 | 0.519 | — |

实测差分在模型(fit11)张成空间的投影能量 0.870、反向 0.823——**不是"模型空间完全装不下实测"**，而是每个模式的方向在主导轴上对不上。实测 PC1 载荷：Z5 −0.92、Z8 −0.94、Z10 −0.95（一组），Z7 +0.91、Z9 +0.61、Z11 +0.43（反号组），Z6 −0.25 弱。

`pc1_sign_check_20261118.py`（`18ae9fba`）做了无需任何拟合的核对：把实测与模型(fit11)各自 PC1 的逐模式符号列出，与方向门通过集合比对——

| 模式 | 实测 PC1 | 模型 fit11 PC1 | 符号一致 | 方向门 corr | 过门 |
|---|---:|---:|:---:|---:|:---:|
| Z5 | −0.92 | −0.87 | ✅ | 0.877 | ✅ |
| Z6 | −0.25 | +0.58 | ❌ | 0.362 | ❌ |
| Z7 | +0.91 | −0.40 | ❌ | 0.026 | ❌ |
| Z8 | −0.94 | −0.81 | ✅ | 0.843 | ✅ |
| Z9 | +0.61 | −0.44 | ❌ | 0.346 | ❌ |
| Z10 | −0.95 | −0.91 | ✅ | 0.791 | ✅ |
| Z11 | +0.43 | +0.53 | ✅ | 0.686 | ✅ |

**模型 fit11 的"PC1 符号一致集合" {Z5, Z8, Z10, Z11} 与其方向门通过集合完全相同（4/7），零拟合参数。** 零残差变体把 Z6/Z7/Z9 的符号纠正过来（+0.58→−0.66 等），通过集合扩到 6/7 符号一致，但 Z11 被翻到 −0.30（实测 +0.43）而掉门，通过数仍是 5/7——这解释了"删残差救回 Z6/Z7/Z9、却塌掉 Z11"以及所有杠杆"通过数封顶 5/7、失败集合轮流换"的饱和现象：本质是 7 个残差/光学自由度在有限容量的主导轴上**此消彼长**，不存在同时对齐 7 档的配置。

### 4.3 无残差候选的强度门代价（补齐 §3 缺口）

§3 表中 `hybrid320_fresh_all` 只有方向门、没有强度门。本轮用与 §2 **完全相同**的代码路径补测（`eval_intensity_no_residual_20261118.py`，`3d8d42f6`，checkpoint `candidates/11_18_fit_no_residual/frozen_parameters.json`，残差全置 0.0——`build_fit_model` 不接受空 dict，但 basis×0 数值等价）：

| 候选 | n | mean CV_pred | mean CV_meas | mean relL2 | mean corr |
|---|---:|---:|---:|---:|---:|
| 11/18 fit（带残差） | 15 | 4.169 | 4.479 | **0.355** | **0.931** |
| 11/18 fit 无残差（in-sample） | 15 | 3.602 | 4.479 | 0.582 | 0.804 |

即"去掉残差"把方向门从 4/7 抬到 5/7，却把强度门 relL2 从 0.355 恶化到 0.582、corr 0.931→0.804——**双门严格互斥**，进一步坐实没有可用 checkpoint。

**机制结论**：在冻结模型类内，方向门与绝对强度门无法同时满足；根因是主导响应轴的符号错配，属模型表达能力与本 epoch 相位→光强映射的结构性失配，不是采集噪声、索引错位或标定标尺问题。继续在冻结参数里搜索**不会**改变这一结论。若要过双门，必须更换能表达实测响应形态的模型类（例：数据驱动的局部 Jacobian / 增加可学习的像差-平移耦合项），或按 11/18 响应形态修订方向门口径与门槛——**两者都需先由用户定夺口径，不在本轮离线工作中擅自改动**，且满足双门前不得进入锚定闭环验收。

复现（均为离线、只读归档；工作目录仓库根）：

```powershell
python data\debug\offline_refit_20261118_freshep\ablate_direction_failure_20261118.py
python data\debug\offline_refit_20261118_freshep\noise_budget_direction_20261118.py
python data\debug\offline_refit_20261118_freshep\cross_mode_match_20261118.py
python data\debug\offline_refit_20261118_freshep\scale_sweep_20261118.py
python data\debug\offline_refit_20261118_freshep\window_sweep_20261118.py
python data\debug\offline_refit_20261118_freshep\sigma_sweep_20261118.py
python data\debug\offline_refit_20261118_freshep\defocus_sweep_20261118.py
python data\debug\offline_refit_20261118_freshep\response_rank_20261118.py
python data\debug\offline_refit_20261118_freshep\pc1_sign_check_20261118.py
python data\debug\offline_refit_20261118_freshep\eval_intensity_no_residual_20261118.py
```

全部脚本 exit 0；各自 JSON 输出在同目录（`*_20261118.json` / `intensity_11_18_fit_no_residual/result.json`）。

## 设备与验收决策

本轮仅离线读取已归档帧与 checkpoint。SLM #1 未占用、Daheng CCD 未占用、激光未占用；无新设备动作，无需设备释放/关闭流程，未执行设备闭环。

**决定：不满足方向门与绝对强度双门，不运行 `shape_transition_timing_20261106.py --execute`，不报告 core CV ≤ 0.10 验收。** 11/18 fit 是当前最接近双门的候选（方向 4/7 + in-sample 强度最好），但其方向门在 Z7（0.026）、Z6（0.362）、Z9（0.347）三档低于 0.50；若继续推进，下一步是 11/18 采集集上的 held-out 强度回放（或新 epoch 的 2-pass 14 对方向屏），而不是再混用跨 epoch 帧。§4 进一步表明：在冻结模型类内双门严格互斥（无残差换方向丢强度），所以继续搜冻结参数不是有效路径；换模型类或修订门口径均需用户先定夺。

## 复现命令

工作目录 `D:\Projects\TIFO\AO-shaping`；均为 pwsh 单行命令。

```powershell
# 1) 方向门（8 候选 × 7 对；exit 0；日志 direction_gate_run.log）
.venv/Scripts/python.exe data\debug\offline_refit_20261118_freshep\eval_direction_gate_20261118.py

# 2) 绝对强度门（3 候选 × 零步回放；exit 0；日志 intensity_run.log）
.venv/Scripts/python.exe data\debug\offline_refit_20261118_freshep\eval_intensity_zero_step_20261118.py
```

两次最终运行退出码均为 0。

## 异常与失败记录

1. **文件系统陈旧快照（外部异常，已绕过）**：方向驱动最初用 importlib 加载 `scripts/slm/direction_fresh_checkpoint_zernike_20261118.py`，两次运行（均 exit 1）在脚本进程内读到旧版本内容（sha256 `d95bc7b6...`，无 `pair_rows` → `AttributeError`），而 `Get-FileHash` 与单独 `python -c` 检查读到的是当前版本（sha256 `BF1517E6...`，含 `pair_rows`）；文件 mtime（2026-11-18 10:18:00）与大小（9714 B）未变。怀疑 FS 层（过滤驱动/同步/杀毒）间歇性提供旧快照。处置：将 gate 脚本的 `CROP/MODES/_AMP/pair_rows/build_forward/crop` 逐字内联进驱动（附来源行号注释），运行时只读一次 gate 文件 sha256 作 provenance（记录值 `bf1517e6...`，与 Get-FileHash 一致）；最终运行 exit 0。
2. **本驱动排除掩码 bug（内部缺陷，已修复并重跑）**：amp=0.10 帧的排除条件写成 `"_0.10" in label`，而实际 label 结尾是 `+0.10`，导致"排除 ztrain_014"的 23 行中心与 n=14 强度统计实际未排除（两值相同，n 均显示 15）。修复为 `label.endswith("+0.10")` 后重跑两个驱动（均 exit 0）：23 行中心仍为 (1226.0, 1106.0)（中心确实不敏感，为真实结果），n=14 强度统计与 n=15 差异 <0.01。
3. 无其他设备/数据异常；amp=0.10 数据缺失的根因见"数据与完整性"（Santec -10032）。
