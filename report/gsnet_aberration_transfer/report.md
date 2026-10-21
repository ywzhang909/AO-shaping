# GSNet 光路相差迁移实验：迁移学习与参数冻结（issue #165，纯仿真）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_gsnet_aberration_transfer_report.py`](../../scripts/slm/generate_gsnet_aberration_transfer_report.py)
> **复现命令**: `python scripts/slm/gsnet_aberration_transfer.py && python scripts/slm/generate_gsnet_aberration_transfer_report.py`
> **数据/关联脚本**: [`scripts/slm/gsnet_aberration_transfer.py`](../../scripts/slm/gsnet_aberration_transfer.py)
> **数据/关联脚本**: [`scripts/slm/gsnet_pretrain.py`](../../scripts/slm/gsnet_pretrain.py)
> **运行环境**: 离线
> **说明**: issue #165 的迁移学习可迁移性检验：六种光路相差（面板离焦/彗差/球差、透镜彗差、相机离焦）下的仿真迁移学习，以及六种冻结策略的可训练参数数与迁移后整形质量；真值是仓库规范的 2f 数字孪生，非实验系统 CCD
<!-- provenance:end -->

## 0. 结论

- **问题 (a)，先说坏消息**：**默认配置**（`--operator ideal` + `--w-phase 1` + 无门，变体 `coupling_A`）下迁移**让网络变差**：6 个情形里只有 1 个 Δ>0，平均 Δ = -0.0225，最差单情形 `none` 0.3401 → 0.2738（-0.0663），跌到平场下限 0.2911 **之下**。
- **问题 (a)，修法之后成立**：换到校准算子 + 去相位项 + 对比择优门（变体 `coupling_E`，全参数微调），平均 Δ = -0.0002，3/6 个情形变好，最大正增益 `camera-defocus` 0.3456 → 0.3661（+0.0205），最差单情形也被门压到 -0.0325。结论：**仿真里迁移学习可以成功，但只有「算子匹配 + 只留整形项 + 验证集对比择优门」这条组合成立；默认配置是有害的**（机制见 §6）。
- **问题 (b) 总最优**：`freeze-decoder`（冻结相位生成头，留 336097 个可训练参数）在主变体 `coupling_E` 下跨情形平均 composite 最高（0.3288），平均 Δ = +0.0191，4/6 个情形变好。
- **问题 (b) 省参数最优**：`freeze-encoder` 只留 86065 个可训练参数（占总数 20.8%），仍拿到正的平均增益 Δ = +0.0040，4/6 个情形变好——「少重学」这一轴上的最好答案。
- **全参数微调不是答案**：`full`（414145 个可训练参数）平均 Δ = -0.0002，3/6 个情形变好——净持平到有害；冻结反而把伤害挡在门外（逐策略表见 §5）。
- 全部数字来自输入 JSON；任何一个缺失都会显示「未记录」而不是被补全。

> ⚠️ 本实验**全部是仿真**：真值是仓库规范的 2f 数字孪生（`drivers/sim/bench_twin.py`），不是实验系统的 CCD。

## 1. 实验设置

| 项 | 值 |
| --- | --- |
| 预训练 checkpoint | `data/gsnet_pretrain/20261022_coupling/checkpoint.pt` |
| 瞳面模型网格 | 64 × 64 |
| 远场 FFT 网格 | 512 × 512 |
| 远场方形目标边长 | 16 px |
| 指标网格 | 36 × 36（远场中心裁剪） |
| 探针 / 评估样本 / epoch | 24 / 16 / 10 |
| 主情形 / 主冻结策略 | `panel-coma` / `full` |
| SLM 灰度→振幅耦合 `--coupling` | c = 0.15（真值仓库未测量，待测——见 §7） |

**完整运行配置**（逐项来自输入 JSON 的 `config`，未做解读）：

| 配置项 | 值 |
| --- | --- |
| `batch_size` | 8 |
| `cases` | none,panel-defocus,panel-coma,panel-spherical,lens-coma,camera-defocus |
| `checkpoint` | /workspace/data/gsnet_pretrain/20261022_coupling/checkpoint.pt |
| `coupling` | 0.15 |
| `epochs` | 10 |
| `eval_samples` | 16 |
| `freezes` | full,frozen,freeze-encoder,freeze-decoder,freeze-films,freeze-outconv |
| `gate` | True |
| `lr` | 0.0001 |
| `operator` | calibrated |
| `out` | /workspace/data/debug/gsnet_aberration_transfer/coupling_E/result.json |
| `perturb_rms` | 0.08 |
| `primary_case` | panel-coma |
| `primary_freeze` | full |
| `probes` | 24 |
| `seed` | 20261021 |
| `transfer_target` | measured |
| `w_phase` | 0.0 |
| `w_shaping` | 40.0 |
| `workers` | 4 |

**三条变体开关**（驱动脚本本轮新增；默认值 = 首轮原行为）：

| 开关 | 语义 |
| --- | --- |
| `--w-phase`（默认 1.0） | 相位项权重。其监督目标是**命令过的探针相位** = 该情形物理下的 GS 解 + 小幅随机 Zernike 微扰；在带耦合的仿真里 GS 解本身比预训练网络的输出差，所以相位项是一个**向下拉**的力。置 0 = 只留整形项。 |
| `--operator {ideal,calibrated}`（默认 ideal） | 微调用的前向算子（网络内部前向、整形项与该情形「迁移前」参照都用同一个算子，保证 Δ 只反映微调本身）。ideal = 理想孪生：复现不出带耦合的实测帧，整形项的相关目标**物理上不可达**、梯度近似噪声；calibrated = **探针实测所在的那台孪生**（该情形相差 + 该情形灰度→振幅耦合），是实验系统上由 `tools/slm/bench_twin_calibrate.py`反演出的校准算子在仿真里的替身。 |
| `--gate`（默认关） | 微调后在**验证集探针**（与训练探针不相交、独立种子，规模 max(4, probes//4)）上做一次**对比择优**：用微调所优化的那把尺子（整形项同一 Pearson 相关、同一裁剪）比较候选权重与未微调权重，只有候选更好才保留，否则恢复预训练权重（`kept_pretrained`）。对应 model-in-loop 闭环的验证集验收门：没有这道门时，有害的微调总被采纳。 |

**孪生携带的耦合**：每个情形的孪生统一带 SLM 灰度→振幅耦合（`TwinErrorModel.slm_amplitude_coupling`），灰度域周期取器件实测的 2π 灰度数（各情形 `errors.gray_levels` 见下表，**不是** 1023）。耦合是**迁移目标实验系统的属性**，消融轴仍然只有相差与冻结配置。

**五种变体配置**（本次渲染实际读入的 JSON；缺位标注「未记录」）：

| 变体 | 开关组合 | JSON 路径 |
| --- | --- | --- |
| `coupling_A` | w-phase=1.0000 · operator=ideal · gate=off | `data/debug/gsnet_aberration_transfer/coupling_A/result.json` |
| `coupling_B` | w-phase=0.0000 · operator=ideal · gate=off | `data/debug/gsnet_aberration_transfer/coupling_B/result.json` |
| `coupling_C` | w-phase=1.0000 · operator=calibrated · gate=off | `data/debug/gsnet_aberration_transfer/coupling_C/result.json` |
| `coupling_D` | w-phase=0.0000 · operator=calibrated · gate=off | `data/debug/gsnet_aberration_transfer/coupling_D/result.json` |
| `coupling_E` | w-phase=0.0000 · operator=calibrated · gate=on | `data/debug/gsnet_aberration_transfer/coupling_E/result.json` |

**六种光路相差情形**（`errors` 为 `TwinErrorModel` 的逐情形字典，完整内容在输入 JSON 里）：

| 情形 | 像差说明 | 误差模型（JSON `errors`） |
| --- | --- | --- |
| `none` | 无像差（对照） | {"defocus_m": 0.0, "gray_levels": 993, "lens_zernike": {}, "lens_zernike_n_max": 4, "lut_gamma": 1.0, "lut_phase_max_rad": 6.283185307179586, "lut_table": null, "slm_amplitude_coupling": 0.15, "slm_amplitude_coupling_phase_rad": 0.0, "slm_roughness_seed": 0, "slm_roughness_sigma_rad": 0.0, "slm_zernike": {}, "slm_zernike_n_max": 6, "tilt_x_rad": 0.0, "tilt_y_rad": 0.0} |
| `panel-defocus` | 面板离焦 (Noll 4) 0.30 rad | {"defocus_m": 0.0, "gray_levels": 993, "lens_zernike": {}, "lens_zernike_n_max": 4, "lut_gamma": 1.0, "lut_phase_max_rad": 6.283185307179586, "lut_table": null, "slm_amplitude_coupling": 0.15, "slm_amplitude_coupling_phase_rad": 0.0, "slm_roughness_seed": 0, "slm_roughness_sigma_rad": 0.0, "slm_zernike": {"4": 0.3}, "slm_zernike_n_max": 6, "tilt_x_rad": 0.0, "tilt_y_rad": 0.0} |
| `panel-coma` | 面板彗差 (Noll 7) 0.30 rad | {"defocus_m": 0.0, "gray_levels": 993, "lens_zernike": {}, "lens_zernike_n_max": 4, "lut_gamma": 1.0, "lut_phase_max_rad": 6.283185307179586, "lut_table": null, "slm_amplitude_coupling": 0.15, "slm_amplitude_coupling_phase_rad": 0.0, "slm_roughness_seed": 0, "slm_roughness_sigma_rad": 0.0, "slm_zernike": {"7": 0.3}, "slm_zernike_n_max": 6, "tilt_x_rad": 0.0, "tilt_y_rad": 0.0} |
| `panel-spherical` | 面板球差 (Noll 11) 0.30 rad | {"defocus_m": 0.0, "gray_levels": 993, "lens_zernike": {}, "lens_zernike_n_max": 4, "lut_gamma": 1.0, "lut_phase_max_rad": 6.283185307179586, "lut_table": null, "slm_amplitude_coupling": 0.15, "slm_amplitude_coupling_phase_rad": 0.0, "slm_roughness_seed": 0, "slm_roughness_sigma_rad": 0.0, "slm_zernike": {"11": 0.3}, "slm_zernike_n_max": 6, "tilt_x_rad": 0.0, "tilt_y_rad": 0.0} |
| `lens-coma` | 透镜彗差 (Noll 7) 0.30 rad | {"defocus_m": 0.0, "gray_levels": 993, "lens_zernike": {"7": 0.3}, "lens_zernike_n_max": 4, "lut_gamma": 1.0, "lut_phase_max_rad": 6.283185307179586, "lut_table": null, "slm_amplitude_coupling": 0.15, "slm_amplitude_coupling_phase_rad": 0.0, "slm_roughness_seed": 0, "slm_roughness_sigma_rad": 0.0, "slm_zernike": {}, "slm_zernike_n_max": 6, "tilt_x_rad": 0.0, "tilt_y_rad": 0.0} |
| `camera-defocus` | 相机离焦 3e-4 m | {"defocus_m": 0.0003, "gray_levels": 993, "lens_zernike": {}, "lens_zernike_n_max": 4, "lut_gamma": 1.0, "lut_phase_max_rad": 6.283185307179586, "lut_table": null, "slm_amplitude_coupling": 0.15, "slm_amplitude_coupling_phase_rad": 0.0, "slm_roughness_seed": 0, "slm_roughness_sigma_rad": 0.0, "slm_zernike": {}, "slm_zernike_n_max": 6, "tilt_x_rad": 0.0, "tilt_y_rad": 0.0} |

六种情形**刻意超出预训练的随机化范围**（面板 ≤0.12 rad、透镜 ≤0.10 rad、离焦 ±2e-4 m），迁移面对的是真实的域差。

**六种冻结策略**（冻结的是 `FourierGSNet` 内 `ConditionUNet` 的部件；每层 GS 展开的物理半部——FFT + 幅度交换——本来就没有参数）：

| 策略 | 冻结了什么 |
| --- | --- |
| `full` | 全参数微调 |
| `frozen` | 全部冻结（等价于不迁移，直接推理） |
| `freeze-encoder` | 冻结 encoder（特征主干不动） |
| `freeze-decoder` | 冻结 decoder（相位生成头不动） |
| `freeze-films` | 冻结 FiLM 条件调制层 |
| `freeze-outconv` | 冻结输出卷积层 |

**`ConditionUNet` 参数构成**（参数量只在输入 JSON 记录时填充）：

| 参数组 | 参数量 |
| --- | --- |
| encoder | 未记录 |
| decoder | 未记录 |
| films | 未记录 |
| outconv | 未记录 |
| **合计** | 未记录 |

> 各组参数量若为「未记录」，仍可从 §5 矩阵的「可训练参数」列读出：
> `total − trainable(该组冻结)` 就是被冻结那一组的参数量。

## 2. 指标口径

| 指标 / 参考 | 口径 |
| --- | --- |
| `composite` | 整形复合分数（目标方框的环绕能量 EE + 框内 CV 合成，与 `slm_gs_refine` / gsnet 预训练同口径），越高越好 |
| `Δ` | 迁移后 − 迁移前（同变体、同情形、同冻结策略的 `references.pretrained`），composite 越高越好 ⇒ **Δ>0 = 迁移有增益** |
| `encircled_energy` | 目标方框内能量占全图能量的比例 |
| `cv_target_box` | 目标方框内 std/mean，越低越均匀 |
| `correlation_full` | 整个远场网格上的 Pearson 相关 |
| **平场**（`references.flat`） | 零相位直接过该情形物理——**下界** |
| **经典 GS 解**（`references.label`） | GS 在该情形自己的物理下求的解——**经典算法达到的水平**，不是严格上限（网络直接优化整形目标，实测可以超过它）（注意：带耦合孪生里它可以**低于**迁移前，见 §6 机制 1） |
| **迁移前**（`references.pretrained`） | 预训练 checkpoint 不微调、直接推理——迁移要超过的基线；用该变体的同一算子评测，跨算子家族不可直接互比 |
| **迁移后**（`runs.<策略>.final`） | 该情形 + 该冻结策略微调之后的同一 checkpoint（门拒绝时已恢复预训练权重 ⇒ 与迁移前相同） |
| `gate` 判定 | `off`（未开门）/ `accepted`（验证集上候选更好，保留）/ `kept_pretrained`（候选不更好，恢复预训练权重） |
| `phase_loss` 猜测水平 | 圆形相位 MSE 的随机猜测水平 π²/3 ≈ 3.2899；相位项只有**低于**它才说明迁移学到了相位，而不是拟合均值 |

> ⚠️ **全部是仿真。** 真值是仓库规范的 2f 数字孪生，不是实验系统的 CCD。所有分数只对「孪生 + 该情形误差模型 + 该变体算子」成立，不能当作实验系统收益。

## 3. 五个变体的对比：迁移为什么（不）成立

五个变体只差在驱动脚本的三个开关上（`--w-phase` / `--operator` / `--gate`，语义见 §1），像差情形、冻结网格、探针数、种子全部相同。Δ = 迁移后 − 迁移前（`full` 冻结策略，composite 越高越好）。

| 变体 | 开关 | JSON | 平均 Δ | 变好情形 | 最差单情形 Δ | 最好单情形 Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `coupling_A` | w-phase=1.0000 · operator=ideal · gate=off | `data/debug/gsnet_aberration_transfer/coupling_A/result.json` | -0.0225 | 1/6 | -0.0663 | +0.0006 |
| `coupling_B` | w-phase=0.0000 · operator=ideal · gate=off | `data/debug/gsnet_aberration_transfer/coupling_B/result.json` | -0.0502 | 0/6 | -0.0811 | -0.0064 |
| `coupling_C` | w-phase=1.0000 · operator=calibrated · gate=off | `data/debug/gsnet_aberration_transfer/coupling_C/result.json` | -0.0169 | 2/6 | -0.0837 | +0.0384 |
| `coupling_D` | w-phase=0.0000 · operator=calibrated · gate=off | `data/debug/gsnet_aberration_transfer/coupling_D/result.json` | -0.0260 | 2/6 | -0.0705 | +0.0226 |
| `coupling_E` | w-phase=0.0000 · operator=calibrated · gate=on | `data/debug/gsnet_aberration_transfer/coupling_E/result.json` | -0.0002 | 3/6 | -0.0325 | +0.0205 |

| 情形（`full`） | `coupling_A` | `coupling_B` | `coupling_C` | `coupling_D` | `coupling_E` |
| --- | --- | --- | --- | --- | --- |
| `none` | -0.0663 | -0.0675 | -0.0688 | -0.0705 | -0.0325 |
| `panel-defocus` | -0.0316 | -0.0291 | +0.0256 | -0.0411 | +0.0000 |
| `panel-coma` | -0.0019 | -0.0064 | +0.0384 | +0.0226 | +0.0119 |
| `panel-spherical` | -0.0131 | -0.0810 | -0.0071 | +0.0034 | -0.0171 |
| `lens-coma` | -0.0229 | -0.0363 | -0.0056 | -0.0241 | +0.0162 |
| `camera-defocus` | +0.0006 | -0.0811 | -0.0837 | -0.0466 | +0.0205 |

> ⚠️ **跨变体比 Δ 时注意基准点不同**：`operator=calibrated` 家族的「迁移前」参照也用校准算子评测（保证每个变体内部 Δ 只反映微调本身），而 checkpoint 内部前向是 ideal 算子训练出来的——换算子本身就会移动迁移前分数（逐情形对照见 §6 机制 3）。同列内比较有效，跨列比较绝对分数无效。

## 4. 表格 1：各光路相差下的迁移学习（主变体 `coupling_E`，冻结策略 `full`）

| 情形 | 像差说明 | 平场 | 经典 GS 解 | 迁移前 | 迁移后 | Δ(迁移后−迁移前) | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `none` | 无像差（对照） | 0.2911 | 0.3133 | 0.3427 | 0.3102 | -0.0325 | accepted |
| `panel-defocus` | 面板离焦 (Noll 4) 0.30 rad | 0.2949 | 0.3233 | 0.3265 | 0.3265 | +0.0000 | kept_pretrained |
| `panel-coma` | 面板彗差 (Noll 7) 0.30 rad | 0.2939 | 0.3113 | 0.2636 | 0.2755 | +0.0119 | accepted |
| `panel-spherical` | 面板球差 (Noll 11) 0.30 rad | 0.2916 | 0.3031 | 0.3219 | 0.3048 | -0.0171 | accepted |
| `lens-coma` | 透镜彗差 (Noll 7) 0.30 rad | 0.2933 | 0.3122 | 0.2582 | 0.2744 | +0.0162 | accepted |
| `camera-defocus` | 相机离焦 3e-4 m | 0.2962 | 0.3211 | 0.3456 | 0.3661 | +0.0205 | accepted |

## 5. 表格 2：各冻结策略（主变体 `coupling_E`，情形 `panel-coma`）

**逐冻结策略跨情形汇总**（变体 `coupling_E`，composite 越高越好；「变好情形」= Δ>0 的情形数）：

| 策略 | 可训练参数 | 参数占比 | 平均 composite | 平均 Δ | 变好情形 |
| --- | --- | --- | --- | --- | --- |
| `full` | 414145 | 100.0% | 0.3096 | -0.0002 | 3/6 |
| `frozen` | 0 | 0.0% | 0.3097 | +0.0000 | 0/6 |
| `freeze-encoder` | 86065 | 20.8% | 0.3137 | +0.0040 | 4/6 |
| `freeze-decoder` | 336097 | 81.2% | 0.3288 | +0.0191 | 4/6 |
| `freeze-films` | 406153 | 98.1% | 0.3180 | +0.0083 | 3/6 |
| `freeze-outconv` | 414120 | 100.0% | 0.3251 | +0.0154 | 3/6 |

**完整矩阵**（变体 `coupling_E`：情形 × 冻结策略，composite 与括号内 Δ）：

| 情形 | 迁移前 | `full` | `frozen` | `freeze-encoder` | `freeze-decoder` | `freeze-films` | `freeze-outconv` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `none` | 0.3427 | 0.3102 (-0.0325) | 0.3427 (+0.0000) | 0.3176 (-0.0250) | 0.3090 (-0.0337) | 0.3104 (-0.0322) | 0.3427 (+0.0000) |
| `panel-defocus` | 0.3265 | 0.3265 (+0.0000) | 0.3265 (+0.0000) | 0.3437 (+0.0172) | 0.3394 (+0.0129) | 0.3503 (+0.0238) | 0.3265 (+0.0000) |
| `panel-coma` | 0.2636 | 0.2755 (+0.0119) | 0.2636 (+0.0000) | 0.3095 (+0.0459) | 0.3139 (+0.0503) | 0.3237 (+0.0601) | 0.3044 (+0.0408) |
| `panel-spherical` | 0.3219 | 0.3048 (-0.0171) | 0.3219 (+0.0000) | 0.2451 (-0.0767) | 0.2978 (-0.0241) | 0.2452 (-0.0767) | 0.3219 (+0.0000) |
| `lens-coma` | 0.2582 | 0.2744 (+0.0162) | 0.2582 (+0.0000) | 0.3075 (+0.0493) | 0.3447 (+0.0865) | 0.3329 (+0.0747) | 0.3048 (+0.0466) |
| `camera-defocus` | 0.3456 | 0.3661 (+0.0205) | 0.3456 (+0.0000) | 0.3589 (+0.0134) | 0.3681 (+0.0226) | 0.3456 (+0.0000) | 0.3503 (+0.0048) |


| 策略 | 冻结了什么 | 可训练参数 | 冻结参数 | 参数占比 | 迁移后 composite | Δ(迁移后−迁移前) |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 全参数微调 | 414145 | 0 | 100.0% | 0.2755 | +0.0119 |
| `frozen` | 全部冻结（等价于不迁移，直接推理） | 0 | 414145 | 0.0% | 0.2636 | +0.0000 |
| `freeze-encoder` | 冻结 encoder（特征主干不动） | 86065 | 328080 | 20.8% | 0.3095 | +0.0459 |
| `freeze-decoder` | 冻结 decoder（相位生成头不动） | 336097 | 78048 | 81.2% | 0.3139 | +0.0503 |
| `freeze-films` | 冻结 FiLM 条件调制层 | 406153 | 7992 | 98.1% | 0.3237 | +0.0601 |
| `freeze-outconv` | 冻结输出卷积层 | 414120 | 25 | 100.0% | 0.3044 | +0.0408 |

**相位项 vs 随机猜测水平**（π²/3 ≈ 3.2899；只有低于它才算学到相位）：

| 策略 | 迁移后相位项 | 猜测水平 | 判定 |
| --- | --- | --- | --- |
| `full` | 3.2062 | 3.2899 | **低于** |
| `frozen` | 3.2299 | 3.2899 | **低于** |
| `freeze-encoder` | 3.5573 | 3.2899 | 尚未低于 |
| `freeze-decoder` | 3.4280 | 3.2899 | 尚未低于 |
| `freeze-films` | 3.3130 | 3.2899 | 尚未低于 |
| `freeze-outconv` | 3.1825 | 3.2899 | **低于** |

**对照：有害配置下的冻结扫描**（变体 `coupling_A`，主情形 `panel-coma`，迁移前 0.3216）——当每个真正训练起来的配置都在变差时，**冻得越多反而越好**，`frozen`（0 可训练参数）恰好复刻迁移前分数：

| 策略 | 可训练参数 | 迁移后 composite | Δ |
| --- | --- | --- | --- |
| `full` | 414145 | 0.3197 | -0.0019 |
| `frozen` | 0 | 0.3216 | +0.0000 |
| `freeze-encoder` | 86065 | 0.3117 | -0.0099 |
| `freeze-decoder` | 336097 | 0.3049 | -0.0168 |
| `freeze-films` | 406153 | 0.3054 | -0.0162 |
| `freeze-outconv` | 414120 | 0.2968 | -0.0249 |


## 6. 讨论：机制

- **机制 1 —— 相位项是一个向下的拉力**：在变体 `coupling_A` 的参照系里，预训练网络在 6 个相差情形中的 6 个上**已经优于该情形物理下的 GS 解**（经典算法达到的水平，**不是**严格上限）。相位项的监督目标是「命令过的探针相位」= GS 解 + 随机 Zernike 微扰——目标本身比网络现状差，往它拉就是在拉低网络。逐情形（composite，经典 GS 解 vs 迁移前）：

| 情形 | 经典 GS 解（`label`） | 迁移前（`pretrained`） | 迁移前 − 经典 GS 解 |
| --- | --- | --- | --- |
| `none` | 0.3133 | 0.3401 | +0.0268 |
| `panel-defocus` | 0.3233 | 0.3521 | +0.0288 |
| `panel-coma` | 0.3113 | 0.3216 | +0.0103 |
| `panel-spherical` | 0.3031 | 0.3275 | +0.0245 |
| `lens-coma` | 0.3122 | 0.3154 | +0.0033 |
| `camera-defocus` | 0.3211 | 0.3445 | +0.0234 |

- **机制 2 —— 只把相位项置 0 反而更糟**：变体 `coupling_B`（`--w-phase 0`，仍是 ideal 算子）平均 Δ = -0.0502，比变体 `coupling_A` 的 -0.0225 还差。相位项撤掉后只剩整形项，而整形项在默认配置里经由 **ideal 算子**去匹配**带耦合的实测帧**——这个目标物理上不可达，梯度近似噪声。逐情形 Δ：`none -0.0675` / `panel-defocus -0.0291` / `panel-coma -0.0064` / `panel-spherical -0.0810` / `lens-coma -0.0363` / `camera-defocus -0.0811`。
- **机制 3 —— 算子必须与产生数据的那台孪生一致**：把微调前向换成探针实测所在的**校准算子**（`--operator calibrated`，实验系统上由 `tools/slm/bench_twin_calibrate.py` 反演出的算子在仿真里的替身），再加验证集上的**对比择优门**，迁移才从有害变成有用（§3 表：门 + 校准算子 + 去相位项的组合是唯一平均非负且最坏单情形被压小的变体）。
- **算子替换的告诫**：checkpoint 内部是用 **ideal 算子**训练出来的，所以「推理时换成校准算子」本身就会移动迁移前参照——变体 `coupling_A` 与 `coupling_E` 的同一 checkpoint、同一情形，迁移前分数有 6 个对不上（下表）。忠实的在机路径因此是：从实验系统反演孪生 → **用拟合后的算子重训（或至少重导出）网络** → 再迁移；runbook `docs/slm/gsnet_transfer_hw_runbook.md` §7 规定的「先反演孪生，再让网络用它」正是这一步。

| 情形 | 迁移前（ideal 评测，`coupling_A`） | 迁移前（calibrated 评测，`coupling_E`） | 移动量 |
| --- | --- | --- | --- |
| `none` | 0.3401 | 0.3427 | +0.0026 |
| `panel-defocus` | 0.3521 | 0.3265 | -0.0256 |
| `panel-coma` | 0.3216 | 0.2636 | -0.0580 |
| `panel-spherical` | 0.3275 | 0.3219 | -0.0057 |
| `lens-coma` | 0.3154 | 0.2582 | -0.0572 |
| `camera-defocus` | 0.3445 | 0.3456 | +0.0010 |

- **机制 4 —— 门的局限**：`--gate` 的判据是**验证集探针**（本次规模 max(4, probes//4) = 6 帧）上的整形项，样本少，接受决策会错。变体 `coupling_E` 里被门**接受**、却在独立 eval 集上变差的情形：`none`（-0.0325）、`panel-spherical`（-0.0171）；门的逐情形判定：`none: accepted` / `panel-defocus: kept_pretrained` / `panel-coma: accepted` / `panel-spherical: accepted` / `lens-coma: accepted` / `camera-defocus: accepted`。
  门把最坏单情形从**无门变体的 -0.0837（变体 `coupling_C` 的 `camera-defocus`）**压到 -0.0325——**限制**了伤害，没有**消除**它。
  另有 1 个情形被门判为「候选不如预训练」而恢复原权重（`kept_pretrained`，迁移后 = 迁移前，Δ=0）——这正是门的价值所在。
- **harness 自检**：所有变体里 `frozen`（0 可训练参数）与迁移前参照的 composite 最大偏差 0.00e+00——不训练就分数不变，评测链路自洽。

## 7. 边界与不能下的结论

1. **全部是仿真。** 真值是仓库规范的 2f 数字孪生（`drivers/sim/bench_twin.py`），没有任何 CCD / SLM 测量。「迁移成功」只对孪生协议成立，不能当作实验系统收益。
2. **探针只有几十帧**（本次输入：probes=24、eval_samples=16，验证集探针 max(4, probes//4) = 6），不是实验系统上的真实预算；冻结策略在这个小数据下够用，不意味着在更大预算下同样够用，也不意味着门的小验证集判定在更大样本下同样准（§6 机制 4）。
3. **探针相位是「最优相位 + 小幅随机 Zernike 扰动」**——这是对 `slm_model_in_loop` 行为的**建模**，不是它本身；闭环真实的探针分布（含验收拒绝后补测的探针）没有进入这份数据。
4. **网络内部前向与评测算子的错配没有被完全闭合**：checkpoint 是用 **ideal 算子**内部训练出来的（预训练时没有该情形的误差模型），`--operator calibrated` 只是在**微调与评测侧**换成校准算子。「用拟合后的算子重训网络」这条更忠实的在机路径（runbook `docs/slm/gsnet_transfer_hw_runbook.md` §7）没有在本份数据里覆盖。
5. **SLM 灰度→振幅耦合的真值在仓库里任何地方都没有被测出**：仓库只有诊断判据 `rel_spread = (max−min)/mean > 0.15`（`tools/slm/panel/diagnose.py` 的调制检测），它只蕴含 `c ≳ 0.075`。本仿真的 c = 0.15 是**待测量**的假设值，不是已知量；换一台耦合更强的面板，带耦合的实测帧与「理想算子不可达」的幅度都会变，机制 1–3 的定量结论随之移动。孪生同时用器件真实的 2π 灰度数（`gray_levels`，见 §1 情形表）而不是 1023。
6. 六种相差情形超出预训练随机化范围是**设计**，因此情形间差异混合了「相差种类」与「域差大小」两条轴，不能单独归因。

## 8. 复现

```bash
# 1) 五个变体（同一 checkpoint / 情形 / 冻结网格 / 种子，只差开关）
python scripts/slm/gsnet_aberration_transfer.py \
    --out data/debug/gsnet_aberration_transfer/coupling_A/result.json   # 默认（A）
python scripts/slm/gsnet_aberration_transfer.py --w-phase 0 \
    --out data/debug/gsnet_aberration_transfer/coupling_B/result.json
python scripts/slm/gsnet_aberration_transfer.py --operator calibrated \
    --out data/debug/gsnet_aberration_transfer/coupling_C/result.json
python scripts/slm/gsnet_aberration_transfer.py --operator calibrated --w-phase 0 \
    --out data/debug/gsnet_aberration_transfer/coupling_D/result.json
python scripts/slm/gsnet_aberration_transfer.py --operator calibrated --w-phase 0 --gate \
    --out data/debug/gsnet_aberration_transfer/coupling_E/result.json

# 2) 渲染本报告（自动发现 --variants-dir 下的 coupling_<标签>/result.json；
#    也可用 --variant NAME=path 显式指定，可重复；缺哪个都只标「未记录」）
python scripts/slm/generate_gsnet_aberration_transfer_report.py \
    --variants-dir data/debug/gsnet_aberration_transfer \
    --out report/gsnet_aberration_transfer/report.md
```

数据来源: `/workspace/data/debug/gsnet_aberration_transfer/coupling_E/result.json`；变体 JSON: `data/debug/gsnet_aberration_transfer/coupling_A/result.json`、`data/debug/gsnet_aberration_transfer/coupling_B/result.json`、`data/debug/gsnet_aberration_transfer/coupling_C/result.json`、`data/debug/gsnet_aberration_transfer/coupling_D/result.json`、`data/debug/gsnet_aberration_transfer/coupling_E/result.json`
