# OpenChamber 上下文 issue 离线复核与后续数据计划（2026-11-19）

> 范围：复核本会话上下文中的 7 个 issue（#234/#235/#236/#238/#242/#244/#247），结合当前 GitHub 状态、仓库采集记录和现存离线数据。此复核未新开 SLM/CCD/激光、未提交或推送；工作树在本轮开始前已存在其它并行未提交修改。本文件只记录判定与可复现的下一步，不把历史配对整形改善归因给某个 issue。

## 1. 当前判定

| Issue | 实时状态 | 有效性/算法结论 | 处理 |
|---|---|---|---|
| #234 | OPEN | 方向有效；`zernike_block`/结构化探针/回灌的特定收益未证明。实现只在分支 `feat/zernike-block-transfer`，离线 P0 `identifiability_rate.structured=0.0`、`matrix_ok=false`、方向门与强度双门无通过 checkpoint。历史 seed→final 改善不是该分支算法的消融。 | 保持 OPEN；新配对数据和双门 checkpoint 缺失。执行计划见 §3。 |
| #235 | OPEN | 有效但需缩窄主张。14 帧 `panel_gray` mismatch=0，实际相位替换后方向 cosine 仍约 0，因此“输入相位不一致是根因”已否定。零级 box centroid 有局部定位改进证据：历史双瓣 30 帧中 argmax 7 个不同像素、径向标准差 0.394 px；质心变体标准差 0.297–0.330 px；实现及测试已落地。该结果不是整形性能收益；`shift_phase`、可信物理 LUT 和独立 pupil（瞳面）-center 仍未闭环。 | 保持 OPEN；把完成的定位项从剩余主张中拆出，按 §3 作独立输入消融。 |
| #236 | OPEN | 损失口径问题有效，但损失重构的收益未证明。主线仍为旧三项；唯一消融未区分方向（中位 −0.043…+0.070），且基线复现 FAILED。未有通过方向门/强度门的 checkpoint。`lane-236-maskcv` 的单测只证明实现/默认兼容，不证明收益。 | 保持 OPEN；无满足前提的独立对比择优数据前不升默认、不合并。计划见 §3。 |
| #238 | OPEN | 灰度-相位标尺与像素串扰问题有效。run4 新数据的 32/48 px fold 估计分别 952.6 `[948,956]`、946.9 `[945,950]`，低于设备回读 993；但束功率漂移分别 +31.53%、−13.33%，整曲线 e-fit 残差高，±1 极小仍有系统分裂。提示不一致，未形成可冻结标定，更未证明逐像元 LUT/串扰核的算法收益。 | 保持 OPEN；不改 LUT/默认。需要低漂移重复/交错锚点数据与独立串扰几何；详见 `issue238_binary_gray_capture_plan_20261119.md` §6.5。 |
| #242 | CLOSED | 有效且已实现：campaign 安全门接入 transfer-desktop，定向测试 19 passed。 | 保持 CLOSED；不重复实现。 |
| #244 | CLOSED | 无效（当前单 SLM 2f 架构不具备第二调制器/独立第二像素网格，逐 ROI 对准前提不成立）。 | 已关闭，无需再采数据。 |
| #247 | OPEN | 物理轴失配/零级-目标级分离有效；但 D2′ 独立 held-out 三变体无可检出收益（axis−baseline `Δr²=−0.00355`，95% CI `[−0.00817,+0.00137]`，含 0），fresh checkpoint 方向门 0/3。可微零级校正未实现，无算法优化有效性证据。 | 保持 OPEN；不得重开 S7 或把零级校正升默认。先补几何常数并完成离线映射校验，见 `next_hw_plan_transfer_candidates_issue247_20261117.md` §任务 B 与 `hw_plan_consolidated_20261118.md` D2/D2′。 |

**总判定**：#235 的零级定位子项已显示可量化、有限的算法/估计器改进；#242 有效功能已完成；#244 已按无效关闭。其余四个仍为真实问题方向，但不能把“有效问题”偷换为“有效算法”：截至本复核，#234/#236/#238/#247 均无其 issue 声称的算法收益闭环证明，继续保持 OPEN。

**本窗口 §2.3 主门实测（补充证据）**：`validate_pairs` 上 `square_seed → square_final` 的配对主门**未通过且方向为负**——`rel_l2` 1.1575→1.2958（Δ=−0.1383，噪声地板 0.0424）、`energy_fraction` 0.4556→0.3919（Δ=−0.0637，噪声地板 0.0146），两项退化均超噪声地板 3 倍以上；目标核 CV 反而下降（0.3271→0.1728），即“CV 降了但能量被推向零级、没填到目标”。这直接以本窗口新数据印证了“改进未证明”，而非仅靠历史 reanalysis。数值与复现见 `hw_window_20261119_preregistration.md` §6，`replay` 因必开设备本轮未执行。

## 2. #238 run4 离线证据解释

- 原始证据目录：`data/experiments/issue238_binary_phase_20261119/run4_period_sweep/{period32,period48}/`。每段 48 个灰度层、每层 20 帧；12-bit；设备回读 2π 灰度 993；两段采集均正常释放 SLM/CCD 与设备锁。
- 复算工具：`scripts/slm/issue238_binary_fit.py`。离线 500 次 frame bootstrap；fit JSON/MD 暂写到 `C:/Users/zhangh/AppData/Local/Temp/opencode/issue238_period{32,48}_fit_audit.{json,md}`，原始 run 未覆盖。仓库内的逐项判读已写入 `issue238_binary_gray_capture_plan_20261119.md` §6.5。
- fold 两次点估计相差 5.7 灰度，提示可重复性；但各自 CI 只反映帧噪声，未含长时漂移及 aperture/阶次几何系统误差。理想 e-domain 拟合分别约 888.9/890.9，且 `rms(e0/e1)` 分别 `0.2359/0.1180` 与 `0.1948/0.0974`，故不能将理想拟合值采纳为标定。不得只选更靠近或更远离 993 的方法作结论。
- 无独立低漂移重复块前，结论限定为“检测到需要解释的响应/回读不一致”，不更新 `docs/slm/slm_lut.md` 工作值，也不宣称测出有效的逐像元 LUT。

## 3. 保持 OPEN 项的明确前置与后续计划

### #234 — structured / zernike block

1. 冻结候选提交（当前 `234f0741` 或其后续审查版）与 master baseline 的 SHA、forward/checkpoint 哈希、相位/模式/振幅集合、评分脚本与预处理哈希；实现仍留分支，不因本计划直接合入。
2. 先取得对独立探针通过的方向模型；用 signed Pearson `r≥0.50`、至少 10/14（不得 `abs(r)`）和冻结强度门 `δ=0.20` 共同放行。若任一门失败，停止算法收益对比择优。
3. 对同一固定目标做 baseline vs `alignment/full/zernike_block` 配对实测，预先冻结 holdout、core/window/mask、曝光、暗场 floor 与 rel-L2/energy-fraction 门槛；训练/回灌不得触碰 validation/holdout。报告所有尝试、逐 pass 与最差 pass，不筛最好结果。
4. `core CV≤0.10` 若仍为 issue 验收要求，需作为独立 endpoint 预注册，不能由模型 CV 替代。具体未决来源见 `transfer_strength_gate_preregistration_20261119.md` 与 `target_shape_ccd_acceptance_plan_20261118.md`。

### #235 — 实际相位 / shift / pupil（瞳面）-center

1. 不再把已证伪的 panel-gray mismatch 当成模型失准根因。将剩余变量拆成物理 LUT、`shift_phase`、pupil（瞳面）-center 三个独立消融，不同时改多个参数。
2. 每个固定控制相位保存命令相位、实际 `panel_gray`、灰度反解相位、SLM 回读的物理 2π 灰度、LUT/驱动哈希、phase sign、shift、面板中心、`--pupil-center-xy`、CCD ROI/曝光/暗场 floor。
3. 同一帧跑两条预测输入路径；以未参与拟合的结构化探针和 ABBA 重复为 holdout，报告方向签名相关、差分相对误差、逐帧区间。中心/定位评估另用独立中心真值报告 bias、SD、双瓣子集失败率；不得把 0.297–0.330 px 的历史帧间 SD 当作绝对准确度。
4. 采集先决与具体记录字段需按项目硬件纪律写入当次 run plan；本计划不授权新设备操作。

### #236 — loss ablation

1. 数据前置：一个同时通过 signed direction 与 strength 双门的 frozen checkpoint；Mono12 中段曝光、匹配暗场和冻结 ROI/core mask。无此 checkpoint 时现有损失对比择优不具辨识力。
2. 离线阶段先在开发验证 split 预登记权重尺度，依次做 baseline、masked CV、Sobel、高频、log、JVP、异方差/Huber 的单项消融，另报交互效应；一次只加一项，不用 holdout 调权。
3. JVP 必须有结构化 ±A 的配对响应帧（ABBA）；径向对称离焦不能替代极性/方向观测。独立 holdout 报 signed difference correlation、符号一致率、rel-L2 与双门通过率，并与同 checkpoint 基线成对比较。
4. 历史 measured-vs-target common_replay 只作背景，不支持 #236 子项归因；统一其统计口径，严格遵循 `historical_measured_shape_metrics_reanalysis_20261119.md` 的 floor/饱和限制。

### #247 — axis mapping / zero-order correction

1. 先离线找出并登记 panel pitch、CCD pitch、焦距/放大率常数；用 11-17 tilt 数据核验映射方向、尺度与角度的定义（88.544° 与后续旋转变体须明确不是同一参数口径）。缺常数则不写入 `FrozenDefocusForward.ramp_matrix`。
2. D2′ 已判“未检出收益”，不得用原三变体结果放行闭环。实现默认关闭，先做合成模板恢复、无校准逐位兼容和梯度/批量一致性单测；离线 held-out CCD replay 只有超出预注册最小效应并且区间排除 0 后才能考虑后续。
3. 在 `target_shape_ccd_acceptance_plan_20261118.md` 冻结目标 rel-L2 上界、energy-fraction 下界、background floor、零级坐标后，方可申请新鲜配对实测。任何后续设备采集是单独授权事项。

## 4. 停止线

- 现有证据可以支持“问题有效”“局部估计器改进”“改进未证明”三种彼此独立的陈述；禁止将其合并为“已验收算法优化”。
- 不因缺数据伪造离线证据、不从历史 `floor=0` 数据重新挑阈值、不把新采数据的帧 bootstrap 当系统不确定度、不以 `abs(Pearson)` 代替有符号方向判据。
- 只有当问题前提被物理/架构事实否定时才关闭为无效；#244 已满足并关闭。缺可归因证据的有效问题保持 OPEN，并使用本节明确的采集/验证门，不做外部 issue 状态变更。
