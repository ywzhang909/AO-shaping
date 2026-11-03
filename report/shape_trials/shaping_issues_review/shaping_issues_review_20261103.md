# 光束整形报告综述：光路失配、物理设备限制与算法缺陷（2026-11-03）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线（只读汇总 `report/shape_trials/` 下已有报告；不产生新的硬件数据）
> **说明**: 对 SPGD、黑盒、GS、model-in-loop、多形状/迁移、物理 refine 与 FourierGSNet 系列整形报告的问题汇总，按光路失配、物理设备受限、算法缺陷、测量与验收口径四类归并
<!-- provenance:end -->

## 0 结论摘要

1. **最大的系统性问题是模型与实测脱节**。多份报告中，模型预测增益比实测高约 100 倍（physical_refine：预测 0.04–0.72，实测 0.001–0.005）；模型只解释实测约 4% 的方差（square_1000，r≈0.21）；Jacobian 对离焦与光滑相位几乎无响应（余弦 ≈ 0）。模型不能作为验收依据。
2. **零级与光路几何是反复出现的根因**。零级位置在不同报告中抖动 8–39 px；CCD 离焦（D=+0.559 m⁻¹，f≈1789 mm）使远场不在焦面；载波与目标框的相对位置在多形状场景下直接把目标推出采样窗。
3. **照明与亮度漂移是主导物理干扰**。同一平场亮度摆动约 86×（slm_pib）；轮内参考峰值下降 30%（no_gs）；激光功率中途从 0.7 调到 1.0（counts/ms 4.36→31.67）。多数报告只做了缓解（同曝光重放、漂移轮剔除），未消除根因。
4. **验收口径与目标函数不匹配**。只优化均匀度（CV）时，光斑被匀化但不成形（填充率 ≤10.7%）；综合目标与 CV 指标冲突；退化解（SPGD 仿真坍缩到 1 px，EE≈0.998）能拿到“成功”的 EE 分数。
5. **多份报告的结论已被后续修复推翻**，本综述只保留仍有效的问题，并注明状态。

## 1 范围与方法

- 覆盖目录：`report/shape_trials/` 下 GS 系列（gs_refine、gs_refine_fwdmodel、gs_refine_zernike、gs_defocus_sim、gs_golden_reference、hw_gs_square_scans、cold_start）、SPGD/黑盒/学习类（spgd_drift_cancel_loop、blackbox_path、gui_closed_loop_shaping、no_gs_ccd_moved）、model-in-loop 系列（model_in_loop、model_in_loop_cv、defocus_differentiable、hw_defocus_zernike、experiment_summary）、多形状/迁移/物理 refine 系列（multishape_capability、multishape_scale、multishape1200、square1200、tri_plane、axial_plane、sim_vs_hw、mask_closed_loop、physical_refine）；以及 `report/benchmarks/device_less_full/` 的 SPGD/GS 仿真指标。
- 每条问题给出：分类、一句话描述、证据位置（仓库相对路径与关键数字）、报告自身给出的状态。
- **局限**：条目来自四组只读摘录，部分长报告正文被截断，标注“待核实”的条目未逐行复核；数字以原报告为准。FourierGSNet 系列四份报告的摘录仍在生成中，未纳入本版（见 §6）。

分类：**A** 光路失配；**B** 物理设备受限；**C** 算法缺陷；**D** 测量与验收口径。状态：已修复 / 已缓解 / 未解决 / 待硬件验证。

## 2 按报告归类的问题

### 2.1 GS 系列

- [A] 零级约距目标 12 px，位移标度 ≈ 7400/P；面板与 CCD 轴互换 — `report/shape_trials/gs_refine/zero_order_blaze_20261015.md` — 未解决；轴互换待硬件验证。
- [A] 面板中心 (960,690)、半径 450 未经光路验证 — `gs_refine_live_20261012.md` — 待硬件验证。
- [A] 液晶 2π 取值 993 而非 1023；各向异性 1.60× 旧模型；仿真场 16.6 mm 对相机 0.55 mm（约 30× 失配）— `gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md` — 未解决。
- [A] 硬件结果与仿真方向相反（仿真 GS 0.8543 对平场 0.7229，硬件 CV 0.723 近似平场）— `gs_golden_reference/gs_golden_reference.md` — 未解决。
- [A] 实测位移系数 α≈0.48，预测 1.00 — `hw_gs_square_scans/hw_gs_square_scans_20261023.md` — 未解决。
- [B] 零级重定位抖动 8 px（Zernike 模式 39 px）；9/50 次 settle 超时 — `hw_gs_square_scans_20261023.md` — 未解决。
- [B] 曝光 0.1 ms 饱和，帧峰值仅 4–7/255（低信号）— `gs_refine_live_20261012.md`、`gs_uniformity_live_20261015.md` — 已缓解（重调曝光），信号不足未解决。
- [C] SPGD 细化增益 0.0043–0.0137，接近或低于噪声底 — `gs_refine_live_20261012.md`、`gui_closed_loop_shaping_20261014.md` — 未解决。
- [C] GS 把热点分裂成多个亮点与暗孔，CV 0.6–0.8，无实心方块 — `gs_issue_live_20261015.md`（IoU 0.068–0.098） — 未解决。
- [C] 仅纯 CV 循环时 CV 0.72313→0.68400，但强度下降 2.3%；四档平滑全部为负 — `gs_uniformity_live_20261015.md` — 未解决。
- [C] 模型交叉验证乐观 0.2631；单种子残差增益 +0.00339 无重复 — `gsnet_hybrid_offline_20261013.md` — 未解决。
- [C] `best_phase` 返回了扰动后的获胜探针而非最佳相位（0.2349→0.0826 rad） — `online_20261010.md` — 已修复（#140）。
- [D] 早期结果在错误图像上计算，已发勘误 — `gs_issue_live_20261015.md` — 已修复（勘误）。
- [D] 自适应框使分数膨胀（43→104 px，最优相位 3.87 vs 0.34 rad） — `gs_defocus_sim/gs_defocus_sim_20261019.md` — 已修复（改为固定框）。
- [D] `ee` 与 `PIB` 为重复指标；单次测量无误差棒 — `gs_issue_live_20261015.md` — 未解决。
- [D] 冷启动 plane 6/6 接受、CV 2.512→0.481，但 EE 0.241→0.173；local_continuation 0/4 — `cold_start/cold_start_20261026.md` — 未解决。
- [C] 相位-光斑模型跨运行 `fit_rms` 漂移 0.008–0.012 rad，根因未知 — `gs_refine_zernike_hardware_plan.md` — 未解决。
- [D] Zernike 均匀度 PIB 差 0.0161 仅约 2 倍抖动（0.0072–0.0109） — `gs_refine_zernike/gs_refine_zernike_uniformity.md` — 未解决。

### 2.2 SPGD、黑盒与学习类

- [C] 照明漂移下 SPGD 四种协议全部不收敛（dec 0.41–0.59，阈 >0.55） — `spgd_drift_cancel_loop/spgd_drift_cancel_loop_20261022.md` — 未解决。
- [C] 噪声门把 29–30/32 个梯度估计置零，更新被饿死 — 同上 — 未解决。
- [B] 同相位噪声底 σ(J)=0.0306（n=24）限制可分辨性 — 同上 — 未解决（物理约束）。
- [B] 噪声底 dJ_noise=0.00040，SPGD 最小可分辨扰动 Δa≈0.2 rad（SNR 2.77）；0.05/0.1 rad 的 SNR 仅 0.27/0.69 — `slm_pib_heuristic_hw/measure_shape_sensitivity/sensitivity.md` — 未解决（定量解释“小步全部随机游走”）。
- [B] 光照漂移 86× 双态摆动（曝光需求 191.5→173.6 ms）— `slm_pib_heuristic_hw/slm_pib_heuristic_hw_report/report.md` — 已缓解（剔除漂移轮，中位 ±25%）。
- [B] 每算法 82–116 次设备加载，仅为指示性对比 — 同上 — 未解决（预算限制）。
- [B] 轮内参考峰值 13.856→9.732 counts/ms（−30%），曝光 7.650→13.564 ms — `no_gs_ccd_moved/live_20261027.md` — 未解决。
- [A] 两轴载波矩阵首行 [563.6, 7349.2]：相机 x 主要由面板 y 驱动；非对角占主导且轮内变化 10–25% — `no_gs_ccd_moved/analysis_20261027.md`（F3/F4） — 未解决。
- [A] 期望 2f 映射 (960,600) 与实测 CCD 中心 (1296,991) 相差 200 px 零级误差 — 同上（F5） — 未解决。
- [A] 光斑只落在 CCD 1/4 区域，1200 px 目标不可行 — 同上（F6） — 未解决。
- [B] 零级能量占全幅 33.8%（ROI 内 20.5%） — 同上（F9） — 未解决。
- [C] 拟合模型全窗 NCC 0.735，质心误差 19.3 px — 同上（F10/F11） — 未解决。
- [D] A–B–B–A 方向性证据为零；独立重放未执行 — `no_gs_ccd_moved/live_20261027.md` — 未解决。
- [C] 闭环仅边际改善（objective −0.035450→−0.029450），接受率 41.67% — 同上 — 未解决。
- [C] spgd-sim 在 square/gaussian/circle 上坍缩为 1 px（fill_ratio 0.004/0.003/0.001），EE≈0.998 看似成功 — `report/benchmarks/device_less_full/gif/spgd-sim_*/spgd-sim_*_metrics.md` — 未解决。
- [C] 数字孪生需补杂散底板 B=0.005 ADU/px 才能复现 bench 亮度耦合 — `spgd_drift_cancel_loop_20261022.md` — 已缓解。
- [D] 固定曝光 r1/r2 36× 曝光差会把亮度差读成质量差 — `blackbox_path/live_20261029.md` — 已缓解（同曝光重放）。
- [C] 模型解释能量 1.02–1.11× 但残余结构 CV 差 2.1–3.3×；H1 相关中位 0.70 未达标 — `blackbox_path/analysis_20261029.md`（F1）、`live_20261029.md` — 未解决。
- [C] 锚点把实测除以 counts_scale=16，与模型原生计数率不一致 — `analysis_20261029.md`（D1） — 已修复。
- [C] `fit.valid()` 用 8-bit 常量 `peak < 250`，12-bit 帧被静默丢弃 — 同上（D2） — 已修复。
- [D] 综合目标以窗内能量换核心平整（CV −6~19%，EE −1.8~3.5%） — 同上（F2） — 未解决。
- [C] 平面起始相位为散斑、不可用；GS 相位可用 — 同上（F3） — 已缓解。
- [C] GUI 三个开环控件在 2f 系统产不出可用方斑，已换闭环 — `gui_closed_loop_shaping_20261014.md` — 已缓解。
- [B] 液晶未完全弛豫，settle 判据提前通过，闭环分数系统性高于重放 — 同上 — 已缓解（交错重放）。
- [A] `find_beam_on_slm` 在散斑下不可靠 — 标定层记录（tools/slm/calibration/calibration.py，2026-10-19） — 未解决。

### 2.3 Model-in-loop 与离焦系列

- [B] 曝光非线性：0.03→0.06 ms 斜率 1.42（理想 4） — `model_in_loop/model_in_loop_live_20261013.md` — 未解决。
- [A] 面板入射位置作废：平场 47→15→2/255 漂移 — `model_in_loop_live_20261015.md` — 未解决（需重扫）。
- [A] 面板中心 (960,670) 作废，新值 (960,720) — `model_in_loop_live_20261019.md` — 已修复。
- [C] 全频随机相位验证相关 0.11–0.28，未过门槛 — 同上 — 未解决。
- [A] 1/f 重建 +0.5875 m⁻¹ 与独立细扫 +0.6667 m⁻¹ 差 0.079 m⁻¹ — 同上 — 未解决。
- [D] 提高匀化权重后 CV 0.8027→0.8989、EE 0.6919→0.5177，两者不同向 — `model_in_loop_uniformity_live_20261016.md` — 未解决。
- [D] rel_sigma 判据峰值整数化导致 MAD=0 — 同上 — 未解决。
- [B] 修复半径 edge/2 与曝光探测截断缺陷 — 同上 — 已修复。
- [C] A/B 两变体均 bake_off_rejected；sweep 几何相关 0.7188 — `model_in_loop_hw_ab_20261021.md` — 未解决。
- [A/C] 落盘相位为零相位平场，未真正写入整形相位 — `model_in_loop_record_20261021_172309.md` — 未解决。
- [A] 聚焦后 CV 0.6030→1.1947 恶化；固定 ROI 对 63 px 质心横移敏感 — `physical_forward_defocus_live_20261022.md` — 未解决。
- [D] 模型采样 U/C 的 CV 绝对误差 0.0975/0.0888，质心误差 7.68/4.64 px，均超门槛 — `physical_forward_sampling_20261022_r2.md` — 未解决。
- [C] 5 折 Δ 折间 sd 0.0561，sign-flip p=0.0625，不显著 — `model_in_loop_cv/model_in_loop_cv.md` — 未解决。
- [C] 模型幅度与实测相差约 29%；轴映射 ramp_matrix 与现场标定互为转置 — `defocus_differentiable/defocus_differentiable_20261023.md` — 未解决 / 待硬件验证。
- [C] 模型 CV 随重拟合单调下降，实测 CV 不动 — `defocus_differentiable_20261023_r2.md` — 未解决。
- [A] 焦距历史摆动 1702.1 / 1789 / 1523.8 mm — `hw_defocus_zernike/hw_defocus_zernike_20261021.md` — 已缓解（26 组独立重标定）。
- [D] SPGD 后 CV 差 0.0113 与噪声 σ 0.00543 同量级，单 seed — `hw_defocus_zernike_20261022.md` — 未解决。
- [B] 驱动板 4 轮不应答，仅一轮判 ok — `experiment_summary_20261022.md` — 待硬件验证。

### 2.4 多形状、迁移与 physical refine 系列

- [A] 阶次重叠 403 px，与 1200/600 目标框冲突；零级落入目标盒 — `multishape_scale/multishape_scale_planning_20261028.md` §11 — 未解决（已登记为已知限制）。
- [A] 迁移相位 float64 与传播网格 float32 不一致；已统一 — `multishape_capability/physical_multishape_reproduction_20261023.md` §13.1 — 已修复。
- [A] period=4 面板像素使载波位移 1865 px，采样窗 220 px，目标在窗外 8.5× — 同上 §14.1 — 已修复（preflight 两级检查，周期 ≥16 px）。
- [A] 相位→面板坐标映射不干净：锚点仿射相对 L2 0.1875 — `multishape_scale/square_1000_diff_abba_20261030.md` §12.3(a) — 未解决。
- [A] 零级系统性偏移 (−5.6, +0.15) px — `physical_refine/physical_refine_results_20261102.md` §3.5 — 未解决。
- [A] 评分窗与记录零级不一致约 37 px：effalign 重算 objective 2.0495 vs 记录 1.9649 — `physical_refine_results_20261103.md`（同一 Sim2f 运行） — 未解决。
- [C] 模型只解释实测约 4% 方差（r≈0.2083）— `square_1000_diff_abba_20261030.md` §12.3 — 未解决。
- [C] Jacobian 方向余弦全面不达标（门槛 ≥0.5）：RX +0.2699、RY +0.4436 — 同上 §13.2 — 未解决（结构性）。
- [C] 对光滑二次相位（离焦）无响应：PV 1→50 rad 时预测变化范数仅 0.03→0.42 — 同上 §13.8 — 未解决。
- [C] 幅度失配 36–66×：预测半差分范数 6.43/8.72，实测 423/317 — 同上 §13.4 — 未解决。
- [D] 像差重拟合救回对角 r 到 0.36，但 Jacobian 方向同步恶化（RY −0.2254） — 同上 §13.6 — 未解决。
- [C] 综合目标三角形下降却 CV 恶化 1.83%（0.251872→0.256491）；接受判据与匀化脱钩 — `physical_multishape_reproduction_20261023.md` §13.1 — 已缓解（补纯 CV 目标）。
- [C] 纯模型 CV² 0.126064→0.087903，实测 CCD CV 0.366→0.380 — 同上 §13.3 — 未解决。
- [B] 墙钟 gap 占 46–60%，单次采集约 2.5 s 无分段计时 — `multishape_scale_live_20261028.md` §5.8 — 已缓解（加 --profile-timing）。
- [B] `np.savez_compressed` 单次 1155 ms；磁盘余量 64.3 GB 不足以存未压缩帧 — 同上 §5.8.2 — 未解决。
- [B] `iterations.pkl` O(n²) 写放大（约 224 GB 写出）— `multishape_scale_wallclock_20261028.md` §5 — 已修复。
- [B] SLM `open()` 偶发 `SantecError -10003`（约 1/20），需断电复位；相机 `InvalidAccess -1005` — `live_20261029_device_session.md` §6 — 已缓解（断电处置，稳定性待硬件验证）。
- [B] 激光功率中途 0.7→1.0，counts/ms 4.36→31.67（8.6×），weak_path 门禁漏掉 23% 差额 — 同上 §9.1/§9.5 — 未解决。
- [D] square 同 panel_gray、同曝光下三次采集峰值 +57%（123→192.7），主瓣换位 63 px — 同上 §8.2 — 未解决。
- [C] filled_fraction ≤10.7%，objective(cv) 只匀化不成形 — `live_20261029_device_session.md` §5.8/§8.5；`next_round_20261029.md` — 未解决。
- [C] 3a jacobian_guard 把 phase_rms_cap 压到 0.02，读数无信息 — `live_20261029_device_session.md` §7 — 未解决。
- [D] 固定 2 pp 接受门槛不足：600 px 常规段拒绝、dense 段接受 — `dual_optimization/experiment_plan_20261027.md` — 未解决。
- [C] 闭环增益随块衰减，首块后 ≤0.46σ — `multishape_capability/shape_comparison_20261023.md` — 未解决。
- [A] 轴向 f_eq 311.32 mm 与 125 mm 冲突（2.5×）；闭式校正律未在线测试 — `axial_move_sim/axial_move_sim_20261029.md` — 待复核。
- [C] 轴向校正最优值落在扫描边界，可辨识性受限 — `axial_plane/axial_plane_20261026.md` — 未解决。
- [C] 阶次扫描实测 pib 0.08264 vs 仿真 0.0037–0.0048；V1 孪生缺零级效率、面板表面、LUT、LCOS 响应、漂移 — `sim_vs_hw/compare_v1_v2_20261023.md`、`sim_vs_hw_20261022.md` — 未解决。
- [C] 仅离线孪生验证，net_open ≈ 平场（无掩模先验）；two_spot 在 nominal 下假通过 — `mask_closed_loop/mask_closed_loop_20261031.md` — 未解决。
- [C] 14 项 Zernike 适配器在高阶漂移下 Δcorr −0.018，对几何漂移无效 — `fast_recal_adapter/fast_recal_adapter.md` — 已验证无效。

## 3 跨报告反复出现的问题

| 主题 | 类别 | 代表证据 | 状态 |
|------|------|----------|------|
| 模型-实测巨大落差（增益约 100× 乐观，r²≈0.04，Jacobian 对离焦失效） | C | square_1000 §12–13；physical_refine §3.3 | 未解决 |
| 零级位置抖动与几何失配（8–39 px 抖动，200 px 零级误差，α≈0.48） | A/B | hw_gs_square_scans；no_gs F5；zero_order_blaze | 未解决 |
| CCD 离焦（D=+0.559 m⁻¹，f≈1789 mm）与中心校正引入 −5.5 px/mm 偏移 | A | 标定层；tri_plane；physical_refine | 未解决 |
| 照明与亮度漂移（86×、−30%、功率跳变 8.6×） | B | slm_pib；no_gs；live_20261029 | 已缓解，根因未消除 |
| 噪声底与可分辨扰动（Δa ≥ 0.2 rad；J 噪声 0.00295–0.0306） | B/C | sensitivity.md；spgd_drift；no_gs | 未解决 |
| 目标函数只匀化不成形（填充 ≤10.7%；退化解 1 px） | C | live_20261029；spgd-sim 指标 | 未解决 |
| 验收口径不闭环（综合目标 ≠ CV；EE 与 PIB 重复；无重复与误差棒） | D | multishape_scale；gs_issue；dual_optimization | 未解决 |
| 单种子、单次测量、无重复（圆形跨小时漂移 7–9 pp） | D | multishape_scale analysis G6；sim_metrics_waves | 未解决 |
| 闭环增益随块衰减、接受门槛不稳 | C/D | shape_comparison；tri_plane | 未解决 |
| 设备 IO 与采集限制（SLM open −10003、磁盘、墙钟 gap） | B | live_20261029；wallclock | 部分缓解 |

## 4 对整形效果影响的排序

1. **模型-实测失配**：所有模型驱动的算法（model-in-loop、GS 增益预测、Jacobian 方向）都依赖它。在修正前，模型只能给方向提示，不能给步长或接受判据。
2. **零级与光路几何**：直接决定光斑能否落进目标窗、能量能否进入目标区域。多形状与尺度实验已多次被它卡住。
3. **照明与亮度漂移**：使单次曝光下的比较不可靠，是多个“接受/拒绝”结论不稳定的直接原因。
4. **目标函数与验收口径**：匀化目标不能保证成形，退化解在 EE 上得分很高。
5. **设备 IO 与墙钟**：限制了可重复实验次数，使统计结论难以成立，但不直接改变光斑质量。

## 5 建议的下一步（按优先级）

1. 把零级定位与面板/CCD 几何标定固化为每轮的前置检查，并统一零级记录与评分窗（当前偏差约 37 px）。
2. 在每轮验收中加入同曝光参考帧与漂移监测，把功率与亮度漂移作为门禁，而不是事后剔除。
3. 把验收目标改为“成形 + 匀化”的复合指标，并固定接受门槛；退化解（面积坍缩、填充率过低）直接拒绝。
4. 先用实测数据对模型做幅度与离焦响应标定，再用它给方向和步长；在失配修正前不以模型增益作为接受依据。
5. 对每个候选做至少两次独立重放，记录误差棒后再下结论。

## 6 待补充

- FourierGSNet 系列（`report/fouriergsnet_pipeline/`、`report/fouriergsnet_sim/`）四份报告的问题摘录仍在生成中，待返回后补入 §2 与 §3。
- 摘录中标注“待核实”的长报告段落未逐行复核；引用的数字以原报告为准。
