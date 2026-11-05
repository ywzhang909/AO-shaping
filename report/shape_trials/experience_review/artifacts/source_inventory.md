# 整形经验报告来源清单

<!-- provenance:start -->
> **生成脚本**: [`scripts/ml/shaping_experience_inventory.py`](../../../../scripts/ml/shaping_experience_inventory.py)
> **复现命令**: `python scripts/ml/shaping_experience_inventory.py`
> **运行环境**: 离线只读扫描 report/ 下整形相关 Markdown；无设备
> **说明**: 169 份报告的标题、行数、SHA-256 与逐行证据摘录；正文主报告为人工撰写。
<!-- provenance:end -->

由脚本自动生成；哈希与逐行证据见 `source_inventory.json`。

| 报告 | 行数 | SHA-256 前 12 位 |
|---|---:|---|
| [Beam-Shaping Benchmark (simulation)](../../../benchmarks/device_less_full/beam_shaping_benchmark_metrics.md) | 32 | `978cf189f1d8` |
| [基准测试: gs × circle](../../../benchmarks/device_less_full/gif/gs_circle/gs_circle_metrics.md) | 16 | `d22271215bae` |
| [Benchmark: gs × gaussian](../../../benchmarks/device_less_full/gif/gs_gaussian/gs_gaussian_metrics.md) | 16 | `55d070b41dba` |
| [Benchmark: gs × square](../../../benchmarks/device_less_full/gif/gs_square/gs_square_metrics.md) | 16 | `c5d968ad4d61` |
| [Benchmark: spgd-sim × circle](../../../benchmarks/device_less_full/gif/spgd-sim_circle/spgd-sim_circle_metrics.md) | 16 | `af1f83f46cb6` |
| [Benchmark: spgd-sim × gaussian](../../../benchmarks/device_less_full/gif/spgd-sim_gaussian/spgd-sim_gaussian_metrics.md) | 16 | `511ea3a98edd` |
| [Benchmark: spgd-sim × square](../../../benchmarks/device_less_full/gif/spgd-sim_square/spgd-sim_square_metrics.md) | 16 | `ce74e824eb4e` |
| [diff-beam 固定边长方形整形 — 硬件经验记录 (Santec SLM200 + 大恒 CCD)](../../../diff_beam/diff_beam_frame_analysis/README.md) | 258 | `d367719649ab` |
| [FourierGSNet 管线集成测试报告](../../../fouriergsnet_pipeline/fouriergsnet_pipeline_report/report.md) | 135 | `4d7ef536d527` |
| [slm-gsnet 实验方形整形运行报告 (2026-10-01)](../../../fouriergsnet_pipeline/hardware_run_20261001.md) | 227 | `609908d46051` |
| [GS 家族整形下一步尝试方案（slm-gs-refine × FourierGSNet）](../../../fouriergsnet_pipeline/next_attempt_plan.md) | 146 | `0855667b9504` |
| [FourierGSNet 离线训练报告](../../../fouriergsnet_pipeline/offline_training/report.md) | 164 | `e6e46f61cc58` |
| [FourierGSNet 硬件验证 TODO](../../../fouriergsnet_pipeline/TODO.md) | 143 | `66d1ca054201` |
| [FourierGSNet 湍流仿真矩阵报告](../../../fouriergsnet_sim/fouriergsnet_sim_report/report.md) | 450 | `5b7edd204620` |
| [hwdataset 语料分布与特征分析](../../../hwdataset_corpus/hwdataset_corpus_report/report.md) | 414 | `5f608aff9d8c` |
| [逆向光束整形模型详解 (PhasorUNet)](../../../inverse_shaping_unet/inverse_shaping_model_report/model.md) | 287 | `9680e53bdc68` |
| [逆向光束整形 U-Net —— 首个真实语料结果](../../../inverse_shaping_unet/inverse_shaping_report/report.md) | 145 | `58af812e1aa6` |
| [Loss x algorithm comparison (offline sim bench)](../../../loss_algorithms_smoke/loss_algorithms/report.md) | 46 | `1e86e93187cf` |
| [物理预测模型改进尝试：逐项贡献、术语与图解读](../../../loss_defects/forward_search_report/forward_search_report.md) | 340 | `06d3523e598a` |
| [逆向整形研究报告（正向 + 反向 pred vs true）](../../../loss_defects/inverse_design_report/inverse_design_report.md) | 345 | `d5cd38d26429` |
| [Process log — forward model & inverse shaping optimisation](../../../loss_defects/PROCESS.md) | 1376 | `3009a250cedf` |
| [Forward-model / loss defect investigation](../../../loss_defects/README.md) | 373 | `c84a51dc6119` |
| [MiiCam 自动曝光算法实验报告](../../../miicam_auto_exposure/miicam_auto_exposure_probe/report.md) | 296 | `63891cd4a6c9` |
| [SAC / Robust-RL 训练运行分析报告](../../../models_analysis/models_report/report.md) | 191 | `37edb3027a2b` |
| `models_analysis/models_report/tmp_sac_run_showcase.md` — tmp_sac_run 临时调试家族最佳 run 成果展示（被 `.gitignore` 匹配，不入库） | 99 | `b7afe559337e` |
| [传统AO仿真环境Reward问题分析与解决](../../../models_analysis/reward_analysis.md) | 170 | `0f6669ecdaa4` |
| [闭环整形目标函数：物理误差感知 Loss 的取舍记录](../../../pib_loss_terms/README.md) | 155 | `62ef810aa5c1` |
| [PIB 优化器功能报告（完整增强版）](../../../pib_optimizer_functional_report.md) | 208 | `e45ff7f3dd85` |
| [SLM 轴向移动的模型代价与再辨识（仿真，2026-10-29）](../../../shape_trials/axial_move_sim/axial_move_sim_20261029.md) | 233 | `44ed6e7a5584` |
| [SLM 偏离前焦面的物理预测模型校正（2026-10-26）](../../../shape_trials/axial_plane/axial_plane_20261026.md) | 95 | `7d99cfc17127` |
| [黑盒路径：离线诊断与修复（2026-10-29）](../../../shape_trials/blackbox_path/analysis_20261029.md) | 211 | `923e0b89b92a` |
| [黑盒路径重放：同曝光物理采集结果（2026-10-29）](../../../shape_trials/blackbox_path/artifacts/live_acquisition_20261029.md) | 165 | `850969ccbe41` |
| [黑盒路径：两次会话的现场记录与验收复核（2026-10-29）](../../../shape_trials/blackbox_path/live_20261029.md) | 137 | `5daada3220ed` |
| [新光路黑盒适配：实验规划（2026-10-29）](../../../shape_trials/blackbox_path/planning_20261029.md) | 67 | `f25420dfc232` |
| [Transfer-chain radiometric units: device-free reproduction (2026-10-29)](../../../shape_trials/blackbox_path/sim_units_20261029.md) | 78 | `0e686002eb92` |
| [无历史权重与无 GS 的冷启动整形实验（2026-10-26）](../../../shape_trials/cold_start/cold_start_20261026.md) | 228 | `37156ebd4d02` |
| [冷启动物理预测模型 CV 误差的曝光偏置诊断（2026-10-26）](../../../shape_trials/cold_start/cv_diagnosis_20261026.md) | 89 | `39fd76868258` |
| [离焦光路下的可微整形（2026-10-23）](../../../shape_trials/defocus_differentiable/defocus_differentiable_20261023.md) | 150 | `79524d5bf52b` |
| [光路变化后的重标定（2026-10-23，第二轮）](../../../shape_trials/defocus_differentiable/defocus_differentiable_20261023_r2.md) | 106 | `2c3b02252a78` |
| [下一轮器件响应矫正与多形状可微整形方案（2026-10-23）](../../../shape_trials/defocus_differentiable/multishape_forward_plan_20261023.md) | 237 | `f13c8c5f4524` |
| [离焦光路下的可微整形测试方案（2026-10-23）](../../../shape_trials/defocus_differentiable_test_plan_20261023.md) | 230 | `14551657bafc` |
| [Experiment Plan: Transfer Learning + Closed-Loop Dual Optimization on Physical Hardware](../../../shape_trials/dual_optimization/experiment_plan_20261027.md) | 495 | `9e1b05107c32` |
| [形状目标整形实验总结报告（2026-10-09 ~ 2026-10-19）](../../../shape_trials/experiment_summary_20261016.md) | 439 | `bd9cb07285fb` |
| [跨系列联合分析与下一步实验计划（2026-10-22）](../../../shape_trials/experiment_summary_20261022.md) | 338 | `a79f83ce851a` |
| [光路变化后如何最快适配：少参数在线微调 (2026-10-13)](../../../shape_trials/fast_recal_adapter/fast_recal_adapter.md) | 322 | `b35b6f32877d` |
| [超视场开环预测、D 最优主动探测与物理约束残差拟合（仿真，2026-11-01）](../../../shape_trials/fov_generalization/fov_generalization_20261101.md) | 451 | `d12657897782` |
| [CCD 焦后离焦下的 GS 匀化仿真（2026-10-19）](../../../shape_trials/gs_defocus_sim/gs_defocus_sim_20261019.md) | 455 | `cc83e89df4a9` |
| [GS 金标参照：模拟上界与平相位对照](../../../shape_trials/gs_golden_reference/gs_golden_reference.md) | 70 | `a8ef4eae5a0a` |
| [GS 方法实验对照与形状验收预检（2026-10-15）](../../../shape_trials/gs_refine/gs_issue_live_20261015.md) | 224 | `cfe2decb3cbe` |
| [GS 重构后真实设备调试（2026-10-12）](../../../shape_trials/gs_refine/gs_refine_live_20261012.md) | 118 | `1dd17a40c73f` |
| [GS → Zernike → SPGD 匀化实验：阶数扫描已就绪，请求实验执行](../../../shape_trials/gs_refine/gs_refine_zernike_hardware_plan.issue.md) | 98 | `54955ea74b54` |
| [GS → Zernike → SPGD 匀化实验：实验测试计划](../../../shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md) | 339 | `b8765e130143` |
| [GS 相位匀化与剔零级均匀度实验试验（2026-10-15）](../../../shape_trials/gs_refine/gs_uniformity_live_20261015.md) | 106 | `f57cfa62f77f` |
| [GSNet 混合瞳面相位离线验证（2026-10-13）](../../../shape_trials/gs_refine/gsnet_hybrid_offline_20261013.md) | 112 | `4e1ae8f3f5bd` |
| [2026-10-10 在线方斑整形复测](../../../shape_trials/gs_refine/online_20261010.md) | 229 | `3852e01a8dab` |
| [仿真指标与守卫五波交付（`feat/sim-metrics-waves`）](../../../shape_trials/gs_refine/sim_metrics_waves.md) | 137 | `cfa2377a2c5e` |
| [SLM 整形的实测离轴载波（2026-10-15）](../../../shape_trials/gs_refine/zero_order_blaze_20261015.md) | 42 | `e51481d568d6` |
| [slm-gs-refine 物理预测模型标定 (2026-10-13)](../../../shape_trials/gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md) | 260 | `4c2670f89dd4` |
| [GS-refine Zernike 阶数扫描 (uniformity)](../../../shape_trials/gs_refine_zernike/gs_refine_zernike_uniformity.md) | 167 | `8057c671b5a2` |
| [GUI 闭环整形功能实验验证（2026-10-14）](../../../shape_trials/gui_closed_loop_shaping/gui_closed_loop_shaping_20261014.md) | 317 | `38520401a891` |
| [`hw_defocus_zernike/` 的输入契约（按轮归档）](../../../shape_trials/hw_defocus_zernike/artifacts/README.md) | 119 | `bb165a6f10a9` |
| [硬件离焦扫描 + 多阶泽尼克匀化测试报告（2026-10-20）](../../../shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261020.md) | 812 | `a320b0c303c9` |
| [硬件离焦扫描 + 多阶泽尼克匀化测试报告（2026-10-21）](../../../shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261021.md) | 869 | `54a4168beda4` |
| [硬件离焦扫描 + 多阶泽尼克匀化测试报告（2026-10-22）](../../../shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261022.md) | 915 | `785358bf4c98` |
| [GS 方形光斑三段扫描：边长 / Zernike 阶数 / 闪耀光栅（2026-10-23）](../../../shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md) | 935 | `588daf43d1cd` |
| [三日报告诊断与局部物理预测模型可微整形（2026-11-03）](../../../shape_trials/local_forward/local_forward_20261103.md) | 87 | `3fa0c433cfcd` |
| [任意形状掩模闭环整形: 标定孪生离线验证 (2026-10-31)](../../../shape_trials/mask_closed_loop/mask_closed_loop_20261031.md) | 194 | `42dddf9b1f2e` |
| [model-in-loop 实验系统 A/B campaign 报告 (20261021)](../../../shape_trials/model_in_loop/model_in_loop_hw_ab_20261021.md) | 311 | `4903054e86c6` |
| [物理预测模型闭环整形实验记录（2026-10-13）](../../../shape_trials/model_in_loop/model_in_loop_live_20261013.md) | 765 | `7523a946d819` |
| [Model-in-loop 实验复核（2026-10-14）](../../../shape_trials/model_in_loop/model_in_loop_live_20261014.md) | 191 | `95e9ad4baf04` |
| [Model-in-loop 实验预检（2026-10-15）](../../../shape_trials/model_in_loop/model_in_loop_live_20261015.md) | 48 | `80a2305120a9` |
| [Model-in-loop 匀化权重与重拟合实验对照（2026-10-16）](../../../shape_trials/model_in_loop/model_in_loop_live_20261016.md) | 180 | `dd0306f6b56b` |
| [相机位置调整后的物理预测模型校正与多形状整形（2026-10-19）](../../../shape_trials/model_in_loop/model_in_loop_live_20261019.md) | 589 | `227a42d1f82e` |
| [物理预测模型闭环整形：暂停期间工作与恢复实验条件](../../../shape_trials/model_in_loop/model_in_loop_next_steps_20261013.md) | 104 | `e3deb4c0ec3c` |
| [物理预测模型闭环：暂停实验交接（2026-10-13）](../../../shape_trials/model_in_loop/model_in_loop_pause_handoff_20261013.md) | 49 | `1dc76f1c23ba` |
| [MIL (model-in-loop) 单次运行记录报告 — 20261021_172309](../../../shape_trials/model_in_loop/model_in_loop_record_20261021_172309.md) | 265 | `35c3b9cc5215` |
| [Model-in-loop 方斑均匀度实验（2026-10-16 下午）](../../../shape_trials/model_in_loop/model_in_loop_uniformity_live_20261016.md) | 451 | `acbfd67b613d` |
| [当前功率与主动离焦条件下的物理预测模型实验（2026-10-22）](../../../shape_trials/model_in_loop/physical_forward_defocus_live_20261022.md) | 352 | `b6318e46c95f` |
| [物理预测的采样、像素积分、曝光偏置与峰宽诊断（2026-10-22，第二阶段）](../../../shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md) | 259 | `126a4e589065` |
| [不确定性采样探针 (R-82/R-86) 验收：带限约束才是关键（2026-10-16）](../../../shape_trials/model_in_loop/uncertainty_probe_hw_20261016.md) | 175 | `379d41e839da` |
| [model-in-loop 验证集探针的 K 折交叉验证](../../../shape_trials/model_in_loop_cv/model_in_loop_cv.md) | 128 | `20cb65eb8e42` |
| [1200 CCD 像元多形状的物理预测模型可微整形（从零训练，2026-10-27）](../../../shape_trials/multishape1200/multishape1200_20261027.md) | 166 | `7053d426899b` |
| [当前配置下的可微整形能力测试（长方形目标 + 末尾叠加倾斜）](../../../shape_trials/multishape_capability/capability_rectangle_20261023_191444.md) | 87 | `726beeda98a0` |
| [当前配置下的可微整形能力测试（正方形目标 + 末尾叠加倾斜）](../../../shape_trials/multishape_capability/capability_square_20261023_182153.md) | 87 | `0569b861a1ce` |
| [当前配置下的可微整形能力测试（五角星目标 + 末尾叠加倾斜）](../../../shape_trials/multishape_capability/capability_star_20261023_191213.md) | 87 | `30ed33bdf1c7` |
| [当前配置下的可微整形能力测试（三角形目标 + 末尾叠加倾斜）](../../../shape_trials/multishape_capability/capability_triangle_20261023_191105.md) | 87 | `a5e8153da074` |
| [2026-10-23 多形状整形诊断与当前光路迁移记录](../../../shape_trials/multishape_capability/diagnosis_and_transfer_20261023.md) | 319 | `501337384dd6` |
| [离焦光路下正方形、长方形、三角形整形：从零探测、物理预测模型微调与实验复现](../../../shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md) | 723 | `85dce8968196` |
| [不同形状的可微整形能力对比（2f 实验系统，0.80 ms）](../../../shape_trials/multishape_capability/shape_comparison_20261023.md) | 105 | `59241a0b0b50` |
| [2026-10-29 设备在线多尺度整形：实验中记录](../../../shape_trials/multishape_scale/live_20261029_device_session.md) | 908 | `1b5c2848a50e` |
| [多形状 × 三尺度同载波闭环整形：实验后分析（2026-10-28）](../../../shape_trials/multishape_scale/multishape_scale_analysis_20261028.md) | 209 | `f2afeafa6d80` |
| [多形状 × 三尺度同载波闭环整形：实验中记录（2026-10-28）](../../../shape_trials/multishape_scale/multishape_scale_live_20261028.md) | 647 | `497a09542ba9` |
| [多形状 × 三尺度同载波闭环整形：规划（2026-10-28）](../../../shape_trials/multishape_scale/multishape_scale_planning_20261028.md) | 248 | `e79c1523ce86` |
| [多形状物理预测模型闭环整形 campaign 的耗时剖析与写放大治理（2026-10-28）](../../../shape_trials/multishape_scale/multishape_scale_wallclock_20261028.md) | 465 | `8b0f4d0211c0` |
| [2026-10-29 之后的下一步：质量改善实施方案](../../../shape_trials/multishape_scale/next_round_20261029.md) | 555 | `cfcc6a74ca9b` |
| [1000 px 正方形：展宽后可微优化的归档与实验协议（2026-10-30）](../../../shape_trials/multishape_scale/square_1000_diff_abba_20261030.md) | 938 | `2bba4f236762` |
| [无 GS 冷启动、曝光偏置与非线性修正：实验后分析（2026-10-27）](../../../shape_trials/no_gs_ccd_moved/analysis_20261027.md) | 155 | `026a9ac090c7` |
| [CCD 移动后无 GS 冷启动：实验中记录（2026-10-27）](../../../shape_trials/no_gs_ccd_moved/live_20261027.md) | 90 | `c4e21e85c449` |
| [光路变化后的在线重标定：效率分析 (2026-10-13)](../../../shape_trials/online_recal_efficiency/online_recal_efficiency.md) | 344 | `e05d94ed21ce` |
| [相机主动离焦条件下的物理预测模型调优与可微整形实验方案](../../../shape_trials/physical_forward_defocus_plan_20261022.md) | 204 | `d2dbff177bab` |
| [无 GS 起点的物理预测模型与可微整形方案（2026-10-27）](../../../shape_trials/physical_forward_no_gs_plan_20261027.md) | 317 | `8b0eee61f91e` |
| [物理会话闭环相位微调与相差拟合：规划（2026-11-02）](../../../shape_trials/physical_refine/physical_refine_plan_20261102.md) | 89 | `0ebf2e22aaa1` |
| [测量传递层、对齐评分与有界零级：离线结果和下一次实验准备（2026-11-02）](../../../shape_trials/physical_refine/physical_refine_results_20261102.md) | 249 | `ccf35d22c14c` |
| [测量传递层、对齐评分与有界零级：离线结果 2026-11-03](../../../shape_trials/physical_refine/physical_refine_results_20261103.md) | 126 | `6f98f560a520` |
| [形状目标试验（2026-10-09）](../../../shape_trials/README.md) | 294 | `973639658d27` |
| [光束整形报告综述：光路失配、物理设备限制与算法缺陷（2026-11-03）](../../../shape_trials/shaping_issues_review/shaping_issues_review_20261103.md) | 287 | `71fb2303df9c` |
| [仿真预训练与无 GS 迁移复现（2026-10-24）](../../../shape_trials/sim_pretrain_transfer/sim_pretrain_transfer_20261024.md) | 215 | `57742c49077f` |
| [仿真 vs 实测 对照尺 (2026-10-22)](../../../shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261022.md) | 243 | `1bc0c9bf94ab` |
| [仿真 vs 实测 对照尺 (2026-10-22)](../../../shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261023.md) | 249 | `d4637999703c` |
| [V1.0 ↔ V2.0 数字孪生 逐项对照对比 (2026-10-23)](../../../shape_trials/sim_vs_hw/compare_v1_v2_20261023.md) | 193 | `54c3b9523737` |
| [数字孪生 vs 第三轮硬件实测: 逐项对照 (2026-10-22)](../../../shape_trials/sim_vs_hw/sim_vs_hw_20261022.md) | 193 | `7b16d2de2642` |
| [数字孪生 vs 第三轮硬件实测: 逐项对照 (2026-10-22)](../../../shape_trials/sim_vs_hw/sim_vs_hw_20261023.md) | 190 | `e653c95aab64` |
| [SPGD 闭环照明漂移对消 A/B (数字孪生, #144 第 1 项 / #36 判据)](../../../shape_trials/spgd_drift_cancel_loop/spgd_drift_cancel_loop_20261022.md) | 83 | `e3b078cc12aa` |
| [1200 CCD 像元正方形的物理预测模型可微匀化（2026-10-26）](../../../shape_trials/square1200/square1200_20261026.md) | 195 | `6ba15ec33503` |
| [1200 CCD 像元方形：严格验收与非单调探索对照（2026-10-26）](../../../shape_trials/square1200_exploration/exploration_20261026.md) | 118 | `0dc5193e1775` |
| [1200 像元方形：迁移学习适配预训练物理预测模型（手动干扰后，2026-10-27）](../../../shape_trials/transfer_square1200/transfer_square1200_20261027.md) | 175 | `93305579cc94` |
| [三角形与五角星：1.5 倍窗口 shaping 与平面相位起始对比（2026-10-27）](../../../shape_trials/tri_plane_20261027/tri_plane_20261027.md) | 80 | `494af3fa79e6` |
| [2f 实验系统孪生-硬件对齐结论 (Gate 0 归因汇总)](../../../sim2f_bench/alignment_20261027/report.md) | 143 | `19ffcc196271` |
| [2f 实验系统数字孪生的远场尺度 / 离焦标定是否已失效 (纯离线诊断)](../../../sim2f_bench/optical_scale_20261027/report.md) | 248 | `64b3f0e80211` |
| [G0-4 未通过归因: 瞳面中心偏移扫描 (纯离线)](../../../sim2f_bench/twin_vs_hw_corpus/20261027/pupil_center_sweep.md) | 292 | `d23ba293b095` |
| [孪生 `ml.sim2f` 与 2f 实验语料的一致性 (纯离线, 只读 `data/debug/`)](../../../sim2f_bench/twin_vs_hw_corpus/20261027/report.md) | 271 | `7b3608036dd9` |
| [2f 孪生上的整形闭环：摊销、闭环估计与静态光学的归因（2026-10-25）](../../../sim2f_shaping/closed_loop_20261025/report.md) | 386 | `f2abd60da1fa` |
| [TU/e 可微仿真四能力：本仓库 ML 仿真复现 + 实验对照（issue #149）](../../../sim_tue_methods/README.md) | 132 | `97c1f1281df2` |
| [数字孪生保真度验证: 理想孪生 vs 误差参数反演后的孪生 (20261021)](../../../simulation/bench_twin/bench_twin_fidelity_20261021.md) | 186 | `1b4cedfabe00` |
| [2f 数字孪生的可信度边界: 绝对画面像、变化方向不像, 以及可微整形在仿真里今天到底做了什么 (20261030)](../../../simulation/sim_credibility/sim_credibility_20261030.md) | 422 | `b7d756e6716b` |
| [SLM #1 + Daheng 2f 实验系统实测标定报告 (2026-10-01)](../../../slm/bench/bench_calibration_20261001.md) | 246 | `58ef2d9fcd00` |
| [SLM 实验系统探针离线报告](../../../slm/bench_probe/report.md) | 281 | `741a26464a5c` |
| [SLM 硬件调试日报 (2026-09-08)](../../../slm/daily/daily_2026-09-08.md) | 155 | `410bd7b300c7` |
| [SLM 硬件实验日报 (2026-09-15)](../../../slm/daily/daily_2026-09-15.md) | 338 | `e226495764e4` |
| [SLM 硬件实验日报 (2026-09-16)](../../../slm/daily/daily_2026-09-16.md) | 119 | `d146103525c9` |
| [可微相位优化 (20261022_213000) — 过程动画配套报告](../../../slm/differentiable_opt/differentiable_opt_animation_20261022.md) | 237 | `8525d996d9b5` |
| [可微分相位优化 (物理预测模型 + 反向传播) 硬件验证报告](../../../slm/differentiable_opt/differentiable_opt_report.md) | 355 | `d63dc01893a9` |
| [焦点门禁的硬件验证 (2026-10-22)](../../../slm/differentiable_opt/focus_gate_verification_20261022.md) | 69 | `507e35855aff` |
| [闪耀光栅衍射效率: 平面态 0 级能转移走多少能量](../../../slm/diffraction_efficiency/hw_blaze_20261023/report.md) | 110 | `9b7910932cae` |
| [衍射效率标定工具的非理想因素仿真（2026-10-22）](../../../slm/diffraction_efficiency/sim_nonideality_20261022.md) | 582 | `c95a894904c4` |
| [焦面补偿：相机离开 2f 焦面后的菲涅尔焦距实测](../../../slm/focus_plane_report/focus_plane_20261019.md) | 216 | `f81ae42d9e8a` |
| [纯 GS 方斑控制组 — 1000 px 目标 / 2f 实验系统](../../../slm/gs_control_1000px/report.md) | 588 | `ff579226593e` |
| [Model-in-the-loop bench calibration (SLM-200 + Daheng)](../../../slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md) | 691 | `bb94f2ea51c4` |
| [Report 2 — SLM 相位生成链路审计 + 矫正效果根因分析](../../../slm/report2.md) | 481 | `93623f5ca3b2` |
| [厂配波前矫正与 SPGD 优化实验记录（0516）](../../../slm/slm_rms_spgd/0516.md) | 16 | `ed9ae6af15a3` |
| [diff-shaping 直连本机硬件使用指南 (Santec SLM200 + MiiCam)](../../../slm/slm_shaping_diff/readme.md) | 163 | `52d2c19e909e` |
| [SLM 方形光斑 SPGD 整形 (`spgd-square`) — 经验总结与故障分析](../../../slm/slm_square_spgd/README.md) | 228 | `19ff90293b5a` |
| [linearity](../../../slm/zernike_linearity/linearity.md) | 102 | `d342cbbb4091` |
| [Zernike 响应矩阵 — 采集现场截图](../../../slm/zernike_response_matrix_report/capture.md) | 13 | `7b0a8c9a0f14` |
| [Zernike 响应矩阵报告 (创建 / 分析 / 检测)](../../../slm/zernike_response_matrix_report/report.md) | 213 | `bfe933783361` |
| [Zernike 响应矩阵重标 — 完整测试报告 (2026-09-16)](../../../slm/zernike_response_matrix_report/report3.md) | 151 | `91984cae862f` |
| [Zernike 相位 → WFS 读数分布报告](../../../slm/zernike_wfs_report/report.md) | 321 | `f9f691e8bbbf` |
| [SPGD delta 扫描 (实验)](../../../slm_pib_bench/explore_delta/delta_scan.md) | 31 | `d797e44bfbcf` |
| [SPGD delta 扫描 (实验)](../../../slm_pib_bench/explore_delta/delta_scan_large.md) | 26 | `7d72aea51668` |
| [SPGD delta 扫描 (实验)](../../../slm_pib_bench/explore_delta/delta_scan_small.md) | 25 | `ccef7a992e37` |
| [SLM 整形目标函数实验报告 —— Pearson 迁移与参数标定](../../../slm_pib_bench/pib_bench_report/EXPERIMENT_REPORT.md) | 262 | `f380b7dd554d` |
| [SLM PIB 整形实验系统验收报告 (离线分析)](../../../slm_pib_bench/pib_bench_report/report.md) | 232 | `54ef01601c6e` |
| [整形目标函数灵敏度 (噪声地板) 实测](../../../slm_pib_heuristic_hw/measure_shape_sensitivity/sensitivity.md) | 24 | `8996e203897a` |
| [SLM 相位优化实验系统基准报告 (目标: 整形评分 (shape))](../../../slm_pib_heuristic_hw/slm_pib_heuristic_hw_report/report.md) | 240 | `aa7c913b175d` |
| `slm_pib_online/measure_shape_sensitivity/sensitivity.md` — 整形目标函数灵敏度 (噪声地板) 实测（被 `.gitignore` 匹配，不入库） | 21 | `a84216f1b583` |
| [整形目标函数对比报告](../../../slm_pib_online/shape_objective_comparison/objective_comparison.md) | 208 | `654697b5f59d` |
| [SLM-PIB 在线验收报告 (Daheng CCD)](../../../slm_pib_online/slm_pib_online_report/report.md) | 259 | `6c6690d06871` |
| `slm_pib_online_hw/shape_objective_comparison/objective_comparison.md` — Shape Objective Comparison Report（被 `.gitignore` 匹配，不入库） | 153 | `135ec40fe2f7` |
| [slm-pib 仿真运行报告 (SPGD 方形目标, 2f-Fourier 数字孪生)](../../../slm_pib_shape_50px/slm_pib_sim_report/report.md) | 75 | `92bb8a4999da` |
| [slm-pib 仿真运行报告 (SPGD 方形目标, 2f-Fourier 数字孪生)](../../../slm_pib_sim/slm_pib_sim_report/report.md) | 157 | `4f55365ac6af` |
| [SLM 整形仿真运行报告 (2f-Fourier 数字孪生)](../../../slm_shaping_sim/slm_shaping_sim_report/report.md) | 58 | `189ffc3bc6b1` |
| [SLM-Zernike 整形优化器报告 (slm_zernike_shaping)](../../../slm_zernike_shaping/slm_zernike_shaping_report/report.md) | 119 | `6f5855df7185` |
| [Strehl Benchmark Report (7 Heuristics + SPGD)](../../../strehl_benchmark/strehl_benchmark_report/report.md) | 70 | `48adf27853f2` |
| [角谱法 (ASM) 与夫琅禾费 FFT 的分界 —— 实测记录](../../../zernike_coeff2amp/asm_vs_fraunhofer_report/asm_vs_fraunhofer.md) | 89 | `7c617ccef3ef` |
| [Zernike 系数 -> 远场强度 前向网络 —— 开发报告](../../../zernike_coeff2amp/zernike_coeff_report/report.md) | 277 | `9c429ec65206` |
| [Zernike 各模式/系数幅度下远场光斑形貌仿真报告](../../../zernike_farfield_sim/zernike_farfield_sim_report/report.md) | 161 | `e0a5f0249fde` |
| [physics vs U-Net on `slm_zernike_shaping`](../../../zernike_phase2amp/models_cv/unet_comparison.md) | 85 | `924caa63ac69` |
| [Zernike 远场模型 —— 开发报告](../../../zernike_phase2amp/zernike_amp_report/report.md) | 703 | `b412ddfd771e` |
| [Zernike 物理预测模型:验证 R² 被"常数基线"吞掉了](../../../zernike_r2_baseline/pod_ridge/report.md) | 391 | `1c1094f75b81` |
