# report/ — 实验报告总索引

> 本目录是**实验报告**的唯一落点，与设备说明文档 (`docs/`) 分开。
> 2026-10-05 从 `docs/` 迁入：报告记录一次测量的结论，设备文档描述设备本身，
> 两者混放会让"这是说明书还是结论"变得不可判定。

## 目录约定

| | 位置 | 内容 | 生命周期 |
|---|---|---|---|
| **实验报告** | `report/<topic>/` | 一次实验/基准/仿真的**结论**：测量值、参数、口径、被推翻的假设 | 追加/重生成，历史结论不删 |
| **设备说明文档** | `docs/` | 设备规格、SDK/驱动用法、装配与操作手册、图集 | 随驱动/固件更新 |

新增报告 = 在 `scripts/_common/provenance.py` 的 `REPORTS` 里加一条，然后
`python scripts/sync_report_provenance.py` 重新生成本索引与各报告的溯源块。
**不要手工编辑本文件** —— 它是生成的。

## 报告 → 生成脚本

下表的 `生成脚本` 全部是**仓库根相对路径**，可直接 `python <脚本>` 复现。
`离线` = 纯 numpy/torch 读产物，仪器可断电；`硬件` = 需要真实台架。


| 报告 | 生成脚本 | 运行环境 | 说明 |
|---|---|---|---|
| [`beam_shaping/papers/beam_shaping_papers.md`](beam_shaping/papers/beam_shaping_papers.md) | `scripts/generate_beam_shaping_papers_report.py` | 离线 | 文献方法在同一光学模型 + 同一 target 上的仿真对比 |
| [`beam_shaping/papers/literature_survey.md`](beam_shaping/papers/literature_survey.md) | `人工撰写` | 人工调研 | 2026-09 文献调研，非生成产物 |
| [`benchmarks/backend_kernels_performance.md`](benchmarks/backend_kernels_performance.md) | `tests/ao_shaping/utils/test_spots_calc.py` | 离线 | 由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时） |
| [`benchmarks/cython_optimizer_performance.md`](benchmarks/cython_optimizer_performance.md) | `scripts/generate_cython_optimizer_report.py` | 离线 | 读 src/calculators/benchmark_results.json；同目录两张 speedup 图由本脚本生成 |
| [`benchmarks/device_less_full/beam_shaping_benchmark_metrics.md`](benchmarks/device_less_full/beam_shaping_benchmark_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元权威网格 (3 算法 x 3 target)；gif/ 为 6 个演化动画 |
| [`benchmarks/device_less_full/gif/gs_circle/gs_circle_metrics.md`](benchmarks/device_less_full/gif/gs_circle/gs_circle_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元网格中 gs x circle 的逐 GIF 指标 |
| [`benchmarks/device_less_full/gif/gs_gaussian/gs_gaussian_metrics.md`](benchmarks/device_less_full/gif/gs_gaussian/gs_gaussian_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元网格中 gs x gaussian 的逐 GIF 指标 |
| [`benchmarks/device_less_full/gif/gs_square/gs_square_metrics.md`](benchmarks/device_less_full/gif/gs_square/gs_square_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元网格中 gs x square 的逐 GIF 指标 |
| [`benchmarks/device_less_full/gif/spgd-sim_circle/spgd-sim_circle_metrics.md`](benchmarks/device_less_full/gif/spgd-sim_circle/spgd-sim_circle_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元网格中 spgd-sim x circle 的逐 GIF 指标；area 已坍缩，勿据此比均匀性 |
| [`benchmarks/device_less_full/gif/spgd-sim_gaussian/spgd-sim_gaussian_metrics.md`](benchmarks/device_less_full/gif/spgd-sim_gaussian/spgd-sim_gaussian_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元网格中 spgd-sim x gaussian 的逐 GIF 指标；area 已坍缩，勿据此比均匀性 |
| [`benchmarks/device_less_full/gif/spgd-sim_square/spgd-sim_square_metrics.md`](benchmarks/device_less_full/gif/spgd-sim_square/spgd-sim_square_metrics.md) | `scripts/generate_beam_shaping_benchmark_report.py` | 离线 | 9 单元网格中 spgd-sim x square 的逐 GIF 指标；area 已坍缩，勿据此比均匀性 |
| [`centroid_test_visualization/centroid_test_report.md`](centroid_test_visualization/centroid_test_report.md) | `scripts/generate_centroid_test_visualization.py` | 离线 | 同目录 10 张 png 为本脚本产物 |
| [`centroid_test_visualization/centroid_test_report_duplicate_run.md`](centroid_test_visualization/centroid_test_report_duplicate_run.md) | `scripts/generate_centroid_test_visualization.py` | 离线 | 同一次报告的更早一次运行，仅 Generated 时间戳不同；保留作对照，正式版见 centroid_test_report.md |
| [`diff_beam/README.md`](diff_beam/README.md) | `scripts/diff_beam_frame_analysis.py` | 离线+硬件 | 正文为人工实测记录 (SLM200 + 大恒 CCD)；本脚本渲染逐帧产物 |
| [`fouriergsnet_pipeline/TODO.md`](fouriergsnet_pipeline/TODO.md) | `人工撰写` | 硬件 | 真机光路验证待办清单 (设备 + 激光就绪后执行) |
| [`fouriergsnet_pipeline/hardware_run_20261001.md`](fouriergsnet_pipeline/hardware_run_20261001.md) | `人工撰写 (+gsnet_runner.py)` | 硬件 | 实机方形整形运行报告；由 slm-gsnet 命令产生 |
| [`fouriergsnet_pipeline/offline_training/report.md`](fouriergsnet_pipeline/offline_training/report.md) | `scripts/generate_gsnet_offline_report.py` | 离线 | 读 data/gsnet_train/run-*/summary.json |
| [`fouriergsnet_pipeline/report.md`](fouriergsnet_pipeline/report.md) | `scripts/generate_fouriergsnet_pipeline_report.py` | 离线 | 5 项集成测试的结果记录 |
| [`fouriergsnet_sim/report.md`](fouriergsnet_sim/report.md) | `scripts/generate_fouriergsnet_sim_report.py (+fouriergsnet_sim_train.py, generate_slm_gsnet_sim_gif.py)` | 离线 | 数据由 fouriergsnet_sim_train.py 产生；gifs/ 由 generate_slm_gsnet_sim_gif.py 追加 |
| [`heuristic_pib/report.md`](heuristic_pib/report.md) | `scripts/generate_heuristic_pib_report.py` | 离线 | 7 种启发式 + PIB 目标；summary.csv 为本脚本产物 |
| [`hwdataset_corpus/report.md`](hwdataset_corpus/report.md) | `scripts/generate_hwdataset_corpus_report.py (+analyze_hwdataset_distributions.py)` | 离线 | data/debug 硬件调试语料的纯元数据统计 (14 张图, 含 4 张数组级分布图); 只读 hw_index_cache.json 与 data/debug/**/*.json, .pkl 仅取元数据; §10/§11 的数组级实测另读 analyze_hwdataset_distributions.py 预产的 distribution_stats.json (该文件缺失时两节降级为提示, 10 张元数据图不受影响, 两阶段复现命令见报告 §14)。§1.1 的「全量重建是否改变结论」交叉核对需另加 --full-index <扫描过全部 pkl 的索引.json> |
| [`loss_algorithms_smoke/report.md`](loss_algorithms_smoke/report.md) | `scripts/compare_loss_algorithms.py` | 离线 | 测试用例跑 --quick 生成的冒烟版本 |
| [`loss_defects/PROCESS.md`](loss_defects/PROCESS.md) | `人工撰写 (+inverse_design_sim_eval.py, inverse_design_accuracy_ladder.py, inverse_design_restarts.py, inverse_restart_selection.py, inverse_objective_alignment.py, inverse_achievable_target.py, gs_vs_gradient_inverse.py, gs_plus_refinement.py, alignment_vs_accuracy.py, roi_robustness.py, restart_claim_robustness.py, freeform_vs_zernike.py)` | 离线 | 逆向整形 14 次尝试的完整过程记录（含 3 处被推翻的结论） |
| [`loss_defects/README.md`](loss_defects/README.md) | `人工撰写 (+sweep_far_field_padding.py)` | 离线 | 前向模型/loss 缺陷排查结论（人工撰写）；同目录 *.json 为各探针面板 |
| [`loss_defects/forward_search_report.md`](loss_defects/forward_search_report.md) | `scripts/generate_forward_search_report.py (+forward_search.py, inverse_strong_phase.py, probe_device_metadata.py, check_ellipse_term.py)` | 离线 | 正向模型改进尝试: 设备参数/椭圆loss/attention/U-Net/图像增强, 3 seed 配对 |
| [`loss_defects/inverse_design_report.md`](loss_defects/inverse_design_report.md) | `scripts/generate_inverse_design_report.py (+freeform_vs_zernike.py)` | 离线 | 逆向整形中文报告：正向/反向 pred vs true 对比图 + ROI 扫描 + 术语表 |
| [`micro_deformable_mirror/freq_test.md`](micro_deformable_mirror/freq_test.md) | `人工撰写` | 硬件 | 微驱动器频率测试记录；images/freq_test/ 为同期截图 |
| [`micro_deformable_mirror/voltage_test.md`](micro_deformable_mirror/voltage_test.md) | `人工撰写` | 硬件 | 微驱动器电压测试记录 (R50Power)；images/voltage_test/ 为同期截图 |
| [`miicam_simulation/miicam_report.md`](miicam_simulation/miicam_report.md) | `tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py` | 离线 | 仿真 MiiCam 的测试报告 (无需设备) |
| [`model_free_ao_survey/report.md`](model_free_ao_survey/report.md) | `人工撰写` | 人工调研 | 无模型 AO 综述 (Selim 等 2026, J. Optics) × 本项目 model-free 栈 (SPGD/GS/RL/ML/正向模型) 逐条映射 + 6 条可借鉴清单 (自适应增益 / TIE / Cn² 自适应超参 / scintillation index / hybrid 调度 / RL 动态基准); 与 zotero_objectives 与 beam_shaping 文献调研互补, 不重复目标函数公式 |
| [`models_analysis/report.md`](models_analysis/report.md) | `scripts/generate_models_report.py` | 离线 | 读 TensorBoard event 文件 |
| [`models_analysis/reward_analysis.md`](models_analysis/reward_analysis.md) | `人工撰写` | 离线 | 传统 AO 仿真环境 reward 问题分析与解决 |
| [`models_analysis/tmp_sac_run_showcase.md`](models_analysis/tmp_sac_run_showcase.md) | `scripts/generate_models_report.py` | 离线 | 临时调试 run 的成果展示 |
| [`oopao_impact/report.md`](oopao_impact/report.md) | `scripts/generate_oopao_impact_report.py` | 离线 | 端到端 AO 环境下的后端影响 |
| [`oopao_vs_numpy/report.md`](oopao_vs_numpy/report.md) | `scripts/generate_oopao_vs_numpy_report.py` | 离线 | 像差 x 湍流 12 场景 x 2 后端对比 |
| [`pib_loss_terms/README.md`](pib_loss_terms/README.md) | `人工撰写` | 离线+硬件 | 闭环整形物理误差感知 Loss 的取舍记录：6 类候选里落地 1 类(对数强度梯度差分, --w_loggrad)、2 类早已存在、3 类否决, 附实测依据 |
| [`pib_optimizer_functional_report.md`](pib_optimizer_functional_report.md) | `人工撰写` | 离线 | PIB 优化器功能说明 (非实验测量) |
| [`sac_ao_20260109_205218/training_report.md`](sac_ao_20260109_205218/training_report.md) | `人工撰写` | 离线 | SAC 训练报告 (2026-01-09) |
| [`slm/bench_calibration_20261001.md`](slm/bench_calibration_20261001.md) | `人工撰写` | 硬件 | SLM #1 + 大恒 2f 台架实测标定记录 |
| [`slm/bench_probe/report.md`](slm/bench_probe/report.md) | `scripts/generate_bench_probe_report.py (+slm_zernike_sweep_probe.py)` | 离线 | 读台架 sweep 落盘的 npz/json/pkl，仪器可断电 |
| [`slm/daily_2026-09-08.md`](slm/daily_2026-09-08.md) | `人工撰写` | 硬件 | SLM 硬件调试日报 |
| [`slm/daily_2026-09-15.md`](slm/daily_2026-09-15.md) | `人工撰写 (+generate_zernike_response_matrix_report.py, generate_zernike_wfs_report.py)` | 硬件 | 硬件实验日报；引用的两个 generate_* 脚本见当日记录 |
| [`slm/daily_2026-09-16.md`](slm/daily_2026-09-16.md) | `人工撰写 (+wfs_probe.py)` | 硬件 | 硬件实验日报 |
| [`slm/model_in_loop_bench_calibration.md`](slm/model_in_loop_bench_calibration.md) | `scripts/model_in_loop_hw_runbook.py` | 硬件 | 正向模型台架几何标定；runbook 采集 + 本文档结论 |
| [`slm/report2.md`](slm/report2.md) | `人工撰写` | 硬件 | SLM 相位生成链路审计 + 矫正效果根因分析；数据由 zernike-matrix 闭环产生 |
| [`slm/report3.md`](slm/report3.md) | `scripts/generate_zernike_response_matrix_report.py` | 离线 | 2026-09-16 响应矩阵重标测试报告 |
| [`slm/slm_rms_spgd/0516.md`](slm/slm_rms_spgd/0516.md) | `人工撰写` | 硬件 | 厂配矫正 + SPGD 实验记录 |
| [`slm/slm_shaping_diff/readme.md`](slm/slm_shaping_diff/readme.md) | `人工撰写` | 硬件 | diff-shaping 直连硬件使用指南与故障排查记录 |
| [`slm/slm_square_spgd/README.md`](slm/slm_square_spgd/README.md) | `人工撰写` | 硬件 | 方形光斑 SPGD 经验总结与故障分析 |
| [`slm/zernike_linearity/linearity.md`](slm/zernike_linearity/linearity.md) | `scripts/generate_zernike_linearity_report.py` | 离线 | 读 data/zernike_correction/raw_scan_*.json |
| [`slm/zernike_response_matrix_report/capture.md`](slm/zernike_response_matrix_report/capture.md) | `人工撰写` | 硬件 | 响应矩阵采集现场截图记录 (image/capture/) |
| [`slm/zernike_response_matrix_report/report.md`](slm/zernike_response_matrix_report/report.md) | `scripts/generate_zernike_response_matrix_report.py` | 离线 | 创建/分析/检测三段；figures/ 为本脚本产物 |
| [`slm/zernike_wfs_report/report.md`](slm/zernike_wfs_report/report.md) | `scripts/generate_zernike_wfs_report.py` | 硬件 | 需 Santec SLM-200 + Thorlabs WFS；phase/ 与 wfs/ 为本脚本产物 |
| [`slm_200/slm_mla_test.md`](slm_200/slm_mla_test.md) | `人工撰写` | 硬件 | SLM-200 MLA 实验记录；images/slm_mla_test/ 为同期截图 |
| [`slm_pib_bench/EXPERIMENT_REPORT.md`](slm_pib_bench/EXPERIMENT_REPORT.md) | `scripts/generate_pib_bench_report.py (+generate_shape_objective_comparison.py, measure_shape_sensitivity.py)` | 离线 | 台架验收总报告；含 Pearson 迁移与参数标定两节 |
| [`slm_pib_bench/delta_scan.md`](slm_pib_bench/delta_scan.md) | `scripts/explore_delta.py` | 硬件 | SPGD delta 扫描 (实机) |
| [`slm_pib_bench/delta_scan_large.md`](slm_pib_bench/delta_scan_large.md) | `scripts/explore_delta.py` | 硬件 | delta 大范围扫描 (实机) |
| [`slm_pib_bench/delta_scan_small.md`](slm_pib_bench/delta_scan_small.md) | `scripts/explore_delta.py` | 硬件 | delta 小范围扫描 (实机) |
| [`slm_pib_bench/report.md`](slm_pib_bench/report.md) | `scripts/generate_pib_bench_report.py` | 离线 | 离线分析报告 (噪声地板 / 门控 / best-vs-sustained) |
| [`slm_pib_heuristic_hw/report.md`](slm_pib_heuristic_hw/report.md) | `scripts/generate_slm_pib_heuristic_hw_report.py` | 硬件 | 真机基准 (Santec SLM-200 + 大恒 MER2-507)；figures/ + summary.csv 为本脚本产物 |
| [`slm_pib_heuristic_hw/sensitivity.md`](slm_pib_heuristic_hw/sensitivity.md) | `scripts/measure_shape_sensitivity.py` | 硬件 | 整形目标函数灵敏度 (噪声地板) 实测；测量内核已下沉到 tools/slm/slm_snr_probe.py |
| [`slm_pib_online/objective_comparison.md`](slm_pib_online/objective_comparison.md) | `scripts/generate_shape_objective_comparison.py` | 离线 | 三种整形目标函数的排序一致性 (Spearman) |
| [`slm_pib_online/report.md`](slm_pib_online/report.md) | `scripts/generate_slm_pib_online_report.py` | 离线 | 在线回归套件验收报告 (读 data/debug/slm_pib_online/) |
| [`slm_pib_online/sensitivity.md`](slm_pib_online/sensitivity.md) | `scripts/measure_shape_sensitivity.py` | 硬件 | 整形目标函数灵敏度 (噪声地板) 实测 |
| [`slm_pib_online_hw/objective_comparison.md`](slm_pib_online_hw/objective_comparison.md) | `scripts/generate_shape_objective_comparison.py` | 离线 | 真机帧上的目标函数排序对比 |
| [`slm_pib_shape_50px/report.md`](slm_pib_shape_50px/report.md) | `scripts/generate_slm_pib_sim_report.py` | 离线 | 50px target 的仿真运行报告 |
| [`slm_pib_sim/report.md`](slm_pib_sim/report.md) | `scripts/generate_slm_pib_sim_report.py (+slm_pib_sim_run.py)` | 离线 | 2f-Fourier 数字孪生跑 slm-pib；数据由 slm_pib_sim_run.py 产生 |
| [`slm_shaping_sim/report.md`](slm_shaping_sim/report.md) | `scripts/generate_slm_shaping_sim_report.py (+run_sim_bench.py)` | 离线 | SLM+CCD 整形仿真: 三个 SLM 族 runner 的收敛与 DM 族烟测对照 |
| [`slm_zernike_shaping/report.md`](slm_zernike_shaping/report.md) | `scripts/generate_slm_zernike_shaping_report.py` | 离线 | 读 slm_zernike_shaping optimizer 的 debug 产物 |
| [`strehl_benchmark/report.md`](strehl_benchmark/report.md) | `scripts/generate_strehl_benchmark_report.py` | 离线 | 8 种算法 + Strehl 目标；交叉对比 heuristic_pib/summary.csv |
| [`zernike_coeff2amp/report.md`](zernike_coeff2amp/report.md) | `scripts/generate_zernike_coeff_report.py (+sweep_coeff_models.py)` | 离线 | 系数->远场前向网络: 数据/归一化闸门/18折文件级 + 5折目标级交叉验证/单折完整训练 |
| [`zernike_farfield_sim/report.md`](zernike_farfield_sim/report.md) | `scripts/generate_zernike_farfield_sim_report.py` | 离线 | Noll 4-15 各模式远场形貌仿真；metrics.csv 为本脚本产物 |
| [`zernike_phase2amp/report.md`](zernike_phase2amp/report.md) | `scripts/generate_zernike_amp_report.py (+sweep_zernike_models.py, compare_models_cv.py)` | 离线 | 9 张图；数据由 sweep_zernike_models.py 与 ml.zernike.train_amp 产生 |
| [`zernike_phase2amp/unet_comparison.md`](zernike_phase2amp/unet_comparison.md) | `scripts/compare_models_cv.py (+compare_unet_baseline.py)` | 离线 | grouped CV 对比 (physics / hybrid / unet) 的原始对照表 |
| [`zotero_objectives/README.md`](zotero_objectives/README.md) | `人工撰写` | 人工调研 | Zotero 扫描：整形目标函数 / 评价函数目录 |

## 尚未生成产物的报告

下列报告已在 `REPORTS` 登记但产物尚未提交（需要硬件，或尚未跑过），因此没有可点击的文件；生成脚本与复现命令见上表同条目：

- `loss_algorithms/report.md` ← `scripts/compare_loss_algorithms.py` (离线)
