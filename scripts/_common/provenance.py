"""Report <-> generating-script provenance: the single source of truth.

**Why this exists.** Every report in ``report/`` answers "where did this number
come from?", but the answer was previously scattered: some reports named their
generator in prose, most named nothing, and a few named a script that had since
been renamed. Worse, the report tree moved out of ``docs/`` on 2026-10-05 while
the generators' hardcoded output paths moved with it -- so a reader could not
tell whether a stale path meant "the report moved" or "the report is wrong".

So the mapping is declared **once**, here, as :data:`REPORTS`, and everything
else is derived from it:

* :func:`provenance_block` renders the ``> 生成脚本`` header that every report
  carries, with a link whose relative depth is computed from the report's own
  location (so it works at any nesting depth).
* :func:`render_index` renders ``report/README.md`` -- the report -> script ->
  environment table.
* ``scripts/sync_report_provenance.py`` inserts the block into every report and
  regenerates the index, idempotently.

**Adding a report** means adding one :data:`REPORTS` entry and re-running the
sync script. Nothing else needs to know the rule.

The keys are report paths relative to the repo root, using ``/`` separators, and
they are matched exactly -- a report nested one level deeper is a different key,
because its links and its generator usually differ too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "END_MARKER",
    "REPORTS",
    "ReportProvenance",
    "START_MARKER",
    "index_rows",
    "provenance_block",
    "provenance_block_for",
    "render_index",
    "script_link",
]

#: HTML-comment fences around an inserted header. A writer emits the same fences
#: so :mod:`scripts.sync_report_provenance` can find and replace its own output
#: instead of stacking a second header on the next run.
START_MARKER = "<!-- provenance:start -->"
END_MARKER = "<!-- provenance:end -->"

#: Marker used in ``script`` when no script generates the document, i.e. it was
#: written by hand. Those still get a provenance block -- "hand-written" plus the
#: hardware it was measured on is exactly the information a reader needs.
HANDWRITTEN = ""


@dataclass(frozen=True)
class ReportProvenance:
    """Where one report comes from.

    Attributes:
        script: Repo-root-relative path of the generating script, or
            :data:`HANDWRITTEN` for a hand-written record.
        command: The exact command that reproduces the report. Empty when
            hand-written (there is nothing to re-run).
        environment: What the run needs -- ``离线`` (offline), ``硬件`` (a real
            bench), or a mix. This is the field a reader checks before trying to
            regenerate, so it is stated per report rather than inferred.
        note: Optional qualifier, e.g. which *other* script produced the inputs.
        extra_scripts: Additional scripts that fed the report without writing it
            (the data producer, when that differs from the renderer).
    """

    script: str
    command: str = ""
    environment: str = "离线"
    note: str = ""
    extra_scripts: tuple[str, ...] = field(default=())


#: Every report under ``report/``, keyed by repo-root-relative path.
#:
#: ``environment`` values, used verbatim in the index:
#:   ``离线``   -- pure numpy/torch on saved artefacts; no instrument involved.
#:   ``硬件``   -- needs a real bench (SLM / CCD / WFS / DM / micro-DM).
#:   ``离线+硬件`` -- renders offline, but the numbers were measured on hardware.
REPORTS: dict[str, ReportProvenance] = {
    "report/zernike_coeff2amp/report.md": ReportProvenance(
        script="scripts/generate_zernike_coeff_report.py",
        command="python scripts/generate_zernike_coeff_report.py",
        environment="离线",
        note="系数->远场前向网络: 数据/归一化闸门/18折文件级 + 5折目标级交叉验证/单折完整训练",
        extra_scripts=("scripts/sweep_coeff_models.py",),
    ),

    # -- beam shaping / benchmark suite -------------------------------------
    "report/beam_shaping/papers/beam_shaping_papers.md": ReportProvenance(
        script="scripts/generate_beam_shaping_papers_report.py",
        command="python scripts/generate_beam_shaping_papers_report.py",
        environment="离线",
        note="文献方法在同一光学模型 + 同一 target 上的仿真对比",
    ),
    "report/beam_shaping/papers/literature_survey.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="人工调研",
        note="2026-09 文献调研，非生成产物",
    ),
    "report/benchmarks/cython_optimizer_performance.md": ReportProvenance(
        script="scripts/generate_cython_optimizer_report.py",
        command="python scripts/generate_cython_optimizer_report.py",
        environment="离线",
        note="读 src/calculators/benchmark_results.json；同目录两张 speedup 图由本脚本生成",
    ),
    "report/benchmarks/backend_kernels_performance.md": ReportProvenance(
        script="tests/ao_shaping/utils/test_spots_calc.py",
        command="python -m pytest tests/ao_shaping/utils/test_spots_calc.py -k benchmark",
        environment="离线",
        note="由 benchmark 测试用例写出（NumPy/Numba/CuPy 逐函数耗时）",
    ),
    "report/benchmarks/device_less_full/beam_shaping_benchmark_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元权威网格 (3 算法 x 3 target)；gif/ 为 6 个演化动画",
    ),
    "report/centroid_test_visualization/centroid_test_report.md": ReportProvenance(
        script="scripts/generate_centroid_test_visualization.py",
        command="python scripts/generate_centroid_test_visualization.py",
        environment="离线",
        note="同目录 10 张 png 为本脚本产物",
    ),
    "report/centroid_test_visualization/centroid_test_report_duplicate_run.md": ReportProvenance(
        script="scripts/generate_centroid_test_visualization.py",
        command="python scripts/generate_centroid_test_visualization.py",
        environment="离线",
        note="同一次报告的更早一次运行，仅 Generated 时间戳不同；保留作对照，正式版见 centroid_test_report.md",
    ),
    # -- SLM 台架 ------------------------------------------------------------
    "report/diff_beam/README.md": ReportProvenance(
        script="scripts/diff_beam_frame_analysis.py",
        command="python scripts/diff_beam_frame_analysis.py --run-dir data/diff_beam/run_<ts> --plot",
        environment="离线+硬件",
        note="正文为人工实测记录 (SLM200 + 大恒 CCD)；本脚本渲染逐帧产物",
    ),
    "report/slm/report2.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM 相位生成链路审计 + 矫正效果根因分析；数据由 zernike-matrix 闭环产生",
    ),
    "report/slm/report3.md": ReportProvenance(
        script="scripts/generate_zernike_response_matrix_report.py",
        command="python scripts/generate_zernike_response_matrix_report.py",
        environment="离线",
        note="2026-09-16 响应矩阵重标测试报告",
    ),
    "report/slm/daily_2026-09-08.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM 硬件调试日报",
    ),
    "report/slm/daily_2026-09-15.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="硬件实验日报；引用的两个 generate_* 脚本见当日记录",
        extra_scripts=(
            "scripts/generate_zernike_response_matrix_report.py",
            "scripts/generate_zernike_wfs_report.py",
        ),
    ),
    "report/slm/daily_2026-09-16.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="硬件实验日报",
        extra_scripts=("scripts/wfs_probe.py",),
    ),
    "report/slm/bench_calibration_20261001.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM #1 + 大恒 2f 台架实测标定记录",
    ),
    "report/slm/model_in_loop_bench_calibration.md": ReportProvenance(
        script="scripts/model_in_loop_hw_runbook.py",
        command="python scripts/model_in_loop_hw_runbook.py --stage all --target-cam-px 40",
        environment="硬件",
        note="正向模型台架几何标定；runbook 采集 + 本文档结论",
    ),
    # NOTE: `docs/slm/model_in_loop_algorithm.md` is generated by
    # `scripts/generate_model_in_loop_report.py` but is deliberately **not** listed
    # here: it is a usage/architecture description rather than the conclusion of a
    # measurement, so it belongs in `docs/`, and this registry renders
    # `report/README.md` with the assumption that every key is under `report/`.
    # It carries its own provenance header instead.
    "report/slm/bench_probe/report.md": ReportProvenance(
        script="scripts/generate_bench_probe_report.py",
        command="python scripts/generate_bench_probe_report.py",
        environment="离线",
        note="读台架 sweep 落盘的 npz/json/pkl，仪器可断电",
        extra_scripts=("src/ao_shaping/tools/slm/slm_zernike_sweep_probe.py",),
    ),
    "report/slm/zernike_linearity/linearity.md": ReportProvenance(
        script="scripts/generate_zernike_linearity_report.py",
        command="python scripts/generate_zernike_linearity_report.py",
        environment="离线",
        note="读 data/zernike_correction/raw_scan_*.json",
    ),
    "report/slm/zernike_response_matrix_report/report.md": ReportProvenance(
        script="scripts/generate_zernike_response_matrix_report.py",
        command="python scripts/generate_zernike_response_matrix_report.py",
        environment="离线",
        note="创建/分析/检测三段；figures/ 为本脚本产物",
    ),
    "report/slm/zernike_wfs_report/report.md": ReportProvenance(
        script="scripts/generate_zernike_wfs_report.py",
        command="python scripts/generate_zernike_wfs_report.py",
        environment="硬件",
        note="需 Santec SLM-200 + Thorlabs WFS；phase/ 与 wfs/ 为本脚本产物",
    ),
    "report/slm/slm_square_spgd/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="方形光斑 SPGD 经验总结与故障分析",
    ),
    "report/slm/slm_rms_spgd/0516.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="厂配矫正 + SPGD 实验记录",
    ),
    "report/slm/slm_shaping_diff/readme.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="diff-shaping 直连硬件使用指南与故障排查记录",
    ),
    "report/slm_200/slm_mla_test.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM-200 MLA 实验记录；images/slm_mla_test/ 为同期截图",
    ),
    # -- micro-DM ------------------------------------------------------------
    "report/micro_deformable_mirror/voltage_test.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="微驱动器电压测试记录 (R50Power)；images/voltage_test/ 为同期截图",
    ),
    "report/micro_deformable_mirror/freq_test.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="微驱动器频率测试记录；images/freq_test/ 为同期截图",
    ),
    # -- FourierGSNet --------------------------------------------------------
    "report/fouriergsnet_pipeline/report.md": ReportProvenance(
        script="scripts/generate_fouriergsnet_pipeline_report.py",
        command="python scripts/generate_fouriergsnet_pipeline_report.py",
        environment="离线",
        note="5 项集成测试的结果记录",
    ),
    "report/fouriergsnet_pipeline/hardware_run_20261001.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="实机方形整形运行报告；由 slm-gsnet 命令产生",
        extra_scripts=("src/ao_shaping/runners/slm/gsnet_runner.py",),
    ),
    "report/fouriergsnet_pipeline/offline_training/report.md": ReportProvenance(
        script="scripts/generate_gsnet_offline_report.py",
        command="python scripts/generate_gsnet_offline_report.py",
        environment="离线",
        note="读 data/gsnet_train/run-*/summary.json",
    ),
    "report/fouriergsnet_sim/report.md": ReportProvenance(
        script="scripts/generate_fouriergsnet_sim_report.py",
        command="python scripts/generate_fouriergsnet_sim_report.py",
        environment="离线",
        note="数据由 fouriergsnet_sim_train.py 产生；gifs/ 由 generate_slm_gsnet_sim_gif.py 追加",
        extra_scripts=(
            "scripts/fouriergsnet_sim_train.py",
            "scripts/generate_slm_gsnet_sim_gif.py",
        ),
    ),
    # -- 优化器基准 / PIB --------------------------------------------------
    "report/heuristic_pib/report.md": ReportProvenance(
        script="scripts/generate_heuristic_pib_report.py",
        command="python scripts/generate_heuristic_pib_report.py",
        environment="离线",
        note="7 种启发式 + PIB 目标；summary.csv 为本脚本产物",
    ),
    "report/strehl_benchmark/report.md": ReportProvenance(
        script="scripts/generate_strehl_benchmark_report.py",
        command="python scripts/generate_strehl_benchmark_report.py",
        environment="离线",
        note="8 种算法 + Strehl 目标；交叉对比 heuristic_pib/summary.csv",
    ),
    "report/slm_pib_bench/EXPERIMENT_REPORT.md": ReportProvenance(
        script="scripts/generate_pib_bench_report.py",
        command="python scripts/generate_pib_bench_report.py",
        environment="离线",
        note="台架验收总报告；含 Pearson 迁移与参数标定两节",
        extra_scripts=(
            "scripts/generate_shape_objective_comparison.py",
            "scripts/measure_shape_sensitivity.py",
        ),
    ),
    "report/slm_pib_bench/report.md": ReportProvenance(
        script="scripts/generate_pib_bench_report.py",
        command="python scripts/generate_pib_bench_report.py",
        environment="离线",
        note="离线分析报告 (噪声地板 / 门控 / best-vs-sustained)",
    ),
    "report/slm_pib_bench/delta_scan.md": ReportProvenance(
        script="scripts/explore_delta.py",
        command="python scripts/explore_delta.py",
        environment="硬件",
        note="SPGD delta 扫描 (实机)",
    ),
    "report/slm_pib_bench/delta_scan_large.md": ReportProvenance(
        script="scripts/explore_delta.py",
        command="python scripts/explore_delta.py",
        environment="硬件",
        note="delta 大范围扫描 (实机)",
    ),
    "report/slm_pib_bench/delta_scan_small.md": ReportProvenance(
        script="scripts/explore_delta.py",
        command="python scripts/explore_delta.py",
        environment="硬件",
        note="delta 小范围扫描 (实机)",
    ),
    "report/slm_pib_heuristic_hw/report.md": ReportProvenance(
        script="scripts/generate_slm_pib_heuristic_hw_report.py",
        command="python scripts/generate_slm_pib_heuristic_hw_report.py",
        environment="硬件",
        note="真机基准 (Santec SLM-200 + 大恒 MER2-507)；figures/ + summary.csv 为本脚本产物",
    ),
    "report/slm_pib_heuristic_hw/sensitivity.md": ReportProvenance(
        script="scripts/measure_shape_sensitivity.py",
        command="python scripts/measure_shape_sensitivity.py",
        environment="硬件",
        note="整形目标函数灵敏度 (噪声地板) 实测；测量内核已下沉到 tools/slm/slm_snr_probe.py",
    ),
    "report/slm_pib_online/report.md": ReportProvenance(
        script="scripts/generate_slm_pib_online_report.py",
        command="python scripts/generate_slm_pib_online_report.py",
        environment="离线",
        note="在线回归套件验收报告 (读 data/debug/slm_pib_online/)",
    ),
    "report/slm_pib_online/objective_comparison.md": ReportProvenance(
        script="scripts/generate_shape_objective_comparison.py",
        command="python scripts/generate_shape_objective_comparison.py",
        environment="离线",
        note="三种整形目标函数的排序一致性 (Spearman)",
    ),
    "report/slm_pib_online/sensitivity.md": ReportProvenance(
        script="scripts/measure_shape_sensitivity.py",
        command="python scripts/measure_shape_sensitivity.py",
        environment="硬件",
        note="整形目标函数灵敏度 (噪声地板) 实测",
    ),
    "report/slm_pib_online_hw/objective_comparison.md": ReportProvenance(
        script="scripts/generate_shape_objective_comparison.py",
        command="python scripts/generate_shape_objective_comparison.py",
        environment="离线",
        note="真机帧上的目标函数排序对比",
    ),
    "report/slm_pib_sim/report.md": ReportProvenance(
        script="scripts/generate_slm_pib_sim_report.py",
        command="python scripts/generate_slm_pib_sim_report.py",
        environment="离线",
        note="2f-Fourier 数字孪生跑 slm-pib；数据由 slm_pib_sim_run.py 产生",
        extra_scripts=("scripts/slm_pib_sim_run.py",),
    ),
    "report/slm_shaping_sim/report.md": ReportProvenance(
        script="scripts/generate_slm_shaping_sim_report.py",
        command="python scripts/generate_slm_shaping_sim_report.py",
        environment="离线",
        note="SLM+CCD 整形仿真: 三个 SLM 族 runner 的收敛与 DM 族烟测对照",
        extra_scripts=("scripts/run_sim_bench.py",),
    ),
    "report/slm_pib_shape_50px/report.md": ReportProvenance(
        script="scripts/generate_slm_pib_sim_report.py",
        command="python scripts/generate_slm_pib_sim_report.py",
        environment="离线",
        note="50px target 的仿真运行报告",
    ),
    "report/slm_zernike_shaping/report.md": ReportProvenance(
        script="scripts/generate_slm_zernike_shaping_report.py",
        command="python scripts/generate_slm_zernike_shaping_report.py",
        environment="离线",
        note="读 slm_zernike_shaping optimizer 的 debug 产物",
    ),
    "report/pib_optimizer_functional_report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="PIB 优化器功能说明 (非实验测量)",
    ),
    # -- ML / 模型 ----------------------------------------------------------
    "report/models_analysis/report.md": ReportProvenance(
        script="scripts/generate_models_report.py",
        command="python scripts/generate_models_report.py",
        environment="离线",
        note="读 TensorBoard event 文件",
    ),
    "report/models_analysis/reward_analysis.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="传统 AO 仿真环境 reward 问题分析与解决",
    ),
    "report/models_analysis/tmp_sac_run_showcase.md": ReportProvenance(
        script="scripts/generate_models_report.py",
        command="python scripts/generate_models_report.py",
        environment="离线",
        note="临时调试 run 的成果展示",
    ),
    "report/zernike_phase2amp/report.md": ReportProvenance(
        script="scripts/generate_zernike_amp_report.py",
        command="python scripts/generate_zernike_amp_report.py",
        environment="离线",
        note="9 张图；数据由 sweep_zernike_models.py 与 ml.zernike.train_amp 产生",
        extra_scripts=(
            "scripts/sweep_zernike_models.py",
            "scripts/compare_models_cv.py",
        ),
    ),
    "report/zernike_phase2amp/unet_comparison.md": ReportProvenance(
        script="scripts/compare_models_cv.py",
        command="python scripts/compare_models_cv.py --analyse",
        environment="离线",
        note="grouped CV 对比 (physics / hybrid / unet) 的原始对照表",
        extra_scripts=("scripts/compare_unet_baseline.py",),
    ),
    "report/zernike_r2_baseline/report.md": ReportProvenance(
        script="scripts/diagnose_pod_ridge.py",
        command=(
            "python scripts/diagnose_pod_ridge.py --k 4 8 16 32 64 128 256 "
            "# 再跑 verify_pod_ridge_canary.py / diagnose_r2_baseline.py / "
            "sweep_target_transform.py"
        ),
        environment="离线",
        note=(
            "验证 R² 被常数基线吞掉：常数预测器在同一 10 折协议下 R²=+0.910，"
            "故改用 skill = 1 - mse_model/mse_const 记分；含 image_mode 默认值"
            "论证的撤回"
        ),
        extra_scripts=(
            "scripts/diagnose_pod_ridge.py",
            "scripts/verify_pod_ridge_canary.py",
            "scripts/diagnose_r2_baseline.py",
            "scripts/sweep_target_transform.py",
        ),
    ),
    # -- 硬件调试语料统计 (hwdataset) ----------------------------------------
    "report/hwdataset_corpus/report.md": ReportProvenance(
        script="scripts/generate_hwdataset_corpus_report.py",
        command="python scripts/generate_hwdataset_corpus_report.py",
        environment="离线",
        note=(
            "data/debug 硬件调试语料的纯元数据统计 (14 张图, 含 4 张数组级分布图); "
            "只读 hw_index_cache.json 与 data/debug/**/*.json, .pkl 仅取元数据; "
            "§10/§11 的数组级实测另读 analyze_hwdataset_distributions.py 预产的 "
            "distribution_stats.json (该文件缺失时两节降级为提示, 10 张元数据图不受影响, "
            "两阶段复现命令见报告 §14)。"
            "§1.1 的「全量重建是否改变结论」交叉核对需另加 "
            "--full-index <扫描过全部 pkl 的索引.json>"
        ),
        extra_scripts=("scripts/analyze_hwdataset_distributions.py",),
    ),
    "report/sac_ao_20260109_205218/training_report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="SAC 训练报告 (2026-01-09)",
    ),
    # -- 仿真 / 其它 --------------------------------------------------------
    "report/zernike_farfield_sim/report.md": ReportProvenance(
        script="scripts/generate_zernike_farfield_sim_report.py",
        command="python scripts/generate_zernike_farfield_sim_report.py",
        environment="离线",
        note="Noll 4-15 各模式远场形貌仿真；metrics.csv 为本脚本产物",
    ),
    "report/oopao_vs_numpy/report.md": ReportProvenance(
        script="scripts/generate_oopao_vs_numpy_report.py",
        command="python scripts/generate_oopao_vs_numpy_report.py",
        environment="离线",
        note="像差 x 湍流 12 场景 x 2 后端对比",
    ),
    "report/oopao_impact/report.md": ReportProvenance(
        script="scripts/generate_oopao_impact_report.py",
        command="python scripts/generate_oopao_impact_report.py",
        environment="离线",
        note="端到端 AO 环境下的后端影响",
    ),
    "report/miicam_simulation/miicam_report.md": ReportProvenance(
        script="tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py",
        command="python -m pytest tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py",
        environment="离线",
        note="仿真 MiiCam 的测试报告 (无需设备)",
    ),
    "report/zotero_objectives/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="人工调研",
        note="Zotero 扫描：整形目标函数 / 评价函数目录",
    ),
    "report/model_free_ao_survey/report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="人工调研",
        note=(
            "无模型 AO 综述 (Selim 等 2026, J. Optics) × 本项目 model-free 栈 "
            "(SPGD/GS/RL/ML/正向模型) 逐条映射 + 6 条可借鉴清单 "
            "(自适应增益 / TIE / Cn² 自适应超参 / scintillation index / hybrid 调度 / "
            "RL 动态基准); 与 zotero_objectives 与 beam_shaping 文献调研互补, 不重复目标函数公式"
        ),
    ),
    "report/pib_loss_terms/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "闭环整形物理误差感知 Loss 的取舍记录：6 类候选里落地 1 类"
            "(对数强度梯度差分, --w_loggrad)、2 类早已存在、3 类否决, 附实测依据"
        ),
    ),
    # -- 基准网格的逐 GIF 指标 (同一脚本按 单元 生成) ----------------------
    "report/benchmarks/device_less_full/gif/gs_circle/gs_circle_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 gs x circle 的逐 GIF 指标",
    ),
    "report/benchmarks/device_less_full/gif/gs_gaussian/gs_gaussian_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 gs x gaussian 的逐 GIF 指标",
    ),
    "report/benchmarks/device_less_full/gif/gs_square/gs_square_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 gs x square 的逐 GIF 指标",
    ),
    "report/benchmarks/device_less_full/gif/spgd-sim_circle/spgd-sim_circle_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 spgd-sim x circle 的逐 GIF 指标；area 已坍缩，勿据此比均匀性",
    ),
    "report/benchmarks/device_less_full/gif/spgd-sim_gaussian/spgd-sim_gaussian_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 spgd-sim x gaussian 的逐 GIF 指标；area 已坍缩，勿据此比均匀性",
    ),
    "report/benchmarks/device_less_full/gif/spgd-sim_square/spgd-sim_square_metrics.md": ReportProvenance(
        script="scripts/generate_beam_shaping_benchmark_report.py",
        command="python scripts/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 spgd-sim x square 的逐 GIF 指标；area 已坍缩，勿据此比均匀性",
    ),
    # -- 正向模型 / loss 缺陷排查 ------------------------------------------
    "report/loss_defects/README.md": ReportProvenance(
        script=HANDWRITTEN,
        command="python -m ml.zernike.train_amp",
        environment="离线",
        note="前向模型/loss 缺陷排查结论（人工撰写）；同目录 *.json 为各探针面板",
        extra_scripts=("scripts/sweep_far_field_padding.py",),
    ),
    "report/loss_defects/PROCESS.md": ReportProvenance(
        script=HANDWRITTEN,
        command="python scripts/inverse_design_sim_eval.py",
        environment="离线",
        note="逆向整形 14 次尝试的完整过程记录（含 3 处被推翻的结论）",
        extra_scripts=(
            "scripts/inverse_design_sim_eval.py",
            "scripts/inverse_design_accuracy_ladder.py",
            "scripts/inverse_design_restarts.py",
            "scripts/inverse_restart_selection.py",
            "scripts/inverse_objective_alignment.py",
            "scripts/inverse_achievable_target.py",
            "scripts/gs_vs_gradient_inverse.py",
            "scripts/gs_plus_refinement.py",
            "scripts/alignment_vs_accuracy.py",
            "scripts/roi_robustness.py",
            "scripts/restart_claim_robustness.py",
            "scripts/freeform_vs_zernike.py",
        ),
    ),
    "report/loss_defects/forward_search_report.md": ReportProvenance(
        script="scripts/generate_forward_search_report.py",
        command="python scripts/forward_search.py && python scripts/generate_forward_search_report.py",
        environment="离线",
        note="正向模型改进尝试: 设备参数/椭圆loss/attention/U-Net/图像增强, 3 seed 配对",
        extra_scripts=(
            "scripts/forward_search.py",
            "scripts/inverse_strong_phase.py",
            "scripts/probe_device_metadata.py",
            "scripts/check_ellipse_term.py",
        ),
    ),
    "report/loss_defects/inverse_design_report.md": ReportProvenance(
        script="scripts/generate_inverse_design_report.py",
        command="python scripts/generate_inverse_design_report.py",
        environment="离线",
        note="逆向整形中文报告：正向/反向 pred vs true 对比图 + ROI 扫描 + 术语表",
        extra_scripts=("scripts/freeform_vs_zernike.py",),
    ),
    # -- 附属记录 ------------------------------------------------------------
    "report/fouriergsnet_pipeline/TODO.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="真机光路验证待办清单 (设备 + 激光就绪后执行)",
    ),
    "report/slm/zernike_response_matrix_report/capture.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="响应矩阵采集现场截图记录 (image/capture/)",
    ),
    # -- loss 算法对比 (compare_loss_algorithms.py 的 --out 家族) -----------
    "report/loss_algorithms/report.md": ReportProvenance(
        script="scripts/compare_loss_algorithms.py",
        command="python scripts/compare_loss_algorithms.py",
        environment="离线",
        note="默认 --out 落点；summary.csv / summary.json / summary.png 为本脚本产物",
    ),
    "report/loss_algorithms_smoke/report.md": ReportProvenance(
        script="scripts/compare_loss_algorithms.py",
        command="python scripts/compare_loss_algorithms.py --quick --out report/loss_algorithms_smoke",
        environment="离线",
        note="测试用例跑 --quick 生成的冒烟版本",
    ),
}


#: Report directories that hold generated figures with no markdown of their own.
#: Kept separate from :data:`REPORTS` because there is no file to attach a header
#: to; the index lists them so the directory is still traceable to its script.
FIGURE_ONLY_DIRS: dict[str, ReportProvenance] = {
    "report/iterative_zernike_shaping": ReportProvenance(
        script="scripts/generate_iterative_zernike_shaping_report.py",
        command="python scripts/generate_iterative_zernike_shaping_report.py",
        environment="离线",
        note="GS 预整形 + 自由相位细化对比 (data.json + *.npy + png)",
    ),
    "report/pearson_gifs": ReportProvenance(
        script="scripts/generate_pearson_pkl_gif.py",
        command="python scripts/generate_pearson_pkl_gif.py",
        environment="离线",
        note="逐 epoch 的 CCD/相位同步 GIF",
    ),
    "report/simulation": ReportProvenance(
        script="scripts/simulate_atmospheric_comparison.py",
        command="python scripts/simulate_atmospheric_comparison.py",
        environment="离线",
        note="湍流对比图 + SPGD Zernike 调参 json",
        extra_scripts=("scripts/tune_sim_spgd_zernike.py",),
    ),
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def script_link(script: str, depth: int) -> str:
    """Markdown link to ``script`` from a report ``depth`` dirs below the root.

    ``depth`` is the number of path segments between the repo root and the
    report's own directory (``report/zernike_phase2amp/report.md`` -> 2). Computed
    rather than hardcoded so a report can be nested differently without its link
    silently breaking.
    """
    if not script:
        return ""
    return f"[`{script}`]({'../' * depth}{script})"


def provenance_block(key: str, depth: int) -> str:
    """Render the ``> 生成脚本`` header for report ``key``.

    ``depth`` again is the report's directory depth below the repo root. Raises
    ``KeyError`` for an unregistered report rather than emitting a block with no
    script in it -- a report with no provenance is the bug this module exists to
    prevent, so it must not be silently produced.
    """
    prov = REPORTS[key]
    lines = []
    if prov.script:
        lines.append(f"> **生成脚本**: {script_link(prov.script, depth)}")
        if prov.command:
            lines.append(f"> **复现命令**: `{prov.command}`")
    else:
        lines.append("> **生成脚本**: 人工撰写，无生成脚本")
    for extra in prov.extra_scripts:
        lines.append(f"> **数据/关联脚本**: {script_link(extra, depth)}")
    lines.append(f"> **运行环境**: {prov.environment}")
    if prov.note:
        lines.append(f"> **说明**: {prov.note}")
    return "\n".join(lines)


def provenance_block_for(key: str, *, depth: int | None = None) -> str:
    """Fenced provenance block for ``key``, or ``""`` when it is unregistered.

    Deprecated in favour of :func:`insert_header`, which also decides *where* the
    block goes. Kept because "give me the block text" is a reasonable thing to
    want on its own.
    """
    if key not in REPORTS:
        return ""
    if depth is None:
        depth = len(Path(key).parts) - 1
    return f"{START_MARKER}\n{provenance_block(key, depth)}\n{END_MARKER}"


def _strip_header(text: str) -> str:
    while START_MARKER in text and END_MARKER in text:
        head, _, rest = text.partition(START_MARKER)
        _, _, tail = rest.partition(END_MARKER)
        text = head + tail.lstrip("\n")
    return text


def insert_header(text: str, key: str, *, depth: int | None = None) -> str:
    """Place the provenance header for ``key`` into a report body, idempotently.

    **This is the single implementation of the placement rule.** Both
    :mod:`scripts.sync_report_provenance` and the writers (report generators, the
    ``TestReport`` harness) call it, because a second placement rule is a second
    opinion: the writer would put the block above the ``# title`` while the sync
    pass moved it below, and ``--check`` would then report drift forever on a file
    that is actually correct.

    Placement, in order of preference:

    1. directly under a level-1 title (the common case);
    2. under a deeper heading (``## 实验目标``), which plays the title role;
    3. at the very top, when the document has no heading at all -- leading blank
       lines are dropped first so repeated calls do not accumulate them.
    """
    block = provenance_block_for(key, depth=depth)
    if not block:
        return text
    lines = _strip_header(text).splitlines()
    anchor = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), None)
    if anchor is None:
        anchor = next(
            (i for i, ln in enumerate(lines) if ln.lstrip().startswith("#")), None
        )
    if anchor is None:
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join([block, ""] + lines).rstrip("\n") + "\n"
    rest = lines[anchor + 1 :]
    while rest and not rest[0].strip():
        rest.pop(0)
    return "\n".join(lines[: anchor + 1] + ["", block, ""] + rest).rstrip("\n") + "\n"


def index_rows() -> list[tuple[str, str, str, str]]:
    """``(report, script, environment, note)`` rows for every known report."""
    rows = []
    for key in sorted(REPORTS):
        prov = REPORTS[key]
        shown = prov.script or "人工撰写"
        if prov.extra_scripts:
            shown += " (+" + ", ".join(p.rsplit("/", 1)[-1] for p in prov.extra_scripts) + ")"
        rows.append((key, shown, prov.environment, prov.note))
    return rows


_INDEX_HEADER = """\
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

"""


def render_index() -> str:
    """Render the full ``report/README.md`` body from :data:`REPORTS`.

    A registered report whose file is **not on disk** is listed in the
    "not yet generated" section instead of the link table: the registry is
    allowed to name a report before it has been produced (that is the normal
    state for anything needing hardware), but a link to a missing file is a dead
    link, and the guard test checks every link in this file.
    """
    out = [_INDEX_HEADER, "| 报告 | 生成脚本 | 运行环境 | 说明 |", "|---|---|---|---|"]
    pending: list[tuple[str, ReportProvenance]] = []
    for report, script, env, note in index_rows():
        prov = REPORTS[report]
        if not Path(report).is_file():
            pending.append((report, prov))
            continue
        rel = report[len("report/") :]
        out.append(f"| [`{rel}`]({rel}) | `{script}` | {env} | {note} |")
    out.append("")
    out.append("## 尚未生成产物的报告")
    out.append("")
    out.append(
        "下列报告已在 `REPORTS` 登记但产物尚未提交（需要硬件，或尚未跑过），"
        "因此没有可点击的文件；生成脚本与复现命令见上表同条目："
    )
    out.append("")
    for report, prov in pending:
        out.append(f"- `{report[len('report/'):]}` ← `{prov.script or '人工撰写'}` ({prov.environment})")
    if pending:
        out.append("")
    return "\n".join(out)
