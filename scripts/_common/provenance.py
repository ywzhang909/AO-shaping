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
    "report/shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（读取已保存的 Santec/Daheng 实验数据）",
        note="三形状从零探测、物理参数微调、GS 与 CCD 锚定闭环复现手册；图片和参数 CSV 由离线脚本生成。",
        extra_scripts=("scripts/slm/generate_multishape_reproduction_figures.py",
                       "src/ao_shaping/tools/slm/transfer/workflow.py",
                       "src/ao_shaping/tools/slm/transfer/artifacts.py",
                       "src/ao_shaping/algorithm/signal_processing/physical_forward_fit.py",
                       "src/ao_shaping/algorithm/signal_processing/multishape_objective.py",
                       "scripts/slm/bootstrap_transfer_checkpoint.py",
                       "scripts/slm/transfer_multishape_experiment.py",
                       "scripts/slm/fit_transfer_multishape.py", "scripts/slm/validate_transfer_shapes.py",
                       "scripts/slm/calibrate_transfer_blur.py"),
    ),
    "report/shape_trials/multishape_capability/diagnosis_and_transfer_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM #1 + Daheng CCD）",
        note="原始 CCD 区域复算、载波几何纠正、物理参数迁移拟合和多形状配对实验；按时间追加记录。",
        extra_scripts=("scripts/slm/transfer_multishape_experiment.py", "scripts/slm/fit_transfer_multishape.py",
                       "scripts/slm/validate_transfer_shapes.py", "scripts/slm/summarize_transfer_shapes.py",
                       "scripts/slm/capture_transfer_response.py", "scripts/slm/analyze_transfer_response.py",
                       "scripts/slm/calibrate_transfer_blur.py", "scripts/slm/analyze_transfer_direction.py"),
    ),
    "report/zernike_coeff2amp/zernike_coeff_report/report.md": ReportProvenance(
        script="scripts/ml/generate_zernike_coeff_report.py",
        command="python scripts/ml/generate_zernike_coeff_report.py",
        environment="离线",
        note="系数->远场前向网络: 数据/归一化闸门/18折文件级 + 5折目标级交叉验证/单折完整训练",
        extra_scripts=("scripts/ml/sweep_coeff_models.py",),
    ),
    "report/zernike_coeff2amp/asm_vs_fraunhofer_report/asm_vs_fraunhofer.md": ReportProvenance(
        script="scripts/oopao/generate_asm_vs_fraunhofer_report.py",
        command="python scripts/oopao/generate_asm_vs_fraunhofer_report.py",
        environment="离线",
        note="角谱法与夫琅禾费 FFT 的分界: 2f 实验系统实测 ASM 返回传播光束而非远场; 含 Parseval/限带/平移约定与 GS 半阵列移位",
        extra_scripts=("tests/ao_shaping/algorithm/test_angular_spectrum.py",),
    ),
    "report/inverse_shaping_unet/inverse_shaping_report/report.md": ReportProvenance(
        script="scripts/ml/generate_inverse_shaping_report.py",
        command="python scripts/ml/generate_inverse_shaping_report.py",
        environment="离线",
        note="逆向光束整形 U-Net 在实验语料上的首个结果: 语料快照固定/预处理决策/"
        "平场退化/6 epoch 训练/按文件配对统计评估/pred-vs-true 可视化; "
        "输入为 ml.shaping.train + ml.shaping.evaluate 的产物",
    ),
    "report/inverse_shaping_unet/inverse_shaping_model_report/model.md": ReportProvenance(
        script="scripts/ml/generate_inverse_shaping_model_report.py",
        command="python scripts/ml/generate_inverse_shaping_model_report.py",
        environment="离线",
        note="PhasorUNet 模型详解: 逐阶段形状/参数量由脚本现场 introspection 真实模型得到; "
        "含 phasor 表示理由/unit_phasor/物理前向 FOV 分桶/损失/unit-vs-non-unit L1 下界实测",
    ),
    # -- beam shaping / benchmark suite -------------------------------------
    "report/benchmarks/cython_optimizer_report/cython_optimizer_performance.md": ReportProvenance(
        script="scripts/dm/generate_cython_optimizer_report.py",
        command="python scripts/dm/generate_cython_optimizer_report.py",
        environment="离线",
        note="读 src/calculators/benchmark_results.json；同目录两张 speedup 图由本脚本生成",
    ),
    "report/benchmarks/backend_kernels_report/backend_kernels_performance.md": ReportProvenance(
        script="scripts/generate_backend_kernels_report.py",
        command="python scripts/generate_backend_kernels_report.py",
        environment="离线",
        note="NumPy/Numba/CuPy 逐函数耗时；原先挂在 tests/ 的零断言用例里写回仓库（#162）",
    ),
    "report/benchmarks/device_less_full/beam_shaping_benchmark_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元权威网格 (3 算法 x 3 target)；gif/ 为 6 个演化动画",
    ),
    "report/centroid_test_visualization/centroid_test_visualization/centroid_test_report.md": ReportProvenance(
        script="scripts/slm/generate_centroid_test_visualization.py",
        command="python scripts/slm/generate_centroid_test_visualization.py",
        environment="离线",
        note="同目录 10 张 png 为本脚本产物",
    ),
    "report/centroid_test_visualization/centroid_test_visualization/centroid_test_report_duplicate_run.md": ReportProvenance(
        script="scripts/slm/generate_centroid_test_visualization.py",
        command="python scripts/slm/generate_centroid_test_visualization.py",
        environment="离线",
        note="同一次报告的更早一次运行，仅 Generated 时间戳不同；保留作对照，正式版见 centroid_test_report.md",
    ),
    # -- SLM 实验系统 ------------------------------------------------------------
    "report/diff_beam/diff_beam_frame_analysis/README.md": ReportProvenance(
        script="scripts/slm/diff_beam_frame_analysis.py",
        command="python scripts/slm/diff_beam_frame_analysis.py --run-dir data/diff_beam/run_<ts> --plot",
        environment="离线+硬件",
        note="正文为人工实测记录 (SLM200 + 大恒 CCD)；本脚本渲染逐帧产物",
    ),
    "report/slm/focus_plane_report/focus_plane_20261019.md": ReportProvenance(
        script="scripts/slm/generate_focus_plane_report.py",
        command="python scripts/slm/generate_focus_plane_report.py",
        environment="离线",
        extra_scripts=(
            "src/ao_shaping/tools/slm/panel/focus_probe.py",
            "scripts/slm/verify_focus_plane_hw.py",
        ),
        note=(
            "相机离开 2f 焦面后的菲涅尔补偿实测 (1/f≈+0.70 m⁻¹, f≈+1420 mm): 逐点自适应曝光的"
            "必要性 (曝光跨度达 96×)、峰值与整帧能量两项交叉验证、4 趟重复性与跨趟光强 7.3× 变化"
        ),
    ),
    "report/slm/diffraction_efficiency/hw_blaze_20261023/report.md": ReportProvenance(
        script="scripts/slm/gen_diffraction_efficiency_hw_report.py",
        command="python scripts/slm/gen_diffraction_efficiency_hw_report.py",
        environment="离线 (读已落盘的实测扫描 CSV)",
        extra_scripts=(
            "scripts/slm/calib_geometry_from_orders.py",
            "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py",
            "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/core.py",
        ),
        note=(
            "闪耀光栅实测衍射效率曲线 + 光斑演化 GIF: 在焦条件下器件满命令只有 1.89π, "
            "0 级仍剩 4.2% —— 2π 灰度外推约 1083, 超出面板上限 1023, 因此"
            "“调灰度把 0 级彻底搬空”在这台器件上不可达; 同时给出离焦把深度抬高到 2.2π 的实测对比"
        ),
    ),
    "report/slm/diffraction_efficiency/sim_nonideality_20261022.md": ReportProvenance(
        script="scripts/slm/generate_diffraction_efficiency_sim_report.py",
        command="python scripts/slm/generate_diffraction_efficiency_sim_report.py",
        environment="离线",
        extra_scripts=(
            "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/simulate.py",
            "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/core.py",
        ),
        note=(
            "闪耀光栅衍射效率标定工具的非理想因素仿真 (填充因子 / 相位 LUT / 面板静态相差 / "
            "光学离焦): 23 个工况里 10 个的欠调制-过调制判定与真值不符, 相位深度拟合偏差最大 "
            "+27% —— 工具的**相对效率曲线**可信, 拟合出的相位深度只在幂律器件上可信; "
            "并给出光斑宽度门禁 (相对在焦参照值的倍数) 的行为与失效边界"
        ),
    ),
    # NOTE: `report/slm/differentiable_opt/differentiable_opt_report.md` 的登记
    # 在**报告文件真正生成之后**再加 —— 索引里的链接必须可解析, 提前登记会让
    # `sync_report_provenance.py --check` 报索引漂移。生成它的脚本是
    # `scripts/slm/hw_differentiable_phase_opt.py`。
    "report/slm/differentiable_opt/differentiable_opt_report.md": ReportProvenance(
        script="scripts/slm/hw_differentiable_phase_opt.py",
        command="python scripts/slm/hw_differentiable_phase_opt.py --execute",
        environment="硬件",
        extra_scripts=(
            "src/ao_shaping/algorithm/signal_processing/differentiable_beam.py",
            "src/ao_shaping/tools/slm/panel/focus_probe.py",
        ),
        note=(
            "正向调优模型 + 反向可微优化 SLM 相位的硬件全流程记录: **开测前先脚本判在焦**"
            "(不在焦即退出 5, 一帧不测), 现场几何标定, 200 次纯模型反向传播的相位/梯度/损失史, "
            "上屏后的 CCD 远场, 模型与实测的对照 (强度相关系数 0.089、峰值偏移 [-6,-3] 远场采样、"
            "类 Strehl 0.450), 测量锚定的硬件闭环迭代 (实测损失 9.08e-4 → 6.50e-4, 梯度范数 "
            "0.12 → 0.79 仍在升), 以及**随机初值 + 强度 MSE 会掉进「把远场摊平」局部极小**"
            "与**相位 RMS 几乎不变**这两个负结果"
        ),
    ),
    "report/shape_trials/defocus_differentiable/defocus_differentiable_20261023.md": ReportProvenance(
        script="scripts/slm/generate_defocus_differentiable_report.py",
        command=(
            "python scripts/slm/generate_defocus_differentiable_report.py "
            "--run-dir data/debug/defocus_differentiable/20261023_101822"
        ),
        environment="硬件",
        extra_scripts=(
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
            "src/ao_shaping/optimizer/wfless/defocus_uniformity_shaping.py",
        ),
        note=(
            "离焦光路（相机未移动）下的可微均匀度优化：冻结物理前向 + 固定 60 px 掩膜 CV。 "
            "现场前置测量（面板斜坡→相机轴映射、去倾斜补偿残差 0.14/0.47 px/mm、可用曝光 "
            "1.6–32 ms、噪声地板 CV 1.1274±0.0055 ⇒ 3σ 0.0165）；C 阶段方向检验 Δ+ = −0.01942 / "
            "Δ− = +0.02132（两向超 3σ 且符号相反 ⇒ 模型梯度方向正确）；D 阶段逐块 A-B-B-A 重放 "
            "(B1/B2 差 0.0001、漂移 +0.0017)；报告含「模型预测 vs 实测」逐 step 指标曲线与并排 GIF。 "
            "**未决**：拟合 ramp_matrix 与现场轴映射互为转置，空间放置需标记测量裁决。"
        ),
    ),
    "report/shape_trials/defocus_differentiable/defocus_differentiable_20261023_r2.md": ReportProvenance(
        script="scripts/slm/generate_recalibration_report.py",
        command="python scripts/slm/generate_recalibration_report.py",
        environment="硬件",
        extra_scripts=(
            "scripts/slm/run_focus_recalibration.py",
            "scripts/slm/capture_defocus_forward_dataset.py",
            "scripts/slm/refit_model_after_optics_change.py",
            "scripts/slm/direction_test_after_optics_change.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
        ),
        note=(
            "光路变化（CCD 移到焦平面）后的重标定第二轮，与 20261023 那一轮**互不覆盖**： "
            "焦点 1783.5 mm/0.5607 m⁻¹/31.4× → inf/0.0000/1.00×（已在焦面）；重拟合把 "
            "defocus_diopter 从 −0.6452 改到 −0.1160（等效焦距 8623 mm，fit loss 1.02→0.14），"
            "零点移到 (1219, 964) 并重建采样窗；B± 方向检验：平场初值两向同号（失败），"
            "结构化探针初值**符号正确但幅度不足**（Δ+ −0.0148 / Δ− +0.0115，0.7–0.9σ）。"
        ),
    ),
    "report/slm/differentiable_opt/differentiable_opt_animation_20261022.md": ReportProvenance(
        script="scripts/slm/render_differentiable_opt_animation.py",
        command=(
            "python scripts/slm/render_differentiable_opt_animation.py "
            "--run-dir data/debug/differentiable_opt/20261022_213000"
        ),
        environment="离线",
        extra_scripts=(
            "scripts/slm/hw_differentiable_phase_opt.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_beam.py",
        ),
        note=(
            "可微相位优化实验的**过程动画**配套报告: 模型侧 200 次反向传播**离线复现**"
            "(同初值 + 同 seed, 与 history.csv 的最大相对偏差 1.06e-07, 最终相位差 6.29e-05 rad), "
            "GIF 展示「随机初值 → 远场演化 → 测量锚定实测帧」并逐帧标注损失/梯度范数, "
            "曲线图把 5 个测量锚定点叠加到模型侧损失/梯度范数曲线上 (拼接时间轴) 并标出 "
            "曝光自适应 2.0 → 72.0 ms; 附落盘数据 (每帧远场/损失/梯度)"
        ),
    ),
    "report/slm/differentiable_opt/focus_gate_verification_20261022.md": ReportProvenance(
        script="scripts/slm/hw_focus_gate_verification.py",
        command="python scripts/slm/hw_focus_gate_verification.py --execute",
        environment="硬件",
        extra_scripts=("scripts/slm/hw_differentiable_phase_opt.py",),
        note=(
            "「不在焦就停止测试」这条限制的**硬件验证**: 未加焦面补偿时零级光斑 "
            "FWHM 71.3 px (补偿后 15.2 px, 即离焦可补偿), 门禁在硬件上按设计工作"
            "——门限收紧到 15 px 以下即退出码 5, 运行目录里只有判定证据, "
            "一帧测量数据都没有"
        ),
    ),
    "report/slm/report2.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM 相位生成链路审计 + 矫正效果根因分析；数据由 zernike-matrix 闭环产生",
    ),
    "report/slm/zernike_response_matrix_report/report3.md": ReportProvenance(
        script="scripts/zernike_bench/generate_zernike_response_matrix_report.py",
        command="python scripts/zernike_bench/generate_zernike_response_matrix_report.py",
        environment="离线",
        note="2026-09-16 响应矩阵重标测试报告",
    ),
    "report/slm/daily/daily_2026-09-08.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM 硬件调试日报",
    ),
    "report/slm/daily/daily_2026-09-15.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="硬件实验日报；引用的两个 generate_* 脚本见当日记录",
        extra_scripts=(
            "scripts/zernike_bench/generate_zernike_response_matrix_report.py",
            "scripts/zernike_bench/generate_zernike_wfs_report.py",
        ),
    ),
    "report/slm/daily/daily_2026-09-16.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="硬件实验日报",
        extra_scripts=("scripts/zernike_bench/wfs_probe.py",),
    ),
    "report/slm/bench/bench_calibration_20261001.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="SLM #1 + 大恒 2f 实验系统实测标定记录",
    ),
    "report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md": ReportProvenance(
        script="scripts/slm/model_in_loop_hw_runbook.py",
        command="python scripts/slm/model_in_loop_hw_runbook.py --stage all --target-cam-px 40",
        environment="硬件",
        note="正向模型实验系统几何标定；runbook 采集 + 本文档结论",
    ),
    # NOTE: `docs/slm/model_in_loop_algorithm.md` is generated by
    # `scripts/slm/generate_model_in_loop_report.py` but is deliberately **not** listed
    # here: it is a usage/architecture description rather than the conclusion of a
    # measurement, so it belongs in `docs/`, and this registry renders
    # `report/README.md` with the assumption that every key is under `report/`.
    # It carries its own provenance header instead.
    "report/slm/bench_probe/report.md": ReportProvenance(
        script="scripts/slm/generate_bench_probe_report.py",
        command="python scripts/slm/generate_bench_probe_report.py",
        environment="离线",
        note="读实验系统 sweep 落盘的 npz/json/pkl，仪器可断电",
        extra_scripts=("src/ao_shaping/tools/zernike/sweep_probe.py",),
    ),
    "report/slm/zernike_linearity/linearity.md": ReportProvenance(
        script="scripts/zernike_bench/generate_zernike_linearity_report.py",
        command="python scripts/zernike_bench/generate_zernike_linearity_report.py",
        environment="离线",
        note="读 data/zernike_correction/raw_scan_*.json",
    ),
    "report/slm/zernike_response_matrix_report/report.md": ReportProvenance(
        script="scripts/zernike_bench/generate_zernike_response_matrix_report.py",
        command="python scripts/zernike_bench/generate_zernike_response_matrix_report.py",
        environment="离线",
        note="创建/分析/检测三段；figures/ 为本脚本产物",
    ),
    "report/slm/zernike_wfs_report/report.md": ReportProvenance(
        script="scripts/zernike_bench/generate_zernike_wfs_report.py",
        command="python scripts/zernike_bench/generate_zernike_wfs_report.py",
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
    "report/fouriergsnet_pipeline/fouriergsnet_pipeline_report/report.md": ReportProvenance(
        script="scripts/slm/generate_fouriergsnet_pipeline_report.py",
        command="python scripts/slm/generate_fouriergsnet_pipeline_report.py",
        environment="离线",
        note="5 项集成测试的结果记录",
    ),
    "report/fouriergsnet_pipeline/hardware_run_20261001.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="实验方形整形运行报告；由 slm-gsnet 命令产生",
        extra_scripts=("src/ao_shaping/runners/slm/gsnet_runner.py",),
    ),
    "report/fouriergsnet_pipeline/offline_training/report.md": ReportProvenance(
        script="scripts/slm/generate_gsnet_offline_report.py",
        command="python scripts/slm/generate_gsnet_offline_report.py",
        environment="离线",
        note="读 data/gsnet_train/run-*/summary.json",
    ),
    "report/fouriergsnet_sim/fouriergsnet_sim_report/report.md": ReportProvenance(
        script="scripts/slm/generate_fouriergsnet_sim_report.py",
        command="python scripts/slm/generate_fouriergsnet_sim_report.py",
        environment="离线",
        note="数据由 fouriergsnet_sim_train.py 产生；gifs/ 由 generate_slm_gsnet_sim_gif.py 追加",
        extra_scripts=(
            "scripts/slm/fouriergsnet_sim_train.py",
            "scripts/slm/generate_slm_gsnet_sim_gif.py",
        ),
    ),
    # -- 优化器基准 / PIB --------------------------------------------------
    "report/heuristic_pib/heuristic_pib_report/report.md": ReportProvenance(
        script="scripts/slm/generate_heuristic_pib_report.py",
        command="python scripts/slm/generate_heuristic_pib_report.py",
        environment="离线",
        note="7 种启发式 + PIB 目标；summary.csv 为本脚本产物",
    ),
    "report/strehl_benchmark/strehl_benchmark_report/report.md": ReportProvenance(
        script="scripts/slm/generate_strehl_benchmark_report.py",
        command="python scripts/slm/generate_strehl_benchmark_report.py",
        environment="离线",
        note="8 种算法 + Strehl 目标；交叉对比 heuristic_pib/summary.csv",
    ),
    "report/slm_pib_bench/pib_bench_report/EXPERIMENT_REPORT.md": ReportProvenance(
        script="scripts/slm/generate_pib_bench_report.py",
        command="python scripts/slm/generate_pib_bench_report.py",
        environment="离线",
        note="实验系统验收总报告；含 Pearson 迁移与参数标定两节",
        extra_scripts=(
            "scripts/slm/generate_shape_objective_comparison.py",
            "scripts/slm/measure_shape_sensitivity.py",
        ),
    ),
    "report/slm_pib_bench/pib_bench_report/report.md": ReportProvenance(
        script="scripts/slm/generate_pib_bench_report.py",
        command="python scripts/slm/generate_pib_bench_report.py",
        environment="离线",
        note="离线分析报告 (噪声地板 / 门控 / best-vs-sustained)",
    ),
    "report/slm_pib_bench/explore_delta/delta_scan.md": ReportProvenance(
        script="scripts/slm/explore_delta.py",
        command="python scripts/slm/explore_delta.py",
        environment="硬件",
        note="SPGD delta 扫描 (实验)",
    ),
    "report/slm_pib_bench/explore_delta/delta_scan_large.md": ReportProvenance(
        script="scripts/slm/explore_delta.py",
        command="python scripts/slm/explore_delta.py",
        environment="硬件",
        note="delta 大范围扫描 (实验)",
    ),
    "report/slm_pib_bench/explore_delta/delta_scan_small.md": ReportProvenance(
        script="scripts/slm/explore_delta.py",
        command="python scripts/slm/explore_delta.py",
        environment="硬件",
        note="delta 小范围扫描 (实验)",
    ),
    "report/slm_pib_heuristic_hw/slm_pib_heuristic_hw_report/report.md": ReportProvenance(
        script="scripts/slm/generate_slm_pib_heuristic_hw_report.py",
        command="python scripts/slm/generate_slm_pib_heuristic_hw_report.py",
        environment="硬件",
        note="实验系统基准 (Santec SLM-200 + 大恒 MER2-507)；figures/ + summary.csv 为本脚本产物",
    ),
    "report/slm_pib_heuristic_hw/measure_shape_sensitivity/sensitivity.md": ReportProvenance(
        script="scripts/slm/measure_shape_sensitivity.py",
        command="python scripts/slm/measure_shape_sensitivity.py",
        environment="硬件",
        note="整形目标函数灵敏度 (噪声地板) 实测；测量内核已下沉到 tools/slm/slm_snr_probe.py",
    ),
    "report/slm_pib_online/slm_pib_online_report/report.md": ReportProvenance(
        script="scripts/slm/generate_slm_pib_online_report.py",
        command="python scripts/slm/generate_slm_pib_online_report.py",
        environment="离线",
        note="在线回归套件验收报告 (读 data/debug/slm_pib_online/)",
    ),
    "report/slm_pib_online/shape_objective_comparison/objective_comparison.md": ReportProvenance(
        script="scripts/slm/generate_shape_objective_comparison.py",
        command="python scripts/slm/generate_shape_objective_comparison.py",
        environment="离线",
        note="三种整形目标函数的排序一致性 (Spearman)",
    ),
    "report/slm_pib_sim/slm_pib_sim_report/report.md": ReportProvenance(
        script="scripts/slm/generate_slm_pib_sim_report.py",
        command="python scripts/slm/generate_slm_pib_sim_report.py",
        environment="离线",
        note="2f-Fourier 数字孪生跑 slm-pib；数据由 slm_pib_sim_run.py 产生",
        extra_scripts=("scripts/slm/slm_pib_sim_run.py",),
    ),
    "report/slm_shaping_sim/slm_shaping_sim_report/report.md": ReportProvenance(
        script="scripts/slm/generate_slm_shaping_sim_report.py",
        command="python scripts/slm/generate_slm_shaping_sim_report.py",
        environment="离线",
        note="SLM+CCD 整形仿真: 三个 SLM 族 runner 的收敛与 DM 族烟测对照",
        extra_scripts=("scripts/slm/run_sim_bench.py",),
    ),
    "report/slm_pib_shape_50px/slm_pib_sim_report/report.md": ReportProvenance(
        script="scripts/slm/generate_slm_pib_sim_report.py",
        command="python scripts/slm/generate_slm_pib_sim_report.py",
        environment="离线",
        note="50px target 的仿真运行报告",
    ),
    "report/slm_zernike_shaping/slm_zernike_shaping_report/report.md": ReportProvenance(
        script="scripts/slm/generate_slm_zernike_shaping_report.py",
        command="python scripts/slm/generate_slm_zernike_shaping_report.py",
        environment="离线",
        note="读 slm_zernike_shaping optimizer 的 debug 产物",
    ),
    "report/pib_optimizer_functional_report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="PIB 优化器功能说明 (非实验测量)",
    ),
    # -- ML / 模型 ----------------------------------------------------------
    "report/models_analysis/models_report/report.md": ReportProvenance(
        script="scripts/rl/generate_models_report.py",
        command="python scripts/rl/generate_models_report.py",
        environment="离线",
        note="读 TensorBoard event 文件",
    ),
    "report/models_analysis/reward_analysis.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="传统 AO 仿真环境 reward 问题分析与解决",
    ),
    "report/zernike_phase2amp/zernike_amp_report/report.md": ReportProvenance(
        script="scripts/ml/generate_zernike_amp_report.py",
        command="python scripts/ml/generate_zernike_amp_report.py",
        environment="离线",
        note="9 张图；数据由 sweep_zernike_models.py 与 ml.zernike.train_amp 产生",
        extra_scripts=(
            "scripts/ml/sweep_zernike_models.py",
            "scripts/ml/compare_models_cv.py",
        ),
    ),
    "report/zernike_phase2amp/models_cv/unet_comparison.md": ReportProvenance(
        script="scripts/ml/compare_models_cv.py",
        command="python scripts/ml/compare_models_cv.py --analyse",
        environment="离线",
        note="grouped CV 对比 (physics / hybrid / unet) 的原始对照表",
        extra_scripts=("scripts/ml/compare_unet_baseline.py",),
    ),
    "report/zernike_r2_baseline/pod_ridge/report.md": ReportProvenance(
        script="scripts/ml/diagnose_pod_ridge.py",
        command=(
            "python scripts/ml/diagnose_pod_ridge.py --k 4 8 16 32 64 128 256 "
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
            "scripts/ml/diagnose_pod_ridge.py",
            "scripts/ml/verify_pod_ridge_canary.py",
            "scripts/ml/diagnose_r2_baseline.py",
            "scripts/ml/sweep_target_transform.py",
        ),
    ),
    # -- 硬件调试语料统计 (hwdataset) ----------------------------------------
    "report/hwdataset_corpus/hwdataset_corpus_report/report.md": ReportProvenance(
        script="scripts/ml/generate_hwdataset_corpus_report.py",
        command="python scripts/ml/generate_hwdataset_corpus_report.py",
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
        extra_scripts=("scripts/ml/analyze_hwdataset_distributions.py",),
    ),
    "report/sac_ao_20260109_205218/training_report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="SAC 训练报告 (2026-01-09)",
    ),
    # -- 仿真 / 其它 --------------------------------------------------------
    "report/zernike_farfield_sim/zernike_farfield_sim_report/report.md": ReportProvenance(
        script="scripts/zernike_bench/generate_zernike_farfield_sim_report.py",
        command="python scripts/zernike_bench/generate_zernike_farfield_sim_report.py",
        environment="离线",
        note="Noll 4-15 各模式远场形貌仿真；metrics.csv 为本脚本产物",
    ),
    "report/oopao_vs_numpy/oopao_vs_numpy_report/report.md": ReportProvenance(
        script="scripts/oopao/generate_oopao_vs_numpy_report.py",
        command="python scripts/oopao/generate_oopao_vs_numpy_report.py",
        environment="离线",
        note="像差 x 湍流 12 场景 x 2 后端对比",
    ),
    "report/oopao_impact/oopao_impact_report/report.md": ReportProvenance(
        script="scripts/oopao/generate_oopao_impact_report.py",
        command="python scripts/oopao/generate_oopao_impact_report.py",
        environment="离线",
        note="端到端 AO 环境下的后端影响",
    ),
    "report/miicam_auto_exposure/miicam_auto_exposure_probe/report.md": ReportProvenance(
        script="tests/ao_shaping/drivers/ccd/test_miicam_auto_exposure_report.py",
        command="python -m pytest tests/ao_shaping/drivers/ccd/test_miicam_auto_exposure_report.py -m hardware -s",
        environment="硬件",
        note=(
            "MiiCam auto_exposure 算法硬件验证 (2026-10-22 起替代 miicam_simulation "
            "仿真报告): 响应台阶扫描 / 脏数据核查 (帧序号-时间戳-重复帧-交叉切换) / "
            "六类目标场景的收敛-边界退出-有界不收敛行为; 报告正文与图由探针脚本生成, "
            "线性度 R2 等为诊断项不计入通过率"
        ),
        extra_scripts=("scripts/ccd/miicam_auto_exposure_probe.py",),
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
            "(对数强度梯度差分, --w-loggrad)、2 类早已存在、3 类否决, 附实测依据"
        ),
    ),
    "report/sim_tue_methods/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note="本目录用本仓库 `model_in_loop_shaping.py` 的 sim 复现 TU/e 可微物理仿真的四项能力（系统松弛 / 混合拟合 / 低参数 vs 深度 / 样本查询与不确定性采样），并与 `shape_trials` 的实验、正向模型标定做定量对照，为论文 7.7 节提供定量证据；T1/T2/T4 定量表由本目录 `t1_noise_relaxation.py`、`t2_mixed_fitting.py`、`t4_uncertainty_sampling.py` 生成（`.venv/bin/python3 report/sim_tue_methods/<脚本>.py` 可复现），T3 复用论文 5.4 节实测数据、无新脚本。全部 sim-only（SimFourierGSNetEnv，64×64，CPU，无硬件），T3 与实验对照用 `shape_trials` 实测 (Santec SLM #1 23020027、Daheng CCD，1064 nm 实际光路)；不触碰硬件写入路径",
    ),
    "report/simulation/bench_twin/bench_twin_fidelity_20261021.md": ReportProvenance(
        script="scripts/slm/generate_bench_twin_report.py",
        command=(
            "python scripts/slm/generate_bench_twin_report.py "
            "--run-dir data/debug/slm_model_in_loop/20261019_103208 "
            "--n-train 5 --maxiter 3 --popsize 5 --maxiter-fine 8 --seed 0"
        ),
        environment="离线",
        note=(
            "2f 实验系统数字孪生 (bench_twin) 的保真度验证与误差参数反演: 读 2026-10-19 那次 "
            "MIL 实验落盘的探针 npz / bench_geometry.json / run_config.json / recorder pkl, "
            "把 300×300 相机像元窗口换算到 115×115 模型采样点 (旋转象限由落盘 model_grid "
            "逐像素反推, 偏差 7.44e-08), 比较理想孪生与反演后孪生在训练集/验证集上的 "
            "SSIM / Pearson r / 标度无关能量比 (验证集: SSIM 0.1526→0.5116, 能量比 "
            "4.7044→1.1656), 并与 MIL 自己的正向模型在 1024² 网格上同口径对拍 (复现其落盘的"
            "验证集相关 0.3918); **焦面标度口径已统一**: 孪生按实测标度 2.614022 经 "
            "BenchTwinConfig.from_measured_scale 反推等效焦距 311.32 mm (解析/实测比 "
            "1.000000000), 标称 122.4060 mm 的解析标度与之差 2.54 倍仅作提醒; 反演走 "
            "bench_twin_calibrate.fit_twin 自己的两阶段路径, 过程 GIF 由拟合期间临时包一层"
            "模块级 twin_loss 取优化器真实访问过的候选序列"
        ),
        extra_scripts=("src/ao_shaping/runners/slm/model_in_loop_runner.py",),
    ),
    # -- 基准网格的逐 GIF 指标 (同一脚本按 单元 生成) ----------------------
    "report/benchmarks/device_less_full/gif/gs_circle/gs_circle_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 gs x circle 的逐 GIF 指标",
    ),
    "report/benchmarks/device_less_full/gif/gs_gaussian/gs_gaussian_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 gs x gaussian 的逐 GIF 指标",
    ),
    "report/benchmarks/device_less_full/gif/gs_square/gs_square_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 gs x square 的逐 GIF 指标",
    ),
    "report/benchmarks/device_less_full/gif/spgd-sim_circle/spgd-sim_circle_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 spgd-sim x circle 的逐 GIF 指标；area 已坍缩，勿据此比均匀度",
    ),
    "report/benchmarks/device_less_full/gif/spgd-sim_gaussian/spgd-sim_gaussian_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 spgd-sim x gaussian 的逐 GIF 指标；area 已坍缩，勿据此比均匀度",
    ),
    "report/benchmarks/device_less_full/gif/spgd-sim_square/spgd-sim_square_metrics.md": ReportProvenance(
        script="scripts/slm/generate_beam_shaping_benchmark_report.py",
        command="python scripts/slm/generate_beam_shaping_benchmark_report.py",
        environment="离线",
        note="9 单元网格中 spgd-sim x square 的逐 GIF 指标；area 已坍缩，勿据此比均匀度",
    ),
    # -- 正向模型 / loss 缺陷排查 ------------------------------------------
    "report/loss_defects/README.md": ReportProvenance(
        script=HANDWRITTEN,
        command="python -m ml.zernike.train_amp",
        environment="离线",
        note="前向模型/loss 缺陷排查结论（人工撰写）；同目录 *.json 为各探针面板",
        extra_scripts=("scripts/ml/sweep_far_field_padding.py",),
    ),
    "report/loss_defects/PROCESS.md": ReportProvenance(
        script=HANDWRITTEN,
        command="python scripts/ml/inverse_design_sim_eval.py",
        environment="离线",
        note="逆向整形 14 次尝试的完整过程记录（含 3 处被推翻的结论）",
        extra_scripts=(
            "scripts/ml/inverse_design_sim_eval.py",
            "scripts/ml/inverse_design_accuracy_ladder.py",
            "scripts/ml/inverse_design_restarts.py",
            "scripts/ml/inverse_restart_selection.py",
            "scripts/ml/inverse_objective_alignment.py",
            "scripts/ml/inverse_achievable_target.py",
            "scripts/ml/gs_vs_gradient_inverse.py",
            "scripts/ml/gs_plus_refinement.py",
            "scripts/ml/alignment_vs_accuracy.py",
            "scripts/ml/roi_robustness.py",
            "scripts/ml/restart_claim_robustness.py",
            "scripts/ml/freeform_vs_zernike.py",
        ),
    ),
    "report/loss_defects/forward_search_report/forward_search_report.md": ReportProvenance(
        script="scripts/ml/generate_forward_search_report.py",
        command="python scripts/ml/forward_search.py && python scripts/ml/generate_forward_search_report.py",
        environment="离线",
        note="正向模型改进尝试: 设备参数/椭圆loss/attention/U-Net/图像增强, 3 seed 配对",
        extra_scripts=(
            "scripts/ml/forward_search.py",
            "scripts/ml/inverse_strong_phase.py",
            "scripts/ml/probe_device_metadata.py",
            "scripts/ml/check_ellipse_term.py",
        ),
    ),
    "report/loss_defects/inverse_design_report/inverse_design_report.md": ReportProvenance(
        script="scripts/ml/generate_inverse_design_report.py",
        command="python scripts/ml/generate_inverse_design_report.py",
        environment="离线",
        note="逆向整形中文报告：正向/反向 pred vs true 对比图 + ROI 扫描 + 术语表",
        extra_scripts=("scripts/ml/freeform_vs_zernike.py",),
    ),
    # -- 方形整形实验记录 ---------------------------------------------------
    "report/shape_trials/defocus_differentiable/multishape_forward_plan_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="结合焦面重标定、旧离焦模型和闪耀响应证据制定器件响应矫正及多形状可微整形实验；没有新增采集。",
        extra_scripts=(
            "scripts/slm/direction_test_after_optics_change.py",
            "scripts/slm/refit_model_after_optics_change.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
            "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/core.py",
        ),
    ),
    "report/shape_trials/defocus_differentiable_test_plan_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="结合可微实验、离焦正向模型及曝光诊断制定探索性短块整形协议；本方案没有新增设备测量。",
        extra_scripts=(
            "scripts/slm/hw_differentiable_phase_opt.py",
            "scripts/slm/refit_defocus_uniformity_model.py",
            "scripts/slm/diagnose_defocus_sampling.py",
            "scripts/slm/validate_defocus_exposure.py",
            "scripts/slm/fit_forward_sampling_variants.py",
        ),
    ),
    "report/shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note="固定物理参数的采样收敛、像素积分与曝光偏置补测；训练集选择有效展宽及联合 CV 微调，逐步保留原始与派生数据。",
        extra_scripts=(
            "scripts/slm/diagnose_defocus_sampling.py",
            "scripts/slm/diagnose_defocus_radiometry.py",
            "scripts/slm/refit_defocus_uniformity_model.py",
            "scripts/slm/summarize_defocus_sampling.py",
            "scripts/slm/validate_defocus_exposure.py",
        ),
    ),
    "report/shape_trials/model_in_loop/physical_forward_defocus_live_20261022.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note="2026-10-22 当前功率与主动离焦下的聚焦、斜坡及未补偿/补偿配对相位采集和低参数物理模型验证；保留失败边界。",
        extra_scripts=(
            "scripts/slm/capture_defocus_forward_dataset.py",
            "scripts/slm/fit_defocus_forward_dataset.py",
            "scripts/slm/evaluate_defocus_forward_dataset.py",
            "scripts/slm/validate_defocus_exposure.py",
            "scripts/slm/summarize_defocus_forward_experiment.py",
            "src/ao_shaping/tools/slm/panel/alignment_probe.py",
        ),
    ),
    "report/shape_trials/physical_forward_defocus_plan_20261022.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="基于现有离焦、Zernike 与 model-in-loop 报告及工作区代码拟定实验方案；没有新增设备测量。",
        extra_scripts=(
            "scripts/slm/hw_gs_defocus_probe.py",
            "scripts/slm/model_in_loop_hw_runbook.py",
            "scripts/slm/model_in_loop_hw_ab.py",
        ),
    ),
    "report/shape_trials/experiment_summary_20261016.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "2026-10-09 ~ 2026-10-16 方斑整形全部报告的汇总: 不含任何新测量, "
            "全部数字来自已有报告与保存的实验产物离线复算; 含研究路径时间线、"
            "公式体系与结论, 供脱离硬件复现"
        ),
    ),
    "report/shape_trials/experiment_summary_20261022.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="人工撰写",
        note=(
            "2026-10-22 MIL + 离焦测试的联合分析: 把 10-21~22 的 MIL 损失审计、"
            "硬件离焦/多阶泽尼克测试与当天四轮 campaign 的失败现场串成一条因果链; "
            "不含任何新测量, 全部数字引自已落盘产物; 含 10 条规律、scripts/slm/ "
            "脚本盘点与去留、以及阶段 0~6 的后续实验计划"
        ),
        extra_scripts=(
            "scripts/slm/analyze_model_in_loop_loss.py",
            "scripts/slm/probe_optical_path_validity.py",
            "scripts/slm/model_in_loop_selfcal_bakeoff.py",
            "scripts/slm/model_in_loop_hw_ab.py",
        ),
    ),
    "report/shape_trials/gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md": ReportProvenance(
        script="scripts/slm/generate_gs_refine_fwdmodel_report.py",
        command="python scripts/slm/generate_gs_refine_fwdmodel_report.py",
        environment="离线+硬件",
        note=(
            "slm-gs-refine 三栏 GIF 第三栏的正向模型标定；本脚本离线重算标定前后对比、"
            "补零扫描与频率响应负结果；数字来自 data/debug/slm_gs_refine/ 的实测产物"
        ),
    ),
    "report/shape_trials/gs_refine_zernike/gs_refine_zernike_uniformity.md": ReportProvenance(
        script="scripts/slm/generate_gs_refine_zernike_report.py",
        command="python scripts/slm/generate_gs_refine_zernike_report.py",
        environment="离线",
        note=(
            "GS 逐像素相位 → Zernike 拟合 (多阶) → SPGD 细化 的阶数扫描；本脚本离线读取 "
            "run_gs_refine_zernike_sweep.py 产出的 summary.json + spgd_nmax*.csv，"
            "渲染拟合残差/CV/PIB 三图与阶数扫描表"
        ),
        extra_scripts=("scripts/slm/run_gs_refine_zernike_sweep.py",),
    ),
    "report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "GS → Zernike → SPGD 匀化实验的实验测试计划 (待执行, 非结论)；含现状 (仿真已确立的"
            "硬件无关结论 / 仿真天花板 / 合并上游 #140 的基线变化 / 复现性边界)、算法三阶段与"
            "锁定量、12 个原子化测试项 (H0~H12, 各带二元判据与止损)、执行顺序依赖图、"
            "交错重放等通用规程、数据记录规范与风险表。采集脚本见关联脚本"
        ),
        extra_scripts=(
            "scripts/slm/run_gs_refine_zernike_sweep.py",
            "scripts/slm/generate_gs_refine_zernike_report.py",
        ),
    ),
    "report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.issue.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "GS → Zernike → SPGD 匀化实验阶数扫描的 GitHub issue 正文（实验测试请求, 非实验结论）; "
            "交付拟合模块/扫描驱动/离线报告/8 阶仿真扫描报告 + 实验测试计划 H0~H12, "
            "记录仿真已确立的硬件无关结论与仿真天花板、合并上游 #140 的基线变化、复现性边界"
        ),
    ),
    "report/shape_trials/gui_closed_loop_shaping/gui_closed_loop_shaping_20261014.md": ReportProvenance(
        script="scripts/generate_gui_shaping_hw_report.py",
        command="python scripts/generate_gui_shaping_hw_report.py",
        environment="离线",
        note="GUI 闭环整形实验验证正文为人工实测记录 (SLM 22030108 + 大恒 FJB24112232)；"
        "图表与汇总由本脚本从 data/debug/slm_gs_refine/20261014_{100441,104105,143035}/ "
        "离线重算，三次会话的交错重放数值由 scripts/replay_gs_refine_hw.py 实验采集；"
        "原 fig4 中心剖面（flat 与 best 共用一条强度轴、峰值相差约 13 倍）已由 "
        "scripts/slm/generate_shape_trial_profiles.py 的逐帧 X/Y 光强截面取代",
        extra_scripts=(
            "scripts/slm/replay_gs_refine_hw.py",
            "scripts/slm/generate_shape_trial_profiles.py",
        ),
    ),
    "report/shape_trials/gs_refine/gs_issue_live_20261015.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "2026-10-15 GS 方法实验对照与形状验收预检 (Santec SLM #1 optionboardID 23020027 "
            "+ 大恒 CCD FJB24112232，1064 nm 实验光路)；固定 0.050 ms 曝光、同种子 87 下 "
            "plain/weighted/adaptive-feedback 五轮对照，仅用于本次光路状态诊断，不构成形状成功验收 "
            "(IoU 0.0684–0.0980，未形成实心方斑)；汇总数据由本地脚本从可信调试记录 "
            "data/debug/slm_gs_refine/20261015_issue_comparison/ 复算 (summary.json)，"
            "含 5 轮 GS 最佳相位与远场渲染图；X/Y 光强截面由 "
            "scripts/slm/generate_shape_trial_profiles.py 从各轮调试 PKL 的 _img 重新生成"
        ),
        extra_scripts=("scripts/slm/generate_shape_trial_profiles.py",),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_uniformity_live_20261016.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "2026-10-16/17 正向模型闭环方斑均匀度实验：固定曝光交错重放比较软边缘 CV "
            "目标与两轮热启动 (SLM 22030108 + 大恒 FJB24112232, 1064 nm)；"
            "并实测照明半径、据以把 GUI 侧栏的曝光与照明半径改为可探测。"
            "2026-10-17 追加节记录两次由实测数据揭出的实现缺陷 (半径用 edge/2 而非 "
            "edge−centre；曝光探测不校验确认帧是否截断) 及其修复"
        ),
        extra_scripts=(
            "src/ao_shaping/gui/slm/bench_settings.py",
            "src/ao_shaping/tools/slm/panel/beam_extent.py",
            "src/ao_shaping/tools/slm/panel/panel_locate.py",
            "src/ao_shaping/utils/image/exposure_control.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261019.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "2026-10-19 相机位置调整后正向模型校正与多形状整形 (Santec #1 22030108 + "
            "大恒 FJB24112232, 1064 nm)：聚焦/彗差组合校正使 FWHM 47.23→13.75 px、"
            "平均镜像误差改善约 30.4%；新几何标准相位族验证集相关性约 0.78–0.87，"
            "但全频随机相位验证集仅约 0.11–0.28 未过门槛，矩形/三角形/五角星均未通过"
            "独立模型门槛，无匀化验收结论；文内两张原标题为「X/Y 光强截面」的表实为相位与"
            "远场图，已由 generate_shape_trial_profiles.py 从记录的原始 CCD 帧补出真正的"
            "X/Y 光强截面（切点为扣背景后的光强质心）"
        ),
        extra_scripts=(
            "scripts/slm/model_in_loop_auto.py",
            "scripts/slm/model_in_loop_hw_runbook.py",
            "scripts/slm/generate_shape_trial_profiles.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_hw_ab_20261021.md": ReportProvenance(
        script="scripts/slm/generate_model_in_loop_hw_ab_report.py",
        command=(
            "python scripts/slm/generate_model_in_loop_hw_ab_report.py "
            "--session data/debug/model_in_loop_hw_ab/20261021_172236 --stamp 20261021"
        ),
        environment="离线+硬件",
        note=(
            "2026-10-21 model-in-loop 实验系统 A/B campaign 的离线报告 (Santec #1 22030108 + "
            "大恒 FJB24112232, 1064 nm)：验证集门槛不动 (0.5) vs 强制放开 (0) 两个变体实测对比 —— "
            "两个变体结局相同 (bake_off_rejected, 整形未打赢平场) ⇒ **本轮验证集门槛不是限制项**, "
            "10-19 那条「门槛按构造不可达」未被沿用; 含当天曝光窗口、sweep 几何标定 "
            "(相关性 0.7188)、全板定位扫描证据、逐变体实测探针与模型预测并排的 GIF, "
            "以及三个拦路缺陷与处置 (曝光上限估计被噪声底毒化 / 全板定位严格模式在噪声底上"
            "误拒 / 驱动改曝光后的陈旧帧)"
        ),
        extra_scripts=(
            "scripts/slm/model_in_loop_hw_ab.py",
            "scripts/slm/diagnose_light_level.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_record_20261021_172309.md": ReportProvenance(
        script="scripts/slm/generate_model_in_loop_record_report.py",
        command=(
            "python scripts/slm/generate_model_in_loop_record_report.py "
            "--run-dir data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/"
            "slm_model_in_loop/20261021_172309 --stamp 20261021_172309"
        ),
        environment="离线",
        note=(
            "2026-10-21 baseline_a 变体的 MIL 单次运行记录 (离线读盘, 不碰设备)："
            "Step A 像差系数 + 信赖域截断、Step B 600 步轨迹、逐探针实测/预测远场与"
            "相关/SSIM、逐轮 CV/EE/IoU、bake_off_rejected 结论 (整形 0.3451 未打赢"
            "平场 0.4025, 落盘相位为零相位平场)；过程动画每帧给相位/实测/预测远场"
            "(叠目标掩模轮廓) + 差值 + 正向预测模型 metrics + 目标 metrics"
        ),
        extra_scripts=(
            "scripts/slm/model_in_loop_hw_ab.py",
            "scripts/slm/generate_shape_trial_profiles.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261016.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "2026-10-16 正向模型闭环 Step B 匀化权重与 Step A 重拟合消融 "
            "(SLM 22030108 + 大恒 FJB24112232, 1064 nm，另有 CPU 仿真)；提高 Step B "
            "均匀度权重未改善方斑 (固定 0.200 ms 交错重放平均 CV 0.8027 升至 0.8989、"
            "EE 0.6919 降至 0.5177)，Step A=0 只有有代价的小幅 CV 趋势 "
            "(CV 0.8028 降至 0.7887，同时 EE 0.7071 降至 0.5888)，两者均被复合质量分拒绝；"
            "结论为保留默认 Step A=80 与 0.4/0.6 权重"
        ),
        extra_scripts=(
            "scripts/slm/check_flat_stability.py",
            "scripts/slm/replay_model_in_loop_config.py",
            "scripts/slm/generate_shape_trial_profiles.py",
            "src/ao_shaping/tools/slm/model_in_loop_replay.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261015.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "2026-10-15 正向模型闭环实验预检 (feat/r71-r84-closed-loop worktree; "
            "SLM 22030108 + 大恒 FJB24112232, 1064 nm)；面板宽扫因中途光路调整而作废，"
            "固定 0.020 ms 曝光平场短复测 21 s 内 ROI 能量漂移 0.52%；未下发任何新形状，"
            "恢复实验须先重扫面板入射位置"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/panel/panel_locate.py",
            "scripts/slm/check_flat_stability.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261014.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="2026-10-14 正向模型闭环实验复核（exp/model-in-loop-20261014 worktree）；"
        "随机相位面板扫描定光斑位置 (960,670)，传播几何沿用 20261013 的 bench_geometry.json，"
        "本日未重新独立标定其尺度/入射振幅/面板配准；图内 X/Y 光强截面由 "
        "generate_shape_trial_profiles.py 从记录的原始 CCD 帧重新生成",
        extra_scripts=("scripts/slm/generate_shape_trial_profiles.py",),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="2026-10-13 正向模型闭环整形实验记录；Step A (Zernike 校准 + NNLS) → Step B (SPGD + Adam) 双路径架构；images/ 动画由 generate_gif_from_pkl.py 渲染",
        extra_scripts=(
            "scripts/slm/generate_gif_from_pkl.py",
            "scripts/slm/generate_shape_trial_profiles.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_next_steps_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="2026-10-13 正向模型闭环暂停后续步骤记录",
    ),
    "report/shape_trials/model_in_loop/model_in_loop_pause_handoff_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note="2026-10-13 正向模型闭环暂停交接记录；当前状态、证据、工程变更摘要",
    ),
    "report/shape_trials/gs_refine/gsnet_hybrid_offline_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note="2026-10-13 GSNet 混合瞳面相位离线验证；8 阶 Zernike + 可训练残差参数化合成验证",
    ),
    "report/shape_trials/gs_refine/sim_metrics_waves.md": ReportProvenance(
        script=HANDWRITTEN,
        command=(
            "python -m pytest tests/ao_shaping/utils/test_beam_metrics.py "
            "tests/ao_shaping/algorithm/signal_processing/test_square_shaping.py "
            "tests/ao_shaping/optimizer/wfless/test_slm_model_in_loop.py "
            "tests/ao_shaping/optimizer/wfless/test_slm_model_in_loop_loop_invariants.py "
            "tests/ao_shaping/optimizer/wfless/test_adaptive_trust_region.py "
            "tests/ao_shaping/display/test_slm_model_in_loop_display.py "
            "tests/ao_shaping/ml/shaping/test_gs_reference.py "
            "tests/ao_shaping/ml/zernike/test_cv_harness.py "
            "tests/ao_shaping/ml/zernike/test_promotion_gate.py "
            "tests/ao_shaping/scripts/test_report_provenance.py"
        ),
        environment="离线",
        note=(
            "五波交付记录 (规范化 Strehl/M² 指标 / GS 金标 / 验证集探针 K 折 / "
            "强度域开关 + 晋升门 / 自适应信赖域), 重点是「默认路径逐位不变」的钉法与"
            "已知的既有基线失败; 两次真实测量见同目录 model_in_loop_cv.md 与 "
            "gs_golden_reference.md"
        ),
    ),
    "report/shape_trials/online_recal_efficiency/online_recal_efficiency.md": ReportProvenance(
        script="scripts/slm/generate_online_recal_efficiency_report.py",
        command="python scripts/slm/generate_online_recal_efficiency_report.py",
        environment="离线",
        note=(
            "光路变化后在线重标定的效率分析。成本模型与常量全部由源码现场推导"
            "(dataclasses/inspect 读默认值 + 扫描源码树统计重复字面量 + 调用 "
            "default_sweep_points() 枚举点数), 故不随代码漂移; §6 文献数字为人工核实; "
            "§7 为提案, 本报告不含任何新测量"
        ),
    ),
    "report/shape_trials/fast_recal_adapter/fast_recal_adapter.md": ReportProvenance(
        script="scripts/ml/exp_fast_recal_adapter_report.py",
        command=(
            "python scripts/slm/exp_fast_recal_adapter.py --seeds 6 --frames 1 2 4 8 "
            "&& python scripts/ml/exp_fast_recal_adapter_report.py"
        ),
        environment="离线",
        note=(
            "光路变化后少参数在线适配实验 (14 低阶 Zernike / 2 几何标量 / 冻结 CNN 三组"
            "互不相交参数); 三类扰动 (低阶像差 / 高阶像差对照 / 几何漂移) x 6 seed 配对, "
            "sign-flip 精确检验; 判据只用 corr/R2/光斑域, 不用 MSE/PSNR/SSIM; "
            "数据为仿真, 非实验系统实测"
        ),
        extra_scripts=("scripts/slm/exp_fast_recal_adapter.py",),
    ),
    "report/shape_trials/gs_golden_reference/gs_golden_reference.md": ReportProvenance(
        script="scripts/slm/generate_gs_golden_reference.py",
        command="python scripts/slm/generate_gs_golden_reference.py",
        environment="离线",
        note=(
            "GS 金标参照 (模拟上界) 与平相位的对照 + GS 迭代预算收敛曲线; "
            "相位由 ml.shaping.gs_reference 委托 canonical gs_shape 产出, 远场用 canonical "
            "forward_intensity, 指标 (含 Strehl / 相对 M²) 全取自 "
            "ao_shaping.utils.image.beam_metrics; GS 是对照而非基线 —— 实验记录为 GS 输给平相"
        ),
    ),
    "report/shape_trials/model_in_loop_cv/model_in_loop_cv.md": ReportProvenance(
        script="scripts/slm/generate_model_in_loop_cv_report.py",
        command="python scripts/slm/generate_model_in_loop_cv_report.py",
        environment="离线",
        note=(
            "model-in-loop 验证集探针的 K 折交叉验证 (sim 后端, 无设备): 固定种子偏移 vs "
            "轮换划分的配对对比; 经 probe_split_hook / holdout_on_sim 两个默认关闭的 "
            "开关驱动真实 optimize_slm_model_in_loop, 不复刻 Step A; "
            "统计量全部取自 ml.zernike.cv_harness 与 ml.zernike.eval_stats"
        ),
    ),
    # -- 附属记录 ------------------------------------------------------------
    "report/fouriergsnet_pipeline/TODO.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="实验系统光路验证待办清单 (设备 + 激光就绪后执行)",
    ),
    "report/fouriergsnet_pipeline/next_attempt_plan.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note="GS 家族整形下一步尝试方案 (slm-gs-refine × FourierGSNet, 2026-10-12)",
    ),
    "report/slm/zernike_response_matrix_report/capture.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="响应矩阵采集现场截图记录 (image/capture/)",
    ),
    # -- loss 算法对比 (compare_loss_algorithms.py 的 --out 家族) -----------
    "report/loss_algorithms_smoke/loss_algorithms/report.md": ReportProvenance(
        script="scripts/ml/compare_loss_algorithms.py",
        command="python scripts/ml/compare_loss_algorithms.py --quick --out report/loss_algorithms_smoke",
        environment="离线",
        note="测试用例跑 --quick 生成的冒烟版本",
    ),
    # -- shape trials ---------------------------------------------------------
    "report/shape_trials/model_in_loop/uncertainty_probe_hw_20261016.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "commit 080e0ac9 (issue #134 R-86) 不确定性采样探针验收；"
            "线上默认 bandlimit_sigma=1.5 下机制不成立（闭环 8 组配对种子 "
            "p=0.4219、机制 0/10 种子胜），关掉带限后闭环 8/8 改善、p=0.0078"
            "（n=8 下限），但实验复现显示无带限探针只把 53% 能量送进相机窗口，"
            "故结论为暂不进闭环、需先把 U(p) 裁到相机窗口"
        ),
        extra_scripts=(
            "scripts/slm/verify_uncertainty_probes_sim.py",
            "scripts/slm/verify_uncertainty_probes_hw.py",
        ),
    ),
    "report/shape_trials/gs_refine/gs_uniformity_live_20261015.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "2026-10-15 GS 相位匀化实验对照 (Santec SLM #1 23020027 + 大恒 "
            "FJB24112232, 1064 nm)；圆周高斯平滑四档均降低面板梯度但未改善"
            "剔零级 CV；纯 CV 闭环在独立交错重放中将剔零级 CV 由 0.72313 "
            "降至 0.68400 (5.41%)，面板梯度 RMS 微升"
        ),
        extra_scripts=("scripts/slm/gs_uniformity_probe.py",),
    ),
    "report/shape_trials/gs_refine/zero_order_blaze_20261015.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "2026-10-15 离轴闪耀载波 (blaze carrier) 的物理与接线约定；实测"
            "位移标度约 7400/P 相机像素，面板与 CCD 轴近似交换；CLI 的 "
            "--blaze-period-px/--blaze-direction 与 GUI 载波选项由此而来；"
            "本轮只读枚举未下发相位，故未声称零级隔离率或衍射效率已改善"
        ),
    ),
    "report/shape_trials/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="方斑整形实验对比试验索引；各次试验的目录与结论由此汇总",
    ),
    "report/shape_trials/gs_refine/gs_refine_live_20261012.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="GS 重构后实验调试正文为人工实测记录 (SLM 23020027 + 大恒 CCD)；"
        "交错重放入口 scripts/slm/replay_gs_refine_hw.py 仅离线部分经测试",
    ),
    "report/shape_trials/gs_defocus_sim/gs_defocus_sim_20261019.md": ReportProvenance(
        script="scripts/slm/generate_gs_defocus_sim_report.py",
        command="python scripts/slm/generate_gs_defocus_sim_report.py",
        environment="离线",
        note=(
            "CCD 焦后离焦下的 GS 匀化仿真 (纯仿真, 无硬件): 五组因果链的收益拆解、"
            "方案A/方案B 等价性、两个互相独立的可信度边界 (方案A 近似质量 ≤20 mm / "
            "几何回绕 ≤10 mm)、零级双分量 + 加权 GS、带限 ASM 有损不可逆、自适应框刷分陷阱; "
            "数字全部来自 scripts/slm/experiment_gs_defocus_offocus.py / "
            "experiment_gs_fresnel_vs_virtuallens.py / experiment_gs_defocus_sweep_visuals.py / "
            "experiment_gs_improved.py 已落盘的 CSV/JSON, 本脚本只渲染 5+1 张图与 3 个 GIF"
        ),
        extra_scripts=(
            "scripts/slm/experiment_gs_defocus_offocus.py",
            "scripts/slm/experiment_gs_fresnel_vs_virtuallens.py",
            "scripts/slm/experiment_gs_defocus_sweep_visuals.py",
            "scripts/slm/experiment_gs_improved.py",
        ),
    ),
    "report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261020.md": ReportProvenance(
        script="scripts/slm/generate_hw_defocus_zernike_report.py",
        command=(
            "python scripts/slm/generate_hw_defocus_zernike_report.py "
            "--groups-root data/debug/hw_groups_20261020 "
            "--groups-root data/debug/hw_zernike_sweep_20261020b "
            "--focus-evidence data/debug/bench_settings/focus_probe/slm1_20261020_093932.json"
        ),
        environment="硬件",
        note=(
            "实验系统离焦扫描 + 多阶泽尼克匀化测试 (Santec SLM #1 22030108 + 大恒 CCD "
            "FJB24112232, 1064 nm, 2f-Fourier 配置) 的**第一轮**: 实验系统常驻离焦的瞳面菲涅尔"
            "焦面矫正、噪声地板、逐 Δz 的定尺框/随斑框双口径、瞳面二次相位的转向分量、"
            "与 gs_defocus_sim_20261019.md 的仿真 G1 曲线对拍, 以及 8 阶泽尼克扫描的"
            "可排序性判定; 数字全部来自 scripts/slm/hw_gs_defocus_probe.py 与 "
            "scripts/slm/run_gs_refine_zernike_sweep.py 的落盘产物, 本脚本只离线重算"
            "逐帧光斑量并渲染 4 张图。**该轮的目标框沿用了模型相位的方斑边长 (44 px), "
            "与硬件相位的光斑不匹配 —— 口径问题在第二轮 (hw_defocus_zernike_20261021.md) "
            "用预先登记的孔径重测, 本轮数字保留作历史对照**"
        ),
        extra_scripts=(
            "scripts/slm/hw_gs_defocus_probe.py",
            "scripts/slm/run_gs_refine_zernike_sweep.py",
        ),
    ),
    "report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261021.md": ReportProvenance(
        script="scripts/slm/generate_hw_defocus_zernike_report.py",
        command=(
            "python scripts/slm/generate_hw_defocus_zernike_report.py "
            "--groups-root data/debug/hw_groups_20261021 "
            "--groups-root data/debug/hw_zernike_sweep_20261021 "
            "--focus-evidence-dir data/debug/hw_groups_20261021/focus_calibration "
            "--report-key report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261021.md"
        ),
        environment="硬件",
        note=(
            "实验系统离焦扫描 + 多阶泽尼克匀化测试的**第二轮 (复测)**: 光路恢复后重新标定"
            "焦面 (f_v = 1523.8 mm, 本轮 13:31 实测 1/f = +0.652 m^-1), 26 组每组一次独立"
            "调用且每组 <=10 分钟 (焦面标定 → Stage A 分 6 块 → 8 个逐阶组 → 离焦噪声组 → "
            "离焦扫描组 → 模型相位对照组 → 8 个各阶远场补拍组), 目标框改为**预先登记**的"
            "该相位自己的成形目标, 离焦扫描带瞳面二次相位的去倾斜自标定; 受阻检测已并入"
            "自动焦面标定 (焦面探测把「光路受阻」与「找不到焦面」分成退出码 3 / 2, "
            "两个采集脚本开跑前做同样的光路自检); 数字全部来自 "
            "scripts/slm/hw_gs_defocus_probe.py 与 scripts/slm/run_gs_refine_zernike_sweep.py "
            "的落盘产物, 本脚本只离线重算逐帧光斑量并渲染图"
        ),
        extra_scripts=(
            "scripts/slm/hw_gs_defocus_probe.py",
            "scripts/slm/run_gs_refine_zernike_sweep.py",
        ),
    ),
    "report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261022.md": ReportProvenance(
        script="scripts/slm/generate_hw_defocus_zernike_report.py",
        command=(
            "python scripts/slm/generate_hw_defocus_zernike_report.py "
            "--groups-root data/debug/hw_groups_20261022 "
            "--groups-root data/debug/hw_zernike_sweep_20261022 "
            "--focus-evidence-dir data/debug/hw_groups_20261022/focus_calibration "
            "--report-key report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261022.md"
        ),
        environment="硬件",
        note=(
            "实验系统离焦扫描 + 多阶泽尼克匀化测试的**第三轮**: 27 组每组一次独立调用且"
            "每组 <=10 分钟 (焦面标定 → Stage A 分 **7** 块 → 8 个逐阶组 → 离焦噪声组 → "
            "离焦扫描组 → 模型相位对照组 → 8 个各阶远场补拍组); 数据落在**新根** "
            "hw_groups_20261022 / hw_zernike_sweep_20261022, 不再覆盖第二轮的原始"
            "记录; 采集计划由 scripts/slm/build_hw_sweep_plan.py **生成** (不再手写 JSON: "
            "上一轮焦距一变要改 20 处)。Stage A 分块数由实测定的: 每块 540 s 墙钟预算只"
            "跑出 39 轮 ⇒ 6 块只到 ~234 轮, 够不到 250 轮的目标 ⇒ 7 块"
        ),
        extra_scripts=(
            "scripts/slm/hw_gs_defocus_probe.py",
            "scripts/slm/run_gs_refine_zernike_sweep.py",
            "scripts/slm/build_hw_sweep_plan.py",
        ),
    ),
    "report/shape_trials/sim_vs_hw/sim_vs_hw_20261022.md": ReportProvenance(
        script="scripts/slm/compare_sim_vs_hw.py",
        command=(
            "python scripts/slm/compare_sim_vs_hw.py --execute "
            "--params report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261022.json"
        ),
        environment="离线",
        note=(
            "数字孪生 vs 第三轮硬件实测的逐项对照: 仿真侧跑的是**同一批采集脚本**的 "
            "`--cam-type sim` 路径 (脚本文件未改, 只在运行时注入口径), 实测侧全部从 "
            "round3_20261022 的归档产物读出 ⇒ 两边的差就是**物理模型**的差"
        ),
        extra_scripts=(
            "scripts/slm/hw_gs_defocus_probe.py",
            "scripts/slm/run_gs_refine_zernike_sweep.py",
            "src/ao_shaping/tools/slm/panel/focus_probe.py",
        ),
    ),
    "report/shape_trials/multishape_capability/capability_square_20261023_182153.md": ReportProvenance(
        script="scripts/slm/generate_capability_report.py",
        command=(
            "python scripts/slm/generate_capability_report.py "
            "--run-dir data/debug/multishape_forward/square_20261023_182153"
        ),
        environment="离线（读设备运行落盘产物）",
        note=(
            "正方形目标（面积 7200 px²）0.8 ms 下的 5 块闭环 + 末尾 best/best+载波对比；模型方向未验证，报告只陈述实测"
        ),
        extra_scripts=(
            "scripts/slm/run_multishape_capability.py",
            "src/ao_shaping/utils/image/target/masks.py",
        ),
    ),
    "report/shape_trials/multishape_capability/capability_rectangle_20261023_191444.md": ReportProvenance(
        script="scripts/slm/generate_capability_report.py",
        command=(
            "python scripts/slm/generate_capability_report.py "
            "--run-dir data/debug/multishape_forward/rectangle08_20261023_191444"
        ),
        environment="离线（读设备运行落盘产物）",
        note=(
            "长方形 120×60（面积 7200 px²）0.8 ms 下的同一协议运行；首块 Δ −0.03737（3.45σ），载波使零点能量占比 3.23%→0.17%"
        ),
        extra_scripts=(
            "scripts/slm/run_multishape_capability.py",
            "src/ao_shaping/utils/image/target/masks.py",
        ),
    ),
    "report/shape_trials/multishape_capability/capability_triangle_20261023_191105.md": ReportProvenance(
        script="scripts/slm/generate_capability_report.py",
        command=(
            "python scripts/slm/generate_capability_report.py "
            "--run-dir data/debug/multishape_forward/triangle08_20261023_191105"
        ),
        environment="离线（读设备运行落盘产物）",
        note=(
            "等边三角形（面积 7200 px²）0.8 ms 下的同一协议运行；首块 Δ −0.03422（3.16σ），载波使零点能量占比 3.81%→0.20%"
        ),
        extra_scripts=(
            "scripts/slm/run_multishape_capability.py",
            "src/ao_shaping/utils/image/target/masks.py",
        ),
    ),
    "report/shape_trials/multishape_capability/capability_star_20261023_191213.md": ReportProvenance(
        script="scripts/slm/generate_capability_report.py",
        command=(
            "python scripts/slm/generate_capability_report.py "
            "--run-dir data/debug/multishape_forward/star08_20261023_191213"
        ),
        environment="离线（读设备运行落盘产物）",
        note=(
            "五角星（面积 7200 px²）0.8 ms 下的同一协议运行；首块 Δ −0.02662（2.46σ），载波使零点能量占比 3.76%→0.20%"
        ),
        extra_scripts=(
            "scripts/slm/run_multishape_capability.py",
            "src/ao_shaping/utils/image/target/masks.py",
        ),
    ),
    "report/shape_trials/multishape_capability/shape_comparison_20261023.md": ReportProvenance(
        script="scripts/slm/summarize_multishape_capability.py",
        command=(
            "python scripts/slm/summarize_multishape_capability.py "
            "--exposure-ms 0.8 --allow-unknown-exposure"
        ),
        environment="离线（只读已落盘 run.json，不重算测量量）",
        note=(
            "四种形状（同面积 7200 px²、同曝光 0.8 ms）的横向对比; 正方形那份 run.json "
            "没有 exposure_ms 字段（脚本改版前的产物），表内以 * 标注, 不静默当成已知值"
        ),
        extra_scripts=(
            "scripts/slm/run_multishape_capability.py",
            "scripts/slm/generate_capability_report.py",
        ),
    ),
    "report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261022.md": ReportProvenance(
        script="scripts/slm/compare_sim_vs_hw.py",
        command="python scripts/slm/compare_sim_vs_hw.py --execute",
        environment="离线",
        note=(
            "同一轮渲染出的**数据表** (逐项数字 + 判定 + 六类差距归因 + 调参建议); "
            "报告正文 `sim_vs_hw_20261022.md` 直接嵌入这张表 (共用同一个渲染函数)"
        ),
    ),
    "report/shape_trials/sim_vs_hw/sim_vs_hw_20261023.md": ReportProvenance(
        script="scripts/slm/compare_sim_vs_hw.py",
        command=(
            "python scripts/slm/compare_sim_vs_hw.py --execute "
            "--params report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261023.json "
            "--out report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261023.md "
            "--report report/shape_trials/sim_vs_hw/sim_vs_hw_20261023.md"
        ),
        environment="离线",
        note=(
            "V2.0 孪生 (ZeroOrderModel / SlmLutModel / CameraNonIdealModel 四个旋钮 "
            "全部开启) vs 第三轮硬件归档的逐项对照; 与 V1.0 的差异见 "
            "`compare_v1_v2_20261023.md`"
        ),
    ),
    "report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261023.md": ReportProvenance(
        script="scripts/slm/compare_sim_vs_hw.py",
        command=(
            "python scripts/slm/compare_sim_vs_hw.py --execute "
            "--params report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261023.json "
            "--out report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261023.md "
            "--report report/shape_trials/sim_vs_hw/sim_vs_hw_20261023.md"
        ),
        environment="离线",
        note=(
            "V2.0 轮次渲染出的**数据表**; 与报告正文 `sim_vs_hw_20261023.md` "
            "同一次运行产出 (--out 写表, --report 写正文)"
        ),
    ),
    "report/shape_trials/sim_vs_hw/compare_v1_v2_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note=(
            "V1.0 与 V2.0 两轮对照报告的人工比对分析: 硬件基准完全相同 "
            "(第三轮 round3_20261022 归档), 差异全部来自孪生模型的非理想旋钮; "
            "两侧数字分别摘自 `sim_vs_hw_20261022.md` 与 `sim_vs_hw_20261023.md`"
        ),
        extra_scripts=("scripts/slm/compare_sim_vs_hw.py",),
    ),
    "report/shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md": ReportProvenance(
        script="scripts/slm/generate_hw_gs_square_scans_report.py",
        command=(
            "python scripts/slm/generate_hw_gs_square_scans_report.py "
            "--run-root data/debug/hw_gs_square_scans/20261023_gs_square "
            "--retest-root data/debug/hw_gs_square_scans/20261023_retest_frozen"
        ),
        environment="离线+硬件",
        extra_scripts=(
            "scripts/slm/hw_gs_square_scans.py",
        ),
        note=(
            "GS 方形光斑三段扫描（边长 18 行 / Zernike 阶数 24 行 / 闪耀光栅 8 行）加激光功率变化后的"
            "复测（光栅 32/40/48、Zernike 17/18/19，同一脚本版本、每配置一个原子进程）。数字全部从落盘 "
            "measurements.csv 解析，表格渲染后回读逐单元自检；复测**推翻**了基线两处「异常」"
            "（光栅周期 40、Zernike n_max=18）——它们是整帧全黑的一次失败采集，不是物理异常；"
            "位移只有 measured_shift_y_px 可读，方向跨批可用但标度与 1/P 形状不可用"
            "（实测 α≈0.48 vs 预测 1.00），故「方框跟着预测走」需改为框固定 + 宽 ROI；"
            "复测与基线是两个功率 regime（20 ms vs 180 ms），绝对值不可横比。"
        ),
    ),
    "report/shape_trials/hw_defocus_zernike/artifacts/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "本系列的**输入契约**归档 (按轮): 两轮各自的分组记录 JSON/CSV、泽尼克阶数扫描的 "
            "summary.json 与 spgd_nmax*.csv、以及 before/after PNG (报告里 16 张泽尼克图的"
            "**直接来源**)、焦面标定证据 JSON; 逐帧 .npz 与相位 .npy 体积大不入库, 仍在 "
            "data/debug/。**第二轮那 11 个组的原始记录已被 10-21 15:57-16:01 第三轮首次"
            "尝试覆盖**, 归档的是现存的那一份, 详见正文"
        ),
        extra_scripts=(
            "scripts/slm/hw_gs_defocus_probe.py",
            "scripts/slm/run_gs_refine_zernike_sweep.py",
        ),
    ),
    "report/shape_trials/spgd_drift_cancel_loop/spgd_drift_cancel_loop_20261022.md": ReportProvenance(
        script="scripts/slm/spgd_drift_cancel_loop.py",
        command=(
            "python scripts/slm/spgd_drift_cancel_loop.py --epochs 32 "
            "--data-root <scratch>/full "
            "--intrinsic <scratch>/intrinsic/results.json"
        ),
        environment="离线",
        note=(
            "#144 第 1 项 (照明漂移对消) / #36 验收判据的数字孪生 A/B: 在 2f-Fourier "
            "孪生上跑真实 slm-pib SPGD 闭环, 对照 control / baseline / abba / "
            "flat_norm 四变体。孪生标定结论 (纯倍增增益对 shape 目标不可见, 须引入"
            "不随功率缩放的杂散底板 B 才能复现 bench 的亮度耦合) 也记在正文; "
            "``--recompute`` 可只从 pkl 重算指标重写报告 (噪声门零置数按优化器判据"
            "离线复算, pkl 不落盘 _gate 列)"
        ),
    ),
    "report/shape_trials/gs_refine/online_20261010.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="2026-10-10 在线复测正文为人工实测记录 (SLM 22030108 + 大恒 CCD "
        "FJB24112232)；逐帧产物由 main.py 各子命令的 --debug 产生，"
        "scripts/slm/compare_shape_debug.py 只渲染对照图；本目录这几次运行的调试 PKL "
        "未随仓库保留，因此其光斑图无法补出 X/Y 光强截面",
        extra_scripts=("scripts/slm/generate_shape_trial_profiles.py",),
    ),
}


#: Report directories that hold generated figures with no markdown of their own.
#: Kept separate from :data:`REPORTS` because there is no file to attach a header
#: to; the index lists them so the directory is still traceable to its script.
FIGURE_ONLY_DIRS: dict[str, ReportProvenance] = {
    "report/iterative_zernike_shaping": ReportProvenance(
        script="scripts/slm/generate_iterative_zernike_shaping_report.py",
        command="python scripts/slm/generate_iterative_zernike_shaping_report.py",
        environment="离线",
        note="GS 预整形 + 全频随机相位细化对比 (data.json + *.npy + png)",
    ),
    "report/pearson_gifs": ReportProvenance(
        script="scripts/slm/generate_pearson_pkl_gif.py",
        command="python scripts/slm/generate_pearson_pkl_gif.py",
        environment="离线",
        note="逐 epoch 的 CCD/相位同步 GIF",
    ),
    "report/simulation": ReportProvenance(
        script="scripts/rl/simulate_atmospheric_comparison.py",
        command="python scripts/rl/simulate_atmospheric_comparison.py",
        environment="离线",
        note="湍流对比图 + SPGD Zernike 调参 json",
        extra_scripts=("scripts/dm/tune_sim_spgd_zernike.py",),
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
            shown += (
                " (+"
                + ", ".join(p.rsplit("/", 1)[-1] for p in prov.extra_scripts)
                + ")"
            )
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
`离线` = 纯 numpy/torch 读产物，仪器可断电；`硬件` = 需要真实实验系统。

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
        out.append(
            f"- `{report[len('report/') :]}` ← `{prov.script or '人工撰写'}` ({prov.environment})"
        )
    if pending:
        out.append("")
    return "\n".join(out)
