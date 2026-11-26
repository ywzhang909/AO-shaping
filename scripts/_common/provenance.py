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
    "report/shape_trials/transfer_geometry_adaptation_20261124.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件采集 + 离线 CPU 拟合/评估（拟合与评估阶段不打开 SLM、CCD 或激光控制器）",
        note=(
            "数值由冻结评估脚本计算，图由同一评估 JSON 直接绘制。"
            "评估脚本 `data/debug/slm_transfer/20261124_transfer_geometry_v1/"
            "evaluate_geometry.py`，SHA256 "
            "`2cd9696b4bfd36360cca36d608042454a9185ee480dd7c2bcdf0bff7c0468d95`；"
            "绘图脚本 `data/debug/slm_transfer/20261124_transfer_geometry_v1/"
            "make_geometry_figures.py`；"
            "复现命令 `python data/debug/slm_transfer/20261124_transfer_geometry_v1/"
            "evaluate_geometry.py`。"
            "预注册 `docs/dev/transfer_geometry_adaptation_prereg_20261124.md`。"
            "本轮检验小幅全局 2×2 几何残差是否改善 transfer 正向预测；validation 与"
            "独立 192 px held-out 两道门均未全部通过，故**不支持启用几何适配，"
            "保留默认关闭**。人工撰写。"
        ),
    ),
    "report/shape_trials/transfer_psf_adaptation_20261124.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线 CPU 拟合与评估（拟合与评估阶段不打开 SLM、CCD 或激光控制器，数据由先前会话采集）",
        note=(
            "数值由冻结评估脚本计算，图由同一评估 JSON 直接绘制。"
            "评估脚本 `data/debug/slm_transfer/20261124_transfer_psf_v1/"
            "evaluate_psf.py`，SHA256 "
            "`828bf746757daad246ecaa8333d29e429e04bfa0e708a4b09908aff35f892a79`；"
            "绘图脚本 `data/debug/slm_transfer/20261124_transfer_psf_v1/"
            "make_psf_figures.py`，SHA256 "
            "`90d29c7d07d68d4c716a4112ef08649f1afad639870110ce8ad82eb6c6113a41`；"
            "复现命令 `python data/debug/slm_transfer/20261124_transfer_psf_v1/"
            "evaluate_psf.py`；训练命令见 §7。"
            "预注册 `docs/dev/transfer_psf_adaptation_prereg_20261124.md`。"
            "本轮只检验单一共享、静态、各向同性、有界 CCD 高斯 PSF；原 capture "
            "validation 门与独立 192 px capture 主门均通过，**支持将静态 PSF 适配作为"
            "显式 opt-in**；默认 `learn_psf=False` 保持不变。人工撰写。"
        ),
    ),
    "report/shape_trials/target_lock_hold_20261123.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "固定 CCD 目标保持·标定与保持物理实验（2026-11-23，SLM #1 1064 nm + Daheng CCD "
            "MER2-507-23GM NIR，cam_id=0）：两次实验（0.10/0.05 ms）标定均成功、保持均未达成，"
            "归因于标定口径错位（整帧 vs 窗口）/适配种子 [WinError 2]/起点光斑不在窗口，"
            "非真实物理漂移；并含倾斜注入幅度扫（≈0.628 px/峰谷-rad）与 registration "
            "等价性探针（整数平移逐点等价）。输入为 data/debug 磁盘证据，未据此完成物理验收。"
        ),
        extra_scripts=(
            "scripts/slm/target_lock_ondevice_20261123.py",
            "scripts/slm/target_lock_probe_20261123.py",
        ),
    ),
    "report/shape_trials/experiment_summary_20261123.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（仅回放已归档数据，不占用 SLM / CCD / 激光）",
        note=(
            "六项已落盘实验的离线有效性汇总: 效率匹配(3 门触发→不可判定)、T4/T5 动态解耦"
            "(30/30 ok, 2.196-2.542 fps)、#238 run1/run4 g2π fold(有效但不可定标)、"
            "#235 零级四法(边界通过 3.0px)、P2a/P2b 方向+强度双门(0/3, 不放行闭环)、"
            "S7b 轴失配三变体(CI 含 0, 无收益)。三分类: 有效 3 / 无效 2 / 不可判定 1; "
            "数字全部复核自已归档 analysis.json / summary.json / run_summary.json / "
            "paired_bootstrap_heldout.json 与既有报告; 本轮未实验、未写标定、未改双门、未发布 #238。人工撰写。"
        ),
    ),
    "report/shape_trials/multishape1200/wall_clock_budget_20261111.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note=(
            "#213 墙钟成本分解：repro1200_20261109 会话（2808.2 s、594 帧）；瓶颈为 "
            "LCOS settle 等待（帧间 gap 1212 s，AB 双峰 5-12 s/次）与模型/seed 构建 "
            "（transport_cv 首帧前约 130 s/shape）；优化器 71 次迭代仅占 0.02-0.03%，"
            "GPU 提速 12.58->7.88 ms/iter 全会话省 0.33 s。输入为磁盘存档，未占用设备。"
        ),
        extra_scripts=(
            "scripts/slm/analyze_session_time_budget.py",
            "src/ao_shaping/tools/kernels/bench_kernels.py",
        ),
    ),
    "report/shape_trials/optical_path_status_20261110.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "11-10 光路 A 下线前诊断状态日志：非饱和曝光阶梯、中心/焦点拟合冲突、"
            "证据保真限制与各 issue 预编写边界；含曝光单位更正补记（30 ms 改为 "
            "30 us，原不一致结论据阶梯数据撤销）；未发布新标定，未据此完成实验系统验收。"
        ),
    ),
    "report/shape_trials/shape_transition/shape_transition_timing_20261107.md": ReportProvenance(
        script="scripts/slm/generate_shape_transition_report_20261107.py",
        command="python scripts/slm/generate_shape_transition_report_20261107.py",
        environment="离线渲染（实验系统数据 2026-11-07：Santec #1 1064 nm + Daheng CCD）",
        note=(
            "随机起步 square 计时与失配取证；输入为 shape_transition_timing_20261106.py "
            "--execute 产物（shape_transition_20261107_randseed / 20261106_rerun）。"
            "§5 另引 anchored 实验系统闭环两种锚定语义（preanchored / raw）的 --execute 产物"
            "（shape_transition_20261107_anchored_pre / shape_transition_20261107_anchored_raw）；"
            "images/ 为本脚本渲染的 4 个实测 CCD GIF"
            "（ccd_square_randseed_20261107 / ccd_square_zerophase_20261106 / "
            "ccd_square_anchored_pre_20261107 / ccd_square_anchored_raw_20261107，"
            "标题含 device_log.json 抓帧时刻与相对首帧墙钟），"
            "figures/ 与 artifacts/mismatch_forensics_20261107.json 同为本脚本产物。"
        ),
        extra_scripts=("scripts/slm/shape_transition_timing_20261106.py",),
    ),
    "report/shape_trials/forward_model_polarity/forward_model_polarity_20261107.md": ReportProvenance(
        script="scripts/slm/generate_forward_model_polarity_report_20261107.py",
        command="python scripts/slm/generate_forward_model_polarity_report_20261107.py",
        environment="离线渲染（实验系统数据 2026-11-07：Santec #1 1064 nm + Daheng CCD）",
        note=(
            "物理预测模型极性核查与单模屏方向门禁：输入为 20261107_focus_scan/focus.json、"
            "defocus_forward_20261107_focus_r2(_fit)、issue234_*_screen_20261107 三组 ladder、"
            "issue235_parity_20261107/parity_registration_scan.json、"
            "loss_compare_20261107_phase_sign_m1/（#236 主变体 −1）与 gated_recurrent_20261107/（#239）"
            "的存档产物；figures/ 五张 PNG 为本脚本产物。报告本体不占用设备。"
        ),
    ),
    "report/shape_trials/loss_compare_20261107/loss_compare_20261107.md": ReportProvenance(
        script="scripts/slm/generate_loss_compare_report_20261107.py",
        command="python scripts/slm/generate_loss_compare_report_20261107.py",
        environment="离线渲染（实验数据 2026-11-07：Santec #1 1064 nm + Daheng CCD 成对 U/C 存档）",
        note=(
            "#236 拟合损失比较（phase_sign=−1 主变体 + +1 历史对照；基线复现 FAILED，结论未定）："
            "输入为 data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json 存档产物，"
            "表格数字全部渲染时读取、零硬编码。报告本体不占用设备。"
        ),
    ),
    "report/shape_trials/gated_recurrent_20261107/gated_recurrent_20261107.md": ReportProvenance(
        script="scripts/slm/generate_gated_recurrent_report_20261107.py",
        command="python scripts/slm/generate_gated_recurrent_report_20261107.py",
        environment="离线渲染（实验数据 2026-11-07：单模屏 ladder pass1/pass2 与 r2 存档，phase_sign=−1）",
        note=(
            "#239 门控循环残差头 对比择优（verdict inconclusive；4 变体×3 种子）："
            "输入为 data/debug/slm_transfer/gated_recurrent_20261107/results.json 存档产物，"
            "表格数字全部渲染时读取、零硬编码。报告本体不占用设备。"
        ),
    ),
    "report/shape_trials/phase_scope_20261107/phase_scope_20261107.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线回放存档数据（2026-11-07；未占用 SLM/相机/激光，未 import ao_shaping.drivers）",
        note=(
            "#235 phase_sign 作用域实验：四变体（command/aberration/aberration_focus/aberration_carrier）"
            "离线拟合与单模屏方向对照；command 精确复现存档 −1（Pearson 路径 Δd 4.7e-10）；"
            "屏方向四变体等价、无法区分作用域；aberration_focus 不稳定需复现；carrier 不可分离。"
            "输入为 data/debug/slm_transfer/phase_scope_20261107/ 存档产物。人工撰写。"
        ),
    ),
    "report/shape_trials/direction_diagnosis_20261112/direction_diagnosis_20261112.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线回放存档数据（2026-11-12；未占用 SLM/相机/激光，未 import ao_shaping.drivers）",
        note=(
            "可微整形物理预测模型 Jacobian 方向诊断合并报告：6 变体×3 幅值方向矩阵 + 7 变体归因"
            "（隔离照明包络/origin/defocus 贡献）。M3（离焦+照明包络）0.616/0.696/0.657 通过 ≥0.50 门禁；"
            "M0/M1 未过；背景/增益对方向度量精确不变；M3 通过是 origin+defocus+包络+gain 的联合自洽属性。"
            "输入为 defocus_forward_20261107_focus_r2_fit/forward_fit.json 与 issue234_*_screen_20261107 存档产物，"
            "产物 data/debug/slm_transfer/direction_variants_20261112/{results,attribution}.json。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/direction_model_variants_20261112.py",
            "scripts/slm/direction_attribution_20261112.py",
        ),
    ),
    "report/shape_trials/anchored_m3_20261116/anchored_m3_20261116.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-16 实验运行后离线整理；报告本体不占用设备）",
        note=(
            "M3（离焦+照明包络，grid 128 / FFT 512 / phase_sign −1.0）锚定闭环两次实验验收："
            "零相位起步实测核心 CV 6.08、随机起步 RMS1.5 实测核心 CV 5.01，均未达 ≤0.10；"
            "model_cv 与 measured_cv_window 同掩膜同网格分叉 ~7.5×，排除掩膜坐标错配与起步盆地，"
            "指向 FrozenDefocusForward 绝对强度表征误差（方向 cosine ≥0.50 是必要条件非充分条件）。"
            "输入为 shape_transition_timing_20261106.py --execute 产物"
            "（data/debug/slm_transfer/anchored_m3_20261116 / anchored_m3_rs15_20261116）与"
            " M3 frozen_parameters.json、20261112 方向诊断、20261107 锚定闭环报告。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/shape_transition_timing_20261106.py",
        ),
    ),
    "report/shape_trials/model_direction_hw_20261117/model_direction_hw_20261117.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-17 实验采集；报告本体不占用设备）",
        note=(
            "冻结模型 14 帧方向诊断（data/debug/square1000_model_diag_20261117_140741）、"
            "窗口对照（win0/win163）、M3 以实测峰为中心复核（m3_peak_centred_20261117.json）、"
            "刀口基线拟合（issue235_knife_baseline_fit_20261117.json）、差分刀口（diff_20261117_170647_v2/summary.json）"
            "与 panel_gray 反解前向一致性（issue235_panel_gray_inversion_20261117.json）、"
            "实测 panel 中心回灌的前向修正（m3_peak_centred_centerfix_20261117.json）。"
            "人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/square_1000_model_diagnostic_20261030.py",
            "scripts/slm/analyze_square1000_model_diagnostic_20261030.py",
            "scripts/slm/direction_peak_centred_20261117.py",
            "scripts/slm/issue235_knife_baseline_fit_20261117.py",
            "scripts/slm/issue235_beam_centre_differential_20261117.py",
            "scripts/slm/issue235_panel_gray_inversion_20261117.py",
        ),
    ),
    "report/shape_trials/fresh_checkpoint_direction_gate_20261118/fresh_checkpoint_direction_gate_20261118.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（仅回放归档帧与 checkpoint；未占用 SLM/CCD/激光）",
        note=(
            "比较 11/18 fit 在 epoch-matched 11/17 square1000 方向帧上的方向门，"
            "并对 11/07 单模屏进行跨 epoch hybrid 诊断及 11/18 强度窗口零步回放；"
            "fresh checkpoint 两个 amp 均 0/3 达到 0.50，hybrid 未通过既有每档 10/14 标准，"
            "held-out relative L2 0.735。结论：不存在已证明通过方向与绝对强度双门的 checkpoint，"
            "不进入硬件闭环验收。输入/输出位于 data/debug/direction_fresh_checkpoint_20261118、"
            "data/debug/hybrid_intensity_gate_20261118 与 data/debug/slm_transfer/axis_mismatch_validation_20261118。"
            "人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/direction_fresh_checkpoint_20261118.py",
            "scripts/slm/direction_fresh_checkpoint_zernike_20261118.py",
            "scripts/slm/build_direction_hybrid_checkpoints_20261118.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
        ),
    ),
    "report/shape_trials/offline_refit_20261118_freshep/offline_refit_20261118_freshep.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（仅回放归档帧与 checkpoint；未占用 SLM/CCD/激光，未启动任何采集进程）",
        note=(
            "用当前 epoch（11/18 采集）fresh 单模 Zernike 屏对 11/18 fit 与各 hybrid 变体做方向门 + 零步绝对强度门回放。"
            "方向门 7 对中 11/18 fit 中位 0.6856、4/7 达 0.50，未达既有 10/14 标准（本轮仅单 pass 7 对，物理不可达）；"
            "强度门 in-sample 零步统计 11/18 fit relL2 0.355、CV 差 0.310 最优。结论：不存在同时过双门的 checkpoint，"
            "不进入硬件闭环验收。输入 data/debug/slm_transfer/current_epoch_zernike_direction_20261118、"
            "输出 data/debug/offline_refit_20261118_freshep。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/direction_fresh_checkpoint_zernike_20261118.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
        ),
    ),
    "report/shape_trials/ideal_path_static_test_20261117/README.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-17 实验采集；报告本体不占用设备）",
        note=(
            "静态整形页同源 generate_phase_radians 模型前向生成相位，SLM 下发 + CCD 全幅采集 5 个图案"
            "（线性光栅 / 棋盘格 / 透镜聚焦 / 圆形光栅 / 达曼光栅），曝光 0.05 ms，观测来源 model_forward"
            "（observed=false）。结论：相位与下发链路正确，但实测远场全部塌缩为中心暗弱小斑，模型前向假设"
            "再次被证伪（透镜方形平顶 vs 实测点状聚焦的证伪结论成立）。"
            "人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/ideal_path_static_test_20261117.py",
            "scripts/slm/render_ideal_path_compare_20261117.py",
        ),
    ),
    "report/shape_trials/issue235_zero_order_four_method_20261120/issue235_zero_order_four_method_20261120.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（仅读 data/debug/issue235_center_20261117/ 归档 flat.npy 与 issue235_knife_baseline_fit_20261117.json，未占用 SLM/CCD/激光）",
        note=(
            "Issue #235 P1 零级中心四口径离线对账：对归档 flat.npy 用生产函数 "
            "ao_shaping.utils.image.zero_order.zero_order_center_by_method 跑 "
            "argmax/centroid/box_centroid/ellipse 四法，六组两两欧氏距离最大 3.0 px（恰在 ≤3 px 判据上，边界通过），"
            "默认椭圆中心 (x=1227, y=1106) 可用（verdict=ellipse_centre_usable）；"
            "CCD 帧中心与 SLM panel 刀口帧中心（find_beam_on_slm_centre_yx / fit_centre_yx，1920×1200）分开报告、不可混减。"
            "数字来自同目录 reconciliation.json。无独立生成脚本，四法直接调用生产函数完成。人工撰写。"
        ),
    ),
    "report/shape_trials/issue247_evidence_summary_20261124/issue247_evidence_summary_20261124.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线汇总（只读历史报告、原始数据与 GitHub issue 评论；未占用 SLM/CCD/激光）",
        note=(
            "Issue #247 全阶段证据汇总：按原始验收项区分物理现象成立、局部标定有效与校正算法收益未证明。"
            "支持零级/目标级分离、探针标度与局部 2D/patch 映射；否定单一全局映射、单一族归因与直接轴校正收益；"
            "根因、跨会话复现、加性判定、合成模板恢复、统一观测校正、默认路径逐位不变以及 Step A/holdout/Step B 收益仍证据不足。"
            "明确披露 11-24 同窗口重复性含检查期新增的 8 px 边界排除、core_exclude_px=41.0 会吞掉距零级 37–40 px 的真实 p192 波瓣，"
            "故只记限定口径下的同窗口可重复。issue 保持 OPEN，不冻结映射、不宣称闭环收益。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/issue247_zero_order_stability_20261117.py",
            "scripts/slm/issue247_axis_map_solve_20261121.py",
            "scripts/slm/issue247_waveform_probe_solve_20261122.py",
            "scripts/slm/issue247_reproducibility_check_20261124.py",
        ),
    ),
    "report/shape_trials/issue247_zero_order_20261117/issue247_zero_order_20261117.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM-200 #1 + Daheng CCD #0，2026-11-17 真线性 tilt 全幅采集；报告本体不占用设备）",
        note=(
            "Issue #247 零级/目标级分离实测：13 组全幅 Mono12 真线性 ±tilt/±离焦探针，"
            "flat 基线最大漂移 2 px、tilt 下零级稳定 ≤2.3 px、目标级两瓣分离 109–119 px；"
            "核心发现是面板轴↔CCD 轴为 +88.54° 的安装旋转（原始 CCD 帧上呈轴交换），"
            "由两轴位移矢量实测出，非写死 90°。"
            "含 P2 余弦探针 ROI 偏离光束的阴性复核。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/issue247_capture_tilt_20261117.py",
            "scripts/slm/issue247_zero_order_stability_20261117.py",
        ),
    ),
    "report/shape_trials/issue247_waveform_attribution_20261123/issue247_waveform_attribution_20261123.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM-200 #1 1064 nm + Daheng MER2-507-23GM cam 0，2026-11-24 采集；报告本体不占用设备）",
        note=(
            "Issue #247 波形族归因窗口 v2：23 图样 × 4 帧 = 92 帧全幅 Mono12 采集，"
            "补齐三族 ± 对照与预注册 held-out 方向。G1 波形依赖门 fail（x 族极差 89.84° pass、"
            "y 族 1.18° fail）、G2 单族异常门 fail（无单一异常族）、G3 轴分离门 fail"
            "（grating 0.35° / tilt 0.74° 轴路由失败，blaze_patch 89.997° pass）、"
            "G4 符号一致性门 fail（blaze_patch/y 偏离 164.01° 超 5° 容差）。"
            "attribution_label = not_assessable，held-out 轴向正确 0/10，映射仍歧义、不冻结。"
            "旧 6 值 verdict tilt_specific 仅兼容保存，不作为本窗口结论。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/issue247_geometry_capture_20261120.py",
            "scripts/slm/issue247_waveform_probe_solve_20261122.py",
            "scripts/slm/probe_optical_path_validity.py",
        ),
    ),
    "report/shape_trials/issue247_reproducibility_20261124/issue247_reproducibility_20261124.md": ReportProvenance(
        script="scripts/slm/issue247_reproducibility_check_20261124.py",
        command=(
            "python scripts/slm/issue247_reproducibility_check_20261124.py "
            "--dir data/debug/hw_window_20261124_issue247_reproducibility_v2 "
            "--baseline-dir data/debug/hw_window_20261123_issue247_waveform_attribution "
            "--probe-dir data/debug/optical_path_probe/issue247_reproducibility_20261124_v2/20261124_103335Z "
            "--out-dir report/shape_trials/issue247_reproducibility_20261124"
        ),
        environment="离线+硬件（Santec SLM-200 #1 1064 nm + Daheng MER2-507-23GM cam 0，2026-11-24 采集；报告本体不占用设备）",
        note=(
            "Issue #247 波形归因信号同窗口重复性复核（预注册 hw_window_20261124_issue247_reproducibility_prereg.md）："
            "同一 SLM 打开周期内 23 图样 × 4 帧 × 3 轮（frozen / reversed / frozen）= 69 唯一图样 / 276 帧，"
            "曝光 0.04 ms、settle 0.8 s、饱和门 0.9 × full_scale、零饱和帧；每轮独立重跑同一分析器。"
            "方向重复性门（至少两轮可评估且任两轮 axis_angle_deg 差 ≤ 2°）：复现 15 / 未复现 0 / 缺测 5，"
            "overall_pass=True；可评估项轮间最大轴角差 1.343°（tilt_y_plus），flat 跨轮最大漂移 0.469 px。"
            "5 个 p192 图样因 principal 峰心落在帧边界 8 px 带内（peak 21.75–25.0 counts / 背景 15.5）"
            "按预注册保留为缺测、不调阈值救回；逐轮 G1–G4 全 fail（verdict=tilt_specific），与 v2 归因窗口一致。"
            "本轮只回答同窗口重复性，不冻结映射、不证明根因、不关闭 issue。"
        ),
        extra_scripts=(
            "scripts/slm/issue247_geometry_capture_20261120.py",
            "scripts/slm/probe_optical_path_validity.py",
        ),
    ),
    "report/shape_trials/transfer_candidates_timing_20261117/transfer_candidates_timing_20261117.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（仅 CUDA，不占 SLM/CCD/激光；2026-11-17 生成）",
        note=(
            "Transfer 优化器 --candidates K 批前向的离线 GPU 微基准：真实 FrozenDefocusForward + "
            "DifferentiableUniformityOptimizer.update()，grid 32/64/128 × 4 形状 × K=1/2/4/8，"
            "单步 7–51 ms、K=8 相对 K=1 为 2.45–4.53×（次线性）；含端到端 301 s I/O 主导口径"
            "对比与 K=1 逐位一致证据。与 candidate_parallelism_20261117（另一模块）区分。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/transfer_candidates_gpu_bench_20261117.py",
        ),
    ),
    "report/shape_trials/model_optim_efficiency_20261116/model_optim_efficiency_20261116.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM-200 #1 + Daheng MER2-507-23GM，2026-11-16 两次实验锚定闭环后离线汇总；报告本体不占用设备）",
        note=(
            "物理预测模型（FrozenDefocusForward / M3，grid 128 / FFT 512）驱动可微闭环的优化效果与效率汇总："
            "模型侧 model_cv 3.89→0.66（随机起步规避焦斑盆地）、方向 cosine 0.696 过 ≥0.50 门禁；"
            "硬件侧实测核心 CV 停在 6.08/5.01 未达 ≤0.10，model_cv 与 measured_cv_window 同掩膜分叉约 7.5×（绝对强度表征误差）。"
            "效率：M3 闭环 ~0.30 s/iter（settle+capture 0.18 / 计算 0.11），比 model640 ~0.92 s/iter 快约 3×；"
            "历史预算证 LCOS settle 占会话 99%、优化器 <0.05%。绝对强度保真度实验数据待 SLM #1 上线补采"
            "（探针 absolute_fidelity_probe_20261116.py，commit 807d456a）。输入为两次锚定运行 timing/device_log/iterations、"
            "direction_variants_20261112、wall_clock_budget_20261111、shape_transition_timing_20261107。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/shape_transition_timing_20261106.py",
            "scripts/slm/absolute_fidelity_probe_20261116.py",
        ),
    ),
    "report/shape_trials/measured_forward_refine_20261107/measured_forward_refine_20261107.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线规划文档与离线探针结果（issue #237；未占用设备）",
        note=(
            "实测正向精化规划 + 离线探针：2D 配准精化、实测 PSF 标定、增益场、CCD 响应 LUT、"
            "杂散光与复振幅；含数据清单、逐项探针计划与验收口径（引 issue 原文）及 §7 离线结果"
            "（配准支持、PSF 不稳健、增益场不支持但退化、LUT 数据不足）。人工撰写。"
        ),
    ),
    "report/shape_trials/shared_lut_crosstalk_20261107/shared_lut_crosstalk_20261107.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线规划文档（issue #238，全部子项未开始实现）",
        note=(
            "SLM 1×1 共享响应 LUT 与面板相位串扰核规划。代码现状：已有可选 CCD 单调响应 LUT"
            "（MonotonicResponse），SLM 侧 LUT 与串扰核未开始实现；含 CCD/SLM LUT 归因歧义风险。人工撰写。"
        ),
    ),
    "report/shape_trials/star_farfield/star_farfield_20261107.md": ReportProvenance(
        script=HANDWRITTEN,
        environment=(
            "离线渲染（实验系统数据 2026-11-07：SLM #1 + Daheng CCD；"
            "shapes=['star']、4 块、acceptance 启用、3 轮）"
        ),
        note=(
            "确认 star 块 CCD 实测远场为弥散圆斑（零级主导），非五角星非方块。"
            "报告正文为人工撰写（分析结论与判定方法），"
            "figures/ 三张 PNG 由 scripts/slm/star_farfield_analysis_20261107.py "
            "渲染（输入 data/debug/slm_transfer/20261107_135252 存档）。报告本体不占用设备。"
        ),
    ),
    "report/shape_trials/physical_refine/forward_model_transfer_plan_20261104.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线规划文档；其 P1 已于 2026-11-05 实验执行（probe_optical_path_validity）",
        note="物理预测模型迁移实验计划（P0–P6 阶段、探针族、zernike_block 档与在线回灌门）；人工撰写。",
    ),
    "report/shape_trials/physical_refine/forward_model_transfer_20261107.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线 CPU（FrozenDefocusForward 2f 数字孪生，注入 7 模 Zernike 残差）；未占用 SLM/CCD/激光硬件",
        note=(
            "issue #229 步骤 3/4/5 工程实施（worktree feat/zernike-block-transfer，主分支代码未动）：结构化单模 "
            "±Zernike 探针 + zernike_block 第三档（梯度掩码+0.5 rad 块 L2）+ 块间在线回灌（ModelSwitchGate）；"
            "20 条新测试 + 328 回归全过、ruff 全净；P0 离线孪生实验：结构化设计 Gram 条件数 1.0 vs 随机 4.05，"
            "远场基逐模 peak/leakage 0.69~0.88 低于 2.0 地板、交叉响应矩阵条件数 17425（与 #229 硬件空结果同因："
            "探针设计不改变光学可辨识性上限，价值在逐模可归因测量）；物理设备实验方案见 §6。人工撰写。"
        ),
        extra_scripts=(),
    ),
    "report/shape_trials/hw_depth_matched/hw_depth_matched_20261105.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件：1064 nm Santec SLM #1 + Daheng CCD（run02 采集 2026-11-05 09:33–09:49）；分析离线 CPU",
        note=(
            "run02_depthmatched 将 ztrain 振幅提至 ±1.0/2.0 rad 消除激励深度混杂；"
            "三个变体离线分析复用 _harness_task2 口径（_harness_task2_run02.py）；"
            "结论：深度混杂不解释 run01 边际，主动探针样本效率无可测收益，twin NCC 结构失配待诊断。"
        ),
        extra_scripts=(),
    ),
    "report/shape_trials/issue230_refeed/issue230_refeed_20261105.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件（待采集）：SLM #1 + Daheng CCD 或 miicam 2.9 µm + SLM-300 + 激光（驱动参数物理相同）；分析离线 CPU",
        note=(
            "issue230 在线回灌验证集 NRMSE 判据；本版为方法/计划稿，采集与离线分析均未执行，"
            "所有结果区标 PENDING 待回填。判据：refeed < stale 且 ≤ fresh 上限 ⇒ 在线回灌提升验证集 NRMSE。"
        ),
        extra_scripts=("scripts/slm/issue230_refeed_probe.py",),
    ),
    "report/shape_trials/zernike_probe_sim/zernike_probe_sim_20261105.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线：纯 CPU 仿真（Sim2f 64x64 数字孪生），无 SLM/CCD/激光硬件",
        note=(
            "基于 Pathak & Jomy 2026 HMWS 五点估计的探测相位与 CCD→Zernike 解析算法仿真验证："
            "PatternHelper.generate_probe_phase + utils/image/zernike_probe.py；"
            "端到端 2f 恢复误差 <0.01 rad（defocus/spherical/astigmatism，a_true=0.2 rad）。"
        ),
        extra_scripts=(
            "tests/ao_shaping/utils/image/test_zernike_probe.py",
            "src/ao_shaping/utils/image/zernike_probe.py",
            "src/ao_shaping/utils/wavefront/pattern_helper.py",
        ),
    ),
    "report/shape_trials/hmws_zernike_probe/hmws_zernike_probe_20261124.md": ReportProvenance(
        script=HANDWRITTEN,
        command=(
            "python scripts/slm/hmws_zernike_probe.py --noll 4 5 "
            "--inject-rad -0.2 0.0 0.2 --bias-rad 0.3 --radius 450 "
            "--bit-depth 12 --exposure-ms 0.08 --auto-exposure --wait-until-idle 600 "
            "--out data/hmws_zernike_probe"
        ),
        environment=(
            "硬件：1064 nm Santec SLM-200 #1 + Daheng MER2-507-23GM CCD；"
            "12-bit 重采相机 12-bit / full_scale=4095，暗帧严格匹配 0.08 ms；分析离线 CPU"
        ),
        note=(
            "Pathak & Jomy 2026 HMWS 五点估计实验系统验证：先六步光路探测 + Fresnel 焦面扫描"
            "（verdict=ok，标定 ambiguous，未补偿），再对 Noll 4/5 各注入 ±0.2/0 rad、"
            "b=0.3 rad、r=450 px 采集五帧。8-bit 首采（曝光 0.06 ms）：Noll 5 线性良好"
            "（slope=1.0753，R²≈0.999996），Noll 4 失效（slope=0.3947，R²=0.9302）。"
            "12-bit 重采（曝光 0.08 ms，严格匹配暗帧）：Noll 5 仍可用（slope=0.8662，"
            "R²=0.9726），Noll 4 反而更差（slope=0.1772，R²=0.2911），位深量化假设被证伪，"
            "失效源于离焦峰值响应本身，不作系数定标。"
        ),
        extra_scripts=(
            "scripts/slm/probe_optical_path_validity.py",
            "tests/ao_shaping/scripts/test_hmws_zernike_probe.py",
            "src/ao_shaping/utils/image/zernike_probe.py",
            "src/ao_shaping/utils/wavefront/pattern_helper.py",
        ),
    ),
    "report/shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md": ReportProvenance(
        "scripts/slm/transfer_uncertainty_experiment.py",
        command=("python scripts/slm/transfer_uncertainty_experiment.py "
                 "--root data/debug/transfer_uncertainty_20261104 "
                 "--render-only"),
        environment="硬件：1064 nm Santec SLM + Daheng CCD；拟合使用 CUDA",
        note=("先使用生产 capture --grid 128 --count 24 --frames 3 --seed 110421 "
              "--save-dataset 采集 capture/，再以 --execute 完成选样采集；"
              "离线渲染用 --render-only，不触碰设备。"),
    ),
    "report/shape_trials/experience_review/shaping_experience_20261103.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线整理 + 11-04 实测收尾（169 份整形相关报告清单、11-03 已归档实验 / 仿真结论；11-04 在 Santec #1 + Daheng 实测补一轮短验证与一轮完整可微优化）",
        note=(
            "按证据等级汇总有效方法、适用条件、失败防线与下次实验固定协议；"
            "E1 数字来自同曝光 CCD 独立复测，仿真修复不标为实测。"
        ),
        extra_scripts=("scripts/ml/shaping_experience_inventory.py",),
    ),
    "report/shape_trials/experience_review/artifacts/source_inventory.md": ReportProvenance(
        script="scripts/ml/shaping_experience_inventory.py",
        command="python scripts/ml/shaping_experience_inventory.py",
        environment="离线只读扫描 report/ 下整形相关 Markdown；无设备",
        note="169 份报告的标题、行数、SHA-256 与逐行证据摘录；正文主报告为人工撰写。",
    ),
    "report/shape_trials/local_forward/local_forward_20261103.md": ReportProvenance(
        script="scripts/ml/sim2f_local_forward_report.py",
        command="python scripts/ml/sim2f_local_forward_report.py",
        environment="离线+硬件（Santec #1 1064 nm；Daheng CCD；2026-11-03 原生帧）",
        note="局部图像响应预测模型自动微分整形；三轮更新与五组独立 ABBA 复测，含失败起点对照。",
        extra_scripts=("scripts/ml/sim2f_local_forward_experiment.py",),
    ),
    "report/slm/gs_control_1000px/report.md": ReportProvenance(
        script="scripts/slm/generate_gs_control_report.py",
        command="python scripts/slm/generate_gs_control_report.py",
        environment="离线渲染 (读硬件采集的帧与相位; 数字为 2f 实验系统实测)",
        extra_scripts=(
            "scripts/slm/hw_gs_square_scans.py",
            "scripts/slm/probe_optical_path_validity.py",
        ),
        note=(
            "纯 GS 方斑控制组 (1000 px 目标): 平相位 / 直接 GS / GS+闪耀 P=10,12 三组实测, "
            "外加同相位物理预测模型的同口径对比; 0 级用横向闪耀移出方框后 pib/cv 才可用; "
            "C 变体 (先矫正光路像差再发 GS) 因设备下线未采集, 图上留灰色占位; "
            "8 bit 量程把方斑平台压到 11-15 counts, 框外那层底的来源未定论"
        ),
    ),
    # -- sim2f 孪生 vs 2f 实验语料 (纯离线) ------------------------------------
    "report/shape_trials/mask_closed_loop/mask_closed_loop_20261031.md": ReportProvenance(
        script="scripts/ml/sim2f_shape_trials.py",
        command=(
            "python scripts/ml/sim2f_shape_trials.py --execute"
            " --out report/shape_trials/mask_closed_loop"
        ),
        environment="离线（session_hw 录制会话标定孪生 + 漂移代理）",
        note=(
            "任意位图形状（mask_target）迭代闭环整形离线验证：标定孪生 vs 裸模型的"
            "两配对，实测验收门在模型失配下逐轮拦截；逐轮明细与每个变体相位在 artifacts/。"
        ),
        extra_scripts=(
            "scripts/ml/sim2f_shape_trials_figures.py",
            "src/ml/sim2f/shape_loop.py",
            "src/ml/sim2f/calibrate.py",
        ),
    ),
    "report/shape_trials/fov_generalization/fov_generalization_20261101.md": ReportProvenance(
        script="scripts/ml/sim2f_fov_extrapolation.py",
        command=(
            "python scripts/ml/sim2f_fov_extrapolation.py"
            " --out report/shape_trials/fov_generalization"
        ),
        environment="离线（带像差与粗糙度的仿真器件 + 冻结标称模型；器件与模型不是同一对象）",
        note=(
            "小视场相机下的超视场开环预测、D 最优主动探测、物理约束残差拟合三路仿真："
            "窗内拟合 + 独立确认相位验收门，未确认一律回退零像差；"
            "逐 trial 明细与主动探测/物理残差原始产物在 artifacts/。"
        ),
        extra_scripts=(
            "scripts/ml/sim2f_probe_active.py",
            "scripts/ml/sim2f_pinn_extrapolation.py",
            "scripts/ml/sim2f_fov_hw.py",
            "scripts/ml/sim2f_fov_hw_session.py",
            "src/ml/sim2f/full_frame.py",
            "src/ml/sim2f/window.py",
            "src/ml/sim2f/extrapolate.py",
            "src/ml/sim2f/probe_active.py",
            "src/ml/sim2f/pinn.py",
        ),
    ),
    "report/shape_trials/physical_refine/physical_refine_plan_20261102.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="规划（无新实测结论；历史数值来自 2026-11-01 标定/生产会话）",
        note=(
            "物理会话闭环相位微调与相差拟合规划：测量传递层与瞳面传播分离，"
            "顺序拟合探测器 MTF，背景默认不拟合；相差仅在平/参考相位重拟合；"
            "sequential_ccd 实测验收与独立重放，模型分数不作 CCD 证据。"
        ),
        extra_scripts=(
            "scripts/ml/sim2f_fov_hw_session.py",
            "scripts/ml/sim2f_fov_hw_refine.py",
            "src/ml/sim2f/calibrate.py",
        ),
    ),
    "report/shape_trials/physical_refine/physical_refine_results_20261102.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线分析（历史物理会话 + 生产会话对齐评分 + 约束残差拟合 + 几何初值孪生干跑；设备下线，无新增 CCD 采集）",
        note=(
            "顺序测量层/相干零级标定的验证集对照、生产会话探测器无关对齐评分与几何支持域、"
            "标定迁移与有界离焦/缩放扫描、约束残差拟合、历史会话驱动的任意形状代理闭环、"
            "几何初值孪生干跑与下一次实验步骤；全幅与逐目标分数只作模型诊断，孪生闭环和干跑只作流程证据，"
            "不作为物理整形成功或 CCD 验收证据。"
        ),
        extra_scripts=(
            "scripts/ml/sim2f_fov_hw_session.py",
            "scripts/ml/sim2f_fov_hw_refine.py",
            "scripts/ml/sim2f_physical_residual.py",
            "src/ml/sim2f/calibrate.py",
            "src/ml/sim2f/measurement.py",
            "src/ml/sim2f/optics.py",
            "src/ml/sim2f/pinn.py",
            "scripts/ml/sim2f_fov_hw_figures.py",
        ),
    ),
    "report/shape_trials/shaping_issues_review/shaping_issues_review_20261103.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线只读综述（汇总 report/shape_trials/ 下既有报告；不产生新硬件数据）",
        note=(
            "光束整形报告问题综述：按光路失配、物理设备限制、算法缺陷、测量与验收口径归类，"
            "数字均来自所引原报告；FourierGSNet 条目经人工摘录核对，未在硬件上复验。"
        ),
    ),
    "report/shape_trials/physical_refine/physical_refine_results_20261103.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线分析 + 2026-11-03 当日硬件采集（daheng CCD、SLM panel、ring refine 三模式与 GS seed refine、零级诊断）；报告含实测 CCD GIF。",
        note=(
            "保留 sections 1-3 离线诊断原值；新增 2026-11-03 硬件测量："
            "raw/effective/effective-aligned 三模式增益表、GS seed refine 负结果、"
            "零级 argmax 双瓣诊断与 measured CCD GIF。"
            "effalign 实测元数据存在未解释偏差（见正文 §5）；"
            "零级诊断仅证据，不修改 _locate_zero_order。"
            "这些数值来源见 artifacts/offline_diagnostics_20261103.json、"
            "artifacts/refine_gains_20261103.json、artifacts/zero_order_diagnosis_20261103.json；"
            "不视为物理整形成功或 CCD 验收证据。"
        ),
        extra_scripts=(
            "scripts/ml/sim2f_fov_hw_refine.py",
            "src/ml/sim2f/calibrate.py",
            "src/ml/sim2f/optics.py",
            "scripts/ml/sim2f_fov_hw_offline_diagnostics.py",
            "scripts/ml/sim2f_fov_hw_figures.py",
            "scripts/ml/sim2f_fov_zero_order_diag.py",
            "scripts/ml/sim2f_fov_zero_order_figure.py",
        ),
    ),
    "report/shape_trials/physical_refine/transfer_mil_hardware_plan_20261118.md": ReportProvenance(
        script=HANDWRITTEN,
        environment=(
            "离线算法验证与规划（2026-11-18/19）；本轮结果未占用设备，"
            "目标形状仍未通过独立 CCD 验收。"
        ),
        note=(
            "记录 SAFARI 借鉴、Poisson 可辩护性、复场与乘性增益候选，以及 B session per-session Gaussian refit；"
            "B-test 24 行仅作冻结后评分，calibration 10 行排除。Gaussian refit 的 held-out 预测显著改善，"
            "但不等于 transfer-shaping 实测提升；未因此修改 optimizer 更新规则。"
            "配套 JSON 保留 mil 原始脚本绝对路径与 SHA256；迁入的诊断脚本按 master 现有 API 适配，"
            "不声称与历史生成脚本字节一致。"
        ),
        extra_scripts=(
            "scripts/slm/diagnose_complex_illumination_b.py",
            "scripts/slm/diagnose_transfer_poisson_weighting.py",
            "scripts/slm/evaluate_multiplicative_gain_b.py",
        ),
    ),
    "report/shape_trials/physical_refine/fit_compare_m3_20261118.md": ReportProvenance(
        script="scripts/slm/compare_transfer_fit_runs.py",
        command=(
            "python scripts/slm/compare_transfer_fit_runs.py "
            "--dataset <M3 dataset 目录> --baseline <baseline fit 输出目录> "
            "--updated <updated fit 输出目录> --out report/shape_trials/physical_refine/"
            "fit_compare_m3_20261118.md"
        ),
        environment="离线对照（同一 dataset 两次 fit 输出；设备未占用）",
        note=(
            "M3 dataset（68 行）baseline(iter 0, loss 0.302758) vs updated(iter 17, loss 0.165202)："
            "correlation 0.7867→0.8814。运行一致性断言通过（manifest_sha256/checkpoint/adaptation/"
            "radiometry/ramp_matrix 相同，cv_measured 逐行一致）。仅前向预测对照，不是整形验收。"
            "复现命令的三个输入目录是 data/debug 下的 fit 产物（语料不入库），占位符需按当次"
            "会话路径替换。"
        ),
    ),
    "report/shape_trials/physical_refine/fit_compare_m3_z21_20261118.md": ReportProvenance(
        script="scripts/slm/compare_transfer_fit_runs.py",
        command=(
            "python scripts/slm/compare_transfer_fit_runs.py "
            "--dataset <M3 dataset 目录> --baseline <baseline fit 输出目录> "
            "--updated <Zernike 5..21 扩展基 fit 输出目录> --out report/shape_trials/"
            "physical_refine/fit_compare_m3_z21_20261118.md"
        ),
        environment="离线对照（同一 dataset 两次 fit 输出；设备未占用）",
        note=(
            "M3 dataset（68 行）baseline(iter 0) vs Zernike 5..21 扩展基(iter 5)：检验残差基容量是否提升下限，"
            "结论为不提升（聚合与 smooth/bandlimited 族均未改善）。运行一致性断言通过。"
        ),
    ),
    "report/shape_trials/physical_refine/fit_compare_m3_z11_vs_z21_20261118.md": ReportProvenance(
        script="scripts/slm/compare_transfer_fit_runs.py",
        command=(
            "python scripts/slm/compare_transfer_fit_runs.py "
            "--dataset <M3 dataset 目录> --baseline <Noll 5..11 fit 输出目录> "
            "--updated <Noll 5..21 fit 输出目录> --out report/shape_trials/"
            "physical_refine/fit_compare_m3_z11_vs_z21_20261118.md"
        ),
        environment="离线对照（同一 dataset 两次 fit 输出；设备未占用）",
        note=(
            "M3 dataset（68 行）Noll 5..11(iter 17) vs Noll 5..21(iter 5)：correlation 0.8814→0.8780（更差）。"
            "据此 5..21 不作候选，残差基维持 5..11。运行一致性断言通过。"
        ),
    ),
    "report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线计划；未连接或占用 SLM、CCD、激光（本轮未执行）",
        note=(
            "目标形状 CCD 验收的条件计划：先补观测模型（静态 PSF/散斑、成像展宽）并给出独立 holdout 下限改善，"
            "再按独立相位合成 + measured-vs-target 指标门槛 + 带锁 replay 执行。当前未放行；"
            "本轮已实现并测试 protocol.target_intensity_rmse（整窗归一化、背景 floor、rel_l2/energy_fraction），"
            "replay_phases 已写入 target_rmse 字段，并新增 replay_transfer_shapes_hardware.py 带锁外壳"
            "（仅 mock 验证，--execute 未执行）。阈值须现场预注册；任何指标均不替代独立 CCD 证据。"
        ),
        extra_scripts=(
            "scripts/slm/validate_transfer_shapes_hardware.py",
            "scripts/slm/replay_transfer_shapes_hardware.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "src/ao_shaping/tools/slm/transfer/protocol.py",
        ),
    ),
    "report/shape_trials/physical_refine/closed_loop_refit_abba_plan_20261124.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线计划；未连接或占用 SLM、CCD、激光（本轮未执行）",
        note=(
            "闭环 refit A/B–ABBA 主实验计划，按文档化契约重建；默认配置与 refit 变体采用同曝光、"
            "A–B–B–A 交错与 N≥3 配对，主判据为实测 core CV。预测指标不替代实测闭环收益，"
            "实验前须重新验证重建的回放脚本。"
        ),
        extra_scripts=(
            "scripts/slm/replay_closed_loop_abba_20261124.py",
            "scripts/slm/refit_transfer_forward.py",
            "scripts/slm/evaluate_transfer_on_new_session.py",
        ),
    ),
    "report/shape_trials/multishape_scale/square_1000_diff_abba_20261030.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线分析与实验；ABBA 及 v6/v7 局部响应诊断均使用 Santec #1 / Daheng Mono12",
        note=(
            "1000 px 正方形：平面起步展宽种子、formation 目标、原生 counts/ms、"
            "同曝光 A-B-B-A 预注册验收；条件性轴向扫描最优在 -0.2 m 边界，未采用。"
            "v6/v7 已完成且无本征饱和，但局部 Jacobian 未通过方向门槛；"
            "离线提议不是实测结果，尚未证明均匀正方形整形成功。"
        ),
        extra_scripts=(
            "scripts/slm/square_1000_diff_abba_20261030.py",
            "scripts/slm/analyze_square1000_diff_live_20261030.py",
            "scripts/slm/square_1000_model_diagnostic_20261030.py",
            "scripts/slm/analyze_square1000_model_diagnostic_20261030.py",
            "scripts/slm/square_1000_model_defect_sim_20261030.py",
            "scripts/slm/square_1000_axial_archive_20261030.py",
            "scripts/slm/square_plane_continuation_20261030.py",
            "src/ao_shaping/tools/slm/transfer/radiometry.py",
            "src/ao_shaping/tools/slm/transfer/axial.py",
            "tests/ao_shaping/tools/slm/test_square1000_diff_abba_protocol.py",
        ),
    ),
    "report/sim2f_bench/twin_vs_hw_corpus/20261027/report.md": ReportProvenance(
        script="scripts/ml/validate_sim2f_vs_hw_corpus.py",
        command="python scripts/ml/validate_sim2f_vs_hw_corpus.py",
        environment="离线（只读 data/debug/ 的 pkl；孪生在 CPU 上跑 torch，无设备、无 GPU 必需）",
        note=(
            "把 G0-4 (`ncc(实测 net 帧, 孪生预测 net 帧) >= 0.5`) 放到 312 条真实 2f 语料上回答: "
            "去重后只有 18 个不同的 (相位, 帧) 对 ⇒ 结论是 not_similar (归一化 L1 在整个 k 扫描上"
            "的最小值 1.2027, 门限 0.5), G0-4 的 min 口径未通过 (+0.453) 而中位口径通过 (+0.625); "
            "含几何标度 k 的完整扫描、逐轮一致性、以及「按序号对半分与族完全混杂」这条预登记缺陷的披露"
        ),
        extra_scripts=(
            "src/ml/hwdataset/index.py",
            "src/ml/hwdataset/records.py",
            "src/ml/hwdataset/transforms.py",
            "src/ml/sim2f/optics.py",
            "src/ml/sim2f/config.py",
        ),
    ),
    # -- sim2f 闭环归因 (ml.sim2f) -------------------------------------------
    "report/sim2f_shaping/closed_loop_20261025/report.md": ReportProvenance(
        script="scripts/ml/generate_sim2f_shaping_report.py",
        command="python scripts/ml/generate_sim2f_shaping_report.py",
        environment="离线（只读 runs/sim2f_fair 的 JSON/npz 产物；无 torch / 无 GPU / 无硬件）",
        note=(
            "2f 实验系统数字孪生 (ml.sim2f) 上的整形闭环: 语料可行性下限 (24 px 特征底线, "
            "line 族从不可达变为可达)、U-Net 在标签实验系统上 48/48 + 8/8 胜过零相位、"
            "换到标定实验系统后五路读数全输给零相位 (net_open 0/7 族)、teacher 对照证明"
            "差别在实验系统而不在摊销、四变体消融把差异归到 10.0847 mm 相机离焦而非 0 级、"
            "噪声阶梯的拒测档位与重跑的逐字段确定性; 含 §8 三条被推翻的结论"
        ),
        extra_scripts=(
            "src/ml/sim2f/loop.py",
            "src/ml/sim2f/train.py",
            "src/ml/sim2f/teacher.py",
            "src/ml/sim2f/optics.py",
            "src/ml/sim2f/synth.py",
            "src/ml/sim2f/losses.py",
            "src/ao_shaping/drivers/sim/bench_twin.py",
            "tests/ao_shaping/ml/sim2f/test_twin_parity.py",
        ),
    ),
    "report/sim2f_bench/twin_vs_hw_corpus/20261027/pupil_center_sweep.md": ReportProvenance(
        script="scripts/ml/calibrate_sim2f_pupil_center.py",
        command="python scripts/ml/calibrate_sim2f_pupil_center.py",
        environment=(
            "离线（只读 runs/sim2f_bench/session 的实测产物 + runs/sim2f_fair 的 "
            "checkpoint/语料；torch 仅重建网络相位与孪生预测帧，无 SLM / 无相机）"
        ),
        note=(
            "G0-4 还差 0.01 的归因: 遍历瞳面中心偏移 (13x13, ±120 px) 看 min "
            "ncc(实测 net, 孪生 net) 能否达到 0.5; 最优偏移 (0,+20) px 落在 ±20 px "
            "判据带内, 但 5 px 细化在 (+10,-15) px 得到 0.5053, 说明门限恰好压在孪生"
            "保真度上; 含重建复现 (2.2e-11) 与网络前向的设备依赖性 (CPU vs CUDA 差 "
            "1.0e-3 ncc)"
        ),
        extra_scripts=(
            "src/ml/sim2f/bench.py",
            "src/ml/sim2f/optics.py",
            "src/ao_shaping/utils/slm/phase_embed.py",
        ),
    ),
    "report/sim2f_bench/optical_scale_20261027/report.md": ReportProvenance(
        script="scripts/ml/diagnose_sim2f_optical_scale.py",
        command="python scripts/ml/diagnose_sim2f_optical_scale.py",
        environment=(
            "离线（只读 data/debug/ 的 pkl；孪生在 CPU 上跑 torch，无设备、无 GPU 必需）"
        ),
        note=(
            "2f 实验系统数字孪生的远场尺度 / 离焦标定是否已失效: H1 (region_span_px, "
            "25 点扫描 600–1200 px) 否决 —— 25 个点上无一点同时满足宽度比 ∈ [0.9, 1.1] "
            "与中位 ncc >= 0.75; H2 (相机离焦 / 零级占比) 亦否决 —— defocus_m 相对差 "
            "1.2 % 未超 20 %, 中位 ncc 提升 +0.0002 未达 0.05。两条假说都被否决, 现有证据"
            "不足以说「标定已失效」: 中位 ncc 仍停在 +0.6247 (门限 +0.7500), 归一化 L1 "
            "仍停在 1.3425 (门限 0.50); 不据此改任何已入库标定量"
        ),
        extra_scripts=(
            "scripts/ml/validate_sim2f_vs_hw_corpus.py",
            "src/ml/hwdataset/records.py",
            "src/ml/sim2f/config.py",
            "src/ml/sim2f/optics.py",
            "src/ao_shaping/drivers/sim/bench_twin.py",
            "src/ao_shaping/tools/slm/bench_twin_calibrate.py",
        ),
    ),
    "report/sim2f_bench/alignment_20261027/report.md": ReportProvenance(
        script=HANDWRITTEN,
        command="",
        environment="离线汇总（只读 runs/sim2f_bench/session_hw 与各 lane 报告产物，无 SLM / 无相机）",
        note=(
            "本报告把 lane 1 (仿真 dry run)、lane 2 (pupil_center_sweep 结论 A)、"
            "optical_scale (H1/H2 均否决) 与 session_hw (G0-2 FAIL / G0-4 FAIL) 四路证据"
            "汇总为一条对齐结论: G0-4 = 孪生保真度上限 (非放置错误), G0-2 = 真实光学漂移; "
            "不放宽任何阈值。"
        ),
        extra_scripts=("src/ml/sim2f/bench.py",),
    ),
    "report/shape_trials/axial_move_sim/axial_move_sim_20261029.md": ReportProvenance(
        script="scripts/slm/generate_axial_move_sim_report.py",
        command="python scripts/slm/generate_axial_move_sim_report.py",
        environment="离线（数字孪生真值，无任何实验测量，无设备）",
        note="SLM 沿光轴移动的模型代价、远场标度律与 (s, ε) 再辨识边界；S1 标称几何只作灵敏度上界，S2 取 2026-10-29 现场读数。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/axial.py",
            "src/ao_shaping/drivers/sim/bench_twin.py",
        ),
    ),
    "report/shape_trials/blackbox_path/planning_20261029.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（写于采集之前，无设备）",
        note=(
            "采集前写定的判据：H1（模型-实测相关）与 H2（CV 降而 EE 不降）的成功/失败阈值、"
            "逐形状报告口径与不接受平均的理由。复核结果见 analysis_20261029.md。"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/protocol.py",
        ),
    ),
    "report/shape_trials/blackbox_path/artifacts/live_acquisition_20261029.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="实验（Santec #1、Daheng Mono12）同曝光同位深重放 + 离线核对",
        note=(
            "把两轮已存相位在同一曝光与原生位深下重放，作为「36 倍曝光差来自整形相位而非光路」的"
            "可比性检验；记录逐帧饱和计数与设备释放状态。"
        ),
        extra_scripts=(
            "scripts/slm/replay_saved_phases_20261029.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
        ),
    ),
    "report/shape_trials/blackbox_path/live_20261029.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="实验（Santec #1、Daheng Mono12，2f 实验系统）采集 + 离线复核（无设备）",
        note=(
            "r1（平面起始相位）与 r2（GS 起始相位）两轮现场记录的逐形状读数、块级接受计数，"
            "以及按 planning_20261029.md 事先写定判据的复核：H1 未通过（r2 相关中位 0.70）、"
            "H2 未通过（CV 降但 EE 同步降）。不跨形状平均。"
        ),
        extra_scripts=(
            "scripts/slm/run_blackbox_path_20261029.py",
            "scripts/slm/summarize_blackbox_path_20261029.py",
            "scripts/slm/replay_saved_phases_20261029.py",
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
        ),
    ),
    "report/shape_trials/blackbox_path/analysis_20261029.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（只读 data/shape_trials/blackbox_path_* 归档产物 + 代码走查，无设备）",
        note=(
            "D1 归一化缺陷（锚点除以 counts_scale=16 而模型是原生计数率）、D2 fit.valid 的 8-bit 门限、"
            "F1 模型对得上能量却对不上残余结构（CV 差 2.1–3.3 倍）、F2 复合分数以 EE 换 CV。"
        ),
        extra_scripts=(
            "scripts/slm/summarize_blackbox_path_20261029.py",
            "scripts/slm/simulate_transfer_units_20261029.py",
            "src/ao_shaping/tools/slm/transfer/radiometry.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
        ),
    ),
    "report/shape_trials/blackbox_path/sim_units_20261029.md": ReportProvenance(
        script="scripts/slm/simulate_transfer_units_20261029.py",
        command="python scripts/slm/simulate_transfer_units_20261029.py",
        environment="离线（合成线性相机 + 数字孪生，无设备）",
        note="无设备复现两条归一化路径：Mono8/Mono12 增益一致性、锚定合成图的目标值、以及块级接受计数。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/radiometry.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/blur.py",
        ),
    ),
    "report/shape_trials/axial_plane/axial_plane_20261026.md": ReportProvenance(
        script="scripts/slm/generate_axial_plane_report.py",
        command="python scripts/slm/generate_axial_plane_report.py",
        environment="实验（Santec #1、Daheng Mono12，SLM 输入平面轴向校正）",
        note="ABCD 条件距离扫描、相位分组验证与可辨识性边界。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/axial.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "src/ao_shaping/algorithm/signal_processing/defocus_forward_core.py",
        ),
    ),
    "report/shape_trials/square1200_exploration/exploration_20261026.md": ReportProvenance(
        script="scripts/slm/generate_square1200_exploration_report.py",
        command="python scripts/slm/generate_square1200_exploration_report.py",
        environment="实验（Santec #1、Daheng Mono12，1200 CCD 像元方形）",
        note="同初相位、同预算的严格 CV 验收与非单调探索对照及固定曝光重放。",
        extra_scripts=(
            "scripts/slm/render_square1200_exploration.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/exploration.py",
            "src/ao_shaping/tools/slm/transfer/artifacts.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
        ),
    ),
    "report/shape_trials/square1200/square1200_20261026.md": ReportProvenance(
        script="scripts/slm/generate_square1200_report.py",
        command="python scripts/slm/generate_square1200_report.py",
        environment="实验（Santec #1 与 Daheng，1064 nm，1200 CCD 像元方形）",
        note="非 GS 物理映射初相位、曝光感知迁移学习、Mono12 及 CCD 核心 CV 验收。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/geometric_seed.py",
            "src/ao_shaping/tools/slm/transfer/camera_probe.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
        ),
    ),
    "report/shape_trials/cold_start/cv_diagnosis_20261026.md": ReportProvenance(
        script="scripts/slm/analyze_cold_start_cv.py",
        command="python scripts/slm/analyze_cold_start_cv.py",
        environment="离线（已存档的 Santec #1 与 Daheng 同相位、同曝光数据）",
        note="训练相位拟合曝光偏置，验证整体窗口与目标核心 CV，并检查传播采样收敛。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/radiometry.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
        ),
    ),
    "report/shape_trials/cold_start/cold_start_20261026.md": ReportProvenance(
        script="scripts/slm/generate_cold_start_report.py",
        command="python scripts/slm/generate_cold_start_report.py",
        environment="离线（重算 Santec #1 与 Daheng 当前光路实验数据）",
        note="无历史权重、平面或随机整形初相位，物理预测模型从零拟合与 CCD 配对验收。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/bootstrap.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/cold_start.py",
            "src/ao_shaping/tools/slm/transfer/progress.py",
        ),
    ),
    "report/shape_trials/sim_pretrain_transfer/sim_pretrain_transfer_20261024.md": ReportProvenance(
        script="scripts/slm/generate_sim_pretrain_transfer_report.py",
        command="python scripts/slm/generate_sim_pretrain_transfer_report.py",
        environment="离线（仿真先验 + 已存档的 Santec/Daheng 实验数据）",
        note="仿真预训练的分组 CV 指标与「无 GS 的逐位复现」；两部分的检查点互不共用，报告 §1 明确分开。",
        extra_scripts=(
            "scripts/slm/reproduce_transfer_block_without_gs.py",
            "scripts/slm/generate_transfer_no_gs_gif.py",
            "src/ao_shaping/tools/slm/transfer/pretrain.py",
            "src/ao_shaping/tools/slm/transfer/workflow.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
            "src/ao_shaping/algorithm/signal_processing/multishape_objective.py",
        ),
    ),
    "report/shape_trials/multishape1200/multishape1200_20261027.md": ReportProvenance(
        script="scripts/slm/generate_multishape1200_report.py",
        command="python scripts/slm/generate_multishape1200_report.py",
        environment="实验（Santec #1 与 Daheng，1064 nm，1200 CCD 像元）",
        note="从零训练物理预测模型 + 四种形状的 CCD 锚定整形 + 初相位实测择优 + Recorder 过程记录。",
        extra_scripts=(
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/recording.py",
            "src/ao_shaping/tools/slm/transfer/geometric_seed.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "src/ao_shaping/algorithm/signal_processing/multishape_objective.py",
        ),
    ),
    "report/shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（读取已保存的 Santec/Daheng 实验数据）",
        note="三形状从零探测、物理参数微调、GS 与 CCD 锚定闭环复现手册；图片和参数 CSV 由离线脚本生成。",
        extra_scripts=(
            "scripts/slm/generate_multishape_reproduction_figures.py",
            "src/ao_shaping/tools/slm/transfer/workflow.py",
            "src/ao_shaping/tools/slm/transfer/artifacts.py",
            "src/ao_shaping/algorithm/signal_processing/physical_forward_fit.py",
            "src/ao_shaping/algorithm/signal_processing/multishape_objective.py",
            "scripts/slm/bootstrap_transfer_checkpoint.py",
            "scripts/slm/transfer_multishape_experiment.py",
            "scripts/slm/fit_transfer_multishape.py",
            "scripts/slm/validate_transfer_shapes.py",
            "scripts/slm/calibrate_transfer_blur.py",
        ),
    ),
    "report/shape_trials/multishape_capability/diagnosis_and_transfer_20261023.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件（Santec SLM #1 + Daheng CCD）",
        note="原始 CCD 区域复算、载波几何纠正、物理参数迁移拟合和多形状配对实验；按时间追加记录。",
        extra_scripts=(
            "scripts/slm/transfer_multishape_experiment.py",
            "scripts/slm/fit_transfer_multishape.py",
            "scripts/slm/validate_transfer_shapes.py",
            "scripts/slm/summarize_transfer_shapes.py",
            "scripts/slm/capture_transfer_response.py",
            "scripts/slm/analyze_transfer_response.py",
            "scripts/slm/calibrate_transfer_blur.py",
            "scripts/slm/analyze_transfer_direction.py",
        ),
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
        "含 phasor 表示理由/unit_phasor/物理预测 FOV 分桶/损失/unit-vs-non-unit L1 下界实测",
    ),
    # -- beam shaping / benchmark suite -------------------------------------
    "report/benchmarks/cython_optimizer_report/cython_optimizer_performance.md": ReportProvenance(
        script="scripts/dm/generate_cython_optimizer_report.py",
        command="python scripts/dm/generate_cython_optimizer_report.py",
        environment="离线",
        note="读 experiments/calculators/benchmark_results.json；同目录两张 speedup 图由本脚本生成",
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
    "report/slm/pib_exit_safety/pib_exit_safety_20261025.md": ReportProvenance(
        script="scripts/slm/generate_pib_exit_safety_report.py",
        command="python scripts/slm/generate_pib_exit_safety_report.py",
        environment="离线 (读硬件采集的 summary.json 与远场裁切)",
        extra_scripts=(
            "scripts/slm/verify_pib_exit_safety_hw.py",
            "tests/ao_shaping/scripts/test_verify_pib_exit_safety_hw.py",
        ),
        note=(
            "issue #145 (H-28) 的硬件验收: 搜索主体被 KeyboardInterrupt / 相机掉线打断后, "
            "面板实际停在哪个状态 —— 固件回读 (内存槽号 / 直接灰度) 与光学比对两条独立证据, "
            "外加退出写入是否绕开了「写到当前显示槽是空操作」的固件陷阱"
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
    "report/slm/lut_checker_20261111/report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线 (读已落盘的 CCD 帧与重建结果)",
        note=(
            "棋盘格 (二维 0/Δ 二元相位) FFT 面双通道强度反演得到的等效相位台阶 Δ_eff(g) "
            "在名义 π 灰度 g≈496 达峰 ~90° 后回落。已排除: 棋盘格几何尺寸 (cells 8–32 谷值 "
            "0.468–0.488)、随灰度的散射/损耗 (全图功率 0.997–1.018)。"
            "首要嫌疑 (优先于电学串扰): 约一半光是不随灰度变化的未调制静态分量 (偏振未匹配等) —— "
            "调制功率占比 q 时标量反演的天花板是 2·arcsin√q, q≈0.5 恰为 90°, "
            "因此“Δ_eff 不到 π”不能断言 LC 真实相位未到 π。"
            "现有 invert_checkerboard_scan 的 LUT 产物不可信, 不应作为已校正 LUT 加载。"
            "正文人工撰写; 配图与 GIF 由 scripts/slm/generate_lut_checker_report.py 离线生成 "
            "(只读已落盘帧, 只写 figures/, 不改写正文)。"
        ),
        extra_scripts=(
            "scripts/slm/generate_lut_checker_report.py",
            "scripts/slm/calib_slm_lut_checkerboard.py",
            "scripts/slm/diag_checker_cells_sweep.py",
            "src/ao_shaping/utils/slm/slm_lut.py",
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
        command="python scripts/slm/hw_differentiable_phase_opt.py --execute --iters 120 --hw-iters 3",
        environment="硬件",
        extra_scripts=(
            "src/ao_shaping/algorithm/signal_processing/differentiable_beam.py",
            "src/ao_shaping/tools/slm/panel/focus_probe.py",
        ),
        note=(
            "正向调优模型 + 反向可微优化 SLM 相位的硬件全流程记录: **开测前先脚本判在焦**"
            "(不在焦即退出 5, 一帧不测), 现场几何标定, 120 次纯模型反向传播的相位/梯度/损失史, "
            "上屏后的 CCD 远场, 模型与实测的对照 (强度相关系数 0.599、峰值偏移 [6.0,-6.0] 远场采样、"
            "类 Strehl 17.226), 测量锚定的硬件闭环迭代 (实测损失 7.018e-4 → 6.969e-4, 梯度范数 "
            "0.949 → 0.634 持续下降), 以及**随机初值 + 强度 MSE 会掉进「把远场摊平」局部极小**"
            "与**相位 RMS 几乎不变**这两个负结果"
        ),
    ),
    "report/slm/differentiable_opt/speedup_before_after_20261116.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        extra_scripts=(
            "scripts/slm/hw_differentiable_phase_opt.py",
            "scripts/slm/render_speedup_figures.py",
        ),
        note=(
            "整形提速「前后对比」报告: MIL Step B 迭代预算 600→150 与光路复用 `reuse_latest` "
            "默认开启落地后, 对比**完整时间消耗 / 迭代次数 / 整形质量**。数据来源全部为实测落盘产物: "
            "Step B 预算离线扫描 (artifacts/stepb_budget_sweep_20261116, 100–600 步的墙钟/CV/EE/FF, "
            "600 步比 100 步多花 39% 墙钟而 CV 不单调改善); Step B 150 步硬件标定 "
            "(CV 13.83→0.651、EE 1.0000, far_field_size=4096); 本轮 3 次物理实跑 "
            "(iters=120/20/120, hw_iters=3, 墙钟 60–68 s, 模型侧纯计算 <1 s、前向传播占 0.79–0.91, "
            "逐次实测曝光 0.08–0.18 ms、焦面扫描曝光 0.04–0.10 ms, 实测损失/梯度范数单调下降, "
            "CCD 读回帧 FWHM 16.6–18.7 px、峰值 136–210、空心度 0.83–0.91); "
            "跨轮模型-硬件一致性 0.089–0.810 历史对照与本轮 0.599–0.663。 "
            "物理对比图由 `scripts/slm/render_speedup_figures.py` 从 artifacts 落盘的 npz 读回帧 "
            "(`ccd_flat`/`ccd_hw_iter_*`/`ccd_optimized`) 与 `history_hw.csv` 渲染, "
            "图名统一 `speedup_` 前缀归属本报告。 "
            "负结果保留: 模型内 loss 下降≠光学改善、曝光对齐下原始峰值比无意义、`get_displayed_phase()` "
            "不可作证据"
        ),
    ),
    "report/slm/differentiable_opt/kaxis_screening_20261123.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="仅 GPU 离线（RTX 5080 Laptop，1024² 网格 cuda）；未占用 SLM / CCD / 激光，全程无实验动作",
        note=(
            "冻结问题 phys_20261116_01 上可微整形加速方案公平筛选（K 轴候选并行 + gate 复现 + 参照变体）："
            "gate 复现逐位一致（max_rel_dev=0.0 ≤ 0.001, PASS）。K 轴 n_candidates∈{1,4,8}（Adam lr=0.05, "
            "120 迭代）**不构成加速**：K>1 终态 cv 仅改善 ~0.05%（0.28381→0.28365，种子噪声 2–4 倍）、"
            "墙钟 +17%（K=4）~+60%（K=8），且 iters_to_k1_cv=115 三者相同（不提前收敛）⇒ 严格支配、K 轴淘汰；"
            "与 candidate_parallelism_20261117（grid 256 square 仅 0.07% 收益）一致。参照变体分开报告："
            "L-BFGS（lr=1.0）第 5 迭代即达 K=1 终态 cv（≈0.67 s vs K=1 全程 5.91 s，快 ≈8.9×），终态 cv "
            "-0.15%、energy +2.5% ⇒ **算法轴加速候选**；cosine 调度质量 -0.42% 无收益。入围物理复验配对 = "
            "K=1 Adam 基线 vs L-BFGS 早停。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/sweep_kaxis_frozen_20261123.py",
            "scripts/_common/differentiable_problem.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py",
        ),
    ),
    "report/slm/differentiable_opt/artifacts/stepb_optimizer_lr_20261116/stepb_optimizer_lr.md": ReportProvenance(
        script="scripts/slm/sweep_stepb_optimizer_lr.py",
        command="python scripts/slm/sweep_stepb_optimizer_lr.py",
        environment="离线",
        note=(
            "MIL Step B 优化器/学习率/调度扫描 (纯离线仿真, 不碰任何硬件设备): lr 阶梯、"
            "余弦/阶梯/预热+余弦调度在同等物理光斑质量约束下能否砍迭代预算。"
        ),
    ),
    "report/slm/differentiable_opt/artifacts/fidelity_ladder_20261116/ladder_summary.md": ReportProvenance(
        script="scripts/slm/hw_forward_model_fidelity_ladder.py",
        command=(
            "python scripts/slm/hw_forward_model_fidelity_ladder.py "
            "--execute --out report/slm/differentiable_opt/artifacts/fidelity_ladder_20261116"
        ),
        environment="离线",
        note=(
            "保真度阶梯的逐图案汇总 (metrics.json/csv 同批产物): dry-run 演练写出占位与判据"
            "说明, 不伪造数字; --execute 后重写为实测对比。主判据 spearman_signal。"
        ),
        extra_scripts=("scripts/slm/_hw_fidelity_ladder_sections.py",),
    ),
    "report/shape_trials/dark_frame_record_20261119.md": ReportProvenance(
        script=HANDWRITTEN,
        command="",
        environment="硬件窗口记录（仅开相机，激光关闭；非脚本产物）",
        note=(
            "同曝光暗场采集与归档的手写记录（MER2-507-23GM, 12-bit, 8 档曝光×16 帧, "
            "归档至 data/ccd_configs/）；暗场采集能力见 ao_shaping.tools.ccd.capture_darks，"
            "但本记录的数据来自当次窗口操作，无单一复现命令。"
        ),
    ),
    "report/shape_trials/target_phase_matrix_20261120.md": ReportProvenance(
        script="scripts/slm/target_phase_matrix_20261120.py",
        command=(
            "python scripts/slm/target_phase_matrix_20261120.py "
            "--checkpoint data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json "
            "--out <输出目录>"
        ),
        environment="离线（未执行 --execute，不占用 SLM/CCD/激光）",
        note=(
            "冻结目标 × 相位矩阵 48 条 (6 形状 × 4 尺寸 × {gs, model}) 离线生成报告；"
            "30 条有效相位文件, 18 条 model 网格不可表示按契约不伪造。checkpoint 为 "
            "data/debug 语料 (不入库), 缺 checkpoint 时脚本诚实退化 (见 "
            "tests/ao_shaping/scripts/test_target_phase_matrix_20261120.py)。"
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
            "离焦光路（相机未移动）下的可微均匀度优化：冻结物理预测 + 固定 60 px 掩膜 CV。 "
            "现场前置测量（面板斜坡→相机轴映射、去倾斜补偿残差 0.14/0.47 px/mm、可用曝光 "
            "1.6–32 ms、噪声地板 CV 1.1274±0.0055 ⇒ 3σ 0.0165）；C 阶段方向检验 Δ+ = −0.01942 / "
            "Δ− = +0.02132（两向超 3σ 且符号相反 ⇒ 模型梯度方向正确）；D 阶段逐块 A-B-B-A 复测 "
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
        note="物理预测模型实验系统几何标定；runbook 采集 + 本文档结论",
    ),
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
        note="2026-09-22 集成测试记录（冻结：正文与生成时间逐字重发；"
        "被测脚本已在 f19dced5 删除，测试已在 baefba2d 删除）",
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
        note="整形目标函数灵敏度 (噪声地板) 实测；测量内核已下沉到 src/ao_shaping/tools/kernels/sweep_analysis.py",
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
    # -- 数据引用清单 --------------------------------------------------------
    "report/data_reference_inventory.md": ReportProvenance(
        script="scripts/_common/data_reference_inventory.py",
        command="python scripts/_common/data_reference_inventory.py",
        environment="离线",
        note=(
            "静态扫描 report/**/*.{md,json} 对 data/ 的引用并判定本机存在性；"
            "只代表报告引用维度，不代表运行时依赖"
        ),
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
    "report/miicam_simulation/miicam_report.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（仿真 MiiCam，无需设备）",
        note=(
            "仿真 MiiCam 的测试报告 (2026-10-15 生成, 2026-10-29 随上游 30cac66d 入库)。"
            "报告自带的 provenance 块原先指向 "
            "`tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py`, 而该文件"
            "**从未入库**（全仓 `git ls-files` 无此路径）——生成器缺失, 故按手工报告登记, "
            "正文与图保留原样; 若要复活可复现性, 需补回该生成器而不是改这一行。"
        ),
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
            "(SPGD/GS/RL/ML/物理预测模型) 逐条映射 + 6 条可借鉴清单 "
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
        note="本目录用本仓库 `model_in_loop_shaping.py` 的 sim 复现 TU/e 可微物理仿真的四项能力（系统松弛 / 混合拟合 / 低参数 vs 深度 / 样本查询与不确定性采样），并与 `shape_trials` 的实验、物理预测模型标定做定量对照，为论文 7.7 节提供定量证据；T1/T2/T4 定量表由本目录 `t1_noise_relaxation.py`、`t2_mixed_fitting.py`、`t4_uncertainty_sampling.py` 生成（`.venv/bin/python3 report/sim_tue_methods/<脚本>.py` 可复现），T3 复用论文 5.4 节实测数据、无新脚本。全部 sim-only（SimFourierGSNetEnv，64×64，CPU，无硬件），T3 与实验对照用 `shape_trials` 实测 (Santec SLM #1 23020027、Daheng CCD，1064 nm 实际光路)；不触碰硬件写入路径",
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
            "4.7044→1.1656), 并与 MIL 自己的物理预测模型在 1024² 网格上同口径对拍 (复现其落盘的"
            "验证集相关 0.3918); **焦面标度口径已统一**: 孪生按实测标度 2.614022 经 "
            "BenchTwinConfig.from_measured_scale 反推等效焦距 311.32 mm (解析/实测比 "
            "1.000000000), 标称 122.4060 mm 的解析标度与之差 2.54 倍仅作提醒; 反演走 "
            "bench_twin_calibrate.fit_twin 自己的两阶段路径, 过程 GIF 由拟合期间临时包一层"
            "模块级 twin_loss 取优化器真实访问过的候选序列"
        ),
        extra_scripts=("src/ao_shaping/runners/slm/model_in_loop_runner.py",),
    ),
    "report/simulation/sim_credibility/sim_credibility_20261030.md": ReportProvenance(
        script="scripts/slm/render_sim_credibility_report.py",
        command="python scripts/slm/render_sim_credibility_report.py",
        environment="离线（只读 data/debug 的 pkl 与 report/ 下两份 JSON；torch 在 CPU 上跑，无设备、无 GPU）",
        note=(
            "2f 数字孪生的可信度边界 + 可微整形在仿真里的现状: 方向判据用 18 个去重 (相位, 帧) 对的 "
            "153 全配对 / 45 同批次配对 / 397 错配对照, 同批次余弦中位 0.0288 (对照 0.0102, 门限 0.5), "
            "符号一致率 0.5056 ⇒ 孪生对变化方向无信息 (判据 false); 绝对画面零均值 ncc 中位 0.7160 与既有 "
            "not_similar 报告的 max-over-k 0.7160 是同一个点, 不构成新证据; 另复现 `run_benchmark` 默认 "
            "sum1 目标下 `uniformity_loss` 恒 0 与 `efficiency_loss` 零点错位 256 倍的退化 (不修), "
            "并用 0/1 掩膜口径重跑四形状的 GS 与 backprop (两个初相位; SPGD 未重跑) 及跨后端、参数敏感性。"
            "**本文件的正文是手写的** (数字取自两份 JSON), 渲染脚本只生成三张图, 不改正文也不改数字"
        ),
        extra_scripts=(
            "scripts/ml/analyze_sim2f_direction_corpus.py",
            "scripts/slm/analyze_sim_differentiable.py",
            "scripts/ml/validate_sim2f_vs_hw_corpus.py",
        ),
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
    # -- 物理预测模型 / loss 缺陷排查 ------------------------------------------
    "report/loss_defects/README.md": ReportProvenance(
        script=HANDWRITTEN,
        command="python -m ml.zernike.train_amp",
        environment="离线",
        note="物理预测模型/loss 缺陷排查结论（人工撰写）；同目录 *.json 为各探针面板",
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
        note="物理预测模型改进尝试: 设备参数/椭圆loss/attention/U-Net/图像增强, 3 seed 配对",
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
        note="结合可微实验、离焦物理预测模型及曝光诊断制定探索性短块整形协议；本方案没有新增设备测量。",
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
        note="2026-10-22 当前功率与主动离焦下的聚焦、斜坡及未补偿/补偿配对相位采集和低参数物理预测模型验证；保留失败边界。",
        extra_scripts=(
            "scripts/slm/capture_defocus_forward_dataset.py",
            "scripts/slm/fit_defocus_forward_dataset.py",
            "scripts/slm/evaluate_defocus_forward_dataset.py",
            "scripts/slm/validate_defocus_exposure.py",
            "scripts/slm/summarize_defocus_forward_experiment.py",
            "src/ao_shaping/tools/slm/panel/alignment_probe.py",
        ),
    ),
    "report/shape_trials/no_gs_ccd_moved/live_20261027.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="实验（Santec #1 / Daheng；CCD 移动后的无 GS 探针与可微优化）",
        note="本轮原始数据、曝光读回、几何、物理预测与同曝光验收的实验中记录。",
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/panel/alignment_probe.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "scripts/slm/analyze_no_gs_session.py",
        ),
    ),
    "report/shape_trials/no_gs_ccd_moved/analysis_20261027.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线分析（本轮 Santec #1 / Daheng 原始数据；设备已释放）",
        note="无 GS CCD 验收、曝光偏置、激活函数位置与下一轮多形状多尺度响应区间方案。",
        extra_scripts=(
            "scripts/slm/analyze_no_gs_run.py",
            "scripts/slm/analyze_no_gs_response.py",
            "src/ao_shaping/tools/slm/transfer/capture_response.py",
            "src/ao_shaping/tools/slm/transfer/response.py",
            "src/ao_shaping/tools/slm/transfer/direction.py",
            "src/ao_shaping/tools/slm/transfer/local_response.py",
            "src/ao_shaping/utils/image/model_response.py",
        ),
    ),
    "report/shape_trials/multishape_scale/multishape_scale_planning_20261028.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线规划（设备已预检；实验开始后于 2026-10-28 追加 H3 勘误与 --no-gs-seed 旗标说明）",
        note=(
            "四形状 × 三尺度（200/600/1200 原生 CCD px）同载波矩阵的实验方案："
            "几何落点、曝光阶梯、通过/拒绝标准与数据目录；不含优化结论。"
            "2026-10-28 拟合后追加勘误：实测载波位移 (14.76, −129.70)，"
            "200 格零级不再与目标分离（排除区与目标重叠 5.61%）。"
        ),
        extra_scripts=(
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
        ),
    ),
    "report/shape_trials/multishape_scale/multishape_scale_live_20261028.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件（Santec #1 SLM + Daheng CCD；WFS / NLight DM / MIICAM 未连接）",
        note=(
            "多形状 × 三尺度同载波闭环整形：实验中记录（2026-10-28）。"
            "Session A（1200 原生 px）的 16 个形状×阶段（square/triangle/star/circle 各 4 段）"
            "全部完成，共 112 个验收块（28 块/形状；journal 16 条 ``complete``、无 error/failed）；"
            "records 已导出（process.csv 179,587 B / process.h5 428,279,000 B / 336 个 sidecar）；"
            "common_replay 完成，common_replay_dense 仍在写盘、本次未消费；s600 / s200 跨尺度未开始。"
            "一次早期 relaunch 在 square/transport_cv 触发饱和守卫而中止，原完成数据未受损。"
            "2026-10-29 追加 §7：跨会话汇总（200 px 会话的净结论）、对 §6 第 3 条时效性的修正、"
            "以及把「反投影在小目标上胜出」降级为「在 circle / triangle 上成立」——"
            "跨尺度部分已跑完 s200 四形状；square 剔除饱和行后仍为真实回归（+0.170/+0.165，"
            "约噪声三倍），star 仅一档可判（no_difference）。"
        ),
        extra_scripts=(
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "scripts/slm/generate_multishape_scale_live_figures.py",
        ),
    ),
    "report/shape_trials/multishape_scale/multishape_scale_analysis_20261028.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线分析（读取 Session A 运行目录的 stage 子目录、common_replay*/ 重放 npz 与 manifest；不连接设备）",
        note=(
            "多形状 × 三尺度同载波闭环整形：实验后分析（2026-10-28）。"
            "三层证据收口（模型验证集 / 逐块 CV / 重放饱和审计）；"
            "新增第（e）条硬约束——CV 随曝光单调上升并饱和、跨曝光档 CV 不可比；"
            "给出目标对照、实验差距（G1–G6）与下一步方案（N1–N6，含具体命令），"
            "并检索 Zotero 文库给出候选改进方法的文献依据（Mastiani 2024 / Cho 2026 GRAPHYCS / Chromatix 2026 等）。"
            "common_replay 的 90 ms 块末 4 条失效记录由 common_replay_r2 洁净重测覆盖。"
        ),
        extra_scripts=(
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "scripts/slm/generate_multishape_scale_live_figures.py",
        ),
    ),
    "report/shape_trials/multishape_scale/multishape_scale_wallclock_20261028.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线分析（读取 Session A/B 运行目录与光路探针落盘；光路复核为实验系统实测探针；修复的硬件收益在 s600_fixed 会话实测中）",
        note=(
            "多形状 × 三尺度闭环整形 campaign 的耗时剖析与写放大治理："
            "按 journal ``status`` 字段重算逐阶段耗时（推翻「相邻同名配对」错配），"
            "定位主导成本为 ``validate.py`` 的 ``iterations.pkl`` 每次优化迭代整体重写"
            "（每条含 1.56 MB 相位副本，O(n²) 落盘，每 campaign ≈224 GB）；"
            "改为每块落盘一次，离线基准 ≈14×、写出字节 15.3×↓、条目逐条一致；"
            "含一次自我纠错（``star/mono12_cv`` 的 1438.8 s 系错误配对，真值为中断 1227.9 s + 恢复 210.9 s）；"
            "并含光路曝光强度变化后的复核（判定 ok，强度 +1.35×、对准保持，质心仅移 2.3 px）。"
        ),
        extra_scripts=(
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "scripts/slm/probe_optical_path_validity.py",
            "scripts/slm/plan_transfer_next_run.py",
        ),
    ),
    "report/shape_trials/multishape_scale/live_20261029_device_session.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件（Santec #1 SLM + Daheng CCD；WFS / NLight DM / MIICAM 未连接）",
        note=(
            "2026-10-29 设备在线多尺度整形会话的实测记录：lane 00–04 在 0.7 功率下完成，"
            "05–08 在激光功率升到 1.0 后从头重跑/新跑，13–15 完成；16 首次失于 SLM 陈旧"
            "句柄（错误码 −10003），断电重置后重跑拿到 4/4 阶段，最后的 `common_replay` "
            "在 18:35 激光停机处截断（12 个归档，无 `released.json`）；31–34 与 17–20 "
            "因停机从未启动（25 条编制的最终状态见 §3 / §9.6）。"
            "**一次会话不跨功率**：0.7 的半成品全部归档，跨功率只读形态不读量级。"
            "200 px 解析初始化（0.7）：circle 15/15 有效、dCV −0.286/−0.126/−0.127 三档 "
            "supported；triangle 13/15、−0.161/−0.173 两档 supported；square 11/15 的 "
            "dCV +0.170/+0.165 **已撤回**——同一批帧换回闭环接受口径给 −0.025（符号相反），"
            "且同一 `panel_gray` 字节 55 s 内峰值 +57%、主瓣换位 63 px ⇒ 效应量上界 "
            "\\|dCV\\| ≲ 0.02，见 issue #217；star 5/15 仅一档 no_difference。"
            "600 px（1.0）：四形状审计 dCV = circle −0.318/−0.314/−0.313、square "
            "−0.619/−0.614/−0.636、triangle −0.348（仅一档可判）、star −0.044/−0.041/"
            "−0.045，均 supported；ee_rel 0.46–0.69，零级排除 0.031，饱和 0（triangle 6 行"
            "单行截断）。"
            "含审计工具缺陷（`clipped` 行被标注却仍进入分组均值）及其本轮修正；"
            "`core_peak/exposure` 被证明不是辐射率（同一天两条 lane 方向相反），"
            "报告只承认平场探针读数与 ≥6× 跳变。"
            "§5.5 逐组合字段台账：本轮 objective 以 `formation=False` 运行，"
            "`l_partition` / `l_boundary` 逐轮恒为 0.0，故闭环**没有**优化形状项；"
            "形状覆盖 / 边界误差已在 §5.8 用同一批归档帧**离线复算**补上"
            "（`scripts/slm/audit_shape_agreement.py` 流式产出 "
            "`artifacts/shape_agreement_20261029.jsonl`，222 帧 / 12 会话：所有组合的 "
            "`filled_fraction` ≤ 10.7%，且两条 CV 改善最大的会话（600 px circle / "
            "square）填充反而降到原来的 1/6–1/8 ⇒ 结论仍只覆盖"
            "「更均匀 + 能量没丢」，且现在有「没长成目标形状」的量化上界）；"
            "Zernike 增量本轮**仍是空白**。"
            "图 9（lane 16 四段逐帧 `cv` / 曝光，68 帧）由 "
            "`scripts/slm/plot_session_process_traces.py` 从各段 `recording/process.csv` "
            "渲染，数值先落 `artifacts/star_stage_traces_20261029.json`：它证的是"
            "「四段都跑了、每帧可追」，不是「变好了」（段内首末差不作收益判据）。"
            "配图由 `generate_multishape_nogs_20261029_figures.py` 渲染："
            "CV 轨迹与逐段首末柱取自 `capture_records.pkl`（与 §5.2 表同源），"
            "冻结光阑审计图直接调用 `audit_block`/`summarise_block`（与 §5.1 表同源），"
            "帧图与 GIF 为 `common_replay/*.npz` 的判据帧。"
        ),
        extra_scripts=(
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "scripts/slm/audit_replay_blocks.py",
            "scripts/slm/audit_shape_agreement.py",
            "scripts/slm/generate_multishape_shape_figure.py",
            "scripts/slm/plot_session_process_traces.py",
            "scripts/slm/probe_optical_path_validity.py",
            "scripts/slm/analyze_session_time_budget.py",
            "scripts/slm/generate_multishape_nogs_20261029_figures.py",
            "scripts/slm/generate_multishape_s600_power1p0_figures.py",
            "scripts/slm/generate_multishape_jacobian_figure.py",
        ),
    ),
    "report/shape_trials/multishape_scale/next_round_20261029.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（issue #213 质量建议 + 2026-10-29 会话实测记录；本文的代码项已在仓内实现并通过单元测试，尚未实验）",
        note=(
            "2026-10-29 之后的质量改善实施方案：按 issue #213 comment 6466195110 的"
            "质量侧优先级（3a Jacobian 逐块校验 / 1b 频域 MTF 正则 / 2b 像素级不均匀性掩膜 / "
            "4c 多目标 Pareto）给出改法、开关、验收判据与失败形态；"
            "依据本轮实测把「饱和先决」与「审计 clipped 语义修正」提到 P0，"
            "并明确 3a 只实现缩 cap（不转纯实测梯度）、1b 默认关闭等实测 MTF。"
            "所有判定统一使用重复导出的不确定度（6 个 ABBA 周期可达 p<0.05），不用固定阈值。"
            "§9 落地状态区分已实现（3a / 1b / P0-2 审计 / 4c 阶段化权重）与未实现"
            "（2b 像素级掩膜 —— 缺实测标定数据，反演无法验证，理由写在 §5 末尾），"
            "并记下本轮走 `cv` 变体导致形状覆盖 / 边界误差未测这一字段缺口。"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/jacobian_guard.py",
            "src/ao_shaping/algorithm/signal_processing/spectral_regulariser.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "scripts/slm/audit_replay_blocks.py",
        ),
    ),
    "report/shape_trials/transfer_square1200/transfer_square1200_20261027.md": ReportProvenance(
        script="scripts/slm/generate_transfer_square1200_report.py",
        command="python scripts/slm/generate_transfer_square1200_report.py",
        environment="实验（Santec #1 与 Daheng，1064 nm，1200 CCD 像元）",
        note=(
            "手动干扰后用迁移适配的物理预测模型整形 1200 像元方形："
            "alignment→full 的验证选择、模型模式与 CCD 锚定两段的 A–B–B–A 实测判决、"
            "方向性检查、Recorder 过程记录与动画；"
            "含「几何被继承导致适配永久卡死」这一机制的修复记录。"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/workflow.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/adaptation.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/paired.py",
            "src/ao_shaping/tools/slm/transfer/recording.py",
            "src/ao_shaping/tools/slm/transfer/capture.py",
            "src/ao_shaping/tools/slm/transfer/radiometry.py",
        ),
    ),
    "report/shape_trials/tri_plane_20261027/tri_plane_20261027.md": ReportProvenance(
        script="scripts/slm/generate_tri_plane_report.py",
        command="python scripts/slm/generate_tri_plane_report.py",
        environment="实验（Santec #1 与 Daheng，1064 nm，1200 CCD 像元）",
        note=(
            "三角形与五角星 1.5 倍窗口 shaping + 平面相位起始对比："
            "窗口开到边长 1.5 倍使目标外光可见；平面起始 3/4 块接受、CV 0.6632→0.5967，"
            "优于 GS 起始的 2/4 块；GIF 渲染修复（整窗裁剪 + 目标轮廓 + 重复采集标注）。"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/recording.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
            "src/ao_shaping/tools/slm/transfer/workflow.py",
            "src/ao_shaping/tools/slm/transfer/cold_start.py",
        ),
    ),
    "report/shape_trials/dual_optimization/experiment_plan_20261027.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（方案文档；设备已预检，实验尚待 G0-2 光路漂移处置）",
        note=(
            "迁移学习 + model-in-loop 双优化流程的实验方案："
            "G0 前置门禁、逐轮交替的判决口径与记录要求；不含任何新测量。"
        ),
        extra_scripts=(
            "src/ao_shaping/runners/slm/model_in_loop_runner.py",
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
        ),
    ),
    "report/shape_trials/physical_forward_no_gs_plan_20261027.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（历史报告分析与下一轮实验规划；未连接设备）",
        note="无 GS 相位起点：分层指标、物理参数与局部响应校正、CCD 验收及缓漂实验设计；不新增实验测量结论。",
        extra_scripts=(
            "src/ao_shaping/utils/image/physical_model_metrics.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/validate.py",
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
    "report/shape_trials/defocus_differentiable/diffopt_efficiency_20261108.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线（纯 CPU；读磁盘冻结 checkpoint，未占用 SLM/CCD/激光）",
        note=(
            "可微优化效率拆解探针: 每步成本(前向/反向/step, fft_size, 线程, 批量起点)、"
            "收敛所需步数(Adam/SGD 学习率扫描, LBFGS)、参数化地板(模式集 Noll 5-15/5-25/5-35)、"
            "部署路径 step_rms_cap 绑定(cumulative vs per_step accept); "
            "结论按「部署路径 cap 欠预算 > 裸 Adam 学习率伪影 > 参数化地板 > 每步毫秒」排序, "
            "并与 multishape_capability 的硬件块边际增益塌陷对齐"
        ),
        extra_scripts=("scripts/slm/diffopt_efficiency_probe_20261108.py",),
    ),
    "report/shape_trials/optical_path_status_20261109.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "实验室两套光路(按 SLM 序列号区分)的设备清单与截至 2026-11-09 的已知状态: "
            "两台 Santec SLM-200 —— 光路A(SLM#1 序列号 22030108, 1064nm 2f-Fourier + Daheng MER2 相机)"
            "为本项目主用台, 11-09 零级未压制(E0/E1≈1.2)但几何映射未变, 已落 locate() 零级掩膜对策; "
            "光路B(SLM#2 序列号 23020026, 532nm 中继 + Thorlabs MLA150M-5C WFS)停留在 2026-09/10 标定; "
            "光路A的 11-09 光阑关闭后整形复现与重放见 aperture_closed_20261109/aperture_closed_shaping_20261109.md"
        ),
    ),
    "report/shape_trials/aperture_closed_20261109/aperture_closed_shaping_20261109.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="实验+离线",
        extra_scripts=(
            "scripts/slm/diffopt_efficiency_probe_20261108.py",
            "scripts/slm/diffopt_dispatch_probe_20261110.py",
            "src/ao_shaping/runners/slm/transfer_shapes_runner.py",
            "src/ao_shaping/tools/slm/transfer/replay_phases.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py",
        ),
        note=(
            "光阑关闭后光路(零级未压制 E0/E1≈1.2, 曝光窗 0.05-0.2ms, 几何未变)重做 "
            "side=1200 square/triangle/star 整形: 全新标定 campaign 复现 + 共同重放成功(0.05/0.2ms) "
            "+ 可微优化 GPU/学习率效率拆解; 含与 10-28 基线同口径对比(跨曝光不作直接对等)"
        ),
    ),
    "report/shape_trials/experiment_summary_20261108.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "2026-11-02 ~ 2026-11-08 形状整形全部报告的汇总: 重点转向动态整形; "
            "不含任何新测量, 全部数字来自已有报告与保存的实验产物离线复算; "
            "含当前现状与最优路线、逐报告指标与任务项, 供脱离硬件复现"
        ),
    ),
    "report/shape_trials/gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md": ReportProvenance(
        script="scripts/slm/generate_gs_refine_fwdmodel_report.py",
        command="python scripts/slm/generate_gs_refine_fwdmodel_report.py",
        environment="离线+硬件",
        note=(
            "slm-gs-refine 三栏 GIF 第三栏的物理预测模型标定；本脚本离线重算标定前后对比、"
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
        "离线重算，三次会话的交错重放数值由 scripts/slm/replay_gs_refine_hw.py 实验采集；"
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
            "2026-10-16/17 物理预测模型闭环方斑均匀度实验：固定曝光交错重放比较软边缘 CV "
            "目标与两轮热启动 (SLM 22030108 + 大恒 FJB24112232, 1064 nm)；"
            "并实测照明半径、据以把 GUI 侧栏的曝光与照明半径改为可探测。"
            "2026-10-17 追加节记录两次由实测数据揭出的实现缺陷 (半径用 edge/2 而非 "
            "edge−centre；曝光探测不校验确认帧是否截断) 及其修复。"
            "2026-10-26 追加节记录会话曝光上限估计的修正 (逐档取最小改为拟合 + 两道斜率证据门)："
            "同一条 2026-10-24 实测阶梯的上限 0.3178 → 0.4696 ms，工作曝光 0.159 → 0.235 ms"
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
            "2026-10-19 相机位置调整后物理预测模型校正与多形状整形 (Santec #1 22030108 + "
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
        script=HANDWRITTEN,
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
        script=HANDWRITTEN,
        environment="离线",
        note=(
            "2026-10-21 baseline_a 变体的 MIL 单次运行记录 (离线读盘, 不碰设备)："
            "Step A 像差系数 + 信赖域截断、Step B 600 步轨迹、逐探针实测/预测远场与"
            "相关/SSIM、逐轮 CV/EE/IoU、bake_off_rejected 结论 (整形 0.3451 未打赢"
            "平场 0.4025, 落盘相位为零相位平场)；过程动画每帧给相位/实测/预测远场"
            "(叠目标掩模轮廓) + 差值 + 物理预测模型 metrics + 目标 metrics"
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
            "2026-10-16 物理预测模型闭环 Step B 匀化权重与 Step A 重拟合消融 "
            "(SLM 22030108 + 大恒 FJB24112232, 1064 nm，另有 CPU 仿真)；提高 Step B "
            "均匀度权重未改善方斑 (固定 0.200 ms 交错重放平均 CV 0.8027 升至 0.8989、"
            "EE 0.6919 降至 0.5177)，Step A=0 只有有代价的小幅 CV 趋势 "
            "(CV 0.8028 降至 0.7887，同时 EE 0.7071 降至 0.5888)，两者均被复合质量分拒绝；"
            "结论为保留默认 Step A=80 与 0.4/0.6 权重"
        ),
        extra_scripts=(
            "scripts/slm/check_flat_stability.py",
            "scripts/slm/generate_shape_trial_profiles.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261015.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "2026-10-15 物理预测模型闭环实验预检 (feat/r71-r84-closed-loop worktree; "
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
        note="2026-10-14 物理预测模型闭环实验复核（exp/model-in-loop-20261014 worktree）；"
        "随机相位面板扫描定光斑位置 (960,670)，传播几何沿用 20261013 的 bench_geometry.json，"
        "本日未重新独立标定其尺度/入射振幅/面板配准；图内 X/Y 光强截面由 "
        "generate_shape_trial_profiles.py 从记录的原始 CCD 帧重新生成",
        extra_scripts=("scripts/slm/generate_shape_trial_profiles.py",),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_live_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="2026-10-13 物理预测模型闭环整形实验记录；Step A (Zernike 校准 + NNLS) → Step B (SPGD + Adam) 双路径架构；images/ 动画由 generate_gif_from_pkl.py 渲染",
        extra_scripts=(
            "scripts/slm/generate_gif_from_pkl.py",
            "scripts/slm/generate_shape_trial_profiles.py",
        ),
    ),
    "report/shape_trials/model_in_loop/model_in_loop_next_steps_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note="2026-10-13 物理预测模型闭环暂停后续步骤记录",
    ),
    "report/shape_trials/model_in_loop/model_in_loop_pause_handoff_20261013.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note="2026-10-13 物理预测模型闭环暂停交接记录；当前状态、证据、工程变更摘要",
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
        script=HANDWRITTEN,
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
            "round3_20261022 的归档产物读出 ⇒ 两边的差就是**物理预测模型**的差"
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
        extra_scripts=("scripts/slm/hw_gs_square_scans.py",),
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
    "report/slm/differentiable_opt/artifacts/model_hw_fidelity_20261116/model_hw_fidelity.md": ReportProvenance(
        script="scripts/slm/analyze_model_hw_fidelity.py",
        command=(
            "python scripts/slm/analyze_model_hw_fidelity.py "
            "--out report/slm/differentiable_opt/artifacts/model_hw_fidelity_20261116"
        ),
        environment="离线",
        note=(
            "可微优化**模型预测 vs 上屏实测**的离线一致性核对: 只读已落盘的运行目录 "
            "(model_farfield.npy + measured_model_grid.npy + run_config.json + history*.csv), "
            "全程不占用 SLM / CCD / 激光。覆盖 10 次运行 (含 phys_20261116_01/02), 逐运行给出 "
            "线性相关系数、归一化 RMS 误差、峰值偏移、FWHM 比与尺度扫描 s*; 用于判定"
            "「可微优化预测远场与实测远场是否几乎一致」——当前结论为不一致, 主因是物理预测模型"
            "缺传播/离焦项与 SLM 驱动链 (LUT/基底相位/面板映射), 实测零填充黑带来自 CCD "
            "网格外区域, 与模型无关"
        ),
        extra_scripts=("scripts/_common/differentiable_problem.py",),
    ),
    "report/slm/differentiable_opt/status_20261117.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "跨系列现状总结 (Q1 模型-实测一致性 / Q2 学习率与调度 / 2π 相位标度 / "
            "原点陈旧根因 / 拟合参数落地)。全部数字可追到 data/debug/differentiable_opt/ 与 "
            "report/slm/differentiable_opt/artifacts/ 下的落盘产物; 核心新证据是阶梯"
            "误用存档几何 (data/slm_geometry_calib.npz, center y=962.63) 与同一晚现场"
            "标定 (y=1104.84) 的 142 px 对照, 只读 pred/frames 的离线原点搜索复核, 以及"
            "把现场标定写回 npz 的快照动作 (旧值存为 slm_geometry_calib_20261023.npz)"
        ),
        extra_scripts=(
            "scripts/slm/persist_live_geometry.py",
            "scripts/slm/hw_forward_model_fidelity_ladder.py",
            "scripts/slm/hw_differentiable_phase_opt.py",
            "scripts/slm/analyze_fwdmodel_fit.py",
        ),
    ),
    "report/slm/differentiable_opt/artifacts/lr_sweep_20261116/lr_sweep.md": ReportProvenance(
        script="scripts/slm/sweep_differentiable_lr.py",
        command=(
            "python scripts/slm/sweep_differentiable_lr.py "
            "--out report/slm/differentiable_opt/artifacts/lr_sweep_20261116"
        ),
        environment="离线",
        note=(
            "可微整形优化器学习率扫描 (120 步, GPU/CUDA): flat lr 上调 (0.2 / 0.4) 终态损失比基线 "
            "lr=0.05 高约 21%, CV 0.373 vs 0.292 —— 调大平 LR **不可行**; 余弦衰减 (峰值 lr=0.1, "
            "无 warmup 无下限) 在第 90 步达到基线终态质量 99%, 终态损失 1.206e-4 vs 1.333e-4, "
            "但压缩预算到 90 步只到 95%, 须保留完整 120 步周期。结论仅离线, 硬件未验证; "
            "硬件验证脚本与命令见 lr_sweep.md 末节"
        ),
        extra_scripts=(
            "scripts/_common/differentiable_problem.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_beam.py",
        ),
    ),
    "report/slm/differentiable_opt/artifacts/fidelity_ladder_20261116/README.md": ReportProvenance(
        script=HANDWRITTEN,
        command="",
        environment="人工撰写（阶梯目录的导读与运行手册; 数据产物由 ladder_summary.md 注册项覆盖）",
        note=(
            "物理预测模型保真度阶梯的产物目录 (占位): 脚本在 SLM #1 上依次上屏多个相位图案, "
            "采集 CCD 远场, 逐图案对比模型预测与实测, 用于判定怀疑的缺失离焦项。整个阶梯只"
            "开/关一次 SLM + 相机 (背靠背开关会触发驱动冻结), 设备占用记入 device_log.json。"
            "目前仅以 --dry-run 离线演练过 (合成相位 + 占位帧), 尚未执行 --execute, 因此本目录"
            "还没有硬件结果, 不伪造数字; 运行手册与判据优先级见正文。主判据为信号区秩相关 "
            "spearman_signal, 不使用全域 spearman_coverage 下结论"
        ),
        extra_scripts=("scripts/slm/_hw_fidelity_ladder_sections.py",),
    ),
    "report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261117/fwdmodel_fit_20261117.md": ReportProvenance(
        script="scripts/slm/analyze_fwdmodel_fit_20261117.py",
        command=(
            "python scripts/slm/analyze_fwdmodel_fit_20261117.py "
            "--out report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261117/fwdmodel_fit_20261117.md"
        ),
        environment="离线 + CUDA GPU (RTX 5080 Laptop, torch 2.11.0+cu128); 不连接设备",
        note=(
            "物理预测模型参数拟合与可微共同优化的离线验证: 阶梯陈旧 y 原点 (存档 962.6 vs 现场 1104.8) 的四列对照, "
            "gs_init 低相关归因 (GS 未收敛), 轴向口径择优与留一验证, 阶段 B 实测历史对照。"
        ),
        extra_scripts=(
            "scripts/slm/analyze_fwdmodel_fit.py",
            "scripts/slm/_hw_fidelity_ladder_sections.py",
            "scripts/slm/hw_differentiable_phase_opt.py",
        ),
    ),
    "report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261116/fwdmodel_fit.md": ReportProvenance(
        script="scripts/slm/analyze_fwdmodel_fit.py",
        command=(
            "python scripts/slm/analyze_fwdmodel_fit.py "
            "--out report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261116"
        ),
        environment="离线",
        note=(
            "前向物理预测模型缺陷的离线拟合归因 (驱动无关, 只读已落盘的相位/实测/模型三联数据): "
            "头号结论是冻结模型的光轴**陈旧 ~179 个 CCD 像素** (2026-11-07 M3 原点 1306,944 vs "
            "2026-11-16 实测零级 1224,1105) —— 仅此一项把 Pearson 从 -0.003 拉到 0.95~0.98、"
            "验证集形状残差降 65%, **离焦不可辨识** (目标敏感度 1e-4), 993/1023 灰度假设"
            "**在方向上被证伪且量级上无关紧要** (3% 相位标度只动远场 NRMSE 0.012)。给出拟合的"
            "横向尺度/中心/模糊等全局量、验证集前后对照、联合优化可辨识性判定与补数据清单"
        ),
        extra_scripts=(
            "scripts/_common/differentiable_problem.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py",
        ),
    ),
    "report/shape_trials/issue235_model_fill_diagnostic_20261117/issue235_model_fill_diagnostic_20261117.md": ReportProvenance(
        # 人工撰写: 脚本只产出 summary.md 与 diagnostic.json, 中文分析正文不在任何
        # 发射器里 (尾节"复现与产物"同理) —— 判 HANDWRITTEN 而非给脚本塞死代码。
        script=HANDWRITTEN,
        environment="离线（回放 2026-11-17 14:07–14:09 已采集的 14 帧会话；本次分析未占用 SLM/相机/激光）",
        note=(
            "#235 冻结模型整幅形态诊断（Q1–Q3）：在同一坐标系（整幅原生平面 1944×2592，实测帧与模型预测同公式）"
            "量化冻结 model640 前向与实测的形态差。Q1 集中度不可预测（显示相位门禁 0 diff，模型预测满幅平顶、实测点状，"
            "整幅 Pearson ≈ −0.010）；Q2 单参数扫描（phase_scale / amplitude_width / sigma_ccd / defocus / 像素尺度）"
            "无一能同时复现实测的功率与半高像素数；Q3 反解表明实测光斑隐含有效照明口径 ~363 px，与冻结包络 (352, 133) px 同量级，"
            "即问题不在口径量级，而在照明包络→远场的形态映射。数字全部由脚本读取 diagnostic.json 渲染。"
        ),
        extra_scripts=(
            "scripts/slm/square_1000_model_diagnostic_20261030.py",
        ),
    ),
    "report/shape_trials/issue235_model_fill_diagnostic_20261117/summary.md": ReportProvenance(
        script="scripts/slm/issue235_model_fill_diagnostic_20261117.py",
        command=(
            "python scripts/slm/issue235_model_fill_diagnostic_20261117.py "
            "--session data/debug/square1000_model_diag_20261117_140741 "
            "--out data/debug/issue235_model_fill_20261117"
        ),
        environment="离线（回放 2026-11-17 14:07–14:09 已采集的 14 帧会话；本次分析未占用 SLM/相机/激光）",
        note=(
            "#235 诊断报告的可读摘要（脚本自动生成）：Q1–Q3 结论要点与关键数字，"
            "数字全部来自同目录 diagnostic.json。与正式报告 issue235_model_fill_diagnostic_20261117.md 同源。"
        ),
        extra_scripts=(
            "scripts/slm/square_1000_model_diagnostic_20261030.py",
        ),
    ),
    "report/shape_trials/speed_decouple_capture_20261120.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线+硬件",
        note=(
            "动态整形速度解耦采集正式报告：30 条 transition（24 GS + 6 model 诊断）全 ok，"
            "原始帧 (52,1944,2592) uint16，冻结曝光 0.08ms；实验约 2.3fps 瓶颈在 SLM 写入+换槽"
            "(中位 0.312s)，非 CCD 读出或像素翻转。数据来自 "
            "data/debug/slm_transfer/speed_decouple_capture_20261120_v2/。transfer light/none 仍暂停。"
        ),
        extra_scripts=(
            "scripts/slm/dynamic_shaping_capture_20261120.py",
            "scripts/slm/build_speed_decouple_plan_20261120.py",
            "scripts/slm/plot_speed_decouple_timing_20261120.py",
        ),
    ),
    "report/shape_trials/candidate_parallelism_20261117.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="仅 GPU 离线（RTX 5080 Laptop，grid 256 cuda）；未占用 SLM / CCD / 激光，全程无实验动作",
        note=(
            "可微整形 8 候选模型内并行（train_beam_shaping n_candidates=8 默认值）的改动、验证与代价记录："
            "单候选路径逐位不变（批量前向 absdiff=0、梯度相对差 6.6e-5、54+37+9 测试通过）；"
            "实测代价 300 迭代生产设置下 8 候选比单候选慢 4.6–5.8×，非凸目标最终损失改善 1.4–2.2%、"
            "单盆地目标（square）几乎无收益（0.07%）；并行相对 K 次串行重启稳定 1.4–2.6×（K=8 时 1.39×）。"
            "结论：这是「用时间换 1–2% 质量」的取舍而非免费加速；单盆地/规则目标建议 n_candidates=1（或 2–4），"
            "非凸/多瓣任意形状质量优先时 8 候选值得。仅 optimizer=adam 支持多候选，lbfgs 自动回退 1。人工撰写。"
        ),
        extra_scripts=(
            "src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_propagation.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_objective.py",
        ),
    ),
    "report/shape_trials/transfer_online_params/transfer_online_params_20261121.md": ReportProvenance(
        script="scripts/slm/visualize_transfer_online_params.py",
        command=(
            "python scripts/slm/visualize_transfer_online_params.py "
            "--session data/debug/slm_transfer/20261105_173655 "
            "--trace data/debug/slm_transfer/debug_weights_real_force_20261121/fit/debug_weights/weights_history.jsonl "
            "--out report/shape_trials/transfer_online_params"
        ),
        environment="离线",
        note=(
            "transfer 在线训练参数变化可视化：闭环轮次图从 rounds.json 读指标、从各轮 frozen_parameters.json "
            "补出参数（6 轮，参数只在 0→1 轮跳变后持平，全部未验收）；逐迭代图从 "
            "debug_weights_real_force_20261121 的 weights_history.jsonl 读真实采集下的 7 次迭代 × 18 参数，"
            "训练损失 0.39325→0.34548。两图的参数线各自按全程极差归一化。纯离线重绘既有存档，未操作设备。"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "src/ao_shaping/tools/slm/transfer/workflow.py",
        ),
    ),
    "report/shape_trials/debug_weights_real_verification_20261121.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件",
        note=(
            "--debug-weights 逐轮权重轨迹落盘的实验系统验证：实验前光路六步探测判定 ok，"
            "短时冷启动采集+拟合 6 轮在 SLM #1 + Daheng CCD 上完整在线实现，"
            "weights_history.jsonl / weights_trajectory.npz / weights_summary.json 三份产物"
            "在真实采集数据上落地，训练损失 0.39325→0.34548；聚焦离线测试 6 用例通过。"
            "debug 数据以标准 JSONL/NPZ 格式保存，支持会话后离线回放；"
            "scripts/slm/visualize_weights_tsne.py 用 t-SNE 将 Zernike 系数轨迹降维到 2D 可视化。"
            "scripts/slm/diagnose_zernike_direction_lr.py 基于 34 条历史轨迹做观察性诊断："
            "方向余弦后 10 轮中位数 0.9978、91.2% 轨迹后段方向一致、步长-损失下降 Spearman 0.7417，"
            "支持「早期/中期可放大 residual lr、噪声地板后收回」的一阶代理处方，不改默认学习率。"
            "口径要点：reuse_latest 命中已有 latest 模型会跳过采集+拟合导致无权重产物，"
            "须用 AO_SHAPING_WEIGHTS_DIR 重定向到空目录强制冷启动。人工撰写。"
        ),
        extra_scripts=(
            "src/ao_shaping/tools/slm/transfer/workflow.py",
            "src/ao_shaping/tools/slm/transfer/fit.py",
            "scripts/slm/probe_optical_path_validity.py",
            "scripts/slm/visualize_weights_tsne.py",
            "scripts/slm/diagnose_zernike_direction_lr.py",
        ),
    ),
    "report/shape_trials/diff_phase_monotonicity_replay_20261121.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note=(
            "可微光束优化逐轮 SLM 相位的离线回放与逐像素单调性诊断："
            "逐轮相位未落盘，但用 scripts/slm/render_differentiable_opt_animation.py 的重放设施"
            "逐位精确重建 121 帧相位栈（120 轮 loss 对 history.csv max_rel_diff=0、"
            "最终相位对 phase_final.npy max_abs_diff=0）。"
            "scripts/slm/replay_diff_phase_monotonicity.py 经 phase_model_to_panel 做中心对齐+"
            "最近邻尺寸缩放后，统计 active 像素（4.9%）的符号一致性：均值 0.611、"
            "单调(>=0.9)像素仅 0.06%；真实 lr 反事实重放显示当前 lr=0.05 近最优，"
            "0.08/0.12 使最终 loss 变差 8%/22%。结论：像素级相位更新不单调，不建议放大 lr，"
            "与 transfer fit 阶段 Zernike 放大 lr 处方不可混用。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/replay_diff_phase_monotonicity.py",
            "scripts/slm/render_differentiable_opt_animation.py",
            "scripts/slm/hw_differentiable_phase_opt.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_beam.py",
        ),
    ),
    "report/shape_trials/multistart_batch_and_cpu_gpu_analysis_20261121.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线",
        note=(
            "可微光束整形 B=2 多起点批量并行 + CPU→GPU 整合分析："
            "scripts/slm/sweep_differentiable_multistart_batch.py 复用 differentiable_problem 重建入口，"
            "一次 batch FFT 前向+一次反传+单 Adam(B,M,M)，每条 trajectory 独立峰值归一化、独立 copy 初相；"
            "真实存档 20261116_114424 跑 120 轮，batch trajectory 0 对照单条串行 loss max_rel_diff=6.56e-7、"
            "最终相位 max_abs_diff=3.34e-6（容差内 PASS，非 bit-exact）；双初相 RMS 差 0.05 证明非重复计算。"
            "端到端吞吐 3.37x（batch 1486 vs 串行 441 fields/s，其中混入去掉 D2H 同步/记账的收益，"
            "纯 batching 约 1.9x，不宣称实验帧率提升）。CPU→GPU 分析：P4 200 步全 GPU，P6b 闭环 5 步的 "
            "scipy map_coordinates / numpy 可搬 torch（grid_sample 有现成先例），但秒级 SLM/CCD I/O 主导延迟；"
            "建议仅做 grid_sample 重采样 + 消除重复 D2H 同步。人工撰写。"
        ),
        extra_scripts=(
            "scripts/slm/sweep_differentiable_multistart_batch.py",
            "scripts/_common/differentiable_problem.py",
            "src/ao_shaping/algorithm/signal_processing/differentiable_beam.py",
            "src/ao_shaping/algorithm/_torch_support.py",
            "scripts/slm/hw_differentiable_phase_opt.py",
            "src/ao_shaping/utils/wavefront/calibration_terms.py",
        ),
    ),
    "report/turbulence_shaping/matrix_20261121/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "3 变体 x 3 档湍流 (cn2=1e-14/2e-13/2e-12, distance=500 m) 对比择优矩阵汇总; "
            "轴为 2026-11-21 实测 σ 标定后冻结 (~0.06/0.27-0.28/0.88-0.90 rad)"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261121/spgd_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 spgd x cn2=1e-14 (实测 σ≈0.064 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/spgd_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 spgd x cn2=2e-13 (实测 σ≈0.273 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/spgd_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 spgd x cn2=2e-12 (实测 σ≈0.884 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/square_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 square x cn2=1e-14 (实测 σ≈0.063 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/square_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 square x cn2=2e-13 (实测 σ≈0.278 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/square_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 square x cn2=2e-12 (实测 σ≈0.895 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/heuristic_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 heuristic (PSO) x cn2=1e-14 (实测 σ≈0.063 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/heuristic_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 heuristic (PSO) x cn2=2e-13 (实测 σ≈0.285 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121/heuristic_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 60 "
            "--out-root data/turb_matrix --report-root report/turbulence_shaping/matrix_20261121"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="矩阵单元 heuristic (PSO) x cn2=2e-12 (实测 σ≈0.885 rad); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "M3 调参轮矩阵汇总 (n_max=8, 120 epochs, 与 matrix_20261121 基线 n_max=4/60ep 同轴同种子); "
            "实测结论: spgd 三档全面大幅改善且对 σ 平坦, square 高 σ 退化修复, heuristic-PSO 提维反而变差"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261121_m3/spgd_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 spgd x cn2=1e-14 (实测 σ≈0.064 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/spgd_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 spgd x cn2=2e-13 (实测 σ≈0.270 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/spgd_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 spgd x cn2=2e-12 (实测 σ≈0.903 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/square_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 square x cn2=1e-14 (实测 σ≈0.062 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/square_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 square x cn2=2e-13 (实测 σ≈0.284 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/square_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 square x cn2=2e-12 (实测 σ≈0.893 rad, n_max=8/120ep); 高 σ 末值退化已修复 (0.557→0.604); 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/heuristic_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 heuristic (PSO) x cn2=1e-14 (实测 σ≈0.063 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/heuristic_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 heuristic (PSO) x cn2=2e-13 (实测 σ≈0.286 rad, n_max=8/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m3/heuristic_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--out-root data/turb_matrix_m3 --report-root report/turbulence_shaping/matrix_20261121_m3"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M3 单元 heuristic (PSO) x cn2=2e-12 (实测 σ≈0.894 rad, n_max=8/120ep); 提维后 best 反劣于基线 n_max=4 (维度固定种群 12 的采样不足); 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m4/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "M4 拐点扫描汇总 (n_max=11, 120 epochs; spgd x heuristic 两变体 x 三档, "
            "与 M3/matrix_20261121 同轴): spgd n8→n11 增益不超出档间波动 (拐点=8); "
            "heuristic-PSO 提维劣化三档复现 (固定种群 12); 本目录为 f5e7550a 冻结"
            "槽位契约后的**配对重跑版** (data/turb_matrix_m4b, 6 单元与基线/M3 逐"
            "单元同 dist_seed, 首值逐位同可配对; 早期未配对版仅 spgd 1e-14 slot 0 "
            "恰好配对, 其结论方向不变、幅度以本版为准)"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261121_m4/spgd_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M4 单元 spgd x cn2=1e-14 (实测 σ≈0.064 rad, n_max=11/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m4/spgd_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M4 单元 spgd x cn2=2e-13 (实测 σ≈0.270 rad, n_max=11/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m4/spgd_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M4 单元 spgd x cn2=2e-12 (实测 σ≈0.903 rad, n_max=11/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m4/heuristic_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M4 单元 heuristic (PSO) x cn2=1e-14 (实测 σ≈0.063 rad, n_max=11/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m4/heuristic_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M4 单元 heuristic (PSO) x cn2=2e-13 (实测 σ≈0.286 rad, n_max=11/120ep); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m4/heuristic_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--variants spgd,heuristic --out-root data/turb_matrix_m4b "
            "--report-root report/turbulence_shaping/matrix_20261121_m4"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M4 单元 heuristic (PSO) x cn2=2e-12 (实测 σ≈0.894 rad, n_max=11/120ep); 提维劣化第三档复现; 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m5a/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --variants gs_refine "
            "--out-root data/turb_matrix_m5a --report-root report/turbulence_shaping/matrix_20261121_m5a"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "M5a/M5b 汇总: GS-refine freeform (12x12 全频网格) 湍流矩阵三档 (gs_refine 变体自带孪生湍流通道, "
            "测量走 SimPibCCD 真光路); 高 σ 末值不降反为三档最高, Zernike square 的高 σ 退化在 freeform 下消失"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261121_m5a/gs_refine_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --variants gs_refine "
            "--out-root data/turb_matrix_m5a --report-root report/turbulence_shaping/matrix_20261121_m5a"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5a 单元 gs_refine x cn2=1e-14 (实测 σ≈0.063 rad); freeform 相位帧取逐轮下发面板灰度; 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m5a/gs_refine_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --variants gs_refine "
            "--out-root data/turb_matrix_m5a --report-root report/turbulence_shaping/matrix_20261121_m5a"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5a 单元 gs_refine x cn2=2e-13 (实测 σ≈0.278 rad); freeform 相位帧取逐轮下发面板灰度; 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m5a/gs_refine_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --variants gs_refine "
            "--out-root data/turb_matrix_m5a --report-root report/turbulence_shaping/matrix_20261121_m5a"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5a 单元 gs_refine x cn2=2e-12 (实测 σ≈0.911 rad); 综合质量 0.279→0.593 (最优 0.623), 高 σ 无退化; 双 GIF + metrics.png",
    ),
    "report/shape_trials/gui_closed_loop_energy_efficiency_20261125.md": ReportProvenance(
        script=HANDWRITTEN,
        command="python scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py",
        environment="离线分析（复用实验系统已采集 JSON，不打开 SLM / CCD / 激光）",
        note="闭环整形完成态能量利用率 vs SLM 衍射效率 (η=E_blaze/F0 口径, 2026-11-26 修订); 正文含 UI 变更说明与三组历史数据复算; 数字表格由脚本复现, 叙述为手写",
    ),
    "report/shape_trials/s1e_postfix_20261126/s1e_postfix_offline_prep_20261126.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="离线准备 + 判读（复用冻结判读脚本；修复本身由实验系统复采核验）",
        note="s1e 同字节探针 postfix 修复 (driver _write_to_memory 非连续相位守卫) 的离线准备报告; 手写正文, 判读复用 scripts/slm/s1e_same_bytes_probe_judge_20261126.py (冻结), 复采脚本 s1e_same_bytes_probe_capture_postfix_20261126.py",
    ),
    "report/shape_trials/s1e_same_bytes_probe_20261126/s1e_same_bytes_probe_20261126.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件采集（s1e 探针会话）+ 离线 CPU 判读",
        note="s1e 同字节探针主报告; 冻结判读 scripts/slm/s1e_same_bytes_probe_judge_20261126.py, 表格见同目录 _tables.md, 采集脚本 s1e_same_bytes_probe_capture_20261124.py",
    ),
    "report/shape_trials/s1e_same_bytes_probe_20261126/s1e_same_bytes_probe_20261126_tables.md": ReportProvenance(
        script="scripts/slm/s1e_same_bytes_probe_judge_20261126.py",
        command="python scripts/slm/s1e_same_bytes_probe_judge_20261126.py --out-dir report/shape_trials/s1e_same_bytes_probe_20261126",
        environment="离线 CPU 判读（读采集目录, 不打开设备）",
        note="s1e 主判读冻结表格 (--out-dir 缺省 stem)",
    ),
    "report/shape_trials/s1e_same_bytes_probe_20261126/s1e_arm_p_20261126.md": ReportProvenance(
        script=HANDWRITTEN,
        environment="硬件采集（s1e arm-p 会话 run_02）+ 离线 CPU 判读",
        note="s1e arm-p 单元报告; 判读命令 python scripts/slm/s1e_same_bytes_probe_judge_20261126.py --capture-dir data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_02 --out-dir report/shape_trials/s1e_same_bytes_probe_20261126 --stem s1e_arm_p_20261126; 表格见 _tables.md",
    ),
    "report/shape_trials/s1e_same_bytes_probe_20261126/s1e_arm_p_20261126_tables.md": ReportProvenance(
        script="scripts/slm/s1e_same_bytes_probe_judge_20261126.py",
        command="python scripts/slm/s1e_same_bytes_probe_judge_20261126.py --capture-dir data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_02 --out-dir report/shape_trials/s1e_same_bytes_probe_20261126 --stem s1e_arm_p_20261126",
        environment="离线 CPU 判读（读采集目录, 不打开设备）",
        note="s1e arm-p 冻结表格",
    ),
    "report/shape_trials/s1e_same_bytes_probe_20261126/s1e_arm_p_d5_20261126_tables.md": ReportProvenance(
        script="scripts/slm/s1e_arm_p_d5_judge_20261126.py",
        command="python scripts/slm/s1e_arm_p_d5_judge_20261126.py",
        environment="离线 CPU 判读（读两个变体冻结 JSON, 不打开设备）",
        note="s1e arm-p vs d5 对比冻结表格 (缺省读 s1e_same_bytes_probe_20261126.json 与 s1e_arm_p_20261126.json)",
    ),
    "report/turbulence_shaping/matrix_20261126_m5b/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --variants model_in_loop "
            "--epochs 120 --n-max 8 --cam-size 256 --out-root data/turb_matrix_m5b "
            "--report-root report/turbulence_shaping/matrix_20261126_m5b"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "M5b 汇总 (model_in_loop 三档, static 屏): MIL 孪生湍流通道首次入列 "
            "(`_SimBench` disturbance 注入, 孪生 pkl 行补 _c/_img); "
            "quality 首→末 0.665→0.756 (σ≈0.071) / 0.666→0.712 最优 0.744 (σ≈0.240) / "
            "0.667→0.721 最优 0.753 (σ≈1.227), 三档末值全部优于平场; "
            "static 屏协议与 dynamic 各变体不可直接同表排序 (屏协议见口径); "
            "#253 择优终判不变 (dynamic 下 n_max=8 spgd-adamod)"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261126_m5b/model_in_loop_cn21e-14/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --variants model_in_loop "
            "--epochs 120 --n-max 8 --cam-size 256 --out-root data/turb_matrix_m5b "
            "--report-root report/turbulence_shaping/matrix_20261126_m5b"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5b 单元 model_in_loop x cn2=1e-14 (实测 σ≈0.071 rad, static 屏); quality 0.665→0.756; 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261126_m5b/model_in_loop_cn22e-13/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --variants model_in_loop "
            "--epochs 120 --n-max 8 --cam-size 256 --out-root data/turb_matrix_m5b "
            "--report-root report/turbulence_shaping/matrix_20261126_m5b"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5b 单元 model_in_loop x cn2=2e-13 (实测 σ≈0.240 rad, static 屏); quality 0.666→0.712 最优 0.744; 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261126_m5b/model_in_loop_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --variants model_in_loop "
            "--epochs 120 --n-max 8 --cam-size 256 --out-root data/turb_matrix_m5b "
            "--report-root report/turbulence_shaping/matrix_20261126_m5b"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5b 单元 model_in_loop x cn2=2e-12 (实测 σ≈1.227 rad, static 屏); quality 0.667→0.721 最优 0.753; 双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m5c_n8p24/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--pop-size 24 --variants heuristic --cn2-grid 2e-12 "
            "--out-root data/turb_matrix_m5c_n8p24b "
            "--report-root report/turbulence_shaping/matrix_20261121_m5c_n8p24"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "M5c 汇总 (heuristic n8 pop24 @2e-12): 与基线/M3/m4b 同 slot 配对 (首值逐位同); "
            "best -0.632 vs 8×12 的 -0.621 (2× 求值, 增益噪声量级) → n8 种群翻倍无收益; "
            "配对口径结论表与 n4/n8/n11 × pop12/24/36 六行对照见正文 M5c 节; "
            "#253 择优终判不变 (n_max=8 spgd-adamod)"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261121_m5c_n8p24/heuristic_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 8 "
            "--pop-size 24 --variants heuristic --cn2-grid 2e-12 "
            "--out-root data/turb_matrix_m5c_n8p24b "
            "--report-root report/turbulence_shaping/matrix_20261121_m5c_n8p24"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5c 单元 heuristic(PSO) x cn2=2e-12 (实测 σ≈0.898 rad, n_max=8/pop=24/120ep, 2908 求值; 首-1.3575/末-0.9161/最优-0.6325); 相位/光斑双 GIF + metrics.png",
    ),
    "report/turbulence_shaping/matrix_20261121_m5c_n11p36/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--pop-size 36 --variants heuristic --cn2-grid 2e-12 "
            "--out-root data/turb_matrix_m5c_n11p36 "
            "--report-root report/turbulence_shaping/matrix_20261121_m5c_n11p36"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note=(
            "M5c 汇总 (heuristic n11 pop36 @2e-12): 同 slot 配对; best -0.702 vs 11×12 的 "
            "-0.840 (3× 求值收复 0.138, 仍劣于 8×12 且远差于 4×12) → 种群重配不能翻案维度排序, "
            "升维劣化主因是 PSO 搜索结构而非种群不足; 六行对照表见正文 M5c 节; "
            "#253 择优终判不变 (n_max=8 spgd-adamod)"
        ),
        extra_scripts=("scripts/slm/generate_turbulence_shaping_report.py",),
    ),
    "report/turbulence_shaping/matrix_20261121_m5c_n11p36/heuristic_cn22e-12/report.md": ReportProvenance(
        script="scripts/slm/run_turb_shaping_matrix.py",
        command=(
            "python scripts/slm/run_turb_shaping_matrix.py --epochs 120 --n-max 11 "
            "--pop-size 36 --variants heuristic --cn2-grid 2e-12 "
            "--out-root data/turb_matrix_m5c_n11p36 "
            "--report-root report/turbulence_shaping/matrix_20261121_m5c_n11p36"
        ),
        environment="离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）",
        note="M5c 单元 heuristic(PSO) x cn2=2e-12 (实测 σ≈0.897 rad, n_max=11/pop=36/120ep, 4360 求值; 首-1.3575/末-0.8253/最优-0.7022); 相位/光斑双 GIF + metrics.png",
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
        if not (Path(__file__).resolve().parents[2] / report).is_file():
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
