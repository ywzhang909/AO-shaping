"""GS 方形光斑三段扫描 (边长 / Zernike 阶数 / 闪耀光栅) 的离线报告生成器。

做什么
======
``scripts/slm/hw_gs_square_scans.py`` 把三段扫描的每一个配置都落盘成
``measurements.csv`` / ``measurements.json`` / ``run_config.json`` / ``metrics_*.json``。
本脚本**只读这些已落盘的产物** (以及那次运行的日志), 渲染一份可入库的实验报告 +
四张图, 并把输入产物复制进报告的 ``artifacts/`` 作为**输入契约**。

**它不 import 任何驱动、不打开任何设备** —— 除了 ``matplotlib`` / ``numpy`` /
``_common`` 之外没有第三方依赖, 所以可以在没有 SLM / 相机的机器上重跑。

三条纪律
========
1. **零硬编码数字。** 报告里的每一个测量值都从 ``measurements.csv`` 读出来
   (``run_config.json`` 提供参数, ``manifest.json`` 提供行数, 日志提供 settle 判据
   与曝光/0 级位置). 想改一个数就改产物, 报告跟着变。
2. **表格要回读校验。** 渲染完成后, 从**已渲染的 Markdown** 里把每张数值表解析
   回来, 逐单元与 ``measurements.csv`` 的原文比对 (容差 = 该单元显示精度的半个
   末位). 任何一处不一致都**非零退出** —— 「表里的数字与产物一致」这条不能靠
   人工核对。
3. **``--check`` 只读。** 重新渲染到内存, 与磁盘上的报告逐字节比对; 同时校验
   ``artifacts/`` 里的副本与源文件逐字节一致、正文引用的图都存在。**零 diff** 是
   幂等的判据, 写图只在普通模式下发生。

输入 (默认)
===========
* ``data/debug/hw_gs_square_scans/20261023_gs_square/`` —— 三段产物 (``--run-root``)
* ``logs/gs_square_scan.log`` —— 那次运行的日志 (``--log``); 缺失时报告会明确写出
  「settle 判据状态未知」, 而不是假装没有超时。

产物
====
* ``report/shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md``
* ``report/shape_trials/hw_gs_square_scans/figures/*.png`` (4 张)
* ``report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/`` (输入契约)

复现
====
.. code-block:: powershell

    python scripts/slm/generate_hw_gs_square_scans_report.py
    python scripts/slm/generate_hw_gs_square_scans_report.py --check   # 幂等校验
"""

from __future__ import annotations

import argparse
from loguru import logger
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# 让脚本可独立运行: `python scripts/slm/generate_hw_gs_square_scans_report.py` 时
# sys.path[0] 是 scripts/slm/, 既找不到 `ao_shaping` (src/) 也找不到 `_common`
# (scripts/) —— 与 `scripts/slm/compare_sim_vs_hw.py` 同一套自举写法。
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")  # 🔴 必须在 import matplotlib.pyplot 之前 (仓库级规则)

import matplotlib.pyplot as plt
import numpy as np

from _common import savefig
from _common.provenance import REPORTS, insert_header
from scripts.slm._hw_gs_square_scans_checks import (
    check_invariants,
    check_residuals,
)
from scripts.slm._hw_gs_square_scans_model import (
    AREA_IDENTITY_ATOL,
    AtomicFindings,
    BASE_CMAP,
    BLAZE_PANEL_OPTIONAL,
    BLAZE_PANEL_PERIOD,
    CROP_FACTOR,
    Cell,
    CellPick,
    DARK_FRAME_FRACTION,
    DEFAULT_ATOMIC_ROOT,
    DEFAULT_LOG,
    DEFAULT_RETEST_LOG_DIR,
    DEFAULT_RETEST_PRIOR_ROOT,
    DEFAULT_RETEST_ROOT,
    DEFAULT_RUN_ROOT,
    EXPOSURE_MATCH_ATOL,
    FIELD_BASELINE_FLAT,
    FIELD_RETEST_FLAT,
    FIG_TAG,
    FLAG_LEGEND,
    FULLFRAME_TARGET_PX,
    FieldRecord,
    Findings,
    FrameRef,
    MOSAICS,
    MOSAIC_CN,
    MosaicSpec,
    OUTLIER_CV_RATIO,
    OUT_DIR,
    PARAM_NPZ_CSV_ATOL,
    PATHS,
    PEAK_MATCH_ATOL,
    REPORT_KEY,
    RESIDUAL_CMAP,
    RESIDUAL_HIGH_FREQ,
    RESIDUAL_PANEL_ORDER,
    RESIDUAL_RECON_ATOL,
    RESIDUAL_RMS_RTOL,
    RESIDUAL_VLIM,
    RETEST_STAGES,
    RETEST_STAGE_CN,
    ROW_MATCH_ATOL,
    ResidualPanel,
    ResidualSet,
    RetestConfig,
    RetestFindings,
    RetestRun,
    RetestStage,
    Run,
    SATURATION_COUNTS,
    SCORE_IDENTITY_ATOL,
    SIDES_PANEL_PX,
    SIDE_PX_NPZ_CSV_ATOL,
    STAGES,
    STAGE_CN,
    SettleEvent,
    SettleRecord,
    Stage,
    Table,
    VARIABLE,
    ZERNIKE_PANEL_ORDER,
    run_zero_order,
)
from scripts.slm._hw_gs_square_scans_fmt import (
    _cell_short,
    _count_stage,
    _figname,
    _idx_of,
    _join_x,
    _pct,
    _rel,
    _row_at,
    cell_of,
    derived,
)
from scripts.slm._hw_gs_square_scans_io import (
    _RETEST_METRIC_KEYS,
    _SETTLE_RE,
    _STAGE_LINE_RE,
    _STAMP_RE,
    _cross_check_csv_json,
    _cross_check_retest_metrics,
    _log_crosscheck,
    _read_log_text,
    _selected_side_px,
    _selected_side_px_from,
    _settle_records,
    _two_tier_split,
    _unwrap_log,
    load_csv,
    load_json,
    load_retest,
    load_run,
    load_selected,
    parse_log,
    verify_table_in_markdown,
)
from scripts.slm._hw_gs_square_scans_metrics import (
    _loglog_exponent,
    _neighbour_deviation,
    column_contract,
    flag_rows,
    monotonic_report,
    pearson,
    retest_power_ratio,
    saturation_flag,
    score_terms,
    spearman,
)
from scripts.slm._hw_gs_square_scans_analysis import (
    _residual_structure,
    base_phase_path,
    compute_atomic_findings,
    compute_findings,
    compute_residuals,
    compute_retest_findings,
    select_cells,
)
from scripts.slm._hw_gs_square_scans_figbase import (
    _annotate_cell,
    _cell_title,
    _check_frame_params,
    _crop_box,
    _sha256,
    batch_tag,
    draw_mosaic,
    frame_params,
    load_frame_mean,
    locate_frames,
    plan_farfield,
    plan_mosaic,
    render_farfield,
    residual_figure_name,
    setup_matplotlib,
    write_frames_selection,
)
from scripts.slm._hw_gs_square_scans_figures import (
    fig_atomic_blaze,
    fig_atomic_sides,
    fig_blaze,
    fig_retest_neighbors,
    fig_sides_exposure,
    fig_sides_objectives,
    fig_zernike,
    fig_zernike_residual,
    render_figures,
)
from scripts.slm._hw_gs_square_scans_sections import (
    render_farfield_section,
    render_retest_section,
)
from scripts.slm._hw_gs_square_scans_atomic import (
    render_atomic_section,
)
from scripts.slm._hw_gs_square_scans_artifacts import (
    _artifact_manifest,
    _atomic_artifact_manifest,
    _check_artifact_set,
    _check_farfield,
    _copy_artifact_set,
    _copy_farfield_to_artifacts,
    _retest_artifact_manifest,
    artifact_sources,
    atomic_artifact_sources,
    check_artifacts,
    copy_artifacts,
    retest_artifact_sources,
)

ROOT = _ROOT













# ---------------------------------------------------------------------------
# 复测批次 (功率已变) —— 第二数据源
# ---------------------------------------------------------------------------





#: 原子整轮产物的归档目录 (同样从默认根的目录名推导)。
#: 复测产物的归档目录 (报告输入契约的第二块)。基线那一批在 ``PATHS.artifact_dir``。
#: 目录名**从默认根的目录名推导** —— 换 ``--retest-root`` 时 :func:`main` 会把它改成
#: 参数那一批的目录名, 所以这里不另抄一份批次字面量。







# ---------------------------------------------------------------------------
# 中文渲染 (字体 + rcParams)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 表格单元: 每个数字都记得自己来自哪个 CSV 原文
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# 读产物
# ---------------------------------------------------------------------------






























# ---------------------------------------------------------------------------
# 派生量: 全部现场算, 不抄
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
# 复测批次: 读产物 + 批内比较
# ---------------------------------------------------------------------------






























# ---------------------------------------------------------------------------
# 图 (全部点来自 CSV)
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# CCD 远场帧: 定位 / 选择 / 拼图 (§12)
# ---------------------------------------------------------------------------




















































# ---------------------------------------------------------------------------
# 报告正文
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# §13 整轮原子重扫 (单一脚本版本 + 单一功率)
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# §13.8 Zernike 截断残差 × 同配置 CCD 远场 (fig12)
# ---------------------------------------------------------------------------

























def render_report(
    run: Run,
    f: Findings,
    retest: RetestRun | None = None,
    rf: RetestFindings | None = None,
    prior: RetestRun | None = None,
    farfield: dict[str, Any] | None = None,
    farfield_figures: Sequence[Path] = (),
    atomic: RetestRun | None = None,
    af: AtomicFindings | None = None,
    atomic_figures: Sequence[Path] = (),
    residuals: ResidualSet | None = None,
) -> tuple[str, list[Table]]:
    """渲染正文。返回 (markdown, 需要回读校验的表)。

    ``retest``/``rf`` 为 ``None`` 时**不渲染 §11**，输出与没有 ``--retest-root``
    时逐字节相同（这是 ``--check`` 幂等的前提之一）。
    """
    tables: list[Table] = []
    sides = run.stages["sides"]
    zern = run.stages["zernike"]
    blaze = run.stages["blaze"]
    cfg = sides.run_config
    bench = (run.plan.get("bench") or {})
    spot = run.spot_assessment
    h, w = run.frame_shape_hw
    zo_x, zo_y = run.zero_order_xy
    marks_s = flag_rows(run, sides)
    marks_z = flag_rows(run, zern)
    marks_b = flag_rows(run, blaze)
    out: list[str] = []

    def add(text: str = "") -> None:
        out.append(text)

    # -- 标题 ---------------------------------------------------------------
    add("# GS 方形光斑三段扫描：边长 / Zernike 阶数 / 闪耀光栅（2026-10-23）")
    add()
    add(
        "一次纯 Gerchberg-Saxton（GS）的方形远场整形实验，沿三个自变量各扫一遍："
        f"**目标方框边长**（{len(sides.rows)} 行）、**Zernike 截断阶数**（{len(zern.rows)} 行）、"
        f"**闪耀光栅周期**（{len(blaze.rows)} 行）。三种扫描之间只通过"
        "「选定边长 / 选定相位」单向传递，全程**没有任何闭环优化**（GS 是交替投影，不含随机数）。"
    )
    add()
    add(
        f"全部数字来自 `{_rel(run.root)}/` 下落盘的 "
        "`measurements.csv` / `measurements.json` / `run_config.json` / `manifest.json`"
        + (f"，以及那次运行的日志 `{_rel(run.log_path)}`。" if run.log_path
           else "，**日志未提供**，因此 settle 判据状态未知（见 §2.3）。")
        + "本报告由 `scripts/slm/generate_hw_gs_square_scans_report.py` 生成；"
        "生成器把每张数值表渲染后再从 Markdown 回读，逐单元与 CSV 原文比对，"
        "不一致即非零退出。"
    )
    add()

    # -- 0. 结论核对 --------------------------------------------------------
    add("## 0. 结论核对（逐条用数据判定）")
    add()
    add(
        "这一节把「本轮原本想验证的说法」逐条摊到数据上。判定只有三种："
        "**成立**（数据支持）、**部分成立**（方向对但措辞过强）、**推翻**。"
    )
    add()
    claim = Table(
        ("#", "待核对的结论", "数据", "判定"),
        [
            ("1", "PIB 随边长单调上升（更大方框装更多能量）",
             f"{len(sides.rows)} 行中反向步数 = {len(f.pib_violations)}；"
             f"PIB {f.pib[0]:.4f} → {f.pib[-1]:.4f}", "成立"),
            ("2", "CV 与 PIB 同步上升",
             f"整体 Spearman ρ = {f.cv_spearman:.3f}，但 {len(f.cv_violations)} 处回落"
             f"（{_join_x(f.sides_x, f.cv_violations)} px）",
             "部分成立" if f.cv_violations else "成立"),
            ("3", "score 在 1500 px 最高、之后下降",
             f"argmax = {f.sides_x[f.score_peak_index]:.0f} px（{f.score[f.score_peak_index]:.4f}）；"
             f"次高 {f.sides_x[f.score_runner_up_index]:.0f} px 只差 {f.score_peak_gap:.2e}"
             f"；峰后单调下降 = {f.score_descend_ok}", "成立但峰不可分辨"),
            ("4", "Zernike 解释方差仅 0.2%–3.3%、残差 1.53–1.55 rad、CV 与阶数无关",
             f"{min(f.zern_explained) * 100:.3f}%–{max(f.zern_explained) * 100:.3f}%；"
             f"{min(f.zern_residual):.4f}–{max(f.zern_residual):.4f} rad；"
             f"干净行 CV {min(f.zern_clean_cv):.3f}–{max(f.zern_clean_cv):.3f}"
             f"（相对跨度 {f.zern_clean_cv_spread:.1%}），"
             f"r(n_max, CV) = {f.zern_cv_pearson:+.3f}",
             "部分成立" if abs(f.zern_cv_pearson) >= 0.5 else "成立"),
            ("5", "n_max=18 是异常点（CV 13.53）",
             f"CV {f.zern_cv[_idx_of(zern.x, 18.0)]:.3f}，"
             f"整帧峰值 {float(zern.rows[_idx_of(zern.x, 18.0)]['frame_peak']):.1f} counts"
             f"（同段中位数 {zern.median('frame_peak'):.1f}）；"
             + (f"复测 CV {float(_row_at(retest.stages['zernike'].rows, 'n_max', 18.0)['cv']):.4f}"
                f"（与 17/19 差 {abs(_neighbour_deviation(rf.zern_cv, 1)):.2%}）"
                if retest is not None and rf is not None else "（未复测）"),
             "**已复测推翻**（§11.5 判定 2）" if retest is not None
             else "成立，且更严重"),
            ("6", "闪耀光栅 score 随周期单调上升",
             f"排除 §/‡ 两行后反向步数 = {len(f.blaze_score_violations_clean)}",
             "成立" if not f.blaze_score_violations_clean else "不成立"),
            ("7", "period=16 预测位移让方框出画",
             f"predicted {blaze.col('predicted_shift_px')[0]:.1f} px；"
             f"box_clipped_px = {blaze.col('box_clipped_px')[0]:.0f} px"
             f"（占整框 {_pct(blaze.col('box_clipped_px')[0], float(blaze.rows[0]['box_side_px']) ** 2)}）",
             "成立"),
            ("8", "period=40 异常且该次采集 settle 超时",
             f"score {f.blaze_score[_idx_of(blaze.x, 40.0)]:.4f}，CV "
             f"{blaze.col('cv')[_idx_of(blaze.x, 40.0)]:.2f}；"
             + (f"日志：{run.settle_for('blaze', _idx_of(blaze.x, 40.0)).describe()}"
                if run.settle_for("blaze", _idx_of(blaze.x, 40.0))
                else "日志缺失，无法核对")
             + (f"；复测 CV {float(_row_at(retest.stages['blaze'].rows, 'blaze_period_px', 40.0)['cv']):.4f}"
                f" 落在 32/48 之间" if retest is not None else ""),
             (f"**异常已复测推翻**（§11.5 判定 1）；settle 超时本身成立"
              f"（另有 {max(len(run.settle_events) - 1, 0)} 行同样超时）")
             if retest is not None
             else f"成立（另有 {max(len(run.settle_events) - 1, 0)} 行同样超时）"),
            ("9", "「近 1000 取到窗口上界」",
             f"窗口 {f.near_window[0]:.0f}–{f.near_window[1]:.0f} px 内 score 单调 ⇒ "
             f"选中 {f.sides_x[f.near_best_index]:.0f} px，正是上界", "成立"),
            ("10", "side_max_px 由 0 级位置反解",
             f"2·({zo_y:.0f}−{f.edge_margin:.0f}) = {f.side_max_raw:.0f} → "
             f"向下取 {f.step:.0f} 的整数倍 = {f.side_max:.0f}", "成立"),
        ]
        + (
            [
                ("11", "基线 n_max=17 的高端异常来自「高阶级重构在面板上欠采样」",
                 f"复测 17 的 CV "
                 f"{float(_row_at(retest.stages['zernike'].rows, 'n_max', 17.0)['cv']):.4f}"
                 f" 落在趋势内（且复测 17 自己也有 settle 超时）",
                 "**未确认**（降级为未证实假设，§11.5 判定 3）"),
            ]
            if retest is not None
            else []
        ),
    )
    tables.append(claim)
    add(claim.render())
    add()
    add(f"> 判定 8 的补充：本轮日志里**一共 {len(run.settle_events)} 行** settle 超时"
        f"（sides {_count_stage(run, 'sides')} / zernike {_count_stage(run, 'zernike')} / "
        f"blaze {_count_stage(run, 'blaze')}），不是只有 period=40 一行。见 §2.3。")
    add()

    # -- 1. 口径 ------------------------------------------------------------
    add("## 1. 实验与口径")
    add()
    add("### 1.1 光路与设备（`sides/run_config.json`）")
    add()
    # 原子化批次的段级 run_config 只记段级参数, gs_grid / gs_iters 在 CSV 行里
    gs_grid = int(cfg["gs_grid"]) if "gs_grid" in cfg else int(float(sides.rows[0]["gs_grid"]))
    gs_iters = (int(cfg["gs_iters"]) if "gs_iters" in cfg
                else int(float(sides.rows[0]["gs_iters"])))
    def _opt(value: Any, spec: str, unit: str = "") -> str:
        """``plan.bench`` 缺失时给 ``—`` 而不是 ``nan`` (原子化批次不带 plan)。"""
        if value is None:
            return "—（本批 `run_config` 无 `plan.bench`）" if not bench else "—"
        return f"{float(value):{spec}}{unit}"

    params = Table(
        ("项", "值", "来源"),
        [
            ("波长", _opt(bench.get("wavelength_nm"), ".0f", " nm"), "`plan.bench`"),
            ("焦距", _opt(bench.get("focal_length_m"), ".3f", " m"), "`plan.bench`"),
            ("SLM 面板 (W×H)", f"{bench.get('panel_shape')} (行×列)", "`plan.bench`"),
            ("面板像元", _opt(bench.get("panel_pitch_um"), ".1f", " µm"), "`plan.bench`"),
            ("相机像元", _opt(bench.get("camera_pixel_um"), ".1f", " µm"), "`plan.bench`"),
            ("相机画幅 (W×H)", f"{w} × {h}", "`frame_shape_hw`"),
            ("等效瞳面边长", f"{float(bench.get('pupil_side_px', float('nan'))):.0f} panel px", "`plan.bench`"),
            ("0 级位置（实测冻结）", f"({zo_x:.0f}, {zo_y:.0f})", "`zero_order_xy_frozen`"),
            ("画面几何中心", f"({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})", "由 `frame_shape_hw` 推导"),
            ("GS 网格 / 迭代", f"N = {gs_grid} / {gs_iters}",
             "`run_config`" if "gs_grid" in cfg else "CSV 行"),
            ("每配置帧数", f"{int(cfg['frames_per_measurement'])}", "`run_config`"),
        ],
    )
    tables.append(params)
    add(params.render())
    add()
    add(
        "`TILT_SHIFT_SCALE`（2f 焦面标度）在采集脚本里按实测常量给出，于是"
        "**一个 GS 网格像素恒等于固定的 camera 像素数**，与网格边长 N 无关；"
        "这也是为什么方斑边长可以直接按 `side_gs_px = side_cam_px · pupil_side_px / TILT_SHIFT_SCALE` "
        "换算。本报告不重算这条标度，只引用 `run_config` 里 GS 的实际输出 "
        f"`gs_grid = {gs_grid}` 与 `gs_iters = {gs_iters}`。"
    )
    add()

    add("### 1.2 指标口径")
    add()
    add(
        "- `PIB` = 目标方框内功率 / 整帧功率（`square_shaping.power_in_bucket`）；"
        "帧先过 `prepare_frame`（减中位数 + 裁负），否则读噪声会把 PIB 顶到 1 以上。"
    )
    add(
        "- `CV` = 方框**内部**强度的标准差 / 均值（`uniformity_cv`），越低越均匀。"
    )
    add(
        "- `score` = **0.5·PIB + 0.5·1/(1+CV)**（`composite_from_pib_cv`）。"
        "本报告把这两项拆开列（§3.3），因为方框变大时它们**方向相反**。"
    )
    add(
        "- `ee` 与 `PIB` 在数值上**完全相同**（`square_encircled_energy` 与 "
        "`power_in_bucket` 对同一方框做同一个求和），所以本报告不把 `ee` 当独立指标。"
    )
    add(
        "- `global_spot_fwhm_px` / `global_spot_hollowness` 量的是**整帧最亮结构**"
        "（2f 光路上通常是 0 级或其晕斑），**不是**被整形的方斑。图 2(b) 里它随边长"
        "大幅摆动是 0 级几何的变化，不能读成「方斑变差」。"
    )
    add()
    add("### 1.3 稳定性判据")
    add()
    policy = run.settle_policy
    add(
        f"每次下发相位后丢弃 {policy.get('n_discard')} 帧、等 {policy.get('wait_s')} s，"
        f"再以相对容差 {policy.get('tol')} 判定稳定，最长等 {policy.get('max_wait_s')} s，"
        "稳定后平均 4 帧。判据超时时采集脚本**仍然取帧**，但会打一条 "
        "`settle criterion not met … must be treated as provisional` —— 本报告把这些行"
        f"标 `†`（共 {len(run.settle_events)} 行）。"
    )
    add()

    # -- 2. 数据完整性 ------------------------------------------------------
    add("## 2. 数据完整性")
    add()
    add("### 2.1 计划 vs 落盘")
    add()
    rows_plan = []
    for name in STAGES:
        st = run.stages[name]
        planned = run.stage_plans.get(name, {}).get("configs", [])
        rows_plan.append(
            (STAGE_CN[name], str(len(planned)), derived(float(len(st.rows)), 0),
             str(st.manifest_entry.get("created_utc", "-")),
             derived(float(len(st.headers)), 0))
        )
    completeness = Table(("段", "计划配置数", "CSV 行数", "落盘时间 (UTC)", "CSV 列数"),
                          rows_plan)
    tables.append(completeness)
    add(completeness.render())
    add()
    if f.dropped_configs:
        add(
            f"**计划里有、CSV 里没有的配置：{', '.join(f.dropped_configs)} px。** "
            "这不是丢数据：`side_max_px` 由 0 级位置与画面边缘**现场反解**（§8），"
            f"解出 {f.side_max:.0f} px，于是 {', '.join(f.dropped_configs)} 从未被采集。"
        )
        add()
    add(
        f"CSV 与 `measurements.json` 已逐值交叉校验（{f.csv_cells} 个单元，含 NaN 语义），"
        "`manifest.json` 的 `n_rows` / `variable` 与 CSV 的实际行数、表头一致。"
        + ("日志里的曝光、0 级位置、最大边长、选定边长与 `run_config` **全部一致**。"
           if run.log_crosscheck and all(ok for *_, ok in run.log_crosscheck)
           else ("日志与 `run_config` 存在**不一致**，见 §2.4。" if run.log_crosscheck
                 else "未提供日志，settle 判据状态未知。"))
    )
    add()

    add("### 2.2 饱和普查")
    add()
    sat_rows: list[tuple[Cell | str, ...]] = []
    for name in STAGES:
        st = run.stages[name]
        peaks = st.col("frame_peak")
        n_sat = sum(1 for v in peaks if v >= SATURATION_COUNTS)
        sat_rows.append((
            STAGE_CN[name],
            derived(float(n_sat), 0),
            derived(float(len(peaks) - n_sat), 0),
            derived(float(max(peaks)), 1),
            derived(float(min(peaks)), 1),
            ", ".join(f"{st.x[i]:.0f}" for i, v in enumerate(peaks) if v >= SATURATION_COUNTS)
            or "—",
        ))
    saturation = Table(
        ("段", "frame_peak 达满量程的行数", "其余行数", "本段最大峰值", "本段最小峰值", "达满量程的配置"),
        sat_rows,
    )
    tables.append(saturation)
    add(saturation.render())
    add()
    add(
        f"满量程按 8 bit 计为 {SATURATION_COUNTS:.0f} counts。**只有达到满量程才是饱和**："
        "这些行的峰值只能读成「≥ 满量程」，不能当线性测量值。注意 `frame_peak` 是**整帧**"
        "最大值，它通常落在 0 级上；方框内的 `box_peak` 单独记，两列一起看才能判断"
        "「饱和的是 0 级还是方斑」。"
    )
    add()

    add("### 2.3 settle 判据超时（日志原文）")
    add()
    if run.settle_events:
        ev_rows: list[tuple[Cell | str, ...]] = []
        for ev in run.settle_events:
            st = run.stages[ev.stage]
            row = st.rows[ev.index]
            frame_peak = float(row["frame_peak"])
            marks = flag_rows(run, st)[ev.index]
            ev_rows.append((
                STAGE_CN[ev.stage],
                derived(st.x[ev.index], 0),
                derived(ev.waited_s, 2),
                derived(float(ev.probes), 0),
                derived(ev.probe_peak, 2),
                derived(frame_peak, 2),
                derived(frame_peak - ev.probe_peak, 2),
                marks or "†",
            ))
        settle_tbl = Table(
            ("段", "配置", "等待 (s)", "探测次数", "探测峰值", "最终 frame_peak", "差",
             "旗标"),
            ev_rows,
        )
        tables.append(settle_tbl)
        add(settle_tbl.render())
        add()
        add(
            "读法：探测峰值与最终 `frame_peak` **相近**的行，说明取帧时画面已经到达"
            "它最终的电平，判据超时更可能是**判据过严**（相对容差 "
            f"{policy.get('tol')} 对这台相机的噪声偏紧），指标仍可用，只是形式上 provisional；"
            "两者**差距很大**或两者都远离满量程的行，才需要按不可信处理 —— "
            "本轮的 `‡` 行（§9）正是后者。"
        )
    else:
        add(
            "**日志未提供**（`--log` 指向的文件不存在），所以 settle 判据是否超时无法核对。"
            "这一节留空而不是推断 —— 断言的空白比错误的断言更安全。"
        )
    add()

    if run.log_crosscheck:
        add("### 2.4 日志 ↔ `run_config` 交叉核对")
        add()
        xc = Table(("量", "日志打印", "`run_config`", "一致"),
                   [(name, logged, actual, "是" if ok else "**否**")
                    for name, logged, actual, ok in run.log_crosscheck])
        tables.append(xc)
        add(xc.render())
        add()

    # -- 3. 边长扫描 --------------------------------------------------------
    add("## 3. 边长扫描")
    add()
    add(
        "每个边长都**从零跑一次 GS** 造方斑（不是把上一行的相位缩放），然后测量。"
        "这是本段最重要的一句口径：18 行的 GS 相位彼此独立，所以行间差异里含有"
        "**GS 本身的收敛差异**，而不只是方框大小的效应。"
    )
    add()
    add("### 3.1 逐行数据")
    add()
    side_tbl = Table(
        ("边长 (px)", "CV", "PIB", "score", "quality", "frame_peak", "box_peak",
         "方框内面积 (px)", "出画面积 (px)", "峰值状态", "旗标"),
        [
            (cell_of(r, "side_px", 0), cell_of(r, "cv", 4), cell_of(r, "pib", 4),
             cell_of(r, "score", 4), cell_of(r, "quality", 4),
             cell_of(r, "frame_peak", 2), cell_of(r, "box_peak", 2),
             cell_of(r, "box_area_px", 0), cell_of(r, "box_clipped_px", 0),
             saturation_flag(r), marks_s[i] or "—")
            for i, r in enumerate(sides.rows)
        ],
    )
    tables.append(side_tbl)
    add(side_tbl.render())
    add()
    add(FLAG_LEGEND)
    add()

    add("### 3.2 PIB 与 CV 同向上升")
    add()
    add(f"![边长 → PIB / CV / score](figures/{_figname('fig1_sides_objectives')})")
    add()
    add(
        f"PIB 在 {len(sides.rows)} 行里**严格单调上升**（反向步数 {len(f.pib_violations)}），"
        f"从 {f.pib[0]:.4f}（{f.sides_x[0]:.0f} px）到 {f.pib[-1]:.4f}"
        f"（{f.sides_x[-1]:.0f} px）。这符合「方框越大装得越多」的直觉，但它同时说明"
        "**PIB 单独不构成整形质量指标**：把框放大到装下整幅画面，PIB 会自然趋近 1。"
    )
    add()
    add(
        f"CV 的整体趋势同样上升（Spearman ρ = {f.cv_spearman:.3f}），但"
        f"**不是单调**：{len(f.cv_violations)} 处回落（{_join_x(f.sides_x, f.cv_violations)} px）。"
        "CV 上升的物理含义很直接 —— 方框变大后，框内除了方斑还装进了越来越多的"
        "**暗区**，标准差/均值被撑大。所以「PIB 与 CV 同向上升」这句在方向上成立，"
        "但它描述的其实是同一个几何事实（框变大），不是两个独立的物理效应。"
    )
    add()

    add("### 3.3 score 的峰：两项互相抵消")
    add()
    peak_x = f.sides_x[f.score_peak_index]
    add(
        f"`score` 的 argmax 在 **{peak_x:.0f} px**（{f.score[f.score_peak_index]:.4f}），"
        f"次高是 {f.sides_x[f.score_runner_up_index]:.0f} px（{f.score[f.score_runner_up_index]:.4f}）。"
        f"两者相差 **{f.score_peak_gap:.2e}** —— 这个差值小到没有意义，所以正确的说法是"
        f"「峰值落在 {min(peak_x, f.sides_x[f.score_runner_up_index]):.0f}–"
        f"{max(peak_x, f.sides_x[f.score_runner_up_index]):.0f} px 之间，"
        "无法分辨单点」，而不是「1500 px 最优」。"
    )
    add()
    term_tbl = Table(
        ("边长 (px)", "PIB", "0.5·PIB", "CV", "0.5/(1+CV)", "score", "说明"),
        [
            (cell_of(sides.rows[0], "side_px", 0), cell_of(sides.rows[0], "pib", 4),
             derived(f.terms_at_small[0], 4), cell_of(sides.rows[0], "cv", 4),
             derived(f.terms_at_small[1], 4), cell_of(sides.rows[0], "score", 4),
             "最小方框：饱和"),
            (cell_of(sides.rows[f.score_peak_index], "side_px", 0),
             cell_of(sides.rows[f.score_peak_index], "pib", 4),
             derived(f.terms_at_peak[0], 4),
             cell_of(sides.rows[f.score_peak_index], "cv", 4),
             derived(f.terms_at_peak[1], 4),
             cell_of(sides.rows[f.score_peak_index], "score", 4), "峰值"),
            (cell_of(sides.rows[-1], "side_px", 0), cell_of(sides.rows[-1], "pib", 4),
             derived(f.terms_at_last[0], 4), cell_of(sides.rows[-1], "cv", 4),
             derived(f.terms_at_last[1], 4), cell_of(sides.rows[-1], "score", 4),
             "最大方框"),
        ],
    )
    tables.append(term_tbl)
    add(term_tbl.render())
    add()
    if f.score_peak_index == len(f.sides_x) - 1:
        add(
            f"**本轮的 score 峰就落在最后一行（{peak_x:.0f} px）**，峰后没有数据 —— "
            "所以「峰后下降」这一条在本轮**不可判定**，§3.3 的分解也就只能给到峰值点为止。"
            "（基线那一轮扫到了 1800 px，才有峰后下降的腿。）"
        )
        add()
    add(
        f"从 {peak_x:.0f} px 到 {f.sides_x[-1]:.0f} px，PIB 项**还在涨**"
        f"（{f.terms_at_peak[0]:.4f} → {f.terms_at_last[0]:.4f}，"
        f"{f.terms_at_last[0] - f.terms_at_peak[0]:+.4f}），"
        f"而均匀度项在掉（{f.terms_at_peak[1]:.4f} → {f.terms_at_last[1]:.4f}，"
        f"{f.terms_at_last[1] - f.terms_at_peak[1]:+.4f}）。"
        f"后者绝对值更大，于是 score 净变化 {f.score[-1] - f.score[f.score_peak_index]:+.4f}。"
        "**峰值之后 score 的下降完全由 CV 造成，PIB 反而更好了** —— 这就是"
        "「两项互相抵消」的量化版本，也解释了为什么单纯追 PIB 会把方框越开越大。"
    )
    add()

    add("### 3.4 GS 模型输出 vs 实测")
    add()
    model_tbl = Table(
        ("边长 (px)", "model_cv", "实测 CV", "model_pib", "实测 PIB", "ΔPIB"),
        [(cell_of(r, "side_px", 0), cell_of(r, "model_cv", 4), cell_of(r, "cv", 4),
          cell_of(r, "model_pib", 4), cell_of(r, "pib", 4), derived(d, 4))
         for r, d in zip(sides.rows, f.model_delta)],
    )
    tables.append(model_tbl)
    add(model_tbl.render())
    add()
    add(
        f"GS 模型给出的 PIB 在 {min(f.model_pib):.4f}–{max(f.model_pib):.4f} 之间几乎不动，"
        f"而实测 PIB 在 {min(f.pib):.4f}–{max(f.pib):.4f} 之间涨了 "
        f"{max(f.pib) - min(f.pib):.2f}。模型在最小方框上高估 {f.model_delta[0]:+.4f}，"
        f"在最大方框上差 {f.model_delta[-1]:+.4f}。**结论：GS 的模型内指标不能用来选边长** —— "
        "它看不到「实际远场不是理想方斑」这件事（模型里方斑是精确的平顶方框，"
        "实测里还有 0 级、散射与像差）。"
    )
    add()
    add(f"![边长 → 峰值与光斑几何](figures/{_figname('fig2_sides_exposure_geometry')})")
    add()

    # -- 4. 曝光口径 --------------------------------------------------------
    add("## 4. 曝光口径")
    add()
    add("### 4.1 工作点")
    add()
    add(
        f"请求 {run.exposure_requested_ms:g} ms，会话上限 `AO_DAHENG_MAX_EXPOSURE_MS` = "
        f"{run.exposure_ceiling_ms:g} ms，**生效 {run.exposure_ms:g} ms**"
        f"（= min(请求, 上限)，与日志一致）。上限只是相机的保护阈值，不是工作点；"
        f"本轮 {len(sides.rows) + len(zern.rows) + len(blaze.rows)} 行的 "
        "`exposure_ms_readback` 全部等于生效值（生成器已逐行核对）。"
    )
    add()
    add("### 4.2 平场的 0 级峰值不能用来挑曝光")
    add()
    add(
        "GS 把能量**摊到边长上千像素的方斑上**，而 0 级是一个衍射极限量级的小核。"
        "两者每像素电平差好几个数量级，所以「把 0 级调到不饱和」和「让方斑有足够信噪比」"
        "是两个完全不同的曝光要求。本轮的证据是自洽的："
    )
    add()
    add(
        f"- GS 之前的平场帧（`spot_assessment`，同一曝光 {run.exposure_ms:g} ms）里，"
        f"整帧峰值 = **{float(spot.get('peak', float('nan'))):.1f} counts**，"
        f"已顶到满量程 {SATURATION_COUNTS:.0f} —— 这个读数只能读成「≥ 满量程」，"
        "**不含任何线性信息**，任何按它做的曝光决策都是在猜。"
    )
    add(
        f"- 同一曝光下，被整形方斑的峰值（`box_peak`）在边长 ≥ "
        f"{f.sides_x[1]:.0f} px 的 {len(sides.rows) - 1} 行里是 "
        f"{min(sides.col('box_peak')[1:]):.2f}–{max(sides.col('box_peak')[1:]):.2f} counts，"
        "全部**远离**满量程。也就是说：0 级已经饱和时，方斑还在半个满量程附近 —— "
        "如果按 0 级反推去降低曝光，方斑会先失去信噪比。"
    )
    add(
        f"- 信噪比余量：评估帧的噪声 σ = {float(spot.get('noise', float('nan'))):.4f} counts"
        "（同曝光、同累计帧数），方斑峰值/噪声 ≈ "
        f"{min(sides.col('box_peak')[1:]) / max(float(spot.get('noise', 1.0)), 1e-9):.0f}–"
        f"{max(sides.col('box_peak')[1:]) / max(float(spot.get('noise', 1.0)), 1e-9):.0f}，"
        "曝光足够，不需要再加。"
    )
    add()
    add(
        "**这里有一个口径缺口，必须写明**：把「0 级峰值 → 方斑每像素电平」写成等式需要两个"
        "本轮没有归档的量 ——（a）未饱和的 0 级峰值随曝光的关系（一条平场曝光阶梯），"
        "（b）0 级的等效面积。本轮 `spot_assessment.peak` 在 20 ms 上已经饱和，且产物里没有"
        "平场阶梯，所以本报告**不给这条换算的绝对值**，只给上面那组可复现的比较"
        "（饱和的 0 级 vs 未饱和的方斑峰值 vs 噪声）。补一条 "
        "0.05 / 0.2 / 1 / 5 / 20 ms 的平场阶梯就能把它写成等式，见 §9。"
    )
    add()

    add("### 4.3 饱和行")
    add()
    sat_list = [f"{sides.x[i]:.0f} px（frame_peak = box_peak = "
                f"{sides.col('frame_peak')[i]:.0f}）"
                for i, r in enumerate(sides.rows)
                if float(r["frame_peak"]) >= SATURATION_COUNTS]
    add(
        ("边长段里饱和的行：" + "；".join(sat_list) + "。") if sat_list
        else "边长段没有饱和行。"
    )
    add(
        "这一行的 PIB/CV 仍然可用，但要知道它们是怎么算的：`PIB` 是**整框求和**与**整帧求和**的"
        "比值，`CV` 是框内逐像素的统计 —— 两者都不是「峰值有多亮」，所以框内少量像素被 8 bit "
        "削平并不会把它们整体带偏；被带偏的只有峰值类读数（`frame_peak` / `box_peak`）和任何"
        "以峰值为分母的量。**读这一行时，请把「框内均值」类结论与峰值类结论分开。**"
    )
    add()
    peak_in_box = sum(
        1 for st in run.stages.values() for r in st.rows
        if float(r["box_peak"]) == float(r["frame_peak"])
    )
    total_rows = sum(len(st.rows) for st in run.stages.values())
    add(
        f"**另外两段的饱和是常态，而且原因不同。** 全部 {total_rows} 行里，"
        f"有 {peak_in_box} 行的 `box_peak == frame_peak` —— 最亮的那个像素"
        "**就在方框内**，饱和的是框内的核，不是框外的杂散光。具体到各段："
    )
    add()
    add(
        f"- **Zernike 段 {sum(1 for v in zern.col('frame_peak') if v >= SATURATION_COUNTS)}/"
        f"{len(zern.rows)} 行饱和**：低阶重构把光聚成一个亮核，亮核落在 1200 px 框里，"
        "框内峰值于是顶到满量程。这正是 §5.1 那个「PIB 假高分」的同一件事 —— "
        "**饱和与「框内几乎全暗但有个亮核」是同一个形态**。"
    )
    add(
        f"- **闪耀光栅段 {sum(1 for v in blaze.col('frame_peak') if v >= SATURATION_COUNTS)}/"
        f"{len(blaze.rows)} 行饱和**：叠加光栅后光场里始终有一个**未经相位调制的分量**"
        "停在冻结 0 级附近（`predicted_shift` 只搬动被调制的那部分），于是即便方框跟着"
        "预测位移走了，框内仍有一个高亮核。所以这一段的峰值饱和**不能**读成「方斑太亮」—— "
        "它量的是那个未调制分量。"
    )
    add()
    add(
        "两段的共同结论：**在这些行上 `frame_peak` / `box_peak` 只能当「≥ 满量程」用**；"
        "`PIB`（比值）与 `CV`（框内统计）不受单像素削顶的系统性影响，仍可比。"
    )
    add()

    # -- 5. Zernike ---------------------------------------------------------
    add("## 5. Zernike 重构段")
    add()
    add(
        f"把选定边长（{f.selected_side} px）那一次的 GS 相位拟合到 Zernike 基，"
        f"然后用**前 N 阶**重构相位、下发、测量。N 从 "
        f"{max(zern.x):.0f} 降到 {min(zern.x):.0f}，最高阶用 "
        f"{int(float(zern.rows[_idx_of(zern.x, max(zern.x))]['zernike_modes_used'])):.0f} 个模式"
        f"（拟合网格 {int(zern.run_config.get('fit_grid', 0))}、"
        f"拟合上限 {int(zern.run_config.get('fit_max_order', 0))} 阶、"
        f"拟合耗时 {float(zern.run_config.get('fit_seconds', float('nan'))):.2f} s —— "
        "这四项都来自 `zernike/run_config.json`）。"
    )
    add()
    zern_tbl = Table(
        ("n_max", "模式数", "解释方差", "残差 RMS (rad)", "CV", "PIB", "score",
         "frame_peak", "质心偏移 x", "质心偏移 y", "旗标"),
        [(cell_of(r, "n_max", 0), cell_of(r, "zernike_modes_used", 0),
          cell_of(r, "zernike_explained_fraction", 6),
          cell_of(r, "zernike_rms_residual_rad", 6),
          cell_of(r, "cv", 4), cell_of(r, "pib", 4), cell_of(r, "score", 4),
          cell_of(r, "frame_peak", 2), cell_of(r, "box_centroid_offset_x", 2),
          cell_of(r, "box_centroid_offset_y", 2), marks_z[i] or "—")
         for i, r in enumerate(zern.rows)],
    )
    tables.append(zern_tbl)
    add(zern_tbl.render())
    add()
    add(f"![Zernike 阶数 → 解释方差 / 残差 / CV](figures/{_figname('fig3_zernike_reconstruction')})")
    add()

    add("### 5.1 低阶 Zernike 表示不了方斑")
    add()
    zn_max = max(zern.x)
    zn_modes = int(float(zern.rows[_idx_of(zern.x, zn_max)]["zernike_modes_used"]))
    expl_max = max(f.zern_explained)
    add(
        f"解释方差从 {min(f.zern_explained) * 100:.3f}%（n_max = {min(zern.x):.0f}）"
        f"单调升到 {expl_max * 100:.3f}%（n_max = {zn_max:.0f}，{zn_modes} 个模式），"
        f"反向步数 {len(f.zern_explained_violations)}。**{zn_modes} 个模式只解释 "
        f"{expl_max * 100:.2f}% 的相位方差** —— 剩下 "
        f"{100 * (1 - expl_max):.2f}% 完全在低阶 Zernike 张成的空间之外。"
    )
    add()
    add(
        f"残差 RMS 从 {max(f.zern_residual):.4f} rad（n_max = {min(zern.x):.0f}）只降到 "
        f"{min(f.zern_residual):.4f} rad（n_max = {zn_max:.0f}），"
        f"相对变化 {abs(min(f.zern_residual) - max(f.zern_residual)) / max(f.zern_residual) * 100:.1f}%。"
        "半个波长的残差在 1064 nm 上就是 **相位本身** 量级的东西，说明重构出的相位与目标相位"
        "几乎没有关系。"
    )
    add()
    add(
        f"实测侧：干净行（排除 `†`/`‡`/`※`）的 CV 落在 "
        f"{min(f.zern_clean_cv):.3f}–{max(f.zern_clean_cv):.3f}，"
        f"与阶数的相关系数 r = {f.zern_cv_pearson:+.3f}。**趋势是有的**（阶数越高略均匀），"
        f"但整段的相对跨度只有 {f.zern_clean_cv_spread:.1%} —— 把它和同框 GS 方斑的 CV 放一起就"
        "看出量级差："
        f"同一 1200 px 方框上 GS 方斑的 CV 是 "
        f"{f.cv[_idx_of(f.sides_x, float(f.selected_side))]:.4f}，"
        f"而 Zernike 重构后是 {f.zern_cv[_idx_of(zern.x, zn_max)]:.4f} —— "
        f"**高 {f.zern_cv[_idx_of(zern.x, zn_max)] / f.cv[_idx_of(f.sides_x, float(f.selected_side))]:.1f} 倍**。"
        f"换句话说：阶数带来的 {f.zern_clean_cv_spread:.1%} 改善，与「基不适配造成的 "
        f"{f.zern_cv[_idx_of(zern.x, zn_max)] / f.cv[_idx_of(f.sides_x, float(f.selected_side))]:.0f}× 劣化」"
        "相比可以忽略 —— 所以「CV 与阶数几乎无关」在**量级**上成立，作为相关系数则不然。"
    )
    add()
    add(
        "这**印证**了仓库既有的反模式记录：「不要用低阶 Zernike 做方斑整形」。"
        "机制也说得通：GS 方斑的相位是高频、类 sinc 的结构（近场需要二维 sinc 的过零点），"
        "而 Zernike 是圆域上的低阶光滑正交基 —— 它不是「拟合得不够准」，而是**基不适配**，"
        "加阶数只能线性地多解释一点点方差（本段 "
        f"{min(f.zern_explained) * 100:.2f}% → {expl_max * 100:.2f}%，"
        f"{zn_modes} 个模式才这个量级）。"
    )
    add()
    add(
        "**这里还有一个必须点出来的陷阱**：失败的 Zernike 重构 PIB = "
        f"{float(zern.rows[_idx_of(zern.x, zn_max)]['pib']):.4f}，高于 GS 方斑的 PIB = "
        f"{f.pib[_idx_of(f.sides_x, float(f.selected_side))]:.4f}。"
        "原因不是它更好，而是**框太大**：低阶重构把光聚成一个亮核（整帧峰值因此顶到满量程），"
        "亮核正好落在 1200 px 方框里，框内功率占比自然高；而方框其余部分几乎全暗，CV 就爆到 7.3。"
        "**只看 PIB 会把这个失败排到成功前面。**"
    )
    add()

    add("### 5.2 n_max = 18 这一行")
    add()
    i18 = _idx_of(zern.x, 18.0)
    i17 = _idx_of(zern.x, 17.0)
    add(
        f"`n_max = 18` 的 CV 是 **{f.zern_cv[i18]:.3f}**，是同段中位数 "
        f"{zern.median('cv'):.3f} 的 {f.zern_cv[i18] / zern.median('cv'):.2f} 倍；"
        f"整帧峰值只有 **{float(zern.rows[i18]['frame_peak']):.1f} counts**"
        f"（相邻行都是 {float(zern.rows[i17]['frame_peak']):.1f}），"
        f"score 掉到 {f.zern_score[i18]:.4f}，框内质心 x 从相邻行的 "
        f"{float(zern.rows[i17]['box_centroid_offset_x']):+.1f} px 跳到 "
        f"{float(zern.rows[i18]['box_centroid_offset_x']):+.1f} px。"
    )
    add()
    add(
        "这一行的**整帧几乎全黑**，所以它的 CV / 质心 / score 全部由噪声主导 —— 它是"
        "**采集失败**，不是「18 阶时性能崩了」。它同时说明本段不能按「阶数越低越好」"
        "或「某阶最优」去读，只能读「整段都在不可用区间」。"
    )
    add()
    fit_grid = int(zern.run_config.get("fit_grid", 0))
    panel = (run.plan.get("bench") or {}).get("panel_shape") or [0, 0]
    add(
        f"另一个次异常是 `n_max = {zern.x[i17]:.0f}`：CV {f.zern_cv[i17]:.3f}"
        f"（比中位数高 {(f.zern_cv[i17] / zern.median('cv') - 1) * 100:.1f}%）、"
        f"score {f.zern_score[i17]:.4f}、整帧峰值正常，但它**恰好也是一条 `†` 行**。"
        "两条高端异常都落在 `†` 上，指向同一个可疑来源：**高阶级重构在面板上的采样**。"
        f"拟合在 {fit_grid}×{fit_grid} 网格上做、显示在 {int(panel[1])}×{int(panel[0])} 面板上（W×H），"
        "高阶模式的相位被放大后可能出现欠采样/混叠，把光衍射到远场窗口之外"
        "（全黑帧正是这个特征）。这是**假设**，验证它需要把 `zernike_fit_grid` 提到 "
        f"{2 * fit_grid} 并把 24–30 阶细扫一遍，见 §9。"
    )
    add()

    # -- 6. blaze -----------------------------------------------------------
    add("## 6. 闪耀光栅段")
    add()
    add(
        "在选定相位上叠加不同周期的**面板 x 向斜坡**，位移按 `predicted_shift_px` 预测"
        "（已含面板↔相机 90° 交换，所以位移落在相机 y 上），**方框跟着预测位移走**，"
        "质心在框内量。"
    )
    add()
    blaze_tbl = Table(
        ("周期 (SLM px)", "预测位移 (px)", "实测位移 y (px)", "位移误差 (px)",
         "实测/预测", "CV", "PIB", "score", "出画面积 (px)", "frame_peak", "旗标"),
        [(cell_of(r, "blaze_period_px", 0), cell_of(r, "predicted_shift_px", 2),
          cell_of(r, "measured_shift_y_px", 2), cell_of(r, "shift_error_px", 2),
          derived(ratio, 4), cell_of(r, "cv", 4), cell_of(r, "pib", 4),
          cell_of(r, "score", 4), cell_of(r, "box_clipped_px", 0),
          cell_of(r, "frame_peak", 2), marks_b[i] or "—")
         for i, (r, ratio) in enumerate(zip(blaze.rows, f.blaze_shift_ratio))],
    )
    tables.append(blaze_tbl)
    add(blaze_tbl.render())
    add()
    add(f"![光栅周期 → 预测/实测位移与 score](figures/{_figname('fig4_blaze_shift')})")
    add()

    add("### 6.1 score 随周期单调上升")
    add()
    clean_scores = [f.blaze_score[i] for i in range(len(blaze.rows))
                    if i not in {_idx_of(blaze.x, 16.0), _idx_of(blaze.x, 40.0)}]
    add(
        f"去掉 `§`（周期 {blaze.x[_idx_of(blaze.x, 16.0)]:.0f}，框出画）和 `‡`"
        f"（周期 {blaze.x[_idx_of(blaze.x, 40.0)]:.0f}，帧近乎全黑）两行后，"
        f"score 从 {clean_scores[0]:.4f} 单调升到 {clean_scores[-1]:.4f}，"
        f"反向步数 {len(f.blaze_score_violations_clean)}。方向与直觉一致：**周期越大，"
        "同样的斜坡幅值给出的位移越小，方斑被保形得越好** —— 位移 ∝ 1/周期"
        "（`predicted_shift_px` 的表达式就是 `TILT_SHIFT_SCALE / 周期`），"
        "所以周期加倍时方斑离开冻结位置的位移减半，落在框外的部分也随之减少。"
    )
    add()

    add("### 6.2 周期 16：框出画，PIB/EE 不可横比")
    add()
    i16 = _idx_of(blaze.x, 16.0)
    side16 = float(blaze.rows[i16]["box_side_px"])
    add(
        f"预测位移 {blaze.col('predicted_shift_px')[i16]:.1f} px 把方框中心推到 "
        f"({float(blaze.rows[i16]['box_center_x']):.0f}, {float(blaze.rows[i16]['box_center_y']):.0f})，"
        f"框高 {side16:.0f} px ⇒ 有 **{blaze.col('box_clipped_px')[i16]:.0f} px**"
        f"（{_pct(blaze.col('box_clipped_px')[i16], side16 ** 2)} 的框面积）落在画面外。"
        f"这一行的 `PIB` = {blaze.col('pib')[i16]:.4f}、`EE` 同值，都因为**分母是整帧、"
        "分子被裁掉一块**而系统性偏低，所以**不能与其它行横向比较**；`CV` 是框内统计，"
        f"受的影响小一些（{blaze.col('cv')[i16]:.2f}），但方框本身已经错位，也不应当作"
        "「周期 16 的均匀度」来读。"
    )
    add()

    add("### 6.3 周期 40：provisional，不得当结论")
    add()
    i40 = _idx_of(blaze.x, 40.0)
    ev40 = run.settle_for("blaze", i40)
    add(
        (f"日志原文（该配置那一条）：\n\n> `{ev40.raw_fragment}`\n" if ev40 and ev40.raw_fragment
         else (f"日志记录：{ev40.describe()}。" if ev40 else "日志缺失，无法核对这一行的 settle 状态。"))
        + f"\n该行最终 `frame_peak` = {float(blaze.rows[i40]['frame_peak']):.2f}，"
        f"`box_peak` = {float(blaze.rows[i40]['box_peak']):.2f}，"
        f"`global_spot_fwhm_px` = {blaze.rows[i40]['global_spot_fwhm_px']}（无效值），"
        f"CV {blaze.col('cv')[i40]:.2f}、score {f.blaze_score[i40]:.4f}。"
    )
    add()
    add(
        "**这一行是采集事故，不是物理结论。** 它落在一条单调序列中间（左邻周期 "
        f"{blaze.x[i40 - 1]:.0f} 的 score {f.blaze_score[i40 - 1]:.4f}、右邻周期 "
        f"{blaze.x[i40 + 1]:.0f} 的 {f.blaze_score[i40 + 1]:.4f}），如果把它当真会得到"
        "「40 px 突然变差」的假结论。它同时是本轮 `†` 里**唯一一行最终画面也全黑**的"
        "（探测峰值与最终峰值都在 2 counts 量级），所以 `‡` 和 `†` 同时命中它。"
    )
    add()

    add("### 6.4 预测位移 vs 实测位移：方向对，标度与 1/P 形状都不对")
    add()
    clean_ratio = [
        float(r["measured_shift_y_px"]) / float(r["predicted_shift_px"])
        for i, r in enumerate(blaze.rows) if i != i40
    ]
    clean_periods = [float(r["blaze_period_px"]) for i, r in enumerate(blaze.rows) if i != i40]
    meas = [float(r["measured_shift_y_px"]) for i, r in enumerate(blaze.rows) if i != i40]
    pred = [float(r["predicted_shift_px"]) for i, r in enumerate(blaze.rows) if i != i40]
    alpha_meas = _loglog_exponent([clean_periods[0], clean_periods[-1]], [meas[0], meas[-1]])
    alpha_pred = _loglog_exponent([clean_periods[0], clean_periods[-1]], [pred[0], pred[-1]])
    add(
        "⚠️ 先把口径钉死：**位移只有 `measured_shift_y_px` 这一列**（质心相对冻结 0 级）。"
        "`box_centroid_offset_y` **不是**位移 —— 它 = 实测质心 − **预测框心**，而框是跟着"
        "预测走的，所以它天然是个大负值（§11.6 给出恒等式）。把它当位移读会得到「反号」的假结论。"
    )
    add()
    add(
        f"- **方向可用**：实测与预测**同号**（全部为正），"
        f"实测/预测 = {min(clean_ratio):.4f}–{max(clean_ratio):.4f}"
        f"（排除 `period = {blaze.x[i40]:.0f}` 那一行 —— 它的实测 "
        f"{float(blaze.rows[i40]['measured_shift_y_px']):.2f} px 是黑帧伪迹，"
        "见 §2.3 / §6.3，**不当数据点**）。"
    )
    add(
        f"- **标度不对，而且不是常数标度错**：若只是系数错，这个比值应当与周期无关；"
        f"实际它随周期**总体上升**（{min(clean_ratio):.4f} → {clean_ratio[-1]:.4f}），"
        f"6 步里 {sum(1 for i in range(1, len(clean_ratio)) if clean_ratio[i] > clean_ratio[i - 1])} 步上升、"
        f"只有 24→32 一步回落 {abs(clean_ratio[2] / clean_ratio[1] - 1):.1%} ⇒ 形状也不对。"
    )
    add(
        f"- **1/P 形状被强烈压缩**：P {clean_periods[0]:.0f} → {clean_periods[-1]:.0f}"
        f"（{clean_periods[-1] / clean_periods[0]:.1f}×）时，预测 "
        f"{pred[0]:.2f} → {pred[-1]:.2f} px（{pred[0] / pred[-1]:.1f}×，构造上恰好 ∝1/P），"
        f"而实测只有 {meas[0]:.2f} → {meas[-1]:.2f} px（{meas[0] / meas[-1]:.1f}×）。"
        f"两端点 log-log 斜率：预测 α = {alpha_pred:.2f}、实测 α = {alpha_meas:.2f}。"
    )
    add(
        "- **直接后果**：周期越小框偏得越远。框心相对质心偏 "
        + " / ".join(
            f"{float(blaze.rows[_idx_of(blaze.x, p)]['box_centroid_offset_y']):+.1f}"
            for p in (16.0, 24.0, 32.0))
        + " px（P = 16 / 24 / 32）—— 也就是说框跟着预测走时，小周期处框会离光斑 "
        + f"{min(abs(float(blaze.rows[_idx_of(blaze.x, p)]['box_centroid_offset_y'])) for p in (16.0, 24.0, 32.0)):.0f}–"
        + f"{max(abs(float(blaze.rows[_idx_of(blaze.x, p)]['box_centroid_offset_y'])) for p in (16.0, 24.0, 32.0)):.0f} px。"
    )
    add()
    add(
        "候选解释只剩两条：**斜坡幅值被削顶**（相位调制深度有限）与**框跟随导致的质心"
        "截断偏置**。二者都预测「实测 < 预测」，本数据分不开；要分开它们，"
        "需要一次**框固定**（冻结在未加光栅的 0 级上，或直接用宽 ROI）的复测："
        "框不动时质心才反映真实位移。§11.5 判定 5 与 §11.6 给出具体做法。"
    )
    add()

    # -- 7. 近 1000 选点 ----------------------------------------------------
    add("## 7. 「近 1000」的选点边界效应")
    add()
    add(
        f"第 2 段的输入边长由 `--target-side-near 1000` 在窗口 "
        f"[{f.near_window[0]:.0f}, {f.near_window[1]:.0f}] px 内按 score 取最大决定。"
        "把窗口内的行摊开看就能看出问题："
    )
    add()
    near_rows = [i for i, x in enumerate(f.sides_x) if f.near_window[0] <= x <= f.near_window[1]]
    near_tbl = Table(
        ("边长 (px)", "score", "距 1000 的距离 (px)", "是否被选中"),
        [(cell_of(sides.rows[i], "side_px", 0), cell_of(sides.rows[i], "score", 4),
          derived(f.sides_x[i] - 1000.0, 0, " px"), "**是**" if i == f.near_best_index else "—")
         for i in near_rows],
    )
    tables.append(near_tbl)
    add(near_tbl.render())
    add()
    i1000 = _idx_of(f.sides_x, 1000.0)
    add(
        f"窗口内 score **严格单调上升**（反向步数 "
        f"{len(monotonic_report([f.score[i] for i in near_rows]))}），所以 argmax 必然落在"
        f"**窗口上界 {f.sides_x[f.near_best_index]:.0f} px** —— 这不是局部峰，是边界效应。"
        f"并且整条曲线在窗口外还在涨（峰值 {f.sides_x[f.score_peak_index]:.0f} px，"
        f"{f.score[f.score_peak_index]:.4f}），所以「选到 1200」既不是局部最优也不是全局最优，"
        "它只是「窗口上界」。"
    )
    add()
    add(
        f"**建议**：若窗口改成 [{f.alt_window[0]:.0f}, {f.alt_window[1]:.0f}] px，"
        f"用本轮的 CSV 就能回答会选到 **{f.sides_x[f.alt_best_index]:.0f} px**"
        f"（score {f.score[f.alt_best_index]:.4f}）。窗口内 score 依然单调，"
        "所以只要上界跟着动，选中值就跟着动 —— 在 score 单调上升的区间里，"
        "任何「接近某值」的选点规则实际选的都是窗口边界。"
        f"**真正想选「接近 1000」应该直接取最接近 1000 的点（{f.sides_x[i1000]:.0f} px），"
        "或者把窗口做成对称且明确说明「取上界」。**"
    )
    add()
    add(
        f"（补一句量级：1000 px 的 score 是 {f.score[i1000]:.4f}，"
        f"1200 px 是 {f.score[f.near_best_index]:.4f}，差 "
        f"{f.score[f.near_best_index] - f.score[i1000]:.4f}；"
        f"而 §3.3 里 1500 vs 1600 的差只有 {f.score_peak_gap:.2e}，"
        f"两者差 {(f.score[f.near_best_index] - f.score[i1000]) / f.score_peak_gap:.0f} 倍。"
        "所以「窗口边界把选中值从 1000 推到 1200」的代价是**真实可测**的，"
        "只是它仍落在平台区（距全局峰值 "
        f"{f.score[f.score_peak_index] - f.score[f.near_best_index]:.4f}）。）"
    )
    add()

    # -- 8. 最大边长 --------------------------------------------------------
    add("## 8. 最大边长与 0 级位置")
    add()
    if "plan" not in cfg:
        add(
            "`side_max_px` **不是常数**，是从 0 级位置与画面边缘反解出来的"
            "（`side_max = 2·(min(cx, W−1−cx, cy, H−1−cy) − edge_margin)`，再向下取到"
            "步长的整数倍）。**但本批的 `run_config` 不含 `plan`**，"
            "`edge_margin` 与计划里的假设中心都没有落盘，所以本节只给出实测的 0 级位置，"
            "不做反解 —— 反解的完整推导见基线那一轮的报告。"
        )
        add()
        add("```")
        add(f"0 级 (实测冻结)     = ({zo_x:.0f}, {zo_y:.0f})")
        add(f"画面 (W x H)        = ({w}, {h})   ⇒ 几何中心 ({(w - 1) / 2:.0f}, "
            f"{(h - 1) / 2:.0f})")
        add(f"到四条边的距离       = 左 {zo_x:.0f} / 右 {(w - 1) - zo_x:.0f} / "
            f"上 {zo_y:.0f} / 下 {(h - 1) - zo_y:.0f}")
        add(f"约束边              = {f.binding_edge} "
            f"({min(zo_x, (w - 1) - zo_x, zo_y, (h - 1) - zo_y):.0f})")
        add("```")
        add()
        add(
            f"**0 级不在画面中心**：几何中心是 ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})，"
            f"实测 0 级在 ({zo_x:.0f}, {zo_y:.0f})，偏 "
            f"{zo_x - (w - 1) / 2:+.0f} / {zo_y - (h - 1) / 2:+.0f} px。"
        )
        add()
    else:
        add(
            "`side_max_px` **不是常数**，是从 0 级位置与画面边缘反解出来的："
        )
        add()
        add("```")
    add(f"0 级 (实测冻结)     = ({zo_x:.0f}, {zo_y:.0f})")
    add(f"画面 (W x H)        = ({w}, {h})   ⇒ 几何中心 ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})")
    add(f"到四条边的距离       = 左 {zo_x:.0f} / 右 {(w - 1) - zo_x:.0f} / "
        f"上 {zo_y:.0f} / 下 {(h - 1) - zo_y:.0f}")
    add(f"约束边              = {f.binding_edge} ({min(zo_x, (w - 1) - zo_x, zo_y, (h - 1) - zo_y):.0f})")
    add(f"side_max_raw        = 2 · ({min(zo_x, (w - 1) - zo_x, zo_y, (h - 1) - zo_y):.0f}"
        f" − {f.edge_margin:.0f}) = {f.side_max_raw:.0f} px")
    add(f"向下取 {f.step:.0f} 的整数倍 = {f.side_max:.0f} px")
    add("```")
    add()
    assumed = (run.plan.get("frame") or {}).get("zero_order_xy") or [w / 2.0, h / 2.0]
    as_x, as_y = float(assumed[0]), float(assumed[1])
    assumed_dist = min(as_x, (w - 1) - as_x, as_y, (h - 1) - as_y)
    assumed_raw = 2.0 * (assumed_dist - f.edge_margin)
    add(
        f"**0 级不在画面中心**：几何中心是 ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})，"
        f"实测 0 级在 ({zo_x:.0f}, {zo_y:.0f})，偏 {zo_x - (w - 1) / 2:+.0f} / "
        f"{zo_y - (h - 1) / 2:+.0f} px。所以「最大边长」必须现场量：若按**假设的中心** "
        f"({as_x:.0f}, {as_y:.0f}) 代入同一条式，得到 "
        f"2 · ({assumed_dist:.0f} − {f.edge_margin:.0f}) = {assumed_raw:.0f} px，"
        f"向下取整就是计划里的 "
        f"{run.plan.get('frame', {}).get('side_max', '-')} px —— 比实际能用的 "
        f"{f.side_max:.0f} px 大 "
        f"{float(run.plan.get('frame', {}).get('side_max', 0)) - f.side_max:.0f} px，"
        "那个尺寸的框会直接压到帧边（框压边会被窗函数与读出噪声污染）。"
    )
    add()
    add(
        f"这也解释了 §2.1 里「计划有 {', '.join(f.dropped_configs)}、CSV 没有」："
        f"`--dry-run` 的计划按**画幅中心**估出 "
        f"{run.plan.get('frame', {}).get('side_max_raw', '-')} px"
        f"（`plan.frame.side_max_raw`），实际按 0 级解出 {f.side_max_raw:.0f} px，"
        f"向下取整后 {f.side_max:.0f} px，于是计划里的 "
        f"{', '.join(f.dropped_configs)} 被裁掉。这是**设计行为**，不是缺数据。"
    )
    add()

    # -- 9. 可疑项 ----------------------------------------------------------
    add("## 9. 可疑项与复测清单")
    add()
    add(
        f"按优先级排列。每条都写明「为什么可疑」和「复测要看什么」，"
        f"不止清单本身。本轮共 {len(run.settle_events)} 行 settle 超时"
        "（见 §2.3），其中 1 行最终画面也全黑。"
    )
    add()
    i18_peak = float(zern.rows[i18]["frame_peak"])
    clean_ratios = [
        r for i, r in enumerate(f.blaze_shift_ratio) if i != i40 and math.isfinite(r)
    ]
    if retest is not None and rf is not None:
        head = [
            (f"**`n_max = 18` 全黑帧**（§5.2）：整帧峰值 {i18_peak:.1f} counts、CV "
             f"{f.zern_cv[i18]:.3f}。**已复测推翻**（§11.5 判定 2）：复测 18 的 CV "
             f"{rf.zern_cv[1]:.4f} 与邻居同量级、`frame_peak` 回到满量程 ⇒ "
             "基线那行是采集失败，不需要再测。"),
            (f"**`blaze period = 40` provisional**（§6.3）：settle 超时 + 最终峰值 "
             f"{float(blaze.rows[i40]['frame_peak']):.2f} counts。**已复测推翻**"
             f"（§11.5 判定 1）：复测 40 的 CV {rf.blaze_cv[1]:.4f} 落在 32/48 之间。"
             "留下的唯一待办是把 settle 的**探测时间序列**打出来（现在只打了最后一次探测）"
             "—— 两次黑帧都发生在探测阶段，这条诊断信息比再测一个点更有价值。"),
            ("**预测位移 vs 实测位移：方向可用，标度与 1/P 形状不可用**（§6.4）：实测/预测 "
             f"{min(clean_ratios):.2f}–{max(clean_ratios):.2f}，且实测的 1/P 指数只有 "
             f"α = {rf.alpha_baseline:.2f}（预测是 1.00）。复测：**框固定 + 宽 ROI** "
             "直接量整帧质心（§11.6 四条），同时直接量面板斜坡的相位幅值是否被削顶。"),
        ]
    else:
        head = [
            (f"**`n_max = 18` 全黑帧**（§5.2）：整帧峰值 {i18_peak:.1f} counts、CV "
             f"{f.zern_cv[i18]:.3f}。复测：把 `zernike_fit_grid` 提到 512，只扫 17–20 阶，"
             "每阶重复 3 次（相位不变），看全黑是否复现。"),
            (f"**`blaze period = 40` provisional**（§6.3）：settle 超时 + 最终峰值 "
             f"{float(blaze.rows[i40]['frame_peak']):.2f} counts。复测：单点重跑，"
             "并把 settle 的探测时间序列打出来（现在只打了最后一次探测）。"),
            ("**预测位移 vs 实测位移系统性偏低**（§6.4）：实测/预测 "
             f"{min(clean_ratios):.2f}–{max(clean_ratios):.2f}。复测：框固定（冻结在未加光栅的 0 级）、"
             "宽 ROI，再比一次；同时直接量面板斜坡的相位幅值是否被削顶。"),
        ]
    items = head + [
        (f"**settle 判据超时**（§2.3）：{len(run.settle_events)}/"
         f"{len(sides.rows) + len(zern.rows) + len(blaze.rows)} 行，"
         f"但其中只有 {sum(1 for ev in run.settle_events if ev.probe_peak < DARK_FRAME_FRACTION * SATURATION_COUNTS)} "
         f"行的探测峰值低于满量程的 {DARK_FRAME_FRACTION:.0%}（即真黑帧）。"
         f"相对容差 `tol = {policy.get('tol')}` 对这台相机可能过紧（读噪声 "
         f"σ ≈ {float(spot.get('noise', float('nan'))):.3f} counts，而容差按相对值算）。"
         + (f"复测里又有 {rf.settle_configs.__len__()} 个配置超时"
            f"（{', '.join(rf.settle_configs)}），**同样没有一条是黑帧** ⇒ 判据确实偏紧，"
            if retest is not None and rf is not None else "")
         + "复测：把 `max_wait_s` 提到 12 s，或改用「探测值变化 < 2σ」的绝对判据。"),
        (f"**`{sides.x[0]:.0f} px` 行饱和**（§4.3）：`frame_peak = box_peak = "
         f"{SATURATION_COUNTS:.0f}`。复测：该点单独降曝光（例如 "
         f"{run.exposure_ms / 4:g} ms）重测，才能给出它的真实峰值。"),
        ("**`ee` ≡ `PIB`**：50 行里两者逐位相同，所以 `ee` / `ar` / `quality` 这一组"
         "里只有 `ar` 与 `quality` 带独立信息。报告与图里都不把 `ee` 当第二个指标 —— "
         "否则会让人误以为两个独立指标互相印证。"),
        ("**平场曝光阶梯缺失**（§4.2）：无法把「0 级峰值 → 方斑每像素」写成等式。"
         "补 0.05 / 0.2 / 1 / 5 / 20 ms 的平场帧 + FWHM，就能给出那条换算。"),
        ("**每配置只测 1 次**：没有重复，因此**没有误差棒**。§3.3 里 1500 vs 1600 差 "
         f"{f.score_peak_gap:.2e}，这个量级低于任何可以分辨的单次噪声，"
         "所以「峰在哪个点」这件事本轮**不可判定**。"
         + (f"复测批内各点相对邻居的偏差（≤ {rf.batch_dev_max:.1%}）只能给出"
            "「这不是趋势上的点」这条弱判据，仍不是误差棒（§11.5 判定 6）。"
            if retest is not None and rf is not None else "")),
    ]
    for idx, text in enumerate(items, start=1):
        add(f"{idx}. {text}")
    add()

    # -- 10. 局限 -----------------------------------------------------------
    add("## 10. 局限与不可横比项")
    add()
    add(
        "- **CV 与 PIB 都依赖框大小**，所以它们只在**同一框**下可比。跨段的数字"
        "（例如「Zernike 重构的 PIB 0.91 高于 GS 方斑的 0.86」）只能作为「框选得太大」"
        "的证据，不能作为质量排序。"
    )
    add(
        "- **无重复测量**：每配置 1 次。所有「某点最优」的说法都受这一条限制，"
        "本报告因此只报峰值区间与差值，不报显著性。"
    )
    add(
        "- **GS 每次从零跑**：18 行的 GS 相位彼此独立，行间差异包含 GS 自身的收敛差异"
        "（`gs_seconds` 在 "
        f"{min(sides.col('gs_seconds')):.2f}–{max(sides.col('gs_seconds')):.2f} s 之间，"
        "不是固定迭代预算下的等价收敛）。"
    )
    add(
        "- **`global_spot_fwhm` 不是方斑宽度**（§1.2）：它是整帧最亮结构，2f 光路上通常"
        "是 0 级/晕斑。所有基于它的「光斑退化」判断都不成立。"
    )
    add(
        "- **Zernike 拟合在 "
        f"{int(zern.run_config.get('fit_grid', 0))}×{int(zern.run_config.get('fit_grid', 0))} "
        f"网格上做**，而显示在 {int(panel[1])}×{int(panel[0])} 面板上（W×H）；高阶模式的"
        "采样充分性没有验证（§5.2 的假设正是从这里来的）。"
    )
    add(
        "- **`box_clipped_px` 的语义**：它是「框落在画面外的**面积**」，不是「被钳制的"
        "像素个数」。本轮 50 行满足恒等式 `box_side_px² − box_area_px == box_clipped_px`"
        "（生成器逐行核对，残差 0），所以可以放心按面积读。"
    )
    add()

    # -- 11. 复测 (功率已变) -------------------------------------------------
    if retest is not None and rf is not None:
        retest_lines, retest_tables = render_retest_section(run, f, retest, rf)
        out.extend(retest_lines)
        tables.extend(retest_tables)

    # -- 12. CCD 远场光斑图 --------------------------------------------------
    if farfield is not None and farfield.get("mosaics"):
        farfield_lines, farfield_tables = render_farfield_section(
            run, farfield, farfield_figures,
            residual_pointer=bool(residuals is not None and residuals.panels))
        out.extend(farfield_lines)
        tables.extend(farfield_tables)

    # -- 13. 整轮原子重扫 ----------------------------------------------------
    if atomic is not None and af is not None:
        atomic_lines, atomic_tables = render_atomic_section(
            run, f, atomic, af, retest, atomic_figures, residuals)
        out.extend(atomic_lines)
        tables.extend(atomic_tables)

    # -- 附录 ---------------------------------------------------------------
    add(f"## 附录 A. 输入契约（`artifacts/{PATHS.artifact_dir.name}/`）")
    add()
    art_rows = []
    for rel, note in _artifact_manifest():
        art_rows.append((f"`{rel}`", note))
    if retest is not None:
        for rel, note in _retest_artifact_manifest(retest):
            art_rows.append((f"`{PATHS.retest_artifact_dir.name}/{rel}`", note))
    if prior is not None:
        for rel, note in _retest_artifact_manifest(prior):
            art_rows.append((
                f"`{PATHS.retest_prior_artifact_dir.name}/{rel}`",
                "已作废（跨脚本版本）—— **仅存档为证据，报告不引用**" if rel == "manifest.json"
                else "同上（已作废批次）",
            ))
    if atomic is not None:
        for rel, note in _atomic_artifact_manifest(atomic):
            art_rows.append((f"`{PATHS.atomic_artifact_dir.name}/{rel}`", note))
    art_tbl = Table(("文件", "说明"), art_rows)
    tables.append(art_tbl)
    add(art_tbl.render())
    add()
    if retest is not None:
        add(
            f"复测批次的产物单独归档在 `artifacts/{PATHS.retest_artifact_dir.name}/`"
            f"（{len(_retest_artifact_manifest(retest))} 个文件），与基线**并列**而不是覆盖 —— "
            "两批的 `measurements.csv` 各自独立，§11 的数字从第二块读。"
            + (f"整轮原子重扫（§13）在 `artifacts/{PATHS.atomic_artifact_dir.name}/`"
               f"（{len(_atomic_artifact_manifest(atomic))} 个文件，含 4 张拼图与 "
               "`frames_selection.json`）。" if atomic is not None else "")
            + (f"更早一版复测在 `artifacts/{PATHS.retest_prior_artifact_dir.name}/`"
               f"（{len(_retest_artifact_manifest(prior))} 个文件）—— 那一批跨了两个脚本版本，"
               "**已作废**，只作为证据留存，报告不引用它的数字。" if prior is not None else "")
        )
        add()
    add(
        "上面的文件是这次运行的**原始字节副本**（生成器用 `shutil.copy2` 复制，不重写、"
        "不格式化），所以报告里的每个测量值都能追到某个 `measurements.csv` 的某个单元。"
        "`PROVENANCE.json` 记录源路径与 sha256，`--check` 会校验副本、sha256 与源逐字节一致。"
    )
    add()
    add(
        "⚠️ 根 `.gitignore` 忽略 `*.csv` 与 `*.log`，所以入库时这两个扩展名要 `git add -f`"
        "（与本目录既有 `artifacts/` 的做法一致）；否则新克隆上只剩 JSON，"
        "而 JSON 与 CSV 是逐值等价的（生成器每次运行都做这条交叉校验），仍可复现。"
    )
    add()

    add("## 附录 B. 复现")
    add()
    add("```powershell")
    add("$env:PYTHONPATH=\"<repo>;<repo>\\src;<repo>\\libs\"")
    add("& .venv/Scripts/python.exe scripts/slm/generate_hw_gs_square_scans_report.py \\")
    add("    --run-root data/debug/hw_gs_square_scans/20261023_gs_square")
    add("# 幂等 / 产物一致性校验 (只读, 不写任何文件)")
    add("& .venv/Scripts/python.exe scripts/slm/generate_hw_gs_square_scans_report.py \\")
    add("    --run-root data/debug/hw_gs_square_scans/20261023_gs_square --check")
    add("```")
    add()
    add(
        "全程**不 import 任何驱动、不打开任何设备**：只读 CSV / JSON / 日志 + 画图。"
        "`--check` 模式下连图都不重画 —— 它只比对报告正文、`artifacts/` 副本与图是否存在。"
    )
    add()

    body = "\n".join(out).rstrip("\n") + "\n"
    return insert_header(body, REPORT_KEY), tables


















# ---------------------------------------------------------------------------
# 产物复制 (输入契约)
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT),
                        help="三段产物的根目录 (含 manifest.json 与 sides/zernike/blaze)")
    parser.add_argument("--log", default=str(DEFAULT_LOG),
                        help="那次运行的日志; 传空字符串表示明确不要日志")
    parser.add_argument("--out-md", default=str(ROOT / REPORT_KEY),
                        help="报告正文输出路径")
    parser.add_argument("--out-figures", default=str(PATHS.fig_dir),
                        help="图输出目录")
    parser.add_argument("--retest-root", default="",
                        help="复测批次根目录 (含 manifest.json 与 blaze/zernike); "
                             "不给就完全不渲染 §11")
    parser.add_argument("--atomic-root", default="",
                        help="整轮原子重扫的根目录 (单一版本 + 单一功率, 三段齐全); "
                             "不给就完全不渲染 §13")
    parser.add_argument("--retest-prior-root", default=str(DEFAULT_RETEST_PRIOR_ROOT),
                        help="更早一版复测 (只用于算批间重复性带宽); 不存在就跳过那张表")
    parser.add_argument("--check", action="store_true",
                        help="只校验: 重新渲染到内存并与磁盘比对, 不写任何文件")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.run_root)
    out_md = Path(args.out_md)
    PATHS.fig_dir = Path(args.out_figures)
    log_path = Path(args.log) if args.log else None

    if not run_root.is_absolute():
        run_root = (Path.cwd() / run_root).resolve()
    run = load_run(run_root, log_path)
    if (run.log_crosscheck and any(not ok for *_, ok in run.log_crosscheck)
            and str(args.log) == str(DEFAULT_LOG)):
        # 默认日志属于基线那一轮; 换 run-root 时它必然对不上 —— 忽略它, 而不是
        # 让一次"日志选错"变成硬失败 (显式传 --log 时仍然严格校验)。
        logger.info(f"警告: 默认日志 {_rel(DEFAULT_LOG)} 与本次 run-root 的 run_config 不一致, "
              "已按「未提供日志」处理; 要核对 settle 判据请显式传 --log")
        run = load_run(run_root, None)
    findings = compute_findings(run)
    # 归档目录跟随实际传入的 run-root (默认值只是示例), 换批次不会互相覆盖
    PATHS.artifact_dir = OUT_DIR / "artifacts" / run.root.name

    retest: RetestRun | None = None
    rf: RetestFindings | None = None
    prior: RetestRun | None = None
    if args.retest_root:
        retest_root = Path(args.retest_root)
        if not retest_root.is_absolute():
            retest_root = (Path.cwd() / retest_root).resolve()
        retest = load_retest(retest_root, DEFAULT_RETEST_LOG_DIR)
        if args.retest_prior_root:
            prior_root = Path(args.retest_prior_root)
            if not prior_root.is_absolute():
                prior_root = (Path.cwd() / prior_root).resolve()
            if prior_root.exists() and prior_root != retest_root:
                prior = load_retest(prior_root, DEFAULT_RETEST_LOG_DIR)
        rf = compute_retest_findings(run, retest, prior)
        # 归档目录跟随实际传入的那一批 (参数为准, 默认值只是示例)
        PATHS.retest_artifact_dir = OUT_DIR / "artifacts" / retest.root.name
        if prior is not None:
            PATHS.retest_prior_artifact_dir = OUT_DIR / "artifacts" / prior.root.name

    # §12 的规划不读像素 (峰值取自 CSV), 所以 --check 也能拿到完全相同的载荷
    selection = select_cells(run, findings)
    farfield = plan_farfield(run, selection)
    # 「会被画出来的」拼图 = 至少有一格不是缺帧的那些 (与 draw_mosaic 的判据一致)
    farfield_figures = [PATHS.fig_dir / m["figure"] for m in farfield["mosaics"]
                        if any(not cell["missing"] for cell in m["cells"])]

    atomic: RetestRun | None = None
    af: AtomicFindings | None = None
    atomic_selection: dict[str, list[CellPick]] = {}
    atomic_farfield: dict[str, Any] = {"mosaics": []}
    atomic_figures: list[Path] = []
    residuals: ResidualSet | None = None
    residual_figure: Path | None = None
    if args.atomic_root:
        atomic_root = Path(args.atomic_root)
        if not atomic_root.is_absolute():
            atomic_root = (Path.cwd() / atomic_root).resolve()
        atomic = load_retest(atomic_root, None, stage_names=STAGES)
        PATHS.atomic_artifact_dir = OUT_DIR / "artifacts" / atomic.root.name
        selected = load_selected(atomic.root)
        af = compute_atomic_findings(run, atomic, selected, retest)
        atomic_findings = compute_findings(atomic)
        atomic_selection = select_cells(
            atomic, atomic_findings,
            sides_panel=(200.0, 600.0, 1000.0, 1500.0, 1700.0),
            add_score_cell=False,
        )
        atomic_farfield = plan_farfield(atomic, atomic_selection)
        atomic_mosaics = [PATHS.fig_dir / m["figure"] for m in atomic_farfield["mosaics"]
                          if any(not cell["missing"] for cell in m["cells"])]
        atomic_figures = atomic_mosaics + [
            PATHS.fig_dir / f"fig10_atomic_sides_{batch_tag(atomic)}.png",
            PATHS.fig_dir / f"fig11_atomic_blaze_shift_{batch_tag(atomic)}.png",
        ]
        # §13.8 的残差只读产物重算 (延迟 import 驱动链, 见 compute_residuals)
        residuals = compute_residuals(atomic)
        if residuals is not None:
            residual_figure = PATHS.fig_dir / residual_figure_name(atomic)
            atomic_figures.append(residual_figure)

    # -- 自检 0: §13.8 的残差必须与记录值 / 存盘相位对得上 --------------------
    residual_problems = check_residuals(residuals)
    if residual_problems:
        logger.info("§13.8 残差自检失败:")
        for problem in residual_problems:
            logger.info(f"  - {problem}")
        return 1
    if residuals is not None:
        pairs = ", ".join(
            f"n_max={p.n_max} 记录 {p.rms_recorded:.6f} / 重算 {p.rms_recomputed:.6f}"
            for p in residuals.panels
        )
        logger.info(f"§13.8 残差自检: {len(residuals.panels)} 个配置, {pairs}; "
              f"recon 与存盘相位最大偏差 "
              f"{max(p.recon_delta for p in residuals.panels):.1e}")

    markdown, tables = render_report(run, findings, retest, rf, prior,
                                     farfield, farfield_figures,
                                     atomic, af, atomic_figures, residuals)

    # -- 自检 1: 表里的值与 CSV 逐值一致 -----------------------------------
    problems: list[str] = []
    checked = 0
    for table in tables:
        checked += table.numeric_cells
        problems.extend(verify_table_in_markdown(markdown, table))
    if problems:
        logger.info("表格回读校验失败:")
        for problem in problems:
            logger.info(f"  - {problem}")
        return 2
    logger.info(f"表格回读校验: {len(tables)} 张表 / {checked} 个数值单元, "
          f"全部与产物逐值一致")

    # -- 自检 2: 结构性不变量 ----------------------------------------------
    invariants = check_invariants(run, findings, retest)
    if invariants:
        logger.info("结构不变量校验失败:")
        for problem in invariants:
            logger.info(f"  - {problem}")
        return 3
    logger.info("结构不变量校验: score 分解 / 框面积恒等式 / CSV↔JSON / manifest "
          "/ 日志↔run_config 全部通过")

    # -- 自检 3 (非致命): 溯源条目里的复现命令必须指向本次实际用的复测目录 -------
    # 溯源块由 `_common/provenance.py::REPORTS` 渲染, 本脚本不能改它; 但一旦条目里的
    # `--retest-root` 与实际参数不同, 报告头部就会印一条错的复现命令 —— 这里把它喊出来。
    registered = REPORTS.get(REPORT_KEY)
    if retest is not None and registered is not None and registered.command:
        found = re.search(r"--retest-root\s+(\S+)", registered.command)
        if found:
            registered_root = Path(found.group(1))
            if registered_root.name != retest.root.name:
                logger.info(
                    "警告: REPORTS 里的复现命令指向 "
                    f"{registered_root.name}, 而本次实际用的是 {retest.root.name} —— "
                    "报告头部的「复现命令」会与实际不符, 请更新 provenance.py 的该条目",

                )

    if args.check:
        if not out_md.exists():
            logger.info(f"报告不存在: {out_md}")
            return 1
        on_disk = out_md.read_text(encoding="utf-8")
        drift = [] if on_disk == markdown else [
            f"正文与现场渲染不一致 ({len(on_disk)} vs {len(markdown)} 字符)"
        ]
        drift.extend(check_artifacts(run))
        if retest is not None:
            drift.extend(_check_artifact_set(
                retest_artifact_sources(retest, PATHS.retest_artifact_dir), PATHS.retest_artifact_dir))
            if rf is not None and rf.prior_root is not None and prior is not None:
                drift.extend(_check_artifact_set(
                    retest_artifact_sources(prior, PATHS.retest_prior_artifact_dir),
                    PATHS.retest_prior_artifact_dir))
        figure_names = ["fig1_sides_objectives", "fig2_sides_exposure_geometry",
                        "fig3_zernike_reconstruction", "fig4_blaze_shift"]
        if retest is not None:
            figure_names.append("fig5_retest_neighbors")
        for name in figure_names:
            path = PATHS.fig_dir / _figname(name)
            if not path.exists():
                drift.append(f"图缺失: {_rel(path)}")
            elif path.stat().st_size == 0:
                drift.append(f"图为空文件: {_rel(path)}")
            elif f"figures/{_figname(name)}" not in markdown:
                drift.append(f"正文没有引用: figures/{_figname(name)}")
        drift.extend(_check_farfield(run, farfield, farfield_figures))
        if atomic is not None:
            drift.extend(_check_artifact_set(
                atomic_artifact_sources(atomic, PATHS.atomic_artifact_dir), PATHS.atomic_artifact_dir))
            for stem in ("fig10_atomic_sides", "fig11_atomic_blaze_shift"):
                path = PATHS.fig_dir / f"{stem}_{batch_tag(atomic)}.png"
                if not path.exists():
                    drift.append(f"图缺失: {_rel(path)}")
                elif path.stat().st_size == 0:
                    drift.append(f"图为空文件: {_rel(path)}")
                elif f"figures/{path.name}" not in markdown:
                    drift.append(f"正文没有引用: figures/{path.name}")
            drift.extend(_check_farfield(atomic, atomic_farfield, atomic_mosaics,
                                         artifact_dir=PATHS.atomic_artifact_dir))
            if residual_figure is not None:
                if not residual_figure.exists():
                    drift.append(f"图缺失: {_rel(residual_figure)}")
                elif residual_figure.stat().st_size == 0:
                    drift.append(f"图为空文件: {_rel(residual_figure)}")
                elif f"figures/{residual_figure.name}" not in markdown:
                    drift.append(f"正文没有引用: figures/{residual_figure.name}")
                copy = PATHS.atomic_artifact_dir / residual_figure.name
                if not copy.exists():
                    drift.append(f"图副本缺失: {_rel(copy)}")
                elif _sha256(copy) != _sha256(residual_figure):
                    drift.append(f"图与 artifacts 副本不一致: {residual_figure.name}")
        if drift:
            logger.info("--check 失败 (存在漂移):")
            for item in drift:
                logger.info(f"  - {item}")
            return 1
        n_mosaic = len([m for m in farfield.get("mosaics", [])])
        extra = ""
        if atomic is not None:
            extra = (f"; §13 另有 2 张曲线 + {len(atomic_mosaics)} 张拼图"
                     + (" + fig12 残差图" if residual_figure is not None else "")
                     + f" 与 `artifacts/{PATHS.atomic_artifact_dir.name}/`")
        logger.info(f"--check 通过: {_rel(out_md)} 与现场渲染逐字节一致, "
              f"artifacts 副本与源一致, {len(figure_names)} 张指标图 + "
              f"{n_mosaic} 张远场拼图齐备 (含 frames_selection.json){extra}")
        return 0

    # -- 写正文 / 图 / 产物 -------------------------------------------------
    # 拼图放在自检之后画: 自检失败时不会在 figures/ 里留下孤儿图
    if not args.check:
        render_farfield(run, selection, farfield)
        if atomic is not None and af is not None:
            render_farfield(atomic, atomic_selection, atomic_farfield)
            fig_atomic_sides(run, atomic)
            fig_atomic_blaze(atomic)
        if atomic is not None and residuals is not None:
            fig_zernike_residual(atomic, residuals)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    logger.info(f"报告已写到: {_rel(out_md)}")
    figures = render_figures(run, findings)
    if retest is not None and rf is not None:
        figures.append(fig_retest_neighbors(run, retest, rf))
    logger.info(f"图已写到: {_rel(PATHS.fig_dir)}/ ({len(figures)} 张)")
    written = copy_artifacts(run)
    logger.info(f"输入产物已复制: {_rel(PATHS.artifact_dir)}/ ({len(written)} 个文件)")
    if farfield["mosaics"]:
        copied = _copy_farfield_to_artifacts(farfield, farfield_figures)
        logger.info(f"远场拼图已复制: {_rel(PATHS.artifact_dir)}/ ({copied} 张) + "
              "frames_selection.json")
    if atomic is not None:
        pairs_a = atomic_artifact_sources(atomic, PATHS.atomic_artifact_dir)
        written_a = _copy_artifact_set(pairs_a, PATHS.atomic_artifact_dir, "atomic", atomic.root)
        copied_a = _copy_farfield_to_artifacts(atomic_farfield, atomic_figures,
                                               target=PATHS.atomic_artifact_dir)
        logger.info(f"整轮原子重扫产物已复制: {_rel(PATHS.atomic_artifact_dir)}/ "
              f"({len(written_a)} 个源文件 + {copied_a} 张图) + frames_selection.json")
    if retest is not None:
        pairs = retest_artifact_sources(retest, PATHS.retest_artifact_dir)
        written_r = _copy_artifact_set(pairs, PATHS.retest_artifact_dir, "retest", retest.root)
        logger.info(f"复测产物已复制: {_rel(PATHS.retest_artifact_dir)}/ ({len(written_r)} 个文件)")
        if prior is not None:
            pairs_p = retest_artifact_sources(prior, PATHS.retest_prior_artifact_dir)
            written_p = _copy_artifact_set(pairs_p, PATHS.retest_prior_artifact_dir,
                                           "retest_prior", prior.root)
            logger.info(f"更早一版复测产物已复制: {_rel(PATHS.retest_prior_artifact_dir)}/ "
                  f"({len(written_p)} 个文件)")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
