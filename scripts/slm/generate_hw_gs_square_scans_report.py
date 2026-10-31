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
from scripts.slm._hw_gs_square_scans_model import ReportCtx
from scripts.slm._hw_gs_square_scans_sections_core import (
    render_s0, render_s1, render_s2, render_s3, render_s4, render_s5,
)
from scripts.slm._hw_gs_square_scans_sections_scan import (
    render_s6, render_s7, render_s8, render_s9, render_s10, render_appendix,
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


    ctx = ReportCtx(
        run=run, f=f, retest=retest, rf=rf, prior=prior,
        farfield=farfield, farfield_figures=farfield_figures,
        atomic=atomic, af=af, atomic_figures=atomic_figures,
        residuals=residuals, tables=tables,
        sides=sides, zern=zern, blaze=blaze, cfg=cfg, bench=bench,
        spot=spot, h=h, w=w, zo_x=zo_x, zo_y=zo_y,
        marks_s=marks_s, marks_z=marks_z, marks_b=marks_b,
    )
    out.extend(render_s0(ctx))
    out.extend(render_s1(ctx))
    out.extend(render_s2(ctx))
    out.extend(render_s3(ctx))
    out.extend(render_s4(ctx))
    out.extend(render_s5(ctx))
    out.extend(render_s6(ctx))
    out.extend(render_s7(ctx))
    out.extend(render_s8(ctx))
    out.extend(render_s9(ctx))
    out.extend(render_s10(ctx))
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


    out.extend(render_appendix(ctx))
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
