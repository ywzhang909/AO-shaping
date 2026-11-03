"""GS 方扫报告的**主扫描正文 §0-§5**（从 facade 的 render_report 机械切段）。

每节一个 `render_sN(ctx)`：函数体与原 render_report 对应段落**逐字节一致**，
新增的只有 ctx 解包与 `add` 闭包；跨段中间量经 ctx 中转。拆分的验收判据是
同一输入渲染出的报告 md 与拆前基线逐字节一致。
"""

from __future__ import annotations



import sys

from pathlib import Path



# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致。
# 用**逐个 insert** 而不是 for 循环：scripts/ 那层的守卫
# (tests/ao_shaping/scripts/test_script_file_form_imports.py) 靠解析
# `sys.path.insert` 的字面参数判断「这个脚本能不能以文件方式启动」，
# for 循环里的循环变量它解析不了（会整文件跳过，等于没检查）。
# src 在 scripts 之前：scripts/ml 会遮蔽 src/ml。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))



from scripts.slm._hw_gs_square_scans_analysis import (
    Any,
    SATURATION_COUNTS,
    _idx_of,
    _neighbour_deviation,
)

from scripts.slm._hw_gs_square_scans_artifacts import (
    STAGES,
    STAGE_CN,
)

from scripts.slm._hw_gs_square_scans_atomic import (
    Cell,
    Table,
    cell_of,
    derived,
)

from scripts.slm._hw_gs_square_scans_figures import (
    _row_at,
    flag_rows,
)

from scripts.slm._hw_gs_square_scans_fmt import (
    _count_stage,
    _figname,
    _join_x,
    _pct,
)

from scripts.slm._hw_gs_square_scans_metrics import (
    saturation_flag,
)

from scripts.slm._hw_gs_square_scans_model import (
    FLAG_LEGEND,
    ReportCtx,
)



def render_s0(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    retest = ctx.retest

    rf = ctx.rf

    tables = ctx.tables

    sides = ctx.sides

    zern = ctx.zern

    blaze = ctx.blaze

    zo_y = ctx.zo_y



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



    return out





def render_s1(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    tables = ctx.tables

    sides = ctx.sides

    cfg = ctx.cfg

    bench = ctx.bench

    h = ctx.h

    w = ctx.w

    zo_x = ctx.zo_x

    zo_y = ctx.zo_y



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



    ctx.policy = policy

    return out





def render_s2(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    tables = ctx.tables

    policy = ctx.policy



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



    return out





def render_s3(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    f = ctx.f

    tables = ctx.tables

    sides = ctx.sides

    marks_s = ctx.marks_s



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



    return out





def render_s4(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    sides = ctx.sides

    zern = ctx.zern

    blaze = ctx.blaze

    spot = ctx.spot



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



    return out





def render_s5(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    tables = ctx.tables

    zern = ctx.zern

    marks_z = ctx.marks_z



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



    ctx.i18 = i18

    ctx.panel = panel

    return out


