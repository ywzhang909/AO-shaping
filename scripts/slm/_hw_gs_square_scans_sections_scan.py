"""GS 方扫报告的**主扫描正文 §6-§10 + 附录**（从 facade 的 render_report 机械切段）。

与 `_hw_gs_square_scans_sections_core` 同一口径：函数体逐字节搬移，只注入
ctx 解包与 `add` 闭包；§11-§13 的胶合与正文收尾仍在 facade。
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
    DARK_FRAME_FRACTION,
    SATURATION_COUNTS,
    _idx_of,
    _loglog_exponent,
    math,
    monotonic_report,
)

from scripts.slm._hw_gs_square_scans_artifacts import (
    PATHS,
    _artifact_manifest,
    _atomic_artifact_manifest,
    _retest_artifact_manifest,
)

from scripts.slm._hw_gs_square_scans_atomic import (
    Table,
    cell_of,
    derived,
)

from scripts.slm._hw_gs_square_scans_fmt import (
    _figname,
    _pct,
)

from scripts.slm._hw_gs_square_scans_model import (
    ReportCtx,
)



def render_s6(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    tables = ctx.tables

    blaze = ctx.blaze

    marks_b = ctx.marks_b



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



    ctx.i40 = i40

    return out





def render_s7(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    f = ctx.f

    tables = ctx.tables

    sides = ctx.sides



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



    return out





def render_s8(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    cfg = ctx.cfg

    h = ctx.h

    w = ctx.w

    zo_x = ctx.zo_x

    zo_y = ctx.zo_y



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



    return out





def render_s9(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    run = ctx.run

    f = ctx.f

    retest = ctx.retest

    rf = ctx.rf

    sides = ctx.sides

    zern = ctx.zern

    blaze = ctx.blaze

    spot = ctx.spot

    i18 = ctx.i18

    i40 = ctx.i40

    policy = ctx.policy



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



    return out





def render_s10(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    sides = ctx.sides

    zern = ctx.zern

    panel = ctx.panel



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



    return out





def render_appendix(ctx: ReportCtx) -> list[str]:

    out: list[str] = []



    def add(text: str = "") -> None:

        out.append(text)



    retest = ctx.retest

    prior = ctx.prior

    atomic = ctx.atomic

    tables = ctx.tables



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



    return out


