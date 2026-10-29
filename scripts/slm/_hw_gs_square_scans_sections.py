"""GS 方扫报告的**复测段与远场段正文**。

:func:`render_retest_section` 写 §11（功率变化后的复测）、
:func:`render_farfield_section` 写 §12（远场拼图规划）。

这两节的共同点是**它们读的都不是主扫描那一批**：复测是另一个功率 regime，
远场是另一组帧。把它们放在主扫描正文之外，主扫描那一坨才不会越写越长。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence


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

from scripts.slm._hw_gs_square_scans_fmt import (
    _cell_short,
    _figname,
    _idx_of,
    _rel,
    _row_at,
    cell_of,
    derived,
)
from scripts.slm._hw_gs_square_scans_metrics import (
    _neighbour_deviation,
    column_contract,
)
from scripts.slm._hw_gs_square_scans_model import (
    CROP_FACTOR,
    Cell,
    DARK_FRAME_FRACTION,
    FIELD_BASELINE_FLAT,
    FIELD_RETEST_FLAT,
    Findings,
    MOSAICS,
    MOSAIC_CN,
    RETEST_STAGES,
    RETEST_STAGE_CN,
    RetestFindings,
    RetestRun,
    Run,
    SATURATION_COUNTS,
    STAGE_CN,
    Table,
    VARIABLE,
)

def render_retest_section(
    run: Run, f: Findings, retest: RetestRun, rf: RetestFindings
) -> tuple[list[str], list[Table]]:
    """§11 复测分节。数字全部从复测 CSV / metrics.json 现读, 现场记录单独标注。"""
    tables: list[Table] = []
    out: list[str] = []
    base_blaze = run.stages["blaze"]
    base_zern = run.stages["zernike"]
    r_blaze = retest.stages["blaze"]
    r_zern = retest.stages["zernike"]

    def add(text: str = "") -> None:
        out.append(text)

    add("## 11. 复测（功率已变）")
    add()
    add(
        "本节是**第二批产物**：`"
        f"{_rel(retest.root)}/`。基线之后激光功率变了（见 §11.1），所以把需要复查的配置"
        "用**原子化调用**在**同一脚本版本**下重跑了一遍：每个配置一个独立进程"
        f"（`mode = {retest.mode}`）、同一 {rf.retest_exposure_ms:g} ms 曝光、同一 "
        f"`--side-px {r_blaze.rows[0]['selected_side_px']}`，共 {rf.retest_rows} 行"
        f"（光栅 {'/'.join(f'{cfg.value:.0f}' for cfg in r_blaze.configs)}、"
        f"Zernike {'/'.join(f'{cfg.value:.0f}' for cfg in r_zern.configs)}）。"
    )
    add()
    add(
        "复测批只有两个用途：(a) 判定基线的两处异常是物理效应还是采集事故；"
        "(b) 在同一批内做**邻居比较**。它**不能**与基线做绝对值比较 —— 理由见 §11.1。"
    )
    add()
    add(
        "**本节与前面所有章节是两套数字，不能混读。** 复测的每一行都由生成器从复测的 "
        "`measurements.csv` 现读（并与同配置的 `metrics.json` 逐值交叉校验），"
        "所以脚本覆盖重跑这 "
        f"{rf.retest_rows} 个配置后，只需重跑一次生成器即可刷新本节 —— "
        "生成器不缓存、不内联任何复测数值。"
    )
    if rf.prior_root is not None:
        add()
        add(
            f"> 脚注：更早一版复测（`{_rel(rf.prior_root)}`）跨了两个脚本版本，**已作废**，"
            "仅作证据留存；本报告不引用它的任何数字。"
        )
    add()

    # -- 11.1 ---------------------------------------------------------------
    add("### 11.1 口径：为什么本节只做批内比较")
    add()
    regime = Table(
        ("项", "基线", "复测"),
        [
            ("产物根", f"`{_rel(run.root)}`", f"`{_rel(retest.root)}`"),
            ("段", "sides / zernike / blaze",
             " / ".join(RETEST_STAGE_CN[s] for s in RETEST_STAGES) + "（**无 sides**）"),
            ("行数", derived(float(sum(len(s.rows) for s in run.stages.values())), 0),
             derived(float(rf.retest_rows), 0)),
            ("生效曝光 (ms)", derived(rf.baseline_exposure_ms, 1),
             derived(rf.retest_exposure_ms, 1)),
            ("会话上限 (ms)", derived(run.exposure_ceiling_ms, 1),
             derived(retest.exposure_ceiling_ms, 1)),
            ("调用方式", "一次进程连续扫三段", f"{retest.mode}（每配置一个进程）"),
            ("目标边长来源", f"第 1 段按 score 选定 {f.selected_side:.0f} px",
             f"`--side-px` 显式覆盖（`selected_source = {retest.selected_source}`）"),
        ],
    )
    tables.append(regime)
    add(regime.render())
    add()
    add(
        f"两批的**工作点差 {rf.retest_exposure_ms / rf.baseline_exposure_ms:.0f}×**"
        f"（{rf.baseline_exposure_ms:g} ms → {rf.retest_exposure_ms:g} ms），"
        "因为激光功率在基线之后变了。功率量级用下面这张表表达 —— "
        "**它是现场记录，不在这两批产物里**，所以单独一张表、单独一列来源："
    )
    add()
    power_rows: list[tuple[Cell | str, ...]] = []
    for record in (FIELD_BASELINE_FLAT, FIELD_RETEST_FLAT):
        anchor = record.anchor()
        for exposure, peak in record.readings:
            power_rows.append((
                record.label,
                derived(exposure, 2),
                derived(peak, 1),
                derived(peak / exposure, 1,
                        "  ← 定斜率" if (exposure, peak) == anchor else ""),
                record.source if (exposure, peak) == anchor else "",
            ))
    power_rows.append((
        "**功率比**",
        "—",
        "—",
        derived(rf.power_ratio, 2, "×"),
        f"{rf.baseline_counts_per_ms:.1f} / {rf.retest_counts_per_ms:.1f} counts·ms⁻¹",
    ))
    power_tbl = Table(("现场记录（**非本批产物**）", "曝光 (ms)", "峰值 (counts)",
                       "counts/ms", "来源"), power_rows)
    tables.append(power_tbl)
    add(power_tbl.render())
    add()
    add(
        f"读法：基线同期 0.5 ms 时 0 级峰值 {FIELD_BASELINE_FLAT.anchor()[1]:g} counts "
        f"（{rf.baseline_counts_per_ms:.1f} counts/ms），复测前 5 ms 时 "
        f"{FIELD_RETEST_FLAT.anchor()[1]:g} counts（{rf.retest_counts_per_ms:.1f} counts/ms）"
        f"⇒ 同一曝光下复测期**弱 {rf.power_ratio:.2f}×**。"
        f"所以复测用 {rf.retest_exposure_ms:g} ms 补偿，折合基线的 "
        f"{rf.equivalent_exposure_ms:.1f} ms —— 与基线 {rf.baseline_exposure_ms:g} ms "
        "几乎相同。**这条折算只说明「亮度口径大致相当」，不等于两批可比**："
    )
    add()
    add(
        "- 斜率只取每段记录里**曝光最大**的那个点。低曝光读数被噪声地板抬高"
        f"（0.05 ms 只有 {FIELD_RETEST_FLAT.readings[0][1]:g} counts），拿它定斜率会把功率比算歪。"
        "同一段记录里 1 ms 与 5 ms 的 counts/ms 相差 "
        f"{abs(FIELD_RETEST_FLAT.readings[2][1] / FIELD_RETEST_FLAT.readings[2][0] - rf.retest_counts_per_ms) / rf.retest_counts_per_ms:.0%}，"
        "所以这条线性外推的可信度就是**约一成**，不要读得更细。"
    )
    add(
        f"- 两批的**成像状态**本身不同：`global_spot_fwhm` 基线 {min(v for v in base_blaze.col('global_spot_fwhm_px') if math.isfinite(v)):.1f}–"
        f"{max(base_blaze.col('global_spot_fwhm_px')):.1f} px、复测 "
        f"{min(r_blaze.col('global_spot_fwhm_px')):.1f}–{max(r_blaze.col('global_spot_fwhm_px')):.1f} px；"
        f"CV 也系统性偏低（光栅 {min(rf.blaze_cv):.2f}–{max(rf.blaze_cv):.2f} vs 基线 "
        f"{min(base_blaze.col('cv')):.2f}–{max(base_blaze.col('cv')):.2f}）。"
        "**「复测 CV 更低」不能读成「复测更好」**，它只是另一个成像状态。"
    )
    add(
        "- 因此本节**只做批内邻居比较**：同一批内、相邻配置之间「谁高谁低」才是可判定的；"
        "跨批的绝对值一律不比。"
    )
    add()

    # -- 11.2 ---------------------------------------------------------------
    add("### 11.2 复测是怎么跑的（原子化 + 显式边长）")
    add()
    add(
        f"每个配置都是独立进程（`mode = {retest.mode}`），边长由 `--side-px "
        f"{r_blaze.rows[0]['selected_side_px']}` 显式给（`selected_source = "
        f"{retest.selected_source}`），因此**每个配置都会重新定位一次 0 级**——"
        "这一点在 §11.6 会变成关键。基线则是「定位一次、冻结 18 行」。"
    )
    add()
    cfg_rows: list[tuple[Cell | str, ...]] = []
    for stage_name in RETEST_STAGES:
        for cfg in retest.stages[stage_name].configs:
            zx, zy = cfg.zero_order
            settle_text = (
                "；".join(ev.describe() for ev in cfg.settle) if cfg.settle else "—"
            )
            cfg_rows.append((
                RETEST_STAGE_CN[stage_name],
                cfg.config,
                cell_of(cfg.row, VARIABLE[stage_name], 0),
                derived(zx, 0),
                derived(zy, 0),
                cell_of(cfg.row, "exposure_ms_readback", 1),
                "无可用局域光斑" if cfg.spot_flagged else "光斑可用",
                derived(float(cfg.spot.get("fwhm_px", float("nan"))), 1),
                settle_text,
            ))
    cfg_tbl = Table(
        ("段", "配置", "自变量", "0 级 x", "0 级 y", "曝光读回 (ms)",
         "平场评估", "平场 FWHM (px)", "settle 告警"),
        cfg_rows,
    )
    tables.append(cfg_tbl)
    add(cfg_tbl.render())
    add()
    add(
        f"两件事先记下：(1) 全部 {rf.retest_rows} 行的 `frame_peak` 都顶到满量程，"
        f"平场评估的 FWHM 是 "
        f"{min(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f}–"
        f"{max(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f} px —— "
        "0 级是一个**饱和平台**，`argmax` 在平台上的位置本身就不稳；"
        f"(2) {len(rf.flagged_configs)} 个配置（{', '.join(rf.flagged_configs)}）的平场评估"
        "直接报了「光路有光但没有可评分的局域光斑」，"
        f"{len(rf.settle_configs)} 个配置有 settle 告警（{', '.join(rf.settle_configs)}），"
        f"但**没有一条**探测峰值是黑帧（低于满量程 {DARK_FRAME_FRACTION:.0%} 的配置："
        f"{', '.join(rf.dark_probe_configs) if rf.dark_probe_configs else '无'}）—— "
        "与基线「超时 ≠ 帧坏」的读法一致（§2.3）。"
    )
    add()

    # -- 11.3 ---------------------------------------------------------------
    add("### 11.3 光栅 32 / 40 / 48：基线与复测并排")
    add()
    add(
        "两列数字来自两批各自的 `measurements.csv`；`—` 表示该批没有这个周期。"
        "「基线 40」那一行是 §6.3 认定的黑帧（`frame_peak = "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 40.0)['frame_peak']):.1f}`），"
        "它的 CV / PIB / score 全部不可用。"
    )
    add()
    blaze_rows: list[tuple[Cell | str, ...]] = []
    for value in [float(r["blaze_period_px"]) for r in r_blaze.rows]:
        for label, row in (("基线", _row_at(base_blaze.rows, "blaze_period_px", value)),
                           ("复测", _row_at(r_blaze.rows, "blaze_period_px", value))):
            note = ""
            if label == "基线" and float(row["frame_peak"]) < DARK_FRAME_FRACTION * base_blaze.median("frame_peak"):
                note = f"**黑帧（frame_peak = {float(row['frame_peak']):.1f}）**"
            elif label == "复测":
                note = "批内邻居"
            blaze_rows.append((
                derived(value, 0),
                label,
                cell_of(row, "cv", 4),
                cell_of(row, "pib", 4),
                cell_of(row, "score", 4),
                cell_of(row, "frame_peak", 2),
                cell_of(row, "box_peak", 2),
                cell_of(row, "box_clipped_px", 0),
                cell_of(row, "box_centroid_offset_x", 2),
                cell_of(row, "box_centroid_offset_y", 2),
                note or "—",
            ))
    blaze_tbl = Table(
        ("周期 (px)", "批", "CV", "PIB", "score", "frame_peak", "box_peak",
         "出画面积 (px)", "偏移 x (px)", "偏移 y (px)", "备注"),
        blaze_rows,
    )
    tables.append(blaze_tbl)
    add(blaze_tbl.render())
    add()

    # -- 11.4 ---------------------------------------------------------------
    add("### 11.4 Zernike 17 / 18 / 19：基线与复测并排")
    add()
    add(
        "基线 18 是 §5.2 的黑帧（`frame_peak = "
        f"{float(_row_at(base_zern.rows, 'n_max', 18.0)['frame_peak']):.1f}`），"
        "基线 17 是 settle 超时行。**解释方差与残差 RMS 在两批里逐位相同**"
        f"（n_max = {', '.join(f'{v:.0f}' for v in rf.fit_bit_identical)} 全部逐位一致）—— "
        "同一 GS 相位 + 同一拟合配置 ⇒ 拟合量是确定性的，所以两批的差别只能来自**测量**。"
    )
    add()
    zern_rows: list[tuple[Cell | str, ...]] = []
    for value in [float(r["n_max"]) for r in r_zern.rows]:
        for label, row in (("基线", _row_at(base_zern.rows, "n_max", value)),
                           ("复测", _row_at(r_zern.rows, "n_max", value))):
            note = ""
            if label == "基线":
                if float(row["frame_peak"]) < DARK_FRAME_FRACTION * base_zern.median("frame_peak"):
                    note = f"**黑帧（frame_peak = {float(row['frame_peak']):.1f}）**"
                elif run.settle_for("zernike", _idx_of(base_zern.x, value)) is not None:
                    note = "settle 超时"
            else:
                note = "批内邻居"
            zern_rows.append((
                derived(value, 0),
                label,
                cell_of(row, "cv", 4),
                cell_of(row, "pib", 4),
                cell_of(row, "score", 4),
                cell_of(row, "frame_peak", 2),
                cell_of(row, "zernike_explained_fraction", 6),
                cell_of(row, "zernike_rms_residual_rad", 6),
                cell_of(row, "box_centroid_offset_x", 2),
                note or "—",
            ))
    zern_tbl = Table(
        ("n_max", "批", "CV", "PIB", "score", "frame_peak", "解释方差",
         "残差 RMS (rad)", "偏移 x (px)", "备注"),
        zern_rows,
    )
    tables.append(zern_tbl)
    add(zern_tbl.render())
    add()

    # -- 11.5 ---------------------------------------------------------------
    add("### 11.5 判定")
    add()
    b40 = _row_at(base_blaze.rows, "blaze_period_px", 40.0)
    b18 = _row_at(base_zern.rows, "n_max", 18.0)
    b17 = _row_at(base_zern.rows, "n_max", 17.0)
    r32 = _row_at(r_blaze.rows, "blaze_period_px", 32.0)
    r40 = _row_at(r_blaze.rows, "blaze_period_px", 40.0)
    r48 = _row_at(r_blaze.rows, "blaze_period_px", 48.0)
    rz17 = _row_at(r_zern.rows, "n_max", 17.0)
    rz18 = _row_at(r_zern.rows, "n_max", 18.0)
    rz19 = _row_at(r_zern.rows, "n_max", 19.0)
    add(
        f"**判定 1 — 基线 blaze 40 的异常（CV {float(b40['cv']):.2f}、score "
        f"{float(b40['score']):.4f}）：推翻。**"
        f"复测 CV 在 32/40/48 上是 {rf.blaze_cv[0]:.4f} / {rf.blaze_cv[1]:.4f} / "
        f"{rf.blaze_cv[2]:.4f}（随周期单调下降 = {rf.blaze_cv_monotone}），"
        f"40 相对左右邻居中位数只偏 **{rf.blaze40_dev:+.2%}**；score 是 "
        f"{float(r32['score']):.4f} / {float(r40['score']):.4f} / {float(r48['score']):.4f}"
        f"（单调上升）。**40 没有异常。**"
        f"基线同一位置偏 {rf.baseline_blaze40_dev:+.1%}（CV {float(b40['cv']):.2f} 对邻居 "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 32.0)['cv']):.2f} / "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 48.0)['cv']):.2f}），"
        "那是黑帧采集失败（`frame_peak = "
        f"{float(b40['frame_peak']):.1f}`、`global_spot_fwhm = nan`、settle 超时且探测峰值 2）"
        "留下的假极值 —— §6.3 的 provisional 可以按「已复测推翻」结案。"
    )
    add()
    add(
        f"**判定 2 — 基线 Zernike 18 的异常（CV {float(b18['cv']):.2f}、frame_peak "
        f"{float(b18['frame_peak']):.1f}、质心 x 从 +120 跳到 −26.8）：推翻。**"
        f"复测 CV 在 17/18/19 上是 {rf.zern_cv[0]:.4f} / {rf.zern_cv[1]:.4f} / "
        f"{rf.zern_cv[2]:.4f}，18 相对左右邻居只偏 **{rf.zern18_dev:+.2%}**；"
        f"复测 18 的 `frame_peak` 回到满量程、质心 x 偏移 "
        f"{float(rz18['box_centroid_offset_x']):+.2f} px —— 与 19 的 "
        f"{float(rz19['box_centroid_offset_x']):+.2f} px 只差 "
        f"{abs(float(rz18['box_centroid_offset_x']) / float(rz19['box_centroid_offset_x']) - 1):.1%}"
        f"（17 是 {float(rz17['box_centroid_offset_x']):+.2f} px），"
        f"基线那个 −26.80 px 的跳变没有复现。基线同一位置偏 {rf.baseline_zern18_dev:+.1%} —— 比复测批内"
        f"任何一点偏离邻居的量（≤ {rf.batch_dev_max:.1%}，见判定 6）大 "
        f"{abs(rf.baseline_zern18_dev) / rf.batch_dev_max:.0f} 倍。**18 阶本身没有问题**；"
        "基线那一行是黑帧。"
    )
    add()
    add(
        "**判定 3 — 我此前把基线 n_max = 17 的高端异常归因为「高阶级重构在面板上欠采样」："
        "未确认（降级为未证实假设）。**"
        f"复测 17 的 CV {rf.zern_cv[0]:.4f} 与 18/19 同量级（相对邻居偏 "
        f"{_neighbour_deviation(rf.zern_cv, 0):+.2%}），"
        "且复测 17 自己**也有** settle 告警（"
        f"{retest.stages['zernike'].configs[0].settle[0].describe() if retest.stages['zernike'].configs[0].settle else '无'}）"
        "却没有任何异常 ⇒ 「超时」本身不产生异常；基线的 17 更像一次**探测帧全黑**"
        f"（settle 探测峰值 2、最终帧 {float(b17['frame_peak']):.1f}）的失败采集。"
        "**要证实欠采样假设，需要的不是再测一个点，而是同一配置的多次重复**："
        "固定 `n_max = 17`、固定相位、连测 N ≥ 5 次，看 CV 的分布是否**系统性地**"
        "偏离 17/18/19 的趋势 —— 单点复测只能排除「单次采集失败」，无法确认机制。"
    )
    add()
    add(
        "**判定 4 — 位移：方向可用，标度与 1/P 形状都不可用。**"
        "⚠️ 先把口径钉死：**位移只有 `measured_shift_y_px` 这一列**（质心相对冻结 0 级）。"
        "`box_centroid_offset_y` **不是**位移 —— 它 = 实测质心 − **预测框心**，"
        "而框是跟着预测走的，所以它天然是个大负值（§11.6 给出恒等式，残差 "
        f"{rf.offset_identity_residual:.1e} px）；把它当位移读会得到「反号」的假结论。"
    )
    add()
    add(
        f"- **方向同号**：P = 32 处基线 {float(_row_at(base_blaze.rows, 'blaze_period_px', 32.0)['measured_shift_y_px']):+.2f} px、"
        f"复测 {float(r32['measured_shift_y_px']):+.2f} px；P = 48 处基线 "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 48.0)['measured_shift_y_px']):+.2f} px、"
        f"复测 {float(r48['measured_shift_y_px']):+.2f} px —— 两批同号、同量级。"
    )
    add(
        f"- **标度不可用**：实测/预测 = 复测 {min(rf.ratio_retest):.3f}–{max(rf.ratio_retest):.3f}、"
        f"基线 {min(rf.ratio_baseline):.3f}–{max(rf.ratio_baseline):.3f}（基线排除黑帧行）。"
        "若只是「标度系数错」，这个比值应当**与周期无关**；"
        f"实际它随周期上升（{len(rf.ratio_baseline)} 个点里 "
        f"{sum(1 for i in range(1, len(rf.ratio_baseline)) if rf.ratio_baseline[i] > rf.ratio_baseline[i - 1])} 步上升）"
        "⇒ 形状也不对。"
    )
    add(
        f"- **1/P 形状被强烈压缩**：基线 P {rf.measured_ends_baseline[0]:.0f} → "
        f"{rf.predicted_ends_baseline[0]:.0f}（{rf.predicted_ends_baseline[0] / rf.measured_ends_baseline[0]:.1f}×）时，"
        f"预测 {rf.predicted_ends_baseline[1]:.2f} → {rf.predicted_ends_baseline[2]:.2f} px"
        f"（{rf.predicted_ends_baseline[1] / rf.predicted_ends_baseline[2]:.1f}×，构造上恰好 ∝1/P），"
        f"实测只有 {rf.measured_ends_baseline[1]:.2f} → {rf.measured_ends_baseline[2]:.2f} px"
        f"（{rf.measured_ends_baseline[1] / rf.measured_ends_baseline[2]:.1f}×）。"
        f"两端点 log-log 斜率：预测 α = {rf.alpha_predicted:.2f}，基线实测 α = "
        f"{rf.alpha_baseline:.2f}；复测只有 {rf.measured_ends_retest[0]:.0f} → "
        f"{rf.predicted_ends_retest[0]:.0f}（{rf.predicted_ends_retest[0] / rf.measured_ends_retest[0]:.2f}×）"
        f"三点，实测 α = {rf.alpha_retest:.2f} —— 跨度太短，只能说明「明显不等于 1」。"
    )
    add(
        "- **直接后果**：周期越小，框偏得越远。复测里框心相对质心偏 "
        + " / ".join(f"{off:+.1f}" for _, off in rf.offset_y_small_period)
        + " px（" + " / ".join(f"P={p:.0f}" for p, _ in rf.offset_y_small_period)
        + "）—— 与基线同量级（§6.4）。"
    )
    add()
    add(
        "**判定 5 — 「方框跟着 `predicted_shift_px` 走」这条做法必须改：成立。**"
        f"复测 {len(r_blaze.rows)} 行上恒等式 `偏移 y ≡ measured_shift_y − predicted_shift_y` "
        f"成立到 {rf.offset_identity_residual:.1e} px，所以框的位置**完全由预测决定**："
        "预测偏大 → 框偏 → 框内质心被框边拉回 → 量出来的位移更小。这是循环依赖，"
        "用这种方式永远验不了 `predicted_shift`。两个可选改法："
    )
    add()
    add(
        "1. **框固定 + 宽 ROI**：框心 = 冻结 0 级（不跟随预测），或直接用 ≥ 半幅的 ROI "
        "对**整帧质心**求位移。这是唯一能独立测位移的做法。"
    )
    add(
        "2. **重标定**：若一定要框跟随，先用改法 1 的实测把 `predicted_shift` 的**标度与 "
        "1/P 指数**拟合出来（本轮实测 α ≈ "
        f"{rf.alpha_baseline:.2f}，而不是 1.00），再拿标定后的式子摆放方框。"
    )
    add()
    add(
        "**判定 6 — 这两处极值不可能是「趋势上的点」：成立（单批即可判定）。**"
        f"复测批内 6 个配置相对左右邻居中位数的偏差最大 **{rf.batch_dev_max:.1%}**"
        f"（这是端点的趋势斜率；两个**中间点**分别只有 {rf.blaze40_dev:+.2%} 与 "
        f"{rf.zern18_dev:+.2%}）；而基线同一统计量是 "
        f"{rf.baseline_blaze40_dev:+.0%} 与 {rf.baseline_zern18_dev:+.0%}，"
        f"相差 {abs(rf.baseline_blaze40_dev) / rf.batch_dev_max:.0f} 倍与 "
        f"{abs(rf.baseline_zern18_dev) / rf.batch_dev_max:.0f} 倍。"
        "也就是说：**同一条平滑趋势上不可能长出这两个值** —— 这个判定只用复测批自己的"
        "内部一致性，不依赖任何跨批重复性，也不依赖基线的任何假设。"
    )
    add()

    # -- 11.6 ---------------------------------------------------------------
    add("### 11.6 0 级每次重定位都会抖，而光斑几乎不动")
    add()
    add(
        "复测是原子化的，所以每个配置都重新在平场帧上 `argmax` 定位 0 级。"
        "把三次独立定位与三次光斑质心摆在一起（全部是光栅段）："
    )
    add()
    shift_rows: list[tuple[Cell | str, ...]] = []
    for cfg in r_blaze.configs:
        zx, zy = cfg.zero_order
        offset = float(cfg.row["box_centroid_offset_y"])
        measured = float(cfg.row["measured_shift_y_px"])
        predicted = float(cfg.row["predicted_shift_px"])
        shift_rows.append((
            derived(cfg.value, 0),
            derived(zx, 0),
            derived(zy, 0),
            cell_of(cfg.row, "box_centroid_x", 2),
            cell_of(cfg.row, "box_centroid_y", 2),
            derived(predicted, 2),
            derived(measured, 2),
            derived(offset, 2),
            derived(offset - (measured - predicted), 6),
        ))
    shift_tbl = Table(
        ("周期 (px)", "0 级 x", "0 级 y", "质心 x", "质心 y", "predicted y (px)",
         "measured y (px)", "偏移 y (px)", "偏移 − (measured − predicted)"),
        shift_rows,
    )
    tables.append(shift_tbl)
    add(shift_tbl.render())
    add()
    add(
        f"三次独立定位的 0 级 x 相差 **{rf.zero_order_x_span:.0f} px**、y 相差 "
        f"**{rf.zero_order_y_span:.0f} px**；而**光斑质心**只差 "
        f"{rf.centroid_x_span:.2f} px / {rf.centroid_y_span:.2f} px。"
        f"于是 `measured_shift_x` 的 {rf.measured_x_span:.2f} px、`measured_shift_y` 的 "
        f"{rf.measured_y_span:.2f} px 波动里，**原点定位占了大头**，不是光斑在动。"
        "原因就在 §11.2 那张表：平场 0 级是**饱和平台**"
        f"（FWHM {min(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f}–"
        f"{max(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f} px、"
        f"`hollowness` 最高到 {max(float(c.spot.get('hollowness', 0.0)) for c in retest.all_configs):.2f}、"
        f"{len(rf.flagged_configs)} 个配置被平场评估直接判为「无可用局域光斑」），"
        "`argmax` 落在平台的哪一格没有物理意义。"
        f"（Zernike 段更明显：三次定位是 "
        + " / ".join(f"{cfg.zero_order[0]:.0f}" for cfg in r_zern.configs)
        + f" px（x 相差 {rf.zero_order_x_span_zern:.0f} px、y 相差 "
        f"{rf.zero_order_y_span_zern:.0f} px），比光栅段还大。）"
    )
    add()
    add("由此得到一条**判别性复测方案**（下一轮直接照做）：")
    add()
    add(
        "1. **先在低曝光定位 0 级**（0 级不饱和的那个曝光，例如基线用的 "
        f"{rf.baseline_exposure_ms:g} ms），把位置写进 `run_config` 冻结；再切到测量曝光。"
        "**不要**在饱和帧上 `argmax`。"
    )
    add(
        "2. **框固定**：框心 = 上一步冻结的 0 级，不跟随 `predicted_shift`；"
        "或者干脆用宽 ROI（≥ 画幅的一半）直接对**整帧质心**求位移。"
    )
    add(
        "3. **同一相位重复 ≥ 3 次**，报质心的分布而不是单值 —— 现在这个量级"
        f"（{rf.measured_y_span:.1f} px）与原点抖动（{rf.zero_order_x_span:.0f} px）同阶，"
        "单次读数分不开「光斑动了」与「原点动了」。"
    )
    add(
        "4. 有了上面三条，`predicted_shift_px` 的**符号与标度**才第一次可以被独立验证；"
        "在那之前，它只能当**摆放方框的初值**，不能当被验证的物理量。"
    )
    add()
    add(f"![复测邻居比较](figures/{_figname('fig5_retest_neighbors')})")
    add()
    add(
        "图 5 是**批内**邻居比较：灰色虚线是基线全段（含它的黑帧点），绿色实线是复测的三点。"
        "两条线的**绝对高度不要互相比较**（§11.1），只看各自段内的顺序与尖峰是否还在。"
    )
    add()

    # -- 11.7 ---------------------------------------------------------------
    add("### 11.7 列口径（两批的列集合不同）")
    add()
    add(
        f"复测的 CSV 列数是 {len(r_blaze.headers)} / {len(r_zern.headers)}"
        f"（光栅 / Zernike），基线是 {len(base_blaze.headers)} / {len(base_zern.headers)}。"
        "本节用到的列逐个核对如下 —— 复测独有的是 `config` / `stage` / `selected_source` / "
        "`base_phase_source` / `fit_source` 这类**运行元数据**，不是指标："
    )
    add()
    needed_blaze = ("blaze_period_px", "cv", "pib", "score", "frame_peak", "box_peak",
                    "box_clipped_px", "box_centroid_offset_x", "box_centroid_offset_y",
                    "measured_shift_y_px", "predicted_shift_px", "exposure_ms_readback")
    needed_zern = ("n_max", "cv", "pib", "score", "frame_peak", "box_peak",
                   "zernike_explained_fraction", "zernike_rms_residual_rad",
                   "box_centroid_offset_x", "box_centroid_offset_y",
                   "exposure_ms_readback")
    contract_rows = [(f"光栅 · {a}", b, c) for a, b, c in
                     column_contract(base_blaze, r_blaze, needed_blaze)]
    contract_rows += [(f"Zernike · {a}", b, c) for a, b, c in
                      column_contract(base_zern, r_zern, needed_zern)]
    contract_tbl = Table(("列", "基线有", "复测有"), contract_rows)
    tables.append(contract_tbl)
    add(contract_tbl.render())
    add()
    missing = [row for row in contract_rows if "**缺**" in (row[1], row[2])]
    add(
        "全部所需列在两批里都存在（缺列数 = "
        f"{len(missing)}）。生成器不假设两批表头相同：它按列名取，取不到就报错退出。"
    )
    add()
    add(
        "**两处容易读错的口径**："
    )
    add()
    add(
        "- **段级 `run_config.json` 的 `zero_order_xy_frozen` 不是每配置的框心。** "
        + "；".join(
            f"{RETEST_STAGE_CN[name]}段级记 "
            f"({float(st.run_config['zero_order_xy_frozen'][0]):.0f}, "
            f"{float(st.run_config['zero_order_xy_frozen'][1]):.0f})，而三个配置各自是 "
            + " / ".join(
                f"({cfg.zero_order[0]:.0f}, {cfg.zero_order[1]:.0f})" for cfg in st.configs)
            for name, st in retest.stages.items()
            if st.run_config.get("zero_order_xy_frozen")
        )
        + "。本节一律用**每配置 `metrics.json` 的 `preflight`** 值 —— 那才是真正摆放方框的那个数。"
    )
    add(
        "- **`fit_seconds` 不是整段耗时**：它是**本进程这一次**拟合的墙钟"
        "（复测里 "
        + " / ".join(
            f"{float(r['fit_seconds']):.2f} s" for r in r_zern.rows
        )
        + "），复测是每配置一个进程，所以它既不累加也不代表阶段时长。本节不用它做任何时长判断。"
        if "fit_seconds" in r_zern.headers
        else "- 复测的 Zernike CSV 没有 `fit_seconds` 列。"
    )
    add()
    return out, tables

def render_farfield_section(
    run: Run, payload: dict[str, Any], figures: Sequence[Path],
    *, residual_pointer: bool = False,
) -> tuple[list[str], list[Table]]:
    """§12 CCD 远场光斑图。数字与配置全部来自 ``payload`` (它由 CSV 与帧路径算出)。

    ``residual_pointer`` 为真时在开头加一句指引 —— 指向 §13.8 的
    「Zernike 残差 × 同配置远场」图。那个图依赖相位文件, 只有整轮原子重扫那一批
    落了 `pupil_phase.npy`, 所以本节自己画不出来。
    """
    tables: list[Table] = []
    out: list[str] = []

    def add(text: str = "") -> None:
        out.append(text)

    add("## 12. CCD 远场光斑图")
    add()
    add(
        "本节把每段**实测远场帧**直接摆出来 —— 不是指标曲线，是相机画面。"
        "帧来自各配置落盘的 `frame_*.npz`（`np.load(...)[\"frame\"]`，"
        "`float32`、画幅与 `run_config.frame_shape_hw` 一致、0–255 counts），"
        "每格是**该配置落盘帧的算术平均**（右下角标 `×N 帧均值`）。"
    )
    add()
    add(
        "生成器在画图前逐格核对两件事：帧的最大值必须等于该行 CSV 的 `frame_peak`，"
        "npz 里那个 0 维 `params`（JSON）的 `cv`/`pib`/`score`/`frame_peak` 必须等于该行 CSV。"
        "两者任一不符就报错退出 —— 否则拼图会张冠李戴。"
    )
    add()
    if residual_pointer:
        add(
            "本节只摆**远场画面**。把 **Zernike 截断的相位残差**与同配置远场并排的那张图"
            "在 §13.8 —— 它需要每个配置的 `pupil_phase.npy`，而只有整轮原子重扫那一批落了"
            "相位文件（基线批没有）。"
        )
        add()

    add("### 12.1 显示了哪些配置")
    add()
    rows: list[tuple[Cell | str, ...]] = []
    missing_cells: list[str] = []
    for mosaic in payload["mosaics"]:
        for cell in mosaic["cells"]:
            stage = run.stages[cell["stage"]]
            row = _row_at(stage.rows, VARIABLE[cell["stage"]], cell["value"][VARIABLE[cell["stage"]]])
            source = ("—" if not cell["sources"] else
                      (_rel(Path(cell["sources"][0]["path"]).parent)
                       if cell["layout"] == "atomic"
                       else _rel(Path(cell["sources"][0]["path"]))))
            frames = (f"{cell['frames_present']} / {cell['frames_planned']}"
                      if cell["frames_planned"] else f"{cell['frames_present']} / —")
            note = cell["note"] or ("缺帧" if cell["missing"] else "")
            if cell["missing"]:
                missing_cells.append(f"{cell['stage']}/{cell['config']}")
            median = cell.get("box_median")
            rows.append((
                MOSAIC_CN.get(mosaic["mosaic"], mosaic["mosaic"]),
                STAGE_CN[cell["stage"]],
                cell["config"],
                _cell_short(cell),
                f"`{source}`",
                frames,
                derived(cell["peak"], 2),
                "饱和" if cell["saturated"] else "未饱和",
                cell_of(row, "cv", 4),
                cell_of(row, "pib", 4),
                (derived(median, 2) if median is not None else "—"),
                note or "—",
            ))
    table = Table(
        ("拼图", "段", "配置", "取值", "源", "帧数 (实际 / 计划)", "峰值 (counts)",
         "峰值状态", "CV", "PIB", "方框中位电平 (counts)", "备注"),
        rows,
    )
    tables.append(table)
    add(table.render())
    add()
    planned_typical = sorted({c["frames_planned"] for m in payload["mosaics"]
                              for c in m["cells"] if c["frames_planned"]})
    add(
        "**帧数口径**：「实际 / 计划」= 该配置**落盘的 npz 个数** / `run_config."
        "frames_per_measurement`。"
        + (f"本轮计划值是 {'/'.join(str(v) for v in planned_typical)}，"
           "而落盘的是**已经做过多帧平均的那一张**（每个 npz 内部已含若干次曝光），"
           "所以「实际 1 / 计划 4」不代表少了 3 帧，而是**平均后只存了一张**；"
           "生成器按实际存在的 npz 求平均，不补帧、不报错，并把两个数都写进 "
           "`frames_selection.json`。" if planned_typical else "")
        + (f"**缺帧的格子**：{', '.join(missing_cells)}（表里已标 `缺帧`，图上不画）。"
           if missing_cells else "")
    )
    add()

    add("### 12.2 四张拼图")
    add()
    by_key = {m["mosaic"]: m for m in payload["mosaics"]}
    drawn = {path.name for path in figures}
    captions = {
        "sides": ("方斑随目标边长变化：小方框里方斑被压得最亮（100–200 px 甚至顶到满量程），"
                  "方框放大后方斑摊开、峰值下降。第 6 格是 score 最高行或它的回退（见表）。"),
        "zernike": ("低阶 Zernike 重构后的远场：与同框的 GS 参照（最后一格）比，"
                    "重构出来的不是方斑而是一个亮核 —— 这正是 §5.1 的结论在画面上的样子。"),
        "blaze": ("叠加闪耀光栅后，方框跟着**预测**位移走；16 px 那一行位移最大，"
                  "框已出画（格内左下角标出出画面积）。"),
        "fullframe": ("整帧不裁剪：0 级（argmax）与画面几何中心并不重合，"
                      "这就是 §8 里「最大边长必须现场量」的由来。"),
    }
    for spec in MOSAICS:
        mosaic = by_key.get(spec.key)
        if mosaic is None:
            continue
        name = mosaic["figure"]
        if name not in drawn:
            add(f"（拼图 `{name}` 没有可画的帧，已跳过。）")
            add()
            continue
        add(f"![{spec.title}](figures/{name})")
        add()
        frames_seen = sorted({c["frames_present"] for c in mosaic["cells"]
                              if not c["missing"]})
        sat = [_cell_short(c) for c in mosaic["cells"] if c["saturated"] and not c["missing"]]
        add(
            f"*{spec.title}。* 对数显示（`LogNorm(vmin=1, vmax=该格峰值)`，"
            "**颜色不代表线性强度**）；"
            + (f"裁剪：以 `box_center` 为中心、边长 `box_side × {CROP_FACTOR}` "
               "（已夹到画幅内）；" if spec.crop else "本图**不裁剪**（整帧）；")
            + "叠加：目标框（虚线）、预测框心（十字）、实测质心（圆点）"
            + ("、0 级 argmax（星）与画面中心（叉）" if not spec.crop else "")
            + "；右下角 `×N 帧均值`。"
            + (f" 饱和格：{', '.join(sat)}。" if sat else "")
            + captions.get(spec.key, "")
        )
        add()
        medians = [(_cell_short(c), c["box_median"]) for c in mosaic["cells"]
                   if not c["missing"] and c.get("box_median") is not None]
        if len(medians) >= 2:
            low = min(medians, key=lambda pair: pair[1])
            high = max(medians, key=lambda pair: pair[1])
            add(
                f"方框内**中位电平**：`{high[0]}` 是 {high[1]:.2f} counts，"
                f"`{low[0]}` 只有 {low[1]:.2f} counts（整列见 §12.1）。"
                + {
                    "sides": ("方框放大后方斑被摊薄到 **1–2 counts 的量化地板**，"
                              "在对数显示下就是一片黑 —— 那是数据本身的动态范围，不是画错；"
                              "读那些格的均匀度请回 CSV 的 CV / PIB。"),
                    "zernike": ("框内大部分像素是 0 counts、峰值却顶到满量程 ⇒ "
                                "光集中在一个亮核上，**这正是 CV 7.3 的画面形态**"
                                "（§5.1 的「PIB 假高分」）。"),
                    "blaze": ("框内中位只有 1 counts 上下、峰值饱和 ⇒ 框里是一个亮核"
                              "加几乎全暗的背景，方斑并没有跟着框走（§6.4）。"),
                }.get(spec.key, "")
            )
            add()

    add("### 12.3 显示口径")
    add()
    add(
        f"1. **裁剪**：拼图 1–3 以该行 CSV 的 `box_center_x/y` 为中心、边长取 "
        f"`box_side_px × {CROP_FACTOR}`，并夹到画幅内（夹过就在 "
        "`frames_selection.json` 里记 `clipped_by_frame`）；拼图 4 **不裁剪**。"
    )
    add(
        "2. **对数拉伸**：`matplotlib.colors.LogNorm(vmin=1, vmax=该格峰值)`。"
        "0 级与方斑的每像素亮度差 2–3 个数量级（§4.2），线性拉伸只会看到 0 级一个点，"
        "方斑的结构全部落在最暗的几档里；对数显示能把两者放进同一张图。"
        "**代价是颜色与强度不成线性** —— 要读绝对强度请回 CSV 的 `frame_peak` / `box_peak`，"
        "不要从颜色反推。"
    )
    add(
        "3. **叠加与饱和**：目标框（虚线）、预测框心（十字）、实测质心（圆点）；"
        "`box_clipped_px > 0` 的格子在左下角标 `框出画 … px`。"
        f"该行 `frame_peak ≥ {SATURATION_COUNTS:.0f}` 时标题末尾加「饱和」——"
        "意味着最亮像素已被 8 bit 削平，画面里那一片的颜色不随强度变化。"
    )
    add()
    return out, tables
