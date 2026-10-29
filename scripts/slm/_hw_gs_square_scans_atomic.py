"""GS 方扫报告的**整轮原子重扫段正文**（§13）。

这是唯一「单一脚本版本 + 单一功率」的完整一轮，所以三段数字第一次真正
可比 —— 也因此它自带一套独立的口径说明与二级分层判据。

单独成模块的原因：它是报告里唯一需要**先做两级分层再说话**的一节，逻辑
密度和 §1-§10 那一坨不是一个量级，混在一起两边都难读。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


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

from scripts.slm._hw_gs_square_scans_figbase import (
    batch_tag,
    residual_figure_name,
)
from scripts.slm._hw_gs_square_scans_fmt import (
    _rel,
    cell_of,
    derived,
)
from scripts.slm._hw_gs_square_scans_model import (
    AtomicFindings,
    CROP_FACTOR,
    Cell,
    Findings,
    MOSAICS,
    PATHS,
    RESIDUAL_RECON_ATOL,
    RESIDUAL_RMS_RTOL,
    ResidualSet,
    RetestRun,
    Run,
    STAGE_CN,
    Table,
)

def render_atomic_section(
    run: Run, f: Findings, atomic: RetestRun, af: AtomicFindings,
    retest: RetestRun | None, figures: Sequence[Path],
    residuals: ResidualSet | None = None,
) -> tuple[list[str], list[Table]]:
    """§13 整轮原子重扫。数字全部从该批 CSV / metrics.json 现读。"""
    tables: list[Table] = []
    out: list[str] = []
    sides = atomic.stages["sides"]
    zern = atomic.stages["zernike"]
    blaze = atomic.stages["blaze"]

    def add(text: str = "") -> None:
        out.append(text)

    add("## 13. 整轮原子重扫（单一版本 + 单一功率）")
    add()
    add(
        "第三批产物：`"
        f"{_rel(atomic.root)}/`。它是**唯一「单一脚本版本 + 单一功率」的完整一轮** —— "
        "三段（边长 / Zernike / 光栅）在同一个 `mode=atomic` 下逐配置重扫，"
        "所以三段数字之间第一次真正可比。"
    )
    add()

    add("### 13.1 口径")
    add()
    gate = Table(
        ("项", "本轮取值", "来源"),
        [
            ("生效曝光 (ms)", derived(af.exposure_ms, 1), "`run_config` / 每配置 `preflight`"),
            ("会话上限 (ms)", derived(af.ceiling_ms, 1), "`run_config`"),
            ("调用方式", f"`mode = {af.mode}`（每配置一个独立进程）", "`manifest` / `run_config`"),
            ("0 级（段级冻结）", f"({af.zero_order[0]:.0f}, {af.zero_order[1]:.0f})",
             "`run_config.zero_order_xy_frozen`"),
            ("可用最大边长 (px)", derived(af.side_max, 0), "`run_config.side_max_px`"),
            ("反解原始值 (px)", derived(af.side_max_raw, 0), "`run_config.side_max_raw_px`"),
            ("边长步长 (px)", derived(af.step, 0), "由 CSV 的 `side_px` 相邻差"),
            ("实际边长行数", derived(float(af.n_sides), 0), "`manifest.n_rows` / CSV"),
            ("选定边长 (px)", derived(af.selected_side, 0), "`sides/selected.json`"),
            ("选定时的 score", derived(af.selected_score, 4), "`sides/selected.json`"),
        ],
    )
    tables.append(gate)
    add(gate.render())
    add()
    add(
        "**1800 与 1900 被拒是保护生效，不是失败。** 按步长 "
        f"{af.step:.0f} px 往上，接下来的两档是 "
        f"{af.rejected[0]:.0f} / {af.rejected[1]:.0f} px，都超过本轮实测的可用最大边长 "
        f"{af.side_max:.0f} px（= 2·(min(cx, W−1−cx, cy, H−1−cy) − edge_margin) 向下取整，"
        f"原始 {af.side_max_raw:.0f} px）。脚本原文的拒绝理由："
    )
    add()
    add("> 边长 `{value}` px 超过本次实测的画面可用最大边长 "
        f"{af.side_max:.0f} px —— 方框会压在帧边上, PIB/EE 就不再是方斑质量了。"
        "改小 --side-px, 或调 --edge-margin-px / 曝光")
    add()
    add("（上面是脚本里的**消息模板**：`{value}` 处就是被拒的那一档边长 —— "
        f"本轮是 {af.rejected[0]:.0f} 与 {af.rejected[1]:.0f}。）")
    add()
    add(
        "这条保护的判据是**方框会不会压到帧边**（压边后窗函数与读出噪声会污染方框内的统计，"
        "PIB/EE 就不再是方斑质量的度量）。所以「少了两行」恰恰是它该做的事："
        f"宁可只有 {af.n_sides} 行可用数据，也不给两个框压边的假点。"
        f"（退出码 2 —— 该脚本把这条越限检查与读帧超时共用同一个硬退出码。）"
    )
    add()

    # -- 13.2 边长段 --------------------------------------------------------
    add("### 13.2 边长段")
    add()
    side_tbl = Table(
        ("边长 (px)", "CV", "PIB", "score", "box_peak", "frame_peak",
         "方框内面积 (px)", "出画面积 (px)", "AR", "质心偏移 x", "质心偏移 y"),
        [(cell_of(r, "side_px", 0), cell_of(r, "cv", 4), cell_of(r, "pib", 4),
          cell_of(r, "score", 4), cell_of(r, "box_peak", 2), cell_of(r, "frame_peak", 2),
          cell_of(r, "box_area_px", 0), cell_of(r, "box_clipped_px", 0),
          cell_of(r, "ar", 4), cell_of(r, "box_centroid_offset_x", 2),
          cell_of(r, "box_centroid_offset_y", 2))
         for r in sides.rows],
    )
    tables.append(side_tbl)
    add(side_tbl.render())
    add()
    plateau_txt = " / ".join(f"{p:.0f}（{s:.4f}）" for p, s in af.plateau)
    tail_txt = " / ".join(f"{p:.0f}（{s:.4f}）" for p, s in af.tail)
    add(
        f"**score 的平台在 {min(p for p, _ in af.plateau):.0f}–"
        f"{max(p for p, _ in af.plateau):.0f} px**：{plateau_txt}，两者只差 "
        f"{af.plateau_gap:.1e}。之后开始掉：{tail_txt}。"
        f"（argmax 在 {af.score_argmax_side:.0f} px。）"
        "也就是说这一轮的「最优边长」不是一个点，而是一段**平台**："
        f"平台宽 {max(p for p, _ in af.plateau) - min(p for p, _ in af.plateau):.0f} px，"
        f"而平台内的高度差只有 {af.plateau_gap:.1e} —— 在单次测量的分辨力内，"
        "峰落在平台哪一点**不可判定**（与 §3.3 基线那一轮同一性质）。"
    )
    add()
    add(
        f"**`box_peak` 与饱和**：全部 {af.n_sides} 行的 `frame_peak` 都是 "
        f"{'/'.join(f'{v:.0f}' for v in af.frame_peak_values)}（0 级削顶），"
        f"而 `box_peak` 只有 {af.box_peak_min:.2f}–{af.box_peak_max:.2f} —— "
        "**削顶的是框外的 0 级，不是方斑本身**（`box_peak < frame_peak` 就证明最亮像素在框外）。"
        "这与 §4 的「0 级饱和」是两回事：那一条说的是平场 0 级不含线性信息，"
        "这一条说的是方斑自己贴着满量程但还没到顶 —— 方斑的读数仍然可信。"
    )
    add()
    if af.ar_outliers:
        add(
            "**一个没解释的离群**：`ar`（方框内长宽比）在 "
            + "、".join(f"{p:.0f} px 处是 {v:.2f}" for p, v in af.ar_outliers)
            + f"，而同列中位数只有 {af.ar_median:.2f}。这一行的 CV/PIB/score 都在趋势上，"
            "所以不是整帧坏，问题只可能出在 AR 这个量本身。**机制未定**，"
            "不要据此删行；要确认需要看该行的方框内二值轮廓（`ar` 由"
            " `square_aspect_ratio` 从框内二值掩膜算出）。"
        )
        add()

    # -- 13.3 Zernike 段 ----------------------------------------------------
    add("### 13.3 Zernike 段")
    add()
    zern_tbl = Table(
        ("n_max", "模式数", "解释方差", "残差 RMS (rad)", "CV", "PIB", "score",
         "质心偏移 x", "质心偏移 y"),
        [(cell_of(r, "n_max", 0), cell_of(r, "zernike_modes_used", 0),
          cell_of(r, "zernike_explained_fraction", 6),
          cell_of(r, "zernike_rms_residual_rad", 6),
          cell_of(r, "cv", 4), cell_of(r, "pib", 4), cell_of(r, "score", 4),
          cell_of(r, "box_centroid_offset_x", 2), cell_of(r, "box_centroid_offset_y", 2))
         for r in zern.rows],
    )
    tables.append(zern_tbl)
    add(zern_tbl.render())
    add()
    add(
        f"**CV 全平**：{af.zern_cv_min:.4f}–{af.zern_cv_max:.4f}（相对跨度 "
        f"{(af.zern_cv_max - af.zern_cv_min) / af.zern_cv_min:.1%}），PIB "
        f"{af.zern_pib_min:.4f}–{af.zern_pib_max:.4f}，score "
        f"{min(float(r['score']) for r in zern.rows):.4f}–"
        f"{max(float(r['score']) for r in zern.rows):.4f}。"
        f"解释方差从 {af.zern_explained_min * 100:.3f}% 单调升到 "
        f"{af.zern_explained_max * 100:.3f}%（{af.zern_modes_max:.0f} 个模式）—— "
        "**阶数加了四倍，画面上几乎没变**，这就是 §5.1「基不适配」的量化版。"
    )
    add()
    if af.tier_split is not None:
        low = sorted(af.tier_low)
        high = sorted(af.tier_high)
        add(
            "**`box_centroid_offset_x` 的两档台阶**："
            f"低档 {min(v for _, v, _ in low):.2f}–{max(v for _, v, _ in low):.2f} px"
            f"（n_max = {'/'.join(f'{n:.0f}' for n, _, _ in low)}），"
            f"高档 {min(v for _, v, _ in high):.2f}–{max(v for _, v, _ in high):.2f} px"
            f"（n_max = {'/'.join(f'{n:.0f}' for n, _, _ in high)}）。"
            "**台阶的位置已经定位：它不在远场里**。把这一列拆开看 ——"
        )
        add()
        add(
            f"- `box_centroid_x`（实测质心）在全部 {len(af.tier_low) + len(af.tier_high)} 个配置上只差 "
            f"**{af.centroid_x_span:.1f} px**；"
        )
        add(
            f"- 两档的差别全在**参考原点**上：低档各配置的 0 级 x 是 "
            f"{min(z for _, _, z in low):.0f}–{max(z for _, _, z in low):.0f}，"
            f"高档是 {min(z for _, _, z in high):.0f}–{max(z for _, _, z in high):.0f}，"
            f"分界落在 x ≈ {af.tier_split:.0f}；"
        )
        add(
            "- 而 `offset_x ≡ centroid_x − 0 级 x`（定义如此），所以台阶 = **原点跳了约 "
            f"{min(z for _, _, z in high) - max(z for _, _, z in low):.0f} px**，不是光斑移动。"
        )
        add()
        add(
            f"注意 `n_max = 21` 落在**低档**（{min(v for n, v, _ in low if n == 21.0):.2f} px）"
            "—— 20 与 22 都在高档。所以这个台阶**不是 `n_max` 的函数**，"
            "而是「那一次进程把 0 级定到了哪一簇」的结果。"
        )
        add()
        add(
            "**为什么原点会跳，仍未定。** 本轮能确定的只有「跳的是参考原点」；"
            "候选机制是每配置独立在饱和平台上 `argmax`（§11.6 已记录同一现象："
            "同一段里 0 级定位跨进程相差 8–39 px），但本轮**无法**把它与别的可能分开。"
            "要定因需要固定 0 级后再复测一次（§11.6 的四条方案）。"
            "**在定因之前，不要把 `box_centroid_offset_x` 的任何台阶读成物理效应。**"
        )
        add()

    # -- 13.4 光栅段 --------------------------------------------------------
    add("### 13.4 闪耀光栅段")
    add()
    blaze_tbl = Table(
        ("周期 (px)", "预测位移 (px)", "实测位移 y (px)", "实测/预测", "CV", "PIB",
         "score", "出画面积 (px)"),
        [(cell_of(r, "blaze_period_px", 0), cell_of(r, "predicted_shift_px", 2),
          cell_of(r, "measured_shift_y_px", 2), derived(ratio, 4),
          cell_of(r, "cv", 4), cell_of(r, "pib", 4), cell_of(r, "score", 4),
          cell_of(r, "box_clipped_px", 0))
         for r, (_, _, _, ratio, _, _, _, _) in zip(blaze.rows, af.blaze)],
    )
    tables.append(blaze_tbl)
    add(blaze_tbl.render())
    add()
    add(
        f"**实测几乎不随周期变**：排除出画那一行后，实测位移 y 只有 "
        f"{af.blaze_meas_min[1]:.2f}（P={af.blaze_meas_min[0]:.0f}）→ "
        f"{af.blaze_meas_max[1]:.2f} px（P={af.blaze_meas_max[0]:.0f}），"
        f"全程只变 **{af.blaze_flat_ratio:.2f}×**；而预测在同一区间变了 "
        f"**{af.blaze_predicted_ratio:.2f}×**（它按 `TILT_SHIFT_SCALE/周期` 构造，"
        "所以是严格的 1/P）。于是实测/预测从 "
        f"{af.blaze_ratio_min:.3f} 一路涨到 {af.blaze_ratio_max:.3f} —— "
        "**比值不是常数，说明错的不是标度系数，而是 1/P 这个形状本身**（与 §6.4 同一结论，"
        "这一轮把它在同一功率下复现了一遍）。"
    )
    add()
    add(
        f"**出画那一行**：P = {'/'.join(f'{p:.0f}' for p in af.blaze_clipped)} 的 "
        "`box_clipped_px > 0`（预测位移把框推到画面外），该行 PIB 不可与其它行横比。"
        "它的 CV/score 也一并在表里给着，但只能当「框错位后的读数」，不是周期 16 的质量。"
    )
    add()

    # -- 13.5 与 §11 复测的同功率交叉核对 -----------------------------------
    add("### 13.5 与 §11 复测的同功率交叉核对")
    add()
    if af.cross:
        add(
            "§11 的复测批与本轮**同为 "
            f"{af.exposure_ms:g} ms**，所以这两批可以做**同功率**对照"
            "（不同于基线的 20 ms）。逐配置比两批的 CV 与 score："
        )
        add()
        cross_tbl = Table(
            ("段", "配置", "CV (本轮)", "CV (§11)", "ΔCV", "score (本轮)", "score (§11)", "Δscore"),
            [(STAGE_CN[stage], derived(value, 0), derived(cv_a, 4), derived(cv_b, 4),
              derived(dcv, 4), derived(sc_a, 4), derived(sc_b, 4), derived(dsc, 4))
             for stage, value, cv_a, cv_b, dcv, sc_a, sc_b, dsc in af.cross],
        )
        tables.append(cross_tbl)
        add(cross_tbl.render())
        add()
        higher = sum(1 for row in af.cross if row[5] > row[6])
        add(
            f"**结论：同功率不同批次的重复性约 1%。** CV 最大差 "
            f"{af.cross_cv_max:.1%}、score 最大差 {af.cross_score_max:.1%}；"
            f"方向**不系统**（{len(af.cross)} 个配置里本轮 score 高 {higher} 个、"
            f"低 {len(af.cross) - higher} 个），也就是说这 1% 是噪声而不是批次偏差。"
            "这条带宽正是 §11.5 判定 1/2 需要的那个量：基线那两处异常偏 "
            f"+72% / +801%，比它大两个数量级以上，所以「黑帧」判定不依赖某一批的运气。"
        )
    else:
        add("（没有 §11 的复测批可比 —— 未给 `--retest-root`。）")
    add()

    # -- 13.6 与基线的定性对照 ----------------------------------------------
    add("### 13.6 与基线的对照（只能定性）")
    add()
    base = run.stages
    base_sides = base["sides"]
    cmp_rows: list[tuple[Cell | str, ...]] = []
    for label, key, stage, pick in (
        ("边长段 score 最高", "score", "sides", max(base_sides.col("score"))),
        ("边长段 CV 最小", "cv", "sides", min(base_sides.col("cv"))),
        ("Zernike 段 CV 最小", "cv", "zernike", min(base["zernike"].col("cv"))),
        ("光栅段 score 最高", "score", "blaze", max(base["blaze"].col("score"))),
    ):
        mine = (max(atomic.stages[stage].col(key))
                if key == "score" else min(atomic.stages[stage].col(key)))
        cmp_rows.append((label, f"{key} = {pick:.4f}", f"{key} = {mine:.4f}", "不可横比"))
    cmp_tbl = Table(("关键量", "基线（20 ms）", "本轮（180 ms）", "可比性"), cmp_rows)
    tables.append(cmp_tbl)
    add(cmp_tbl.render())
    add()
    add(
        f"两批的曝光差 {af.exposure_ms / run.exposure_ms:.0f}×（功率也差一个量级，见 §11.1），"
        "所以**任何绝对值都不能横比**。这张表只用来回答一个问题：**趋势是否一致** —— "
        "例如「CV 随边长上升」「score 有平台」「低阶 Zernike 的 CV 与阶数无关」，"
        "这些**形状**在两批里是一致的，而高度不是。"
    )
    add()

    # -- 13.7 图 ------------------------------------------------------------
    add("### 13.7 图")
    add()
    drawn = {p.name for p in figures}
    for stem, caption in (
        ("fig10_atomic_sides", "边长 → CV / PIB / score（本轮实线，基线虚线仅作定性对照）"),
        ("fig11_atomic_blaze_shift", "光栅周期 → 预测位移（1/P）vs 实测位移（近水平）"),
    ):
        name = f"{stem}_{batch_tag(atomic)}.png"
        if name in drawn:
            add(f"![{caption}](figures/{name})")
            add()
            add(f"*{caption}。*")
            add()
    for spec in MOSAICS:
        name = f"{spec.stem}_{batch_tag(atomic)}.png"
        if name in drawn:
            add(f"![{spec.title}（本轮）](figures/{name})")
            add()
            add(f"*{spec.title}（本轮 {af.exposure_ms:g} ms）。* "
                "口径与 §12 完全一致：对数显示（`LogNorm(vmin=1, vmax=该格峰值)`，"
                "**颜色不代表线性强度**）、"
                + (f"裁剪 `box_side × {CROP_FACTOR}`、" if spec.crop else "整帧不裁剪、")
                + "目标框/预测框心/实测质心叠加、右下角 `×N 帧均值`。"
                "每格的源 npz 与 sha256 见 `artifacts/"
                f"{PATHS.atomic_artifact_dir.name}/frames_selection.json`。")
            add()
    add(
        "拼图里没有 1800/1900 两格 —— 这两行本来就没有被采集（§13.1）。"
        "边长格取 200 / 600 / 1000 / 1500 / 1700 五档（§13 不追加 score 格："
        "本轮 score 是平台而不是单点，追加一格会误导）。"
    )
    add()

    # -- 13.8 相位残差 × 同配置远场 -----------------------------------------
    if residuals is None or not residuals.panels:
        return out, tables

    panels = residuals.panels
    first = panels[0]      # n_max 最大
    last = panels[-1]      # n_max 最小
    rms_drop = 1.0 - first.rms_recorded / last.rms_recorded
    max_explained = max(p.explained for p in panels)
    unrepresented = 1.0 - max_explained
    max_rel = max(abs(p.rms_recomputed - p.rms_recorded) / max(abs(p.rms_recorded), 1e-12)
                  for p in panels)
    cvs = [float(p.row["cv"]) for p in panels]
    pibs = [float(p.row["pib"]) for p in panels]
    freqs = [p.mean_freq for p in panels]
    highs = [p.high_share for p in panels]
    ratios = [p.radial_ratio for p in panels]
    m4s = [p.m4_share for p in panels]

    add("### 13.8 相位残差 × 同配置远场")
    add()
    add(
        "本节把**被 Zernike 截断掉的那部分相位**与同配置的相机画面并排摆出来。"
        "只有这一批落了相位文件（`zernike/nmax_*/pupil_phase.npy`），基线批没有 ⇒ "
        "这张图只能放在这里（§12 只有远场画面）。"
    )
    add()
    add("口径（与采集脚本的生产路径逐行一致）：")
    add()
    add(f"- 基准相位：`{_rel(residuals.base_path)}`（512², raw rad；边长由 `sides/selected.json` 决定）；")
    add(
        f"- `reference` = `upsample_bilinear_to_shape(base, ({residuals.fit_grid}, "
        f"{residuals.fit_grid}))`，`fit_grid` = {residuals.fit_grid}；"
    )
    add("- `recon` = `generate_noll(coeffs[:modes_for_order(n_max)])` —— 高阶系数**置零**，"
        "所以「第 N 阶重构」严格是「把高阶丢掉」，不是另一个模型的解；")
    add("- `residual` = `recon − reference`，只统计单位圆内（`ZernikeGenerator.mask`）。")
    add()
    add(
        "⚠ 每个 `zernike/nmax_<N>/pupil_phase.npy` 是**该阶数的重构相位**，不是基准；"
        "拿它当基准残差恒为 0。基准在边长段的 `side_<S>/pupil_phase.npy`。"
    )
    add()

    residual_figure = residual_figure_name(atomic)
    if residual_figure in drawn:
        add(f"![Zernike 截断残差 × 同配置 CCD 远场](figures/{residual_figure})")
        add()
        add(
            "*第一行 = 残差相位（`RdBu_r`，固定 ±π，**四列共用同一色标**，单位圆外没有光"
            "所以不画）；第二行 = 同配置 CCD 远场（对数显示 `LogNorm(vmin=1, vmax=四列最大峰值)`、"
            f"裁剪 `box_side × {CROP_FACTOR}`、叠加目标框/预测框心/实测质心、右下角 `×N 帧均值`）；"
            "左上角 = GS 基准相位（`twilight`，±π）。残差在图上已卷到 ±π（面板施加的本来就是 "
            "mod 2π 的相位），而标题与表里的 RMS 取的是**未卷**差值 —— 与 `metrics.json` 同口径。*"
        )
        add()

    res_tbl = Table(
        ("n_max", "模式数", "残差 RMS (rad)", "解释比例", "CV", "PIB",
         "残差里的可见结构（一句话）"),
        [(cell_of(p.row, "n_max", 0), cell_of(p.row, "zernike_modes_used", 0),
          cell_of(p.row, "zernike_rms_residual_rad", 6),
          # 解释比例按 % 显示: 与图标题、与正文的百分比是同一个表示, 免得读者
          # 在 0.032515 与 3.25% 之间换算 (CSV 里存的是分数)。
          derived(float(p.row["zernike_explained_fraction"]) * 100.0, 3, "%"),
          cell_of(p.row, "cv", 4), cell_of(p.row, "pib", 4),
          p.structure_note())
         for p in panels],
    )
    tables.append(res_tbl)
    add(res_tbl.render())
    add()
    add(
        "前三列与 §13.3 同源（同一批 CSV 列；解释比例在这里按 % 显示，CSV 里存的是分数），"
        "新增的只有最后一列 —— 它是从**残差数组**现场算出来的四个量：平均空间频率、"
        "`>0.25 cyc/px`（周期 4 个拟合格）的能量占比、外缘（`r > 0.7`）与中心"
        "（`r ≤ 0.35`）的 RMS 比，以及方位向 `m = 4` 分量的占比。"
    )
    add()
    add(
        "**残差不是「拟合误差」，也不是噪声 —— 它是被这个基截断掉的高频结构。** "
        f"判据是残差随阶数的走向：模式数从 {last.modes:.0f}（`n_max = {last.n_max}`）加到 "
        f"{first.modes:.0f}（`n_max = {first.n_max}`），残差 RMS 只从 "
        f"{last.rms_recorded:.4f} 降到 {first.rms_recorded:.4f} rad（{rms_drop:.1%}），"
        f"解释比例从 {last.explained:.2%} 升到 {first.explained:.2%}。"
        "若残差是「没拟合好」，多给 "
        f"{first.modes - last.modes:.0f} 个自由度应该让它大幅下降；它几乎不动，"
        f"说明方斑所需的相位里有 {unrepresented:.1%} 根本不在前 {first.n_max} 阶张成的空间里。"
    )
    add()
    add(
        "**它在图上长什么样，以及看不到什么。** 四列是同一片细颗粒：平均空间频率 "
        f"{min(freqs):.2f}–{max(freqs):.2f} cyc/px（周期约 4 个拟合格，已经是 "
        f"{residuals.fit_grid} 网格能表示的最细尺度），{min(highs):.0%}–{max(highs):.0%} 的"
        "能量在 `>0.25 cyc/px`；径向几乎不衰减（外缘/中心 RMS 比 "
        f"{min(ratios):.2f}–{max(ratios):.2f}），方位向也没有方斑该有的 4 重对称"
        f"（`m = 4` 只占 {min(m4s):.1%}–{max(m4s):.1%}）。所以**方斑的「边缘与角」不在这张"
        "相位图上** —— 它们不是画在瞳面上的轮廓：目标是均匀方斑（远场幅度是二维 sinc 型，"
        "边与角全在高频里），要把圆盘照明变成它，瞳面相位必须携带同样高频的结构，而这些"
        "结构恰好是 Zernike 截断丢掉的那一层。这一点可以当场验证：把残差相位单独做一次"
        "傅里叶变换，得到的仍是一个**方形**的均匀区域，而不是亮核（这项检查不在图里，"
        "它的用处是把「高频结构」与「噪声」分开 —— 噪声的变换不会保持方形的边）。"
    )
    add()
    add(
        "**远场这一行就是「塌」的样子。** 第二行与 §12 拼图 2 是同一件事的两面："
        f"前 N 阶重构的远场是一个亮核（CV {min(cvs):.3f}–{max(cvs):.3f}，"
        f"PIB {min(pibs):.3f}–{max(pibs):.3f}），而同框的 GS 基准远场是方斑"
        "（§12 拼图 2 最后一格）。CV 在四列之间只差 "
        f"{(max(cvs) - min(cvs)) / min(cvs):.1%} —— 阶数加了四倍画面几乎没变，"
        "这正是 §13.3「基不适配」在相位侧的对应物：改的是相位的高频内容，远场却基本不动，"
        "因为那部分内容根本没有被表示出来。"
    )
    add()
    add(
        "**自检（生成与 `--check` 两条路径都跑，不只是生成时）。** 重算的残差 RMS 必须与 "
        "`metrics.json` 的 `zernike_rms_residual_rad` 相对误差 < "
        f"{RESIDUAL_RMS_RTOL:.0%}，重算相位必须与存盘的 `pupil_phase.npy` 最大偏差 < "
        # `:.0e` 会印成 1e-09; 这里只要值对, 顺手去掉指数里的前导零
        f"{format(RESIDUAL_RECON_ATOL, '.0e').replace('e-0', 'e-')} —— "
        "任一不满足就打印两边值并非零退出。"
        f"本轮 {len(panels)} 个配置全部通过（最大相对误差 {max_rel:.2%}，"
        f"最大偏差 {max(p.recon_delta for p in panels):.1e}）。"
    )
    add()
    return out, tables
