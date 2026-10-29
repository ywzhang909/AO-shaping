"""GS 方扫报告的**指标计算层**。

把 IO 层读到的原始行算成报告要说的事：边长 / Zernike / 闪耀三段各自的
PIB-CV-score 走向、复测的批间带宽、整轮原子重扫的两级分层、以及相位残差
的存盘重算核对。

这一层**只读不说**：产出 :class:`Findings` / :class:`RetestFindings` /
:class:`AtomicFindings` / :class:`ResidualSet`，交给渲染层去写。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np


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
    _idx_of,
    _rel,
)
from scripts.slm._hw_gs_square_scans_io import (
    _selected_side_px,
    _two_tier_split,
    load_json,
)
from scripts.slm._hw_gs_square_scans_metrics import (
    _loglog_exponent,
    _neighbour_deviation,
    monotonic_report,
    pearson,
    retest_power_ratio,
    score_terms,
    spearman,
)
from scripts.slm._hw_gs_square_scans_model import (
    AREA_IDENTITY_ATOL,
    AtomicFindings,
    BLAZE_PANEL_OPTIONAL,
    BLAZE_PANEL_PERIOD,
    CellPick,
    DARK_FRAME_FRACTION,
    EXPOSURE_MATCH_ATOL,
    FULLFRAME_TARGET_PX,
    Findings,
    OUTLIER_CV_RATIO,
    RESIDUAL_HIGH_FREQ,
    RESIDUAL_PANEL_ORDER,
    RESIDUAL_RECON_ATOL,
    RESIDUAL_RMS_RTOL,
    RETEST_STAGES,
    ROW_MATCH_ATOL,
    ResidualPanel,
    ResidualSet,
    RetestFindings,
    RetestRun,
    Run,
    SATURATION_COUNTS,
    SCORE_IDENTITY_ATOL,
    SIDES_PANEL_PX,
    Stage,
    VARIABLE,
    ZERNIKE_PANEL_ORDER,
    run_zero_order,
)

def compute_findings(run: Run) -> Findings:
    sides = run.stages["sides"]
    zern = run.stages["zernike"]
    blaze = run.stages["blaze"]

    xs = sides.x
    pib = sides.col("pib")
    cv = sides.col("cv")
    score = sides.col("score")
    order = sorted(range(len(score)), key=lambda i: -score[i])
    peak = order[0]
    runner = order[1]

    cv_viol = monotonic_report(cv, increasing=True)
    pib_viol = monotonic_report(pib, increasing=True)
    descend = all(score[i] < score[i - 1] for i in range(peak + 1, len(score)))

    near_lo, near_hi = (float(v) for v in (run.stages["sides"].run_config["near_window_px"]))
    near_idx = [i for i, x in enumerate(xs) if near_lo <= x <= near_hi]
    near_best = max(near_idx, key=lambda i: score[i])
    alt_lo, alt_hi = 900.0, 1100.0
    alt_idx = [i for i, x in enumerate(xs) if alt_lo <= x <= alt_hi]
    alt_best = max(alt_idx, key=lambda i: score[i])

    zern_cv = zern.col("cv")
    zern_med = zern.median("cv")
    clean = [
        i for i, r in enumerate(zern.rows)
        if run.settle_for("zernike", i) is None
        and float(r["frame_peak"]) >= DARK_FRAME_FRACTION * zern.median("frame_peak")
        and float(r["cv"]) <= OUTLIER_CV_RATIO * zern_med
    ]

    b_score = blaze.col("score")
    b_clip = [i for i, r in enumerate(blaze.rows) if float(r["box_clipped_px"]) > 0]
    b_dark = [i for i, r in enumerate(blaze.rows)
              if float(r["frame_peak"]) < DARK_FRAME_FRACTION * blaze.median("frame_peak")]
    b_excluded = sorted(set(b_clip) | set(b_dark))
    b_clean = [i for i in range(len(b_score)) if i not in b_excluded]

    shift_ratio = []
    for row in blaze.rows:
        pred = float(row["predicted_shift_px"])
        meas = float(row["measured_shift_y_px"])
        shift_ratio.append(meas / pred if pred else float("nan"))

    model_pib = sides.col("model_pib")
    model_delta = [m - p for m, p in zip(model_pib, pib)]

    cfg = sides.run_config
    zo_x, zo_y = run.zero_order_xy
    h, w = run.frame_shape_hw
    margin = float(cfg["plan"]["frame"]["edge_margin_px"]) if "plan" in cfg else 0.0
    dist = {
        "左": zo_x,
        "右": (w - 1) - zo_x,
        "上": zo_y,
        "下": (h - 1) - zo_y,
    }
    binding = min(dist, key=lambda k: dist[k])
    raw = 2.0 * (dist[binding] - margin)
    step = float(min(b - a for a, b in zip(xs, xs[1:]))) if len(xs) > 1 else 1.0
    side_max = math.floor(raw / step) * step

    planned = [float(v) for v in run.stage_plans.get("sides", {}).get("configs", [])]
    dropped = [f"{int(v)}" for v in planned if v not in set(xs)]

    return Findings(
        sides_x=xs, pib=pib, cv=cv, score=score,
        score_peak_index=peak, score_runner_up_index=runner,
        score_peak_gap=score[peak] - score[runner],
        pib_violations=pib_viol, cv_violations=cv_viol,
        cv_spearman=spearman(xs, cv),
        score_descend_ok=descend,
        terms_at_peak=score_terms(pib[peak], cv[peak]),
        terms_at_last=score_terms(pib[-1], cv[-1]),
        terms_at_small=score_terms(pib[0], cv[0]),
        near_window=(near_lo, near_hi), near_best_index=near_best,
        alt_window=(alt_lo, alt_hi), alt_best_index=alt_best,
        selected_side=_selected_side_px(run),
        zern_explained=zern.col("zernike_explained_fraction"),
        zern_residual=zern.col("zernike_rms_residual_rad"),
        zern_cv=zern_cv, zern_score=zern.col("score"),
        zern_clean_cv=[zern_cv[i] for i in clean],
        zern_clean_cv_spread=(
            (max(zern_cv[i] for i in clean) - min(zern_cv[i] for i in clean))
            / min(zern_cv[i] for i in clean)
        ),
        zern_cv_pearson=pearson([zern.x[i] for i in clean], [zern_cv[i] for i in clean]),
        zern_explained_violations=monotonic_report(zern.col("zernike_explained_fraction")),
        blaze_score=b_score,
        blaze_score_violations_clean=monotonic_report(
            [b_score[i] for i in b_clean]),
        blaze_score_violations_all=monotonic_report(b_score),
        blaze_shift_ratio=shift_ratio,
        model_pib=model_pib, model_delta=model_delta,
        side_max_raw=raw, side_max=side_max, step=step, edge_margin=margin,
        binding_edge=binding,
        dropped_configs=dropped,
        csv_cells=sum(len(r) for s in run.stages.values() for r in s.rows),
    )

def compute_retest_findings(
    run: Run, retest: RetestRun, prior: RetestRun | None = None
) -> RetestFindings:
    base_cpm, now_cpm, ratio = retest_power_ratio()
    blaze = retest.stages["blaze"]
    zern = retest.stages["zernike"]
    base_blaze = run.stages["blaze"]
    base_zern = run.stages["zernike"]

    b_cv = blaze.col("cv")
    blaze_monotone = all(b_cv[i] < b_cv[i - 1] for i in range(1, len(b_cv)))
    i40 = _idx_of([float(r["blaze_period_px"]) for r in blaze.rows], 40.0)
    z_cv = zern.col("cv")
    zern_monotone = all(z_cv[i] > z_cv[i - 1] for i in range(1, len(z_cv)))
    i18 = _idx_of([float(r["n_max"]) for r in zern.rows], 18.0)

    # 基线同位置的偏差 (同一套「相对左右邻居中位数」的口径)
    base_b_cv = base_blaze.col("cv")
    base_z_cv = base_zern.col("cv")
    base_blaze40_dev = _neighbour_deviation(
        base_b_cv, _idx_of([float(r["blaze_period_px"]) for r in base_blaze.rows], 40.0)
    )
    base_zern18_dev = _neighbour_deviation(
        base_z_cv, _idx_of([float(r["n_max"]) for r in base_zern.rows], 18.0)
    )

    # 偏移量恒等式: offset_y == measured_shift_y − predicted_shift_y
    residual = 0.0
    for cfg in blaze.configs:
        offset = float(cfg.row["box_centroid_offset_y"])
        identity = float(cfg.row["measured_shift_y_px"]) - float(cfg.row["predicted_shift_px"])
        residual = max(residual, abs(offset - identity))

    z0 = [cfg.zero_order for cfg in blaze.configs]
    centroid_x = [float(cfg.row["box_centroid_x"]) for cfg in blaze.configs]
    centroid_y = [float(cfg.row["box_centroid_y"]) for cfg in blaze.configs]
    measured_x = [float(cfg.row["measured_shift_x_px"]) for cfg in blaze.configs]
    measured_y = [float(cfg.row["measured_shift_y_px"]) for cfg in blaze.configs]

    # 位移: 只用 measured_shift_y (相对冻结 0 级)。基线黑帧行(period=40)剔除。
    base_med_peak = base_blaze.median("frame_peak")
    base_clean = [
        row for row in base_blaze.rows
        if float(row["frame_peak"]) >= DARK_FRAME_FRACTION * base_med_peak
    ]
    ratio_baseline = [
        float(r["measured_shift_y_px"]) / float(r["predicted_shift_px"]) for r in base_clean
    ]
    ratio_retest = [
        float(cfg.row["measured_shift_y_px"]) / float(cfg.row["predicted_shift_px"])
        for cfg in blaze.configs
    ]

    def ends(rows: Sequence[dict[str, str]], measured_key: str, predicted_key: str):
        ordered = sorted(rows, key=lambda r: float(r["blaze_period_px"]))
        lo, hi = ordered[0], ordered[-1]
        return (
            (float(lo["blaze_period_px"]), float(lo[measured_key]), float(hi[measured_key])),
            (float(hi["blaze_period_px"]), float(lo[predicted_key]), float(hi[predicted_key])),
        )

    ret_meas, ret_pred = ends([cfg.row for cfg in blaze.configs],
                              "measured_shift_y_px", "predicted_shift_px")
    base_meas, base_pred = ends(base_clean, "measured_shift_y_px", "predicted_shift_px")

    # 小周期处「框心 − 质心」偏多少 (框跟着预测走时框偏得有多远)
    offset_y_small = sorted(
        [(float(cfg.row["blaze_period_px"]), float(cfg.row["box_centroid_offset_y"]))
         for cfg in blaze.configs],
        key=lambda pair: pair[0],
    )

    # 与基线的同周期对照 (基线黑帧行不参与 —— 它的偏移量无意义)
    offset_delta: list[tuple[float, float, float]] = []
    for row in blaze.rows:
        value = float(row["blaze_period_px"])
        b_row = None
        for cand in base_blaze.rows:
            if abs(float(cand["blaze_period_px"]) - value) < ROW_MATCH_ATOL:
                b_row = cand
                break
        if b_row is None or float(b_row["frame_peak"]) < DARK_FRAME_FRACTION * base_med_peak:
            continue
        offset_delta.append((
            value,
            float(b_row["box_centroid_offset_y"]),
            float(row["box_centroid_offset_y"]) - float(b_row["box_centroid_offset_y"]),
        ))

    # 拟合量跨会话逐位一致? (同一 GS 相位 + 同一拟合配置 ⇒ 应当逐位相同)
    identical: list[float] = []
    differs: list[float] = []
    for cfg in zern.configs:
        b_row = None
        for row in base_zern.rows:
            if abs(float(row["n_max"]) - cfg.value) < ROW_MATCH_ATOL:
                b_row = row
                break
        if b_row is None:
            continue
        same = (
            float(b_row["zernike_rms_residual_rad"]) == float(cfg.row["zernike_rms_residual_rad"])
            and float(b_row["zernike_explained_fraction"])
            == float(cfg.row["zernike_explained_fraction"])
            and float(b_row["zernike_modes_used"]) == float(cfg.row["zernike_modes_used"])
        )
        (identical if same else differs).append(cfg.value)

    # 批内一致性: 单批 6 个点各自相对左右邻居中位数的偏差, 取绝对值最大者。
    # (三点时"邻居中位数"就是线性插值, 所以中间点的这个量等价于"曲率残差"。)
    devs = []
    for stage_name in RETEST_STAGES:
        st = retest.stages[stage_name]
        cvs = st.col("cv")
        for i in range(len(cvs)):
            if i in (0, len(cvs) - 1):
                continue
            dev = _neighbour_deviation(cvs, i)
            if math.isfinite(dev):
                devs.append(abs(dev))
        # 端点也算进去 (它是趋势斜率, 不是噪声), 用于给出"批内最大偏差"的上界
        for i in (0, len(cvs) - 1):
            neighbours = [cvs[j] for j in ((i - 1,) if i else (i + 1,))]
            if neighbours and neighbours[0]:
                devs.append(abs(cvs[i] / neighbours[0] - 1.0))

    flagged = [cfg.config for cfg in retest.all_configs if cfg.spot_flagged]
    settle_cfgs = [cfg.config for cfg in retest.all_configs if cfg.settle]
    dark_probe = [
        cfg.config for cfg in retest.all_configs
        if any(ev.probe_peak < DARK_FRAME_FRACTION * SATURATION_COUNTS for ev in cfg.settle)
    ]

    return RetestFindings(
        power_ratio=ratio,
        baseline_counts_per_ms=base_cpm,
        retest_counts_per_ms=now_cpm,
        equivalent_exposure_ms=retest.exposure_ms / ratio,
        baseline_exposure_ms=run.exposure_ms,
        retest_exposure_ms=retest.exposure_ms,
        blaze_cv=b_cv,
        blaze_cv_monotone=blaze_monotone,
        blaze40_dev=_neighbour_deviation(b_cv, i40),
        baseline_blaze40_dev=base_blaze40_dev,
        zern_cv=z_cv,
        zern_cv_monotone=zern_monotone,
        zern18_dev=_neighbour_deviation(z_cv, i18),
        baseline_zern18_dev=base_zern18_dev,
        offset_identity_residual=residual,
        ratio_retest=ratio_retest,
        ratio_baseline=ratio_baseline,
        alpha_retest=_loglog_exponent(
            [ret_meas[0], ret_pred[0]], [ret_meas[1], ret_meas[2]]),
        alpha_baseline=_loglog_exponent(
            [base_meas[0], base_pred[0]], [base_meas[1], base_meas[2]]),
        alpha_predicted=_loglog_exponent(
            [ret_meas[0], ret_pred[0]], [ret_pred[1], ret_pred[2]]),
        measured_ends_retest=ret_meas,
        measured_ends_baseline=base_meas,
        predicted_ends_retest=ret_pred,
        predicted_ends_baseline=base_pred,
        offset_y_small_period=offset_y_small,
        zero_order_x_span=max(x for x, _ in z0) - min(x for x, _ in z0),
        zero_order_y_span=max(y for _, y in z0) - min(y for _, y in z0),
        zero_order_x_span_zern=(lambda zz: max(x for x, _ in zz) - min(x for x, _ in zz))(
            [cfg.zero_order for cfg in zern.configs]),
        zero_order_y_span_zern=(lambda zz: max(y for _, y in zz) - min(y for _, y in zz))(
            [cfg.zero_order for cfg in zern.configs]),
        centroid_x_span=max(centroid_x) - min(centroid_x),
        centroid_y_span=max(centroid_y) - min(centroid_y),
        measured_x_span=max(measured_x) - min(measured_x),
        measured_y_span=max(measured_y) - min(measured_y),
        offset_y_vs_baseline=offset_delta,
        fit_bit_identical=identical,
        fit_differs=differs,
        flagged_configs=flagged,
        settle_configs=settle_cfgs,
        dark_probe_configs=dark_probe,
        retest_rows=sum(len(st.rows) for st in retest.stages.values()),
        batch_dev_max=max(devs) if devs else float("nan"),
        prior_root=prior.root if prior is not None else None,
    )

def select_cells(
    run: Run,
    f: Findings,
    *,
    sides_panel: Sequence[float] = SIDES_PANEL_PX,
    add_score_cell: bool = True,
) -> dict[str, list[CellPick]]:
    """按写死的规则从**实际存在的 CSV 行**里挑格子。

    ``sides_panel`` / ``add_score_cell`` 让 §13 的整轮重扫用另一套边长格
    (200/600/1000/1500/1700, 且不追加 score 格) —— 其余规则完全一样。
    """
    sides = run.stages["sides"]
    zern = run.stages["zernike"]
    blaze = run.stages["blaze"]

    def pick(stage: Stage, target: float, role: str, note: str = "") -> CellPick | None:
        key = VARIABLE[stage.name]
        for row in stage.rows:
            if abs(float(row[key]) - float(target)) < ROW_MATCH_ATOL:
                return CellPick(stage=stage.name, row=row, role=role, note=note)
        return None

    # -- 拼图 1: 5 个固定边长 + score 最高行 ---------------------------------
    sides_picks: list[CellPick] = []
    for value in sides_panel:
        got = pick(sides, value, "固定格")
        if got is not None:
            sides_picks.append(got)
    best = CellPick(stage="sides", row=sides.rows[f.score_peak_index],
                    role="score 最高", note=f"score 最高 ({f.score[f.score_peak_index]:.4f})")
    taken = {p.value for p in sides_picks}
    if not add_score_cell:
        pass
    elif best.value not in taken:
        sides_picks.append(best)
    else:
        # 规则里的回退("换 1000")在本轮会与固定格重复, 所以退一步: 取最接近 1000 的
        # **未入选**行, 并在表里写明这是回退 —— 否则第 6 格只是第 3 格的复制品。
        candidates = [r for r in sides.rows if float(r["side_px"]) not in taken]
        if candidates:
            nearest = min(candidates,
                          key=lambda r: abs(float(r["side_px"]) - FULLFRAME_TARGET_PX))
            sides_picks.append(CellPick(
                stage="sides", row=nearest, role="回退",
                note=(f"score 最高行 ({best.value:.0f} px) 已在固定格内 ⇒ "
                      f"取最接近 1000 的未入选行 {float(nearest['side_px']):.0f} px"),
            ))
        else:
            sides_picks.append(CellPick(
                stage="sides", row=best.row, role="重复",
                note=f"score 最高行 ({best.value:.0f} px) 已在固定格内, 且没有别的行可选"))

    # -- 拼图 2: 4 个阶数 + 同 run-root 的 GS 参照 ---------------------------
    zern_picks: list[CellPick] = []
    for order in ZERNIKE_PANEL_ORDER:
        got = pick(zern, order, "固定格")
        if got is not None:
            zern_picks.append(got)
    ref_side = float(_selected_side_px(run))
    gs_ref = pick(sides, ref_side, "GS 参照",
                  note=f"同 run-root 边长段的 {ref_side:.0f} px 行（未截断）")
    if gs_ref is not None:
        zern_picks.append(gs_ref)

    # -- 拼图 3: 3 个周期 + 16 (有就画) --------------------------------------
    blaze_picks: list[CellPick] = []
    for period in BLAZE_PANEL_PERIOD:
        got = pick(blaze, period, "固定格")
        if got is not None:
            blaze_picks.append(got)
    optional = pick(blaze, BLAZE_PANEL_OPTIONAL, "补充格",
                    note="该行方框出画, PIB 不可与其它行横比")
    if optional is not None:
        blaze_picks.append(optional)

    # -- 拼图 4: 最接近 1000 px 的一行, 整帧不裁剪 ---------------------------
    nearest_row = min(sides.rows,
                      key=lambda r: abs(float(r["side_px"]) - FULLFRAME_TARGET_PX))
    full_pick = CellPick(stage="sides", row=nearest_row, role="整帧",
                         note="不裁剪, 用于核对 0 级与画面中心的相对位置")

    return {"sides": sides_picks, "zernike": zern_picks,
            "blaze": blaze_picks, "fullframe": [full_pick]}

def compute_atomic_findings(
    run: Run, atomic: RetestRun, selected: dict[str, Any],
    retest: RetestRun | None,
) -> AtomicFindings:
    sides = atomic.stages["sides"]
    zern = atomic.stages["zernike"]
    blaze = atomic.stages["blaze"]
    sides_cfg = sides.run_config

    xs = sides.col("side_px")
    scores = sides.col("score")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    peak = order[0]
    score_max = scores[peak]
    plateau = [(xs[i], scores[i]) for i in range(len(xs)) if scores[i] >= score_max - 1e-3]
    tail = [(xs[i], scores[i]) for i in range(len(xs)) if xs[i] > max(p for p, _ in plateau)]
    plateau_gap = (max(s for _, s in plateau) - min(s for _, s in plateau)) if len(plateau) > 1 else 0.0

    box_peaks = sides.col("box_peak")
    ars = sides.col("ar")
    ar_median = float(np.median(ars))
    ar_outliers = [(xs[i], ars[i]) for i in range(len(ars))
                   if abs(ars[i] / ar_median - 1.0) > 0.5]

    z_cv = zern.col("cv")
    z_pib = zern.col("pib")
    z_exp = zern.col("zernike_explained_fraction")
    z_off = zern.col("box_centroid_offset_x")
    z_centroid = zern.col("box_centroid_x")
    z_n = zern.col("n_max")
    z0x = [cfg.zero_order[0] for cfg in zern.configs]
    split = _two_tier_split(z0x)
    tier_low: list[tuple[float, float, float]] = []
    tier_high: list[tuple[float, float, float]] = []
    for i in range(len(z_n)):
        bucket = tier_low if split is None or z0x[i] < split else tier_high
        bucket.append((z_n[i], z_off[i], z0x[i]))

    b_p = blaze.col("blaze_period_px")
    b_pred = blaze.col("predicted_shift_px")
    b_meas = blaze.col("measured_shift_y_px")
    blaze_rows = [
        (b_p[i], b_pred[i], b_meas[i],
         b_meas[i] / b_pred[i] if b_pred[i] else float("nan"),
         blaze.col("cv")[i], blaze.col("pib")[i], blaze.col("score")[i],
         blaze.col("box_clipped_px")[i])
        for i in range(len(b_p))
    ]
    clipped = [b_p[i] for i in range(len(b_p)) if blaze.col("box_clipped_px")[i] > 0]
    clean = [row for row in blaze_rows if row[7] <= 0]
    meas_clean = [row[2] for row in clean]
    pred_clean = [row[1] for row in clean]

    # 与 §11 复测的同功率交叉核对 (只比两批都有的配置)
    cross: list[tuple[str, float, float, float, float, float, float, float]] = []
    if retest is not None:
        for stage_name in ("blaze", "zernike"):
            if stage_name not in retest.stages:
                continue
            for cfg in retest.stages[stage_name].configs:
                try:
                    mine = atomic.stages[stage_name].row_of(cfg.value)
                except KeyError:
                    continue
                cv_a = float(mine["cv"])
                cv_b = float(cfg.row["cv"])
                sc_a = float(mine["score"])
                sc_b = float(cfg.row["score"])
                cross.append((
                    stage_name, cfg.value, cv_a, cv_b,
                    abs(cv_a - cv_b) / ((cv_a + cv_b) / 2.0),
                    sc_a, sc_b, abs(sc_a - sc_b) / ((sc_a + sc_b) / 2.0),
                ))

    return AtomicFindings(
        exposure_ms=atomic.exposure_ms,
        ceiling_ms=atomic.exposure_ceiling_ms,
        mode=atomic.mode,
        zero_order=run_zero_order(sides_cfg),
        side_max=float(sides_cfg.get("side_max_px", float("nan"))),
        side_max_raw=float(sides_cfg.get("side_max_raw_px", float("nan"))),
        step=float(min(b - a for a, b in zip(xs, xs[1:]))) if len(xs) > 1 else 1.0,
        rejected=(
            float(sides_cfg.get("side_max_px", float("nan"))) + 100.0,
            float(sides_cfg.get("side_max_px", float("nan"))) + 200.0,
        ),
        n_sides=len(xs),
        selected_side=float(selected.get("side_px", float("nan"))),
        selected_score=float(selected.get("rank_value", float("nan"))),
        score_max=score_max,
        score_argmax_side=xs[peak],
        plateau=plateau,
        plateau_gap=plateau_gap,
        tail=tail,
        box_peak_min=min(box_peaks),
        box_peak_max=max(box_peaks),
        frame_peak_values=sorted({float(v) for v in sides.col("frame_peak")}),
        ar_median=ar_median,
        ar_outliers=ar_outliers,
        zern_cv_min=min(z_cv),
        zern_cv_max=max(z_cv),
        zern_pib_min=min(z_pib),
        zern_pib_max=max(z_pib),
        zern_explained_min=min(z_exp),
        zern_explained_max=max(z_exp),
        zern_modes_max=max(zern.col("zernike_modes_used")),
        tier_split=split,
        tier_low=tier_low,
        tier_high=tier_high,
        centroid_x_span=max(z_centroid) - min(z_centroid),
        blaze=blaze_rows,
        blaze_meas_min=((min(clean, key=lambda r: r[2])[0],
                         min(clean, key=lambda r: r[2])[2]) if clean
                        else (float("nan"), float("nan"))),
        blaze_meas_max=((max(clean, key=lambda r: r[2])[0],
                         max(clean, key=lambda r: r[2])[2]) if clean
                        else (float("nan"), float("nan"))),
        blaze_ratio_min=min(row[3] for row in clean) if clean else float("nan"),
        blaze_ratio_max=max(row[3] for row in clean) if clean else float("nan"),
        blaze_flat_ratio=(max(meas_clean) / min(meas_clean)) if clean else float("nan"),
        blaze_predicted_ratio=(max(pred_clean) / min(pred_clean)) if clean else float("nan"),
        blaze_clipped=clipped,
        cross=cross,
        cross_cv_max=max((row[4] for row in cross), default=float("nan")),
        cross_score_max=max((row[7] for row in cross), default=float("nan")),
    )

def base_phase_path(root: Path, metrics: dict[str, Any]) -> Path:
    """GS 基准相位 (512², raw rad) 的路径。

    🔴 **基准不在 `<stage>/nmax_<N>/` 里**。每个 zernike 配置目录的 `pupil_phase.npy`
    是**该阶数的重构相位**; 拿它当基准会得到残差恒为 0 (踩过)。真正的基准是边长段
    选定那一格的相位 `<root>/sides/side_<S>/pupil_phase.npy`, 路径优先取 `metrics`
    里记的 `base_phase_info.side_dir` (绝对路径), 否则按 `selected_side_px` 反推。
    """
    candidates: list[Path] = []
    side_dir = (metrics.get("base_phase_info") or {}).get("side_dir")
    if side_dir:
        candidates.append(Path(str(side_dir)) / "pupil_phase.npy")
        candidates.append(root / str(side_dir) / "pupil_phase.npy")
    side = int(float(metrics.get("selected_side_px") or 0))
    if side:
        candidates.append(root / "sides" / f"side_{side}" / "pupil_phase.npy")
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"找不到 GS 基准相位 (selected_side_px={side}, "
        f"base_phase_info={metrics.get('base_phase_info')!r}) —— 残差不能算"
    )

def _residual_structure(residual: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """残差相位的三个结构量 (径向分布 / 空间频率 / 方位向 4 重对称)。"""
    grid = int(residual.shape[0])
    centre = (grid - 1) / 2.0
    yy, xx = np.mgrid[0:grid, 0:grid]
    radius = np.hypot(yy - centre, xx - centre) / (grid / 2.0)
    inner = mask & (radius <= 0.35)
    outer = mask & (radius > 0.70)
    rms_in = float(np.sqrt(np.mean(residual[inner] ** 2))) if inner.any() else float("nan")
    rms_out = float(np.sqrt(np.mean(residual[outer] ** 2))) if outer.any() else float("nan")

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(grid))
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    krad = np.hypot(ky, kx)
    total = float(spectrum.sum())
    mean_freq = float((spectrum * krad).sum() / total) if total > 0 else float("nan")
    high_share = float(spectrum[krad > RESIDUAL_HIGH_FREQ].sum() / total) if total > 0 else float("nan")

    power = np.where(mask, residual ** 2, 0.0)
    theta = np.arctan2(yy - centre, xx - centre)
    m4 = float(np.abs((power * np.exp(4j * theta)).sum()) / power.sum()) if power.sum() > 0 else float("nan")
    return {
        "mean_freq": mean_freq,
        "high_share": high_share,
        "radial_ratio": (rms_out / rms_in) if rms_in else float("nan"),
        "m4_share": m4,
    }

def compute_residuals(atomic: RetestRun) -> ResidualSet | None:
    """逐配置重算「前 N 阶重构 − GS 基准相位」, 并算结构描述量。

    口径 (与 `logs/check_zernike_phase_residual.py` 逐行一致)::

        base      = <root>/sides/side_<S>/pupil_phase.npy        # 512², raw rad
        reference = upsample_bilinear_to_shape(base, (fit_grid, fit_grid))
        recon     = generate_noll(coeffs[:modes_for_order(n_max)])  # 高阶置零
        residual  = recon − reference   (只统计 ZernikeGenerator.mask 内)

    Zernike 数学**全部复用** `scripts/slm/hw_gs_square_scans.py` 与
    `ao_shaping.utils.wavefront.zernike_calc`, 本函数不自己算任何基函数。那两个模块
    是**延迟导入**的: 它们会连带 import 驱动包, 而本脚本的模块级 import 必须保持
    「不 import 任何驱动」, 好在没有产物时也能跑。

    Returns:
        :class:`ResidualSet`; 该批**没有任何** `pupil_phase.npy` 时返回 ``None``
        (基线批就是这样 —— 那时完全不渲染 §13.8)。
    """
    from ao_shaping.utils.image.resample import upsample_bilinear_to_shape
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
    from scripts.slm.hw_gs_square_scans import ZernikeFit, reconstruct_from_fit

    stage = atomic.stages["zernike"]
    stage_dir = atomic.root / "zernike"
    wanted = [
        cfg for cfg in stage.configs
        if int(round(cfg.value)) in set(RESIDUAL_PANEL_ORDER)
        and (stage_dir / cfg.config / "pupil_phase.npy").exists()
    ]
    if not wanted:
        return None
    wanted.sort(key=lambda cfg: -cfg.value)

    base_path = base_phase_path(atomic.root, wanted[0].metrics)
    base = np.asarray(np.load(base_path), dtype=np.float64)
    generators: dict[tuple[int, int], Any] = {}
    panels: list[ResidualPanel] = []
    for cfg in wanted:
        other = base_phase_path(atomic.root, cfg.metrics)
        if other.resolve() != base_path.resolve():
            raise ValueError(
                f"{cfg.config} 的基准相位 {_rel(other)} 与首列的 {_rel(base_path)} 不同 —— "
                "同一张图里的残差必须共用一个基准"
            )
        grid = int(cfg.metrics["fit_grid"])
        order = int(cfg.metrics["fit_max_order"])
        reference = upsample_bilinear_to_shape(base, (grid, grid))
        key = (grid, order)
        if key not in generators:
            generators[key] = ZernikeGenerator((grid, grid), radius=grid / 2.0, n_orders=order)
        gen = generators[key]
        fit = ZernikeFit(
            coefficients=np.asarray(cfg.metrics["fit_coefficients_noll"], dtype=np.float64),
            fit_grid=grid,
            max_order=order,
            reference=reference,
            mask=np.asarray(gen.mask, dtype=bool),
        )
        n_max = int(round(cfg.value))
        recon, stats = reconstruct_from_fit(fit, n_max, generator=gen)
        mask = np.asarray(fit.mask, dtype=bool)
        saved = np.asarray(np.load(stage_dir / cfg.config / "pupil_phase.npy"), dtype=np.float64)
        # 显示用的是**卷到 ±π** 的残差: 面板施加的相位就是 mod 2π 的, 卷过之后 ±π 的
        # 色标才逐格有意义。统计量仍取 reconstruct_from_fit 的**未卷**差值 (与记录值同口径)。
        residual = np.where(mask, np.mod(recon - reference + math.pi, 2.0 * math.pi) - math.pi, 0.0)
        panels.append(
            ResidualPanel(
                n_max=n_max,
                config=cfg.config,
                row=cfg.row,
                modes=float(cfg.row["zernike_modes_used"]),
                rms_recorded=float(cfg.row["zernike_rms_residual_rad"]),
                explained=float(cfg.row["zernike_explained_fraction"]),
                rms_recomputed=float(stats["zernike_rms_residual_rad"]),
                recon_delta=float(np.abs(recon - saved).max()),
                residual=residual,
                mask=mask,
                **_residual_structure(residual, mask),
            )
        )
    return ResidualSet(base=base, base_path=base_path, fit_grid=int(panels[0].mask.shape[0]),
                       panels=panels)


