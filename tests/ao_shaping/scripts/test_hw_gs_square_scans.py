"""``scripts/slm/hw_gs_square_scans.py`` 的离线测试 —— 不碰任何设备。

覆盖的是**会静默错掉**的那几条:

* 最大边长 (0 级**不在**画面中心时也要对 —— 本实验系统的 0 级实测偏离中心数百 px);
* 边长扫描网格与「近 1000」选点 (区间内没有点时必须**报错**, 不能静默退化成全局最优);
* 阶数 → Zernike 模式数 (两套索引漂移时先在这里炸);
* 光栅周期 → 预测位移 (面板 x 斜坡 → 相机 y 的 90° 交换);
* 边长 → GS 网格像素的换算 (整条链路: GS → 方框 → 面板相位), 以及 GS 是确定性的;
* 源码里**没有**任何黑盒随机梯度迭代 (本脚本只用 GS);
* ``--dry-run`` 能跑完、打印验收所需的每一项、并且真的没 import 驱动包。

原子模式 (一个进程一组参数) 另有几条**跨进程**契约, 也全在这里钉住:

* 配置身份 (``config-slug``) 与命令清单的行数/顺序 (sides 19 → select 1 →
  zernike 24 → blaze 8 → summarize 1; ``--side-step`` 一改行数就跟着变);
* 累积件按 slug 去重 (同一配置重跑不产生第二行)、CSV 按自变量排序、JSON 原子替换;
* ``selected.json`` 是第 2/3 段的**唯一**输入: 缺失时报错 (退出码 6) 且点名
  ``--scan select``, 绝不静默退化成全局最优或默认 1000;
* 基准相位优先复用第 1 段存盘的 pupil 相位 (命中时**不跑 GS**), 形状不符才重算,
  两种都在 ``base_phase_source`` 里如实记录。

帧是合成的, 相位是本地算的, 设备一次都没打开。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.slm.hw_gs_square_scans import (
    CONFIG_PUPIL_PHASE_NAME,
    EXIT_NO_MEASUREMENTS,
    EXIT_NO_NEAR_SIDE,
    EXIT_SELECTION_MISSING,
    Bench,
    NoNearSideError,
    Selection,
    SelectionMissing,
    _resolve_base_pupil,
    atomic_gs_grid,
    atomic_selection,
    blaze_shift_table,
    box_fits,
    build_parser,
    build_plan,
    build_bench,
    command_plan_lines,
    config_slug,
    flatten_row,
    gs_seconds,
    load_cached_pupil_phase,
    load_selection,
    max_side_from_frame,
    measure_box,
    modes_for_order,
    parse_frame_shape,
    parse_orders_spec,
    parse_xy,
    pupil_phase_to_panel,
    require_accumulation_root,
    run_gs_square,
    run_select,
    save_selection,
    select_near_side,
    side_scan_list,
    update_manifest_atomic,
    upsert_measurements,
    write_config_artifacts,
    write_json_atomic,
)
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE
from ao_shaping.utils.wavefront.zernike_calc import noll_indices, zernike_modes

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "hw_gs_square_scans.py"

#: dry-run / ``--plan-commands`` 必须显式给像元: 从型号配置读像元要 import 驱动包,
#: 而那条路径保证「不 import 任何 ao_shaping.drivers.*」。这里统一用本实验系统
#: Daheng MER2-507-23GM 的实测值, 与被钉住的历史读数 (焦距/位移) 同源。
_DRY_PITCH = ("--camera-pixel-um", "2.2")


# ---------------------------------------------------------------------------
# 合成帧
# ---------------------------------------------------------------------------
def spot_frame(
    shape: tuple[int, int] = (300, 320),
    *,
    cy: int = 150,
    cx: int = 170,
    peak: float = 50.0,
    sigma: float = 6.0,
    noise: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """一个偏离画面中心的高斯光斑 + 读出噪声。"""
    height, width = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    frame = peak * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)
    )
    return frame + rng.normal(0.0, noise, shape)


def uniform_square_frame(
    shape: tuple[int, int] = (300, 320),
    *,
    cy: int = 150,
    cx: int = 170,
    side: int = 100,
    level: float = 50.0,
    noise: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """一个平顶方斑 + 读出噪声 (用来验指标口径)。"""
    rng = np.random.default_rng(seed)
    frame = rng.normal(0.0, noise, shape)
    half = side // 2
    frame[cy - half : cy + half, cx - half : cx + half] += level
    return frame


# ---------------------------------------------------------------------------
# 最大边长
# ---------------------------------------------------------------------------
class TestMaxSide:
    """``side_max = 2·(min(cx, W-1-cx, cy, H-1-cy) - margin)``, 再按步长取整。"""

    def test_centred_zero_order(self):
        raw, floored = max_side_from_frame((300, 320), (160.0, 150.0), margin_px=10, step_px=1)
        # min(160, 319-160=159, 150, 299-150=149) = 149 ⇒ 2*(149-10) = 278
        assert raw == 278
        assert floored == 278

    def test_zero_order_off_centre_is_not_assumed_centred(self):
        """0 级靠近上边缘 ⇒ 上限由**行**方向决定, 不是画幅中心。"""
        raw, _ = max_side_from_frame((300, 320), (160.0, 40.0), margin_px=10, step_px=1)
        assert raw == 2 * (40 - 10)

    def test_default_margin_is_the_canonical_constant(self):
        from ao_shaping.utils.image.frame_analysis import TARGET_EDGE_MARGIN_PX

        assert max_side_from_frame((300, 320), (160.0, 150.0))[0] == 2 * (
            149 - TARGET_EDGE_MARGIN_PX
        )

    def test_step_flooring(self):
        raw, floored = max_side_from_frame((300, 320), (160.0, 150.0), margin_px=10, step_px=100)
        assert raw == 278
        assert floored == 200
        assert floored % 100 == 0 and floored <= raw

    def test_zero_order_at_the_edge_gives_zero(self):
        assert max_side_from_frame((300, 320), (2.0, 150.0), margin_px=10) == (0, 0)

    def test_non_finite_centre_is_refused(self):
        with pytest.raises(ValueError):
            max_side_from_frame((300, 320), (float("nan"), 150.0))

    def test_detected_from_a_real_frame(self):
        """在合成帧上 ``argmax`` 定位 0 级后再算 —— 与实测路径同一条链。"""
        from ao_shaping.utils.image.beam_metrics import zero_order_center

        frame = spot_frame(cy=60, cx=250, peak=80.0)
        cx, cy = zero_order_center(frame, refine=False, despike_k=3)
        assert (cx, cy) == (250, 60), "argmax 应当落在光斑上"
        raw, _ = max_side_from_frame(frame.shape, (cx, cy), margin_px=10, step_px=1)
        assert raw == 2 * (min(250, 320 - 1 - 250, 60, 300 - 1 - 60) - 10)

    def test_agrees_with_the_canonical_frame_analysis_helper(self):
        """与 ``frame_analysis.max_target_side_at_zero_order`` 同一条式子 —— 不是第二套定义。

        那个函数要一个**帧**(它自己在 ``center=None`` 时会去 argmax), 也不返回未取整
        值 / 不按步长取整, 所以这里没法直接调用它; 这条测试就是「同一公式」的证据。

        唯一的有意差异在**退化分支**: 0 级离画面边缘比留白还近时本函数返回 0
        (调用方据此报错「网格不可用」), 而那个函数把它抬到 ``min_side``。
        """
        from ao_shaping.utils.image.frame_analysis import max_target_side_at_zero_order

        frame = spot_frame(cy=70, cx=240, peak=90.0)
        for center in ((240.0, 70.0), (100.0, 150.0), (11.0, 11.0)):
            mine = max_side_from_frame(frame.shape, center, margin_px=10, step_px=1)[0]
            theirs = max_target_side_at_zero_order(
                frame, center, margin_px=10, min_side=1
            )
            assert mine == theirs, f"center={center}"
        # 退化: 可用半径为负 ⇒ 本函数给 0 (让 side_scan_list 显式报错)
        assert max_side_from_frame(frame.shape, (5.0, 5.0), margin_px=10) == (0, 0)


class TestSideScanList:
    def test_plain_arithmetic_progression(self):
        assert side_scan_list(100, 400, 100) == (100, 200, 300, 400)

    def test_last_point_is_at_or_below_max(self):
        assert side_scan_list(100, 950, 100) == (100, 200, 300, 400, 500, 600, 700, 800, 900)

    def test_empty_range_is_an_error_not_an_empty_scan(self):
        with pytest.raises(ValueError) as excinfo:
            side_scan_list(100, 50, 100)
        assert "最大边长" in str(excinfo.value)

    def test_bad_step_is_refused(self):
        with pytest.raises(ValueError):
            side_scan_list(100, 400, 0)


# ---------------------------------------------------------------------------
# 近 1000 选点
# ---------------------------------------------------------------------------
class TestSelectNearSide:
    def rows(self):
        return [
            {"side_px": 800, "score": 0.60},
            {"side_px": 900, "score": 0.75},
            {"side_px": 1000, "score": 0.70},
            {"side_px": 1100, "score": 0.65},
            {"side_px": 1300, "score": 0.99},  # 区间外, 不得被选中
        ]

    def test_picks_the_best_inside_the_window(self):
        picked = select_near_side(self.rows(), target_px=1000, tolerance_px=200)
        assert picked["side_px"] == 900

    def test_a_higher_score_outside_the_window_is_ignored(self):
        """不能让「全局最优」偷偷替换掉近 1000 的选点。"""
        assert select_near_side(self.rows(), target_px=1000, tolerance_px=50)["side_px"] == 1000

    def test_rank_key_is_honoured(self):
        rows = self.rows()
        for row in rows:
            row["quality"] = 1.0 - row["score"]
        # 窗口内 quality 最高的是 800 (对应 score 最低的那个)。
        assert select_near_side(rows, target_px=1000, tolerance_px=200, key="quality")[
            "side_px"
        ] == 800

    def test_non_finite_rows_are_skipped(self):
        rows = [
            {"side_px": 900, "score": float("nan")},
            {"side_px": 1000, "score": 0.4},
        ]
        assert select_near_side(rows, target_px=1000, tolerance_px=200)["side_px"] == 1000

    def test_no_candidate_raises_with_the_real_list(self):
        rows = [{"side_px": 100, "score": 0.9}, {"side_px": 200, "score": 0.8}]
        with pytest.raises(NoNearSideError) as excinfo:
            select_near_side(rows, target_px=1000, tolerance_px=200)
        message = str(excinfo.value)
        assert "[800, 1200]" in message
        assert "100" in message and "200" in message, "异常里必须带上实际边长列表"

    def test_exit_code_is_its_own(self):
        assert EXIT_NO_NEAR_SIDE not in (0, 2, 3, 4)


# ---------------------------------------------------------------------------
# 阶数 / 网格
# ---------------------------------------------------------------------------
class TestOrders:
    def test_mode_count_matches_the_canonical_lists(self):
        for n_max in (1, 4, 7, 11, 15, 20, 25, 30):
            expected = (n_max + 1) * (n_max + 2) // 2
            assert modes_for_order(n_max) == expected
            assert len(noll_indices(n_max)) == expected
            assert len(zernike_modes(n_max)) == expected

    def test_range_syntax(self):
        assert parse_orders_spec("7..30") == tuple(range(30, 6, -1))
        assert len(parse_orders_spec("7..30")) == 24

    def test_dash_and_comma_and_dedup(self):
        assert parse_orders_spec("7-9") == (9, 8, 7)
        assert parse_orders_spec("7,7,9") == (9, 7)

    def test_descending_order_preserved(self):
        orders = parse_orders_spec("7..30")
        assert orders[0] == 30 and orders[-1] == 7

    def test_out_of_range_is_refused(self):
        with pytest.raises(ValueError):
            parse_orders_spec("7..31")
        with pytest.raises(ValueError):
            parse_orders_spec("0")

    def test_garbage_is_refused(self):
        with pytest.raises(ValueError):
            parse_orders_spec("abc")
        with pytest.raises(ValueError):
            parse_orders_spec("  ")


class TestFrameParsing:
    def test_frame_shape(self):
        assert parse_frame_shape("1944x2592") == (1944, 2592)
        assert parse_frame_shape("1944,2592") == (1944, 2592)
        with pytest.raises(ValueError):
            parse_frame_shape("1944")

    def test_xy(self):
        assert parse_xy("930,1050") == (930.0, 1050.0)
        with pytest.raises(ValueError):
            parse_xy("930")


# ---------------------------------------------------------------------------
# 闪耀光栅: 周期 → 预测位移
# ---------------------------------------------------------------------------
class TestBlaze:
    def bench(self) -> Bench:
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH])
        return build_bench(args)

    def test_shift_magnitude_is_the_bench_scale_over_period(self):
        table = blaze_shift_table(
            (16, 32, 128), 1064.0, (1944, 2592), side_px=1000.0, center=(1296.0, 972.0)
        )
        for row in table:
            assert row["predicted_shift_px"] == pytest.approx(
                TILT_SHIFT_SCALE / row["blaze_period_px"], rel=1e-9
            )

    def test_panel_x_ramp_moves_the_spot_along_camera_y(self):
        """面板↔相机 90° 交换: 面板 x 斜坡 → 相机 **y** 位移, x 分量为 0。"""
        table = blaze_shift_table(
            (16,), 1064.0, (1944, 2592), side_px=1000.0, center=(1296.0, 972.0)
        )
        assert table[0]["predicted_shift_x_px"] == 0.0
        assert table[0]["predicted_shift_y_px"] > 0.0

    def test_shorter_period_shifts_further(self):
        table = blaze_shift_table(
            (16, 32, 64), 1064.0, (1944, 2592), side_px=1000.0, center=(1296.0, 972.0)
        )
        shifts = [row["predicted_shift_px"] for row in table]
        assert shifts == sorted(shifts, reverse=True)

    def test_box_that_leaves_the_frame_is_flagged(self):
        # 0 级偏下: 周期 16 的 462 px 位移把 600 px 的方框推出底边, 周期 128 不会。
        table = blaze_shift_table(
            (16, 128), 1064.0, (1944, 2592), side_px=600.0, center=(1296.0, 1300.0)
        )
        flags = {row["blaze_period_px"]: row["fits_frame"] for row in table}
        assert flags[16] is False
        assert flags[128] is True

    def test_box_fits(self):
        assert box_fits((100.0, 100.0), 50.0, (300, 320)) is True
        assert box_fits((10.0, 100.0), 50.0, (300, 320)) is False
        assert box_fits((150.0, 150.0), 300.0, (300, 320)) is False


# ---------------------------------------------------------------------------
# GS → 面板相位 (整条换算链)
# ---------------------------------------------------------------------------
class TestGsMapping:
    @pytest.fixture(scope="class")
    def bench(self) -> Bench:
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH])
        return build_bench(args)

    def test_one_gs_sample_is_a_fixed_number_of_camera_pixels(self, bench):
        """``Δx_cam = TILT_SHIFT_SCALE / pupil_side`` —— 与网格边长**无关**。"""
        assert bench.pupil_side_px == 900
        assert bench.camera_px_per_gs_px == pytest.approx(TILT_SHIFT_SCALE / 900.0)

    def test_side_conversion_is_linear(self, bench):
        assert bench.side_cam_to_gs(1000.0) == pytest.approx(1000.0 / (TILT_SHIFT_SCALE / 900.0))
        assert bench.side_cam_to_gs(2 * 1000.0) == pytest.approx(2 * bench.side_cam_to_gs(1000.0))

    def test_grid_grows_with_the_span_and_is_a_power_of_two(self, bench):
        grid = bench.gs_grid_for(3000.0)
        assert grid & (grid - 1) == 0, "网格边长必须是 2 的幂"
        assert grid * bench.camera_px_per_gs_px >= 3000.0
        assert bench.gs_grid_for(3000.0) >= bench.gs_grid_for(1000.0)

    def test_gs_phase_concentrates_energy_in_the_requested_box(self, bench):
        """端到端: 边长 → GS 网格像素 → 方斑模型远场。

        这一条把「边长 → 网格像素」的换算钉住: 换算偏了, 模型侧的 PIB 会立刻掉下来。
        """
        result = run_gs_square(400.0, bench, n_grid=128, iterations=60)
        amp = np.asarray(result.amplitude)
        assert amp.shape == (128, 128)
        n = 128
        side_gs = int(round(bench.side_cam_to_gs(400.0)))
        assert side_gs == 49  # 400 / 8.222
        from ao_shaping.algorithm.signal_processing.square_shaping import (
            power_in_bucket,
            uniformity_cv,
        )
        from ao_shaping.utils.image.targets import square_mask

        mask = square_mask((n, n), n // 2, n // 2, side_gs)
        intensity = amp**2
        assert power_in_bucket(intensity, mask) > 0.7
        assert 0.02 < uniformity_cv(intensity, mask) < 0.6

    def test_gs_is_deterministic(self, bench):
        """初值 = 目标的反向传播 ⇒ 没有随机数, 两次结果逐位相同。"""
        first = run_gs_square(300.0, bench, n_grid=64, iterations=20)
        second = run_gs_square(300.0, bench, n_grid=64, iterations=20)
        assert np.array_equal(first.phase, second.phase)

    def test_panel_phase_is_raw_and_inside_the_panel(self, bench):
        result = run_gs_square(300.0, bench, n_grid=64, iterations=20)
        panel = pupil_phase_to_panel(result.phase, bench)
        assert panel.shape == bench.panel_shape
        assert np.isfinite(panel).all()
        cx, cy = bench.beam_center_panel
        assert abs(float(panel[cy, cx])) > 0.0, "照明中心处必须有相位"
        assert float(np.abs(panel[0, 0])) == 0.0, "瞳面方框之外必须为 0"

    def test_gs_time_model_is_monotone(self):
        assert gs_seconds(512, 200) > gs_seconds(256, 200) > 0.0
        assert gs_seconds(512, 400) == pytest.approx(2 * gs_seconds(512, 200))


# ---------------------------------------------------------------------------
# 指标口径
# ---------------------------------------------------------------------------
class TestMeasureBox:
    def test_a_uniform_square_scores_well(self):
        frame = uniform_square_frame(cy=150, cx=170, side=100, level=50.0)
        metrics = measure_box(frame, (170.0, 150.0), 100.0)
        assert metrics["cv"] < 0.05
        assert metrics["pib"] > 0.9
        assert metrics["ar"] == pytest.approx(1.0, abs=0.05)
        assert 0.9 < metrics["score"] <= 1.0

    def test_centroid_is_measured_inside_the_box_not_on_the_frame(self):
        """整帧最亮的可能是静止的 0 级; 质心必须在**框内**算。"""
        frame = uniform_square_frame(cy=60, cx=70, side=40, level=50.0)
        # 画一个比方斑亮得多的静止 0 级, 位置远离方框。
        frame[250, 280] += 5000.0
        metrics = measure_box(frame, (70.0, 60.0), 40.0)
        assert metrics["box_centroid_x"] == pytest.approx(70.0, abs=2.0)
        assert metrics["box_centroid_y"] == pytest.approx(60.0, abs=2.0)
        assert metrics["frame_peak"] > metrics["box_peak"]

    def test_a_box_hanging_off_the_frame_reports_the_clipped_area(self):
        frame = uniform_square_frame(cy=20, cx=20, side=100)
        metrics = measure_box(frame, (20.0, 20.0), 100.0)
        assert metrics["box_clipped_px"] > 0
        assert metrics["box_area_px"] < 100 * 100

    def test_background_subtraction_keeps_pib_at_or_below_one(self):
        """raw 帧里的对称读出噪声会把 PIB 顶到 1 以上 —— 指标必须用过背景的帧。"""
        rng = np.random.default_rng(3)
        frame = rng.normal(0.0, 1.0, (400, 400))
        frame[200:260, 200:260] += 20.0
        metrics = measure_box(frame, (230.0, 230.0), 60.0)
        assert 0.0 <= metrics["pib"] <= 1.0


# ---------------------------------------------------------------------------
# 计划 (dry-run)
# ---------------------------------------------------------------------------
class TestPlan:
    def test_plan_carries_everything_the_dry_run_must_show(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH])
        bench = build_bench(args)
        plan = build_plan(args, bench)
        scan = plan["scan"]
        assert scan["sides"][0] == 100
        assert scan["sides"] == list(range(100, scan["sides"][-1] + 1, 100))
        assert scan["near_candidates"] == [800, 900, 1000, 1100, 1200]
        assert len(scan["orders"]) == 24
        assert scan["blaze_periods"] == [16, 24, 32, 40, 48, 64, 96, 128]
        assert plan["frame"]["side_max"] > 0
        assert plan["stage_plans"]["sides"]["n_configs"] == len(scan["sides"])
        assert plan["estimate"]["total_s"] > 0.0

    def test_explicit_side_max_is_clamped_by_the_frame(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--side-max", "5000"])
        plan = build_plan(args, build_bench(args))
        assert plan["frame"]["side_max"] < 5000
        assert "限制" in plan["frame"]["side_max_note"]

    def test_assumed_zero_order_shows_up_as_an_assumption(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH])
        assert build_plan(args, build_bench(args))["frame"]["zero_order_assumed"] is True
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--dry-run-zero-order", "900,1100"])
        plan = build_plan(args, build_bench(args))
        assert plan["frame"]["zero_order_assumed"] is False
        assert plan["frame"]["zero_order_xy"] == [900.0, 1100.0]

    def test_off_centre_zero_order_shrinks_the_max_side(self):
        """0 级离左边只有 200 px ⇒ 可用半径 190 ⇒ 原始 380, 取到步长整数倍 = 300。"""
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--dry-run-zero-order", "200,1000"])
        plan = build_plan(args, build_bench(args))
        assert plan["frame"]["side_max_raw"] == 380
        assert plan["frame"]["side_max"] == 300

    def test_a_single_stage_plan_only_lists_that_stage(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--scan", "sides"])
        plan = build_plan(args, build_bench(args))
        assert plan["stages"] == ["sides"]
        assert set(plan["stage_plans"]) == {"sides"}

    def test_explicit_gs_grid_too_small_is_refused(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--gs-grid", "64"])
        with pytest.raises(SystemExit) as excinfo:
            build_plan(args, build_bench(args))
        assert "卷绕" in str(excinfo.value)

    def test_bad_blaze_period_is_refused(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--blaze-periods", "1", "16"])
        with pytest.raises(SystemExit):
            build_plan(args, build_bench(args))


# ---------------------------------------------------------------------------
# 纪律
# ---------------------------------------------------------------------------
class TestDiscipline:
    def test_source_has_no_random_gradient_iteration(self):
        """本脚本只用 GS (交替投影), 不含任何黑盒随机梯度迭代。"""
        source = SCRIPT.read_text(encoding="utf-8")
        assert "spgd" not in source.lower(), "本脚本不允许出现随机梯度迭代"
        assert "gerchberg_saxton" in source

    def test_source_reuses_the_canonical_imports(self):
        """禁止另写一套: GS / 指标 / 光栅 / Zernike 都必须 import 仓库实现。"""
        source = SCRIPT.read_text(encoding="utf-8")
        for needed in (
            "from ao_shaping.algorithm.signal_processing.gerchberg_saxton import",
            "from ao_shaping.algorithm.signal_processing.square_shaping import",
            "from ao_shaping.utils.wavefront.zero_order_overlay import",
            "from ao_shaping.utils.wavefront.zernike_calc import",
            "from ao_shaping.utils.slm.phase_embed import",
            "from ao_shaping.utils.image.resample import",
        ):
            assert needed in source, f"缺少对仓库既有实现的复用: {needed}"

    def test_dry_run_runs_without_hardware_and_prints_the_plan(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run", *_DRY_PITCH],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        for required in (
            "最大边长",
            "实际边长列表",
            "窗口内候选",
            "阶数 → 模式数",
            "预测位移",
            "预计耗时",
            "dry-run 保证通过",
        ):
            assert required in out, f"dry-run 输出缺少 {required!r}"
        assert "sys.modules 里没有任何 ao_shaping.drivers" in out

    def test_atomic_mode_is_advertised_in_the_whole_segment_plan(self):
        """整段计划里必须写出原子模式的推荐用法 (否则没人会知道有这条路)。"""
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run", *_DRY_PITCH],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--plan-commands" in proc.stdout
        assert "一个进程只跑一组参数" in proc.stdout


# ---------------------------------------------------------------------------
# 原子模式: 配置身份
# ---------------------------------------------------------------------------
class TestAtomicSelection:
    def test_slug_is_one_config_identity_per_stage(self):
        assert config_slug("sides", 700) == "side_700"
        assert config_slug("zernike", 11) == "nmax_11"
        assert config_slug("blaze", 32) == "blaze_32"
        # 同一个值 → 同一个 slug (去重键必须稳定)
        assert config_slug("sides", 700.0) == config_slug("sides", 700)

    def test_each_stage_picks_up_its_own_value_selector(self):
        cases = (
            (["--dry-run", *_DRY_PITCH, "--scan", "sides", "--side-px", "700"], "sides", 700),
            (["--dry-run", *_DRY_PITCH, "--scan", "zernike", "--n-max", "11"], "zernike", 11),
            (["--dry-run", *_DRY_PITCH, "--scan", "blaze", "--blaze-period", "32"], "blaze", 32),
        )
        for argv, stage, value in cases:
            args = build_parser().parse_args(argv)
            picked = atomic_selection(args, (stage,))
            assert picked == {
                "stage": stage, "value": value,
                "variable": {"sides": "side_px", "zernike": "n_max",
                             "blaze": "blaze_period_px"}[stage],
            }

    def test_whole_segment_has_no_atomic_selector(self):
        """不给值选择器 = 旧版整段路径 (保留, 不是原子模式)。"""
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--scan", "sides"])
        assert atomic_selection(args, ("sides",)) is None
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--scan", "all"])
        assert atomic_selection(args, ("sides", "zernike", "blaze")) is None
        # --scan all + --side-px 仍是「显式覆盖」, 不是原子模式
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--scan", "all", "--side-px", "700"])
        assert atomic_selection(args, ("sides", "zernike", "blaze")) is None

    def test_plan_records_the_atomic_config(self):
        args = build_parser().parse_args(
            ["--dry-run", *_DRY_PITCH, "--scan", "zernike", "--n-max", "11", "--run-root", "x"]
        )
        plan = build_plan(args, build_bench(args))
        assert plan["atomic"]["stage"] == "zernike"
        assert plan["atomic"]["value"] == 11
        assert plan["stages"] == ["zernike"]

    def test_out_of_range_values_are_refused(self):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--scan", "zernike", "--n-max", "31"])
        with pytest.raises(SystemExit):
            atomic_selection(args, ("zernike",))
        args = build_parser().parse_args(
            ["--dry-run", *_DRY_PITCH, "--scan", "blaze", "--blaze-period", "1"]
        )
        with pytest.raises(SystemExit):
            atomic_selection(args, ("blaze",))
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--scan", "sides", "--side-px", "0"])
        with pytest.raises(SystemExit):
            atomic_selection(args, ("sides",))

    def test_atomic_runs_need_an_explicit_batch_root(self, capsys):
        """每个进程各取一个时间戳 = 同一批散在多个目录; 所以必须显式给根。"""
        bare = argparse.Namespace(run_root=None, out_root=None)
        with pytest.raises(SystemExit) as excinfo:
            require_accumulation_root(bare)
        assert excinfo.value.code == 2
        assert "--run-root" in capsys.readouterr().err
        given = argparse.Namespace(run_root="data/x", out_root=None)
        assert require_accumulation_root(given) == ROOT / "data" / "x"


# ---------------------------------------------------------------------------
# 命令清单 (--plan-commands)
# ---------------------------------------------------------------------------
class TestCommandPlan:
    def plan(self, argv=()):
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--run-root", "data/demo", *argv])
        plan_args = argparse.Namespace(**vars(args))
        plan_args.scan = "all"
        plan_args.side_px = None
        plan_args.n_max = None
        plan_args.blaze_period = None
        return build_plan(plan_args, build_bench(plan_args))

    def test_line_counts_match_the_config_counts(self):
        plan = self.plan()
        lines = command_plan_lines(plan, build_parser().parse_args(["--run-root", "data/demo"]))
        assert len(lines) == 19 + 1 + 24 + 8 + 1 == 53
        assert sum("--scan sides" in line for line in lines) == 19
        assert sum("--scan select" in line for line in lines) == 1
        assert sum("--scan zernike" in line for line in lines) == 24
        assert sum("--scan blaze" in line for line in lines) == 8
        assert sum("--scan summarize" in line for line in lines) == 1

    def test_dependency_order(self):
        """先 sides, 再 select, 然后 zernike, 再 blaze, 最后 summarize。"""
        plan = self.plan()
        args = build_parser().parse_args(["--run-root", "data/demo"])
        lines = command_plan_lines(plan, args)
        kinds = [
            next(k for k in ("sides", "select", "zernike", "blaze", "summarize")
                 if f"--scan {k}" in line)
            for line in lines
        ]
        blocks = [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k]
        assert blocks == ["sides", "select", "zernike", "blaze", "summarize"]
        # 值按各自声明的顺序 (sides 升序 / 阶数降序 / 周期升序)
        sides = [int(line.split("--side-px ")[1].split()[0]) for line in lines[:19]]
        assert sides == sorted(sides) and sides[0] == 100 and sides[-1] == 1900
        orders = [int(line.split("--n-max ")[1].split()[0]) for line in lines[20:44]]
        assert orders == sorted(orders, reverse=True) and orders[0] == 30 and orders[-1] == 7
        periods = [int(line.split("--blaze-period ")[1].split()[0]) for line in lines[44:52]]
        assert periods == [16, 24, 32, 40, 48, 64, 96, 128]

    def test_every_line_carries_the_shared_args(self):
        """逐行复制即等价: 相机/曝光/根目录/GS 这些必须在**每一行**上。"""
        plan = self.plan(["--cam-type", "daheng", "--exposure-ms", "20"])
        args = build_parser().parse_args(
            ["--run-root", "data/demo", "--cam-type", "daheng", "--exposure-ms", "20"]
        )
        for line in command_plan_lines(plan, args):
            assert line.startswith("python scripts/slm/hw_gs_square_scans.py")
            for needed in (
                "--run-root data/demo",
                "--cam-type daheng",
                "--exposure-ms 20",
                "--gs-iters 200",
                "--gs-grid auto",
                "--zernike-fit-grid 256",
            ):
                assert needed in line, f"命令缺 {needed!r}: {line}"

    def test_side_step_changes_the_sides_line_count(self):
        """行数必须与**实际配置数**逐值一致 (不是写死的 19)。"""
        plan = self.plan(["--side-step", "20"])
        args = build_parser().parse_args(["--run-root", "data/demo", "--side-step", "20"])
        lines = command_plan_lines(plan, args)
        sides_lines = [line for line in lines if "--scan sides" in line]
        assert len(sides_lines) == len(plan["scan"]["sides"]) == 92
        assert len(sides_lines) != 19
        assert "--side-px 1920" in sides_lines[-1]

    def test_zernike_and_blaze_lines_do_not_override_the_selection(self):
        """它们的值选择器是 --n-max / --blaze-period; 带上 --side-px 会绕过 selected.json。"""
        plan = self.plan()
        args = build_parser().parse_args(["--run-root", "data/demo"])
        for line in command_plan_lines(plan, args):
            if "--scan zernike" in line or "--scan blaze" in line:
                assert "--side-px" not in line

    def test_subprocess_prints_the_same_number_of_lines(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--plan-commands", *_DRY_PITCH, "--run-root", "data/demo"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        commands = [ln for ln in proc.stdout.splitlines() if ln.startswith("python ")]
        assert len(commands) == 53
        assert commands[0].startswith("python scripts/slm/hw_gs_square_scans.py --execute")
        assert "--scan select" in commands[19]
        assert "--scan summarize" in commands[-1]
        assert "dry-run 保证通过" in proc.stdout
        assert "分批建议" in proc.stdout


# ---------------------------------------------------------------------------
# 跨进程累积: 去重 / 排序 / 原子写
# ---------------------------------------------------------------------------
def _row(**extra):
    base = {
        "config": "side_700", "side_px": 700, "cv": 0.4, "pib": 0.8,
        "score": 0.7, "preflight": {"zero_order_xy_frozen": [10.0, 20.0]},
    }
    base.update(extra)
    return base


class TestAccumulation:
    def test_upsert_is_idempotent_per_config(self, tmp_path):
        stage = tmp_path / "sides"
        upsert_measurements(stage, _row(), stage="sides")
        upsert_measurements(stage, _row(cv=0.5), stage="sides")
        rows = list(csv.DictReader((stage / "measurements.csv").open(encoding="utf-8")))
        assert len(rows) == 1, "同一配置重跑必须就地替换, 不能多一行"
        assert float(rows[0]["cv"]) == 0.5, "留下的应当是**新**那一行"
        assert json.loads((stage / "measurements.json").read_text(encoding="utf-8"))[0][
            "config"
        ] == "side_700"

    def test_other_configs_are_untouched(self, tmp_path):
        stage = tmp_path / "sides"
        for side in (700, 900, 1000):
            upsert_measurements(
                stage, _row(config=f"side_{side}", side_px=side), stage="sides"
            )
        upsert_measurements(stage, _row(cv=0.9), stage="sides")
        rows = list(csv.DictReader((stage / "measurements.csv").open(encoding="utf-8")))
        assert [r["config"] for r in rows] == ["side_700", "side_900", "side_1000"]
        assert float(rows[1]["cv"]) == 0.4, "别的配置不该被改"

    def test_rows_are_sorted_by_the_variable_not_by_write_order(self, tmp_path):
        stage = tmp_path / "sides"
        for side in (1000, 700, 900):
            upsert_measurements(
                stage, _row(config=f"side_{side}", side_px=side), stage="sides"
            )
        rows = list(csv.DictReader((stage / "measurements.csv").open(encoding="utf-8")))
        assert [int(r["side_px"]) for r in rows] == [700, 900, 1000]

    def test_nested_fields_stay_out_of_the_csv(self):
        flat = flatten_row(_row())
        assert "preflight" not in flat
        assert flat["config"] == "side_700"
        assert isinstance(flat["cv"], float)

    def test_manifest_replaces_the_same_group(self, tmp_path):
        entry = {
            "group": "sides", "variable": "side_px", "meta": {}, "dir": "sides",
            "n_rows": 1, "created_utc": "2026-10-23T00:00:00+00:00",
        }
        update_manifest_atomic(tmp_path, dict(entry))
        update_manifest_atomic(tmp_path, {**entry, "n_rows": 5})
        other = {**entry, "group": "zernike", "variable": "n_max", "dir": "zernike"}
        update_manifest_atomic(tmp_path, other)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert [e["group"] for e in manifest] == ["sides", "zernike"]
        assert manifest[0]["n_rows"] == 5, "同名段就地替换"

    def test_json_write_is_atomic_and_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "x.json"
        write_json_atomic(target, {"a": 1})
        write_json_atomic(target, {"a": 2})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
        assert list(target.parent.glob("*.tmp*")) == [], "不得留下临时文件"

    def test_a_corrupt_manifest_does_not_block_new_entries(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
        update_manifest_atomic(
            tmp_path,
            {"group": "blaze", "variable": "blaze_period_px", "meta": {}, "dir": "blaze",
             "n_rows": 1, "created_utc": "2026-10-23T00:00:00+00:00"},
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert [e["group"] for e in manifest] == ["blaze"]

    def test_config_artifacts_write_metrics_last(self, tmp_path):
        cfg = tmp_path / "sides" / "side_700"
        write_config_artifacts(
            cfg,
            phase_panel=np.zeros((8, 10)),
            pupil_phase=np.ones((4, 4)),
            frame=None,
            row=_row(),
        )
        assert (cfg / "phase_000.npy").is_file()
        assert (cfg / CONFIG_PUPIL_PHASE_NAME).is_file()
        assert (cfg / "metrics.json").is_file()
        assert np.asarray(np.load(cfg / CONFIG_PUPIL_PHASE_NAME)).shape == (4, 4)
        assert json.loads((cfg / "metrics.json").read_text(encoding="utf-8"))["config"] == "side_700"


# ---------------------------------------------------------------------------
# selected.json: 第 2/3 段的唯一输入
# ---------------------------------------------------------------------------
def _sides_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _select_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        ["--scan", "select", "--run-root", str(tmp_path), *extra]
    )


class TestSelectedJson:
    def test_roundtrip_keeps_the_side_and_the_phase_path(self, tmp_path):
        phase = tmp_path / "sides" / "side_900" / CONFIG_PUPIL_PHASE_NAME
        phase.parent.mkdir(parents=True)
        np.save(phase, np.zeros((8, 8)))
        save_selection(
            tmp_path,
            {"side_px": 900, "gs_grid": 8, "gs_iters": 200,
             "phase_path": "sides/side_900/pupil_phase.npy"},
        )
        picked = load_selection(tmp_path)
        assert picked.side_px == 900
        assert picked.source == "selected.json"
        assert picked.gs_grid == 8
        assert picked.phase_path == phase, "相对路径要按 run-root 解析, 不是按 cwd"

    def test_missing_file_names_the_command_to_run(self, tmp_path):
        with pytest.raises(SelectionMissing) as excinfo:
            load_selection(tmp_path)
        message = str(excinfo.value)
        assert "--scan select" in message
        assert str(tmp_path) in message

    def test_corrupt_file_is_refused_not_defaulted(self, tmp_path):
        (tmp_path / "sides").mkdir(parents=True)
        (tmp_path / "sides" / "selected.json").write_text("{oops", encoding="utf-8")
        with pytest.raises(SelectionMissing):
            load_selection(tmp_path)
        (tmp_path / "sides" / "selected.json").write_text(
            json.dumps({"side_px": 0}), encoding="utf-8"
        )
        with pytest.raises(SelectionMissing):
            load_selection(tmp_path)

    def test_cli_override_marks_itself_as_not_a_selection(self, tmp_path):
        picked = load_selection(tmp_path, side_px_override=1234)
        assert picked.side_px == 1234
        assert picked.source == "cli"
        assert picked.phase_path is None


class TestSelectCommand:
    def rows(self):
        return [
            {"config": "side_800", "side_px": 800, "score": 0.60, "quality": 0.5,
             "gs_grid": 512, "gs_iters": 200, "side_ok": 1},
            {"config": "side_900", "side_px": 900, "score": 0.75, "quality": 0.4,
             "gs_grid": 512, "gs_iters": 200},
            {"config": "side_1000", "side_px": 1000, "score": 0.70, "quality": 0.3,
             "gs_grid": 512, "gs_iters": 200},
            {"config": "side_1300", "side_px": 1300, "score": 0.99, "quality": 0.9,
             "gs_grid": 512, "gs_iters": 200},
        ]

    def test_picks_the_best_inside_the_window(self, tmp_path):
        _sides_csv(tmp_path / "sides" / "measurements.csv", self.rows())
        path = run_select(_select_args(tmp_path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["side_px"] == 900, "窗口外那个 0.99 不得被选中"
        assert payload["rank_by"] == "score"
        assert payload["gs_grid"] == 512
        assert payload["phase_path"].endswith("side_900/pupil_phase.npy")
        assert payload["n_sides_rows"] == 4

    def test_rank_key_is_honoured(self, tmp_path):
        _sides_csv(tmp_path / "sides" / "measurements.csv", self.rows())
        payload = json.loads(
            run_select(_select_args(tmp_path, "--rank-by", "quality")).read_text(
                encoding="utf-8"
            )
        )
        assert payload["side_px"] == 1300 or payload["side_px"] == 800
        assert payload["rank_by"] == "quality"

    def test_no_candidate_in_the_window_keeps_its_own_exit_code(self, tmp_path):
        _sides_csv(
            tmp_path / "sides" / "measurements.csv",
            [{"config": "side_100", "side_px": 100, "score": 0.9}],
        )
        with pytest.raises(SystemExit) as excinfo:
            run_select(_select_args(tmp_path))
        assert excinfo.value.code == EXIT_NO_NEAR_SIDE

    def test_empty_csv_is_a_different_error(self, tmp_path):
        (tmp_path / "sides").mkdir(parents=True)
        with pytest.raises(SystemExit) as excinfo:
            run_select(_select_args(tmp_path))
        assert excinfo.value.code == EXIT_NO_MEASUREMENTS

    def test_selected_json_is_never_written_on_failure(self, tmp_path):
        _sides_csv(
            tmp_path / "sides" / "measurements.csv",
            [{"config": "side_100", "side_px": 100, "score": 0.9}],
        )
        with pytest.raises(SystemExit):
            run_select(_select_args(tmp_path))
        assert not (tmp_path / "sides" / "selected.json").exists()


# ---------------------------------------------------------------------------
# 基准相位: 缓存复用 vs 重算
# ---------------------------------------------------------------------------
class TestBasePhaseCache:
    def bench(self) -> Bench:
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH])
        return build_bench(args)

    def selection(self, side: int = 900, grid: int = 8) -> Selection:
        return Selection(
            side_px=side, source="selected.json", phase_path=None,
            gs_grid=grid, gs_iters=20, rank_by="score", score=0.7,
        )

    def write_cache(self, tmp_path: Path, side: int, array: np.ndarray) -> Path:
        path = tmp_path / "sides" / f"side_{side}" / CONFIG_PUPIL_PHASE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        return path

    def test_a_matching_cache_is_used_and_reported(self, tmp_path):
        array = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        self.write_cache(tmp_path, 900, array)
        phase, source, info = _resolve_base_pupil(
            tmp_path, self.bench(), self.selection(), gs_grid=8, gs_iters=20
        )
        assert source == "cache"
        assert np.array_equal(phase, array), "缓存必须**逐位**照用, 不做任何再处理"
        assert str(tmp_path) in info["loaded"]

    def test_cache_hit_does_not_run_gs(self, tmp_path, monkeypatch):
        """GS 是确定性的 ⇒ 命中缓存时不该再算一次 (那 7.5 s 是白花的)。"""
        import scripts.slm.hw_gs_square_scans as mod

        self.write_cache(tmp_path, 900, np.zeros((8, 8)))

        def boom(*args, **kwargs):
            raise AssertionError("命中缓存时不得调用 run_gs_square")

        monkeypatch.setattr(mod, "run_gs_square", boom)
        _, source, _ = _resolve_base_pupil(
            tmp_path, self.bench(), self.selection(), gs_grid=8, gs_iters=20
        )
        assert source == "cache"

    def test_shape_mismatch_forces_a_recompute(self, tmp_path, monkeypatch):
        import scripts.slm.hw_gs_square_scans as mod

        self.write_cache(tmp_path, 900, np.zeros((4, 4)))  # 与 gs_grid=8 不符
        calls = []

        def fake_gs(side_cam_px, bench, *, n_grid, iterations):
            calls.append((float(side_cam_px), int(n_grid)))
            return SimpleNamespace(phase=np.ones((int(n_grid), int(n_grid))))

        monkeypatch.setattr(mod, "run_gs_square", fake_gs)
        phase, source, info = _resolve_base_pupil(
            tmp_path, self.bench(), self.selection(), gs_grid=8, gs_iters=20
        )
        assert source == "recomputed"
        assert calls == [(900.0, 8)]
        assert phase.shape == (8, 8)
        assert "不符" in info["reason"]

    def test_missing_cache_recomputes_and_says_why(self, tmp_path, monkeypatch):
        import scripts.slm.hw_gs_square_scans as mod

        monkeypatch.setattr(
            mod, "run_gs_square",
            lambda side, bench, *, n_grid, iterations: SimpleNamespace(
                phase=np.zeros((int(n_grid), int(n_grid)))
            ),
        )
        _, source, info = _resolve_base_pupil(
            tmp_path, self.bench(), self.selection(), gs_grid=8, gs_iters=20
        )
        assert source == "recomputed"
        assert info["reason"] == "文件不存在"

    def test_non_finite_cache_is_refused(self, tmp_path):
        array = np.zeros((8, 8))
        array[0, 0] = np.nan
        path = self.write_cache(tmp_path, 900, array)
        loaded, info = load_cached_pupil_phase(tmp_path, 900, gs_grid=8, phase_path=path)
        assert loaded is None
        assert "非有限" in info["reason"]

    def test_gs_grid_prefers_the_recorded_one(self):
        """原子模式的网格: 显式 > 第 1 段记录的 > 按窗口反解。"""
        bench = self.bench()
        args = build_parser().parse_args(["--dry-run", *_DRY_PITCH])
        assert atomic_gs_grid(args, bench, side_max_px=1000, max_shift_px=0, hint=1024) == 1024
        auto = atomic_gs_grid(args, bench, side_max_px=1000, max_shift_px=0)
        assert auto == bench.gs_grid_for(1000)
        explicit = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--gs-grid", "2048"])
        assert atomic_gs_grid(explicit, bench, side_max_px=1000, max_shift_px=0) == 2048
        too_small = build_parser().parse_args(["--dry-run", *_DRY_PITCH, "--gs-grid", "64"])
        with pytest.raises(SystemExit):
            atomic_gs_grid(too_small, bench, side_max_px=1000, max_shift_px=0)


# ---------------------------------------------------------------------------
# 原子模式的 dry-run 与退出码
# ---------------------------------------------------------------------------
class TestAtomicDryRun:
    def test_one_config_only(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run", *_DRY_PITCH, "--scan", "sides",
             "--side-px", "700", "--run-root", "data/demo"],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        for required in (
            "原子模式计划", "side_700", "side_gs_px", "GS 网格", "曝光请求",
            "预计耗时", "dry-run 保证通过",
        ):
            assert required in out, f"原子 dry-run 缺少 {required!r}"
        assert "实际边长列表" not in out, "原子模式不该列整段网格"

    def test_zernike_plan_states_the_dependency(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run", *_DRY_PITCH, "--scan", "zernike",
             "--n-max", "11", "--run-root", "data/demo"],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "selected.json" in proc.stdout
        assert "--scan select" in proc.stdout
        assert "78 个模式" in proc.stdout

    def test_blaze_plan_shows_the_predicted_shift(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run", *_DRY_PITCH, "--scan", "blaze",
             "--blaze-period", "32", "--run-root", "data/demo"],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "预测位移" in proc.stdout
        assert f"{TILT_SHIFT_SCALE / 32:.1f}" in proc.stdout

    def test_missing_selection_exits_with_its_own_code(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--execute", "--cam-type", "sim",
             "--scan", "zernike", "--n-max", "11", "--run-root", str(tmp_path)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        assert proc.returncode == EXIT_SELECTION_MISSING, proc.stderr
        combined = proc.stdout + proc.stderr
        assert "--scan select" in combined
        assert "selected.json" in combined
        assert not (tmp_path / "zernike" / "nmax_11" / "metrics.json").exists()


# ---------------------------------------------------------------------------
# 端到端 (数字孪生): 一个进程一组参数 → 累积 → 复用缓存相位
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestAtomicChainInSim:
    """每个进程只跑**一组**参数, 三条命令串起来得到一份可用的累积产物。

    这是唯一会**开设备**的测试 —— ``--cam-type sim`` (数字孪生, 不碰任何硬件)。它验的
    是原子化本身: 一个进程一组、CSV 不重复、第 2 段复用第 1 段存盘的 pupil 相位。
    """

    def run_cli(self, *argv: str):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )

    def require_sim_backend(self, proc) -> None:
        if proc.returncode != 0 and "ModuleNotFoundError" in (proc.stderr or ""):
            pytest.skip(f"数字孪生后端不可用: {proc.stderr.strip().splitlines()[-1:]}")

    def test_sides_then_select_then_zernike_reuses_the_cached_phase(self, tmp_path):
        root = tmp_path / "run"
        # 仿真画幅小 (0 级偏下), 所以选点目标放低 —— 用的是与正式口径**同一套参数**,
        # 只是数值落在仿真能装下的窗口里。
        near = ["--target-side-near", "300", "--near-tolerance", "100"]

        sides = self.run_cli(
            "--execute", "--cam-type", "sim", "--scan", "sides",
            "--side-px", "300", "--run-root", str(root), *near,
        )
        self.require_sim_backend(sides)
        assert sides.returncode == 0, sides.stderr
        row = json.loads((root / "sides" / "side_300" / "metrics.json").read_text(
            encoding="utf-8"))
        assert row["config"] == "side_300"
        assert "preflight" in row, "原子进程必须把 pre-flight 结论写进 metrics.json"
        assert row["preflight"]["zero_order_xy_frozen"]
        assert (root / "sides" / "side_300" / CONFIG_PUPIL_PHASE_NAME).is_file()

        sel = self.run_cli("--scan", "select", "--run-root", str(root), *near)
        assert sel.returncode == 0, sel.stderr
        selection = json.loads((root / "sides" / "selected.json").read_text(encoding="utf-8"))
        assert selection["side_px"] == 300
        assert selection["gs_grid"] == row["gs_grid"]

        for attempt in (1, 2):
            proc = self.run_cli(
                "--execute", "--cam-type", "sim", "--scan", "zernike",
                "--n-max", "11", "--run-root", str(root), *near,
            )
            assert proc.returncode == 0, proc.stderr
            metrics = json.loads(
                (root / "zernike" / "nmax_11" / "metrics.json").read_text(encoding="utf-8")
            )
            assert metrics["selected_side_px"] == 300
            assert metrics["selected_source"] == "selected.json"
            assert metrics["base_phase_source"] == "cache", (
                f"第 {attempt} 次: 应当复用第 1 段存盘的 pupil 相位"
            )
            rows = list(csv.DictReader(
                (root / "zernike" / "measurements.csv").open(encoding="utf-8")))
            assert len(rows) == 1, f"第 {attempt} 次: 同一配置不得写第二行"
            if attempt == 1:
                first_score = float(rows[0]["score"])
            else:
                assert float(rows[0]["score"]) == first_score, "重跑同一配置结果应逐位相同"

        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert [e["group"] for e in manifest] == ["sides", "zernike"]
        assert [e["n_rows"] for e in manifest] == [1, 1]

