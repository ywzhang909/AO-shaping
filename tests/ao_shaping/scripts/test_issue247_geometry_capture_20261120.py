"""Issue #247 离线验证: 几何采集脚本（光栅比例 / 光斑梯 / 倾斜符号）。

全部检查均为离线、无硬件：

1. ``--help`` 退出码 0 且列出关键参数（含 ``--tilt-period-px``）；
2. ``--self-test`` 无需设备即退出 0；
3. import 脚本本身不会加载任何相机/SLM 驱动；
4. plan 模式（默认）只打印 JSON，不创建 ``--out`` 目录；
5. plan JSON 覆盖全部图样 id、帧数、缺失项与声明常量；
6. 光栅图样：严格两灰阶 {0, 512}、周期正确、占空比 50%；
7. 光斑梯（blazed spot ladder）：单向、阶数不等，x->-x 与 rot180 下均非不变，
   每个 600x600 patch 的 ramp 恰为 2*pi / period；
8. 倾斜：真实棱镜斜坡（2*pi/period）、跨轴相位跨度、条纹数 >= 10、
   严格反对称，且在最悲观 (pitch, focal) 组合下位移 >= 5 CCD px；
   被否决的 0.5 rad 峰值弱设计必须判为不可行；
9. plan / summary schema 键齐全；
10. 波形判别计划（``--plan waveform``）：15 个 id 的顺序/数量/帧数、局部 blaze patch
    只沿单轴变化且 patch 外为 0、period=128 时 blaze_x_plus 与 tilt_x_plus 相位
    逐元素相同、``--patterns`` 子集与未知 id 硬失败。
11. 同窗口重复性计划（``--plan reproduction``）：69 个唯一 id / 276 帧、轮次顺序
    r1 冻结 / r2 反序 / r3 冻结、``base_pattern_id`` + ``repeat_index`` 元数据、
    同一 base 相位摘要逐轮一致、``--patterns`` 被拒绝。
12. ``capture_settings``：所有计划（含 reproduction）都记录已解析的 settle_s /
    exposure_ms / saturation_fraction / wait_until_idle_s / 设备选择，summary 镜像
    同一块，CLI 覆盖值生效。

本测试只用 ``tmp_path``，不 import 任何驱动，也不读写 ``data/``。
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = importlib.import_module("scripts.slm.issue247_geometry_capture_20261120")

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "slm" / "issue247_geometry_capture_20261120.py"

EXPECTED_PATTERN_IDS = (
    "flat",
    "grating_x_p128",
    "grating_y_p128",
    "grating_x_p256",
    "grating_y_p256",
    "ladder_x",
    "ladder_y",
    "tilt_x_plus",
    "tilt_x_minus",
    "tilt_y_plus",
    "tilt_y_minus",
    "blaze2d_px128_py192_sx+1_sy+1",
    "blaze2d_px128_py192_sx+1_sy-1",
    "blaze2d_px192_py128_sx+1_sy+1",
    "blaze2d_px192_py128_sx+1_sy-1",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# 1. --help
# ---------------------------------------------------------------------------
def test_help_exits_zero():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    for flag in (
        "--out",
        "--execute",
        "--self-test",
        "--plan",
        "--patterns",
        "--tilt-period-px",
        "--exposure-ms",
        "--settle-s",
        "--saturation-fraction",
        "--wait-until-idle",
        "--slm-number",
        "--wavelength-nm",
        "--cam-id",
    ):
        assert flag in result.stdout
    # --plan 的选择必须包含本轮新增的 reproduction（CLI 名称即契约）。
    assert "reproduction" in result.stdout


# ---------------------------------------------------------------------------
# 2. --self-test 无需设备
# ---------------------------------------------------------------------------
def test_self_test_exits_zero_without_devices():
    result = _run("--self-test")
    assert result.returncode == 0, result.stderr
    assert "all invariants passed" in result.stderr


# ---------------------------------------------------------------------------
# 3. 脚本 import 不加载驱动
# ---------------------------------------------------------------------------
def test_importing_script_does_not_load_drivers():
    # 干净子进程：仅 import 脚本，确认没有相机/SLM 驱动被加载。
    probe = (
        "import sys; import scripts.slm.issue247_geometry_capture_20261120; "
        "tokens = ('daheng', 'gxipy', 'santec', '_slm_win'); "
        "print(sorted(n for n in sys.modules if any(t in n for t in tokens)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


# ---------------------------------------------------------------------------
# 4. plan 模式不创建目录、输出 JSON
# ---------------------------------------------------------------------------
def test_plan_mode_creates_no_directory(tmp_path: Path):
    target = tmp_path / "fresh-plan-out"
    result = _run("--out", str(target))
    assert result.returncode == 0, result.stderr
    assert not target.exists(), "plan-only mode must not create --out"
    plan = json.loads(result.stdout)
    assert plan["out"] == str(target)
    assert "requires --execute" in result.stderr


# ---------------------------------------------------------------------------
# 5. plan JSON 覆盖图样表 / 帧数 / 缺失项 / 声明常量
# ---------------------------------------------------------------------------
def test_plan_json_covers_all_patterns_and_constants(tmp_path: Path):
    target = tmp_path / "plan-out"
    result = _run("--out", str(target))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)

    ids = tuple(plan["pattern_ids"])
    assert ids == EXPECTED_PATTERN_IDS
    assert plan["n_patterns"] == len(EXPECTED_PATTERN_IDS)

    frames = plan["frames_per_pattern"]
    assert frames["flat"] >= 2
    for pid in ids:
        if pid != "flat":
            assert frames[pid] >= 4, pid
    assert plan["n_frames_expected"] == sum(frames.values())

    assert plan["grating_periods_px"] == [128, 256]
    assert plan["grating_duty"] == 0.5
    assert plan["encoding_max_gray"] == 1023

    # 倾斜：默认 128 px 请求值；两轴都记录跨度/条纹数与悲观 CCD 位移。
    tilt = plan["tilt"]
    assert tilt["requested_period_px"] == 128.0
    assert tilt["feasible"] is True
    assert set(tilt["axes"]) == {"x", "y"}
    for name in ("x", "y"):
        geometry = tilt["axes"][name]
        assert geometry["fringe_count"] >= 10.0, name
        assert geometry["pessimistic_ccd_px"] >= 5.0, name
        assert geometry["phase_span_rad"] > 0.0, name
    assert tilt["expected_ccd_px_formula"]

    # 光斑梯：base 96 px，+1/+3 两档，非镜像对称。
    ladder = plan["ladder"]
    assert ladder["base_period_px"] == 96.0
    assert ladder["patch_size_px"] == 600
    assert ladder["orders"] == [1, 3]
    assert ladder["periods_px"]["1"] == 96.0
    assert ladder["periods_px"]["3"] == 32.0
    assert ladder["spots"]["x"] == [[0, 0], [1, 0], [3, 0]]
    assert ladder["spots"]["y"] == [[0, 0], [0, 1], [0, 3]]
    assert ladder["invariant_under_sign_flip"]["x"] is False
    assert ladder["invariant_under_rot180"]["y"] is False
    assert ladder["expected_ccd_px_formula"]

    declared = plan["declared_constants"]
    assert declared["panel_shape"] == [1200, 1920]
    assert declared["native_ccd_shape"] == [1944, 2592]
    assert declared["encoding_max_gray"] == 1023
    assert declared["tilt_min_fringes"] == 10.0
    assert declared["tilt_min_ccd_px"] == 5.0
    assert declared["ladder_orders"] == [1, 3]

    missing = plan["missing"]
    assert missing, "physical pitches / magnification must be declared missing"
    assert all(isinstance(reason, str) and reason for reason in missing.values())
    assert "declared.tilt_expected_ccd_px" in missing
    assert "declared.ladder_expected_ccd_px" in missing
    # 物理尺寸只允许 null + 明确原因，禁止硬编码猜测。
    assert declared["slm_pixel_pitch_um"] is None
    assert declared["camera_pixel_size_um"] is None
    assert declared["focal_magnification"] is None


# ---------------------------------------------------------------------------
# 5b. --patterns 只能缩小到批准范围，未知 id 必须硬失败
# ---------------------------------------------------------------------------
def test_patterns_flag_narrows_the_run_and_rejects_unknown_ids(tmp_path: Path):
    subset = (
        "flat",
        "blaze2d_px128_py192_sx+1_sy+1",
        "blaze2d_px128_py192_sx+1_sy-1",
        "blaze2d_px192_py128_sx+1_sy+1",
        "blaze2d_px192_py128_sx+1_sy-1",
    )
    result = _run("--out", str(tmp_path / "subset"), "--patterns", ",".join(subset))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert tuple(plan["pattern_ids"]) == subset
    assert plan["n_patterns"] == len(subset)
    assert plan["n_frames_expected"] == sum(plan["frames_per_pattern"].values())
    # summary 只声明本次采集的 blaze2d 设计，求解器不会去找没采的方案。
    assert [d["id"] for d in plan["blaze2d_designs"]] == list(subset[1:])
    # 默认行为不变：不传 --patterns 仍是完整冻结清单。
    full = json.loads(_run("--out", str(tmp_path / "full")).stdout)
    assert tuple(full["pattern_ids"]) == EXPECTED_PATTERN_IDS

    bad = _run("--out", str(tmp_path / "bad"), "--patterns", "flat,bogus_id")
    assert bad.returncode != 0
    assert "unknown pattern id" in (bad.stderr + bad.stdout)


# ---------------------------------------------------------------------------
# 6. 光栅图样不变量
# ---------------------------------------------------------------------------
def test_grating_patterns_two_levels_period_and_duty():
    patterns = [p for p in SCRIPT.build_patterns() if p["kind"] == "grating"]
    assert len(patterns) == 4
    for item in patterns:
        arr = SCRIPT.pattern_array(item, SCRIPT.DEFAULT_PANEL_SHAPE)
        levels = np.unique(arr)
        assert list(map(int, levels)) == [0, SCRIPT.PI_GRAY], item["id"]
        period = int(item["period_px"])
        line = arr[0, :] if item["axis"] == 1 else arr[:, 0]
        assert np.array_equal(line[:-period], line[period:]), item["id"]
        assert float(np.mean(line[:period] == SCRIPT.PI_GRAY)) == 0.5, item["id"]


def test_grating_rejects_odd_or_tiny_period():
    with pytest.raises(ValueError):
        SCRIPT.grating_phase(SCRIPT.DEFAULT_PANEL_SHAPE, period_px=7, axis=1)
    with pytest.raises(ValueError):
        SCRIPT.grating_phase(SCRIPT.DEFAULT_PANEL_SHAPE, period_px=128, axis=2)


# ---------------------------------------------------------------------------
# 7. 光斑梯（180° 分支判别）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [0, 1])
def test_ladder_spots_break_sign_and_rotation_symmetry(axis: int):
    spots = SCRIPT.ladder_spot_orders(axis)
    assert len(spots) == 1 + len(SCRIPT.LADDER_ORDERS)
    assert spots[0] == (0, 0)
    # 阶梯不是 x -> -x 的镜像不变，也不是 180° 旋转不变。
    assert SCRIPT.spots_invariant_under_sign_flip(spots) is False
    assert SCRIPT.spots_invariant_under_rot180(spots) is False
    # 全部落在单一轴、同侧：镜像与旋转可分离。
    on_axis = 0 if axis == 1 else 1
    assert all(s[1 - on_axis] == 0 for s in spots)
    assert [s[on_axis] for s in spots] == [0, *SCRIPT.LADDER_ORDERS]


def test_ladder_amplitudes_are_unequal():
    periods = [period for _order, period in SCRIPT.ladder_periods()]
    assert periods == [96.0, 32.0]
    assert len(set(periods)) == len(periods)


@pytest.mark.parametrize("axis", [0, 1])
def test_ladder_patch_ramp_is_two_pi_per_period(axis: int):
    phase = SCRIPT.ladder_phase(SCRIPT.DEFAULT_PANEL_SHAPE, axis=axis)
    side = SCRIPT.LADDER_PATCH_SIZE_PX
    for (row, col), (_order, period) in zip(
        SCRIPT.LADDER_PATCH_ORIGINS, SCRIPT.ladder_periods()
    ):
        patch = phase[row : row + side, col : col + side]
        line = patch[0, :] if axis == 1 else patch[:, 0]
        expected_span = SCRIPT.TWO_PI * (side - 1) / period
        assert abs(abs(line[-1] - line[0]) - expected_span) <= 5e-3 * expected_span
        steps = np.diff(line)
        assert np.allclose(steps, steps[0], rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 8. 倾斜：真实棱镜斜坡 + 可测性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [0, 1])
def test_tilt_ramp_is_prismatic_and_antisymmetric(axis: int):
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    geometry = SCRIPT.tilt_geometry(
        shape, period_px=SCRIPT.DEFAULT_TILT_PERIOD_PX, axis=axis
    )
    period = geometry["period_px"]
    plus = SCRIPT.tilt_phase(shape, period_px=period, axis=axis, sign=+1)
    minus = SCRIPT.tilt_phase(shape, period_px=period, axis=axis, sign=-1)

    # 严格反对称（rad，非 mod 而是逐点取负）且零活塞。
    assert np.array_equal(plus, -minus)
    assert float(plus.mean()) == pytest.approx(0.0, abs=1e-9)

    # 斜率恰为 2*pi / period，跨轴跨度为 2*pi*n/period（0.5% 容差）。
    n = geometry["axis_length_px"]
    line = plus[0, :] if axis == 1 else plus[:, 0]
    assert np.allclose(np.diff(line), SCRIPT.TWO_PI / period, rtol=0, atol=1e-9)
    expected_span = SCRIPT.TWO_PI * n / period
    assert abs(float(np.ptp(plus)) - expected_span) <= 5e-3 * expected_span
    assert geometry["fringe_count"] >= SCRIPT.TILT_MIN_FRINGES
    assert geometry["fringe_count"] == pytest.approx(n / period)


def test_tilt_feasibility_under_pessimistic_geometry():
    plan = SCRIPT.build_plan(argparse.Namespace(out=None))
    tilt = plan["tilt"]
    assert tilt["feasible"] is True

    # 最悲观 (pitch, focal, ccd pixel)：位移仍 >= 5 CCD px。
    pessimistic = tilt["pessimistic_geometry"]
    assert pessimistic["pitch_um"] == SCRIPT.TILT_PESSIMISTIC_PITCH_UM
    assert pessimistic["focal_mm"] == SCRIPT.TILT_PESSIMISTIC_FOCAL_MM
    for name in ("x", "y"):
        geometry = tilt["axes"][name]
        manual = SCRIPT.tilt_expected_ccd_px(
            period_px=geometry["period_px"],
            pitch_um=pessimistic["pitch_um"],
            focal_mm=pessimistic["focal_mm"],
            ccd_pixel_um=pessimistic["ccd_pixel_um"],
        )
        assert manual == pytest.approx(geometry["pessimistic_ccd_px"])
        assert manual >= SCRIPT.TILT_MIN_CCD_PX

    # 被否决的 0.5 rad 峰值（等价 1.0 rad 全跨）必须判为不可行。
    weak = tilt["rejected_weak_design"]
    assert weak["phase_span_rad"] == 1.0
    assert weak["feasible"] is False
    assert SCRIPT.raw_tilt_feasible(weak["equivalent_period_px"], 1920) is False


# ---------------------------------------------------------------------------
# 9. schema 键
# ---------------------------------------------------------------------------
def test_plan_and_summary_schema_keys_present():
    plan = SCRIPT.build_plan(argparse.Namespace(out=None))
    for key in SCRIPT.PLAN_KEYS:
        assert key in plan, key
    summary = SCRIPT.summary_skeleton()
    for key in SCRIPT.SUMMARY_KEYS:
        assert key in summary, key
    assert plan["schema"] == SCRIPT.PLAN_SCHEMA
    assert summary["schema"] == SCRIPT.SUMMARY_SCHEMA


def test_named_pattern_axes_match_panel_column_row_semantics():
    """x patterns vary across panel columns; y patterns vary across panel rows."""
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    patterns = (
        ("grating_x", SCRIPT.grating_phase(shape, period_px=128, axis=1), 1),
        ("grating_y", SCRIPT.grating_phase(shape, period_px=128, axis=0), 0),
        ("tilt_x", SCRIPT.tilt_phase(shape, period_px=128, axis=1, sign=1), 1),
        ("tilt_y", SCRIPT.tilt_phase(shape, period_px=128, axis=0, sign=1), 0),
        ("ladder_x", SCRIPT.ladder_phase(shape, axis=1), 1),
        ("ladder_y", SCRIPT.ladder_phase(shape, axis=0), 0),
    )

    for name, phase, expected_axis in patterns:
        col_gradient = float(np.abs(np.diff(phase, axis=1)).sum())
        row_gradient = float(np.abs(np.diff(phase, axis=0)).sum())
        if expected_axis == 1:
            assert col_gradient > row_gradient, name
        else:
            assert row_gradient > col_gradient, name


# ---------------------------------------------------------------------------
# 10. 波形判别计划（--plan waveform）
# ---------------------------------------------------------------------------
WAVEFORM_IDS = (
    "flat_pre",
    "blaze_x_plus",
    "blaze_x_minus",
    "blaze_y_plus",
    "blaze_y_minus",
    "grating_x_p128",
    "grating_y_p128",
    "tilt_x_plus",
    "tilt_y_plus",
    "flat_mid",
    "grating_x_p256",
    "grating_y_p256",
    "blaze_patch_x",
    "blaze_patch_y",
    "flat_post",
)


def test_default_plan_stays_frozen(tmp_path: Path):
    """不加 --plan 时仍必须是原 15 图样冻结清单（顺序与 id 一律不变）。"""
    result = _run("--out", str(tmp_path / "frozen-plan"))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["plan_name"] == "frozen"
    assert tuple(plan["pattern_ids"]) == EXPECTED_PATTERN_IDS


def test_waveform_plan_ids_order_and_frames(tmp_path: Path):
    result = _run("--plan", "waveform", "--out", str(tmp_path / "wf-plan"))
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "wf-plan").exists(), "plan-only 模式不得创建 --out"
    plan = json.loads(result.stdout)
    assert plan["plan_name"] == "waveform"
    assert tuple(plan["pattern_ids"]) == WAVEFORM_IDS
    assert plan["n_patterns"] == 15
    assert plan["n_frames_expected"] == 60
    for pid in WAVEFORM_IDS:
        assert plan["frames_per_pattern"][pid] == 4, pid
    # 波形计划里没有 blaze2d 图样，派生清单不得残留。
    assert plan["blaze2d_designs"] == []

    by_id = {p["id"]: p for p in plan["patterns"]}
    for pid, source in (
        ("blaze_x_plus", "tilt_x_plus"),
        ("blaze_x_minus", "tilt_x_minus"),
        ("blaze_y_plus", "tilt_y_plus"),
        ("blaze_y_minus", "tilt_y_minus"),
    ):
        assert by_id[pid]["identical_builder_to"] == source, pid
        assert by_id[pid]["period_px"] == 128.0, pid
        assert by_id[pid]["requested_period_px"] == 128.0, pid
    for pid in ("flat_pre", "flat_mid", "flat_post"):
        assert by_id[pid]["kind"] == "flat", pid


def test_waveform_plan_subset_and_unknown_id(tmp_path: Path):
    ok = _run(
        "--plan",
        "waveform",
        "--patterns",
        "flat_pre,blaze_patch_x",
        "--out",
        str(tmp_path / "wf-subset"),
    )
    assert ok.returncode == 0, ok.stderr
    subset = json.loads(ok.stdout)
    assert subset["plan_name"] == "waveform"
    assert subset["pattern_ids"] == ["flat_pre", "blaze_patch_x"]
    assert subset["n_patterns"] == 2

    bad = _run(
        "--plan",
        "waveform",
        "--patterns",
        "flat_pre,bogus_id",
        "--out",
        str(tmp_path / "wf-bad"),
    )
    assert bad.returncode != 0
    assert "unknown pattern id" in bad.stderr
    assert not (tmp_path / "wf-bad").exists()


def test_blaze_patch_phase_is_zero_outside_and_single_axis_inside():
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    row, col = SCRIPT.LADDER_PATCH_ORIGINS[0]
    side = SCRIPT.LADDER_PATCH_SIZE_PX
    inside = np.zeros(shape, dtype=bool)
    inside[row : row + side, col : col + side] = True
    for axis in (1, 0):
        phase = SCRIPT.blaze_patch_phase(shape, axis=axis)
        assert np.all(phase[~inside] == 0.0), axis
        patch = phase[row : row + side, col : col + side]
        row_delta = float(np.abs(np.diff(patch, axis=0)).sum())
        col_delta = float(np.abs(np.diff(patch, axis=1)).sum())
        if axis == 1:
            assert row_delta == 0.0 and col_delta > 0.0, axis
        else:
            assert col_delta == 0.0 and row_delta > 0.0, axis


def test_blaze_patch_phase_rejects_bad_arguments():
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    with pytest.raises(ValueError):
        SCRIPT.blaze_patch_phase(shape, axis=2)
    with pytest.raises(ValueError):
        SCRIPT.blaze_patch_phase(shape, axis=1, period_px=0)
    with pytest.raises(ValueError):
        # patch 落到面板外：与 ladder_phase 一样抛 ValueError，不静默裁剪。
        SCRIPT.blaze_patch_phase(shape, axis=1, origin=(int(shape[0]) - 10, 0))


def test_waveform_pattern_phases_follow_declared_axis():
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    by_id = {p["id"]: p for p in SCRIPT.build_waveform_patterns()}
    # period 恰 128 px 时，blaze_x_plus 与 tilt_x_plus 是同一份数组。
    assert np.array_equal(
        SCRIPT.pattern_phase(by_id["blaze_x_plus"], shape),
        SCRIPT.pattern_phase(by_id["tilt_x_plus"], shape),
    )
    for pid, expected_axis in (
        ("grating_x_p128", 1),
        ("grating_y_p128", 0),
    ):
        phase = SCRIPT.pattern_phase(by_id[pid], shape)
        row_delta = float(np.abs(np.diff(phase, axis=0)).sum())
        col_delta = float(np.abs(np.diff(phase, axis=1)).sum())
        if expected_axis == 1:
            assert row_delta == 0.0 and col_delta > 0.0, pid
        else:
            assert col_delta == 0.0 and row_delta > 0.0, pid

    # 局部 patch：差分只在 patch 内部量（patch 边界与 patch 外的 0 之外
    # 必然产生一条非零差分，这正是「patch 外为 0」的推论，不是轴错位）。
    row0, col0 = SCRIPT.LADDER_PATCH_ORIGINS[0]
    side = SCRIPT.LADDER_PATCH_SIZE_PX
    for pid, expected_axis in (("blaze_patch_x", 1), ("blaze_patch_y", 0)):
        phase = SCRIPT.pattern_phase(by_id[pid], shape)
        inside = np.zeros(shape, dtype=bool)
        inside[row0 : row0 + side, col0 : col0 + side] = True
        assert np.all(phase[~inside] == 0.0), pid
        patch = phase[row0 : row0 + side, col0 : col0 + side]
        row_delta = float(np.abs(np.diff(patch, axis=0)).sum())
        col_delta = float(np.abs(np.diff(patch, axis=1)).sum())
        if expected_axis == 1:
            assert row_delta == 0.0 and col_delta > 0.0, pid
        else:
            assert col_delta == 0.0 and row_delta > 0.0, pid


# ---------------------------------------------------------------------------
# 11. 波形族归因计划（--plan attribution）：23 图样 / 92 帧 / fit + held_out
# ---------------------------------------------------------------------------
ATTRIBUTION_IDS = (
    "flat_pre",
    "grating_x_p128",
    "tilt_x_plus",
    "tilt_x_minus",
    "patch_x_plus_p128",
    "patch_x_minus_p128",
    "grating_y_p128",
    "tilt_y_plus",
    "tilt_y_minus",
    "patch_y_plus_p128",
    "patch_y_minus_p128",
    "flat_mid",
    "grating_x_p192",
    "tilt_x_plus_p192",
    "tilt_x_minus_p192",
    "patch_x_plus_p192",
    "patch_x_minus_p192",
    "grating_y_p192",
    "tilt_y_plus_p192",
    "tilt_y_minus_p192",
    "patch_y_plus_p192",
    "patch_y_minus_p192",
    "flat_post",
)


def test_attribution_plan_ids_order_count_frames_and_roles(tmp_path: Path):
    result = _run("--plan", "attribution", "--out", str(tmp_path / "attr-plan"))
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "attr-plan").exists(), "plan-only 模式不得创建 --out"
    plan = json.loads(result.stdout)
    assert plan["plan_name"] == "attribution"
    assert tuple(plan["pattern_ids"]) == ATTRIBUTION_IDS
    assert tuple(SCRIPT.ATTRIBUTION_PATTERN_IDS) == ATTRIBUTION_IDS
    assert plan["n_patterns"] == 23
    assert plan["n_frames_expected"] == 92
    for pid in ATTRIBUTION_IDS:
        assert plan["frames_per_pattern"][pid] == 4, pid
    assert plan["blaze2d_designs"] == []

    patterns = plan["patterns"]
    assert tuple(p["id"] for p in patterns) == ATTRIBUTION_IDS
    by_id = {p["id"]: p for p in patterns}
    fit = {pid for pid, role in plan["role_by_pattern"].items() if role == "fit"}
    held_out = {
        pid for pid, role in plan["role_by_pattern"].items() if role == "held_out"
    }
    assert len(fit) == 12 and len(held_out) == 11
    assert not (fit & held_out)
    assert fit | held_out == set(ATTRIBUTION_IDS)

    for pid in ATTRIBUTION_IDS:
        expected_role = (
            "held_out" if pid.endswith("_p192") or pid == "flat_post" else "fit"
        )
        assert by_id[pid]["role"] == expected_role, pid
        assert plan["role_by_pattern"][pid] == expected_role, pid

    for pid in ("flat_pre", "flat_mid", "flat_post"):
        assert by_id[pid]["kind"] == "flat", pid
        assert "period_px" not in by_id[pid], pid
        assert "requested_period_px" not in by_id[pid], pid
    for pid in ATTRIBUTION_IDS:
        item = by_id[pid]
        if pid.startswith(("grating_", "tilt_", "patch_")):
            expected_period = 192 if pid.endswith("_p192") else 128
            assert float(item["period_px"]) == float(expected_period), pid
        if pid.startswith("grating_"):
            assert item["kind"] == "grating", pid
            assert "sign" not in item, pid


def test_attribution_plan_subset_preserves_order_and_rejects_unknown(tmp_path: Path):
    ok = _run(
        "--plan",
        "attribution",
        "--patterns",
        "patch_y_minus_p192,flat_pre,grating_x_p128",
        "--out",
        str(tmp_path / "attr-subset"),
    )
    assert ok.returncode == 0, ok.stderr
    subset = json.loads(ok.stdout)
    assert subset["plan_name"] == "attribution"
    assert subset["pattern_ids"] == [
        "flat_pre",
        "grating_x_p128",
        "patch_y_minus_p192",
    ]
    assert subset["n_patterns"] == 3
    assert subset["n_frames_expected"] == 12
    assert subset["role_by_pattern"] == {
        "flat_pre": "fit",
        "grating_x_p128": "fit",
        "patch_y_minus_p192": "held_out",
    }

    bad = _run(
        "--plan",
        "attribution",
        "--patterns",
        "flat_pre,bogus_id",
        "--out",
        str(tmp_path / "attr-bad"),
    )
    assert bad.returncode != 0
    assert "unknown pattern id" in bad.stderr
    assert not (tmp_path / "attr-bad").exists()


def test_blaze_patch_phase_sign_is_elementwise_negation():
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    row, col = SCRIPT.LADDER_PATCH_ORIGINS[0]
    side = SCRIPT.LADDER_PATCH_SIZE_PX
    inside = np.zeros(shape, dtype=bool)
    inside[row : row + side, col : col + side] = True
    for axis in (1, 0):
        default = SCRIPT.blaze_patch_phase(shape, axis=axis, period_px=128.0)
        plus = SCRIPT.blaze_patch_phase(shape, axis=axis, period_px=128.0, sign=1)
        minus = SCRIPT.blaze_patch_phase(shape, axis=axis, period_px=128.0, sign=-1)
        assert np.array_equal(default, plus), axis
        assert np.array_equal(minus, -default), axis
        assert np.all(plus[~inside] == 0.0), axis
        assert np.all(minus[~inside] == 0.0), axis
    with pytest.raises(ValueError):
        SCRIPT.blaze_patch_phase(shape, axis=1, sign=0)


def test_pattern_phase_reads_blaze_patch_sign_with_default_unchanged():
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    base = {
        "kind": "blaze_patch",
        "axis": 1,
        "period_px": 128.0,
        "side_px": SCRIPT.LADDER_PATCH_SIZE_PX,
        "origin": list(SCRIPT.LADDER_PATCH_ORIGINS[0]),
    }
    default_phase = SCRIPT.pattern_phase(base, shape)
    assert np.array_equal(
        default_phase,
        SCRIPT.blaze_patch_phase(shape, axis=1, period_px=128.0),
    ), "缺省 sign 必须仍是 +1"
    assert np.array_equal(
        SCRIPT.pattern_phase({**base, "sign": -1}, shape), -default_phase
    )

    by_id = {p["id"]: p for p in SCRIPT.build_attribution_patterns()}
    for name in ("x", "y"):
        assert np.array_equal(
            SCRIPT.pattern_phase(by_id[f"patch_{name}_plus_p128"], shape),
            -SCRIPT.pattern_phase(by_id[f"patch_{name}_minus_p128"], shape),
        )


def test_attribution_tilt_y_keeps_exact_period_and_sign_antisymmetry():
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    by_id = {p["id"]: p for p in SCRIPT.build_attribution_patterns()}
    for pid in (
        "tilt_y_plus",
        "tilt_y_minus",
        "tilt_y_plus_p192",
        "tilt_y_minus_p192",
    ):
        item = by_id[pid]
        expected_period = 192.0 if pid.endswith("_p192") else 128.0
        assert int(item["axis"]) == 0, pid
        assert float(item["period_px"]) == expected_period, pid
        assert float(item["requested_period_px"]) == expected_period, pid
        assert float(item["period_px"]) != 120.0, pid  # tilt_geometry 的 y clamp

    for plus_id, minus_id in (
        ("tilt_x_plus", "tilt_x_minus"),
        ("tilt_x_plus_p192", "tilt_x_minus_p192"),
        ("tilt_y_plus", "tilt_y_minus"),
        ("tilt_y_plus_p192", "tilt_y_minus_p192"),
    ):
        assert np.array_equal(
            SCRIPT.pattern_phase(by_id[plus_id], shape),
            -SCRIPT.pattern_phase(by_id[minus_id], shape),
        ), (plus_id, minus_id)

    for pid, period in (("tilt_y_plus", 128.0), ("tilt_y_plus_p192", 192.0)):
        line = SCRIPT.pattern_phase(by_id[pid], shape)[:, 0]
        assert np.allclose(np.diff(line), SCRIPT.TWO_PI / period, rtol=0, atol=1e-9), pid


def test_frozen_and_waveform_plans_are_unchanged_regression(tmp_path: Path):
    """新增 attribution 不得改变原 two plans 的 id / 帧数 / phase / schema。"""
    frozen_run = _run("--out", str(tmp_path / "frozen"))
    assert frozen_run.returncode == 0, frozen_run.stderr
    frozen = json.loads(frozen_run.stdout)
    assert frozen["plan_name"] == "frozen"
    assert tuple(frozen["pattern_ids"]) == EXPECTED_PATTERN_IDS
    assert frozen["n_patterns"] == 15
    assert frozen["n_frames_expected"] == 60

    waveform_run = _run("--plan", "waveform", "--out", str(tmp_path / "wf"))
    assert waveform_run.returncode == 0, waveform_run.stderr
    waveform = json.loads(waveform_run.stdout)
    assert waveform["plan_name"] == "waveform"
    assert tuple(waveform["pattern_ids"]) == WAVEFORM_IDS
    assert waveform["n_patterns"] == 15
    assert waveform["n_frames_expected"] == 60

    # frozen / waveform plan JSON 不引入新键；两种老 plan 的图样也没有 role。
    for plan in (frozen, waveform):
        assert "role_by_pattern" not in plan, plan["plan_name"]
        assert "reproduction" not in plan, plan["plan_name"]
        for key in SCRIPT.PLAN_KEYS:
            assert key in plan, key
    for builder in (SCRIPT.build_patterns, SCRIPT.build_waveform_patterns):
        assert all("role" not in item for item in builder())

    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    wf_by_id = {p["id"]: p for p in SCRIPT.build_waveform_patterns()}
    assert np.array_equal(
        SCRIPT.pattern_phase(wf_by_id["blaze_x_plus"], shape),
        SCRIPT.pattern_phase(wf_by_id["tilt_x_plus"], shape),
    )
    for pid in ("blaze_patch_x", "blaze_patch_y"):
        item = wf_by_id[pid]
        assert np.array_equal(
            SCRIPT.pattern_phase(item, shape),
            SCRIPT.blaze_patch_phase(
                shape,
                axis=int(item["axis"]),
                period_px=float(item["period_px"]),
            ),
        ), pid


# ---------------------------------------------------------------------------
# 11. 同窗口重复性计划（--plan reproduction）
# ---------------------------------------------------------------------------
def test_reproduction_plan_ids_rounds_frames_and_metadata():
    plan = SCRIPT.build_plan(
        argparse.Namespace(out=Path("__plan__"), plan="reproduction", patterns=None)
    )
    ids = tuple(plan["pattern_ids"])
    assert len(ids) == 69
    assert len(set(ids)) == 69
    assert plan["n_patterns"] == 69
    assert plan["n_frames_expected"] == 276
    assert set(plan["frames_per_pattern"].values()) == {4}

    meta = plan["reproduction"]
    assert meta["rounds"] == ["r1", "r2", "r3"]
    assert meta["round_order"] == ["frozen", "reversed", "frozen"]
    assert meta["base_pattern_count"] == 23
    assert meta["n_unique_patterns"] == 69
    assert meta["n_frames_expected"] == 276
    assert meta["frames_per_pattern"] == 4
    assert meta["phase_hash_identical_across_rounds"] is True
    assert meta["single_slm_session"] is True
    assert "role_by_pattern" not in plan

    # 冻结轮次顺序：r1/r3 按冻结 attribution 顺序，r2 反序；三轮依次排布。
    base_ids = list(SCRIPT.ATTRIBUTION_PATTERN_IDS)
    round_ids = {
        label: [pid for pid in ids if pid.endswith(f"__{label}")]
        for label in ("r1", "r2", "r3")
    }
    assert round_ids["r1"] == [f"{pid}__r1" for pid in base_ids]
    assert round_ids["r2"] == [f"{pid}__r2" for pid in reversed(base_ids)]
    assert round_ids["r3"] == [f"{pid}__r3" for pid in base_ids]
    assert list(ids) == round_ids["r1"] + round_ids["r2"] + round_ids["r3"]

    # 每条 pattern 记录 base id 与 1 基 repeat_index；kind / role / 帧数沿用 base。
    attrib = {p["id"]: p for p in SCRIPT.build_attribution_patterns()}
    for item in plan["patterns"]:
        assert item["id"] == f"{item['base_pattern_id']}__r{item['repeat_index']}"
        assert int(item["repeat_index"]) in (1, 2, 3)
        ref = attrib[item["base_pattern_id"]]
        assert item["kind"] == ref["kind"]
        assert item.get("role") == ref.get("role")
        assert int(item["frames"]) == 4


def test_reproduction_phase_hashes_are_identical_across_rounds():
    reproduction = SCRIPT.build_reproduction_patterns()
    attrib = {p["id"]: p for p in SCRIPT.build_attribution_patterns()}
    for pid in SCRIPT.ATTRIBUTION_PATTERN_IDS:
        rounds = [p for p in reproduction if p["base_pattern_id"] == pid]
        assert [int(p["repeat_index"]) for p in rounds] == [1, 2, 3], pid
        # 相位摘要必须与 base 图样逐字节一致（相同 phase，不携带轮次信息）。
        assert {p["phase_sha256"] for p in rounds} == {attrib[pid]["phase_sha256"]}, pid
    # 摘要已覆盖全部 23 个 base；再对三种 kind 抽查像素级一致。
    shape = SCRIPT.DEFAULT_PANEL_SHAPE
    for pid in ("grating_x_p128", "tilt_y_minus_p192", "patch_x_plus_p128"):
        arrays = [
            SCRIPT.pattern_array(p, shape)
            for p in reproduction
            if p["base_pattern_id"] == pid
        ]
        assert np.array_equal(arrays[0], arrays[1]), pid
        assert np.array_equal(arrays[0], arrays[2]), pid


def test_reproduction_plan_rejects_patterns_subset():
    with pytest.raises(SystemExit):
        SCRIPT.build_plan(
            argparse.Namespace(
                out=Path("__plan__"), plan="reproduction", patterns="flat_pre"
            )
        )


def test_reproduction_cli_plan_only_creates_no_directory(tmp_path: Path):
    target = tmp_path / "repro-out"
    result = _run("--plan", "reproduction", "--out", str(target))
    assert result.returncode == 0, result.stderr
    assert not target.exists(), "plan-only mode must not create --out"
    plan = json.loads(result.stdout)
    assert plan["plan_name"] == "reproduction"
    assert plan["n_patterns"] == 69
    assert plan["n_frames_expected"] == 276
    assert plan["reproduction"]["rounds"] == ["r1", "r2", "r3"]
    assert "requires --execute" in result.stderr


# ---------------------------------------------------------------------------
# 12. capture_settings：已解析的采集设置（plan.json / summary.json 共有）
# ---------------------------------------------------------------------------
CAPTURE_SETTING_KEYS = (
    "plan_name",
    "settle_s",
    "exposure_ms",
    "saturation_fraction",
    "wait_until_idle_s",
    "slm_number",
    "wavelength_nm",
    "cam_id",
)


def test_capture_settings_defaults_present_for_every_plan():
    """所有计划都带同一块 capture_settings，冻结值可直接被离线检查器核对。"""
    for plan_name in ("frozen", "waveform", "attribution", "reproduction"):
        plan = SCRIPT.build_plan(argparse.Namespace(out=None, plan=plan_name))
        block = plan["capture_settings"]
        # 键集合固定且顺序固定，检查器按名读取不必容错。
        assert tuple(block) == CAPTURE_SETTING_KEYS, plan_name
        assert block["plan_name"] == plan_name
        # 预注册冻结项：settle 0.8 s、饱和门 0.9、等待 1800 s。
        assert block["settle_s"] == 0.8, plan_name
        assert isinstance(block["settle_s"], float)
        assert block["saturation_fraction"] == 0.9, plan_name
        assert block["wait_until_idle_s"] == 1800.0, plan_name
        # 曝光只记录请求值（默认 0.05；硬件轮次由操作者显式传 0.04）。
        assert block["exposure_ms"] == SCRIPT.DEFAULT_EXPOSURE_MS, plan_name
        # 设备选择默认值。
        assert (block["slm_number"], block["wavelength_nm"], block["cam_id"]) == (
            1,
            1064,
            0,
        ), plan_name
        # 必须可 JSON 序列化（plan.json 直接写盘）。
        assert json.loads(json.dumps(plan))["capture_settings"] == block


def test_capture_settings_follow_resolved_cli_values():
    args = argparse.Namespace(
        out=None,
        plan="reproduction",
        settle_s=0.8,
        exposure_ms=0.04,
        saturation_fraction=0.9,
        wait_until_idle=1800.0,
        slm_number=2,
        wavelength_nm=1064,
        cam_id=1,
    )
    plan = SCRIPT.build_plan(args)
    block = plan["capture_settings"]
    assert block == {
        "plan_name": "reproduction",
        "settle_s": 0.8,
        "exposure_ms": 0.04,
        "saturation_fraction": 0.9,
        "wait_until_idle_s": 1800.0,
        "slm_number": 2,
        "wavelength_nm": 1064,
        "cam_id": 1,
    }


def test_summary_mirrors_plan_capture_settings():
    plan = SCRIPT.build_plan(argparse.Namespace(out=None, plan="attribution"))
    summary = SCRIPT.summary_skeleton(plan["capture_settings"])
    assert summary["capture_settings"] == plan["capture_settings"]
    assert summary["capture_settings"]["plan_name"] == "attribution"
    # 未传块时键仍存在（schema 一致），但内容为空。
    assert SCRIPT.summary_skeleton()["capture_settings"] == {}
    assert "capture_settings" in SCRIPT.PLAN_KEYS
    assert "capture_settings" in SCRIPT.SUMMARY_KEYS


def test_cli_plan_only_records_resolved_capture_settings(tmp_path: Path):
    target = tmp_path / "settings-out"
    result = _run(
        "--plan",
        "reproduction",
        "--settle-s",
        "0.8",
        "--exposure-ms",
        "0.04",
        "--saturation-fraction",
        "0.9",
        "--wait-until-idle",
        "1800",
        "--out",
        str(target),
    )
    assert result.returncode == 0, result.stderr
    assert not target.exists(), "plan-only mode must not create --out"
    plan = json.loads(result.stdout)
    block = plan["capture_settings"]
    assert block["plan_name"] == "reproduction"
    assert block["settle_s"] == 0.8
    assert block["exposure_ms"] == 0.04
    assert block["saturation_fraction"] == 0.9
    assert block["wait_until_idle_s"] == 1800.0
    # 计划内容不受影响：capture_settings 只是新增块。
    assert plan["n_patterns"] == 69
    assert plan["n_frames_expected"] == 276
    assert plan["reproduction"]["rounds"] == ["r1", "r2", "r3"]
