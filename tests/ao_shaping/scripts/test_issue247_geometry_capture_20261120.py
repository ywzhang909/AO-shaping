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
9. plan / summary schema 键齐全。

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
