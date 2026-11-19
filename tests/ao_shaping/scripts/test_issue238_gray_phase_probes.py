"""Issue #238 离线验证: 原始 Mono12 gray->phase + crosstalk 采集脚本。

四组检查 (一律不碰硬件):

1. ``--help`` 退出码 0;
2. plan 模式: 干净子进程里打印 JSON、不创建 ``--out``、不 import 相机/SLM 驱动;
3. ``_run`` 遇到已存在的 ``--out``: 返回 2, 假锁恰好 acquire+release 一次,
   且不 import 任何驱动 (out.mkdir 失败发生在任何驱动 import 之前);
4. 纯图样/扫描函数: 65 点、端点 1023、blaze 峰值 == g、checker 周期 16。
"""

from __future__ import annotations

import argparse
import builtins
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = importlib.import_module("scripts.slm.issue238_gray_phase_probes")

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "slm" / "issue238_gray_phase_probes.py"

DEVICE_MODULES = ("daheng", "gxipy", "santec", "_slm_win")


def _device_modules_loaded() -> list[str]:
    return [name for name in sys.modules if any(tok in name for tok in DEVICE_MODULES)]


# ---------------------------------------------------------------------------
# 1. --help
# ---------------------------------------------------------------------------
def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--out", "--execute", "--exposure-bracket", "--sweep-step",
                 "--wait-until-idle", "--slm-number", "--cam-id"):
        assert flag in result.stdout


# ---------------------------------------------------------------------------
# 2. Plan mode: side-effect-free, no driver import (clean subprocess)
# ---------------------------------------------------------------------------
def test_plan_mode_is_side_effect_free_and_imports_no_driver(tmp_path):
    out = tmp_path / "must_not_exist"
    code = f"""
import importlib, json, sys
m = importlib.import_module("scripts.slm.issue238_gray_phase_probes")
rc = m.main(["--out", r"{out}"])
print("__RC__", rc)
print("__DEVMODS__", [n for n in sys.modules if any(t in n for t in {DEVICE_MODULES!r})])
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "libs"), str(ROOT), str(ROOT / "scripts")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "__RC__ 0" in result.stdout
    assert "__DEVMODS__ []" in result.stdout, result.stdout
    assert "skip: requires --execute" in result.stderr
    assert not out.exists(), "plan mode must not create --out"
    # The JSON plan was printed to stdout.
    payload = result.stdout.split("__RC__")[0]
    plan = json.loads(payload)
    assert plan["g_ref"] == 1023
    assert plan["two_pi_gray_true"] == "pending offline fit"
    assert plan["n_sweep"] == 65


# ---------------------------------------------------------------------------
# 3. _run with existing --out: return 2, lock released once, no driver import
# ---------------------------------------------------------------------------
class _FakeLock:
    def __init__(self):
        self.release_count = 0

    def release(self):
        self.release_count += 1


def test_run_existing_out_releases_lock_without_driver_import(tmp_path, monkeypatch):
    existing = tmp_path / "already_here"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not touch", encoding="utf-8")

    fake = _FakeLock()
    monkeypatch.setattr(SCRIPT, "acquire_device_lock", lambda wait_s: fake)

    baseline = set(_device_modules_loaded())
    real_import = builtins.__import__

    def guard(name: str, *a, **k):
        if any(tok in name for tok in DEVICE_MODULES):
            raise AssertionError(f"device driver imported on failure path: {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)

    args = argparse.Namespace(out=existing, wait_until_idle=0)
    plan: dict = {}
    result = SCRIPT._run(args, plan)

    assert result == 2
    assert fake.release_count == 1
    assert set(_device_modules_loaded()) == baseline
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "do not touch"
    # No release.json / device_log.json when we never created the directory.
    assert not (existing / "release.json").exists()


# ---------------------------------------------------------------------------
# 4. Pure pattern / sweep helpers
# ---------------------------------------------------------------------------
def test_gray_sweep_has_65_points_and_endpoint_1023():
    sweep = SCRIPT.gray_sweep(16)
    assert len(sweep) == 65
    assert sweep[0] == 0
    assert sweep[-1] == 1023
    assert sweep[1] == 16
    assert SCRIPT.G_REF == 1023
    assert SCRIPT.TWO_PI_GRAY_ASSUMED == 993


def test_blaze_peak_equals_gray_and_period_is_16():
    shape = (120, 160)
    g = 480
    arr = SCRIPT.blaze_gray(shape, gray=g, period_px=16, axis=1)
    assert arr.shape == shape
    assert arr.dtype == np.uint16
    assert int(arr.max()) == g
    # One full period repeats along x.
    np.testing.assert_array_equal(arr[:, :16], arr[:, 16:32])
    # Sawtooth: within one period the first and last pixels are the extremes.
    assert int(arr[0, 0]) == 0
    assert int(arr[0, 15]) == g


def test_checkerboard_has_16px_period():
    shape = (120, 160)
    arr = SCRIPT.checkerboard_gray(shape, gray=480, period_px=16)
    assert arr.dtype == np.uint16
    # Full checker period is 16 px (8 px cells) in both x and y.
    np.testing.assert_array_equal(arr[:, :16], arr[:, 16:32])
    np.testing.assert_array_equal(arr[:16, :], arr[16:32, :])
    # Alternating 8 px blocks within a row; parity flips on the next cell row.
    np.testing.assert_array_equal(arr[0, :8], np.zeros(8, dtype=np.uint16))
    np.testing.assert_array_equal(arr[0, 8:16], np.full(8, 480, dtype=np.uint16))
    np.testing.assert_array_equal(arr[8, :8], np.full(8, 480, dtype=np.uint16))
    np.testing.assert_array_equal(arr[8, 8:16], np.zeros(8, dtype=np.uint16))


def test_clamp_warns_and_never_exceeds_1023():
    shape = (32, 32)
    arr = SCRIPT.blaze_gray(shape, gray=1023, period_px=16, axis=1)
    assert int(arr.max()) == 1023
    assert arr.dtype == np.uint16
    # _clamp_gray reports overflow explicitly.
    over = np.full((4, 4), 2000, dtype=np.int64)
    clamped, warning = SCRIPT._clamp_gray(over)
    assert int(clamped.max()) == 1023
    assert warning is not None and "clamped" in warning


def test_crosstalk_grid_center_differs_from_neighbors():
    shape = (1200, 1920)
    arr = SCRIPT.crosstalk_gray(
        shape, gray=993, neighbor_gray=0, grid_n=3, cell_px=128, gap_px=64
    )
    assert arr.shape == shape
    assert arr.dtype == np.uint16
    # The very center of the panel is the modulated cell (non-flat).
    center_block = arr[600 - 20 : 600 + 20, 960 - 20 : 960 + 20]
    assert center_block.max() > 0
    # A corner cell is flat zero.
    side = 3 * 128 + 2 * 64
    y0 = (1200 - side) // 2
    x0 = (1920 - side) // 2
    corner = arr[y0 + 10 : y0 + 40, x0 + 10 : x0 + 40]
    assert corner.max() == 0


# ---------------------------------------------------------------------------
# 5. Binary phase-grating mode (gamma <-> 2*pi-gray degeneracy breaker)
# ---------------------------------------------------------------------------
def _binary_args(**overrides) -> argparse.Namespace:
    base = dict(
        mode="binary",
        out=Path("unused"),
        wait_until_idle=0,
        sweep_step=16,
        period_px=16,
        checker_frames=20,
        exposure_bracket=(0.02, 0.04, 0.08, 0.16),
        frames_per_level=16,
        dense_lo=864,
        dense_hi=1023,
        dense_step=8,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _blaze_args(**overrides) -> argparse.Namespace:
    return _binary_args(mode="blaze", **overrides)


def test_binary_gray_is_two_level_square_wave():
    shape = (120, 160)
    g = 512
    arr = SCRIPT.binary_gray(shape, gray=g, period_px=16, duty=0.5, axis=1)
    assert arr.shape == shape
    assert arr.dtype == np.uint16
    # Exactly two levels: 0 and g (symmetric grating -> both +/-1 orders).
    assert sorted(set(arr.ravel().tolist())) == [0, g]
    # 50% duty: 8 px at g then 8 px at 0, repeating with period 16.
    np.testing.assert_array_equal(arr[0, :8], np.full(8, g, dtype=np.uint16))
    np.testing.assert_array_equal(arr[0, 8:16], np.zeros(8, dtype=np.uint16))
    np.testing.assert_array_equal(arr[:, :16], arr[:, 16:32])
    # Columns are constant (grating varies along x only).
    np.testing.assert_array_equal(arr[:, 0], np.full(shape[0], g, dtype=np.uint16))

    # axis=0 rotates the grating (rows constant, varies along y).
    arr_y = SCRIPT.binary_gray(shape, gray=g, period_px=16, duty=0.5, axis=0)
    np.testing.assert_array_equal(arr_y[0, :], np.full(shape[1], g, dtype=np.uint16))
    np.testing.assert_array_equal(arr_y[:16, :], arr_y[16:32, :])

    with pytest.raises(ValueError):
        SCRIPT.binary_gray(shape, gray=g, period_px=16, duty=0.0)
    with pytest.raises(ValueError):
        SCRIPT.binary_gray(shape, gray=g, period_px=1)


def test_binary_sweep_levels_sorted_unique_and_brackets_zero():
    levels = SCRIPT.binary_sweep_levels(
        coarse_step=32, dense_lo=864, dense_hi=1023, dense_step=8
    )
    assert levels == tuple(sorted(set(levels)))
    assert 0 in levels
    assert 1023 in levels
    # Coarse grid present (step 32) and dense band present (step 8 near 2*pi).
    assert 32 in levels
    for g in (864, 872, 880, 1016, 1023):
        assert g in levels
    # Dense band yields more points than the coarse grid alone above 864.
    above = [v for v in levels if v >= 864]
    assert len(above) >= 18
    with pytest.raises(ValueError):
        SCRIPT.binary_sweep_levels(coarse_step=0)


def test_plan_mode_binary_has_bracket_sweep_and_drift():
    plan = SCRIPT.build_plan(_binary_args())
    assert plan["schema"] == "issue238-gray-phase-probes/v2-binary"
    assert plan["mode"] == "binary"
    groups = {p["group"] for p in plan["patterns"]}
    assert {"reference", "bracket", "sweep", "drift"} <= groups
    # No crosstalk/checker work in the focused binary plan.
    assert "crosstalk" not in groups
    assert plan["full_scale_required"] == 4095
    assert plan["binary_duty"] == 0.5
    # Every sweep item is a binary grating with the per-level frame count.
    sweep_items = [p for p in plan["patterns"] if p["group"] == "sweep"]
    assert len(sweep_items) == plan["n_sweep"]
    assert {p["kind"] for p in sweep_items} == {"binary"}
    assert {p["frames"] for p in sweep_items} == {16}
    assert [p["gray"] for p in sweep_items] == plan["binary_sweep"]
    # Bracket group carries one frame per bracket step at the flat pattern.
    bracket_items = [p for p in plan["patterns"] if p["group"] == "bracket"]
    assert len(bracket_items) == 1
    assert bracket_items[0]["frames"] == 4
    # Frame accounting is consistent with the plan.
    assert plan["n_frames_expected"] == sum(p["frames"] for p in plan["patterns"])


def test_plan_mode_default_stays_blaze():
    plan = SCRIPT.build_plan(_blaze_args())
    assert plan["schema"] == "issue238-gray-phase-probes/v1"
    assert plan.get("mode", "blaze") == "blaze"
    assert plan["n_sweep"] == 65
    groups = {p["group"] for p in plan["patterns"]}
    assert "crosstalk" in groups and "checker" in groups
    assert plan["full_scale_required"] == 4095


# ---------------------------------------------------------------------------
# 6. choose_exposure: headroom-aware bracket selection (2026-11-19 fix)
# ---------------------------------------------------------------------------
def _usable_rows():
    # Real 2026-11-19 bracket: full scale 4095, saturation 0.9.
    return [
        {"readback_ms": 0.02, "peak": 598.0, "saturated_pixels": 0},
        {"readback_ms": 0.04, "peak": 1007.0, "saturated_pixels": 0},
        {"readback_ms": 0.08, "peak": 2113.0, "saturated_pixels": 0},
        {"readback_ms": 0.16, "peak": 3519.0, "saturated_pixels": 0},
    ]


def test_choose_exposure_prefers_headroom_over_brightest():
    # 0.16 ms (peak 3519 = 86% of full scale) is unsaturated in the bracket
    # but tips 6 flat pixels over the line on re-capture; 0.08 ms must win.
    chosen = SCRIPT.choose_exposure(_usable_rows(), 4095)
    assert chosen["readback_ms"] == 0.08


def test_choose_exposure_includes_row_exactly_at_ceiling():
    preferred = SCRIPT.PREFERRED_PEAK_FRACTION * 4095
    rows = _usable_rows() + [
        {"readback_ms": 0.1, "peak": preferred, "saturated_pixels": 0}
    ]
    assert SCRIPT.choose_exposure(rows, 4095)["readback_ms"] == 0.1


def test_choose_exposure_falls_back_when_all_above_ceiling():
    rows = [
        {"readback_ms": 0.04, "peak": 3000.0, "saturated_pixels": 0},
        {"readback_ms": 0.08, "peak": 3200.0, "saturated_pixels": 0},
    ]
    # preferred = 2866.5, both above -> brightest usable wins (0.08).
    assert SCRIPT.choose_exposure(rows, 4095)["readback_ms"] == 0.08


def test_choose_exposure_rejects_empty():
    with pytest.raises(ValueError):
        SCRIPT.choose_exposure([], 4095)
