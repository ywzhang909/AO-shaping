"""Issue #238 束功率参考轨 (drift decoupling) 的离线单测 (2026-11-20)。

覆盖四组纯函数与计划行为, **一律不碰硬件、不写 data/**:

1. ``interpolate_power_scale``: 合成 +30% 线性漂移在区间内恢复误差 <1%,
   区间外按文档的「边缘常数保持」外推;
2. ``abba_schedule``: 3 个 block 的参考槽严格 pre .. post 括住各自 block,
   相邻 block 不共用参考帧;
3. plan 层: ``--power-reference off`` 不产生 ``power_reference`` 键、帧数与
   旧参数完全一致; ``abba`` 恰好增加 ``2 * frames_per_ref * N`` 帧, 且
   ``block_index`` / ``ref_position`` / 时间顺序一致;
4. 窗口: ``palindrome`` 周期顺序、``as-given`` 不变、单周期回文被
   ``validate`` 拒绝、plan-only 不建目录。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CAPTURE = importlib.import_module("scripts.slm.issue238_gray_phase_probes")
SWEEP = importlib.import_module("scripts.experiments.run_issue238_period_sweep_20261119")

DEVICE_MODULES = ("daheng", "gxipy", "santec", "_slm_win")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _binary_args(**over):
    base = dict(
        mode="binary",
        period_px=32,
        frames_per_level=3,
        dense_lo=864,
        dense_hi=1023,
        dense_step=8,
        exposure_bracket=[0.02, 0.04, 0.08, 0.16],
        power_reference="off",
        power_reference_frames=2,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _legacy_binary_args():
    """Pre-change caller: no power-reference attributes at all."""
    return argparse.Namespace(
        mode="binary",
        period_px=32,
        frames_per_level=3,
        dense_lo=864,
        dense_hi=1023,
        dense_step=8,
        exposure_bracket=[0.02, 0.04, 0.08, 0.16],
    )


def _window_args(**over):
    base = dict(
        run_dir=None,
        periods_px=[32, 48],
        period_order="as-given",
        frames_per_level=4,
        dense_lo=None,
        dense_hi=None,
        dense_step=None,
        checker_frames=None,
        settle_s=None,
        exposure_bracket=None,
        power_reference="off",
        power_reference_frames=2,
        wait_until_idle=600.0,
        slm_number=1,
        cam_id=0,
        execute=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _strip_timestamps(plan):
    payload = json.loads(json.dumps(plan))
    payload.pop("timestamp_utc", None)
    return payload


# ---------------------------------------------------------------------------
# 1. estimator
# ---------------------------------------------------------------------------
def test_interpolate_power_scale_recovers_linear_drift():
    ref_times = [0.0, 33.0, 66.0, 100.0]
    ref_totals = [100.0, 110.0, 120.0, 130.0]  # +30% over the run
    frame_times = [10.0, 25.0, 50.0, 75.0, 95.0]
    scale = CAPTURE.interpolate_power_scale(frame_times, ref_times, ref_totals)
    mean_total = float(np.mean(ref_totals))
    expected = np.array([100.0 + 0.3 * t for t in frame_times]) / mean_total
    assert scale.shape == (len(frame_times),)
    assert np.all(np.abs(scale - expected) <= 0.01 * expected)


def test_interpolate_power_scale_edges_hold_nearest_reference():
    ref_times = [0.0, 33.0, 66.0, 100.0]
    ref_totals = [100.0, 110.0, 120.0, 130.0]
    mean_total = float(np.mean(ref_totals))
    scale = CAPTURE.interpolate_power_scale([-10.0, 0.0, 100.0, 150.0],
                                            ref_times, ref_totals)
    # Flat (constant) extrapolation, never a fitted slope beyond the ends.
    assert scale[0] == pytest.approx(100.0 / mean_total)
    assert scale[1] == pytest.approx(100.0 / mean_total)
    assert scale[2] == pytest.approx(130.0 / mean_total)
    assert scale[3] == pytest.approx(130.0 / mean_total)
    assert scale[3] != pytest.approx(145.0 / mean_total)


def test_interpolate_power_scale_rejects_bad_references():
    with pytest.raises(ValueError):
        CAPTURE.interpolate_power_scale([0.0], [1.0, 2.0], [3.0])
    with pytest.raises(ValueError):
        CAPTURE.interpolate_power_scale([0.0], [2.0, 1.0], [3.0, 3.0])
    with pytest.raises(ValueError):
        CAPTURE.interpolate_power_scale([0.0], [1.0], [0.0])


# ---------------------------------------------------------------------------
# 2. ABBA schedule
# ---------------------------------------------------------------------------
def test_abba_schedule_brackets_each_block_without_sharing_refs():
    schedule = CAPTURE.abba_schedule([2, 3, 1], 2)
    for block_index, size in enumerate([2, 3, 1]):
        roles = [s["role"] for s in schedule if s["block_index"] == block_index]
        assert roles == ["pre", "pre"] + ["sweep"] * size + ["post", "post"]
    refs = [
        (s["block_index"], s["role"], s["ref_slot"])
        for s in schedule
        if s["role"] != "sweep"
    ]
    assert len(refs) == 2 * 2 * 3
    assert len(set(refs)) == len(refs)
    # The schedule is globally pre .. post around each block, in plan order.
    assert [s["block_index"] for s in schedule] == sorted(
        s["block_index"] for s in schedule
    )
    assert schedule[0]["role"] == "pre"
    assert schedule[-1]["role"] == "post"


def test_abba_schedule_rejects_empty_inputs():
    with pytest.raises(ValueError):
        CAPTURE.abba_schedule([3], 0)
    with pytest.raises(ValueError):
        CAPTURE.abba_schedule([], 2)


# ---------------------------------------------------------------------------
# 3. plans
# ---------------------------------------------------------------------------
def test_plan_off_is_unchanged_and_has_no_power_reference_key():
    legacy = CAPTURE.build_plan(_legacy_binary_args())
    explicit_off = CAPTURE.build_plan(_binary_args(power_reference="off"))
    assert "power_reference" not in explicit_off
    assert _strip_timestamps(legacy) == _strip_timestamps(explicit_off)
    assert legacy["n_frames_expected"] == explicit_off["n_frames_expected"]
    assert legacy["n_patterns"] == explicit_off["n_patterns"]


def test_plan_abba_adds_exactly_two_sided_reference_frames():
    frames_per_side = 3
    off = CAPTURE.build_plan(_binary_args())
    abba = CAPTURE.build_plan(
        _binary_args(power_reference="abba", power_reference_frames=frames_per_side)
    )
    assert abba["power_reference"]["mode"] == "abba"
    assert abba["power_reference"]["frames_per_side"] == frames_per_side
    assert (
        abba["n_frames_expected"] - off["n_frames_expected"]
        == 2 * frames_per_side * 1
    )
    refs = [p for p in abba["patterns"] if p["group"] == "power_reference"]
    assert len(refs) == 2 * frames_per_side
    assert [p["ref_position"] for p in refs] == (
        ["pre"] * frames_per_side + ["post"] * frames_per_side
    )
    assert {p["block_index"] for p in refs} == {0}
    block = abba["power_reference"]["blocks"][0]
    assert len(block["pre_pattern_indices"]) == frames_per_side
    assert len(block["post_pattern_indices"]) == frames_per_side
    sweep_indices = [i for i, p in enumerate(abba["patterns"]) if p["group"] == "sweep"]
    assert max(block["pre_pattern_indices"]) < min(sweep_indices)
    assert min(block["post_pattern_indices"]) > max(sweep_indices)
    # Same 1-frame-per-reference-item treatment as ordinary plan items.
    assert all(p["frames"] == 1 for p in refs)
    assert all(p["kind"] == CAPTURE.POWER_REFERENCE_KIND for p in refs)


def test_plan_abba_is_binary_only():
    with pytest.raises(ValueError):
        CAPTURE.build_plan(
            argparse.Namespace(
                mode="blaze",
                exposure_bracket=[0.02, 0.04, 0.08, 0.16],
                power_reference="abba",
                power_reference_frames=2,
            )
        )


def test_summary_power_reference_is_descriptive_and_ordered():
    records = [
        dict(index=0, group="power_reference", ref_position="pre", block_index=0,
             block_id="block0_p32", period_px=32, frame_total=100, monotonic_ns=0),
        dict(index=1, group="power_reference", ref_position="pre", block_index=0,
             block_id="block0_p32", period_px=32, frame_total=102, monotonic_ns=1_000_000_000),
        dict(index=2, group="sweep", ref_position=None, block_index=0,
             block_id="block0_p32", period_px=32, frame_total=104, monotonic_ns=2_000_000_000),
        dict(index=3, group="sweep", ref_position=None, block_index=0,
             block_id="block0_p32", period_px=32, frame_total=106, monotonic_ns=3_000_000_000),
        dict(index=4, group="sweep", ref_position=None, block_index=0,
             block_id="block0_p32", period_px=32, frame_total=108, monotonic_ns=4_000_000_000),
        dict(index=5, group="power_reference", ref_position="post", block_index=0,
             block_id="block0_p32", period_px=32, frame_total=110, monotonic_ns=5_000_000_000),
        dict(index=6, group="power_reference", ref_position="post", block_index=0,
             block_id="block0_p32", period_px=32, frame_total=112, monotonic_ns=6_000_000_000),
    ]
    section = CAPTURE.summarize_power_reference(records, frames_per_side=2)
    assert section["descriptive_only"] is True
    assert "offline" in section["note"]
    block = section["blocks"][0]
    assert block["pre"]["raw_totals"] == [100.0, 102.0]
    assert block["post"]["raw_totals"] == [110.0, 112.0]
    assert block["drift_ratio_post_over_pre"] == pytest.approx(111.0 / 101.0)
    scales = [f["scale"] for f in section["frames"]]
    assert scales == sorted(scales)  # monotonically rising beam power
    assert len(scales) == 3
    # No g2pi claim anywhere in the section.
    assert "two_pi" not in json.dumps(section)


# ---------------------------------------------------------------------------
# 4. window wrapper
# ---------------------------------------------------------------------------
def test_ordered_periods_palindrome_and_as_given():
    assert SWEEP.ordered_periods([32, 48, 64], "as-given") == [32, 48, 64]
    assert SWEEP.ordered_periods([32, 48, 64], "palindrome") == [
        32,
        48,
        64,
        64,
        48,
        32,
    ]
    assert SWEEP.ordered_periods([32, 48], "palindrome") == [32, 48, 48, 32]


def test_validate_rejects_single_period_palindrome():
    problems = SWEEP.validate(
        _window_args(run_dir="x", periods_px=[32], period_order="palindrome")
    )
    assert any("palindrome" in p for p in problems)
    assert SWEEP.validate(
        _window_args(run_dir="x", periods_px=[32, 48], period_order="palindrome")
    ) == []


def test_build_segments_palindrome_order_and_unique_names(tmp_path):
    args = _window_args(
        periods_px=[32, 48, 64],
        period_order="palindrome",
        run_dir=str(tmp_path / "win"),
        power_reference="abba",
        power_reference_frames=2,
    )
    segments = SWEEP.build_segments(args, Path(args.run_dir))
    assert [s.name for s in segments] == [
        "period32",
        "period48",
        "period64",
        "period64_r2",
        "period48_r2",
        "period32_r2",
    ]
    assert [s.out.name for s in segments] == [s.name for s in segments]
    # ABBA flags forwarded to every segment; default off forwards nothing.
    for segment in segments:
        argv = list(segment.argv)
        assert argv[argv.index("--power-reference") + 1] == "abba"
        assert argv[argv.index("--power-reference-frames") + 1] == "2"
    off = SWEEP.build_segments(
        _window_args(periods_px=[32, 48], run_dir=str(tmp_path / "win")),
        tmp_path / "win",
    )
    assert all("--power-reference" not in list(s.argv) for s in off)
    assert [s.name for s in off] == ["period32", "period48"]


def test_plan_only_window_creates_no_directory(tmp_path):
    run_dir = tmp_path / "window_must_not_exist"
    rc = SWEEP.main(["--run-dir", str(run_dir), "--periods-px", "32", "48"])
    assert rc == 0
    assert not run_dir.exists()


def test_no_device_driver_is_imported():
    loaded = [n for n in sys.modules if any(tok in n for tok in DEVICE_MODULES)]
    assert loaded == []
