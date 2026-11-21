"""Issue #238 period-sweep 窗口: 暗场标志透传 (离线, 零设备)。

窗口脚本只是 ``issue238_gray_phase_probes.py`` 的瘦封装。B1 计划用
``--dark-frames 20`` 采集遮光暗场, 因此窗口必须:

1. 缺省 (``--dark-frames 0``) 不产生任何 dark 标志 —— 历史 argv 逐字不变;
2. 显式给出时把 ``--dark-frames`` (与可选 ``--dark-at``) 原样转发给探针;
3. 拒绝负帧数, 以及只给 ``--dark-at`` 却不给 ``--dark-frames`` 的组合。

本测试只 import 脚本, 不 import 任何驱动、不建目录、不碰设备。
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pytest

WINDOW = importlib.import_module("scripts.experiments.run_issue238_period_sweep_20261119")


def _args(**overrides: object) -> argparse.Namespace:
    """全部 ``validate`` / ``shared_flags`` / ``build_segments`` 读取的属性。"""
    base: dict[str, object] = dict(
        run_dir=None,
        periods_px=[32, 48],
        period_order="as-given",
        frames_per_level=20,
        dense_lo=None,
        dense_hi=None,
        dense_step=None,
        checker_frames=None,
        settle_s=None,
        exposure_bracket=None,
        power_reference="off",
        power_reference_frames=2,
        dark_frames=0,
        dark_at=None,
        wait_until_idle=0.0,
        slm_number=1,
        cam_id=0,
        cooldown_s=200.0,
        segment_timeout_s=3600.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_shared_flags_have_no_dark_tokens():
    flags = WINDOW.shared_flags(_args())
    assert "--dark-frames" not in flags
    assert "--dark-at" not in flags


def test_dark_frames_forwarded_without_explicit_at():
    flags = WINDOW.shared_flags(_args(dark_frames=20))
    assert flags[flags.index("--dark-frames") + 1] == "20"
    # 未给 --dark-at 时不转发, 让探针按 N>0 默认 both。
    assert "--dark-at" not in flags


def test_dark_frames_and_at_forwarded():
    flags = WINDOW.shared_flags(_args(dark_frames=20, dark_at="pre"))
    assert flags[flags.index("--dark-frames") + 1] == "20"
    assert flags[flags.index("--dark-at") + 1] == "pre"


def test_build_segments_carries_dark_flags_to_probe_argv():
    run_dir = Path("C:/tmp/issue238_dark_test")
    segments = WINDOW.build_segments(_args(dark_frames=20, dark_at="both"), run_dir)
    assert len(segments) == 2
    for segment in segments:
        argv = list(segment.argv)
        assert argv[argv.index("--dark-frames") + 1] == "20"
        assert argv[argv.index("--dark-at") + 1] == "both"
        # 探针在 --mode binary 下才接受暗场, 窗口必须始终传 binary。
        assert argv[argv.index("--mode") + 1] == "binary"


def test_validate_rejects_negative_dark_frames():
    problems = WINDOW.validate(_args(run_dir="x", dark_frames=-1))
    assert any("--dark-frames" in p for p in problems)


def test_validate_rejects_dark_at_without_frames():
    problems = WINDOW.validate(_args(run_dir="x", dark_frames=0, dark_at="pre"))
    assert any("--dark-at" in p for p in problems)


def test_validate_accepts_dark_request_when_consistent():
    problems = WINDOW.validate(_args(run_dir="x", dark_frames=20, dark_at="both"))
    assert not any("--dark" in p for p in problems)
