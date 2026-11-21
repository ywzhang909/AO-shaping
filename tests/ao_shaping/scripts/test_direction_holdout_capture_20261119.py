"""``scripts/slm/direction_holdout_capture_20261119.py`` 的离线测试。

只验证纯逻辑（计划表与探针序列），不打开 SLM/CCD/激光：
- ``build_plan`` 必须给出 14 行、顺序为每模式先 +amp 后 -amp、split 全为 test；
- ``make_probes`` 的标定段必须等于 ``capture.probes(...)[:-1]``，末尾补零相位
  ``reference_end``。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/slm/direction_holdout_capture_20261119.py"
)


def _load():
    """按路径加载脚本模块（``scripts/`` 不是可导入包的手工脚本目录）。"""
    spec = importlib.util.spec_from_file_location(
        "direction_holdout_capture_20261119_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_holdout = _load()


def test_build_plan_has_fourteen_rows_all_test_split() -> None:
    """14 行 = 7 模式 × 2 符号，split 一律 test。"""
    plan = capture_holdout.build_plan()
    assert len(plan) == 14
    assert {row[1] for row in plan} == {"test"}
    assert [row[2] for row in plan] == [5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11]


def test_build_plan_labels_and_ordering_plus_before_minus() -> None:
    """标签编码模式与带符号幅度，且每个模式先 +amp 后 -amp。"""
    plan = capture_holdout.build_plan()
    assert plan[0] == ("holdout_000_Z5_+0.05", "test", 5, 0.05)
    assert plan[1] == ("holdout_001_Z5_-0.05", "test", 5, -0.05)
    assert plan[-1] == ("holdout_013_Z11_-0.05", "test", 11, -0.05)
    for index, (label, split, mode, amp) in enumerate(plan):
        assert label == f"holdout_{index:03d}_Z{mode}_{amp:+.2f}"


def test_build_plan_respects_custom_amp() -> None:
    """幅度可覆盖，标签随之格式化。"""
    plan = capture_holdout.build_plan((5, 6), amp=0.1)
    assert [row[0] for row in plan] == ["holdout_000_Z5_+0.10", "holdout_001_Z5_-0.10",
                                        "holdout_002_Z6_+0.10", "holdout_003_Z6_-0.10"]


def test_make_probes_keeps_calibration_prefix_and_appends_reference_end() -> None:
    """标定段 == capture.probes[:-1]；随后 14 个 holdout；末尾零相位 reference_end。"""
    plan = capture_holdout.build_plan()
    rows = list(capture_holdout.make_probes(plan, seed=1104229)(128, 0, 1104229))

    expected_calibration = list(
        capture_holdout.capture.probes(128, 0, 1104229, (64, 128))
    )[:-1]
    assert [row[0] for row in rows[: len(expected_calibration)]] == [
        row[0] for row in expected_calibration
    ]
    assert len(expected_calibration) == 9

    holdout = rows[len(expected_calibration) : -1]
    assert [row[0] for row in holdout] == [row[0] for row in plan]
    assert all(row[1] == "test" for row in holdout)
    assert all(np.asarray(row[2]).shape == (128, 128) for row in holdout)

    label, split, phase = rows[-1]
    assert label == "reference_end"
    assert split == "calibration"
    assert not np.any(np.asarray(phase))
