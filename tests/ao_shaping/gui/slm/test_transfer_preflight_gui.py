"""GUI 侧的启动前检查接线（迁移学习闭环整形）。

2026-10-26 实测：操作员把载波周期填成 4 panel px，运行在第一个 CCD 分块才报
``empty target core`` —— 采集与拟合已经跑了几分钟。这些测试钉住「按「开始」
之前就拒绝，并给出可执行的改法」。
"""

from __future__ import annotations

from typing import Any

import pytest

from ao_shaping.gui.slm.shaping_controls import TransferMethod


def _values(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "cam_type": "daheng",
        "cam_id": 0,
        "slm_number": 1,
        "shapes": ["square"],
        "side": 80.0,
        "period": 64.0,
        "count": 24,
        "iterations": 60,
        "blocks": 4,
        "updates": 10,
        "frames": 3,
        "seed": 102319,
        "adaptation": "auto",
        "accept_metric": "loss",
        "objective": "shape",
        "seed_policy": "strict",
        "abba": True,
        "save_dataset": False,
        "device": "auto",
        "initial_phase": "gs",
    }
    base.update(overrides)
    return base


def _method() -> TransferMethod:
    return TransferMethod("slm1", 1)


def test_the_carrier_period_that_broke_the_run_is_rejected_before_the_run() -> None:
    """period=4：按钮该是禁用的，提示里要有「怎么改」。"""
    message = _method().validate(_values(period=4.0))

    assert message is not None
    assert "4" in message
    assert "panel px" in message
    assert "→" in message


def test_a_probe_count_that_is_not_a_multiple_of_six_is_rejected() -> None:
    message = _method().validate(_values(count=10))

    assert message is not None
    assert "10" in message


def test_the_default_parameters_pass_the_preflight() -> None:
    """默认参数必须通过（否则检查本身就成了噪音）。"""
    assert _method().validate(_values()) is None


def test_a_target_that_cannot_fit_the_window_is_rejected() -> None:
    """边长 1000 px 超出采样窗（半跨 220 px ⇒ 上限 424 px）：启动前就拒绝。

    边长 200 px 这一档**参数级**放得过（它要等实测位移才知道放不放得下），
    由拟合之后的几何检查兜底 —— 见
    ``tests/ao_shaping/tools/slm/test_transfer_preflight.py``。
    """
    message = _method().validate(_values(side=1000.0))

    assert message is not None
    assert "边长" in message
