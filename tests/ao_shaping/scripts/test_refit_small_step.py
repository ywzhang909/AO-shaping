"""小步重拟合（冻结 d / 照明 / 残差）的机制测试。

只测**冻结机制**与**参数解析**，不构造真实 `PhysicalForwardFit`（那需要真实
数据与大幅 FFT）。冻结机制是这次改动的全部风险点：`d`、`center`、`log_width`、
`residual` 在每步优化后必须被逐位还原，否则"小步"名不副实。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.slm.refit_model_after_optics_change import (
    build_parser,
    restore_frozen,
    snapshot_parameters,
)


class _StubModel:
    """与 `PhysicalForwardFit` 同名的可冻结参数的最小替身。"""

    def __init__(self) -> None:
        self.d = torch.tensor(0.5)
        self.origin = torch.tensor([1.0, 2.0])
        self.center = torch.tensor([10.0, 20.0])
        self.log_width = torch.tensor([1.0, 1.0])
        self.residual = torch.tensor([0.1, -0.2, 0.3])


def test_snapshot_covers_all_frozen_parameters() -> None:
    model = _StubModel()
    snapshots = snapshot_parameters(model)
    assert set(snapshots) == {"d", "origin", "center", "log_width", "residual"}


def test_restore_puts_back_exact_values() -> None:
    model = _StubModel()
    before = snapshot_parameters(model)

    # 模拟优化器把参数推走
    with torch.no_grad():
        model.d.add_(1.0)
        model.center.mul_(3.0)
        model.residual.add_(0.5)

    restore_frozen(model, before)

    for name, value in before.items():
        assert torch.equal(getattr(model, name), value), name


def test_restore_only_touches_named_parameters() -> None:
    model = _StubModel()
    before = snapshot_parameters(model)
    only_origin = {"origin": before["origin"]}

    with torch.no_grad():
        model.d.add_(2.0)          # 未列入冻结 ⇒ 应保持不变
        model.origin.add_(7.0)     # 列入 ⇒ 应被还原

    restore_frozen(model, only_origin)

    assert torch.equal(model.origin, before["origin"])
    assert float(model.d) == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("flags", "expect_frozen"),
    [
        (["--small-step"], {"d", "center", "log_width", "residual"}),
        (["--freeze-defocus"], {"d"}),
        (["--freeze-amplitude"], {"center", "log_width"}),
        (["--freeze-residual"], {"residual"}),
        (["--freeze-defocus", "--freeze-residual"], {"d", "residual"}),
        ([], set()),
    ],
)
def test_flag_to_frozen_set(flags: list[str], expect_frozen: set[str]) -> None:
    """flag 组合必须映射成预期的冻结集合（`--small-step` 是三者之和）。"""
    args = build_parser().parse_args(
        ["--dataset", "d", "--checkpoint", "c", "--out", "o", *flags]
    )
    small = bool(getattr(args, "small_step", False))
    learn_amplitude = not bool(args.freeze_amplitude or small)
    frozen = {
        "d": bool(args.freeze_defocus or small),
        "center": not learn_amplitude,
        "log_width": not learn_amplitude,
        "residual": bool(args.freeze_residual or small),
    }
    assert {name for name, is_frozen in frozen.items() if is_frozen} == expect_frozen


def test_default_parser_keeps_learning_everything() -> None:
    args = build_parser().parse_args(["--dataset", "d", "--checkpoint", "c", "--out", "o"])
    assert np.allclose(
        [args.freeze_defocus, args.freeze_amplitude, args.freeze_residual, args.small_step],
        [False, False, False, False],
    )

