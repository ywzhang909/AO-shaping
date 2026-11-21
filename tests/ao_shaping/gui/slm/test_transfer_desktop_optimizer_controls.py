"""可微优化控件参数映射与早停边界。"""

from __future__ import annotations

from ao_shaping.gui.slm.transfer_desktop.params import (
    TransferFormValues,
    build_session,
    validate_form,
)


def test_optimizer_defaults_match_transfer_pipeline() -> None:
    values = TransferFormValues()

    assert values.phase_lr == 0.30
    assert values.iterations == 200
    assert values.patience == 8
    assert values.min_delta == 1e-4


def test_build_session_passes_optimizer_controls(tmp_path) -> None:
    values = TransferFormValues(
        phase_lr=0.075,
        iterations=321,
        patience=11,
        min_delta=2.5e-5,
    )

    session = build_session(values, out=tmp_path / "session")

    assert session.phase_lr == 0.075
    assert session.iterations == 321
    assert session.patience == 11
    assert session.min_delta == 2.5e-5


def test_optimizer_controls_reject_invalid_values() -> None:
    cases = (
        (TransferFormValues(phase_lr=0.0), "学习率"),
        (TransferFormValues(iterations=0), "拟合迭代"),
        (TransferFormValues(patience=0), "patience"),
        (TransferFormValues(min_delta=-1e-4), "min_delta"),
    )

    for values, expected in cases:
        errors, _warnings = validate_form(values)
        assert any(expected in error for error in errors), (expected, errors)
