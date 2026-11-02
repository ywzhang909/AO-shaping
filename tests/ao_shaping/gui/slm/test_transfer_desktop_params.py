"""桌面页纯参数层的契约（默认值 / 校验顺序 / 会话目录 / 字段透传）。

这些测试与 Streamlit 页共用同一口径，但**不 import** 那个页面模块（它 import
Streamlit）。默认值以字面量钉住 ``TRANSFER_RECIPE``：改桌面页默认值而漏改另一
处时，这里会直接报出具体字段。
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ao_shaping.gui.slm.transfer_desktop import params
from ao_shaping.gui.slm.transfer_desktop.params import (
    CAMERA_CHOICES,
    SHAPE_CHOICES,
    SHAPE_LABELS,
    STAGE_LABELS,
    TransferFormValues,
    build_session,
    plan_session_dir,
    validate_form,
)
from ao_shaping.tools.slm.transfer import preflight


def _values(**overrides: object) -> TransferFormValues:
    return replace(TransferFormValues(), **overrides)


# --- 常量与默认值 -----------------------------------------------------------


def test_the_public_contract_is_the_documented_nine_names() -> None:
    assert set(params.__all__) == {
        "CAMERA_CHOICES",
        "SHAPE_CHOICES",
        "SHAPE_LABELS",
        "STAGE_LABELS",
        "TransferFormValues",
        "build_session",
        "effective_window_span",
        "plan_session_dir",
        "validate_form",
    }


def test_camera_choices_match_the_bench_camera_options() -> None:
    assert CAMERA_CHOICES == ("daheng", "miicam", "sim")


def test_shape_choices_are_the_four_transfer_shapes_in_display_order() -> None:
    assert SHAPE_CHOICES == ("square", "rectangle", "triangle", "star")
    assert SHAPE_LABELS == {
        "square": "正方形",
        "rectangle": "长方形",
        "triangle": "三角形",
        "star": "五角星",
    }
    assert "circle" not in SHAPE_CHOICES


def test_stage_labels_cover_every_stage_run_session_emits() -> None:
    assert set(STAGE_LABELS) == {
        "capture",
        "reuse_capture",
        "nominal_initialization",
        "fit",
        "CCD_model_blocks",
        "exposure_response_capture",
        "exposure_response_analysis",
        "blur_calibration",
        "CCD_anchored_blocks",
        "GS_and_differentiable_synthesis",
        "transferred_phase_differentiable_synthesis",
        "raw_data_audit",
        "summary",
        "direction_check",
    }


def test_stage_labels_do_not_drift_from_the_shared_progress_mapping() -> None:
    from ao_shaping.gui.slm.transfer_progress import STAGE_LABELS as shared

    assert STAGE_LABELS == shared


def test_form_defaults_mirror_the_streamlit_recipe() -> None:
    values = TransferFormValues()

    assert values.shapes == ("square",)
    assert values.side == 80.0
    assert values.period == 64.0
    assert values.count == 24
    assert values.iterations == 60
    assert values.blocks == 4
    assert values.updates == 10
    assert values.frames == 3
    assert values.seed == 102319
    assert values.camera == "daheng"
    assert values.cam_id == 0
    assert values.slm_number == 1
    assert values.calibration is True
    assert values.adaptation == "auto"
    assert values.objective == "shape"
    assert values.accept_metric == "loss"
    assert values.seed_policy == "transfer"
    assert values.abba is False
    assert values.adaptive_exposure is False
    assert values.save_dataset is False
    assert values.device == "auto"
    assert values.checkpoint == ""
    assert values.seed_run == ""


# --- 校验顺序 ---------------------------------------------------------------


def test_a_missing_torch_is_reported_before_anything_else(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    errors, warnings = validate_form(_values(camera="sim", shapes=(), period=4.0))

    assert len(errors) == 1
    assert errors[0].startswith("迁移学习闭环需要 torch")
    assert warnings == []


def test_a_non_daheng_camera_is_reported_before_shapes_and_preflight() -> None:
    pytest.importorskip("torch")

    errors, warnings = validate_form(_values(camera="sim", shapes=(), period=4.0))

    assert len(errors) == 1
    assert "大恒" in errors[0]
    assert "sim" in errors[0]
    assert warnings == []


def test_empty_shapes_are_reported_before_the_preflight() -> None:
    pytest.importorskip("torch")

    errors, warnings = validate_form(_values(shapes=(), period=4.0))

    assert errors == ["至少选择一个目标形状"]
    assert warnings == []


def test_the_default_parameters_pass_with_no_errors_or_warnings() -> None:
    pytest.importorskip("torch")

    errors, warnings = validate_form(TransferFormValues())

    assert errors == []
    assert warnings == []


def test_the_carrier_period_that_broke_the_run_is_an_error_with_a_fix() -> None:
    pytest.importorskip("torch")

    errors, warnings = validate_form(_values(period=4.0))

    assert warnings == []
    assert len(errors) == 1
    assert "panel px" in errors[0]
    assert "→" in errors[0]


def test_an_uncalibrated_period_is_a_warning_not_an_error() -> None:
    pytest.importorskip("torch")

    errors, warnings = validate_form(_values(period=100.0))

    assert errors == []
    assert len(warnings) == 1
    assert "100" in warnings[0]
    assert "→" in warnings[0]


def test_a_probe_count_that_is_not_a_multiple_of_six_is_rejected() -> None:
    pytest.importorskip("torch")

    errors, _ = validate_form(_values(count=10))

    assert len(errors) == 1
    assert "10" in errors[0]


def test_every_validation_entry_is_display_text_not_a_problem_object() -> None:
    pytest.importorskip("torch")

    errors, warnings = validate_form(_values(period=100.0, side=4.0, count=10))

    assert all(isinstance(text, str) for text in (*errors, *warnings))


# --- 会话目录 ---------------------------------------------------------------


def test_plan_session_dir_uses_the_shared_debug_convention(tmp_path) -> None:
    planned = plan_session_dir(tmp_path)

    assert planned.parent == tmp_path / "debug" / "slm_transfer"
    assert re.fullmatch(r"\d{8}_\d{6}(-\d+)?", planned.name)
    assert not planned.exists()
    assert not planned.parent.exists()


def test_plan_session_dir_never_reuses_an_existing_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(params, "format_ts", lambda: "20260101_000000")
    first = plan_session_dir(tmp_path)
    first.mkdir(parents=True)

    second = plan_session_dir(tmp_path)

    assert first.name == "20260101_000000"
    assert second.name == "20260101_000000-2"
    assert second.parent == first.parent
    assert re.fullmatch(r"\d{8}_\d{6}(-\d+)?", second.name)


def test_plan_session_dir_defaults_to_the_data_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(params, "format_ts", lambda: "20260101_000000")

    planned = plan_session_dir()

    assert planned == Path("data") / "debug" / "slm_transfer" / "20260101_000000"
    assert not planned.parent.exists()


# --- build_session 字段透传 -------------------------------------------------


def test_build_session_passes_every_form_field_through(tmp_path) -> None:
    out = tmp_path / "session"
    values = TransferFormValues(
        shapes=("triangle", "star"),
        side=42.5,
        period=128.0,
        count=36,
        iterations=12,
        blocks=2,
        updates=3,
        frames=5,
        seed=7,
        camera="daheng",
        cam_id=2,
        slm_number=3,
        calibration=False,
        adaptation="full",
        objective="cv",
        accept_metric="cv",
        seed_policy="strict",
        abba=True,
        adaptive_exposure=True,
        save_dataset=True,
        device="cpu",
        checkpoint=str(tmp_path / "fit" / "frozen_parameters.json"),
        seed_run=str(tmp_path / "anchored"),
    )

    session = build_session(values, out=out)

    assert session.out == out
    assert session.checkpoint == tmp_path / "fit" / "frozen_parameters.json"
    assert session.seed_run == tmp_path / "anchored"
    assert session.shapes == ("triangle", "star")
    assert session.count == 36
    assert session.iterations == 12
    assert session.blocks == 2
    assert session.updates == 3
    assert session.frames == 5
    assert session.side == 42.5
    assert session.period == 128.0
    assert session.seed == 7
    assert session.camera == "daheng"
    assert session.cam_id == 2
    assert session.slm_number == 3
    assert session.calibration is False
    assert session.adaptation == "full"
    assert session.accept_metric == "cv"
    assert session.objective == "cv"
    assert session.seed_policy == "strict"
    assert session.abba is True
    assert session.adaptive_exposure is True
    assert session.save_dataset is True
    assert session.device == "cpu"


def test_build_session_keeps_the_pipeline_defaults_for_unshown_fields(
    tmp_path,
) -> None:
    session = build_session(TransferFormValues(), out=tmp_path / "session")

    assert session.dataset is None
    assert session.initial_phase == "gs"
    assert session.window_span == 220.0
    assert session.window_step == 2.0
    assert session.phase_lr == 0.01
    assert session.bit_depth == 8
    assert session.capture_grid == 128


def test_calibration_true_becomes_none_so_the_pipeline_decides(tmp_path) -> None:
    session = build_session(TransferFormValues(), out=tmp_path / "session")

    assert session.calibration is None


def test_calibration_none_lets_the_shape_set_decide(tmp_path) -> None:
    three = TransferFormValues(shapes=("square", "rectangle", "triangle"))
    one = TransferFormValues(shapes=("square",))

    assert build_session(three, out=tmp_path / "a").calibration_enabled is True
    assert build_session(one, out=tmp_path / "b").calibration_enabled is False
    off = replace(three, calibration=False)
    assert build_session(off, out=tmp_path / "c").calibration_enabled is False


def test_build_session_plans_the_session_dir_when_out_is_omitted(
    tmp_path, monkeypatch
) -> None:
    sentinel = tmp_path / "planned"
    monkeypatch.setattr(params, "plan_session_dir", lambda: sentinel)

    session = build_session(TransferFormValues())

    assert session.out == sentinel
    assert not sentinel.exists()


def test_build_session_strips_blank_history_paths(tmp_path) -> None:
    blank = TransferFormValues(checkpoint="   ", seed_run="")
    stripped = build_session(blank, out=tmp_path / "a")
    assert stripped.checkpoint is None
    assert stripped.seed_run is None

    spaced = TransferFormValues(checkpoint="  ck  ", seed_run="  seed  ")
    cleaned = build_session(spaced, out=tmp_path / "b")
    assert cleaned.checkpoint == Path("ck")
    assert cleaned.seed_run == Path("seed")


def test_build_session_normalises_a_widget_sequence_to_a_tuple(tmp_path) -> None:
    values = TransferFormValues(shapes=["square", "star"])

    session = build_session(values, out=tmp_path / "session")

    assert session.shapes == ("square", "star")


def test_build_session_output_passes_transfer_session_validation(tmp_path) -> None:
    session = build_session(TransferFormValues(), out=tmp_path / "session")

    session.validate()


# --- 模块边界 ---------------------------------------------------------------


def test_the_module_imports_no_qt_streamlit_torch_or_page_modules() -> None:
    source = Path(params.__file__).read_text(encoding="utf-8")
    module_names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            module_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_names.add(node.module)

    for prefix in ("PySide6", "streamlit", "torch"):
        assert not any(
            name == prefix or name.startswith(f"{prefix}.") for name in module_names
        )
    assert module_names.isdisjoint(
        {
            "ao_shaping.gui.slm.shaping_controls",
            "ao_shaping.gui.slm.transfer_candidates",
        }
    )


def test_the_module_is_utf8_without_a_bom() -> None:
    raw = Path(params.__file__).read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")


# --- 采样窗半跨的派生 -------------------------------------------------------


def test_small_targets_keep_the_pipeline_default_half_span() -> None:
    """小目标行为必须逐位不变: 220 px 半跨是管线默认值。"""
    assert params.effective_window_span(80) == 220.0
    assert params.effective_window_span(424) == 220.0
    # 424 = 2 * (220 - 8): 恰好放得下, 不该被推到 220 以上。
    assert params.effective_window_span(424, 0.0) == 220.0


def test_large_targets_derive_a_window_that_holds_the_target() -> None:
    """派生口径 = max(220, side/2 + 留白): 采样窗要放下整个目标 + 两侧留白。"""
    assert params.effective_window_span(900) >= 458.0
    assert params.effective_window_span(900) == 900 / 2 + 8.0


def test_an_explicit_window_span_is_returned_verbatim() -> None:
    """命令行验证过的显式半跨 (930) 不能被派生逻辑改写。"""
    assert params.effective_window_span(900, 930.0) == 930.0
    assert params.effective_window_span(80, 300.0) == 300.0


def test_side_900_passes_validation_through_the_derived_half_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 只测「半跨怎么传给 preflight」, 不把 torch 安装状态混进来。
    monkeypatch.setattr(params, "_torch_missing", lambda: None)
    errors, _ = validate_form(_values(side=900.0))

    assert errors == []
    problems = preflight.check_parameters(
        side=900.0,
        period=64.0,
        count=24,
        iterations=60,
        blocks=4,
        updates=10,
        frames=3,
        shapes=("square",),
        half_span_px=params.effective_window_span(900.0),
    )
    assert [problem.code for problem in problems] == []


def test_an_explicit_span_below_the_derived_one_still_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式半跨优先: 写了 220 就必须按 220 判, 900 px 目标要被拦住。"""
    monkeypatch.setattr(params, "_torch_missing", lambda: None)
    errors, _ = validate_form(_values(side=900.0, window_span=220.0))

    assert any(problem.code == "side_exceeds_window" for problem in _problems(900.0, 220.0))
    assert any("采样窗" in error for error in errors)


def _problems(side: float, half_span: float) -> list[preflight.Problem]:
    return preflight.check_parameters(
        side=side,
        period=64.0,
        count=24,
        iterations=60,
        blocks=4,
        updates=10,
        frames=3,
        shapes=("square",),
        half_span_px=half_span,
    )


def test_build_session_carries_the_effective_window_span(tmp_path) -> None:
    values = TransferFormValues(side=900.0)
    session = build_session(values, out=tmp_path / "session")

    assert session.window_span == params.effective_window_span(values.side)


def test_build_session_keeps_an_explicit_window_span(tmp_path) -> None:
    values = TransferFormValues(side=900.0, window_span=930.0)
    session = build_session(values, out=tmp_path / "session")

    assert session.window_span == 930.0
