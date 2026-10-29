"""campaign 参数派生的测试（免设备、无 streamlit / 无 torch）。

这里的每条断言都钉住一个"GUI 只给形状 + 尺寸"必须成立的性质：
完整 kwargs、窗口档位、来源标签、覆盖留下被替换值、默认值与 runner 的 click 默认值
一致、矩阵合计 16 段 / 112 块、时间估算落在实测基线附近。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer import campaign
from ao_shaping.tools.slm.transfer.campaign import (
    SIZE_WINDOWS,
    CampaignSettings,
    FieldSource,
    MatrixRow,
    Setting,
    campaign_matrix,
    derive_settings,
    estimate_disk_gb,
    estimate_seconds,
    matrix_totals,
    window_for_side,
)

#: 报告 §3.2 的四个目标形状（``TRANSFER_SHAPE_CHOICES`` 里的 ``rectangle`` 不在
#: 那次 campaign 里，所以这里显式列出而不是用整个词表）。
CAMPAIGN_SHAPES = ("square", "triangle", "star", "circle")

#: 期望的 runner 关键字参数（``--out`` / ``--execute`` 之外的全部）。
EXPECTED_KWARGS = {
    "shapes", "side", "window_span", "period", "zero_radius", "capture_grid",
    "fit_iterations", "control_grid", "fft_size", "frames", "replay_exposures_auto",
    "slm_number", "camera", "initial_phase", "gs_seed",
}


def test_shape_and_size_are_the_only_inputs_needed() -> None:
    """只给 (形状, 尺寸) 就有完整 kwargs，且没有红停项。"""
    settings = derive_settings(("square",), 1200.0)

    assert isinstance(settings, CampaignSettings)
    assert settings.shapes == ("square",)
    assert settings.side == 1200.0
    kwargs = settings.kwargs()
    assert set(kwargs) == EXPECTED_KWARGS
    assert kwargs["shapes"] == ("square",)
    assert kwargs["side"] == 1200.0
    assert kwargs["window_span"] == 900.0  # 尺寸派生，不是 CLI 默认 820
    assert kwargs["initial_phase"] == "auto"
    assert kwargs["gs_seed"] is False  # 无 GS 方案
    assert kwargs["replay_exposures_auto"] is True
    assert settings.blocking() == ()
    assert all(isinstance(setting, Setting) for setting in settings.settings.values())


def test_zero_xy_and_exposure_are_not_runner_flags() -> None:
    """实测输入不进 kwargs：runner 自己测零点、曝光由实验系统探针定。"""
    settings = derive_settings(("square",), 1200.0, zero_xy=(1294.8, 982.4), exposure_ms=135.196)

    assert "zero_xy" not in settings.kwargs()
    assert "exposure_ms" not in settings.kwargs()
    assert settings.settings["zero_xy"].value == (1294.8, 982.4)
    assert settings.settings["exposure_ms"].value == 135.196


def test_window_bands_and_nearest_band() -> None:
    """三档尺寸命中报告 §5.2 的窗口半跨；档外尺寸就近取并说明方向。"""
    assert SIZE_WINDOWS == {200.0: 420.0, 600.0: 620.0, 1200.0: 900.0}

    for side, span in ((200.0, 420.0), (600.0, 620.0), (1200.0, 900.0)):
        got, note = window_for_side(side)
        assert got == span
        assert "§5.2 尺寸窗口 420/620/900" in note
        assert "命中" in note

    span, note = window_for_side(800.0)
    assert span == 620.0  # 最近档（600），不是最小可容纳档
    assert "不在 200/600/1200 档位上" in note
    assert "偏小" in note


def test_provenance_tags() -> None:
    """每个参数都带来源：实测 / 派生 / 默认 / 手工四类齐全。"""
    measured = derive_settings(("square",), 1200.0, zero_xy=(1.0, 2.0), exposure_ms=50.0)
    assert measured.settings["zero_xy"].source is FieldSource.MEASURED
    assert measured.settings["exposure_ms"].source is FieldSource.MEASURED

    derived = derive_settings(("square",), 1200.0)
    assert derived.settings["window_span"].source is FieldSource.DERIVED
    assert derived.settings["replay_exposures_auto"].source is FieldSource.DERIVED
    assert derived.settings["gs_seed"].source is FieldSource.DERIVED
    # 派生说明必须点名它取代的 CLI 默认值与被替换掉的口径
    assert "820" in derived.settings["window_span"].note

    for name in ("period", "capture_grid", "fit_iterations", "control_grid", "fft_size",
                 "frames", "slm_number", "camera", "zero_radius", "initial_phase"):
        assert derived.settings[name].source is FieldSource.DEFAULT, name

    manual = derive_settings(("square",), 1200.0, overrides={"period": 64.0})
    assert manual.settings["period"].source is FieldSource.MANUAL


def test_zero_xy_missing_is_a_warning_not_a_blocker() -> None:
    """零级未测只给警告：几何检查退回标称位置，但运行前要补光路探针。"""
    settings = derive_settings(("square",), 1200.0)

    assert "zero_xy" not in settings.settings
    assert any("零级位置未测" in warning for warning in settings.warnings)
    assert all("探针" in warning for warning in settings.warnings)
    assert settings.blocking() == ()


def test_exposure_default_is_auto_with_note() -> None:
    """未给曝光时是 ``auto``，说明要指向自动重放曝光/曝光探针。"""
    setting = derive_settings(("square",), 1200.0).settings["exposure_ms"]

    assert setting.value == "auto"
    assert setting.source is FieldSource.DEFAULT
    assert "--replay-exposures-auto" in setting.note


def test_overrides_become_manual_and_record_the_replaced_value() -> None:
    """覆盖一律 MANUAL，并在说明里记下被替换的「来源 + 值」—— 这是可审计性的全部。"""
    settings = derive_settings(("square",), 1200.0, overrides={"period": 64.0})

    setting = settings.settings["period"]
    assert setting.value == 64.0
    assert setting.source is FieldSource.MANUAL
    assert "128.0" in setting.note
    assert "default" in setting.note
    assert settings.kwargs()["period"] == 64.0

    span_override = derive_settings(("square",), 1200.0, overrides={"window_span": 1000.0})
    note = span_override.settings["window_span"].note
    assert "900" in note and "手工覆盖" in note
    assert span_override.kwargs()["window_span"] == 1000.0


def test_explicit_keyword_arguments_are_manual() -> None:
    """函数参数显式给值 = 人工改动，同样标 MANUAL 并记下默认值。"""
    settings = derive_settings(("square",), 1200.0, frames=5, capture_grid=512,
                               initial_phase="backprojection")

    assert settings.settings["frames"].source is FieldSource.MANUAL
    assert "3" in settings.settings["frames"].note
    assert settings.settings["capture_grid"].source is FieldSource.MANUAL
    assert settings.settings["initial_phase"].source is FieldSource.MANUAL
    assert settings.kwargs()["frames"] == 5


def test_explicit_replay_ladder_disables_auto() -> None:
    """显式重放 ladder 会关掉自动模式（并留痕），因为二者不可同时生效。"""
    settings = derive_settings(("square",), 1200.0, replay_exposures=(20.0, 50.0, 90.0))

    assert settings.settings["replay_exposures_auto"].value is False
    assert settings.settings["replay_exposures_auto"].source is FieldSource.MANUAL
    assert settings.kwargs()["replay_exposures_auto"] is False
    assert settings.kwargs()["replay_exposures"] == (20.0, 50.0, 90.0)


def test_unknown_override_is_warned_not_swallowed() -> None:
    """拼错的参数名要报警告而不是被静默丢弃 —— runner 会以未知参数报错。"""
    settings = derive_settings(("square",), 1200.0, overrides={"windowspann": 1.0})

    assert any("windowspann" in warning for warning in settings.warnings)
    assert settings.kwargs()["windowspann"] == 1.0


@pytest.mark.parametrize(
    "shapes,side",
    [
        ((), 1200.0),  # 空
        (("hexagon",), 1200.0),  # 未知形状
        (("square", "square"), 1200.0),  # 重复：同目录写两遍
        ("square", 1200.0),  # 误把单个字符串当序列
        (("square",), 0.0),  # 边长非正
        (("square",), -5.0),
        (("square",), float("nan")),
    ],
)
def test_invalid_inputs_raise_chinese_errors(shapes, side) -> None:
    """非法输入必须抛 ValueError（GUI 直接当红停原因显示），消息是中文。"""
    with pytest.raises(ValueError) as excinfo:
        derive_settings(shapes, side)

    message = str(excinfo.value)
    assert any("\u4e00" <= ch <= "\u9fff" for ch in message), message


def test_blocking_reports_a_target_that_does_not_fit_the_window() -> None:
    """目标放不进采样窗是红停项（窗口按半跨硬裁，不会自己报错）。"""
    settings = derive_settings(("square",), 1200.0, overrides={"window_span": 300.0})

    problems = settings.blocking()
    assert len(problems) == 1
    assert "600" in problems[0] or "608" in problems[0]


def test_default_table_matches_runner_click_defaults() -> None:
    """``RUNNER_CLI_DEFAULTS`` 与 runner 的 click 默认值逐项相同（没有第二份真相）。"""
    from ao_shaping.runners.slm.transfer_shapes_runner import run as command

    declared = {param.name: param.default for param in command.params}
    for name, expected in campaign.RUNNER_CLI_DEFAULTS.items():
        assert name in declared, f"{name} 不是 runner 的 click 参数"
        assert declared[name] == expected, f"{name}: 表里 {expected!r} ≠ runner {declared[name]!r}"


def test_matrix_is_sixteen_segments_and_112_blocks() -> None:
    """4 形状 × 4 阶段 = 16 段 / 112 块，逐行块数与报告 §3.2 一致。"""
    rows = campaign_matrix(CAMPAIGN_SHAPES)

    assert len(rows) == 16
    assert all(isinstance(row, MatrixRow) for row in rows)
    assert matrix_totals(rows) == {"segments": 16, "blocks": 112, "updates": 280, "shapes": 4}

    one_shape = campaign_matrix(("square",))
    assert [(row.stage, row.blocks, row.bit_depth) for row in one_shape] == [
        ("transport_cv", 4, 8),
        ("mono12_cv", 8, 12),
        ("mono12_continue", 12, 12),
        ("mono12_dense", 4, 12),
    ]
    assert sum(row.updates for row in one_shape) == 70


def test_estimate_seconds_brackets_the_measured_baseline() -> None:
    """期望值 = 块数 × 24 s（= 规划脚本 0.4 min/块），且以 ±40 % 夹住实测 2675 s。"""
    rows = campaign_matrix(CAMPAIGN_SHAPES)

    optimistic, expected, pessimistic = estimate_seconds(rows)
    assert campaign.MINUTES_PER_BLOCK == 0.4
    assert campaign.SECONDS_PER_BLOCK == 24.0
    assert expected == pytest.approx(112 * 24.0)
    assert optimistic < expected < pessimistic
    # 实测四形状整形合计 2675.0 s（wallclock 报告 §8.3），基线偏差 < 1 %
    assert abs(expected - 2675.0) / 2675.0 < 0.01
    assert optimistic == pytest.approx(expected * 0.6)
    assert pessimistic == pytest.approx(expected * 1.4)


def test_estimate_disk_matches_the_measured_campaign_bytes() -> None:
    """落盘估算按阶段查表，四形状合计 14.72 GB ≈ 实测 14.7 GB。"""
    rows = campaign_matrix(CAMPAIGN_SHAPES)

    assert estimate_disk_gb(rows) == pytest.approx(14.72)
    assert estimate_disk_gb(campaign_matrix(("square",))) == pytest.approx(3.68)
    # 大 campaign 要有磁盘提示
    assert any("落盘" in warning for warning in derive_settings(CAMPAIGN_SHAPES, 1200.0).warnings)


def _load_plan_script():
    """把规划脚本当模块加载（它只 import protocol/campaign，不开设备）。"""
    path = Path(__file__).resolve().parents[5] / "scripts" / "slm" / "plan_multishape_campaign.py"
    spec = importlib.util.spec_from_file_location("_plan_multishape_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plan_script_uses_the_single_source() -> None:
    """规划脚本的窗口表与单块基线必须来自本模块（搬常量时最容易留第二份副本）。"""
    script = _load_plan_script()

    assert dict(script.SIZE_WINDOWS) == dict(SIZE_WINDOWS)
    assert script.DEFAULT_MINUTES_PER_BLOCK == campaign.MINUTES_PER_BLOCK
