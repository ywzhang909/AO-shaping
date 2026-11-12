"""采集网格自动升档 / FFT 可达窗：纯离线判定（不开设备、不占相机）。

同一条可达公式服务四个读者：硬保护、preflight 预告、桌面校验文案、会话升档。
这里把它们钉在一起，避免任何一处另立数字。
"""
from __future__ import annotations

import pytest

from ao_shaping.algorithm.signal_processing.multishape_objective import (
    capture_grid_ladder,
    fft_max_side_px,
    required_capture_grid,
)
from ao_shaping.gui.slm.transfer_desktop.params import (
    default_capture_grid,
    describe_capture_grid,
)
from ao_shaping.tools.slm.transfer.preflight import check_parameters
from ao_shaping.tools.slm.transfer.workflow import TransferSession, decide_capture_grid
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE


def _params(**overrides):
    values = dict(
        side=1800.0, period=120.0, count=6, iterations=10, blocks=1, updates=1,
        frames=3, shapes=("triangle",),
    )
    values.update(overrides)
    return values


def test_ladder_matches_session_tiling_rule():
    """梯子里的每一档都必须过 TransferSession 的 tiling 校验，反之亦然。"""
    for grid in capture_grid_ladder():
        TransferSession(out="unused", capture_grid=grid).validate()
    for grid in (16, 100, 128, 640, 1920):
        session = TransferSession(out="unused", capture_grid=grid)
        try:
            session.validate()
        except ValueError:
            assert grid not in capture_grid_ladder()
        else:
            assert grid in capture_grid_ladder()


def test_required_grid_1800_triangle_is_640_not_480():
    assert required_capture_grid(1800.0, TILT_SHIFT_SCALE, shapes=("triangle",), period=120.0) == 640


def test_required_grid_rectangle_uses_long_axis():
    assert required_capture_grid(1800.0, TILT_SHIFT_SCALE, shapes=("rectangle",), period=120.0) == 1920


def test_required_grid_unreachable_returns_none():
    assert required_capture_grid(20000.0, TILT_SHIFT_SCALE, shapes=("square",)) is None


def test_decide_raises_only_when_opted_in():
    session = TransferSession(out="unused", side=1800.0, shapes=("triangle",))
    assert decide_capture_grid(session).chosen == 128  # 默认关：原样
    object.__setattr__(session, "tilt", True)
    object.__setattr__(session, "auto_capture_grid", True)
    decision = decide_capture_grid(session)
    assert decision.raised and decision.chosen == 640 and decision.requested == 128


def test_decide_never_lowers_pinned_grid():
    session = TransferSession(
        out="unused", side=1800.0, shapes=("triangle",), capture_grid=960,
        auto_capture_grid=True,
    )
    decision = decide_capture_grid(session)
    assert decision.chosen == 960 and not decision.raised


def test_decide_leaves_sufficient_grid_untouched():
    session = TransferSession(out="unused", side=200.0, shapes=("square",), auto_capture_grid=True)
    decision = decide_capture_grid(session)
    assert decision.chosen == 128 and not decision.raised


def test_preflight_grid_none_adds_no_reach_problem():
    problems = check_parameters(**_params())
    assert not any(p.code == "side_exceeds_fft_reach" for p in problems)


def test_preflight_warns_with_numbers_when_grid_too_narrow():
    problems = check_parameters(**_params(), grid_size=128)
    reach = [p for p in problems if p.code == "side_exceeds_fft_reach"]
    assert len(reach) == 1 and reach[0].severity == "warning"
    assert "1800" in reach[0].message and "128" in reach[0].message
    assert "640" in reach[0].hint


def test_preflight_quiet_when_grid_reaches():
    problems = check_parameters(**_params(), grid_size=640)
    assert not any(p.code == "side_exceeds_fft_reach" for p in problems)


def test_preflight_and_guard_agree_on_boundary():
    """同一公式：preflight 报越界的点，硬保护的最大边长也确实更小。"""
    limit = fft_max_side_px(128, TILT_SHIFT_SCALE, offset_px=TILT_SHIFT_SCALE / 120.0)
    assert limit < 1800.0
    assert fft_max_side_px(640, TILT_SHIFT_SCALE, offset_px=TILT_SHIFT_SCALE / 120.0) >= 1800.0


def test_desktop_reports_raise_as_neutral_info():
    info, error = describe_capture_grid(
        side=1800.0, shapes=("triangle",), aspect_ratio=1.0, tilt=True, period=120.0,
    )
    assert error is None
    assert info is not None and "128 → 640" in info and "1800" in info


def test_desktop_silent_when_default_grid_already_reaches():
    assert describe_capture_grid(
        side=200.0, shapes=("square",), aspect_ratio=1.0, tilt=True, period=120.0,
    ) == (None, None)


def test_desktop_blocks_when_no_grid_reaches():
    info, error = describe_capture_grid(
        side=20000.0, shapes=("square",), aspect_ratio=1.0, tilt=False, period=120.0,
    )
    assert info is None
    assert error is not None and "20000" in error


def test_desktop_baseline_is_session_default():
    assert default_capture_grid() == TransferSession(out="unused").capture_grid


def test_describe_never_raises_on_bad_input():
    assert describe_capture_grid(
        side=float("nan"), shapes=(), aspect_ratio=1.0, tilt=False, period=0.0,
    ) == (None, None)


def test_reuse_skips_when_latest_grid_cannot_reach(tmp_path, monkeypatch):
    """光路一致但 latest 网格太窄 → 跳过复用（不抛），说明处能看到原因。"""
    import json

    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    monkeypatch.setenv("SLM_MODELS_DIR", str(tmp_path / "models"))
    latest = tmp_path / "models" / "SLM-A" / "latest"
    latest.mkdir(parents=True)
    (latest / "frozen_parameters.json").write_text(
        json.dumps({"origin": [10.0, 20.0], "grid_size": 128}), encoding="utf-8"
    )
    (latest / probe_mod.FLAT_FROZEN_SIDECAR).write_text(
        json.dumps({
            "exposure_range_ms": [0.02, 25.0], "camera_format": "MONO8",
            "bit_depth": 8, "slm_serial": "SLM-A",
        }),
        encoding="utf-8",
    )
    config = TransferSession(
        tmp_path / "run", reuse_latest=True, side=1800.0, shapes=("triangle",),
    )
    config.out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        probe_mod, "measure_probe",
        lambda **kwargs: probe_mod.OpticalProbe(
            zero_xy=(10.0, 20.0), exposure_range_ms=(0.02, 25.0),
            camera_format="MONO8", bit_depth=8, slm_serial="SLM-A",
        ),
    )
    assert workflow_mod._attempt_optical_reuse(config, config.out) is None
    record = json.loads((config.out / "optical_probe.json").read_text(encoding="utf-8"))
    assert record["skipped"] is True
    assert record["reason"] == probe_mod.REUSE_SKIP_GRID
    assert probe_mod.REUSE_SKIP_GRID in probe_mod.format_probe_skip_line(record["reason"])


def test_reuse_still_applies_when_latest_grid_reaches(tmp_path, monkeypatch):
    """latest 网格够得着 → 复用计划照旧返回（新判据不误伤）。"""
    import json

    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    monkeypatch.setenv("SLM_MODELS_DIR", str(tmp_path / "models"))
    latest = tmp_path / "models" / "SLM-A" / "latest"
    latest.mkdir(parents=True)
    (latest / "frozen_parameters.json").write_text(
        json.dumps({"origin": [10.0, 20.0], "grid_size": 640}), encoding="utf-8"
    )
    (latest / probe_mod.FLAT_FROZEN_SIDECAR).write_text(
        json.dumps({
            "exposure_range_ms": [0.02, 25.0], "camera_format": "MONO8",
            "bit_depth": 8, "slm_serial": "SLM-A",
        }),
        encoding="utf-8",
    )
    config = TransferSession(
        tmp_path / "run", reuse_latest=True, side=1800.0, shapes=("triangle",),
    )
    config.out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        probe_mod, "measure_probe",
        lambda **kwargs: probe_mod.OpticalProbe(
            zero_xy=(10.0, 20.0), exposure_range_ms=(0.02, 25.0),
            camera_format="MONO8", bit_depth=8, slm_serial="SLM-A",
        ),
    )
    assert workflow_mod._attempt_optical_reuse(config, config.out) is not None


def test_reuse_plan_zero_is_current_measured_not_frozen_origin(tmp_path, monkeypatch):
    """复用计划的零点取**本次实测**, 不用 latest 里的旧 origin。

    旧 origin 与本场零点可差到 DEFAULT_MAX_ZERO_DRIFT_PX (40 px) 仍判同一条光路,
    而下游裁剪/定位窗口都按零点锚定 —— 用旧值会把窗口带偏。这里让两者差 2 px
    (在 same_path 的零点容忍 5 px 之内), 计划必须带本次实测值。
    """
    import json

    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    monkeypatch.setenv("SLM_MODELS_DIR", str(tmp_path / "models"))
    latest = tmp_path / "models" / "SLM-A" / "latest"
    latest.mkdir(parents=True)
    # 冻结参数里的旧零点 (12, 18) 与本次实测 (10, 20) 差 2 px。
    (latest / "frozen_parameters.json").write_text(
        json.dumps({"origin": [12.0, 18.0], "grid_size": 640}), encoding="utf-8"
    )
    (latest / probe_mod.FLAT_FROZEN_SIDECAR).write_text(
        json.dumps({
            "exposure_range_ms": [0.02, 25.0], "camera_format": "MONO8",
            "bit_depth": 8, "slm_serial": "SLM-A",
        }),
        encoding="utf-8",
    )
    config = TransferSession(
        tmp_path / "run", reuse_latest=True, side=1800.0, shapes=("triangle",),
    )
    config.out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        probe_mod, "measure_probe",
        lambda **kwargs: probe_mod.OpticalProbe(
            zero_xy=(10.0, 20.0), exposure_range_ms=(0.02, 25.0),
            camera_format="MONO8", bit_depth=8, slm_serial="SLM-A",
        ),
    )
    plan = workflow_mod._attempt_optical_reuse(config, config.out)
    assert plan is not None
    assert plan.zero_xy == [10.0, 20.0]  # 本次实测, 不是旧 origin [12.0, 18.0]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
