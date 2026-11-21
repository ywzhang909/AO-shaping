"""Offline tests for the SLM phase-change-speed tool (no hardware/SDK)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.drivers.ccd.common import CAMERA_TYPES, register_camera
import ao_shaping.tools.slm.phase_speed as phase_speed
from ao_shaping.tools.slm.phase_speed import (
    PhaseStepResult,
    analyze_step,
    build_pattern,
    far_field_metric,
)
from ao_shaping.tools.slm.phase_speed.__main__ import main
from ao_shaping.tools.slm.phase_speed.measure import run_speed_measurement


# ── 纯算法: analyze_step ────────────────────────────────────────────────────


def test_analyze_step_response_and_settle_index_to_ms() -> None:
    # 0,0,0, 1,1,1,1,1,1 — 在 idx=3 达到 90%, idx=3 起全在 ±10% 带内.
    trace = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = analyze_step(trace, fps=100.0, baseline_window=3, settle_window=3)

    assert result.step_metric_before == pytest.approx(0.0)
    assert result.step_metric_after == pytest.approx(1.0)
    assert result.t_response_ms == pytest.approx(30.0)  # 3 * 1000/100
    assert result.t_settle_ms == pytest.approx(30.0)
    assert result.frames_to_settle == 3
    assert any("SLM+CCD" in note for note in result.notes)
    assert any("帧率" in note for note in result.notes)


def test_analyze_step_settle_band_requires_all_remaining_samples() -> None:
    # idx=3 短暂到位, idx=4 又跳出 10% 带, idx=5 起稳定 ⇒ settle 应在 idx=5.
    trace = [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    result = analyze_step(trace, fps=50.0, baseline_window=3, settle_window=3)

    assert result.t_response_ms == pytest.approx(60.0)  # idx=3 * 20ms
    assert result.frames_to_settle == 5
    assert result.t_settle_ms == pytest.approx(100.0)


def test_analyze_step_never_settles_returns_none() -> None:
    # 最后一个样本还在带外 ⇒ 无稳定点.
    trace = [0.0, 0.0, 0.0, 1.0, 1.0, 0.3]
    result = analyze_step(trace, fps=10.0, baseline_window=3, settle_window=3)

    assert result.t_response_ms == pytest.approx(300.0)
    assert result.t_settle_ms is None
    assert result.frames_to_settle is None
    assert any("未进入稳定带" in note for note in result.notes)


def test_analyze_step_no_transition_is_graceful() -> None:
    trace = [5.0, 5.0, 5.0, 5.0, 5.0]
    result = analyze_step(trace, fps=30.0)

    assert result.t_response_ms is None
    assert result.t_settle_ms is None
    assert result.frames_to_settle is None
    assert result.step_metric_before == pytest.approx(5.0)
    assert result.step_metric_after == pytest.approx(5.0)
    assert any("阶跃" in note for note in result.notes)


def test_analyze_step_descending_step() -> None:
    trace = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    result = analyze_step(trace, fps=20.0, baseline_window=3, settle_window=3)

    assert result.t_response_ms == pytest.approx(150.0)  # idx=3
    assert result.frames_to_settle == 3


# ── 纯算法: far_field_metric ───────────────────────────────────────────────


def test_far_field_metric_argmax_off_center() -> None:
    frame = np.full((16, 16), 2.0)
    frame[3, 11] = 200.0  # 0 级不在帧中心
    metric = far_field_metric(frame)

    assert metric == pytest.approx(198.0)  # 200 - median(2)


def test_far_field_metric_explicit_center_and_no_background() -> None:
    frame = np.full((8, 8), 5.0)
    frame[6, 1] = 42.0

    assert far_field_metric(frame, zero_center=(6, 1)) == pytest.approx(37.0)
    assert far_field_metric(frame, zero_center=(6, 1), subtract_background=False) == 42.0


def test_far_field_metric_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        far_field_metric(np.zeros((4,)))


# ── 图案生成 ───────────────────────────────────────────────────────────────


def test_build_pattern_flat_is_raw_uint16() -> None:
    pattern = build_pattern("flat", (6, 8), 993, 16.0)
    assert pattern.dtype == np.uint16
    assert pattern.shape == (6, 8)
    assert int(pattern.min()) == int(pattern.max()) == 993


def test_build_pattern_blaze_has_period_and_is_uint16() -> None:
    pattern = build_pattern("blaze", (6, 8), 512, 4.0)
    assert pattern.dtype == np.uint16
    assert pattern.shape == (6, 8)
    # 每行相同, 且一个周期内 4 个不同灰度.
    assert np.array_equal(pattern[0], pattern[1])
    assert len(np.unique(pattern[0])) == 4


# ── 硬件后端: FakeSLM + FakeCamera ────────────────────────────────────────


class FakeSLM:
    """鸭子类型 SLM: 记录写入图案与槽位, 不接硬件."""

    def __init__(self) -> None:
        self.Panel_Res = (8, 12)  # (w, h)
        self.displayed_slot = 3
        self.writes: list[tuple[np.ndarray, int, float]] = []

    def __enter__(self) -> "FakeSLM":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get_displayed_memory_number(self) -> int:
        return self.displayed_slot

    def display_data(
        self,
        pattern: np.ndarray,
        memory_number: int,
        wait_time_s: float = 0.0,
    ) -> None:
        self.displayed_slot = memory_number
        self.writes.append((pattern.copy(), memory_number, wait_time_s))


class FakeCamera:
    """鸭子类型相机: 预置帧序列, 逐帧返回, 记录采集参数."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0
        self.calls: list[tuple[int, bool]] = []

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.calls.append((n_sample, skip_first))
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def get_exposure_time_ms(self) -> float:
        return 5.0

    def effective_full_scale(self, img: np.ndarray) -> float:
        return 255.0


def _make_frames(n_baseline: int, n_steps: int) -> list[np.ndarray]:
    """基线亮度 ~2, 阶跃后亮度 ~102, 0 级固定在 (5, 9)."""
    frames: list[np.ndarray] = []
    for i in range(n_baseline + n_steps):
        peak = 2.0 if i < n_baseline else 102.0
        frame = np.full((12, 16), 2.0)
        frame[5, 9] = peak
        frames.append(frame)
    return frames


def _make_config(tmp_path: Path, **overrides: Any) -> Any:
    from ao_shaping.tools.slm.phase_speed.__main__ import SpeedConfig

    values: dict[str, Any] = dict(
        out_dir=tmp_path,
        slm_number=1,
        wavelength_nm=1064,
        cam_type="daheng",
        cam_id=0,
        exposure_ms=5.0,
        gray_a=0,
        gray_b=993,
        n_frames=8,
        settle_s=0.0,
        pattern="flat",
        period_px=16.0,
        save_frames=True,
        tag="test",
        baseline_window=3,
        settle_window=3,
        settle_fraction=0.1,
        response_fraction=0.9,
    )
    values.update(overrides)
    return SpeedConfig(**values)


def _run_with_fakes(tmp_path: Path, slm: FakeSLM, cam: FakeCamera, **overrides: Any) -> Any:
    config = _make_config(tmp_path, **overrides)
    return run_speed_measurement(
        config,
        slm_factory=lambda _cfg: slm,
        camera_factory=lambda _cfg: cam,
    )


def test_run_speed_measurement_rotates_slots_and_raw_flat(tmp_path: Path) -> None:
    slm = FakeSLM()
    cam = FakeCamera(_make_frames(n_baseline=3, n_steps=8))
    outcome = _run_with_fakes(tmp_path, slm, cam)

    # 写入两次: A 与 B, 用不同内存槽 (禁止连续同槽).
    assert len(slm.writes) == 2
    slot_a, slot_b = slm.writes[0][1], slm.writes[1][1]
    assert slot_a != slot_b
    # 平场是 raw uint16 np.full, 不走弧度→灰度转换.
    for (pattern, _slot, _wait), gray in zip(slm.writes, (0, 993)):
        assert pattern.dtype == np.uint16
        assert int(pattern.min()) == int(pattern.max()) == gray
    # 采集契约: n_sample=1, skip_first=False; 基线 3 + 阶跃 8 帧.
    assert len(cam.calls) == 11
    assert all(call == (1, False) for call in cam.calls)
    # 结果: 阶跃确实被检出, 时间非负.
    assert outcome.result.t_response_ms is not None
    assert outcome.result.t_response_ms >= 0.0
    assert outcome.result.t_settle_ms is not None
    assert outcome.result.frames_to_settle is not None


def test_run_speed_measurement_timing_fps_and_artifacts(tmp_path: Path) -> None:
    slm = FakeSLM()
    cam = FakeCamera(_make_frames(n_baseline=3, n_steps=8))
    outcome = _run_with_fakes(tmp_path, slm, cam, save_frames=True)

    payload = json.loads((tmp_path / "test_result.json").read_text(encoding="utf-8"))
    assert payload["fps_measured"] > 0.0
    assert payload["fps_requested"] == pytest.approx(200.0)  # 1000/5ms
    assert len(payload["frame_intervals_ms"]) == 7  # 8 阶跃帧 → 7 间隔
    assert any("SLM+CCD" in c for c in payload["caveats"])
    assert payload["device"]["slot_a"] != payload["device"]["slot_b"]
    assert (tmp_path / "test_frames.npz").exists()
    assert outcome.fps_measured > 0.0


class RegistryCamera(FakeCamera):
    """注册表测试用相机: 模块级定义, 便于 create_camera 按 module:qualname 解析."""

    def __init__(self, cam_id=0, exposure_time_ms=5.0, **kwargs: Any) -> None:
        super().__init__(_make_frames(n_baseline=3, n_steps=8))
        self.constructed_cam_id = cam_id
        self.constructed_exposure = exposure_time_ms


def test_run_speed_measurement_uses_camera_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """默认相机工厂经 create_camera 注册表构造 (相机类型注册/调用可验证)."""
    # 注册表是全局的, 用唯一名字注册并在测试后清理, 不污染其他测试.
    name = "_phase_speed_test_cam"
    register_camera(name, RegistryCamera, accepted_kwargs={"cam_id", "exposure_time_ms"})

    created: dict[str, Any] = {}
    original = phase_speed.measure.create_camera

    def tracking_create(cam_type: str, *args: Any, **kwargs: Any) -> Any:
        obj = original(cam_type, *args, **kwargs)
        created["type"] = cam_type
        created["obj"] = obj
        return obj

    monkeypatch.setattr(phase_speed.measure, "create_camera", tracking_create)
    config = _make_config(tmp_path, cam_type=name)

    slm = FakeSLM()
    result = run_speed_measurement(
        config,
        slm_factory=lambda _cfg: slm,
    )
    assert created["type"] == name
    assert created["obj"].constructed_cam_id == 0
    assert created["obj"].constructed_exposure == 5.0
    assert result.result.t_response_ms is not None

    CAMERA_TYPES.pop(name, None)


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_help_shows_flags() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    for flag in (
        "--out-dir",
        "--slm-number",
        "--wavelength-nm",
        "--cam-type",
        "--cam-id",
        "--exposure-ms",
        "--gray-a",
        "--gray-b",
        "--n-frames",
        "--settle-s",
        "--save-frames",
        "--pattern",
        "--period-px",
    ):
        assert flag in result.output
    assert "SLM+CCD" in result.output
    assert "帧率" in result.output
    assert "blaze" in result.output
