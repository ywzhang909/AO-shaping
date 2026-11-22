"""实测闭环整形的离线契约（不打开 SLM / CCD / 激光）。

钉住的都是"错了会静默给出好看结论"的地方：

* **门禁先于判断**：饱和、低信号、曝光不符、质心跳变的帧不得进入指标比较，更不得
  因为"分数 0 看起来最好"而被接受；
* **分数只比形状**：实测与目标各自按均值归一化，换曝光不整体平移轮次分数；
  窗口内能量占比单独报告，不混进接受判据；
* **回滚是真的回滚**：候选被拒或门禁失败后，设备上留下的是最后确认过的相位；
* **训练样本只来自被接受的轮次**，且验证族按事先声明的规则划分；
* **没有开场标定就不适配**：`insufficient`，不是降低门槛。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.gui.slm.closed_loop_shaping import (
    AdaptationPolicy,
    CapturedFrame,
    ClosedLoopConfig,
    ClosedLoopRunner,
    GateThresholds,
    LoopState,
    OnlineTransferAdapter,
    analysis_grid,
    check_frame,
    compare_to_target,
    describe_gate,
    promotion_decision,
    refine_phase,
    sample_on_grid,
    score_from_metrics,
    target_field,
    window_fraction,
)

# --------------------------------------------------------------------------- #
# 测试替身：传感器 40x40，分析窗口 5x5 位于 (8..12, 8..12)
# --------------------------------------------------------------------------- #

_SPAN = 4
_WINDOW = 5
_SENSOR = 40
_CENTRE = (10.0, 10.0)


def _xy() -> np.ndarray:
    return analysis_grid(_CENTRE, _SPAN)


def _target() -> np.ndarray:
    # 采样步长 2 px：±2 px 恰好覆盖 3x3 个采样点。
    return target_field(_xy(), half_width=2.0, half_height=2.0)


def _embed(pattern: np.ndarray) -> np.ndarray:
    """把窗口图案散到取景框的**采样点**上，其余像素为 0。

    闭环只在这批采样点上取强度（``sample_on_grid``），所以画满整个 9x9 物理像素
    会造出"窗口内有一半能量落在非采样点"的假泄漏。
    """
    frame = np.zeros((_SENSOR, _SENSOR), dtype=np.float64)
    grid = _xy()
    rows = grid[..., 1].astype(int)
    cols = grid[..., 0].astype(int)
    frame[rows, cols] = pattern
    return frame


def _blob(offset: float = 1.6) -> np.ndarray:
    """一个偏心的高斯斑：形状与目标明显不符，但峰值足够、不过曝。"""
    grid = _xy()
    gy = grid[..., 1] - _CENTRE[1]
    gx = grid[..., 0] - _CENTRE[0]
    return 60.0 * np.exp(-((gx - offset) ** 2 + (gy + offset) ** 2) / (2 * 0.9**2)) + 4.0


def _good() -> np.ndarray:
    return 60.0 * _target() + 6.0


class _Bench:
    """假光路: 第 n 次下发得到的帧由 ``frames(n)`` 决定（n 从 1 开始）。"""

    def __init__(self, frames) -> None:
        self.applied: list[np.ndarray] = []
        self._frames = frames
        self._frame_id = 0
        self.closed = False
        self.opened = 0
        self.calls: list[str] = []
        self.exposure_ms = 1.0
        self.full_scale = 255.0

    def open(self) -> None:
        self.opened += 1
        self.calls.append("open")

    def describe(self) -> dict:
        self.calls.append("describe")
        return {"name": "fake", "sensor": _SENSOR}

    def apply_phase(self, phase: np.ndarray) -> str:
        self.calls.append("apply_phase")
        self.applied.append(np.array(phase, dtype=np.float64, copy=True))
        return f"phase-{len(self.applied)}"

    def capture(self) -> CapturedFrame:
        self.calls.append("capture")
        self._frame_id += 1
        return CapturedFrame(
            frame=np.asarray(self._frames(len(self.applied)), dtype=np.float64),
            frame_id=self._frame_id,
            exposure_ms=self.exposure_ms,
            full_scale=self.full_scale,
            monotonic_ns=self._frame_id,
        )

    def close(self) -> None:
        self.closed = True


class _SquareForward:
    """够用的假物理预测模型：把控制网格池化到分析网格。

    真实 ``FrozenDefocusForward`` 吃 (grid, grid) 相位、吐 (H, W) 相机采样强度，
    两者尺度不同；这个替身用自适应池化复现该映射，好让信赖域测试量的真是控制网格。
    """

    def __init__(self, target: np.ndarray) -> None:
        self.grid_shape = np.asarray(target).shape

    def predict(self, phase, exposure_ms=None):
        import torch
        import torch.nn.functional as F

        p = phase if torch.is_tensor(phase) else torch.as_tensor(np.asarray(phase))
        if p.ndim == 2:
            p = p[None]
        pooled = F.adaptive_avg_pool2d(p, self.grid_shape)
        return 0.5 + 0.5 * torch.tanh(pooled)  # (1, H, W)，取值 0..1


def _config(**overrides) -> ClosedLoopConfig:
    base = dict(
        xy=_xy(),
        target=_target(),
        start_phase=np.zeros((8, 8), dtype=np.float64),
        max_rounds=2,
        candidate_policy="auto",
        adapt=True,
        exposure_ms=1.0,
    )
    base.update(overrides)
    return ClosedLoopConfig(**base)


# --------------------------------------------------------------------------- #
# 门禁
# --------------------------------------------------------------------------- #


def test_gate_accepts_a_clean_frame() -> None:
    gate = check_frame(_embed(_good()), full_scale=255.0, exposure_ms=1.0)
    assert gate.ok, gate.reasons
    assert gate.saturation_pixels == 0
    assert "通过" in describe_gate(gate)


def test_gate_rejects_saturated_frame() -> None:
    frame = _embed(_good())
    frame[5, 5] = 255.0
    gate = check_frame(frame, full_scale=255.0, exposure_ms=1.0)
    assert not gate.ok
    assert gate.reasons == ("saturated",)


def test_gate_rejects_low_signal() -> None:
    gate = check_frame(_embed(_target()) * 5.0, full_scale=255.0, exposure_ms=1.0)
    assert not gate.ok
    assert "low_signal" in gate.reasons


def test_gate_rejects_exposure_mismatch_and_invalid() -> None:
    assert "exposure_mismatch" in check_frame(
        _embed(_good()), full_scale=255.0, exposure_ms=3.0, expected_exposure_ms=1.0
    ).reasons
    assert "exposure_invalid" in check_frame(
        _embed(_good()), full_scale=255.0, exposure_ms=0.0
    ).reasons


def test_gate_flags_centroid_jump() -> None:
    frame = np.zeros((_SENSOR, _SENSOR), dtype=np.float64)
    frame[30, 30] = 80.0
    gate = check_frame(
        frame,
        full_scale=255.0,
        exposure_ms=1.0,
        previous_centroid=(2.0, 2.0),
        thresholds=GateThresholds(max_centroid_jump_px=5.0),
    )
    assert "centroid_jump" in gate.reasons
    assert not gate.ok


def test_gate_rejects_empty_and_nonfinite() -> None:
    assert "empty_frame" in check_frame(
        np.zeros((0, 0)), full_scale=255.0, exposure_ms=1.0
    ).reasons
    bad = _embed(_good())
    bad[0, 0] = np.nan
    assert "nonfinite" in check_frame(bad, full_scale=255.0, exposure_ms=1.0).reasons


# --------------------------------------------------------------------------- #
# 网格、目标与分数
# --------------------------------------------------------------------------- #


def test_analysis_grid_matches_transfer_geometry() -> None:
    grid = analysis_grid((100.0, 50.0), 3)
    expected_y, expected_x = np.mgrid[-3:4:2, -3:4:2]
    assert grid.shape == (4, 4, 2)
    np.testing.assert_array_equal(grid[..., 0], expected_x + 100)
    np.testing.assert_array_equal(grid[..., 1], expected_y + 50)


def test_target_field_shapes() -> None:
    rect = _target()
    assert rect.shape == (_WINDOW, _WINDOW)
    assert rect.sum() == 9  # 中心 3x3
    ring = target_field(_xy(), half_width=2.0, half_height=2.0, shape="ring")
    assert ring[2, 2] == 0.0  # 环内挖空
    assert ring.sum() < 10


def test_target_field_rejects_empty_target() -> None:
    with pytest.raises(ValueError):
        target_field(_xy(), half_width=0.1, half_height=0.1, centre_xy=(50.0, 50.0))


def test_score_is_scale_invariant() -> None:
    measured = _good()
    target = _target()
    base = compare_to_target(measured, target)
    scaled = compare_to_target(measured * 7.0, target)
    assert base["correlation"] == pytest.approx(scaled["correlation"])
    assert score_from_metrics(base) == pytest.approx(score_from_metrics(scaled))


def test_score_prefers_the_matching_shape() -> None:
    target = _target()
    good = score_from_metrics(compare_to_target(_good(), target))
    bad = score_from_metrics(compare_to_target(_blob(), target))
    assert good < bad


def test_compare_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        compare_to_target(np.zeros_like(_target()), _target())
    with pytest.raises(ValueError):
        compare_to_target(_good()[:2], _target())


def test_window_fraction_reports_leakage() -> None:
    frame = _embed(_good())
    fraction = window_fraction(frame, _xy())
    assert fraction == pytest.approx(1.0)
    frame[0, 0] = 500.0  # 窗口外漏光
    assert window_fraction(frame, _xy()) < 1.0


def test_sample_on_grid_rejects_out_of_bounds_window() -> None:
    with pytest.raises(ValueError):
        sample_on_grid(np.zeros((8, 8)), analysis_grid((10.0, 10.0), 4))


# --------------------------------------------------------------------------- #
# 晋级判据与候选生成
# --------------------------------------------------------------------------- #


def test_promotion_requires_a_real_margin() -> None:
    assert promotion_decision(0.50, 0.52, margin=0.01)[0]
    assert not promotion_decision(0.515, 0.52, margin=0.01)[0]
    assert not promotion_decision(0.52, 0.52, margin=0.01)[0]
    assert not promotion_decision(float("nan"), 0.5, margin=0.01)[0]
    assert not promotion_decision(None, 0.5, margin=0.01)[0]


def test_refine_phase_respects_the_trust_region() -> None:
    proposal = refine_phase(
        _SquareForward(_target()),
        _target(),
        np.zeros((8, 8)),
        iterations=5,
        lr=0.5,
        max_step_rad=0.2,
        phase_rms_cap=0.3,
    )
    assert proposal.phase.shape == (8, 8)
    assert len(proposal.history) == 5
    assert proposal.max_step_rad <= 0.2 + 1e-6
    assert proposal.rms_rad <= 0.3 + 1e-6
    assert np.isfinite(proposal.predicted_score)
    assert not np.allclose(proposal.phase, 0.0)


def test_refine_phase_rejects_a_wrong_sized_prediction() -> None:
    class _Mismatch:
        def predict(self, phase, exposure_ms=None):
            import torch

            return torch.zeros((1, 2, 2))

    with pytest.raises(ValueError, match="analysis grid"):
        refine_phase(_Mismatch(), _target(), np.zeros((8, 8)), iterations=2)


def test_config_allows_control_grid_different_from_analysis_grid() -> None:
    # SLM 控制网格 (grid_size) 与相机分析窗口 (span) 尺度不同，不能要求同形。
    cfg = _config(start_phase=np.zeros((8, 8)))
    assert cfg.start_phase.shape == (8, 8)
    assert cfg.xy.shape[:2] == (_WINDOW, _WINDOW)
    with pytest.raises(ValueError, match="2-D control-grid"):
        _config(start_phase=np.zeros((2, 8, 8)))
    with pytest.raises(ValueError, match="candidate_policy"):
        _config(candidate_policy="maybe")
    with pytest.raises(ValueError, match="exposure_ms"):
        _config(exposure_ms=0.0)


# --------------------------------------------------------------------------- #
# 编排：接受、回滚、门禁、确认
# --------------------------------------------------------------------------- #


def test_runner_accepts_only_a_measured_improvement() -> None:
    bench = _Bench(lambda n: _embed(_blob()) if n == 1 else _embed(_good()))
    runner = ClosedLoopRunner(
        bench, config=_config(max_rounds=1), forward=_SquareForward(_target())
    )
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    baseline, round_one = summary.rounds
    assert baseline.decision == "baseline"
    assert round_one.decision == "auto"
    assert round_one.measured_score is not None
    assert round_one.measured_score < baseline.measured_score
    assert round_one.predicted_score is not None  # 模型估计与实测分开留档
    assert summary.applied_score == pytest.approx(round_one.measured_score)
    assert bench.closed


def test_runner_opens_the_bench_before_any_device_call() -> None:
    """设备必须先 open 再 describe/下发/采集: 真实 HardwareBench 的硬契约。"""
    bench = _Bench(lambda n: _embed(_good()))
    runner = ClosedLoopRunner(
        bench, config=_config(max_rounds=1), forward=_SquareForward(_target())
    )
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert bench.opened == 1
    assert bench.calls[0] == "open"
    assert bench.calls.count("open") == 1
    # 所有设备调用都排在 open 之后 (没有第二次 open 混在中间)。
    assert bench.calls.index("describe") > bench.calls.index("open")
    assert bench.closed


def test_runner_reports_a_failed_open_without_touching_the_device() -> None:
    """open 失败要落成可读故障, 且不得再下发相位或采集。"""

    class _NoDeviceBench(_Bench):
        def open(self) -> None:
            super().open()
            raise RuntimeError("SLM 驱动板无应答")

    bench = _NoDeviceBench(lambda n: _embed(_good()))
    runner = ClosedLoopRunner(
        bench, config=_config(max_rounds=1), forward=_SquareForward(_target())
    )
    summary = runner.run()

    assert summary.state is LoopState.FAULT
    assert "SLM 驱动板无应答" in summary.reason
    assert bench.applied == []
    assert "apply_phase" not in bench.calls
    assert "capture" not in bench.calls
    assert bench.closed


def test_runner_rolls_back_a_worse_candidate() -> None:
    start = np.zeros((8, 8), dtype=np.float64)
    # 基线是好形状（低分），每个候选都更差（blob，高分）→ 全部拒绝。
    bench = _Bench(lambda n: _embed(_good()) if n == 1 else _embed(_blob()))
    runner = ClosedLoopRunner(
        bench, config=_config(max_rounds=5, max_rounds_without_improvement=2),
        forward=_SquareForward(_target()),
    )
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert "未被接受" in summary.reason
    assert [r.decision for r in summary.rounds[1:]] == ["reject", "reject"]
    assert np.allclose(summary.applied_phase, start)
    # 每轮候选之后都重新下发已确认相位：最后落在起始相位上。
    assert np.allclose(bench.applied[-1], start)


def test_runner_refuses_to_accept_a_gate_failed_frame() -> None:
    def frames(n: int) -> np.ndarray:
        if n == 1:
            return _embed(_good())
        saturated = _embed(_good())
        saturated[2, 2] = 255.0
        return saturated

    bench = _Bench(frames)
    runner = ClosedLoopRunner(
        bench,
        config=_config(max_rounds=8, max_consecutive_gate_failures=2),
        forward=_SquareForward(_target()),
    )
    summary = runner.run()

    assert summary.state is LoopState.FAULT
    assert "帧门禁" in summary.reason
    failed = [r for r in summary.rounds if r.decision == "gate_fail"]
    assert len(failed) == 2
    assert all(r.measured_score is None for r in failed)
    # 门禁失败后回滚：设备上的最后一个相位仍是起点相位。
    assert np.allclose(bench.applied[-1], summary.applied_phase)


def test_runner_faults_when_the_baseline_frame_fails_the_gate() -> None:
    saturated = _embed(_good())
    saturated[0, 0] = 255.0
    bench = _Bench(lambda n: saturated)
    runner = ClosedLoopRunner(
        bench, config=_config(), forward=_SquareForward(_target())
    )
    summary = runner.run()

    assert summary.state is LoopState.FAULT
    assert "起点相位" in summary.reason
    assert summary.applied_phase is None
    assert bench.closed


def test_runner_waits_for_operator_confirmation() -> None:
    bench = _Bench(lambda n: _embed(_blob()) if n == 1 else _embed(_good()))
    holder: dict[str, ClosedLoopRunner] = {}

    def on_event(event) -> None:
        if event.kind == "awaiting_decision":
            holder["runner"].request_accept()

    runner = ClosedLoopRunner(
        bench,
        config=_config(max_rounds=1, candidate_policy="confirm"),
        forward=_SquareForward(_target()),
        on_event=on_event,
    )
    holder["runner"] = runner
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert summary.rounds[1].decision == "accept"


def test_runner_stop_during_confirmation_keeps_the_applied_phase() -> None:
    bench = _Bench(lambda n: _embed(_blob()) if n == 1 else _embed(_good()))
    holder: dict[str, ClosedLoopRunner] = {}

    def on_event(event) -> None:
        if event.kind == "awaiting_decision":
            holder["runner"].request_stop()

    runner = ClosedLoopRunner(
        bench,
        config=_config(max_rounds=1, candidate_policy="confirm"),
        forward=_SquareForward(_target()),
        on_event=on_event,
    )
    holder["runner"] = runner
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert "停止" in summary.reason
    assert np.allclose(bench.applied[-1], summary.applied_phase)


def test_runner_registers_training_samples_only_from_accepted_rounds() -> None:
    class _SpyAdapter:
        def __init__(self) -> None:
            self.added: list[dict] = []

        def ready(self) -> bool:
            return False

        def refit(self):  # pragma: no cover - ready() 恒 False
            raise AssertionError("adapter must not refit while not ready")

        def add_round(self, **kwargs):
            self.added.append(kwargs)
            return "sample_0000.npz"

        def forward(self, xy):
            return None

    spy = _SpyAdapter()
    # 基线差 → 第 1 轮好（接受）→ 第 2 轮差（拒绝）。
    bench = _Bench(
        lambda n: _embed(_blob()) if n in {1, 3} else _embed(_good())
    )
    runner = ClosedLoopRunner(
        bench,
        config=_config(max_rounds=2, validation_every=0),
        forward=_SquareForward(_target()),
        adapter=spy,
    )
    summary = runner.run()

    assert [r.decision for r in summary.rounds[1:]] == ["auto", "reject"]
    assert len(spy.added) == 1
    assert spy.added[0]["split"] == "train"
    assert spy.added[0]["label"] == "round_01"
    assert spy.added[0]["full_scale"] == pytest.approx(255.0)


def test_runner_reserves_declared_validation_rounds() -> None:
    class _SpyAdapter:
        def __init__(self) -> None:
            self.splits: list[str] = []

        def ready(self) -> bool:
            return False

        def refit(self):  # pragma: no cover
            raise AssertionError

        def add_round(self, **kwargs):
            self.splits.append(kwargs["split"])
            return "x.npz"

        def forward(self, xy):
            return None

    spy = _SpyAdapter()
    bench = _Bench(lambda n: _embed(_blob()) if n == 1 else _embed(_good()))
    runner = ClosedLoopRunner(
        bench,
        config=_config(max_rounds=2, validation_every=2),
        forward=_SquareForward(_target()),
        adapter=spy,
    )
    runner.run()
    # 第 2 个被接受的轮次留作验证族，绝不进训练。
    assert spy.splits == ["train", "validation"]


# --------------------------------------------------------------------------- #
# 在线适配器（只测磁盘契约与"没有标定就不适配"）
# --------------------------------------------------------------------------- #


def test_adapter_without_calibration_stays_insufficient(tmp_path: Path) -> None:
    adapter = OnlineTransferAdapter(
        checkpoint=tmp_path / "ckpt", session_dir=tmp_path / "session", span=_SPAN
    )
    assert not adapter.seeded
    assert not adapter.ready()
    result = adapter.refit()
    assert result.status == "insufficient"
    assert "标定" in result.detail
    assert not result.promoted


def test_adapter_seed_reports_missing_reference(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("[]", encoding="utf-8")
    adapter = OnlineTransferAdapter(
        checkpoint=tmp_path / "ckpt",
        session_dir=tmp_path / "session",
        span=_SPAN,
        source_dataset=source,
    )
    assert not adapter.seeded
    assert adapter.seed_error is not None and "reference_start" in adapter.seed_error


def test_adapter_writes_a_replayable_dataset(tmp_path: Path) -> None:
    adapter = OnlineTransferAdapter(
        checkpoint=tmp_path / "ckpt", session_dir=tmp_path / "session", span=_SPAN
    )
    name = adapter.add_round(
        phase=np.zeros((8, 8)),
        frame=_embed(_good()),
        exposure_ms=2.0,
        spot_xy=_CENTRE,
        label="round_01",
    )
    payload = np.load(tmp_path / "session" / name)
    assert set(payload.files) == {"model_phase", "raw_frames"}
    assert payload["model_phase"].shape == (8, 8)
    assert payload["raw_frames"].shape == (1, _SENSOR, _SENSOR)
    assert payload["raw_frames"].dtype == np.uint8

    manifest = __import__("json").loads(
        (tmp_path / "session" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest[-1]["split"] == "train"
    assert manifest[-1]["exposure_ms"] == pytest.approx(2.0)
    assert adapter.counts() == (1, 0)


def test_adapter_keeps_16bit_rows_above_8bit_scale(tmp_path: Path) -> None:
    adapter = OnlineTransferAdapter(
        checkpoint=tmp_path / "ckpt", session_dir=tmp_path / "session", span=_SPAN
    )
    name = adapter.add_round(
        phase=np.zeros((8, 8)),
        frame=_embed(_good()),
        exposure_ms=1.0,
        spot_xy=_CENTRE,
        label="round_01",
        full_scale=4095.0,
    )
    payload = np.load(tmp_path / "session" / name)
    assert payload["raw_frames"].dtype == np.uint16
    rows = __import__("json").loads(
        (tmp_path / "session" / "manifest.json").read_text(encoding="utf-8")
    )
    assert rows[-1]["full_scale"] == pytest.approx(4095.0)
    # 帧按 12-bit 满量程记录（dtype=uint16 但值在 0..4095）——这是真实 Mono12
    # 采集的形状；valid 用 full_scale 作比例基准，此行必须仍然有效。
    from ao_shaping.tools.slm.transfer.fit import valid

    assert valid(rows[-1])


def test_adapter_rejects_bad_round_arguments(tmp_path: Path) -> None:
    adapter = OnlineTransferAdapter(
        checkpoint=tmp_path / "ckpt", session_dir=tmp_path / "session", span=_SPAN
    )
    with pytest.raises(ValueError):
        adapter.add_round(
            phase=np.zeros((8, 8)), frame=_embed(_good()), exposure_ms=0.0,
            spot_xy=_CENTRE, label="x",
        )
    with pytest.raises(ValueError):
        adapter.add_round(
            phase=np.zeros((8, 8)), frame=_embed(_good()), exposure_ms=1.0,
            spot_xy=_CENTRE, label="x", split="test",
        )


# --------------------------------------------------------------------------- #
# hold 模式（同一固定目标 + 固定 CCD 像素位置）
# --------------------------------------------------------------------------- #


class _TimedBench(_Bench):
    """假光路：采集时刻取自真实的单调时钟，hold 模式据此判定帧是否晚于写入。

    ``stale_after`` 不为 None 时，第 ``stale_after`` 帧之后的帧都带上一个
    **构造时**的旧时刻 —— 也就是早于本次运行的任何相位写入。
    """

    def __init__(self, frames, *, stale_after: int | None = None) -> None:
        super().__init__(frames)
        self._stale_after = stale_after
        self._stale_ns = time.monotonic_ns() if stale_after is not None else None

    def capture(self) -> CapturedFrame:
        self.calls.append("capture")
        self._frame_id += 1
        stale = self._stale_after is not None and self._frame_id > self._stale_after
        return CapturedFrame(
            frame=np.asarray(self._frames(len(self.applied)), dtype=np.float64),
            frame_id=self._frame_id,
            exposure_ms=self.exposure_ms,
            full_scale=self.full_scale,
            monotonic_ns=int(self._stale_ns) if stale else time.monotonic_ns(),
        )


def test_hold_config_rejects_unattended_conflicts_and_bad_values() -> None:
    with pytest.raises(ValueError, match="candidate_policy"):
        _config(mode="hold", candidate_policy="confirm")
    with pytest.raises(ValueError, match="hold_target_ccd_px"):
        _config(mode="hold", hold_target_ccd_px=0.0)
    with pytest.raises(ValueError, match="hold_target_ccd_px"):
        _config(mode="hold", hold_target_ccd_px=float("nan"))
    with pytest.raises(ValueError, match="mode"):
        _config(mode="stare")


def test_hold_runs_to_the_round_cap_without_confirmation() -> None:
    events: list = []
    bench = _TimedBench(lambda n: _embed(_good()))
    runner = ClosedLoopRunner(
        bench,
        config=_config(mode="hold", max_rounds=4),
        forward=_SquareForward(_target()),
        on_event=events.append,
    )
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert "轮数上限" in summary.reason
    # 基线 + 4 轮全部跑完，且从不进入"等待人工确认"。
    assert len(summary.rounds) == 5
    assert [r.decision for r in summary.rounds[1:]] == ["hold"] * 4
    assert all(event.kind != "awaiting_decision" for event in events)
    assert bench.closed


def test_hold_never_accepts_a_frame_captured_before_the_write() -> None:
    bench = _TimedBench(lambda n: _embed(_good()), stale_after=1)
    runner = ClosedLoopRunner(
        bench,
        config=_config(mode="hold", max_rounds=1),
        forward=_SquareForward(_target()),
    )
    summary = runner.run()

    # 旧帧不构成证据：不 ACCEPT、不改善最佳，设备回滚到最佳（起点）。
    assert summary.state is LoopState.STOPPED
    assert summary.rounds[-1].decision == "invalid"
    assert "old_frame" in summary.rounds[-1].note
    assert np.allclose(summary.applied_phase, summary.best_phase)
    assert np.allclose(summary.applied_phase, np.zeros((8, 8)))
    assert np.allclose(bench.applied[-1], np.zeros((8, 8)))


def test_hold_reports_a_rollback_with_ccd_pixel_centroid_error() -> None:
    events: list = []
    bench = _TimedBench(lambda n: _embed(_good()) if n == 1 else _embed(_blob()))
    runner = ClosedLoopRunner(
        bench,
        config=_config(mode="hold", max_rounds=1),
        forward=_SquareForward(_target()),
        on_event=events.append,
    )
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert summary.rounds[-1].decision == "rollback"
    assert np.allclose(bench.applied[-1], np.zeros((8, 8)))

    payloads = [event.payload for event in events if event.kind == "lock"]
    rollbacks = [payload for payload in payloads if payload["action"] == "rollback"]
    assert len(rollbacks) == 1
    payload = rollbacks[0]
    assert payload["round_index"] == 1
    assert payload["lock_state"] == "converging"
    assert set(payload) == {
        "round_index",
        "lock_state",
        "action",
        "score",
        "best_score",
        "centroid_error_ccd_px",
        "target_centroid_xy",
        "centroid_xy",
    }
    # 分析网格步长是 2 CCD px：CCD 误差必须是网格误差的两倍。
    assert payload["centroid_error_ccd_px"] > 0.0
    gx, gy = payload["centroid_xy"]
    tx, ty = payload["target_centroid_xy"]
    grid_error = float(np.hypot(gx - tx, gy - ty))
    assert payload["centroid_error_ccd_px"] == pytest.approx(2.0 * grid_error, rel=1e-9)


def test_hold_freezes_after_repeated_invalid_frames() -> None:
    events: list = []
    bench = _TimedBench(lambda n: _embed(_good()), stale_after=1)
    runner = ClosedLoopRunner(
        bench,
        config=_config(mode="hold", max_rounds=6),
        forward=_SquareForward(_target()),
        on_event=events.append,
    )
    summary = runner.run()

    assert summary.state is LoopState.FAULT
    assert "监督器" in summary.reason
    assert "repeated_invalid_frames" in summary.reason
    # 基线 + 3 个无效帧后由监督器冻结；冻结后不再生成、不再下发候选。
    assert len(summary.rounds) == 4
    assert [r.decision for r in summary.rounds[1:]] == ["invalid", "invalid", "fault"]
    assert bench.calls.count("capture") == 4
    assert bench.calls.count("apply_phase") == 1 + 2 * 3
    assert np.allclose(bench.applied[-1], np.zeros((8, 8)))
    faults = [event.payload for event in events if event.kind == "lock" and event.payload["action"] == "fault"]
    assert len(faults) == 1
    assert faults[0]["lock_state"] == "frozen"


def test_hold_stops_on_request_and_keeps_the_last_accepted_phase() -> None:
    events: list = []
    holder: dict[str, ClosedLoopRunner] = {}

    def on_event(event) -> None:
        events.append(event)
        if (
            event.kind == "lock"
            and isinstance(event.payload, dict)
            and event.payload.get("action") == "accept"
            and event.payload.get("round_index") == 1
        ):
            holder["runner"].request_stop()

    bench = _TimedBench(lambda n: _embed(_blob()) if n == 1 else _embed(_good()))
    runner = ClosedLoopRunner(
        bench,
        config=_config(mode="hold", max_rounds=4),
        forward=_SquareForward(_target()),
        on_event=on_event,
    )
    holder["runner"] = runner
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert "停止" in summary.reason
    assert summary.rounds[-1].decision == "accept"
    # 只在轮次边界停：设备上留下的是最后被接受的那个最佳相位。
    assert np.allclose(summary.applied_phase, bench.applied[-1])
    assert not np.allclose(summary.applied_phase, np.zeros((8, 8)))
    assert "stop_requested" in [event.kind for event in events]


def test_optimize_mode_still_stops_when_candidates_stop_improving() -> None:
    bench = _TimedBench(lambda n: _embed(_good()) if n == 1 else _embed(_blob()))
    runner = ClosedLoopRunner(
        bench,
        config=_config(max_rounds=6, max_rounds_without_improvement=2),
        forward=_SquareForward(_target()),
    )
    assert runner.config.mode == "optimize"
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert "未被接受" in summary.reason
    assert [r.decision for r in summary.rounds[1:]] == ["reject", "reject"]
