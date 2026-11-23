"""固定目标锁定监督器的离线测试（无相机、无 SLM、无 Qt）。

覆盖四类契约：

1. **新鲜度与配对**：没有登记写入、写序号不符、采集早于写入、重复采集序号
   都只算门禁失败，不进入评分，更不改变最佳值。
2. **图像质量门禁**：尺寸、非有限值、过曝、低信号、平场与绝对位置越界。
3. **固定目标语义**：目标质心在 arm 时固定，平移后的同形状光斑评分变差，
   不做重新居中。
4. **决策与冻结**：每帧至多一个事件；连续无效、长期无改善、来回振荡都只
   冻结一次，之后 observe 不再发事件。
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.gui.slm import target_lock as tl

SHAPE = (64, 64)
TARGET = np.zeros(SHAPE, dtype=np.float64)
TARGET[26:38, 24:40] = 200.0  # 16x12 矩形，质心约在 (31.5, 31.5)


def shifted(dx: int = 0, dy: int = 0, peak: float = 200.0) -> np.ndarray:
    image = np.zeros(SHAPE, dtype=np.float64)
    y0, y1 = 26 + dy, 38 + dy
    x0, x1 = 24 + dx, 40 + dx
    image[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = peak
    return image


def frame(
    image: np.ndarray | None = None,
    *,
    write_seq: int = 1,
    capture_seq: int = 1,
    captured_at_s: float = 1.1,
    exposure_ms: float = 5.0,
) -> tl.MeasuredFrame:
    return tl.MeasuredFrame(
        image=TARGET.copy() if image is None else image,
        write_seq=write_seq,
        capture_seq=capture_seq,
        captured_at_s=captured_at_s,
        exposure_ms=exposure_ms,
    )


def armed(
    config: tl.LockConfig | None = None,
    callbacks: tl.LockCallbacks | None = None,
) -> tl.TargetLockSupervisor:
    supervisor = tl.TargetLockSupervisor(config, callbacks)
    supervisor.arm(tl.FixedTarget(TARGET.copy()))
    return supervisor


def count_actions(callbacks: dict[str, int]) -> tl.LockCallbacks:
    return tl.LockCallbacks(
        on_propose=lambda _d: callbacks.__setitem__("propose", callbacks["propose"] + 1),
        on_accept=lambda _d: callbacks.__setitem__("accept", callbacks["accept"] + 1),
        on_rollback=lambda _d: callbacks.__setitem__("rollback", callbacks["rollback"] + 1),
        on_fault=lambda _d: callbacks.__setitem__("fault", callbacks["fault"] + 1),
    )


# --------------------------------------------------------------- 配对与新鲜度
def test_observe_before_arm_is_refused_without_side_effects():
    calls: list[str] = []
    supervisor = tl.TargetLockSupervisor(callbacks=tl.LockCallbacks(on_accept=calls.append))
    with pytest.raises(RuntimeError, match="arm"):
        supervisor.observe(frame())
    assert calls == []


def test_write_registration_must_be_strictly_increasing():
    supervisor = armed()
    supervisor.expect_write(5, written_at_s=1.0)
    with pytest.raises(ValueError, match="严格递增"):
        supervisor.expect_write(5, written_at_s=2.0)
    with pytest.raises(ValueError, match="严格递增"):
        supervisor.expect_write(4, written_at_s=2.0)
    supervisor.expect_write(6, written_at_s=2.0)


def test_missing_write_registration_gates_the_frame():
    supervisor = armed()
    assert supervisor.observe(frame()) is None
    assert supervisor.last_gate_reasons == ("no_pending_write",)
    assert supervisor.state is tl.LockState.CONVERGING


def test_unpaired_write_sequence_is_gated_even_if_frame_is_fresh():
    supervisor = armed()
    supervisor.expect_write(1, written_at_s=1.0)
    assert supervisor.observe(frame(write_seq=2)) is None
    assert "unpaired_frame" in supervisor.last_gate_reasons
    assert supervisor.best_score is None


def test_frame_captured_at_or_before_write_is_old():
    supervisor = armed()
    supervisor.expect_write(1, written_at_s=10.0)
    assert supervisor.observe(frame(captured_at_s=10.0)) is None
    assert "old_frame" in supervisor.last_gate_reasons


def test_duplicate_capture_sequence_is_gated():
    supervisor = armed()
    supervisor.expect_write(1, written_at_s=1.0)
    assert supervisor.observe(frame(capture_seq=1)) is not None
    assert supervisor.observe(frame(capture_seq=1, captured_at_s=1.1)) is None
    assert "duplicate_frame" in supervisor.last_gate_reasons


# --------------------------------------------------------------- 图像门禁
@pytest.mark.parametrize(
    ("image", "reason", "config"),
    [
        (np.zeros((32, 32)), "invalid_shape", None),
        (np.full(SHAPE, np.nan), "not_finite", None),
        (TARGET - 5.0, "negative_intensity", None),
        (shifted(peak=255.0) + 300.0, "overexposed", None),
        (np.zeros(SHAPE), "low_signal", None),
        (np.full(SHAPE, 120.0), "flat_frame", None),
        (shifted(dx=20), "location_mismatch", tl.LockConfig(max_centroid_error_px=5.0)),
    ],
)
def test_invalid_frames_are_gated_with_a_reason(image, reason, config):
    supervisor = armed(config)
    supervisor.expect_write(1, written_at_s=1.0)
    assert supervisor.observe(frame(image)) is None
    assert reason in supervisor.last_gate_reasons
    assert supervisor.best_score is None


def test_overexposure_threshold_uses_saturated_pixel_fraction():
    saturated = shifted()
    saturated[0:12, 0:12] = 255.0  # 144 px > 1% of 4096
    supervisor = armed(tl.LockConfig(max_saturated_fraction=0.01))
    supervisor.expect_write(1, written_at_s=1.0)
    assert supervisor.observe(frame(saturated)) is None
    assert supervisor.last_gate_reasons == ("overexposed",)


# --------------------------------------------------------------- 固定目标语义
def test_target_centroid_is_fixed_and_not_recentered():
    supervisor = armed(tl.LockConfig(max_centroid_error_px=100.0))
    target_centroid = supervisor.target_centroid
    assert target_centroid is not None

    supervisor.expect_write(1, written_at_s=1.0)
    aligned = supervisor.observe(frame(shifted(), capture_seq=1))
    assert aligned is not None
    assert aligned.score == pytest.approx(0.0, abs=1e-9)
    assert aligned.centroid_error_px == pytest.approx(0.0, abs=1e-9)

    supervisor.expect_write(2, written_at_s=2.0)
    moved = supervisor.observe(
        frame(shifted(dx=6), write_seq=2, capture_seq=2, captured_at_s=2.1)
    )
    assert moved is not None
    assert moved.centroid_xy is not None
    assert moved.centroid_xy[0] - target_centroid[0] == pytest.approx(6.0, abs=0.1)
    # 同形状、整体平移 6 px：若被重新居中，score 仍会是 0；固定网格会变差。
    assert moved.score > aligned.score + 0.05
    assert supervisor.target_centroid == target_centroid


# --------------------------------------------------------------- 决策与冻结
def test_baseline_then_improvement_accepts_and_best_write_seq_tracks_winner():
    supervisor = armed(tl.LockConfig(improvement_epsilon=0.01))
    supervisor.expect_write(1, written_at_s=1.0)
    baseline = supervisor.observe(frame(shifted(dx=1), capture_seq=1))
    assert baseline is not None and baseline.action is tl.LockAction.ACCEPT
    assert baseline.reason == "baseline_established"
    baseline_score = supervisor.best_score
    assert baseline_score is not None and baseline_score > 0.0

    supervisor.expect_write(2, written_at_s=2.0)
    improved = supervisor.observe(
        frame(shifted(), write_seq=2, capture_seq=2, captured_at_s=2.1)
    )
    assert improved is not None and improved.action is tl.LockAction.ACCEPT
    assert improved.reason == "improved"
    assert improved.score < baseline_score
    assert supervisor.best_write_seq == 2


def test_worse_candidate_rolls_back_and_keeps_best():
    supervisor = armed()
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    best = supervisor.best_score

    supervisor.expect_write(2, written_at_s=2.0)
    degraded = supervisor.observe(
        frame(shifted(dx=8), write_seq=2, capture_seq=2, captured_at_s=2.1)
    )
    assert degraded is not None and degraded.action is tl.LockAction.ROLLBACK
    assert degraded.reason == "degraded"
    assert degraded.best_score == best
    assert supervisor.best_write_seq == 1


def test_quiet_non_improvement_transitions_to_maintaining_not_fault():
    supervisor = armed(
        tl.LockConfig(converge_patience=2, max_non_improving=5, oscillation_amplitude=0.5)
    )
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    for seq in (2, 3):
        supervisor.expect_write(seq, written_at_s=float(seq))
        decision = supervisor.observe(
            frame(shifted(), write_seq=seq, capture_seq=seq, captured_at_s=seq + 0.1)
        )
        assert decision is not None
    assert supervisor.state is tl.LockState.MAINTAINING
    assert decision.action is tl.LockAction.PROPOSE
    assert supervisor.fault_count == 0


def test_maintenance_tolerates_long_quiet_periods_without_faulting():
    supervisor = armed(
        tl.LockConfig(converge_patience=2, max_non_improving=3, oscillation_amplitude=0.5)
    )
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    decision = None
    for seq in range(2, 10):
        supervisor.expect_write(seq, written_at_s=float(seq))
        decision = supervisor.observe(
            frame(shifted(), write_seq=seq, capture_seq=seq, captured_at_s=seq + 0.1)
        )
    assert supervisor.state is tl.LockState.MAINTAINING
    assert decision is not None and decision.action is tl.LockAction.PROPOSE
    assert supervisor.fault_count == 0


def test_maintenance_still_freezes_on_repeated_drift():
    supervisor = armed(
        tl.LockConfig(
            converge_patience=2,
            max_non_improving=3,
            oscillation_amplitude=10.0,
        )
    )
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    for seq in (2, 3):
        supervisor.expect_write(seq, written_at_s=float(seq))
        supervisor.observe(
            frame(shifted(), write_seq=seq, capture_seq=seq, captured_at_s=seq + 0.1)
        )
    assert supervisor.state is tl.LockState.MAINTAINING

    fault = None
    for seq in (4, 5, 6):
        supervisor.expect_write(seq, written_at_s=float(seq))
        fault = supervisor.observe(
            frame(shifted(dx=8), write_seq=seq, capture_seq=seq, captured_at_s=seq + 0.1)
        )
    assert fault is not None and fault.action is tl.LockAction.FAULT
    assert fault.reason == "non_improving_corrections"


def test_repeated_invalid_frames_freeze_exactly_once():
    calls = {"propose": 0, "accept": 0, "rollback": 0, "fault": 0}
    supervisor = armed(
        tl.LockConfig(max_invalid_frames=2),
        count_actions(calls),
    )
    supervisor.expect_write(1, written_at_s=1.0)
    assert supervisor.observe(frame(np.zeros(SHAPE), capture_seq=1)) is None
    fault = supervisor.observe(frame(np.zeros(SHAPE), capture_seq=2, captured_at_s=1.1))
    assert fault is not None and fault.action is tl.LockAction.FAULT
    assert fault.reason == "repeated_invalid_frames"
    assert supervisor.state is tl.LockState.FROZEN
    assert supervisor.fault_count == 1

    # 冻结后既不发事件，也不再改变任何计数。
    for seq in range(3, 6):
        assert supervisor.observe(frame(capture_seq=seq, captured_at_s=1.0 + seq)) is None
    assert calls == {"propose": 0, "accept": 0, "rollback": 0, "fault": 1}


def test_non_improving_candidates_freeze_when_oscillation_is_disabled():
    supervisor = armed(
        tl.LockConfig(
            max_non_improving=3,
            converge_patience=5,
            oscillation_amplitude=10.0,
        )
    )
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    fault = None
    for seq in (2, 3, 4):
        supervisor.expect_write(seq, written_at_s=float(seq))
        fault = supervisor.observe(
            frame(shifted(dx=6), write_seq=seq, capture_seq=seq, captured_at_s=seq + 0.1)
        )
    assert fault is not None and fault.action is tl.LockAction.FAULT
    assert fault.reason == "non_improving_corrections"
    assert supervisor.state is tl.LockState.FROZEN


def test_alternating_scores_freeze_as_oscillation():
    supervisor = armed()
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    fault = None
    for seq, dx in enumerate((2, 8, 2, 8, 2), start=2):
        supervisor.expect_write(seq, written_at_s=float(seq))
        fault = supervisor.observe(
            frame(
                shifted(dx=dx),
                write_seq=seq,
                capture_seq=seq,
                captured_at_s=seq + 0.1,
            )
        )
    assert fault is not None and fault.action is tl.LockAction.FAULT
    assert fault.reason == "oscillating_corrections"
    assert supervisor.state is tl.LockState.FROZEN


# --------------------------------------------------------------- 回调节奏与复位
def test_callbacks_receive_one_decision_per_frame_and_actions_route_correctly():
    calls: list[tl.LockAction] = []
    callbacks = tl.LockCallbacks(
        on_propose=lambda d: calls.append(d.action),
        on_accept=lambda d: calls.append(d.action),
        on_rollback=lambda d: calls.append(d.action),
        on_fault=lambda d: calls.append(d.action),
    )
    supervisor = armed(tl.LockConfig(converge_patience=2), callbacks)
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(dx=1), capture_seq=1))  # baseline accept
    supervisor.expect_write(2, written_at_s=2.0)
    supervisor.observe(frame(shifted(), write_seq=2, capture_seq=2, captured_at_s=2.1))
    supervisor.expect_write(3, written_at_s=3.0)
    supervisor.observe(
        frame(shifted(dx=8), write_seq=3, capture_seq=3, captured_at_s=3.1)
    )
    supervisor.expect_write(4, written_at_s=4.0)
    supervisor.observe(frame(shifted(), write_seq=4, capture_seq=4, captured_at_s=4.1))
    assert calls == [
        tl.LockAction.ACCEPT,
        tl.LockAction.ACCEPT,
        tl.LockAction.ROLLBACK,
        tl.LockAction.PROPOSE,
    ]


def test_frozen_supervisor_refuses_new_writes_until_reset():
    supervisor = armed(tl.LockConfig(max_invalid_frames=1))
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(np.zeros(SHAPE)))
    assert supervisor.state is tl.LockState.FROZEN
    with pytest.raises(RuntimeError, match="冻结"):
        supervisor.expect_write(2, written_at_s=2.0)

    supervisor.reset()
    assert supervisor.state is tl.LockState.IDLE
    supervisor.arm(tl.FixedTarget(TARGET.copy()))
    supervisor.expect_write(1, written_at_s=1.0)
    decision = supervisor.observe(frame(shifted()))
    assert decision is not None and decision.action is tl.LockAction.ACCEPT


def test_snapshot_is_a_plain_read_only_view():
    supervisor = armed()
    supervisor.expect_write(1, written_at_s=1.0)
    supervisor.observe(frame(shifted(), capture_seq=1))
    snapshot = supervisor.snapshot()
    assert snapshot["state"] == "converging"
    assert snapshot["best_write_seq"] == 1
    assert snapshot["pending_write_seq"] == 1
    assert snapshot["score_history"] == [snapshot["best_score"]]


# --------------------------------------------------------------------------- #
# LockConfig 阈值标定：默认阈值必须能被实测噪声替换，且替换倍数是可审计的
# --------------------------------------------------------------------------- #


def test_calibration_stats_read_noise_signal_and_repeatability():
    rng = np.random.default_rng(7)
    images = []
    for _ in range(6):
        image = shifted().copy()
        image += rng.normal(0.0, 2.0, size=SHAPE)
        images.append(np.clip(image, 0.0, 255.0))

    sample = tl.calibration_stats_from_frames(images, target=TARGET, full_scale=255.0)

    assert 0.3 < sample.dark_mean < 2.5  # 截断到 0 后的背景均值
    assert 0.5 < sample.dark_sigma < 2.5
    assert sample.peak_counts > 190.0
    assert sample.total_counts > 0.0
    assert sample.full_scale == 255.0
    assert sample.centroid_sigma_px >= 0.0
    assert sample.score_sigma > 0.0
    assert sample.pixel_count == SHAPE[0] * SHAPE[1]


def test_calibration_stats_rejects_shape_mismatch_and_empty_frames():
    with pytest.raises(ValueError, match="至少需要一帧"):
        tl.calibration_stats_from_frames([], target=TARGET)
    with pytest.raises(ValueError, match="不一致"):
        tl.calibration_stats_from_frames([np.zeros((8, 8))], target=TARGET)


def test_calibrated_thresholds_scale_with_measured_noise():
    quiet = tl.CalibrationSample(
        dark_mean=1.0,
        dark_sigma=1.0,
        peak_counts=200.0,
        total_counts=10000.0,
        saturated_fraction=0.0,
        full_scale=255.0,
        centroid_sigma_px=0.2,
        score_sigma=0.01,
        pixel_count=4096,
    )
    noisy = tl.CalibrationSample(
        dark_mean=5.0,
        dark_sigma=4.0,
        peak_counts=200.0,
        total_counts=10000.0,
        saturated_fraction=0.02,
        full_scale=255.0,
        centroid_sigma_px=0.2,
        score_sigma=0.04,
        pixel_count=4096,
    )
    q = tl.calibrate_lock_thresholds(
        quiet, position_tolerance_ccd_px=5.0, grid_spacing_px=2.0
    )
    n = tl.calibrate_lock_thresholds(
        noisy, position_tolerance_ccd_px=5.0, grid_spacing_px=2.0
    )

    assert q.min_peak_counts == pytest.approx(21.0)
    assert n.min_peak_counts == pytest.approx(85.0)
    assert q.saturation_counts == pytest.approx(0.98 * 255.0)
    assert n.max_saturated_fraction == pytest.approx(0.05)  # 0.02*3 被封顶
    assert q.max_centroid_error_px == pytest.approx(2.5)
    assert n.improvement_epsilon == pytest.approx(0.2)
    assert n.rollback_epsilon == pytest.approx(0.6)
    assert n.oscillation_amplitude == pytest.approx(0.2)
    assert n.min_peak_counts > q.min_peak_counts
    assert n.improvement_epsilon > q.improvement_epsilon
    # 标定结果必须仍是合法 LockConfig，能直接交给监督器使用。
    assert armed(n).config is n


def test_calibrate_rejects_nonpositive_geometry():
    sample = tl.CalibrationSample(
        dark_mean=0.0,
        dark_sigma=1.0,
        peak_counts=200.0,
        total_counts=10000.0,
        saturated_fraction=0.0,
        full_scale=255.0,
        centroid_sigma_px=0.2,
        score_sigma=0.01,
        pixel_count=4096,
    )
    with pytest.raises(ValueError, match="grid_spacing"):
        tl.calibrate_lock_thresholds(
            sample, position_tolerance_ccd_px=5.0, grid_spacing_px=0.0
        )
    with pytest.raises(ValueError, match="position_tolerance"):
        tl.calibrate_lock_thresholds(
            sample, position_tolerance_ccd_px=0.0, grid_spacing_px=2.0
        )
