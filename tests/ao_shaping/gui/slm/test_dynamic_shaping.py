"""动态整形过渡引擎的离线单元测试（不触碰任何硬件）。

覆盖用户核心诉求的数学保证：端点逐码精确、中间帧走最短环绕路径、
每个像素在过渡中最多走半周期、缓动曲线起止平缓。执行器（:class:`DynamicShapingWorker`）
的硬件交互不在此测试范围。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.gui.slm.dynamic_shaping import (
    MAX_GRAY_CODE,
    TransitionEstimate,
    TransitionPlan,
    blend_gray,
    ease,
    estimate_transition,
    pixel_flip_wait_s,
    wrapped_gray_delta,
)


# --------------------------------------------------------------------- 缓动


@pytest.mark.parametrize("kind", ["smooth", "linear", "ease_in", "ease_out"])
def test_ease_endpoints_are_exact(kind: str) -> None:
    assert ease(0.0, kind) == pytest.approx(0.0)
    assert ease(1.0, kind) == pytest.approx(1.0)


def test_ease_smooth_is_s_curve() -> None:
    # 3t²-2t³：中点斜率平缓，起止都为 0 斜率。
    assert ease(0.5, "smooth") == pytest.approx(0.5)
    assert ease(0.25, "smooth") == pytest.approx(0.25 * 0.25 * (3 - 0.5))


def test_ease_clamps_out_of_range() -> None:
    assert ease(-0.5, "linear") == 0.0
    assert ease(1.5, "linear") == 1.0


def test_ease_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ease(0.5, "nope")


# ----------------------------------------------------------- 最短环绕码差


def test_wrapped_delta_stays_in_half_period() -> None:
    start = np.array([[0, 100, 512]], dtype=np.int32)
    target = np.array([[1023, 900, 1023]], dtype=np.int32)
    delta = wrapped_gray_delta(start, target)
    assert delta.min() >= -512
    assert delta.max() <= 511


def test_wrapped_delta_takes_short_path_across_boundary() -> None:
    # 0 → 1023：物理上 1023≡0，两者相邻，最短路径是 1 步（引擎取 -1，经 0 回绕）。
    delta = wrapped_gray_delta(np.array([[0]]), np.array([[1023]]))
    assert abs(int(delta[0, 0])) == 1

    # 1022 → 0：经 1023 回绕，最短路径是 2 步（1022→1023→0）。
    delta = wrapped_gray_delta(np.array([[1022]]), np.array([[0]]))
    assert abs(int(delta[0, 0])) == 2


def test_wrapped_delta_identical_is_zero() -> None:
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    assert np.all(wrapped_gray_delta(a, a) == 0)


def test_wrapped_delta_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        wrapped_gray_delta(np.zeros((4, 4)), np.zeros((8, 8)))


# --------------------------------------------------------------- 灰度混合


def test_blend_endpoints_are_exact() -> None:
    start = np.array([[10, 200, 1022]], dtype=np.uint16)
    target = np.array([[900, 50, 0]], dtype=np.uint16)
    assert np.array_equal(blend_gray(start, target, 0.0), start)
    assert np.array_equal(blend_gray(start, target, 1.0), target)


def test_blend_output_always_in_valid_range() -> None:
    rng = np.random.default_rng(0)
    start = rng.integers(0, MAX_GRAY_CODE + 1, size=(16, 16)).astype(np.uint16)
    target = rng.integers(0, MAX_GRAY_CODE + 1, size=(16, 16)).astype(np.uint16)
    for t in np.linspace(0.0, 1.0, 25):
        frame = blend_gray(start, target, t)
        assert frame.dtype == np.uint16
        assert frame.min() >= 0
        assert frame.max() <= MAX_GRAY_CODE


def test_blend_never_overshoots_half_period() -> None:
    # 每个中间帧相对起点的环绕距离应随 t 单调、且不超过半周期。
    start = np.array([[0]], dtype=np.uint16)
    target = np.array([[1023]], dtype=np.uint16)  # delta = +1
    half = np.abs(wrapped_gray_delta(start, target)).max()
    distances = []
    for t in np.linspace(0.0, 1.0, 31):
        frame = blend_gray(start, target, t)
        d = np.abs(wrapped_gray_delta(start, frame)).max()
        distances.append(d)
        assert d <= half
    assert distances == sorted(distances)  # 单调不减


def test_blend_clamps_progress() -> None:
    start = np.array([[0]], dtype=np.uint16)
    target = np.array([[500]], dtype=np.uint16)
    assert np.array_equal(blend_gray(start, target, -3.0), start)
    assert np.array_equal(blend_gray(start, target, 9.0), target)


# ------------------------------------------------------------- 时间轴计划


def test_plan_frame_count_and_progress() -> None:
    plan = TransitionPlan(duration_s=1.0, fps=10)
    # 1s × 10fps + 1 = 11 帧。
    assert plan.frame_count == 11
    assert plan.frame_interval_s == pytest.approx(0.1)
    assert plan.progress_of_frame(0) == 0.0
    assert plan.progress_of_frame(10) == 1.0
    assert plan.progress_of_frame(99) == 1.0  # 越界钳位


def test_plan_validation_raises() -> None:
    with pytest.raises(ValueError):
        TransitionPlan(duration_s=0.01)  # 低于下限
    with pytest.raises(ValueError):
        TransitionPlan(duration_s=1.0, fps=0)  # 帧率越界
    with pytest.raises(ValueError):
        TransitionPlan(duration_s=1.0, easing="bogus")
    with pytest.raises(ValueError):
        TransitionPlan(duration_s=1.0, dwell_ms=99999.0)


# ------------------------------------------------------------- 速率估算


def test_estimate_transition_fields() -> None:
    start = np.zeros((8, 8), dtype=np.uint16)
    # 用半周期（512 码）作目标，避免 0↔1023 被引擎正确识别为相邻 1 步。
    target = np.full((8, 8), 512, dtype=np.uint16)
    plan = TransitionPlan(duration_s=2.0, fps=10)
    est = estimate_transition(plan, start, target)
    assert isinstance(est, TransitionEstimate)
    assert est.frames == plan.frame_count
    assert est.max_step_gray == pytest.approx(512 / (plan.frame_count - 1))
    # 首帧满幅翻转 200 ms 计入预计时长。
    assert est.expected_duration_s >= 0.2


def test_pixel_flip_wait_is_linear_and_capped() -> None:
    assert pixel_flip_wait_s(0.0) == 0.0
    # 满幅（1023 码）对应满翻转上限。
    assert pixel_flip_wait_s(MAX_GRAY_CODE) == pytest.approx(0.2)
    # 半幅约为上限的一半。
    assert pixel_flip_wait_s(MAX_GRAY_CODE / 2) == pytest.approx(0.1)


def test_estimate_flags_panel_limited_when_dwell_exceeds_interval() -> None:
    start = np.zeros((8, 8), dtype=np.uint16)
    target = np.full((8, 8), 512, dtype=np.uint16)
    # 用户设定的每步停留远超帧间隔 → 实际被停留拖慢，面板限速成立。
    plan = TransitionPlan(duration_s=0.5, fps=30, dwell_ms=500.0)
    est = estimate_transition(plan, start, target)
    assert est.limited_by_panel is True
    assert est.dwell_s == pytest.approx(0.5)
