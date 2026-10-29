"""逐块 Jacobian 方向校验与信任域调度 (免设备)。"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.jacobian_guard import (
    Agreement,
    TrustRegionSchedule,
    direction_agreement,
)


def _frames(seed=0, size=48):
    rng = np.random.default_rng(seed)
    # A smooth-ish pedestal plus a structure both ends will share.
    yy, xx = np.mgrid[:size, :size]
    base = 1.0 + 0.2 * np.cos(2 * np.pi * xx / size)
    return base + 0.001 * rng.normal(size=(size, size))


def test_identical_step_direction_agrees():
    anchor = _frames(1)
    yy, xx = np.mgrid[:48, :48]
    step = 0.05 * np.cos(2 * np.pi * (xx + yy) / 12)
    agreement = direction_agreement(anchor, anchor + step, anchor, anchor + step)
    assert agreement.verdict == "agree"
    assert agreement.correlation == pytest.approx(1.0, abs=1e-6)
    assert agreement.relative_l2 == pytest.approx(0.0, abs=1e-6)


def test_opposite_step_direction_contradicts():
    anchor = _frames(2)
    yy, xx = np.mgrid[:48, :48]
    step = 0.05 * np.cos(2 * np.pi * (xx + yy) / 12)
    agreement = direction_agreement(anchor, anchor + step, anchor, anchor - step)
    assert agreement.verdict == "contradict"
    assert agreement.correlation == pytest.approx(-1.0, abs=1e-6)


def test_uncorrelated_step_is_weak():
    anchor = _frames(3)
    rng = np.random.default_rng(99)
    model_step = rng.normal(size=(48, 48))
    measured_step = rng.normal(size=(48, 48))
    agreement = direction_agreement(anchor, anchor + 0.01 * model_step,
                                    anchor, anchor + 0.01 * measured_step)
    assert agreement.verdict in {"weak", "contradict"}
    assert abs(agreement.correlation) < 0.5


def test_silent_bench_is_insufficient_signal():
    anchor = _frames(4)
    agreement = direction_agreement(anchor, anchor + 0.05, anchor, anchor)
    assert agreement.verdict == "insufficient_signal"
    assert np.isnan(agreement.correlation)


def test_scale_free_relative_l2_ignores_gain():
    anchor = _frames(5)
    yy, xx = np.mgrid[:48, :48]
    step = 0.02 * np.cos(2 * np.pi * xx / 10)
    small = direction_agreement(anchor, anchor + step, anchor, anchor + step)
    # Same direction, measured at 4x the model's magnitude.
    large = direction_agreement(anchor, anchor + step, anchor, anchor + 4 * step)
    assert small.verdict == large.verdict == "agree"
    assert large.relative_l2 == pytest.approx(small.relative_l2, abs=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        direction_agreement(np.zeros((4, 4)), np.zeros((4, 4)),
                            np.zeros((4, 4)), np.zeros((4, 5)))


def test_schedule_shrinks_on_contradiction_and_has_a_floor():
    schedule = TrustRegionSchedule(cap=0.2, floor=0.05)
    contradicted = Agreement(0.1, 1.0, 1.0, 1.0, "contradict")
    assert schedule.observe(contradicted) == pytest.approx(0.1)
    assert schedule.observe(contradicted) == pytest.approx(0.05)
    assert schedule.observe(contradicted) == pytest.approx(0.05)


def test_schedule_grows_on_agreement_but_never_above_the_recipe():
    schedule = TrustRegionSchedule(cap=0.2, grow=1.25)
    agreed = Agreement(0.95, 0.0, 1.0, 1.0, "agree")
    schedule.observe(Agreement(0.1, 1.0, 1.0, 1.0, "contradict"))
    assert schedule.cap == pytest.approx(0.1)
    for _ in range(10):
        schedule.observe(agreed)
    assert schedule.cap == pytest.approx(0.2)


def test_schedule_ignores_weak_and_insufficient_evidence():
    schedule = TrustRegionSchedule(cap=0.2)
    start = schedule.cap
    schedule.observe(Agreement(0.6, 0.5, 1.0, 1.0, "weak"))
    schedule.observe(Agreement(float("nan"), float("nan"), 1.0, 0.0, "insufficient_signal"))
    assert schedule.cap == pytest.approx(start)


def test_schedule_validates_its_bounds():
    with pytest.raises(ValueError):
        TrustRegionSchedule(cap=0.0)
    with pytest.raises(ValueError):
        TrustRegionSchedule(cap=0.2, floor=0.3)
    with pytest.raises(ValueError):
        TrustRegionSchedule(cap=0.2, shrink=1.0)
    with pytest.raises(ValueError):
        TrustRegionSchedule(cap=0.2, grow=1.0)
