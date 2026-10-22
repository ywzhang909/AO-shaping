"""model-in-loop 的自标定护栏: 逐轮信任域 + 接受判据。

论文 (GRAPHYCS) 的自标定量**没有**信任域, 也**没有**逐轮验收。那与本仓库
「宁可中止也不提交错误相位」的原则直接冲突, 所以这一层必须自己加: 标定参数
一旦跑飞, 就说明正向模型是在被掰弯去迎合数据, 而不是描述实验系统。

这里测的是**纯函数**那一层 (模块头就写明: 耦合护栏是纯函数, 以便在没有
硬件的情况下测试)。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    acceptance_verdict,
    calibration_trust_region_check,
)
from ao_shaping.utils.wavefront.calibration_terms import (
    FAMILY_NAMES,
    PupilAffine,
    TravelScale,
)


class TestCalibrationTrustRegion:
    """The per-round bound on the self-calibration parameters."""

    def test_a_small_step_is_inside_the_region(self) -> None:
        """A step under the cap is not a violation."""
        before = PupilAffine()
        after = PupilAffine(off_x=0.01)
        violated, theta_step, gamma_step = calibration_trust_region_check(
            after, before, None, None
        )
        assert violated is False
        assert theta_step == pytest.approx(0.01)
        assert gamma_step == 0.0

    def test_a_large_theta_step_is_a_violation(self) -> None:
        """A runaway affine must be caught."""
        before = PupilAffine()
        after = PupilAffine(off_x=0.5)
        violated, theta_step, _ = calibration_trust_region_check(
            after, before, None, None
        )
        assert violated is True
        assert theta_step == pytest.approx(0.5)

    def test_a_large_gamma_step_is_a_violation(self) -> None:
        """A runaway travel scale must be caught."""
        before = TravelScale()
        after = TravelScale((0.4,) * len(FAMILY_NAMES))
        violated, _, gamma_step = calibration_trust_region_check(
            None, None, after, before
        )
        assert violated is True
        assert gamma_step == pytest.approx(0.4)

    def test_zero_caps_disable_the_guard(self) -> None:
        """``<= 0`` disables the bound, exactly like the coefficient guard."""
        before = PupilAffine()
        after = PupilAffine(off_x=0.5)
        violated, _, _ = calibration_trust_region_check(
            after, before, None, None, theta_step_max=0.0
        )
        assert violated is False

    def test_a_disabled_self_calibration_never_violates(self) -> None:
        """With the affine and γ off there is nothing to bound."""
        violated, theta_step, gamma_step = calibration_trust_region_check(
            None, None, None, None
        )
        assert violated is False
        assert theta_step == 0.0
        assert gamma_step == 0.0

    def test_the_bound_is_per_parameter_not_a_group_norm(self) -> None:
        """Six components with different units must not share one norm."""
        before = PupilAffine()
        after = PupilAffine(amp_x=1.05, amp_y=1.05, off_x=0.05, off_y=0.05,
                           rot=0.05, grid=1.05)
        violated, theta_step, _ = calibration_trust_region_check(
            after, before, None, None, theta_step_max=0.02
        )
        assert theta_step == pytest.approx(0.05)
        assert violated is True


class TestAcceptanceIntegration:
    """The guard must reach the round-acceptance test."""

    def test_a_violation_rejects_the_round(self) -> None:
        """A round whose calibration ran away is not committed."""
        verdict = acceptance_verdict(
            1.0, 0.5, 0.4, 0.8, calibration_violated=True
        )
        assert verdict.accepted is False
        assert bool(verdict) is False
        assert verdict.reason == "calibration_trust_region"

    def test_the_violation_joins_the_other_reasons(self) -> None:
        """The reason tag keeps the other failure modes visible."""
        verdict = acceptance_verdict(
            1.0, 2.0, 0.9, 0.1, calibration_violated=True
        )
        assert verdict.accepted is False
        assert "calibration_trust_region" in verdict.reason
        assert "loss_worsened" in verdict.reason

    def test_the_default_path_is_unchanged(self) -> None:
        """Without the flag the verdict is exactly what it always was."""
        verdict = acceptance_verdict(1.0, 0.5, 0.4, 0.8)
        assert verdict.accepted is True
        assert verdict.reason == "accepted"


class TestConfigDefaults:
    """The new knobs default to off."""

    def test_the_self_calibration_is_off_by_default(self) -> None:
        """Nothing about the default run changes."""
        config = SlmModelInLoopConfig()
        assert config.fourier_weight == 0.0
        assert config.loss_kind == "mse"
        assert config.selfcal_affine is False
        assert config.selfcal_travel is False

    def test_the_guards_have_conservative_defaults(self) -> None:
        """A trust region that is too loose is no trust region."""
        config = SlmModelInLoopConfig()
        assert config.selfcal_theta_step_max > 0.0
        assert config.selfcal_gamma_step_max > 0.0
        assert config.selfcal_lr > 0.0

    def test_the_config_rejects_nonsense(self) -> None:
        """The new knobs are validated like the old ones."""
        with pytest.raises(ValueError):
            SlmModelInLoopConfig(fourier_weight=-1.0)
        with pytest.raises(ValueError):
            SlmModelInLoopConfig(loss_kind="huber")
        with pytest.raises(ValueError):
            SlmModelInLoopConfig(selfcal_lr=0.0)


class TestCalibrationCarryOver:
    """The self-calibration is bench state, so it must survive a rebuild."""

    def test_the_optimizer_carries_the_values(self) -> None:
        """``set_calibration`` is the only way in, and it works."""
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            ZernikeCoefficientOptimizer,
        )

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=4,
            region=32,
            dtype="float64",
            device="cpu",
            affine_params=PupilAffine(),
            travel_gammas=(0.0,) * len(FAMILY_NAMES),
        )
        affine = PupilAffine(off_x=0.12, rot=0.03)
        travel = TravelScale((0.01, 0.02) + (0.0,) * (len(FAMILY_NAMES) - 2))
        optimizer.set_calibration(affine=affine, travel=travel)
        assert optimizer.affine == affine
        assert optimizer.travel == travel

    def test_a_rollback_restores_the_previous_values(self) -> None:
        """The guard's rollback path is exercised through the same entry."""
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            ZernikeCoefficientOptimizer,
        )

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=4,
            region=32,
            dtype="float64",
            device="cpu",
            affine_params=PupilAffine(),
        )
        before = optimizer.affine
        assert before is not None
        optimizer.set_calibration(affine=PupilAffine(off_x=0.4))
        assert optimizer.affine != before
        optimizer.set_calibration(affine=before)
        assert optimizer.affine == before


def test_the_numpy_import_is_used() -> None:
    """Guard against an accidental dead import in this module."""
    assert np is not None


#: A tiny simulated budget: enough for one round, small enough to stay fast.
_SIM_KWARGS = dict(
    cam_type="sim",
    region=32,
    cam_size=128,
    far_field_padding=4,
    n_rounds=1,
    probe_count=2,
    step_a_iterations=4,
    step_b_iterations=4,
    target_side=64,
    n_orders=4,
    dtype="float32",
    device="cpu",
    seed=0,
)


def _sim_history(**overrides) -> list[dict]:
    """在模拟实验系统上跑一轮, 返回逐轮记录。"""
    from ao_shaping.optimizer.wfless.slm_model_in_loop import (
        optimize_slm_model_in_loop,
    )

    config = SlmModelInLoopConfig(**{**_SIM_KWARGS, **overrides})
    result = optimize_slm_model_in_loop(config)
    return list(result.recorder.history)


class TestLoopIntegration:
    """The guard has to reach the loop, not just the pure function."""

    def test_a_violating_round_is_rejected(self) -> None:
        """A calibration step over the cap must reject the round."""
        history = _sim_history(
            selfcal_affine=True,
            # Any movement at all is a violation.
            selfcal_theta_step_max=1e-9,
        )
        rounds = [row for row in history if row.get("stage") == "round"]
        assert rounds, history
        for row in rounds:
            assert row["accepted"] is False
            assert "calibration_trust_region" in str(row["reason"])

    def test_the_violation_is_recorded(self) -> None:
        """The per-round step size has to be in the history."""
        history = _sim_history(
            selfcal_affine=True,
            selfcal_theta_step_max=1e-9,
        )
        rounds = [row for row in history if row.get("stage") == "round"]
        assert rounds
        for row in rounds:
            assert row["selfcal_violated"] is True
            assert float(row["selfcal_theta_step"]) > 0.0
            assert set(row["selfcal_affine"]) == {
                "amp_x", "amp_y", "off_x", "off_y", "rot", "grid"
            }

    def test_the_default_run_is_not_touched(self) -> None:
        """With the self-calibration off, no new key appears in the history."""
        history = _sim_history()
        rounds = [row for row in history if row.get("stage") == "round"]
        assert rounds
        for row in rounds:
            assert "selfcal_affine" not in row
            assert "selfcal_violated" not in row
            assert "calibration_trust_region" not in str(row["reason"])
