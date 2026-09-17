"""Offline tests for the ``slm_zernike_pib`` dynamic SPGD schedule.

Locks the safety caps. The schedule used to return base lr=6 / delta=5 against
the +/-5 Zernike coefficient clip used by the SPGD loop, so every step saturated
the clip and the optimizer diverged (hardware-observed: bucket PIB 0.70 -> 0.08,
peak brightness 199 -> 15). A single step must stay well inside the clip range.

No hardware is opened and no hardware behaviour is asserted here; the module
import degrades gracefully when the SDKs are absent.
"""

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SCHEDULE_MAX_DELTA,
    SCHEDULE_MAX_LR,
    learning_schedule,
)


class TestScheduleSafetyCaps:
    def test_extreme_radius_is_capped(self):
        lr, delta = learning_schedule(power_radius=1e6)
        assert lr <= SCHEDULE_MAX_LR
        assert delta <= SCHEDULE_MAX_DELTA

    def test_no_history_early_return_is_capped(self):
        """epoch-0 call passes empty history -> early-return branch."""
        lr, delta = learning_schedule(
            power_radius=1e6, gradient_history=[], pib_history=[], epoch=0
        )
        assert lr <= SCHEDULE_MAX_LR
        assert delta <= SCHEDULE_MAX_DELTA

    def test_warmup_and_diverging_history_stay_capped(self):
        grads = [1.0] * 10
        pibs = [1.0 - 0.01 * i for i in range(10)]
        lr, delta = learning_schedule(
            power_radius=1e6, gradient_history=grads, pib_history=pibs, epoch=0
        )
        assert lr <= SCHEDULE_MAX_LR
        assert delta <= SCHEDULE_MAX_DELTA

    def test_caps_are_a_fraction_of_the_clip_range(self):
        # Coefficient clip is +/-5 (span 10). A perturbation equal to the whole
        # span makes the finite difference saturate; keep it well below.
        assert SCHEDULE_MAX_DELTA < 5.0
        assert SCHEDULE_MAX_LR < 5.0

    def test_small_radius_is_positive(self):
        lr, delta = learning_schedule(power_radius=1.0)
        assert lr > 0
        assert delta > 0
