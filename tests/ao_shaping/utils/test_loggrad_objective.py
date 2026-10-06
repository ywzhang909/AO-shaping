"""Integration tests for the ``w_loggrad`` structure modifier on ShapingObjective.

The term is a LOSS (lower is better). ``_raw_base`` returns NATIVE polarity, so
the modifier is combined as ``j + sign * w * lg``: SUBTRACTED in ``max`` mode,
ADDED in ``min`` mode. In both cases a structurally WORSE frame must move ``j``
in the direction the optimiser is driving, which is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.goal_functions.target_func import ImageTargetFunc
from ao_shaping.utils.image.target.objective import (
    GUARD_PENALTY,
    SHAPING_OBJECTIVE_CHOICES,
    ShapeScoringParams,
    ShapingObjective,
    ShapingObjectiveParams,
)

H = W = 128
CENTER = (W / 2.0, H / 2.0)
_YY, _XX = np.mgrid[0:H, 0:W]
_D2 = (_XX - CENTER[0]) ** 2 + (_YY - CENTER[1]) ** 2
SIGMA = 10.0


def clean_blob() -> np.ndarray:
    """The bench's natural spot profile - used as the structural reference."""
    return 80.0 * np.exp(-_D2 / (2.0 * SIGMA**2))


def speckled_blob(amp: float = 0.9, seed: int = 11) -> np.ndarray:
    """Same envelope, extra fine structure: must score strictly worse."""
    rng = np.random.default_rng(seed)
    base = clean_blob()
    return np.clip(base * (1.0 + amp * rng.normal(size=base.shape)), 0.0, None)


def make_objective(
    *,
    objective: str = "shape",
    mode: str = "max",
    w_loggrad: float = 0.0,
    max_roi_energy_loss: float = 0.0,
    init_img: np.ndarray | None = None,
) -> ShapingObjective:
    params = ShapingObjectiveParams(
        objective=objective,
        mode=mode,
        shape="circle",
        size=40.0,
        aspect_ratio=1.0,
        reference_center=CENTER,
        max_roi_energy_loss=max_roi_energy_loss,
        scoring=ShapeScoringParams(),
        w_loggrad=w_loggrad,
    )
    seed = clean_blob() if init_img is None else init_img
    return ShapingObjective(params, ImageTargetFunc(W, H, CENTER), seed)


class TestDefaultOff:
    def test_raw_equals_base_exactly(self) -> None:
        """With the feature off, ``raw`` must be ``_raw_base`` byte for byte."""
        obj = make_objective(w_loggrad=0.0)
        for img in (clean_blob(), speckled_blob()):
            assert obj.raw(img) == obj._raw_base(img)

    @pytest.mark.parametrize("name", SHAPING_OBJECTIVE_CHOICES)
    def test_metric_is_never_called_when_off(self, name, monkeypatch) -> None:
        """Belt and braces: the metric must not even be reached."""
        import ao_shaping.utils.image.target.objective as mod

        def _boom(*_a, **_k):
            raise AssertionError("metric called while w_loggrad == 0")

        monkeypatch.setattr(mod, "log_gradient_difference_metric", _boom)
        obj = make_objective(objective=name, w_loggrad=0.0)
        assert np.isfinite(obj(clean_blob()).j)
        assert obj(clean_blob()).loggrad == 0.0

    def test_reference_not_retained_when_off(self) -> None:
        """The default path must not pay the memory cost."""
        assert make_objective(w_loggrad=0.0)._loggrad_reference is None
        assert make_objective(w_loggrad=0.5)._loggrad_reference is not None


class TestPolarity:
    def test_matching_profile_scores_zero(self) -> None:
        """A frame identical to the reference has no excess structure."""
        obj = make_objective(w_loggrad=1.0)
        assert obj(clean_blob()).loggrad == pytest.approx(0.0, abs=1e-9)

    def test_speckle_scores_worse_than_clean(self) -> None:
        obj = make_objective(w_loggrad=1.0)
        assert obj(speckled_blob()).loggrad > obj(clean_blob()).loggrad

    def test_max_mode_lowers_reward_for_worse_frame(self) -> None:
        """max-mode: ``j`` is a reward, so a worse frame must LOWER it."""
        on = make_objective(mode="max", w_loggrad=1.0)
        off = make_objective(mode="max", w_loggrad=0.0)
        bad = speckled_blob()
        assert on(bad).j < off(bad).j

    def test_min_mode_raises_cost_for_worse_frame(self) -> None:
        """min-mode: ``j`` is a cost, so a worse frame must RAISE it."""
        on = make_objective(mode="min", w_loggrad=1.0)
        off = make_objective(mode="min", w_loggrad=0.0)
        bad = speckled_blob()
        assert on(bad).j > off(bad).j

    def test_raw_applies_the_same_polarity_as_call(self) -> None:
        """``raw`` carries the modifier too - pin it separately.

        Mutation check: dropping ``sign`` from ``raw`` alone leaves every
        ``__call__``-based test green, so this assertion is the only thing that
        catches that half of the bug.
        """
        bad = speckled_blob()
        base = make_objective(mode="max", w_loggrad=0.0)
        on = make_objective(mode="max", w_loggrad=1.0)
        assert on.raw(bad)[0] < base.raw(bad)[0], "max-mode raw() must lower j"
        on_min = make_objective(mode="min", w_loggrad=1.0)
        base_min = make_objective(mode="min", w_loggrad=0.0)
        assert on_min.raw(bad)[0] > base_min.raw(bad)[0], "min-mode raw() must raise j"

    def test_weight_scales_the_shift_linearly(self) -> None:
        """Doubling the weight must double the shift, not leave it unchanged."""
        bad = speckled_blob()
        base = make_objective(mode="max", w_loggrad=0.0)(bad).j
        one = make_objective(mode="max", w_loggrad=1.0)(bad).j - base
        two = make_objective(mode="max", w_loggrad=2.0)(bad).j - base
        assert one != 0.0
        assert two == pytest.approx(2.0 * one, rel=1e-9)


class TestGuardDominance:
    def test_guard_bypasses_modifier_even_with_huge_weight(self) -> None:
        """``GUARD_PENALTY`` must win for ANY weight, not just small ones."""
        seed = clean_blob()
        obj = make_objective(w_loggrad=1e9, max_roi_energy_loss=0.1, init_img=seed)
        # Kill the in-ROI energy so the guard certainly fires.
        starved = np.full_like(seed, 0.01)
        result = obj(starved)
        assert result.j <= -GUARD_PENALTY + 1.0
        assert result.loggrad == 0.0, "modifier must be skipped on the guard path"

    def test_guard_disabled_still_applies_modifier(self) -> None:
        on = make_objective(mode="max", w_loggrad=1.0, max_roi_energy_loss=0.0)
        off = make_objective(mode="max", w_loggrad=0.0, max_roi_energy_loss=0.0)
        assert on(speckled_blob()).j != off(speckled_blob()).j

    def test_penalty_hits_both_j_and_tracking(self) -> None:
        """Regression: penalty on ``j`` alone let the exit path write a
        guard-forbidden phase back to the SLM."""
        seed = clean_blob()
        obj = make_objective(w_loggrad=0.0, max_roi_energy_loss=0.1, init_img=seed)
        res = obj(np.full_like(seed, 0.01))
        assert res.j <= -GUARD_PENALTY + 1.0
        assert res.tracking <= -GUARD_PENALTY + 1.0


class TestRobustness:
    def test_shape_mismatch_degrades_gracfully(self) -> None:
        """A re-armed window changes the frame geometry; must not raise."""
        obj = make_objective(w_loggrad=1.0)
        res = obj(clean_blob()[:64, :64])
        assert np.isfinite(res.j)
        assert res.loggrad == 0.0

    def test_reference_frame_with_nan_does_not_break(self) -> None:
        seed = clean_blob()
        seed = seed.copy()
        seed[5, 5] = np.nan
        obj = make_objective(w_loggrad=1.0, init_img=seed)
        assert np.isfinite(obj(speckled_blob()).loggrad)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_invalid_weight_rejected(self, bad) -> None:
        with pytest.raises(ValueError):
            ShapingObjectiveParams(
                objective="shape",
                mode="max",
                shape="circle",
                size=40.0,
                aspect_ratio=1.0,
                reference_center=CENTER,
                scoring=ShapeScoringParams(),
                w_loggrad=bad,
            )


class TestTelemetry:
    def test_metric_panel_key_set_is_unchanged(self) -> None:
        """The panel's key set is a pinned recorder-schema contract.

        ``test_shaping_objective.py`` asserts ``set(metric_panel(...))`` equals a
        fixed ``EXPECTED_KEYS``, and ``metric_panel`` documents that it keeps the
        exact same ten keys. So the term is reported through
        ``ObjectiveResult.loggrad`` instead, never as a new panel column.
        """
        panel = make_objective(w_loggrad=1.0).metric_panel(speckled_blob())
        assert "loggrad" not in panel
        assert len(panel) == 10

    def test_result_carries_loggrad(self) -> None:
        on = make_objective(w_loggrad=1.0)(speckled_blob())
        off = make_objective(w_loggrad=0.0)(speckled_blob())
        assert on.loggrad > 0.0
        assert off.loggrad == 0.0