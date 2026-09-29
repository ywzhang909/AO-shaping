"""Regression anchors for ``ShapingObjective`` and its weight helpers.

``ShapingObjective`` replaced the seven-branch objective-closure dispatch that
used to live inline in ``optimizer/wfless/slm_zernike_pib.py``. These tests pin
the extracted behavior:

* every objective family returns exactly what the underlying pure target
  function returns (no re-derivation, no re-ordering of arguments);
* the ``rms_pib`` weight state, its init resolution and the adaptive update;
* the in-ROI energy-loss safety guard (penalty sign, violation counter, the
  objectives it applies to) and the live bucket radius;
* the cross-objective ``m_*`` metric panel keys and values.

The frames are small and hand-constructible, so every assertion is exact. The
``ImageTargetFunc`` is only used as the injected bucket/radius helper - it is
never imported by ``targets.py`` itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.goal_functions.target_func import ImageTargetFunc
from ao_shaping.utils.image.targets import (
    ObjectiveResult,
    ShapeScoringParams,
    ShapingObjective,
    ShapingObjectiveParams,
    _resolve_init_weights,
    _update_dynamic_weights,
    rmse_shape_metric,
    rms_pib_terms,
    roi_energy_loss,
    roi_pib_metric,
    shape_metric,
)

CENTER = (10.0, 10.0)
SIZE = 6.0
ASPECT = 1.0
SHAPE = "circle"


def make_frame(seed: int = 0, spread: float = 2.0) -> np.ndarray:
    """A smooth non-negative blob centred on ``CENTER`` (uint16-like frame)."""
    yy, xx = np.mgrid[0:21, 0:21]
    cx, cy = CENTER
    img = 1000.0 * np.exp(
        -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * spread**2))
    )
    img += seed
    return img.astype(np.float64)


def make_params(objective: str, mode: str = "max", **overrides) -> ShapingObjectiveParams:
    params = {
        "objective": objective,
        "mode": mode,
        "shape": SHAPE,
        "size": SIZE,
        "aspect_ratio": ASPECT,
        "reference_center": CENTER,
        "max_roi_energy_loss": 0.0,
        "ideal_spot_radius": 3,
        "r_bucket": 2.0,
    }
    params.update(overrides)
    return ShapingObjectiveParams(**params)


def make_objective(objective: str, mode: str = "max", init_img=None, **overrides):
    init = make_frame() if init_img is None else init_img
    return ShapingObjective(make_params(objective, mode, **overrides), None, init)


class TestValidation:
    """Constructor rejects unusable configurations instead of failing later."""

    def test_unknown_objective(self) -> None:
        with pytest.raises(ValueError, match="objective must be one of"):
            make_objective("does_not_exist")

    def test_bad_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            make_objective("roi_pib", mode="sideways")

    @pytest.mark.parametrize("objective", ["pib", "radiu", "avg_radiu"])
    def test_target_func_backed_objective_requires_target_func(
        self, objective: str
    ) -> None:
        with pytest.raises(ValueError, match="needs a target_func"):
            make_objective(objective)

    def test_roi_only_objective_accepts_no_target_func(self) -> None:
        # roi_pib / rmse / shape / rms_pib are pure target-math objectives.
        for objective in ("roi_pib", "rmse", "shape", "rms_pib"):
            make_objective(objective)


class TestObjectiveFamilies:
    """``raw`` reproduces the pure target function of each family exactly."""

    def test_roi_pib_matches_roi_pib_metric(self) -> None:
        obj = make_objective("roi_pib")
        img = make_frame()
        expected = roi_pib_metric(img, CENTER, SHAPE, SIZE, ASPECT)
        j, ratio = obj.raw(img)
        assert j == pytest.approx(expected[0])
        assert ratio == pytest.approx(expected[1])

    def test_rmse_matches_rmse_shape_metric(self) -> None:
        obj = make_objective("rmse", mode="min")
        img = make_frame()
        expected = rmse_shape_metric(img, CENTER, SHAPE, SIZE, ASPECT)
        j, ratio = obj.raw(img)
        assert j == pytest.approx(expected[0])
        assert ratio == pytest.approx(expected[1])

    def test_shape_matches_shape_metric_without_schedule(self) -> None:
        scoring = ShapeScoringParams(
            w_uniformity=2.0, w_peak=1.0, w_displacement=0.0, log_uniformity=False
        )
        obj = make_objective("shape", scoring=scoring)
        img = make_frame()
        expected = shape_metric(
            img,
            CENTER,
            CENTER,
            SHAPE,
            SIZE,
            ASPECT,
            w_uniformity=2.0,
            w_peak=1.0,
            w_displacement=0.0,
            stage=None,
            log_uniformity=False,
        )
        j, ratio = obj.raw(img)
        assert j == pytest.approx(expected[0])
        assert ratio == pytest.approx(expected[1])

    def test_rms_pib_is_the_weighted_sum_of_its_three_terms(self) -> None:
        obj = make_objective("rms_pib", w_pib_init=0.5, w_rms_init=0.25)
        img = make_frame()
        # (0.5, 0.25, 0.25) - the missing ee weight takes the remaining mass.
        assert obj.weights == pytest.approx((0.5, 0.25, 0.25))
        pib_t, rms_t = rms_pib_terms(img, CENTER, SHAPE, SIZE, ASPECT)
        ee_t = pytest.approx(1.0)  # the frame IS the baseline
        j, ratio = obj.raw(img)
        assert ratio == pytest.approx(pib_t)
        assert j == pytest.approx(0.5 * pib_t + 0.25 * rms_t + 0.25 * 1.0)
        assert obj.terms == pytest.approx((j, pib_t, rms_t, 1.0))

    def test_rms_pib_ee_term_tracks_the_window_energy_baseline(self) -> None:
        init = make_frame()
        obj = ShapingObjective(make_params("rms_pib"), None, init)
        # Half the light still in the window -> ee_term == 0.5.
        res = obj(0.5 * init)
        assert res.terms[3] == pytest.approx(0.5)
        # No light left at all -> ee_term == 0.
        assert obj(np.zeros_like(init)).terms[3] == pytest.approx(0.0)

    def test_pib_family_uses_the_injected_target_func(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("pib", r_bucket=2.0), target_func, init)
        img = make_frame(1.0)
        expected = target_func.pib(img, 2.0)
        j, ratio = obj.raw(img)
        assert j == pytest.approx(expected[0])
        assert ratio == pytest.approx(expected[1])

    def test_radiu_family(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("radiu", mode="min"), target_func, init)
        img = make_frame(1.0)
        j, ratio = obj.raw(img)
        assert j == pytest.approx(float(target_func.radius(img, energy=0.99)))
        assert ratio == 0.0

    def test_avg_radiu_family(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("avg_radiu"), target_func, init)
        img = make_frame(1.0)
        expected = target_func.avg_radius(img, moment=1.0)
        j, ratio = obj.raw(img)
        assert j == pytest.approx(expected[0])
        assert ratio == pytest.approx(expected[1])


class TestLiveBucket:
    """``r_bucket`` must stay live - the search shrinks it mid-run."""

    def test_set_bucket_changes_what_pib_and_the_panel_read(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("pib", r_bucket=1.0), target_func, init)
        img = make_frame(1.0)
        at_one = obj.raw(img)[0]
        obj.set_bucket(3.0)
        at_three = obj.raw(img)[0]
        assert at_one != pytest.approx(at_three)
        assert at_three == pytest.approx(target_func.pib(img, 3.0)[0])
        assert obj.metric_panel(img)["m_pib"] == pytest.approx(
            target_func.pib(img, 3.0)[1]
        )


class TestTrackingValue:
    """``tracking_value`` is the "best" value the search tracks and logs."""

    def test_pib_uses_the_fixed_ideal_radius_ratio(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("pib", ideal_spot_radius=3), target_func, init)
        img = make_frame(1.0)
        res = obj(img)
        assert obj.tracking_value(img, res) == pytest.approx(
            target_func.pib(img, 3.0)[1]
        )

    @pytest.mark.parametrize("objective", ["roi_pib", "rmse", "shape", "rms_pib"])
    def test_non_pib_objectives_track_j(self, objective: str) -> None:
        obj = make_objective(objective)
        img = make_frame()
        res = obj(img)
        assert obj.tracking_value(img, res) == pytest.approx(res.j)


class TestEnergyGuard:
    """The in-ROI energy-loss guard abandons (strongly penalises) evaluations."""

    def test_disabled_by_default(self) -> None:
        obj = make_objective("roi_pib")
        assert obj(np.zeros_like(make_frame())).j == pytest.approx(
            obj.raw(np.zeros_like(make_frame()))[0]
        )
        assert obj.guard_violations == 0

    def test_max_mode_penalises_by_minus_1e3_and_keeps_the_true_ratio(self) -> None:
        init = make_frame()
        obj = ShapingObjective(
            make_params("roi_pib", max_roi_energy_loss=0.2), None, init
        )
        res = obj(np.zeros_like(init))  # all the light is gone
        true_j, true_ratio = obj.raw(np.zeros_like(init))
        assert res.j == pytest.approx(true_j - 1e3)
        assert res.ratio == pytest.approx(true_ratio)
        assert obj.guard_violations == 1

    def test_min_mode_penalises_by_plus_1e3(self) -> None:
        init = make_frame()
        obj = ShapingObjective(
            make_params("rmse", mode="min", max_roi_energy_loss=0.2), None, init
        )
        res = obj(np.zeros_like(init))
        assert res.j == pytest.approx(obj.raw(np.zeros_like(init))[0] + 1e3)

    def test_energy_within_the_limit_is_not_penalised(self) -> None:
        init = make_frame()
        obj = ShapingObjective(
            make_params("roi_pib", max_roi_energy_loss=0.5), None, init
        )
        # A half-energy frame loses ~50% of the in-ROI energy, right at the edge.
        res = obj(0.5 * init)
        assert res.j == pytest.approx(obj.raw(0.5 * init)[0])
        assert obj.guard_violations == 0

    def test_violation_counter_accumulates(self) -> None:
        init = make_frame()
        obj = ShapingObjective(
            make_params("roi_pib", max_roi_energy_loss=0.1), None, init
        )
        for _ in range(4):
            obj(np.zeros_like(init))
        assert obj.guard_violations == 4

    @pytest.mark.parametrize("objective", ["radiu", "avg_radiu"])
    def test_guard_does_not_apply_without_a_target_roi(
        self, objective: str
    ) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(
            make_params(objective, max_roi_energy_loss=0.2), target_func, init
        )
        obj(np.zeros_like(init))
        assert obj.guard_violations == 0

    def test_guard_reference_is_the_fixed_roi_of_the_initial_frame(self) -> None:
        init = make_frame()
        obj = ShapingObjective(
            make_params("roi_pib", max_roi_energy_loss=0.2), None, init
        )
        ref = obj._guard_ref_energy
        assert ref == pytest.approx(
            roi_pib_metric(init, CENTER, SHAPE, SIZE, ASPECT)[0]
        )
        assert roi_energy_loss(ref, obj._fixed_roi_energy(init)) == pytest.approx(0.0)


class TestMetricPanel:
    """The panel is recorded on EVERY epoch, so its keys must be stable."""

    EXPECTED_KEYS = {
        "m_shape",
        "m_energy",
        "m_rmse",
        "m_roi_pib",
        "m_pib",
        "m_pib7",
        "m_rms_pib",
        "m_rms_t",
        "m_ee",
        "m_brt",
    }

    def test_keys_are_exactly_the_ten_panel_columns(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("roi_pib", r_bucket=2.0), target_func, init)
        assert set(obj.metric_panel(init)) == self.EXPECTED_KEYS

    def test_values_match_the_underlying_functions(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("roi_pib", r_bucket=2.0), target_func, init)
        img = make_frame(1.0)
        panel = obj.metric_panel(img)

        shape_score, energy = shape_metric(
            img,
            CENTER,
            CENTER,
            SHAPE,
            SIZE,
            ASPECT,
            w_uniformity=3.0,
            w_peak=0.5,
            w_displacement=0.5,
            stage=None,
            log_uniformity=True,
        )
        pib_t, rms_t = rms_pib_terms(img, CENTER, SHAPE, SIZE, ASPECT)
        rmse, _ = rmse_shape_metric(img, CENTER, SHAPE, SIZE, ASPECT)
        roi_score, _ = roi_pib_metric(img, CENTER, SHAPE, SIZE, ASPECT)
        frame_sum = float(np.asarray(img, dtype=np.float64).sum())
        ee = min(max(frame_sum / float(np.sum(init)), 0.0), 1.0)

        assert panel["m_shape"] == pytest.approx(shape_score)
        assert panel["m_energy"] == pytest.approx(energy)
        assert panel["m_rmse"] == pytest.approx(rmse)
        assert panel["m_roi_pib"] == pytest.approx(roi_score)
        assert panel["m_pib"] == pytest.approx(target_func.pib(img, 2.0)[1])
        assert panel["m_pib7"] == pytest.approx(target_func.pib(img, 3.0)[1])
        assert panel["m_rms_pib"] == pytest.approx((pib_t + rms_t + ee) / 3.0)
        assert panel["m_rms_t"] == pytest.approx(rms_t)
        assert panel["m_ee"] == pytest.approx(ee)
        assert panel["m_brt"] == pytest.approx(float(np.max(img)))

    def test_ee_of_the_initial_frame_is_one(self) -> None:
        init = make_frame()
        target_func = ImageTargetFunc.build_from_init_image(init)
        obj = ShapingObjective(make_params("roi_pib"), target_func, init)
        assert obj.metric_panel(init)["m_ee"] == pytest.approx(1.0)

    def test_panel_keeps_all_ten_keys_without_a_target_func(self) -> None:
        # A ROI-only objective may be built without a ``target_func``; the panel
        # must still expose the same recorder schema (NaN bucket columns).
        init = make_frame()
        obj = ShapingObjective(make_params("roi_pib"), None, init)
        panel = obj.metric_panel(init)
        assert set(panel) == self.EXPECTED_KEYS
        assert np.isnan(panel["m_pib"])
        assert np.isnan(panel["m_pib7"])
        assert not np.isnan(panel["m_shape"])
        assert not np.isnan(panel["m_ee"])


class TestAdaptWeights:
    """``adapt_weights`` must consume the PASSED result, not the last terms."""

    def test_returns_empty_for_non_rms_pib(self) -> None:
        obj = make_objective("roi_pib")
        res = obj(make_frame())
        assert obj.adapt_weights(res) == ()

    def test_uses_the_passed_result_not_the_latest_evaluation(self) -> None:
        obj = make_objective("rms_pib")
        # Hold a "positive perturbation" result, then evaluate a much worse frame
        # (as the SPGD negative evaluation does) before adapting.
        held = obj(make_frame(1.0))
        obj(make_frame(500.0))
        adapted_from_held = obj.adapt_weights(held)
        assert len(adapted_from_held) == 3

        # Adapting from the same held snapshot twice must give the same answer
        # as adapting from it once - i.e. the intervening evaluation did not
        # change what was adapted.
        assert obj.weights == pytest.approx(adapted_from_held)

    def test_weights_stay_normalised_and_respect_the_floor(self) -> None:
        obj = make_objective("rms_pib", w_floor=0.1)
        for seed in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            obj.adapt_weights(obj(make_frame(seed)))
        w_pib, w_rms, w_ee = obj.weights
        assert w_pib + w_rms + w_ee == pytest.approx(1.0)
        assert min(w_pib, w_rms, w_ee) >= 0.1 - 1e-12

    def test_first_call_only_records_the_baseline(self) -> None:
        obj = make_objective("rms_pib")
        before = obj.weights
        obj.adapt_weights(obj(make_frame(1.0)))
        assert obj.weights == pytest.approx(before)


class TestResolveInitWeights:
    """``_resolve_init_weights`` keeps the triple summing to 1."""

    def test_all_none_gives_thirds(self) -> None:
        assert _resolve_init_weights(None, None, None) == pytest.approx(
            (1 / 3, 1 / 3, 1 / 3)
        )

    def test_partial_keeps_given_and_splits_the_rest_equally(self) -> None:
        assert _resolve_init_weights(0.6, None, None) == pytest.approx(
            (0.6, 0.2, 0.2)
        )
        assert _resolve_init_weights(None, 0.5, None) == pytest.approx(
            (0.25, 0.5, 0.25)
        )

    def test_all_given_is_normalised(self) -> None:
        assert _resolve_init_weights(1.0, 1.0, 2.0) == pytest.approx(
            (0.25, 0.25, 0.5)
        )

    def test_partial_summing_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to <= 1"):
            _resolve_init_weights(0.7, 0.7, None)

    def test_all_given_summing_to_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 0"):
            _resolve_init_weights(0.0, 0.0, 0.0)


class TestUpdateDynamicWeights:
    """The adaptive update keeps the invariants the ``rms_pib`` loop relies on."""

    def test_first_call_is_a_no_op(self) -> None:
        state: dict = {}
        assert _update_dynamic_weights(
            state, pib=0.1, rms=0.2, ee=0.9, j=0.4
        ) == pytest.approx((1 / 3, 1 / 3, 1 / 3))

    def test_no_improving_step_keeps_the_weights(self) -> None:
        state: dict = {"w_pib": 0.5, "w_rms": 0.3, "w_ee": 0.2}
        _update_dynamic_weights(state, pib=0.10, rms=0.20, ee=0.90, j=0.40)  # baseline
        out = _update_dynamic_weights(
            state, pib=0.05, rms=0.10, ee=0.50, j=0.20
        )  # everything got worse
        assert out == pytest.approx((0.5, 0.3, 0.2))

    def test_two_term_mode_still_returns_two_weights_summing_to_one(self) -> None:
        state: dict = {}
        first = _update_dynamic_weights(state, pib=0.1, rms=0.2, j=0.3)
        assert len(first) == 2
        _update_dynamic_weights(state, pib=0.5, rms=0.2, j=0.7)
        out = _update_dynamic_weights(state, pib=0.9, rms=0.2, j=1.1)
        assert len(out) == 2
        assert sum(out) == pytest.approx(1.0)

    def test_the_term_that_improves_j_more_gets_the_higher_weight(self) -> None:
        state: dict = {}
        _update_dynamic_weights(state, pib=0.10, rms=0.20, ee=0.90, j=0.40)
        _update_dynamic_weights(state, pib=0.10, rms=0.80, ee=0.90, j=0.60)
        _update_dynamic_weights(state, pib=0.10, rms=0.99, ee=0.90, j=0.63)
        w_pib, w_rms, w_ee = _update_dynamic_weights(
            state, pib=0.10, rms=0.999, ee=0.90, j=0.633
        )
        assert w_rms > w_pib
        assert w_rms > w_ee


class TestObjectiveResult:
    """The result is a frozen snapshot so a held result cannot be overwritten."""

    def test_terms_default_to_zero(self) -> None:
        res = ObjectiveResult(1.0, 0.5)
        assert res.terms == (0.0, 0.0, 0.0, 0.0)

    def test_is_frozen(self) -> None:
        res = ObjectiveResult(1.0, 0.5)
        with pytest.raises(Exception):
            res.j = 2.0  # type: ignore[misc]
