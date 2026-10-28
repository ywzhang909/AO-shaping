"""Tests for the campaign-level measurement protocol (device-free, NumPy only).

Every assertion here pins a defect the 2026-10-28 multishape review actually
found in the archived data, so a future edit that re-introduces it fails loudly
instead of silently producing another unsound comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import protocol as protocol_module
from ao_shaping.tools.slm.transfer.protocol import (
    FixedAperture,
    aperture_metrics,
    compare_repeats,
    insufficient_keys,
    judge_pair,
    plan_reference_blocks,
    repeat_statistics,
    replay_audit,
    shape_agreement,
    verdict_with_uncertainty,
)


def _aperture(**overrides) -> FixedAperture:
    fields = dict(centre_xy=(100.0, 100.0), half_side=40.0)
    fields.update(overrides)
    return FixedAperture(**fields)


def _flat_box(core_value: float = 100.0, total_extra: float = 0.0) -> np.ndarray:
    """A 200x200 frame with a uniform 81x81 target box, plus optional outside energy.

    The box extent matches :meth:`FixedAperture.box_mask`'s inclusive bound, so the
    core (a subset of it) is uniform and its CV is exactly zero.
    """
    frame = np.zeros((200, 200), dtype=float)
    frame[60:141, 60:141] = core_value
    if total_extra:
        frame[0:20, 0:20] = total_extra
    return frame





# -- FixedAperture ---------------------------------------------------------

def test_box_mask_is_independent_of_the_camera_window_step():
    aperture = _aperture()
    box = aperture.box_mask((200, 200))
    # The aperture lives in full-frame sensor pixels, so the mask is the same
    # geometry whatever sampling a stage used.
    assert box.sum() == 81 * 81
    assert box[100, 100] and not box[100, 58]




def test_core_mask_erodes_and_removes_the_zero_order_disk():
    aperture = _aperture(core_margin_px=4.0, zero_xy=(100.0, 100.0), zero_radius=6.0)
    core = aperture.core_mask((200, 200))
    assert not core[100, 100], "zero-order disk must be cut out of the core"
    assert not core[100, 62], "core must be eroded by core_margin_px"
    assert core[75, 75]


def test_missing_zero_centre_is_rejected():
    with pytest.raises(ValueError, match="zero_xy"):
        _aperture(zero_radius=5.0)


# -- aperture_metrics ------------------------------------------------------

def test_flat_core_reports_near_zero_cv_and_full_relative_energy():
    metrics = aperture_metrics(_flat_box(100.0), _aperture(full_scale=255.0))
    assert metrics["cv"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["ee_rel"] == pytest.approx(1.0)
    assert metrics["zero_excluded_fraction"] == pytest.approx(0.0)
    assert metrics["saturated_fraction"] == 0.0


def test_empty_core_reports_none_not_perfect_uniformity():
    # An emptied target is the failure mode CV alone cannot see, so "no signal"
    # must not read as "perfectly flat".
    metrics = aperture_metrics(np.zeros((200, 200)), _aperture())
    assert metrics["cv"] is None
    assert metrics["ee_rel"] is None
    assert metrics["mean"] == 0.0


def test_relative_energy_separates_a_dim_core_from_a_lost_core():
    aperture = _aperture()
    dim = aperture_metrics(_flat_box(100.0), aperture)
    leaking = aperture_metrics(_flat_box(100.0, total_extra=2000.0), aperture)
    # Same core shape -> same CV; only relative energy says the light left.
    assert leaking["cv"] == pytest.approx(dim["cv"])
    assert leaking["ee_rel"] < dim["ee_rel"] - 0.3



def test_zero_excluded_fraction_is_reported_so_the_auditor_sees_it():
    # The s200 target box overlaps the zero-order disk by 5.61 %; a number with
    # that much of the box removed must say so.
    narrow = _aperture(core_margin_px=2.0, zero_xy=(100.0, 100.0), zero_radius=20.0)
    assert aperture_metrics(_flat_box(), narrow)["zero_excluded_fraction"] > 0.2


def test_unknown_full_scale_leaves_saturation_unknown():
    assert aperture_metrics(_flat_box(), _aperture())["saturated_fraction"] is None


def test_saturated_pixels_are_counted():
    # An 11x11 clipped patch inside the box: 121 of the 40000 full-frame pixels.
    # (The fraction is of the whole frame, not the box, so a later re-window of the
    # same frame cannot silently change it.)
    frame = _flat_box(100.0)
    frame[70:81, 70:81] = 255.0
    metrics = aperture_metrics(frame, _aperture(full_scale=255.0))
    assert metrics["saturated_fraction"] == pytest.approx(121 / 40000)



def test_non_monochrome_input_is_rejected():
    with pytest.raises(ValueError, match="monochrome"):
        aperture_metrics(np.zeros((4, 4, 3)), _aperture())


# -- repeat_statistics -----------------------------------------------------

def test_repeat_spread_uses_student_t_not_normal_z():
    stats = repeat_statistics([1.0, 2.0, 3.0])
    sample_std = 1.0
    # t(0.975, 2) = 4.3027, not the normal 1.96: three repeats must not pretend
    # to a normal-distribution certainty.
    assert stats["half_width"] == pytest.approx(4.302653 * sample_std / np.sqrt(3), rel=1e-4)


def test_student_t_matches_published_table():
    for dof, expected in ((1, 12.7062), (2, 4.3027), (4, 2.7764), (30, 2.0423)):
        assert protocol_module._student_t_975(dof) == pytest.approx(expected, abs=2e-4)


def test_single_measurement_admits_no_spread():
    stats = repeat_statistics([0.64])
    assert stats["n"] == 1 and stats["std"] is None and stats["half_width"] is None


def test_non_finite_values_are_dropped_not_counted():
    assert repeat_statistics([0.6, float("nan"), 0.7])["n"] == 2


def test_compare_repeats_keys_the_rows():
    rows = [dict(cv=0.6, ee_rel=0.5), dict(cv=0.62, ee_rel=0.49), dict(cv=0.61, ee_rel=0.5)]
    result = compare_repeats(rows)
    assert set(result) == {"cv", "ee_rel"}
    assert result["cv"]["n"] == 3


# -- verdict_with_uncertainty ---------------------------------------------

_STATS = {"cv": repeat_statistics([0.70, 0.71, 0.705])}


def test_cv_win_with_lost_energy_is_rejected():
    verdict = verdict_with_uncertainty(
        dict(cv=0.70, ee_rel=0.50), dict(cv=0.40, ee_rel=0.30),
        stats={"cv": repeat_statistics([0.70, 0.71, 0.705])})
    assert verdict["supported"] is False
    assert verdict["reason"] == "energy_lost_with_cv"


def test_clipped_candidate_is_rejected_even_when_it_clears_the_noise_band():
    verdict = verdict_with_uncertainty(
        dict(cv=0.70, ee_rel=0.50),
        dict(cv=0.50, ee_rel=0.52, saturated_fraction=0.01),
        stats=_STATS)
    assert verdict["supported"] is False
    assert verdict["reason"] == "candidate_clipped"


def test_clean_cv_win_beyond_noise_is_supported():
    verdict = verdict_with_uncertainty(
        dict(cv=0.70, ee_rel=0.50), dict(cv=0.60, ee_rel=0.51), stats=_STATS)
    assert verdict["supported"] is True
    assert verdict["reason"] == "cv_improved_beyond_noise"
    assert verdict["noise_half_width"] > 0


def test_without_repeats_the_verdict_says_so_instead_of_inventing_a_threshold():
    verdict = verdict_with_uncertainty(dict(cv=0.70, ee_rel=0.5), dict(cv=0.60, ee_rel=0.51))
    assert verdict["supported"] is None
    assert verdict["reason"] == "insufficient_repeats"


def test_wrong_direction_and_missing_cv_are_reported_by_name():
    assert verdict_with_uncertainty(dict(cv=0.5), dict(cv=0.6))["reason"] == "wrong_direction"
    assert verdict_with_uncertainty(dict(cv=None), dict(cv=0.6))["reason"] == "missing_cv"


def test_min_effect_floor_applies_on_top_of_the_statistics():
    verdict = verdict_with_uncertainty(
        dict(cv=0.700, ee_rel=0.5), dict(cv=0.699, ee_rel=0.5),
        stats=_STATS, min_effect=0.005)
    assert verdict["supported"] is False
    assert verdict["reason"] == "below_min_effect"


def test_invalid_improve_direction_is_rejected():
    with pytest.raises(ValueError, match="improve"):
        verdict_with_uncertainty(dict(cv=0.5), dict(cv=0.4), improve="sideways")


def test_identical_phase_twice_is_a_tie_not_a_regression():
    # A strict improvement test would answer "wrong_direction" here, which reads
    # like a regression to anyone scanning the table.
    verdict = judge_pair(dict(cv=0.64, ee_rel=0.5), dict(cv=0.64, ee_rel=0.5))
    assert verdict["supported"] is False
    assert verdict["reason"] == "no_difference"


def test_pair_within_replicate_noise_is_not_a_difference():
    noise = repeat_statistics([0.640, 0.655, 0.645])
    verdict = judge_pair(dict(cv=0.640, ee_rel=0.5), dict(cv=0.652, ee_rel=0.5), noise=noise)
    assert verdict["reason"] == "no_difference"


def test_real_difference_still_needs_to_clear_noise():
    noise = repeat_statistics([0.64, 0.65, 0.64])
    worse = judge_pair(dict(cv=0.60, ee_rel=0.5), dict(cv=0.70, ee_rel=0.5), noise=noise)
    assert worse["reason"] == "wrong_direction"
    better = judge_pair(dict(cv=0.70, ee_rel=0.5), dict(cv=0.50, ee_rel=0.51), noise=noise)
    assert better["supported"] is True


def test_pair_with_a_real_change_but_no_noise_estimate_is_insufficient():
    verdict = judge_pair(dict(cv=0.70, ee_rel=0.5), dict(cv=0.50, ee_rel=0.51))
    assert verdict["supported"] is None
    assert verdict["reason"] == "insufficient_repeats"


def test_clipped_pair_is_still_rejected():
    verdict = judge_pair(dict(cv=0.70, ee_rel=0.5),
                         dict(cv=0.40, ee_rel=0.52, saturated_fraction=0.02),
                         noise=repeat_statistics([0.70, 0.705, 0.699]))
    assert verdict["reason"] == "candidate_clipped"


# -- shape_agreement ------------------------------------------------------

def test_square_versus_itself_is_perfect_agreement():
    target = np.zeros((60, 60))
    target[20:40, 20:40] = 1.0
    report = shape_agreement(target * 100.0, target)
    assert report["iou"] == pytest.approx(1.0)
    assert report["boundary_mean_px"] == pytest.approx(0.0)
    assert report["coverage_fraction"] == pytest.approx(1.0)
    assert insufficient_keys(report) == []


def test_offset_spot_reports_boundary_error_not_just_iou():
    target = np.zeros((60, 60))
    target[20:40, 20:40] = 1.0
    measured = np.zeros((60, 60))
    measured[23:43, 20:40] = 100.0
    report = shape_agreement(measured, target)
    assert 0.0 < report["iou"] < 1.0
    # Three-sample shift: half the measured boundary (top/bottom edges) sits 3
    # samples away, the other half only 1, so the mean shell distance is 2.
    assert report["boundary_mean_px"] == pytest.approx(2.0, abs=0.5)
    assert report["boundary_p95_px"] == pytest.approx(3.0, abs=0.5)


def test_empty_measurement_scores_zero_and_raises_no_boundary():
    target = np.zeros((60, 60))
    target[20:40, 20:40] = 1.0
    report = shape_agreement(np.zeros((60, 60)), target)
    assert report["iou"] == 0.0 and report["coverage_fraction"] == 0.0
    assert report["boundary_mean_px"] is None


def test_shape_mismatch_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="differ"):
        shape_agreement(np.zeros((10, 10)), np.ones((20, 20)))


# -- replay_audit ---------------------------------------------------------

def test_replay_audit_withholds_clipped_and_unrecorded_frames():
    audit = replay_audit(
        [dict(source="ok", peak=4000.0),
         dict(source="clipped", peak=4014.0),
         dict(source="unknown", peak=None)],
        full_scale=4095.0)
    assert [frame["source"] for frame in audit["usable"]] == ["ok"]
    reasons = {frame["source"]: frame["reason"] for frame in audit["withheld"]}
    assert reasons == {"clipped": "at_or_above_guard", "unknown": "peak_not_recorded"}
    assert audit["n"] == 3


def test_replay_audit_on_8bit_capture_uses_the_8bit_full_scale():
    # The failure the guard exists for: the 8-bit anchor stage clipped at 250/255.
    audit = replay_audit([dict(source="a8", peak=250.0)], full_scale=255.0)
    assert audit["usable"] == []


def test_replay_audit_rejects_a_nonsense_full_scale():
    with pytest.raises(ValueError, match="full_scale"):
        replay_audit([dict(peak=1.0)], full_scale=0.0)


# -- plan_reference_blocks ------------------------------------------------

def test_repeat_budget_states_its_own_resolution():
    plan = plan_reference_blocks(
        [dict(name="ref_s1200", block_seconds=24.0, planned_repeats=6)], block_seconds=20.0)
    row = plan["conditions"][0]
    assert row["device_seconds"] == pytest.approx(144.0)
    assert row["min_attainable_pvalue"] == pytest.approx(2 / 64)
    # Sign-flip's floor is 2/2**n, so 5 repeats alone still cannot reach 0.05
    # (2/32 = 0.0625) even though the design review called 5 "enough".
    assert plan["min_repeats_for_p005"] == 6
    assert plan["total_device_seconds"] == pytest.approx(144.0)


def test_five_repeats_land_just_above_significance():
    row = plan_reference_blocks([dict(name="ref", planned_repeats=5)], block_seconds=10.0)["conditions"][0]
    assert row["min_attainable_pvalue"] > 0.05



def test_min_attainable_pvalue_agrees_with_the_statistics_module():
    # The protocol module re-derives this to stay import-light; it must not drift
    # from the reporting statistics' single source of truth.
    from ml.zernike.eval_stats import min_attainable_pvalue

    for repeats in (2, 3, 4, 5, 8):
        assert protocol_module._min_attainable_pvalue(repeats) == pytest.approx(
            min_attainable_pvalue(repeats))


def test_empty_condition_list_is_rejected():
    with pytest.raises(ValueError, match="no conditions"):
        plan_reference_blocks([], block_seconds=10.0)
