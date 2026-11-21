# -*- coding: utf-8 -*-
"""Frozen strength-gate primitives: analytic relL2, zero-order mask, δ = 0.20.

The criterion is the a priori one from
``docs/dev/transfer_upstream_qualification_plan_20261119.md`` §2. These tests
pin the arithmetic (not a run's numbers): an absolute relL2 with DC
subtraction, the literal ``max(..., 1e-9)`` empty-mask branch, the zero-order
disk geometry on the window grid, and the frozen threshold value.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.drivers.ccd import dark_frames as ccd_dark_frames
from ao_shaping.tools.slm.transfer.strength_gate import (
    STRENGTH_GATE_DELTA,
    gate_rel_l2,
    masked_absolute_rel_l2,
    match_dark_exposure,
    window_zero_mask,
)


def test_match_dark_exposure_exact_and_missing():
    """An exact key is returned (the key, not its value); a miss is None."""
    darks = {0.05: "dark-50us", 0.1: "dark-100us"}

    assert match_dark_exposure(darks, 0.05) == 0.05
    assert match_dark_exposure(darks, 0.2) is None
    assert match_dark_exposure(darks, 0.0) is None
    assert match_dark_exposure(darks, float("nan")) is None
    assert match_dark_exposure(darks, -0.05) is None


def test_match_dark_exposure_tolerates_float32_round_trip_by_default():
    """0.051f == 0.050999999046325684 must still select the 0.051 dark."""
    darks = {0.05: "a", np.float32(0.051).item(): "b"}

    assert match_dark_exposure(darks, 0.051) == np.float32(0.051).item()
    assert match_dark_exposure(darks, np.float32(0.051).item()) == np.float32(0.051).item()


def test_match_dark_exposure_rel_tol_only_widens_when_requested():
    """Strict default never crosses exposure levels; rel_tol does so explicitly."""
    darks = {0.05: "a", 0.1: "b"}

    assert match_dark_exposure(darks, 0.051) is None
    assert match_dark_exposure(darks, 0.051, rel_tol=0.02) == 0.05
    # Nearest key wins when a loose tolerance spans two entries.
    assert match_dark_exposure(darks, 0.075, rel_tol=1.0) in (0.05, 0.1)


def test_match_dark_exposure_rejects_negative_rel_tol():
    with pytest.raises(ValueError):
        match_dark_exposure({0.05: "a"}, 0.05, rel_tol=-0.1)


def test_dark_library_manifest_yields_mean_keyed_exposure_map(tmp_path):
    """A v1 manifest + ``mean``-keyed npz parses into an exposure -> entry map."""
    import gc

    frame = np.full((4, 4), 2.0, dtype=np.float32)
    array_path = tmp_path / "dark-exp-0.05ms.npz"
    np.savez(array_path, mean=frame)
    manifest = {
        "schema": "ao_shaping.ccd.dark_frames/v1",
        "serial": "TEST-SN",
        "entries": [{
            "exposure_ms": 0.05, "file": array_path.name, "frames": 3,
            "shape": [4, 4], "bit_depth": 8, "full_scale": 255.0,
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["schema"] == ccd_dark_frames.SCHEMA
    entries = tuple(ccd_dark_frames.DarkFrameEntry.from_dict(item)
                    for item in data["entries"])
    exposure_map = {entry.exposure_ms: entry for entry in entries}

    assert match_dark_exposure(exposure_map, 0.05) == 0.05
    entry = exposure_map[0.05]
    with np.load(tmp_path / entry.file) as saved:
        assert "mean" in saved.files
        assert "exposure_ms" not in saved.files
        assert np.asarray(saved["mean"]).shape == tuple(entry.shape)
    assert entry.bit_depth == 8 and entry.full_scale == 255.0
    gc.collect()  # release the np.load handle on Windows before tmp_path cleanup


def test_delta_is_the_frozen_twenty_percent():
    """δ is pre-registered before any candidate is scored; 0.20 is not tunable."""
    assert STRENGTH_GATE_DELTA == 0.20


def test_masked_rel_l2_matches_hand_arithmetic_full_mask():
    """pred - meas = [1, 2, 3, 4] and ||meas|| = sqrt(30): relL2 = 1 exactly."""
    pred = np.array([[2.0, 4.0], [6.0, 8.0]])
    meas = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones((2, 2), dtype=bool)

    result = masked_absolute_rel_l2(pred, meas, mask)

    assert result["rel_l2"] == pytest.approx(1.0, rel=0, abs=1e-12)
    assert result["energy_meas"] == pytest.approx(np.sqrt(30.0), rel=1e-12)
    assert result["n_pixels"] == 4


def test_masked_rel_l2_is_absolute_not_scale_invariant():
    """A shape-perfect 2x-bright model must fail the absolute gate (relL2 = 1)."""
    meas = np.array([[3.0, 4.0], [0.0, 5.0]])
    pred = 3.0 * meas

    result = masked_absolute_rel_l2(pred, meas, np.ones_like(meas, dtype=bool))

    assert result["rel_l2"] == pytest.approx(2.0, rel=1e-12)


def test_masked_rel_l2_subtracts_both_dc_terms():
    """pred = meas + 5 with dc_pred = 5 removes the offset completely."""
    meas = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = meas + 5.0

    result = masked_absolute_rel_l2(pred, meas, np.ones((2, 2), dtype=bool),
                                    dc_pred=5.0, dc_meas=0.0)

    assert result["rel_l2"] == 0.0
    assert result["energy_meas"] == pytest.approx(np.sqrt(30.0), rel=1e-12)


def test_masked_rel_l2_uses_only_masked_cells():
    """mask = [[T, F], [F, T]] keeps (1, 3): ||residual|| = ||meas|| = sqrt(17)."""
    pred = np.array([[2.0, 0.0], [0.0, 8.0]])
    meas = np.array([[1.0, 0.0], [0.0, 4.0]])
    mask = np.array([[True, False], [False, True]])

    result = masked_absolute_rel_l2(pred, meas, mask)

    assert result["n_pixels"] == 2
    assert result["energy_meas"] == pytest.approx(np.sqrt(17.0), rel=1e-12)
    assert result["rel_l2"] == pytest.approx(1.0, rel=1e-12)


def test_masked_rel_l2_empty_mask_hits_the_floor_branch():
    """No kept pixel: 0 / max(0, 1e-9) = 0.0 and n_pixels = 0 (not a pass).

    The caller must treat ``n_pixels == 0`` as *not assessable*; the returned
    number itself is the literal floor branch of the frozen formula.
    """
    zeros = np.zeros((3, 3))

    result = masked_absolute_rel_l2(zeros, zeros, np.zeros((3, 3), dtype=bool))

    assert result["rel_l2"] == 0.0
    assert result["energy_meas"] == 0.0
    assert result["n_pixels"] == 0


def test_masked_rel_l2_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        masked_absolute_rel_l2(np.zeros((2, 2)), np.zeros((3, 3)), np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        masked_absolute_rel_l2(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((3, 3), dtype=bool))


def test_masked_rel_l2_rejects_non_finite_inputs():
    pred = np.array([[1.0, np.nan], [0.0, 0.0]])

    with pytest.raises(ValueError):
        masked_absolute_rel_l2(pred, np.zeros((2, 2)), np.ones((2, 2), dtype=bool))


def test_zero_mask_radius_zero_keeps_the_whole_window():
    mask = window_zero_mask((7, 5), window_step=2.0, zero_radius_px=0.0)

    assert mask.shape == (7, 5)
    assert mask.all()


def test_zero_mask_excludes_the_disk_around_the_grid_centre():
    """(11, 11) grid, radius 2 window px: cells up to 2 away drop, 3 away stay."""
    mask = window_zero_mask((11, 11), window_step=2.0, zero_radius_px=2.0)

    assert not mask[5, 5]           # centre
    assert not mask[5, 7]           # distance 2 (inclusive)
    assert not mask[3, 5]           # distance 2 the other axis
    assert mask[5, 8]               # distance 3
    assert mask[0, 0]               # far corner
    assert mask.sum() == 121 - 13   # 13 lattice cells within distance 2


def test_zero_mask_honours_reference_xy_in_window_pixels():
    default = window_zero_mask((11, 11), window_step=2.0, zero_radius_px=2.0)
    shifted = window_zero_mask((11, 11), window_step=2.0, zero_radius_px=2.0,
                               reference_xy=(3.0, 4.0))

    assert not shifted[4, 3]
    assert shifted[5, 5]
    assert not np.array_equal(default, shifted)
    assert shifted.sum() == 121 - 13


def test_zero_mask_is_invariant_to_window_step():
    """The radius is in window pixels, so the mask is the same at any step.

    ``window_step`` only rescales the equivalent full-frame radius that the
    runner records, never which window cells fall inside the disk.
    """
    step_two = window_zero_mask((21, 21), window_step=2.0, zero_radius_px=5.0)
    step_four = window_zero_mask((21, 21), window_step=4.0, zero_radius_px=5.0)

    assert np.array_equal(step_two, step_four)


def test_zero_mask_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        window_zero_mask((0, 5), window_step=2.0, zero_radius_px=0.0)
    with pytest.raises(ValueError):
        window_zero_mask((5, 5), window_step=0.0, zero_radius_px=1.0)
    with pytest.raises(ValueError):
        window_zero_mask((5, 5), window_step=2.0, zero_radius_px=-1.0)


def test_gate_rel_l2_subtracts_counts_per_ms_without_an_exposure_factor():
    """Both DC terms are counts/ms, so the gate never multiplies by exposure.

    This pins the runner's archived dimensional bug: subtracting
    ``dark_rate * exposure`` under-subtracts the dark by ``1/exposure`` (a factor
    of ~20 at the 0.051 ms hold-out exposure) and silently moves the verdict.
    """
    exposure = 0.051            # replicate_a hold-out exposure (ms)
    meas = np.array([[10.0, 12.0], [14.0, 16.0]])
    pred = meas.copy()
    dark_rate = np.full((2, 2), 4.0)   # counts/ms at the probe exposure
    bias_rate = 3.0                    # bias_counts / exposure, counts/ms
    mask = np.ones((2, 2), dtype=bool)

    gate = gate_rel_l2(pred, meas, mask, bias_rate=bias_rate, dark_rate=dark_rate)

    expected = masked_absolute_rel_l2(pred - bias_rate, meas - dark_rate, mask)
    assert gate["rel_l2"] == pytest.approx(expected["rel_l2"], rel=1e-12)
    assert gate["energy_meas"] == pytest.approx(expected["energy_meas"], rel=1e-12)
    assert gate["n_pixels"] == 4

    wrong = masked_absolute_rel_l2(pred - bias_rate, meas - dark_rate * exposure, mask)
    assert gate["rel_l2"] != pytest.approx(wrong["rel_l2"], rel=1e-9)


def test_gate_rel_l2_accepts_per_pixel_bias_and_dark_maps():
    """A dark frame projected onto the window is a map, not a scalar."""
    meas = np.array([[10.0, 12.0], [14.0, 16.0]])
    pred = np.array([[9.0, 13.0], [15.0, 17.0]])
    dark_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    bias_map = np.array([[0.5, 0.5], [0.5, 0.5]])
    mask = np.ones((2, 2), dtype=bool)

    gate = gate_rel_l2(pred, meas, mask, bias_rate=bias_map, dark_rate=dark_map)

    expected = masked_absolute_rel_l2(pred - bias_map, meas - dark_map, mask)
    assert gate["rel_l2"] == pytest.approx(expected["rel_l2"], rel=1e-12)
    assert gate["energy_meas"] == pytest.approx(expected["energy_meas"], rel=1e-12)
