"""Hermetic tests for the robust-SPGD integrity gates in slm-zernike-pib.

Covers the three environment-robustness primitives added for the ep15 J-collapse
fix (see docs/slm/report2.md):

- ``_frame_fold_check`` / ``_update_fold_baseline`` — bright-state fold gate
  (measured discrete brightness states with ``corr(J, max_brt) = -0.9996`` on
  the bench; a folded frame must never be treated as a gradient).
- ``_rolling_sigma`` / ``_noise_gate`` — noise-aware update gate: when
  ``|diff| <= k * sigma_hat`` the SPGD step is pure measurement noise and must
  be zeroed so the coefficients stall honestly instead of random-walking.
- ``ShapingObjective.set_reference_center`` — per-evaluation re-location of the
  target ROI onto the freshly measured spot (Fix 1), so environmental beam
  drift neither collapses the ROI energy nor false-fires the energy guard.

These are pure functions / pure-objective behaviours: no hardware, offline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SlmZernikePibConfig,
    _frame_fold_check,
    _noise_gate,
    _rolling_sigma,
    _update_fold_baseline,
)
from ao_shaping.utils.image.targets import (
    ShapingObjective,
    ShapingObjectiveParams,
)


# ---------------------------------------------------------------------------
# Local helpers (repo convention: no fixtures, helpers inline)
# ---------------------------------------------------------------------------

def make_params(
    objective: str = "roi_pib", mode: str = "max", **overrides: Any
) -> ShapingObjectiveParams:
    """ShapingObjectiveParams with the test conventions of
    ``tests/ao_shaping/utils/test_shaping_objective.py``."""
    params: dict[str, Any] = dict(
        objective=objective,
        mode=mode,
        reference_center=(10.0, 10.0),
        shape="circle",
        size=6.0,
        aspect_ratio=1.5,
        r_bucket=2.0,
        ideal_spot_radius=3,
        max_roi_energy_loss=0.0,
    )
    params.update(overrides)
    return ShapingObjectiveParams(**params)


def make_blob(center: tuple[float, float], spread: float = 1.5, size: int = 21) -> np.ndarray:
    """Normalised Gaussian blob on a ``size`` x ``size`` grid."""
    yy, xx = np.mgrid[0:size, 0:size]
    return 1000.0 * np.exp(
        -((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2.0 * spread**2)
    )


GRID = 21
FULL_PEAK = 100.0
FULL_SUM = GRID * GRID * FULL_PEAK  # 44100.0


# ---------------------------------------------------------------------------
# _frame_fold_check
# ---------------------------------------------------------------------------

class TestFrameFoldCheck:
    def test_disabled_when_fold_ratio_non_positive(self) -> None:
        frame = np.full((GRID, GRID), FULL_PEAK)
        for ratio in (0.0, -1.0):
            is_fold, peak, frame_sum = _frame_fold_check(frame, FULL_PEAK, FULL_SUM, ratio)
            assert is_fold is False
            assert peak == pytest.approx(FULL_PEAK)
            assert frame_sum == pytest.approx(FULL_SUM)

    def test_no_baseline_disables_check(self) -> None:
        frame = np.full((GRID, GRID), 1.0)
        is_fold, peak, frame_sum = _frame_fold_check(frame, None, None, 0.5)
        assert is_fold is False
        assert peak == pytest.approx(1.0)
        assert frame_sum == pytest.approx(GRID * GRID)

    def test_peak_fold_detected(self) -> None:
        # Peak below 50% of the 100.0 baseline -> folded, regardless of sum.
        frame = np.full((GRID, GRID), 40.0)
        is_fold, peak, frame_sum = _frame_fold_check(frame, FULL_PEAK, FULL_SUM, 0.5)
        assert is_fold is True
        assert peak == pytest.approx(40.0)
        assert frame_sum == pytest.approx(GRID * GRID * 40.0)

    def test_sum_fold_detected(self) -> None:
        # High isolated peak but total energy well below 50% -> folded.
        frame = np.zeros((GRID, GRID))
        frame[:4, :11] = FULL_PEAK  # 44 px @ 100 -> sum 4400.0, peak 100.0
        is_fold, peak, frame_sum = _frame_fold_check(frame, FULL_PEAK, FULL_SUM, 0.5)
        assert is_fold is True
        assert peak == pytest.approx(FULL_PEAK)
        assert frame_sum == pytest.approx(4400.0)

    def test_valid_frame_passes(self) -> None:
        frame = np.full((GRID, GRID), 90.0)
        is_fold, peak, frame_sum = _frame_fold_check(frame, FULL_PEAK, FULL_SUM, 0.5)
        assert is_fold is False
        assert peak == pytest.approx(90.0)
        assert frame_sum == pytest.approx(GRID * GRID * 90.0)


# ---------------------------------------------------------------------------
# _update_fold_baseline
# ---------------------------------------------------------------------------

class TestUpdateFoldBaseline:
    def test_seeds_from_first_valid_frame(self) -> None:
        peak, frame_sum = _update_fold_baseline(None, None, FULL_PEAK, FULL_SUM)
        assert peak == pytest.approx(FULL_PEAK)
        assert frame_sum == pytest.approx(FULL_SUM)

    def test_ema_update_default_alpha(self) -> None:
        peak, frame_sum = _update_fold_baseline(FULL_PEAK, FULL_SUM, 90.0, 39690.0)
        assert peak == pytest.approx(0.1 * 90.0 + 0.9 * FULL_PEAK)  # 99.0
        assert frame_sum == pytest.approx(0.1 * 39690.0 + 0.9 * FULL_SUM)  # 43659.0

    def test_ema_update_custom_alpha(self) -> None:
        peak, frame_sum = _update_fold_baseline(FULL_PEAK, FULL_SUM, 90.0, 39690.0, alpha=0.5)
        assert peak == pytest.approx(95.0)
        assert frame_sum == pytest.approx(41895.0)


# ---------------------------------------------------------------------------
# _rolling_sigma
# ---------------------------------------------------------------------------

class TestRollingSigma:
    def test_empty_and_single_are_degenerate(self) -> None:
        assert _rolling_sigma([]) == 0.0
        assert _rolling_sigma([1.0]) == 0.0

    def test_two_point_std(self) -> None:
        assert _rolling_sigma([1.0, 3.0]) == pytest.approx(1.0)

    def test_multi_point_std(self) -> None:
        assert _rolling_sigma([1.0, 2.0, 3.0]) == pytest.approx(np.std([1.0, 2.0, 3.0]))

    def test_non_finite_history_yields_zero(self) -> None:
        assert _rolling_sigma([float("nan"), 1.0]) == 0.0

    def test_constant_history_yields_zero(self) -> None:
        assert _rolling_sigma([2.0, 2.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# _noise_gate
# ---------------------------------------------------------------------------

class TestNoiseGate:
    def test_disabled_when_k_non_positive(self) -> None:
        assert _noise_gate(0.5, 1.0, 0.0) is False
        assert _noise_gate(0.5, 1.0, -1.0) is False

    def test_disabled_on_unusable_sigma(self) -> None:
        assert _noise_gate(0.5, 0.0, 3.0) is False
        assert _noise_gate(0.5, float("nan"), 3.0) is False
        assert _noise_gate(0.5, -1.0, 3.0) is False

    def test_gates_small_diff(self) -> None:
        assert _noise_gate(2.5, 1.0, 3.0) is True

    def test_gates_negative_diff_symmetrically(self) -> None:
        assert _noise_gate(-2.5, 1.0, 3.0) is True

    def test_boundary_is_inclusive(self) -> None:
        assert _noise_gate(3.0, 1.0, 3.0) is True

    def test_passes_large_diff(self) -> None:
        assert _noise_gate(5.0, 1.0, 3.0) is False


# ---------------------------------------------------------------------------
# ShapingObjective.set_reference_center (Fix 1: per-evaluation re-location)
# ---------------------------------------------------------------------------

class TestReferenceCenterRelocation:
    def test_recentering_recovers_roi_energy_after_beam_shift(self) -> None:
        """A blob that has drifted off the seeded centre scores ~0; after
        ``set_reference_center`` onto the measured spot the same frame scores
        high - the objective follows the beam, not the seed."""
        drift = make_blob((16.0, 16.0))
        obj = ShapingObjective(make_params(), None, make_blob((10.0, 10.0)))

        _, ratio_off = obj.raw(drift)
        assert ratio_off < 0.05

        obj.set_reference_center((16.0, 16.0))
        _, ratio_on = obj.raw(drift)
        assert ratio_on > 0.7
        assert ratio_on > 10.0 * max(ratio_off, 1e-9)

    def test_live_center_moves_but_seed_stays_frozen(self) -> None:
        obj = ShapingObjective(make_params(), None, make_blob((10.0, 10.0)))
        obj.set_reference_center((16.0, 16.0))
        assert obj._reference_center == (16.0, 16.0)
        assert obj._params.reference_center == (10.0, 10.0)

    def test_energy_guard_follows_live_center(self) -> None:
        """Armed from the seed-centred frame, a drift frame without re-location
        false-fires the guard (abandoned, j - 1e3); with re-location the guard
        passes and the score is the true ROI fraction."""
        obj = ShapingObjective(
            make_params(max_roi_energy_loss=0.5), None, make_blob((10.0, 10.0))
        )
        drift = make_blob((16.0, 16.0))

        abandoned = obj(drift).j
        assert abandoned <= -900.0  # raw ~0.0 minus the 1e3 penalty
        assert obj._violations == 1

        obj.set_reference_center((16.0, 16.0))
        passed = obj(drift).j
        assert passed > 0.5
        assert obj._violations == 1  # no new violation after re-location

    def test_no_guard_when_max_roi_energy_loss_disabled(self) -> None:
        obj = ShapingObjective(make_params(max_roi_energy_loss=0.0), None, make_blob((10.0, 10.0)))
        j = obj(make_blob((16.0, 16.0))).j
        assert j > -900.0  # guard disabled -> raw value, not abandoned


# ---------------------------------------------------------------------------
# SlmZernikePibConfig sanity (plain dataclass contract)
# ---------------------------------------------------------------------------

class TestSlmZernikePibConfig:
    def test_plain_dataclass_requires_center_and_epochs(self) -> None:
        cfg = SlmZernikePibConfig(center="max", epochs=5)
        assert cfg.center == "max"
        assert cfg.epochs == 5
        assert cfg.algorithm == "spgd"
        assert cfg.optimizer_type == "adamod"
        assert cfg.slm.n_max >= 4  # engine default, overridden by runners/tests
        assert cfg.camera.cam_type == "daheng"