"""Regression tests for the ``--delta`` pin semantics.

Before this, ``SpgdParams.delta`` defaulted to ``0.1`` and Click could not tell
"the user typed ``--delta 0.1``" from "the user typed nothing". Since the
``lr == 0`` adaptive schedule *reassigns* ``delta`` on every epoch, an explicit
``--delta`` was silently overwritten unless the caller also passed an explicit
``--lr`` -- making the flag a no-op in the common case.

Hardware relevance: the 2026-10-01 ``slm-gsnet spgd`` run was launched with
``--delta 0.2 --lr 0.02``, i.e. it dodged the bug only by accident.
"""

from __future__ import annotations

import pytest

from ao_shaping.optimizer.wfless.slm_square_shaping import SlmSquareConfig
from ao_shaping.runners.runner_common import (
    DEFAULT_SPGD_DELTA,
    SpgdParams,
    resolve_spgd_delta,
)


class TestResolveSpgdDelta:
    def test_omitted_flag_is_not_pinned_and_keeps_the_historical_default(self):
        value, pinned = resolve_spgd_delta(None)
        assert value == pytest.approx(DEFAULT_SPGD_DELTA)
        assert pinned is False

    @pytest.mark.parametrize("requested", [0.2, 0.35, 1.0])
    def test_explicit_value_is_returned_and_pinned(self, requested: float):
        value, pinned = resolve_spgd_delta(requested)
        assert value == pytest.approx(requested)
        assert pinned is True

    def test_negative_delta_is_normalised_to_magnitude(self):
        # SPGD perturbs +/-delta, so a negative request is a magnitude typo, not
        # a reversed sign. abs() keeps that from silently inverting the search.
        value, pinned = resolve_spgd_delta(-0.3)
        assert value == pytest.approx(0.3)
        assert pinned is True

    def test_default_parameter_overrides_the_module_default(self):
        value, pinned = resolve_spgd_delta(None, default=0.2)
        assert value == pytest.approx(0.2)
        assert pinned is False

    def test_spgd_params_delta_defaults_to_none(self):
        """None is the sentinel that makes "unset" distinguishable."""
        assert SpgdParams().delta is None


class TestDeltaPinnedReachesTheConfig:
    def test_config_carries_the_pin_flag(self):
        cfg = SlmSquareConfig(delta=0.2, delta_pinned=True)
        assert cfg.delta == pytest.approx(0.2)
        assert cfg.delta_pinned is True

    def test_config_pin_defaults_to_false_for_backwards_compat(self):
        """A hand-built SlmSquareConfig keeps the old schedule-owns-delta path."""
        assert SlmSquareConfig().delta_pinned is False


class TestSquareRunnerWiring:
    """`slm-gsnet spgd` must forward the pin decision into the config."""

    @staticmethod
    def _cfg_with(delta: float | None):
        # Imported lazily: the runner module pulls in the SDK-backed camera
        # stack at import time, which is irrelevant to these pure-config tests.
        from ao_shaping.runners.slm.gsnet_runner import _build_square_config

        class _Objective:
            target_side = 100
            target_mean_brightness = 0.0
            side_factor = 1.5
            w_uniformity = 0.4
            w_efficiency = 0.6
            w_aspect = 0.0
            w_pbr = 0.0
            target_max_brightness = 200
            objective = "quality"

        class _Cam:
            cam_size = 300
            exposure_time_ms = 0.0
            cam_id = 0
            cam_type = "daheng"

        class _Slm:
            slm_number = 1
            slm_wavelength = 1064
            n_max = 4
            zernike_radius = 450

        class _Run:
            seed = None

        class _Cfg:
            run = _Run()
            camera = _Cam()
            slm = _Slm()
            objective = _Objective()
            center = "shape"
            search = SpgdParams(delta=delta, lr=0.0)

        return _build_square_config(_Cfg())

    def test_omitted_delta_leaves_the_schedule_in_charge(self):
        cfg = self._cfg_with(None)
        assert cfg.delta == pytest.approx(DEFAULT_SPGD_DELTA)
        assert cfg.delta_pinned is False

    def test_explicit_delta_is_pinned_even_with_auto_lr(self):
        """The exact case that used to be silently discarded (lr defaults to 0)."""
        cfg = self._cfg_with(0.2)
        assert cfg.delta == pytest.approx(0.2)
        assert cfg.delta_pinned is True

    def test_square_run_keeps_the_new_safety_defaults(self):
        """Guards established from the 2026-10-01 hardware verdict."""
        cfg = self._cfg_with(None)
        assert cfg.init_amplitude_rad == pytest.approx(0.0), "must start flat"
        assert cfg.max_roi_energy_loss == pytest.approx(0.6), "ROI energy guard"