"""Offline tests for the SLM-Zernike heuristic wiring (no hardware).

The shared driver itself is covered by
``tests/ao_shaping/algorithm/test_heuristic_search.py``; here we only pin the
integration points of ``optimize_slm_zernike_pib`` after the refactor onto it.
"""

from __future__ import annotations

import pytest

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    ALGORITHM_CHOICES,
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)


def test_algorithm_choices_come_from_the_shared_driver():
    assert ALGORITHM_CHOICES == heuristic_algorithm_choices()
    assert ALGORITHM_CHOICES[0] == "spgd"
    assert set(ALGORITHM_CHOICES[1:]) == {"ga", "pso", "sa", "hc", "rs", "cem", "de"}


def test_config_exposes_algorithm_and_pop_size():
    config = SlmZernikePibConfig(center="shape", epochs=1)

    assert config.algorithm == "spgd"
    assert config.pop_size is None


def test_unknown_algorithm_rejected_before_touching_hardware():
    # Validation runs before the camera/SLM context managers, so this stays
    # offline: no device is opened.
    with pytest.raises(ValueError, match="algorithm must be one of"):
        optimize_slm_zernike_pib(
            SlmZernikePibConfig(center="shape", epochs=1, algorithm="bogus")
        )


def test_nested_config_uses_canonical_runner_defaults():
    config = SlmZernikePibConfig(center="shape", epochs=1)

    assert config.center == "shape"
    assert config.epochs == 1
    assert config.camera.name == "pib"
    assert config.camera.target_shape is None
    assert config.camera.r_bucket == 0
    assert config.slm.zernike_radius == 600.0
    assert not hasattr(config, "r_bucket")


def test_nested_groups_carry_the_full_parameter_set():
    from ao_shaping.runners.runner_common import (
        CameraParamsPib,
        ObjectiveTarget,
        SlmParamsPib,
    )

    config = SlmZernikePibConfig(
        center="shape",
        epochs=17,
        algorithm="ga",
        pop_size=17,
        random_seed=23,
        optimizer_type="muno",
        delta=0.4,
        lr=0.3,
        shrink_iter=8,
        shrink_ratio=0.75,
        show=True,
        record_phase=True,
        camera=CameraParamsPib(
            target=ObjectiveTarget(name="shape", target_shape="circle"),
            cam_id=3,
            cam_type="sim",
            cam_size=123,
            exposure_time_ms=4.5,
            target_max_brightness=31,
            r_bucket=11,
            target_size=48.0,
            target_aspect_ratio=1.5,
            target_center_smooth=5,
            shape_schedule=True,
            max_roi_energy_loss=0.4,
            w_uniformity=3.0,
            w_peak=0.7,
            w_displacement=0.2,
            log_uniformity=True,
            w_ema_decay=0.8,
            w_floor=0.2,
            w_temperature=9.0,
            w_pib_init=0.5,
            w_rms_init=0.3,
            w_ee_init=0.2,
        ),
        slm=SlmParamsPib(
            slm_number=2,
            slm_wavelength=532,
            n_max=3,
            zernike_radius=19.0,
            init_c="0.1,0.2",
            shift_x=-2,
            shift_y=4,
        ),
        kwargs={"beta1": 0.8},
    )

    assert config.center == "shape"
    assert config.epochs == 17
    assert config.algorithm == "ga"
    assert config.pop_size == 17
    assert config.random_seed == 23
    assert config.optimizer_type == "muno"
    assert config.delta == 0.4
    assert config.lr == 0.3
    assert config.shrink_iter == 8
    assert config.shrink_ratio == 0.75
    assert config.show is True
    assert config.record_phase is True
    assert config.kwargs == {"beta1": 0.8}

    assert config.camera.name == "shape"
    assert config.camera.cam_id == 3
    assert config.camera.cam_type == "sim"
    assert config.camera.cam_size == 123
    assert config.camera.exposure_time_ms == 4.5
    assert config.camera.target_max_brightness == 31
    assert config.camera.r_bucket == 11
    assert config.camera.target_shape == "circle"
    assert config.camera.target_size == 48.0
    assert config.camera.target_aspect_ratio == 1.5
    assert config.camera.target_center_smooth == 5
    assert config.camera.shape_schedule is True
    assert config.camera.max_roi_energy_loss == 0.4
    assert config.camera.w_uniformity == 3.0
    assert config.camera.w_peak == 0.7
    assert config.camera.w_displacement == 0.2
    assert config.camera.log_uniformity is True
    assert config.camera.w_ema_decay == 0.8
    assert config.camera.w_floor == 0.2
    assert config.camera.w_temperature == 9.0
    assert config.camera.w_pib_init == 0.5
    assert config.camera.w_rms_init == 0.3
    assert config.camera.w_ee_init == 0.2

    assert config.slm.slm_number == 2
    assert config.slm.slm_wavelength == 532
    assert config.slm.n_max == 3
    assert config.slm.zernike_radius == 19.0
    assert config.slm.init_c == "0.1,0.2"
    assert config.slm.shift_x == -2
    assert config.slm.shift_y == 4


def test_explicit_nested_groups_are_used_verbatim():
    from ao_shaping.runners.runner_common import (
        CameraParamsPib,
        ObjectiveTarget,
        SlmParamsPib,
    )

    camera = CameraParamsPib(
        target=ObjectiveTarget(name="pib"), r_bucket=7, cam_type="sim"
    )
    slm = SlmParamsPib(slm_number=3, zernike_radius=480.0)

    config = SlmZernikePibConfig(
        center="shape",
        epochs=1,
        camera=camera,
        slm=slm,
    )

    assert config.camera is camera
    assert config.slm is slm
    assert config.camera.name == "pib"
    assert config.camera.r_bucket == 7
    assert config.slm.n_max == 4
    assert config.slm.zernike_radius == 480.0


def test_pib_rejects_a_zero_radius_while_the_parent_class_keeps_zero_as_default():
    """The two SLM param classes disagree on what ``0`` means, and both are right.

    X-6 filed this as "new validation rejects old fixtures" and left the decision
    open. It is not ambiguous once the class definitions are read:

    * ``SlmParams`` (slm-gsnet family) defaults ``zernike_radius`` to ``0.0`` and
      its help says "0 = default" -- the optimizers then substitute their own
      aperture radius. Zero genuinely means "unset" there.
    * ``SlmParamsPib`` (slm-pib family) *overrides* the field with
      ``DEFAULT_ZERNIKE_RADIUS`` and its help says "default = 600 px". Its
      ``__post_init__`` rejects a non-positive radius, because for this class 0
      is a typo, not a request for the default.

    Loosening the check to satisfy a fixture would have quietly given the
    slm-pib family the parent's sentinel and made the documented 600 px default
    unreachable. Pin both sides instead, so neither contract can drift.
    """
    from ao_shaping.runners.runner_common import SlmParams, SlmParamsPib

    assert SlmParamsPib().zernike_radius == 600.0
    with pytest.raises(ValueError, match="zernike_radius must be positive"):
        SlmParamsPib(zernike_radius=0.0)
    with pytest.raises(ValueError, match="zernike_radius must be positive"):
        SlmParamsPib(zernike_radius=-1.0)

    assert SlmParams().zernike_radius == 0.0
