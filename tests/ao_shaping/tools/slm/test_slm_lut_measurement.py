from __future__ import annotations

import numpy as np

from ao_shaping.tools.slm.slm_lut_runner import (
    _check_spot_drift,
    _joint_exposure_check,
    _measure_power,
)


class FakeCamera:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.exposures: list[float] = []

    def reset_exposure_time(self, exposure_ms: float) -> None:
        self.exposures.append(exposure_ms)

    def get_numpy_image(self, **kwargs) -> np.ndarray:
        return self.frame


def test_measure_power_subtracts_border_background() -> None:
    frame = np.ones((9, 9), dtype=np.float64)
    frame[4, 4] = 10.0

    power, returned = _measure_power(frame, (4, 4), 5)

    assert power == 9.0
    assert returned is frame


def test_joint_exposure_halves_on_saturation() -> None:
    frame = np.zeros((10, 10), dtype=np.float64)
    frame[5, 5] = 900.0
    camera = FakeCamera(np.ones((10, 10), dtype=np.float64))

    updated, exposure_ms, adjusted = _joint_exposure_check(
        frame,
        [((5, 5), 3)],
        full_well=1000.0,
        camera=camera,
        current_exposure_ms=1.0,
        bright_floor=0.02,
        saturation_stop=0.8,
    )

    assert adjusted
    assert exposure_ms == 0.5
    assert camera.exposures == [0.5]
    assert updated.shape == frame.shape


def test_joint_exposure_doubles_on_underexposure() -> None:
    frame = np.ones((10, 10), dtype=np.float64)
    camera = FakeCamera(np.full((10, 10), 2.0, dtype=np.float64))

    updated, exposure_ms, adjusted = _joint_exposure_check(
        frame,
        [((5, 5), 3)],
        full_well=1000.0,
        camera=camera,
        current_exposure_ms=1.0,
        bright_floor=0.02,
        saturation_stop=0.8,
    )

    assert adjusted
    assert exposure_ms == 2.0
    assert camera.exposures == [2.0]
    assert updated.shape == frame.shape


def test_joint_exposure_leaves_well_exposed_frame_unchanged() -> None:
    frame = np.full((10, 10), 100.0, dtype=np.float64)
    camera = FakeCamera(frame)

    updated, exposure_ms, adjusted = _joint_exposure_check(
        frame,
        [((5, 5), 3)],
        full_well=1000.0,
        camera=camera,
        current_exposure_ms=1.0,
        bright_floor=0.02,
        saturation_stop=0.8,
    )

    assert not adjusted
    assert exposure_ms == 1.0
    assert camera.exposures == []
    assert updated is frame


def test_check_spot_drift_relocates_to_expected_peak() -> None:
    frame = np.zeros((20, 20), dtype=np.float64)
    frame[10, 10] = 100.0

    center = _check_spot_drift(
        (2, 2),
        frame,
        (10, 10),
        period=64,
        spot_window=5,
    )

    assert center == (10, 10)


def test_check_spot_drift_keeps_center_within_threshold() -> None:
    frame = np.zeros((20, 20), dtype=np.float64)

    center = _check_spot_drift(
        (10, 10),
        frame,
        (10, 10),
        period=64,
        spot_window=5,
    )

    assert center == (10, 10)
