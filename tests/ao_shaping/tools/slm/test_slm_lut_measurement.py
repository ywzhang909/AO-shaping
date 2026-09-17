from __future__ import annotations

import numpy as np

from ao_shaping.tools.slm.slm_lut_runner import (
    _check_spot_drift,
    _joint_exposure_settle,
    _measure_power,
)


class FakeCamera:
    """Returns a frame whose pixel values scale linearly with exposure_ms.

    Real cameras do this: doubling exposure doubles the ROI mean/max.  The
    settle loop needs the re-captured frame to reflect the new exposure,
    otherwise it can never converge.
    """

    def __init__(self, frame: np.ndarray, base_exposure_ms: float = 1.0):
        self.frame = frame
        self.base_exposure_ms = base_exposure_ms
        self.exposures: list[float] = []

    def reset_exposure_time(self, exposure_ms: float) -> None:
        self.exposures.append(exposure_ms)

    def get_numpy_image(self, **kwargs) -> np.ndarray:
        cur = self.exposures[-1] if self.exposures else self.base_exposure_ms
        scale = cur / self.base_exposure_ms
        return np.clip(self.frame * scale, 0, np.iinfo(np.uint16).max)


def test_measure_power_subtracts_border_background() -> None:
    frame = np.ones((9, 9), dtype=np.float64)
    frame[4, 4] = 10.0

    power, returned = _measure_power(frame, (4, 4), 5)

    assert power == 9.0
    assert returned is frame


def test_joint_exposure_settle_halves_on_saturation() -> None:
    # ROI max (900/1000 = 0.9) > saturation_stop (0.8) -> exposure halves.
    # After halving the re-captured frame scales down (max=450 -> 0.45) so the
    # loop converges on round 2.
    frame = np.zeros((10, 10), dtype=np.float64)
    frame[5, 5] = 900.0
    camera = FakeCamera(frame)

    exposure_ms = _joint_exposure_settle(
        [((5, 5), 3)],
        full_well=1000.0,
        camera=camera,
        exposure_ms=1.0,
        bright_floor=0.02,
        saturation_stop=0.8,
        max_rounds=2,
        settle_s=0.0,
    )

    assert exposure_ms == 0.5
    assert camera.exposures == [1.0, 0.5]


def test_joint_exposure_settle_doubles_on_underexposure() -> None:
    # ROI mean (10/1000 = 0.01) < bright_floor (0.02) -> exposure doubles.
    # After doubling the re-captured frame scales up (mean=20 -> 0.02) which is
    # NOT < bright_floor, so the loop converges on round 2.
    frame = np.full((10, 10), 10.0, dtype=np.float64)
    camera = FakeCamera(frame)

    exposure_ms = _joint_exposure_settle(
        [((5, 5), 3)],
        full_well=1000.0,
        camera=camera,
        exposure_ms=1.0,
        bright_floor=0.02,
        saturation_stop=0.8,
        max_rounds=2,
        settle_s=0.0,
    )

    assert exposure_ms == 2.0
    assert camera.exposures == [1.0, 2.0]


def test_joint_exposure_settle_leaves_well_exposed_unchanged() -> None:
    # ROI mean (100/1000 = 0.1) >= bright_floor, max (100/1000 = 0.1) <= stop
    frame = np.full((10, 10), 100.0, dtype=np.float64)
    camera = FakeCamera(frame)

    exposure_ms = _joint_exposure_settle(
        [((5, 5), 3)],
        full_well=1000.0,
        camera=camera,
        exposure_ms=1.0,
        bright_floor=0.02,
        saturation_stop=0.8,
        settle_s=0.0,
    )

    assert exposure_ms == 1.0
    assert camera.exposures == [1.0]  # one probe only, no retry


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
