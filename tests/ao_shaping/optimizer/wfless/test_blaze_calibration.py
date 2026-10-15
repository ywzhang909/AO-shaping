"""Off-axis carrier selection uses measured CCD geometry."""

import numpy as np
import pytest

from ao_shaping.optimizer.wfless._blaze_calibration import select_blaze


def _fake_capture(phase: np.ndarray) -> np.ndarray:
    frame = np.zeros((300, 400), dtype=np.float64)
    frame[150, 200] = 80.0
    # A pure grating is a constant phase step along one panel axis.
    step_x = float(np.mean(np.diff(phase, axis=1)))
    step_y = float(np.mean(np.diff(phase, axis=0)))
    if abs(step_x) > 1e-5:
        shift = round(7400 * abs(step_x) / (2 * np.pi))
        if 150 + shift < 300:
            frame[150 + shift, 200] = 100.0
    elif abs(step_y) > 1e-5:
        shift = round(7400 * abs(step_y) / (2 * np.pi))
        if 200 + shift < 400:
            frame[150, 200 + shift] = 100.0
    return frame


def test_select_blaze_measures_signal_and_preserves_zero_reference() -> None:
    choice = select_blaze(
        _fake_capture, (80, 100), target_side=40, window_size=100,
        period_px=128, direction="vertical",
    )
    assert choice.zero_xy == (200, 150)
    assert choice.signal_xy == (200, 208)
    assert choice.params.blaze_period_px == 128
    assert choice.separation_px == 58


def test_select_blaze_refuses_overlap_or_off_sensor() -> None:
    with pytest.raises(RuntimeError, match="no measured blaze order"):
        select_blaze(
            _fake_capture, (80, 100), target_side=90, window_size=120,
            period_px=192, direction="vertical",
        )
    with pytest.raises(RuntimeError, match="no measured blaze order"):
        select_blaze(
            _fake_capture, (80, 100), target_side=40, window_size=100,
            period_px=32, direction="vertical",
        )


def test_auto_chooses_smallest_measured_separation_that_clears_target() -> None:
    choice = select_blaze(
        _fake_capture, (80, 100), target_side=40, window_size=100,
    )
    assert choice.zero_xy == (200, 150)
    assert choice.separation_px >= 32
    assert choice.params.blaze_period_px >= 128
