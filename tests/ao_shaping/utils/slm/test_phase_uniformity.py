"""Circular phase smoothing should change physical gradients, not wrap cuts."""

import numpy as np

from ao_shaping.utils.slm.phase_uniformity import (
    circular_smooth_phase,
    wrapped_gradient_rms,
)


def test_wrap_boundary_is_not_a_large_gradient() -> None:
    phase = np.tile(np.array([3.13, -3.13]), (8, 4))
    pupil = np.ones_like(phase, dtype=bool)
    assert wrapped_gradient_rms(phase, pupil) < 0.03
    smoothed = circular_smooth_phase(phase, 1.0, pupil)
    assert np.all(np.abs(smoothed) > 3.0)
    assert wrapped_gradient_rms(smoothed, pupil) < wrapped_gradient_rms(phase, pupil)


def test_smoothing_preserves_outside_pupil() -> None:
    phase = np.zeros((16, 16))
    phase[4:12, 4:12] = np.indices((8, 8)).sum(axis=0) * 0.4
    pupil = np.zeros_like(phase, dtype=bool)
    pupil[4:12, 4:12] = True
    result = circular_smooth_phase(phase, 2.0, pupil)
    assert np.array_equal(result[~pupil], phase[~pupil])
    assert wrapped_gradient_rms(result, pupil) < wrapped_gradient_rms(phase, pupil)
