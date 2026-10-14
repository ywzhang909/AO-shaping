"""The paired comparator must score only light the CCD could observe."""

import numpy as np

from scripts.compare_model_in_loop_unet import _input, _score


def test_unobserved_energy_does_not_change_metrics() -> None:
    mask = np.zeros((192, 192), dtype=bool)
    mask[40:152, 40:152] = True
    measured = np.zeros((192, 192), dtype=float)
    measured[80:112, 80:112] = 1.0
    prediction = measured.copy()
    prediction[~mask] = 100.0
    actual = _score(prediction, measured, 36, mask)
    baseline = _score(measured, measured, 36, mask)
    assert actual == baseline


def test_phase_input_is_two_channel_phasor() -> None:
    x = _input(np.zeros((64, 64), dtype=np.float32))
    assert tuple(x.shape) == (2, 192, 192)
    assert np.allclose(x[0], 1)
    assert np.allclose(x[1], 0)
    with_exposure = _input(np.zeros((64, 64), dtype=np.float32), 0.2,
                           exposure_channel=True)
    assert tuple(with_exposure.shape) == (3, 192, 192)
    assert np.allclose(with_exposure[2], np.log10(0.5))
