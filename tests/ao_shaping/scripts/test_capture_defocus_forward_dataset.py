"""Offline phase diversity and seed separation before physical acquisition."""
import numpy as np

from scripts.slm.capture_defocus_forward_dataset import probe_grid


def test_probe_grid_families_and_raw_radians():
    for index, family in enumerate(("zernike", "smooth", "bandlimited")):
        name, phase = probe_grid(index, 64, 1022)
        assert name == family
        assert phase.shape == (64, 64)
        assert np.isfinite(phase).all()
        assert phase.min() < 0 < phase.max()
        assert phase.std() > .1
        np.testing.assert_array_equal(phase, probe_grid(index, 64, 1022)[1])


def test_independent_phase_seeds():
    for index in range(3):
        assert not np.array_equal(probe_grid(index, 64, 1022)[1],
                                  probe_grid(index + 3, 64, 1022)[1])
