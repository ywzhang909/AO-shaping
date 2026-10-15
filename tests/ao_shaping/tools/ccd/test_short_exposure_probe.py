"""Safety checks for the bounded Daheng exposure probe."""

import numpy as np
import pytest

from ao_shaping.tools.ccd.short_exposure_probe import (
    _flat_on_exit, validate_exposures,
)


@pytest.mark.parametrize("exposures", [[0.02, 0.4], [0.03, 0.06, 0.03]])
def test_exposure_limits_accept_bounded_sequences(exposures, monkeypatch):
    # With no env ceiling the static layer cannot bound the ladder; the
    # runtime image probe does.
    monkeypatch.delenv("AO_DAHENG_MAX_EXPOSURE_MS", raising=False)
    validate_exposures(exposures)


@pytest.mark.parametrize("exposures", [[0.03, 0.6], [0.01, 0.02], [float("nan"), 0.03]])
def test_exposure_limits_reject_unsafe_sequences(exposures, monkeypatch):
    monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
    with pytest.raises(ValueError, match="Daheng exposure"):
        validate_exposures(exposures)


def test_restores_flat_when_capture_fails():
    class Slm:
        displayed = None

        def create_phase_from_array(self, phase):
            assert np.count_nonzero(phase) == 0
            return phase

        def display_data(self, phase):
            self.displayed = phase

    slm = Slm()
    with pytest.raises(RuntimeError, match="camera failed"):
        with _flat_on_exit(slm):
            raise RuntimeError("camera failed")
    assert slm.displayed.shape == (1200, 1920)
