"""Circular phase filtering must respect the SLM's 2*pi equivalence."""

import numpy as np
import pytest

from scripts.slm.evaluate_model_in_loop_phase_bandwidth import circular_phase_blur


def test_circular_phase_blur_is_invariant_to_full_turns() -> None:
    phase = np.array([[0.0, np.pi / 3], [-np.pi / 2, np.pi]])
    shifted = phase + 2 * np.pi * np.array([[0, 1], [-2, 3]])
    filtered = circular_phase_blur(phase, 0.6)
    shifted_filtered = circular_phase_blur(shifted, 0.6)
    np.testing.assert_allclose(np.exp(1j * filtered), np.exp(1j * shifted_filtered),
                               atol=1e-12)


def test_circular_phase_blur_zero_is_identity_and_rejects_invalid_width() -> None:
    phase = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(circular_phase_blur(phase, 0), phase)
    with pytest.raises(ValueError, match="sigma"):
        circular_phase_blur(phase, -0.1)
