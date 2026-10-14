"""Small fitting and scoring contracts for saved-probe PSF comparisons."""

import numpy as np
import pytest

from scripts.evaluate_model_in_loop_psf import _score, fit_gain_leakage


def test_gain_leakage_recovers_nonnegative_components() -> None:
    model = np.array([[1.0, 2.0], [3.0, 4.0]])
    flat = np.array([[4.0, 3.0], [2.0, 1.0]])
    measured = 2 * model + 0.5 * flat
    weights, error = fit_gain_leakage([model], [measured], flat)
    np.testing.assert_allclose(weights, [2.0, 0.5], atol=1e-12)
    assert error < 1e-12


def test_score_excludes_unobserved_model_energy() -> None:
    measured = np.array([[1.0, 2.0], [3.0, 0.0]])
    predicted = np.array([[1.0, 2.0], [3.0, 100.0]])
    mask = np.array([[True, True], [True, False]])
    correlation, error = _score(predicted, measured, mask)
    assert correlation == pytest.approx(1.0)
    assert error == pytest.approx(0.0)
