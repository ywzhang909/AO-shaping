"""Issue #238: offline tests for the binary phase-grating fitter (no hardware).

Checks the exact Fourier model and that the e-domain fit recovers a known
2*pi gray from a synthetic curve, which is the property that breaks the
gamma <-> 2*pi-gray degeneracy of the blaze fit.
"""

from __future__ import annotations

import importlib
import math

import numpy as np

FIT = importlib.import_module("scripts.slm.issue238_binary_fit")


def test_binary_eta_endpoints_match_fourier_result():
    # phi = 0 (no phase step): everything in the 0 order.
    e0, e1 = FIT.binary_eta(np.array([0.0]))
    assert e0[0] == 1.0
    assert e1[0] == 0.0

    # phi = pi: the 0 order vanishes, +-1 orders carry (2/pi)^2 each.
    e0, e1 = FIT.binary_eta(np.array([math.pi]))
    assert abs(e0[0]) < 1e-12
    assert abs(e1[0] - 4.0 / math.pi**2) < 1e-12

    # phi = 2*pi: back to the 0 order (2*pi periodicity -> no branch ambiguity).
    e0, e1 = FIT.binary_eta(np.array([2.0 * math.pi]))
    assert abs(e0[0] - 1.0) < 1e-12
    assert abs(e1[0]) < 1e-12


def test_model_fractions_are_normalised_and_periodic():
    gray = np.array([100.0, 500.0, 993.0, 1200.0])
    e0, e1 = FIT.model_fractions(gray, g2pi=993.0, gamma=0.77)
    assert np.all(e0 >= 0.0) and np.all(e1 >= 0.0)
    assert np.allclose(e0 + 2.0 * e1, 1.0)

    # Exact 2*pi gray -> all power in the 0 order.
    at_two_pi = FIT.model_fractions(np.array([993.0]), 993.0, 0.77)
    assert abs(at_two_pi[0][0] - 1.0) < 1e-12
    assert abs(at_two_pi[1][0]) < 1e-12

    # Periodicity: g2pi + 2*pi-equivalent phase gives the same fractions.
    # phi(g) = 2*pi*(g/993)^0.77 ; phi(993) = 2*pi and phi is monotone.
    g_beyond = 993.0 * (2.0) ** (1.0 / 0.77)
    far = FIT.model_fractions(np.array([g_beyond]), 993.0, 0.77)
    assert abs(far[0][0] - 1.0) < 1e-9


def test_fit_recovers_known_two_pi_gray_and_gamma():
    g2pi_true, gamma_true = 993.0, 0.77
    gray = np.array(sorted(set(range(0, 1024, 32)) | set(range(864, 1024, 8))),
                    dtype=float)
    e0, e1 = FIT.model_fractions(gray, g2pi_true, gamma_true)
    result = FIT.fit_g2pi(gray, e0, e1)
    assert result["success"]
    assert abs(result["g2pi"] - g2pi_true) < 2.0, result
    assert abs(result["gamma"] - gamma_true) < 0.02, result
    assert result["rms"] < 1e-6


def test_fold_estimator_is_gamma_independent():
    """The high-gray fold-back zero gives g2pi with no gamma assumption."""
    gray = np.array(sorted(set(range(0, 1024, 32)) | set(range(864, 1024, 8))),
                    dtype=float)
    for gamma in (0.65, 0.77, 0.95):
        e0, e1 = FIT.model_fractions(gray, 993.0, gamma)
        fold = FIT.g2pi_direct_fold(gray, e0, e1)
        assert fold["available"]
        assert abs(fold["g2pi"] - 993.0) < 8.0, (gamma, fold)


def test_free_duty_fit_recovers_non_half_duty():
    """A 45% duty grating must be recovered, not mistaken for a different g2pi."""
    gray = np.array(sorted(set(range(0, 1024, 32)) | set(range(864, 1024, 8))),
                    dtype=float)
    e0, e1 = FIT.model_fractions(gray, 993.0, 0.77, duty=0.45)
    fixed = FIT.fit_g2pi(gray, e0, e1, duty=0.5)
    free = FIT.fit_g2pi_free_duty(gray, e0, e1)
    assert abs(free["duty"] - 0.45) < 0.03, free
    assert abs(free["g2pi"] - 993.0) < 5.0, free
    # Forcing the wrong duty must fit worse than the free-duty model.
    assert free["rms"] < fixed["rms"]


def test_fit_separates_candidate_hypotheses():
    """The fit must not confuse a 938 device with a 993 device."""
    gray = np.array(sorted(set(range(0, 1024, 32)) | set(range(864, 1024, 8))),
                    dtype=float)
    for true in (938.0, 993.0):
        e0, e1 = FIT.model_fractions(gray, true, 0.77)
        got = FIT.fit_g2pi(gray, e0, e1)["g2pi"]
        assert abs(got - true) < 2.0
        assert abs(got - (993.0 if true == 938.0 else 938.0)) > 30.0


def test_fractions_from_powers_splits_zero_and_first_orders():
    # 3 orders: 0 order power 2, +1 1, -1 1 -> e0 = 0.5, e1 = 0.25.
    powers = np.array([[2.0, 1.0, 1.0]])
    e0, e1 = FIT.fractions(powers)
    assert np.allclose(e0, 0.5)
    assert np.allclose(e1, 0.25)

    # Zero total power must not raise (clip guard).
    e0, e1 = FIT.fractions(np.zeros((1, 3)))
    assert np.isfinite(e0).all() and np.isfinite(e1).all()
