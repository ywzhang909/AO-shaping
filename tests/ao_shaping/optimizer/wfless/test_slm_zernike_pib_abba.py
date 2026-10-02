"""Offline tests for the ABBA SPGD capture order in ``slm_zernike_pib``.

No hardware: the tests exercise the pure sign-sequence helper and reconstruct
the sign-mean algebra the optimizer uses, so the drift-cancellation property is
verified without a camera or an SLM.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_zernike_pib import _spgd_capture_signs


class TestSpgdCaptureSigns:
    """The helper is the single source of the capture order."""

    def test_default_order_is_two_frame(self) -> None:
        assert _spgd_capture_signs(False) == (1, -1)

    def test_abba_order_is_four_frame_palindrome(self) -> None:
        signs = _spgd_capture_signs(True)
        assert signs == (1, -1, -1, 1)
        assert len(signs) == 4

    def test_abba_is_a_palindrome_in_time(self) -> None:
        """``+ - - +``: + samples sit at the outer, - at the inner positions."""
        signs = _spgd_capture_signs(True)
        assert sum(signs) == 0
        # The first moment vanishing is what makes a drift linear in time
        # contribute nothing to mean(+) - mean(-).
        assert sum(sign * step for step, sign in enumerate(signs)) == 0

    def test_abba_default_off_preserves_two_captures(self) -> None:
        """The opt-in must not change the default epoch cost."""
        assert len(_spgd_capture_signs(False)) == 2
        assert len(_spgd_capture_signs(True)) == 4


# Objective response per unit perturbation sign: J(+) = +s, J(-) = -s.
_SIGNAL = 1.0
# A perfect measurement returns J(+) - J(-) = 2*s.
_TRUTH = 2.0 * _SIGNAL


def _readings(signs: tuple[int, ...], drift_per_step: float) -> list[float]:
    """Objective readings for ``signs`` under a drift linear in time."""
    return [
        _SIGNAL * sign + drift_per_step * step for step, sign in enumerate(signs)
    ]


def _sign_means(values: list[float], signs: tuple[int, ...]) -> tuple[float, float]:
    """Per-sign means, exactly as the optimizer aggregates ``_captures``."""
    pos = [v for v, s in zip(values, signs, strict=True) if s > 0]
    neg = [v for v, s in zip(values, signs, strict=True) if s < 0]
    return float(np.mean(pos)), float(np.mean(neg))


class TestDriftCancellation:
    """A drift linear in time cancels for ABBA but not for ``+ -``."""

    @pytest.mark.parametrize("drift", [0.0, 0.5, 2.0])
    def test_abba_recovers_signal_under_linear_drift(self, drift: float) -> None:
        """ABBA returns ``2*s`` regardless of the drift slope."""
        signs = _spgd_capture_signs(True)
        pos, neg = _sign_means(_readings(signs, drift), signs)
        assert pos - neg == pytest.approx(_TRUTH)

    @pytest.mark.parametrize("drift", [0.5, 2.0])
    def test_two_frame_order_is_contaminated_by_drift(self, drift: float) -> None:
        """``+ -`` cannot cancel: the residual grows with the drift slope."""
        signs = _spgd_capture_signs(False)
        pos, neg = _sign_means(_readings(signs, drift), signs)
        assert pos - neg == pytest.approx(_TRUTH - drift)

    def test_abba_uses_twice_as_many_captures(self) -> None:
        """Documented cost of the drift cancellation: 2x captures per epoch."""
        assert len(_spgd_capture_signs(True)) == 2 * len(_spgd_capture_signs(False))

    def test_abba_beats_alternating_at_equal_capture_count(self) -> None:
        """``+ - + -`` costs the same 4 captures yet keeps a ``-drift`` residual."""
        drift = 1.0
        abba = _spgd_capture_signs(True)
        alternating = (1, -1, 1, -1)
        abba_pos, abba_neg = _sign_means(_readings(abba, drift), abba)
        alt_pos, alt_neg = _sign_means(_readings(alternating, drift), alternating)
        assert abba_pos - abba_neg == pytest.approx(_TRUTH)
        assert alt_pos - alt_neg == pytest.approx(_TRUTH - drift)

    def test_no_drift_makes_both_orders_agree(self) -> None:
        """Without drift the order is irrelevant - both read the true signal."""
        for abba in (False, True):
            signs = _spgd_capture_signs(abba)
            pos, neg = _sign_means(_readings(signs, 0.0), signs)
            assert pos - neg == pytest.approx(_TRUTH)