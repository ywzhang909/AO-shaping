"""Regression anchors for the PatternHelper grayscale conversion contract.

History (2026-09): ``PatternHelper._zernike_to_uint16`` used a **min-max
normalisation** ``(p - pmin) / (pmax - pmin) * 1023``, which made the output
**scale-invariant**: Zernike coefficients ×1 and ×4 produced byte-identical
uint16 patterns (``np.array_equal == True``) — the coefficient amplitude was
silently destroyed. The method was removed as part of the raw-only SLM phase
contract; ``generate_zernike_polynomial`` now returns raw unwrapped radians
and the grayscale conversion is owned by ``to_uint16`` (mod-2π wrap → scale)
or the canonical ``phase_to_slm_grayscale(phase, slm=None)`` pure fallback.

These tests lock the fixed behaviour: coefficient amplitude MUST survive the
conversion (×1 ≠ ×4), output MUST be uint16 with preserved shape and values
in ``[0, 1023]``. They would fail (RED) against the legacy min-max
implementation and pass (GREEN) against the current code.
"""

from __future__ import annotations

import numpy as np

from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

RESOLUTION = (128, 128)
BITS = 10
MAX_GRAY = 2**BITS - 1  # 1023


def _phase(amplitude: float) -> np.ndarray:
    """Raw unwrapped defocus (2,0) phase at the given coefficient amplitude."""
    helper = PatternHelper(RESOLUTION, bits=BITS)
    return helper.generate_zernike_polynomial({(2, 0): amplitude})


class TestPatternHelperGrayscaleScaleDependence:
    """Coefficient amplitude must survive the radian → grayscale conversion."""

    def test_raw_phase_preserves_coefficient_amplitude(self):
        p1 = _phase(1.0)
        p4 = _phase(4.0)
        assert not np.array_equal(p1, p4)
        assert p4.std() > p1.std()  # amplitude scales linearly, no min-max

    def test_to_uint16_is_scale_dependent(self):
        """Legacy min-max bug: ×1 and ×4 produced byte-identical uint16."""
        helper = PatternHelper(RESOLUTION, bits=BITS)
        g1 = helper.to_uint16(_phase(1.0))
        g4 = helper.to_uint16(_phase(4.0))
        assert not np.array_equal(g1, g4)

    def test_phase_to_slm_grayscale_pure_fallback_is_scale_dependent(self):
        """The canonical converter (slm=None pure fallback) must also be
        scale-dependent — the delegation target of the removed method."""
        g1 = phase_to_slm_grayscale(_phase(1.0), slm=None)
        g4 = phase_to_slm_grayscale(_phase(4.0), slm=None)
        assert not np.array_equal(g1, g4)


class TestPatternHelperGrayscaleFormat:
    """uint16 dtype, preserved shape, values within [0, 1023]."""

    def test_to_uint16_dtype_shape_range(self):
        helper = PatternHelper(RESOLUTION, bits=BITS)
        phase = helper.generate_zernike_polynomial({(2, 0): 1.0, (4, 0): 0.5})
        gray = helper.to_uint16(phase)
        assert gray.dtype == np.uint16
        assert gray.shape == RESOLUTION
        assert gray.min() >= 0
        assert gray.max() <= MAX_GRAY

    def test_phase_to_slm_grayscale_dtype_shape_range(self):
        phase = _phase(1.0)
        gray = phase_to_slm_grayscale(phase, slm=None)
        assert gray.dtype == np.uint16
        assert gray.shape == phase.shape
        assert gray.min() >= 0
        assert gray.max() <= MAX_GRAY

    def test_to_uint16_matches_pure_fallback(self):
        """``to_uint16`` (mod-2π wrap → scale) and the canonical pure fallback
        must agree for the default 10-bit grayscale."""
        helper = PatternHelper(RESOLUTION, bits=BITS)
        phase = _phase(1.0)
        assert np.array_equal(
            helper.to_uint16(phase), phase_to_slm_grayscale(phase, slm=None)
        )
