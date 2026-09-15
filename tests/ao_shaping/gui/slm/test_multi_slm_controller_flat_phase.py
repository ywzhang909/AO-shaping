"""Unit tests for multi_slm_controller — flat phase (平场) generation fix.

The bug: ``generate_phase_gray("平场", gray=N)`` previously routed through
``create_phase_from_array()`` which treats input as **radians** and applies
modulo 2π before converting to grayscale.  The fix returns raw uint16
grayscale directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm.multi_slm_controller import (
    _build_control,
    _reset_context_widgets,
    generate_phase_gray,
)


# ---------------------------------------------------------------------------
# Mock SLM helpers
# ---------------------------------------------------------------------------

def _mock_slm(
    width: int = 1920,
    height: int = 1200,
    bits: int = 10,
    pixel_pitch_um: float = 8.0,
    wavelength: int = 1064,
) -> MagicMock:
    """Create a minimal mock SLM that satisfies ``generate_phase_gray``."""
    slm = MagicMock()
    slm.Panel_Res = (width, height)
    slm.Pitch_um = pixel_pitch_um
    slm.Gray_Scale_bits = bits
    slm.wavelength = wavelength
    return slm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGeneratePhaseGrayFlatPhase:
    """Test the "平场" (flat phase) branch of ``generate_phase_gray``."""

    SLM = _mock_slm()

    @pytest.mark.parametrize("gray", [0, 1, 127, 512, 1023])
    def test_returns_correct_dtype_and_shape(self, gray: int):
        """Output must be uint16 with SLM panel dimensions."""
        result = generate_phase_gray(self.SLM, "平场", {"flat_gray": gray})
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint16, (
            f"Expected uint16, got {result.dtype}"
        )
        assert result.shape == (1200, 1920), (
            f"Expected (1200, 1920), got {result.shape}"
        )

    @pytest.mark.parametrize("gray", [0, 1, 127, 512, 1023])
    def test_returns_uniform_fill(self, gray: int):
        """Every pixel must equal the requested gray value."""
        result = generate_phase_gray(self.SLM, "平场", {"flat_gray": gray})
        assert np.all(result == gray), (
            f"Not all pixels equal {gray}: "
            f"min={result.min()}, max={result.max()}"
        )

    def test_output_is_literal_uint16_not_from_radian_conversion(self):
        """Regression: input=100 must produce output=100, not ~937.

        Pre-fix ``create_phase_from_array`` would treat 100 as radians,
        compute 100 mod 2π ≈ 5.75, then map to grayscale ≈ 937.
        """
        result = generate_phase_gray(self.SLM, "平场", {"flat_gray": 100})
        assert np.all(result == 100), (
            f"Expected 100, got {result.max()} — radian conversion leak"
        )

    def test_zero_to_max_gray_span(self):
        """Verify full grayscale span 0..1023 produces different images."""
        g0 = generate_phase_gray(self.SLM, "平场", {"flat_gray": 0})
        g512 = generate_phase_gray(self.SLM, "平场", {"flat_gray": 512})
        g1023 = generate_phase_gray(self.SLM, "平场", {"flat_gray": 1023})

        assert np.all(g0 == 0)
        assert np.all(g512 == 512)
        assert np.all(g1023 == 1023)

        # Different gray values → different images
        assert not np.array_equal(g0, g512)
        assert not np.array_equal(g512, g1023)

    def test_does_not_call_create_phase_from_array_on_slm(self):
        """The fix should NOT invoke slm.create_phase_from_array."""
        slm = _mock_slm()
        generate_phase_gray(slm, "平场", {"flat_gray": 0})
        slm.create_phase_from_array.assert_not_called()

    def test_different_gray_values_produce_different_arrays(self):
        """Sanity: two different gray values must not return identical arrays."""
        r1 = generate_phase_gray(self.SLM, "平场", {"flat_gray": 10})
        r2 = generate_phase_gray(self.SLM, "平场", {"flat_gray": 20})
        assert not np.array_equal(r1, r2), (
            "Two different gray values produced identical arrays"
        )


class TestGeneratePhaseGrayOtherPatterns:
    """Smoke tests that other pattern types still route through radian path.

    These are NOT comprehensive — just ensure we didn't break other branches.
    """

    SLM = _mock_slm()

    @pytest.fixture
    def _realistic_slm(self):
        """SLM mock whose ``create_phase_from_array`` returns real uint16.

        The real method wraps radians → [0, 2π) → [0, max_grayscale].
        Our mock mimics this so smoke tests can verify dtype/shape.
        """
        slm = _mock_slm()
        max_gray = 1023

        def _fake_create_phase_from_array(phase_rad: np.ndarray) -> np.ndarray:
            wrapped = np.mod(np.asarray(phase_rad, dtype=np.float64), 2 * np.pi)
            return np.round(wrapped / (2 * np.pi) * max_gray).astype(np.uint16)

        slm.create_phase_from_array = _fake_create_phase_from_array
        return slm

    def test_linear_grating_returns_uint16(self, _realistic_slm):
        """Linear grating still produces uint16 output."""
        result = generate_phase_gray(
            _realistic_slm, "线性光栅", {"period": 8, "phase_range": 2}
        )
        assert result.dtype == np.uint16
        assert result.shape == (1200, 1920)

    def test_circular_grating_returns_uint16(self, _realistic_slm):
        """Circular grating still produces uint16 output."""
        result = generate_phase_gray(
            _realistic_slm, "圆形光栅", {"radius": 100, "phase_range": 2}
        )
        assert result.dtype == np.uint16
        assert result.shape == (1200, 1920)


class TestResetContextWidgets:
    """Context-synced widget keys are popped from session_state so the next
    rerun re-initializes them from the new control defaults."""

    def test_reset_context_widgets_pops_synced_keys(self, monkeypatch):
        session = {
            "slm1_flat_gray": 512,
            "slm1_halfhalf_flat_gray": 512,
            "slm1_blazed_calc_wl": 1064,
            "slm1_vortex_wavelength": 1064,
            "slm1_phase_source": "暂无",
        }
        monkeypatch.setattr(st, "session_state", session)
        _reset_context_widgets(1)
        assert "slm1_flat_gray" not in session
        assert "slm1_halfhalf_flat_gray" not in session
        assert "slm1_blazed_calc_wl" not in session
        assert "slm1_vortex_wavelength" not in session
        # Unrelated keys survive.
        assert session["slm1_phase_source"] == "暂无"

    def test_build_control_connected_propagates_max_gray(self, monkeypatch):
        slm = _mock_slm(width=128, height=96)
        slm._max_gray = 993
        monkeypatch.setattr(st, "session_state", {"slm1": slm})
        ctrl = _build_control("平场", 1)
        assert ctrl.max_gray == 993
