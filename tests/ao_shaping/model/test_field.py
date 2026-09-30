"""Tests for ao_shaping.model.field data structures."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ao_shaping.model import (
    AmplitudeMap,
    ComplexField,
    PhaseMap,
    FieldMetadata,
    closest_opt_band,
    oopao_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_phase(shape=(128, 128)) -> np.ndarray:
    """Simple defocus-like phase map (radians)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (min(h, w) / 2)
    phase = np.zeros(shape, dtype=np.float64)
    pupil = rho <= 1.0
    phase[pupil] = (2.0 * rho[pupil] ** 2 - 1.0) * 1.5
    return phase


def _make_amplitude(shape=(128, 128)) -> np.ndarray:
    """Gaussian-like amplitude map."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    sigma = min(h, w) / 6.0
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# FieldMetadata
# ---------------------------------------------------------------------------
class TestFieldMetadata:
    def test_defaults(self):
        md = FieldMetadata()
        assert isinstance(md.timestamp, datetime)
        assert md.source == "unknown"
        assert md.wavelength_m is None
        assert md.extra == {}

    def test_custom_values(self):
        md = FieldMetadata(
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            source="Santec SLM-200 #1",
            wavelength_m=1064e-9,
            extra={"temperature_c": 25.0},
        )
        assert md.source == "Santec SLM-200 #1"
        assert md.wavelength_m == 1064e-9
        assert md.extra["temperature_c"] == 25.0

    def test_round_trip_serialisation(self):
        md = FieldMetadata(
            timestamp=datetime(2024, 6, 1, 8, 30, 0),
            source="sim.digitaltwin",
            wavelength_m=1550e-9,
            extra={"seed": 42, "model": "Atmosphere"},
        )
        d = md.to_dict()
        assert isinstance(d["timestamp"], str)
        md2 = FieldMetadata.from_dict(d)
        assert md2.timestamp == md.timestamp
        assert md2.source == md.source
        assert md2.wavelength_m == md.wavelength_m
        assert md2.extra == md.extra

    def test_from_dict_defaults(self):
        md = FieldMetadata.from_dict({})
        assert isinstance(md.timestamp, datetime)
        assert md.source == "unknown"
        assert md.wavelength_m is None


# ---------------------------------------------------------------------------
# PhaseMap
# ---------------------------------------------------------------------------
class TestPhaseMap:
    def test_construction(self):
        phase = _make_phase()
        pm = PhaseMap(phase=phase, pitch_size_um=8.0)
        assert pm.phase.shape == (128, 128)
        assert pm.phase.dtype == np.float64
        assert pm.pitch_size_um == 8.0
        assert pm.metadata.source == "unknown"

    def test_rejects_1d_array(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            PhaseMap(phase=np.zeros(10), pitch_size_um=8.0)

    def test_rejects_3d_array(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            PhaseMap(phase=np.zeros((10, 10, 2)), pitch_size_um=8.0)

    def test_rejects_nan(self):
        bad = np.ones((10, 10))
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            PhaseMap(phase=bad, pitch_size_um=8.0)

    def test_rejects_inf(self):
        bad = np.ones((10, 10))
        bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            PhaseMap(phase=bad, pitch_size_um=8.0)

    def test_rejects_negative_pitch(self):
        with pytest.raises(ValueError, match="pitch_size_um must be positive"):
            PhaseMap(phase=np.zeros((10, 10)), pitch_size_um=-1.0)

    def test_rejects_zero_pitch(self):
        with pytest.raises(ValueError, match="pitch_size_um must be positive"):
            PhaseMap(phase=np.zeros((10, 10)), pitch_size_um=0.0)

    def test_shape_property(self):
        pm = PhaseMap(phase=_make_phase((64, 96)), pitch_size_um=8.0)
        assert pm.shape == (64, 96)
        assert pm.size == 64 * 96

    def test_dtype_property(self):
        pm = PhaseMap(phase=_make_phase(), pitch_size_um=8.0)
        assert pm.dtype == np.float64

    def test_array_is_copied(self):
        """Mutating the input array after construction must not affect PhaseMap."""
        phase = _make_phase((16, 16))
        original = phase.copy()
        pm = PhaseMap(phase=phase, pitch_size_um=8.0)
        phase[0, 0] = 999.0
        assert pm.phase[0, 0] != 999.0
        np.testing.assert_array_equal(pm.phase, original)

    def test_to_opd(self):
        phase = np.full((10, 10), np.pi / 2)
        pm = PhaseMap(phase=phase, pitch_size_um=8.0)
        opd = pm.to_opd(wavelength_m=1064e-9)
        # OPD = phase * λ / (2π) = (π/2) * 1064e-9 / (2π) = 1064e-9 / 4
        expected = 1064e-9 / 4.0
        np.testing.assert_allclose(opd, expected)

    def test_to_oopao_opd_alias(self):
        pm = PhaseMap(phase=np.zeros((10, 10)), pitch_size_um=8.0)
        np.testing.assert_array_equal(pm.to_opd(500e-9), pm.to_oopao_opd(500e-9))

    def test_to_oopao_phase(self):
        phase = _make_phase((10, 10))
        pm = PhaseMap(phase=phase, pitch_size_um=8.0)
        result = pm.to_oopao_phase()
        np.testing.assert_array_equal(result, phase)
        # returned copy, not a view
        assert result is not pm.phase

    def test_to_oopao_source_without_oopao_raises(self):
        if oopao_available():
            pytest.skip("OOPAO is available; test only runs when unavailable")
        pm = PhaseMap(phase=np.zeros((10, 10)), pitch_size_um=8.0)
        with pytest.raises(RuntimeError, match="OOPAO is not available"):
            pm.to_oopao_source(500e-9)

    def test_to_oopao_source_with_oopao(self):
        if not oopao_available():
            pytest.skip("OOPAO not installed")
        phase = _make_phase((32, 32))
        pm = PhaseMap(
            phase=phase,
            pitch_size_um=8.0,
            metadata=FieldMetadata(wavelength_m=1064e-9, source="sim"),
        )
        src = pm.to_oopao_source(wavelength_m=1064e-9)
        assert src is not None
        np.testing.assert_allclose(src.phase, phase, rtol=1e-10)

    def test_from_oopao_source_without_oopao(self):
        if oopao_available():
            pytest.skip("OOPAO is available")
        with pytest.raises((RuntimeError, ValueError, AttributeError, TypeError)):
            PhaseMap.from_oopao_source(None, pitch_size_um=8.0)

    def test_repr(self):
        pm = PhaseMap(phase=np.zeros((10, 10)), pitch_size_um=8.0)
        r = repr(pm)
        assert "PhaseMap" in r
        assert "(10, 10)" in r
        assert "8.0" in r

    def test_from_list_coerced_to_array(self):
        data = [[0.0, 1.0], [2.0, 3.0]]
        pm = PhaseMap(phase=data, pitch_size_um=8.0)
        assert isinstance(pm.phase, np.ndarray)
        assert pm.phase.shape == (2, 2)


# ---------------------------------------------------------------------------
# AmplitudeMap
# ---------------------------------------------------------------------------
class TestAmplitudeMap:
    def test_construction(self):
        amp = _make_amplitude()
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        assert am.amplitude.shape == (128, 128)
        assert am.pitch_size_um == 8.0

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            AmplitudeMap(amplitude=np.ones(10), pitch_size_um=8.0)

    def test_rejects_nan(self):
        bad = np.ones((10, 10))
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            AmplitudeMap(amplitude=bad, pitch_size_um=8.0)

    def test_rejects_negative_values(self):
        bad = np.ones((10, 10))
        bad[0, 0] = -0.5
        with pytest.raises(ValueError, match="non-negative"):
            AmplitudeMap(amplitude=bad, pitch_size_um=8.0)

    def test_rejects_negative_pitch(self):
        with pytest.raises(ValueError, match="pitch_size_um must be positive"):
            AmplitudeMap(amplitude=np.ones((10, 10)), pitch_size_um=-1.0)

    def test_shape_property(self):
        am = AmplitudeMap(amplitude=_make_amplitude((64, 96)), pitch_size_um=8.0)
        assert am.shape == (64, 96)

    def test_intensity_property(self):
        amp = np.full((10, 10), 2.0)
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        np.testing.assert_allclose(am.intensity, 4.0)

    def test_total_power_property(self):
        amp = np.full((10, 10), 3.0)
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        # total power = sum(intensity) = 100 * 9 = 900
        assert am.total_power == 900.0

    def test_to_intensity(self):
        amp = np.full((10, 10), 2.0)
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        result = am.to_intensity()
        np.testing.assert_array_equal(result, 4.0)
        assert result is not am.intensity  # copy

    def test_to_oopao_intensity_alias(self):
        am = AmplitudeMap(amplitude=np.ones((10, 10)), pitch_size_um=8.0)
        np.testing.assert_array_equal(am.to_intensity(), am.to_oopao_intensity())

    def test_to_oopao_amplitude(self):
        amp = _make_amplitude((10, 10))
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        result = am.to_oopao_amplitude()
        np.testing.assert_array_equal(result, amp)

    def test_to_oopao_source_without_oopao_raises(self):
        if oopao_available():
            pytest.skip("OOPAO is available")
        am = AmplitudeMap(amplitude=np.ones((10, 10)), pitch_size_um=8.0)
        with pytest.raises(RuntimeError, match="OOPAO is not available"):
            am.to_oopao_source(500e-9)

    def test_to_oopao_source_with_oopao(self):
        if not oopao_available():
            pytest.skip("OOPAO not installed")
        amp = _make_amplitude((32, 32))
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        src = am.to_oopao_source(wavelength_m=1064e-9)
        assert src is not None
        np.testing.assert_allclose(src.intensity, amp**2, rtol=1e-10)

    def test_from_oopao_source_with_oopao(self):
        if not oopao_available():
            pytest.skip("OOPAO not installed")
        amp = _make_amplitude((32, 32))
        am = AmplitudeMap(amplitude=amp, pitch_size_um=8.0)
        src = am.to_oopao_source(wavelength_m=1064e-9)
        am2 = AmplitudeMap.from_oopao_source(src, pitch_size_um=8.0)
        np.testing.assert_allclose(am2.amplitude, amp, rtol=1e-8)

    def test_repr(self):
        am = AmplitudeMap(amplitude=np.ones((10, 10)), pitch_size_um=8.0)
        r = repr(am)
        assert "AmplitudeMap" in r


# ---------------------------------------------------------------------------
# ComplexField
# ---------------------------------------------------------------------------
class TestComplexField:
    def test_construction_from_complex(self):
        field = np.ones((32, 32), dtype=np.complex128)
        cf = ComplexField(field=field, pitch_size_um=8.0)
        assert cf.field.shape == (32, 32)
        assert cf.field.dtype == np.complex128

    def test_construction_from_real_promoted_to_complex(self):
        field = np.ones((10, 10), dtype=np.float64)
        cf = ComplexField(field=field, pitch_size_um=8.0)
        assert np.iscomplexobj(cf.field)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            ComplexField(field=np.ones(10, dtype=np.complex128), pitch_size_um=8.0)

    def test_rejects_nan(self):
        bad = np.ones((10, 10), dtype=np.complex128)
        bad[0, 0] = np.nan + 1j * 0
        with pytest.raises(ValueError, match="non-finite"):
            ComplexField(field=bad, pitch_size_um=8.0)

    def test_rejects_inf(self):
        bad = np.ones((10, 10), dtype=np.complex128)
        bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            ComplexField(field=bad, pitch_size_um=8.0)

    def test_rejects_negative_pitch(self):
        with pytest.raises(ValueError, match="pitch_size_um must be positive"):
            ComplexField(
                field=np.ones((10, 10), dtype=np.complex128), pitch_size_um=-1.0
            )

    def test_from_phase_amplitude(self):
        phase = _make_phase((32, 32))
        amp = _make_amplitude((32, 32))
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=amp, pitch_size_um=8.0
        )
        np.testing.assert_allclose(cf.amplitude, amp, rtol=1e-10)
        np.testing.assert_allclose(cf.phase, phase, rtol=1e-10, atol=1e-10)

    def test_from_phase_amplitude_shape_mismatch(self):
        phase = np.zeros((32, 32))
        amp = np.ones((16, 16))
        with pytest.raises(ValueError, match="must have the same shape"):
            ComplexField.from_phase_amplitude(
                phase=phase, amplitude=amp, pitch_size_um=8.0
            )

    def test_from_phase_amplitude_rejects_negative_amplitude(self):
        phase = np.zeros((10, 10))
        amp = -np.ones((10, 10))
        with pytest.raises(ValueError, match="non-negative"):
            ComplexField.from_phase_amplitude(
                phase=phase, amplitude=amp, pitch_size_um=8.0
            )

    def test_from_phase_amplitude_with_metadata(self):
        phase = np.zeros((16, 16))
        amp = np.ones((16, 16))
        md = FieldMetadata(source="test", wavelength_m=500e-9)
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=amp, pitch_size_um=8.0, metadata=md
        )
        assert cf.metadata.source == "test"
        assert cf.metadata.wavelength_m == 500e-9

    def test_shape_property(self):
        cf = ComplexField(
            field=np.ones((64, 96), dtype=np.complex128), pitch_size_um=8.0
        )
        assert cf.shape == (64, 96)

    def test_amplitude_property(self):
        phase = np.full((10, 10), np.pi / 4)
        amp = np.full((10, 10), 2.0)
        cf = ComplexField.from_phase_amplitude(phase, amp, pitch_size_um=8.0)
        np.testing.assert_allclose(cf.amplitude, 2.0)

    def test_phase_property(self):
        phase = np.full((10, 10), np.pi / 4)
        amp = np.ones((10, 10))
        cf = ComplexField.from_phase_amplitude(phase, amp, pitch_size_um=8.0)
        np.testing.assert_allclose(cf.phase, np.pi / 4)

    def test_intensity_property(self):
        phase = np.zeros((10, 10))
        amp = np.full((10, 10), 3.0)
        cf = ComplexField.from_phase_amplitude(phase, amp, pitch_size_um=8.0)
        np.testing.assert_allclose(cf.intensity, 9.0)

    def test_total_power_property(self):
        amp = np.full((10, 10), 3.0)
        cf = ComplexField(field=amp.astype(np.complex128), pitch_size_um=8.0)
        assert cf.total_power == 900.0

    def test_peak_intensity_property(self):
        field = np.ones((10, 10), dtype=np.complex128)
        field[5, 5] = 3.0 + 4.0j  # intensity = 25 at this pixel
        cf = ComplexField(field=field, pitch_size_um=8.0)
        assert cf.peak_intensity == 25.0

    def test_phase_unwrapped(self):
        phase = _make_phase((32, 32))
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=np.ones((32, 32)), pitch_size_um=8.0
        )
        unwrapped = cf.phase_unwrapped
        assert unwrapped.shape == phase.shape

    def test_to_phase(self):
        phase = np.full((10, 10), np.pi / 3)
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=np.ones((10, 10)), pitch_size_um=8.0
        )
        result = cf.to_phase()
        np.testing.assert_allclose(result, phase, atol=1e-10)
        assert result is not cf.phase

    def test_to_amplitude(self):
        amp = _make_amplitude((10, 10))
        cf = ComplexField.from_phase_amplitude(
            phase=np.zeros((10, 10)), amplitude=amp, pitch_size_um=8.0
        )
        result = cf.to_amplitude()
        np.testing.assert_allclose(result, amp, rtol=1e-10)

    def test_to_intensity(self):
        amp = np.full((10, 10), 2.0)
        cf = ComplexField.from_phase_amplitude(
            phase=np.zeros((10, 10)), amplitude=amp, pitch_size_um=8.0
        )
        result = cf.to_intensity()
        np.testing.assert_allclose(result, 4.0)

    def test_to_opd(self):
        phase = np.full((10, 10), np.pi)
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=np.ones((10, 10)), pitch_size_um=8.0
        )
        opd = cf.to_opd(wavelength_m=1064e-9)
        # OPD = π * 1064e-9 / (2π) = 1064e-9 / 2
        np.testing.assert_allclose(opd, 1064e-9 / 2.0)

    def test_to_oopao_source_without_oopao_raises(self):
        if oopao_available():
            pytest.skip("OOPAO is available")
        cf = ComplexField(
            field=np.ones((10, 10), dtype=np.complex128),
            pitch_size_um=8.0,
        )
        with pytest.raises(RuntimeError, match="OOPAO is not available"):
            cf.to_oopao_source(500e-9)

    def test_to_oopao_source_with_oopao(self):
        if not oopao_available():
            pytest.skip("OOPAO not installed")
        phase = _make_phase((32, 32))
        amp = _make_amplitude((32, 32))
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=amp, pitch_size_um=8.0
        )
        src = cf.to_oopao_source(wavelength_m=1064e-9)
        assert src is not None
        np.testing.assert_allclose(src.phase, phase, rtol=1e-10)
        np.testing.assert_allclose(src.intensity, amp**2, rtol=1e-10)

    def test_from_oopao_source_with_oopao(self):
        if not oopao_available():
            pytest.skip("OOPAO not installed")
        phase = _make_phase((32, 32))
        amp = _make_amplitude((32, 32))
        cf = ComplexField.from_phase_amplitude(
            phase=phase, amplitude=amp, pitch_size_um=8.0
        )
        src = cf.to_oopao_source(wavelength_m=1064e-9)
        cf2 = ComplexField.from_oopao_source(src, pitch_size_um=8.0)
        np.testing.assert_allclose(cf2.phase, phase, atol=1e-8)
        np.testing.assert_allclose(cf2.amplitude, amp, atol=1e-6)

    def test_repr(self):
        cf = ComplexField(
            field=np.ones((10, 10), dtype=np.complex128),
            pitch_size_um=8.0,
        )
        r = repr(cf)
        assert "ComplexField" in r


# ---------------------------------------------------------------------------
# closest_opt_band
# ---------------------------------------------------------------------------
class TestClosestOptBand:
    def test_eos_1064nm(self):
        assert closest_opt_band(1064e-9) == "EOS"

    def test_h_1654nm(self):
        assert closest_opt_band(1654e-9) == "H"

    def test_j2_1550nm(self):
        assert closest_opt_band(1550e-9) == "J2"

    def test_r_640nm(self):
        assert closest_opt_band(640e-9) == "R"

    def test_b_440nm(self):
        assert closest_opt_band(440e-9) == "B"

    def test_close_to_boundary(self):
        # 645 nm is halfway between R (640) and R2 (650) — R wins by tie-break?
        # Actually 645 is 5nm from both; min() picks the first match in the
        # list, which is R at 640nm.
        result = closest_opt_band(645e-9)
        assert result in ("R", "R2")

    def test_far_from_all_bands(self):
        result = closest_opt_band(5000e-9)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# OOPAO availability
# ---------------------------------------------------------------------------
class TestOopaoAvailability:
    def test_returns_bool(self):
        assert isinstance(oopao_available(), bool)

    def test_available_when_oopao_importable(self):
        # This test passes if OOPAO is installed; if not, it should be False
        if oopao_available():
            from ao_shaping.drivers.sim._oopao_compat import Source

            assert Source is not None
        else:
            assert oopao_available() is False
