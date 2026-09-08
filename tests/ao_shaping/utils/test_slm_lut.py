"""Offline tests for the pure-math SLM gray-to-phase LUT module."""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.slm_lut import (
    LUTData,
    blaze_ramp_gray,
    build_inverse_lut,
    depth_pattern,
    invert_depth_scan,
    invert_offset_scan,
    load_lut,
    normalize_efficiency,
    offset_pattern,
    save_lut,
    sinc_inv,
    stack_halves,
)


class TestBlazeRampGray:
    """Tests for blaze_ramp_gray."""

    def test_shape_and_range(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        assert ramp.shape == (1, width)
        assert ramp.min() >= 0
        assert ramp.max() <= peak_gray

    def test_max_equals_peak_gray(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        assert ramp.max() == peak_gray

    def test_monotonic_within_period(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        first = ramp[0, :period]
        assert np.all(np.diff(first) >= 0)

    def test_period_repeats(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        np.testing.assert_array_equal(ramp[0, :period], ramp[0, period : 2 * period])


class TestDepthPattern:
    """Tests for depth_pattern."""

    def test_dtype_and_shape(self):
        pattern = depth_pattern(period=16, peak_gray=1023, height=100, width=200)
        assert pattern.dtype == np.uint16
        assert pattern.shape == (100, 200)

    def test_value_range(self):
        peak_gray = 1023
        pattern = depth_pattern(period=16, peak_gray=peak_gray, height=50, width=100)
        assert pattern.min() >= 0
        assert pattern.max() <= peak_gray

    def test_rows_identical(self):
        pattern = depth_pattern(period=16, peak_gray=1023, height=10, width=64)
        for r in range(1, pattern.shape[0]):
            np.testing.assert_array_equal(pattern[r], pattern[0])


class TestOffsetPattern:
    """Tests for offset_pattern."""

    def test_dtype_and_shape(self):
        pattern = offset_pattern(
            period=16, peak_gray=1023, gray_offset=100, bits=10, height=100, width=200
        )
        assert pattern.dtype == np.uint16
        assert pattern.shape == (100, 200)

    def test_value_range_within_bits(self):
        bits = 10
        pattern = offset_pattern(
            period=16, peak_gray=1023, gray_offset=100, bits=bits, height=50, width=100
        )
        assert pattern.min() >= 0
        assert pattern.max() < 2**bits

    def test_modulo_wrap(self):
        # gray_offset pushes values past 2**bits -> wraps around
        bits = 10
        period, peak_gray, gray_offset = 16, 1023, 900
        pattern = offset_pattern(
            period=period,
            peak_gray=peak_gray,
            gray_offset=gray_offset,
            bits=bits,
            height=1,
            width=period,
        )
        ramp = blaze_ramp_gray(period, peak_gray, period)
        expected = np.mod(ramp + gray_offset, 2**bits).astype(np.uint16)
        np.testing.assert_array_equal(pattern[0], expected[0])


class TestStackHalves:
    """Tests for stack_halves."""

    def test_stack_vertical(self):
        h, w = 20, 30
        ref = np.full((h // 2, w), 1, dtype=np.uint16)
        test = np.full((h // 2, w), 2, dtype=np.uint16)
        out = stack_halves(ref, test)
        assert out.shape == (h, w)
        np.testing.assert_array_equal(out[: h // 2], ref)
        np.testing.assert_array_equal(out[h // 2 :], test)

    def test_mismatched_axis_raises(self):
        ref = np.zeros((10, 20))
        test = np.zeros((10, 25))  # width mismatch
        with pytest.raises(ValueError):
            stack_halves(ref, test)


class TestNormalizeEfficiency:
    """Tests for normalize_efficiency."""

    def test_normalize(self):
        eta = np.array([0.0, 2.0, 4.0, 1.0])
        out = normalize_efficiency(eta)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 0.25])

    def test_zero_max_returns_zeros(self):
        eta = np.array([0.0, 0.0, 0.0])
        out = normalize_efficiency(eta)
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_negative_max_returns_zeros(self):
        eta = np.array([-1.0, -2.0])
        out = normalize_efficiency(eta)
        np.testing.assert_array_equal(out, np.zeros(2))


class TestSincInv:
    """Tests for sinc_inv."""

    def test_inverse_roundtrip(self):
        for y in [0.01, 0.1, 0.5, 0.9, 0.999, 1.0]:
            x = sinc_inv(np.array([y]))[0]
            assert x <= 0
            np.testing.assert_allclose(np.sinc(x), y, atol=1e-6)

    def test_y_equals_one_is_zero(self):
        x = sinc_inv(np.array([1.0]))[0]
        assert x == 0.0

    def test_y_leq_zero_is_neg_one(self):
        x = sinc_inv(np.array([0.0]))[0]
        assert x == -1.0

    def test_y_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            sinc_inv(np.array([1.5]))

    def test_nan_propagates(self):
        x = sinc_inv(np.array([np.nan]))[0]
        assert np.isnan(x)

    def test_vectorized(self):
        ys = np.array([0.1, 0.5, 0.9])
        xs = sinc_inv(ys)
        np.testing.assert_allclose(np.sinc(xs), ys, atol=1e-6)


class TestInvertDepthScan:
    """Tests for invert_depth_scan (the critical recovery test)."""

    def test_recovers_linear_response(self):
        gray_for_2pi = 993
        g_values = np.arange(0, 1024, 16, dtype=float)
        phi_lin = 2 * np.pi * g_values / gray_for_2pi
        # Simulate blazed-grating efficiency with tiny noise.
        u = (phi_lin - 2 * np.pi) / (2 * np.pi)
        rng = np.random.default_rng(0)
        eta = np.sinc(u) ** 2 + 1e-6 * rng.standard_normal(g_values.size)
        phi_rec = invert_depth_scan(eta)
        np.testing.assert_allclose(phi_rec, phi_lin, atol=2e-2)

    def test_peak_is_two_pi(self):
        eta = np.array([0.0, 0.5, 1.0, 0.5, 0.2])
        phi = invert_depth_scan(eta)
        assert phi[2] == pytest.approx(2 * np.pi)

    def test_output_float64(self):
        eta = np.array([0.0, 0.5, 1.0, 0.5, 0.2])
        phi = invert_depth_scan(eta)
        assert phi.dtype == np.float64


class TestInvertOffsetScan:
    """Tests for invert_offset_scan."""

    def test_flat_eta_monotone_and_arcsin(self):
        gray_for_2pi = 993
        # Scan within [0, gray_for_2pi] so no saturation region is entered.
        g_values = np.arange(0, gray_for_2pi + 1, 16, dtype=float)
        eta = np.full(g_values.size, 5.0)  # constant
        phi = invert_offset_scan(eta, g_values, gray_for_2pi)
        assert np.all(np.diff(phi) >= 0)
        # dphi = 2*arcsin(sqrt(1)) = pi for constant eta
        i = 10
        expected = 2 * np.pi * (g_values[i] / gray_for_2pi) + np.pi
        assert phi[i] == pytest.approx(expected, rel=1e-6)

    def test_saturation_after_gray_for_2pi(self):
        gray_for_2pi = 993
        g_values = np.arange(0, 1024, 16, dtype=float)
        eta = np.full(g_values.size, 5.0)
        phi = invert_offset_scan(eta, g_values, gray_for_2pi)
        mask = g_values > gray_for_2pi
        if np.any(mask):
            np.testing.assert_allclose(phi[mask], 2 * np.pi, atol=1e-6)


class TestBuildInverseLut:
    """Tests for build_inverse_lut."""

    def test_monotone_phi(self):
        g_values = np.arange(0, 1024, 16)
        phi = 2 * np.pi * g_values / 993.0
        t, gray_lut = build_inverse_lut(phi, g_values, n=256)
        assert gray_lut.dtype == np.uint16
        assert t.shape == (256,)
        assert gray_lut.shape == (256,)
        assert np.all(np.diff(gray_lut) >= 0)
        # matches argmin rule
        for i in range(256):
            idx = np.argmin(np.abs(phi - t[i]))
            assert gray_lut[i] == g_values[idx]

    def test_non_monotone_raises(self):
        phi = np.array([0.0, 3.0, 2.0, 4.0])
        g_values = np.arange(4)
        with pytest.raises(ValueError):
            build_inverse_lut(phi, g_values)


class TestSaveLoadLut:
    """Tests for save_lut / load_lut roundtrip."""

    def test_roundtrip_npz(self, tmp_path):
        g_values = np.arange(0, 1024, 16, dtype=float)
        phi = 2 * np.pi * g_values / 993.0
        inverse_gray = np.arange(64, dtype=np.uint16)
        meta = {"gray_for_2pi": 993, "wavelength_nm": 1064}
        save_lut(tmp_path, g_values, phi, inverse_gray, meta)
        data = load_lut(tmp_path)
        assert isinstance(data, LUTData)
        np.testing.assert_array_equal(data.g_values, g_values)
        np.testing.assert_array_equal(data.phi, phi)
        np.testing.assert_array_equal(data.inverse_gray, inverse_gray)
        assert data.meta == meta

    def test_roundtrip_csv_fallback(self, tmp_path):
        g_values = np.arange(0, 1024, 16, dtype=float)
        phi = 2 * np.pi * g_values / 993.0
        inverse_gray = np.arange(64, dtype=np.uint16)
        save_lut(tmp_path, g_values, phi, inverse_gray)
        # Remove npz to force CSV fallback.
        (tmp_path / "lut.npz").unlink()
        data = load_lut(tmp_path)
        np.testing.assert_allclose(data.g_values, g_values)
        np.testing.assert_allclose(data.phi, phi)
        np.testing.assert_array_equal(data.inverse_gray, inverse_gray)

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lut(tmp_path)
