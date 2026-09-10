from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.zernike_utils import (
    coefficients_to_array,
    generate_zernike_phase,
    list_zernike_modes,
    parse_zernike_coefficients,
)
from ao_shaping.utils.zernike_calc import calc_n_zernike_terms, noll_to_nm


class TestListZernikeModes:
    """Tests for list_zernike_modes()."""

    def test_n_max_1_returns_3_modes(self):
        """n_max=1 → 3 modes: piston, vertical tilt, horizontal tilt."""
        modes = list_zernike_modes(1)
        assert len(modes) == 3
        noll_indices = [m[0] for m in modes]
        assert noll_indices == [1, 2, 3]

    def test_n_max_2_returns_6_modes(self):
        """n_max=2 → 6 modes."""
        modes = list_zernike_modes(2)
        assert len(modes) == 6

    def test_n_max_4_returns_15_modes(self):
        """n_max=4 → (4+1)(4+2)/2 = 15 modes."""
        modes = list_zernike_modes(4)
        assert len(modes) == 15

    def test_returns_tuple_with_4_elements(self):
        """Each mode is (noll_index, n, m, name)."""
        modes = list_zernike_modes(2)
        for m in modes:
            assert len(m) == 4
            noll_idx, n, m_val, name = m
            assert isinstance(noll_idx, int)
            assert isinstance(n, int)
            assert isinstance(m_val, int)
            assert isinstance(name, str)

    def test_noll_index_matches_noll_to_nm(self):
        """Noll index from list should match noll_to_nm conversion."""
        modes = list_zernike_modes(4)
        for noll_idx, n, m, _name in modes:
            n_calc, m_calc = noll_to_nm(noll_idx)
            assert n == n_calc
            assert m == m_calc

    def test_sorted_by_noll_index(self):
        """Modes should be sorted by Noll index."""
        modes = list_zernike_modes(4)
        noll_indices = [m[0] for m in modes]
        assert noll_indices == sorted(noll_indices)

    def test_known_noll_mapping(self):
        """Verify specific known Noll → (n, m) mappings."""
        modes = list_zernike_modes(4)
        mode_dict = {(n, m): noll_idx for noll_idx, n, m, _ in modes}
        # Standard Noll ordering (Noll 1976 / aotools)
        assert mode_dict[(0, 0)] == 1   # Piston
        assert mode_dict[(1, 1)] == 2   # Vertical tilt
        assert mode_dict[(1, -1)] == 3  # Horizontal tilt
        assert mode_dict[(2, 0)] == 4   # Defocus
        assert mode_dict[(2, -2)] == 5  # Astigmatism (vertical)
        assert mode_dict[(2, 2)] == 6   # Astigmatism (horizontal)
        assert mode_dict[(4, 0)] == 11  # Spherical


class TestParseZernikeCoefficients:
    """Tests for parse_zernike_coefficients()."""

    def test_none_input(self):
        """None → empty dict."""
        result = parse_zernike_coefficients(None)
        assert result == {}

    def test_empty_dict(self):
        """Empty dict → empty dict."""
        result = parse_zernike_coefficients({})
        assert result == {}

    def test_noll_index_str_keys(self):
        """Dict with string Noll index keys."""
        # Noll 5 = (2, -2), Noll 13 = (4, -2)
        raw = {"5": 1.0, "13": 0.5}
        result = parse_zernike_coefficients(raw)
        assert result[(2, -2)] == 1.0
        assert result[(4, -2)] == 0.5
        assert len(result) == 2

    def test_noll_index_int_keys(self):
        """Dict with integer Noll index keys."""
        raw = {5: 1.0, 13: 0.5}
        result = parse_zernike_coefficients(raw)
        assert result[(2, -2)] == 1.0
        assert result[(4, -2)] == 0.5

    def test_nm_tuple_keys(self):
        """Dict with (n, m) tuple keys passes through."""
        raw = {(2, 0): 1.0, (4, 0): 0.5}
        result = parse_zernike_coefficients(raw)
        assert result[(2, 0)] == 1.0
        assert result[(4, 0)] == 0.5
        assert len(result) == 2

    def test_list_format(self):
        """List in Noll order: index 3 = Noll 4 = (2,0) defocus."""
        # Noll 1-13: index 0=piston, 1=tilt_v, 2=tilt_h, 3=defocus, ...
        # Index 3 (0-based) = Noll 4 = (2, 0) defocus
        # Index 12 (0-based) = Noll 13 = (4, -2)
        raw = [0.0, 0.0, 0.0, 1.0] + [0.0] * 8 + [0.5]
        result = parse_zernike_coefficients(raw)
        assert (2, 0) in result
        assert result[(2, 0)] == 1.0
        assert (4, -2) in result
        assert result[(4, -2)] == 0.5

    def test_numpy_array_format(self):
        """Numpy array in Noll order."""
        arr = np.zeros(13, dtype=float)
        arr[3] = 1.0  # Noll 4 → (2, 0)
        arr[12] = 0.5  # Noll 13 → (4, -2)
        result = parse_zernike_coefficients(arr)
        assert result[(2, 0)] == 1.0
        assert result[(4, -2)] == 0.5

    def test_n_max_filters_high_order(self):
        """n_max parameter filters out modes above the specified order."""
        raw = {(2, 0): 1.0, (4, 0): 0.5, (6, 0): 0.3}
        result = parse_zernike_coefficients(raw, n_max=4)
        assert (2, 0) in result
        assert (4, 0) in result
        assert (6, 0) not in result

    def test_n_max_with_list_format(self):
        """n_max filters high-order modes in list format."""
        # Noll 15 = (4, -4) — n=4 ≤ 4, should NOT be filtered
        # Noll 16+ would be n=5+, which should be filtered
        raw = [0.0] * 20
        raw[0] = 1.0  # Noll 1 → (0, 0)
        raw[3] = 1.0  # Noll 4 → (2, 0)
        raw[14] = 1.0  # Noll 15 → (4, -4) — n=4, should stay
        raw[15] = 1.0  # Noll 16 → (5, -1) — n=5, should be filtered
        result = parse_zernike_coefficients(raw, n_max=4)
        assert (0, 0) in result
        assert (2, 0) in result
        assert (4, -4) in result
        # All modes should have n ≤ 4
        nm_keys = set(result.keys())
        assert all(n <= 4 for n, m in nm_keys)

    def test_zeros_filtered_out(self):
        """Zero-valued coefficients are excluded from result."""
        raw = [0.0, 0.0, 0.0]
        result = parse_zernike_coefficients(raw)
        assert result == {}

    def test_int_values_accepted(self):
        """Integer values are accepted and converted to float."""
        raw = {5: 1, 13: 2}
        result = parse_zernike_coefficients(raw)
        assert result[(2, -2)] == 1.0
        assert result[(4, -2)] == 2.0

    def test_invalid_noll_index_negative_skipped(self):
        """Negative Noll index is skipped."""
        raw = {-1: 1.0, 5: 1.0}
        result = parse_zernike_coefficients(raw)
        assert len(result) == 1
        assert (2, -2) in result

    def test_invalid_type_raises(self):
        """Unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported coefficient type"):
            parse_zernike_coefficients("invalid")  # type: ignore[arg-type]


class TestCoefficientsToArray:
    """Tests for coefficients_to_array()."""

    def test_round_trip_nm_dict(self):
        """(n, m) dict → array should place values at correct Noll positions."""
        n_max = 4
        nk = calc_n_zernike_terms(n_max)
        # (2, 0) = Noll 4 → index 3
        # (4, 0) = Noll 11 → index 10
        raw = {(2, 0): 1.0, (4, 0): 0.5}
        arr = coefficients_to_array(raw, n_max=n_max)
        assert arr.shape == (nk,)
        assert arr.dtype == np.float64
        assert arr[3] == 1.0   # Noll 4
        assert arr[10] == 0.5  # Noll 11

    def test_empty_dict_returns_zeros(self):
        """Empty dict → all-zero array."""
        arr = coefficients_to_array({}, n_max=4)
        assert arr.shape == (calc_n_zernike_terms(4),)
        assert np.all(arr == 0.0)

    def test_n_max_controls_array_length(self):
        """Array length matches calc_n_zernike_terms(n_max)."""
        for n_max in [1, 2, 3, 4]:
            arr = coefficients_to_array({}, n_max=n_max)
            assert arr.shape == (calc_n_zernike_terms(n_max),)

    def test_high_order_modes_outside_n_max_ignored(self):
        """Modes with n > n_max are placed outside array bounds (ignored)."""
        # (6, 0) has n=6, with n_max=4 the array has only 15 elements
        raw = {(6, 0): 1.0}
        arr = coefficients_to_array(raw, n_max=4)
        assert arr.shape == (15,)
        assert np.all(arr == 0.0)


class TestGenerateZernikePhase:
    """Tests for generate_zernike_phase()."""

    def test_output_shape(self):
        """Output shape should be (height, width)."""
        width, height = 200, 100
        coeffs = {(2, 0): 1.0}
        img = generate_zernike_phase(coeffs, resolution=(width, height), n_max=4)
        assert img.shape == (height, width)

    def test_output_is_float(self):
        """Output should be float (not uint16)."""
        coeffs = {(2, 0): 1.0}
        img = generate_zernike_phase(coeffs, resolution=(100, 100), n_max=4)
        assert np.issubdtype(img.dtype, np.floating)

    def test_empty_coefficients_returns_zeros(self):
        """No coefficients → all-zero phase."""
        img = generate_zernike_phase({}, resolution=(100, 100))
        assert np.all(img == 0)

    def test_none_coefficients_returns_zeros(self):
        """None → all-zero phase."""
        img = generate_zernike_phase(None, resolution=(100, 100))
        assert np.all(img == 0)

    def test_accepts_list_input(self):
        """List of coefficients in Noll order."""
        # Noll 4 = (2, 0) defocus → index 3
        raw = [0.0, 0.0, 0.0, 1.0]
        img = generate_zernike_phase(raw, resolution=(100, 100), n_max=4)
        assert img.shape == (100, 100)

    def test_accepts_numpy_array_input(self):
        """Numpy array of coefficients."""
        arr = np.zeros(6)
        arr[3] = 1.0  # Noll 4 → (2, 0)
        img = generate_zernike_phase(arr, resolution=(100, 100), n_max=2)
        assert img.shape == (100, 100)

    def test_accepts_noll_dict_input(self):
        """Dict with Noll index keys."""
        raw = {4: 1.0}  # Noll 4 → (2, 0)
        img = generate_zernike_phase(raw, resolution=(100, 100), n_max=4)
        assert img.shape == (100, 100)

    def test_accepts_nm_tuple_dict_input(self):
        """Dict with (n, m) tuple keys."""
        raw = {(2, 0): 1.0}
        img = generate_zernike_phase(raw, resolution=(100, 100), n_max=4)
        assert img.shape == (100, 100)

    def test_different_modes_produce_different_patterns(self):
        """Different Zernike modes should produce different phase patterns."""
        img_defocus = generate_zernike_phase(
            {(2, 0): 1.0}, resolution=(100, 100), n_max=4
        )
        img_astig = generate_zernike_phase(
            {(2, 2): 1.0}, resolution=(100, 100), n_max=4
        )
        assert not np.array_equal(img_defocus, img_astig)

    def test_custom_radius(self):
        """Custom radius parameter should not crash."""
        coeffs = {(2, 0): 1.0}
        img = generate_zernike_phase(
            coeffs, resolution=(100, 100), n_max=4, radius=30.0
        )
        assert img.shape == (100, 100)

    def test_has_nan_outside_aperture(self):
        """Pixels outside the circular aperture should be NaN."""
        coeffs = {(2, 0): 1.0}
        img = generate_zernike_phase(
            coeffs, resolution=(200, 200), n_max=4, radius=50.0
        )
        # Center pixels (inside aperture) should be finite
        assert np.isfinite(img[100, 100])
        # Corner pixels (outside aperture) should be NaN
        assert np.isnan(img[0, 0])
