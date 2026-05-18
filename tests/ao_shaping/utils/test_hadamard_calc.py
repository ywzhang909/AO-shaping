import numpy as np
import pytest

from ao_shaping.utils.hadamard_calc import (
    HadamardGenerator,
    calc_n_hadamard_modes,
    hadamard_mode_2d,
    is_hadamard_order,
)


class TestIsHadamardOrder:
    """Tests for is_hadamard_order function."""

    def test_valid_orders(self):
        """Test that powers of 2 are recognized as valid orders."""
        valid_orders = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for order in valid_orders:
            assert is_hadamard_order(order), f"{order} should be a valid order"

    def test_invalid_orders_non_power_of_2(self):
        """Test that non-powers of 2 are rejected."""
        invalid_orders = [3, 5, 6, 7, 9, 10, 12, 20, 100]
        for order in invalid_orders:
            assert not is_hadamard_order(order), f"{order} should not be a valid order"

    def test_edge_cases(self):
        """Test edge cases like 0, 1, and negative numbers."""
        assert not is_hadamard_order(0)
        assert not is_hadamard_order(1)
        assert not is_hadamard_order(-1)
        assert not is_hadamard_order(-8)


class TestCalcNHadamardModes:
    """Tests for calc_n_hadamard_modes function."""

    def test_valid_orders(self):
        """Test calculation of total modes for valid orders."""
        assert calc_n_hadamard_modes(2) == 4
        assert calc_n_hadamard_modes(4) == 16
        assert calc_n_hadamard_modes(8) == 64
        assert calc_n_hadamard_modes(16) == 256

    def test_invalid_order_raises(self):
        """Test that invalid orders raise ValueError."""
        with pytest.raises(ValueError):
            calc_n_hadamard_modes(3)
        with pytest.raises(ValueError):
            calc_n_hadamard_modes(0)
        with pytest.raises(ValueError):
            calc_n_hadamard_modes(-1)


class TestHadamardMode2D:
    """Tests for hadamard_mode_2d function."""

    def test_mode_shape(self):
        """Test that generated modes have correct shape."""
        order = 8
        mode = hadamard_mode_2d(0, 0, order)
        assert mode.shape == (order, order)

    def test_mode_values(self):
        """Test that mode values are ±1."""
        order = 8
        mode = hadamard_mode_2d(3, 5, order)
        unique_values = np.unique(mode)
        assert np.all(np.isin(unique_values, [-1, 1]))

    def test_invalid_indices(self):
        """Test that invalid indices raise ValueError."""
        with pytest.raises(ValueError):
            hadamard_mode_2d(-1, 0, 8)
        with pytest.raises(ValueError):
            hadamard_mode_2d(0, -1, 8)
        with pytest.raises(ValueError):
            hadamard_mode_2d(8, 0, 8)
        with pytest.raises(ValueError):
            hadamard_mode_2d(0, 8, 8)

    def test_orthogonality(self):
        """Test that different modes are orthogonal (dot product is 0)."""
        order = 8
        mode1 = hadamard_mode_2d(0, 0, order).ravel()
        mode2 = hadamard_mode_2d(1, 0, order).ravel()
        dot_product = np.dot(mode1, mode2)
        assert abs(dot_product) < 1e-10


class TestHadamardGeneratorInit:
    """Tests for HadamardGenerator initialization."""

    def test_default_init(self):
        """Test initialization with default parameters."""
        gen = HadamardGenerator(resolution=(1920, 1080))
        assert gen.resolution == (1920, 1080)
        assert gen.n_modes == 64  # 8x8
        assert gen.radius == 1.0

    def test_custom_mode_order(self):
        """Test initialization with custom mode order."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=16)
        assert gen.n_modes == 256  # 16x16

    def test_invalid_mode_order(self):
        """Test that invalid mode orders raise ValueError."""
        with pytest.raises(ValueError):
            HadamardGenerator(resolution=(100, 100), mode_order=3)
        with pytest.raises(ValueError):
            HadamardGenerator(resolution=(100, 100), mode_order=0)
        with pytest.raises(ValueError):
            HadamardGenerator(resolution=(100, 100), mode_order=1)

    def test_invalid_mask_type(self):
        """Test that invalid mask type raises ValueError."""
        with pytest.raises(ValueError):
            HadamardGenerator(resolution=(100, 100), mask_type="invalid")

    def test_custom_radius(self):
        """Test initialization with custom radius."""
        gen = HadamardGenerator(resolution=(100, 100), radius=0.5)
        assert gen.radius == 0.5


class TestHadamardGeneratorMask:
    """Tests for HadamardGenerator mask functionality."""

    def test_circular_mask_shape(self):
        """Test that circular mask has correct shape."""
        gen = HadamardGenerator(resolution=(100, 100), mask_type="circular")
        assert gen.mask.shape == (100, 100)

    def test_rectangular_mask_all_ones(self):
        """Test that rectangular mask is all ones."""
        gen = HadamardGenerator(resolution=(100, 100), mask_type="rectangular")
        assert np.all(gen.mask == 1)

    def test_circular_mask_center_inside(self):
        """Test that center of circular mask is inside aperture."""
        gen = HadamardGenerator(resolution=(101, 101), mask_type="circular")
        center = gen.mask[50, 50]
        assert center == 1

    def test_circular_mask_corner_outside(self):
        """Test that corners of circular mask are outside aperture."""
        gen = HadamardGenerator(resolution=(101, 101), mask_type="circular")
        corner = gen.mask[0, 0]
        assert corner == 0

    def test_custom_radius_mask(self):
        """Test that custom radius affects mask."""
        gen_small = HadamardGenerator(resolution=(101, 101), mask_type="circular", radius=0.3)
        gen_large = HadamardGenerator(resolution=(101, 101), mask_type="circular", radius=0.9)
        assert np.sum(gen_small.mask) < np.sum(gen_large.mask)


class TestHadamardGeneratorGenerate:
    """Tests for HadamardGenerator generate methods."""

    def test_set_bits_required(self):
        """Test that generate() requires set_bits() first."""
        gen = HadamardGenerator(resolution=(100, 100))
        with pytest.raises(ValueError):
            gen.generate(0)

    def test_generate_single_mode_shape(self):
        """Test that generated mode has correct shape."""
        gen = HadamardGenerator(resolution=(1920, 1080))
        gen.set_bits(10)
        mode = gen.generate(0, amplitude=0.5)
        assert mode.shape == (1080, 1920)

    def test_generate_single_mode_dtype(self):
        """Test that generated mode has correct dtype."""
        gen = HadamardGenerator(resolution=(100, 100))
        gen.set_bits(10)
        mode = gen.generate(0)
        assert mode.dtype == np.uint16

    def test_generate_single_mode_range(self):
        """Test that generated mode values are within expected range."""
        gen = HadamardGenerator(resolution=(100, 100))
        gen.set_bits(10)
        mode = gen.generate(0, amplitude=1.0)
        assert np.all(mode >= 0)
        assert np.all(mode <= 1023)  # 2^10 - 1

    def test_generate_row_col_equivalence(self):
        """Test that generate(0) equals generate_row_col(0, 0)."""
        gen = HadamardGenerator(resolution=(100, 100))
        gen.set_bits(10)
        mode1 = gen.generate(0)
        mode2 = gen.generate_row_col(0, 0)
        np.testing.assert_array_equal(mode1, mode2)

    def test_mode_index_range(self):
        """Test that out-of-range mode indices raise IndexError."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=8)
        gen.set_bits(10)
        with pytest.raises(IndexError):
            gen.generate(64)  # n_modes = 64, valid indices are 0-63
        with pytest.raises(IndexError):
            gen.generate(-1)

    def test_generate_row_col_invalid(self):
        """Test that invalid row/col indices raise ValueError."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=8)
        gen.set_bits(10)
        with pytest.raises(ValueError):
            gen.generate_row_col(-1, 0)
        with pytest.raises(ValueError):
            gen.generate_row_col(8, 0)


class TestHadamardGeneratorGenerateModes:
    """Tests for HadamardGenerator generate_modes methods."""

    def test_generate_modes_shape(self):
        """Test that combined mode has correct shape."""
        gen = HadamardGenerator(resolution=(100, 100))
        gen.set_bits(10)
        coeffs = np.random.randn(64)
        phase = gen.generate_modes(coeffs)
        assert phase.shape == (100, 100)

    def test_generate_modes_empty(self):
        """Test that zero coefficients produce zero phase."""
        gen = HadamardGenerator(resolution=(100, 100))
        gen.set_bits(10)
        coeffs = np.zeros(64)
        phase = gen.generate_modes(coeffs)
        # Should be mid-range (0.5 * max_val) after normalization
        expected = (2**10 - 1) / 2
        assert np.allclose(phase, expected, atol=1)

    def test_generate_modes_dict(self):
        """Test generation from coefficient dictionary."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=8)
        gen.set_bits(10)
        coeffs = {(0, 0): 0.5, (1, 2): 0.3}
        phase = gen.generate_modes_dict(coeffs)
        assert phase.shape == (100, 100)
        assert phase.dtype == np.uint16

    def test_generate_modes_dict_invalid_key(self):
        """Test that invalid dictionary keys raise ValueError."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=8)
        gen.set_bits(10)
        with pytest.raises(ValueError):
            gen.generate_modes_dict({(10, 0): 1.0})  # u=10 is out of range


class TestHadamardGeneratorOrthogonality:
    """Tests for Walsh-Hadamard mode orthogonality."""

    def test_orthogonality(self):
        """Test that different modes are orthogonal."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=8)
        gen.set_bits(10)
        mode1 = gen.generate(0).astype(np.float64)
        mode2 = gen.generate(1).astype(np.float64)

        # Normalize to [-1, 1] for orthogonality test
        max_val = 2**10 - 1
        mode1_norm = (mode1 / max_val) * 2 - 1
        mode2_norm = (mode2 / max_val) * 2 - 1

        # Apply mask
        mask = gen.mask > 0
        dot_product = np.sum(mode1_norm[mask] * mode2_norm[mask])
        assert abs(dot_product) < 100  # Allow some numerical error


class TestHadamardGeneratorProperties:
    """Tests for HadamardGenerator properties."""

    def test_n_modes(self):
        """Test n_modes property."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=16)
        assert gen.n_modes == 256

    def test_hadamard_matrix_shape(self):
        """Test hadamard_matrix property shape."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=8)
        H = gen.hadamard_matrix
        assert H.shape == (8, 8)
        assert np.all(np.isin(np.unique(H), [-1, 1]))

    def test_resolution_property(self):
        """Test resolution property."""
        gen = HadamardGenerator(resolution=(1920, 1080))
        assert gen.resolution == (1920, 1080)

    def test_radius_property(self):
        """Test radius property."""
        gen = HadamardGenerator(resolution=(100, 100), radius=0.7)
        assert gen.radius == 0.7

    def test_get_valid_pixels(self):
        """Test get_valid_pixels method."""
        gen = HadamardGenerator(resolution=(100, 100))
        valid = gen.get_valid_pixels()
        assert len(valid) > 0
        assert len(valid) <= 100 * 100

    def test_get_mode_indices(self):
        """Test get_mode_indices method."""
        gen = HadamardGenerator(resolution=(100, 100), mode_order=4)
        indices = gen.get_mode_indices()
        assert len(indices) == 16
        assert (0, 0) in indices
        assert (3, 3) in indices

    def test_mask_copy(self):
        """Test that mask property returns a copy."""
        gen = HadamardGenerator(resolution=(100, 100))
        mask1 = gen.mask
        mask1[0, 0] = 99
        mask2 = gen.mask
        assert mask2[0, 0] != 99
