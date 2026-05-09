import numpy as np
import pytest

from ao_shaping.utils.wavefront_calc import (
    normalize_01,
    centroid_calculation,
    calculate_derotation,
    to_color,
)


class TestNormalize01:
    def test_normalize_range(self):
        matrix = np.array([[0, 5], [10, 15]], dtype=np.float64)
        result = normalize_01(matrix)
        assert np.min(result) == pytest.approx(0.0)
        assert np.max(result) == pytest.approx(1.0)

    def test_normalize_constant_matrix(self):
        matrix = np.full((3, 3), 5.0)
        result = normalize_01(matrix)
        assert np.all(result == 0)

    def test_normalize_preserves_shape(self):
        matrix = np.random.rand(10, 20)
        result = normalize_01(matrix)
        assert result.shape == matrix.shape

    def test_normalize_negative_values(self):
        matrix = np.array([[-10, 0], [5, 10]], dtype=np.float64)
        result = normalize_01(matrix)
        assert np.min(result) == pytest.approx(0.0)
        assert np.max(result) == pytest.approx(1.0)


class TestCentroidCalculation:
    def test_uniform_matrix_center(self):
        matrix = np.ones((100, 100))
        cx, cy = centroid_calculation(matrix)
        assert cx == pytest.approx(50.5, abs=0.5)
        assert cy == pytest.approx(50.5, abs=0.5)

    def test_single_pixel(self):
        matrix = np.zeros((10, 10))
        matrix[5, 3] = 1.0
        cx, cy = centroid_calculation(matrix)
        assert cx == pytest.approx(4.0, abs=0.01)
        assert cy == pytest.approx(6.0, abs=0.01)

    def test_symmetric_gaussian_center(self):
        x = np.arange(50)
        y = np.arange(50)
        xx, yy = np.meshgrid(x, y)
        matrix = np.exp(-((xx - 25) ** 2 + (yy - 25) ** 2) / 100)
        cx, cy = centroid_calculation(matrix)
        assert cx == pytest.approx(26, abs=1)
        assert cy == pytest.approx(26, abs=1)


class TestCalculateDerotation:
    def test_zero_angle_identity(self):
        x, y = calculate_derotation(3.0, 4.0, 0.0)
        assert x == pytest.approx(3.0, abs=1e-6)
        assert y == pytest.approx(4.0, abs=1e-6)

    def test_90_degree_rotation(self):
        x, y = calculate_derotation(1.0, 0.0, np.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(-1.0, abs=1e-6)

    def test_180_degree_rotation(self):
        x, y = calculate_derotation(1.0, 0.0, np.pi)
        assert x == pytest.approx(-1.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_inverse_property(self):
        x0, y0 = 3.0, 4.0
        theta = 0.7
        x_rot, y_rot = calculate_derotation(x0, y0, theta)
        x_back, y_back = calculate_derotation(x_rot, y_rot, -theta)
        assert x_back == pytest.approx(x0, abs=1e-4)
        assert y_back == pytest.approx(y0, abs=1e-4)

    def test_returns_6_decimal_precision(self):
        x, y = calculate_derotation(1.0, 1.0, 0.1)
        x_str = str(x)
        assert x_str.count(".") <= 1 or len(x_str.split(".")[1]) <= 6


class TestToColor:
    def test_output_shape(self):
        matrix = np.random.rand(50, 50)
        result = to_color(matrix)
        assert result.shape == (50, 50, 3)

    def test_output_dtype(self):
        matrix = np.random.rand(10, 10)
        result = to_color(matrix)
        assert result.dtype == np.uint8

    def test_zero_matrix(self):
        matrix = np.zeros((5, 5))
        result = to_color(matrix)
        assert np.all(result == 0)

    def test_custom_max_val(self):
        matrix = np.ones((5, 5)) * 0.5
        result = to_color(matrix, max_val=0.5)
        assert np.all(result[:, :, 0] >= 254)

    def test_three_channels_equal(self):
        matrix = np.random.rand(8, 8)
        result = to_color(matrix)
        assert np.all(result[:, :, 0] == result[:, :, 1])
        assert np.all(result[:, :, 1] == result[:, :, 2])
