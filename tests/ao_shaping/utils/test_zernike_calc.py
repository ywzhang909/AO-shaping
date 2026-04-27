from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.zernike_calc import ZernikeGenerator, fit_zernike, zernike_radial


class TestZernikeGeneratorSquare:
    """Tests for square mode in ZernikeGenerator."""

    def test_square_wider_than_tall(self):
        """Test square mode with width > height (landscape)."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output must match requested resolution
        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"

    def test_square_taller_than_wide(self):
        """Test square mode with height > width (portrait)."""
        width, height = 1080, 1920
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output must match requested resolution
        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"

    def test_square_already_square(self):
        """Test square mode when already square (no cropping needed)."""
        width, height = 1000, 1000
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output must match requested resolution
        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"

    def test_square_false_preserves_aspect(self):
        """Test square=False preserves original non-square shape."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=False)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output unchanged when square=False
        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"

    def test_square_generate_noll(self):
        """Test square mode with generate_noll."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        coeffs = np.ones(10)
        img = gen.generate_noll(coeffs)

        # Output must match requested resolution
        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"

    def test_square_generate_polynomial(self):
        """Test square mode with generate_polynomial."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        coeffs = {(0, 0): 1.0, (1, -1): 0.3, (2, 0): 0.2}
        img = gen.generate_polynomial(coeffs)

        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"

    def test_square_empty_coefficients(self):
        """Test square mode with empty coefficients."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        coeffs = {}
        img = gen.generate_polynomial(coeffs)

        assert img.shape == (height, width), f"Expected ({height}, {width}), got {img.shape}"


class TestZernikeGeneratorBasic:
    """Basic tests for ZernikeGenerator."""

    def test_init_default_radius(self):
        gen = ZernikeGenerator((800, 600))
        assert gen.resolution == (800, 600)
        assert gen.radius == 300.0

    def test_init_custom_radius(self):
        gen = ZernikeGenerator((800, 600), radius=250.0)
        assert gen.radius == 250.0

    def test_set_bits(self):
        gen = ZernikeGenerator((800, 600))
        gen.set_bits(8)
        assert gen._max_val == 255

    def test_generate_without_set_bits_raises(self):
        gen = ZernikeGenerator((800, 600))
        with pytest.raises(ValueError, match="Call set_bits"):
            gen.generate(2, 0)

    def test_generate_piston(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(0, 0, amplitude=1.0)
        assert img.shape == (100, 100)
        assert img.dtype == np.uint16
        assert img.max() >= 0

    def test_generate_tilt(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(1, 1, amplitude=0.5)
        assert img.shape == (100, 100)
        assert img.max() > 0

    def test_generate_defocus(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(2, 0, amplitude=0.5)
        assert img.shape == (100, 100)

    def test_generate_negative_m(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(1, -1, amplitude=0.5)
        assert img.shape == (100, 100)

    def test_generate_polynomial(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        coeffs = {(0, 0): 1.0, (1, -1): 0.3, (2, 0): 0.2}
        img = gen.generate_polynomial(coeffs)
        assert img.shape == (100, 100)

    def test_generate_polynomial_with_zero_coeffs(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        coeffs = {(0, 0): 0.0, (1, 1): 0.0}
        img = gen.generate_polynomial(coeffs)
        assert img.shape == (100, 100)

    def test_mask_property(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        mask = gen.mask
        assert mask.shape == (100, 100)
        assert mask.dtype == bool

    def test_R_property(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        R = gen.R
        assert R.shape == (100, 100)
        assert R.max() > 1.0

    def test_Theta_property(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        Theta = gen.Theta
        assert Theta.shape == (100, 100)


class TestFitZernike:
    def test_fit_zernike_default(self):
        np.random.seed(42)
        wavefront = np.random.rand(100, 100)
        coeffs = fit_zernike(wavefront, n_max=2)
        assert isinstance(coeffs, dict)
        assert (0, 0) in coeffs

    def test_fit_zernike_custom_radius(self):
        np.random.seed(42)
        wavefront = np.random.rand(100, 100)
        coeffs = fit_zernike(wavefront, n_max=2, radius=50.0)
        assert (0, 0) in coeffs

    def test_fit_zernike_single_mode(self):
        np.random.seed(42)
        wavefront = np.random.rand(50, 50)
        coeffs = fit_zernike(wavefront, n_max=1)
        assert isinstance(coeffs, dict)

    def test_fit_zernike_contains_expected_modes(self):
        np.random.seed(42)
        wavefront = np.random.rand(60, 60)
        coeffs = fit_zernike(wavefront, n_max=2)
        expected_keys = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
        for key in expected_keys:
            assert key in coeffs


class TestZernikeRadial:
    def test_zernike_radial_piston(self):
        r = np.linspace(0, 1, 10)
        result = zernike_radial(0, 0, r)
        assert result.shape == r.shape

    def test_zernike_radial_tilt(self):
        r = np.linspace(0, 1, 10)
        result = zernike_radial(1, 1, r)
        assert result.shape == r.shape

    def test_zernike_radial_defocus(self):
        r = np.linspace(0, 1, 10)
        result = zernike_radial(2, 0, r)
        assert result.shape == r.shape