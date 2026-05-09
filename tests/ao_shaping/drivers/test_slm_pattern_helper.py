from __future__ import annotations

import numpy as np
from ao_shaping.utils.pattern_helper import PatternHelper


class TestPatternHelperTurbulence:
    def test_generate_turbulence_screen_with_aotools(self):
        np.random.seed(42)
        ph = PatternHelper((256, 256), bits=10)
        ph.init_turbulence_screen(r0=10e-2, L0=1.0, pixel_scale=0.1/256, random_seed=42)
        screen = ph.generate_turbulence_screen()
        assert screen.shape == (256, 256)
        assert screen.dtype == np.uint16
        assert screen.max() > 0

    def test_generate_turbulence_screen_deterministic_with_seed(self):
        ph1 = PatternHelper((256, 256), bits=10)
        ph1.init_turbulence_screen(r0=10e-2, L0=1.0, pixel_scale=0.1/256, random_seed=123)
        screen1 = ph1.generate_turbulence_screen()

        ph2 = PatternHelper((256, 256), bits=10)
        ph2.init_turbulence_screen(r0=10e-2, L0=1.0, pixel_scale=0.1/256, random_seed=456)
        screen2 = ph2.generate_turbulence_screen()

        assert not np.array_equal(screen1, screen2)

    def test_generate_turbulence_screen_different_params(self):
        ph1 = PatternHelper((256, 256), bits=10)
        ph1.init_turbulence_screen(r0=10e-2, L0=1.0, pixel_scale=0.1/256, random_seed=42)
        screen1 = ph1.generate_turbulence_screen()

        ph2 = PatternHelper((256, 256), bits=10)
        ph2.init_turbulence_screen(r0=10e-2, L0=1.0, pixel_scale=0.1/256, random_seed=99)
        screen2 = ph2.generate_turbulence_screen()

        assert not np.array_equal(screen1, screen2)

    def test_generate_turbulence_screen_without_init_raises_error(self):
        ph = PatternHelper((128, 128), bits=10)
        try:
            ph.generate_turbulence_screen()
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "not initialized" in str(e).lower()


class TestPatternHelperZernike:
    def test_generate_zernike_single(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_zernike(2, 0, amplitude=1)
        assert img.shape == (200, 200)
        assert img.dtype == np.uint16
        assert img.max() > 0
        assert img.min() == 0

    def test_generate_zernike_negative_m(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_zernike(1, -1, amplitude=0.5)
        assert img.shape == (200, 200)
        assert img.dtype == np.uint16
        assert img.max() > 0

    def test_generate_zernike_polynomial(self):
        ph = PatternHelper((200, 200), bits=10)
        coeffs = {(0, 0): 1.0, (1, -1): 0.3, (2, 0): 0.2}
        img = ph.generate_zernike_polynomial(coefficients=coeffs)
        assert img.shape == (200, 200)
        assert img.dtype == np.uint16

    def test_generate_zernike_polynomial_default(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_zernike_polynomial()
        assert img.shape == (200, 200)
        assert img.dtype == np.uint16


class TestPatternHelperBasics:
    def test_init(self):
        ph = PatternHelper((800, 600), bits=10)
        assert ph.resolution == (800, 600)
        assert ph.bits == 10

    def test_generate_focus(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_focus(focal_length=0.15)
        assert img.shape == (200, 200)

    def test_generate_checkerboard(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_checkerboard(period=50)
        assert img.shape == (200, 200)

    def test_generate_binary_grating_horizontal(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_binary_grating(a=2, b=3, direction="horizontal")
        assert img.shape == (200, 200)

    def test_generate_binary_grating_vertical(self):
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_binary_grating(a=2, b=3, direction="vertical")
        assert img.shape == (200, 200)

    def test_generate_microlens_array(self):
        ph = PatternHelper((400, 400), bits=10)
        img = ph.generate_microlens_array(lens_size=100, focal_length=0.1)
        assert img.shape == (400, 400)


class TestSLMPatternHelperMerged:
    """SLMPatternHelper was merged into PatternHelper. Test its methods."""

    def test_init(self):
        ph = PatternHelper((600, 400), bits=10)
        assert ph._width == 600
        assert ph._height == 400

    def test_linear_grating(self):
        ph = PatternHelper((600, 400), bits=10)
        img = ph.linear_grating(period=100)
        assert img.shape == (400, 600)

    def test_circular_grating(self):
        ph = PatternHelper((600, 400), bits=10)
        img = ph.circular_grating(radius=100)
        assert img.shape == (400, 600)

    def test_lens(self):
        ph = PatternHelper((600, 400), bits=10)
        img = ph.lens(focal_length=0.15, wavelength=532e-9, pixel_size=8e-6)
        assert img.shape == (400, 600)

    def test_hologram(self):
        ph = PatternHelper((600, 400), bits=10)
        img = ph.hologram(period=100)
        assert img.shape == (400, 600)
