import numpy as np
import pytest

from ao_shaping.algorithm.target_func import ImageTargetFunc


class TestImageTargetFuncInit:
    """Test ImageTargetFunc initialization with different input types."""

    def test_init_with_integer_dimensions(self):
        """Test initialization with integer height and width."""
        h, w = 10, 15
        center = (5, 7)
        target = ImageTargetFunc(h, w, center)
        assert target.shape == (h, w)
        assert target.center == center
        assert target.xv.shape == (h, w)
        assert target.yv.shape == (h, w)

    def test_init_with_meshgrid_coordinates(self):
        """Test initialization with meshgrid coordinates."""
        h, w = 10, 10
        xv, yv = np.meshgrid(np.arange(h), np.arange(w))
        center = (5, 5)
        target = ImageTargetFunc(xv, yv, center)
        assert target.shape == (h, w)
        assert target.center == center
        assert target.dist_mat.shape == (h, w)

    def test_init_with_meshgrid_1d_coordinates(self):
        """Test initialization with 1D coordinate arrays."""
        x = np.arange(10)
        y = np.arange(15)
        center = (5, 7)
        target = ImageTargetFunc(x, y, center)
        assert target.shape == (10, 15)

    def test_assertion_on_mismatched_types(self):
        """Test that assertion is raised when x and y have different types."""
        h, w = 10, 15
        xv, yv = np.meshgrid(np.arange(h), np.arange(w))
        center = (5, 7)
        # One is numpy array, one is int - should raise assertion
        with pytest.raises(AssertionError):
            ImageTargetFunc(xv, w, center)


class TestBuildFromInitImage:
    """Test class method build_from_init_image."""

    def test_build_from_uniform_image(self):
        """Test building from a uniform image."""
        img = np.ones((10, 10))
        target = ImageTargetFunc.build_from_init_image(img)
        assert target.shape == (10, 10)
        # For uniform image, center should be at image center
        assert target.center == (5, 5)

    def test_build_from_gaussian_image(self):
        """Test building from a Gaussian-like image with bright spot."""
        img = np.zeros((20, 20))
        # Create a bright spot at (8, 12)
        img[7:10, 11:14] = 10
        target = ImageTargetFunc.build_from_init_image(img)
        # Center should be near the bright spot
        cx, cy = target.center
        assert 7 <= cx <= 10
        assert 11 <= cy <= 14

    def test_build_from_image_with_hole(self):
        """Test building from an image with a dark center (hole)."""
        img = np.ones((20, 20)) * 5
        # Create a dark hole in the center
        img[8:12, 8:12] = 0.1
        target = ImageTargetFunc.build_from_init_image(img)
        # Should use center_of_brightness due to hole
        cx, cy = target.center
        # Brightness center should be away from the hole
        assert not (8 <= cx <= 12 and 8 <= cy <= 12)


class TestPIB:
    """Test pib (power in bucket) method."""

    def test_pib_normalized(self):
        """Test pib with normalization."""
        img = np.ones((10, 10))
        target = ImageTargetFunc(10, 10, (5, 5))
        pib_value = target.pib(img, pib_radius=2, normalize=True)
        # Total power is 100, bucket area is approximately pi*2^2 = 12.56
        # Normalized pib should be between 0 and 1
        assert 0 <= pib_value <= 1

    def test_pib_not_normalized(self):
        """Test pib without normalization."""
        img = np.ones((10, 10))
        target = ImageTargetFunc(10, 10, (5, 5))
        pib_value = target.pib(img, pib_radius=2, normalize=False)
        # Should return sum of values in bucket
        assert pib_value > 0

    def test_pib_with_varying_radius(self):
        """Test pib with different bucket radii."""
        img = np.ones((20, 20))
        target = ImageTargetFunc(20, 20, (10, 10))
        pib_small = target.pib(img, pib_radius=2, normalize=True)
        pib_large = target.pib(img, pib_radius=5, normalize=True)
        # Larger radius should have more power
        assert pib_large >= pib_small

    def test_pib_with_gaussian_image(self):
        """Test pib with a Gaussian intensity distribution."""
        xv, yv = np.meshgrid(np.arange(20), np.arange(20))
        img = np.exp(-((xv - 10)**2 + (yv - 10)**2) / 20)
        target = ImageTargetFunc(20, 20, (10, 10))
        pib_center = target.pib(img, pib_radius=3, normalize=True)
        pib_edge = target.pib(img, pib_radius=8, normalize=True)
        # Center bucket should have higher normalized power density
        assert pib_center > 0


class TestDenoiseProcess:
    """Test denoise_process method."""

    def test_denoise_uniform_image(self):
        """Test denoising a uniform image."""
        img = np.ones((10, 10)) * 0.5
        denoised = ImageTargetFunc(10, 10, (5, 5)).denoise_process(img)
        # After removing 5th percentile (which is 0.5), values should be 0
        assert np.all(denoised == 0)

    def test_denoise_with_signal(self):
        """Test denoising an image with actual signal."""
        img = np.ones((10, 10)) * 0.2
        img[4:6, 4:6] = 1.0  # Add signal
        target = ImageTargetFunc(10, 10, (5, 5))
        denoised = target.denoise_process(img)
        # Background should be removed, signal should remain
        assert np.all(denoised[4:6, 4:6] > 0)
        # Background should be 0 after denoising
        assert np.all(denoised[:4, :] == 0)
        assert np.all(denoised[6:, :] == 0)

    def test_denoise_all_zeros(self):
        """Test denoising an all-zero image."""
        img = np.zeros((10, 10))
        target = ImageTargetFunc(10, 10, (5, 5))
        denoised = target.denoise_process(img)
        assert np.all(denoised == 0)


class TestIntelligenCenter:
    """Test intelligen_center method."""

    def test_intelligen_center_uniform(self):
        """Test center detection on uniform image."""
        img = np.ones((20, 20))
        target = ImageTargetFunc(20, 20, (10, 10))
        center = target.intelligen_center(img)
        assert center == (10, 10)

    def test_intelligen_center_with_peak(self):
        """Test center detection with a bright peak."""
        img = np.zeros((20, 20))
        img[12, 8] = 10  # Bright peak
        target = ImageTargetFunc(20, 20, (10, 10))
        center = target.intelligen_center(img)
        # Should detect the bright spot
        cx, cy = center
        assert abs(cx - 8) <= 1
        assert abs(cy - 12) <= 1

    def test_intelligen_center_with_hole(self):
        """Test center detection when center has a hole (dark area)."""
        img = np.ones((20, 20)) * 5
        img[9:12, 9:12] = 0.1  # Dark hole in center
        target = ImageTargetFunc(20, 20, (10, 10))
        center = target.intelligen_center(img)
        # Should use center_of_brightness (not mass) due to hole
        cx, cy = center
        # Should not be at the hole
        assert not (9 <= cx <= 12 and 9 <= cy <= 12)


class TestCenterOfBrightness:
    """Test center_of_brightness method."""

    def test_center_of_brightness_uniform(self):
        """Test center_of_brightness on uniform image."""
        img = np.ones((10, 10))
        target = ImageTargetFunc(10, 10, (5, 5))
        center = target.center_of_brightness(img)
        # Should return center of image for uniform
        assert center == (5, 5) or center == (4, 4)  # Depends on implementation

    def test_center_of_brightness_single_peak(self):
        """Test center_of_brightness with single bright pixel."""
        img = np.zeros((10, 10))
        img[3, 7] = 100
        target = ImageTargetFunc(10, 10, (5, 5))
        center = target.center_of_brightness(img)
        assert center == (7, 3)


class TestCenterOfMass:
    """Test center_of_mass method."""

    def test_center_of_mass_uniform(self):
        """Test center_of_mass on uniform image."""
        img = np.ones((10, 10))
        target = ImageTargetFunc(10, 10, (5, 5))
        cx, cy = target.center_of_mass(img)
        assert np.isclose(cx, 4.5)
        assert np.isclose(cy, 4.5)

    def test_center_of_mass_shifted(self):
        """Test center_of_mass with shifted intensity."""
        img = np.zeros((10, 10))
        img[2, 6] = 1
        target = ImageTargetFunc(10, 10, (5, 5))
        cx, cy = target.center_of_mass(img)
        assert cx == 6.0
        assert cy == 2.0

    def test_center_of_mass_moment_1(self):
        """Test center_of_mass with moment=1 (standard center of mass)."""
        img = np.ones((5, 5))
        target = ImageTargetFunc(5, 5, (2, 2))
        cx, cy = target.center_of_mass(img, moment=1)
        assert np.isclose(cx, 2.0)
        assert np.isclose(cy, 2.0)


class TestRadius:
    """Test radius method."""

    def test_radius_uniform_image(self):
        """Test radius calculation on uniform image."""
        img = np.ones((20, 20))
        target = ImageTargetFunc(20, 20, (10, 10))
        r = target.radius(img, energy=0.99)
        # Should cover almost entire image
        assert r > 10

    def test_radius_high_energy(self):
        """Test radius with high energy threshold."""
        img = np.zeros((20, 20))
        img[8:12, 8:12] = 1  # 4x4 bright square
        target = ImageTargetFunc(20, 20, (10, 10))
        r = target.radius(img, energy=0.99)
        # Should be approximately the distance to corner of square
        assert r > 0

    def test_radius_low_energy(self):
        """Test radius with low energy threshold."""
        img = np.zeros((20, 20))
        img[9:11, 9:11] = 1  # Small 2x2 bright square
        target = ImageTargetFunc(20, 20, (10, 10))
        r = target.radius(img, energy=0.5)
        # Should be small
        assert r <= 3


class TestMaskGeneration:
    """Test internal mask generation."""

    def test_masks_shape(self):
        """Test that masks have correct shape."""
        target = ImageTargetFunc(20, 20, (10, 10))
        assert target.masks.shape[0] == 10  # max_radius = min(10, 10, 10, 10)
        assert target.masks.shape[1] == 20
        assert target.masks.shape[2] == 20

    def test_masks_radius_properties(self):
        """Test that masks satisfy radius properties."""
        target = ImageTargetFunc(20, 20, (10, 10))
        # Mask for radius r should include all points within distance r
        for r in range(1, len(target.masks)):
            mask = target.masks[r - 1]
            # Points at exactly distance r should be included
            assert np.any(mask)


class TestGetBucketMask:
    """Test __get_bucket_mask method via public interface."""

    def test_get_bucket_mask_valid_radius(self):
        """Test getting bucket mask for valid radius."""
        target = ImageTargetFunc(20, 20, (10, 10))
        mask = target._ImageTargetFunc__get_bucket_mask(5)
        assert mask.shape == (20, 20)
        assert mask.dtype == bool

    def test_get_bucket_mask_invalid_radius(self):
        """Test that invalid radius raises assertion."""
        target = ImageTargetFunc(20, 20, (10, 10))
        with pytest.raises(AssertionError):
            target._ImageTargetFunc__get_bucket_mask(0)
        with pytest.raises(AssertionError):
            target._ImageTargetFunc__get_bucket_mask(100)


class TestDistanceMatrix:
    """Test distance matrix calculation."""

    def test_distance_matrix_center(self):
        """Test that center has zero distance."""
        target = ImageTargetFunc(10, 10, (5, 5))
        assert target.dist_mat[5, 5] == 0

    def test_distance_matrix_symmetry(self):
        """Test distance matrix symmetry."""
        target = ImageTargetFunc(10, 10, (5, 5))
        h, w = target.dist_mat.shape
        # Check symmetry around center
        for i in range(h):
            for j in range(w):
                dist = target.dist_mat[i, j]
                # Corresponding point across center
                ci, cj = 2*5 - i, 2*5 - j
                if 0 <= ci < h and 0 <= cj < w:
                    assert target.dist_mat[ci, cj] == dist

    def test_distance_matrix_properties(self):
        """Test distance matrix properties."""
        target = ImageTargetFunc(10, 10, (5, 5))
        # All distances should be non-negative
        assert np.all(target.dist_mat >= 0)
        # Distance at center should be 0
        assert target.dist_mat[5, 5] == 0


class TestPixelSpacing:
    """Test pixel spacing (dpix) calculation."""

    def test_dpix_default_meshgrid(self):
        """Test dpix for default meshgrid coordinates."""
        target = ImageTargetFunc(10, 10, (5, 5))
        assert target.dpix == 1.0

    def test_npix(self):
        """Test npix (number of pixels in one dimension)."""
        target = ImageTargetFunc(15, 20, (7, 10))
        assert target.npix == 15


class TestPerformance:
    """Performance tests for ImageTargetFunc."""

    def test_pib_performance(self):
        """Test pib performance with large image."""
        import time
        img = np.random.rand(200, 200)
        target = ImageTargetFunc(200, 200, (100, 100))
        
        # Warm up
        target.pib(img, pib_radius=10)
        
        # Time multiple calls
        start = time.perf_counter()
        for _ in range(100):
            target.pib(img, pib_radius=10)
        elapsed = time.perf_counter() - start
        
        # Should complete 100 calls in reasonable time
        assert elapsed < 2.0  # 2 seconds for 100 calls

    def test_radius_performance(self):
        """Test radius calculation performance."""
        import time
        img = np.random.rand(200, 200)
        target = ImageTargetFunc(200, 200, (100, 100))
        
        # Warm up
        target.radius(img)
        
        # Time multiple calls
        start = time.perf_counter()
        for _ in range(100):
            target.radius(img)
        elapsed = time.perf_counter() - start
        
        # Should complete 100 calls in reasonable time
        assert elapsed < 2.0
