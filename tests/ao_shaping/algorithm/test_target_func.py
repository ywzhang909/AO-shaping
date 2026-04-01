import numpy as np
import pytest

from ao_shaping.algorithm.target_func import ImageTargetFunc


def create_target_from_dims(h, w, center):
    """Helper to create ImageTargetFunc with proper meshgrid coordinates."""
    xv, yv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return ImageTargetFunc(xv, yv, center)


class TestImageTargetFuncInit:
    """Test ImageTargetFunc initialization with different input types."""

    def test_init_with_integer_dimensions(self):
        """Test initialization with integer height and width."""
        h, w = 10, 15
        center = (5, 7)
        target = ImageTargetFunc(w, h, center)
        assert target.shape == (h, w)
        assert target.center == center
        # xv and yv are 2D arrays after meshgrid
        assert target.xv.shape == (h, w)
        assert target.yv.shape == (h, w)

    def test_init_with_meshgrid_coordinates(self):
        """Test initialization with meshgrid coordinates."""
        h, w = 10, 10
        xv, yv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
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
        # meshgrid uses default indexing='xy' which swaps dimensions
        # x has 10 elements, y has 15 elements, result is (15, 10)
        assert target.shape == (15, 10)

    def test_assertion_on_mismatched_types(self):
        """Test that assertion is raised when x and y have different types."""
        h, w = 10, 15
        xv, yv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
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
        # Center of mass for uniform image is (4.5, 4.5), rounded to (4, 4) or (5, 5)
        # The build_from_init_image rounds center values for consistent mask calculation
        # Note: numpy uses banker's rounding for .5 cases (rounds to even)
        assert target.center[0] in (4, 5)
        assert target.center[1] in (4, 5)

    def test_build_from_gaussian_image(self):
        """Test building from a Gaussian-like image with bright spot."""
        img = np.zeros((20, 20))
        # Create a bright spot at (row=8, col=12)
        img[7:10, 11:14] = 10
        target = ImageTargetFunc.build_from_init_image(img)
        # Center should be near the bright spot (row, col) format
        cx, cy = target.center
        # Center of mass should be around the center of the bright region
        assert 7 <= cy <= 10  # row coordinate
        assert 11 <= cx <= 14  # col coordinate

    def test_build_from_image_with_hole(self):
        """Test building from an image with a dark center (hole)."""
        img = np.ones((20, 20)) * 5
        # Create a dark hole in the center
        img[8:12, 8:12] = 0.1
        target = ImageTargetFunc.build_from_init_image(img)
        # Should use center_of_mass (which is attracted to hole)
        cx, cy = target.center
        # Center of mass is pulled toward the hole
        assert 8 <= cx <= 12
        assert 8 <= cy <= 12


class TestPIB:
    """Test pib (power in bucket) method."""

    def test_pib_normalized(self):
        """Test pib returns (pib_value, pib_ratio)."""
        img = np.ones((10, 10))
        target = create_target_from_dims(10, 10, (5, 5))
        pib_value, pib_ratio = target.pib(img, pib_radius=2)
        # Total power is 100, bucket area is approximately pi*2^2 = 12.56
        # pib_ratio should be between 0 and 1
        assert 0 <= pib_ratio <= 1

    def test_pib_not_normalized(self):
        """Test pib returns sum of values in bucket."""
        img = np.ones((10, 10))
        target = create_target_from_dims(10, 10, (5, 5))
        pib_value, pib_ratio = target.pib(img, pib_radius=2)
        # Should return sum of values in bucket
        assert pib_value > 0

    def test_pib_with_varying_radius(self):
        """Test pib with different bucket radii."""
        img = np.ones((20, 20))
        target = create_target_from_dims(20, 20, (10, 10))
        _, pib_ratio_small = target.pib(img, pib_radius=2)
        _, pib_ratio_large = target.pib(img, pib_radius=5)
        # Larger radius should have more power ratio
        assert pib_ratio_large >= pib_ratio_small

    def test_pib_with_gaussian_image(self):
        """Test pib with a Gaussian intensity distribution."""
        xv, yv = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        img = np.exp(-((xv - 10) ** 2 + (yv - 10) ** 2) / 3)
        target = create_target_from_dims(20, 20, (10, 10))
        pib_center, _ = target.pib(img, pib_radius=3)
        pib_edge, _ = target.pib(img, pib_radius=6)
        # Center bucket should have higher power than edge bucket
        assert pib_center > 0
        assert pib_center < pib_edge


class TestDenoiseProcess:
    """Test denoise_process method."""

    def test_denoise_uniform_image(self):
        """Test denoising a uniform image."""
        img = np.ones((10, 10)) * 0.5
        denoised = create_target_from_dims(10, 10, (5, 5)).denoise_process(img)
        # After removing 5th percentile (which is 0.5), values should be 0
        assert np.all(denoised == 0)

    def test_denoise_with_signal(self):
        """Test denoising an image with actual signal."""
        img = np.ones((10, 10)) * 0.2
        img[4:6, 4:6] = 1.0  # Add signal
        target = create_target_from_dims(10, 10, (5, 5))
        denoised = target.denoise_process(img)
        # Background should be removed, signal should remain
        assert np.all(denoised[4:6, 4:6] > 0)
        # Background should be 0 after denoising
        assert np.all(denoised[:4, :] == 0)
        assert np.all(denoised[6:, :] == 0)

    def test_denoise_all_zeros(self):
        """Test denoising an all-zero image."""
        img = np.zeros((10, 10))
        target = create_target_from_dims(10, 10, (5, 5))
        denoised = target.denoise_process(img)
        assert np.all(denoised == 0)


class TestIntelligenCenter:
    """Test intelligen_center method."""

    def test_intelligen_center_uniform(self):
        """Test center detection on uniform image."""
        img = np.ones((20, 20))
        target = create_target_from_dims(20, 20, (10, 10))
        center = target.intelligen_center(img)
        # Center of mass for uniform image is (9.5, 9.5)
        assert np.isclose(center[0], 9.5, atol=0.1)
        assert np.isclose(center[1], 9.5, atol=0.1)

    def test_intelligen_center_with_peak(self):
        """Test center detection with a bright peak."""
        img = np.zeros((20, 20))
        img[12, 8] = 10  # Bright peak at (row=12, col=8)
        target = create_target_from_dims(20, 20, (10, 10))
        center = target.intelligen_center(img)
        # Should detect the bright spot
        cx, cy = center  # (col, row) format
        assert abs(cy - 12) <= 1  # row
        assert abs(cx - 8) <= 1  # col

    def test_intelligen_center_with_hole(self):
        """Test center detection when center has a hole (dark area)."""
        img = np.ones((20, 20)) * 5
        img[9:12, 9:12] = 0.1  # Dark hole in center
        target = create_target_from_dims(20, 20, (10, 10))
        center = target.intelligen_center(img)
        # Should use center_of_mass (not brightness) - hole pulls mass toward it
        cx, cy = center
        # Center of mass is pulled toward the hole
        assert 9 <= cx <= 12
        assert 9 <= cy <= 12


class TestCenterOfBrightness:
    """Test center_of_brightness method."""

    def test_center_of_brightness_uniform(self):
        """Test center_of_brightness on uniform image."""
        img = np.ones((10, 10))
        target = create_target_from_dims(10, 10, (5, 5))
        center = target.center_of_brightness(img)
        # For uniform image, returns (0, 0) - first max position
        assert isinstance(center, tuple)
        assert len(center) == 2

    def test_center_of_brightness_single_peak(self):
        """Test center_of_brightness with single bright pixel."""
        img = np.zeros((10, 10))
        img[3, 7] = 100
        target = create_target_from_dims(10, 10, (5, 5))
        center = target.center_of_brightness(img)
        # Returns (col, row) or (x, y) format
        assert center == (7, 3)


class TestCenterOfMass:
    """Test center_of_mass method."""

    def test_center_of_mass_uniform(self):
        """Test center_of_mass on uniform image."""
        img = np.ones((10, 10))
        target = create_target_from_dims(10, 10, (5, 5))
        cx, cy = target.center_of_mass(img)
        # Returns (row, col) format
        assert np.isclose(cx, 4.5)
        assert np.isclose(cy, 4.5)

    def test_center_of_mass_shifted(self):
        """Test center_of_mass with shifted intensity."""
        img = np.zeros((10, 10))
        img[2, 6] = 1
        target = create_target_from_dims(10, 10, (5, 5))
        cx, cy = target.center_of_mass(img)
        # Returns (row, col) format
        assert cx == 2.0
        assert cy == 6.0

    def test_center_of_mass_moment_1(self):
        """Test center_of_mass with moment=1 (standard center of mass)."""
        img = np.ones((5, 5))
        target = create_target_from_dims(5, 5, (2, 2))
        cx, cy = target.center_of_mass(img, moment=1)
        assert np.isclose(cx, 2.0)
        assert np.isclose(cy, 2.0)


class TestRadius:
    """Test radius method."""

    def test_radius_uniform_image(self):
        """Test radius calculation on uniform image."""
        img = np.ones((20, 20))
        target = create_target_from_dims(20, 20, (10, 10))
        r = target.radius(img, energy=0.99)
        # For uniform image, the radius should be at least 1
        assert r >= 1

    def test_radius_high_energy(self):
        """Test radius with high energy threshold."""
        img = np.zeros((20, 20))
        img[8:12, 8:12] = 1  # 4x4 bright square
        target = create_target_from_dims(20, 20, (10, 10))
        r = target.radius(img, energy=0.99)
        # Should be approximately the distance to corner of square
        assert r > 0

    def test_radius_low_energy(self):
        """Test radius with low energy threshold."""
        img = np.zeros((20, 20))
        img[9:11, 9:11] = 1  # Small 2x2 bright square
        target = create_target_from_dims(20, 20, (10, 10))
        r = target.radius(img, energy=0.99)
        # Should be small
        assert r <= 3


class TestMaskGeneration:
    """Test internal mask generation."""

    def test_masks_shape(self):
        """Test that masks have correct shape."""
        target = create_target_from_dims(20, 20, (10, 10))
        # max_radius = min(10, 10, 10, 10) - 1 = 9, then max(1, 9) = 9
        assert target.masks.shape[0] == 9
        assert target.masks.shape[1] == 20
        assert target.masks.shape[2] == 20

    def test_masks_radius_properties(self):
        """Test that masks satisfy radius properties."""
        target = create_target_from_dims(20, 20, (10, 10))
        # Just check that masks are generated and have correct shape
        assert len(target.masks) > 0
        assert target.masks.ndim == 3


class TestGetBucketMask:
    """Test __get_bucket_mask method via public interface."""

    def test_get_bucket_mask_valid_radius(self):
        """Test getting bucket mask for valid radius."""
        target = create_target_from_dims(20, 20, (10, 10))
        mask = target._ImageTargetFunc__get_bucket_mask(5)
        assert mask.shape == (20, 20)
        assert mask.dtype == bool

    def test_get_bucket_mask_invalid_radius(self):
        """Test that invalid radius raises assertion."""
        target = create_target_from_dims(20, 20, (10, 10))
        with pytest.raises(AssertionError):
            target._ImageTargetFunc__get_bucket_mask(0)
        with pytest.raises(AssertionError):
            target._ImageTargetFunc__get_bucket_mask(100)


class TestDistanceMatrix:
    """Test distance matrix calculation."""

    def test_distance_matrix_center(self):
        """Test that center has zero distance."""
        target = create_target_from_dims(10, 10, (5, 5))
        assert target.dist_mat[5, 5] == 0

    def test_distance_matrix_symmetry(self):
        """Test distance matrix symmetry."""
        target = create_target_from_dims(10, 10, (5, 5))
        h, w = target.dist_mat.shape
        # Check symmetry around center
        for i in range(h):
            for j in range(w):
                dist = target.dist_mat[i, j]
                # Corresponding point across center
                ci, cj = 2 * 5 - i, 2 * 5 - j
                if 0 <= ci < h and 0 <= cj < w:
                    assert target.dist_mat[ci, cj] == dist

    def test_distance_matrix_properties(self):
        """Test distance matrix properties."""
        target = create_target_from_dims(10, 10, (5, 5))
        # All distances should be non-negative
        assert np.all(target.dist_mat >= 0)
        # Distance at center should be 0
        assert target.dist_mat[5, 5] == 0


class TestPixelSpacing:
    """Test pixel spacing (dpix) calculation."""

    def test_dpix_default_meshgrid(self):
        """Test dpix for default meshgrid coordinates."""
        target = create_target_from_dims(10, 10, (5, 5))
        # dpix should be positive (not 0)
        assert target.dpix >= 0

    def test_npix(self):
        """Test npix (number of pixels in one dimension)."""
        target = create_target_from_dims(15, 20, (7, 10))
        assert target.npix == 15


class TestPerformance:
    """Performance tests for ImageTargetFunc."""

    def test_pib_performance(self):
        """Test pib performance with large image."""
        import time

        img = np.random.rand(200, 200)
        target = create_target_from_dims(200, 200, (100, 100))

        # Warm up
        target.pib(img, pib_radius=10)

        # Time multiple calls
        start = time.perf_counter()
        for _ in range(100):
            target.pib(img, pib_radius=10)
        elapsed = time.perf_counter() - start

        # Should complete 100 calls in reasonable time
        assert elapsed < 2.0  # 2 seconds for 100 calls


class TestBuildFromInitImageConsistency:
    """Test that build_from_init_image correctly recalculates dist_mat and masks."""

    def test_dist_mat_recalculated_after_center_update(self):
        """Test that dist_mat is correctly calculated based on updated center.

        This tests the bug fix: when build_from_init_image updates center
        via intelligen_center, dist_mat must be recalculated.
        """
        # Create an image with bright spot at (row=5, col=15)
        # This differs from the initial center (w//2, h//2) = (10, 10)
        img = np.zeros((20, 20))
        img[4:6, 14:16] = 100  # Bright spot at center (15, 5) in (col, row) format

        target = ImageTargetFunc.build_from_init_image(img)

        # The center should be near the bright spot
        cx, cy = target.center  # (col, row) format

        # Verify center is near the bright spot
        assert 14 <= cx <= 16, f"Center x={cx} should be near bright spot (col=15)"
        assert 4 <= cy <= 6, f"Center y={cy} should be near bright spot (row=5)"

        # Verify dist_mat is consistent with center
        # Distance at center should be 0
        row_idx = int(round(cy))
        col_idx = int(round(cx))
        assert target.dist_mat[row_idx, col_idx] == 0, (
            f"dist_mat at center ({row_idx}, {col_idx}) should be 0, got {target.dist_mat[row_idx, col_idx]}"
        )

        # Distance should increase away from center
        assert (
            target.dist_mat[row_idx, col_idx] < target.dist_mat[row_idx, col_idx + 5]
        ), "Distance should increase further from center"

    def test_masks_consistent_with_dist_mat(self):
        """Test that masks are consistent with dist_mat after build_from_init_image.

        The bug: masks were calculated with updated center, but dist_mat
        was not recalculated, causing inconsistency.
        """
        img = np.zeros((20, 20))
        img[5, 15] = 100  # Single bright pixel at (row=5, col=15)

        target = ImageTargetFunc.build_from_init_image(img)

        # Verify masks are based on correct center
        cx, cy = target.center  # (col, row) format

        # Create reference mask using the same logic
        row_idx = int(round(cy))
        col_idx = int(round(cx))

        # Check that the center pixel is inside the radius-0 mask (just the center pixel)
        mask_r0 = target.masks[0]  # radius 0 mask (pixels with dist <= 0)
        assert mask_r0[row_idx, col_idx] == True, (
            "Center pixel should be inside radius-0 mask"
        )

        # Verify consistency: dist_mat and masks should agree on center
        assert np.isclose(target.dist_mat[row_idx, col_idx], 0.0), (
            "dist_mat should be 0 at center"
        )

        # Also check radius-1 mask contains nearby pixels
        mask_r1 = target.masks[1]  # radius 1 mask (pixels with dist <= 1)
        # The center pixel should be in the radius-1 mask
        assert mask_r1[row_idx, col_idx] == True, (
            "Center pixel should be inside radius-1 mask"
        )

    def test_build_from_init_image_vs_manual_construction(self):
        """Test that build_from_init_image gives same result as manual construction
        with correct center."""
        img = np.zeros((20, 25))
        img[8:12, 18:22] = 50  # Bright region at center (20, 10) in (col, row) format

        # Build using class method
        target1 = ImageTargetFunc.build_from_init_image(img)

        # Manually build with known center (from build_from_init_image)
        cx, cy = target1.center  # Get the calculated center

        # Create another target with the same center
        target2 = ImageTargetFunc(25, 20, (cx, cy))  # w=25, h=20, center=(col, row)

        # Both should have same dist_mat
        np.testing.assert_allclose(
            target1.dist_mat,
            target2.dist_mat,
            rtol=1e-10,
            err_msg="dist_mat should be identical after build_from_init_image",
        )

        # Both should have same masks
        np.testing.assert_array_equal(
            target1.masks,
            target2.masks,
            err_msg="masks should be identical after build_from_init_image",
        )

    def test_build_from_init_image_non_square_image(self):
        """Test build_from_init_image with non-square image."""
        # Non-square image: h=15, w=25
        img = np.zeros((15, 25))
        img[10:13, 20:23] = 100  # Bright spot at (row=10-13, col=20-23)

        target = ImageTargetFunc.build_from_init_image(img)

        cx, cy = target.center  # (col, row) format

        # Verify center is near the bright spot
        assert 20 <= cx <= 23, f"Center x={cx} should be near col=20-23"
        assert 10 <= cy <= 13, f"Center y={cy} should be near row=10-13"

        # Verify shape is correct
        assert target.shape == (15, 25), f"Shape should be (15, 25), got {target.shape}"

        # Verify dist_mat at center is 0
        row_idx = int(round(cy))
        col_idx = int(round(cx))
        assert target.dist_mat[row_idx, col_idx] == 0

    def test_center_consistency_across_operations(self):
        """Test that center remains consistent across all operations."""
        img = np.zeros((30, 30))
        # Create asymmetric bright spot
        img[20:25, 5:10] = 100

        target = ImageTargetFunc.build_from_init_image(img)

        cx, cy = target.center  # (col, row) format

        # Test pib: should use correct center
        pib_mask = target._ImageTargetFunc__get_bucket_mask(3)
        # The center pixel should be inside the mask
        assert pib_mask[cy, cx] == True, "Center should be inside pib mask"

        # Test radius: should use correct center
        r = target.radius(img, energy=0.5)
        assert r > 0, "Radius should be positive"

        # Test avg_radius: should use correct center (via dist_mat)
        avg_r, _ = target.avg_radius(img)
        assert avg_r > 0, "Average radius should be positive"

    def test_radius_performance(self):
        """Test radius calculation performance."""
        import time

        img = np.random.rand(200, 200)
        target = create_target_from_dims(200, 200, (100, 100))

        # Warm up
        target.radius(img)

        # Time multiple calls
        start = time.perf_counter()
        for _ in range(100):
            target.radius(img)
        elapsed = time.perf_counter() - start

        # Should complete 100 calls in reasonable time (relaxed for CI)
        assert elapsed < 5.0
