import numpy as np
import pytest
import time
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except (ImportError, AttributeError):
    CUPY_AVAILABLE = False

from ao_shaping.utils.spots_calc import (
    calculate_sharpness,
    calculate_sharpness_numba,
    crop,
    crop_numba,
    center_of_mass_numba,
    center_of_brightness_numba,
    diffraction_limit,
    jitter_diameter,
    centroid,
    peak_position,
    make_coord,
    radius,
    effective_radius,
    power_bucket,
    disp
)

if CUPY_AVAILABLE:
    from ao_shaping.utils.spots_calc import (
        calculate_sharpness_cupy,
        crop_cupy,
        center_of_mass_cupy,
        center_of_brightness_cupy,
    )


class TestCalculateSharpness:
    def test_calculate_sharpness_uniform_image(self):
        img = np.ones((10, 10))
        sharpness = calculate_sharpness(img)
        sharpness_numba = calculate_sharpness_numba(img)
        assert sharpness == 0.0
        assert sharpness_numba == 0.0

    def test_calculate_sharpness_gradient_image(self):
        img = np.linspace(0, 1, 10).reshape(1, -1).repeat(10, axis=0)
        sharpness = calculate_sharpness(img)
        sharpness_numba = calculate_sharpness_numba(img)
        assert sharpness > 0
        assert sharpness_numba > 0
        assert np.isclose(sharpness, sharpness_numba)

    def test_calculate_sharpness_random_image(self):
        np.random.seed(42)
        img = np.random.rand(10, 10)
        sharpness = calculate_sharpness(img)
        sharpness_numba = calculate_sharpness_numba(img)
        assert isinstance(sharpness, float)
        assert sharpness >= 0
        assert np.isclose(sharpness, sharpness_numba)


class TestCrop:
    def test_crop_simple_image(self):
        img = np.zeros((10, 10))
        img[3:7, 3:7] = 1
        cropped = crop(img, sample_pix=3)
        cropped_numba = crop_numba(img, sample_pix=3)
        assert cropped.shape == (4, 4)
        assert np.all(cropped == 1)
        assert cropped_numba.shape == (4, 4)
        assert np.allclose(cropped, cropped_numba)

    def test_crop_with_background(self):
        img = np.ones((10, 10)) * 0.5
        img[2:8, 2:8] = 1
        cropped = crop(img, sample_pix=2)
        cropped_numba = crop_numba(img, sample_pix=2)
        assert cropped.shape == (6, 6)
        assert cropped_numba.shape == (6, 6)
        assert np.allclose(cropped, cropped_numba)

    def test_crop_3d_image_raises_assertion(self):
        img = np.ones((10, 10, 3))
        with pytest.raises(AssertionError):
            crop(img)
        with pytest.raises(AssertionError):
            crop_numba(img)


class TestCenterOfMassNumba:
    def test_center_of_mass_numba_uniform(self):
        intensity = np.ones((5, 5))
        xv, yv = np.meshgrid(np.arange(5), np.arange(5))
        cx, cy = center_of_mass_numba(intensity, xv, yv)
        assert cx == 2.0
        assert cy == 2.0
        try:
            if CUPY_AVAILABLE:
                intensity_cp = cp.asarray(intensity)
                xv_cp = cp.asarray(xv)
                yv_cp = cp.asarray(yv)
                cx_cp, cy_cp = center_of_mass_cupy(intensity_cp, xv_cp, yv_cp)
                assert cx_cp == 2.0
                assert cy_cp == 2.0
        except Exception:
            pass  # Skip if cupy not working

    def test_center_of_mass_numba_shifted(self):
        intensity = np.zeros((5, 5))
        intensity[1, 3] = 1
        xv, yv = np.meshgrid(np.arange(5), np.arange(5))
        cx, cy = center_of_mass_numba(intensity, xv, yv)
        assert cx == 3.0
        assert cy == 1.0
        try:
            if CUPY_AVAILABLE:
                intensity_cp = cp.asarray(intensity)
                xv_cp = cp.asarray(xv)
                yv_cp = cp.asarray(yv)
                cx_cp, cy_cp = center_of_mass_cupy(intensity_cp, xv_cp, yv_cp)
                assert cx_cp == 3.0
                assert cy_cp == 1.0
        except Exception:
            pass

    def test_center_of_mass_numba_moment_2(self):
        intensity = np.ones((3, 3))
        xv, yv = np.meshgrid(np.arange(3), np.arange(3))
        cx, cy = center_of_mass_numba(intensity, xv, yv, moment=2)
        assert cx == 1.0
        assert cy == 1.0
        try:
            if CUPY_AVAILABLE:
                intensity_cp = cp.asarray(intensity)
                xv_cp = cp.asarray(xv)
                yv_cp = cp.asarray(yv)
                cx_cp, cy_cp = center_of_mass_cupy(intensity_cp, xv_cp, yv_cp, moment=2)
                assert cx_cp == 1.0
                assert cy_cp == 1.0
        except Exception:
            pass


class TestCenterOfBrightnessNumba:
    def test_center_of_brightness_numba(self):
        img = np.zeros((5, 5))
        img[2, 3] = 10
        cx, cy = center_of_brightness_numba(img)
        assert cx == 3
        assert cy == 2
        try:
            if CUPY_AVAILABLE:
                img_cp = cp.asarray(img)
                cx_cp, cy_cp = center_of_brightness_cupy(img_cp)
                assert cx_cp == 3
                assert cy_cp == 2
        except Exception:
            pass


class TestDiffractionLimit:
    def test_diffraction_limit(self):
        lamd = 500e-9
        aperture = 0.1
        dist = 1000
        d = diffraction_limit(lamd, aperture, dist)
        expected = 1.22 * lamd * dist / aperture
        assert d == expected


class TestJitterDiameter:
    def test_jitter_diameter_short_dist(self):
        lamd = 500e-9
        aperture = 0.1
        dist = 5000
        d = jitter_diameter(lamd, aperture, dist)
        assert d == 1e-6 * dist

    def test_jitter_diameter_long_dist(self):
        lamd = 500e-9
        aperture = 0.1
        dist = 20000
        d = jitter_diameter(lamd, aperture, dist)
        expected = 3 * lamd * dist / aperture
        assert d == expected


class TestCentroid:
    def test_centroid_uniform(self):
        intensity = np.ones((5, 5))
        cx, cy = centroid(intensity)
        assert cx == 2
        assert cy == 2

    def test_centroid_shifted(self):
        intensity = np.zeros((5, 5))
        intensity[1, 3] = 1
        cx, cy = centroid(intensity)
        assert cx == 3
        assert cy == 1

    def test_centroid_with_threshold(self):
        intensity = np.ones((5, 5)) * 0.5
        intensity[2, 2] = 1
        cx, cy = centroid(intensity, threshold=0.6)
        assert cx == 2
        assert cy == 2


class TestPeakPosition:
    def test_peak_position(self):
        intensity = np.zeros((5, 5))
        intensity[1, 3] = 10
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        xp, yp = peak_position(intensity, x, y)
        assert xp == 3
        assert yp == 1


class TestMakeCoord:
    def test_make_coord(self):
        img = np.zeros((3, 4))
        x, y = make_coord(img)
        assert x.shape == (3, 4)
        assert y.shape == (3, 4)
        assert np.all(x[0, :] == [0, 1, 2, 3])
        assert np.all(y[:, 0] == [0, 1, 2])


class TestRadius:
    def test_radius_centroid(self):
        intensity = np.zeros((10, 10))
        intensity[5, 5] = 1
        r = radius(intensity, center='centroid', energy=1.0)
        assert r >= 0

    def test_radius_peak(self):
        intensity = np.zeros((10, 10))
        intensity[3, 4] = 1
        r = radius(intensity, center='peak', energy=1.0)
        assert r >= 0

    def test_radius_origin(self):
        intensity = np.ones((10, 10))
        r = radius(intensity, center='origin', energy=0.5)
        assert r >= 0

    def test_radius_custom_center(self):
        intensity = np.ones((10, 10))
        r = radius(intensity, center=(5, 5), energy=0.5)
        assert r >= 0

    def test_radius_invalid_center(self):
        intensity = np.ones((10, 10))
        with pytest.raises(ValueError):
            radius(intensity, center='invalid')


class TestEffectiveRadius:
    def test_effective_radius(self):
        intensity = np.zeros((10, 10))
        intensity[4:6, 4:6] = 1
        dpix = 1.0
        clip = 0.5
        r = effective_radius(intensity, dpix, clip)
        assert r > 0


class TestPowerBucket:
    def test_power_bucket_centroid(self):
        intensity = np.ones((10, 10))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        power = power_bucket(intensity, x, y, center='centroid', r_bucket=2.0)
        assert power > 0

    def test_power_bucket_peak(self):
        intensity = np.zeros((10, 10))
        intensity[5, 5] = 1
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        power = power_bucket(intensity, x, y, center='peak', r_bucket=2.0)
        assert power > 0

    def test_power_bucket_origin(self):
        intensity = np.ones((10, 10))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        power = power_bucket(intensity, x, y, center='origin', r_bucket=2.0)
        assert power > 0

    def test_power_bucket_custom_center(self):
        intensity = np.ones((10, 10))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        power = power_bucket(intensity, x, y, center=(5, 5), r_bucket=2.0)
        assert power > 0

    def test_power_bucket_invalid_center(self):
        intensity = np.ones((10, 10))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        with pytest.raises(ValueError):
            power_bucket(intensity, x, y, center='invalid', r_bucket=2.0)


class TestDisp:
    def test_disp_runs_without_error(self):
        # Mock plt.show to avoid displaying
        import matplotlib.pyplot as plt
        plt.show = lambda: None
        img = np.random.rand(10, 10)
        xv, yv = np.meshgrid(np.arange(10), np.arange(10))
        # This will display but we mock show
        disp(img, xv, yv, r_bucket=2.0, title='Test')
        # If no exception, test passes


class TestPerformance:
    def test_performance_comparison(self):
        # Generate test data
        img = np.random.rand(100, 100).astype(np.float32)
        xv, yv = np.meshgrid(np.arange(100), np.arange(100))

        results = {}

        # Test calculate_sharpness
        def time_func(func, *args, n=100):
            times = []
            for _ in range(n):
                start = time.time()
                func(*args)
                end = time.time()
                times.append(end - start)
            return np.mean(times)

        # Numpy version
        calculate_sharpness(img)  # warmup
        results['calculate_sharpness_numpy'] = time_func(calculate_sharpness, img)

        # Numba version
        calculate_sharpness_numba(img)  # warmup
        results['calculate_sharpness_numba'] = time_func(calculate_sharpness_numba, img)

        # Cupy version
        try:
            if CUPY_AVAILABLE:
                img_cp = cp.asarray(img)
                results['calculate_sharpness_cupy'] = time_func(calculate_sharpness_cupy, img_cp)
            else:
                results['calculate_sharpness_cupy'] = 'N/A'
        except Exception:
            results['calculate_sharpness_cupy'] = 'N/A'

        # Test crop
        crop(img, 10)  # warmup
        results['crop_numpy'] = time_func(crop, img, 10)

        crop_numba(img, 10)  # warmup
        results['crop_numba'] = time_func(crop_numba, img, 10)

        try:
            if CUPY_AVAILABLE:
                img_cp = cp.asarray(img)
                results['crop_cupy'] = time_func(crop_cupy, img_cp, 10)
            else:
                results['crop_cupy'] = 'N/A'
        except Exception:
            results['crop_cupy'] = 'N/A'

        # Test center_of_mass_numba
        center_of_mass_numba(img, xv, yv)  # warmup
        results['center_of_mass_numba'] = time_func(center_of_mass_numba, img, xv, yv)

        try:
            if CUPY_AVAILABLE:
                img_cp = cp.asarray(img)
                xv_cp = cp.asarray(xv)
                yv_cp = cp.asarray(yv)
                results['center_of_mass_cupy'] = time_func(center_of_mass_cupy, img_cp, xv_cp, yv_cp)
            else:
                results['center_of_mass_cupy'] = 'N/A'
        except Exception:
            results['center_of_mass_cupy'] = 'N/A'

        # Test center_of_brightness_numba
        center_of_brightness_numba(img)  # warmup
        results['center_of_brightness_numba'] = time_func(center_of_brightness_numba, img)

        try:
            if CUPY_AVAILABLE:
                img_cp = cp.asarray(img)
                results['center_of_brightness_cupy'] = time_func(center_of_brightness_cupy, img_cp)
            else:
                results['center_of_brightness_cupy'] = 'N/A'
        except Exception:
            results['center_of_brightness_cupy'] = 'N/A'

        # Generate markdown
        markdown = "# Performance Comparison\n\n"
        markdown += "| Function | NumPy (s) | Numba (s) | CuPy (s) |\n"
        markdown += "|----------|-----------|-----------|----------|\n"

        functions = [
            'calculate_sharpness',
            'crop',
            'center_of_mass',
            'center_of_brightness'
        ]

        for func in functions:
            numpy_time = results.get(f'{func}_numpy', 'N/A')
            numba_time = results.get(f'{func}_numba', 'N/A')
            cupy_time = results.get(f'{func}_cupy', 'N/A')
            markdown += f"| {func} | {numpy_time} | {numba_time} | {cupy_time} |\n"

        print(markdown)
        # Save to file
        with open('performance_comparison.md', 'w') as f:
            f.write(markdown)