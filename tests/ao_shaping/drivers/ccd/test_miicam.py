import time

import numpy as np
import pytest


pytestmark = pytest.mark.skip(reason="Requires MIICAM 4100 series camera hardware and SDK")


class TestMIICAMCamera:
    """Test cases for MIICAM camera driver."""

    @pytest.fixture
    def miicam_module(self):
        """Import MIICAM module, skip if not available."""
        try:
            from ao_shaping.drivers.ccd.miicam import driver as miicam

            return miicam
        except ImportError:
            pytest.skip("MIICAM module not available")

    @pytest.fixture
    def CameraStreamManager(self, miicam_module):
        """Import CameraStreamManager from MIICAM module."""
        return miicam_module.CameraStreamManager

    def test_get_cam_list(self, CameraStreamManager):
        """Test getting list of available cameras."""
        cam_list = CameraStreamManager.get_cam_list()
        print(f"\nAvailable cameras: {cam_list}")
        assert isinstance(cam_list, list)

    def test_camera_init_and_close(self, CameraStreamManager):
        """Test camera initialization and proper closing."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            assert cam.cam is not None
            assert cam.cam_width > 0
            assert cam.cam_height > 0
            print(f"\nCamera initialized: {cam.cam_width}x{cam.cam_height}")
            print(f"Serial number: {cam._sn}")

    def test_exposure_time(self, CameraStreamManager):
        """Test setting and getting exposure time."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=50.0) as cam:
            new_exposure = cam.reset_exposure_time(30.0)
            assert new_exposure == 30.0
            print(f"\nExposure time set to: {new_exposure}ms")

            new_exposure = cam.reset_exposure_time(0)
            assert abs(new_exposure - 0.011) < 0.001
            print(f"Exposure time (min clamped): {new_exposure}ms")

            new_exposure = cam.reset_exposure_time(0.5)
            assert abs(new_exposure - 0.5) < 0.001
            print(f"Exposure time (sub-ms): {new_exposure}ms")

            new_exposure = cam.reset_exposure_time(20000)
            assert abs(new_exposure - 10000.0) < 0.001
            print(f"Exposure time (max clamped): {new_exposure}ms")

    def test_reset_window_full(self, CameraStreamManager):
        """Test resetting window to full resolution."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            size, center = cam.reset_window(center=(0, 0), size=(0, 0))
            print(f"\nFull window - size: {size}, center: {center}")
            assert cam.cam_width > 0
            assert cam.cam_height > 0

    def test_reset_window_roi(self, CameraStreamManager):
        """Test resetting window to ROI."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            full_width = cam.cam_width
            full_height = cam.cam_height

            roi_size = (full_width // 2, full_height // 2)
            center = (full_width // 2, full_height // 2)
            size, new_center = cam.reset_window(center=center, size=roi_size)
            print(f"\nROI window - size: {size}, center: {new_center}")
            assert size[0] <= full_width
            assert size[1] <= full_height

    def test_get_numpy_image_single(self, CameraStreamManager):
        """Test capturing single image."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8
            assert img.shape == (cam.cam_height, cam.cam_width)
            print(f"\nSingle image shape: {img.shape}")
            print(f"Image min: {img.min()}, max: {img.max()}")

    def test_get_numpy_image_averaged(self, CameraStreamManager):
        """Test capturing and averaging multiple images."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            img = cam.get_numpy_image(n_sample=5, skip_first=True)
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8
            assert img.shape == (cam.cam_height, cam.cam_width)
            print(f"\nAveraged image (5 samples) shape: {img.shape}")
            print(f"Image min: {img.min()}, max: {img.max()}")

    def test_skip_sampling_mode(self, CameraStreamManager):
        """Test camera with skip_sampling (binning) enabled."""
        try:
            with CameraStreamManager(
                cam_id=0, exposure_time_ms=20, skip_sampling=True
            ) as cam:
                img = cam.get_numpy_image(n_sample=1, skip_first=False)
                assert isinstance(img, np.ndarray)
                print(f"\nBinned image shape: {img.shape}")
        except Exception as e:
            pytest.skip(f"Binning not supported on this camera: {e}")

    def test_auto_exposure_enable(self, CameraStreamManager):
        """Test enabling and disabling auto exposure."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            state = cam.get_auto_exposure_state()
            print(f"\nInitial state: {state}")

            cam.enable_auto_exposure(enable=True)
            state = cam.get_auto_exposure_state()
            print(f"After enable: {state}")
            assert state["enabled"] is True
            assert state["mode"] == 1

            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(
                f"Auto exposure image: shape={img.shape}, min={img.min()}, max={img.max()}"
            )

            cam.enable_auto_exposure(enable=False)
            state = cam.get_auto_exposure_state()
            print(f"After disable: {state}")
            assert state["enabled"] is False

    def test_auto_exposure_target(self, CameraStreamManager):
        """Test setting auto exposure target."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            cam.enable_auto_exposure(enable=True)

            target = cam.set_auto_exposure_target(150)
            print(f"\nSet target to: {target}")
            assert target == 150

            state = cam.get_auto_exposure_state()
            print(f"State: {state}")

            target = cam.set_auto_exposure_target(500)
            assert target == 220

            target = cam.set_auto_exposure_target(5)
            assert target == 16

    def test_auto_exposure_range(self, CameraStreamManager):
        """Test setting auto exposure range."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            cam.enable_auto_exposure(enable=True)

            result = cam.set_auto_exposure_range(
                max_time_ms=500,
                min_time_ms=10,
                max_gain=500,
                min_gain=50,
            )
            print(f"\nSet range result: {result}")

            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(
                f"Image with custom range: shape={img.shape}, min={img.min()}, max={img.max()}"
            )

    def test_auto_exposure_target_brightness_200_240(self, CameraStreamManager):
        """Test that with 1ms exposure (skip_first=False), max brightness is reasonable.

        This test uses 1ms exposure time with skip_first=False to get the first frame
        which typically has higher brightness values.
        """
        with CameraStreamManager(cam_id=0, exposure_time_ms=1) as cam:
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            max_val = int(img.max())
            mean_val = float(img.mean())

            print(f"\n1ms exposure (first frame) stats:")
            print(f"  Max: {max_val}")
            print(f"  Mean: {mean_val:.1f}")
            print(f"  Min: {int(img.min())}")

            if max_val < 200:
                if mean_val < 10:
                    pytest.skip("Scene too dark - need more light")

            # Allow for saturation (max_val can be 255 if scene is very bright)
            assert max_val >= 195, f"Max brightness {max_val} too low (expected >= 195)"

    def test_context_manager(self, CameraStreamManager):
        """Test that context manager properly closes camera."""
        cam = CameraStreamManager(cam_id=0, exposure_time_ms=20)
        with cam:
            assert cam.cam is not None
        assert cam.cam is None

    def test_8bit_mode(self, CameraStreamManager):
        """Test camera initialization and capture in 8-bit mode."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20, bit_depth=8) as cam:
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8
            assert img.shape == (cam.cam_height, cam.cam_width)
            assert cam._bit_depth == 8
            print(f"\n8-bit mode: shape={img.shape}, dtype={img.dtype}")

    def test_16bit_mode(self, CameraStreamManager):
        """Test camera initialization and capture in 16-bit mode (high bit depth)."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20, bit_depth=16) as cam:
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint16
            assert img.shape == (cam.cam_height, cam.cam_width)
            assert cam._bit_depth == 16
            print(f"\n16-bit mode: shape={img.shape}, dtype={img.dtype}")

    def test_bit_depth_default_is_8(self, CameraStreamManager):
        """Test that default bit_depth is 8."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            assert cam._bit_depth == 8
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert img.dtype == np.uint8

    def test_8bit_16bit_image_capture(self, CameraStreamManager):
        """Test that both 8-bit and 16-bit modes can capture valid images."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=50, bit_depth=8) as cam:
            img_8bit = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert img_8bit.dtype == np.uint8
            assert img_8bit.shape == (cam.cam_height, cam.cam_width)
            max_8bit = int(img_8bit.max())

        # Allow camera hardware to settle between sessions
        time.sleep(2)

        with CameraStreamManager(cam_id=0, exposure_time_ms=50, bit_depth=16) as cam:
            img_16bit = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert img_16bit.dtype == np.uint16
            assert img_16bit.shape == (cam.cam_height, cam.cam_width)
            max_16bit = int(img_16bit.max())

        assert img_8bit.shape == img_16bit.shape
        print(f"\n8-bit max: {max_8bit}, 16-bit max: {max_16bit}")


def test_quick_capture():
    """Quick test to verify camera works without hardware."""
    try:
        from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager

        cam_list = CameraStreamManager.get_cam_list()
        print(f"\nFound {len(cam_list)} camera(s)")
        for cam in cam_list:
            print(f"  - {cam.id}")
    except ImportError as e:
        pytest.skip(f"MIICAM module not available: {e}")
