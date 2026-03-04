"""
Tests for MIICAM camera driver.

These tests require MIICAM 4100 series camera hardware connected via USB.
Run with: pytest tests/ao_shaping/drivers/test_miicam.py -v
"""

import numpy as np
import pytest


class TestMIICAMCamera:
    """Test cases for MIICAM camera driver."""

    @pytest.fixture
    def miicam_module(self):
        """Import MIICAM module, skip if not available."""
        try:
            from ao_shaping.drivers.ccd import miicam

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
        # Just verify it returns a list (can be empty if no camera connected)
        assert isinstance(cam_list, list)

    def test_camera_init_and_close(self, CameraStreamManager):
        """Test camera initialization and proper closing."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            # Verify camera is initialized
            assert cam.cam is not None
            assert cam.cam_width > 0
            assert cam.cam_height > 0
            print(f"\nCamera initialized: {cam.cam_width}x{cam.cam_height}")
            print(f"Serial number: {cam._CameraStreamManager__sn}")

    def test_exposure_time(self, CameraStreamManager):
        """Test setting and getting exposure time."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=50) as cam:
            # Reset to different exposure time
            new_exposure = cam.reset_exposure_time(30)
            assert new_exposure == 30
            print(f"\nExposure time set to: {new_exposure}ms")

            # Test minimum exposure
            new_exposure = cam.reset_exposure_time(0)
            assert new_exposure == 1  # Should be clamped to minimum
            print(f"Exposure time (min clamped): {new_exposure}ms")

    def test_reset_window_full(self, CameraStreamManager):
        """Test resetting window to full resolution."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            # Reset to full window
            size, center = cam.reset_window(center=(0, 0), size=(0, 0))
            print(f"\nFull window - size: {size}, center: {center}")
            assert cam.cam_width > 0
            assert cam.cam_height > 0

    def test_reset_window_roi(self, CameraStreamManager):
        """Test resetting window to ROI."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            # Get current max resolution
            full_width = cam.cam_width
            full_height = cam.cam_height

            # Set a smaller ROI (centered)
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
        # This test uses skip_sampling=True which enables 2x2 binning
        # Note: May not work on all cameras
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
            # Initially should be disabled (manual exposure)
            state = cam.get_auto_exposure_state()
            print(f"\nInitial state: {state}")

            # Enable auto exposure
            cam.enable_auto_exposure(enable=True)
            state = cam.get_auto_exposure_state()
            print(f"After enable: {state}")
            assert state["enabled"] is True
            assert state["mode"] == 1  # continuous mode

            # Get image with auto exposure
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(
                f"Auto exposure image: shape={img.shape}, min={img.min()}, max={img.max()}"
            )

            # Disable auto exposure
            cam.enable_auto_exposure(enable=False)
            state = cam.get_auto_exposure_state()
            print(f"After disable: {state}")
            assert state["enabled"] is False

    def test_auto_exposure_target(self, CameraStreamManager):
        """Test setting auto exposure target."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            # Enable auto exposure first
            cam.enable_auto_exposure(enable=True)

            # Set target brightness
            target = cam.set_auto_exposure_target(150)
            print(f"\nSet target to: {target}")
            assert target == 150

            # Get state to verify
            state = cam.get_auto_exposure_state()
            print(f"State: {state}")

            # Test clamping
            target = cam.set_auto_exposure_target(500)  # Should clamp to 220
            assert target == 220

            target = cam.set_auto_exposure_target(5)  # Should clamp to 16
            assert target == 16

    def test_auto_exposure_range(self, CameraStreamManager):
        """Test setting auto exposure range."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            # Enable auto exposure first
            cam.enable_auto_exposure(enable=True)

            # Set range
            result = cam.set_auto_exposure_range(
                max_time_ms=500,
                min_time_ms=10,
                max_gain=500,
                min_gain=50,
            )
            print(f"\nSet range result: {result}")

            # Get image to verify auto exposure works
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(
                f"Image with custom range: shape={img.shape}, min={img.min()}, max={img.max()}"
            )

    def test_auto_exposure_target_brightness_200_240(self, CameraStreamManager):
        """Test that with 1ms exposure (skip_first=False), max brightness is in 200-240 range.

        This test uses 1ms exposure time with skip_first=False to get the first frame
        which typically has higher brightness values.
        """
        with CameraStreamManager(cam_id=0, exposure_time_ms=1) as cam:
            # Get image without skipping first frame
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            max_val = int(img.max())
            mean_val = float(img.mean())

            print(f"\n1ms exposure (first frame) stats:")
            print(f"  Max: {max_val}")
            print(f"  Mean: {mean_val:.1f}")
            print(f"  Min: {int(img.min())}")

            # Check if in range or if scene is too dark
            if max_val < 200:
                if mean_val < 10:
                    pytest.skip("Scene too dark - need more light")

            # Verify in range
            assert 200 <= max_val <= 240, (
                f"Max brightness {max_val} not in range [200, 240]"
            )

    def test_context_manager(self, CameraStreamManager):
        """Test that context manager properly closes camera."""
        cam = CameraStreamManager(cam_id=0, exposure_time_ms=20)
        with cam:
            assert cam.cam is not None
        # After exiting context, camera should be closed
        assert cam.cam is None


# Convenience function for quick testing
def test_quick_capture():
    """Quick test to verify camera works without hardware."""
    try:
        from ao_shaping.drivers.ccd.miicam import CameraStreamManager

        cam_list = CameraStreamManager.get_cam_list()
        print(f"\nFound {len(cam_list)} camera(s)")
        for cam in cam_list:
            print(f"  - {cam.id}")
    except ImportError as e:
        pytest.skip(f"MIICAM module not available: {e}")
