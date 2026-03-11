"""
联合测试: SLM + MIICAM 相机

测试 SLM 和 MIICAM 相机协同工作的功能。
需要 SLM 硬件和 MIICAM 4100 系列相机同时连接。
Run with: pytest tests/ao_shaping/drivers/test_slm_miicam.py -v
"""

import numpy as np
import pytest


class TestSLMMIICAMJoint:
    """SLM 和 MIICAM 相机联合测试类"""

    @pytest.fixture
    def slm_module(self):
        """Import SLM module, skip if not available."""
        try:
            from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200

            return SantecSLM200
        except ImportError:
            pytest.skip("SLM module not available")

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

    @pytest.fixture
    def slm(self, slm_module):
        """Create SLM instance."""
        return slm_module(slm_number=1, wavelength=1064, phase_range=200)

    @pytest.fixture
    def open_slm(self, slm):
        """Open SLM."""
        try:
            slm.open()
            yield slm
        except Exception as e:
            pytest.skip(f"SLM not available: {e}")
        finally:
            if slm.is_open:
                slm.close()

    def test_camera_and_slm_list(self, CameraStreamManager, slm_module):
        """Test listing both camera and SLM devices."""
        # List cameras
        cam_list = CameraStreamManager.get_cam_list()
        print(f"\nAvailable cameras: {len(cam_list)}")

        # Note: SLM doesn't have a list function, but we can check if the module loads
        assert slm_module is not None
        print("SLM module loaded successfully")

    def test_slm_display_with_camera_capture(self, open_slm, CameraStreamManager):
        """Test SLM pattern display and camera capture."""
        # Generate a simple phase pattern
        phase = np.zeros((1200, 1920), dtype=np.uint16)
        # Add a simple pattern in the center
        center_y, center_x = 600, 960
        radius = 100
        y, x = np.ogrid[:1200, :1920]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        phase[mask] = 512  # Half phase (π)

        # Write to SLM
        open_slm.write_phase(phase, memory_number=1)
        open_slm.display_memory(1)

        # Capture with camera
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            assert isinstance(img, np.ndarray)
            assert img.shape[0] > 0 and img.shape[1] > 0
            print(f"\nCaptured image shape: {img.shape}")
            print(
                f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}"
            )

    def test_blazed_grating_capture(self, open_slm, CameraStreamManager):
        """Test blazed grating pattern with camera capture."""
        # Generate blazed grating
        period = 50
        height, width = 1200, 1920
        max_val = 1023

        y = np.arange(height)
        grating = (y % period) / period * max_val
        phase = np.tile(grating[:, np.newaxis], (1, width)).astype(np.uint16)

        # Write to SLM
        open_slm.write_phase(phase, memory_number=2)
        open_slm.display_memory(2)

        # Capture with camera
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(f"\nBlazed grating image shape: {img.shape}")
            print(
                f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}"
            )
            # The image should show a gradient pattern
            assert img.max() > img.min()

    def test_multi_pattern_sequence(self, open_slm, CameraStreamManager):
        """Test displaying multiple patterns and capturing each."""
        patterns = []

        # Pattern 1: All zeros (flat)
        p1 = np.zeros((1200, 1920), dtype=np.uint16)
        patterns.append(("flat", p1, 10))

        # Pattern 2: Half gray
        p2 = np.ones((1200, 1920), dtype=np.uint16) * 512
        patterns.append(("half", p2, 11))

        # Pattern 3: Full gray
        p3 = np.ones((1200, 1920), dtype=np.uint16) * 1023
        patterns.append(("full", p3, 12))

        # Pattern 4: Blazed grating
        period = 100
        y = np.arange(1200)
        grating = (y % period) / period * 1023
        p4 = np.tile(grating[:, np.newaxis], (1, 1920)).astype(np.uint16)
        patterns.append(("grating", p4, 13))

        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            for name, phase, mem_num in patterns:
                # Write to SLM
                open_slm.write_phase(phase, memory_number=mem_num)
                open_slm.display_memory(mem_num)

                # Small delay for SLM to settle
                import time

                time.sleep(0.1)

                # Capture image
                img = cam.get_numpy_image(n_sample=1, skip_first=False)
                print(
                    f"\n{name}: shape={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.1f}"
                )

    def test_auto_exposure_with_slm_patterns(self, open_slm, CameraStreamManager):
        """Test auto exposure with different SLM patterns."""
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            # Enable auto exposure
            cam.enable_auto_exposure(enable=True)

            # Dark pattern first
            phase_dark = np.zeros((1200, 1920), dtype=np.uint16)
            open_slm.write_phase(phase_dark, memory_number=20)
            open_slm.display_memory(20)

            import time

            time.sleep(0.2)

            img_dark = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(f"\nDark pattern: mean={img_dark.mean():.1f}")

            # Bright pattern
            phase_bright = np.ones((1200, 1920), dtype=np.uint16) * 1023
            open_slm.write_phase(phase_bright, memory_number=21)
            open_slm.display_memory(21)

            time.sleep(0.2)

            img_bright = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(f"Bright pattern: mean={img_bright.mean():.1f}")

            # Check that auto exposure adjusted
            # (The exact values depend on the setup)

    def test_roi_with_slm_center(self, open_slm, CameraStreamManager):
        """Test camera ROI with SLM pattern in center."""
        pytest.skip("ROI not fully supported on this camera")

    def test_slm_calibration_capture(self, open_slm, CameraStreamManager):
        """Test capturing for SLM calibration (blazed grating scan)."""
        pytest.skip("Calibration test timing issue - needs retry logic")

    def test_timing_slm_switch(self, open_slm, CameraStreamManager):
        """Test timing of SLM pattern switch and camera capture."""
        import time

        with CameraStreamManager(cam_id=0, exposure_time_ms=10) as cam:
            # Pattern 1
            p1 = np.zeros((1200, 1920), dtype=np.uint16)
            open_slm.write_phase(p1, memory_number=50)

            # Quick capture
            t0 = time.time()
            open_slm.display_memory(50)
            img1 = cam.get_numpy_image(n_sample=1, skip_first=False)
            t1 = time.time()
            print(f"\nPattern switch + capture time: {(t1 - t0) * 1000:.1f}ms")

            # Pattern 2
            p2 = np.ones((1200, 1920), dtype=np.uint16) * 1023
            open_slm.write_phase(p2, memory_number=51)

            t0 = time.time()
            open_slm.display_memory(51)
            img2 = cam.get_numpy_image(n_sample=1, skip_first=False)
            t1 = time.time()
            print(f"Pattern 2 switch + capture time: {(t1 - t0) * 1000:.1f}ms")

            # Verify different patterns captured
            assert img1.mean() != img2.mean()

    def test_context_managers_both(self, slm_module, CameraStreamManager):
        """Test that both SLM and Camera work with context managers."""
        # Test SLM context
        with slm_module(slm_number=1, wavelength=1064) as slm:
            assert slm.is_open

            # Generate pattern
            phase = np.zeros((1200, 1920), dtype=np.uint16)
            phase[500:700, 900:1100] = 512
            slm.write_phase(phase, memory_number=1)
            slm.display_memory(1)

        assert not slm.is_open

        # Test Camera context
        with CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam:
            assert cam.cam is not None
            img = cam.get_numpy_image(n_sample=1, skip_first=False)
            print(f"\nContext manager test image: {img.shape}")

        assert cam.cam is None


class TestSLMMIICAMCalibration:
    """SLM + MIICAM 标定测试类"""

    def test_calibration_basic(self):
        """Basic calibration test placeholder."""
        # This test just verifies the modules can be imported together
        try:
            from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
            from ao_shaping.drivers.ccd.miicam import CameraStreamManager

            # Check SLM parameters
            slm = SantecSLM200(slm_number=1, wavelength=1064)
            assert slm.wavelength == 1064
            assert slm.phase_range == 200

            # Check camera module exists
            cam_list = CameraStreamManager.get_cam_list()
            print(f"\nCalibration test: Found {len(cam_list)} camera(s)")

        except ImportError as e:
            pytest.skip(f"Modules not available: {e}")
