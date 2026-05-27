"""Hardware integration tests for all devices.

These tests require actual hardware devices to be connected.
They will be skipped if hardware is not available.

Run with:
    pytest tests/ao_shaping/drivers/test_hardware_integration.py -v -s

Environment variables to control which devices to test:
    TEST_WFS=1      - Enable WFS tests (default: 1)
    TEST_SLM=1      - Enable SLM tests (default: 1)
    TEST_DM=1       - Enable DM tests (default: 1)
    TEST_CCD=1      - Enable CCD tests (default: 1)
"""

import os
import pytest
import numpy as np
from pathlib import Path

# Check which hardware tests to enable
ENABLE_WFS = os.environ.get("TEST_WFS", "1") == "1"
ENABLE_SLM = os.environ.get("TEST_SLM", "1") == "1"
ENABLE_DM = os.environ.get("TEST_DM", "1") == "1"
ENABLE_CCD = os.environ.get("TEST_CCD", "1") == "1"


# ============================================================================
# WFS Hardware Integration Tests
# ============================================================================

class TestWFSHardware:
    """Thorlabs WFS hardware integration tests."""

    @pytest.fixture(scope="class")
    def wfs(self):
        """Create and initialize WFS instance."""
        if not ENABLE_WFS:
            pytest.skip("WFS tests disabled (set TEST_WFS=1 to enable)")

        try:
            from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager, MlaRes
            from ao_shaping.drivers import Thorlab_WFS

            wfs = Thorlab_WFS(MlaRes.Res768, use_custom_ref=False, high_speed=True)
            wfs.open()
            print(f"\n[WFS] Connected: {wfs.device_name}, SN: {wfs.serial_num}")
            print(f"[WFS] MLA: {wfs.mla_index}, Spots: {wfs.num_spots_x}x{wfs.num_spots_y}")
            yield wfs
            wfs.close()
            print("[WFS] Closed")
        except Exception as e:
            pytest.skip(f"WFS hardware not available: {e}")

    def test_connection(self, wfs):
        """Test WFS connection and basic info."""
        print(f"  Device: {wfs.device_name}")
        print(f"  Serial: {wfs.serial_num}")
        print(f"  MLA: {wfs.mla_index.name}")
        print(f"  Resolution: {wfs.image_pix}")
        print(f"  Spots: {wfs.num_spots_x} x {wfs.num_spots_y}")
        assert wfs.device_name, "Device name should not be empty"
        assert wfs.serial_num, "Serial number should not be empty"

    def test_mla_name(self, wfs):
        """Test get_mla_name returns valid string."""
        mla_name = wfs.get_mla_name()
        print(f"  MLA Name: {mla_name}")
        assert isinstance(mla_name, str)
        assert len(mla_name) > 0

    def test_exposure_time_range(self, wfs):
        """Test exposure time range query."""
        min_exp, max_exp, step = wfs.get_exposure_time_range()
        print(f"  Exposure range: {min_exp:.3f} - {max_exp:.3f} ms (step: {step:.3f})")
        assert min_exp > 0
        assert max_exp > min_exp

    def test_take_image(self, wfs):
        """Test image capture."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        print(f"  Image captured: {wfs.image_pix}")
        assert wfs._image_captured

    def test_get_spotfield_image(self, wfs):
        """Test spotfield image retrieval."""
        wfs.take_image()
        img = wfs.get_spotfiled_image()
        print(f"  Spotfield image shape: {img.shape}, dtype: {img.dtype}")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

    def test_get_spots_statics(self, wfs):
        """Test spot statistics retrieval."""
        wfs.take_image()
        intensities, (cx, cy) = wfs.get_spots_statics()
        print(f"  Intensities shape: {intensities.shape}")
        print(f"  Centroid X range: [{cx.min():.3f}, {cx.max():.3f}]")
        print(f"  Centroid Y range: [{cy.min():.3f}, {cy.max():.3f}]")
        assert intensities.shape == (wfs.num_spots_x, wfs.num_spots_y)

    def test_get_wavefront(self, wfs):
        """Test wavefront measurement."""
        wfs.take_image()
        wf, stats = wfs.get_wavefront(cancel_tile=False)
        print(f"  Wavefront shape: {wf.shape}")
        print(f"  RMS: {stats['rms']:.4f}, Weighted RMS: {stats['wighted_rms']:.4f}")
        print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        assert isinstance(wf, np.ndarray)
        assert wf.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert 'rms' in stats

    def test_get_wavefront_with_tilt_cancel(self, wfs):
        """Test wavefront with tilt cancellation."""
        wfs.take_image()
        wf, stats = wfs.get_wavefront(cancel_tile=True)
        print(f"  Wavefront (tilt cancelled) RMS: {stats['rms']:.4f}")
        assert isinstance(wf, np.ndarray)

    def test_get_spot_deviation(self, wfs):
        """Test spot deviation measurement."""
        wfs.take_image()
        dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
        print(f"  Deviation X shape: {dev_x.shape}, range: [{dev_x.min():.4f}, {dev_x.max():.4f}]")
        print(f"  Deviation Y shape: {dev_y.shape}, range: [{dev_y.min():.4f}, {dev_y.max():.4f}]")
        assert dev_x.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert dev_y.shape == (wfs.num_spots_x, wfs.num_spots_y)

    def test_get_stable_spot_deviation(self, wfs):
        """Test stable spot deviation with intensity threshold."""
        wfs.take_image()
        dev_x, dev_y = wfs.get_stable_spot_deviation(intensity_threshold=0.0, cancel_tile=False)
        print(f"  Stable deviation X shape: {dev_x.shape}")
        print(f"  Stable deviation Y shape: {dev_y.shape}")
        assert dev_x.shape == (wfs.num_spots_x, wfs.num_spots_y)

    def test_get_zernike(self, wfs):
        """Test Zernike coefficient calculation."""
        wfs.take_image()
        zernike_coeffs = wfs.get_zernike(zernike_order=4)
        print(f"  Zernike coefficients: {len(zernike_coeffs)} terms")
        print(f"  First 5 coeffs: {zernike_coeffs[:5]}")
        assert isinstance(zernike_coeffs, np.ndarray)
        assert len(zernike_coeffs) > 0

    def test_optimize_pupil(self, wfs):
        """Test pupil optimization."""
        cx, cy, dx, dy = wfs.optimize_pupil()
        print(f"  Pupil: center=({cx:.2f}, {cy:.2f}), diameter=({dx:.2f}, {dy:.2f})")
        assert dx > 0 and dy > 0

    def test_stable_sampling(self, wfs):
        """Test stable sampling feature."""
        # Enable stable sampling
        wfs.stable_sample_enable = True
        wfs.stable_sample_n = 3
        wfs.stable_variance_threshold = 1.0
        wfs.stable_max_attempts = 10

        wfs.take_image()
        wf, stats = wfs.get_wavefront(cancel_tile=False)
        print(f"  Stable sampling: RMS={stats['rms']:.4f}")
        print(f"  Samples collected: {wfs.stable_sample_n}")

        # Reset
        wfs.stable_sample_enable = False

    def test_save_load_config(self, wfs):
        """Test configuration save/load cycle."""
        # Save current config
        wfs.save_config()
        print(f"  Config saved for SN: {wfs.serial_num}")

        # Load config
        config = wfs.load_config()
        print(f"  Config loaded: {list(config.keys())}")
        assert isinstance(config, dict)

    def test_context_manager(self):
        """Test WFS context manager protocol."""
        from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager, MlaRes
        from ao_shaping.drivers import Thorlab_WFS

        with Thorlab_WFS(MlaRes.Res768, use_custom_ref=False) as wfs:
            print(f"  Context manager: Connected to {wfs.device_name}")
            wfs.take_image()
            wf, stats = wfs.get_wavefront()
            print(f"  RMS in context: {stats['rms']:.4f}")
        print("  Context manager: Closed")


# ============================================================================
# SLM Hardware Integration Tests
# ============================================================================

class TestSLMHardware:
    """Santec SLM hardware integration tests."""

    @pytest.fixture(scope="class")
    def slm(self):
        """Create and initialize SLM instance."""
        if not ENABLE_SLM:
            pytest.skip("SLM tests disabled (set TEST_SLM=1 to enable)")

        try:
            from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200

            slm = SantecSLM200(slm_number=1, wavelength=532, phase_range=200)
            slm.open()
            print(f"\n[SLM] Connected: SN={slm.serial_number}")
            yield slm
            slm.close()
            print("[SLM] Closed")
        except Exception as e:
            pytest.skip(f"SLM hardware not available: {e}")

    def test_connection(self, slm):
        """Test SLM connection."""
        print(f"  Serial: {slm.serial_number}")
        print(f"  Wavelength: {slm.wavelength} nm")
        print(f"  Phase range: {slm.phase_range}")
        assert slm.is_open

    def test_wavelength_info(self, slm):
        """Test wavelength info query."""
        info = slm.get_wavelength_info()
        print(f"  Wavelength info: {info}")
        assert info is not None

    def test_set_grayscale(self, slm):
        """Test grayscale setting."""
        slm.set_grayscale(512)
        current = slm.get_current_grayscale()
        print(f"  Set grayscale: 512, Current: {current}")
        assert current is not None

    def test_write_phase(self, slm):
        """Test phase writing to memory."""
        phase = np.zeros((1200, 1920), dtype=np.uint16)
        slm.write_phase(phase, memory_number=1)
        print("  Phase written to memory 1")

    def test_display_memory(self, slm):
        """Test displaying from memory."""
        slm.display_memory(1)
        mem_num = slm.get_displayed_memory_number()
        print(f"  Displaying memory: {mem_num}")

    def test_display_data(self, slm):
        """Test direct phase display."""
        phase = np.zeros((1200, 1920), dtype=np.uint16)
        slm.display_data(phase, wait_time_s=0.1)
        print("  Phase displayed directly")

    def test_context_manager(self):
        """Test SLM context manager protocol."""
        from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200

        with SantecSLM200(slm_number=1, wavelength=532) as slm:
            print(f"  Context manager: Connected to SLM SN={slm.serial_number}")
            slm.set_grayscale(256)
        print("  Context manager: Closed")


class TestZernikeSLMHardware:
    """ZernikeSLM wrapper hardware integration tests."""

    @pytest.fixture(scope="class")
    def zernike_slm(self):
        """Create and initialize ZernikeSLM instance."""
        if not ENABLE_SLM:
            pytest.skip("SLM tests disabled (set TEST_SLM=1 to enable)")

        try:
            from ao_shaping.drivers.slm import ZernikeSLM

            slm = ZernikeSLM(slm_number=1, wavelength=532, n_max=4)
            slm.open()
            print(f"\n[ZernikeSLM] Connected, n_max=4")
            yield slm
            slm.close()
            print("[ZernikeSLM] Closed")
        except Exception as e:
            pytest.skip(f"ZernikeSLM hardware not available: {e}")

    def test_connection(self, zernike_slm):
        """Test ZernikeSLM connection."""
        info = zernike_slm.get_hardware_info()
        print(f"  Hardware info: {info}")
        assert zernike_slm.is_connected()

    def test_send_zernike(self, zernike_slm):
        """Test sending Zernike coefficients."""
        coeffs = np.zeros(15)  # n_max=4 -> 15 terms
        phase = zernike_slm.send_zernike(coeffs)
        print(f"  Zernike sent, phase shape: {phase.shape}")
        assert isinstance(phase, np.ndarray)

    def test_set_flat(self, zernike_slm):
        """Test setting flat phase."""
        zernike_slm.set_flat()
        print("  Flat phase set")
        current = zernike_slm.get_current_phase()
        print(f"  Current phase shape: {current.shape}")

    def test_zernike_coeffs(self, zernike_slm):
        """Test Zernike coefficient retrieval."""
        coeffs = zernike_slm.get_current_zernike_coeffs()
        print(f"  Current Zernike coeffs: {len(coeffs)} terms")
        print(f"  First 5: {coeffs[:5]}")


# ============================================================================
# DM Hardware Integration Tests
# ============================================================================

class TestDMHardware:
    """NLight DM hardware integration tests."""

    @pytest.fixture(scope="class")
    def dm(self):
        """Create and initialize DM instance."""
        if not ENABLE_DM:
            pytest.skip("DM tests disabled (set TEST_DM=1 to enable)")

        try:
            from ao_shaping.drivers.dm.NLight import NLight

            dm = NLight()
            dm.open()
            print(f"\n[DM] Connected: {dm.DM_Num} actuators")
            print(f"[DM] Voltage range: [{dm.V_Min}, {dm.V_Max}] V")
            yield dm
            dm.close()
            print("[DM] Closed")
        except Exception as e:
            pytest.skip(f"DM hardware not available: {e}")

    def test_connection(self, dm):
        """Test DM connection."""
        print(f"  Actuators: {dm.DM_Num}")
        print(f"  Voltage range: [{dm.V_Min}, {dm.V_Max}] V")
        assert dm.DM_Num > 0

    def test_send_voltages(self, dm):
        """Test sending voltage array."""
        voltages = np.zeros(dm.DM_Num)
        dm.send_voltages(voltages, wait_time_s=0.1)
        print(f"  Sent {len(voltages)} zero voltages")
        positions = dm.get_actuator_positions()
        print(f"  Actuator positions: {len(positions)}")

    def test_send_voltages_with_pattern(self, dm):
        """Test sending patterned voltages."""
        voltages = np.random.uniform(dm.V_Min, dm.V_Max, dm.DM_Num)
        dm.send_voltages(voltages, wait_time_s=0.1)
        print(f"  Sent random voltages: min={voltages.min():.1f}, max={voltages.max():.1f}")

    def test_check_dm_unit_grad_safe(self, dm):
        """Test voltage gradient safety check."""
        voltages = np.zeros(dm.DM_Num)
        is_safe = dm.check_dm_unit_grad_safe(voltages)
        print(f"  Zero voltages safe: {is_safe}")

        # Test with large gradient
        voltages[0] = dm.V_Max
        voltages[1] = dm.V_Min
        is_safe = dm.check_dm_unit_grad_safe(voltages)
        print(f"  Large gradient safe: {is_safe}")

    def test_get_neighbors(self, dm):
        """Test neighbor retrieval."""
        neighbors = dm.get_neighbors(0)
        print(f"  Actuator 0 neighbors: {len(neighbors)}")
        assert isinstance(neighbors, list)

    def test_reset_all(self, dm):
        """Test resetting all actuators."""
        dm.reset_all()
        print("  All actuators reset to zero")

    def test_context_manager(self):
        """Test DM context manager protocol."""
        from ao_shaping.drivers.dm.NLight import NLight

        with NLight() as dm:
            print(f"  Context manager: Connected to DM with {dm.DM_Num} actuators")
            dm.reset_all()
        print("  Context manager: Closed")


# ============================================================================
# CCD Hardware Integration Tests
# ============================================================================

class TestCCDHardware:
    """CCD camera hardware integration tests."""

    @pytest.fixture(scope="class")
    def daheng_cam(self):
        """Create and initialize Daheng camera."""
        if not ENABLE_CCD:
            pytest.skip("CCD tests disabled (set TEST_CCD=1 to enable)")

        try:
            from ao_shaping.drivers.ccd.daheng import DahengCamManager

            cam = DahengCamManager(cam_id=0)
            cam.initialize()
            print(f"\n[Daheng] Connected: SN={cam.sn}")
            yield cam
            cam.close()
            print("[Daheng] Closed")
        except Exception as e:
            pytest.skip(f"Daheng camera not available: {e}")

    @pytest.fixture(scope="class")
    def miicam(self):
        """Create and initialize MIICAM camera."""
        if not ENABLE_CCD:
            pytest.skip("CCD tests disabled (set TEST_CCD=1 to enable)")

        try:
            from ao_shaping.drivers.ccd.miicam_driver import CameraStreamManager

            cam = CameraStreamManager(cam_id=0)
            cam.initialize()
            print(f"\n[MIICAM] Connected")
            yield cam
            cam.close()
            print("[MIICAM] Closed")
        except Exception as e:
            pytest.skip(f"MIICAM not available: {e}")

    def test_daheng_connection(self, daheng_cam):
        """Test Daheng camera connection."""
        print(f"  Serial: {daheng_cam.sn}")
        print(f"  Exposure: {daheng_cam.exposure_time} ms")
        assert daheng_cam.sn

    def test_daheng_get_image(self, daheng_cam):
        """Test Daheng image capture."""
        img = daheng_cam.get_numpy_image(n_sample=3, skip_first=True)
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

    def test_daheng_exposure(self, daheng_cam):
        """Test Daheng exposure time setting."""
        original = daheng_cam.exposure_time
        daheng_cam.exposure_time = 50.0
        print(f"  Set exposure: 50.0 ms, Current: {daheng_cam.exposure_time}")
        daheng_cam.exposure_time = original

    def test_miicam_connection(self, miicam):
        """Test MIICAM camera connection."""
        print(f"  Connected to MIICAM")

    def test_miicam_get_image(self, miicam):
        """Test MIICAM image capture."""
        img = miicam.get_numpy_image(n_sample=3, skip_first=True)
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        assert isinstance(img, np.ndarray)

    def test_miicam_exposure(self, miicam):
        """Test MIICAM exposure time setting."""
        miicam.reset_exposure_time(50.0)
        print(f"  Set exposure: 50.0 ms")


# ============================================================================
# Full Pipeline Hardware Integration Test
# ============================================================================

class TestFullPipeline:
    """Full pipeline test: WFS + SLM + DM working together."""

    @pytest.fixture(scope="class")
    def full_setup(self):
        """Initialize all hardware devices."""
        if not (ENABLE_WFS and ENABLE_SLM and ENABLE_DM):
            pytest.skip("Full pipeline requires WFS, SLM, and DM (set TEST_WFS=1 TEST_SLM=1 TEST_DM=1)")

        setup = {}
        try:
            from ao_shaping.drivers import Thorlab_WFS, MlaRes
            from ao_shaping.drivers.slm import ZernikeSLM
            from ao_shaping.drivers.dm.NLight import NLight

            print("\n[Pipeline] Initializing all devices...")

            setup['wfs'] = Thorlab_WFS(MlaRes.Res768, use_custom_ref=False, high_speed=True)
            setup['wfs'].open()
            print(f"[Pipeline] WFS: {setup['wfs'].device_name}")

            setup['slm'] = ZernikeSLM(slm_number=1, wavelength=532, n_max=4)
            setup['slm'].open()
            print(f"[Pipeline] SLM: Connected")

            setup['dm'] = NLight()
            setup['dm'].open()
            print(f"[Pipeline] DM: {setup['dm'].DM_Num} actuators")

            yield setup

        except Exception as e:
            pytest.skip(f"Full pipeline not available: {e}")
        finally:
            for key, dev in setup.items():
                try:
                    dev.close()
                    print(f"[Pipeline] {key}: Closed")
                except:
                    pass

    def test_full_pipeline_wavefront(self, full_setup):
        """Test full pipeline: SLM phase -> WFS measurement."""
        wfs = full_setup['wfs']
        slm = full_setup['slm']

        # Set flat phase on SLM
        slm.set_flat()

        # Take image and measure wavefront
        wfs.take_image()
        wf, stats = wfs.get_wavefront(cancel_tile=True)

        print(f"\n  Pipeline: Flat phase RMS = {stats['rms']:.4f}")

        # Send random Zernike coefficients
        import numpy as np
        coeffs = np.random.uniform(-5, 5, 15)
        coeffs[0] = 0  # No piston
        slm.send_zernike(coeffs)

        wfs.take_image()
        wf2, stats2 = wfs.get_wavefront(cancel_tile=True)

        print(f"  Pipeline: Random Zernike RMS = {stats2['rms']:.4f}")
        print(f"  Pipeline: RMS change = {stats2['rms'] - stats['rms']:.4f}")

    def test_full_pipeline_dm_correction(self, full_setup):
        """Test full pipeline: DM voltage correction."""
        wfs = full_setup['wfs']
        dm = full_setup['dm']

        # Reset DM
        dm.reset_all()

        # Measure initial wavefront
        wfs.take_image()
        wf1, stats1 = wfs.get_wavefront()
        print(f"\n  Pipeline: Initial DM RMS = {stats1['rms']:.4f}")

        # Send small random voltages
        voltages = np.random.uniform(-50, 50, dm.DM_Num)
        dm.send_voltages(voltages, wait_time_s=0.1)

        # Measure after correction
        wfs.take_image()
        wf2, stats2 = wfs.get_wavefront()
        print(f"  Pipeline: After DM correction RMS = {stats2['rms']:.4f}")

        # Reset DM
        dm.reset_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
