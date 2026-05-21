"""Hardware integration tests for all devices.

Tests all functionality with actual hardware connected.
Results are output via print statements.

Run with: pytest tests/ao_shaping/drivers/hardware/ -v --hardware -s
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.mark.hardware
class TestWFS:
    """Thorlabs WFS hardware integration tests."""

    def test_initialize_and_info(self, wfs):
        """Test WFS initialization and hardware info."""
        print(f"\nWFS Device: {wfs.device_name}")
        print(f"Serial: {wfs.serial_num}")
        print(f"MLA Index: {wfs.mla_index}")
        print(f"Resolution: {wfs.image_pix}")
        print(f"Spots: {wfs.num_spots_x}x{wfs.num_spots_y}")
        assert wfs.device_name != ""
        assert wfs.serial_num != ""
        assert wfs.num_spots_x > 0
        assert wfs.num_spots_y > 0

    def test_mla_name(self, wfs):
        """Test get_mla_name returns valid MLA name."""
        mla_name = wfs.get_mla_name()
        print(f"\nMLA Name: {mla_name}")
        assert isinstance(mla_name, str)
        assert len(mla_name) > 0

    def test_exposure_range(self, wfs):
        """Test exposure time range query."""
        min_exp, max_exp, step = wfs.get_exposure_time_range()
        print(f"\nExposure Range: {min_exp:.3f} - {max_exp:.3f} ms (step: {step:.3f})")
        assert min_exp > 0
        assert max_exp > min_exp

    def test_optimize_pupil(self, wfs):
        """Test pupil optimization."""
        pupil = wfs.optimize_pupil()
        print(f"\nPupil: center=({pupil[0]:.2f}, {pupil[1]:.2f}), "
              f"diameter=({pupil[2]:.2f}, {pupil[3]:.2f})")
        assert len(pupil) == 4

    def test_take_image(self, wfs):
        """Test image capture."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        img = wfs.get_spotfiled_image()
        print(f"\nImage shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_spots_statics(self, wfs):
        """Test spot statistics."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        intensities, (cx, cy) = wfs.get_spots_statics()
        print(f"\nSpots shape: {intensities.shape}")
        print(f"Max intensity: {intensities.max():.1f}")
        print(f"Mean intensity: {intensities.mean():.1f}")
        assert intensities.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert cx.shape == intensities.shape
        assert cy.shape == intensities.shape

    def test_wavefront(self, wfs):
        """Test wavefront measurement."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        wf, stats = wfs.get_wavefront(cancel_tile=False)
        print(f"\nWavefront shape: {wf.shape}")
        print(f"RMS: {stats['rms']:.4f}")
        print(f"Weighted RMS: {stats['wighted_rms']:.4f}")
        print(f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        print(f"Mean: {stats['mean']:.4f}")
        assert wf.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert stats['rms'] >= 0

    def test_wavefront_with_tilt_cancel(self, wfs):
        """Test wavefront with tilt cancellation."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        wf_no_tilt, stats_no_tilt = wfs.get_wavefront(cancel_tile=True)
        wf_with_tilt, stats_with_tilt = wfs.get_wavefront(cancel_tile=False)
        print(f"\nWavefront with tilt - RMS: {stats_with_tilt['rms']:.4f}")
        print(f"Wavefront without tilt - RMS: {stats_no_tilt['rms']:.4f}")
        assert wf_no_tilt.shape == wf_with_tilt.shape

    def test_spot_deviation(self, wfs):
        """Test spot deviation measurement."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        dev_x, dev_y = wfs.get_spot_deviation(cancel_tile=False)
        print(f"\nDeviation shape: {dev_x.shape}")
        print(f"Dev X - mean: {dev_x.mean():.4f}, std: {dev_x.std():.4f}")
        print(f"Dev Y - mean: {dev_y.mean():.4f}, std: {dev_y.std():.4f}")
        assert dev_x.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert dev_y.shape == dev_x.shape

    def test_zernike(self, wfs):
        """Test Zernike coefficient calculation."""
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        zernike = wfs.get_zernike(zernike_order=4)
        print(f"\nZernike coefficients (order 4): {len(zernike)} terms")
        for i, z in enumerate(zernike):
            print(f"  Z{i+1}: {z:.4f}")
        assert len(zernike) > 0

    def test_stable_wavefront(self, wfs_stable_sampling):
        """Test stable wavefront sampling."""
        wfs_stable_sampling.take_image(n_sample=3, dynamicNoiseCut=True)
        wf, stats = wfs_stable_sampling.get_wavefront(cancel_tile=False)
        print(f"\nStable Wavefront - RMS: {stats['rms']:.4f}")
        print(f"Stable sampling enabled: {wfs_stable_sampling.stable_sample_enable}")
        print(f"Stable samples: {wfs_stable_sampling.stable_sample_n}")
        assert wf.shape == (wfs_stable_sampling.num_spots_x, wfs_stable_sampling.num_spots_y)

    def test_stable_spot_deviation(self, wfs_stable_sampling):
        """Test stable spot deviation sampling."""
        wfs_stable_sampling.take_image(n_sample=3, dynamicNoiseCut=True)
        dev_x, dev_y = wfs_stable_sampling.get_spot_deviation(cancel_tile=False)
        print(f"\nStable Deviation - mean X: {dev_x.mean():.4f}, mean Y: {dev_y.mean():.4f}")
        assert dev_x.shape == (wfs_stable_sampling.num_spots_x, wfs_stable_sampling.num_spots_y)

    def test_build_subaperture_mask(self, wfs):
        """Test subaperture mask building."""
        mask, valid_indices = wfs.build_subaperture_mask(
            n_avg=5, threshold_ratio=0.3, edge_clip=1
        )
        print(f"\nSubaperture mask: {mask.shape}")
        print(f"Valid subapertures: {np.sum(mask)}/{mask.size}")
        print(f"Valid indices: {len(valid_indices)}")
        assert mask.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert np.sum(mask) > 0

    def test_ref_filename(self, wfs):
        """Test reference filename generation."""
        filename = wfs._get_ref_filename()
        print(f"\nRef filename: {filename}")
        assert filename.endswith(".ref")
        assert filename.startswith("WFS_")

    def test_save_load_config(self, wfs):
        """Test configuration save/load cycle."""
        wfs.save_config()
        config = wfs.load_config()
        print(f"\nConfig keys: {list(config.keys())}")
        print(f"Config mla_index: {config.get('mla_index')}")
        print(f"Config exposure_time: {config.get('exposure_time')}")
        assert len(config) > 0


@pytest.mark.hardware
class TestZernikeSLM:
    """ZernikeSLM hardware integration tests."""

    def test_initialize(self, zernike_slm):
        """Test ZernikeSLM initialization."""
        info = zernike_slm.get_hardware_info()
        print(f"\nZernikeSLM Info: {info}")
        assert zernike_slm.is_connected()

    def test_set_flat(self, zernike_slm):
        """Test setting flat phase."""
        zernike_slm.set_flat()
        print("\nSet flat phase: OK")

    def test_send_zernike(self, zernike_slm):
        """Test sending Zernike coefficients."""
        coeffs = np.zeros(zernike_slm._n_zernike)
        coeffs[4] = 2.0  # Add defocus
        zernike_slm.send_zernike(coeffs)
        print(f"\nSent Zernike coeffs: {coeffs}")
        current = zernike_slm.get_current_zernike_coeffs()
        print(f"Current coeffs: {current}")

    def test_send_zernike_to_memory(self, zernike_slm):
        """Test sending Zernike to memory slot."""
        coeffs = np.zeros(zernike_slm._n_zernike)
        coeffs[5] = 1.0  # Add astigmatism
        zernike_slm.send_zernike_to_memory(coeffs, memory_number=1)
        zernike_slm.display_memory(1)
        print(f"\nSent to memory slot 1: {coeffs}")

    def test_grayscale(self, zernike_slm):
        """Test grayscale control."""
        zernike_slm.set_grayscale(500)
        gs = zernike_slm._slm.get_current_grayscale()
        print(f"\nGrayscale set to 500, current: {gs}")

    def test_phase_retrieval(self, zernike_slm):
        """Test phase retrieval."""
        zernike_slm.set_flat()
        phase = zernike_slm.get_current_phase()
        print(f"\nPhase shape: {phase.shape}")
        print(f"Phase stats: min={phase.min():.2f}, max={phase.max():.2f}")


@pytest.mark.hardware
class TestDM:
    """NLight DM hardware integration tests."""

    def test_initialize(self, dm):
        """Test DM initialization."""
        print(f"\nDM Num: {dm.DM_Num}")
        print(f"V Min: {dm.V_Min}, V Max: {dm.V_Max}")
        assert dm.DM_Num > 0

    def test_send_voltages(self, dm):
        """Test sending voltages."""
        voltages = np.zeros(dm.DM_Num)
        dm.send_voltages(voltages, 0.1)
        print(f"\nSent {len(voltages)} zero voltages")
        positions = dm.get_actuator_positions()
        print(f"Actuator positions shape: {positions.shape}")

    def test_send_random_voltages(self, dm):
        """Test sending random voltages."""
        voltages = np.random.uniform(dm.V_Min, dm.V_Max, dm.DM_Num)
        safe = dm.check_dm_unit_grad_safe(voltages)
        print(f"\nRandom voltages - safe: {safe}")
        print(f"Voltage range: {voltages.min():.1f} - {voltages.max():.1f}")
        if safe:
            dm.send_voltages(voltages, 0.1)
            print("Sent random voltages: OK")

    def test_reset_all(self, dm):
        """Test reset all actuators."""
        dm.reset_all()
        print("\nReset all actuators: OK")

    def test_neighbors(self, dm):
        """Test neighbor lookup."""
        neighbors = dm.get_neighbors(0)
        print(f"\nNeighbors of actuator 0: {neighbors}")
        assert len(neighbors) > 0

    def test_hv_control(self, dm):
        """Test high voltage control."""
        dm.set_hv(True)
        print("\nHV ON: OK")
        dm.set_hv(False)
        print("HV OFF: OK")


@pytest.mark.hardware
class TestDahengCam:
    """Daheng camera hardware integration tests."""

    def test_initialize(self, daheng_cam):
        """Test camera initialization."""
        print(f"\nCam SN: {daheng_cam.sn}")
        print(f"Exposure: {daheng_cam.exposure_time:.1f} ms")
        assert daheng_cam.sn is not None

    def test_capture_image(self, daheng_cam):
        """Test image capture."""
        img = daheng_cam.get_numpy_image(n_sample=3, skip_first=True)
        print(f"\nImage shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_set_exposure(self, daheng_cam):
        """Test exposure time setting."""
        daheng_cam.reset_exposure_time(50)
        print(f"\nSet exposure to 50ms, actual: {daheng_cam.exposure_time:.1f}")

    def test_set_window(self, daheng_cam):
        """Test ROI window setting."""
        daheng_cam.reset_window(center=(100, 100), size=200)
        print("\nSet ROI window: center=(100,100), size=200")


@pytest.mark.hardware
class TestMiiCam:
    """MiiCam hardware integration tests."""

    def test_initialize(self, miicam):
        """Test camera initialization."""
        print(f"\nMiiCam initialized")
        print(f"Image format: {miicam.image_format}")

    def test_capture_image(self, miicam):
        """Test image capture."""
        img = miicam.get_numpy_image(n_sample=3, skip_first=True)
        print(f"\nImage shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_set_exposure(self, miicam):
        """Test exposure time setting."""
        miicam.reset_exposure_time(50)
        print(f"\nSet exposure to 50ms")

    def test_auto_exposure(self, miicam):
        """Test auto exposure control."""
        miicam.enable_auto_exposure(True)
        state = miicam.get_auto_exposure_state()
        print(f"\nAuto exposure state: {state}")
        miicam.enable_auto_exposure(False)


@pytest.mark.hardware
class TestWFS_SLM_Integration:
    """WFS + SLM integration tests."""

    def test_zernike_wavefront_response(self, wfs, zernike_slm):
        """Test WFS response to SLM Zernike changes."""
        zernike_slm.set_flat()
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        wf_flat, stats_flat = wfs.get_wavefront(cancel_tile=True)
        print(f"\nFlat wavefront RMS: {stats_flat['rms']:.4f}")

        coeffs = np.zeros(zernike_slm._n_zernike)
        coeffs[4] = 3.0  # Add defocus
        zernike_slm.send_zernike(coeffs)
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        wf_defocus, stats_defocus = wfs.get_wavefront(cancel_tile=True)
        print(f"Defocus wavefront RMS: {stats_defocus['rms']:.4f}")

        zernike_slm.set_flat()
        print(f"Restored flat: OK")

    def test_zernike_deviation_response(self, wfs, zernike_slm):
        """Test spot deviation response to SLM changes."""
        zernike_slm.set_flat()
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        dev_flat_x, dev_flat_y = wfs.get_spot_deviation(cancel_tile=False)
        print(f"\nFlat deviation X std: {dev_flat_x.std():.4f}")

        coeffs = np.zeros(zernike_slm._n_zernike)
        coeffs[1] = 2.0  # Add tilt X
        zernike_slm.send_zernike(coeffs)
        wfs.take_image(n_sample=3, dynamicNoiseCut=True)
        dev_tilt_x, dev_tilt_y = wfs.get_spot_deviation(cancel_tile=False)
        print(f"Tilt deviation X std: {dev_tilt_x.std():.4f}")

        zernike_slm.set_flat()
