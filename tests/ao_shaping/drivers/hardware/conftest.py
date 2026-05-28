"""Hardware integration test fixtures.

Provides fixtures for testing with actual hardware devices:
- WFS (Thorlabs Wavefront Sensor)
- SLM (Santec SLM-200 / ZernikeSLM)
- DM (NLight Deformable Mirror)
- CCD Cameras (Daheng / MiiCam)

Usage:
    pytest tests/ao_shaping/drivers/hardware/ -v --hardware

Run with actual hardware connected. Tests skip automatically if devices unavailable.
"""
from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run hardware integration tests (requires connected devices)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring actual hardware"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hardware"):
        return
    skip_hardware = pytest.mark.skip(reason="need --hardware option to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)


@pytest.fixture(scope="session")
def tmp_hardware_dir(tmp_path_factory):
    """Create temporary directory for hardware test artifacts."""
    return tmp_path_factory.mktemp("hardware_test")


# ---------------------------------------------------------------------------
# WFS Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wfs():
    """Thorlabs WFS instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS
        wfs = ThorlabWFS(MlaRes.Res768, use_custom_ref=False, high_speed=True)
        wfs.open()
        yield wfs
        wfs.close()
    except Exception as e:
        pytest.skip(f"WFS hardware not available: {e}")


@pytest.fixture(scope="module")
def wfs_with_custom_ref():
    """WFS with custom reference plane. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers import MlaRes, ThorlabWFS
        wfs = ThorlabWFS(MlaRes.Res768, use_custom_ref=True, high_speed=False)
        wfs.open()
        yield wfs
        wfs.close()
    except Exception as e:
        pytest.skip(f"WFS hardware not available: {e}")


@pytest.fixture(scope="module")
def wfs_stable_sampling():
    """WFS with stable sampling enabled."""
    try:
        from ao_shaping.drivers import MlaRes, ThorlabWFS
        wfs = ThorlabWFS(
            MlaRes.Res768,
            use_custom_ref=False,
            high_speed=True,
            stable_sample_enable=True,
            stable_sample_n=3,
            stable_variance_threshold=0.5,
            stable_max_attempts=20,
        )
        wfs.open()
        yield wfs
        wfs.close()
    except Exception as e:
        pytest.skip(f"WFS hardware not available: {e}")


# ---------------------------------------------------------------------------
# SLM Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def slm():
    """Santec SLM-200 instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
        slm = SantecSLM200(slm_number=1, wavelength=1064, phase_range=200)
        slm.open()
        yield slm
        slm.close()
    except Exception as e:
        pytest.skip(f"SLM hardware not available: {e}")


@pytest.fixture(scope="module")
def zernike_slm():
    """ZernikeSLM instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.slm import ZernikeSLM
        zslm = ZernikeSLM(slm_number=1, wavelength=1064, n_max=4)
        zslm.open()
        yield zslm
        zslm.close()
    except Exception as e:
        pytest.skip(f"ZernikeSLM hardware not available: {e}")


# ---------------------------------------------------------------------------
# DM Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dm():
    """NLight DM instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.dm.NLight import NLight
        dm = NLight()
        dm.open()
        yield dm
        dm.close()
    except Exception as e:
        pytest.skip(f"DM hardware not available: {e}")


# ---------------------------------------------------------------------------
# CCD Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def daheng_cam():
    """Daheng camera instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.ccd.daheng import DahengCamManager
        cam = DahengCamManager(cam_id=0, exposure_time_ms=100)
        cam.initialize()
        yield cam
        cam.close()
    except Exception as e:
        pytest.skip(f"Daheng camera not available: {e}")


@pytest.fixture(scope="module")
def miicam():
    """MiiCam instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.ccd.miicam_driver import CameraStreamManager
        cam = CameraStreamManager(cam_id=0, exposure_time_ms=100)
        cam.initialize()
        yield cam
        cam.close()
    except Exception as e:
        pytest.skip(f"MiiCam not available: {e}")
