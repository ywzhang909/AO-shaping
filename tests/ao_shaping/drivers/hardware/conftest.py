"""Hardware integration test fixtures.

Provides fixtures for testing with actual hardware devices:
- WFS (Thorlabs Wavefront Sensor)
- SLM (Santec SLM-200 / ZernikeSLM)
- DM (NLight Deformable Mirror)
- CCD Cameras (Daheng / MiiCam)

Usage:
    pytest tests/ao_shaping/drivers/hardware/ -v --hardware

Run with actual hardware connected. Tests skip automatically if devices unavailable.

Exception-handling contract (issue #174 batch 3):
    Each device fixture keeps a *broad* ``except Exception`` around
    ``import + construct + open/initialize + close``. This is intentional,
    not a BLE001 blind-except: the driver packages guard their native-SDK
    imports internally (``import gxipy`` / ``import miicam`` are wrapped), so
    the fixture's own ``import`` line never raises; the failure that should
    trigger ``pytest.skip`` surfaces in ``open()``/``initialize()`` and its
    type differs per device:

    - Thorlab WFS   -> OSError (WFS_64.dll / VISA missing)
    - Santec SLM    -> SantecError (``_slm_win`` / SDK import fails)
    - NLight DM     -> Exception ("Drv_UDPST.dll not found.")
    - Daheng CCD    -> NameError (``gx`` is undefined when the SDK import fails)
    - MiiCam        -> AttributeError (``miicam`` module is set to ``None``)
    - (device present but not responding) -> ConnectionAbortedError / *Error

    There is no single narrow base class meaning "device unavailable", so
    narrowing to ``ImportError`` (or any specific subclass) would turn a
    hardware-absent *skip* into an *error*. Do not narrow these.
"""
from pathlib import Path

import numpy as np
import pytest

from tests.ao_shaping.drivers._hardware_gate import (
    add_hardware_option,
    skip_marked_hardware_tests,
)


def pytest_addoption(parser):
    add_hardware_option(parser)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring actual hardware"
    )


def pytest_collection_modifyitems(config, items):
    skip_marked_hardware_tests(config, items)


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
    """Santec SLM-200 instance. Skips if hardware unavailable.

    Note: the constructor signature is (slm_number, use_120hz, wavelength,
    video_mode, shift_x, shift_y, correction_csv_path) -- there is no
    ``phase_range`` parameter (it used to be a legacy kwarg; the 200-unit
    phase range is now fixed internally by ``SLM_Ctrl_WriteWL``). Passing it
    raised ``TypeError`` on every call, which this fixture's broad ``except``
    swallowed and reported as "hardware not available", masking the real
    defect. See issue #174 batch 3.
    """
    try:
        from ao_shaping.drivers.slm.santec import Santec
        slm = Santec(slm_number=1, wavelength=1064)
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
        from ao_shaping.drivers.dm.nlight import NLight
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
        from ao_shaping.drivers.ccd.daheng import DahengCamera
        cam = DahengCamera(cam_id=0, exposure_time_ms=100)
        cam.initialize()
        yield cam
        cam.close()
    except Exception as e:
        pytest.skip(f"Daheng camera not available: {e}")


@pytest.fixture(scope="module")
def miicam():
    """MiiCam instance. Skips if hardware unavailable."""
    try:
        from ao_shaping.drivers.ccd.miicam.driver import MIICamera
        cam = MIICamera(cam_id=0, exposure_time_ms=100)
        cam.initialize()
        yield cam
        cam.close()
    except Exception as e:
        pytest.skip(f"MiiCam not available: {e}")
