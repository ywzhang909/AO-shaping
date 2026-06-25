"""Hardware drivers package.

This package provides unified interfaces for various hardware devices,
including cameras, SLMs, DMs, and wavefront sensors.
"""

from loguru import logger

from ao_shaping.drivers.device_base import (
    Device,
    DeviceCapability,
    DeviceError,
    DeviceMetadata,
    DeviceNotFoundError,
    DeviceParameter,
    DeviceState,
    DeviceType,
)
from ao_shaping.drivers.device_registry import (
    DeviceRegistry,
    RegisteredDevice,
    get_global_registry,
)
from ao_shaping.drivers.dm.NLight import NLight as NlightDM
from ao_shaping.drivers.wfs import MlaRes, ThorlabWFS

__all__ = [
    # Base classes
    "Device",
    "DeviceCapability",
    "DeviceError",
    "DeviceMetadata",
    "DeviceNotFoundError",
    "DeviceParameter",
    "DeviceRegistry",
    "DeviceState",
    "DeviceType",
    "RegisteredDevice",
    "get_global_registry",
    # Hardware
    "ThorlabWFS",
    "MlaRes",
    "NlightDM",
]

CameraStreamManager = None

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamManager

    __all__ += ["DahengCamManager"]
    CameraStreamManager = DahengCamManager
    __all__ += ["CameraStreamManager"]
except Exception as daheng_error:
    try:
        from ao_shaping.drivers.ccd.miicam import CameraStreamManager

        __all__ += ["CameraStreamManager"]
        logger.warning(
            f"Daheng CameraStreamManager not available; using MIICAM fallback: {daheng_error}"
        )
    except Exception as miicam_error:
        logger.warning(
            f"CameraStreamManager not available. Daheng import failed: {daheng_error}; "
            f"MIICAM import failed: {miicam_error}"
        )
        CameraStreamManager = None

try:
    from ao_shaping.drivers.ccd.ffmpeg import FFmpegCamera, FFmpegCameraError

    __all__ += ["FFmpegCamera", "FFmpegCameraError"]
except ImportError as e:
    logger.debug(f"FFmpegCamera not available: {e}")
    FFmpegCamera = None
    FFmpegCameraError = None

try:
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200, SantecSLM200Error

    __all__ += ["SantecSLM200", "SantecSLM200Error"]
except ImportError as e:
    logger.warning(f"SantecSLM200 not available: {e}")
    SantecSLM200 = None
    SantecSLM200Error = None

try:
    from ao_shaping.drivers.visa_base import (
        VisaError,
        VisaInstrument,
        VisaInstrumentFactory,
        VisaResourceManager,
        is_pyvisa_available,
        list_visa_resources,
        open_visa_instrument,
    )

    __all__ += [
        "VisaResourceManager",
        "VisaInstrument",
        "VisaInstrumentFactory",
        "VisaError",
        "is_pyvisa_available",
        "list_visa_resources",
        "open_visa_instrument",
    ]
except ImportError as e:
    logger.debug(f"PyVISA components not available: {e}")
    VisaResourceManager = None
    VisaInstrument = None
    VisaInstrumentFactory = None
    VisaError = None
    is_pyvisa_available = lambda: False
    list_visa_resources = None
    open_visa_instrument = None



from ao_shaping.drivers.mock_devices import (
    MockCamera,
    MockCameraError,
    MockDM,
    MockDMError,
    MockFilter,
    MockFilterError,
    MockLaser,
    MockLaserError,
    MockSLM,
    MockSLMError,
    MockStage,
    MockStageError,
    MockWFS,
    MockWFSError,
)

__all__ += [
    "MockCamera",
    "MockCameraError",
    "MockDM",
    "MockDMError",
    "MockFilter",
    "MockFilterError",
    "MockLaser",
    "MockLaserError",
    "MockSLM",
    "MockSLMError",
    "MockStage",
    "MockStageError",
    "MockWFS",
    "MockWFSError",
]

from ao_shaping.drivers.sim import (
    OpticalDevice,
    SimulatedAperture,
    SimulatedATP,
    SimulatedCCD,
    SimulatedDevice,
    SimulatedLaser,
    SimulatedLens,
    SimulatedSLM,
    SimulatedThermalScreen,
    SimulatedTurbulentScreen,
    WavefrontProcessor,
)

__all__ += [
    "SimulatedCCD",
    "SimulatedLaser",
    "SimulatedSLM",
    "SimulatedLens",
    "SimulatedAperture",
    "SimulatedTurbulentScreen",
    "SimulatedThermalScreen",
    "SimulatedATP",
    "SimulatedDevice",
    "OpticalDevice",
    "WavefrontProcessor",
]
