"""Hardware drivers package.

This package provides unified interfaces for various hardware devices,
including cameras, SLMs, DMs, and wavefront sensors.
"""

# Device base classes for digital twin management
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

# Hardware-specific imports
from .wfs.thorlab_wfs import WFSManager as Thorlab_WFS
from .wfs.thorlab_wfs import MlaRes
from .dm.NLight import NLight as NlightDM

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
    "Thorlab_WFS",
    "MlaRes",
    "NlightDM",
]

import logging


logger = logging.getLogger(__name__)


# Try to import CameraStreamManager, preferring Daheng and falling back to MIICAM.
try:
    from .ccd.daheng import DahengCamManager

    __all__ += ["DahengCamManager"]
except Exception as daheng_error:
    try:
        from .ccd.miicam import CameraStreamManager

        __all__ += ["DahengCamManager"]
        logger.warning(
            "Daheng CameraStreamManager not available; using MIICAM fallback: %s",
            daheng_error,
        )
    except Exception as miicam_error:
        logger.warning(
            "CameraStreamManager not available. Daheng import failed: %s; "
            "MIICAM import failed: %s",
            daheng_error,
            miicam_error,
        )
        DahengCamManager = None

# Try to import FFmpegCamera, but make it optional
try:
    from .ccd.ffmpeg import FFmpegCamera, FFmpegCameraError

    __all__ += ["FFmpegCamera", "FFmpegCameraError"]
except ImportError as e:
    logger.debug(f"FFmpegCamera not available: {e}")
    FFmpegCamera = None
    FFmpegCameraError = None

# Try to import SantecSLM200, but make it optional
try:
    from .slm.santec_slm200 import SantecSLM200, SantecSLM200Error

    __all__ += ["SantecSLM200", "SantecSLM200Error"]
except ImportError as e:
    logger.warning(f"SantecSLM200 not available: {e}")
    SantecSLM200 = None
    SantecSLM200Error = None

# Try to import PyVISA base components, but make them optional
try:
    from .visa_base import (
        VisaResourceManager,
        VisaInstrument,
        VisaInstrumentFactory,
        VisaError,
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

# Try to import SLM VISA wrapper, but make it optional
try:
    from .slm.santec_slm200_visa import SantecSLM200Visa, create_slm_visa_instrument

    __all__ += ["SantecSLM200Visa", "create_slm_visa_instrument"]
except ImportError as e:
    logger.debug(f"SantecSLM200Visa not available: {e}")
    SantecSLM200Visa = None
    create_slm_visa_instrument = None

# Import mock devices for testing
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

# Import simulated devices (wrapping sim.digitaltwin)
from ao_shaping.drivers.sim import (
    SimulatedCCD,
    SimulatedLaser,
    SimulatedSLM,
    SimulatedLens,
    SimulatedAperture,
    SimulatedTurbulentScreen,
    SimulatedThermalScreen,
    SimulatedATP,
    # Base classes
    SimulatedDevice,
    OpticalDevice,
    WavefrontProcessor,
)

__all__ += [
    # Simulated devices
    "SimulatedCCD",
    "SimulatedLaser",
    "SimulatedSLM",
    "SimulatedLens",
    "SimulatedAperture",
    "SimulatedTurbulentScreen",
    "SimulatedThermalScreen",
    "SimulatedATP",
    # Base classes
    "SimulatedDevice",
    "OpticalDevice",
    "WavefrontProcessor",
]
