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

try:
    from ao_shaping.drivers.ccd import DahengCamera
    __all__ += ["DahengCamera"]
except Exception as e:
    logger.debug(f"DahengCamera not available: {e}")
    DahengCamera = None
    
try:
    from ao_shaping.drivers.ccd import MIICamera, MIICAMError
    __all__ += ["MIICamera", "MIICAMError"]
except Exception as e:
    logger.debug(f"MIICamera not available: {e}")
    MIICamera = None
    MIICAMError = None

try:
    from ao_shaping.drivers.ccd.ffmpeg import FFmpegCamera, FFmpegCameraError

    __all__ += ["FFmpegCamera", "FFmpegCameraError"]
except ImportError as e:
    logger.debug(f"FFmpegCamera not available: {e}")
    FFmpegCamera = None
    FFmpegCameraError = None

try:
    from ao_shaping.drivers.slm.santec import Santec, SantecError

    __all__ += ["Santec", "SantecError"]
except ImportError as e:
    logger.warning(f"Santec not available: {e}")
    Santec = None
    SantecError = None


from ao_shaping.drivers.mock_devices import (
    MockADC,
    MockADCError,
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
    "MockADC",
    "MockADCError",
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

# ---------------------------------------------------------------------------
# ADC driver (NI DAQ)
# ---------------------------------------------------------------------------
try:
    from ao_shaping.drivers.adc import NidaqADC, NidaqADCError, NidaqADCNotFoundError

    __all__ += ["NidaqADC", "NidaqADCError", "NidaqADCNotFoundError"]
except ImportError as e:
    logger.debug(f"NidaqADC not available: {e}")
    NidaqADC = None
    NidaqADCError = None
    NidaqADCNotFoundError = None

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
