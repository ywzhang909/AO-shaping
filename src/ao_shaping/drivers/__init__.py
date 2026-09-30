"""Hardware drivers package.

This package provides unified interfaces for various hardware devices,
including cameras, SLMs, DMs, and wavefront sensors.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

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

# Camera backends resolve lazily (PEP 562). They are deliberately NOT bound at
# module scope: a defined global would shadow ``__getattr__`` and re-introduce
# the eager import. Eagerly importing them made every ``import
# ao_shaping.drivers`` — including pure-python consumers such as the sim and
# mock backends — load the native MiiCam SDK and the gxipy bindings. A backend
# that cannot be imported still degrades to ``None``, as before.
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "MIICamera": ("ao_shaping.drivers.ccd.miicam", "MIICamera"),
    "DahengCamera": ("ao_shaping.drivers.ccd.daheng", "DahengCamera"),
}

__all__ += ["MIICamera", "DahengCamera"]

if TYPE_CHECKING:
    # Static analyzers need the names to be visible; at runtime they are bound
    # by ``__getattr__`` on first access instead.
    from ao_shaping.drivers.ccd.daheng import DahengCamera
    from ao_shaping.drivers.ccd.miicam import MIICamera


def __getattr__(name: str) -> Any:
    """Resolve an SDK-backed camera class on first access (PEP 562)."""
    try:
        module_path, attr = _LAZY_BACKENDS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    try:
        value = getattr(import_module(module_path), attr)
    except Exception as e:  # mirrors the previous catch-all degradation
        logger.debug(f"{name} driver not available: {e}")
        value = None

    globals()[name] = value
    return value

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
