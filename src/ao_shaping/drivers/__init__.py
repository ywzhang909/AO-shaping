"""Hardware drivers package.

This package provides unified interfaces for various hardware devices,
including cameras, SLMs, DMs, and wavefront sensors.
"""

from typing import TYPE_CHECKING

from loguru import logger

from ao_shaping.drivers._lazy import install_lazy_attrs
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

# SDK- and hardware-backed drivers resolve lazily (PEP 562). They are
# deliberately NOT bound at module scope: a defined global would shadow
# ``__getattr__`` and re-introduce the eager import. Eagerly importing them made
# every ``import ao_shaping.drivers`` — including pure-python consumers such as
# the sim and mock backends — load native SDKs (MIIUSB.dll, gxipy, SLMFunc.dll)
# or pay for large driver modules. A backend that cannot be imported still
# degrades to ``None``, as before.
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "MIICamera": ("ao_shaping.drivers.ccd.miicam", "MIICamera"),
    "DahengCamera": ("ao_shaping.drivers.ccd.daheng", "DahengCamera"),
    "FFmpegCamera": ("ao_shaping.drivers.ccd.ffmpeg", "FFmpegCamera"),
    "FFmpegCameraError": ("ao_shaping.drivers.ccd.ffmpeg", "FFmpegCameraError"),
    "Santec": ("ao_shaping.drivers.slm.santec", "Santec"),
    "SantecError": ("ao_shaping.drivers.slm.santec", "SantecError"),
    "NidaqADC": ("ao_shaping.drivers.adc", "NidaqADC"),
    "NidaqADCError": ("ao_shaping.drivers.adc", "NidaqADCError"),
    "NidaqADCNotFoundError": ("ao_shaping.drivers.adc", "NidaqADCNotFoundError"),
    "ZernikeSLM": ("ao_shaping.drivers.slm.zernike_slm", "ZernikeSLM"),
    "ZernikeSLMError": ("ao_shaping.drivers.slm.zernike_slm", "ZernikeSLMError"),
    "NlightDM": ("ao_shaping.drivers.dm.nlight", "NLight"),
    "HadamardDM": ("ao_shaping.drivers.dm.hadamard_dm", "HadamardDM"),
    "ZernikeDM": ("ao_shaping.drivers.dm.zernike_dm", "ZernikeDM"),
    "MicroDM": ("ao_shaping.drivers.dm.micro", "MicroDM"),
    "AsyncMicroDM": ("ao_shaping.drivers.dm.micro", "AsyncMicroDM"),
    "SimulateDM": ("ao_shaping.drivers.sim.dm", "SimulateDM"),
    "SimMicroDM": ("ao_shaping.drivers.sim.dm", "SimMicroDM"),
}

__all__ += [
    "MIICamera",
    "DahengCamera",
    "FFmpegCamera",
    "FFmpegCameraError",
    "Santec",
    "SantecError",
    "NidaqADC",
    "NidaqADCError",
    "NidaqADCNotFoundError",
    "ZernikeSLM",
    "ZernikeSLMError",
    "NlightDM",
    "HadamardDM",
    "ZernikeDM",
    "MicroDM",
    "AsyncMicroDM",
    "SimulateDM",
    "SimMicroDM",
]

if TYPE_CHECKING:
    # Static analyzers need the names to be visible; at runtime they are bound
    # by ``__getattr__`` on first access instead.
    from ao_shaping.drivers.adc import NidaqADC, NidaqADCError, NidaqADCNotFoundError
    from ao_shaping.drivers.ccd.daheng import DahengCamera
    from ao_shaping.drivers.ccd.ffmpeg import FFmpegCamera, FFmpegCameraError
    from ao_shaping.drivers.ccd.miicam import MIICamera
    from ao_shaping.drivers.dm.micro import AsyncMicroDM
    from ao_shaping.drivers.dm.hadamard_dm import HadamardDM
    from ao_shaping.drivers.dm.micro import MicroDM
    from ao_shaping.drivers.dm.nlight import NLight as NlightDM
    from ao_shaping.drivers.dm.zernike_dm import ZernikeDM
    from ao_shaping.drivers.sim.dm import SimMicroDM, SimulateDM
    from ao_shaping.drivers.slm.santec import Santec, SantecError
    from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM, ZernikeSLMError


__getattr__ = install_lazy_attrs(globals(), _LAZY_BACKENDS, __name__)


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
