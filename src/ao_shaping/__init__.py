"""AO-Shaping: Adaptive Optics Beam Shaping System.

A reinforcement learning based adaptive optics system for wavefront
correction and beam shaping. Integrates multiple optimization algorithms
including WFS-based and wavefront-sensorless methods.

Main Features:
- Wavefront RMS optimization using WFS
- Power-in-Bucket (PIB) optimization
- Reinforcement learning (SAC) integration
- Hardware drivers for cameras, DMs, SLMs, WFS

Example:
    from ao_shaping import CameraStreamManager, NlightDM, optimizer_rms, optimize_pib

    # Wavefront optimization
    recorder = optimizer_rms(epochs=1000)

    # PIB optimization
    recorder = optimize_pib(center="mass", epochs=4000)
"""

from __future__ import annotations

# ============================================================================
# Version
# ============================================================================
__version__ = "0.2.0"

# ============================================================================
# Direct imports - no circular dependencies in this structure:
# optimizer → drivers, utils, algorithm
# drivers → utils  
# utils → (external only)
# algorithm → utils
# ============================================================================

# Base device classes
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

# Mock devices for testing
from ao_shaping.drivers.mock_devices import (
    MockCamera,
    MockDM,
    MockSLM,
    MockWFS,
)
from ao_shaping.drivers.sim.atmos.screens import SimulatedTurbulentScreen

# Simulated devices
from ao_shaping.drivers.sim.base import (
    OpticalDevice,
    SimulatedDevice,
    SimulatedDeviceError,
    WavefrontProcessor,
)
from ao_shaping.drivers.sim.ccd import SimulatedCCD
from ao_shaping.drivers.sim.laser import SimulatedLaser
from ao_shaping.drivers.sim.optics import SimulatedSLM

# Hardware device aliases (gracefully skip if SDK not available)
try:
    from ao_shaping.drivers import (
        CameraStreamManager,
        DahengCamManager,
        FFmpegCamera,
        MlaRes,
        NlightDM,
        SantecSLM200,
        ThorlabWFS,
    )
except ImportError:
    # SDK not available - provide None aliases
    CameraStreamManager = None
    DahengCamManager = None
    NlightDM = None
    ThorlabWFS = None
    SantecSLM200 = None
    FFmpegCamera = None
    MlaRes = None

# Optimizers (loaded here to expose in package namespace)
from ao_shaping.optimizer.wf.rms import optimizer_rms
from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.display import ImageVoltagesDisplay

# Utilities
from ao_shaping.utils.file import Recorder, logger
from ao_shaping.utils.spots_calc import (
    calculate_sharpness,
    centroid,
    radius,
)
from ao_shaping.utils.wavefront_calc import (
    ZernikeCentroidCalculator,
    normalize_01,
    to_color,
)
from ao_shaping.utils.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
    generate_noll_polynomial,
    zernike_radial,
)

# ============================================================================
# Public API - Explicit __all__ for clear interface
# ============================================================================
__all__ = [
    # Version
    "__version__",
    # Base device classes
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
    # Simulated devices
    "SimulatedCCD",
    "SimulatedLaser",
    "SimulatedSLM",
    "SimulatedTurbulentScreen",
    "SimulatedDevice",
    "OpticalDevice",
    "WavefrontProcessor",
    # Mock devices for testing
    "MockCamera",
    "MockDM",
    "MockSLM",
    "MockWFS",
    # Hardware device aliases
    "CameraStreamManager",
    "DahengCamManager",
    "NlightDM",
    "ThorlabWFS",
    "SantecSLM200",
    "FFmpegCamera",
    "MlaRes",
    # Optimizers
    "optimizer_rms",
    "optimize_pib",
# Utilities
    "logger",
    "Recorder",
    "ImageVoltagesDisplay",
    "centroid",
    "radius",
    "calculate_sharpness",
    "ZernikeGenerator",
    "zernike_radial",
    "generate_noll_polynomial",
    "calc_n_zernike_terms",
    "ZernikeCentroidCalculator",
    "normalize_01",
    "to_color",
]