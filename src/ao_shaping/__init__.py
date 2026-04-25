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
# Lazy submodule access - avoid circular imports at package load time
# ============================================================================
def __getattr__(name: str):
    """Lazy import to avoid circular import issues in the package."""
    
    # Core module imports (these may trigger circular imports, load lazily)
    _core_modules = {
        "optimizer_rms": ("ao_shaping.optimizer.wf.rms", "optimizer_rms"),
        "optimize_pib": ("ao_shaping.optimizer.wfless.pib", "optimize_pib"),
    }
    
    if name in _core_modules:
        import importlib
        module_name, attr_name = _core_modules[name]
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    
    # Base device classes
    if name in ("Device", "DeviceState", "DeviceType", "DeviceError", "DeviceParameter"):
        from ao_shaping.drivers import device_base
        return getattr(device_base, name)
    
    if name == "DeviceMetadata":
        from ao_shaping.drivers import device_base
        return getattr(device_base, name)
    
    # Simulated devices
    if name in ("SimulatedCCD", "SimulatedLaser", "SimulatedSLM", "SimulatedTurbulentScreen",
              "SimulatedDevice", "OpticalDevice", "WavefrontProcessor"):
        from ao_shaping.drivers import sim
        return getattr(sim, name)
    
    # Mock devices
    if name in ("MockCamera", "MockDM", "MockSLM", "MockWFS"):
        from ao_shaping.drivers import mock_devices
        return getattr(mock_devices, name)
    
    # Device aliases (conditionally loaded)
    if name == "CameraStreamManager":
        from ao_shaping import drivers
        return getattr(drivers, "CameraStreamManager", None)
    
    if name == "DahengCamManager":
        from ao_shaping import drivers
        return getattr(drivers, "DahengCamManager", None)
    
    if name == "NlightDM":
        from ao_shaping import drivers
        return getattr(drivers, "NlightDM", None)
    
    if name == "Thorlab_WFS":
        from ao_shaping import drivers
        return getattr(drivers, "Thorlab_WFS", None)
    
    if name == "SantecSLM200":
        from ao_shaping import drivers
        return getattr(drivers, "SantecSLM200", None)
    
    if name == "FFmpegCamera":
        from ao_shaping import drivers
        return getattr(drivers, "FFmpegCamera", None)
    
    if name == "MlaRes":
        from ao_shaping import drivers
        return getattr(drivers, "MlaRes", None)
    
    # Utilities - load directly to avoid circular import
    if name in ("logger", "Recorder"):
        from ao_shaping import utils
        return getattr(utils, name, None)
    
    if name == "ImageVoltagesDisplay":
        from ao_shaping.utils import display as _display
        return getattr(_display, "ImageVoltagesDisplay", None)
    
    # Spots calculation
    if name in ("centroid", "radius", "calculate_sharpness", "get_centroid"):
        from ao_shaping.utils import spots_calc
        return getattr(spots_calc, name)
    
    # Wavefront utilities
    if name in ("zernike_coeffs", "zernike_fit", "wavefront_rms"):
        from ao_shaping.utils import wavefront_calc
        return getattr(wavefront_calc, name)
    
    # Zernike polynomial utilities
    if name in ("ZernikePolynomial", "zernike_polynomial", "generate_zernike_basis",
               "calculate_zernike_coefficients"):
        from ao_shaping.utils import zernike_calc
        return getattr(zernike_calc, name)
    
    raise AttributeError(f"module 'ao_shaping' has no attribute '{name}'")


# ============================================================================
# Public API - Explicit __all__ for clear interface
# ============================================================================
__all__ = [
    # Version
    "__version__",
    # Base device classes
    "Device",
    "DeviceState",
    "DeviceType",
    "DeviceError",
    "DeviceParameter",
    "DeviceMetadata",
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
    # Hardware device aliases (conditionally available)
    "CameraStreamManager",
    "DahengCamManager",
    "NlightDM",
    "Thorlab_WFS",
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
    "get_centroid",
    "zernike_coeffs",
    "zernike_fit",
    "wavefront_rms",
    "ZernikePolynomial",
    "zernike_polynomial",
    "generate_zernike_basis",
    "calculate_zernike_coefficients",
]