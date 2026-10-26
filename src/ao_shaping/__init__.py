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
    from ao_shaping import MIICamera, NlightDM, optimizer_rms_dm, optimize_pib

    # Wavefront optimization
    recorder = optimizer_rms_dm(epochs=1000)

    # PIB optimization
    recorder = optimize_pib(center="mass", epochs=4000)"""

from __future__ import annotations

# Redirect numba's JIT cache out of the repository tree *before* any submodule
# (or a third-party it pulls in) can import numba. ``@njit(cache=True)`` writes
# ``.nbi``/``.nbc`` next to the source module by default, which (a) breaks the
# "import must not touch disk" contract and (b) fails outright on a read-only
# mount. Pointing NUMBA_CACHE_DIR at a temp dir keeps the cache but moves it out
# of ``src/``. Forced only when unset/empty so an explicit user override wins.
import os
import tempfile

_cachedir = os.environ.get("NUMBA_CACHE_DIR", "")
if not _cachedir or not os.path.isabs(_cachedir):
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), "ao_shaping_numba_cache"
    )

from importlib import import_module
from typing import TYPE_CHECKING, Any

# ============================================================================
# Public API -- resolved on first access (PEP 562)
# ============================================================================
#
# Import contract (R-20). It is restated here because this module is what broke
# it: Python initialises ``ao_shaping/__init__.py`` before ANY ``ao_shaping.*``
# submodule, so the 17 eager ``from ao_shaping... import`` statements this file
# used to hold made every consumer -- including purely offline code -- pay for
# numpy, matplotlib, pandas and scipy up front. Measured with this module
# emptied, ``import ao_shaping.utils`` costs nothing, so this module was the sole
# violator and ``utils/`` was always compliant.
#
# Every module named below was measured to pull numpy, and most pull matplotlib,
# pandas and scipy too; hence the map rather than a handful of exceptions.
#
# _LAZY_ATTRS is ordinary public API: if its module is missing, that is a real
# error and it propagates. _LAZY_SOFT is the hardware backends, which keep the
# pre-existing behaviour of degrading to None when no SDK is installed.

_LAZY_ATTRS: dict[str, str] = {
    "Device": "ao_shaping.drivers.device_base.Device",
    "DeviceCapability": "ao_shaping.drivers.device_base.DeviceCapability",
    "DeviceError": "ao_shaping.drivers.device_base.DeviceError",
    "DeviceMetadata": "ao_shaping.drivers.device_base.DeviceMetadata",
    "DeviceNotFoundError": "ao_shaping.drivers.device_base.DeviceNotFoundError",
    "DeviceParameter": "ao_shaping.drivers.device_base.DeviceParameter",
    "DeviceRegistry": "ao_shaping.drivers.device_registry.DeviceRegistry",
    "DeviceState": "ao_shaping.drivers.device_base.DeviceState",
    "DeviceType": "ao_shaping.drivers.device_base.DeviceType",
    "ImageVoltagesDisplay": "ao_shaping.display.image_display.ImageVoltagesDisplay",
    "MockCamera": "ao_shaping.drivers.mock_devices.MockCamera",
    "MockDM": "ao_shaping.drivers.mock_devices.MockDM",
    "MockSLM": "ao_shaping.drivers.mock_devices.MockSLM",
    "MockWFS": "ao_shaping.drivers.mock_devices.MockWFS",
    "OpticalDevice": "ao_shaping.drivers.sim.base.OpticalDevice",
    "Recorder": "ao_shaping.utils.io.file.Recorder",
    "RegisteredDevice": "ao_shaping.drivers.device_registry.RegisteredDevice",
    "SimulatedCCD": "ao_shaping.drivers.sim.ccd.SimulatedCCD",
    "SimulatedDevice": "ao_shaping.drivers.sim.base.SimulatedDevice",
    "SimulatedDeviceError": "ao_shaping.drivers.sim.base.SimulatedDeviceError",
    "SimulatedLaser": "ao_shaping.drivers.sim.laser.SimulatedLaser",
    "SimulatedSLM": "ao_shaping.drivers.sim.optics.SimulatedSLM",
    "SimulatedTurbulentScreen": "ao_shaping.drivers.sim.atmos.screens.SimulatedTurbulentScreen",
    "WavefrontProcessor": "ao_shaping.drivers.sim.base.WavefrontProcessor",
    "ZernikeCentroidCalculator": "ao_shaping.utils.wavefront.wavefront_calc.ZernikeCentroidCalculator",
    "ZernikeGenerator": "ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator",
    "calc_n_zernike_terms": "ao_shaping.utils.wavefront.zernike_calc.calc_n_zernike_terms",
    "calculate_sharpness": "ao_shaping.utils.image.spots_calc.calculate_sharpness",
    "centroid": "ao_shaping.utils.image.spots_calc.centroid",
    "generate_noll_polynomial": "ao_shaping.utils.wavefront.zernike_calc.generate_noll_polynomial",
    "get_global_registry": "ao_shaping.drivers.device_registry.get_global_registry",
    "logger": "ao_shaping.utils.io.file.logger",
    "normalize_01": "ao_shaping.utils.wavefront.wavefront_calc.normalize_01",
    "optimize_pib": "ao_shaping.optimizer.wfless.pib.optimize_pib",
    "optimizer_rms_dm": "ao_shaping.optimizer.wf.rms.optimizer_rms_dm",
    "radius": "ao_shaping.utils.image.spots_calc.radius",
    "to_color": "ao_shaping.utils.wavefront.wavefront_calc.to_color",
    "zernike_radial": "ao_shaping.utils.wavefront.zernike_calc.zernike_radial",
}

#: Hardware backends -- an absent SDK degrades to None instead of raising.
_LAZY_SOFT: dict[str, str] = {
    "FFmpegCamera": "ao_shaping.drivers.FFmpegCamera",
    "MlaRes": "ao_shaping.drivers.MlaRes",
    "NlightDM": "ao_shaping.drivers.NlightDM",
    "Santec": "ao_shaping.drivers.Santec",
    "ThorlabWFS": "ao_shaping.drivers.ThorlabWFS",
}

# Camera backends delegate to ao_shaping.drivers, which applies the same lazy
# contract one level down. Importing a camera backend is not side-effect free:
# miicam.driver runs _setup_miicam_sdk(), rewriting sys.path and ctypes.CDLL-
# loading the native MIIUSB.dll, while daheng.driver imports the gxipy bindings.
_LAZY_CAMERAS = ("MIICamera", "DahengCamera")

__version__ = "0.2.0"

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from ao_shaping.display.image_display import ImageVoltagesDisplay
    from ao_shaping.drivers import FFmpegCamera, MlaRes, NlightDM, Santec, ThorlabWFS
    from ao_shaping.drivers.ccd.daheng import DahengCamera
    from ao_shaping.drivers.ccd.miicam import MIICamera
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
    from ao_shaping.drivers.mock_devices import MockCamera, MockDM, MockSLM, MockWFS
    from ao_shaping.drivers.sim.atmos.screens import SimulatedTurbulentScreen
    from ao_shaping.drivers.sim.base import (
        OpticalDevice,
        SimulatedDevice,
        SimulatedDeviceError,
        WavefrontProcessor,
    )
    from ao_shaping.drivers.sim.ccd import SimulatedCCD
    from ao_shaping.drivers.sim.laser import SimulatedLaser
    from ao_shaping.drivers.sim.optics import SimulatedSLM
    from ao_shaping.optimizer.wf.rms import optimizer_rms_dm
    from ao_shaping.optimizer.wfless.pib import optimize_pib
    from ao_shaping.utils.image.spots_calc import calculate_sharpness, centroid, radius
    from ao_shaping.utils.io.file import Recorder, logger
    from ao_shaping.utils.wavefront.wavefront_calc import (
        ZernikeCentroidCalculator,
        normalize_01,
        to_color,
    )
    from ao_shaping.utils.wavefront.zernike_calc import (
        ZernikeGenerator,
        calc_n_zernike_terms,
        generate_noll_polynomial,
        zernike_radial,
    )


def __getattr__(name: str) -> Any:
    """Resolve a public attribute on first access (PEP 562)."""
    if name in _LAZY_CAMERAS:
        from ao_shaping import drivers

        value = getattr(drivers, name)
        globals()[name] = value
        return value

    target = _LAZY_ATTRS.get(name) or _LAZY_SOFT.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.rpartition(".")
    try:
        value = getattr(import_module(module_path), attr)
    except ImportError:
        if name not in _LAZY_SOFT:
            raise
        value = None

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


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
    "MIICamera",
    "DahengCamera",
    "NlightDM",
    "ThorlabWFS",
    "Santec",
    "FFmpegCamera",
    "MlaRes",
    # Optimizers
    "optimizer_rms_dm",
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
