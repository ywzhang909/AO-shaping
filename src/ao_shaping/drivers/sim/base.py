"""Simulated device base classes.

This module provides base classes for simulated optical devices,
extending the Device framework with simulation-specific functionality.

Architecture:
    - SimulatedDevice: Base class for all simulation devices
    - OpticalDevice: Specialized base for optical simulation devices
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import (
    Device,
    DeviceError,
    DeviceState,
    DeviceType,
)


class SimulatedDeviceError(DeviceError):
    """Base exception for simulated device errors."""
    pass


class SimulatedDevice(Device):
    """Base class for simulated devices.
    
    This class extends the Device base class with simulation-specific
    functionality, providing a foundation for all simulated hardware
    devices in the AO-Shaping framework.
    
    Attributes:
        device_type: DeviceType.OTHER for simulated devices
        manufacturer: "Simulation"
        model: Specific model name
    
    Example:
        >>> class MySimulatedDevice(SimulatedDevice):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._register_parameters()
        ...
        ...     def compute(self, *args):
        ...         # Simulation logic here
        ...         return result
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Generic Simulated Device"

    def __init__(
        self,
        device_id: str = "",
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """Initialize simulated device.
        
        Args:
            device_id: Unique device identifier. If empty, auto-generated.
            enable_noise: Whether to add noise to simulations.
            random_seed: Random seed for reproducible results.
        """
        super().__init__(device_id)

        self._simulation_enabled = True
        self._enable_noise = enable_noise
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        # Update metadata for simulation
        self._metadata.manufacturer = self.manufacturer
        self._metadata.model = self.model

        logger.debug(f"SimulatedDevice {self.__class__.__name__} initialized")

    # ========== Device Base Class Implementation ==========

    def open(self) -> None:
        """Open simulated device connection."""
        self._set_state(DeviceState.CONNECTING)
        # Simulate connection delay
        self._set_state(DeviceState.READY)
        logger.info(f"Simulated device {self.device_id} opened")

    def close(self) -> None:
        """Close simulated device connection."""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Simulated device {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get simulated hardware information.
        
        Returns:
            Dictionary containing simulation-specific hardware info.
        """
        return {
            "device_type": "simulation",
            "model": self.model,
            "manufacturer": self.manufacturer,
            "simulation_enabled": self._simulation_enabled,
            "noise_enabled": self._enable_noise,
            "random_seed": self._random_seed,
        }

    # ========== Simulation-Specific Methods ==========

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Execute simulation computation.
        
        This method must be implemented by subclasses to perform
        the actual simulation calculation.
        
        Returns:
            Simulation result (type depends on implementation).
        """
        pass

    def reset(self) -> None:
        """Reset simulation state to initial conditions."""
        logger.debug(f"Simulated device {self.device_id} reset")

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible simulations.
        
        Args:
            seed: Random seed value.
        """
        self._random_seed = seed
        self._rng = np.random.default_rng(seed)
        logger.debug(f"Random seed set to {seed}")

    def set_noise(self, enabled: bool) -> None:
        """Enable or disable noise in simulations.
        
        Args:
            enabled: Whether to add noise.
        """
        self._enable_noise = enabled
        logger.debug(f"Noise {'enabled' if enabled else 'disabled'}")

    def _generate_noise(self, shape: tuple, scale: float = 1.0) -> np.ndarray:
        """Generate noise array.
        
        Args:
            shape: Output array shape.
            scale: Noise amplitude scale factor.
            
        Returns:
            Noise array.
        """
        if self._enable_noise:
            return self._rng.normal(0, scale, shape)
        return np.zeros(shape)


class OpticalDevice(SimulatedDevice):
    """Base class for optical simulation devices.
    
    This class extends SimulatedDevice with optical-specific
    functionality, including wavelength handling and wavefront processing.
    
    Attributes:
        wavelength: Operating wavelength in nanometers.
    
    Example:
        >>> class MyOpticDevice(OpticalDevice):
        ...     def __init__(self, wavelength=1064):
        ...         super().__init__()
        ...         self.wavelength = wavelength
        ...
        ...     def process(self, wave):
        ...         # Process wavefront
        ...         return processed_wave
    """

    def __init__(
        self,
        device_id: str = "",
        wavelength: float = 1064.0,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """Initialize optical simulation device.
        
        Args:
            device_id: Unique device identifier.
            wavelength: Operating wavelength in nm.
            enable_noise: Whether to add noise.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(device_id, enable_noise, random_seed)

        self.wavelength = wavelength
        self._input_wave: Any | None = None
        self._output_wave: Any | None = None

    def set_input(self, wave: Any) -> None:
        """Set input wavefront for processing.
        
        Args:
            wave: Input wavefront object.
        """
        self._input_wave = wave
        logger.debug(f"Input wave set for {self.__class__.__name__}")

    def get_output(self) -> Any:
        """Get processed output wavefront.
        
        Returns:
            Output wavefront object.
        """
        return self._output_wave

    @abstractmethod
    def process(self, wave: Any) -> Any:
        """Process input wavefront.
        
        This method must be implemented by subclasses to perform
        the actual optical processing.
        
        Args:
            wave: Input wavefront to process.
            
        Returns:
            Processed wavefront.
        """
        pass


class WavefrontProcessor(OpticalDevice):
    """Base class for wavefront processing devices.
    
    Specialized optical device for wavefront manipulation,
    such as SLMs, lenses, and apertures.
    """

    def __init__(
        self,
        device_id: str = "",
        wavelength: float = 1064.0,
        npix: int = 512,
        dpix: float = 1e-3,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """Initialize wavefront processor.
        
        Args:
            device_id: Unique device identifier.
            wavelength: Operating wavelength in nm.
            npix: Number of pixels in wavefront array.
            dpix: Pixel size in meters.
            enable_noise: Whether to add noise.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(device_id, wavelength, enable_noise, random_seed)

        self.npix = npix
        self.dpix = dpix
        self._phase_pattern: np.ndarray | None = None

    def set_phase(self, phase: np.ndarray) -> None:
        """Set phase pattern.
        
        Args:
            phase: 2D phase array in radians.
        """
        if phase.shape != (self.npix, self.npix):
            logger.warning(
                f"Phase shape {phase.shape} doesn't match device "
                f"({self.npix}, {self.npix})"
            )
        self._phase_pattern = phase

    def get_phase(self) -> np.ndarray | None:
        """Get current phase pattern.
        
        Returns:
            Current phase pattern or None.
        """
        return self._phase_pattern
