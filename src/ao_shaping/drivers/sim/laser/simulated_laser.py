"""Simulated laser device.

This module provides a simulated laser that inherits from Device,
allowing it to integrate with the existing device framework while using
numerical simulation internally.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.base import OpticalDevice, SimulatedDeviceError


class SimulatedLaserError(SimulatedDeviceError):
    """Exception for simulated laser errors."""
    pass


class SimulatedLaser(OpticalDevice):
    """Simulated laser source.
    
    This class provides a simulated laser that generates wavefronts
    based on specified parameters. It inherits from OpticalDevice
    to integrate with the optical simulation framework.
    
    Attributes:
        power: Laser power in watts.
        wavelength: Operating wavelength in nanometers.
        aperture: Beam aperture diameter in meters.
        beam_quality: Beam quality factor M².
    
    Example:
        >>> laser = SimulatedLaser(power=100, wavelength=1064)
        >>> with laser:
        ...     wave = laser.generate()
        ...     print(f"Generated wave with power {wave.power}")
    """
    
    device_type = DeviceType.LASER
    manufacturer = "Simulation"
    model = "Simulated Laser"
    
    def __init__(
        self,
        device_id: str = "",
        power: float = 100.0,
        wavelength: float = 1064.0,
        aperture: float = 0.2,
        beam_quality: float = 1.0,
        enable_noise: bool = True,
        random_seed: Optional[int] = None,
    ):
        """Initialize simulated laser.
        
        Args:
            device_id: Unique device identifier.
            power: Laser power in watts.
            wavelength: Wavelength in nanometers.
            aperture: Beam aperture in meters.
            beam_quality: Beam quality factor M².
            enable_noise: Whether to add noise.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(device_id, wavelength, enable_noise, random_seed)
        
        self.power = power
        self.wavelength = wavelength
        self.aperture = aperture
        self.beam_quality = beam_quality
        
        self._output_enabled = False
        self._current_wave = None
        
        self._register_parameters()
        self._register_capabilities()
        
        logger.debug(
            f"SimulatedLaser initialized: "
            f"power={power}W, wavelength={wavelength}nm, "
            f"aperture={aperture}m"
        )
    
    def _register_parameters(self) -> None:
        """Register laser parameters."""
        self.register_parameter(
            "power",
            default_value=self.power,
            min_value=0.0,
            max_value=1000.0,
            unit="W",
            description="Laser output power",
        )
        self.register_parameter(
            "wavelength",
            default_value=self.wavelength,
            min_value=300.0,
            max_value=2000.0,
            unit="nm",
            description="Laser wavelength",
        )
        self.register_parameter(
            "aperture",
            default_value=self.aperture,
            min_value=0.01,
            max_value=1.0,
            unit="m",
            description="Beam aperture diameter",
        )
        self.register_parameter(
            "beam_quality",
            default_value=self.beam_quality,
            min_value=1.0,
            max_value=10.0,
            unit="",
            description="Beam quality factor M²",
        )
    
    def _register_capabilities(self) -> None:
        """Register laser capabilities."""
        self.register_capability(
            "generate",
            description="Generate laser wavefront",
            return_type=object,
        )
        self.register_capability(
            "set_power",
            description="Set laser power",
            parameters=["power"],
        )
        self.register_capability(
            "set_wavelength",
            description="Set laser wavelength",
            parameters=["wavelength"],
        )
    
    # ========== SimulatedDevice Implementation ==========
    
    def compute(self, *args, **kwargs) -> Any:
        """Generate laser wavefront."""
        return self.generate()
    
    # ========== Laser-Specific Methods ==========
    
    def generate(self, npix: int = 512, dpix: float = 1e-3) -> Any:
        """Generate laser wavefront.
        
        Args:
            npix: Number of pixels.
            dpix: Pixel size in meters.
            
        Returns:
            Wavefront object (compatible with sim.digitaltwin.Wave).
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")
        
        if not self._output_enabled:
            logger.warning("Laser output is disabled")
        
        self._set_state(DeviceState.BUSY)
        
        try:
            # Generate wavefront using digitaltwin if available
            wave = self._create_wavefront(npix, dpix)
            
            if self.beam_quality > 1.0:
                wave = self._apply_beam_quality(wave)
            
            self._current_wave = wave
            self._output_enabled = True
            
            logger.debug(f"Generated wavefront: power={self.power}W")
            return wave
        finally:
            self._set_state(DeviceState.READY)
    
    def _create_wavefront(self, npix: int, dpix: float) -> Any:
        """Create wavefront object.
        
        Args:
            npix: Number of pixels.
            dpix: Pixel size.
            
        Returns:
            Wavefront object.
        """
        # Try to use digitaltwin base classes
        try:
            from sim.digitaltwin.base import Wave as DTWave
            
            wave = DTWave()
            wave.change_grid(npix, dpix)
            wave.wavelength = self.wavelength * 1e-9  # Convert to meters
            wave.refractive = 1.0  # Air
            
            # Generate Gaussian beam
            r = wave.r
            radius = self.aperture / 2 / np.sqrt(2)
            amplitude = np.exp(-(r / radius) ** 2)
            
            # Set wavefront
            wave.wavefront = amplitude * np.exp(0j)
            
            # Scale power
            from sim.digitaltwin import utilities as utils
            intensity = utils.wf2intensity(wave.wavefront, wave.refractive)
            power = intensity.sum() * dpix ** 2
            wave.scale_power(self.power)
            
            return wave
        except ImportError:
            # Fallback: create simple wavefront dict
            logger.warning("Using fallback wavefront (sim.digitaltwin not available)")
            return self._create_simple_wavefront(npix, dpix)
    
    def _create_simple_wavefront(self, npix: int, dpix: float) -> dict:
        """Create simple wavefront dict (fallback).
        
        Args:
            npix: Number of pixels.
            dpix: Pixel size.
            
        Returns:
            Simple wavefront dictionary.
        """
        # Create coordinate grids
        x = np.linspace(-npix * dpix / 2, npix * dpix / 2 - dpix, npix)
        y = np.linspace(-npix * dpix / 2, npix * dpix / 2 - dpix, npix)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx ** 2 + yy ** 2)
        
        # Gaussian beam
        radius = self.aperture / 2 / np.sqrt(2)
        amplitude = np.exp(-(r / radius) ** 2)
        
        wavefront = amplitude * np.exp(0j)
        
        return {
            "wavefront": wavefront,
            "npix": npix,
            "dpix": dpix,
            "x": xx,
            "y": yy,
            "r": r,
            "wavelength": self.wavelength * 1e-9,
            "refractive": 1.0,
            "power": self.power,
        }
    
    def _apply_beam_quality(self, wave: Any) -> Any:
        """Apply beam quality degradation.
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Wavefront with beam quality applied.
        """
        # Simplified: add phase distortion based on M²
        try:
            from sim.digitaltwin import screens as dt_screens
            from sim.digitaltwin import base as dt_base
            
            # Create fake environment for turbulent screen
            env = dt_base.Environment()
            env.Cn2 = 1e-15  # Weak turbulence
            env.L0 = 1.0
            env.l0 = 0.01
            
            # Apply slight distortion
            screen = dt_screens.TurbulentScreen(0.1, env, harmonic=0)
            screen.out(wave)
            
            return wave
        except ImportError:
            return wave
    
    def set_power(self, power: float) -> None:
        """Set laser power.
        
        Args:
            power: Power in watts.
        """
        if not (0 <= power <= 1000):
            raise ValueError(f"Power {power} out of range [0, 1000]")
        
        self.power = power
        self.set_parameter_value("power", power)
        logger.info(f"Laser power set to {power} W")
    
    def set_wavelength(self, wavelength: float) -> None:
        """Set laser wavelength.
        
        Args:
            wavelength: Wavelength in nanometers.
        """
        if not (300 <= wavelength <= 2000):
            raise ValueError(f"Wavelength {wavelength} out of range [300, 2000]")
        
        self.wavelength = wavelength
        self.set_parameter_value("wavelength", wavelength)
        logger.info(f"Laser wavelength set to {wavelength} nm")
    
    def enable_output(self, enabled: bool) -> None:
        """Enable or disable laser output.
        
        Args:
            enabled: True to enable, False to disable.
        """
        self._output_enabled = enabled
        logger.info(f"Laser output {'enabled' if enabled else 'disabled'}")
    
    def is_output_enabled(self) -> bool:
        """Check if output is enabled."""
        return self._output_enabled
    
    # ========== OpticalDevice Implementation ==========
    
    def process(self, wave: Any) -> Any:
        """Process wavefront (pass-through for laser).
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Same wavefront.
        """
        return wave
