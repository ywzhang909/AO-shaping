"""Simulated SLM and optical elements.

This module provides simulated optical devices including SLM (Spatial Light Modulator),
Lens, Aperture, and other optical elements that integrate with the simulation framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.base import SimulatedDevice, SimulatedDeviceError, WavefrontProcessor


class SimulatedSLMError(SimulatedDeviceError):
    """Exception for simulated SLM errors."""
    pass


class SimulatedSLM(WavefrontProcessor):
    """Simulated Spatial Light Modulator.
    
    This class provides a simulated SLM that applies phase patterns
    to wavefronts. It inherits from WavefrontProcessor.
    
    Attributes:
        resolution: SLM resolution (width, height).
        phase_range: Maximum phase modulation range in radians.
    
    Example:
        >>> slm = SimulatedSLM(resolution=(1920, 1080))
        >>> with slm:
        ...     phase = np.random.rand(1080, 1920) * 2 * np.pi
        ...     slm.set_phase(phase)
        ...     output = slm.process(input_wave)
    """

    device_type = DeviceType.SLM
    manufacturer = "Simulation"
    model = "Simulated SLM"

    def __init__(
        self,
        device_id: str = "",
        resolution: tuple = (1920, 1080),
        phase_range: float = 2 * np.pi,
        wavelength: float = 1064.0,
        enable_noise: bool = False,
        random_seed: int | None = None,
    ):
        """Initialize simulated SLM.
        
        Args:
            device_id: Unique device identifier.
            resolution: SLM resolution (width, height).
            phase_range: Maximum phase range in radians.
            wavelength: Operating wavelength in nm.
            enable_noise: Whether to add phase noise.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(
            device_id,
            wavelength,
            resolution[1],  # npix = height
            8e-6,  # dpix ~ 8μm for typical SLM
            enable_noise,
            random_seed,
        )

        self._resolution = resolution
        self.phase_range = phase_range

        self._current_phase: np.ndarray | None = None
        self._phase_loaded = False

        self._register_parameters()
        self._register_capabilities()

        logger.debug(
            f"SimulatedSLM initialized: resolution={resolution}, "
            f"phase_range={phase_range:.2f}π"
        )

    def _register_parameters(self) -> None:
        """Register SLM parameters."""
        self.register_parameter(
            "phase_range",
            default_value=self.phase_range,
            min_value=np.pi,
            max_value=4 * np.pi,
            unit="rad",
            description="Maximum phase modulation range",
        )
        self.register_parameter(
            "gamma",
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            unit="",
            description="Gamma correction factor",
        )

    def _register_capabilities(self) -> None:
        """Register SLM capabilities."""
        self.register_capability(
            "set_phase",
            description="Set phase pattern",
            parameters=["phase"],
        )
        self.register_capability(
            "process",
            description="Apply phase to wavefront",
            parameters=["wave"],
            return_type=object,
        )

    # ========== SimulatedDevice Implementation ==========

    def compute(self, *args, **kwargs) -> Any:
        """Apply phase to wavefront."""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    # ========== SLM-Specific Methods ==========

    def set_phase(self, phase: np.ndarray) -> None:
        """Set phase pattern on SLM.
        
        Args:
            phase: 2D phase array in radians.
        """
        if phase.shape != self._resolution[::-1]:  # Note: (height, width)
            raise ValueError(
                f"Phase shape {phase.shape} doesn't match "
                f"SLM resolution {self._resolution}"
            )

        # Normalize to phase range
        gamma = self.get_parameter_value("gamma")
        if gamma != 1.0:
            phase = np.power(phase / self.phase_range, 1.0 / gamma) * self.phase_range

        self._current_phase = np.clip(phase, 0, self.phase_range)
        self._phase_loaded = True

        logger.debug(f"Phase pattern loaded: shape={phase.shape}")

    def get_phase(self) -> np.ndarray | None:
        """Get current phase pattern.
        
        Returns:
            Current phase pattern or None.
        """
        return self._current_phase.copy() if self._current_phase is not None else None

    def clear(self) -> None:
        """Clear phase pattern (set to zero)."""
        self._current_phase = np.zeros(self._resolution[::-1])
        self._phase_loaded = False
        logger.debug("Phase pattern cleared")

    # ========== WavefrontProcessor Implementation ==========

    def process(self, wave: Any) -> Any:
        """Apply SLM phase pattern to wavefront.
        
        Args:
            wave: Input wavefront (compatible with sim.digitaltwin.Wave).
            
        Returns:
            Wavefront with phase applied.
        """
        if not self.is_connected():
            raise RuntimeError("SLM not connected")

        if not self._phase_loaded:
            logger.warning("No phase pattern loaded, returning input wave")
            return wave

        self._set_state(DeviceState.BUSY)

        try:
            # Try to use digitaltwin optics
            try:
                from sim.digitaltwin.optics import SLM as DT_SLM

                # Create SLM and apply
                slm = DT_SLM(self._current_phase, wave.dpix)
                slm.out(wave)

                return wave
            except ImportError:
                # Fallback: apply phase directly
                return self._apply_phase_direct(wave)
        finally:
            self._set_state(DeviceState.READY)

    def _apply_phase_direct(self, wave: Any) -> Any:
        """Apply phase directly to wavefront (fallback).
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Modified wavefront.
        """
        # Resize phase if needed
        phase = self._current_phase
        if hasattr(wave, 'npix') and hasattr(wave, 'dpix'):
            if phase.shape != (wave.npix, wave.npix):
                try:
                    from sim.digitaltwin import utilities as utils
                    phase = utils.matrix_size_trans(
                        phase, self.dpix, wave.npix, wave.dpix
                    )
                except ImportError:
                    # Simple resize
                    from scipy import ndimage
                    factor = wave.npix / phase.shape[0]
                    phase = ndimage.zoom(phase, factor)

        # Apply phase
        if hasattr(wave, 'wavefront'):
            wave.change_wf(phase=phase)

        return wave


class SimulatedLens(SimulatedDevice):
    """Simulated focusing lens.
    
    Example:
        >>> lens = SimulatedLens(focus_length=0.5)
        >>> output = lens.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Simulated Lens"

    def __init__(
        self,
        device_id: str = "",
        focus_length: float = 0.5,
        wavelength: float = 1064.0,
    ):
        """Initialize simulated lens.
        
        Args:
            device_id: Unique device identifier.
            focus_length: Focal length in meters (positive = converging).
            wavelength: Wavelength in nm.
        """
        super().__init__(device_id)

        self.focus_length = focus_length
        self.wavelength = wavelength

    def compute(self, *args, **kwargs) -> Any:
        """Apply lens to wavefront."""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    def process(self, wave: Any) -> Any:
        """Apply lens phase to wavefront.
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Focused wavefront.
        """
        if not self.is_connected():
            raise RuntimeError("Lens not connected")

        # Try digitaltwin
        try:
            from sim.digitaltwin.optics import Lens as DT_Lens

            lens = DT_Lens(self.focus_length)
            lens.out(wave)
            return wave
        except ImportError:
            # Fallback: apply directly
            return self._apply_lens_direct(wave)

    def _apply_lens_direct(self, wave: Any) -> Any:
        """Apply lens phase directly (fallback)."""
        if not hasattr(wave, 'r'):
            return wave

        lamd = self.wavelength * 1e-9
        focus_phase = -np.pi * wave.r ** 2 / lamd / self.focus_length

        if hasattr(wave, 'change_wf'):
            wave.change_wf(phase=focus_phase)

        return wave


class SimulatedAperture(SimulatedDevice):
    """Simulated optical aperture.
    
    Example:
        >>> aperture = SimulatedAperture(radius=0.05)
        >>> output = aperture.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Simulated Aperture"

    def __init__(
        self,
        device_id: str = "",
        radius: float = 0.05,
    ):
        """Initialize simulated aperture.
        
        Args:
            device_id: Unique device identifier.
            radius: Aperture radius in meters (positive = aperture, negative = obstruction).
        """
        super().__init__(device_id)

        self.radius = radius

    def compute(self, *args, **kwargs) -> Any:
        """Apply aperture to wavefront."""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    def process(self, wave: Any) -> Any:
        """Apply aperture to wavefront.
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Masked wavefront.
        """
        if not self.is_connected():
            raise RuntimeError("Aperture not connected")

        # Try digitaltwin
        try:
            from sim.digitaltwin.optics import Aperture as DT_Aperture

            aperture = DT_Aperture(self.radius)
            aperture.out(wave)
            return wave
        except ImportError:
            # Fallback
            return self._apply_aperture_direct(wave)

    def _apply_aperture_direct(self, wave: Any) -> Any:
        """Apply aperture directly (fallback)."""
        if not hasattr(wave, 'r'):
            return wave

        if self.radius > 0:
            mask = (np.sign(self.radius - wave.r) + 1) / 2
        elif self.radius < 0:
            mask = (np.sign(wave.r + self.radius) + 1) / 2
        else:
            mask = 1

        if hasattr(wave, 'wavefront'):
            wave.wavefront = np.real(wave.wavefront) * mask + 1j * np.imag(wave.wavefront) * mask

        return wave
