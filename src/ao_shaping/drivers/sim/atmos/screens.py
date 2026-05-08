"""Atmospheric simulation - Phase screens and propagation.

This module provides simulated atmospheric turbulence and thermal blooming effects
using phase screens, which can be used as device-compatible wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.beam_backend import make_beam_config, turbulence_phase
from ao_shaping.drivers.sim.base import SimulatedDevice, WavefrontProcessor


class SimulatedTurbulentScreen(WavefrontProcessor):
    """Simulated turbulent phase screen.
    
    This class provides a simulated turbulent phase screen that applies
    atmospheric turbulence effects to wavefronts.
    
    Example:
        >>> screen = SimulatedTurbulentScreen(Cn2=1e-15, L0=1.0, l0=0.01)
        >>> with screen:
        ...     output = screen.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Turbulent Phase Screen"

    def __init__(
        self,
        device_id: str = "",
        dist: float = 1.0,
        Cn2: float = 1e-15,
        L0: float = 1.0,
        l0: float = 0.01,
        harmonic: int = 1,
    ):
        """Initialize turbulent phase screen.
        
        Args:
            device_id: Unique device identifier.
            dist: Propagation distance in meters.
            Cn2: Refractive index structure constant.
            L0: Outer scale in meters.
            l0: Inner scale in meters.
            harmonic: Sub-harmonic order.
        """
        super().__init__(device_id, wavelength=1064.0, npix=512, dpix=1e-3)

        self.dist = dist
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0
        self.harmonic = harmonic

        self._screen = None
        self._opd = None

        logger.debug(
            f"TurbulentScreen initialized: Cn2={Cn2}, "
            f"L0={L0}m, l0={l0}m"
        )

    def _register_parameters(self) -> None:
        """Register parameters."""
        self.register_parameter(
            "Cn2",
            default_value=self.Cn2,
            min_value=1e-20,
            max_value=1e-12,
            unit="m^{-2/3}",
            description="Refractive index structure constant",
        )
        self.register_parameter(
            "L0",
            default_value=self.L0,
            min_value=0.1,
            max_value=100.0,
            unit="m",
            description="Outer scale",
        )
        self.register_parameter(
            "l0",
            default_value=self.l0,
            min_value=1e-4,
            max_value=0.1,
            unit="m",
            description="Inner scale",
        )

    # ========== SimulatedDevice Implementation ==========

    def compute(self, *args, **kwargs) -> Any:
        """Apply turbulent screen to wavefront."""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    # ========== WavefrontProcessor Implementation ==========

    def process(self, wave: Any) -> Any:
        """Apply turbulent phase screen to wavefront.
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Wavefront with turbulence applied.
        """
        if not self.is_connected():
            raise RuntimeError("Turbulent screen not connected")

        self._set_state(DeviceState.BUSY)
        try:
            from sim.digitaltwin import screens as dt_screens
            from sim.digitaltwin import base as dt_base

            env = dt_base.Environment()
            env.Cn2 = self.Cn2
            env.L0 = self.L0
            env.l0 = self.l0

            screen = dt_screens.TurbulentScreen(self.dist, env, self.harmonic)
            screen.out(wave)

            self._opd = screen.opd
            return wave
        except Exception as exc:
            logger.warning(f"digitaltwin turbulence path unavailable ({exc}), using fallback")
            return self._apply_turbulence_fallback(wave)
        finally:
            self._set_state(DeviceState.READY)

    def _apply_turbulence_fallback(self, wave: Any) -> Any:
        """Apply turbulence directly (fallback)."""
        npix = getattr(wave, "npix", 512)
        dpix = getattr(wave, "dpix", 1e-3)
        wavelength = getattr(wave, "wavelength", getattr(wave, "lamd", 1064e-9))

        phase = self._generate_kolmogorov_phase(
            npix=npix,
            dpix=dpix,
            wavelength=float(wavelength),
            cn2=self.Cn2,
            l0=self.l0,
            l_max=self.L0,
            distance=self.dist,
        )
        self._opd = phase

        if hasattr(wave, 'change_wf'):
            wave.change_wf(phase=phase)

        return wave

    def _generate_kolmogorov_phase(
        self,
        npix: int,
        dpix: float,
        wavelength: float,
        cn2: float,
        l0: float,
        l_max: float,
        distance: float,
    ) -> np.ndarray:
        """Generate Von Kármán/Kolmogorov turbulence phase screen.
        
        Args:
            npix: Number of pixels.
            dpix: Pixel size.
            
        Returns:
            Phase screen array.
        """
        if cn2 <= 0 or distance <= 0:
            return np.zeros((npix, npix), dtype=float)
        beam_cfg = make_beam_config(
            n_grid=npix,
            aperture_size=npix * dpix,
            wavelength=wavelength,
            cn2=cn2,
            l_max=l_max,
            l_min=l0,
            propagation_distance=distance,
        )
        return turbulence_phase(
            beam_cfg,
            cn2=cn2,
            l_max=l_max,
            l_min=l0,
            propagation_distance=distance,
            rng=self._rng,
        )

    def get_opd(self) -> np.ndarray | None:
        """Get current OPD (Optical Path Difference).
        
        Returns:
            OPD array or None.
        """
        return self._opd.copy() if self._opd is not None else None


class SimulatedThermalScreen(WavefrontProcessor):
    """Simulated thermal blooming phase screen.
    
    Example:
        >>> screen = SimulatedThermalScreen(absorb=1e-5, wind=2.0)
        >>> with screen:
        ...     output = screen.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Thermal Phase Screen"

    def __init__(
        self,
        device_id: str = "",
        dist: float = 1.0,
        absorb: float = 1e-5,
        wind_x: float = 2.0,
        wind_y: float = 0.0,
        solve_mode: str = "FFT_non_Isobaric",
    ):
        """Initialize thermal phase screen.
        
        Args:
            device_id: Unique device identifier.
            dist: Propagation distance in meters.
            absorb: Absorption coefficient.
            wind_x: Wind velocity in x direction (m/s).
            wind_y: Wind velocity in y direction (m/s).
            solve_mode: Solution mode ('Green', 'FFT_Isobaric', 'FFT_non_Isobaric').
        """
        super().__init__(device_id, wavelength=1064.0, npix=512, dpix=1e-3)

        self.dist = dist
        self.absorb = absorb
        self.wind_x = wind_x
        self.wind_y = wind_y
        self.solve_mode = solve_mode

        self._opd = None

        logger.debug(
            f"ThermalScreen initialized: absorb={absorb}, "
            f"wind=({wind_x}, {wind_y}) m/s"
        )

    # ========== SimulatedDevice Implementation ==========

    def compute(self, *args, **kwargs) -> Any:
        """Apply thermal screen to wavefront."""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    # ========== WavefrontProcessor Implementation ==========

    def process(self, wave: Any) -> Any:
        """Apply thermal blooming phase screen to wavefront.
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Wavefront with thermal effects applied.
        """
        if not self.is_connected():
            raise RuntimeError("Thermal screen not connected")

        self._set_state(DeviceState.BUSY)

        try:
            # Try digitaltwin
            try:
                from sim.digitaltwin import screens as dt_screens
                from sim.digitaltwin import base as dt_base

                # Create environment
                env = dt_base.Environment()
                env.absorb = self.absorb
                env.wind_x = self.wind_x
                env.wind_y = self.wind_y
                env.density = 1.177  # Standard air
                env.Cp = 1005
                env.Cv = 718
                env.temperature = 288
                env.Cs2 = 331.3 ** 2
                env.gravity = 9.81

                # Create and apply screen
                screen = dt_screens.ThermalScreen(self.dist, env, self.solve_mode)
                screen.out(wave)

                self._opd = screen.opd
                return wave
            except ImportError:
                logger.warning("sim.digitaltwin not available, using fallback")
                return wave  # Fallback: no-op
        finally:
            self._set_state(DeviceState.READY)

    def get_opd(self) -> np.ndarray | None:
        """Get current OPD."""
        return self._opd.copy() if self._opd is not None else None


class SimulatedATP(SimulatedDevice):
    """Simulated Atmospheric Propagation (ATP).
    
    This class simulates laser propagation through the atmosphere
    including turbulence and thermal blooming effects.
    
    Example:
        >>> atp = SimulatedATP(prop_dist=3000, layers=10, Cn2=1e-15)
        >>> with atp:
        ...     output = atp.propagate(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Atmospheric Propagation"

    def __init__(
        self,
        device_id: str = "",
        prop_dist: float = 3000.0,
        layers: int = 10,
        Cn2: float = 1e-15,
        Thermal: bool = False,
        Turbulent: bool = True,
    ):
        """Initialize atmospheric propagation.
        
        Args:
            device_id: Unique device identifier.
            prop_dist: Propagation distance in meters.
            layers: Number of phase screen layers.
            Cn2: Refractive index structure constant.
            Thermal: Enable thermal blooming.
            Turbulent: Enable turbulence.
        """
        super().__init__(device_id)

        self.prop_dist = prop_dist
        self.layers = layers
        self.Cn2 = Cn2
        self.Thermal = Thermal
        self.Turbulent = Turbulent

        self._atp = None

        logger.debug(
            f"ATP initialized: distance={prop_dist}m, layers={layers}, "
            f"Cn2={Cn2}, Turbulent={Turbulent}, Thermal={Thermal}"
        )

    # ========== SimulatedDevice Implementation ==========

    def compute(self, *args, **kwargs) -> Any:
        """Propagate wave through atmosphere."""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.propagate(args[0])

    def propagate(self, wave: Any) -> Any:
        """Propagate wave through atmosphere.
        
        Args:
            wave: Input wavefront.
            
        Returns:
            Propagated wavefront.
        """
        if not self.is_connected():
            raise RuntimeError("ATP not connected")

        self._set_state(DeviceState.BUSY)

        try:
            # Try digitaltwin
            try:
                from sim.digitaltwin import atp as dt_atp
                from sim.digitaltwin import base as dt_base

                # Create initial environment
                env = dt_base.Environment()
                env.absorb = 5e-6
                env.scatter = 5e-5
                env.wind_x = 2.0
                env.wind_y = 0.0
                env.density = 1.177
                env.Cp = 1005
                env.Cv = 718
                env.temperature = 288
                env.nT = -8.6e-7
                env.atm = 1.0
                env.Cs2 = 331.3 ** 2
                env.gravity = 9.81
                env.Cn2 = self.Cn2
                env.L0 = 1.0
                env.l0 = 0.01

                # Create ATP
                atp = dt_atp.ATP(
                    env_init=env,
                    prop_dist=self.prop_dist,
                    layers=self.layers,
                    Turbulent=self.Turbulent,
                    Thermal=self.Thermal,
                )

                # Propagate
                atp.out(wave)

                self._atp = atp
                return wave
            except ImportError:
                logger.warning("sim.digitaltwin not available, ATP not executed")
                return wave
        finally:
            self._set_state(DeviceState.READY)

    def set_env_params(self, **kwargs) -> None:
        """Set environmental parameters.
        
        Args:
            **kwargs: Environmental parameters.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"ATP parameter set: {key}={value}")
