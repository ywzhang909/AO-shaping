"""Wave generation and propagation devices.

This module provides simulated wavefront generation (laser sources)
and propagation (free-space, lens focus) using validated physics
from sim.digitaltwin.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.base import OpticalDevice, SimulatedDevice, SimulatedDeviceError, WavefrontProcessor


class WaveDeviceError(SimulatedDeviceError):
    pass


class WaveGenerator(OpticalDevice):
    """Simulated wavefront generator.

    Generates plane waves or Gaussian beams using validated physics
    from sim.digitaltwin.base.Wave.

    Example:
        >>> gen = WaveGenerator(npix=256, dpix=0.1e-3, wavelength=1550e-9)
        >>> with gen:
        ...     wave = gen.generate()
        >>> # wave is a sim.digitaltwin.base.Wave object
    """

    device_type = DeviceType.LASER
    manufacturer = "Simulation"
    model = "Wave Generator"

    def __init__(
        self,
        device_id: str = "",
        npix: int = 256,
        dpix: float = 0.1e-3,
        wavelength: float = 1550e-9,
        aperture: float = 0.0,
        beam_type: str = "plane",
        random_seed: Optional[int] = None,
    ):
        super().__init__(device_id, wavelength, npix, dpix, False, random_seed)

        self.npix = npix
        self.dpix = dpix
        self.aperture = aperture
        self.beam_type = beam_type

        self._dt_wave: Any = None
        self._current_wave: Any = None

    def generate(self) -> Any:
        """Generate a wavefront.

        Returns:
            Wave object (sim.digitaltwin.base.Wave).
        """
        if not self.is_connected():
            raise RuntimeError("WaveGenerator not connected")

        self._set_state(DeviceState.BUSY)
        try:
            from sim.digitaltwin import base as dt_base

            wave = dt_base.Wave()
            wave.change_grid(self.npix, self.dpix)
            wave.wavelength = self.wavelength
            wave.refractive = 1.0
            wave.wavefront = np.ones((self.npix, self.npix), dtype=complex)

            if self.beam_type == "gaussian":
                radius = self.aperture / 2 / np.sqrt(2) if self.aperture > 0 else self.npix * self.dpix / 4
                amplitude = np.exp(-(wave.r / radius) ** 2)
                wave.wavefront = amplitude * np.exp(0j)

            if self.aperture > 0:
                mask = (np.sign(self.aperture / 2 - wave.r) + 1) / 2
                wave.wavefront *= mask

            self._current_wave = wave
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def process(self, wave: Any) -> Any:
        return wave


class WavePropagator(SimulatedDevice):
    """Simulated wavefront propagator.

    Propagates wavefronts using angular spectrum method
    (sim.digitaltwin.utilities.wave_angle_spectrum).

    Example:
        >>> prop = WavePropagator(prop_dist=0.5)
        >>> with prop:
        ...     wave_out = prop.propagate(wave_in)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Wave Propagator"

    def __init__(
        self,
        device_id: str = "",
        prop_dist: float = 0.5,
    ):
        super().__init__(device_id)
        self.prop_dist = prop_dist

    def propagate(self, wave: Any) -> Any:
        """Propagate wavefront.

        Args:
            wave: Input wavefront (sim.digitaltwin.base.Wave).

        Returns:
            Propagated wavefront.
        """
        if not self.is_connected():
            raise RuntimeError("WavePropagator not connected")

        self._set_state(DeviceState.BUSY)
        try:
            from sim.digitaltwin import utilities as utils

            utils.wave_angle_spectrum(wave, self.prop_dist)
            return wave
        finally:
            self._set_state(DeviceState.READY)


class LensApplier(WavefrontProcessor):
    """Simulated thin lens.

    Applies lens phase to wavefronts: phase = -pi * r^2 / (lambda * f).

    Example:
        >>> lens = LensApplier(focal_length=0.5)
        >>> with lens:
        ...     wave_out = lens.apply(wave_in)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Thin Lens"

    def __init__(
        self,
        device_id: str = "",
        focal_length: float = 0.5,
        wavelength: float = 1550e-9,
        npix: int = 256,
        dpix: float = 0.1e-3,
    ):
        super().__init__(device_id, wavelength, npix, dpix)
        self.focal_length = focal_length

    def apply(self, wave: Any) -> Any:
        """Apply lens phase to wavefront.

        Args:
            wave: Input wavefront.

        Returns:
            Wavefront with lens phase applied.
        """
        if not self.is_connected():
            raise RuntimeError("LensApplier not connected")

        self._set_state(DeviceState.BUSY)
        try:
            if not hasattr(wave, "r") or not hasattr(wave, "lamd"):
                raise ValueError("Wave must be a sim.digitaltwin.base.Wave object")

            focus_phase = -np.pi * wave.r ** 2 / wave.lamd / self.focal_length
            wave.change_wf(phase=focus_phase)
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def process(self, wave: Any) -> Any:
        return self.apply(wave)


class ApertureApplier(WavefrontProcessor):
    """Simulated circular aperture.

    Applies a circular pupil mask to wavefronts.

    Example:
        >>> ap = ApertureApplier(radius=0.05)
        >>> with ap:
        ...     wave_out = ap.apply(wave_in)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Circular Aperture"

    def __init__(
        self,
        device_id: str = "",
        radius: float = 0.05,
        wavelength: float = 1550e-9,
        npix: int = 256,
        dpix: float = 0.1e-3,
    ):
        super().__init__(device_id, wavelength, npix, dpix)
        self.radius = radius

    def apply(self, wave: Any) -> Any:
        """Apply aperture mask to wavefront.

        Args:
            wave: Input wavefront.

        Returns:
            Masked wavefront.
        """
        if not self.is_connected():
            raise RuntimeError("ApertureApplier not connected")

        self._set_state(DeviceState.BUSY)
        try:
            if not hasattr(wave, "r"):
                raise ValueError("Wave must be a sim.digitaltwin.base.Wave object")

            mask = (np.sign(self.radius - wave.r) + 1) / 2
            real_wf = np.real(wave.wavefront)
            imag_wf = np.imag(wave.wavefront)
            wave.wavefront = real_wf * mask + 1j * imag_wf * mask
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def process(self, wave: Any) -> Any:
        return self.apply(wave)


class WaveMetric:
    """Wavefront metric computation utilities.

    Provides PIB, Strehl, centroid, and radius computation
    using sim.digitaltwin.params.WaveIndex.
    """

    @staticmethod
    def power_bucket(wave: Any, r_bucket: float, center: str = "origin") -> float:
        """Compute Power-In-Bucket.

        Args:
            wave: Wave object with intensity.
            r_bucket: Bucket radius.
            center: 'origin', 'centroid', 'peak', or (x, y) tuple.

        Returns:
            Power within the bucket.
        """
        from sim.digitaltwin import params as dt_params

        return dt_params.WaveIndex.power_bucket(wave.intensity, wave.x, wave.y, center, r_bucket)

    @staticmethod
    def radius(intensity: np.ndarray, x: np.ndarray, y: np.ndarray,
               center: str = "origin", energy: float = 0.865) -> float:
        """Compute radius containing a fraction of total energy.

        Args:
            intensity: Intensity array.
            x, y: Coordinate arrays.
            center: Center type.
            energy: Energy fraction.

        Returns:
            Radius.
        """
        from sim.digitaltwin import params as dt_params

        return dt_params.WaveIndex.radius(intensity, x, y, center, energy)

    @staticmethod
    def centroid(intensity: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Compute intensity centroid.

        Args:
            intensity: Intensity array.
            x, y: Coordinate arrays.

        Returns:
            (cx, cy) centroid position.
        """
        from sim.digitaltwin import params as dt_params

        return dt_params.WaveIndex.centroid(intensity, x, y)


def create_wave(npix: int, dpix: float, wavelength: float) -> Any:
    """Create a wave object.

    Args:
        npix: Number of pixels.
        dpix: Pixel size (m).
        wavelength: Wavelength (m).

    Returns:
        Wave object.
    """
    from sim.digitaltwin import base as dt_base

    wave = dt_base.Wave()
    wave.change_grid(npix, dpix)
    wave.wavelength = wavelength
    wave.refractive = 1.0
    wave.wavefront = np.ones((npix, npix), dtype=complex)
    return wave


def apply_aperture(wave: Any, radius: float) -> None:
    """Apply circular aperture to wavefront.

    Args:
        wave: Wave object.
        radius: Aperture radius (m).
    """
    mask = (np.sign(radius - wave.r) + 1) / 2
    wave.wavefront = wave.wavefront * mask


def apply_focus(wave: Any, focal_length: float) -> None:
    """Apply thin lens focus phase to wavefront.

    Args:
        wave: Wave object.
        focal_length: Focal length (m).
    """
    focus_phase = -np.pi * wave.r ** 2 / wave.lamd / focal_length
    wave.change_wf(phase=focus_phase)


def propagate(wave: Any, distance: float) -> None:
    """Propagate wavefront using angular spectrum.

    Args:
        wave: Wave object.
        distance: Propagation distance (m).
    """
    from sim.digitaltwin import utilities as utils

    utils.wave_angle_spectrum(wave, distance)


def power_bucket(intensity: np.ndarray, x: np.ndarray, y: np.ndarray,
                 center: str, r_bucket: float) -> float:
    """Compute power-in-bucket.

    Args:
        intensity: Intensity array.
        x, y: Coordinate arrays.
        center: Center type.
        r_bucket: Bucket radius.

    Returns:
        Power within bucket.
    """
    from sim.digitaltwin import params as dt_params

    return dt_params.WaveIndex.power_bucket(intensity, x, y, center, r_bucket)


def radius_metric(intensity: np.ndarray, x: np.ndarray, y: np.ndarray,
                  center: str, energy: float) -> float:
    """Compute radius containing fraction of total energy.

    Args:
        intensity: Intensity array.
        x, y: Coordinate arrays.
        center: Center type.
        energy: Energy fraction.

    Returns:
        Radius.
    """
    from sim.digitaltwin import params as dt_params

    return dt_params.WaveIndex.radius(intensity, x, y, center, energy)
