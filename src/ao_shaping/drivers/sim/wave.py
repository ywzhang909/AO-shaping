from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ao_shaping.drivers.sim import beam_simulation as bs

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.beam_backend import (
    gaussian_pupil,
    make_beam_config,
    propagate as beam_propagate,
    to_bs_config,
)
from ao_shaping.drivers.sim.base import OpticalDevice, SimulatedDevice, SimulatedDeviceError, WavefrontProcessor


class WaveDeviceError(SimulatedDeviceError):
    """Wave simulation related errors."""


@dataclass
class SimWave:
    """Lightweight wave container used when sim.digitaltwin is unavailable."""

    wavefront: np.ndarray
    wavelength: float
    dpix: float
    refractive: float = 1.0

    @property
    def npix(self) -> int:
        return int(self.wavefront.shape[0])

    @property
    def x(self) -> np.ndarray:
        coord = (np.arange(self.npix) - self.npix // 2) * self.dpix
        return np.tile(coord, (self.npix, 1))

    @property
    def y(self) -> np.ndarray:
        coord = (np.arange(self.npix) - self.npix // 2) * self.dpix
        return np.tile(coord[:, None], (1, self.npix))

    @property
    def r(self) -> np.ndarray:
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.wavefront) ** 2

    @property
    def lamd(self) -> float:
        return self.wavelength

    def change_wf(self, phase: np.ndarray | None = None, amplitude: np.ndarray | None = None) -> None:
        """Apply phase and/or amplitude to the wavefront."""
        if amplitude is not None:
            self.wavefront = self.wavefront * amplitude
        if phase is not None:
            self.wavefront = self.wavefront * np.exp(1j * phase)


class WaveGenerator(OpticalDevice):
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
        random_seed: int | None = None,
    ):
        super().__init__(device_id=device_id, wavelength=wavelength, enable_noise=False, random_seed=random_seed)
        self.npix = npix
        self.dpix = dpix
        self.aperture = aperture
        self.beam_type = beam_type

    def generate(self) -> Any:
        if not self.is_connected():
            raise RuntimeError("WaveGenerator not connected")

        self._set_state(DeviceState.BUSY)
        try:
            if self.beam_type == "gaussian":
                beam_cfg = make_beam_config(
                    n_grid=self.npix,
                    aperture_size=self.npix * self.dpix,
                    wavelength=self.wavelength,
                    cn2=0.0,
                    l_max=1.0,
                    l_min=1e-3,
                    propagation_distance=0.0,
                    beam_waist=self.aperture / 3.5 if self.aperture > 0 else self.npix * self.dpix / 4,
                )
                amplitude = np.abs(
                    gaussian_pupil(
                        beam_cfg,
                        aperture_radius=self.aperture / 2 if self.aperture > 0 else None,
                    )
                )
            else:
                amplitude = np.ones((self.npix, self.npix), dtype=float)

            if self.aperture > 0:
                mask = (_wave_grid(self.npix, self.dpix)[2] <= self.aperture / 2).astype(float)
                amplitude *= mask

            return SimWave(
                wavefront=amplitude.astype(np.complex128),
                wavelength=self.wavelength,
                dpix=self.dpix,
            )
        finally:
            self._set_state(DeviceState.READY)

    def process(self, wave: Any) -> Any:
        return wave

    def compute(self, *args, **kwargs) -> Any:
        return self.generate()


class WavePropagator(SimulatedDevice):
    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Wave Propagator"

    def __init__(self, device_id: str = "", prop_dist: float = 0.5):
        super().__init__(device_id)
        self.prop_dist = prop_dist

    def propagate(self, wave: Any) -> Any:
        if not self.is_connected():
            raise RuntimeError("WavePropagator not connected")

        self._set_state(DeviceState.BUSY)
        try:
            propagate(wave, self.prop_dist)
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def compute(self, *args, **kwargs) -> Any:
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.propagate(args[0])


class LensApplier(WavefrontProcessor):
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
        if not self.is_connected():
            raise RuntimeError("LensApplier not connected")
        self._set_state(DeviceState.BUSY)
        try:
            apply_focus(wave, self.focal_length)
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def process(self, wave: Any) -> Any:
        return self.apply(wave)

    def compute(self, *args, **kwargs) -> Any:
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.apply(args[0])


class ApertureApplier(WavefrontProcessor):
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
        if not self.is_connected():
            raise RuntimeError("ApertureApplier not connected")
        self._set_state(DeviceState.BUSY)
        try:
            apply_aperture(wave, self.radius)
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def process(self, wave: Any) -> Any:
        return self.apply(wave)

    def compute(self, *args, **kwargs) -> Any:
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.apply(args[0])


class WaveMetric:
    @staticmethod
    def power_bucket(wave: Any, r_bucket: float, center: str = "origin") -> float:
        return power_bucket(wave.intensity, wave.x, wave.y, center, r_bucket)

    @staticmethod
    def radius(
        intensity: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        center: str = "origin",
        energy: float = 0.865,
    ) -> float:
        return radius_metric(intensity, x, y, center, energy)

    @staticmethod
    def centroid(intensity: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        total = float(np.sum(intensity))
        if total <= 0:
            return 0.0, 0.0
        return float(np.sum(intensity * x) / total), float(np.sum(intensity * y) / total)


def _wave_grid(npix: int, dpix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord = (np.arange(npix) - npix // 2) * dpix
    xx, yy = np.meshgrid(coord, coord)
    return xx, yy, np.sqrt(xx**2 + yy**2)


def _as_sim_wave(wave: Any) -> SimWave:
    if isinstance(wave, SimWave):
        return wave
    if hasattr(wave, "wavefront") and hasattr(wave, "dpix"):
        wavelength = float(getattr(wave, "wavelength", getattr(wave, "lamd", 1550e-9)))
        return SimWave(np.asarray(wave.wavefront), wavelength=wavelength, dpix=float(wave.dpix))
    raise WaveDeviceError("Unsupported wave object type")


def create_wave(npix: int, dpix: float, wavelength: float) -> SimWave:
    return SimWave(np.ones((npix, npix), dtype=np.complex128), wavelength=float(wavelength), dpix=float(dpix))


def apply_aperture(wave: Any, radius: float) -> None:
    sim_wave = _as_sim_wave(wave)
    mask = (sim_wave.r <= radius).astype(float)
    sim_wave.wavefront *= mask


def apply_focus(wave: Any, focal_length: float) -> None:
    sim_wave = _as_sim_wave(wave)
    beam_cfg = make_beam_config(
        n_grid=sim_wave.npix,
        aperture_size=sim_wave.npix * sim_wave.dpix,
        wavelength=sim_wave.wavelength,
        cn2=0.0,
        l_max=1.0,
        l_min=1e-3,
        propagation_distance=focal_length,
    )
    beam_bs_cfg = to_bs_config(beam_cfg)
    focus_phase = bs.lens_phase(f=focal_length, cfg=beam_bs_cfg)
    sim_wave.change_wf(phase=focus_phase)


def propagate(wave: Any, distance: float) -> None:
    """Angular spectrum propagation using internal beam simulation backend."""
    sim_wave = _as_sim_wave(wave)
    beam_cfg = make_beam_config(
        n_grid=sim_wave.npix,
        aperture_size=sim_wave.npix * sim_wave.dpix,
        wavelength=sim_wave.wavelength,
        cn2=0.0,
        l_max=1.0,
        l_min=1e-3,
        propagation_distance=distance,
    )
    sim_wave.wavefront = beam_propagate(sim_wave.wavefront, beam_cfg, distance)


def power_bucket(
    intensity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center: str,
    r_bucket: float,
) -> float:
    if center == "origin":
        cx, cy = 0.0, 0.0
    elif center == "peak":
        idx = np.unravel_index(np.argmax(intensity), intensity.shape)
        cx, cy = float(x[idx]), float(y[idx])
    elif center == "centroid":
        total = np.sum(intensity)
        cx = float(np.sum(intensity * x) / total)
        cy = float(np.sum(intensity * y) / total)
    else:
        raise ValueError(f"Unsupported center mode: {center}")

    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(np.sum(intensity[rr <= r_bucket]))


def radius_metric(
    intensity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center: str,
    energy: float,
) -> float:
    total = float(np.sum(intensity))
    if total <= 0:
        return 0.0

    if center == "origin":
        cx, cy = 0.0, 0.0
    elif center == "centroid":
        cx, cy = WaveMetric.centroid(intensity, x, y)
    else:
        raise ValueError(f"Unsupported center mode: {center}")

    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    sort_idx = np.argsort(rr.ravel())
    cum_energy = np.cumsum(intensity.ravel()[sort_idx])
    threshold = energy * total
    pos = np.searchsorted(cum_energy, threshold)
    return float(rr.ravel()[sort_idx[min(pos, len(sort_idx) - 1)]])
