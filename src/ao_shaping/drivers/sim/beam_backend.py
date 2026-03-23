from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.drivers.sim import beam_simulation as bs


@dataclass(frozen=True)
class BeamSimConfig:
    n_grid: int
    aperture_size: float
    wavelength: float
    cn2: float
    l_max: float
    l_min: float
    propagation_distance: float
    beam_waist: float

    @property
    def pixel_size(self) -> float:
        return self.aperture_size / self.n_grid


def make_beam_config(
    *,
    n_grid: int,
    aperture_size: float,
    wavelength: float,
    cn2: float,
    l_max: float,
    l_min: float,
    propagation_distance: float,
    beam_waist: float | None = None,
) -> BeamSimConfig:
    waist = beam_waist if beam_waist is not None else aperture_size / 3.5
    return BeamSimConfig(
        n_grid=int(n_grid),
        aperture_size=float(aperture_size),
        wavelength=float(wavelength),
        cn2=float(cn2),
        l_max=float(l_max),
        l_min=float(l_min),
        propagation_distance=float(propagation_distance),
        beam_waist=float(waist),
    )


def to_bs_config(cfg: BeamSimConfig) -> bs.Config:
    return bs.Config(
        wavelength=cfg.wavelength,
        size_x=cfg.n_grid,
        size_y=cfg.n_grid,
        pixel_size=cfg.pixel_size,
        w0=cfg.beam_waist,
        z_default=cfg.propagation_distance,
        Cn2=cfg.cn2,
        l_max=cfg.l_max,
        l_min=cfg.l_min,
    )


def grid(cfg: BeamSimConfig) -> tuple[np.ndarray, np.ndarray]:
    beam_cfg = to_bs_config(cfg)
    return beam_cfg.grid()


def gaussian_pupil(cfg: BeamSimConfig, *, aperture_radius: float | None = None) -> np.ndarray:
    beam_cfg = to_bs_config(cfg)
    field = bs.gauss(cfg=beam_cfg).astype(np.complex128)
    x, y = beam_cfg.grid()
    radius = np.sqrt(x**2 + y**2)
    cutoff = aperture_radius if aperture_radius is not None else cfg.aperture_size / 2
    return field * (radius <= cutoff)


def turbulence_phase(
    cfg: BeamSimConfig,
    *,
    cn2: float | None = None,
    l_max: float | None = None,
    l_min: float | None = None,
    propagation_distance: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    beam_cfg = to_bs_config(cfg)
    cn2_value = beam_cfg.Cn2 if cn2 is None else float(cn2)
    if cn2_value <= 0:
        return np.zeros((cfg.n_grid, cfg.n_grid), dtype=float)

    l_max_value = beam_cfg.l_max if l_max is None else float(l_max)
    l_min_value = beam_cfg.l_min if l_min is None else float(l_min)
    distance = beam_cfg.z_default if propagation_distance is None else float(propagation_distance)
    k = 2 * np.pi / beam_cfg.wavelength
    r0 = (0.423 * (k**2) * cn2_value * max(distance, 1e-9)) ** (-3 / 5)

    fx = np.fft.fftshift(np.fft.fftfreq(beam_cfg.size_x, beam_cfg.pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(beam_cfg.size_y, beam_cfg.pixel_size))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    freq = np.sqrt(fx_grid**2 + fy_grid**2)

    fm = 5.92 / (2 * np.pi * max(l_min_value, 1e-12))
    f0 = 1.0 / max(l_max_value, 1e-12)
    psd_phi = (
        0.023
        * r0 ** (-5 / 3)
        * np.exp(-(freq / fm) ** 2)
        / ((freq**2 + f0**2) ** (11 / 6))
    )
    psd_phi[beam_cfg.size_y // 2, beam_cfg.size_x // 2] = 0.0

    delta_fx = 1.0 / (beam_cfg.size_x * beam_cfg.pixel_size)
    delta_fy = 1.0 / (beam_cfg.size_y * beam_cfg.pixel_size)
    if rng is None:
        real = np.random.normal(size=psd_phi.shape)
        imag = np.random.normal(size=psd_phi.shape)
    else:
        real = rng.normal(size=psd_phi.shape)
        imag = rng.normal(size=psd_phi.shape)
    rand_spec = real + 1j * imag
    coeff = rand_spec * np.sqrt(psd_phi) * np.sqrt(delta_fx * delta_fy)
    return np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(coeff)))) * (
        beam_cfg.size_x * beam_cfg.size_y
    )


def apply_lens(field: np.ndarray, cfg: BeamSimConfig, focal_length: float) -> np.ndarray:
    beam_cfg = to_bs_config(cfg)
    return bs.apply_lens(E=field, cfg=beam_cfg, f=focal_length)


def propagate(field: np.ndarray, cfg: BeamSimConfig, distance: float) -> np.ndarray:
    beam_cfg = to_bs_config(cfg)
    return bs.propagation(field, z=distance, cfg=beam_cfg)


def focal_plane(field: np.ndarray, cfg: BeamSimConfig, focal_length: float) -> np.ndarray:
    beam_cfg = to_bs_config(cfg)
    return bs.lens_fft_propagation_to_focal(E=field, f=focal_length, cfg=beam_cfg)
