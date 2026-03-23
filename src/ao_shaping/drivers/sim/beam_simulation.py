from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import factorial


@dataclass
class Config:
    wavelength: float = 810e-9
    size_x: int = 300
    size_y: int = 300
    pixel_size: float = 8e-6
    w0: float = 0.45e-3
    z_default: float = 1e-2
    Cn2: float = 3e-13
    l_max: float = 1e-1
    l_min: float = 1e-3

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.arange(-self.size_x / 2, self.size_x / 2) * self.pixel_size
        y = np.arange(-self.size_y / 2, self.size_y / 2) * self.pixel_size
        return np.meshgrid(x, y)


def gauss(*, cfg: Config) -> np.ndarray:
    x, y = cfg.grid()
    radius = np.sqrt(x**2 + y**2)
    return np.exp(-(radius**2) / cfg.w0**2)


def _ft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def _ift(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def propagation(field: np.ndarray, *, z: float, cfg: Config) -> np.ndarray:
    size_y, size_x = field.shape
    fx = np.fft.fftshift(np.fft.fftfreq(size_x, cfg.pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(size_y, cfg.pixel_size))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    arg = (2 * np.pi) ** 2 * ((1.0 / cfg.wavelength) ** 2 - fx_grid**2 - fy_grid**2)
    kz = np.sqrt(np.abs(arg))
    transfer = np.exp(1j * z * kz)
    return _ift(_ft(field) * transfer)


def lens_phase(*, f: float, cfg: Config) -> np.ndarray:
    x, y = cfg.grid()
    r_sq = x**2 + y**2
    k = 2 * np.pi / cfg.wavelength
    return -k * r_sq / (2 * f)


def apply_lens(*, E: np.ndarray, f: float, cfg: Config) -> np.ndarray:
    return E * np.exp(1j * lens_phase(f=f, cfg=cfg))


def lens_fft_propagation_to_focal(*, E: np.ndarray, f: float, cfg: Config, z_prop: float = 0.0) -> np.ndarray:
    return propagation(apply_lens(E=E, f=f, cfg=cfg), z=f + z_prop, cfg=cfg)


def normalize_rho(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radius = np.sqrt(x**2 + y**2)
    r_max = np.max(radius)
    return radius / r_max if r_max > 0 else radius


def zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if m >= 0:
        angular = np.cos(m * theta)
    else:
        m = abs(m)
        angular = np.sin(m * theta)

    radial = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + m) // 2 - k)
                * factorial((n - m) // 2 - k)
            )
        )
        radial += coeff * rho ** (n - 2 * k)
    return radial * angular


def noll_to_nm(noll_index: int) -> tuple[int, int]:
    n = 0
    m = 0
    noll = 1
    while noll < noll_index:
        m += 1
        if m > n:
            n += 1
            m = -n
        else:
            m = -m
        noll += 1
    return n, m


def generate_zernike_map(noll_index: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, m = noll_to_nm(noll_index)
    rho = normalize_rho(x, y)
    theta = np.arctan2(y, x)
    return zernike_polynomial(n, m, rho, theta)


__all__ = [
    "Config",
    "apply_lens",
    "gauss",
    "generate_zernike_map",
    "lens_fft_propagation_to_focal",
    "lens_phase",
    "propagation",
]
