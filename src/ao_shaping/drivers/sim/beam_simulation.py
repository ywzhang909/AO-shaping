from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def zernike_polynomial(
    n: int, m: int, rho: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """单个 Zernike 多项式 ``Z(rho, theta)`` (aotools/Noll 约定)。

    供 ``optimizer/rl/envs.py`` 的 ``_sample_disturbance`` 在任意分辨率网格上求值。
    径向部分直接复用全仓库统一的 :func:`ao_shaping.utils.wavefront.zernike_calc.
    zernike_radial` (aotools ``RZern``, 与 ``fit_zernike`` / ``ZernikeGenerator``
    同一套归一化), 角向部分按标准约定:

        R_n^|m|(rho) * { cos(|m| theta)  if m >= 0
                        sin(|m| theta)  if m < 0 }

    延迟 import 径向函数, 避免 ``drivers/sim`` 在包初始化早期触发
    ``utils/wavefront`` 的 import 链。

    ``rho`` 应为归一化半径 (单位圆内 0..1); 圆外取值为径向多项式的自然延拓,
    调用方 (``_sample_disturbance``) 用 ``mask`` 把圆外置零, 因此此处不做截断。
    """
    from ao_shaping.utils.wavefront.zernike_calc import zernike_radial

    am = abs(int(m))
    r = np.asarray(rho, dtype=np.float64)
    radial = zernike_radial(int(n), am, r)
    th = np.asarray(theta, dtype=np.float64)
    if m >= 0:
        return radial * np.cos(am * th)
    return radial * np.sin(am * th)


__all__ = [
    "Config",
    "apply_lens",
    "gauss",
    "lens_fft_propagation_to_focal",
    "lens_phase",
    "normalize_rho",
    "propagation",
    "zernike_polynomial",
]
