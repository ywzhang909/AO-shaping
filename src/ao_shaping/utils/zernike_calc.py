from __future__ import annotations

import numpy as np
from numba import njit, prange

@njit(cache=True)
def _zernike_radial_numba(n: int, m: int, r: np.ndarray) -> np.ndarray:
    R_out = np.zeros_like(r)
    abs_m = abs(m)
    for k in range((n - abs_m) // 2 + 1):
        coef = 1.0
        for i in range(1, k + 1):
            coef *= -1 * (n - i + 1) / i
        for i in range(1, (n + abs_m) // 2 - k + 1):
            coef /= i
        for i in range(1, (n - abs_m) // 2 - k + 1):
            coef /= i
        R_out += coef * r ** (n - 2 * k)
    return R_out


@njit(cache=True, parallel=True)
def _compute_zernike_basis_numba(
    n_arr: np.ndarray,
    m_arr: np.ndarray,
    R: np.ndarray,
    Theta: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    n_terms = len(n_arr)
    height, width = R.shape
    bases = np.zeros((n_terms, height, width), dtype=np.float64)

    for i in prange(n_terms):
        n, m = n_arr[i], m_arr[i]
        abs_m = abs(m)

        R_rad = np.zeros_like(R)
        for k in range((n - abs_m) // 2 + 1):
            coef = 1.0
            for ki in range(1, k + 1):
                coef *= -1 * (n - ki + 1) / ki
            for ki in range(1, (n + abs_m) // 2 - k + 1):
                coef /= ki
            for ki in range(1, (n - abs_m) // 2 - k + 1):
                coef /= ki
            R_rad += coef * R ** (n - 2 * k)

        Z = R_rad * np.cos(m * Theta) if m >= 0 else R_rad * np.sin(abs_m * Theta)

        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    bases[i, y, x] = Z[y, x]

    return bases


@njit(cache=True)
def _generate_phase_from_bases_numba(
    bases: np.ndarray,
    coeffs: np.ndarray,
    max_val: float,
    bits: int,
) -> np.ndarray:
    n_terms, height, width = bases.shape
    phase_total = np.zeros((height, width), dtype=np.float64)

    for i in range(n_terms):
        amp = coeffs[i]
        if abs(amp) > 1e-10:
            for y in range(height):
                for x in range(width):
                    phase_total[y, x] += bases[i, y, x] * amp * 2 * np.pi

    phase_wrapped = np.mod(phase_total, 2 * np.pi)
    return (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)


class ZernikeGenerator:
    def __init__(
        self,
        resolution: tuple[int, int],
        radius: float | None = None,
    ) -> None:
        height, width = resolution[1], resolution[0]
        if radius is None:
            radius = min(height, width) / 2

        self._height = height
        self._width = width
        self._radius = radius
        self._max_val: float | None = None

        x = (np.arange(width, dtype=np.float64) - width / 2) / radius
        y = (np.arange(height, dtype=np.float64) - height / 2) / radius
        X, Y = np.meshgrid(x, y)
        self._R = np.sqrt(X**2 + Y**2)
        self._Theta = np.arctan2(Y, X)
        self._mask = (self._R <= 1.0).astype(np.float64)

        self._basis_cache: np.ndarray | None = None
        self._n_arr: np.ndarray | None = None
        self._m_arr: np.ndarray | None = None
        self._n_terms: int = 0

    def precompute_bases(self, n_terms: int) -> None:
        from ao_shaping.utils.zernike_calc import noll_to_nm

        n_arr = np.zeros(n_terms, dtype=np.int32)
        m_arr = np.zeros(n_terms, dtype=np.int32)

        for j in range(n_terms):
            n, m = noll_to_nm(j + 1)
            n_arr[j] = n
            m_arr[j] = m

        self._n_arr = n_arr
        self._m_arr = m_arr
        self._n_terms = n_terms
        self._basis_cache = _compute_zernike_basis_numba(
            n_arr, m_arr, self._R, self._Theta, self._mask
        )

    def set_bits(self, bits: int) -> None:
        self._max_val = 2**bits - 1

    def generate_noll(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """使用Noll索引的一维系数数组生成相位（Numba加速）"""
        max_val = self._max_val
        if max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")

        if self._basis_cache is not None and len(coefficients) <= self._n_terms:
            return _generate_phase_from_bases_numba(
                self._basis_cache[:len(coefficients)],
                coefficients,
                max_val,
                10,
            )

        raise ValueError(
            "Call precompute_bases() first with sufficient terms, "
            f"or increase n_terms. Got {len(coefficients)} coeffs, "
            f"cached {self._n_terms} terms."
        )

    def generate_polynomial(
        self,
        coefficients: dict[tuple[int, int], float],
    ) -> np.ndarray:
        max_val = self._max_val
        if max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")

        if not coefficients:
            return np.zeros((self._height, self._width), dtype=np.uint16)

        max_j = max(nm_to_noll(n, m) for (n, m) in coefficients.keys())
        coeffs_array = np.zeros(max_j, dtype=np.float64)

        for (n, m), amp in coefficients.items():
            j = nm_to_noll(n, m) - 1
            coeffs_array[j] = amp

        return self.generate_noll(coeffs_array)

    def generate(
        self,
        n: int,
        m: int,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        from ao_shaping.utils.zernike_calc import nm_to_noll

        max_val = self._max_val
        if max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")

        j = nm_to_noll(n, m) - 1
        coeffs = np.zeros(j + 1, dtype=np.float64)
        coeffs[j] = amplitude
        return self.generate_noll(coeffs)

    @property
    def mask(self) -> np.ndarray:
        return (self._mask > 0).astype(np.uint8)

    @property
    def R(self) -> np.ndarray:
        return self._R.copy()

    @property
    def Theta(self) -> np.ndarray:
        return self._Theta.copy()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def radius(self) -> float:
        return self._radius


def fit_zernike(
    wavefront: np.ndarray,
    n_max: int = 4,
    radius: float | None = None,
) -> dict[tuple[int, int], float]:
    height, width = wavefront.shape
    if radius is None:
        radius = min(height, width) / 2

    x = (np.arange(width, dtype=np.float64) - width / 2) / radius
    y = (np.arange(height, dtype=np.float64) - height / 2) / radius
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = R <= 1.0

    coeffs: dict[tuple[int, int], float] = {}
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 != 0:
                continue
            if m >= 0:
                Z = _zernike_radial_numba(n, m, R) * np.cos(m * Theta)
            else:
                Z = _zernike_radial_numba(n, -m, R) * np.sin(-m * Theta)
            Z = Z * mask

            coefficient = np.mean(wavefront * Z) / (np.mean(Z * Z) + 1e-10)
            coeffs[(n, m)] = float(coefficient)

    return coeffs


def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    return _zernike_radial_numba(n, m, r)


def noll_to_nm(j: int) -> tuple[int, int]:
    if j < 1:
        raise ValueError(f"Noll索引必须>=1，当前: {j}")

    n = 0
    count = 0
    while count < j:
        for m in range(-n, n + 1, 2):
            count += 1
            if count == j:
                return (n, m)
        n += 1
    raise ValueError(f"无法转换为(n,m): j={j}")


def nm_to_noll(n: int, m: int) -> int:
    j = 0
    for n_i in range(n + 1):
        for m_i in range(-n_i, n_i + 1, 2):
            j += 1
            if n_i == n and m_i == m:
                return j
    raise ValueError(f"无法转换为Noll索引: n={n}, m={m}")


def calc_n_zernike_terms(n_max: int) -> int:
    """Calculate number of Zernike terms up to order n_max.

    Args:
        n_max: Maximum Zernike order.

    Returns:
        Total number of Zernike terms including piston.
    """
    count = 0
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            count += 1
    return count


def generate_noll_polynomial(
    coeffs: np.ndarray,
    resolution: tuple[int, int],
    radius: float | None = None,
    bits: int = 10,
) -> np.ndarray:
    gen = ZernikeGenerator(resolution=resolution, radius=radius)
    gen.set_bits(bits)
    gen.precompute_bases(len(coeffs))
    return gen.generate_noll(coeffs)
