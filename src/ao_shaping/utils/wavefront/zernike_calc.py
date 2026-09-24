from __future__ import annotations

from typing import TypeVar

import numpy as np
from zernike import RZern

# Zernike polynomial naming (Noll's scheme)
ZERNIKE_NAMES: dict[tuple[int, int], str] = {
    # n=0 (radial order 0): 1 mode
    (0, 0): "Piston / 活塞",

    # n=1 (radial order 1): 2 modes
    (1, -1): "Tip / X倾斜",
    (1, 1): "Tilt / Y倾斜",

    # n=2 (radial order 2): 3 modes
    (2, 0): "Defocus / 离焦",
    (2, -2): "Astigmatism 45° / 45°像散",
    (2, 2): "Astigmatism 0° / 0°像散",

    # n=3 (radial order 3): 4 modes
    (3, -1): "Coma Y / Y彗差",
    (3, 1): "Coma X / X彗差",
    (3, -3): "Trefoil Y / Y三叶像差",
    (3, 3): "Trefoil X / X三叶像差",

    # n=4 (radial order 4): 5 modes
    (4, 0): "Spherical / 球差",
    (4, -2): "Secondary Astig 45° / 二级45°像散",
    (4, 2): "Secondary Astig 0° / 二级0°像散",
    (4, -4): "Tetrafoil Y / Y四叶像差",
    (4, 4): "Tetrafoil X / X四叶像差",

    # n=5 (radial order 5): 6 modes
    (5, -1): "Secondary Coma Y / 二级Y彗差",
    (5, 1): "Secondary Coma X / 二级X彗差",
    (5, -3): "Secondary Trefoil Y / 二级Y三叶像差",
    (5, 3): "Secondary Trefoil X / 二级X三叶像差",
    (5, -5): "Pentafoil Y / Y五叶像差",
    (5, 5): "Pentafoil X / X五叶像差",

    # n=6 (radial order 6): 7 modes
    (6, 0): "Secondary Spherical / 二级球差",
    (6, -2): "Tertiary Astig 45° / 三级45°像散",
    (6, 2): "Tertiary Astig 0° / 三级0°像散",
    (6, -4): "Secondary Tetrafoil Y / 二级Y四叶像差",
    (6, 4): "Secondary Tetrafoil X / 二级X四叶像差",
    (6, -6): "Hexafoil Y / Y六叶像差",
    (6, 6): "Hexafoil X / X六叶像差",

    # n=7 (radial order 7): 8 modes
    (7, -1): "Tertiary Coma Y / 三级Y彗差",
    (7, 1): "Tertiary Coma X / 三级X彗差",
    (7, -3): "Tertiary Trefoil Y / 三级Y三叶像差",
    (7, 3): "Tertiary Trefoil X / 三级X三叶像差",
    (7, -5): "Secondary Pentafoil Y / 二级Y五叶像差",
    (7, 5): "Secondary Pentafoil X / 二级X五叶像差",
    (7, -7): "Heptafoil Y / Y七叶像差",
    (7, 7): "Heptafoil X / X七叶像差",
}

_ZernikeT = TypeVar("_ZernikeT", bound="ZernikeGenerator")


def get_zernike_name(n: int, m: int) -> str:
    """Get Zernike polynomial name from (n, m) indices.

    Args:
        n: Radial order.
        m: Azimuthal order.

    Returns:
        Zernike name string, or empty string if not in lookup table.
    """
    return ZERNIKE_NAMES.get((n, m), f"n={n},m={m}")

def calc_n_zernike_terms(n_max: int) -> int:
    """Calculate the number of Zernike terms up to order n_max.

    Args:
        n_max: Maximum Zernike radial order.

    Returns:
        Number of Zernike terms (including piston).
    """
    return (n_max + 1) * (n_max + 2) // 2


def zernike_modes(n_max: int | str) -> list[tuple[int, int]]:
    """Valid (n, m) index pairs up to radial order ``n_max``.

    Parity rule: ``n - |m|`` must be even (m steps by 2 for each n).
    This is the inverse of :func:`calc_n_zernike_terms` — given the radial
    order it enumerates every (n, m) pair in the standard Noll ordering
    (n ascending, m ascending within each n).

    Pure Zernike math — no GUI / no session_state dependency, so it belongs
    in :mod:`ao_shaping.utils.zernike_calc` alongside
    :func:`get_zernike_name` and :func:`calc_n_zernike_terms`.

    Args:
        n_max: Maximum Zernike radial order. Coerced to ``int`` so a widget
            string value (e.g. ``"5"``) is accepted transparently.

    Returns:
        List of ``(n, m)`` tuples. Length equals
        :func:`calc_n_zernike_terms` ``= (n_max+1)(n_max+2)//2``.
    """
    return [
        (n, m)
        for n in range(int(n_max) + 1)
        for m in range(-n, n + 1)
        if (n - abs(m)) % 2 == 0
    ]


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m) Zernike indices (aotools convention).

    Standalone function using the aotools RZern library for conversion.
    Follows the standard Noll indexing convention (Noll 1976).
    Supports any Noll index (not limited to 1-15).

    NOTE: This is the canonical implementation used across the codebase
    (including the wf/ optimizers); it supports any Noll index (>= 1).

    Args:
        j: Noll index (1-based).

    Returns:
        Tuple of (n, m) radial and azimuthal orders.

    Raises:
        ValueError: If j < 1.
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    # Create a temporary RZern with enough orders to cover index j
    # noll2nm needs at least ceil((sqrt(8*j-7)-1)/2) radial orders
    import math
    n_needed = max(1, math.ceil((math.sqrt(8 * j - 7) - 1) / 2))
    cart = RZern(n_needed)
    result = cart.noll2nm(j)
    if isinstance(result, tuple):
        return (int(result[0]), int(result[1]))
    return (int(result[0][0]), int(result[1][0]))


def fit_zernike(phase: np.ndarray, n_max: int = 10) -> np.ndarray:
    """Fit Zernike coefficients to a phase map.

    Args:
        phase: 2D phase array.
        n_max: Maximum Zernike order.

    Returns:
        1D array of Zernike coefficients (Noll order).
    """
    height, width = phase.shape
    cart = RZern(n_max)
    ddx = np.linspace(-1.0, 1.0, width)
    ddy = np.linspace(-1.0, 1.0, height)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    return cart.fit_cart_grid(phase)[0]


def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Generate radial Zernike polynomial R_n^m(rho).

    Args:
        n: Radial order.
        m: Azimuthal order.
        rho: Radial coordinate array (normalized to [0,1]).

    Returns:
        Radial polynomial values.
    """
    # Create sufficient order for the given (n,m)
    cart = RZern(n)
    # Use k-th Zernike radial polynomial (0-indexed internally)
    # The package uses Noll index -> radial polynomial internally
    k = cart.nm2noll(n, m) - 1  # Convert to 0-based Noll index
    return cart.radial(k, rho)


def generate_noll_polynomial(
    n: int,
    m: int,
    resolution: tuple[int, int],
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a single Zernike polynomial pattern.

    Args:
        n: Radial order.
        m: Azimuthal order.
        resolution: Output resolution (width, height).
        amplitude: Coefficient amplitude.

    Returns:
        2D phase pattern.
    """
    height, width = resolution[1], resolution[0]
    cart = RZern(n)
    ddx = np.linspace(-1.0, 1.0, width)
    ddy = np.linspace(-1.0, 1.0, height)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    coeffs = np.zeros(cart.nk)
    j = cart.nm2noll(n, m) - 1  # Convert to 0-based
    if j < cart.nk:
        coeffs[j] = amplitude
    return cart.eval_grid(coeffs, matrix=True)


# Single active slot for the expensive RZern cart + coordinate grid.
# Keyed by ``(width, height, radius, n_orders)``; any parameter change —
# most importantly a *radius* change — fully rebuilds and replaces the
# previous entry, so only the current aperture's basis (~500 MB at full
# panel resolution) is ever retained in memory.
#
# 验证记录 (2026-09-15, 实证脚本 + 测试锚点):
#   1) 同键重复构造 → 返回同一 cart 对象 (id 相同), 归一化网格与生成结果
#      逐元素一致 → 缓存"加载"命中, 无重复 make_cart_grid。
#   2) radius 变化 → 新 cart 对象 (非复用); 替换槽后旧 cart 与 xv/yv 经
#      weakref 确认引用归 None (GC 释放) → "完全重建", 内存恒定单份。
#   3) 实测 ZZ 字节数 = width*height*nk*8, nk=(n_orders+1)(n_orders+2)//2:
#      100x100/n=6 → 2,240,000 B; 1920x1200/n=6 → ~515 MB; n=10 → ~1.2 GB。
#   4) 回归测试锚点 (tests/ao_shaping/utils/test_zernike_calc.py):
#      test_grid_cache_reused_across_instances / test_radius_change_rebuilds_cart /
#      test_radius_change_releases_old_cart / test_n_orders_change_rebuilds_cart。
_grid_cache: tuple[
    tuple[int, int, float, int], tuple[RZern, np.ndarray, np.ndarray]
] | None = None


def _build_cached_grid(
    width: int, height: int, radius: float, n_orders: int,
) -> tuple[RZern, np.ndarray, np.ndarray]:
    """Build (and cache) the RZern cart + normalized coordinate grid.

    The grid is normalized in *units of radius*: ``(pixel - center) / radius``,
    so the unit circle corresponds exactly to ``radius`` pixels, and the radial
    coordinate ``R`` runs 0..1 across the aperture. The result is cached as a
    **single active slot** keyed by ``(width, height, radius, n_orders)``:
    repeated generation for the same panel/aperture reuses the expensive
    ``RZern.make_cart_grid`` polar tables, while changing the radius (or any
    other key component) fully rebuilds and *releases* the previous basis
    instead of accumulating cached variants.

    Returns:
        ``(cart, xv, yv)`` where ``cart`` is the ``RZern`` instance whose
        cartesian grid has been set, and ``xv``/``yv`` are the normalized
        coordinate grids (shape ``(height, width)``).
    """
    global _grid_cache
    radius = float(radius)
    key = (width, height, radius, n_orders)
    if _grid_cache is not None and _grid_cache[0] == key:
        return _grid_cache[1]

    cart = RZern(n_orders)
    # Pixel offsets from panel centre, normalized by the aperture radius.
    ddx = (np.arange(width) - (width - 1) / 2.0) / radius
    ddy = (np.arange(height) - (height - 1) / 2.0) / radius
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    _grid_cache = (key, (cart, xv, yv))
    return cart, xv, yv


class ZernikeGenerator:
    """Zernike polynomial generator using the zernike package.

    This class wraps the zernike package (from Jacopo Antonello) to provide
    Zernike polynomial generation functionality.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        radius: float | None = None,
        square: bool = True,
        n_orders: int = 6,
    ) -> None:
        """Initialize Zernike polynomial generator.

        Args:
            resolution: Target resolution as (width, height).
            radius: Aperture radius in *pixels*. Defaults to min(height, width) / 2.
                The coordinate grid is normalized by this radius, so the unit
                circle (``sqrt(x²+y²) <= 1`` from :attr:`mask`) corresponds
                exactly to ``radius`` pixels — changing radius genuinely changes
                the aperture size.
            square: If True and resolution is non-square, generate on square grid
                    (max dimension) then crop back to target resolution.
                    This ensures proper aspect ratio for circular patterns.
            n_orders: Number of radial orders (default 6). Determines max Zernike modes.

        Returns:
            None
        """
        height, width = resolution[1], resolution[0]
        if radius is None:
            radius = min(height, width) / 2

        self._height = height
        self._width = width
        self._radius = float(radius)
        self._max_val: float | None = None
        self._n_orders: int = n_orders
        self._square = square

        # Reuse the expensive RZern cart + coordinate grids across instances
        # (keyed by resolution/radius/order). ``make_cart_grid`` builds the polar
        # lookup tables over the full grid — the dominant cost when regenerating.
        self._cart, self.xv, self.yv = _build_cached_grid(
            self._width, self._height, self._radius, self._n_orders,
        )
        self.ddx = self.xv[0, :]
        self.ddy = self.yv[:, 0]

    def nm_to_noll(self, n: int, m: int) -> int:
        """Convert (n, m) Zernike indices to Noll index.

        Args:
            n: Radial order.
            m: Azimuthal order (can be negative).

        Returns:
            Noll index (1-based).
        """
        return self._cart.nm2noll(n, m)


    def noll_to_nm(self, j: int) -> tuple[int, int]:
        """Convert Noll index to (n, m) Zernike indices (aotools convention).

        Uses the aotools RZern library for conversion, which follows the
        standard Noll indexing convention (Noll 1976). Supports any Noll index.

        NOTE: This convention differs from the hardcoded lookup table in
        `optimizer/wf/ga_zernike.py` and `optimizer/wf/rms_by_zernike.py`.
        The aotools version is the canonical implementation.

        Args:
            j: Noll index (1-based).

        Returns:
            Tuple of (n, m) radial and azimuthal orders.
        """
        result = self._cart.noll2nm(j)
        # Handle both tuple and array returns
        if isinstance(result, tuple):
            return (int(result[0]), int(result[1]))
        # Handle array return type
        return (int(result[0][0]), int(result[1][0]))

    def set_bits(self, bits: int) -> None:
        """Set output bit depth.

        Args:
            bits: Number of bits for output (e.g., 10 for 0-1023).
        """
        self._max_val = 2**bits - 1

    def generate_noll(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Generate phase from Noll-index coefficients.

        Args:
            coefficients: 1D array of coefficients (Noll index, 0-based).

        Returns:
            2D array of phase values.
        """

        # Pad coefficients to match available terms
        coeffs = np.zeros(self._cart.nk, dtype=np.float64)
        n_coeffs = min(len(coefficients), self._cart.nk)
        coeffs[:n_coeffs] = coefficients[:n_coeffs]

        # Generate phase (apply crop for square mode)
        return self._cart.eval_grid(coeffs, matrix=True)

    def generate_polynomial(
        self,
        coefficients: dict[tuple[int, int], float],
    ) -> np.ndarray:
        """Generate phase from (n, m) coefficients.

        Args:
            coefficients: Dictionary mapping (n, m) to amplitude.

        Returns:
            2D array of phase values.
        """
        if not coefficients:
            shape = (self._height, self._width)
            return np.zeros(shape, dtype=np.float64)

        # Convert (n, m) to Noll index and create coefficient array
        coeffs = np.zeros(self._cart.nk, dtype=np.float64)
        for (n, m), amp in coefficients.items():
            j = self._cart.nm2noll(n, m) - 1  # Convert to 0-based index
            if j < self._cart.nk:
                coeffs[j] = amp

        # Use generate_noll which handles cropping internally
        return self.generate_noll(coeffs)

    def generate(
        self,
        n: int,
        m: int,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate single Zernike polynomial.

        Args:
            n: Radial order.
            m: Azimuthal order.
            amplitude: Amplitude factor.

        Returns:
            2D array of phase values.
        """
        max_val = self._max_val
        if max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")
        j = self._cart.nm2noll(n, m) - 1 # Convert to 0-based index
        coeffs = np.zeros(self._cart.nk, dtype=np.float64)
        if j < self._cart.nk:
            coeffs[j] = amplitude

        return self.generate_noll(coeffs)

    def fit(self, phase):
        return self._cart.fit_cart_grid(phase)[0]

    @property
    def Theta(self) -> np.ndarray:
        """Get angular coordinates.
        Returns:
            2D array of angles.
        """
        return np.arctan2(self.yv, self.xv)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get resolution (width, height)."""
        return (self._width, self._height)

    @property
    def radius(self) -> float:
        """Get radius."""
        return self._radius

    @property
    def mask(self) -> np.ndarray:
        """Get circular aperture mask.
        Returns:
            2D binary array where 1 indicates inside aperture.
        """
        # The zernike package uses unit circle, so mask is where radius <= 1
        mask = np.sqrt(self.xv**2 + self.yv**2) <= 1.0
        return mask.astype(np.uint8)

    @property
    def R(self) -> np.ndarray:
        """Get radial coordinates.
        Returns:
            2D array of radial distances.
        """
        return np.sqrt(self.xv**2 + self.yv**2)
    
    @property
    def n_modes(self) -> int:
        return self._cart.nk
