from __future__ import annotations

import numpy as np
from zernike import RZern

# Zernike polynomial naming (Noll's scheme)
ZERNIKE_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "Piston",
    (1, -1): "Tip",
    (1, 1): "Tilt",
    (2, 0): "Defocus",
    (2, -2): "Astigmatism 45°",
    (2, 2): "Astigmatism 0°",
    (3, -1): "Coma Y",
    (3, 1): "Coma X",
    (3, -3): "Trefoil Y",
    (3, 3): "Trefoil X",
    (4, 0): "Spherical",
    (4, -2): "Secondary Astig 45°",
    (4, 2): "Secondary Astig 0°",
    (4, -4): "Tetrafoil Y",
    (4, 4): "Tetrafoil X",
}


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
            radius: Aperture radius. Defaults to min(height, width) / 2.
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
        self._radius = radius
        self._max_val: float | None = None
        self._n_orders: int = n_orders
        self._square = square

        # Effective resolution for Zernike generation
        scaler = self._height / self._width

        # Create zernike RZern object
        self._cart = RZern(self._n_orders)

        # Create normalized coordinate grid (in units of radius)
        self.ddx = np.linspace(-1.0, 1.0, self._width)
        self.ddy = np.linspace(-1.0, 1.0, self._height) * scaler
        self.xv, self.yv = np.meshgrid(self.ddx, self.ddy)
        self._cart.make_cart_grid(self.xv, self.yv)

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
