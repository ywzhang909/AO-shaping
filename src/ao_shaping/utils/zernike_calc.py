from __future__ import annotations

import numpy as np

from zernike import RZern, FitZern


class ZernikeGenerator:
    """Zernike polynomial generator using the zernike package.
    
    This class wraps the zernike package (from Jacopo Antonello) to provide
    Zernike polynomial generation functionality.
    """

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
        self._n_orders: int = 6  # default to 6 radial orders

        # Create zernike RZern object
        self._cart = RZern(self._n_orders)

        # Create normalized coordinate grid (in units of radius)
        ddx = np.linspace(-1.0, 1.0, width)
        ddy = np.linspace(-1.0, 1.0, height)
        xv, yv = np.meshgrid(ddx, ddy)
        self._cart.make_cart_grid(xv, yv)

    def precompute_bases(self, n_terms: int) -> None:
        """Precompute Zernike bases up to n_terms.
        
        Args:
            n_terms: Number of Zernike terms to compute.
        """
        # Calculate required radial orders to get n_terms
        # Number of terms = (n_orders + 1) * (n_orders + 2) / 2
        # Solve for n_orders: n_terms >= (n_orders + 1) * (n_orders + 2) / 2
        n = 0
        count = 0
        while count < n_terms:
            n += 1
            count = n * (n + 1) // 2 + 1  # +1 for piston
        
        self._n_orders = n
        self._cart = RZern(self._n_orders)
        
        # Re-create grid
        ddx = np.linspace(-1.0, 1.0, self._width)
        ddy = np.linspace(-1.0, 1.0, self._height)
        xv, yv = np.meshgrid(ddx, ddy)
        self._cart.make_cart_grid(xv, yv)

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
        max_val = self._max_val
        if max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")

        # Pad coefficients to match available terms
        coeffs = np.zeros(self._cart.nk, dtype=np.float64)
        n_coeffs = min(len(coefficients), self._cart.nk)
        coeffs[:n_coeffs] = coefficients[:n_coeffs]

        # Generate phase
        phase = self._cart.eval_grid(coeffs, matrix=True)

        # Scale to output range
        if max_val > 0:
            # Normalize to [0, max_val]
            phase_min = phase.min()
            phase_max = phase.max()
            if phase_max > phase_min:
                phase = (phase - phase_min) / (phase_max - phase_min) * max_val
                phase = np.clip(phase, 0, max_val)  # Ensure within bounds
            else:
                # All values are the same - set to middle of range
                phase = np.full_like(phase, max_val // 2)

        return phase.astype(np.uint16)

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
            return np.zeros((self._height, self._width), dtype=np.uint16)

        # Convert (n, m) to Noll index and create coefficient array
        coeffs = np.zeros(self._cart.nk, dtype=np.float64)
        for (n, m), amp in coefficients.items():
            j = nm_to_noll(n, m) - 1  # Convert to 0-based index
            if j < self._cart.nk:
                coeffs[j] = amp

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

        j = nm_to_noll(n, m) - 1  # Convert to 0-based index
        coeffs = np.zeros(self._cart.nk, dtype=np.float64)
        if j < self._cart.nk:
            coeffs[j] = amplitude

        return self.generate_noll(coeffs)

    @property
    def mask(self) -> np.ndarray:
        """Get circular aperture mask.
        
        Returns:
            2D binary array where 1 indicates inside aperture.
        """
        # The zernike package uses unit circle, so mask is where radius <= 1
        mask = np.sqrt(self._cart.xv**2 + self._cart.yv**2) <= 1.0
        return mask.astype(np.uint8)

    @property
    def R(self) -> np.ndarray:
        """Get radial coordinates.
        
        Returns:
            2D array of radial distances.
        """
        return np.sqrt(self._cart.xv**2 + self._cart.yv**2)

    @property
    def Theta(self) -> np.ndarray:
        """Get angular coordinates.
        
        Returns:
            2D array of angles.
        """
        return np.arctan2(self._cart.yv, self._cart.xv)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get resolution (width, height)."""
        return (self._width, self._height)

    @property
    def radius(self) -> float:
        """Get radius."""
        return self._radius


def fit_zernike(
    wavefront: np.ndarray,
    n_max: int = 4,
    radius: float | None = None,
) -> dict[tuple[int, int], float]:
    """Fit Zernike polynomials to wavefront data.
    
    Args:
        wavefront: 2D array of wavefront values.
        n_max: Maximum radial order.
        radius: Aperture radius (defaults to min dimensions / 2).
        
    Returns:
        Dictionary mapping (n, m) to fitted coefficients.
    """
    height, width = wavefront.shape
    if radius is None:
        radius = min(height, width) / 2

    # Create Zernike fitter - note: FitZern expects (L, K) = (rows, cols)
    pol = RZern(n_max)
    ip = FitZern(pol, height, width)

    # Make a Cartesian grid for the wavefront (not polar)
    ddx = np.linspace(-1.0, 1.0, width)
    ddy = np.linspace(-1.0, 1.0, height)
    xv, yv = np.meshgrid(ddx, ddy)
    pol.make_cart_grid(xv, yv)

    # Fit wavefront using Cartesian grid
    coeffs = pol.fit_cart_grid(wavefront)[0]

    # Convert to (n, m) dictionary
    result: dict[tuple[int, int], float] = {}
    for j, amp in enumerate(coeffs):
        n, m = noll_to_nm(j + 1)
        result[(n, m)] = float(amp)

    return result


def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Calculate radial part of Zernike polynomial.
    
    Args:
        n: Radial order.
        m: Azimuthal order.
        r: Radial coordinates.
        
    Returns:
        Radial polynomial values.
    """
    # Use zernike package for calculation
    cart = RZern(n)
    L = len(r)
    ddx = np.linspace(-1.0, 1.0, L)
    ddy = np.zeros(L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)

    coeffs = np.zeros(cart.nk)
    j = nm_to_noll(n, m) - 1
    if j < cart.nk:
        coeffs[j] = 1.0

    return cart.eval_grid(coeffs)


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m).
    
    Args:
        j: Noll index (1-based).
        
    Returns:
        (n, m) tuple.
    """
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
    """Convert (n, m) to Noll index.
    
    Args:
        n: Radial order.
        m: Azimuthal order.
        
    Returns:
        Noll index (1-based).
    """
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
    """Generate phase from Noll coefficients.
    
    Args:
        coeffs: 1D array of Noll coefficients.
        resolution: (width, height).
        radius: Aperture radius.
        bits: Output bit depth.
        
    Returns:
        2D phase array.
    """
    gen = ZernikeGenerator(resolution=resolution, radius=radius)
    gen.set_bits(bits)
    gen.precompute_bases(len(coeffs))
    return gen.generate_noll(coeffs)
