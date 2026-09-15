from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import hadamard
from scipy.ndimage import zoom


def is_hadamard_order(n: int) -> bool:
    """Check if n is a valid Hadamard matrix order.

    Valid orders are powers of 2 (2, 4, 8, 16, 32, 64, 128, 256, ...).

    Args:
        n: Order to check.

    Returns:
        True if n is a valid Hadamard order, False otherwise.
    """
    if n < 2:
        return False
    return (n & (n - 1)) == 0


def calc_n_hadamard_modes(mode_order: int) -> int:
    """Calculate the total number of 2D Walsh-Hadamard modes.

    For a mode_order of N, there are N x N = N² 2D modes formed by
    the outer products of 1D Walsh functions.

    Args:
        mode_order: The order N of the Hadamard matrix.

    Returns:
        Number of 2D Walsh-Hadamard modes (N²).

    Raises:
        ValueError: If mode_order is not a valid Hadamard order.
    """
    if not is_hadamard_order(mode_order):
        raise ValueError(f"mode_order must be a power of 2 >= 2, got {mode_order}")
    return mode_order * mode_order


def hadamard_mode_2d(u: int, v: int, order: int) -> np.ndarray:
    """Generate a single N×N Walsh-Hadamard mode (before resize to SLM grid).

    The 2D Walsh-Hadamard mode W_{u,v} is the outer product of
    two 1D Walsh functions: W_{u,v} = H[u,:] ⊗ H[v,:]

    Args:
        u: Row sequency index (0 to order-1).
        v: Column sequency index (0 to order-1).
        order: The order N of the Hadamard matrix.

    Returns:
        N×N array containing the 2D Walsh-Hadamard mode.

    Raises:
        ValueError: If u or v are out of range for the given order.
    """
    if not (0 <= u < order):
        raise ValueError(f"u must be in [0, {order}), got {u}")
    if not (0 <= v < order):
        raise ValueError(f"v must be in [0, {order}), got {v}")

    H = hadamard(order)
    mode = np.outer(H[u, :], H[v, :])
    return mode


class HadamardGenerator:
    """Walsh-Hadamard mode generator for SLM/DM phase pattern generation.

    Generates 2D Walsh-Hadamard basis functions from 1D Hadamard matrices
    via Kronecker product. Supports circular and rectangular pupil masks.

    The Walsh-Hadamard modes form a complete orthogonal basis with values ±1,
    useful for wavefront sensing, phase modulation, and adaptive optics.

    Attributes:
        resolution: Output resolution as (width, height).
        mode_order: The order N of the underlying Hadamard matrix (power of 2).
        mask_type: Type of pupil mask ("circular" or "rectangular").
        radius: Aperture radius in normalized coordinates.

    Example:
        >>> gen = HadamardGenerator(resolution=(1920, 1080), mode_order=8)
        >>> gen.set_bits(10)
        >>> mode = gen.generate(0, amplitude=0.5)  # First mode
        >>> coeffs = np.random.randn(64) * 0.1
        >>> phase = gen.generate_modes(coeffs)  # Combined phase
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        mode_order: int = 8,
        mask_type: str = "circular",
        radius: float | None = None,
    ) -> None:
        """Initialize the Hadamard mode generator.

        Args:
            resolution: Target resolution as (width, height) in pixels.
            mode_order: The order N of the Hadamard matrix. Must be a power of 2.
                       Default is 8, giving 64 total 2D modes.
            mask_type: Pupil mask type. "circular" for round aperture,
                      "rectangular" for full rectangular aperture.
            radius: Aperture radius in normalized coordinates [-1, 1].
                   Defaults to 1.0 (full pupil).

        Raises:
            ValueError: If mode_order is not a power of 2 >= 2.
            ValueError: If mask_type is not "circular" or "rectangular".
        """
        if not is_hadamard_order(mode_order):
            raise ValueError(
                f"mode_order must be a power of 2 >= 2 (e.g., 2, 4, 8, 16, 32, 64, 128), "
                f"got {mode_order}"
            )
        if mask_type not in ("circular", "rectangular"):
            raise ValueError(f"mask_type must be 'circular' or 'rectangular', got '{mask_type}'")

        self._width, self._height = resolution
        self._mode_order = mode_order
        self._mask_type = mask_type
        self._radius = radius if radius is not None else 1.0
        self._max_val: float | None = None

        # Generate the Hadamard matrix (N x N with ±1 entries)
        self._hadamard = hadamard(mode_order).astype(np.float64)

        # Create normalized coordinate grids [-1, 1]
        self._x = np.linspace(-1.0, 1.0, self._width)
        self._y = np.linspace(-1.0, 1.0, self._height)
        self._xv, self._yv = np.meshgrid(self._x, self._y)

        # Precompute the pupil mask
        self._mask = self._compute_mask()

        # Precompute valid pixel indices for efficiency
        self._valid_pixels = np.where(self._mask.ravel())[0]
        self._n_valid_pixels = len(self._valid_pixels)

    def _compute_mask(self) -> np.ndarray:
        """Compute the pupil mask based on mask_type and radius.

        Returns:
            Binary mask array of shape (height, width) with 1 inside aperture,
            0 outside.
        """
        if self._mask_type == "circular":
            r = np.sqrt(self._xv**2 + self._yv**2)
            return (r <= self._radius).astype(np.uint8)
        else:  # rectangular
            return np.ones((self._height, self._width), dtype=np.uint8)

    def _uv_to_idx(self, u: int, v: int) -> int:
        """Convert (u, v) sequency indices to flat mode index.

        Args:
            u: Row sequency (0 to mode_order-1).
            v: Column sequency (0 to mode_order-1).

        Returns:
            Flat mode index in [0, mode_order²).
        """
        return u * self._mode_order + v

    def _idx_to_uv(self, mode_index: int) -> tuple[int, int]:
        """Convert flat mode index to (u, v) sequency indices.

        Args:
            mode_index: Flat index in [0, mode_order²).

        Returns:
            Tuple of (u, v) sequency indices.

        Raises:
            IndexError: If mode_index is out of range.
        """
        if not (0 <= mode_index < self.n_modes):
            raise IndexError(
                f"mode_index must be in [0, {self.n_modes}), got {mode_index}"
            )
        u = mode_index // self._mode_order
        v = mode_index % self._mode_order
        return u, v

    def _compute_mode(self, u: int, v: int) -> np.ndarray:
        """Compute a single 2D Walsh-Hadamard mode and resize to SLM resolution.

        Args:
            u: Row sequency (0 to mode_order-1).
            v: Column sequency (0 to mode_order-1).

        Returns:
            Resized mode array of shape (height, width).
        """
        # Get the N×N mode
        mode_nxn = hadamard_mode_2d(u, v, self._mode_order)

        # Resize to SLM resolution using interpolation
        zoom_y = self._height / self._mode_order
        zoom_x = self._width / self._mode_order
        mode_resized = zoom(mode_nxn, (zoom_y, zoom_x), order=1)

        # Apply pupil mask
        mode_resized = mode_resized * self._mask

        return mode_resized

    def set_bits(self, bits: int) -> None:
        """Set the output bit depth for SLM display.

        Args:
            bits: Number of bits (e.g., 10 for 0-1023 range).

        Example:
            >>> gen.set_bits(10)
            >>> # Output values will be scaled to [0, 1023]
        """
        self._max_val = 2**bits - 1

    def generate(self, mode_index: int, amplitude: float = 1.0) -> np.ndarray:
        """Generate a single 2D Walsh-Hadamard mode by flat index.

        Args:
            mode_index: Flat index into the modes [0, mode_order²).
            amplitude: Amplitude scaling factor. Values ±1 are scaled by this.

        Returns:
            2D phase array of shape (height, width). If set_bits() was called,
            returns uint16 in range [0, max_val]. Otherwise returns float64
            with values approximately in [-amplitude, amplitude].

        Raises:
            ValueError: If set_bits() has not been called (for uint16 output).
            IndexError: If mode_index is out of range.
        """
        u, v = self._idx_to_uv(mode_index)
        return self.generate_row_col(u, v, amplitude)

    def generate_row_col(self, u: int, v: int, amplitude: float = 1.0) -> np.ndarray:
        """Generate a single 2D Walsh-Hadamard mode by (row, col) sequency indices.

        Args:
            u: Row sequency (0 to mode_order-1).
            v: Column sequency (0 to mode_order-1).
            amplitude: Amplitude scaling factor.

        Returns:
            2D phase array of shape (height, width).

        Raises:
            ValueError: If set_bits() has not been called.
            ValueError: If u or v are out of range.
        """
        if self._max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")

        # Compute the mode
        mode = self._compute_mode(u, v)

        # Scale from ±1 to [0, max_val] * amplitude
        # First normalize to [0, 1], then scale
        mode_normalized = (mode + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        result = mode_normalized * amplitude * self._max_val

        return result.astype(np.uint16) if self._max_val is not None else result

    def generate_modes(self, coefficients: np.ndarray) -> np.ndarray:
        """Generate combined phase from a coefficient vector.

        Linearly combines multiple Walsh-Hadamard modes weighted by coefficients.

        Args:
            coefficients: 1D array of coefficients for each mode.
                         Length should be ≤ n_modes. Shorter arrays are
                         zero-padded, longer arrays are truncated.

        Returns:
            2D phase array of shape (height, width) with combined pattern.
            If set_bits() was called, returns uint16. Otherwise float64.

        Raises:
            ValueError: If set_bits() has not been called.

        Example:
            >>> gen.set_bits(10)
            >>> coeffs = np.zeros(64)
            >>> coeffs[0] = 0.5  # First mode at half amplitude
            >>> coeffs[5] = 0.3  # Sixth mode
            >>> phase = gen.generate_modes(coeffs)
        """
        if self._max_val is None:
            raise ValueError("Call set_bits() first to configure output scale")

        # Initialize output
        phase = np.zeros((self._height, self._width), dtype=np.float64)

        # Trim or pad coefficients
        n_coeffs = min(len(coefficients), self.n_modes)

        # Linear combination of modes
        for idx in range(n_coeffs):
            if abs(coefficients[idx]) < 1e-10:
                continue
            u, v = self._idx_to_uv(idx)
            mode = self._compute_mode(u, v)
            phase += coefficients[idx] * mode

        # Normalize and scale to output range
        # Walsh-Hadamard values are ±1, so sum can range from -sum(abs(coeffs)) to +sum(abs(coeffs))
        # We normalize by max possible sum to get into [-1, 1], then map to [0, max_val]
        max_possible = np.sum(np.abs(coefficients[:n_coeffs]))
        if max_possible > 1e-10:
            phase = phase / max_possible  # Normalize to [-1, 1]

        # Map from [-1, 1] to [0, max_val]
        phase_normalized = (phase + 1.0) / 2.0
        result = phase_normalized * self._max_val

        return result.astype(np.uint16)

    def generate_modes_dict(
        self, coefficients: dict[tuple[int, int], float]
    ) -> np.ndarray:
        """Generate combined phase from a dictionary of coefficients.

        Args:
            coefficients: Dictionary mapping {(u, v): amplitude} where u and v
                         are the row and column sequency indices.

        Returns:
            2D phase array of shape (height, width).

        Raises:
            ValueError: If set_bits() has not been called.
            ValueError: If any (u, v) key is out of range.
        """
        # Convert dict to array
        coeffs_array = np.zeros(self.n_modes)
        for (u, v), amp in coefficients.items():
            idx = self._uv_to_idx(u, v)
            if not (0 <= idx < self.n_modes):
                raise ValueError(f"Invalid mode indices ({u}, {v}) for mode_order {self._mode_order}")
            coeffs_array[idx] = amp

        return self.generate_modes(coeffs_array)

    @property
    def n_modes(self) -> int:
        """Total number of 2D Walsh-Hadamard modes (mode_order²)."""
        return self._mode_order * self._mode_order

    @property
    def mask(self) -> np.ndarray:
        """Get the pupil mask.

        Returns:
            2D binary array of shape (height, width) where 1 indicates
            inside the aperture.
        """
        return self._mask.copy()

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the output resolution as (width, height)."""
        return (self._width, self._height)

    @property
    def radius(self) -> float:
        """Get the aperture radius in normalized coordinates."""
        return self._radius

    @property
    def hadamard_matrix(self) -> np.ndarray:
        """Get the raw N×N Hadamard matrix.

        Returns:
            The underlying Hadamard matrix with shape (mode_order, mode_order)
            containing values ±1.
        """
        return self._hadamard.copy()

    def get_valid_pixels(self) -> np.ndarray:
        """Get indices of valid pixels within the pupil mask.

        Returns:
            1D array of flat indices for pixels inside the aperture.
        """
        return self._valid_pixels.copy()

    def get_mode_indices(self) -> list[tuple[int, int]]:
        """Get list of all (u, v) sequency index pairs.

        Returns:
            List of (u, v) tuples for all n_modes modes.
        """
        indices = []
        for idx in range(self.n_modes):
            indices.append(self._idx_to_uv(idx))
        return indices
