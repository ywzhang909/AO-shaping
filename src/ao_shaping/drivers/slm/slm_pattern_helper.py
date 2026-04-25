# 相位图案生成函数
from __future__ import annotations

import numpy as np

from ao_shaping.utils.zernike_calc import ZernikeGenerator, nm_to_noll

from aotools.turbulence import PhaseScreenKolmogorov
from aotools import ft_phase_screen



class PatternHelper:
    def __init__(self, resolution: tuple[int, int], bits: int = 10) -> None:
        self.resolution = resolution
        self.bits = bits
        height, width = resolution[1], resolution[0]
        self._max_val = 2**bits - 1
        self._height = height
        self._width = width

        # Cached coordinate arrays (lazy-computed)
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._xx: np.ndarray | None = None
        self._yy: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._Theta: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._pixel_x: np.ndarray | None = None
        self._pixel_y: np.ndarray | None = None

    @property
    def x(self) -> np.ndarray:
        """1D x coordinates (centered at 0)."""
        if self._x is None:
            self._x = np.arange(self._width, dtype=np.float64) - self._width // 2
        return self._x

    @property
    def y(self) -> np.ndarray:
        """1D y coordinates (centered at 0)."""
        if self._y is None:
            self._y = np.arange(self._height, dtype=np.float64) - self._height // 2
        return self._y

    @property
    def xx(self) -> np.ndarray:
        """2D meshgrid x coordinates."""
        if self._xx is None:
            self._xx, self._yy = np.meshgrid(self.x, self.y)
        return self._xx

    @property
    def yy(self) -> np.ndarray:
        """2D meshgrid y coordinates."""
        if self._yy is None:
            self._xx, self._yy = np.meshgrid(self.x, self.y)
        return self._yy

    @property
    def R(self) -> np.ndarray:
        """Radial distance from center."""
        if self._R is None:
            self._R = np.sqrt(self.xx**2 + self.yy**2)
        return self._R

    @property
    def Theta(self) -> np.ndarray:
        """Azimuthal angle from center."""
        if self._Theta is None:
            self._Theta = np.arctan2(self.yy, self.xx)
        return self._Theta

    @property
    def mask(self) -> np.ndarray:
        """Circular pupil mask (R <= 1.0)."""
        if self._mask is None:
            radius = min(self._height, self._width) / 2
            self._mask = (self.R <= radius).astype(np.float64)
        return self._mask

    @property
    def pixel_x(self) -> np.ndarray:
        """Pixel x coordinates (centered at 0, in pixel units)."""
        if self._pixel_x is None:
            self._pixel_x = np.arange(self._width, dtype=np.float64) - self._width / 2
        return self._pixel_x

    @property
    def pixel_y(self) -> np.ndarray:
        """Pixel y coordinates (centered at 0, in pixel units)."""
        if self._pixel_y is None:
            self._pixel_y = np.arange(self._height, dtype=np.float64) - self._height / 2
        return self._pixel_y

    def generate_checkerboard(self, period: int = 100) -> np.ndarray:
        max_val = self._max_val

        y = np.arange(self._height) // period
        x = np.arange(self._width) // period
        X, Y = np.meshgrid(x, y)

        checker = (X + Y) % 2
        img = (checker * max_val).astype(np.uint16)

        return img

    def generate_binary_grating(
        self, a: int = 2, b: int = 3, direction: str = "horizontal"
    ) -> np.ndarray:
        height, width = self._height, self._width
        max_val = (2**self.bits - 1) // 2

        if direction == "horizontal":
            y = np.arange(height)
            grating = np.where(y % (a + b) < b, 0, max_val)
            img = np.tile(grating[:, np.newaxis], (1, width))
        else:
            x = np.arange(width)
            grating = np.where(x % (a + b) < b, 0, max_val)
            img = np.tile(grating[np.newaxis, :], (height, 1))

        return img.astype(np.uint16)

    def generate_microlens_array(
        self,
        lens_size: int = 200,
        focal_length: float = 0.1,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
    ) -> np.ndarray:
        height, width = self._height, self._width
        max_val = self._max_val

        x = (np.arange(lens_size, dtype=np.float64) - lens_size / 2) * pixel_size
        y = (np.arange(lens_size, dtype=np.float64) - lens_size / 2) * pixel_size
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2

        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))
        phase_wrapped = np.mod(phase, 2 * np.pi)
        lens_pattern = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        n_y = height // lens_size + 1
        n_x = width // lens_size + 1

        array = np.tile(lens_pattern, (n_y, n_x))

        img = array[:height, :width]

        return img

    def generate_turbulence_screen(
        self,
        Cn2: float = 1e-14,
        L: float = 1000,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
        screen_size: float | None = None,
        L0: float | None = None,
        l0: float | None = None,
        random_seed: int | None = None,
        method: str = "kolmogorov",
    ) -> np.ndarray:
        """Generate turbulence phase screen.

        Args:
            Cn2: Refractive index structure constant (m^(2/3)).
                Default 1e-14 corresponds to weak turbulence.
            L: Propagation path length in meters.
            wavelength: Wavelength in meters.
            pixel_size: Pixel size in meters.
            screen_size: Physical size of the screen in meters.
                Defaults to max(height, width) * pixel_size.
            L0: Outer scale in meters. Defaults to 10 * screen_size.
            l0: Inner scale in meters. Defaults to pixel_size * 2.
            random_seed: Random seed for reproducibility.
                Only used when method='kolmogorov'.
            method: 'kolmogorov' (PhaseScreenKolmogorov) or 'vankarman' (ft_phase_screen).
                Defaults to 'kolmogorov'.

        Returns:
            Turbulence phase screen in radians (normalized to [0, 2π)).
        """
        height, width = self._height, self._width
        max_val = self._max_val

        if screen_size is None:
            screen_size = max(height, width) * pixel_size

        if L0 is None:
            L0 = 10 * screen_size
        if l0 is None:
            l0 = pixel_size * 2


        # Compute Fried parameter r0 from Cn2
        r0 = (wavelength**2 / (Cn2 * L * 0.033 * (2 * np.pi) ** 2)) ** (3 / 5)

        try:
            if method == "kolmogorov":
                # Use PhaseScreenKolmogorov for Kolmogorov turbulence
                screen = PhaseScreenKolmogorov(
                    nx_size=height,
                    pixel_scale=pixel_size,
                    r0=r0,
                    L0=L0,
                    random_seed=random_seed,
                )
                phase_screen = screen.scrn
            else:
                # Fallback to ft_phase_screen for Von Karman
                screen = ft_phase_screen(r0, height, pixel_size, L0, l0)
                phase_screen = screen[:height, :width]
        except np.linalg.LinAlgError:
            # PhaseScreenKolmogorov can fail for certain L0/pixel_scale combinations
            # Fall back to ft_phase_screen
            screen = ft_phase_screen(r0, height, pixel_size, L0, l0)
            phase_screen = screen[:height, :width]

        # Normalize to [0, 2π) and convert to uint16
        phase_min = phase_screen.min()
        phase_max = phase_screen.max()
        phase_normalized = (
            (phase_screen - phase_min)
            / (phase_max - phase_min + 1e-10)
            * 2 * np.pi
        )

        img = (phase_normalized / (2 * np.pi) * max_val).astype(np.uint16)
        return img

    def generate_zernike(
        self,
        n: int,
        m: int,
        amplitude: float = 1.0,
        radius: float | None = None,
    ) -> np.ndarray:
        """Generate single Zernike mode phase pattern.

        Args:
            n: Zernike radial order.
            m: Zernike azimuthal frequency.
            amplitude: Zernike coefficient amplitude.
            radius: Pupil radius in pixels. Defaults to half of min dimension.

        Returns:
            Phase pattern as uint16 (0 to 2^bits-1).
        """
        gen = ZernikeGenerator(resolution=(self._height, self._width), radius=radius)
        gen.set_bits(self.bits)
        # generate() internally needs bases cached; precompute up to needed Noll index
        j = nm_to_noll(n, m)
        gen.precompute_bases(j)
        return gen.generate(n, m, amplitude)

    def generate_zernike_polynomial(
        self,
        coefficients: dict[tuple[int, int], float] | None = None,
        radius: float | None = None,
    ) -> np.ndarray:
        """Generate multi-mode Zernike polynomial phase pattern.

        Args:
            n_max: Maximum radial order. Used to precompute Zernike bases.
            coefficients: Dict of {(n, m): amplitude}. Defaults to piston only.
            radius: Pupil radius in pixels. Defaults to half of min dimension.

        Returns:
            Phase pattern as uint16 (0 to 2^bits-1).
        """
        gen = ZernikeGenerator(resolution=(self._height, self._width), radius=radius)
        gen.set_bits(self.bits)

        if coefficients is None:
            coefficients = {}
        max_noll = max(nm_to_noll(n, m) for (n, m) in coefficients) if coefficients else 1
        n_terms = max_noll
        gen.precompute_bases(n_terms)

        if not coefficients:
            return np.zeros((self._height, self._width), dtype=np.uint16)

        return gen.generate_polynomial(coefficients)

    def to_uint16(self, phase_radians: np.ndarray) -> np.ndarray:
        """Convert phase in radians to uint16 for SLM display.

        Args:
            phase_radians: Phase array in radians (0 to phase_range)

        Returns:
            Phase array in uint16 format (0 to 2^bits - 1)
        """
        phase_wrapped = np.mod(phase_radians, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * self._max_val).astype(np.uint16)
        return img

    def generate_focus(
        self,
        focal_length: float,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """Generate focus pattern (lens phase)."""
        max_val = self._max_val
        R2 = self.xx**2 + self.yy**2
        phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
        if not wrap_phase:
            return phase

        phase_wrapped = np.mod(phase, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)
        return img

    def generate_dammann_grating(
        self, order: int = 3, fill_factor: float = 0.5
    ) -> np.ndarray:
        """
        Generate a Dammann grating phase pattern

        Args:
            order: Number of diffraction orders in each direction (typically 2, 3, 4)
            fill_factor: Ratio of transparent area in each cell (0.0 to 1.0)

        Returns:
            Phase pattern (0 or max_val)
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        if order <= 0:
            order = 1

        # Calculate the size of each grating element
        elem_width = width // order
        elem_height = height // order

        # Create the Dammann grating pattern
        img = np.zeros((height, width), dtype=np.uint16)

        # Fill each grating element with alternating phase values
        for i in range(order):
            for j in range(order):
                # Define the region for this grating element
                y_start = i * elem_height
                y_end = min((i + 1) * elem_height, height)
                x_start = j * elem_width
                x_end = min((j + 1) * elem_width, width)

                # Determine phase based on position (alternating 0 and pi)
                if (i + j) % 2 == 0:
                    # Set to max phase (pi phase shift)
                    img[y_start:y_end, x_start:x_end] = max_val
                else:
                    # Set to zero phase
                    img[y_start:y_end, x_start:x_end] = 0

        return img

    def linear_grating(
        self,
        period: float,
        phase_range: float = 2 * np.pi,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """Generate linear (blazed) grating pattern."""
        max_val = self._max_val
        phase = (self.xx / period) * phase_range

        if not wrap_phase:
            return np.mod(phase, phase_range)

        phase_wrapped = np.mod(phase, phase_range)
        img = (phase_wrapped / phase_range * max_val).astype(np.uint16)
        return img

    def circular_grating(
        self,
        radius: float,
        phase_range: float = 2 * np.pi,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """Generate circular (radial) grating pattern."""
        max_val = self._max_val
        rr = np.sqrt(self.xx**2 + self.yy**2)
        phase = (rr / radius) * phase_range

        if not wrap_phase:
            return np.mod(phase, phase_range)

        phase_wrapped = np.mod(phase, phase_range)
        img = (phase_wrapped / phase_range * max_val).astype(np.uint16)
        return img

    def lens(
        self,
        focal_length: float,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
    ) -> np.ndarray:
        """Generate lens (focus) phase pattern."""
        xx = self.pixel_x * pixel_size
        yy = self.pixel_y * pixel_size
        xx, yy = np.meshgrid(xx, yy)
        r2 = xx**2 + yy**2
        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))
        return np.mod(phase, 2 * np.pi)

    def hologram(self, period: float, phase_range: float = 2 * np.pi) -> np.ndarray:
        """Generate hologram (alias for linear_grating).

        Args:
            period: Grating period in pixels
            phase_range: Maximum phase range in radians (default 2π)

        Returns:
            Phase pattern in radians (0 to phase_range), wrapped
        """
        return self.linear_grating(period=period, phase_range=phase_range, wrap_phase=False)

    def dammann_grating(
        self,
        width: int,
        height: int,
        order: int = 3,
        phase_range: float = 2 * np.pi,
    ) -> np.ndarray:
        """Generate a Dammann grating phase pattern.

        A Dammann grating is a binary-phase grating that generates uniform diffraction orders.
        It creates a specific number of equally intense spots at regular intervals.

        Args:
            width: Width of the output pattern in pixels
            height: Height of the output pattern in pixels
            order: Number of diffraction orders (typically 2, 3, 4, etc.)
            phase_range: Total phase range in radians (default 2π)

        Returns:
            Phase pattern array in radians
        """
        # Create coordinate grids
        if order <= 1:
            order = 2  # Minimum order is 2

        # Calculate the Dammann grating pattern
        # For a 1D Dammann grating, the phase follows a specific sequence to create uniform orders
        # For 2D, we can combine two 1D gratings orthogonally

        # Calculate the spatial frequency for the specified order
        # The grating period is chosen such that it creates the desired number of orders
        period_x = width // order
        period_y = height // order

        # Create the phase pattern based on Dammann grating principles
        # This implementation creates a pattern that generates uniform diffraction orders
        phase_x = (self.xx // period_x) % 2 * np.pi  # Alternate 0 and π phases
        phase_y = (self.yy // period_y) % 2 * np.pi  # Alternate 0 and π phases

        # Combine both dimensions (XOR-like behavior)
        return np.mod(phase_x + phase_y, phase_range)

