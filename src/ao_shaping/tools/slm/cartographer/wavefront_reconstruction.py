"""Fourier wavefront reconstruction from sparse centroid displacements.

Implements the Fourier transform method for wavefront reconstruction
from Hartmann-Shack centroid displacements as described in the paper.

The WFS provides slopes (gradients) at subaperture centers; Fourier
reconstruction converts these to phase via the relationship:
    φ(x,y) = F⁻¹{ F{Wslopes} / (i2πf) }
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FourierReconstructionConfig:
    """Configuration for Fourier wavefront reconstruction.

    Attributes:
        output_resolution: Resolution of output wavefront grid (height, width).
        wavelength_nm: Wavelength in nm (for unit conversion).
        lenslet_pitch_mm: Microlens pitch in mm.
        focal_length_mm: Microlens focal length in mm.
        pad_factor: Zero-padding factor for FFT (reduces wrap-around).
        filter_type: Frequency filter type ('none', 'lowpass', 'wiener').
        cutoff_freq: Cutoff frequency for lowpass filter (cycles/pupil).
        regularization: Regularization parameter for inverse filter.
        interpolate_dense: If True, interpolate sparse input to dense grid.
        dense_grid_resolution: Resolution for dense interpolation grid.
    """

    output_resolution: tuple[int, int] | None = None
    wavelength_nm: float = 532.0
    lenslet_pitch_mm: float = 0.15
    focal_length_mm: float = 4.6
    pad_factor: float = 2.0
    filter_type: str = "wiener"
    cutoff_freq: float = 0.5
    regularization: float = 1e-4
    interpolate_dense: bool = True
    dense_grid_resolution: tuple[int, int] | None = None


class FourierWavefrontReconstructor:
    """Fourier-based wavefront reconstructor from centroid displacements.

    Takes sparse centroid displacements from WFS subapertures and reconstructs
    the continuous wavefront phase distribution using FFT-based integration.

    Args:
        config: Reconstruction configuration.
        num_spots: (num_spots_x, num_spots_y) from WFS.
        pupil_pitch_mm: Spacing between adjacent subaperture centers (mm).
    """

    def __init__(
        self,
        config: FourierReconstructionConfig,
        num_spots: tuple[int, int],
        pupil_pitch_mm: float = 0.15,
    ):
        self.config = config
        self.num_spots = num_spots
        self.pupil_pitch_mm = pupil_pitch_mm
        self._freq_filter: np.ndarray | None = None

    def reconstruct(
        self,
        displacement_x: np.ndarray,
        displacement_y: np.ndarray,
        lens_to_pupil_scale: float = 1.0,
    ) -> np.ndarray:
        """Reconstruct wavefront from centroid displacements.

        Args:
            displacement_x: 2D array of X displacements (num_spots_y, num_spots_x).
            displacement_y: 2D array of Y displacements (num_spots_y, num_spots_x).
            lens_to_pupil_ratio: Conversion factor from WFS pixel displacement
                to pupil-plane spatial frequency.

        Returns:
            2D wavefront array in radians (or waves, depending on scale).
        """
        dx = np.asarray(displacement_x, dtype=np.float64)
        dy = np.asarray(displacement_y, dtype=np.float64)

        if dx.ndim != 2 or dy.ndim != 2:
            raise ValueError(
                f"Displacement arrays must be 2D, got shapes: {dx.shape}, {dy.shape}"
            )

        ny, nx = dx.shape

        if self.config.interpolate_dense:
            dx_dense, dy_dense = interpolate_sparse_to_dense(
                dx,
                dy,
                resolution=self.config.dense_grid_resolution
                or self.config.output_resolution,
            )
        else:
            dx_dense, dy_dense = dx, dy

        return self._fourier_reconstruct(dx_dense, dy_dense, lens_to_pupil_scale)

    def _fourier_reconstruct(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Core Fourier reconstruction algorithm.

        Wavefront gradients (slopes) are related to displacements by:
            ∂W/∂x = displacement_x * (f / pitch)
            ∂W/∂y = displacement_y * (f / pitch)

        Phase is recovered by Fourier-domain integration:
            W(x,y) = F⁻¹{ F{∂W/∂x + i∂W/∂y} / (i·k_x + i·k_y) }
        """
        ny, nx = dx.shape

        # Scale displacements to wavefront slopes
        # slope = displacement * (focal_length / pixel_pitch)
        px = dy * scale
        py = dx * scale
        slope_x = px  # ∂W/∂x
        slope_y = py  # ∂W/∂y

        # Pad arrays to reduce FFT wrap-around
        pad_y = int(ny * (self.config.pad_factor - 1) / 2)
        pad_x = int(nx * (self.config.pad_factor - 1) / 2)
        padded_shape = (ny + 2 * pad_y, nx + 2 * pad_x)

        slope_x_padded = np.pad(
            slope_x, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant"
        )
        slope_y_padded = np.pad(
            slope_y, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant"
        )

        # Create complex slope field: S = ∂W/∂x + i·∂W/∂y
        complex_slope = slope_x_padded + 1j * slope_y_padded

        # Forward FFT of slope
        fft_slope = np.fft.fft2(complex_slope)

        # Create frequency grids
        ky = np.fft.fftfreq(padded_shape[0]).reshape(-1, 1)
        kx = np.fft.fftfreq(padded_shape[1]).reshape(1, -1)

        # Integration in frequency domain
        # Avoid division by zero at DC (k=0)
        denom = 1j * (2 * np.pi * (kx + ky))
        denom[0, 0] = 1.0  # Avoid division by zero; DC component set to zero

        # Apply regularization / Wiener filtering
        if self.config.filter_type == "wiener":
            mag_sq = np.abs(denom) ** 2
            wiener_filter = np.conj(denom) / (mag_sq + self.config.regularization)
            fft_phase = fft_slope * wiener_filter
        elif self.config.filter_type == "lowpass":
            cutoff = self.config.cutoff_freq
            mask = (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)
            denom_safe = np.where(np.abs(denom) < 1e-10, 1e10, denom)
            fft_phase = np.where(
                mask,
                fft_slope / denom_safe,
                0.0,
            )
        else:
            denom_safe = np.where(np.abs(denom) < 1e-10, 1e10, denom)
            fft_phase = fft_slope / denom_safe

        # Inverse FFT to get wavefront
        wavefront_padded = np.fft.ifft2(fft_phase).real

        # Remove padding
        wavefront = wavefront_padded[pad_y : pad_y + ny, pad_x : pad_x + nx]

        # Remove piston (subtract mean)
        mean_val = np.nanmean(wavefront)
        wavefront = wavefront - mean_val

        return wavefront


def reconstruct_from_displacements(
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    num_spots: tuple[int, int],
    config: FourierReconstructionConfig | None = None,
) -> np.ndarray:
    """Convenience function for single-call Fourier reconstruction.

    Args:
        displacement_x: X displacement array (num_spots_y, num_spots_x).
        displacement_y: Y displacement array (num_spots_y, num_spots_x).
        num_spots: (num_spots_x, num_spots_y) tuple from WFS.
        config: Optional reconstruction config.

    Returns:
        Reconstructed wavefront array in radians.
    """
    if config is None:
        config = FourierReconstructionConfig()

    reconstructor = FourierWavefrontReconstructor(
        config=config,
        num_spots=num_spots,
    )
    return reconstructor.reconstruct(displacement_x, displacement_y)


def interpolate_sparse_to_dense(
    sparse_x: np.ndarray,
    sparse_y: np.ndarray,
    resolution: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse subaperture data to a dense grid.

    Uses cubic spline interpolation to map sparse WFS subaperture
    coordinates to a finer regular grid.

    Args:
        sparse_x: Sparse X displacement data (ny, nx).
        sparse_y: Sparse Y displacement data (ny, nx).
        resolution: Target (height, width) for dense grid.
            Defaults to 4× the sparse resolution.

    Returns:
        (dense_x, dense_y) interpolated arrays.
    """
    from scipy.interpolate import RectBivariateSpline

    ny, nx = sparse_x.shape

    if resolution is None:
        target_ny = ny * 4
        target_nx = nx * 4
    else:
        target_ny, target_nx = resolution

    # Original coordinate grid (normalized to [0, 1])
    x_orig = np.linspace(0, 1, nx)
    y_orig = np.linspace(0, 1, ny)

    # Target coordinate grid
    x_dense = np.linspace(0, 1, target_nx)
    y_dense = np.linspace(0, 1, target_ny)

    try:
        spline_x = RectBivariateSpline(y_orig, x_orig, sparse_x, kx=3, ky=3)
        spline_y = RectBivariateSpline(y_orig, x_orig, sparse_y, kx=3, ky=3)

        dense_x = spline_x(y_dense, x_dense)
        dense_y = spline_y(y_dense, x_dense)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Spline interpolation failed ({e}), falling back to bilinear")
        from scipy.interpolate import RegularGridInterpolator

        interp_x = RegularGridInterpolator(
            (y_orig, x_orig), sparse_x, method="linear", bounds_error=False
        )
        interp_y = RegularGridInterpolator(
            (y_orig, x_orig), sparse_y, method="linear", bounds_error=False
        )

        xv, yv = np.meshgrid(x_dense, y_dense)
        dense_x = interp_x(np.column_stack([yv.ravel(), xv.ravel()])).reshape(
            target_ny, target_nx
        )
        dense_y = interp_y(np.column_stack([yv.ravel(), xv.ravel()])).reshape(
            target_ny, target_nx
        )

    return dense_x, dense_y
