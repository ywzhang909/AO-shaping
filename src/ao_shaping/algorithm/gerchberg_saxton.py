"""Gerchberg-Saxton algorithm for hologram generation.

This module implements the classical Gerchberg-Saxton phase retrieval algorithm
with Angular Spectrum Method (ASM) for optical wave propagation.

The algorithm iteratively constrains the amplitude at the source (SLM) plane
and target (far-field) plane to compute the optimal phase pattern for the SLM.

Reference:
    - Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the
      determination of phase from image and diffraction plane pictures.
      Optik, 35, 237-246.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from loguru import logger


@dataclass
class GSResult:
    """Result container for Gerchberg-Saxton algorithm.
    
    Attributes:
        phase: Computed phase pattern for SLM (radians, 0-2π)
        amplitude: Final amplitude at target plane
        error_history: List of error values per iteration
        iterations: Number of iterations performed
        converged: Whether the algorithm converged
    """
    phase: np.ndarray
    amplitude: np.ndarray
    error_history: list[float]
    iterations: int
    converged: bool


def angular_spectrum_propagate(
    field: np.ndarray,
    dx: float,
    z: float,
    wavelength: float,
) -> np.ndarray:
    """Propagate optical field using Angular Spectrum Method (ASM).
    
    The Angular Spectrum Method propagates a complex optical field from one
    plane to another using Fourier optics. It's accurate for near-field and
    far-field propagation.
    
    Args:
        field: Complex field array (2D numpy array)
        dx: Pixel spacing (meters)
        z: Propagation distance (meters, positive=forward, negative=backward)
        wavelength: Light wavelength (meters)
    
    Returns:
        Propagated complex field (same shape as input)
    
    Example:
        >>> # Forward propagate by 10cm
        >>> propagated = angular_spectrum_propagate(field, dx=8e-6, z=0.1, wavelength=633e-9)
    """
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D array, got {field.ndim}D")

    Ny, Nx = field.shape
    k = 2 * np.pi / wavelength

    # Spatial frequencies
    fx = np.fft.fftfreq(Nx, dx)
    fy = np.fft.fftfreq(Ny, dx)
    FX, FY = np.meshgrid(fx, fy)

    # Propagator: H = exp(i * kz * z)
    # kz = sqrt(k^2 - (2π*fx)^2 - (2π*fy)^2)
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    # Evanescent wave filtering (optional but recommended)
    kz_squared = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(kz_squared, 0))  # Clip negative values (evanescent waves)

    # Propagator phase factor
    H = np.exp(1j * kz * z)

    # Handle evanescent waves (damped propagation)
    evanescent_mask = kz_squared < 0
    H[evanescent_mask] = 0

    # FFT → multiply by propagator → IFFT
    F = fft2(field)
    F_propagated = F * ifftshift(H)  # ifftshift aligns with fft2 output

    return ifft2(F_propagated)


def gerchberg_saxton(
    source_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    iterations: int = 50,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    error_threshold: float | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
) -> GSResult:
    """Gerchberg-Saxton algorithm for phase retrieval.
    
    Computes the optimal phase pattern to apply at the source plane (SLM)
    to produce a desired intensity distribution at the target plane.
    
    Algorithm:
        1. Initialize field A at source plane
        2. For each iteration:
           a. Apply source amplitude constraint: B = source_amp * exp(i*phase(A))
           b. Propagate forward to target plane: C = ASM(B, +z)
           c. Apply target amplitude constraint: D = target_amp * exp(i*phase(C))
           d. Propagate backward to source plane: A = ASM(D, -z)
        3. Extract final phase: phase = angle(A)
    
    Args:
        source_amplitude: 2D array, amplitude constraint at SLM plane
                         (typically uniform illumination, shape matches SLM)
        target_amplitude: 2D array, desired amplitude at target plane
                         (square root of target intensity image)
        iterations: Number of GS iterations (default: 50)
        cell_spacing: Pixel size in meters (default: 8e-6 for SLM200)
        distance: Propagation distance in meters (default: 0.1)
        wavelength: Light wavelength in meters (default: 1064e-9 for YAG laser)
        error_threshold: Optional convergence threshold (mean squared error)
        progress_callback: Optional callback function(iteration, error) for monitoring
    
    Returns:
        GSResult containing computed phase, amplitude, error history, and convergence info
    
    Raises:
        ValueError: If input arrays have wrong dimensions or parameters are invalid
    
    Example:
        >>> # Create target amplitude from image
        >>> target_img = np.loadtxt('target_pattern.csv', delimiter=',')
        >>> target_amp = np.sqrt(target_img / target_img.max())  # Normalize and sqrt
        >>> 
        >>> # Uniform source amplitude
        >>> source_amp = np.ones((1200, 1920))
        >>> 
        >>> # Run GS algorithm
        >>> result = gerchberg_saxton(
        ...     source_amplitude=source_amp,
        ...     target_amplitude=target_amp,
        ...     iterations=100,
        ...     cell_spacing=8e-6,
        ...     distance=0.15,
        ...     wavelength=1064e-9,
        ... )
        >>> 
        >>> # Use computed phase
        >>> slm_phase = result.phase  # Radians, 0-2π
    """
    # Validate inputs
    if source_amplitude.ndim != 2 or target_amplitude.ndim != 2:
        raise ValueError("Input amplitudes must be 2D arrays")

    if source_amplitude.shape != target_amplitude.shape:
        raise ValueError(
            f"Source and target shapes must match: "
            f"{source_amplitude.shape} vs {target_amplitude.shape}"
        )

    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    if cell_spacing <= 0 or distance <= 0 or wavelength <= 0:
        raise ValueError("Physical parameters must be positive")

    logger.info(
        f"Starting Gerchberg-Saxton algorithm: "
        f"iterations={iterations}, distance={distance*1000:.1f}mm, "
        f"λ={wavelength*1e9:.0f}nm, pixel={cell_spacing*1e6:.1f}µm"
    )

    Ny, Nx = source_amplitude.shape

    # Initialize field A with target back-propagated to source plane
    # This gives a better starting point than random initialization
    logger.debug("Initializing field with back-propagated target")
    A = angular_spectrum_propagate(
        target_amplitude.astype(np.complex128),
        cell_spacing,
        -distance,  # Backward propagation
        wavelength,
    )

    error_history = []

    # Main GS iteration loop
    for i in range(iterations):
        # Step 1: Apply source plane amplitude constraint
        # B = source_amplitude * exp(i * phase(A))
        phase_A = np.angle(A)
        B = source_amplitude * np.exp(1j * phase_A)

        # Step 2: Forward propagate to target plane
        C = angular_spectrum_propagate(B, cell_spacing, distance, wavelength)

        # Step 3: Apply target plane amplitude constraint
        # D = target_amplitude * exp(i * phase(C))
        phase_C = np.angle(C)
        D = target_amplitude * np.exp(1j * phase_C)

        # Step 4: Backward propagate to source plane
        A = angular_spectrum_propagate(D, cell_spacing, -distance, wavelength)

        # Calculate error (mean squared error between |C| and target)
        amplitude_C = np.abs(C)
        mse = np.mean((amplitude_C - target_amplitude) ** 2)
        error_history.append(float(mse))

        # Progress callback
        if progress_callback is not None:
            progress_callback(i, float(mse))

        # Log progress every 10 iterations
        if (i + 1) % 10 == 0 or i == 0:
            logger.debug(f"Iteration {i+1}/{iterations}, MSE={mse:.6f}")

        # Check convergence
        if error_threshold is not None and mse < error_threshold:
            logger.info(f"Converged at iteration {i+1} with MSE={mse:.6f}")
            break

    # Extract final results
    final_phase = np.angle(A)

    # Forward propagate one more time to get target plane amplitude
    final_B = source_amplitude * np.exp(1j * final_phase)
    final_C = angular_spectrum_propagate(final_B, cell_spacing, distance, wavelength)
    final_amplitude = np.abs(final_C)

    # Check if we converged
    converged = error_threshold is not None and error_history[-1] < error_threshold

    logger.info(
        f"GS algorithm completed: final MSE={error_history[-1]:.6f}, "
        f"converged={converged}"
    )

    return GSResult(
        phase=final_phase,
        amplitude=final_amplitude,
        error_history=error_history,
        iterations=len(error_history),
        converged=converged,
    )


def adaptive_gerchberg_saxton(
    source_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    measured_amplitude_callback: Callable[[np.ndarray], np.ndarray],
    outer_iterations: int = 5,
    inner_iterations: int = 30,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    feedback_weight: float = 0.3,
) -> GSResult:
    """Adaptive Gerchberg-Saxton with experimental feedback.
    
    This variant incorporates actual measured amplitude from the experimental
    setup to refine the phase pattern iteratively. It's useful when the
    theoretical model doesn't perfectly match reality.
    
    Args:
        source_amplitude: Amplitude constraint at SLM plane
        target_amplitude: Desired amplitude at target plane
        measured_amplitude_callback: Function that takes a phase pattern,
            displays it on SLM, captures image with CCD, and returns
            measured amplitude (square root of intensity)
        outer_iterations: Number of adaptive feedback loops
        inner_iterations: Number of GS iterations per feedback loop
        cell_spacing: Pixel spacing in meters
        distance: Propagation distance in meters
        wavelength: Light wavelength in meters
        feedback_weight: Weight for blending measured vs simulated (0-1)
    
    Returns:
        GSResult with final computed phase
    
    Example:
        >>> def capture_amplitude(phase_pattern):
        ...     slm.display_phase(phase_pattern)
        ...     img = camera.get_image()
        ...     return np.sqrt(img)  # Amplitude from intensity
        >>> 
        >>> result = adaptive_gerchberg_saxton(
        ...     source_amplitude,
        ...     target_amplitude,
        ...     capture_amplitude,
        ...     outer_iterations=5,
        ...     inner_iterations=20,
        ... )
    """
    logger.info(f"Starting adaptive GS: {outer_iterations} outer loops")

    # Start with standard GS
    result = gerchberg_saxton(
        source_amplitude,
        target_amplitude,
        iterations=inner_iterations,
        cell_spacing=cell_spacing,
        distance=distance,
        wavelength=wavelength,
    )

    current_phase = result.phase

    for outer_i in range(outer_iterations):
        logger.info(f"Adaptive iteration {outer_i+1}/{outer_iterations}")

        # Get measured amplitude from experiment
        measured_amp = measured_amplitude_callback(current_phase)

        # Blend target with measured (feedback)
        # This allows the algorithm to adapt to real-world imperfections
        blended_target = (
            (1 - feedback_weight) * target_amplitude +
            feedback_weight * measured_amp * target_amplitude / (measured_amp + 1e-10)
        )

        # Run GS with blended target
        result = gerchberg_saxton(
            source_amplitude,
            blended_target,
            iterations=inner_iterations,
            cell_spacing=cell_spacing,
            distance=distance,
            wavelength=wavelength,
        )

        current_phase = result.phase

    return result


def calculate_reconstruction_error(
    computed_phase: np.ndarray,
    source_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
) -> dict[str, float]:
    """Calculate various error metrics for GS reconstruction quality.
    
    Args:
        computed_phase: Phase pattern computed by GS algorithm
        source_amplitude: Source plane amplitude constraint
        target_amplitude: Target plane amplitude constraint
        cell_spacing: Pixel spacing in meters
        distance: Propagation distance in meters
        wavelength: Light wavelength in meters
    
    Returns:
        Dictionary with error metrics:
            - mse: Mean squared error
            - nmse: Normalized MSE
            - correlation: Correlation coefficient
            - efficiency: Optical efficiency
    """
    # Propagate computed phase to target plane
    field = source_amplitude * np.exp(1j * computed_phase)
    propagated = angular_spectrum_propagate(field, cell_spacing, distance, wavelength)
    computed_amplitude = np.abs(propagated)

    # Normalize for comparison
    target_norm = target_amplitude / (target_amplitude.max() + 1e-10)
    computed_norm = computed_amplitude / (computed_amplitude.max() + 1e-10)

    # MSE
    mse = np.mean((computed_norm - target_norm) ** 2)

    # Normalized MSE
    nmse = mse / (np.mean(target_norm**2) + 1e-10)

    # Correlation coefficient
    correlation = np.corrcoef(
        computed_norm.flatten(),
        target_norm.flatten()
    )[0, 1]

    # Optical efficiency (energy in target region / total energy)
    efficiency = np.sum(computed_amplitude**2) / (np.sum(source_amplitude**2) + 1e-10)

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "correlation": float(correlation),
        "efficiency": float(efficiency),
    }
