# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, cos, sin, exp, M_PI

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CDTYPE_t

def _compute_propagator(
    int ny,
    int nx,
    DTYPE_t dx,
    DTYPE_t z,
    DTYPE_t wavelength
):
    """Precompute ASM propagator (already ifftshift aligned fft2 output) for reuse across calls.
    
    Propagator only depends on (grid shape, pixel spacing, distance, wavelength), independent of input field.
    GS iteration repeatedly calls angular_spectrum_propagate, caching avoids reconstructing meshgrid and exp propagation factor
    (pure scalar waste on multi-million-pixel grids).
    """
    cdef DTYPE_t k = 2 * M_PI / wavelength
    cdef cnp.ndarray[DTYPE_t, ndim=1] fx = np.fft.fftfreq(nx, dx)
    cdef cnp.ndarray[DTYPE_t, ndim=1] fy = np.fft.fftfreq(ny, dx)
    cdef cnp.ndarray[DTYPE_t, ndim=2] FX, FY
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    cdef cnp.ndarray[DTYPE_t, ndim=2] kx = 2 * M_PI * FX
    cdef cnp.ndarray[DTYPE_t, ndim=2] ky = 2 * M_PI * FY
    cdef cnp.ndarray[DTYPE_t, ndim=2] kz_squared = k**2 - kx**2 - ky**2
    cdef cnp.ndarray[DTYPE_t, ndim=2] kz = np.sqrt(np.maximum(kz_squared, 0))
    cdef cnp.ndarray[CDTYPE_t, ndim=2] H = np.exp(1j * kz * z)
    H[kz_squared < 0] = 0  # evanescent waves
    # Use np.fft.ifftshift instead of np.ifftshift
    return np.fft.ifftshift(H)

def angular_spectrum_propagate(
    cnp.ndarray[CDTYPE_t, ndim=2] field,
    DTYPE_t dx,
    DTYPE_t z,
    DTYPE_t wavelength
):
    """Propagate optical field using angular spectrum method (ASM).
    
    Angular spectrum method uses Fourier optics to propagate complex optical field from one plane to another.
    It is exact for both near-field and far-field propagation.
    
    Args:
        field: Complex optical field array (2D numpy array)
        dx: Pixel pitch (meters)
        z: Propagation distance (meters, positive=forward, negative=backward)
        wavelength: Wavelength of light (meters)
    
    Returns:
        Propagated complex optical field (same shape as input)
    """
    cdef int ny = field.shape[0]
    cdef int nx = field.shape[1]
    
    # FFT → multiply by propagator → IFFT
    cdef cnp.ndarray[CDTYPE_t, ndim=2] F = np.fft.fft2(field)
    cdef cnp.ndarray[CDTYPE_t, ndim=2] H = _compute_propagator(ny, nx, dx, z, wavelength)
    cdef cnp.ndarray[CDTYPE_t, ndim=2] F_propagated = F * H
    
    return np.fft.ifft2(F_propagated)

def gerchberg_saxton(
    cnp.ndarray[DTYPE_t, ndim=2] source_amplitude,
    cnp.ndarray[DTYPE_t, ndim=2] target_amplitude,
    int iterations=50,
    DTYPE_t cell_spacing=8e-6,
    DTYPE_t distance=0.1,
    DTYPE_t wavelength=1064e-9,
    object error_threshold=None,  # Changed to object to allow None
    # Note: Callbacks are omitted in Cython version for simplicity
    # In a real implementation, these would need to be handled differently
    int propagation=0  # 0 for asm, 1 for fft
):
    """Gerchberg-Saxton algorithm for phase recovery.
    
    Computes the optimal phase pattern to apply to the source plane (SLM) 
    to produce a desired intensity distribution on the target plane.
    
    Algorithm:
        1. Initialize optical field A on source plane
        2. For each iteration:
           a. Apply source amplitude constraint: B = source_amp * exp(i*phase(A))
           b. Forward propagate to target plane: C = ASM(B, +z)
           c. Apply target amplitude constraint: D = target_amp * exp(i*phase(C))
           d. Backward propagate to source plane: A = ASM(D, -z)
        3. Extract final phase: phase = angle(A)
    
    Args:
        source_amplitude: 2D array, amplitude constraint on source plane
                         (usually uniform illumination, shape matching SLM)
        target_amplitude: 2D array, desired amplitude on target plane
                         (square root of target intensity image)
        iterations: Number of GS iterations (default: 50)
        cell_spacing: Pixel size in meters (SLM200 default 8e-6)
        distance: Propagation distance in meters (default: 0.1)
        wavelength: Wavelength of light in meters (YAG laser default 1064e-9)
        error_threshold: Optional convergence threshold (mean squared error)
        propagation: Propagation model, 0 for asm (angular spectrum method, default) or 
                     1 for fft (single FFT of Fourier plane -- focal plane treated as source plane's Fourier transform).
                     "fft" completely omits the overhead of constructing propagator each iteration, 
                     speed comparable to single pair of FFT/IFFT far-field GS.
    
    Returns:
        tuple of (phase, amplitude, error_history, iterations, converged)
        where:
            phase: Computed phase pattern for SLM (radians, 0-2π)
            amplitude: Final amplitude on target plane
            error_history: List of error values corresponding to each iteration
            iterations: Actual number of iterations performed
            converged: Whether algorithm converged
    
    Raises:
        ValueError: If input arrays have wrong dimensions or parameters invalid
    """
    # Validate input
    if source_amplitude.ndim != 2 or target_amplitude.ndim != 2:
        raise ValueError("Input amplitudes must be 2D arrays")
    
    if source_amplitude.shape[0] != target_amplitude.shape[0] or source_amplitude.shape[1] != target_amplitude.shape[1]:
        raise ValueError(
            f"Source and target shapes must match: "
            f"({source_amplitude.shape[0]}, {source_amplitude.shape[1]}) vs ({target_amplitude.shape[0]}, {target_amplitude.shape[1]})"
        )
    
    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")
    
    if cell_spacing <= 0 or distance <= 0 or wavelength <= 0:
        raise ValueError("Physical parameters must be positive")
    
    if propagation not in (0, 1):
        raise ValueError(f"propagation must be 0 (asm) or 1 (fft), got {propagation}")
    
    # Define all cdef variables at the beginning of the function
    cdef int Ny = source_amplitude.shape[0]
    cdef int Nx = source_amplitude.shape[1]
    cdef cnp.ndarray[CDTYPE_t, ndim=2] A
    cdef cnp.ndarray[DTYPE_t, ndim=1] error_history = np.zeros(iterations, dtype=np.float64)
    cdef int actual_iterations = 0
    cdef int i
    cdef DTYPE_t mse
    cdef cnp.ndarray[DTYPE_t, ndim=2] phase_A
    cdef cnp.ndarray[CDTYPE_t, ndim=2] B
    cdef cnp.ndarray[CDTYPE_t, ndim=2] C
    cdef cnp.ndarray[DTYPE_t, ndim=2] phase_C
    cdef cnp.ndarray[CDTYPE_t, ndim=2] D
    cdef cnp.ndarray[DTYPE_t, ndim=2] amplitude_C
    cdef cnp.ndarray[CDTYPE_t, ndim=2] final_B
    cdef cnp.ndarray[CDTYPE_t, ndim=2] final_C
    cdef cnp.ndarray[DTYPE_t, ndim=2] final_phase
    cdef cnp.ndarray[DTYPE_t, ndim=2] final_amplitude
    cdef int converged
    cdef cnp.ndarray[DTYPE_t, ndim=1] trimmed_error_history
    cdef DTYPE_t error_threshold_val  # To hold the actual threshold value for comparison
    
    # Handle error_threshold - if None, set to a large value so convergence check never triggers
    if error_threshold is None:
        error_threshold_val = 1e10  # Large value
    else:
        error_threshold_val = <DTYPE_t>error_threshold
    
    # Initialize field A with back-propagated target
    # This gives a better starting point than random initialization
    if propagation == 0:  # asm
        A = angular_spectrum_propagate(
            target_amplitude.astype(np.complex128),
            cell_spacing,
            -distance,  # backward propagation
            wavelength
        )
    else:  # fft
        # Fourier transform backward propagation: inverse FFT of target plane
        A = np.fft.ifft2(np.fft.ifftshift(target_amplitude.astype(np.complex128)))
    
    # Main GS iteration loop
    for i in range(iterations):
        actual_iterations = i + 1
        
        # Step 1: Apply source amplitude constraint
        # B = source_amplitude * exp(i * phase(A))
        phase_A = np.angle(A)
        B = source_amplitude * np.exp(1j * phase_A)
        
        if propagation == 0:  # asm
            # Step 2: Forward propagate to target plane
            C = angular_spectrum_propagate(
                B, cell_spacing, distance, wavelength
            )
        else:  # fft
            # Step 2: Forward propagate to Fourier (focal) plane -- single FFT
            C = np.fft.fftshift(np.fft.fft2(B))
        
        # Step 3: Apply target amplitude constraint
        # D = target_amplitude * exp(i * phase(C))
        phase_C = np.angle(C)
        D = target_amplitude * np.exp(1j * phase_C)
        
        if propagation == 0:  # asm
            # Step 4: Backward propagate to source plane
            A = angular_spectrum_propagate(
                D, cell_spacing, -distance, wavelength
            )
        else:  # fft
            # Step 4: Backward propagate to source plane -- single IFFT
            A = np.fft.ifft2(np.fft.ifftshift(D))
        
        # Compute error (|C| vs target amplitude mean squared error)
        amplitude_C = np.abs(C)
        mse = np.mean((amplitude_C - target_amplitude) ** 2)
        error_history[i] = mse
        
        # Check for convergence
        if mse < error_threshold_val:
            break
    
    # Extract final results
    final_phase = np.angle(A)
    
    # Forward propagate once more to get target plane amplitude
    final_B = source_amplitude * np.exp(1j * final_phase)
    if propagation == 0:  # asm
        final_C = angular_spectrum_propagate(
            final_B, cell_spacing, distance, wavelength
        )
    else:  # fft
        final_C = np.fft.fftshift(np.fft.fft2(final_B))
    final_amplitude = np.abs(final_C)
    
    # Check if converged
    converged = 1 if (error_threshold is not None and error_history[actual_iterations-1] < error_threshold_val) else 0
    
    # Resize error_history to actual size
    trimmed_error_history = error_history[:actual_iterations]
    
    return (
        final_phase,
        final_amplitude,
        trimmed_error_history,
        actual_iterations,
        converged
    )