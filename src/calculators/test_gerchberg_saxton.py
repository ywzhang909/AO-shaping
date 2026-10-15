#!/usr/bin/env python
"""
Test script to verify the Cython implementation of Gerchberg-Saxton algorithm
matches the original Python implementation.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import our local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import both implementations
try:
    import gerchberg_saxton_cython
    CYTHON_AVAILABLE = True
except ImportError as e:
    print(f"Cython module not available: {e}")
    CYTHON_AVAILABLE = False

# Import the original Python implementation
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton as python_gerchberg_saxton

def test_gerchberg_saxton():
    """Test that the Cython implementation matches the Python implementation."""
    if not CYTHON_AVAILABLE:
        print("Skipping test because Cython module is not available")
        return
    
    # Create test data
    np.random.seed(42)  # For reproducibility
    size = 64
    source_amplitude = np.ones((size, size), dtype=np.float64)
    # Create a simple target amplitude (a circle in the center)
    target_amplitude = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    radius = size // 8
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 <= radius**2:
                target_amplitude[i, j] = 1.0
    
    # Normalize the target amplitude
    target_amplitude = target_amplitude / np.sqrt(np.sum(target_amplitude**2))
    
    # Parameters for the algorithm
    iterations = 20
    cell_spacing = 8e-6
    distance = 0.1
    wavelength = 1064e-9
    error_threshold = 1e-6
    
    print("Running Gerchberg-Saxton algorithm comparison...")
    print(f"Array size: {size}x{size}")
    print(f"Iterations: {iterations}")
    
    # Run the Python implementation
    print("Running Python implementation...")
    python_result = python_gerchberg_saxton(
        source_amplitude=source_amplitude,
        target_amplitude=target_amplitude,
        iterations=iterations,
        cell_spacing=cell_spacing,
        distance=distance,
        wavelength=wavelength,
        error_threshold=error_threshold,
        propagation="asm"  # Use angular spectrum method
    )
    
    # Run the Cython implementation
    print("Running Cython implementation...")
    cython_result = gerchberg_saxton_cython.gerchberg_saxton(
        source_amplitude=source_amplitude,
        target_amplitude=target_amplitude,
        iterations=iterations,
        cell_spacing=cell_spacing,
        distance=distance,
        wavelength=wavelength,
        error_threshold=error_threshold,
        propagation=0  # 0 for asm
    )
    
    # Unpack results - handle both tuple and GSResult object
    if hasattr(python_result, 'phase'):
        # It's a GSResult object
        python_phase = python_result.phase
        python_amplitude = python_result.amplitude
        python_error_history = python_result.error_history
        python_iterations = python_result.iterations
        python_converged = python_result.converged
    else:
        # It's a tuple
        python_phase, python_amplitude, python_error_history, python_iterations, python_converged = python_result
    
    # Cython result is always a tuple
    cython_phase, cython_amplitude, cython_error_history, cython_iterations, cython_converged = cython_result
    
    # Compare results
    print("\nComparison Results:")
    print(f"Iterations - Python: {python_iterations}, Cython: {cython_iterations}")
    print(f"Converged - Python: {python_converged}, Cython: {cython_converged}")
    
    # Check if the results are close
    phase_close = np.allclose(python_phase, cython_phase, rtol=1e-10, atol=1e-10)
    amplitude_close = np.allclose(python_amplitude, cython_amplitude, rtol=1e-10, atol=1e-10)
    error_history_close = np.allclose(python_error_history, cython_error_history, rtol=1e-10, atol=1e-10)
    
    print(f"Phase arrays close: {phase_close}")
    print(f"Amplitude arrays close: {amplitude_close}")
    print(f"Error history close: {error_history_close}")
    
    if phase_close and amplitude_close and error_history_close:
        print("\n✓ All tests passed! The Cython implementation matches the Python implementation.")
        return True
    else:
        print("\n✗ Test failed! The implementations do not match.")
        return False

if __name__ == "__main__":
    success = test_gerchberg_saxton()
    sys.exit(0 if success else 1)