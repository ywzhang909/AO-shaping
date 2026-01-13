"""
Test script for VectorWavePropagator methods
"""
import numpy as np
from src.ao_shaping.sim.devices import VectorWavePropagator, OpticParameters

def test_propagation_methods():
    """Test different propagation methods in VectorWavePropagator"""
    # Create propagator instance
    propagator = VectorWavePropagator(
        N=64,                  # Grid size
        L=100e-6,             # Physical size (100 μm)
        wavelength=1550e-9    # Wavelength (1550 nm)
    )
    
    # Create a simple test phase pattern (e.g., lens phase)
    x = np.linspace(-50e-6, 50e-6, 64)
    y = np.linspace(-50e-6, 50e-6, 64)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    focal_length = 0.05  # 50 mm
    
    # Lens phase pattern
    lens_phase = -np.pi * radius**2 / (1550e-9 * focal_length)
    
    # Create OpticParameters instance with required parameters
    params = OpticParameters(
        _num_elements_x=64,      # Number of elements in x
        _num_elements_y=64,      # Number of elements in y
        period=100e-6 / 64,      # Period (physical size / grid points)
        wavelength=1550e-9,      # Wavelength
        focal_length=focal_length  # Focal length
    )
    
    print("Testing VectorWavePropagator methods...")
    
    # Convert phase to complex field for angular spectrum method
    complex_field = np.exp(1j * lens_phase)
    
    # Test traditional angular spectrum propagation
    try:
        result1 = propagator.propagate(complex_field)
        print(f"✓ Angular spectrum propagation successful, result shape: {result1.shape}")
    except Exception as e:
        print(f"✗ Angular spectrum propagation failed: {e}")
    
    # Test beam transform propagation
    try:
        result2, efficiency = propagator.trans_beam(lens_phase, params, focal_length)
        print(f"✓ Beam transform propagation successful, result shape: {result2.G.shape}, efficiency: {efficiency:.4f}")
    except Exception as e:
        print(f"✗ Beam transform propagation failed: {e}")
    
    # Test unified propagation interface with beam transform method
    try:
        result3, efficiency = propagator.unified_propagate(lens_phase, method='beam_transform', 
                                                          params=params, mon_dist=focal_length)
        print(f"✓ Unified propagation (beam_transform) successful, efficiency: {efficiency:.4f}")
    except Exception as e:
        print(f"✗ Unified propagation (beam_transform) failed: {e}")
    
    # Test unified propagation interface with angular spectrum method
    try:
        result4, _ = propagator.unified_propagate(complex_field, method='angular_spectrum')
        print(f"✓ Unified propagation (angular_spectrum) successful, result shape: {result4.shape}")
    except Exception as e:
        print(f"✗ Unified propagation (angular_spectrum) failed: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_propagation_methods()