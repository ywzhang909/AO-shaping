"""
Demo script for GS algorithm visualization.

This script demonstrates how to use the GS visualization utilities
to generate animated GIFs of the iteration process.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
src_root = Path(__file__).resolve().parent / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from ao_shaping.utils.gs_visualization import gerchberg_saxton_with_visualization


def create_target(shape: str, size: tuple = (128, 128)) -> np.ndarray:
    """Create a target amplitude distribution."""
    height, width = size
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    Y, X = np.meshgrid(y, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    
    if shape == "gaussian":
        intensity = np.exp(-R**2 / (2 * 0.3**2))
    elif shape == "circle":
        intensity = (R < 0.5).astype(float)
    elif shape == "square":
        intensity = ((np.abs(X) < 0.4) & (np.abs(Y) < 0.4)).astype(float)
    elif shape == "annular":
        intensity = ((R > 0.2) & (R < 0.5)).astype(float)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    return np.sqrt(intensity)


def main():
    """Run GS visualization demo."""
    print("=" * 60)
    print("Gerchberg-Saxton Visualization Demo")
    print("=" * 60)
    
    # Parameters
    size = (128, 128)
    output_dir = Path("logs/gs_demo")
    
    # Create source (uniform illumination)
    source = np.ones(size)
    
    # Run for each shape
    shapes = ["gaussian", "circle", "square"]
    
    for shape in shapes:
        print(f"\nProcessing {shape} shape...")
        
        # Create target
        target = create_target(shape, size)
        
        # Run GS with visualization
        result = gerchberg_saxton_with_visualization(
            source_amplitude=source,
            target_amplitude=target,
            iterations=30,
            cell_spacing=8e-6,
            distance=0.1,
            wavelength=1064e-9,
            output_dir=output_dir,
            save_animation=True,
            animation_fps=5,
            skip_frames=2,
        )
        
        print(f"  ✓ Animation saved to: {result.animation_path}")
        print(f"  ✓ Final MSE: {result.error_history[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Check the GIFs in: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
