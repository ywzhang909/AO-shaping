"""Tests for Gerchberg-Saxton algorithm with GIF animation output.

These tests run GS algorithm and save the iteration process as animated GIFs
to the logs/gs_viz folder for visual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# Ensure src is in path
_src_root = Path(__file__).resolve().parents[3] / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from ao_shaping.utils.gs_visualization import (
    gerchberg_saxton_with_visualization,
    create_gs_iteration_frame,
    save_frames_as_gif,
    GSVizCallback,
)


# Default test parameters
PHYSICAL_PARAMS = {
    "cell_spacing": 8e-6,
    "distance": 0.1,
    "wavelength": 1064e-9,
}

# Output directory for GIFs
LOG_DIR = Path("logs/gs_viz")


def create_test_target(shape: str, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Create a test target amplitude distribution."""
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


class TestGSVizualization:
    """Test GS algorithm with GIF visualization."""
    
    @pytest.mark.parametrize("shape_name", ["gaussian", "circle", "square", "annular"])
    def test_gs_with_gif_output(self, shape_name):
        """Test GS algorithm and save iteration process as GIF."""
        size = (128, 128)
        
        # Create target
        target_amplitude = create_test_target(shape_name, size)
        source_amplitude = np.ones(size)
        
        # Run GS with visualization
        result = gerchberg_saxton_with_visualization(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=30,
            output_dir=LOG_DIR,
            save_animation=True,
            animation_fps=5,
            skip_frames=2,  # Save every 2nd frame
            **PHYSICAL_PARAMS,
        )
        
        # Verify result
        assert result.phase.shape == size
        assert result.amplitude.shape == size
        assert len(result.error_history) > 0
        assert result.error_history[-1] < result.error_history[0]
        
        # Verify GIF was created
        assert result.animation_path is not None
        assert result.animation_path.exists()
        assert result.animation_path.suffix == ".gif"
        
        print(f"\n✓ {shape_name}: GIF saved to {result.animation_path}")
    
    def test_gs_convergence_animation(self):
        """Create animation showing GS convergence over many iterations."""
        size = (128, 128)
        
        # Create circular target
        target_amplitude = create_test_target("circle", size)
        source_amplitude = np.ones(size)
        
        # Run with more iterations
        result = gerchberg_saxton_with_visualization(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=50,
            output_dir=LOG_DIR,
            save_animation=True,
            animation_fps=10,
            skip_frames=1,  # Save all frames
            **PHYSICAL_PARAMS,
        )
        
        # Verify convergence
        initial_error = result.error_history[0]
        final_error = result.error_history[-1]
        improvement = (initial_error - final_error) / initial_error
        
        print(f"\n✓ Convergence test:")
        print(f"  Initial error: {initial_error:.6f}")
        print(f"  Final error: {final_error:.6f}")
        print(f"  Improvement: {improvement:.1%}")
        print(f"  GIF: {result.animation_path}")
        
        assert improvement > 0.1, "Should have at least 10% improvement"
    
    def test_compare_shapes_side_by_side(self):
        """Create side-by-side comparison of different shapes."""
        size = (128, 128)
        shapes = ["gaussian", "circle", "square"]
        
        gif_paths = []
        
        for shape_name in shapes:
            target_amplitude = create_test_target(shape_name, size)
            source_amplitude = np.ones(size)
            
            result = gerchberg_saxton_with_visualization(
                source_amplitude=source_amplitude,
                target_amplitude=target_amplitude,
                iterations=25,
                output_dir=LOG_DIR / "comparison",
                save_animation=True,
                animation_fps=5,
                skip_frames=2,
                **PHYSICAL_PARAMS,
            )
            
            gif_paths.append((shape_name, result.animation_path))
        
        print("\n✓ Side-by-side comparison GIFs:")
        for shape, path in gif_paths:
            print(f"  {shape}: {path}")
    
    def test_high_resolution_animation(self):
        """Test with higher resolution for better visualization."""
        size = (256, 256)
        
        target_amplitude = create_test_target("annular", size)
        source_amplitude = np.ones(size)
        
        result = gerchberg_saxton_with_visualization(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=40,
            output_dir=LOG_DIR / "high_res",
            save_animation=True,
            animation_fps=8,
            skip_frames=2,
            **PHYSICAL_PARAMS,
        )
        
        print(f"\n✓ High resolution (256x256) animation saved to {result.animation_path}")


class TestGSVizCallback:
    """Test the GSVizCallback class for collecting states."""
    
    def test_callback_collects_states(self):
        """Test that callback properly collects iteration states."""
        from ao_shaping.algorithm.gerchberg_saxton import (
            angular_spectrum_propagate,
        )
        
        size = (64, 64)
        target_amplitude = create_test_target("circle", size)
        source_amplitude = np.ones(size)
        
        callback = GSVizCallback(source_amplitude)
        
        # Run a few iterations manually
        A = angular_spectrum_propagate(
            target_amplitude.astype(np.complex128),
            PHYSICAL_PARAMS["cell_spacing"],
            -PHYSICAL_PARAMS["distance"],
            PHYSICAL_PARAMS["wavelength"],
        )
        
        for i in range(5):
            phase_A = np.angle(A)
            B = source_amplitude * np.exp(1j * phase_A)
            C = angular_spectrum_propagate(
                B,
                PHYSICAL_PARAMS["cell_spacing"],
                PHYSICAL_PARAMS["distance"],
                PHYSICAL_PARAMS["wavelength"],
            )
            amplitude_C = np.abs(C)
            mse = np.mean((amplitude_C - target_amplitude) ** 2)
            
            # Add state manually
            callback.add_state(phase_A, amplitude_C, mse)
            
            # Continue iteration
            phase_C = np.angle(C)
            D = target_amplitude * np.exp(1j * phase_C)
            A = angular_spectrum_propagate(
                D,
                PHYSICAL_PARAMS["cell_spacing"],
                -PHYSICAL_PARAMS["distance"],
                PHYSICAL_PARAMS["wavelength"],
            )
        
        # Verify states collected
        assert len(callback.phase_history) == 5
        assert len(callback.amplitude_history) == 5
        assert len(callback.error_history) == 5
        
        # Save animation
        output_path = LOG_DIR / "callback_test.gif"
        callback.save_animation(
            target_amplitude=target_amplitude,
            output_path=output_path,
            frame_duration=300,
        )
        
        assert output_path.exists()
        print(f"\n✓ Callback test animation saved to {output_path}")


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running GS visualization tests...")
    print(f"GIFs will be saved to: {LOG_DIR.absolute()}")
    print("=" * 60)
    
    # Create output directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("\n" + "=" * 60)
    print("Test complete! Check the following directories for GIFs:")
    print(f"  - {LOG_DIR.absolute()}")
    print(f"  - {(LOG_DIR / 'comparison').absolute()}")
    print(f"  - {(LOG_DIR / 'high_res').absolute()}")
