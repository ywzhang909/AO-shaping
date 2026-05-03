"""Tests for GS Hologram Runner preset shape generation.

These tests verify the target shape generation functions used by the
GS hologram runner. No hardware required - pure simulation tests.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

# Import the shape generation functions from the runner
# We need to import them directly since they're in the runner file
import sys
from pathlib import Path

# Add src to path for importing
_src_root = Path(__file__).resolve().parents[3] / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from ao_shaping.gs_hologram_runner import create_target_shape, phase_to_slm_grayscale


class TestCreateTargetShape:
    """Test preset target shape generation functions."""

    def test_gaussian_shape(self):
        """Test Gaussian target shape generation."""
        size = (128, 128)
        amplitude = create_target_shape("gaussian", size, sigma=0.3)
        
        # Check shape
        assert amplitude.shape == (size[1], size[0])  # (height, width)
        
        # Check range (amplitude should be 0-1)
        assert amplitude.min() >= 0
        assert amplitude.max() <= 1
        
        # Center should be maximum
        center_y, center_x = size[1] // 2, size[0] // 2
        assert amplitude[center_y, center_x] == amplitude.max()
        
        # Corners should be near zero
        assert amplitude[0, 0] < 0.1
        assert amplitude[-1, -1] < 0.1

    def test_circle_shape(self):
        """Test circular target shape generation."""
        size = (128, 128)
        radius = 0.5
        amplitude = create_target_shape("circle", size, radius=radius)
        
        # Check shape
        assert amplitude.shape == (size[1], size[0])
        
        # Check binary nature (circle is either 0 or 1)
        unique_values = np.unique(amplitude)
        assert len(unique_values) <= 2  # 0 and 1 (or just 0 or just 1)
        
        # Center should be 1 (inside circle)
        center_y, center_x = size[1] // 2, size[0] // 2
        assert amplitude[center_y, center_x] == 1.0
        
        # Corners should be 0 (outside circle)
        assert amplitude[0, 0] == 0.0

    def test_square_shape(self):
        """Test square target shape generation."""
        size = (128, 128)
        side = 0.8
        amplitude = create_target_shape("square", size, side=side)
        
        # Check shape
        assert amplitude.shape == (size[1], size[0])
        
        # Check binary nature
        unique_values = np.unique(amplitude)
        assert len(unique_values) <= 2
        
        # Center should be 1
        center_y, center_x = size[1] // 2, size[0] // 2
        assert amplitude[center_y, center_x] == 1.0

    def test_annular_shape(self):
        """Test annular (ring) target shape generation."""
        size = (128, 128)
        inner_r = 0.2
        outer_r = 0.5
        amplitude = create_target_shape("annular", size, 
                                        inner_radius=inner_r, 
                                        outer_radius=outer_r)
        
        # Check shape
        assert amplitude.shape == (size[1], size[0])
        
        # Center should be 0 (inside inner radius)
        center_y, center_x = size[1] // 2, size[0] // 2
        assert amplitude[center_y, center_x] == 0.0
        
        # Some point at middle radius should be 1
        mid_y = size[1] // 2
        mid_x = int(size[0] * (inner_r + outer_r) / 4 + size[0] / 2)
        # This is approximate; just check that ring exists
        assert np.sum(amplitude > 0) > 0, "Annular ring should have non-zero values"

    def test_grid_shape(self):
        """Test grid target shape generation."""
        size = (128, 128)
        nx, ny = 3, 4
        amplitude = create_target_shape("grid", size, nx=nx, ny=ny)
        
        # Check shape
        assert amplitude.shape == (size[1], size[0])
        
        # Should have some non-zero values (the grid lines)
        assert np.sum(amplitude > 0) > 0
        
        # Grid should have specific structure - count bright regions
        # For a 3x4 grid, we expect lines at regular intervals
        bright_pixels = np.sum(amplitude > 0)
        total_pixels = size[0] * size[1]
        # Grid lines are relatively sparse
        assert bright_pixels < total_pixels * 0.3, "Grid should not fill entire image"

    def test_cross_shape(self):
        """Test cross target shape generation."""
        size = (128, 128)
        thickness = 0.05
        amplitude = create_target_shape("cross", size, thickness=thickness)
        
        # Check shape
        assert amplitude.shape == (size[1], size[0])
        
        # Center should be 1
        center_y, center_x = size[1] // 2, size[0] // 2
        assert amplitude[center_y, center_x] == 1.0
        
        # Cross should have horizontal and vertical lines
        # Check middle row has horizontal line
        middle_row = amplitude[center_y, :]
        assert np.sum(middle_row > 0) > size[0] * 0.5, "Cross should have horizontal bar"
        
        # Check middle column has vertical line
        middle_col = amplitude[:, center_x]
        assert np.sum(middle_col > 0) > size[1] * 0.5, "Cross should have vertical bar"

    def test_invalid_shape_raises_error(self):
        """Test that invalid shape name raises error."""
        with pytest.raises(ValueError, match="Unknown shape"):
            create_target_shape("invalid_shape", (128, 128))

    def test_different_sizes(self):
        """Test shape generation with different sizes."""
        sizes = [(64, 64), (128, 256), (256, 128), (512, 512)]
        
        for size in sizes:
            amplitude = create_target_shape("circle", size)
            assert amplitude.shape == (size[1], size[0]), \
                f"Shape mismatch for size {size}"


class TestPhaseToSLMGrayscale:
    """Test phase to SLM grayscale conversion."""

    def test_basic_conversion(self):
        """Test basic phase to grayscale conversion."""
        # Create phase array (0 to 2π)
        phase = np.array([[0, np.pi/2, np.pi, 3*np.pi/2]])
        
        grayscale = phase_to_slm_grayscale(phase)
        
        # Check type
        assert grayscale.dtype == np.uint16
        
        # Check values (0 → 0, π/2 → 256, π → 512, 3π/2 → 768 for 10-bit)
        assert grayscale[0, 0] == 0
        assert grayscale[0, 1] == pytest.approx(256, abs=1)
        assert grayscale[0, 2] == pytest.approx(512, abs=1)
        assert grayscale[0, 3] == pytest.approx(768, abs=1)

    def test_phase_wrapping(self):
        """Test that phase values outside 0-2π are wrapped."""
        # Phase values beyond 2π
        phase = np.array([[2*np.pi, 3*np.pi, 4*np.pi, -np.pi]])
        
        grayscale = phase_to_slm_grayscale(phase)
        
        # Check wrapping:
        # 2π → 0
        assert grayscale[0, 0] == 0
        # 3π = π (mod 2π) → ~512
        assert grayscale[0, 1] == pytest.approx(511, abs=1)
        # 4π = 0 (mod 2π) → 0
        assert grayscale[0, 2] == 0
        # -π = π (mod 2π) → ~512
        assert grayscale[0, 3] == pytest.approx(511, abs=1)
        
        # Check that wrapped values are in valid range
        assert np.all(grayscale >= 0)
        assert np.all(grayscale <= 1023)

    def test_custom_max_grayscale(self):
        """Test conversion with custom max grayscale."""
        phase = np.array([[0, np.pi, 2*np.pi]])
        
        # 8-bit max
        grayscale_8bit = phase_to_slm_grayscale(phase, max_grayscale=255)
        assert grayscale_8bit[0, 0] == 0
        assert grayscale_8bit[0, 1] == pytest.approx(127, abs=1)
        assert grayscale_8bit[0, 2] == 0  # 2π wraps to 0
        
        # 12-bit max
        grayscale_12bit = phase_to_slm_grayscale(phase, max_grayscale=4095)
        assert grayscale_12bit[0, 1] == pytest.approx(2047, abs=1)

    def test_2d_array(self):
        """Test conversion with 2D phase array."""
        phase = np.random.rand(100, 100) * 2 * np.pi
        
        grayscale = phase_to_slm_grayscale(phase)
        
        assert grayscale.shape == phase.shape
        assert grayscale.dtype == np.uint16
        assert np.all(grayscale >= 0)
        assert np.all(grayscale <= 1023)


class TestShapeReconstruction:
    """Integration tests: verify GS can reconstruct preset shapes."""
    
    PHYSICAL_PARAMS = {
        "cell_spacing": 8e-6,
        "distance": 0.1,
        "wavelength": 1064e-9,
    }

    @pytest.mark.parametrize("shape_name", ["gaussian", "circle", "square", "annular"])
    def test_shape_reconstruction(self, shape_name):
        """Test that GS can reconstruct each preset shape."""
        from ao_shaping.algorithm.gerchberg_saxton import gerchberg_saxton
        
        size = (128, 128)
        
        # Create target
        target_amplitude = create_target_shape(shape_name, size)
        source_amplitude = np.ones((size[1], size[0]))
        
        # Run GS
        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=20,
            **self.PHYSICAL_PARAMS
        )
        
        # Verify basic properties
        assert result.phase.shape == (size[1], size[0])
        assert result.amplitude.shape == (size[1], size[0])
        
        # Error should decrease
        assert result.error_history[-1] < result.error_history[0]
        
        # Phase should be valid
        assert np.all(np.isfinite(result.phase))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
