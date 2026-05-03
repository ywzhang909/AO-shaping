"""Visualization utilities for Gerchberg-Saxton algorithm.

Provides functions to render GS iteration process as animated GIFs,
showing phase pattern evolution and intensity reconstruction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Try to import PIL for GIF creation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from loguru import logger


def create_gs_iteration_frame(
    phase: np.ndarray,
    target_amplitude: np.ndarray,
    reconstructed_amplitude: np.ndarray,
    iteration: int,
    total_iterations: int,
    error: float,
    figsize: Tuple[int, int] = (12, 4),
) -> np.ndarray:
    """Create a single frame showing GS iteration state.
    
    Args:
        phase: Current phase pattern (radians)
        target_amplitude: Target amplitude distribution
        reconstructed_amplitude: Current reconstructed amplitude
        iteration: Current iteration number
        total_iterations: Total number of iterations
        error: Current error value
        figsize: Figure size (width, height) in inches
    
    Returns:
        RGB array of the rendered frame
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Phase pattern
    ax = axes[0]
    phase_display = np.mod(phase, 2 * np.pi)
    im = ax.imshow(phase_display, cmap='hsv', vmin=0, vmax=2*np.pi)
    ax.set_title(f'Phase Pattern\nIter {iteration}/{total_iterations}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 2: Target amplitude
    ax = axes[1]
    im = ax.imshow(target_amplitude, cmap='hot')
    ax.set_title('Target Amplitude')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 3: Reconstructed amplitude
    ax = axes[2]
    im = ax.imshow(reconstructed_amplitude, cmap='hot')
    ax.set_title(f'Reconstructed\nMSE: {error:.6f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Render to RGB array using buffer_rgba
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())
    ncols, nrows = fig.canvas.get_width_height()
    buf = buf.reshape(nrows, ncols, 4)  # RGBA
    
    # Convert RGBA to RGB
    buf_rgb = buf[:, :, :3]
    
    plt.close(fig)
    
    return buf_rgb


def save_frames_as_gif(
    frames: List[np.ndarray],
    output_path: Path,
    duration: float = 200.0,
    loop: int = 0,
) -> None:
    """Save list of frames as animated GIF.
    
    Args:
        frames: List of RGB arrays (H, W, 3)
        output_path: Output file path
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for GIF creation. "
                         "Install with: pip install Pillow")
    
    if not frames:
        logger.warning("No frames to save")
        return
    
    # Convert numpy arrays to PIL Images
    pil_images = []
    for i, frame in enumerate(frames):
        # Ensure contiguous array
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        pil_img = Image.fromarray(frame)
        pil_images.append(pil_img)
        
        if (i + 1) % 10 == 0:
            logger.debug(f"Converted {i+1}/{len(frames)} frames")
    
    # Save as GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    
    logger.info(f"GIF saved to: {output_path}")


def render_gs_animation(
    phase_history: List[np.ndarray],
    target_amplitude: np.ndarray,
    reconstructed_history: List[np.ndarray],
    error_history: List[float],
    output_path: Path,
    frame_duration: float = 200.0,
    skip_frames: int = 1,
) -> None:
    """Render GS algorithm animation as GIF.
    
    Args:
        phase_history: List of phase patterns from each iteration
        target_amplitude: Target amplitude distribution
        reconstructed_history: List of reconstructed amplitudes
        error_history: List of error values
        output_path: Output GIF file path
        frame_duration: Frame duration in milliseconds
        skip_frames: Render every Nth frame (for faster processing)
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available, cannot create GIF")
        return
    
    logger.info(f"Rendering GS animation with {len(phase_history)} iterations...")
    
    frames = []
    total_iterations = len(phase_history)
    
    for i in range(0, total_iterations, skip_frames):
        frame = create_gs_iteration_frame(
            phase=phase_history[i],
            target_amplitude=target_amplitude,
            reconstructed_amplitude=reconstructed_history[i],
            iteration=i + 1,
            total_iterations=total_iterations,
            error=error_history[i],
        )
        frames.append(frame)
        
        if (i // skip_frames + 1) % 10 == 0:
            logger.debug(f"Rendered {i+1}/{total_iterations} frames")
    
    # Always include last frame
    if (total_iterations - 1) % skip_frames != 0:
        frame = create_gs_iteration_frame(
            phase=phase_history[-1],
            target_amplitude=target_amplitude,
            reconstructed_amplitude=reconstructed_history[-1],
            iteration=total_iterations,
            total_iterations=total_iterations,
            error=error_history[-1],
        )
        frames.append(frame)
    
    save_frames_as_gif(frames, output_path, duration=frame_duration)
    logger.info(f"Animation saved: {output_path}")


class GSVizCallback:
    """Callback class to collect GS iteration data for visualization."""
    
    def __init__(self, source_amplitude: np.ndarray):
        """Initialize callback.
        
        Args:
            source_amplitude: Source plane amplitude for reconstruction
        """
        self.source_amplitude = source_amplitude
        self.phase_history: List[np.ndarray] = []
        self.amplitude_history: List[np.ndarray] = []
        self.error_history: List[float] = []
    
    def __call__(self, iteration: int, error: float) -> None:
        """Callback function for GS algorithm.
        
        This is called after each GS iteration. We need to compute
        the phase and amplitude from the current state.
        
        Note: This requires access to the current field state.
        For now, we'll record the error and rely on post-hoc reconstruction.
        """
        self.error_history.append(error)
    
    def add_state(self, phase: np.ndarray, amplitude: np.ndarray, error: float) -> None:
        """Manually add a state snapshot.
        
        Args:
            phase: Current phase pattern
            amplitude: Current reconstructed amplitude
            error: Current error value
        """
        self.phase_history.append(phase.copy())
        self.amplitude_history.append(amplitude.copy())
        self.error_history.append(error)
    
    def save_animation(
        self,
        target_amplitude: np.ndarray,
        output_path: Path,
        frame_duration: float = 200.0,
        skip_frames: int = 1,
    ) -> None:
        """Save collected states as animation.
        
        Args:
            target_amplitude: Target amplitude distribution
            output_path: Output GIF file path
            frame_duration: Frame duration in milliseconds
            skip_frames: Render every Nth frame
        """
        if not self.phase_history:
            logger.warning("No states collected, cannot create animation")
            return
        
        render_gs_animation(
            phase_history=self.phase_history,
            target_amplitude=target_amplitude,
            reconstructed_history=self.amplitude_history,
            error_history=self.error_history,
            output_path=output_path,
            frame_duration=frame_duration,
            skip_frames=skip_frames,
        )


def gerchberg_saxton_with_visualization(
    source_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    iterations: int = 50,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    output_dir: Optional[Path] = None,
    save_animation: bool = True,
    animation_fps: int = 5,
    skip_frames: int = 1,
):
    """Run GS algorithm with visualization.
    
    This is a modified version of gerchberg_saxton that collects
    intermediate states for animation.
    
    Args:
        source_amplitude: Source plane amplitude
        target_amplitude: Target plane amplitude
        iterations: Number of iterations
        cell_spacing: Pixel spacing (m)
        distance: Propagation distance (m)
        wavelength: Wavelength (m)
        output_dir: Directory to save outputs (default: logs/gs_viz)
        save_animation: Whether to save GIF animation
        animation_fps: Animation frame rate (frames per second)
        skip_frames: Save every Nth frame
    
    Returns:
        GSResult with additional 'animation_path' attribute
    """
    from ao_shaping.algorithm.gerchberg_saxton import (
        angular_spectrum_propagate,
        GSResult,
    )
    
    if output_dir is None:
        output_dir = Path("logs/gs_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    k = 2 * np.pi / wavelength
    Ny, Nx = source_amplitude.shape
    
    # Initialize with target back-propagated
    A = angular_spectrum_propagate(
        target_amplitude.astype(np.complex128),
        cell_spacing,
        -distance,
        wavelength,
    )
    
    # Collect states
    phase_history = []
    amplitude_history = []
    error_history = []
    
    # Main iteration loop
    for i in range(iterations):
        # Source plane constraint
        phase_A = np.angle(A)
        B = source_amplitude * np.exp(1j * phase_A)
        
        # Forward propagate
        C = angular_spectrum_propagate(B, cell_spacing, distance, wavelength)
        
        # Target plane constraint
        phase_C = np.angle(C)
        D = target_amplitude * np.exp(1j * phase_C)
        
        # Backward propagate
        A = angular_spectrum_propagate(D, cell_spacing, -distance, wavelength)
        
        # Calculate error
        amplitude_C = np.abs(C)
        mse = np.mean((amplitude_C - target_amplitude) ** 2)
        
        # Record state
        if i % skip_frames == 0 or i == iterations - 1:
            phase_history.append(phase_A.copy())
            amplitude_history.append(amplitude_C.copy())
            error_history.append(float(mse))
        
        if (i + 1) % 10 == 0:
            logger.debug(f"Iteration {i+1}/{iterations}, MSE={mse:.6f}")
    
    # Final results
    final_phase = np.angle(A)
    final_B = source_amplitude * np.exp(1j * final_phase)
    final_C = angular_spectrum_propagate(final_B, cell_spacing, distance, wavelength)
    final_amplitude = np.abs(final_C)
    
    # Save animation
    animation_path = None
    if save_animation and PIL_AVAILABLE:
        timestamp = np.datetime64('now').astype(str).replace(':', '-')
        animation_path = output_dir / f"gs_animation_{timestamp}.gif"
        
        frame_duration = 1000 / animation_fps  # Convert fps to ms
        
        render_gs_animation(
            phase_history=phase_history,
            target_amplitude=target_amplitude,
            reconstructed_history=amplitude_history,
            error_history=error_history,
            output_path=animation_path,
            frame_duration=frame_duration,
            skip_frames=1,  # Already skipped during collection
        )
    
    result = GSResult(
        phase=final_phase,
        amplitude=final_amplitude,
        error_history=error_history,
        iterations=len(error_history),
        converged=False,
    )
    
    # Add animation path as attribute
    result.animation_path = animation_path
    
    return result
