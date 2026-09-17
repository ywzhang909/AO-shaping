"""Synthetic beam-shaping dataset and ground-truth phase generation.

Generates ``(source_intensity, target_intensity, gt_phase)`` triples for
training FourierGSNet. The ground-truth phase for each target is computed with
the classical batched FFT-based Gerchberg–Saxton algorithm — the same physics
that FourierGSNet unrolls, making the network learn to *reproduce and refine*
GS behaviour in a single forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Target intensity generation
# --------------------------------------------------------------------------- #

def make_target(
    shape: str,
    size: int,
    grid: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a target far-field intensity mask on a ``(grid, grid)`` grid.

    Supported shapes:

    - ``square``: filled square of side ``size`` px.
    - ``circle``: filled disc of diameter ``size`` px.
    - ``gaussian``: smooth 2-D Gaussian with FWHM ~ ``size`` px.
    - ``ring``: annulus with outer diameter ``size``, inner diameter ``size/2``.
    - ``cross``: two overlapping rectangles (horizontal + vertical).
    - ``checker``: two-by-two checkerboard of ``size/2`` px blocks.

    Args:
        shape: One of the shapes above.
        size: Characteristic size of the target in pixels.
        grid: Grid side length in pixels.
        rng: Optional RNG used for non-deterministic variants.

    Returns:
        Float32 intensity mask ``(grid, grid)`` in ``[0, 1]``.
    """
    size = int(size)
    if size < 4:
        size = 4
    if size > grid - 2:
        size = grid - 2

    g = np.arange(grid, dtype=np.float32)
    yy, xx = np.meshgrid(g - (grid - 1) / 2.0, g - (grid - 1) / 2.0, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)

    if shape == "square":
        half = size / 2.0
        mask = (np.abs(xx) < half) & (np.abs(yy) < half)
    elif shape == "circle":
        mask = r <= size / 2.0
    elif shape == "ring":
        mask = (r <= size / 2.0) & (r >= size / 4.0)
    elif shape == "cross":
        half = size / 2.0
        w = max(2, size // 8)
        mask = (np.abs(yy) < w) | (np.abs(xx) < half)
    elif shape == "checker":
        half = size / 2.0
        block = max(2, int(half / 2.0))
        cell_x = np.floor((xx + half) / (2 * block)).astype(int)
        cell_y = np.floor((yy + half) / (2 * block)).astype(int)
        inside = (np.abs(xx) < half) & (np.abs(yy) < half)
        mask = inside & ((cell_x + cell_y) % 2 == 0)
    elif shape == "gaussian":
        sigma = max(1.0, size / 4.0)
        mask = np.exp(-(r**2) / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown target shape: {shape!r}")

    intensity = mask.astype(np.float32)
    if intensity.max() > 0:
        intensity /= intensity.max()
    return intensity


# --------------------------------------------------------------------------- #
# Batched FFT Gerchberg–Saxton ground-truth phase
# --------------------------------------------------------------------------- #

def compute_gs_phase(
    source_amp: torch.Tensor,
    target_amp: torch.Tensor,
    iterations: int = 60,
) -> torch.Tensor:
    """Run batched classical FFT-GS to obtain a ground-truth SLM phase.

    Args:
        source_amp: Source-plane amplitude ``(B, 1, H, W)``.
        target_amp: Far-field amplitude ``(B, 1, H, W)``.
        iterations: Number of GS iterations (default: 60).

    Returns:
        GT source-plane phase ``(B, 1, H, W)`` radians.
    """
    # Warm start: back-propagated target amplitude.
    phase = torch.angle(
        torch.fft.ifft2(torch.fft.ifftshift(target_amp, dim=(-2, -1)))
    )
    for _ in range(iterations):
        pupil = source_amp * torch.exp(1j * phase)
        far = torch.fft.fftshift(torch.fft.fft2(pupil), dim=(-2, -1))
        far_phase = torch.angle(far)
        far_c = target_amp * torch.exp(1j * far_phase)
        pupil_back = torch.fft.ifft2(
            torch.fft.ifftshift(far_c, dim=(-2, -1))
        )
        phase = torch.angle(pupil_back)
    return phase


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def make_source_intensity(
    grid: int,
    source_type: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a source-plane (SLM) intensity pattern.

    Args:
        grid: Grid side length in pixels.
        source_type: ``"uniform"`` (flat-top) or ``"gaussian"`` (default).
        rng: Optional RNG (unused, kept for API symmetry).

    Returns:
        Float32 intensity ``(grid, grid)`` in ``[0, 1]``.
    """
    g = np.arange(grid, dtype=np.float32)
    yy, xx = np.meshgrid(g - (grid - 1) / 2.0, g - (grid - 1) / 2.0, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)

    if source_type == "uniform":
        # Flat-top with soft edges (super-Gaussian), fraction of grid.
        sigma = 0.42 * grid
        p = 8
        return np.exp(-((r / sigma) ** (2 * p))).astype(np.float32)
    if source_type == "gaussian":
        sigma = 0.28 * grid
        return np.exp(-(r**2) / (2 * sigma**2)).astype(np.float32)
    raise ValueError(f"Unknown source_type: {source_type!r}")


class GSShapingDataset(Dataset):
    """Synthetic FourierGSNet training/eval dataset.

    Each item is ``(source_intensity, target_intensity, gt_phase)``. Targets
    are randomly drawn from a mix of shapes with random sizes, and GT phase is
    precomputed with batched FFT-GS at construction time (cached in memory).
    """

    def __init__(
        self,
        n_samples: int,
        grid: int = 64,
        source_type: str = "gaussian",
        shapes: tuple[str, ...] = ("square", "circle", "gaussian", "ring", "cross"),
        gs_iterations: int = 60,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """Initialize and pre-generate the dataset.

        Args:
            n_samples: Number of samples to pre-generate.
            grid: Grid side length (default: 64).
            source_type: ``"gaussian"`` or ``"uniform"`` source illumination.
            shapes: Pool of target shapes to sample from.
            gs_iterations: FFT-GS iterations for GT phase (default: 60).
            seed: RNG seed for reproducible generation.
            device: Device to store tensors on (``"cpu"`` default).
        """
        super().__init__()
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if grid < 16:
            raise ValueError(f"grid must be >= 16, got {grid}")

        self.n_samples = n_samples
        self.grid = grid
        self.seed = seed
        self.device = device
        rng = np.random.default_rng(seed)

        source_np = make_source_intensity(grid, source_type, rng)
        source_amp_np = np.sqrt(np.maximum(source_np, 0.0))

        self.source_intensity = torch.from_numpy(
            source_np[None, None].astype(np.float32)
        ).repeat(n_samples, 1, 1, 1)
        self.target_intensity = torch.empty(n_samples, 1, grid, grid)
        self.gt_phase = torch.empty(n_samples, 1, grid, grid)

        # Pre-generate targets (batched construction).
        targets_np: list[np.ndarray] = []
        for i in range(n_samples):
            shape = shapes[int(rng.integers(0, len(shapes)))]
            size = int(rng.integers(grid // 8, grid // 2))
            targets_np.append(make_target(shape, size, grid, rng))
        target_stack = np.stack(targets_np)                       # (N, grid, grid)
        self.target_intensity = torch.from_numpy(
            target_stack[:, None].astype(np.float32)
        )

        # Batched GT phase via GS.
        source_amp = torch.from_numpy(
            np.broadcast_to(source_amp_np[None, None], (n_samples, 1, grid, grid)).copy()
        ).float()
        target_amp = torch.sqrt(self.target_intensity + 1e-12)
        self.gt_phase = compute_gs_phase(source_amp, target_amp, gs_iterations)

        if device != "cpu":
            self.source_intensity = self.source_intensity.to(device)
            self.target_intensity = self.target_intensity.to(device)
            self.gt_phase = self.gt_phase.to(device)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(source_intensity, target_intensity, gt_phase)``."""
        return (
            self.source_intensity[idx],
            self.target_intensity[idx],
            self.gt_phase[idx],
        )

    def predict_far_intensity(self, phase: torch.Tensor) -> torch.Tensor:
        """Compute the far-field intensity produced by a phase on the source beam.

        Uses the same physics as the model (FFT with fftshift), so it is a
        valid simulation-side reconstruction.

        Args:
            phase: Source-plane phase ``(B, 1, H, W)``.

        Returns:
            Far-field intensity ``(B, 1, H, W)`` (arbitrary scale).
        """
        source_amp = torch.sqrt(self.source_intensity[0:1] + 1e-12)
        # Broadcast the single source to batch of ``phase``.
        source_amp = source_amp.expand(phase.shape[0], -1, -1, -1)
        field = source_amp * torch.exp(1j * phase)
        far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
        return torch.abs(far) ** 2