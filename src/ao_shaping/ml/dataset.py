"""Dataset classes for phase prediction training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PhasePredictionDataset(Dataset):
    """Dataset for dual-camera phase prediction.

    Expects data saved by slm_phase_capture.py with consolidated .pt files.

    Directory structure:
        data_dir/
        ├── sample_0000/
        │   ├── sample.pt          # {"phase": tensor, "daheng": tensor, "miicam": tensor}
        │   └── metadata.json
        ├── sample_0001/
        │   └── ...
        └── global_metadata.json
    """

    def __init__(
        self,
        data_dir: str | Path,
        target_size: tuple[int, int] | None = None,
        normalize_phase: bool = True,
        normalize_images: bool = True,
        use_daheng: bool = True,
        use_miicam: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing sample_XXXX subdirectories.
            target_size: Resize images and phase to (height, width). None for original.
            normalize_phase: Normalize phase to [0, 1].
            normalize_images: Normalize images to [-1, 1].
            use_daheng: Include Daheng camera channel.
            use_miicam: Include MiiCam camera channel.
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize_phase = normalize_phase
        self.normalize_images = normalize_images
        self.use_daheng = use_daheng
        self.use_miicam = use_miicam

        # Discover samples
        self.sample_dirs = sorted(self.data_dir.glob("sample_*"))
        self.samples = [d for d in self.sample_dirs if (d / "sample.pt").exists()]

        if not self.samples:
            raise ValueError(
                f"No samples found in {data_dir}. Expected sample_XXXX/sample.pt files."
            )

        # Load global metadata
        self.global_meta = {}
        meta_path = self.data_dir / "global_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.global_meta = json.load(f)

        self.in_channels = int(use_daheng) + int(use_miicam)
        if self.in_channels == 0:
            raise ValueError("At least one camera must be enabled.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]
        data = torch.load(sample_dir / "sample.pt", weights_only=False)

        # Load phase target
        phase = data["phase"]  # (H, W)
        if phase.ndim == 2:
            phase = phase.unsqueeze(0)  # (1, H, W)

        # Load camera images
        channels = []
        if self.use_daheng and "daheng" in data:
            daheng = data["daheng"]
            if daheng.ndim == 2:
                daheng = daheng.unsqueeze(0)
            channels.append(daheng)
        if self.use_miicam and "miicam" in data:
            miicam = data["miicam"]
            if miicam.ndim == 2:
                miicam = miicam.unsqueeze(0)
            channels.append(miicam)

        if not channels:
            raise ValueError(f"No camera data found in {sample_dir}")

        # Concatenate channels
        image = torch.cat(channels, dim=0)  # (C, H, W)

        # Resize if needed
        if self.target_size is not None:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            phase = torch.nn.functional.interpolate(
                phase.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        # Normalize images to [-1, 1]
        if self.normalize_images:
            # Per-channel min-max normalization
            img_min = image.amin(dim=(1, 2), keepdim=True)
            img_max = image.amax(dim=(1, 2), keepdim=True)
            img_range = img_max - img_min
            img_range = torch.where(
                img_range == 0, torch.ones_like(img_range), img_range
            )
            image = 2.0 * (image - img_min) / img_range - 1.0

        # Normalize phase to [0, 1]
        if self.normalize_phase:
            phase_max = phase.amax()
            if phase_max > 0:
                phase = phase / phase_max

        return {
            "image": image,  # (C, H, W)
            "phase": phase,  # (1, H, W)
            "sample_idx": data.get("sample_idx", idx),
            "phase_type": data.get("phase_type", "unknown"),
            "path": str(sample_dir),
        }

    @property
    def phase_shape(self) -> tuple[int, int] | None:
        """Infer phase shape from first sample."""
        if not self.samples:
            return None
        data = torch.load(self.samples[0] / "sample.pt", weights_only=False)
        phase = data["phase"]
        if self.target_size:
            return self.target_size
        return tuple(phase.shape[-2:])


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    train_split: float = 0.8,
    target_size: tuple[int, int] | None = None,
    num_workers: int = 0,
    use_daheng: bool = True,
    use_miicam: bool = True,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val dataloaders from a dataset directory.

    Args:
        data_dir: Root directory with sample_XXXX subdirectories.
        batch_size: Batch size for both loaders.
        train_split: Fraction of data for training.
        target_size: Resize to (height, width).
        num_workers: DataLoader workers.
        use_daheng: Include Daheng camera.
        use_miicam: Include MiiCam camera.
        seed: Random seed for split.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = PhasePredictionDataset(
        data_dir=data_dir,
        target_size=target_size,
        use_daheng=use_daheng,
        use_miicam=use_miicam,
    )

    # Train/val split
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    split = int(n * train_split)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
