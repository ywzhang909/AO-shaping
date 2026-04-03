"""Dataset classes for phase prediction training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import ndimage
from torch import nn
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
        target_size: tuple[int, int] | None = (512, 512),
        normalize_phase: bool = True,
        normalize_images: bool = True,
        use_daheng: bool = True,
        use_miicam: bool = True,
        brightness_normalize: bool = True,
        equalize_histogram: bool = False,
        # Augmentation parameters
        random_shift: bool = True,
        shift_range: int = 20,
        random_blur: bool = True,
        blur_sigma_range: tuple[float, float] = (0.1, 1.5),
    ):
        """
        Args:
            data_dir: Root directory containing sample_XXXX subdirectories.
            target_size: Resize images and phase to (height, width). None for original.
            normalize_phase: Normalize phase to [0, 1].
            normalize_images: Normalize images to [-1, 1].
            use_daheng: Include Daheng camera channel.
            use_miicam: Include MiiCam camera channel.
            brightness_normalize: Normalize brightness across dataset (per-channel z-score).
            equalize_histogram: Apply histogram equalization for contrast enhancement.
            random_shift: Apply random translation augmentation.
            shift_range: Maximum pixels to shift in each direction.
            random_blur: Apply random Gaussian blur augmentation.
            blur_sigma_range: Range (min, max) for random Gaussian blur sigma.
        """
        self.data_dir = Path(data_dir)
        # Use default (512, 512) if target_size is None
        self.target_size = target_size if target_size is not None else (512, 512)
        self.normalize_phase = normalize_phase
        self.normalize_images = normalize_images
        self.use_daheng = use_daheng
        self.use_miicam = use_miicam
        self.brightness_normalize = brightness_normalize
        self.equalize_histogram = equalize_histogram

        # Augmentation parameters
        self.random_shift = random_shift
        self.shift_range = shift_range
        self.random_blur = random_blur
        self.blur_sigma_range = blur_sigma_range

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

        # Pre-compute dataset statistics for brightness normalization
        self._compute_dataset_stats()

    def _find_center(self, img: torch.Tensor) -> tuple[float, float]:
        """Find center of mass for an image tensor.

        Args:
            img: Image tensor (H, W) or (1, H, W)

        Returns:
            Tuple of (row, col) center coordinates
        """
        if img.ndim == 3:
            img = img.squeeze(0)
        # Convert to numpy for scipy
        arr = img.cpu().numpy()
        # Normalize to make bright spots stand out
        arr_norm = arr - arr.min()
        if arr_norm.max() > 0:
            center = ndimage.center_of_mass(arr_norm)
            return float(center[0]), float(center[1])
        # Fallback to image center
        return arr.shape[0] / 2, arr.shape[1] / 2

    def _compute_dataset_stats(self) -> None:
        """Pre-compute mean and std for each camera channel across dataset."""
        self.stats = {
            "daheng": {"mean": None, "std": None},
            "miicam": {"mean": None, "std": None},
        }

        if not self.brightness_normalize:
            return

        # Sample first 30 images to compute statistics
        n_samples = min(30, len(self.samples))
        daheng_values, miicam_values = [], []

        for i in range(n_samples):
            data = torch.load(self.samples[i] / "sample.pt", weights_only=False)
            if self.use_daheng and "daheng" in data:
                daheng_values.append(data["daheng"].float())
            if self.use_miicam and "miicam" in data:
                miicam_values.append(data["miicam"].float())

        if daheng_values and self.use_daheng:
            daheng_cat = torch.cat(daheng_values)
            self.stats["daheng"]["mean"] = daheng_cat.mean().item()
            self.stats["daheng"]["std"] = daheng_cat.std().item()

        if miicam_values and self.use_miicam:
            miicam_cat = torch.cat(miicam_values)
            self.stats["miicam"]["mean"] = miicam_cat.mean().item()
            self.stats["miicam"]["std"] = miicam_cat.std().item()

    def _normalize_brightness(self, img: torch.Tensor, camera: str) -> torch.Tensor:
        """Apply z-score normalization based on dataset statistics.

        Args:
            img: Image tensor (C, H, W) for the specific camera
            camera: 'daheng' or 'miicam'

        Returns:
            Normalized tensor
        """
        if not self.brightness_normalize:
            return img

        camera_stats = self.stats.get(camera, {})
        mean = camera_stats.get("mean")
        std = camera_stats.get("std")

        if mean is None or std is None or std <= 0:
            return img

        # Z-score normalize: (x - mean) / std
        # Then scale to reasonable range for neural network
        normalized = (img - mean) / std
        # Scale to roughly [-1, 1] range based on typical deviations
        normalized = normalized.clamp(-3, 3) / 3

        return normalized

    def _equalize_histogram(self, img: torch.Tensor) -> torch.Tensor:
        """Apply histogram equalization for contrast enhancement.

        Args:
            img: Image tensor (H, W) or (C, H, W)

        Returns:
            Equalized tensor
        """
        # Convert to numpy for skimage
        if img.ndim == 3:
            # Process each channel separately
            result = []
            for c in range(img.shape[0]):
                arr = img[c].cpu().numpy()
                # Simple histogram equalization
                hist, bins = np.histogram(
                    arr.flatten(), 256, range=(arr.min(), arr.max())
                )
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                # Interpolate
                equalized = np.interp(arr.flatten(), bins[:-1], cdf_normalized)
                result.append(torch.from_numpy(equalized.reshape(arr.shape)))
            return torch.stack(result)
        else:
            arr = img.cpu().numpy()
            hist, bins = np.histogram(arr.flatten(), 256, range=(arr.min(), arr.max()))
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            equalized = np.interp(arr.flatten(), bins[:-1], cdf_normalized)
            return torch.from_numpy(equalized.reshape(arr.shape))

    def _apply_augmentations(
        self,
        image: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation to image and phase.

        Args:
            image: Image tensor (C, H, W)
            phase: Phase tensor (1, H, W)

        Returns:
            Augmented (image, phase) tensors
        """
        # Random translation (shift)
        if self.random_shift and self.shift_range > 0:
            shift_x = np.random.randint(-self.shift_range, self.shift_range + 1)
            shift_y = np.random.randint(-self.shift_range, self.shift_range + 1)

            # Apply same shift to all channels of image
            image = self._shift_tensor(image, shift_x, shift_y)
            # Apply same shift to phase
            phase = self._shift_tensor(phase, shift_x, shift_y)

        # Random Gaussian blur
        if self.random_blur:
            sigma = np.random.uniform(
                self.blur_sigma_range[0], self.blur_sigma_range[1]
            )
            image = self._apply_gaussian_blur(image, sigma)
            # Also blur phase slightly to maintain consistency
            phase = self._apply_gaussian_blur(phase, sigma * 0.5)

        return image, phase

    def _shift_tensor(
        self,
        tensor: torch.Tensor,
        shift_x: int,
        shift_y: int,
    ) -> torch.Tensor:
        """Shift a tensor by (shift_x, shift_y) pixels.

        Args:
            tensor: Input tensor (C, H, W) or (1, H, W)
            shift_x: Horizontal shift (positive = right)
            shift_y: Vertical shift (positive = down)

        Returns:
            Shifted tensor with zero-padding
        """
        if shift_x == 0 and shift_y == 0:
            return tensor

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        c, h, w = tensor.shape
        shifted = torch.zeros_like(tensor)

        # Calculate source and destination bounds
        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)

        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)

        if (src_x_end > src_x_start) and (src_y_end > src_y_start):
            shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = tensor[
                :, src_y_start:src_y_end, src_x_start:src_x_end
            ]

        return shifted

    def _apply_gaussian_blur(
        self,
        tensor: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Apply Gaussian blur to a tensor.

        Args:
            tensor: Input tensor (C, H, W) or (1, H, W)
            sigma: Gaussian kernel standard deviation

        Returns:
            Blurred tensor
        """
        if sigma <= 0.01:  # Skip negligible blur
            return tensor

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        c, h, w = tensor.shape

        # Calculate kernel size (odd number, ~6*sigma covers 99.7% of Gaussian)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        # Create 1D Gaussian kernel
        x = (
            torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
            - kernel_size // 2
        )
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        # Apply convolution to each channel
        result = []
        for i in range(c):
            channel = tensor[i : i + 1, :, :]
            # Pad to maintain size
            padded = torch.nn.functional.pad(
                channel,
                (
                    kernel_size // 2,
                    kernel_size // 2,
                    kernel_size // 2,
                    kernel_size // 2,
                ),
                mode="reflect",
            )
            blurred = torch.nn.functional.conv2d(padded, kernel_2d, padding=0)
            result.append(blurred)

        return torch.cat(result, dim=0)

    def _crop_and_resize(
        self,
        img: torch.Tensor,
        center: tuple[float, float],
        crop_size: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Crop around center and resize to target size.

        Args:
            img: Image tensor (H, W) or (1, H, W)
            center: (row, col) center coordinates
            crop_size: Size of square crop around center
            target_size: Target (height, width) for resize

        Returns:
            Cropped and resized tensor
        """
        if img.ndim == 3:
            img = img.squeeze(0)

        h, w = img.shape
        center_row, center_col = center

        # Calculate crop bounds
        half = crop_size // 2
        top = int(max(0, center_row - half))
        bottom = int(min(h, center_row + half))
        left = int(max(0, center_col - half))
        right = int(min(w, center_col + half))

        # Crop
        cropped = img[top:bottom, left:right]

        # If crop is smaller than expected (near edges), pad
        if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
            pad_h = crop_size - cropped.shape[0]
            pad_w = crop_size - cropped.shape[1]
            cropped = torch.nn.functional.pad(
                cropped,
                (
                    max(0, pad_w // 2),
                    max(0, pad_w - pad_w // 2),
                    max(0, pad_h // 2),
                    max(0, pad_h - pad_h // 2),
                ),
                mode="constant",
                value=cropped.min().item() if cropped.numel() > 0 else 0,
            )

        # Resize to target
        resized = (
            torch.nn.functional.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )

        return resized

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]
        data = torch.load(sample_dir / "sample.pt", weights_only=False)

        # Load phase target
        phase = data["phase"]  # (H, W)
        if phase.ndim == 2:
            phase = phase.unsqueeze(0)  # (1, H, W)

        # Determine crop size: use the smaller image dimension
        # Default to 512 or smaller if images are smaller
        default_crop = 512

        # Load camera images with center detection and cropping
        channels = []
        if self.use_daheng and "daheng" in data:
            daheng = data["daheng"]
            if daheng.ndim == 2:
                daheng = daheng.unsqueeze(0)
            # Find center and crop around it
            center_row, center_col = self._find_center(daheng)
            # Use reasonable crop size (512 or smaller)
            crop_size = min(default_crop, daheng.shape[-2], daheng.shape[-1])
            daheng_cropped = self._crop_and_resize(
                daheng, (center_row, center_col), crop_size, self.target_size
            )
            channels.append(daheng_cropped.unsqueeze(0))
        if self.use_miicam and "miicam" in data:
            miicam = data["miicam"]
            if miicam.ndim == 2:
                miicam = miicam.unsqueeze(0)
            # Find center and crop around it
            center_row, center_col = self._find_center(miicam)
            crop_size = min(default_crop, miicam.shape[-2], miicam.shape[-1])
            miicam_cropped = self._crop_and_resize(
                miicam, (center_row, center_col), crop_size, self.target_size
            )
            channels.append(miicam_cropped.unsqueeze(0))

        if not channels:
            raise ValueError(f"No camera data found in {sample_dir}")

        # Concatenate channels - now all channels have same size
        image = torch.cat(channels, dim=0)  # (C, H, W)

        # Resize phase if needed (only if not already resized above)
        if self.target_size is not None:
            phase = torch.nn.functional.interpolate(
                phase.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        # Apply random augmentations (shift + blur)
        image, phase = self._apply_augmentations(image, phase)

        # Apply brightness normalization (z-score) per camera channel
        if self.brightness_normalize:
            # Daheng is channel 0, MiiCam is channel 1
            if self.use_daheng and self.use_miicam:
                daheng_img = image[0:1, :, :]
                miicam_img = image[1:2, :, :]
                daheng_norm = self._normalize_brightness(daheng_img, "daheng")
                miicam_norm = self._normalize_brightness(miicam_img, "miicam")
                image = torch.cat([daheng_norm, miicam_norm], dim=0)
            elif self.use_daheng:
                image = self._normalize_brightness(image, "daheng")
            elif self.use_miicam:
                image = self._normalize_brightness(image, "miicam")

        # Apply histogram equalization if enabled
        if self.equalize_histogram:
            if image.shape[0] == 2:
                # Apply to each channel
                daheng_eq = self._equalize_histogram(image[0])
                miicam_eq = self._equalize_histogram(image[1])
                image = torch.stack([daheng_eq, miicam_eq])
            else:
                image = self._equalize_histogram(image)

        # Normalize images to [-1, 1] (final normalization)
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
    target_size: tuple[int, int] | None = (512, 512),
    num_workers: int = 0,
    use_daheng: bool = True,
    use_miicam: bool = True,
    seed: int = 42,
    # Augmentation parameters
    random_shift: bool = True,
    shift_range: int = 20,
    random_blur: bool = True,
    blur_sigma_range: tuple[float, float] = (0.1, 1.5),
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
        random_shift: Apply random translation augmentation.
        shift_range: Maximum pixels to shift in each direction.
        random_blur: Apply random Gaussian blur augmentation.
        blur_sigma_range: Range (min, max) for random Gaussian blur sigma.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = PhasePredictionDataset(
        data_dir=data_dir,
        target_size=target_size,
        use_daheng=use_daheng,
        use_miicam=use_miicam,
        random_shift=random_shift,
        shift_range=shift_range,
        random_blur=random_blur,
        blur_sigma_range=blur_sigma_range,
    )

    # Train/val split
    # Need to use filtered sample indices, not raw dataset indices
    n = len(dataset.samples)
    all_indices = list(range(n))
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=generator).tolist()
    split = int(n * train_split)
    train_indices = shuffled[:split]
    val_indices = shuffled[split:]

    # Create subsets using filtered indices
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
