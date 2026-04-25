"""Phase prediction dataset for dual-camera phase map prediction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.utils.data import Dataset

# External dependency - keep as absolute import
from ao_shaping.utils.zernike_calc import (
    ZernikeGenerator,
    noll_to_nm,
    zernike_radial,
    calc_n_zernike_terms,
)


def coefficients_to_phase_map(
    coefficients: np.ndarray,
    size: tuple[int, int] = (256, 256),
    pupil_radius: float | None = None,
) -> np.ndarray:
    """Convert Zernike coefficients to 2D phase map.

    Args:
        coefficients: Zernike coefficient array (n_terms,).
        size: Output phase map size (H, W).
        pupil_radius: Pupil radius in pixels. If None, use min(H, W) / 2.

    Returns:
        Phase map (H, W) in radians.
    """
    import numpy as np

    h, w = size
    if pupil_radius is None:
        pupil_radius = min(h, w) / 2 - 2

    # Create coordinate grid
    cy, cx = np.ogrid[:h, :w]
    x = (cx - w / 2) / pupil_radius
    y = (cy - h / 2) / pupil_radius

    # Normalize to unit circle
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Apply circular mask
    mask = rho <= 1.0

    # Generate phase map using ZernikeGenerator
    gen = ZernikeGenerator(resolution=(w, h), radius=pupil_radius)
    gen.set_bits(16)  # Use high precision for phase
    gen.precompute_bases(len(coefficients))
    phase = gen.generate_noll(coefficients.astype(np.float64))

    # Convert from uint16 to radians and apply mask
    phase = phase.astype(np.float64) / (2**16) * 2 * np.pi
    phase = phase * mask

    # Subtract piston (mean within pupil)
    if mask.any():
        phase = phase - np.mean(phase[mask]) * mask

    return phase


def load_zernike_coefficients(csv_path: Path, n_terms: int | None = None) -> np.ndarray:
    """Load Zernike coefficients from CSV file.

    Args:
        csv_path: Path to CSV file with coefficients.
        n_terms: Number of terms to load. If None, load all.

    Returns:
        Array of Zernike coefficients.
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"Empty or invalid CSV: {csv_path}")

    # First row is header, second row is values
    values = [float(v) for v in rows[1]]

    if n_terms is not None:
        values = values[:n_terms]

    return np.array(values, dtype=np.float32)


class PhasePredictionDataset(Dataset):
    """Dataset for dual-camera phase prediction.

    Expects data saved by slm_phase_capture.py with separate .npy and .csv files.

    Directory structure:
        data_dir/
        ├── sample_0000/
        │   ├── daheng_frame.npy      # Focus/far-field image
        │   ├── miicam_frame.npy     # Pupil/near-field image
        │   ├── phase.csv           # Zernike coefficients (for backwards compatibility)
        │   ├── phase.npy           # Phase map (H, W) in radians
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
        cache_data: bool = False,  # Cache loaded data in memory for faster access
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
            cache_data: Cache loaded data in memory for faster access (use with caution for large datasets).
            random_shift: Apply random translation augmentation.
            shift_range: Maximum pixels to shift in each direction.
            random_blur: Apply random Gaussian blur augmentation.
            blur_sigma_range: Range (min, max) for random Gaussian blur sigma.
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size if target_size is not None else (512, 512)
        self.normalize_phase = normalize_phase
        self.normalize_images = normalize_images
        self.use_daheng = use_daheng
        self.use_miicam = use_miicam
        self.brightness_normalize = brightness_normalize
        self.equalize_histogram = equalize_histogram
        self.cache_data = cache_data

        # Augmentation parameters
        self.random_shift = random_shift
        self.shift_range = shift_range
        self.random_blur = random_blur
        self.blur_sigma_range = blur_sigma_range

        # Initialize cache if enabled
        self._cache = {} if cache_data else None

        # Discover samples
        self.sample_dirs = sorted(self.data_dir.glob("sample_*"))
        self.samples = [d for d in self.sample_dirs if self._is_valid_sample(d)]

        if not self.samples:
            raise ValueError(
                f"No valid samples found in {data_dir}. Expected sample_XXXX with .npy files."
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

    def _is_valid_sample(self, sample_dir: Path) -> bool:
        """Check if sample directory has required files."""
        daheng = sample_dir / "daheng_frame.npy"
        miicam = sample_dir / "miicam_frame.npy"
        phase_npy = sample_dir / "phase.npy"
        phase_csv = sample_dir / "phase.csv"  # For backwards compatibility

        if self.use_daheng and not daheng.exists():
            return False
        if self.use_miicam and not miicam.exists():
            return False

        return phase_npy.exists() or phase_csv.exists()

    def _find_center(self, img: torch.Tensor) -> tuple[float, float]:
        """Find center of mass for an image tensor."""
        if img.ndim == 3:
            img = img.squeeze(0)
        arr = img.cpu().numpy()
        arr_norm = arr - arr.min()
        if arr_norm.max() > 0:
            center = ndimage.center_of_mass(arr_norm)
            return float(center[0]), float(center[1])
        return arr.shape[0] / 2, arr.shape[1] / 2

    def _compute_dataset_stats(self) -> None:
        """Pre-compute mean and std for each camera channel across dataset."""
        self.stats = {
            "daheng": {"mean": None, "std": None},
            "miicam": {"mean": None, "std": None},
        }

        if not self.brightness_normalize:
            return

        n_samples = min(30, len(self.samples))
        daheng_values, miicam_values = [], []

        for i in range(n_samples):
            sample_dir = self.samples[i]

            if self.use_daheng:
                daheng_path = sample_dir / "daheng_frame.npy"
                if daheng_path.exists():
                    daheng_np = np.load(daheng_path)
                    daheng_values.append(torch.from_numpy(daheng_np).float())

            if self.use_miicam:
                miicam_path = sample_dir / "miicam_frame.npy"
                if miicam_path.exists():
                    miicam_np = np.load(miicam_path)
                    miicam_values.append(torch.from_numpy(miicam_np).float())

        if daheng_values and self.use_daheng:
            daheng_cat = torch.cat(daheng_values)
            self.stats["daheng"]["mean"] = daheng_cat.mean().item()
            self.stats["daheng"]["std"] = daheng_cat.std().item()

        if miicam_values and self.use_miicam:
            miicam_cat = torch.cat(miicam_values)
            self.stats["miicam"]["mean"] = miicam_cat.mean().item()
            self.stats["miicam"]["std"] = miicam_cat.std().item()

    def _normalize_brightness(self, img: torch.Tensor, camera: str) -> torch.Tensor:
        """Apply z-score normalization based on dataset statistics."""
        if not self.brightness_normalize:
            return img

        camera_stats = self.stats.get(camera, {})
        mean = camera_stats.get("mean")
        std = camera_stats.get("std")

        if mean is None or std is None or std <= 0:
            return img

        normalized = (img - mean) / std
        normalized = normalized.clamp(-3, 3) / 3

        return normalized

    def _equalize_histogram(self, img: torch.Tensor) -> torch.Tensor:
        """Apply histogram equalization for contrast enhancement."""
        if img.ndim == 3:
            result = []
            for c in range(img.shape[0]):
                arr = img[c].cpu().numpy()
                hist, bins = np.histogram(arr.flatten(), 256, range=(arr.min(), arr.max()))
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
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
        """Apply random augmentation to image and phase."""
        if self.random_shift and self.shift_range > 0:
            shift_x = np.random.randint(-self.shift_range, self.shift_range + 1)
            shift_y = np.random.randint(-self.shift_range, self.shift_range + 1)
            image = self._shift_tensor(image, shift_x, shift_y)
            phase = self._shift_tensor(phase, shift_x, shift_y)

        if self.random_blur:
            sigma = np.random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
            image = self._apply_gaussian_blur(image, sigma)
            phase = self._apply_gaussian_blur(phase, sigma * 0.5)

        return image, phase

    def _shift_tensor(
        self,
        tensor: torch.Tensor,
        shift_x: int,
        shift_y: int,
    ) -> torch.Tensor:
        """Shift a tensor by (shift_x, shift_y) pixels."""
        if shift_x == 0 and shift_y == 0:
            return tensor

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        c, h, w = tensor.shape
        shifted = torch.zeros_like(tensor)

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
        """Apply Gaussian blur to a tensor."""
        if sigma <= 0.01:
            return tensor

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        c, h, w = tensor.shape

        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        x = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        result = []
        for i in range(c):
            channel = tensor[i : i + 1, :, :]
            padded = torch.nn.functional.pad(
                channel,
                (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
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
        """Crop around center and resize to target size."""
        if img.ndim == 3:
            img = img.squeeze(0)

        h, w = img.shape
        center_row, center_col = center

        half = crop_size // 2
        top = int(max(0, center_row - half))
        bottom = int(min(h, center_row + half))
        left = int(max(0, center_col - half))
        right = int(min(w, center_col + half))

        cropped = img[top:bottom, left:right]

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
        sample_path = str(self.samples[idx])

        if self.cache_data and sample_path in self._cache:
            cached_result = self._cache[sample_path]
            image = cached_result["image"].clone()
            phase = cached_result["phase"].clone()
        else:
            sample_dir = self.samples[idx]
            channels = []

            if self.use_daheng:
                daheng_path = sample_dir / "daheng_frame.npy"
                if not daheng_path.exists():
                    raise ValueError(f"Daheng frame not found: {daheng_path}")

                daheng_np = np.load(daheng_path)
                daheng = torch.from_numpy(daheng_np).float()
                if daheng.ndim == 2:
                    daheng = daheng.unsqueeze(0)

                center_row, center_col = self._find_center(daheng)
                crop_size = min(512, daheng.shape[-2], daheng.shape[-1])
                daheng_cropped = self._crop_and_resize(
                    daheng, (center_row, center_col), crop_size, self.target_size
                )
                channels.append(daheng_cropped.unsqueeze(0))

            if self.use_miicam:
                miicam_path = sample_dir / "miicam_frame.npy"
                if not miicam_path.exists():
                    raise ValueError(f"MiiCam frame not found: {miicam_path}")

                miicam_np = np.load(miicam_path)
                miicam = torch.from_numpy(miicam_np).float()
                if miicam.ndim == 2:
                    miicam = miicam.unsqueeze(0)

                center_row, center_col = self._find_center(miicam)
                crop_size = min(512, miicam.shape[-2], miicam.shape[-1])
                miicam_cropped = self._crop_and_resize(
                    miicam, (center_row, center_col), crop_size, self.target_size
                )
                channels.append(miicam_cropped.unsqueeze(0))

            if not channels:
                raise ValueError(f"No camera data found in {sample_dir}")

            image = torch.cat(channels, dim=0)

            # Load phase target
            phase_path_npy = sample_dir / "phase.npy"
            phase_path_csv = sample_dir / "phase.csv"

            if phase_path_npy.exists():
                phase_np = np.load(phase_path_npy)
                if phase_np.ndim != 2:
                    raise ValueError(f"Expected 2D phase map, got {phase_np.ndim}D: {phase_path_npy}")
                phase = torch.from_numpy(phase_np).float()
                if phase.ndim == 2:
                    phase = phase.unsqueeze(0)
            elif phase_path_csv.exists():
                coeffs = load_zernike_coefficients(phase_path_csv)
                phase_np = coefficients_to_phase_map(coeffs, size=self.target_size)
                phase = torch.from_numpy(phase_np).float().unsqueeze(0)
            else:
                raise ValueError(f"No phase data found in {sample_dir}")

            if self.cache_data:
                self._cache[sample_path] = {
                    "image": image.clone(),
                    "phase": phase.clone(),
                    "sample_idx": idx,
                    "phase_type": "unknown",
                    "path": sample_path,
                }

        if self.target_size is not None and phase.shape[-2:] != self.target_size:
            phase = torch.nn.functional.interpolate(
                phase.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        image, phase = self._apply_augmentations(image, phase)

        # Brightness normalization
        if self.brightness_normalize:
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

        if self.equalize_histogram:
            if image.shape[0] == 2:
                daheng_eq = self._equalize_histogram(image[0])
                miicam_eq = self._equalize_histogram(image[1])
                image = torch.stack([daheng_eq, miicam_eq])
            else:
                image = self._equalize_histogram(image)

        if self.normalize_images:
            img_min = image.amin(dim=(1, 2), keepdim=True)
            img_max = image.amax(dim=(1, 2), keepdim=True)
            img_range = img_max - img_min
            img_range = torch.where(img_range == 0, torch.ones_like(img_range), img_range)
            image = 2.0 * (image - img_min) / img_range - 1.0

        if self.normalize_phase:
            phase_max = phase.amax()
            if phase_max > 0:
                phase = phase / phase_max

        return {
            "image": image,
            "phase": phase,
            "sample_idx": idx,
            "phase_type": "unknown",
            "path": sample_path,
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
    cache_data: bool = False,
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
        cache_data: Cache loaded data in memory.
        random_shift: Apply random translation augmentation.
        shift_range: Maximum pixels to shift.
        random_blur: Apply random Gaussian blur.
        blur_sigma_range: Range for blur sigma.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = PhasePredictionDataset(
        data_dir=data_dir,
        target_size=target_size,
        use_daheng=use_daheng,
        use_miicam=use_miicam,
        cache_data=cache_data,
        random_shift=random_shift,
        shift_range=shift_range,
        random_blur=random_blur,
        blur_sigma_range=blur_sigma_range,
    )

    n = len(dataset.samples)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=generator).tolist()
    split = int(n * train_split)
    train_indices = shuffled[:split]
    val_indices = shuffled[split:]

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