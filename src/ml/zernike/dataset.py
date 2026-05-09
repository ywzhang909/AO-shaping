"""Zernike coefficient prediction dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

# External dependency - keep as absolute import
from ao_shaping.utils.zernike_calc import (
    ZernikeGenerator,
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
    h, w = size
    if pupil_radius is None:
        pupil_radius = min(h, w) / 2 - 2

    cy, cx = np.ogrid[:h, :w]
    x = (cx - w / 2) / pupil_radius
    y = (cy - h / 2) / pupil_radius

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    mask = rho <= 1.0

    gen = ZernikeGenerator(resolution=(w, h), radius=pupil_radius)
    gen.set_bits(16)
    gen.precompute_bases(len(coefficients))
    phase = gen.generate_noll(coefficients.astype(np.float64))

    phase = phase.astype(np.float64) / (2**16) * 2 * np.pi
    phase = phase * mask

    if mask.any():
        phase = phase - np.mean(phase[mask]) * mask

    return phase


def load_zernike_coefficients(csv_path: Path, n_terms: int | None = None) -> np.ndarray:
    """Load Zernike coefficients from CSV file."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"Empty or invalid CSV: {csv_path}")

    values = [float(v) for v in rows[1]]

    if n_terms is not None:
        values = values[:n_terms]

    return np.array(values, dtype=np.float32)


class ZernikeCoefficientDataset(Dataset):
    """Dataset for Zernike coefficient prediction from camera images.

    Loads from separate .npy and .csv files:
        data_dir/
        ├── sample_0000/
        │   ├── daheng_frame.npy      # Focus/far-field image
        │   ├── miicam_frame.npy     # Pupil/near-field image
        │   ├── phase.csv           # Zernike coefficients
        │   └── metadata.json
        └── global_metadata.json

    Input modes:
        - "focus": Daheng (1 channel)
        - "pupil": MiiCam (1 channel)
        - "combined": Daheng + MiiCam (2 channels)

    Output: Zernike coefficient vector (n_zernike_terms,)
    """

    def __init__(
        self,
        data_dir: str | Path,
        input_mode: Literal["focus", "pupil", "combined"] = "combined",
        n_zernike_terms: int = 55,
        n_max: int = 10,
        target_size: tuple[int, int] | None = (256, 256),
        normalize_images: bool = True,
        brightness_normalize: bool = True,
        cache_data: bool = False,
        random_shift: bool = True,
        shift_range: int = 20,
        random_blur: bool = True,
        blur_sigma_range: tuple[float, float] = (0.1, 1.5),
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.input_mode = input_mode
        self.n_zernike_terms = n_zernike_terms
        self.n_max = n_max
        self.target_size = target_size if target_size else (256, 256)
        self.normalize_images = normalize_images
        self.brightness_normalize = brightness_normalize
        self.cache_data = cache_data
        self.augment = augment and random_shift
        self.shift_range = shift_range
        self.random_blur = random_blur and augment
        self.blur_sigma_range = blur_sigma_range

        self.in_channels = 2 if input_mode == "combined" else 1
        self._cache = {} if cache_data else None

        self.sample_dirs = sorted(self.data_dir.glob("sample_*"))
        self.samples = [d for d in self.sample_dirs if self._is_valid_sample(d)]

        if not self.samples:
            raise ValueError(f"No valid samples found in {data_dir}.")

        self.global_meta = {}
        meta_path = self.data_dir / "global_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.global_meta = json.load(f)

        self._compute_stats()

    def _is_valid_sample(self, sample_dir: Path) -> bool:
        daheng = sample_dir / "daheng_frame.npy"
        miicam = sample_dir / "miicam_frame.npy"
        phase = sample_dir / "phase.csv"

        if self.input_mode == "focus":
            return daheng.exists() and phase.exists()
        elif self.input_mode == "pupil":
            return miicam.exists() and phase.exists()
        else:
            return daheng.exists() and miicam.exists() and phase.exists()

    def _compute_stats(self) -> None:
        self.stats = {"daheng": {}, "miicam": {}}

        if not self.brightness_normalize:
            return

        n_samples = min(30, len(self.samples))
        daheng_vals, miicam_vals = [], []

        for i in range(n_samples):
            sample_dir = self.samples[i]
            if (sample_dir / "daheng_frame.npy").exists():
                daheng_vals.append(torch.from_numpy(np.load(sample_dir / "daheng_frame.npy")).float())
            if (sample_dir / "miicam_frame.npy").exists():
                miicam_vals.append(torch.from_numpy(np.load(sample_dir / "miicam_frame.npy")).float())

        if daheng_vals:
            daheng_cat = torch.cat(daheng_vals)
            self.stats["daheng"]["mean"] = daheng_cat.mean().item()
            self.stats["daheng"]["std"] = daheng_cat.std().item()

        if miicam_vals:
            miicam_cat = torch.cat(miicam_vals)
            self.stats["miicam"]["mean"] = miicam_cat.mean().item()
            self.stats["miicam"]["std"] = miicam_cat.std().item()

    def _normalize_image(self, img: torch.Tensor, camera: str) -> torch.Tensor:
        if self.brightness_normalize:
            stats = self.stats.get(camera, {})
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is not None and std is not None and std > 0:
                img = (img - mean) / std
                img = img.clamp(-3, 3) / 3

        if self.normalize_images:
            img_min = img.amin()
            img_max = img.amax()
            img_range = img_max - img_min
            if img_range > 0:
                img = 2.0 * (img - img_min) / img_range - 1.0

        return img

    def _apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        if self.shift_range > 0:
            shift_x = np.random.randint(-self.shift_range, self.shift_range + 1)
            shift_y = np.random.randint(-self.shift_range, self.shift_range + 1)
            if shift_x != 0 or shift_y != 0:
                image = self._shift_tensor(image, shift_x, shift_y)

        if self.random_blur:
            sigma = np.random.uniform(*self.blur_sigma_range)
            if sigma > 0.1:
                image = self._apply_gaussian_blur(image, sigma)

        return image

    def _shift_tensor(self, t: torch.Tensor, sx: int, sy: int) -> torch.Tensor:
        if sx == 0 and sy == 0:
            return t

        if t.ndim == 2:
            t = t.unsqueeze(0)

        c, h, w = t.shape
        shifted = torch.zeros_like(t)

        src_x_s = max(0, -sx)
        src_x_e = min(w, w - sx)
        src_y_s = max(0, -sy)
        src_y_e = min(h, h - sy)

        dst_x_s = max(0, sx)
        dst_x_e = min(w, w + sx)
        dst_y_s = max(0, sy)
        dst_y_e = min(h, h + sy)

        if src_x_e > src_x_s and src_y_e > src_y_s:
            shifted[:, dst_y_s:dst_y_e, dst_x_s:dst_x_e] = t[
                :, src_y_s:src_y_e, src_x_s:src_x_e
            ]

        return shifted

    def _apply_gaussian_blur(self, t: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma < 0.1:
            return t

        if t.ndim == 2:
            t = t.unsqueeze(0)

        c = t.shape[0]
        result = []
        for ch in range(c):
            channel = t[ch : ch + 1, :, :]

            kernel_size = max(3, int(6 * sigma + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1

            pad = kernel_size // 2
            padded = torch.nn.functional.pad(channel, (pad, pad, pad, pad), mode="reflect")
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=t.device) / (kernel_size**2)
            blurred = torch.nn.functional.conv2d(padded, kernel, padding=0, groups=1)
            result.append(blurred)

        return torch.cat(result, dim=0)

    def _find_center(self, img: torch.Tensor) -> tuple[float, float]:
        arr = img.squeeze().cpu().numpy()
        arr_norm = arr - arr.min()
        if arr_norm.max() > 0:
            center = ndimage.center_of_mass(arr_norm)
            return float(center[0]), float(center[1])
        return arr.shape[0] / 2, arr.shape[1] / 2

    def _crop_resize(
        self, img: torch.Tensor, center: tuple[float, float], crop_size: int
    ) -> torch.Tensor:
        if img.ndim == 3:
            img = img.squeeze(0)

        h, w = img.shape
        row, col = center

        top = int(max(0, row - crop_size // 2))
        bottom = top + crop_size
        left = int(max(0, col - crop_size // 2))
        right = left + crop_size

        if bottom > h:
            bottom = h
            top = max(0, bottom - crop_size)
        if right > w:
            right = w
            left = max(0, right - crop_size)

        cropped = img[top:bottom, left:right]

        if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
            pad_h = max(0, crop_size - cropped.shape[0])
            pad_w = max(0, crop_size - cropped.shape[1])
            cropped = torch.nn.functional.pad(
                cropped,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="constant",
                value=cropped.min().item() if cropped.numel() > 0 else 0,
            )

        resized = (
            torch.nn.functional.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=(self.target_size[0], self.target_size[1]),
                mode="bilinear",
                align_corners=False,
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
            coefficients = cached_result["coefficients"].clone()
        else:
            sample_dir = self.samples[idx]
            channels = []

            if self.input_mode in ("focus", "combined"):
                daheng_path = sample_dir / "daheng_frame.npy"
                if not daheng_path.exists():
                    raise ValueError(f"Daheng frame not found: {daheng_path}")

                daheng_np = np.load(daheng_path)
                daheng = torch.from_numpy(daheng_np).float()
                row, col = self._find_center(daheng)
                crop_size = min(512, daheng.shape[-2], daheng.shape[-1])
                daheng_proc = self._crop_resize(daheng, (row, col), crop_size)
                daheng_norm = self._normalize_image(daheng_proc, "daheng")
                if daheng_norm.ndim == 2:
                    daheng_norm = daheng_norm.unsqueeze(0)
                channels.append(daheng_norm)

            if self.input_mode in ("pupil", "combined"):
                miicam_path = sample_dir / "miicam_frame.npy"
                if not miicam_path.exists():
                    raise ValueError(f"MiiCam frame not found: {miicam_path}")

                miicam_np = np.load(miicam_path)
                miicam = torch.from_numpy(miicam_np).float()
                row, col = self._find_center(miicam)
                crop_size = min(512, miicam.shape[-2], miicam.shape[-1])
                miicam_proc = self._crop_resize(miicam, (row, col), crop_size)
                miicam_norm = self._normalize_image(miicam_proc, "miicam")
                if miicam_norm.ndim == 2:
                    miicam_norm = miicam_norm.unsqueeze(0)
                channels.append(miicam_norm)

            image = torch.cat(channels, dim=0)

            phase_path = sample_dir / "phase.csv"
            if not phase_path.exists():
                raise ValueError(f"Phase CSV not found: {phase_path}")

            coeffs = load_zernike_coefficients(phase_path, self.n_zernike_terms)
            coefficients = torch.from_numpy(coeffs)

            if self.cache_data:
                self._cache[sample_path] = {
                    "image": image.clone(),
                    "coefficients": coefficients.clone(),
                    "sample_idx": idx,
                    "path": sample_path,
                }

        if self.augment:
            image = self._apply_augmentation(image)

        return {
            "image": image,
            "coefficients": coefficients,
            "sample_idx": idx,
            "path": sample_path,
        }


def create_zernike_loaders(
    data_dir: str | Path,
    batch_size: int = 8,
    train_split: float = 0.7,
    val_split: float = 0.15,
    input_mode: Literal["focus", "pupil", "combined"] = "combined",
    n_zernike_terms: int = 55,
    n_max: int = 10,
    target_size: tuple[int, int] = (256, 256),
    num_workers: int = 0,
    seed: int = 42,
    augment: bool = True,
    cache_data: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders for Zernike coefficient prediction."""
    dataset = ZernikeCoefficientDataset(
        data_dir=data_dir,
        input_mode=input_mode,
        n_zernike_terms=n_zernike_terms,
        n_max=n_max,
        target_size=target_size,
        augment=augment,
        cache_data=cache_data,
    )

    n = len(dataset.samples)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=generator).tolist()

    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    train_indices = shuffled[:train_end]
    val_indices = shuffled[train_end:val_end]
    test_indices = shuffled[val_end:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
