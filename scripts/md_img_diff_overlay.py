"""Overlay all diff images by taking the maximum value at each pixel.

Loads all PNGs from the diff directory (recursively), computes the
pixel-wise maximum across all images, and saves the result.

Usage:
    python scripts/md_img_diff_overlay.py
    python scripts/md_img_diff_overlay.py --input data/md_test/md_img-100v_diff --output data/md_test/md_img-100v_overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/md_test/md_img-100v_diff"),
        help="Directory containing diff images (recursively scanned)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/md_test/md_img-100v_overlay.png"),
        help="Output path for the overlay image",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        help="Image file suffixes to include (default: png jpg jpeg bmp tif tiff)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    # Collect all image files
    images = sorted(
        p
        for p in args.input.rglob("*")
        if p.suffix.lower() in args.suffixes and p.is_file()
    )
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    print(f"Found {len(images)} diff images")

    # Load first image to get shape
    first = np.asarray(Image.open(images[0]), dtype=np.float64)
    h, w = first.shape[:2]
    is_rgb = first.ndim == 3 and first.shape[2] == 3
    print(f"Image shape: {h}x{w} ({'RGB' if is_rgb else 'grayscale'})")

    # Compute pixel-wise maximum
    if is_rgb:
        overlay = np.zeros((h, w, 3), dtype=np.float64)
    else:
        overlay = np.zeros((h, w), dtype=np.float64)

    for i, path in enumerate(images):
        img = np.asarray(Image.open(path), dtype=np.float64)
        if img.shape[:2] != (h, w):
            print(f"  Skipping {path.name}: shape {img.shape[:2]} != ({h}, {w})")
            continue
        overlay = np.maximum(overlay, img)
        if (i + 1) % 50 == 0 or i == len(images) - 1:
            print(f"  Processed {i + 1}/{len(images)}")

    # Save result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    result.save(args.output)
    print(f"Saved overlay -> {args.output}")

    # Stats
    if is_rgb:
        # Convert to grayscale for stats
        gray = np.mean(overlay, axis=2)
    else:
        gray = overlay
    total_pixels = h * w
    nonzero_pixels = int((gray > 0).sum())
    coverage = nonzero_pixels / total_pixels * 100
    print(f"Coverage: {nonzero_pixels}/{total_pixels} pixels have signal ({coverage:.1f}%)")
    print(f"Max intensity: {gray.max():.1f}")


if __name__ == "__main__":
    main()
