"""Complete Micro-DM diff analysis pipeline with shared per-IP reference.

This script orchestrates the full pipeline for each controller IP:
  1. Diff computation against a SHARED reference (default:
     ``<input>/192.168.0.101/192.168.0.101-000.png`` — one common baseline for
     all IPs; override with ``--ref``)
  2. Max aggregation overlay (per-IP and global)
  3. Per-IP animated GIF generation
  4. Combined GIF with IP labels and metadata

Every source image including channel 000 is diffed (1:1 filename mapping).

Usage:
    python scripts/md_img_pipeline.py --input data/md_test/md_img-80v
    python scripts/md_img_pipeline.py --input data/md_test/md_img-100v --threshold 12
    python scripts/md_img_pipeline.py --input data/md_test/md_img-80v --skip-diff
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Reuse functions from existing scripts
sys.path.insert(0, str(Path(__file__).parent))
from md_img_diff_centroid import (
    dominant_blob_centroid,
    load_gray,
    notch_fft,
    render_colormap,
)

# --- Constants ---

_CHANNEL_RE = re.compile(r"-(\d{3})")

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/consola.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _channel_of(path: Path) -> int:
    m = _CHANNEL_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse channel number from {path.name}")
    return int(m.group(1))


_CENTROID_RE = re.compile(r"_cx(-?\d+(?:\.\d+)?)_cy(-?\d+(?:\.\d+)?)")


def _centroid_from_name(name: str) -> tuple[float, float] | None:
    """Parse centroid from a filename like `...-001_cx821.0_cy245.6.png`."""
    m = _CENTROID_RE.search(name)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


class _CentroidDB:
    """Per-IP centroid lookup keyed by original filename (stem).

    Prefers a `centroids.csv` next to the diff images (written by
    `process_ip_diff`), falls back to parsing the `_cx..._cy...` suffix from
    legacy filenames.
    """

    def __init__(self, diff_dir: Path) -> None:
        self._by_stem: dict[str, tuple[float, float] | None] = {}
        csv_path = diff_dir / "centroids.csv"
        if csv_path.is_file():
            with open(csv_path, encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                    stem = parts[0]
                    cx_s, cy_s = parts[2], parts[3]
                    self._by_stem[stem] = (
                        (float(cx_s), float(cy_s))
                        if cx_s not in ("", "None", "nan")
                        else None
                    )

    def get(self, path: Path) -> str:
        name = path.name
        if name in self._by_stem:
            c = self._by_stem[name]
            if c is None:
                return "?"
            return f"({c[0]:.0f}, {c[1]:.0f})"
        c = _centroid_from_name(name)
        if c is None:
            return "?"
        return f"({c[0]:.0f}, {c[1]:.0f})"


def _centroid_of(path: Path) -> str:
    return _CentroidDB(path.parent).get(path)


def _load_metadata(ip_dir: Path) -> dict:
    meta_path = ip_dir / "metadata.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _format_metadata(meta: dict) -> str:
    if not meta:
        return ""
    parts = []
    if "voltage" in meta:
        parts.append(f"V={meta['voltage']:.0f}V")
    if "exposure_ms" in meta:
        parts.append(f"exp={meta['exposure_ms']}ms")
    if "camera_type" in meta:
        parts.append(f"cam={meta['camera_type']}")
    if "timestamp" in meta:
        parts.append(meta["timestamp"][:19])
    return " | ".join(parts)


# --- Step 1: Per-IP Diff Computation ---


def process_ip_diff(
    ip_dir: Path,
    diff_dir: Path,
    ref: np.ndarray,
    threshold: float,
    cmap: str,
    vmax: float | None,
    notch: bool,
) -> list[tuple[Path, Path, tuple[float, float] | None]]:
    """Compute diff images for one IP against a SHARED reference.

    The reference is the pre-loaded (and optionally notch-filtered) grayscale
    array passed in — by default the channel-000 frame of the first controller
    (``192.168.0.101-000``) is used for ALL IPs, giving one common baseline
    across the whole dataset.

    Every source image (including channel 000 of each IP) gets a diff, so the
    output keeps a full 1:1 channel correspondence with the source (000..N).
    Output images keep the EXACT original filename, and the per-channel
    centroids are written to ``centroids.csv`` inside the IP's diff folder.

    Returns a list of (src, out, centroid_or_None).
    """
    ref_shape = ref.shape
    out_dir = diff_dir / ip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, Path, tuple[float, float] | None]] = []
    for src in sorted(ip_dir.glob("*.png")):
        img = load_gray(src)
        if img.shape != ref_shape:
            continue
        if notch:
            img = notch_fft(img)
        diff = ref - img
        ad = np.abs(diff)
        denoised = np.where(ad >= threshold, ad, 0.0)
        out = render_colormap(denoised, threshold, vmax=vmax, cmap=cmap)
        centroid = dominant_blob_centroid(diff, threshold)

        # Keep the original filename -> 1:1 mapping with the source image
        out_path = out_dir / src.name
        Image.fromarray(out).save(out_path)
        results.append((src, out_path, centroid))

    # Write centroid table: filename, channel, cx, cy
    csv_path = out_dir / "centroids.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("filename,channel,cx,cy\n")
        for src, _, centroid in sorted(results, key=lambda r: _channel_of(r[0])):
            ch = _channel_of(src)
            if centroid is None:
                f.write(f"{src.name},{ch},,\n")
            else:
                cx, cy = centroid
                f.write(f"{src.name},{ch},{cx:.1f},{cy:.1f}\n")

    return results


# --- Step 2: Per-IP Overlay ---


def compute_ip_overlay(diff_dir: Path, out_path: Path) -> float:
    """Compute pixel-wise maximum overlay for one IP's diff images."""
    images = sorted(diff_dir.glob("*.png"))
    if not images:
        return 0.0

    first = np.asarray(Image.open(images[0]), dtype=np.float64)
    h, w = first.shape[:2]
    is_rgb = first.ndim == 3 and first.shape[2] == 3

    if is_rgb:
        overlay = np.zeros((h, w, 3), dtype=np.float64)
    else:
        overlay = np.zeros((h, w), dtype=np.float64)

    for path in images:
        img = np.asarray(Image.open(path), dtype=np.float64)
        if img.shape[:2] != (h, w):
            continue
        overlay = np.maximum(overlay, img)

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    gray = np.mean(overlay, axis=2) if is_rgb else overlay
    coverage = float((gray > 0).sum()) / (h * w) * 100
    return coverage


# --- Step 3: Per-IP GIF ---


def render_ip_gif(
    diff_dir: Path,
    out_path: Path,
    scale: float,
    fps: int,
) -> Path:
    """Create animated GIF from one IP's diff images."""
    frames = sorted(diff_dir.glob("*.png"), key=_channel_of)
    if not frames:
        raise FileNotFoundError(f"No diff images under {diff_dir}")

    duration_ms = int(1000 / fps)
    font = _load_font(max(12, int(28 * scale)))
    centroids = _CentroidDB(diff_dir)

    pil_frames: list[Image.Image] = []
    for f in frames:
        img = Image.open(f)
        if scale != 1.0:
            w, h = img.size
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        if img.mode != "RGB":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        ch = _channel_of(f)
        label = f"ch {ch:02d}  {centroids.get(f)}"
        draw.text((8, 8), label, fill=(255, 255, 255), font=font)
        pil_frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


# --- Step 4: Combine GIFs ---


def combine_gifs(
    gif_dir: Path,
    source_dirs: list[Path],
    out_path: Path,
    fps: int,
) -> Path:
    """Combine per-IP GIFs into one master GIF with labels."""
    gif_files = sorted(gif_dir.glob("*.gif"))
    if not gif_files:
        raise FileNotFoundError(f"No .gif files under {gif_dir}")

    meta_map: dict[str, dict] = {}
    for src in source_dirs:
        if src.is_dir():
            meta_map[src.name] = _load_metadata(src)

    font = _load_font(14)
    font_small = _load_font(11)

    all_frames: list[Image.Image] = []
    ip_labels: list[tuple[str, int, int]] = []

    for gif_path in gif_files:
        ip_name = gif_path.stem
        with Image.open(gif_path) as gif:
            n_frames = gif.n_frames
            ip_labels.append((ip_name, len(all_frames), n_frames))
            for frame_idx in range(n_frames):
                gif.seek(frame_idx)
                frame = gif.copy().convert("RGB")
                all_frames.append(frame)

    if not all_frames:
        raise RuntimeError("No frames loaded from any GIF")

    width, height = all_frames[0].size
    for ip_idx, (ip_name, start, count) in enumerate(ip_labels):
        meta = meta_map.get(ip_name, {})
        meta_str = _format_metadata(meta)
        ip_label = f"{ip_name}-{ip_idx + 1:02d}"

        for i in range(count):
            frame = all_frames[start + i]
            draw = ImageDraw.Draw(frame)

            if meta_str:
                draw.text((6, 6), meta_str, fill=(255, 255, 255), font=font_small)

            bbox = draw.textbbox((0, 0), ip_label, font=font)
            text_h = bbox[3] - bbox[1]
            x, y = 6, height - text_h - 10
            draw.text((x + 1, y + 1), ip_label, fill=(0, 0, 0), font=font)
            draw.text((x, y), ip_label, fill=(255, 50, 50), font=font)

    palette_frames = [
        f.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for f in all_frames
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    palette_frames[0].save(
        out_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input directory with per-IP subfolders (e.g. data/md_test/md_img-80v)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output root directory (default: <input>_processed)",
    )
    parser.add_argument(
        "--threshold", type=float, default=15.0,
        help="Dark-blob threshold (default: 15)",
    )
    parser.add_argument(
        "--cmap", choices=("gray", "jet"), default="jet",
        help="Colormap for diff images (default: jet)",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Fixed jet color scale maximum (default: per-image max)",
    )
    parser.add_argument(
        "--notch", action=argparse.BooleanOptionalAction, default=True,
        help="FFT-notch to remove fringes (default: on)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.25,
        help="Downscale factor for GIFs (default: 0.25)",
    )
    parser.add_argument(
        "--fps", type=float, default=8.0,
        help="GIF frame rate (default: 8)",
    )
    parser.add_argument(
        "--skip-diff", action="store_true",
        help="Skip diff computation (use existing diff images)",
    )
    parser.add_argument(
        "--ref", type=Path, default=None,
        help="Shared reference image for ALL IPs (default: "
             "<input>/<first-ip>/<first-ip>-000.png, i.e. 192.168.0.101-000)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    # Set up output directories
    if args.output is None:
        args.output = args.input.parent / f"{args.input.name}_processed"
    diff_root = args.output / "diff"
    overlay_dir = args.output / "overlay"
    gif_dir = args.output / "gif"

    ip_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    print(f"Found {len(ip_dirs)} controller IPs")

    # Load the SHARED reference once: channel-000 of the first controller
    # (192.168.0.101) is used for ALL IPs so every diff shares one baseline.
    shared_ref: np.ndarray | None = None
    if not args.skip_diff and ip_dirs:
        if args.ref is not None:
            ref_path = args.ref
        else:
            ref_path = ip_dirs[0] / f"{ip_dirs[0].name}-000.png"
        if ref_path.is_file():
            shared_ref = load_gray(ref_path)
            if args.notch:
                shared_ref = notch_fft(shared_ref)
            print(f"Shared reference: {ref_path.name} (from {ref_path.parent.name})")
        else:
            print(f"WARNING: shared reference not found: {ref_path}")

    for ip_idx, ip_dir in enumerate(ip_dirs):
        ip_name = ip_dir.name
        print(f"\n[{ip_idx + 1}/{len(ip_dirs)}] Processing {ip_name}...")

        # Step 1: Diff computation
        if not args.skip_diff:
            try:
                if shared_ref is None:
                    raise FileNotFoundError("No shared reference available")
                results = process_ip_diff(
                    ip_dir, diff_root, shared_ref,
                    args.threshold, args.cmap, args.vmax, args.notch
                )
                n_valid = sum(1 for _, _, c in results if c is not None)
                print(f"  Diff: {len(results)} images, {n_valid} with valid centroids")
            except Exception as e:
                print(f"  Diff FAILED: {e}")
                continue

        # Step 2: Overlay
        ip_diff_dir = diff_root / ip_name
        if ip_diff_dir.is_dir():
            overlay_path = overlay_dir / f"{ip_name}_overlay.png"
            coverage = compute_ip_overlay(ip_diff_dir, overlay_path)
            print(f"  Overlay: {coverage:.1f}% coverage -> {overlay_path.name}")

        # Step 3: GIF
        if ip_diff_dir.is_dir():
            gif_path = gif_dir / f"{ip_name}.gif"
            try:
                render_ip_gif(ip_diff_dir, gif_path, args.scale, args.fps)
                print(f"  GIF: {gif_path.name} ({gif_path.stat().st_size / 1024:.0f} KiB)")
            except Exception as e:
                print(f"  GIF FAILED: {e}")

    # Step 4: Global overlay
    print(f"\nComputing global overlay...")
    all_diff_files = list(diff_root.rglob("*.png"))
    if all_diff_files:
        first = np.asarray(Image.open(all_diff_files[0]), dtype=np.float64)
        h, w = first.shape[:2]
        is_rgb = first.ndim == 3 and first.shape[2] == 3
        overlay = np.zeros((h, w, 3) if is_rgb else (h, w), dtype=np.float64)
        for path in all_diff_files:
            img = np.asarray(Image.open(path), dtype=np.float64)
            if img.shape[:2] == (h, w):
                overlay = np.maximum(overlay, img)
        global_overlay_path = args.output / "global_overlay.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(global_overlay_path)
        gray = np.mean(overlay, axis=2) if is_rgb else overlay
        coverage = float((gray > 0).sum()) / (h * w) * 100
        print(f"  Global overlay: {coverage:.1f}% coverage -> {global_overlay_path}")

    # Step 5: Combined GIF
    if gif_dir.is_dir() and list(gif_dir.glob("*.gif")):
        combined_path = args.output / "combined.gif"
        try:
            combine_gifs(gif_dir, [args.input], combined_path, args.fps)
            print(f"\nCombined GIF: {combined_path} "
                  f"({combined_path.stat().st_size / 1024:.0f} KiB)")
        except Exception as e:
            print(f"\nCombined GIF FAILED: {e}")

    print(f"\nPipeline complete. Output -> {args.output}")


if __name__ == "__main__":
    main()
