"""Render per-IP animated GIFs from the per-channel images.

For every subfolder under INPUT_DIR (one per controller IP), combine the
50 per-channel PNGs (sorted by channel number, extracted from the
filename) into an animated GIF. Frames are downscaled by --scale to keep
the GIF size reasonable; each frame is labelled with its channel number
and the centroid written in the filename (if present).

Usage:
    python scripts/md_img_diff_to_gif.py
    python scripts/md_img_diff_to_gif.py --input data/md_test/md_img-100v_diff_jet \
        --output data/md_test/md_img-100v_gif --scale 0.25 --fps 8
    python scripts/md_img_diff_to_gif.py --input data/md_test/md_img-80v \
        --output data/md_test/md_img-80v_gif --scale 0.25 --fps 8

    # Combine all per-IP GIFs into one
    python scripts/md_img_diff_to_gif.py --input data/md_test/md_img-80v \
        --output data/md_test/md_img-80v_gif --combine --scale 0.25 --fps 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Filenames look like: <ip>-<channel>_cx<X>_cy<Y>.png  or  <ip>-<channel>.png
# (channel is the 3-digit number right after the last dash of the base name).
_CHANNEL_RE = re.compile(r"-(\d{3})")

# Windows system font for frame labels (falls back gracefully).
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/consola.ttf",
)


def _channel_of(path: Path) -> int:
    m = _CHANNEL_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse channel number from {path.name}")
    return int(m.group(1))


def _centroid_of(path: Path) -> str:
    m = re.search(r"_cx(-?\d+(?:\.\d+)?)_cy(-?\d+(?:\.\d+)?)", path.name)
    if m is None:
        return "?"
    return f"({float(m.group(1)):.0f}, {float(m.group(2)):.0f})"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_ip_gif(
    ip_dir: Path,
    out_path: Path,
    scale: float,
    fps: int,
) -> Path:
    """Combine the per-channel diff images of one IP into an animated GIF."""
    frames = sorted(ip_dir.glob("*.png"), key=_channel_of)
    if not frames:
        raise FileNotFoundError(f"No PNG frames under {ip_dir}")

    duration_ms = int(1000 / fps)
    font = _load_font(max(12, int(28 * scale)))

    pil_frames: list[Image.Image] = []
    for i, f in enumerate(frames):
        img = Image.open(f)
        if scale != 1.0:
            w, h = img.size
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        # Convert to RGB if needed for text drawing
        if img.mode != "RGB":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        ch = _channel_of(f)
        label = f"ch {ch:02d}  {_centroid_of(f)}"
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


def _load_metadata(ip_dir: Path) -> dict:
    """Load metadata.json from the IP directory if it exists."""
    meta_path = ip_dir / "metadata.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _format_metadata(meta: dict) -> str:
    """Format metadata dict into compact display string."""
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


def combine_gifs(
    gif_dir: Path,
    source_dirs: list[Path],
    out_path: Path,
    scale: float,
    fps: int,
) -> Path:
    """Combine per-IP GIFs into one master GIF.

    For each frame in each IP's GIF, draws:
      - top-left: metadata (voltage, exposure, camera, timestamp)
      - bottom-left: red label "IP-NN" (IP index)
    """
    gif_files = sorted(gif_dir.glob("*.gif"))
    if not gif_files:
        raise FileNotFoundError(f"No .gif files under {gif_dir}")

    # Build a mapping: ip_name -> metadata dict
    meta_map: dict[str, dict] = {}
    for src in source_dirs:
        if src.is_dir():
            meta_map[src.name] = _load_metadata(src)

    font_size = max(12, int(24 * scale))
    font = _load_font(font_size)
    font_small = _load_font(max(10, int(18 * scale)))

    # Collect frames from all GIFs
    all_frames: list[Image.Image] = []
    ip_labels: list[tuple[str, int, int]] = []  # (ip_name, start_idx, count)

    for gif_idx, gif_path in enumerate(gif_files):
        ip_name = gif_path.stem  # e.g. "192.168.0.101"
        with Image.open(gif_path) as gif:
            n_frames = gif.n_frames
            ip_labels.append((ip_name, len(all_frames), n_frames))
            for frame_idx in range(n_frames):
                gif.seek(frame_idx)
                frame = gif.copy().convert("RGB")
                # Scale if needed
                if scale != 1.0:
                    w, h = frame.size
                    frame = frame.resize(
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        Image.Resampling.LANCZOS,
                    )
                all_frames.append(frame)

    if not all_frames:
        raise RuntimeError("No frames loaded from any GIF")

    # Draw labels on each frame
    width, height = all_frames[0].size
    for ip_idx, (ip_name, start, count) in enumerate(ip_labels):
        meta = meta_map.get(ip_name, {})
        meta_str = _format_metadata(meta)
        ip_label = f"{ip_name}-{ip_idx + 1:02d}"

        for i in range(count):
            frame = all_frames[start + i]
            draw = ImageDraw.Draw(frame)

            # Top-left: metadata (white)
            if meta_str:
                draw.text((6, 6), meta_str, fill=(255, 255, 255), font=font_small)

            # Bottom-left: IP label (red)
            bbox = draw.textbbox((0, 0), ip_label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = 6
            y = height - text_h - 10
            # Draw shadow for readability
            draw.text((x + 1, y + 1), ip_label, fill=(0, 0, 0), font=font)
            draw.text((x, y), ip_label, fill=(255, 50, 50), font=font)

    # Convert to palette for GIF
    palette_frames = [
        f.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        for f in all_frames
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/md_test/md_img-100v_diff_jet"),
        help="Directory containing one subfolder per IP with per-channel PNGs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/md_test/md_img-100v_gif"),
        help="Directory where per-IP .gif files are written",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Downscale factor applied to each frame (default: 0.25)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Animation frame rate (default: 8)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="After generating per-IP GIFs, combine them into one master GIF "
        "with IP labels (bottom-left, red) and metadata (top-left, white)",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip per-IP GIF generation; only combine existing GIFs in --output "
        "into one master GIF",
    )
    parser.add_argument(
        "--combine-name",
        type=str,
        default="combined.gif",
        help="Filename for the combined GIF (default: combined.gif)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    made: list[Path] = []

    # Step 1: generate per-IP GIFs (unless --combine-only)
    if not args.combine_only:
        for ip_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
            out_path = args.output / f"{ip_dir.name}.gif"
            try:
                render_ip_gif(ip_dir, out_path, args.scale, args.fps)
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"[{ip_dir.name}] FAILED: {e}")
                continue
            made.append(out_path)
            print(f"[{ip_dir.name}] {out_path.name} "
                  f"({out_path.stat().st_size / 1024:.0f} KiB)")

        print(f"\n{len(made)} GIFs -> {args.output}")

    # Step 2: combine all per-IP GIFs into one
    if args.combine or args.combine_only:
        combine_path = args.output / args.combine_name
        try:
            combine_gifs(
                gif_dir=args.output,
                source_dirs=[args.input],
                out_path=combine_path,
                scale=1.0,  # per-IP GIFs are already scaled
                fps=args.fps,
            )
            print(f"\nCombined GIF -> {combine_path} "
                  f"({combine_path.stat().st_size / 1024:.0f} KiB)")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Combine FAILED: {e}")


if __name__ == "__main__":
    main()
