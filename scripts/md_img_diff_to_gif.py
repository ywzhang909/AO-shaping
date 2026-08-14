"""Render per-IP animated GIFs from the jet-cmap diff images.

For every subfolder under INPUT_DIR (one per controller IP), combine the
50 per-channel diff PNGs (sorted by channel number, extracted from the
filename) into an animated GIF. Frames are downscaled by --scale to keep
the GIF size reasonable; each frame is labelled with its channel number
and the centroid written in the filename.

Usage:
    python scripts/md_img_diff_to_gif.py
    python scripts/md_img_diff_to_gif.py --input data/md_test/md_img-100v_diff_jet \
        --output data/md_test/md_img-100v_gif --scale 0.25 --fps 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Filenames look like: <ip>-<channel>_cx<X>_cy<Y>.png  (channel is the
# 3-digit number right after the last dash of the base name).
_CHANNEL_RE = re.compile(r"-(\d{3})_cx")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/md_test/md_img-100v_diff_jet"),
        help="Directory containing one subfolder per IP with jet diff PNGs",
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
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    made: list[Path] = []
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


if __name__ == "__main__":
    main()
