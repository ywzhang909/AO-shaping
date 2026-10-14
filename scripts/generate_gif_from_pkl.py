from __future__ import annotations

import sys
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path


def generate_gif(pkl_path: Path, fps: float = 1.0) -> Path:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keys = sorted(data.keys())
    print("Keys:", keys)

    frames = []
    for k in keys:
        entry = data[k]
        img = entry.get("_img")
        phase = entry.get("_phase")

        if img is None or phase is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # CCD image (left)
        im0 = axes[0].imshow(img, cmap="hot", vmin=0, vmax=np.percentile(img, 99))
        title_text = "Epoch {} - CCD Image\nScore: {:.4f}".format(
            k, entry.get("score", 0)
        )
        axes[0].set_title(title_text)
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Phase (right) - show central region
        h, w = phase.shape
        crop = min(500, h, w)
        cy, cx = h // 2, w // 2
        phase_crop = phase[
            cy - crop // 2 : cy + crop // 2, cx - crop // 2 : cx + crop // 2
        ]
        im1 = axes[1].imshow(phase_crop, cmap="twilight", vmin=0, vmax=1023)
        axes[1].set_title("Epoch {} - SLM Phase (center crop)".format(k))
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3])
        plt.close(fig)

    gif_path = pkl_path.with_suffix(".gif")
    imageio.mimsave(gif_path, frames, fps=fps)
    print("Saved GIF to", gif_path)
    return gif_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_gif_from_pkl.py <pkl_path> [fps]")
        sys.exit(1)

    pkl_path = Path(sys.argv[1])
    if not pkl_path.with_suffix(".pkl").exists():
        # Allow passing a directory with a single pkl
        if pkl_path.is_dir():
            pkls = sorted(pkl_path.glob("*.pkl"))
            if pkls:
                pkl_path = pkls[0]
            else:
                print("No .pkl found in", pkl_path)
                sys.exit(1)
        else:
            print("File not found:", pkl_path)
            sys.exit(1)

    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    generate_gif(pkl_path, fps)


if __name__ == "__main__":
    main()
