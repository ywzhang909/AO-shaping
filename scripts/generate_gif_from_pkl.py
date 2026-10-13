import sys
import pickle
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    forward_intensity,
)
from ao_shaping.utils.slm.phase_embed import pupil_side


MAX_GRAYSCALE = 1023
TWO_PI_GRAY_1064NM = 993


def _load_config(pkl_path: Path) -> dict:
    sidecar = pkl_path.with_suffix(".json")
    config = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    probe = pkl_path.parent / "forward_diagnostic.json"
    if probe.exists():
        diag = json.loads(probe.read_text(encoding="utf-8"))
        if "pupil_center_xy" in diag and diag["pupil_center_xy"] is not None:
            config["pupil_center"] = tuple(diag["pupil_center_xy"])
        if "wavelength_nm" in diag and diag["wavelength_nm"] is not None:
            config["slm_wavelength"] = int(diag["wavelength_nm"])
    return config


def _pupil_geometry(pkl_path: Path, center_override: tuple[int, int] | None):
    config = _load_config(pkl_path)
    center = center_override or config.get("pupil_center")
    if center is None:
        center = (960, 600)
    radius = config.get("beam_radius_px")
    if radius is None:
        return None
    center = tuple(float(c) for c in center[:2])
    return center, float(radius)


def _build_bench_config(config: dict) -> ShapingBenchConfig:
    beam_radius = float(config.get("beam_radius_px", 450.0))
    n_grid = pupil_side(beam_radius)
    panel_pixel_um = float(config.get("panel_pixel_um", 8.0))
    aperture_m = n_grid * panel_pixel_um * 1e-6
    slm_wavelength = int(config.get("slm_wavelength", 0))
    if slm_wavelength == 0:
        slm_wavelength = 1064
    wavelength = slm_wavelength * 1e-9
    focal_length = float(config.get("focal_length_m", 0.1224))
    far_field_padding = int(config.get("far_field_padding", 3))
    seed = int(config.get("seed", 0))
    return ShapingBenchConfig(
        n_grid=n_grid,
        aperture_size=aperture_m,
        wavelength=wavelength,
        focal_length=focal_length,
        far_field_padding=far_field_padding,
        target_side_px=8,
        seed=seed,
    )


def _gray_to_radians(gray: np.ndarray, wavelength_nm: int) -> np.ndarray:
    max_gray = TWO_PI_GRAY_1064NM if wavelength_nm == 1064 else MAX_GRAYSCALE
    return np.asarray(gray, dtype=np.float64) / max_gray * (2.0 * np.pi)


def _extract_pupil_phase(
    gray: np.ndarray,
    beam_radius: float,
    center_xy: tuple[float, ...],
) -> np.ndarray:
    h, w = gray.shape
    side = pupil_side(beam_radius)
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = side // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    return gray[y0:y1, x0:x1]


def generate_gif(
    pkl_path: Path,
    fps: float = 1.0,
    *,
    pupil_center: tuple[int, int] | None = None,
) -> Path:
    geometry = _pupil_geometry(pkl_path, pupil_center)
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

        fig, axes = plt.subplots(
            1, 3 if geometry else 2, figsize=(15 if geometry else 10, 5)
        )

        # CCD image (left)
        im0 = axes[0].imshow(img, cmap="hot", vmin=0, vmax=np.percentile(img, 99))
        score_text = "Epoch {} - CCD Image\nScore: {:.4f}, PIB: {:.4f}".format(
            k, entry.get("score", 0), entry.get("pib", 0)
        )
        axes[0].set_title(score_text)
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # SLM Phase (center)
        h, w = phase.shape
        crop = min(500, h, w)
        cx, cy = geometry[0] if geometry else (w // 2, h // 2)
        cx, cy = int(round(cx)), int(round(cy))
        cx = min(max(cx, crop // 2), w - (crop - crop // 2))
        cy = min(max(cy, crop // 2), h - (crop - crop // 2))
        phase_crop = phase[
            cy - crop // 2 : cy + crop // 2, cx - crop // 2 : cx + crop // 2
        ]
        im1 = axes[1].imshow(phase_crop, cmap="twilight", vmin=0, vmax=MAX_GRAYSCALE)
        axes[1].set_title("Epoch {} - SLM pupil region".format(k))
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        if geometry:
            cfg_dict = _load_config(pkl_path)
            bench_cfg = _build_bench_config(cfg_dict)
            slm_wavelength = int(cfg_dict.get("slm_wavelength", 0)) or 1064
            pupil_gray = _extract_pupil_phase(phase, geometry[1], geometry[0])
            pupil_rad = _gray_to_radians(pupil_gray, slm_wavelength)
            sim_far = forward_intensity(pupil_rad, bench_cfg)
            sim_far = sim_far / max(sim_far.max(), 1e-20)
            sh, sw = sim_far.shape
            s_crop = min(sh, sw, 500)
            sim_crop = sim_far[
                sh // 2 - s_crop // 2 : sh // 2 + s_crop // 2,
                sw // 2 - s_crop // 2 : sw // 2 + s_crop // 2,
            ]
            im2 = axes[2].imshow(
                sim_crop, cmap="hot", vmin=0, vmax=np.percentile(sim_crop, 99)
            )
            axes[2].set_title(
                "Epoch {} - simulated FFT far field\n(lambda={}nm, f={:.1f}mm)".format(
                    k, slm_wavelength, bench_cfg.focal_length * 1000
                )
            )
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

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
        print(
            "Usage: python generate_gif_from_pkl.py <pkl_path> [fps] [pupil_x,pupil_y]"
        )
        sys.exit(1)

    pkl_path = Path(sys.argv[1])
    if not pkl_path.with_suffix(".pkl").exists():
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
    center: tuple[int, int] | None = None
    if len(sys.argv) > 3:
        parts = tuple(map(int, sys.argv[3].split(",")))
        if len(parts) == 2:
            center = parts
    generate_gif(pkl_path, fps, pupil_center=center)


if __name__ == "__main__":
    main()
