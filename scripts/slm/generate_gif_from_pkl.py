import sys
import pickle
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

from loguru import logger

from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    camera_window_from_far_field,
    forward_intensity,
)
from ao_shaping.utils.slm.phase_embed import pupil_side
from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
    RECORD_PHASE,
)


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


def _pupil_geometry(
    pkl_path: Path, center_override: tuple[int, int] | None
) -> tuple[tuple[float, float], float] | None:
    """Resolve the pupil geometry needed to simulate panel 3.

    Returns ``None`` when either the pupil centre or the beam radius is unknown.
    There is deliberately no fallback centre: a hardcoded one silently produces a
    simulation that is *plausible but wrong*, which is worse than no simulation
    because the GIF then shows three panels and invites the reader to compare
    them. Sources, in priority order: explicit CLI override, ``forward_diagnostic.json``
    (written by the bench probe), then the pkl's JSON sidecar.
    """
    config = _load_config(pkl_path)
    center = center_override or config.get("pupil_center")
    radius = config.get("beam_radius_px")
    if center is None or radius is None:
        return None
    # Validate explicitly rather than trusting the sidecar: a malformed
    # ``pupil_center`` (wrong length / non-numeric) used to reach the
    # ``cx, cy = geometry[0]`` unpack and raise mid-render.
    try:
        cx, cy = (float(c) for c in center)
    except (TypeError, ValueError):
        logger.warning("忽略无法解析的 pupil_center: {!r}", center)
        return None
    return (cx, cy), float(radius)


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
    if geometry is None:
        logger.warning(
            "缺少瞳面几何, 跳过第三栏仿真 (pupil centre / beam radius unknown). "
            "来源需为: {}.json, 同目录 forward_diagnostic.json, "
            "或命令行第三参数 pupil_x,pupil_y。",
            pkl_path.stem,
        )
    # Reconstruct a Recorder first (unified load path), then project its
    # history into the epoch-keyed dict this renderer expects.
    from ao_shaping.utils.io.file import Recorder
    from ao_shaping.utils.io.recorder_keys import RECORD_EPOCH

    rec = Recorder.from_saved(pkl_path)
    data = {}
    for i, r in enumerate(rec.history):
        if not isinstance(r, dict):
            continue
        epoch = r.get(RECORD_EPOCH, i)
        data[epoch] = r
    if not data:
        logger.warning("no records restored from {}", pkl_path)
        return pkl_path.with_suffix(".gif")

    keys = sorted(data.keys())
    logger.info(f"Keys: {keys}")

    frames = []
    for k in keys:
        entry = data[k]
        img = entry.get(RECORD_IMAGE)
        phase = entry.get(RECORD_PHASE)

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
            # Put the simulation on the SAME camera pixel grid as panel 1, so the
            # two are directly comparable. Cropping "relative pixels" out of the
            # far field (what this used to do) leaves panel 3 ~30x larger-scale
            # than the measured window on this bench and not comparable at all.
            sim_crop = camera_window_from_far_field(
                sim_far, bench_cfg, np.asarray(img).shape
            )
            sim_crop = sim_crop / max(sim_crop.max(), 1e-20)
            im2 = axes[2].imshow(
                sim_crop, cmap="hot", vmin=0, vmax=np.percentile(sim_crop, 99)
            )
            axes[2].set_title(
                "Epoch {} - simulated far field\n(camera px, lambda={}nm, f={:.1f}mm)".format(
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
    logger.info(f"Saved GIF to {gif_path}")
    return gif_path


def main():
    if len(sys.argv) < 2:
        logger.info(
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
                logger.info(f"No .pkl found in {pkl_path}")
                sys.exit(1)
        else:
            logger.info(f"File not found: {pkl_path}")
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
