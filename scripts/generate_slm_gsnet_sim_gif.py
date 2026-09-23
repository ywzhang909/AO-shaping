"""Generate the slm-gsnet offline-sim verification GIF (+ markdown snippet).

Reads the newest ``data/debug/slm_gsnet_*/**/*.pkl`` debug artifact produced by::

    python src/ao_shaping/main.py slm-gsnet spgd --cam_type sim --epochs 40 --debug

and renders two animated GIFs into ``docs/fouriergsnet_sim/gifs/``:

* ``slm_gsnet_spgd_sim_far.gif`` — far-field (CCD) evolution, inferno;
* ``slm_gsnet_spgd_sim_phase.gif`` — freeform phase grid (24×24) evolution,
  twilight (mod 2π, cyclic).

The pkl stores ``{epoch: record}`` with ``_img`` (CCD far-field) and ``_c``
(the freeform phase vector, length ``phase_grid²`` = 576).

**Fully offline** — reads saved artefacts only, no hardware, no camera.

Usage::

    python scripts/generate_slm_gsnet_sim_gif.py
    python scripts/generate_slm_gsnet_sim_gif.py --pkl data/debug/slm_gsnet_xxx/20260923_*/xxx.pkl
    python scripts/generate_slm_gsnet_sim_gif.py -o docs/fouriergsnet_sim
"""

from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Repo bootstrap: ROOT + src on sys.path, then Agg BEFORE pyplot
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# _frames_to_gif: prefer the repo reference implementation, fall back to a
# verbatim copy (it is a pure module-level function).
# ---------------------------------------------------------------------------
try:
    from generate_diff_shaping_report import _frames_to_gif  # noqa: E402
    logger.debug("复用 scripts/generate_diff_shaping_report._frames_to_gif")
except ImportError as exc:  # noqa: F841

    def _frames_to_gif(frames: list[np.ndarray], out_path: Path, cmap: str) -> None:
        """Downscale + quantize 2D arrays into an animated GIF (repo convention)."""
        from PIL import Image

        pil_frames: list[Image.Image] = []
        for arr in frames:
            a = arr.astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            if hi - lo < 1e-12:
                a = np.zeros_like(a)
            else:
                a = (a - lo) / (hi - lo)
            cm = plt.get_cmap(cmap)
            rgba = (cm(a)[:, :, :3] * 255.0).astype(np.uint8)
            img = Image.fromarray(rgba)
            w, h = img.size
            scale = 128.0 / max(w, h)
            if scale < 1.0:
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            pil_frames.append(
                img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // 15,
            loop=0,
            optimize=True,
        )


# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

PHASE_CMAP = "twilight"  # cyclic — appropriate for mod 2π phase
FAR_CMAP = "inferno"  # far-field intensity
PHASE_GRID = 24  # freeform phase_grid² used by the slm-gsnet runner


def _resolve_pkl(pkl: str | None) -> Path:
    if pkl:
        return Path(pkl).resolve()
    candidates = sorted(glob.glob(str(ROOT / "data/debug/slm_gsnet_*/*/*.pkl")))
    if not candidates:
        raise click.ClickException(
            "未找到 data/debug/slm_gsnet_*/*/*.pkl — 先运行 "
            "`python src/ao_shaping/main.py slm-gsnet spgd --cam_type sim --epochs 40 --debug`"
        )
    return Path(candidates[-1])


def _load_run(pkl: Path) -> dict:
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    epochs = sorted(data.keys())
    if not epochs:
        raise click.ClickException(f"pkl 无记录: {pkl}")
    return data


def _make_far_frames(data: dict) -> list[np.ndarray]:
    frames = [data[e]["_img"] for e in sorted(data) if "_img" in data[e]]
    if not frames:
        raise click.ClickException("pkl 无 _img (CCD far-field) 记录")
    logger.info("提取 {} 帧 far-field (_img)", len(frames))
    return frames


def _make_phase_frames(data: dict) -> list[np.ndarray]:
    frames = []
    for e in sorted(data):
        if "_c" not in data[e]:
            continue
        c = np.asarray(data[e]["_c"], dtype=float)
        if c.size != PHASE_GRID * PHASE_GRID:
            continue
        frames.append(np.mod(c.reshape(PHASE_GRID, PHASE_GRID), 2.0 * np.pi))
    if not frames:
        raise click.ClickException("pkl 无 _c (freeform phase) 记录")
    logger.info("提取 {} 帧 freeform phase (_c→{}×{})", len(frames), PHASE_GRID, PHASE_GRID)
    return frames


@click.command()
@click.option(
    "--pkl",
    default=None,
    help="slm-gsnet debug pkl 路径 (默认取 data/debug/slm_gsnet_* 最新)",
)
@click.option(
    "-o", "--output",
    default="docs/fouriergsnet_sim",
    show_default=True,
    help="报告输出目录 (GIF 写入 <output>/gifs/)",
)
def cli(pkl: str | None, output: str) -> None:
    """slm-gsnet 离线仿真验证 GIF 生成器 (无硬件)."""
    pkl_path = _resolve_pkl(pkl)
    data = _load_run(pkl_path)
    logger.info("读取 {} ({} 个 epoch 记录)", pkl_path, len(data))

    gif_dir = (ROOT / output).resolve() / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    far_gif = gif_dir / "slm_gsnet_spgd_sim_far.gif"
    phase_gif = gif_dir / "slm_gsnet_spgd_sim_phase.gif"

    _frames_to_gif(_make_far_frames(data), far_gif, FAR_CMAP)
    _frames_to_gif(_make_phase_frames(data), phase_gif, PHASE_CMAP)

    logger.info("GIF 已写入 {} / {}", far_gif, phase_gif)
    print(f"far:   ![{far_gif.stem}](gifs/{far_gif.name})")
    print(f"phase: ![{phase_gif.stem}](gifs/{phase_gif.name})")


if __name__ == "__main__":
    cli()