"""Generate synchronized CCD-image + SLM-phase evolution GIFs from debug pkl records.

Reads every ``data/debug/*/*.pkl`` debug artifact produced by the
``optimize_slm_zernike_pib`` runner, and renders an **animated GIF** that
synchronously displays:

* left  panel -- CCD far-field image (``record["_img"]``), ``inferno`` colormap;
* right panel -- SLM phase pattern (``record["_phase"]``, uint16 grayscale)
  converted to radians and shown with the cyclic ``twilight`` colormap;
* frame overlay -- epoch number and the **Pearson correlation** value
  (``record["pearson"]``) when present.

Only pickles that contain **both** ``_img`` and ``_phase`` produce a GIF.  When a
``pearson`` key exists it is annotated on every frame.

Output GIFs are written to ``report/pearson_gifs/<pkl_stem>.gif``.

Usage::

    python scripts/generate_pearson_pkl_gif.py
    python scripts/generate_pearson_pkl_gif.py --pkl data/debug/slm_pib_pearson_20260929_160921/20260929_160921/slm_pib_pearson_20260929_160921_20260929_160921.pkl
    python scripts/generate_pearson_pkl_gif.py --output-dir report/pearson_gifs --max-frames 300 --fps 12
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

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

CCD_CMAP = "inferno"
PHASE_CMAP = "twilight"  # cyclic -- appropriate for wrapped phase
_PHASE_GRAY_MAX = 992.0  # grayscale value corresponding to 2*pi on this SLM


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _collect_pkls() -> list[Path]:
    """Return every ``*.pkl`` under ``data/debug`` (sorted newest-first by mtime)."""
    candidates = sorted(glob.glob(str(ROOT / "data/debug/*/*.pkl"))) + sorted(
        glob.glob(str(ROOT / "data/debug/*/*/*.pkl"))
    )
    paths = list(dict.fromkeys(Path(p) for p in candidates))  # de-dup, keep order
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def _load_run(pkl: Path) -> dict:
    """Load a pkl; return ``{epoch: record}`` dict."""
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"pkl 无有效记录: {pkl}")
    return data


def _has_phase_phase(record: dict) -> bool:
    """True when the record carries a CCD image plus *some* phase data.

    Phase may be stored as ``_phase`` (2-D SLM grayscale) or ``_c``
    (1-D freeform phase vector that can be reshaped to a square grid, as
    produced by the ``slm-gsnet`` runner).
    """
    if "_img" not in record:
        return False
    return "_phase" in record or "_c" in record


def _extract_phase(record: dict) -> np.ndarray:
    """Return a 2-D phase array from a record.

    Prefers ``_phase`` (uint16 SLM grayscale).  Falls back to ``_c``,
    reshaped to its square root dimension and converted to a 0..2*pi
    radian representation so the twilight colormap cycles correctly.
    """
    if "_phase" in record:
        return np.asarray(record["_phase"])
    c = np.asarray(record["_c"], dtype=np.float64)
    side = int(np.sqrt(c.size))
    if side * side != c.size:
        raise ValueError(f"无法将 _c (size={c.size}) 重组为正方形相位")
    return np.mod(c.reshape(side, side), 2.0 * np.pi)


def _subsample_epochs(epochs: list[int], max_frames) -> list[int]:
    """Evenly subsample the epoch list to <= ``max_frames`` entries."""
    if not max_frames or max_frames <= 0 or len(epochs) <= max_frames:
        return list(epochs)
    idx = np.linspace(0, len(epochs) - 1, int(max_frames), dtype=int)
    return [epochs[i] for i in idx]


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------
def _ccd_to_rgb(cimg: np.ndarray, cmap: str = CCD_CMAP) -> np.ndarray:
    """Normalise a CCD intensity frame to 0..1 (per-frame) and apply a colormap."""
    a = np.asarray(cimg, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        a = np.zeros_like(a)
    else:
        a = (a - lo) / (hi - lo)
    cm = plt.get_cmap(cmap)
    rgb = (cm(a)[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def _phase_to_rgb(phase: np.ndarray, cmap: str = PHASE_CMAP) -> np.ndarray:
    """Convert a phase array to cyclic-colormap RGB.

    Accepts either uint16 SLM grayscale (0..2*pi maps to ~992) or float
    radians.  Output is normalised to 0..1 for the cyclic colormap.
    """
    phase = np.asarray(phase)
    if np.issubdtype(phase.dtype, np.integer):
        gray = phase.astype(np.float64)
        a = np.mod(gray / _PHASE_GRAY_MAX * 2.0 * np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    else:
        a = np.mod(phase.astype(np.float64), 2.0 * np.pi) / (2.0 * np.pi)
    cm = plt.get_cmap(cmap)
    rgb = (cm(a)[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def _resize_keep(panel: Image.Image, target_h: int) -> Image.Image:
    if panel.height == target_h:
        return panel
    nw = max(1, int(panel.width * target_h / panel.height))
    return panel.resize((nw, target_h), Image.Resampling.LANCZOS)


def _load_font(size: int):
    for cand in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------
def _frames_to_synchronized_gif(
    ccd_frames: list[np.ndarray],
    phase_frames: list[np.ndarray],
    out_path: Path,
    pearson_values: list,
    epochs: list,
    fps: float = 12.0,
    max_dim: int = 768,
) -> Path:
    """Render a single synchronized panel GIF.

    Each frame is a 1x2 horizontal composite (CCD | phase) with a top bar
    holding the epoch number and the Pearson value.
    """
    assert len(ccd_frames) == len(phase_frames) == len(pearson_values) == len(epochs)
    n = len(ccd_frames)
    if n == 0:
        raise ValueError("no frames to render")

    bar_h = 40
    font_size = max(14, int(20 * (max_dim / 1024)))
    font = _load_font(font_size)
    font_small = _load_font(max(10, int(font_size * 0.75)))

    # Pre-render each CCD and phase panel as RGB.
    ccd_panels = [
        Image.fromarray(_ccd_to_rgb(cimg)).convert("RGB") for cimg in ccd_frames
    ]
    phase_panels = [
        Image.fromarray(_phase_to_rgb(pimg)).convert("RGB") for pimg in phase_frames
    ]

    # Equalise panel heights.
    h = min(max_dim, max(1, max(p.height for p in ccd_panels + phase_panels)))
    ccd_panels = [_resize_keep(p, h) for p in ccd_panels]
    phase_panels = [_resize_keep(p, h) for p in phase_panels]

    # Cap total width so the GIF stays manageable.
    total_w = ccd_panels[0].width + phase_panels[0].width
    cap = max_dim * 2
    if total_w > cap:
        scale = cap / total_w
        nw = max(1, int(ccd_panels[0].width * scale))
        nh = max(1, int(h * scale))
        ccd_panels = [p.resize((nw, nh), Image.Resampling.LANCZOS) for p in ccd_panels]
        phase_panels = [
            p.resize((nw, nh), Image.Resampling.LANCZOS) for p in phase_panels
        ]
        h = nh
        bar_h = max(12, int(bar_h * scale))
        font = _load_font(max(10, int(font_size * scale)))
        font_small = _load_font(max(9, int(font_size * scale * 0.75)))

    gap = 4
    w1 = ccd_panels[0].width
    w2 = phase_panels[0].width
    canvas_w = w1 + gap + w2
    canvas_h = h + bar_h

    pil_frames = []
    for i in range(n):
        frame = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
        draw = ImageDraw.Draw(frame)
        frame.paste(ccd_panels[i], (0, bar_h))
        frame.paste(phase_panels[i], (w1 + gap, bar_h))

        # Draw separator line
        draw.line(
            (w1 + gap // 2, bar_h, w1 + gap // 2, canvas_h), fill=(80, 80, 80), width=1
        )

        # Top bar -- epoch + Pearson
        pearson = pearson_values[i]
        if pearson is not None:
            label = f"epoch {epochs[i]:>5d}   Pearson = {pearson:.4f}"
        else:
            label = f"epoch {epochs[i]:>5d}"
        draw.text((8, 8), label, fill=(255, 255, 255), font=font)
        draw.text(
            (8, bar_h - 14),
            "CCD (inferno)            SLM phase (twilight)",
            fill=(200, 200, 200),
            font=font_small,
        )
        pil_frames.append(
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / fps))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


# ---------------------------------------------------------------------------
# Per-pkl processing
# ---------------------------------------------------------------------------
def process_pkl(pkl: Path, out_dir: Path, pearson_only: bool = False, **gif_kwargs):
    """Render one pkl -> one synchronized GIF.

    Returns ``(ok, message)`` where *message* carries context about Pearson
    membership.  When ``pearson_only`` is set, the file is still loaded once
    (unavoidable for pickle) but the GIF is only written when at least one
    record carries a ``pearson`` key.
    """
    try:
        data = _load_run(pkl)
    except Exception as exc:
        return False, f"load failed: {exc}"

    epochs = sorted(data.keys())
    records = [data[e] for e in epochs]

    valid_epochs = [e for e, r in zip(epochs, records) if _has_phase_phase(r)]
    if not valid_epochs:
        return False, "no records with both _img and _phase"

    has_pearson = any(
        "pearson" in data[e]
        or "pearson" in pkl.name.lower()
        or "pearson" in str(pkl.parent).lower()
        for e in valid_epochs
    )
    if pearson_only and not has_pearson:
        return False, "no-pearson (skipped)"

    max_frames = gif_kwargs.get("max_frames")
    valid_epochs = _subsample_epochs(valid_epochs, max_frames)

    ccd_frames = [data[e]["_img"] for e in valid_epochs]
    phase_frames = [_extract_phase(data[e]) for e in valid_epochs]
    pearson_values = [data[e].get("pearson") for e in valid_epochs]

    out_name = pkl.name
    if out_name.endswith(".pkl"):
        out_name = out_name[:-4]
    out_path = out_dir / f"{out_name}.gif"

    _frames_to_synchronized_gif(
        ccd_frames=ccd_frames,
        phase_frames=phase_frames,
        out_path=out_path,
        pearson_values=pearson_values,
        epochs=valid_epochs,
        fps=gif_kwargs.get("fps", 12.0),
        max_dim=gif_kwargs.get("max_dim", 768),
    )

    tag = " (pearson)" if has_pearson else " (no-pearson)"
    size_kb = out_path.stat().st_size / 1024
    return (
        True,
        f"ok{tag} {len(ccd_frames)} frames -> {out_path.name} ({size_kb:.0f} KiB)",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--pkl", default=None, help="single pkl to process (default: all data/debug pkls)"
)
@click.option(
    "-o",
    "--output-dir",
    default="report/pearson_gifs",
    show_default=True,
    help="output directory for GIFs",
)
@click.option(
    "--max-frames",
    default=300,
    show_default=True,
    help="cap number of GIF frames per pkl (0 = no cap)",
)
@click.option("--fps", default=12.0, show_default=True, help="GIF frame rate")
@click.option(
    "--max-dim",
    default=768,
    show_default=True,
    help="max dimension of each panel (CCD or phase)",
)
@click.option(
    "--pearson-only",
    is_flag=True,
    default=False,
    help="only generate GIFs for pkls that contain a 'pearson' field",
)
def cli(pkl, output_dir, max_frames, fps, max_dim, pearson_only):
    """Generate synchronized CCD+phase evolution GIFs from debug pkls.

    Reads pkl debug records and produces one animated GIF per pickle,
    showing the CCD far-field (left) and SLM phase pattern (right)
    evolving in lockstep, annotated with the Pearson correlation value
    when present.
    """
    out_dir = (ROOT / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_kwargs = {
        "max_frames": max_frames if max_frames > 0 else None,
        "fps": fps,
        "max_dim": max_dim,
    }

    if pkl:
        pkls = [Path(pkl).resolve()]
    else:
        pkls = _collect_pkls()
        if not pkls:
            raise click.ClickException(
                "未找到 data/debug/*.pkl -- 先运行优化器生成 debug 记录"
            )

    logger.info("发现 {} 个 pkl 文件", len(pkls))

    made = 0
    skipped = 0
    for p in pkls:
        ok, msg = process_pkl(p, out_dir, pearson_only=pearson_only, **gif_kwargs)
        if ok:
            made += 1
            print(f"[{p.name}] {msg}")
        else:
            skipped += 1
            print(f"[{p.name}] SKIP: {msg}")

    print(f"\n{made} GIF(s) -> {out_dir}")
    if skipped:
        print(f"{skipped} skipped")


if __name__ == "__main__":
    cli()
