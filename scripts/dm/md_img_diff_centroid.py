"""Batch difference processing for per-channel DM images.

For every image under INPUT_DIR:
    1. diff = reference - image          (signed float difference)
    2. denoise by threshold               (keep |diff| >= threshold, else 0)
    3. save the thresholded |diff| to OUTPUT_DIR, mirroring input subfolders
    4. filename gains the intensity-weighted centroid of the DOMINANT DARK
       blob (largest connected component of diff > threshold), which is the
       per-channel DM response signal:
            <original_stem>_cx<X>_cy<Y>.png

NOTE: a whole-image centroid is useless here - the diff is dominated by
noise (3*sigma keeps 88-97% of all pixels). The dominant dark blob is the
real, localized per-channel signal (empirically |diff| values 15-25 at the
moved spot). Use --threshold 15 (override with --threshold).

Colormap:
    --cmap gray  -> grayscale |diff| (0..255)
    --cmap jet   -> jet colormap over [threshold, vmax] with black
                    background; differences pop as blue->cyan->green->
                    yellow->red as |diff| grows (default). vmax defaults
                    to the per-image max |diff| (>= threshold + 1) so the
                    full jet range is used; override with --vmax for a
                    fixed scale comparable across images.

FFT notch (--notch, default ON):
    The raw camera images carry fixed-pattern interference fringes
    (freq peaks at +-(19,4), +-(19,7), +-(6,-2), +-(5,2), +-(2,7) in
    fftshifted space; x period ~141 px). Notching the RAW reference and
    image BEFORE diffing removes these fringes without moving the DM
    response signal (validated: fringe energy 15.6M -> 0, centroid of
    119-030 stays (1790,1241)). Disable with --no-notch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

# Fixed-pattern interference fringe peaks in fftshifted frequency space
# (fx, fy) with fx along x-axis. Empirically derived from RAW frames.
FRINGE_PEAKS: tuple[tuple[int, int], ...] = (
    (19, 4), (19, 7), (-19, -4), (-19, -7),
    (6, -2), (-6, 2), (5, 2), (-5, -2), (2, 7), (-2, -7),
)


def load_gray(path: Path) -> np.ndarray:
    """Load an image as float64 grayscale array."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.float64)


def notch_fft(img: np.ndarray, peaks: tuple[tuple[int, int], ...] = FRINGE_PEAKS,
              width: float = 1.0) -> np.ndarray:
    """Gaussian notch filter suppressing fixed fringe frequencies.

    Applies a Gaussian notch (1 - exp(-d^2 / 2w^2), with the very center
    of each peak zeroed) to the fftshifted spectrum, then inverse-FTs back
    to real space. width=1.0 fully removes the fringe energy (-> 0) while
    preserving localized DM signals (validated on 4 ground-truth images).

    The frequency mask depends only on the image shape, so it is built
    once and cached per shape (rebuilding it is ~2s, ~3x the FFT itself).
    """
    mask = _get_notch_mask(img.shape, peaks, width)
    spec = np.fft.fftshift(np.fft.fft2(img))
    return np.fft.ifft2(np.fft.ifftshift(spec * mask)).real


_notch_cache: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...], float], np.ndarray] = {}


def _get_notch_mask(
    shape: tuple[int, int],
    peaks: tuple[tuple[int, int], ...],
    width: float,
) -> np.ndarray:
    """Build (or fetch from cache) the Gaussian notch frequency mask."""
    key = (shape, peaks, width)
    cached = _notch_cache.get(key)
    if cached is not None:
        return cached
    h, w = shape
    cy, cx = h // 2, w // 2
    ys, xs = np.mgrid[0:h, 0:w]
    mask = np.ones((h, w))
    for fx, fy in peaks:
        y0, x0 = cy + fy, cx + fx
        d2 = (xs - x0) ** 2 + (ys - y0) ** 2
        mask *= 1.0 - np.exp(-d2 / (2.0 * width * width))
        mask[d2 <= 1.0] = 0.0
    _notch_cache[key] = mask
    return mask


def render_colormap(
    denoised: np.ndarray,
    threshold: float,
    vmax: float | None = None,
    cmap: str = "jet",
) -> np.ndarray:
    """Map thresholded |diff| to an RGB display image.

    gray:  uint8 grayscale, identical to the plain |diff| (0..255).
    jet:   values in [threshold, vmax] mapped through the jet colormap,
           background (0) left black, so the difference signal stands out.
           vmax defaults to the per-image max |diff| (floored at
           threshold + 1) to use the full jet range.

    Returns a uint8 RGB (H, W, 3) array.
    """
    if cmap == "gray":
        return np.clip(denoised, 0, 255).astype(np.uint8)

    if cmap != "jet":
        raise ValueError(f"Unknown colormap: {cmap!r} (supported: gray, jet)")

    import matplotlib.colors as mcolors

    from matplotlib import colormaps

    scale = vmax if vmax is not None else max(float(denoised.max()), threshold + 1.0)
    if scale <= threshold:
        scale = threshold + 1.0
    norm = mcolors.Normalize(vmin=threshold, vmax=scale, clip=True)
    rgba = colormaps["jet"](norm(denoised))  # (H, W, 4) float in [0, 1]
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    # black background where no difference survived the threshold
    rgb[denoised <= 0] = 0
    return rgb


def dominant_blob_centroid(diff: np.ndarray, threshold: float) -> tuple[float, float] | None:
    """Intensity-weighted centroid of the LARGEST connected dark blob.

    Dark blob = pixels where (diff - ref image) > threshold, i.e. the image
    is significantly darker than the reference. The largest connected
    component is the per-channel DM response signal.

    Returns (cx, cy) in (x, y) order, or None if no dark pixel survives.
    """
    from scipy import ndimage

    mask = diff > threshold
    if not mask.any():
        return None
    # scipy stubs type label() returns poorly; cast to real array shape
    label_result = cast(tuple[np.ndarray, int], ndimage.label(mask))
    labels: np.ndarray = label_result[0]
    n = int(label_result[1])
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1  # label index of the largest blob
    ys, xs = np.where(labels == main)
    w = diff[ys, xs]
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())


def process(
    ref_path: Path,
    input_dir: Path,
    output_dir: Path,
    threshold: float | None,
    cmap: str = "jet",
    vmax: float | None = None,
    notch: bool = True,
    suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> list[tuple[Path, float, tuple[float, float] | None, str]]:
    """Run the pipeline over all images in input_dir (recursive).

    Returns a summary list of (src, threshold_used, centroid_or_None, out_name).
    """
    ref = load_gray(ref_path)
    ref_h, ref_w = ref.shape
    if notch:
        ref = notch_fft(ref)

    images = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in suffixes and p.is_file()
    )
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}")

    results: list[tuple[Path, float, tuple[float, float] | None, str]] = []
    for src in images:
        img = load_gray(src)
        if img.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch: ref {ref.shape} vs {src} {img.shape}"
            )
        if notch:
            img = notch_fft(img)
        diff = ref - img
        thr = threshold if threshold is not None else 15.0

        ad = np.abs(diff)
        denoised = np.where(ad >= thr, ad, 0.0)
        out = render_colormap(denoised, thr, vmax=vmax, cmap=cmap)

        centroid = dominant_blob_centroid(diff, thr)

        rel = src.relative_to(input_dir)
        stem = src.stem
        if centroid is not None:
            cx, cy = centroid
            out_name = f"{stem}_cx{cx:.1f}_cy{cy:.1f}.png"
        else:
            out_name = f"{stem}_cxNone_cyNone.png"
        out_path = output_dir / rel.parent / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(out_path)
        results.append((src, thr, centroid, out_name))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subtract a reference image from each image, threshold-denoise, "
        "save with centroid coordinates in filename."
    )
    parser.add_argument("--ref", required=True, type=Path, help="Reference image path")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/md_test/md_img-100v"),
        help="Input directory (recursively scanned)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/md_test/md_img-100v_diff"),
        help="Output directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Dark-blob threshold on diff (= reference - image). Default: 15 "
        "(empirically the per-channel DM signal sits at |diff| 15-25)",
    )
    parser.add_argument(
        "--cmap",
        choices=("gray", "jet"),
        default="jet",
        help="Output colormap: jet (default, black background + blue->red "
        "difference enhancement) or gray (plain |diff|)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fixed jet color scale maximum (default: per-image max |diff|, "
        "floored at threshold+1). Use for a comparable scale across images",
    )
    parser.add_argument(
        "--notch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FFT-notch the RAW reference and image to remove fixed-pattern "
        "fringes before diffing (default: on). Disable with --no-notch",
    )
    args = parser.parse_args()

    if not args.ref.is_file():
        sys.exit(f"Reference image not found: {args.ref}")

    results = process(args.ref, args.input, args.output, args.threshold,
                      cmap=args.cmap, vmax=args.vmax, notch=args.notch)

    from collections import Counter

    n_none = sum(1 for _, _, c, _ in results if c is None)
    print(f"Processed {len(results)} images -> {args.output}")
    if n_none:
        print(f"  {n_none} images had no pixels above threshold (saved as _cxNone_cyNone)")

    # Quick sanity overview: distinct centroids
    centroids = [c for _, _, c, _ in results if c is not None]
    if centroids:
        arr = np.array(centroids)
        spread = arr.max(axis=0) - arr.min(axis=0)
        print(f"  centroid x range: {arr[:,0].min():.1f}..{arr[:,0].max():.1f} "
              f"(spread {spread[0]:.1f} px)")
        print(f"  centroid y range: {arr[:,1].min():.1f}..{arr[:,1].max():.1f} "
              f"(spread {spread[1]:.1f} px)")
        # distinct centroid count per controller
        per_dir: Counter[str] = Counter()
        for src, _, c, _ in results:
            if c is not None:
                per_dir[src.parent.name] += 1
        for name, cnt in sorted(per_dir.items()):
            print(f"    {name}: {cnt} distinct-pixel centroids")


if __name__ == "__main__":
    main()
