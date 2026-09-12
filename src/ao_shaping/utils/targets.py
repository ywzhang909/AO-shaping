"""Target image generation utilities for beam shaping (SLM grid / CCD space).

Migrated from ``ao_shaping.algorithm.beam_shaping_utils`` (target-pattern
generation family) and ``ao_shaping.gui.ccd.target_shape_helper`` (mask
generation) so that "how to build a target" lives in one pure-NumPy module.

Scope
-----
- ``create_target_shape`` / ``create_target_mask`` / ``load_target_image``:
  parametric and image-based target intensity patterns on the SLM grid.
- ``compute_square_side`` / ``build_square_target_amplitude``: square-target
  sizing and amplitude masks for GS-style shaping.
- ``crop_resize_to_grid`` / ``square_target_from_measurement``:
  CCD-frame driven targets (0-order located by frame-global ``argmax``;
  exposure/brightness invariant).
- ``generate_target_mask`` / ``build_ccd_target``: binary / energy-normalised
  (sum == 1) masks centred at an arbitrary pixel position — used by the SLM
  calibration GUI and by hardware closed-loop targets.

All functions are pure (no hardware access, no logging): the module is a leaf
of ``ao_shaping.utils`` and must not import from ``algorithm`` / ``drivers`` /
``optimizer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Parametric target pattern generation (SLM grid space)
# ---------------------------------------------------------------------------
def create_target_shape(
    shape: Literal[
        "gaussian", "circle", "square", "annular", "grid", "cross",
        "rectangle", "pentagon",
    ],
    size: int | tuple[int, int],
    radius_ratio: float = 0.3,
    side: int | None = None,
    as_amplitude: bool = False,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    line_width: float = 0.02,
    thickness: float = 0.05,
    aspect_ratio: float = 2.0,
) -> np.ndarray:
    """Create a 2D target intensity or amplitude pattern.

    Args:
        shape: Pattern type. ``"gaussian"``, ``"circle"``, ``"square"``,
            ``"annular"`` (ring), ``"grid"`` (grid lines), ``"cross"``,
            ``"rectangle"`` (aspect-ratio rectangle, long axis horizontal),
            or ``"pentagon"`` (regular pentagon, point-up).
        size: Side length of the square output array, or ``(height, width)``
            tuple for a rectangular grid (e.g. the SLM panel 1200x1920).
        radius_ratio: Fraction of half-size defining the base radius
            (used for gaussian sigma, circle radius, and annular defaults).
        side: Side length in grid pixels. For ``"square"`` it is the full
            side; for ``"rectangle"`` it is the SHORT side (the long side is
            ``side * aspect_ratio``). Defaults to the radius-derived size.
        as_amplitude: If True, return ``sqrt(intensity)`` instead of
            intensity. Useful for Gerchberg-Saxton style algorithms that
            operate on amplitude masks.
        inner_radius: Inner radius of an annular ring (pixels). Defaults to
            ``0.2 * radius``.
        outer_radius: Outer radius of an annular ring (pixels). Defaults to
            ``0.5 * min(height, width) / 2``.
        nx: Number of vertical grid lines. Only used when ``shape == "grid"``.
        ny: Number of horizontal grid lines. Only used when ``shape == "grid"``.
        line_width: Grid line width as a fraction of half-size. Only used when
            ``shape == "grid"`` or ``shape == "cross"``.
        thickness: Cross arm half-width as a fraction of half-size. Only used
            when ``shape == "cross"``.
        aspect_ratio: Width:height ratio of the rectangle (long axis
            horizontal). Only used when ``shape == "rectangle"``.

    Returns:
        Float32 array of shape ``(height, width)`` with values in ``[0, 1]``
        (intensity by default, or amplitude when ``as_amplitude=True``).

    Raises:
        ValueError: If ``shape`` is not recognized.
    """
    if isinstance(size, tuple):
        height, width = int(size[0]), int(size[1])
    else:
        height = width = int(size)

    y, x = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    radius = radius_ratio * min(height, width) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if shape == "gaussian":
        sigma = radius / 2
        pattern = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    elif shape == "circle":
        pattern = (r <= radius).astype(np.float32)
    elif shape == "square":
        half = (side if side is not None else int(radius)) / 2.0
        pattern = ((np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)).astype(
            np.float32
        )
    elif shape == "annular":
        inner_r = float(inner_radius if inner_radius is not None else 0.2 * radius)
        outer_r = float(
            outer_radius if outer_radius is not None else 0.5 * min(height, width) / 2
        )
        pattern = ((r >= inner_r) & (r <= outer_r)).astype(np.float32)
    elif shape == "grid":
        nx = int(nx if nx is not None else 5)
        ny = int(ny if ny is not None else 5)
        lw = float(line_width) * min(height, width) / 2
        pattern = np.zeros((height, width), dtype=np.float32)
        for i in range(nx):
            x_pos = cx + (2 * i / max(nx - 1, 1) - 1) * (width / 2)
            pattern[np.abs(x - x_pos) <= lw] = 1.0
        for j in range(ny):
            y_pos = cy + (2 * j / max(ny - 1, 1) - 1) * (height / 2)
            pattern[np.abs(y - y_pos) <= lw] = 1.0
    elif shape == "cross":
        th = float(thickness) * min(height, width) / 2
        pattern = ((np.abs(x - cx) <= th) | (np.abs(y - cy) <= th)).astype(np.float32)
    elif shape == "rectangle":
        # ``side`` is the SHORT side; the long side (horizontal) is
        # ``side * aspect_ratio``. The boolean mask clips to grid bounds.
        short_side = side if side is not None else int(radius)
        half_h = short_side / 2.0
        half_w = short_side * aspect_ratio / 2.0
        pattern = ((np.abs(x - cx) <= half_w) & (np.abs(y - cy) <= half_h)).astype(
            np.float32
        )
    elif shape == "pentagon":
        # Regular pentagon, point-up (one vertex at top), circumradius
        # ``radius``. Vectorized convex-polygon point-in-polygon test over the
        # 5 edges (half-plane checks) — no matplotlib dependency.
        angles = -np.pi / 2 + 2 * np.pi * np.arange(5) / 5
        vx = cx + radius * np.cos(angles)
        vy = cy + radius * np.sin(angles)
        inside = np.ones((height, width), dtype=bool)
        for i in range(5):
            x1, y1 = vx[i], vy[i]
            x2, y2 = vx[(i + 1) % 5], vy[(i + 1) % 5]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            inside &= cross >= 0
        pattern = inside.astype(np.float32)
    else:
        raise ValueError(
            f"Unknown shape: {shape}. Use 'gaussian', 'circle', 'square', "
            "'annular', 'grid', 'cross', 'rectangle', or 'pentagon'."
        )

    if as_amplitude:
        return np.sqrt(pattern).astype(np.float32)
    return (
        (pattern / pattern.max()).astype(np.float32) if pattern.max() > 0 else pattern
    )


def create_target_mask(
    shape: str,
    grid_size: tuple[int, int],
    size: int,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Create a normalised target intensity mask.

    Unified target generation helper that supports all shapes used by both the
    differentiable beam-shaping runners (``diff_beam_runner`` /
    ``diff_shaping_runner``). Shapes ``square``, ``circle``, ``gaussian`` and
    ``spot`` are supported.

    Args:
        shape: One of ``"square"``, ``"circle"``, ``"gaussian"``, ``"spot"``.
        grid_size: ``(H, W)`` of the output array.
        size: Characteristic dimension in pixels.
            * square — side length
            * circle — diameter
            * gaussian — ``sigma`` defaults to ``size / 6`` if not given
            * spot — diameter (focused spot, same as circle)
        sigma: Override for the Gaussian standard deviation (pixels).

    Returns:
        ``(H, W)`` ``float64`` mask with values in ``[0, 1]``, centred.
    """
    valid_shapes = {"square", "circle", "gaussian", "spot"}
    if shape not in valid_shapes:
        raise ValueError(f"Invalid shape {shape!r}. Must be one of {valid_shapes}")
    if len(grid_size) != 2:
        raise ValueError(f"grid_size must be a 2-tuple, got {len(grid_size)}D")

    H, W = grid_size
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W]

    if shape == "square":
        half = size / 2.0
        mask = ((np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)).astype(
            np.float64
        )

    elif shape in {"circle", "spot"}:
        radius = size / 2.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (r2 <= radius**2).astype(np.float64)

    else:  # gaussian
        sig = sigma if sigma is not None else size / 6.0
        mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2))
        peak = mask.max()
        if peak > 0:
            mask = mask / peak

    return mask.astype(np.float64)


def load_target_image(path: str | Path) -> np.ndarray:
    """Load a grayscale image as a normalized float target intensity.

    The file may be a ``.npy`` (array) or an image (``.png``/``.jpg``/...).
    The result is normalized to ``[0, 1]`` and cast to float32.

    Args:
        path: File path to the target image or ``.npy`` array.

    Returns:
        Float32 2D array with values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target image not found: {path}")

    if path.suffix.lower() == ".npy":
        target = np.load(path)
    else:
        try:
            from skimage import io  # type: ignore

            target = io.imread(str(path))
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "scikit-image required to load image targets. "
                "Install with: pip install scikit-image"
            ) from exc

    if target.ndim > 2:
        target = target[..., 0] if target.shape[-1] > 0 else target.mean(axis=-1)
    target = target.astype(np.float32)
    tmax = target.max()
    if tmax > 0:
        target = target / tmax
    return target


# ---------------------------------------------------------------------------
# Square beam shaping (GS): spot measurement -> square sizing -> target
# ---------------------------------------------------------------------------
def compute_square_side(
    spot_diameter_cam_px: float,
    factor: float = 1.5,
    p_cam: float = 8e-6,
    d_slm: float = 8e-6,
) -> int:
    """Auto-compute square side (SLM-grid pixels) from the beam spot size.

    The requested physical beam size is ``factor`` times the measured spot
    diameter. If the camera pixel pitch differs from the SLM pixel pitch the
    size is rescaled accordingly::

        side = factor * spot_diameter_cam * (p_cam / d_slm)

    Args:
        spot_diameter_cam_px: Measured spot diameter in camera pixels.
        factor: Square-to-spot size factor (default 1.5).
        p_cam: Camera pixel pitch in meters (default matches SLM pitch).
        d_slm: SLM pixel pitch in meters (default 8e-6).

    Returns:
        Square side length in SLM-grid pixels.
    """
    return int(round(float(factor) * float(spot_diameter_cam_px) * (p_cam / d_slm)))


def build_square_target_amplitude(
    height: int,
    width: int,
    side: int,
) -> np.ndarray:
    """Build a centered square target amplitude on an (height, width) grid.

    Args:
        height: Grid height in pixels.
        width: Grid width in pixels.
        side: Square side length in pixels.

    Returns:
        Float array (height, width) with 1 inside the square, 0 outside.
    """
    target = np.zeros((height, width), dtype=np.float64)
    half = side // 2
    cy = height // 2
    cx = width // 2
    y0 = max(cy - half, 0)
    y1 = min(cy + side - half, height)
    x0 = max(cx - half, 0)
    x1 = min(cx + side - half, width)
    target[y0:y1, x0:x1] = 1.0
    return target


# ---------------------------------------------------------------------------
# CCD frame -> SLM-grid target (0-order located by argmax)
# ---------------------------------------------------------------------------
def _resize_bilinear(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Separable bilinear resize with ``np.interp``-style pixel-center mapping.

    Pure-NumPy replacement for ``skimage.transform.resize`` so the runner does
    not depend on a (fragile on this machine) scikit-image build. Output pixel
    centers are mapped to source coordinates as ``(i + 0.5) * src / dst - 0.5``
    and clamped, matching the standard coordinate convention.
    """
    src_h, src_w = arr.shape
    dst_h, dst_w = out_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return arr.copy()

    ys = (np.arange(dst_h) + 0.5) * src_h / dst_h - 0.5
    xs = (np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.0)
    xs = np.clip(xs, 0.0, src_w - 1.0)

    y0 = np.floor(ys).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wy = (ys - y0)[:, None]
    interp_rows = arr[y0, :] * (1.0 - wy) + arr[y1, :] * wy

    x0 = np.floor(xs).astype(np.intp)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wx = (xs - x0)[None, :]
    return interp_rows[:, x0] * (1.0 - wx) + interp_rows[:, x1] * wx


def crop_resize_to_grid(
    frame: np.ndarray,
    grid_h: int = 1200,
    grid_w: int = 1920,
) -> np.ndarray:
    """Crop a CCD frame to the SLM grid aspect ratio and resize to the grid.

    The 0-order spot is located as the frame global argmax (never assumed to
    be the geometric frame center). The crop is the largest centered rectangle
    with aspect ratio ``grid_h/grid_w`` that fits in the frame, zero-padded if
    it extends past the frame bounds, then resized to the full SLM grid.
    """
    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_h, frame_w = frame.shape
    cy, cx = np.unravel_index(np.argmax(frame), frame.shape)

    crop_h = min(frame_h, round(frame_w * grid_h / grid_w))
    crop_w = min(round(crop_h * grid_w / grid_h), frame_w)

    y0 = cy - crop_h // 2
    x0 = cx - crop_w // 2
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    y0c = max(y0, 0)
    y1c = min(y1, frame_h)
    x0c = max(x0, 0)
    x1c = min(x1, frame_w)

    crop = frame[y0c:y1c, x0c:x1c]
    if crop.shape != (crop_h, crop_w):
        pad_top = y0c - y0
        pad_bottom = (crop_h - crop.shape[0]) - pad_top
        pad_left = x0c - x0
        pad_right = (crop_w - crop.shape[1]) - pad_left
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )

    return _resize_bilinear(crop, (grid_h, grid_w)).astype(np.float32)


def square_target_from_measurement(
    frame: np.ndarray,
    side_px: float,
    grid_h: int = 1200,
    grid_w: int = 1920,
) -> tuple[np.ndarray, dict]:
    """Build a uniform-square grid target with a FIXED side in CCD space.

    Exposure/brightness invariant by construction:

    1. Background = 10th percentile of the frame, subtracted (signal stats).
    2. Beam center located at the **frame global maximum (argmax)** of the
       signal — the 0-order spot. This is the project rule (AGENTS.md: locate
       the 0-order by ``argmax``, never by geometry or centroid), and is the
       only method that survives the stray-light halo on the full 2592x1944
       Daheng frame (measured: intensity centroid dragged 150-450 px off;
       argmax stable within ~4 px across runs). Anchors on the *current*
       0-order spot, never on the frame center.
    3. A uniform CCD-space square of **fixed** side ``side_px`` camera pixels
       is centered on the centroid with value ``1 / n_pixels`` (n_pixels =
       actual filled pixel count, so the square sums to exactly 1 regardless
       of sub-pixel centroid alignment).
    4. The CCD-space square is mapped to the SLM grid via
       ``crop_resize_to_grid`` and normalized to ``[0, 1]``.

    Evaluation is exposure-invariant: compare ``frame / frame.sum()`` against
    ``info["target_ccd"]`` (sum == 1) — both sides are normalized, so changing
    exposure or total brightness never changes the target square.

    Args:
        frame: Raw far-field CCD frame of the current beam (flat phase).
        side_px: Fixed square side length in camera pixels (e.g. 20 for a
            20×20 px square on the CCD).
        grid_h: SLM grid height (pixels).
        grid_w: SLM grid width (pixels).

    Returns:
        ``(target_intensity, info)`` — grid-space float32 target in ``[0, 1]``
        for the optimization algorithms, plus a dict: ``side_cam_px`` (fixed
        side in CCD px), ``target_ccd`` (CCD-space float32 normalized square,
        sum == 1, same shape as ``frame`` — use as the evaluation target),
        ``n_pixels`` (actual filled pixel count), ``max_brightness``,
        ``background``, ``total_intensity``, ``centroid``
        ``[cy, cx]`` and ``side_grid_bins`` ``[rows, cols]``.

    Raises:
        ValueError: If no usable signal remains after background subtraction
            or ``side_px`` is not positive.
    """
    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_h, frame_w = frame.shape
    if side_px <= 0:
        raise ValueError("side_px 必须 > 0")

    bg = float(np.percentile(frame, 10))
    signal = np.clip(frame - bg, 0.0, None)

    total = float(signal.sum())
    smax = float(signal.max())
    if smax <= 0 or total <= 0:
        raise ValueError("实测帧无有效信号 (去背景后总和/峰值 <= 0) — 请检查曝光或光束")

    # Center: 0-order spot located by frame GLOBAL MAX (argmax), per the
    # project rule (AGENTS.md): "0-order = frame global max, never assume it
    # sits at the frame/ROI center". Hardware-verified 2026-09-10 on the
    # 2f bench: the real spot sits at argmax (row,col) ≈ (705,1781) / (701,1780)
    # (two runs, stable), while the intensity centroid is dragged 150-450 px
    # away by pervasive stray light (6.9-7.3% of pixels nonzero, halo energy
    # >> spot) and the corner-max binary centroid drifts 26 px+ at higher
    # exposure. argmax is the only method that pins the true spot reliably.
    #
    # NOTE: do NOT use the full-frame intensity centroid here. Measured on
    # this bench it lands hundreds of px away from the real spot because the
    # constant-offset background removal (10th percentile) cannot subtract the
    # *spatially structured* stray-light halo, whose total energy dominates
    # the moments. The Runner's per-frame recorder calls this same function
    # indirectly via ``centroid`` only for *logging* — do not confuse the log
    # value with the actual spot location.
    #
    # 2026-09-11 flat-phase frame (1200 us, full 2592x1944, saved as
    # data/diff_beam/flat_tiff_20260911/flat_1200us.tiff/.npy) — quantitative
    # reason the centroid fails and why argmax survives:
    #   * 96.37% of pixels are 0; the remaining 3.51% are a *single gray step*
    #     of value 1 spread over EVERY row and column, carrying 62.46% of the
    #     frame energy. Median == 0, so any median/percentile-based background
    #     subtraction leaves this layer untouched and the intensity centroid is
    #     dragged to (834,1578) vs true spot argmax (689,1776).
    #   * Threshold-sweep centroid: for t>=2 the centroid of ``img > t``
    #     instantly converges to (694-695, 1776-1778) and stays within ~1 px up
    #     to t=200. I.e. masking ONLY the gray==1 layer is enough to make any
    #     intensity centroid accurate — but argmax needs no threshold choice at
    #     all, so it remains the robust default.
    #   * corner-max binary centroid fails here too: the gray==1 layer reaches
    #     the frame corners, so a corner-derived threshold (1.0) cannot filter
    #     it and the centroid still lands (715,1834).
    cy, cx = np.unravel_index(np.argmax(signal), signal.shape)
    cy, cx = int(cy), int(cx)

    # Fixed-side square in *camera* space, centered at the measured centroid.
    # Value = 1 / n_pixels (actual filled pixel count): the square sums to
    # exactly 1 (normalized target) no matter how the sub-pixel centroid maps
    # onto the pixel grid. Exposure/brightness changes only scale the raw
    # frame, never the normalized target.
    side = float(np.clip(side_px, 1.0, min(frame_h, frame_w)))

    half = side / 2.0
    y0 = max(0, int(np.floor(cy - half)))
    y1 = min(frame_h, int(np.ceil(cy + half)))
    x0 = max(0, int(np.floor(cx - half)))
    x1 = min(frame_w, int(np.ceil(cx + half)))
    square_ccd = np.zeros_like(frame)
    n_pixels = (y1 - y0) * (x1 - x0)
    if n_pixels > 0:
        square_ccd[y0:y1, x0:x1] = 1.0 / n_pixels

    target = crop_resize_to_grid(square_ccd, grid_h, grid_w)
    tmax = target.max()
    if tmax > 0:
        target = target / tmax
    target = target.astype(np.float32)

    crop_h = min(frame_h, round(frame_w * grid_h / grid_w))
    crop_w = round(crop_h * grid_w / grid_h)
    info = {
        "side_cam_px": side,
        "target_ccd": square_ccd.astype(np.float32),
        "n_pixels": n_pixels,
        "max_brightness": smax,
        "background": bg,
        "total_intensity": total,
        "centroid": [cy, cx],
        "side_grid_bins": [side * grid_h / crop_h, side * grid_w / crop_w],
    }
    return target, info


# ---------------------------------------------------------------------------
# Arbitrary-centre masks (binary / energy-normalised)
# ---------------------------------------------------------------------------
# 兼容 target_shape_helper 的中文 shape key:-GUI 界面语言
_SHAPE_ALIASES: dict[str, str] = {
    "圆形": "circle",
    "方形": "square",
    "长方形": "rectangle",
}


def generate_target_mask(
    shape_type: str,
    params: dict,
    center: tuple[float, float] | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Generate a binary target mask (float32, 0/1) centred at ``center``.

    Extracted from ``target_shape_helper`` (``_generate_target`` +
    ``_make_rectangle``). Supports circles, squares and rectangles at an
    arbitrary pixel position (sub-pixel centre allowed).

    Args:
        shape_type: ``"circle"`` / ``"square"`` / ``"rectangle"``, or the
            Chinese GUI keys ``"圆形"`` / ``"方形"`` / ``"长方形"`` for
            backward compatibility with the SLM calibration UI.
        params: Shape parameters — ``radius`` (circle),
            ``side`` (square), or ``rect_w``/``rect_h`` (rectangle).
        center: ``(cx, cy)`` pixel position, or ``None`` to centre the shape
            on the frame (parametric, via :func:`create_target_shape`).
        height: Output mask height (rows).
        width: Output mask width (cols).

    Returns:
        ``float32`` binary mask (0/1).
    """
    shape = _SHAPE_ALIASES.get(shape_type, shape_type)

    if center is None:
        if shape == "circle":
            radius = params["radius"]
            radius_ratio = radius / (min(height, width) / 2)
            return create_target_shape(
                "circle", (height, width), radius_ratio=radius_ratio
            )
        if shape == "square":
            return create_target_shape(
                "square", (height, width), side=params["side"]
            )
        # rectangle — fall through to manual generation
        return _rectangle_mask(params["rect_w"], params["rect_h"], None, height, width)

    cx, cy = center

    if shape == "circle":
        radius = params["radius"]
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        return mask.astype(np.float32)

    if shape == "square":
        side = params["side"]
        half = side / 2.0
        target = np.zeros((height, width), dtype=np.float32)
        y0 = max(0, int(cy - half))
        y1 = min(height, int(cy + half))
        x0 = max(0, int(cx - half))
        x1 = min(width, int(cx + half))
        target[y0:y1, x0:x1] = 1.0
        return target

    # rectangle
    return _rectangle_mask(params["rect_w"], params["rect_h"], center, height, width)


def _rectangle_mask(
    rw: int,
    rh: int,
    center: tuple[float, float] | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Generate a rectangle mask of size (rh, rw) centred at ``center``."""
    if center is None:
        y0 = max(0, (height - rh) // 2)
        x0 = max(0, (width - rw) // 2)
    else:
        cx, cy = center
        y0 = max(0, int(cy - rh / 2))
        x0 = max(0, int(cx - rw / 2))
    y1 = min(height, y0 + rh)
    x1 = min(width, x0 + rw)
    target = np.zeros((height, width), dtype=np.float32)
    target[y0:y1, x0:x1] = 1.0
    return target


def build_ccd_target(
    frame_shape: tuple[int, int],
    shape: Literal["square", "circle", "rectangle"],
    params: dict,
    center: tuple[float, float],
) -> np.ndarray:
    """Build an energy-normalised CCD-space target (sum == 1).

    Exposure/brightness invariant by construction: the mask value is
    ``1 / n_pixels`` (actual filled pixel count) so the target sums to exactly
    1 regardless of how the floating-point centre maps onto the pixel grid —
    the same normalisation used by :func:`square_target_from_measurement`.

    Args:
        frame_shape: ``(height, width)`` of the CCD frame.
        shape: ``"square"`` (params ``side``), ``"circle"``
            (params ``radius``) or ``"rectangle"`` (params ``rect_w`` /
            ``rect_h``).
        params: Shape parameters (see ``shape``).
        center: ``(cx, cy)`` spot centre in camera pixels.

    Returns:
        ``float32`` frame-shaped mask with values ``[0, 1]``, sum == 1
        (or all-zero if the shape parameter is degenerate).
    """
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"frame_shape 必须为正, got {(height, width)}")

    target = np.zeros((height, width), dtype=np.float32)
    cx, cy = center

    if shape == "square":
        side = float(np.clip(params["side"], 1.0, min(height, width)))
        half = side / 2.0
        y0 = max(0, int(np.floor(cy - half)))
        y1 = min(height, int(np.ceil(cy + half)))
        x0 = max(0, int(np.floor(cx - half)))
        x1 = min(width, int(np.ceil(cx + half)))
    elif shape == "circle":
        radius = float(np.clip(params["radius"], 1.0, min(height, width)))
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        n_pixels = int(mask.sum())
        if n_pixels > 0:
            target[mask] = 1.0 / n_pixels
        return target
    elif shape == "rectangle":
        rw = float(np.clip(params["rect_w"], 1.0, width))
        rh = float(np.clip(params["rect_h"], 1.0, height))
        y0 = max(0, int(np.floor(cy - rh / 2)))
        y1 = min(height, int(np.ceil(cy + rh / 2)))
        x0 = max(0, int(np.floor(cx - rw / 2)))
        x1 = min(width, int(np.ceil(cx + rw / 2)))
    else:
        raise ValueError(
            f"Unknown shape {shape!r}. Use 'square', 'circle' or 'rectangle'."
        )

    n_pixels = (y1 - y0) * (x1 - x0)
    if n_pixels > 0:
        target[y0:y1, x0:x1] = 1.0 / n_pixels
    return target