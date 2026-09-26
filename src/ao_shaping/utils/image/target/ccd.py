"""CCD-frame driven targets: crop/resize, arbitrary-centre masks and the
frame -> SLM-grid pipeline.

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ao_shaping.utils.image.spots_calc import centroid
from ao_shaping.utils.image.target.patterns import create_target_shape


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


_SHAPE_ALIASES: dict[str, str] = {
    "圆形": "circle",
    "方形": "square",
    "长方形": "rectangle",
    "ellipse": "circle",  # not separately supported; map to circle
    "annulus": "annular",  # common spelling alias
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
            return create_target_shape("square", (height, width), side=params["side"])
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


def build_target_from_frame(
    frame: np.ndarray,
    shape: Literal["square", "circle", "rectangle"],
    params: dict,
    grid_h: int = 1200,
    grid_w: int = 1920,
    *,
    center_method: Literal["argmax", "centroid", "centroid_thresh"] = "argmax",
) -> tuple[np.ndarray, dict]:
    """Build an exposure-invariant target from a raw CCD frame.

    This is the single entry point used by both the hardware closed-loop
    runner (``shaping_test.py``) and the SLM calibration GUI
    (``target_shape_helper``). It wraps :func:`build_ccd_target` +
    :func:`crop_resize_to_grid` and anchors the shape on the measured beam
    centre.

    Exposure/brightness invariant by construction:

    1. Background = 10th percentile of the frame, subtracted.
    2. Beam centre located by ``center_method`` (default ``"argmax"`` — the
       project rule: 0-order = frame global max, never assume it sits at the
       frame/ROI centre). ``"centroid"`` / ``"centroid_thresh"`` are provided
       for the GUI comparison view.
    3. The CCD-space mask is built with :func:`build_ccd_target`: value =
       ``1 / n_pixels`` so the target sums to exactly 1 regardless of
       sub-pixel centroid alignment.
    4. The CCD-space mask is mapped to the SLM grid via
       :func:`crop_resize_to_grid` and peak-normalised to ``[0, 1]``.

    Args:
        frame: Raw far-field CCD frame of the current beam (flat phase).
        shape: ``"square"`` (params ``side``), ``"circle"``
            (params ``radius``) or ``"rectangle"`` (params ``rect_w`` /
            ``rect_h``).
        params: Shape parameters (see ``shape``).
        grid_h: SLM grid height (pixels).
        grid_w: SLM grid width (pixels).
        center_method: Centre-finding method — ``"argmax"`` (default, robust
            to stray-light halo), ``"centroid"`` or ``"centroid_thresh"``.

    Returns:
        ``(target_intensity, info)`` — grid-space float32 target in ``[0, 1]``
        for the optimization algorithms, plus a dict with keys:
        ``target_ccd`` (CCD-space float32 normalised mask, sum == 1, same
        shape as ``frame``), ``centroid`` ``[cy, cx]``, ``n_pixels``,
        ``max_brightness``, ``background``, ``total_intensity``,
        ``side_cam_px`` (the characteristic dimension in CCD px, for logging),
        ``side_grid_bins`` ``[rows, cols]`` (mapped grid extent).

    Raises:
        ValueError: If no usable signal remains after background subtraction
            or ``shape``/``params`` are invalid.
    """
    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_h, frame_w = frame.shape

    bg = float(np.percentile(frame, 10))
    signal = np.clip(frame - bg, 0.0, None)
    smax = float(signal.max())
    total = float(signal.sum())
    if smax <= 0 or total <= 0:
        raise ValueError("实测帧无有效信号 (去背景后总和/峰值 <= 0) — 请检查曝光或光束")

    # Beam centre — argmax is the project default (robust to the stray-light
    # halo that drags the intensity centroid hundreds of px away). The GUI
    # may opt for centroid / thresholded-centroid for visual comparison.
    if center_method == "argmax":
        cy, cx = np.unravel_index(np.argmax(signal), signal.shape)
    elif center_method == "centroid":
        cx, cy = centroid(signal, moment=1, threshold=0.0, return_float=True)
    elif center_method == "centroid_thresh":
        cx, cy = centroid(signal, moment=1, threshold=0.1, return_float=True)
    else:
        raise ValueError(f"未知的中心计算方法: {center_method}")
    cy, cx = int(cy), int(cx)

    target_ccd = build_ccd_target(
        frame_shape=frame.shape,
        shape=shape,
        params=params,
        center=(float(cx), float(cy)),
    )

    target = crop_resize_to_grid(target_ccd, grid_h, grid_w)
    tmax = target.max()
    if tmax > 0:
        target = target / tmax
    target = target.astype(np.float32)

    # Characteristic CCD-space dimension for logging.
    if shape == "square":
        side_cam_px = float(params["side"])
    elif shape == "circle":
        side_cam_px = float(params["radius"]) * 2.0
    else:  # rectangle
        side_cam_px = float(params["rect_w"])

    crop_h = min(frame_h, round(frame_w * grid_h / grid_w))
    crop_w = round(crop_h * grid_w / grid_h)
    tmax_ccd = float(target_ccd.max())
    info = {
        "side_cam_px": side_cam_px,
        "target_ccd": target_ccd,
        "n_pixels": int(round(1.0 / tmax_ccd)) if tmax_ccd > 0 else 0,
        "max_brightness": smax,
        "background": bg,
        "total_intensity": total,
        "centroid": [cy, cx],
        "side_grid_bins": [
            side_cam_px * grid_h / crop_h,
            side_cam_px * grid_w / crop_w,
        ],
    }
    return target, info
