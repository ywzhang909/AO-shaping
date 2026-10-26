"""多曝光图像融合 (Mertens / Kautz / Van Reeth, PG'07 + CGF 2009)。

纯 NumPy 复刻经典的「三种测度 / 三种融合」曝光融合算法, 位于
:mod:`ao_shaping.utils.image` 叶子层: 无硬件访问、无 ``cv2``、无 ``scipy`` /
``skimage``。卷积与双线性重采样全部手写, 与
:func:`ao_shaping.utils.image.target.metrics._loggrad_blur` /
:func:`ao_shaping.utils.image.target.ccd._resize_bilinear` 保持同一约定。

三层结构
-----------------
1. **Stage 0 —— 曝光归一化** (:func:`exposure_fusion`): 探测器计数与曝光时间
   成正比, 所以每帧必须先除以自己的曝光再算权重。
2. **Stage 1 —— 三种测度** (:func:`compute_fusion_weights`): 对比度、饱和度、
   良曝光度, 按 ``alphas`` 相乘后跨帧归一化。
3. **Stage 2/3 —— 金字塔与三种输出** (:func:`gaussian_pyramid`,
   :func:`laplacian_pyramid`, :func:`exposure_fusion`): 朴素平均 / 权重高斯平滑
   / 多分辨率拉普拉斯金字塔。

单位契约
-----------------
本仓库全程使用 **0-255 灰度「探测器计数」**, 不是 ``[0, 1]``
(见 :data:`DETECTOR_FULL_SCALE`)。内部一律 float64, 返回一律 float32 且已
乘回 ``full_scale``, 因此调用方拿到的就是原始量纲。

单色决策 (最关键的一条, 不要「修」它)
-----------------
对 **2-D 输入完全跳过饱和度**。2-D 帧的「跨通道标准差」恒为 0; 一旦照公式算,
每帧的原始权重都是 0, 归一化时 ``eps`` 会把它们抬成**完全相同**的值, 于是每个
像素都拿到同一个 ``1/K`` 权重 —— 融合静默退化成整组曝光的**算术平均**, 完全没有
逐像素选帧。所以 2-D 只用对比度 + 良曝光度; 3-D 才计算真正的彩色饱和度。见
:func:`compute_fusion_weights`。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from loguru import logger

__all__ = [
    "DETECTOR_FULL_SCALE",
    "compute_fusion_weights",
    "exposure_fusion",
    "gaussian_pyramid",
    "laplacian_pyramid",
]


#: Full-scale detector value, in the 0-255 gray units the whole repo uses.
#:
#: Duplicates ``ao_shaping.drivers.ccd.common.DETECTOR_FULL_SCALE`` on purpose:
#: ``utils/`` is a leaf layer and may not import ``drivers/`` at module scope
#: (enforced by ``tests/ao_shaping/test_conventions.py``). Keep the two in sync.
DETECTOR_FULL_SCALE: float = 255.0


#: The 4-neighbour Laplacian, ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]``.
#:
#: This matrix is **rank 2, not rank 1** -- the often-quoted "it is separable,
#: just run ``[-1, 2, -1]`` twice" is a different operator (that one produces
#: ``[[1, -2, 1], [-2, 4, -2], [1, -2, 1]]``, an 8-neighbour Laplacian). Since
#: this module's contract is the literal kernel above, it is applied as a direct
#: 2-D correlation in :func:`_convolve2d` rather than via the separable path.
LAPLACIAN_KERNEL_2D: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


#: Anti-aliasing width, in pixels, of the low-pass applied before each pyramid
#: decimation. 1.0 px is the usual Burt-Adelson pre-filter order of magnitude and
#: is small enough that the low-pass stays close to the Burt-Adelson identity
#: used to validate the band construction.
PYRAMID_SIGMA: float = 1.0

#: numpy rank of a spatial stack in this module, i.e. ``(C, H, W)``. Every filter here takes
#: either that rank or a bare map, and the rank test is how the two are told apart -- it
#: counts axes, not channels.
NDIM_SPATIAL_STACK = 3

#: numpy rank of a single-channel map, i.e. ``(H, W)``. Luminance measures, fusion weights
#: and per-level corrections are computed on this rank. Kept separate from
#: NDIM_SPATIAL_STACK rather than merged into one rank constant.
NDIM_SINGLE_CHANNEL = 2


#: Width of the hard weight-map blur used by the ``"gaussian_smoothed"`` blend,
#: as a fraction of ``min(height, width)``.
#:
#: The reference implementation hard-codes a 101-tap kernel with ``sigma = 17``
#: at full resolution; 17 / 256 = 0.0664 = 1/15, so ``min(H, W) / 15`` reproduces
#: it at 256 px while scaling to any frame size. Floored at 1.0 px so a small
#: frame still gets a real (if narrow) blur.
WEIGHT_BLUR_SIGMA_RATIO: float = 1.0 / 15.0


# ---------------------------------------------------------------------------
# Pure-NumPy convolution / resampling kernels
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalised 1D Gaussian kernel; ``sigma == 0`` is the identity kernel.

    Mirrors :func:`ao_shaping.utils.image.target.metrics._loggrad_kernel` -- same
    ``ceil(3 * sigma)`` support, same normalisation. Hand-rolled rather than
    pulled from ``scipy.ndimage`` for the same reason that module documents:
    ``utils/`` is a leaf and must not gain a ``scipy`` import to save a line.
    """
    if sigma == 0.0:
        return np.ones(1, dtype=np.float64)
    radius = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _fit_kernel(kernel: np.ndarray, height: int, width: int) -> np.ndarray:
    """Trim a 1D kernel until ``np.pad(mode="reflect")`` is legal.

    ``mode="reflect"`` requires the pad width to be **smaller** than the axis it
    pads, so a level only a couple of pixels across cannot take the full radius.
    The kernel is shortened (and re-normalised) rather than the padding changed,
    which keeps the sliding-window offsets -- and therefore the output geometry --
    exactly as they were.
    """
    while kernel.size > 1 and (kernel.size // 2) >= min(height, width):
        trimmed = kernel[1:-1]
        total = float(trimmed.sum())
        kernel = trimmed / total if total > 0.0 else trimmed
    return kernel


def _blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable convolution with reflect padding, pure NumPy.

    Hand-rolled on purpose: this is a leaf of :mod:`ao_shaping.utils`, so it
    must not import ``scipy.ndimage`` to save ~10 lines. Keep it that way.

    Accepts ``(H, W)`` or ``(C, H, W)``; a 3-D input is filtered per channel
    along the two trailing axes.
    """
    spatial = img if img.ndim == NDIM_SPATIAL_STACK else img[None, ...]
    height, width = int(spatial.shape[-2]), int(spatial.shape[-1])
    kernel = _fit_kernel(kernel, height, width)
    radius = (kernel.size - 1) // 2
    pad_hw = ((0, 0), (radius, radius), (radius, radius))

    padded = np.pad(spatial, pad_hw, mode="reflect")
    rows = np.zeros_like(spatial)
    for offset, weight in enumerate(kernel):
        rows += weight * padded[..., :height, offset : offset + width]
    padded = np.pad(rows, pad_hw, mode="reflect")
    out = np.zeros_like(spatial)
    for offset, weight in enumerate(kernel):
        out += weight * padded[..., offset : offset + height, :width]
    return out if img.ndim == NDIM_SPATIAL_STACK else out[0]


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct 2-D correlation with reflect padding, pure NumPy.

    Used for the (non-separable) 3x3 Laplacian of :data:`LAPLACIAN_KERNEL_2D`
    only; every band-limiting blur goes through the separable :func:`_blur`.
    """
    radius_y, radius_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    spatial = img if img.ndim == NDIM_SPATIAL_STACK else img[None, ...]
    height, width = int(spatial.shape[-2]), int(spatial.shape[-1])
    if min(height, width) <= max(radius_y, radius_x):
        # ``mode="reflect"`` is illegal at this size; a level this small carries
        # no resolvable Laplacian anyway.
        return np.zeros_like(spatial)

    pad_hw = ((0, 0), (radius_y, radius_y), (radius_x, radius_x))
    padded = np.pad(spatial, pad_hw, mode="reflect")
    out = np.zeros_like(spatial)
    for dy, krow in enumerate(kernel):
        for dx, weight in enumerate(krow):
            out += weight * padded[..., dy : dy + height, dx : dx + width]
    return out if img.ndim == NDIM_SPATIAL_STACK else out[0]


def _resize_bilinear(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Separable bilinear resize with ``np.interp``-style pixel-center mapping.

    Pure-NumPy replacement for ``skimage.transform.resize`` / ``cv2.resize`` so
    the leaf layer needs neither. Output pixel centers are mapped to source
    coordinates as ``(i + 0.5) * src / dst - 0.5`` and clamped, the standard
    coordinate convention.

    This is a deliberate **promote-with-copy** of
    :func:`ao_shaping.utils.image.target.ccd._resize_bilinear`: that helper is
    private to ``target/`` and importing another package's private name would be
    a layering smell. Keep the two bodies in sync.

    Args:
        arr: 2-D ``(H, W)`` or 3-D ``(C, H, W)`` float array; only the two
            trailing axes are resized.
        out_shape: target ``(height, width)``.

    Returns:
        Array with shape ``arr.shape[:-2] + out_shape``.
    """
    lead = arr.shape[:-2]
    src = arr.reshape(-1, *arr.shape[-2:])
    src_h, src_w = src.shape[-2], src.shape[-1]
    dst_h, dst_w = out_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return arr.astype(np.float64, copy=True)

    ys = (np.arange(dst_h) + 0.5) * src_h / dst_h - 0.5
    xs = (np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.0)
    xs = np.clip(xs, 0.0, src_w - 1.0)

    y0 = np.floor(ys).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wy = (ys - y0)[:, None]
    interp_rows = src[..., y0, :] * (1.0 - wy) + src[..., y1, :] * wy

    x0 = np.floor(xs).astype(np.intp)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wx = (xs - x0)[None, :]
    out = interp_rows[..., x0] * (1.0 - wx) + interp_rows[..., x1] * wx
    return out.reshape(*lead, dst_h, dst_w)


def _expand_to(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Bilinear-upsample ``x`` to exactly ``shape`` (no-op if already equal)."""
    if x.shape == tuple(shape):
        return x
    return _resize_bilinear(x, (int(shape[-2]), int(shape[-1])))


def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Coerce to float64 and replace non-finite samples with ``0.0``.

    Non-finite camera samples are an expected condition, not a caller bug, so
    they are scrubbed rather than rejected -- same convention as
    :func:`ao_shaping.utils.image.target.metrics._sanitize_frame` and
    :func:`ao_shaping.utils.image.beam_metrics.prepare_frame`. Shape mistakes
    still raise.
    """
    out = np.asarray(arr, dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _frame_full_scale(frame: np.ndarray, default: float) -> float:
    """Resolve one frame's saturation level in grey units.

    Duck-typed mirror of ``ao_shaping.drivers.ccd.common.full_scale``, inlined
    because ``utils/`` may not import ``drivers/``: a frame may carry its own
    ``.full_scale`` attribute (``drivers.ccd.base.AveragedFrame`` does, and it
    keeps the *source* pixel format's limit after averaging into float), else an
    integer dtype gives its own ``iinfo`` limit, else the caller's ``default``.

    Note this is the *per-frame* saturation level. The ``full_scale`` keyword of
    :func:`exposure_fusion` is the **output** unit system, which must match the
    frames the caller hands in.
    """
    attr = getattr(frame, "full_scale", None)
    if attr is not None:
        return float(attr)
    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    return float(default)


# ---------------------------------------------------------------------------
# Stage 1: the three weight measures
# ---------------------------------------------------------------------------


def _validate_alphas(alphas: Sequence[float]) -> tuple[float, float, float]:
    """Coerce ``alphas`` to exactly three finite floats.

    Returns a literal 3-tuple (not a variadic one) so the width check also
    narrows the type for the caller.

    Raises:
        ValueError: If ``alphas`` does not have exactly three finite entries.
    """
    values = tuple(float(value) for value in alphas)
    if len(values) != 3:
        raise ValueError(
            "alphas must be (contrast, saturation, well_exposedness), got "
            f"{len(values)} entries: {values!r}"
        )
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"alphas entries must be finite, got {values!r}")
    return values[0], values[1], values[2]


def _measure_weights(
    x: np.ndarray,
    alphas: tuple[float, float, float],
    best_illumination: float,
    sigma: float,
) -> np.ndarray:
    """Contrast / saturation / well-exposedness measures of one normalised frame.

    Args:
        x: ``(H, W)`` or ``(C, H, W)`` frame already in ``[0, 1]`` grey units.
        alphas: ``(contrast, saturation, well_exposedness)`` exponents.
        best_illumination: target grey level for the well-exposedness Gaussian.
        sigma: width of that Gaussian, in normalised grey units.

    Returns:
        ``(H, W)`` float64 array of **unnormalised** weights.
    """
    alpha_c, alpha_s, alpha_e = (float(a) for a in alphas)

    # (C, H, W) -> (H, W): the luminance the contrast measure is defined on.
    gray = x if x.ndim == NDIM_SINGLE_CHANNEL else x.mean(axis=0)
    contrast = np.abs(_convolve2d(gray, LAPLACIAN_KERNEL_2D))

    weights = np.power(contrast, alpha_c)

    # Monochrome decision (see the module docstring): the saturation term
    # sqrt(mean_c((x_c - mean_c(x))**2)) is identically ZERO on 2-D input.
    # Omit the factor rather than multiplying a zero in -- and note that the
    # ``eps`` normalisation floor does NOT rescue that: every frame's raw weight
    # is 0, so ``W += eps`` makes them all identical and ``W /= W.sum`` hands
    # every pixel the same 1/K. The "fusion" would silently degrade to a plain
    # average of the bracket, with no per-pixel frame selection at all. Do not
    # reintroduce the factor to "fix" a mono regression.
    if x.ndim == NDIM_SPATIAL_STACK:
        saturation = np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))
        weights = weights * np.power(saturation, alpha_s)

    gaussian = np.exp(-0.5 * ((x - best_illumination) ** 2) / (sigma * sigma))
    # Product over CHANNELS. ``np.prod(g, axis=None)`` would flatten a 2-D frame
    # into a scalar (and that scalar is exp(-O(10^3)) for a dim bracket, i.e. an
    # exact zero), so the mono case must NOT go through ``np.prod`` at all.
    well_exposed = np.prod(gaussian, axis=0) if x.ndim == NDIM_SPATIAL_STACK else gaussian
    return weights * np.power(well_exposed, alpha_e)


def compute_fusion_weights(
    frame: np.ndarray,
    *,
    alphas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    best_illumination: float = 0.5,
    sigma: float = 0.2,
) -> np.ndarray:
    """Unnormalised fusion weights of a **single** frame, as a 2-D ``(H, W)`` map.

    ``W = C^alpha_c * S^alpha_s * E^alpha_e`` evaluated on
    ``frame / DETECTOR_FULL_SCALE``. There is nothing to normalise against -- a
    single frame always has weight 1 after the stack normalisation done in
    :func:`exposure_fusion` -- so this returns the raw product.

    2-D input is **monochrome**: the saturation factor is omitted entirely,
    because ``sqrt(mean_c((x_c - mean_c(x))**2))`` is identically 0 for one
    channel. Keeping it would not zero the weight map outright -- the ``eps``
    floor would lift every frame's weight to the same value -- but the result
    would be a uniform 1/K blend at every pixel, i.e. a plain average of the
    bracket with no per-pixel frame selection. 3-D input uses the real colour
    saturation. The result is ``(H, W)`` either way.

    Non-finite samples are scrubbed to ``0.0`` rather than rejected, matching the
    rest of ``utils/image/``. Input is divided by
    :data:`DETECTOR_FULL_SCALE` because this entry point has no ``full_scale``
    parameter; :func:`exposure_fusion` resolves each frame's own scale and
    calls :func:`_measure_weights` directly.

    Args:
        frame: ``(H, W)`` monochrome or ``(C, H, W)`` colour frame in 0-255
            detector counts (the unit system the whole repo uses).
        alphas: ``(contrast, saturation, well_exposedness)`` exponents.
        best_illumination: grey level the well-exposedness Gaussian is centred
            on; Mertens uses ``0.5``.
        sigma: width of that Gaussian in normalised grey units; Mertens uses
            ``0.2``.

    Returns:
        ``(H, W)`` float64 weights, ``>= 0``.

    Raises:
        ValueError: If ``sigma`` is not finite and positive, or ``frame`` is not
            2-D or 3-D.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be a finite positive float, got {sigma!r}")

    arr = np.asarray(frame)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"frame must be 2-D (mono) or 3-D (C, H, W), got {arr.ndim}-D shape "
            f"{arr.shape}"
        )
    x = _sanitize(arr) / DETECTOR_FULL_SCALE
    return _measure_weights(
        x, _validate_alphas(alphas), float(best_illumination), sigma
    )


# ---------------------------------------------------------------------------
# Stage 2: pyramids
# ---------------------------------------------------------------------------


def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid: level 0 is the input, each further level is the previous
    one low-passed and decimated by ``[::2, ::2]``.

    Naive ``[::2, ::2]`` on an odd axis **ceils** (``(65, 64) -> (33, 32)``). That
    is acceptable only because the expand step targets exact shapes (see
    :func:`laplacian_pyramid`); the reconstruction is therefore exact even when
    the decimation lost a row or column.

    Args:
        img: ``(H, W)`` or ``(C, H, W)`` array; converted to float64.
        levels: requested number of levels, ``>= 1``.

    Returns:
        At most ``levels`` arrays, coarsest last. Truncated early if a further
        level would fall below 2 px on either axis (such a level cannot be
        blurred or expanded meaningfully).

    Raises:
        ValueError: If ``levels < 1``.
    """
    levels = int(levels)
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    base = _sanitize(img)
    pyramid = [base]
    kernel = _gaussian_kernel_1d(PYRAMID_SIGMA)
    while len(pyramid) < levels:
        previous = pyramid[-1]
        if min(previous.shape[-2], previous.shape[-1]) < 4:
            break
        nxt = _blur(previous, kernel)[..., ::2, ::2]
        if min(nxt.shape[-2], nxt.shape[-1]) < 1:
            break
        pyramid.append(nxt)
    return pyramid


def laplacian_pyramid(gauss_pyr: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Laplacian pyramid in the Mertens form.

    ``bands[i] = gauss_pyr[i] - expand_to(gauss_pyr[i + 1], gauss_pyr[i].shape)``
    for ``i`` in ``0 .. n - 2``, and ``bands[-1] = gauss_pyr[-1]``.

    **The upsampled operand is the Gaussian level, not the previously upsampled
    result.** Chaining the expansion instead compounds the resampling error once
    per level and destroys the reconstruction identity -- that is the classic
    trap this construction exists to avoid.

    Args:
        gauss_pyr: Non-empty Gaussian pyramid, finest first
            (:func:`gaussian_pyramid` output).

    Returns:
        A list of the same length, same shapes as ``gauss_pyr``.

    Raises:
        ValueError: If ``gauss_pyr`` is empty.
    """
    pyramid = [_sanitize(level) for level in gauss_pyr]
    if not pyramid:
        raise ValueError("gauss_pyr must be a non-empty sequence of levels")

    bands = [
        pyramid[i] - _expand_to(pyramid[i + 1], pyramid[i].shape)
        for i in range(len(pyramid) - 1)
    ]
    bands.append(pyramid[-1])
    return bands


def _collapse(bands: Sequence[np.ndarray]) -> np.ndarray:
    """Rebuild an image from its Laplacian pyramid, coarse-to-fine.

    ``acc = bands[-1]``, then for ``n`` from ``len - 2`` down to ``0``:
    ``acc = bands[n] + expand_to(acc, bands[n].shape)``.

    The operand that gets expanded is the running accumulator; it is the *target
    shape* that comes from ``bands[n]``. This is the exact inverse of
    :func:`laplacian_pyramid`, which conversely expands the Gaussian level -- see
    the note there before "simplifying" either side.
    """
    accumulated = bands[-1]
    for n in range(len(bands) - 2, -1, -1):
        accumulated = bands[n] + _expand_to(accumulated, bands[n].shape)
    return accumulated


def _blend(
    frames: Sequence[np.ndarray],
    weights: np.ndarray,
    full_scale: float,
) -> np.ndarray:
    """``sum_k x_k * W_k``, clipped to ``[0, 1]`` and returned in grey units.

    ``frames`` are normalised (``[0, 1]``-ish) and ``weights`` is the ``(K, H, W)``
    stack that already sums to 1 per pixel. The 2-D weight map is broadcast over
    the channel axis for colour input.
    """
    total = np.zeros_like(frames[0])
    for frame, weight in zip(frames, weights, strict=True):
        # (H, W) -> (1, H, W) so it lines up with a (C, H, W) frame;
        # ``weight[..., None]`` would be (H, W, 1) and misalign the axes instead.
        factor = weight if frame.ndim == NDIM_SINGLE_CHANNEL else weight[None, ...]
        total = total + frame * factor
    return np.clip(total, 0.0, 1.0) * full_scale


# ---------------------------------------------------------------------------
# Stage 3: the three outputs
# ---------------------------------------------------------------------------


def _validate_sequence(sequence: Sequence[np.ndarray]) -> list[np.ndarray]:
    frames = [np.asarray(frame) for frame in sequence]
    if not frames:
        raise ValueError("sequence must contain at least 2 frames, got 0")
    if len(frames) < 2:
        raise ValueError(
            f"sequence must contain at least 2 frames to fuse, got {len(frames)}"
        )
    shape = frames[0].shape
    for index, frame in enumerate(frames):
        if frame.ndim not in (2, 3):
            raise ValueError(
                f"frame {index} must be 2-D (mono) or 3-D (C, H, W), got "
                f"{frame.ndim}-D shape {frame.shape}"
            )
        if frame.shape != shape:
            raise ValueError(
                f"frame {index} has shape {frame.shape}, but frame 0 has {shape}"
            )
    return frames


def _validate_exposures(
    exposures_ms: Sequence[float] | None,
    n_frames: int,
) -> list[float] | None:
    """Validate the Stage-0 exposure bracket, or return ``None`` to skip it."""
    if exposures_ms is None:
        return None
    exposures = [float(value) for value in exposures_ms]
    if len(exposures) != n_frames:
        raise ValueError(
            f"exposures_ms has {len(exposures)} entries but sequence has "
            f"{n_frames} frames"
        )
    for index, value in enumerate(exposures):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"exposures_ms[{index}] must be a finite positive float, got {value!r}"
            )
    return exposures


def _exposure_scales(exposures: Sequence[float]) -> np.ndarray:
    """Per-frame linear gains that land every frame on frame 0's exposure.

    Detector counts scale linearly with exposure time, so dividing frame ``k`` by
    ``exposures[k] / exposures[0]`` puts all frames on a common scale. Mertens'
    well-exposedness term assumes each frame has something near mid-grey; a
    bracketed frame at 0.12 ms peaking at 33/255 has nothing near 0.5, so its
    ``E`` collapses and the whole weight stack ends up on whichever frame *is*
    near mid-grey. This is the Stage-0 fix.
    """
    reference = exposures[0]
    return np.array([reference / value for value in exposures], dtype=np.float64)


def exposure_fusion(
    sequence: Sequence[np.ndarray],
    *,
    exposures_ms: Sequence[float] | None = None,
    alphas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    best_illumination: float = 0.5,
    sigma: float = 0.2,
    eps: float = 1e-12,
    levels: int = 5,
    full_scale: float = DETECTOR_FULL_SCALE,
) -> dict[str, np.ndarray]:
    """Fuse a bracketed exposure sequence into three images.

    Returns the same scene three ways, all in the caller's original grey units:

    * ``"naive"`` -- direct ``sum_k x_k * W_k`` with no smoothing. This is the
      rough result, speckled with black/white dots wherever one frame dominates.
    * ``"gaussian_smoothed"`` -- each weight map hard-blurred (sigma =
      ``min(H, W) / 15``, the reference's ``sigma = 17`` at 256 px), then
      re-normalised and fused.
    * ``"laplacian_pyramid"`` -- a Laplacian pyramid per frame and a Gaussian
      pyramid of the **already-normalised** weights, fused per scale and
      collapsed coarse-to-fine. The best of the three.

    **The stack normalisation in Stage 1 is not optional.** Without
    ``W_k += eps; W_k /= W_k.sum(axis=0)`` the raw weights are ~1e-11 for a
    typical bracket, and the fused output collapses to exactly zero -- the first
    draft of this module omitted it and produced an all-black image.

    Stage 0 (exposure normalisation) runs first: each frame is multiplied by
    ``exposures_ms[0] / exposures_ms[k]`` so all frames reach the weight stage on
    one exposure scale. Pass ``None`` to skip it, which reproduces the
    handheld-style reference behaviour.

    Args:
        sequence: ``>= 2`` frames, all the same shape, ``(H, W)`` or
            ``(C, H, W)``, in 0-255 detector counts. A frame carrying its own
            ``.full_scale`` attribute (``drivers.ccd.base.AveragedFrame``) is
            honoured for the internal normalisation.
        exposures_ms: Per-frame exposure times in milliseconds, same length as
            ``sequence``, all finite and positive; ``None`` skips Stage 0.
        alphas: ``(contrast, saturation, well_exposedness)`` exponents.
        best_illumination: grey level the well-exposedness Gaussian is centred on.
        sigma: width of that Gaussian in normalised grey units.
        eps: Floor added to the weight stack before normalising, so a pixel with
            no evidence anywhere still gets a finite share.
        levels: Maximum number of pyramid levels per output.
        full_scale: The caller's **output** unit system. The returned arrays are
            ``clip(fused, 0, 1) * full_scale``, so pass ``65535`` for a uint16
            bracket. Per-frame *input* normalisation additionally honours a
            frame's own ``.full_scale`` attribute and an integer dtype's
            ``iinfo`` limit.

    Returns:
        ``{"naive": ..., "gaussian_smoothed": ..., "laplacian_pyramid": ...}``,
        each ``float32`` with the input's spatial shape, finite and within
        ``[0, full_scale]``.

    Raises:
        ValueError: If ``sequence`` is empty or shorter than 2 frames, a frame is
            neither 2-D nor 3-D, frames disagree on shape, ``exposures_ms`` has
            the wrong length or a non-finite/non-positive entry, ``levels < 1``,
            or ``sigma`` / ``full_scale`` are non-positive.
    """
    frames = _validate_sequence(sequence)
    exposures = _validate_exposures(exposures_ms, len(frames))

    levels = int(levels)
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be a finite positive float, got {sigma!r}")
    full_scale = float(full_scale)
    if not np.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError(
            f"full_scale must be a finite positive float, got {full_scale!r}"
        )

    scales = _exposure_scales(exposures) if exposures is not None else None
    normalised: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        frame_scale = _frame_full_scale(frame, full_scale)
        if abs(frame_scale - full_scale) > 1e-6 * full_scale:
            logger.warning(
                "frame {} saturates at {} but full_scale={} was declared; the "
                "returned images are in the declared {} units",
                index,
                frame_scale,
                full_scale,
                full_scale,
            )
        x = _sanitize(frame) / frame_scale
        if scales is not None:
            # A pure linear gain, so it commutes with the ``/ full_scale`` step
            # above and could equally have been applied to the raw counts.
            x = x * float(scales[index])
        normalised.append(x)

    exponents = _validate_alphas(alphas)
    weight_stack = np.stack(
        [
            _measure_weights(x, exponents, float(best_illumination), sigma)
            for x in normalised
        ],
        axis=0,
    )
    # NOT OPTIONAL -- see the docstring. This is the line whose absence turns the
    # whole output black.
    weight_stack = weight_stack + float(eps)
    weight_stack = weight_stack / weight_stack.sum(axis=0, keepdims=True)

    height, width = int(frames[0].shape[-2]), int(frames[0].shape[-1])
    # The reference hard-codes sigma = 17 at full resolution; see
    # :data:`WEIGHT_BLUR_SIGMA_RATIO`.
    blur_sigma = max(1.0, min(height, width) * WEIGHT_BLUR_SIGMA_RATIO)

    results = {
        "naive": _blend(normalised, weight_stack, full_scale),
        "gaussian_smoothed": _blend(
            normalised,
            _blur_weight_stack(weight_stack, eps, sigma=blur_sigma),
            full_scale,
        ),
        "laplacian_pyramid": _laplacian_fusion(
            normalised, weight_stack, levels, full_scale
        ),
    }
    return {key: np.asarray(value, dtype=np.float32) for key, value in results.items()}


def _blur_weight_stack(
    weight_stack: np.ndarray,
    eps: float,
    sigma: float = PYRAMID_SIGMA,
) -> np.ndarray:
    """Blur every weight map hard, then re-normalise the stack.

    A normalised kernel applied with reflect padding conserves a partition of
    unity exactly, so the blurred stack already sums to 1 per pixel; the
    re-normalisation is kept anyway so the ``"gaussian_smoothed"`` and
    ``"naive"`` outputs share one stated invariant rather than an argument about
    padding, and so a caller passing a truncated kernel cannot silently break it.
    """
    kernel = _gaussian_kernel_1d(sigma)
    blurred = np.stack(
        [_blur(weight, kernel) for weight in weight_stack], axis=0
    )
    blurred = blurred + float(eps)
    return blurred / blurred.sum(axis=0, keepdims=True)


def _laplacian_fusion(
    frames: Sequence[np.ndarray],
    weight_stack: np.ndarray,
    levels: int,
    full_scale: float,
) -> np.ndarray:
    """Multi-resolution blend: per-scale weighted Laplacian bands, collapsed.

    ``fused[n] = sum_k lap_k[n] * wpyr_k[n]`` (the 2-D weight level is broadcast
    over the channel axis), then coarse-to-fine exactly as
    :func:`_collapse`. The weight pyramid is built from the **already
    normalised** stack, which is what makes the per-scale products average to 1.
    """
    weight_pyramids = [gaussian_pyramid(weight, levels) for weight in weight_stack]
    band_pyramids = [
        laplacian_pyramid(gaussian_pyramid(frame, levels)) for frame in frames
    ]

    n_scales = min(len(bands) for bands in band_pyramids + weight_pyramids)
    fused: list[np.ndarray] = []
    for n in range(n_scales):
        level = np.zeros_like(band_pyramids[0][n])
        for bands, wpyr in zip(band_pyramids, weight_pyramids, strict=True):
            weight = wpyr[n]
            factor = weight if level.ndim == NDIM_SINGLE_CHANNEL else weight[None, ...]
            level = level + bands[n] * factor
        fused.append(level)

    return np.clip(_collapse(fused), 0.0, 1.0) * full_scale