"""图像重采样通用工具.

提供保持宽高比 (aspect-ratio) 的裁剪与双线性缩放方法, 用于将相机图像
(不同分辨率/宽高比) 重采样到目标网格 (如 SLM 相位网格), 避免整帧直接
缩放产生的比例失真.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def crop_to_aspect(
    img: np.ndarray,
    aspect_ratio: float,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """按目标宽高比围绕指定中心裁剪图像.

    裁剪后返回的子图宽高比约等于 ``aspect_ratio`` (宽/高), 中心点保留在
    子图内. 当原始宽高比已接近目标宽高比时不裁剪或仅裁剪最小长度.

    Args:
        img: 输入 2D 图像 (H, W).
        aspect_ratio: 目标宽高比 ``w / h`` (>0).
        center: 裁剪中心 ``(cy, cx)``; 默认 ``None`` 取图像中心.

    Returns:
        裁剪后的 2D 数组, 形状 (crop_h, crop_w).
    """
    ih, iw = img.shape
    if aspect_ratio <= 0:
        raise ValueError(f"aspect_ratio must be > 0, got {aspect_ratio}")

    if center is None:
        cy, cx = ih // 2, iw // 2
    else:
        cy, cx = int(center[0]), int(center[1])

    current_ar = iw / ih
    if iw / ih > aspect_ratio:
        # 图像过宽: 裁剪宽度以匹配目标宽高比
        crop_w = max(1, min(iw, int(ih * aspect_ratio)))
        x0 = max(0, min(iw - crop_w, int(cx - crop_w / 2)))
        return img[:, x0 : x0 + crop_w]
    if current_ar < aspect_ratio:
        # 图像过高: 裁剪高度以匹配目标宽高比
        crop_h = max(1, min(ih, int(iw / aspect_ratio)))
        y0 = max(0, min(ih - crop_h, int(cy - crop_h / 2)))
        return img[y0 : y0 + crop_h, :]
    return img


def resample_to_grid(
    img: np.ndarray,
    target_shape: tuple[int, int],
    center: tuple[int, int] | None = None,
    order: int = 1,
) -> np.ndarray:
    """将图像重采样到目标网格尺寸, 保持宽高比 (裁剪 + 双线性缩放).

    流程: 以给定中心 (默认光斑峰值/图像中心) 按目标宽高比裁剪, 再双线性
    缩放至 ``target_shape``. 若输入形状已与目标一致则原样返回 (不复制).

    Args:
        img: 输入 2D 图像 (H, W).
        target_shape: 目标形状 ``(height, width)``.
        center: 裁剪中心 ``(cy, cx)``; 默认 ``None`` 时取图像峰值位置
            (``argmax``), 保证高亮光斑位于裁剪窗口中心.
        order: ``scipy.ndimage.zoom`` 插值阶数, 1=双线性 (默认).

    Returns:
        重采样后的 2D float64 数组, 形状 ``target_shape``.

    Raises:
        ValueError: 输入非 2D 或 ``target_shape`` 非正整数.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"img must be 2D, got {img.ndim}D")
    if len(target_shape) != 2 or target_shape[0] < 1 or target_shape[1] < 1:
        raise ValueError(f"target_shape must be (h, w) positive ints, got {target_shape}")

    th, tw = target_shape
    ih, iw = img.shape
    if (ih, iw) == (th, tw):
        return img

    if center is None:
        peak_y, peak_x = np.unravel_index(int(np.argmax(img)), img.shape)
        center = (int(peak_y), int(peak_x))

    target_ar = tw / th
    crop = crop_to_aspect(img, target_ar, center=center)

    zoom_y, zoom_x = th / crop.shape[0], tw / crop.shape[1]
    return np.asarray(zoom(crop, (zoom_y, zoom_x), order=order), dtype=np.float64)