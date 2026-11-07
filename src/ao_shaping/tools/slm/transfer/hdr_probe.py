"""HDR 探测显示: 多曝光融合图, 供实时画面色调映射展示。

与 :mod:`ao_shaping.utils.image.hdr` 的分工: 后者是纯 NumPy 融合 (无硬件);
本模块是**采集侧**的薄包装 —— 给定已打开的相机, 调 ``cam.capture_hdr(...)``
拿宽动态 ``HdrResult.fused`` (float32, 线性 radiance × 参考曝光), 供
``live_view.tone_map_hdr`` 重新映射回 0..255 uint8 展示。

为什么单独一个模块而不是直接在采集循环里调 ``capture_hdr``: 这是**探测**开关,
任何失败 (相机不支持 HDR、曝光阶梯非法、融合异常) 都只记 warning 并返回
``None``, 采集继续走单帧 —— 探测绝不能拖垮一次记录采集。

Qt-free by construction (与 ``live_frame`` 同一约定): 本模块在**测量线程**被
``validate`` 的采集循环调用, 不碰任何 GUI 工具包。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from loguru import logger

__all__ = [
    "DEFAULT_HDR_EXPOSURES_MS",
    "capture_hdr_display",
]

#: 默认曝光阶梯 (ms), 与 ``scripts/slm/capture_hdr_metrics.py`` 的约定一致:
#: 最短档保峰值, 最长档救暗区。阶梯要够到暗区 (最暗目标区在最长档下 ≳10 counts)。
DEFAULT_HDR_EXPOSURES_MS: tuple[float, ...] = (1.0, 4.0, 16.0)


def capture_hdr_display(
    cam: Any,
    exposures_ms: Sequence[float] = DEFAULT_HDR_EXPOSURES_MS,
    *,
    n_sample: int = 1,
) -> npt.NDArray[np.float32] | None:
    """多曝光融合的 HDR 显示图 (``HdrResult.fused``), 失败返回 ``None``。

    只做显示增强: 调用方 (``validate`` 采集循环, ``args.hdr`` 开启时) 用它生成
    payload 的 ``hdr_image``; 采集记录 (``frame``) 仍走单帧 ``get_numpy_image``,
    不受影响。任何异常都吞掉并记 warning —— HDR 探测是 best-effort, 不能让
    一次记录采集因为显示增强而失败。

    参数:
        cam: 已打开的相机 (实现 ``capture_hdr`` 的 ``BaseCamera`` 子类)。
        exposures_ms: 曝光阶梯 (ms), 至少两档。
        n_sample: 每档采样帧数 (``>1`` 时驱动内部平均)。

    返回:
        ``HdrResult.fused`` 的 float32 副本 (线性 radiance × 参考曝光, 宽动态),
        或 ``None`` (相机不支持 / 融合失败 / 无有效像素)。
    """
    try:
        result = cam.capture_hdr(list(exposures_ms), n_sample=n_sample)
    except Exception as exc:  # 探测绝不冒泡
        logger.warning("HDR display capture failed: {}", exc)
        return None
    fused = getattr(result, "fused", None)
    if fused is None:
        return None
    arr = np.asarray(fused, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr
