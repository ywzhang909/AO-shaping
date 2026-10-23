"""多曝光 HDR 融合：把同一场景的若干不同曝光帧合成一张线性化图。

**为什么需要**：大目标框（如 1000 px）里，峰值像素在短曝光下已接近饱和，而框内
大部分区域在同一曝光下只有 0–1 counts、落在读出噪声里。单曝光无法同时照顾两者：
短曝光把暗区判成"空"、长曝光把峰值打爆。实测证据（2026-10-23，4 ms，同一光路）：

| 掩膜 | peak | 目标区均值 | 填充率 |
|---|---|---|---|
| 60 px | 111 | 22.3 | 0.20 |
| 1000 px | 147 | 0.377 | 0.0026 |

⇒ 大框的均值落在噪声地板上，任何 CV/EE 都是在噪声上算出来的。

本模块按**每像素选"未饱和的最长曝光"**做融合（越长曝光信噪比越好），除以该曝光
还原成"每毫秒计数"（radiance），再乘以参考曝光给出与参考帧同量纲的融合图。
输出同时给 `valid` 掩膜与 `source_index`，便于事后判定哪些像素是可信的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


class HdrFuseError(ValueError):
    """输入不合法（形状/曝光/数量）。"""


@dataclass(frozen=True)
class HdrResult:
    """融合结果（全部与输入同形状）。"""

    radiance: npt.NDArray[np.floating]      # 每毫秒计数（已减暗电平）
    fused: npt.NDArray[np.floating]         # radiance × reference_exposure_ms
    valid: npt.NDArray[np.bool_]            # 至少一帧未饱和且高于暗电平
    source_index: npt.NDArray[np.int16]     # 该像素取自第几帧（-1 = 无效）
    reference_exposure_ms: float
    saturated_pixels: int                   # 所有帧都饱和的像素数
    invalid_pixels: int                     # 无可用帧的像素数

    @property
    def coverage(self) -> float:
        """有效像素占比。"""
        total = self.valid.size
        return float(self.valid.sum() / total) if total else 0.0


def hdr_fuse(frames: Sequence[npt.NDArray[np.floating]], exposures_ms: Sequence[float], *,
             dark: float = 0.0, full_scale: float = 255.0, saturation_fraction: float = 0.98,
             min_counts: float = 0.0, reference_exposure_ms: float | None = None) -> HdrResult:
    """把多曝光帧融合成线性化结果。

    Args:
        frames: 同一场景的若干帧，形状一致；顺序不限（内部按曝光排序处理）。
        exposures_ms: 与 ``frames`` 一一对应的曝光（ms，必须为正）。
        dark: 暗电平（counts），从每帧减去。
        full_scale: 相机数字满量程（8 bit ⇒ 255，12 bit ⇒ 4095）。
        saturation_fraction: 视作饱和的比例阈值（默认 0.98）。
        min_counts: 低于它视作不可用（例如负值或纯暗）；默认 0。
        reference_exposure_ms: 融合图``fused``所乘的参考曝光；缺省取最短曝光。

    Returns:
        :class:`HdrResult`。

    Raises:
        HdrFuseError: 帧数为空、形状不一致、曝光非正或长度不匹配。
    """
    if len(frames) == 0:
        raise HdrFuseError("frames must not be empty")
    if len(frames) != len(exposures_ms):
        raise HdrFuseError("frames and exposures_ms must have the same length")
    exposures = np.asarray(exposures_ms, dtype=np.float64)
    if not np.all(np.isfinite(exposures)) or np.any(exposures <= 0):
        raise HdrFuseError("exposures_ms must be finite and positive")
    arrays = [np.asarray(frame, dtype=np.float64) for frame in frames]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise HdrFuseError("all frames must share one shape")

    order = np.argsort(exposures)            # 短 → 长
    saturation_level = float(saturation_fraction) * float(full_scale)

    radiance = np.full(shape, np.nan)
    source = np.full(shape, -1, dtype=np.int16)
    all_saturated = np.ones(shape, dtype=bool)
    for index in order:                      # 依次用更长的曝光补空位 ⇒ 最终取最长的可用
        frame = arrays[index]
        usable = (frame < saturation_level) & ((frame - dark) > float(min_counts))
        all_saturated &= ~(frame < saturation_level)
        radiance = np.where(usable, (frame - dark) / exposures[index], radiance)
        source = np.where(usable, np.int16(index), source)

    valid = source >= 0
    reference = float(reference_exposure_ms) if reference_exposure_ms is not None else float(exposures.min())
    fused = np.where(valid, radiance * reference, np.nan)
    return HdrResult(radiance=radiance, fused=fused, valid=valid, source_index=source,
                     reference_exposure_ms=reference,
                     saturated_pixels=int((all_saturated & ~valid).sum()),
                     invalid_pixels=int((~valid).sum()))


def robust_mean(image: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_], *,
                clip_sigma: float | None = 3.0) -> float:
    """掩膜内的稳健均值（可选按中位数 ± clip_sigma·MAD 剔除离群），用于 HDR 图的目标区均值。"""
    values = np.asarray(image)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if clip_sigma is None:
        return float(values.mean())
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad <= 0:
        # 半数以上取值相同 ⇒ MAD 退化，改用高分位截断（常见于"大片平场 + 少量亮斑"）
        cap = float(np.percentile(values, 99.0))
        keep = values <= cap
        return float(values[keep].mean()) if keep.any() else median
    keep = np.abs(values - median) <= clip_sigma * mad
    return float(values[keep].mean()) if keep.any() else median
