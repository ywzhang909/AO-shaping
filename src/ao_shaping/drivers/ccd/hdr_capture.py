"""HDR capture operations shared by camera backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ao_shaping.drivers.ccd.base import AveragedFrame


class HdrCaptureMixin:
    def _validated_ladder(self, exposures_ms: Sequence[float], n_sample: int) -> tuple[list[float], int]:
        """校验曝光阶梯与每档采样数，返回 (曝光列表, 采样数)。"""
        exposures = [float(value) for value in exposures_ms]
        if len(exposures) < 2:
            raise ValueError(f"exposure ladder needs at least two exposures, got {exposures!r}")
        for index, value in enumerate(exposures):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"exposures_ms[{index}] must be a finite positive float, got {value!r}")
        samples = int(n_sample)
        if samples < 1:
            raise ValueError(f"n_sample must be >= 1, got {n_sample!r}")
        return exposures, samples

    def _acquire_exposure_ladder(self, exposures: Sequence[float], samples: int, *,
                                 restore_exposure: bool = True) -> tuple[list[np.ndarray], list[float], float | None]:
        """按曝光阶梯采集：返回 (每档帧, **实际生效**的曝光, 进入时的曝光)。

        只依赖基类的 ``reset_exposure_time`` / ``get_numpy_image``，因此所有后端
        （daheng / miicam / sim / ffmpeg / image_folder）零改动可用。约定：

        * 每档固定 ``skip_first=True`` —— 改曝光后的首帧可能是陈旧帧；
        * 记录**实际生效**的曝光（后端会把越界值钳到设备范围），融合必须吃实际值；
        * ``restore_exposure`` 时在 ``finally`` 里写回进入时的曝光（异常路径同样恢复）。
        """
        from ao_shaping.drivers.ccd import common
        from ao_shaping.drivers.ccd.base import CameraError

        original: float | None = None
        try:
            original = common.get_camera_exposure_ms(self)
        except (AttributeError, CameraError):
            logger.warning(
                "{} exposes no readable exposure time; leaving the camera at the "
                "last ladder exposure {:.4f} ms", type(self).__name__, float(exposures[-1]))
        frames: list[np.ndarray] = []
        applied: list[float] = []
        try:
            for ms in exposures:
                common.set_camera_exposure_ms(self, ms)
                if original is None:
                    applied.append(float(ms))
                else:
                    effective = float(common.get_camera_exposure_ms(self))
                    applied.append(effective)
                    if abs(effective - float(ms)) > 1e-9:
                        logger.warning(
                            "exposure {} ms was clamped to {} ms; the fusion normalises "
                            "this frame by the value actually used", ms, effective)
                frames.append(self.get_numpy_image(samples, skip_first=True))
        finally:
            if restore_exposure and original is not None:
                common.set_camera_exposure_ms(self, original)
        return frames, applied, original

    def capture_hdr(
        self,
        exposures_ms: Sequence[float],
        *,
        n_sample: int = 1,
        restore_exposure: bool = True,
        dark: float = 0.0,
        read_noise_counts: float = 0.0,
        min_snr: float = 0.0,
        reference_exposure_ms: float | None = None,
    ):
        """多曝光阶梯采集并融合成**辐射度** HDR 图（counts/ms），与显示融合分开。

        与 :meth:`capture_fused` 共用同一套采集契约（每档丢首帧、读回实际曝光、
        退出恢复曝光），区别只在融合语义：

        * :meth:`capture_fused` 走 ``utils.image.exposure_fusion`` 的拉普拉斯金字塔，
          输出**显示参考**的 0–255 图 —— 观感好，但幅度被权重混合改写，不能用来
          判"目标区均值是否超过噪声地板"。
        * 本方法走 ``utils.image.hdr.hdr_fuse``，输出**线性 radiance**（每毫秒计数）
          + ``valid`` / ``source_index`` / ``coverage``，可直接喂
          ``utils.image.hdr.box_metrics`` 做 CV / EE / 填充率 / 可测性判定。

        参数:
            exposures_ms: 曝光阶梯（ms，至少两档）。**阶梯要够到暗区**：最暗的目标区
                在最长一档下需要 ≳10 counts，否则该区域只会得到噪声整流值。
            n_sample: 每档采样帧数（``>1`` 时驱动内部平均）。
            restore_exposure: 退出时是否恢复进入时的曝光。
            dark: 暗电平（counts），从每帧减去。
            read_noise_counts: 读出噪声（counts），用于扣除噪声整流偏置 σ·√(2/π)。
            min_snr: 只接受 ``原始计数 >= min_snr×σ`` 的像素；不满足的标无效。
            reference_exposure_ms: 融合图 ``fused`` 使用的参考曝光；缺省取最短一档。

        返回:
            ``HdrResult``（``radiance`` / ``fused`` / ``valid`` / ``source_index`` /
            ``coverage`` / ``saturated_pixels`` / ``invalid_pixels``），并带上本次采集
            的 ``full_scale`` 与 ``applied_exposures_ms`` 便于事后核对。

        异常:
            ValueError: 阶梯少于两档、含非有限/非正曝光，或 ``n_sample < 1``。
        """
        from dataclasses import replace

        from ao_shaping.utils.image.hdr import hdr_fuse

        exposures, samples = self._validated_ladder(exposures_ms, n_sample)
        frames, applied, _ = self._acquire_exposure_ladder(exposures, samples,
                                                           restore_exposure=restore_exposure)
        detector_full_scale = float(self.effective_full_scale(frames[0]))
        result = hdr_fuse(frames, applied, dark=dark, full_scale=detector_full_scale,
                          read_noise_counts=read_noise_counts, min_snr=min_snr,
                          reference_exposure_ms=reference_exposure_ms)
        return replace(result, full_scale=detector_full_scale,
                       applied_exposures_ms=tuple(applied))

    def capture_fused(
        self,
        exposures_ms: Sequence[float],
        *,
        n_sample: int = 1,
        restore_exposure: bool = True,
    ) -> AveragedFrame:
        """多曝光阶梯采集, 并融合成一张高动态范围合成图。

        依次把相机切到 ``exposures_ms`` 的每一档曝光、每档采一帧, 再交给
        ``ao_shaping.utils.image.exposure_fusion.exposure_fusion`` 的
        ``"laplacian_pyramid"`` 输出融合: 短曝光保住长曝光里被削平的层次, 长曝光
        补回短曝光的信噪比。本方法只用基类已有的抽象接口, 所以 daheng / miicam /
        ffmpeg / image_folder / sim 各后端**零改动**即可调用。

        约定与不变量
        ---------------------
        * **单位**: ``exposures_ms`` 与相机当前曝光同为**毫秒 (ms)**。写曝光统一
          走 ``common.set_camera_exposure_ms``, 它优先调用 ``reset_exposure_time``
          (MIICAM 的 Stop→设置→重启拉流必需)。
        * **每档丢弃第一帧**: 改曝光后的第一帧可能是**陈旧帧**
          (``reset_exposure_time`` 的文档即记录了 MIICAM 的这种情况), 所以每一档
          都固定 ``skip_first=True``, 不交给调用方决定。
        * **退出时恢复调用方的曝光**: 无论采集还是融合抛异常, ``finally`` 都会把
          曝光写回进入本方法时的值。``restore_exposure=False`` 时相机停在阶梯的
          **最后一档**。若后端压根没有可读的曝光属性, 跳过恢复并记一条 warning。
        * **输出量纲**: 与输入帧完全相同的 0-255 探测器计数 (**不是** ``[0, 1]``),
          dtype 为 ``float32``, 取值落在 ``[0, full_scale]`` 内。
        * **曝光按实际生效值归一化**: 后端可能把越界的曝光钳到设备范围, 所以每档
          都读回硬件**实际**的值再交给融合, 而不是用请求值 (否则被钳的那一档权重
          会整体偏掉)。发生钳位时会记一条 warning。

        参数:
            exposures_ms: 曝光阶梯, 单位毫秒, 至少两档, 全部为有限正数。真实值会
                原样传给融合算法 —— 少了它, 良曝光项会让长曝光帧以十倍以上的平均
                权重独自胜出, 融合退化成单帧。
            n_sample: 每档的采样帧数, 必须 ``>= 1``。``> 1`` 时
                :meth:`get_numpy_image` 返回 ``AveragedFrame`` (float32), 融合
                照常进行。
            restore_exposure: 退出时是否把曝光恢复成进入本方法前的值。

        返回:
            AveragedFrame: 融合结果。``full_scale`` 是**实测**的探测器饱和值
            (uint8 后端 255, uint16 后端 65535), 取自采集到的第一帧, 与
            :meth:`effective_full_scale` 同源。

        异常:
            ValueError: ``exposures_ms`` 少于两档, 或含非有限 / 非正的曝光,
                或 ``n_sample < 1``。
        """
        # ``utils/`` 从不 import ``drivers/``, 因此这里放模块作用域是合法的 ——
        # 但本模块刻意只 import abc / numpy / loguru, 保持导入面最小; 同理延后。
        from ao_shaping.drivers.ccd.base import AveragedFrame
        from ao_shaping.utils.image.exposure_fusion import exposure_fusion

        exposures, samples = self._validated_ladder(exposures_ms, n_sample)
        frames, applied, _ = self._acquire_exposure_ladder(exposures, samples,
                                                           restore_exposure=restore_exposure)

        #: 同 capture_hdr_ladder: 12-bit 后端的原始单帧没有 AveragedFrame 标注,
        #: 按 dtype (uint16) 会得到 65535, 满量程须按本相机**生效位深**解析 (issue #246)。
        detector_full_scale = float(self.effective_full_scale(frames[0]))
        fused = exposure_fusion(
            frames,
            exposures_ms=applied,
            full_scale=detector_full_scale,
        )["laplacian_pyramid"]

        # ``AveragedFrame.__new__`` 要读 ``np.iinfo(source_dtype).max``, 所以只能传
        # **整数** dtype: ``n_sample > 1`` 时 ``get_numpy_image`` 返回的是 float32 的
        # ``AveragedFrame``, 而 ``np.iinfo(np.float32)`` 会抛 ValueError。
        source_dtype = np.dtype(np.uint8 if detector_full_scale <= 255.0 else np.uint16)
        result = AveragedFrame(fused, source_dtype)
        # iinfo 的猜测只对两种标准位深恰好等于真实饱和值; 14 bit 之类 (16383) 没有
        # 对应的标准 dtype, 因此把实测值显式带上。
        result.full_scale = detector_full_scale
        return result
