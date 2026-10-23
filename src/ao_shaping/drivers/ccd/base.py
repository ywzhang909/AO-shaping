"""相机驱动基类接口。"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from loguru import logger


class AveragedFrame(np.ndarray):
    """Float camera counts with the source pixel format's saturation level."""

    full_scale: float

    def __new__(cls, values: np.ndarray, source_dtype: np.dtype) -> "AveragedFrame":
        frame = np.asarray(values, dtype=np.float32).view(cls)
        frame.full_scale = float(np.iinfo(source_dtype).max)
        return frame

    def __array_finalize__(self, obj: object) -> None:
        self.full_scale = getattr(obj, "full_scale", 255.0)


class CameraError(Exception):
    """相机错误异常的基类。"""

    pass


class BaseCamera(ABC):
    """相机驱动的抽象基类。

    所有相机驱动都应继承本类并实现其中的抽象方法。本类提供上下文管理器支持,
    以便自动清理资源。

    单位与约定
    ---------------------
    * 曝光时间: Python 层统一用**毫秒 (ms)**。
      部分 SDK (大恒 gxipy、MIICAM) 内部用**微秒 (µs)**;
      各实现必须在 SDK 边界换算 (写入时 `ms * 1000 -> µs`,
      读取时 `µs / 1000 -> ms`)。
    * 亮度: 原始像素值; dtype 取决于位深
      (通常是 ``uint8`` 或 ``uint16``)。
    * ROI 窗口: ``(width, height)`` 与 ``(center_x, center_y)``, 单位像素。
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
    ):
        """初始化相机配置。

        参数:
            cam_id: 相机设备索引。
            exposure_time_ms: 初始曝光时间, 单位毫秒。
                各驱动共同的合法范围: ``0.011 ms`` 到
                ``10000 ms``。硬件支持时也允许亚毫秒值。
            skip_sampling: 是否启用 binning/skipping 以加快采集。
        """
        self.cam_id = int(cam_id)
        self.exposure_time_ms = float(exposure_time_ms)
        self.skip_sampling = skip_sampling

        self.cam = None
        self._sn: str | None = None
        self.cam_width: int = 0
        self.cam_height: int = 0

    def __enter__(self):
        """上下文管理器入口 —— 初始化相机。"""
        self.initialize()
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器出口 —— 清理相机资源。

        各实现应关闭相机并释放资源。
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """初始化相机设备。

        本方法应当:
        1. 关闭此前已打开的相机设备。
        2. 按 cam_id 查找并打开指定相机。
        3. 设置初始曝光时间、增益与像素格式。
        4. 若 skip_sampling 为 True, 配置分辨率与 binning。
        5. 启动数据流。

        异常:
            ConnectionAbortedError: 未找到相机设备。
            CameraError: 相机初始化失败。
        """
        pass

    @abstractmethod
    def open(self) -> None:
        """打开相机设备 (initialize 的别名)。

        这是给不使用上下文管理器的调用方准备的显式方法。
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭相机设备并释放资源。"""
        pass

    @abstractmethod
    def reset_exposure_time(self, time_ms: float) -> float:
        """设置相机曝光时间。

        曝光时间在 Python 层用**毫秒 (ms)** 表示。写入硬件时, 各实现可能需要先
        换算成 SDK 的原生单位 (通常是微秒)。

        部分相机 (如 MIICAM) 在拉流期间改曝光需要走 ``Stop -> set -> Start``
        循环; 改动后的第一帧可能是陈旧帧, 因此调用方应丢弃它
        (``skip_first=True``)。

        参数:
            time_ms: 新的曝光时间, 单位毫秒。
                典型合法范围: ``0.011 ms`` 到 ``10000 ms``。
                超出硬件范围的值应被钳制。

        返回:
            float: 实际设定的曝光时间, 单位毫秒。

        异常:
            AssertionError: 相机未初始化。
        """
        pass

    @abstractmethod
    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """重设相机 ROI 窗口的大小与位置。

        参数:
            center: 期望的窗口中心位置 (x, y)。
            size: 期望的窗口大小 (width, height)。
                用 (0, 0) 表示最大分辨率。

        返回:
            实际设定的 ((width, height), (center_x, center_y))。

        异常:
            AssertionError: 相机未初始化, 或 center 非法。
        """
        pass

    @abstractmethod
    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """采集相机图像, 可选做平均。

        参数:
            n_sample: 平均的采样数。必须 > 0。
            skip_first: 是否跳过第一帧。这在改曝光或改 ROI 之后尤其有用,
                因为返回的第一帧可能是陈旧或不稳定的。

        返回:
            np.ndarray: 单帧保留相机原始类型；多帧平均保留小数灰度值
            (``float32``)，不提前量化为整数。

        异常:
            AssertionError: n_sample 不是正数。
        """
        pass

    @abstractmethod
    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光。

        参数:
            enable: True 启用, False 关闭。
            mode: 自动曝光模式 (各实现自定义)。

        返回:
            bool: 成功返回 True, 不支持返回 False。
        """
        pass

    @abstractmethod
    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态。

        返回:
            含以下内容的字典:
                - enabled: bool —— 自动曝光是否已启用
                - mode: int —— 当前模式
                - target: int —— 当前目标亮度
        """
        pass

    @abstractmethod
    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光的时间与增益范围。

        参数:
            max_time_ms: 最大曝光时间, 单位 ms。
            min_time_ms: 最小曝光时间, 单位 ms。
            max_gain: 最大增益值。
            min_gain: 最小增益值。

        返回:
            bool: 成功返回 True, 不支持返回 False。
        """
        pass

    @staticmethod
    @abstractmethod
    def get_cam_list():
        """获取可用相机设备列表。

        返回:
            可用相机设备列表 (格式各实现自定义)。
        """
        pass

    def _get_grid(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """为图像生成坐标网格。

        参数:
            width: 图像宽度。
            height: 图像高度。

        返回:
            (xv, yv) meshgrid 数组构成的 tuple。
        """
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv

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

        original: float | None = None
        try:
            original = common.get_camera_exposure_ms(self)
        except AttributeError:
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

        from ao_shaping.drivers.ccd import common
        from ao_shaping.utils.image.hdr import hdr_fuse

        exposures, samples = self._validated_ladder(exposures_ms, n_sample)
        frames, applied, _ = self._acquire_exposure_ladder(exposures, samples,
                                                           restore_exposure=restore_exposure)
        detector_full_scale = float(common.full_scale(frames[0]))
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
            ``common.full_scale`` 同源。

        异常:
            ValueError: ``exposures_ms`` 少于两档, 或含非有限 / 非正的曝光,
                或 ``n_sample < 1``。
        """
        # ``common`` 在 common.py 里 ``from ao_shaping.drivers.ccd.base import
        # BaseCamera``, 所以模块作用域的 import 会构成循环依赖、直接炸掉
        # ``import ao_shaping.drivers.ccd``。仅为此理由延后到函数体内。
        from ao_shaping.drivers.ccd import common

        # ``utils/`` 从不 import ``drivers/``, 因此这里放模块作用域是合法的 ——
        # 但本模块刻意只 import abc / numpy / loguru, 保持导入面最小; 同理延后。
        from ao_shaping.utils.image.exposure_fusion import exposure_fusion

        exposures, samples = self._validated_ladder(exposures_ms, n_sample)
        frames, applied, _ = self._acquire_exposure_ladder(exposures, samples,
                                                           restore_exposure=restore_exposure)

        detector_full_scale = float(common.full_scale(frames[0]))
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
