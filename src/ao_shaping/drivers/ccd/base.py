"""相机驱动基类接口。"""

from __future__ import annotations
from abc import ABC, abstractmethod
import threading
from typing import Any, Callable, ClassVar

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.hdr_capture import HdrCaptureMixin


class AveragedFrame(np.ndarray):
    """Float camera counts with the source pixel format's saturation level."""

    full_scale: float

    def __new__(
        cls,
        values: np.ndarray,
        source_dtype: np.dtype,
        full_scale: float | None = None,
    ) -> "AveragedFrame":
        frame = np.asarray(values, dtype=np.float32).view(cls)
        #: 默认按源 dtype 推导满量程; 但**某些后端 dtype 与位深不符**
        #: (大恒 MONO12 帧 dtype 是 uint16, 真实满量程 4095 而非 65535),
        #: 此时必须由后端显式传入生效位深的满量程, 否则一切按满量程折算的
        #: 判据 (自动曝光峰值、饱和保护) 都错量纲。
        frame.full_scale = float(full_scale) if full_scale is not None else float(
            np.iinfo(source_dtype).max
        )
        return frame

    def __array_finalize__(self, obj: object) -> None:
        self.full_scale = getattr(obj, "full_scale", 255.0)


class CameraError(Exception):
    """相机错误异常的基类。"""

    pass


class BaseCamera(HdrCaptureMixin, ABC):
    """相机驱动的抽象基类。

    所有相机驱动都应继承本类并实现其中的抽象方法。本类提供上下文管理器支持,
    以便自动清理资源。

    单位与约定
    ---------------------
    * 曝光时间: Python 层统一用**毫秒 (ms)**。
      部分 SDK (大恒 gxipy、MIICAM) 内部用**微秒 (µs)**;
      各实现必须在 SDK 边界换算 (写入时 `ms * 1000 -> µs`,
      读取时 `µs / 1000 -> ms`)。
    * 曝光契约 (基类提供默认实现, 后端按需覆写):
      :meth:`get_exposure_time_ms` 读**生效**曝光 (有设备回读的后端必须覆写,
      否则静默退回缓存 = issue #32); :meth:`get_exposure_range` 给量程
      ``(min_ms, max_ms)``; :meth:`set_auto_exposure_target` 设 SDK 目标亮度;
      :meth:`auto_exposure` 是统一的自动曝光入口, ``target_max`` / ``tolerance``
      一律 **0-255 灰度**。调用方一律走这四个入口, 不要再读
      ``exposure_time_ms`` 缓存、也不要按后端分支。
    * 亮度: 原始像素值; dtype 取决于位深
      (通常是 ``uint8`` 或 ``uint16``)。
    * ROI 窗口: ``(width, height)`` 与 ``(center_x, center_y)``, 单位像素。
    * 像元尺寸: :attr:`pixel_size_um`, 单位**微米 (µm)**, 是**类属性** ——
      它由型号决定, 与实例状态无关。``0`` 表示该后端不提供或尚未登记。
    """

    #: 传感器像元边长 (µm)。0 = 未知或不适用。
    #:
    #: 子类按型号给出: 大恒从型号配置/已知型号表解析 (见
    #: :mod:`ao_shaping.drivers.ccd.pixel_size`), MiiCam 写死 2.9, 仿真后端给出
    #: 孪生口径。**调用方不得再硬编码数值**, 需要像元尺寸时读本属性 (或走
    #: :meth:`require_pixel_size_um` 拿到带根因的报错)。
    pixel_size_um: float = 0.0

    #: 后端是否提供原生推流回调 (MiiCam)。``False`` 时 :meth:`start_stream` 用
    #: 后台线程轮询 :meth:`get_numpy_image` 产帧 (Daheng / ffmpeg / 仿真等)。
    supports_callback_stream: bool = False

    #: 轮询取流两次抓帧之间的额外间隔 (秒); 0 表示由 ``get_numpy_image`` 自身的
    #: 阻塞/曝光时间限速。后端若一次性返回缓存帧, 可调大以避免空转。
    stream_poll_interval_s: float = 0.0

    #: 后端可选的位深集合 (bit); 空元组 = 该后端**不提供**位深选择。
    #:
    #: 位深决定帧的 dtype 与满量程 (12 bit 是 ``uint16`` 只填低 12 位), 所以它是
    #: **采集契约的一部分** 而不是驱动内部细节: 饱和判据、线性度、峰值统计都按
    #: 满量程折算, 拿到错位深就等于拿错量纲。实现方在这里登记设备真能切出的档位。
    supported_bit_depths: ClassVar[tuple[int, ...]] = ()

    #: 当前**生效**位深 (bit); ``None`` = 后端不提供位深选择, 或设备尚未打开。
    #:
    #: 类级默认 ``None``, 实例由后端在 ``open()`` / :meth:`set_bit_depth` 里写入
    #: **设备回读真值** (不是请求值 —— SDK 对 ``PixelFormat`` 的写入可能被静默保留
    #: 旧值, 同 ``IntFeature.set`` 的坑)。调用方一律读本属性, 不要读驱动私有缓存。
    #: 位深是**满量程的唯一来源** (issue #246): 一切满量程折算取值经
    #: :meth:`_full_scale` / :meth:`effective_full_scale`, 不按帧 dtype 推导。
    bit_depth: int | None = None

    # 取流线程状态 (惰性初始化; 未覆写 __init__ 的后端也能直接用 start_stream)
    _stream_thread: threading.Thread | None = None
    _stream_stop: threading.Event | None = None

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
        self._stream_thread: threading.Thread | None = None
        self._stream_stop: threading.Event | None = None

    def __enter__(self):
        """上下文管理器入口 —— 初始化相机。"""
        self.initialize()
        return self

    def _ensure_open(self, detail: str = "camera not initialized") -> None:
        """设备未初始化 ⇒ 抛 :class:`CameraError`。

        取代此前散在三个驱动里的 ``assert self.cam, "camera not initialized"``:
        它校验的是**外部设备状态** (``initialize()`` 是否真的成功), 而 `assert` 在
        ``python -O`` 下被整条移除 —— 之后就是拿空句柄去调 SDK。SLM 侧早已有同一
        范式 (``Santec._ensure_open``), 这里是相机侧的对应入口。
        """
        if not self._device_open():
            raise CameraError(detail)

    def _device_open(self) -> bool:
        """设备是否已初始化。用 ``cv2`` 视频流 / 图像序列回放等后端时覆写本方法。"""
        return self.cam is not None

    # ------------------------------------------------------------ 取流 (GUI 用)
    def start_stream(self, on_frame: Callable[[np.ndarray], None] | None = None) -> None:
        """启动后台取流, 每来一帧调用 ``on_frame(img)``。

        默认实现用**轮询线程**: 反复调用 :meth:`get_numpy_image` (单帧, 不丢首帧),
        因此任何实现了抽象 ``get_numpy_image`` 的后端都零改动可用。原生支持回调
        推流的后端 (``supports_callback_stream=True``, 目前是 MiiCam) 应覆写本方法,
        避免多一层线程与抓帧延迟。

        回调运行在取流线程里, 必须保持轻量: 只入信箱, 不做分析, 不回头调 SDK。

        参数:
            on_frame: 帧回调, 签名 ``(img)``。``None`` 表示只启动流。
        """
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._ensure_open()
        stop = threading.Event()
        thread = threading.Thread(
            target=self._stream_loop,
            args=(on_frame, stop),
            name=f"{type(self).__name__}-stream",
            daemon=True,
        )
        self._stream_stop = stop
        self._stream_thread = thread
        thread.start()

    def stop_stream(self) -> None:
        """停止 :meth:`start_stream` 启动的取流线程 (幂等)。"""
        stop = getattr(self, "_stream_stop", None)
        thread = getattr(self, "_stream_thread", None)
        self._stream_stop = None
        self._stream_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _stream_loop(
        self,
        on_frame: Callable[[np.ndarray], None] | None,
        stop: threading.Event,
    ) -> None:
        """轮询取流循环 (基类默认实现)。"""
        interval = max(0.0, float(getattr(self, "stream_poll_interval_s", 0.0) or 0.0))
        while not stop.is_set():
            try:
                img = self.get_numpy_image(1, skip_first=False)
            # policy: 单帧抓取失败不能让取流线程退出 (否则界面永久停帧), 限流后继续
            except Exception:  # policy: 后端 SDK 异常类型不可枚举
                logger.exception("{} 轮询取流失败, 200ms 后重试", type(self).__name__)
                stop.wait(0.2)
                continue
            if on_frame is not None:
                try:
                    on_frame(img)
                # policy: 用户回调异常绝不能打死取流线程
                except Exception:  # policy: 同上
                    logger.exception("帧回调 (on_frame) 抛出异常, 已忽略")
            if interval > 0.0:
                stop.wait(interval)

    def set_exposure_ms(self, time_ms: float) -> float:
        """设置曝光 (ms) 并返回生效值 —— 采集中改曝光的安全入口。

        默认委托 :meth:`reset_exposure_time`。MiiCam 覆写为「回调流存续期间单次
        Stop→put→Start」的快速通道; Daheng 等支持运行中写曝光的后端用默认实现
        即可 (其 ``reset_exposure_time`` 会自行消费陈旧帧)。
        """
        return self.reset_exposure_time(time_ms)

    # ------------------------------------------------------------ 曝光契约
    #: 跨后端通用的曝光量程回落 (后端既不暴露 SDK 量程, 也没有 min/max 属性时)。
    DEFAULT_EXPOSURE_RANGE_MS: ClassVar[tuple[float, float]] = (0.011, 10_000.0)

    #: 自动曝光统一契约的默认档位 (0-255 灰度), 见 :meth:`auto_exposure`。
    AUTO_EXPOSURE_TARGET_MAX: ClassVar[float] = 40.0
    AUTO_EXPOSURE_TOLERANCE: ClassVar[float] = 5.0
    AUTO_EXPOSURE_MAX_ITERATIONS: ClassVar[int] = 20
    AUTO_EXPOSURE_N_SAMPLE: ClassVar[int] = 1

    #: 比例环单步放大上限 —— 三个实现 (基类 / 大恒 / MiiCam) 必须一致, 否则
    #: 「同一个 target 在不同后端上收敛到不同曝光」无法解释。
    AUTO_EXPOSURE_MAX_RATIO: ClassVar[float] = 3.0

    #: SDK 自动曝光目标亮度的记录值 (0-255 灰度)。**类属性**: 没调用过
    #: ``super().__init__`` 的后端 (大恒) 也能安全读写。
    _auto_exposure_target: int = 0

    def get_exposure_time_ms(self) -> float:
        """当前**生效**的曝光时间 (ms) —— 曝光契约的唯一读取入口。

        🔴 返回值必须是**设备真正在用**的曝光, 不是原始请求。两个硬件后端都
        把请求钳制并量化到整数微秒后才下发 SDK (大恒 ``ExposureTime.set(int(ms*1000))``、
        MiiCam ``put_ExpoTime(int(ms*1000))``), 拿未经量化的缓存当回读, 就是在虚报
        一个设备并未使用的曝光 (issue #32)。

        基类只能读驱动侧缓存 —— **有设备回读能力的后端必须覆写本方法**:

        * ``DahengCamera``: 读 GenICam ``ExposureTime`` 特征 (设备真值);
        * ``MIICamera``: SDK 无可靠 getter, 返回**已钳制 + 量化**的缓存。

        没实现本方法的后端会静默退回缓存, 那正是 issue #32 的缺陷形态。

        返回:
            float: 生效曝光, 单位毫秒。

        异常:
            CameraError: 后端连 ``exposure_time_ms`` / ``exposure_time`` 缓存都没有。
        """
        for attr in ("exposure_time_ms", "exposure_time"):
            value = getattr(self, attr, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        raise CameraError(
            f"{type(self).__name__} 没有可读的曝光时间: 既未实现 "
            "get_exposure_time_ms(), 也没有 exposure_time_ms / exposure_time 缓存。"
        )

    def get_exposure_range(self) -> tuple[float, float]:
        """设备支持的曝光范围 ``(min_ms, max_ms)``。

        基类读后端的 ``min_exposure_ms`` / ``max_exposure_ms`` (MiiCam / SimPibCCD
        提供), 都没有时回落到 :attr:`DEFAULT_EXPOSURE_RANGE_MS`。
        有 SDK 量程特征的后端 (大恒) 覆写本方法返回**设备真值**。
        """
        lo = getattr(self, "min_exposure_ms", None)
        hi = getattr(self, "max_exposure_ms", None)
        if (
            isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
            and not isinstance(lo, bool)
            and not isinstance(hi, bool)
            and float(hi) > float(lo) > 0.0
        ):
            return float(lo), float(hi)
        return self.DEFAULT_EXPOSURE_RANGE_MS

    def set_auto_exposure_target(self, target: int) -> int:
        """设置 SDK 自动曝光的目标亮度 (0-255 灰度), 返回**生效**值。

        基类没有可写的 SDK 特征, 只做驱动侧记录 (``get_auto_exposure_state`` 会
        把它报出去)。后端有对应特征时必须覆写: 大恒写 ``ExpectedGrayValue``,
        MiiCam 写 ``put_AutoExpoTarget`` 并钳到该 SDK 的 16-220。
        不支持自动曝光的回放型后端 (ffmpeg / image_folder) 覆写为抛
        :class:`NotImplementedError` —— 那是能力缺失, 不是「设了没生效」。

        参数:
            target: 目标亮度 (0-255 灰度), 越界会被钳制。

        返回:
            int: 实际生效的目标值。
        """
        value = int(round(float(target)))
        clamped = int(max(0, min(255, value)))
        if clamped != value:
            logger.warning(
                "auto exposure target {} 越界, 钳制到 {}", value, clamped
            )
        self._auto_exposure_target = clamped
        return clamped

    def _full_scale(self, source_dtype: np.dtype | type | None = None) -> float:
        """当前**生效位深**对应的满量程 (8/12/16 bit → 255/4095/65535)。

        优先级: bit_depth (设备回读真值) > source_dtype 的整数上限 > 255.0。

        位深是满量程的唯一来源 (issue #246): 部分后端的帧 dtype 与生效位深不符
        (大恒 MONO12 帧 dtype 是 ``uint16``, 真实满量程 4095 而非 65535), 按
        dtype 推导会得到错量纲, 因此设备回读的位深永远优先。
        """
        if self.bit_depth is not None:
            return float((1 << int(self.bit_depth)) - 1)
        if source_dtype is not None:
            try:
                # 非整数 dtype (float32 多帧均值等) 没有整数上限, 落入兜底。
                return float(np.iinfo(source_dtype).max)
            except ValueError:
                pass
        # ``DETECTOR_FULL_SCALE`` 在 common.py, 而 common 会 import base ——
        # 只能函数体内局部 import, 与本文件其它局部 import 的风格一致。
        from ao_shaping.drivers.ccd.common import DETECTOR_FULL_SCALE

        return float(DETECTOR_FULL_SCALE)

    def effective_full_scale(self, img: np.ndarray) -> float:
        """一帧的生效满量程: 帧上的显式标注 (AveragedFrame.full_scale) 优先,
        否则按本相机生效位深解析 (绝不按 img.dtype, 多帧均值恒为 float32)。

        帧标注优先于位深是刻意的: 标注是采集时刻写下的**实测**饱和值 (融合图
        在换位深/多档混合时无法用单一位深表达), 比位深推算更可信。
        """
        explicit = getattr(img, "full_scale", None)
        if explicit is not None:
            return float(explicit)
        return self._full_scale(getattr(img, "dtype", None))

    def _auto_exposure_peak(self, img: np.ndarray) -> float:
        """把一帧的峰值折算到 **0-255 灰度** —— 自动曝光契约的量纲。

        ``target_max`` / ``tolerance`` 按定义是 0-255 灰度, 但 8/12/16-bit 后端的
        原始帧峰值分别封顶在 255/4095/65535, 直接拿来和目标比会让高位深后端
        「peak 恒大于 high」而把曝光一路压到下限。满量程从
        :class:`AveragedFrame` 的 ``full_scale`` (或多帧均值推断) 取得 ——
        绝不用 ``img.dtype``, 因为多帧平均后 dtype 恒为 ``float32``。

        三个自动曝光实现 (基类通用环 / 大恒 / MiiCam) 必须走同一个折算入口,
        否则「同一个 target 在不同后端上收敛到不同曝光」无法解释。
        位深未知的后端仍退回帧标注/dtype (见 :meth:`_full_scale`)。

        参数:
            img: 采集到的帧 (uint8 / uint16 / 多帧均值 ``AveragedFrame`` 皆可)。

        返回:
            float: 折算到 0-255 灰度的峰值。
        """
        from ao_shaping.drivers.ccd.common import normalize_peak_to_gray

        return normalize_peak_to_gray(img, full_scale=self.effective_full_scale(img))

    def auto_exposure(
        self,
        target_max: float = AUTO_EXPOSURE_TARGET_MAX,
        tolerance: float = AUTO_EXPOSURE_TOLERANCE,
        twice_valid: bool = True,
        max_iterations: int = AUTO_EXPOSURE_MAX_ITERATIONS,
        n_sample: int = AUTO_EXPOSURE_N_SAMPLE,
    ) -> np.ndarray:
        """统一自动曝光契约的**通用**实现 (比例迭代)。

        两个硬件后端 (:meth:`DahengCamera.auto_exposure` /
        :meth:`MIICamera.auto_exposure`) 覆写本方法以加入各自 SDK 的行为
        (大恒的 ``ExposureAuto="Once"`` 快路径、MiiCam 的 Stop→set→Start 节奏),
        但**参数与语义完全一致**, 调用方不需要按后端分支:

        * ``target_max`` / ``tolerance`` 一律是 **0-255 灰度** (与 SDK
          ``ExpectedGrayValue`` 及历史 ``autoset_exposure_time_ms`` 一致, **不是 0-1**);
          16-bit 帧按其满量程折算到同一量纲, 因此 ``target_max=40`` 在 uint8 与
          uint16 后端上代表同一个亮度;
        * ``twice_valid=True`` 需要**连续两次**落入 ``target±tolerance`` 才收敛;
        * 返回**调整后采集到的那一帧** (不重拍)。

        单步调整 ``new = exp * target / peak``, 放大上限
        :attr:`AUTO_EXPOSURE_MAX_RATIO`; 到量程边界仍不达标时记 warning 并以边界
        曝光退出 (不空转到 ``max_iterations``)。

        本方法**绝不调用** ``common.auto_exposure`` —— 那个 helper 的职责是「有原生
        实现就委托给后端」, 反向调用会构成无限递归。

        参数:
            target_max: 目标峰值亮度 (0-255 灰度)。
            tolerance: 峰值容差 (0-255, 绝对值), 必须 > 0。
            twice_valid: 是否要求连续两次命中才收敛。
            max_iterations: 最大迭代次数 (至少 1 次测量)。
            n_sample: 每次估计峰值的采样帧数。

        返回:
            np.ndarray: 收敛 (或退出) 时采集到的图像。

        异常:
            ValueError: ``tolerance <= 0``。
            CameraError: 相机未初始化。
        """
        self._ensure_open()
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        target_val = float(target_max)
        low = int(max(target_val - tolerance, 10))
        high = int(min(target_val + tolerance, 254))
        min_exp, max_exp = self.get_exposure_range()
        min_exp, max_exp = float(min_exp), float(max_exp)

        logger.info(
            "Auto exposure start: target={:.0f}±{:.0f} (range=[{}, {}]ms, max_iter={})",
            target_val,
            tolerance,
            min_exp,
            max_exp,
            max_iterations,
        )

        img = self.get_numpy_image(n_sample, skip_first=True)
        twice_ok = False
        for i in range(max(1, int(max_iterations))):
            peak = self._auto_exposure_peak(img)

            if low <= peak <= high:
                if twice_ok or not twice_valid:
                    logger.info(
                        "Auto exposure converged at iter {}: exp={:.4f}ms, max={:.1f}",
                        i + 1,
                        self.get_exposure_time_ms(),
                        peak,
                    )
                    return img
                twice_ok = True
            else:
                twice_ok = False
                current = self.get_exposure_time_ms()
                if current <= 0:
                    break
                ratio = min(target_val / max(peak, 1.0), self.AUTO_EXPOSURE_MAX_RATIO)
                requested = float(np.clip(current * ratio, min_exp, max_exp))
                if abs(requested - current) < 1e-9:
                    break
                # 用**生效值**判边界: 后端会把请求钳到设备量程, 拿请求值比较会
                # 在「请求 = min 但设备钳到 > min」时永不退出。
                applied = float(self.set_exposure_ms(requested))
                if applied <= min_exp and peak > high:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to min",
                        target_val,
                        peak,
                    )
                    break
                if applied >= max_exp and peak < low:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to max",
                        target_val,
                        peak,
                    )
                    break

            img = self.get_numpy_image(n_sample, skip_first=True)

        logger.info(
            "Auto exposure finished: exp={:.3f}ms, max={:.1f}",
            self.get_exposure_time_ms(),
            # 与循环内判据同量纲 (0-255 灰度): 直接打印原始峰值会让 uint16 /
            # AveragedFrame 后端的日志比 target_max 大约 257 倍, 无法核对收敛。
            self._auto_exposure_peak(img),
        )
        return img

    def require_pixel_size_um(self) -> float:
        """返回像元边长 (µm); 未提供时抛 :class:`CameraError`。

        返回 ``0`` 比抛异常危险得多: 调用方会拿它去乘像素坐标, 得到 0 的物理尺寸,
        然后在几何校验里报一个与根因无关的错 (或静默算出 0 焦距)。这里直接把根因
        和补救办法说出来。
        """
        value = float(getattr(self, "pixel_size_um", 0.0) or 0.0)
        if not (value > 0.0):
            raise CameraError(
                f"{type(self).__name__} 没有可用的像元尺寸 (pixel_size_um={value:g})。"
                "大恒相机请登记型号像元尺寸: "
                "python -m ao_shaping.drivers.ccd.pixel_size --prompt; "
                "其它后端请显式设置 pixel_size_um。"
            )
        return value

    def set_bit_depth(self, bits: int) -> int:
        """切换输出位深 (bit), 返回**设备生效**的位深。

        基类默认**不提供**该能力: 没有登记 :attr:`supported_bit_depths` 的后端
        (ffmpeg / image_folder / 通用仿真) 在这里抛 :class:`CameraError`, 而不是把
        请求值当成生效值静默返回 —— 位深错了, 满量程与饱和判据就全错了。
        位深是**满量程的唯一来源** (issue #246): 生效值写入 :attr:`bit_depth` 后,
        一切满量程折算经 :meth:`_full_scale` / :meth:`effective_full_scale` 取值,
        不按帧 dtype 推导 (高位置深帧常被塞进 ``uint16`` 容器, dtype 并不可信)。

        覆写方的契约 (与曝光契约同源: 请求值不等于生效值):

        1. 只接受 :attr:`supported_bit_depths` 里的档位, 其余**报错**而不是回落;
        2. 写入设备后必须**回读真值** (SDK 可能静默保留旧 ``PixelFormat``, 同
           ``IntFeature.set`` 的坑);
        3. 把回读到的真值写进 :attr:`bit_depth` 并返回;
        4. 档位真的变了之后消费掉流水线里残留的旧格式帧 (同改曝光的
           ``EXPOSURE_CHANGE_DISCARD_FRAMES`` 纪律), 使「换位深后的第一帧就是新格式」。

        参数:
            bits: 请求位深 (bit)。

        返回:
            int: 设备实际生效的位深。

        抛出:
            CameraError: 该后端不支持位深选择。
        """
        raise CameraError(
            f"{type(self).__name__} does not support bit depth selection "
            f"(supported_bit_depths={tuple(self.supported_bit_depths)})"
        )

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

        契约: 成功返回 ``None``; 失败抛异常并携带具体信息
        (``ConnectionAbortedError`` 未找到设备 / ``CameraError`` 初始化失败)。
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
            CameraError: 相机未初始化。
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
            CameraError: 相机未初始化。
            AssertionError: center 非法。
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
    def get_cam_list() -> list[Any]:
        """获取可用相机设备列表。

        返回:
            可用相机设备列表 (格式各实现自定义: SDK 设备信息对象、设备索引、
            目录名或型号名)。
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

