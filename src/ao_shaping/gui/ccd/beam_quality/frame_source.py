"""帧源抽象: 任意 ``BaseCamera`` 后端 (MiiCam / Daheng / ...) / 离屏仿真源。

线程模型 (与参考 SDK demo 的反面做法相反, 这里刻意如此):

* 生产者 (SDK 回调线程 / 仿真线程) **只做一件事**: 把最新一帧塞进容量为 1 的
  ``queue.Queue``, 满了就丢掉最旧的一帧。绝不做分析、绝不能回调 SDK。
* 消费者是 **UI 线程**, 用 :meth:`drain_latest` 取走"最新一帧", 落后时直接
  跳过中间帧 (只处理最新 → 显示延迟有界, 不会随采集速率堆积)。
* 生命周期 (open/close/改曝光) 由 UI 线程串行调用, 由实例锁保护。

``MIICamera`` 的导入**在函数体内**完成 (惰性): 该模块在 import 期就会加载原生
SDK 与 DLL, 而无 SDK 的机器也必须能 import 本模块做离线测试/演示。
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from loguru import logger

# 帧回调签名: (图像, 接收时刻 monotonic 秒)
FrameCallback = Callable[[np.ndarray, float], None]

#: 跨驱动兜底曝光量程 (ms): 源未打开或后端不暴露量程时, 界面用它占位
FALLBACK_EXPOSURE_RANGE_MS: tuple[float, float] = (0.011, 10000.0)


@runtime_checkable
class FrameSource(Protocol):
    """帧源统一接口 (MiiCam 与仿真源一致)。"""

    @property
    def is_running(self) -> bool:
        """是否正在产帧。"""
        ...

    @property
    def fps(self) -> float:
        """实测帧率 (EMA)。"""
        ...

    @property
    def exposure_ms(self) -> float:
        """当前曝光 (ms); 仿真源只是记录值, 真实相机为设备回读/驱动缓存。"""
        ...

    @property
    def exposure_range_ms(self) -> tuple[float, float]:
        """设备真实量程 ``(min_ms, max_ms)``; 仿真源用宽量程。"""
        ...

    def start(self, on_frame: FrameCallback | None = None) -> None:
        """开始产帧。"""
        ...

    def stop(self) -> None:
        """停止产帧 (幂等); 真实相机在此退出 SDK 回调线程并关闭设备。"""
        ...

    def drain_latest(self) -> tuple[np.ndarray, float] | None:
        """取走最新一帧 ``(img, timestamp)``; 没有新帧返回 ``None``。"""
        ...

    def set_exposure_ms(self, ms: float) -> float:
        """设置曝光 (ms), 返回生效值。"""
        ...


class _LatestFrameQueue:
    """容量 1 的"最新帧"信箱 + 帧率 EMA (生产/消费两侧线程安全)。"""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._fps = 0.0
        self._last_ts: float | None = None
        self._dropped = 0

    def push(self, img: np.ndarray, timestamp: float) -> None:
        """放入最新帧; 槽位已满则丢弃旧帧 (最新优先)。"""
        with self._lock:
            if self._last_ts is not None:
                dt = timestamp - self._last_ts
                if 1e-6 < dt < 10.0:
                    instant = 1.0 / dt
                    self._fps = (
                        instant
                        if self._fps <= 0.0
                        else 0.85 * self._fps + 0.15 * instant
                    )
            self._last_ts = timestamp
        if not self._offer(img, timestamp):
            self._dropped += 1
            # 挤出旧帧后重试一次; 仍失败说明消费者恰好抢先, 放弃本帧
            self._discard_oldest()
            self._offer(img, timestamp)

    def _offer(self, img: np.ndarray, timestamp: float) -> bool:
        try:
            self._queue.put_nowait((img, timestamp))
        except queue.Full:
            return False
        return True

    def _discard_oldest(self) -> None:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def drain(self) -> tuple[np.ndarray, float] | None:
        """取走最新帧, 丢弃积压 (容量 1, 但保持循环以防边界情况)。"""
        latest: tuple[np.ndarray, float] | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def reset(self) -> None:
        """清空信箱与帧率统计 (启停时调用)。"""
        with self._lock:
            self._fps = 0.0
            self._last_ts = None
            self._dropped = 0
        self._discard_oldest()

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def dropped(self) -> int:
        return self._dropped


class CameraFrameSource:
    """通用相机帧源 (回调/轮询 + 最新帧信箱)。

    通过 :mod:`ao_shaping.drivers.ccd.common` 构造任意 ``BaseCamera`` 后端
    (``miicam`` / ``daheng`` / ...), 对外暴露统一的 :class:`FrameSource` 接口。
    采集中的曝光变更一律走 ``cam.set_exposure_ms`` (MiiCam 有快速通道,
    其它后端回落到 ``reset_exposure_time``)。

    - 推流型后端 (MiiCam, ``supports_callback_stream=True``): SDK 回调线程只入
      信箱, 不做分析。
    - 轮询型后端 (Daheng): 单独取流线程以 ``stream_fps`` 调 ``get_numpy_image``
      产帧, 仍只入信箱。
    """

    def __init__(
        self,
        camera_type: str = "miicam",
        cam_id: int = 0,
        exposure_ms: float = 20.0,
        camera: object | None = None,
        **camera_kwargs: object,
    ) -> None:
        self._camera_type = str(camera_type).lower()
        self._cam_id = int(cam_id)
        self._exposure_ms = float(exposure_ms)
        self._camera_kwargs = dict(camera_kwargs)
        #: 已构造好的后端实例 (测试注入口); None 时在 start() 里按类型构造。
        self._camera = camera
        self._cam: object | None = None
        self._running = False
        self._lock = threading.RLock()
        self._frames = _LatestFrameQueue()
        self._on_frame: FrameCallback | None = None
        self.last_frame_info: object | None = None

    # ---------------------------------------------------------------- 状态
    @property
    def camera_type(self) -> str:
        return self._camera_type

    @property
    def cam_id(self) -> int:
        return self._cam_id

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._frames.fps

    @property
    def exposure_ms(self) -> float:
        return self._exposure_ms

    @property
    def dropped_frames(self) -> int:
        """因消费不及时而被挤掉的帧数。"""
        return self._frames.dropped

    @property
    def exposure_range_ms(self) -> tuple[float, float]:
        """设备真实曝光量程 (ms); 未打开时退回跨驱动兜底量程。

        量程是设备特征 (大恒真实量程如 ``(0.02, 1000.0)``), 只能在 open 之后读,
        因此这里惰性 import 助手并包住当前相机实例。
        """
        cam = self._cam
        if cam is None:
            return FALLBACK_EXPOSURE_RANGE_MS
        from ao_shaping.drivers.ccd.common import get_camera_exposure_range

        try:
            return get_camera_exposure_range(cam)
        # policy: 量程读不回来只影响界面占位, 不能让打开源后的回调抛异常
        except Exception:
            logger.warning("读取相机曝光量程失败, 退回兜底量程")
            return FALLBACK_EXPOSURE_RANGE_MS

    # ---------------------------------------------------------------- 生命周期
    def start(self, on_frame: FrameCallback | None = None) -> None:
        """打开相机并启动取流。失败时抛异常且不留下半开状态。"""
        with self._lock:
            if self._running:
                return
            cam = self._create_camera()
            try:
                cam.open()
                self._cam = cam
                self._on_frame = on_frame
                cam.start_stream(self._on_camera_frame)
            # policy: 起流失败必须先把设备关掉再抛, 否则下一次 open 会撞 REPEAT_OPENED
            except Exception:
                try:
                    cam.close()
                except Exception:  # policy: 关闭阶段错误类型不可枚举
                    logger.warning("起流失败后关闭相机也失败")
                self._cam = None
                self._running = False
                raise
            self._running = True
            # 以设备实际生效值刷新缓存 (后端可能钳制/量化曝光)
            self._refresh_exposure_from_cam()

    def stop(self) -> None:
        """幂等停止取流并关闭设备。"""
        with self._lock:
            if not self._running:
                # 未启动过: 仍尝试关掉可能半开的设备
                cam = self._cam
                self._cam = None
                if cam is not None:
                    try:
                        cam.stop_stream()
                    except Exception:  # policy: 同上
                        logger.warning("停止取流失败 (未运行态)")
                    try:
                        cam.close()
                    except Exception:  # policy: 同上
                        logger.warning("关闭相机失败 (未运行态)")
                return
            self._running = False
            cam = self._cam
            self._cam = None
            if cam is not None:
                try:
                    cam.stop_stream()
                # policy: SDK 关闭阶段错误类型不可枚举
                except Exception:
                    logger.warning("停止取流失败 (继续关闭设备)")
                try:
                    cam.close()
                # policy: 同上
                except Exception:
                    logger.warning("关闭相机失败")
            self._frames.reset()
            self.last_frame_info = None

    # ---------------------------------------------------------------- 取帧
    def drain_latest(self) -> tuple[np.ndarray, float] | None:
        return self._frames.drain()

    # ---------------------------------------------------------------- 曝光
    def set_exposure_ms(self, ms: float) -> float:
        """设置曝光 (ms), 返回生效值。采集中/未采集都走 cam.set_exposure_ms。"""
        with self._lock:
            cam = self._cam
            if cam is None:
                # 未打开: 只记录请求值, start() 时带入构造函数
                self._exposure_ms = float(ms)
                return self._exposure_ms
            try:
                applied = float(cam.set_exposure_ms(float(ms)))
            except Exception:
                logger.exception("设置相机曝光失败 (请求 {:.3f}ms)", ms)
                raise
            self._exposure_ms = applied
            logger.info("相机曝光已设为 {:.3f}ms", applied)
            return applied

    # ---------------------------------------------------------------- 内部
    def _create_camera(self) -> object:
        """惰性构造后端相机 (SDK 只在此刻加载)。"""
        if self._camera is not None:
            return self._camera
        from ao_shaping.drivers.ccd import common

        kwargs = dict(self._camera_kwargs)
        # MiiCam 用回调模式; 其它后端回落到默认轮询
        if self._camera_type == "miicam":
            kwargs.setdefault("capture_mode", "callback")
        try:
            return common.create_camera(
                self._camera_type,
                cam_id=self._cam_id,
                exposure_time_ms=self._exposure_ms,
                **kwargs,
            )
        except Exception:
            logger.exception("构造相机后端 {} 失败", self._camera_type)
            raise

    def _on_camera_frame(self, img: np.ndarray) -> None:
        """SDK/轮询线程回调 —— 只入信箱, 严禁重活。"""
        timestamp = time.monotonic()
        self.last_frame_info = getattr(self._cam, "last_frame_info", None)
        self._frames.push(np.asarray(img), timestamp)
        callback = self._on_frame
        if callback is not None:
            try:
                callback(np.asarray(img), timestamp)
            # policy: 用户回调异常绝不能抛回厂商/取流线程
            except Exception:
                logger.exception("帧回调 (on_frame) 抛出异常, 已忽略")

    def _refresh_exposure_from_cam(self) -> None:
        cam = self._cam
        if cam is None:
            return
        try:
            self._exposure_ms = float(cam.get_exposure_time_ms())
        # policy: 曝光读不回来不影响取流, 退回构造时的设定值
        except Exception:
            try:
                self._exposure_ms = float(
                    getattr(cam, "exposure_time_ms", self._exposure_ms)
                )
            # policy: 属性缺失/不是数字的后端保持旧值, 不在取流路径上抛
            except (TypeError, ValueError):
                pass


# 向后兼容别名 (旧名仍被测试/外部引用)
MIICamFrameSource = CameraFrameSource


class SimBeamSource:
    """离屏仿真源: 独立线程生成带噪声/漂移/强度起伏的合成高斯光束。

    用于无硬件开发、UI 联调与冒烟测试; 接口与 :class:`MIICamFrameSource` 一致,
    且**不依赖 Qt** (纯 threading), 因此可在没有 QApplication 的测试里直接跑。
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        exposure_ms: float = 20.0,
        seed: int = 0,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._interval = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0
        self._exposure_ms = float(exposure_ms)
        self._running = False
        self._lock = threading.Lock()
        self._frames = _LatestFrameQueue()
        self._on_frame: FrameCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng(seed)

        yy, xx = np.mgrid[0 : self._height, 0 : self._width]
        self._xx = xx.astype(np.float32)
        self._yy = yy.astype(np.float32)
        # 基准光束参数: 中心/宽度/强度围绕这些值缓慢漂移
        self._cx0 = 0.5 * self._width
        self._cy0 = 0.5 * self._height
        self._sigma_x0 = 0.09 * self._width
        self._sigma_y0 = 0.07 * self._height
        self._amplitude0 = 210.0
        self._phase = 0.0

    # ---------------------------------------------------------------- 状态
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def fps(self) -> float:
        return self._frames.fps

    @property
    def exposure_ms(self) -> float:
        return self._exposure_ms

    @property
    def exposure_range_ms(self) -> tuple[float, float]:
        """仿真源没有设备约束, 用跨驱动兜底宽量程占位。"""
        return FALLBACK_EXPOSURE_RANGE_MS

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self._height, self._width)

    # ---------------------------------------------------------------- 生命周期
    def start(self, on_frame: FrameCallback | None = None) -> None:
        with self._lock:
            if self._running:
                return
            self._on_frame = on_frame
            self._stop_event.clear()
            self._frames.reset()
            self._thread = threading.Thread(
                target=self._run, name="SimBeamSource", daemon=True
            )
            self._running = True
            self._thread.start()
        logger.info(
            "仿真帧源已启动: {}x{} @ {:.1f} fps",
            self._width,
            self._height,
            1.0 / self._interval,
        )

    def stop(self) -> None:
        """停止仿真线程并等待其退出 (幂等)。"""
        with self._lock:
            thread = self._thread
            self._thread = None
            self._running = False
            self._on_frame = None
        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover - 生成一帧远快于 2s
                logger.warning("仿真帧源线程未在 2s 内退出")
        self._frames.reset()

    # ---------------------------------------------------------------- 取帧
    def drain_latest(self) -> tuple[np.ndarray, float] | None:
        return self._frames.drain()

    def set_exposure_ms(self, ms: float) -> float:
        """仿真源把曝光映射为亮度增益 (线性缩放, 上限 255)。"""
        with self._lock:
            self._exposure_ms = float(ms)
        return self._exposure_ms

    # ---------------------------------------------------------------- 生成
    def _run(self) -> None:
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            frame = self._make_frame()
            timestamp = time.monotonic()
            self._frames.push(frame, timestamp)
            callback = self._on_frame
            if callback is not None:
                try:
                    callback(frame, timestamp)
                # policy: 用户回调异常不终止仿真线程 (与实验系统源的取舍一致)
                except Exception:  # policy: 见上
                    logger.exception("仿真帧回调抛出异常, 已忽略")
            next_deadline += self._interval
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_deadline = time.monotonic()

    def _make_frame(self) -> np.ndarray:
        """合成一帧: 高斯光束 + 高斯噪声 + 缓慢漂移/宽度/强度起伏。"""
        self._phase += 0.05
        phase = self._phase
        cx = self._cx0 + 14.0 * np.sin(0.31 * phase)
        cy = self._cy0 + 10.0 * np.cos(0.23 * phase)
        sigma_x = self._sigma_x0 * (1.0 + 0.12 * np.sin(0.11 * phase))
        sigma_y = self._sigma_y0 * (1.0 + 0.10 * np.cos(0.17 * phase))
        gain = 1.0
        with self._lock:
            # 曝光仅作亮度增益的示意映射, 便于 UI 上的"应用曝光"看得到反应
            gain = max(0.05, min(self._exposure_ms / 20.0, 3.0))
        amplitude = self._amplitude0 * gain * (0.9 + 0.1 * np.sin(0.53 * phase))

        beam = amplitude * np.exp(
            -(((self._xx - cx) ** 2) / (2.0 * sigma_x**2))
            - (((self._yy - cy) ** 2) / (2.0 * sigma_y**2))
        )
        noise = self._rng.normal(0.0, 2.5, size=beam.shape)
        frame = np.clip(beam + noise + 3.0, 0.0, 255.0).astype(np.uint8)
        return frame
