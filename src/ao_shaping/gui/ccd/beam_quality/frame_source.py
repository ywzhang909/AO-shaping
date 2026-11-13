"""帧源抽象: 任意 ``BaseCamera`` 后端 (MiiCam / Daheng / ...) / 离屏仿真源。

线程模型 (与参考 SDK demo 的反面做法相反, 这里刻意如此):

* 生产者 (SDK 回调线程 / 仿真线程) **只做一件事**: 把最新一帧塞进容量为 1 的
  ``queue.Queue``, 满了就丢掉最旧的一帧。绝不做分析、绝不能回调 SDK。
* 消费者是 **UI 线程**, 用 :meth:`drain_latest` 取走"最新一帧", 落后时直接
  跳过中间帧 (只处理最新 → 显示延迟有界, 不会随采集速率堆积)。
* 生命周期 (open/close/改曝光) 由 UI 线程串行调用, 由实例锁保护。

**HDR 与取流从不同时占有相机**: 一次多曝光 bracket 自己逐档改曝光、自己抓帧, 和
取流线程抢的是同一个 SDK 句柄。所以「HDR 获取」打开期间取流是**停着**的, 由一个
节拍线程一轮接一轮 ``capture_hdr``; 恢复取流由该线程自己收尾 (关开关 / 失败 / 阶梯
不合法三种出口都一样), 设备操作因此永远不回 UI 线程。规则与实现见
:meth:`CameraFrameSource.set_hdr`。

``MIICamera`` 的导入**在函数体内**完成 (惰性): 该模块在 import 期就会加载原生
SDK 与 DLL, 而无 SDK 的机器也必须能 import 本模块做离线测试/演示。模块级只导入
:mod:`ao_shaping.drivers.ccd.base` (异常类型 ``CameraError``) —— 它与
``ccd/__init__`` 都是 SDK-free 的, 不破坏这条约定。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import CameraError
from ao_shaping.gui.ccd.beam_quality.hdr_preview import validate_exposure_ladder

# 帧回调签名: (图像, 接收时刻 monotonic 秒)
FrameCallback = Callable[[np.ndarray, float], None]

#: 跨驱动兜底曝光量程 (ms): 源未打开或后端不暴露量程时, 界面用它占位
FALLBACK_EXPOSURE_RANGE_MS: tuple[float, float] = (0.011, 10000.0)

#: 两轮 HDR bracket 之间的余量 (秒): 让 UI 有时间读信箱、画图, 也让操作员按得停。
#: 真正决定预览节拍的是 bracket 本身 (每档一次曝光切换 + 一次读出), 这个数只是下限。
HDR_CYCLE_GAP_S: float = 0.25

#: 关闭源时等 HDR 节拍线程收尾的上限 (秒)。取值要盖住一轮三档 bracket 的正常耗时,
#: 又不能让「关闭」按钮看起来卡死; 超时就继续关设备 —— 那一轮采集会失败并被节拍
#: 线程自己的 best-effort 兜住, 不会把异常抛回 UI 线程。
HDR_JOIN_TIMEOUT_S: float = 15.0


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
    """容量 1 的"最新帧"信箱 + 帧率 EMA (生产/消费两侧线程安全)。

    信箱里每一帧都带一个来源标记 (``"stream"`` 单曝光原始帧 / ``"hdr"`` 派生显示帧),
    取帧时与帧一起交给消费者: 界面必须分清手里这帧能不能拿来测量 —— HDR 预览是
    融合 + 色调映射后的 0..255 显示图, 当成原始计数存快照/导 CSV 就是假数据。
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[np.ndarray, float, str]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._fps = 0.0
        self._last_ts: float | None = None
        self._dropped = 0

    def push(self, img: np.ndarray, timestamp: float, kind: str = "stream") -> None:
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
        self._offer_latest(img, timestamp, kind)

    def push_display(self, img: np.ndarray, timestamp: float) -> None:
        """放入一帧**派生画面** (HDR 预览), 不计入帧率 EMA。

        帧率是「相机流健康度」的读数: HDR 预览由后台线程按 bracket 节奏推, 把它
        一起算进来会让 fps 变成 bracket 节拍、并在相机流上凭空多出挤帧, 操作员就
        再也没法从这两个数判断流是否正常。派生画面只占信箱位置。
        """
        self._offer_latest(img, timestamp, "hdr")

    def _offer_latest(self, img: np.ndarray, timestamp: float, kind: str) -> None:
        """替换信箱里的最新一帧; 挤掉旧帧则计一次丢弃并重试一次。"""
        if not self._offer(img, timestamp, kind):
            self._dropped += 1
            # 挤出旧帧后重试一次; 仍失败说明消费者恰好抢先, 放弃本帧
            self._discard_oldest()
            self._offer(img, timestamp, kind)

    def _offer(self, img: np.ndarray, timestamp: float, kind: str) -> bool:
        try:
            self._queue.put_nowait((img, timestamp, kind))
        except queue.Full:
            return False
        return True

    def _discard_oldest(self) -> None:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def drain_with_kind(self) -> tuple[np.ndarray, float, str] | None:
        """取走最新一帧 ``(img, timestamp, kind)``, 丢弃积压。

        容量 1, 但保持循环以防边界情况; 返回的元组里最后一项就是该帧的来源标记,
        因此"哪一帧"和"哪一帧的来历"永远是配套的, 不存在读到别的帧的来历这种竞态。
        """
        latest: tuple[np.ndarray, float, str] | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def drain(self) -> tuple[np.ndarray, float] | None:
        """取走最新一帧 ``(img, timestamp)`` (不关心来源时的简写)。"""
        latest = self.drain_with_kind()
        if latest is None:
            return None
        return latest[0], latest[1]

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
        self._stopping = False
        self._lock = threading.RLock()
        self._frames = _LatestFrameQueue()
        self._on_frame: FrameCallback | None = None
        self.last_frame_info: object | None = None
        #: 最近一次 :meth:`drain_latest` 取回的帧是哪来的: ``"stream"`` 取流原始帧 /
        #: ``"hdr"`` 多曝光融合预览。**测量只认原始帧** (界面据此把 HDR 预览挡在分析之外)。
        self.last_frame_kind: str = "stream"
        # ------------------------------------------------ HDR 预览 (与取流争用同一相机)
        #: 操作员意图: 预览走 HDR bracket。实际是否生效见 ``hdr_active``。
        self._hdr_enabled = False
        self._hdr_ladder_ms: tuple[float, ...] = ()
        #: 正在跑的 bracket 线程 (同一时刻至多一个; 相机 handle 只有一个)
        self._hdr_thread: threading.Thread | None = None
        #: HDR 回路是否在跑 (= 界面当前预览是不是 HDR)
        self._hdr_active = False
        #: 取流是否已为 HDR 停下 (退出时要恢复)
        self._hdr_stream_stopped = False
        #: 一次 bracket 是否正在采集 (曝光旋钮此刻归 bracket 所有)
        self._hdr_capturing = False
        self._hdr_n_sample = 1
        #: 节拍序号: 阶梯改动/关开关时 +1, 让在跑的那一轮换设定或收手
        self._hdr_cycle = 0
        #: 最近一次 HDR 失败原因 (None = 没失败过); 界面据此如实显示回退状态
        self._last_hdr_error: str | None = None
        #: bracket 期间收到的曝光请求: 攒到这里, 下一轮开头再下发设备。
        #: 不能在 bracket 中途写 —— ``capture_hdr`` 的 finally 会把进入时的曝光
        #: 改回去, 中途写的值会被无声撤销, 操作员看到滑块动了、下一帧却还是老曝光。
        self._pending_exposure_ms: float | None = None

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
        """打开相机并启动取流 (HDR 开着时直接交给节拍线程独占相机)。

        失败时抛异常且不留下半开状态。
        """
        with self._lock:
            if self._running:
                return
            cam = self._create_camera()
            # HDR 意图在 open 之前就定下来了: 那样连"先起流再停流"这一趟都不必走
            # (MiiCam 的 start/stop_grabbing 是重建采集管线, 白花一两秒)
            hdr_first = bool(self._hdr_enabled)
            try:
                cam.open()
                self._cam = cam
                self._on_frame = on_frame
                if not hdr_first:
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
            self._stopping = False
            self.last_frame_kind = "stream"
            # 以设备实际生效值刷新缓存 (后端可能钳制/量化曝光)
            self._refresh_exposure_from_cam()
            if hdr_first:
                logger.info(
                    "HDR 获取已开启, 直接以 bracket 节拍起流: 阶梯 {} ms",
                    "/".join(f"{value:g}" for value in self._hdr_ladder_ms),
                )
                self._hdr_stream_stopped = True
                self._start_hdr_loop()

    def stop(self) -> None:
        """幂等停止 HDR 节拍/取流并关闭设备。"""
        with self._lock:
            # 先立"别再产帧"的牌子: 节拍线程在每个检查点看它, 尽快自己收手
            self._stopping = True
            hdr_thread = self._hdr_thread
        if hdr_thread is not None and hdr_thread.is_alive():
            # 节拍线程手里是**同一个相机 handle**: 必须等它把 bracket 收尾再关设备,
            # 否则就是"一边改曝光抓帧一边 close"。等不到 (长阶梯 > 超时) 也只能继续关:
            # 那一轮的采集会失败并被自己的 best-effort 兜住, 不会把异常抛回这里。
            hdr_thread.join(timeout=HDR_JOIN_TIMEOUT_S)
            if hdr_thread.is_alive():
                logger.warning(
                    "HDR 节拍线程 {}s 内没有退出, 继续关闭设备 (该轮采集将失败)",
                    HDR_JOIN_TIMEOUT_S,
                )
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
            self._hdr_stream_stopped = False
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
        """取走最新一帧 ``(img, 采集时刻)``; 它的来历写进 :attr:`last_frame_kind`。"""
        latest = self._frames.drain_with_kind()
        if latest is None:
            return None
        self.last_frame_kind = latest[2]
        return latest[0], latest[1]

    # ---------------------------------------------------------------- 曝光
    def set_exposure_ms(self, ms: float) -> float:
        """设置曝光 (ms), 返回生效值。采集中/未采集都走 cam.set_exposure_ms。

        HDR bracket 正在采集时**不碰设备** (见 :meth:`set_hdr`): 曝光旋钮此刻归
        bracket 所有, 写进去会污染正在积分的那一档。这里把值记下来作为"下次开流用
        的曝光", 返回它, 让上层界面显示自己发出的值而不是被无视的请求 ——
        bracket 一结束, 恢复取流用的就是这个值。
        """
        with self._lock:
            cam = self._cam
            if cam is None:
                # 未打开: 只记录请求值, start() 时带入构造函数
                self._exposure_ms = float(ms)
                return self._exposure_ms
            if self._hdr_capturing:
                # bracket 正在逐档改曝光: 此刻写设备会污染正在积分的那一档, 而且
                # capture_hdr 退出时会把曝光**还原成进 bracket 前的值**, 直接写等于
                # 写完就被抹掉。所以记下待写值, 下一轮开采前先写进去。
                self._exposure_ms = float(ms)
                self._pending_exposure_ms = float(ms)
                logger.info(
                    "HDR bracket 正在采集, 曝光 {}ms 排队到下一轮开头生效", float(ms)
                )
                return self._exposure_ms
            try:
                applied = float(cam.set_exposure_ms(float(ms)))
            except Exception:
                logger.exception("设置相机曝光失败 (请求 {:.3f}ms)", ms)
                raise
            self._exposure_ms = applied
            logger.info("相机曝光已设为 {:.3f}ms", applied)
            return applied

    # ---------------------------------------------------------------- HDR 预览
    def set_hdr(
        self,
        enabled: bool,
        ladder_ms: Sequence[float] | None = None,
        *,
        n_sample: int = 1,
    ) -> None:
        """切换预览的 HDR bracket 模式, 并设定曝光阶梯 (ms) 与每档采样帧数。

        **为什么要独占相机** — :meth:`BaseCamera.capture_hdr` 自己逐档改曝光、自己
        走 ``get_numpy_image`` 抓帧, 是**阻塞**通路; 它与 ``start_stream`` 的取流线程
        同时抓帧会抢同一个 SDK 句柄。所以 HDR 打开期间取流是**停着**的, 节拍线程
        一轮接一轮:

            停流 (一次) → capture_hdr → 推 HDR 预览 → 停一下 → 下一轮

        预览节拍因此从"相机帧率"掉到"每轮 bracket 一次", 这是 HDR 的真实代价,
        界面上必须看得出来 (源侧 fps 停在停流前的值, 模式行显示 HDR 预览)。
        退出 HDR (关开关 / 一轮失败 / 阶梯不合法) 时由节拍线程恢复取流。

        **语义与 transfer 侧的 HDR 探测一致**: HDR 只作用于显示预览。测量/记录用的
        单曝光 raw 帧不经过这里, 曝光值本身也不被改动 (bracket 结束由
        ``capture_hdr`` 恢复进入时的曝光)。

        失败 (相机不支持 / 阶梯非法 / 融合抛错) 只记 :attr:`last_hdr_error`、把
        :attr:`hdr_enabled` 落回 ``False`` 并退回单曝光取流, **绝不抛出** —— 预览
        增强不能把画面弄死, 也不能在没人管的情况下反复重试砸相机。阶梯的合法性
        规则与界面用的是同一份 (:mod:`.hdr_preview`), 免得脚本直接调这里时绕过红线。
        """
        want = bool(enabled)
        ladder = tuple(float(value) for value in (ladder_ms or ()))
        with self._lock:
            was_ladder = self._hdr_ladder_ms
            effective = ladder or was_ladder
            if want:
                try:
                    effective = validate_exposure_ladder(effective)
                except ValueError as exc:
                    # policy: 非法阶梯不抛给调用方 (调用方可能就是 UI 线程): 记下原因,
                    # 老老实实退回单曝光取流, 由界面把这句话回显给操作员
                    self._hdr_enabled = False
                    self._last_hdr_error = str(exc)
                    if self._hdr_active:
                        self._bump_hdr_cycle()
                    return
                self._hdr_ladder_ms = effective
            self._hdr_enabled = want
            self._hdr_n_sample = max(1, int(n_sample))
            if not want:
                if self._hdr_active:
                    # 让在跑的那一轮在下一个检查点自己收手 (不 join: 这里可能在 UI 线程)
                    self._bump_hdr_cycle()
                return
            self._last_hdr_error = None
            if effective != was_ladder:
                self._bump_hdr_cycle()
            if self._running and self._cam is not None and self._hdr_thread is None:
                self._start_hdr_loop()

    def _start_hdr_loop(self) -> None:
        """在源锁内起一轮接一轮的 HDR 线程 (调用方持锁且已确认未在跑)。"""
        self._hdr_active = True
        thread = threading.Thread(
            target=self._hdr_loop, name="beam-quality-hdr", daemon=True
        )
        self._hdr_thread = thread
        thread.start()

    def _bump_hdr_cycle(self) -> None:
        """让正在跑的那一轮立刻重读设定 (阶梯变了 / 开关关了)。"""
        self._hdr_cycle += 1

    def run_one_hdr_cycle(self) -> bool:
        """尽快补一轮 HDR 预览 (阶梯刚改完, 不想等下一拍)。

        相机只有一个 handle, 因此**永不**为了补拍去起第二个线程: HDR 回路在跑时
        只是把节拍序号 +1, 让回路在下一个检查点用新阶梯立刻重采; 回路没跑时
        (源未开 / 后端不支持 / 上一轮失败退回) 返回 ``False``, 由界面如实显示。
        """
        with self._lock:
            if not self._hdr_enabled:
                self._last_hdr_error = "HDR 未开启"
                return False
            if not self._hdr_ladder_ms or len(self._hdr_ladder_ms) < 2:
                self._last_hdr_error = "曝光阶梯至少要两档"
                return False
            self._bump_hdr_cycle()
            if self._hdr_thread is not None:
                return True
            if self._cam is None or self._stopping:
                return False
            self._start_hdr_loop()
            return True

    @property
    def hdr_supported(self) -> bool:
        """当前后端是否能多曝光 bracket (看有没有 ``capture_hdr``; 不碰设备)。"""
        cam = self._cam
        if cam is not None:
            return callable(getattr(cam, "capture_hdr", None))
        return True

    @property
    def hdr_enabled(self) -> bool:
        """界面上「HDR 获取」开关的状态。"""
        return self._hdr_enabled

    @property
    def hdr_active(self) -> bool:
        """HDR 回路是否真的在跑 (= 当前预览是 HDR)。开关打开但源未运行/后端不支持时为 False。"""
        return self._hdr_active

    @property
    def hdr_ladder_ms(self) -> tuple[float, ...]:
        return self._hdr_ladder_ms

    @property
    def last_hdr_error(self) -> str | None:
        """最近一次 HDR 失败的原因 (None = 没失败过)。"""
        return self._last_hdr_error

    def _hdr_loop(self) -> None:
        """节拍线程: 整段 HDR 模式**独占相机** (取流停着), 一轮 bracket 接一轮。

        为什么不在两轮之间把取流放回来:

        * ``capture_hdr`` 推出的预览帧刚进信箱, 恢复的取流就用手头第一帧单曝光 raw
          把它顶掉 —— 操作员看到的是 HDR 闪一下被打回原形;
        * ``MiiCam.reset_exposure_time`` 的文档自己写明「非取流态 (HDR 逐档改曝光,
          ``capture_hdr`` 内部) 走重建采集管线这条路」—— 停流正是驱动预期的状态。

        于是预览节拍 = 一轮 bracket 的耗时 (外加 :data:`HDR_CYCLE_GAP_S` 余量),
        信箱里留的是最后一次融合结果。退出时 (关开关 / 失败 / 阶梯不合法) 由本线程
        负责恢复取流 —— 设备操作留在工作线程, 不放回 UI 线程。
        """
        failure: str | None = None
        try:
            while True:
                with self._lock:
                    cycle = self._hdr_cycle
                    cam = self._cam
                    ladder = self._hdr_ladder_ms
                    n_sample = self._hdr_n_sample
                    keep_going = (
                        self._running
                        and not self._stopping
                        and self._hdr_enabled
                        and cam is not None
                        and len(ladder) >= 2
                    )
                if not keep_going or cam is None:
                    return
                if not self._hold_camera_for_hdr(cam):
                    return
                self._apply_pending_exposure(cam)
                fused = self._capture_hdr_display(cam, ladder, n_sample)
                with self._lock:
                    self._hdr_capturing = False
                    still_on = (
                        self._running
                        and not self._stopping
                        and self._hdr_enabled
                        and self._cam is cam
                    )
                    stale = self._hdr_cycle != cycle
                    error = self._last_hdr_error
                if not still_on:
                    return
                if fused is None:
                    failure = error or "HDR 采集失败"
                    return
                if stale:
                    # 采集期间阶梯又改了/要求立刻补一轮: 这帧属于旧设定, 丢掉直接重来
                    continue
                self._frames.push_display(fused, time.monotonic())
                logger.debug(
                    "HDR 预览已更新: 阶梯 {} ms (节拍 = 一轮 bracket, 取流期间暂停)",
                    "/".join(f"{value:g}" for value in ladder),
                )
                if not self._wait_next_cycle(cycle):
                    return
        finally:
            with self._lock:
                self._hdr_active = False
                self._hdr_capturing = False
                self._hdr_thread = None
                self._hdr_stream_stopped = False
                resume = (
                    self._running
                    and not self._stopping
                    and self._cam is not None
                )
                cam = self._cam
            if failure is not None:
                # 失败退回单曝光: 先把开关落回 False (否则界面的补拍请求会砸着相机
                # 无限重试), 再由下面的 _resume_stream 恢复取流 —— 画面不能停在 HDR
                # 之前的旧帧上, 也不能停在"看起来还开着 HDR"的状态上
                logger.warning("HDR 预览失败, 退回单曝光取流: {}", failure)
                with self._lock:
                    self._hdr_enabled = False
                    self._last_hdr_error = failure
            if resume and cam is not None:
                self._resume_stream(cam)

    def _wait_next_cycle(self, cycle: int) -> bool:
        """两轮之间的短睡 (分片, 便于关开关/改阶梯时立即响应)。

        返回 ``False`` = 该收手了; ``True`` = 继续下一轮 (可能是被新的请求叫醒的)。
        """
        deadline = time.monotonic() + HDR_CYCLE_GAP_S
        while time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            with self._lock:
                if not (self._running and not self._stopping and self._hdr_enabled):
                    return False
                if self._hdr_cycle != cycle:
                    return True
        return True

    def _hold_camera_for_hdr(self, cam: object) -> bool:
        """确保取流已停、相机归 bracket 独占 (幂等: 只在第一次真正停流)。"""
        with self._lock:
            if self._hdr_stream_stopped:
                self._hdr_capturing = True
                return True
        try:
            cam.stop_stream()  # type: ignore[attr-defined]
        except (CameraError, OSError) as exc:
            # 停流的失败面是可枚举的: 基类实现只做「置停止事件 + join 取流线程」(不抛),
            # MiiCam 覆写要退出 SDK 回调模式, 其失败形态是厂商 HRESULT 包装 (win32 下
            # ``HRESULTException`` 是 ``OSError`` 子类), 设备未初始化则是 ``CameraError``
            # (drivers/ccd/base.py::_ensure_open)。停不下来就绝不能边取流边 bracket
            # (两者抢同一个设备), 所以这里认错并退回单曝光。
            logger.exception("HDR 前停止取流失败")
            with self._lock:
                self._last_hdr_error = f"停止取流失败: {exc}"
            return False
        with self._lock:
            self._hdr_stream_stopped = True
            self._hdr_capturing = True
        return True

    def _apply_pending_exposure(self, cam: object) -> None:
        """把 bracket 期间攒下的曝光请求写进设备 (每轮开采前一次)。

        只有在 HDR 开着、有人动过曝光滑块时才有内容; 写完就清空, 不会每轮重复下发。
        失败只记日志: 预览节拍不能因为一次曝光写失败就死掉。
        """
        with self._lock:
            pending = self._pending_exposure_ms
            if pending is None:
                return
            self._pending_exposure_ms = None
        try:
            applied = float(cam.set_exposure_ms(pending))  # type: ignore[attr-defined]
        except (CameraError, OSError, TypeError, ValueError):
            # 设备侧同样可枚举: ``set_exposure_ms`` 的入口是 ``_ensure_open``
            # (CameraError), 往下是写曝光 + 重启拉流的厂商调用 (HRESULTException 是
            # OSError 子类; 大恒经 ``ExposureTime.set``, MiiCam 经 Stop→put_ExpoTime→
            # StartPullModeWithCallback, 重试耗尽后原样抛出), 加上把生效值换算成
            # float 时的 TypeError / ValueError。一次写失败只丢这一条排队请求。
            logger.exception("HDR 轮间下发排队曝光失败 (请求 {:.3f}ms)", pending)
            return
        with self._lock:
            self._exposure_ms = applied
        logger.info("排队曝光已在 HDR 轮间生效: {:.3f}ms", applied)

    def _resume_stream(self, cam: object) -> None:
        """退出 HDR 后恢复取流; 失败只记日志 (界面上的「关闭/打开」还能救回来)。

        这里**不看** ``_hdr_enabled``: 关开关时它当然是 False, 失败退回时也是刚被
        落成 False —— 两种情况都必须把单曝光流放回来。
        """
        with self._lock:
            if not (self._running and not self._stopping and self._cam is cam):
                return
            exposure = self._exposure_ms
        try:
            cam.start_stream(self._on_camera_frame)  # type: ignore[attr-defined]
        except (CameraError, OSError, RuntimeError):
            # 可枚举: 基类轮询路径的失败面是 ``_ensure_open`` (CameraError) 与
            # ``Thread.start()`` 起不了线程 (RuntimeError, 线程资源耗尽);
            # ``MIICamera`` 覆写为 SDK 回调启动, 失败形态是厂商 HRESULT 包装
            # (win32 下 OSError 子类)。恢复失败只记日志: 界面上的「关闭/打开」还能救回来。
            logger.exception("HDR 退出后恢复取流失败 (曝光 {:.3f}ms)", exposure)
            with self._lock:
                self._last_hdr_error = "恢复取流失败"

    def _capture_hdr_display(
        self, cam: object, ladder: tuple[float, ...], n_sample: int
    ) -> np.ndarray | None:
        """跑一次多曝光 bracket 并色调映射成 0..255 预览图; 失败返回 None。

        融合走 :func:`capture_hdr_display` (``BaseCamera.capture_hdr`` +
        ``utils.image.hdr.hdr_fuse``), 色调映射走 ``live_view.tone_map_hdr``
        (Reinhard) —— 都是既有实现, 这里不重写第二套 HDR。两者都在函数体内惰性
        导入: 无 SDK 的机器也要能 import 本模块做离线测试, 且
        ``transfer_desktop.live_view`` 会拉起 Qt 控件, 不该在模块导入期被 CCD 侧拖进来。
        """
        from ao_shaping.tools.slm.transfer.hdr_probe import capture_hdr_display
        from ao_shaping.gui.slm.transfer_desktop.live_view import tone_map_hdr

        try:
            fused = capture_hdr_display(cam, list(ladder), n_sample=n_sample)
        except (TypeError, ValueError) as exc:
            # 这里守的**不是**设备调用: ``hdr_probe.capture_hdr_display`` 把自己的
            # ``cam.capture_hdr`` 包在 ``except Exception`` 里 (它的设计就是"探测绝不
            # 冒泡"), 设备侧失败一律变成 ``None`` + warning。能越出它的那行只剩它
            # 出 try 之后做的 ``np.asarray(fused, dtype=np.float32)``, 其失败形态是
            # 融合结果的类型/形状不合法 (TypeError / ValueError)。
            logger.exception("HDR 采集抛出未预期异常")
            with self._lock:
                self._last_hdr_error = str(exc) or type(exc).__name__
            return None
        if fused is None:
            with self._lock:
                if self._last_hdr_error is None:
                    # capture_hdr_display 把原因吞在日志里了 (它的设计就是"绝不打断主流程"),
                    # 这里只能给操作员一句能看懂的: 是没出帧, 不是画面坏了
                    self._last_hdr_error = "HDR 未取到融合帧 (相机不支持 / 无有效像素), 详见日志"
            return None
        display = np.asarray(tone_map_hdr(fused, method="reinhard"), dtype=np.uint8)
        with self._lock:
            # 本轮成功: 清掉旧的失败原因, 界面不再显示回退提示
            self._last_hdr_error = None
        return display

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
        except (ValueError, ImportError):
            # 按 ``drivers/ccd/common.py::create_camera`` 的 ``Raises:`` 收窄: 未知类型
            # ValueError, 后端 / SDK 不可用 ImportError (daheng 把构造期的 NameError
            # 归一化成了它)。其它异常是注册表或构造签名的接线 bug, 让它冒出去。
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
        self._stopping = False
        self._lock = threading.Lock()
        self._frames = _LatestFrameQueue()
        self._on_frame: FrameCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng(seed)
        # ---------------------------------------------- HDR 预览 (仿真: 同一线程换节拍)
        # 仿真源没有"取流线程"与 bracket 争用设备, 所以这里用**同一个**产帧线程:
        # HDR 开着时它就不发单曝光帧 (等价于真源的"取流暂停"), 改发融合帧。
        # 界面侧的状态语义与 :class:`CameraFrameSource` 完全一致, 便于离线演示/测试。
        self._hdr_enabled = False
        self._hdr_ladder_ms: tuple[float, ...] = ()
        self._hdr_cycle = 0
        self._last_hdr_error: str | None = None
        #: 补拍一轮的敲门砖: ``run_one_hdr_cycle`` set 它, 产帧线程立刻重画一帧
        self._hdr_kick = threading.Event()
        #: 同 :class:`CameraFrameSource.last_frame_kind`: 取回的这帧是单曝光还是 HDR 预览
        self.last_frame_kind: str = "stream"

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
            self._stopping = False
            self._stop_event.clear()
            self._hdr_kick.clear()
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
            self._stopping = True
            self._hdr_enabled = False
            self._on_frame = None
        self._stop_event.set()
        self._hdr_kick.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover - 生成一帧远快于 2s
                logger.warning("仿真帧源线程未在 2s 内退出")
        self._frames.reset()

    # ---------------------------------------------------------------- 取帧
    def drain_latest(self) -> tuple[np.ndarray, float] | None:
        """取走最新一帧 ``(img, 采集时刻)``; 它的来历写进 :attr:`last_frame_kind`。"""
        latest = self._frames.drain_with_kind()
        if latest is None:
            return None
        self.last_frame_kind = latest[2]
        return latest[0], latest[1]

    def set_exposure_ms(self, ms: float) -> float:
        """仿真源把曝光映射为亮度增益 (线性缩放, 上限 255)。"""
        with self._lock:
            self._exposure_ms = float(ms)
        return self._exposure_ms

    # ---------------------------------------------------------------- HDR 预览
    @property
    def hdr_supported(self) -> bool:
        """仿真源永远支持: 它是按阶梯逐档多渲染几帧, 再走**同一条**融合通路。"""
        return True

    @property
    def hdr_enabled(self) -> bool:
        return self._hdr_enabled

    @property
    def hdr_active(self) -> bool:
        """产帧线程当前是不是真在发 HDR 帧 (开关开着但没运行/阶梯不够时不算)。"""
        with self._lock:
            return bool(self._running and self._hdr_enabled and len(self._hdr_ladder_ms) >= 2)

    @property
    def hdr_ladder_ms(self) -> tuple[float, ...]:
        return self._hdr_ladder_ms

    @property
    def last_hdr_error(self) -> str | None:
        return self._last_hdr_error

    def set_hdr(
        self,
        enabled: bool,
        ladder_ms: Sequence[float] | None = None,
        *,
        n_sample: int = 1,
    ) -> None:
        """切换仿真 HDR 预览 (对外语义与 :meth:`CameraFrameSource.set_hdr` 一致)。

        仿真源没有取流线程与 bracket 争用同一台设备, 所以不需要"停流/恢复": 产帧
        线程在 HDR 开着时改发融合帧 (等价于真源里"取流暂停"), 关掉就回到常规节拍。
        ``n_sample`` 接住但不使用 —— 逐档渲染没有重复采样降噪的需求。
        """
        del n_sample
        want = bool(enabled)
        ladder = tuple(float(value) for value in (ladder_ms or ()))
        with self._lock:
            effective = ladder or self._hdr_ladder_ms
            if want:
                try:
                    effective = validate_exposure_ladder(effective)
                except ValueError as exc:
                    # policy: 与真源一致 —— 非法阶梯只记原因并留在单曝光节拍, 不抛出
                    self._hdr_enabled = False
                    self._last_hdr_error = str(exc)
                    self._hdr_kick.set()
                    return
            changed_ladder = want and effective != self._hdr_ladder_ms
            self._hdr_enabled = want
            if want:
                self._hdr_ladder_ms = effective
                self._last_hdr_error = None
            if changed_ladder:
                self._hdr_cycle += 1
            # 叫醒产帧线程: 关掉时别傻等一轮 HDR 余量才回到常规节拍
            self._hdr_kick.set()

    def run_one_hdr_cycle(self) -> bool:
        """让产帧线程立刻补一帧 HDR 预览 (阶梯刚改完, 不想等下一拍)。"""
        with self._lock:
            if not self._hdr_enabled:
                self._last_hdr_error = "HDR 未开启"
                return False
            if len(self._hdr_ladder_ms) < 2:
                self._last_hdr_error = "曝光阶梯至少要两档"
                return False
            if not self._running:
                return False
            self._hdr_cycle += 1
            self._hdr_kick.set()
            return True

    # ---------------------------------------------------------------- 生成
    def _run(self) -> None:
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                hdr = bool(self._hdr_enabled and len(self._hdr_ladder_ms) >= 2)
            frame = self._make_hdr_frame() if hdr else self._make_frame()
            timestamp = time.monotonic()
            if hdr:
                # 与真源同一语义: HDR 预览帧只占信箱, 不参与帧率统计
                self._frames.push_display(frame, timestamp)
            else:
                self._frames.push(frame, timestamp)
            callback = self._on_frame
            if callback is not None:
                try:
                    callback(frame, timestamp)
                # policy: 用户回调异常不终止仿真线程 (与实验系统源的取舍一致)
                except Exception:  # policy: 见上
                    logger.exception("仿真帧回调抛出异常, 已忽略")
            if hdr:
                # HDR 节拍: 歇一轮余量, 或被"重拍一轮"/关开关立刻叫醒
                self._hdr_kick.wait(HDR_CYCLE_GAP_S)
                self._hdr_kick.clear()
                next_deadline = time.monotonic()
                continue
            next_deadline += self._interval
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_deadline = time.monotonic()

    def _make_frame(self) -> np.ndarray:
        """合成一帧单曝光图像: 高斯光束 + 高斯噪声 + 缓慢漂移/宽度/强度起伏。"""
        self._phase += 0.05
        # 曝光仅作亮度增益的示意映射, 便于 UI 上的"应用曝光"看得到反应
        frame = self._render_beam(self._phase, self._exposure_gain(), clamp_display=True)
        noise = self._rng.normal(0.0, 2.5, size=frame.shape)
        return np.clip(frame + noise + 3.0, 0.0, 255.0).astype(np.uint8)

    def _make_hdr_frame(self) -> np.ndarray:
        """合成一帧 HDR 预览: 按阶梯逐档渲染 → **真实的** ``hdr_fuse`` → Reinhard。

        这里不另写一套融合/色调映射: 仿真源只负责"拍出"这一 bracket (同一相位、
        每档曝光各自的计数与噪声), 之后交给与实验系统完全相同的
        :func:`~ao_shaping.utils.image.hdr.hdr_fuse` 与
        ``transfer_desktop.live_view.tone_map_hdr``。所以离线按开关看到的是真算法的
        输出, 而不是"看起来像 HDR"的贴图。

        阶梯各档按**真实曝光线性缩放** (不像滑块那样截到 3 倍增益), 长档因此会真的
        削顶 —— 融合通路才有东西可救: 中心回到短档的保形值, 外晕从长档捞回来。
        默认阶梯 (1/4/16 ms) 都不足以让仿真场景饱和, 画面因此与 16 ms 单帧接近;
        想看到差别, 把阶梯拉到 ``1, 4, 16, 64``。
        融合/色调映射一旦出错 (无 Qt / 阶梯不合法) 退回单曝光帧并如实记下原因。
        """
        from ao_shaping.gui.slm.transfer_desktop.live_view import tone_map_hdr
        from ao_shaping.utils.image.hdr import hdr_fuse

        self._phase += 0.05
        phase = self._phase
        with self._lock:
            ladder = self._hdr_ladder_ms
        if len(ladder) < 2:  # pragma: no cover - 调用方已筛过, 留作自保
            return self._make_frame()
        shape = (self._height, self._width)
        try:
            samples = [
                self._render_beam(phase, ms * 0.05, clamp_display=False)
                + self._rng.normal(0.0, 2.5, size=shape)
                + 3.0
                for ms in ladder
            ]
            result = hdr_fuse(
                samples, ladder, dark=3.0, full_scale=255.0, read_noise_counts=2.5
            )
            display = np.asarray(tone_map_hdr(result.fused, method="reinhard"), dtype=np.uint8)
        # policy: 融合/色调映射不能杀掉产帧线程 —— 退回单曝光帧并记下原因
        except Exception as exc:
            logger.warning("仿真 HDR 预览失败, 退回单曝光帧: {}", exc)
            with self._lock:
                self._last_hdr_error = str(exc) or type(exc).__name__
            return self._make_frame()
        with self._lock:
            self._last_hdr_error = None
        return display

    def _exposure_gain(self) -> float:
        """当前曝光设定的示意增益 (线性, 截到 3 倍, 与滑块的可感知范围一致)。"""
        with self._lock:
            return max(0.05, min(self._exposure_ms / 20.0, 3.0))

    def _render_beam(self, phase: float, gain: float, *, clamp_display: bool) -> np.ndarray:
        """按给定相位与增益合成**未加噪声**的光束计数图 (float, 未截顶)。

        ``clamp_display`` 只决定要不要把增益截到 8 bit 显示范围: 单曝光通路沿用
        界面上的示意增益 (已经截过), HDR 通路要的是"长档真的饱和", 所以不截。
        """
        cx = self._cx0 + 14.0 * np.sin(0.31 * phase)
        cy = self._cy0 + 10.0 * np.cos(0.23 * phase)
        sigma_x = self._sigma_x0 * (1.0 + 0.12 * np.sin(0.11 * phase))
        sigma_y = self._sigma_y0 * (1.0 + 0.10 * np.cos(0.17 * phase))
        value = max(float(gain), 0.0)
        if clamp_display:
            value = min(value, 3.0)
        amplitude = self._amplitude0 * value * (0.9 + 0.1 * np.sin(0.53 * phase))
        return amplitude * np.exp(
            -(((self._xx - cx) ** 2) / (2.0 * sigma_x**2))
            - (((self._yy - cy) ** 2) / (2.0 * sigma_y**2))
        )
