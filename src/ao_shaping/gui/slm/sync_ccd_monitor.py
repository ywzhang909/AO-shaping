"""SLM 派发与 CCD 采样的解耦监测（Qt 线程侧）。

语义边界（务必按此向用户解释）
------------------------------
* 这是**按共享派发进度做时间对齐的并行采样**：SLM 线程照原节拍写帧，CCD 在
  自己的 ``QThread`` 里按固定轮询间隔取 ``drain_latest()``；采样频率与派发
  帧率互相独立。
* **不是逐帧一一对应**：相机比派发慢时，中间派发帧没有对应图像；相机比派发
  快时，多余图像被丢弃。每条样本只记录「采样那一刻派发到第几帧 / 第几秒」。
* 本模块不提高 SLM 实机帧率，也不改变闭环页的串行语义；没有 CCD 或相机起流
  失败时只让采样线程自己 ``failed``，不触碰 SLM 派发。
* 只保留最新一帧（``CameraFrameSource`` 内部就是容量 1 的信箱），不建无界
  队列。``frame_ready`` 每次只带一帧；UI 回调要轻（换引用/画最新一帧），并把
  ``interval_s`` 保持在 0.1 s 量级，否则 Qt 事件队列本身会积压图像对象。

时间口径
--------
派发进度由 :class:`DispatchSnapshot` 提供（UI 在 ``DynamicShapingWorker`` 的
``progress`` / ``finished_ok`` / ``stopped`` / ``failed`` slot 里更新它）。
可选的 :class:`~ao_shaping.utils.slm.transition_timeline.TransitionPlan` 只用来
把派发墙钟秒数换算成过渡进度 ``dispatch_progress_t``（与 ``blend_gray`` 的
``t = elapsed / duration_s`` 同一口径），不改 ``transition_timeline`` 本身。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PySide6.QtCore import QThread, Signal

from ao_shaping.gui.ccd.beam_quality.frame_source import (
    CameraFrameSource,
    FrameSource,
)
from ao_shaping.utils.slm.transition_timeline import TransitionPlan

__all__ = [
    "CCDSyncSampler",
    "DispatchSnapshot",
    "STRATEGIES",
    "frame_stats",
    "plan_progress_t",
]

#: 采样策略:
#:
#: * ``every``    —— 每 ``every_n`` 个派发帧取一帧（只在有新相机帧时取）；
#: * ``periodic`` —— 以 ``interval_s`` 为最小间隔连续取帧，与派发帧号无关；
#: * ``at-end``   —— 过渡结束后只取最后一帧。
STRATEGIES: tuple[str, ...] = ("every", "periodic", "at-end")


class DispatchSnapshot:
    """SLM 派发进度的线程安全快照。

    UI 侧接线（示例）::

        state = DispatchSnapshot()
        worker.progress.connect(
            lambda sent, total, elapsed: state.update(
                frame_index=sent, elapsed_s=elapsed
            )
        )
        worker.finished_ok.connect(lambda *_: state.mark_done())
        worker.stopped.connect(lambda *_: state.mark_done())
        worker.failed.connect(lambda *_: state.mark_done())

    ``frame_index`` 与 ``DynamicShapingWorker.progress`` 一致是 **1 基**
    （已发帧数）；初值 ``-1`` 表示「还没发出第一帧」，与
    ``scripts/slm/dynamic_shaping_capture_20261120.py`` 的 ``SharedState`` 同口径。
    同一时刻只有采样线程读、UI 线程写，锁只保护这三个标量，代价可忽略。
    """

    __slots__ = ("_lock", "_frame_index", "_elapsed_s", "_done")

    def __init__(
        self,
        frame_index: int = -1,
        elapsed_s: float = 0.0,
        done: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._frame_index = int(frame_index)
        self._elapsed_s = max(0.0, float(elapsed_s))
        self._done = bool(done)

    @property
    def frame_index(self) -> int:
        with self._lock:
            return self._frame_index

    @property
    def elapsed_s(self) -> float:
        with self._lock:
            return self._elapsed_s

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    def update(
        self,
        frame_index: int | None = None,
        elapsed_s: float | None = None,
        *,
        done: bool | None = None,
    ) -> None:
        """更新已提供的字段；``None`` 表示保持原值。"""
        with self._lock:
            if frame_index is not None:
                self._frame_index = int(frame_index)
            if elapsed_s is not None:
                self._elapsed_s = max(0.0, float(elapsed_s))
            if done is not None:
                self._done = bool(done)

    def mark_done(self, elapsed_s: float | None = None) -> None:
        """worker 结束（成功/停止/失败）都调这个，采样线程据此收尾。"""
        self.update(elapsed_s=elapsed_s, done=True)

    def snapshot(self) -> tuple[int, float, bool]:
        """一次性取 ``(frame_index, elapsed_s, done)``，三者互相一致。"""
        with self._lock:
            return (self._frame_index, self._elapsed_s, self._done)


def plan_progress_t(plan: TransitionPlan | None, elapsed_s: float) -> float | None:
    """派发墙钟秒数 → 过渡线性进度 0…1；没有 plan 时返回 ``None``。

    与 ``transition_timeline.blend_gray`` 的墙钟口径一致：慢硬件只会跳帧，
    不会拉长 ``duration_s``。
    """
    if plan is None:
        return None
    duration = float(plan.duration_s)
    if duration <= 0.0:
        return None
    return min(1.0, max(0.0, float(elapsed_s) / duration))


def frame_stats(img: np.ndarray) -> dict[str, Any]:
    """轻量统计：峰值、均值、扣中值背景后的质心（纯 numpy，不做椭圆拟合）。

    采样线程每帧都要调它，所以刻意不引入 ``analyzer.analyze_frame`` 的
    scipy 椭圆/剖面拟合。质心口径与 ``analyzer`` 相同：扣全图中值背景后的
    一阶矩；全黑帧退化为画面中心（不假装找到了光斑）。
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        # (H, W, C) 取首通道；(C, H, W) 取首通道同样成立
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    if arr.ndim != 2 or arr.size == 0:
        shape = tuple(int(v) for v in getattr(arr, "shape", ()))
        return {"peak": 0.0, "mean": 0.0, "centroid": (0.0, 0.0), "shape": shape}

    work = arr.astype(np.float32, copy=False)
    height, width = int(work.shape[0]), int(work.shape[1])
    peak = float(work.max())
    mean = float(work.mean())
    background = float(np.median(work))
    weight = work - background
    np.maximum(weight, 0.0, out=weight)
    total = float(weight.sum())
    if total > 0.0:
        col_sums = weight.sum(axis=0, dtype=np.float64)
        row_sums = weight.sum(axis=1, dtype=np.float64)
        cx = float(np.dot(col_sums, np.arange(width)) / total)
        cy = float(np.dot(row_sums, np.arange(height)) / total)
    else:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
    return {
        "peak": peak,
        "mean": mean,
        "centroid": (cx, cy),
        "shape": (height, width),
    }


class CCDSyncSampler(QThread):
    """在独立线程里采 CCD，并按共享派发进度给每帧打时间标签。

    Signals:
        frame_ready: ``dict``，含 ``frame`` (原始 ndarray)、统计与采样元数据；
            每次只带最新一帧。
        status_changed: 可读状态文案（连接/采集中/收尾/失败）。
        finished_summary: 线程结束时必发一次的汇总 ``dict``（含 ``status``、
            ``samples``、``output_path`` 等）；``failed`` 之后也会发。
        failed: 相机/采集异常的可读原因；只影响本采样线程。

    一次性对象：退出后不可再次 ``start()``（``QThread`` 限制），需要重采就新建。
    """

    frame_ready = Signal(object)
    status_changed = Signal(str)
    finished_summary = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        camera_type: str,
        cam_id: int | str,
        exposure_ms: float,
        dispatch_state: DispatchSnapshot,
        strategy: str,
        every_n: int = 1,
        interval_s: float = 0.1,
        output_dir: str | Path | None = None,
        plan: TransitionPlan | None = None,
        source_factory: Callable[..., FrameSource] | None = None,
    ) -> None:
        super().__init__()
        if strategy not in STRATEGIES:
            raise ValueError(
                f"未知采样策略 {strategy!r}（可选 {list(STRATEGIES)}）"
            )
        if int(every_n) < 1:
            raise ValueError(f"every_n 必须 >= 1，收到 {every_n}")
        if float(interval_s) <= 0.0:
            raise ValueError(f"interval_s 必须 > 0，收到 {interval_s}")
        if float(exposure_ms) <= 0.0:
            raise ValueError(f"exposure_ms 必须 > 0，收到 {exposure_ms}")
        if not isinstance(dispatch_state, DispatchSnapshot):
            raise TypeError("dispatch_state 必须是 DispatchSnapshot")

        self._camera_type = str(camera_type).strip().lower()
        self._cam_id = cam_id
        self._exposure_ms = float(exposure_ms)
        self._dispatch = dispatch_state
        self._strategy = strategy
        self._every_n = int(every_n)
        self._interval_s = float(interval_s)
        self._output_dir = None if output_dir is None else Path(output_dir)
        self._plan = plan
        self._source_factory = source_factory or CameraFrameSource

        self._stop_event = threading.Event()
        self._pending: tuple[np.ndarray, float] | None = None
        self._last_emit_mono: float | None = None
        self._last_emit_index: int | None = None
        self._t0 = 0.0
        self._thread_id: int | None = None

    # ---------------------------------------------------------------- 生命周期
    def request_stop(self) -> None:
        """请求停止（不等待）；线程会在当前轮询间隔内收尾。"""
        self._stop_event.set()

    def stop(self, wait_ms: int = 3000) -> bool:
        """请求停止并等待线程退出；返回是否已退出（幂等）。"""
        self.request_stop()
        if not self.isRunning():
            return True
        if threading.get_ident() == self._thread_id:
            return False  # 采样线程自己调 stop：不能 join 自己
        if int(wait_ms) <= 0:
            return False
        return bool(self.wait(int(wait_ms)))

    # ---------------------------------------------------------------- 线程体
    def run(self) -> None:
        """构造帧源 → 起流 → 轮询 ``drain_latest`` → 收尾 ``stop``。"""
        self._thread_id = threading.get_ident()
        self._t0 = time.monotonic()
        started_at = datetime.now()
        rows: list[dict[str, Any]] = []
        source: FrameSource | None = None
        status = "finished"
        error: str | None = None
        try:
            self._emit_status("正在连接 CCD…")
            source = self._make_source()
            source.start()
            self._emit_status("CCD 并行采样中（按派发进度时间对齐，非逐帧对应）")
            self._loop(source, rows)
            if self._stop_event.is_set():
                status = "stopped"
        # policy: 采样失败只让本线程结束，绝不能反向影响 SLM 派发线程
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("CCD 同步采样失败: {}", error)
            self.failed.emit(error)
        finally:
            if source is not None:
                try:
                    source.stop()
                # policy: 关闭阶段错误类型不可枚举；不能盖住真正的失败原因
                except Exception as exc:
                    logger.warning("CCD 同步采样收尾时停止帧源失败: {}", exc)
            summary = self._summary(status, error, started_at, rows, source)
            try:
                path = self._write_summary(summary)
            except Exception as exc:  # policy: 落盘失败不改变采样结论
                logger.warning("CCD 采样汇总落盘失败: {}", exc)
                path = None
            if path is not None:
                summary["output_path"] = str(path)
            self._emit_status(
                {
                    "finished": "CCD 采样完成",
                    "stopped": "CCD 采样已停止",
                    "failed": "CCD 采样失败",
                }.get(status, "CCD 采样结束")
            )
            self.finished_summary.emit(summary)

    def _loop(self, source: FrameSource, rows: list[dict[str, Any]]) -> None:
        while not self._stop_event.is_set():
            fresh = source.drain_latest()
            now = time.monotonic()
            dispatch = self._dispatch.snapshot()
            if fresh is not None:
                self._pending = fresh
                if not dispatch[2] and self._should_emit(dispatch, now):
                    self._emit_sample(
                        source, fresh[0], fresh[1], dispatch, now, rows
                    )
            if dispatch[2]:
                self._emit_final(source, rows)
                return
            self._stop_event.wait(self._interval_s)

    # ---------------------------------------------------------------- 选帧策略
    def _should_emit(self, dispatch: tuple[int, float, bool], now: float) -> bool:
        index, _elapsed, done = dispatch
        if done or index < 0:
            return False
        if self._strategy == "at-end":
            return False
        if self._strategy == "every":
            if self._last_emit_index is None:
                return True
            return index - self._last_emit_index >= self._every_n
        # periodic
        if self._last_emit_mono is None:
            return True
        return now - self._last_emit_mono >= self._interval_s

    def _emit_final(self, source: FrameSource, rows: list[dict[str, Any]]) -> None:
        """派发 done 后补最后一帧（不同策略各取所需的「收尾样本」）。"""
        dispatch = self._dispatch.snapshot()
        now = time.monotonic()
        if self._strategy == "at-end":
            frame = self._pending or self._wait_for_frame(source)
            if frame is not None:
                self._emit_sample(
                    source, frame[0], frame[1], dispatch, now, rows
                )
            return
        if self._strategy == "every":
            index = dispatch[0]
            if self._pending is None or index < 0:
                return
            if (
                self._last_emit_index is not None
                and index - self._last_emit_index < self._every_n
            ):
                return
            self._emit_sample(
                source, self._pending[0], self._pending[1], dispatch, now, rows
            )
            return
        # periodic：收尾至少留一帧，但不违反最小间隔
        if self._pending is None:
            return
        if self._last_emit_mono is not None and now - self._last_emit_mono < self._interval_s:
            return
        self._emit_sample(
            source, self._pending[0], self._pending[1], dispatch, now, rows
        )

    def _wait_for_frame(self, source: FrameSource) -> tuple[np.ndarray, float] | None:
        """at-end 收尾时短暂等一帧（相机可能恰好没有缓存）；超时返回 ``None``。"""
        deadline = time.monotonic() + max(1.0, 5.0 * self._interval_s)
        while time.monotonic() < deadline and not self._stop_event.is_set():
            frame = source.drain_latest()
            if frame is not None:
                return frame
            time.sleep(min(0.01, self._interval_s))
        return self._pending

    # ---------------------------------------------------------------- 样本与汇总
    def _emit_sample(
        self,
        source: FrameSource,
        frame: np.ndarray,
        timestamp: float,
        dispatch: tuple[int, float, bool],
        now: float,
        rows: list[dict[str, Any]],
    ) -> None:
        stats = frame_stats(frame)
        index, elapsed, done = dispatch
        row: dict[str, Any] = {
            "sample_index": len(rows) + 1,
            "strategy": self._strategy,
            # 派发侧（时间对齐用；与相机帧数没有一一对应关系）
            "dispatch_frame_index": int(index),
            "dispatch_elapsed_s": float(elapsed),
            "dispatch_progress_t": plan_progress_t(self._plan, elapsed),
            "dispatch_frame_progress": (
                float(self._plan.progress_of_frame(index - 1))
                if self._plan is not None and index >= 0
                else None
            ),
            "dispatch_done": bool(done),
            # 相机侧
            "camera_timestamp": float(timestamp),
            "camera_elapsed_s": float(now - self._t0),
            "camera_exposure_ms": float(
                getattr(source, "exposure_ms", self._exposure_ms)
            ),
            "peak": stats["peak"],
            "mean": stats["mean"],
            "centroid": stats["centroid"],
            "frame_shape": [int(v) for v in stats["shape"]],
        }
        rows.append(row)
        self._last_emit_mono = now
        self._last_emit_index = int(index)
        # 图像与行一起交给 UI；行本身不含 ndarray，便于直接落盘
        self.frame_ready.emit({**row, "frame": frame})

    def _summary(
        self,
        status: str,
        error: str | None,
        started_at: datetime,
        rows: list[dict[str, Any]],
        source: FrameSource | None,
    ) -> dict[str, Any]:
        index, elapsed, done = self._dispatch.snapshot()
        return {
            "status": status,
            "error": error,
            "strategy": self._strategy,
            "every_n": self._every_n,
            "interval_s": self._interval_s,
            "camera_type": self._camera_type,
            "cam_id": self._cam_id,
            "exposure_ms": float(getattr(source, "exposure_ms", self._exposure_ms)),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": float(time.monotonic() - self._t0),
            "sample_count": len(rows),
            "dispatch_frame_index": int(index),
            "dispatch_elapsed_s": float(elapsed),
            "dispatch_done": bool(done),
            # 明确口径，免得下游把并行采样误读成逐帧同步
            "alignment": "parallel sampling time-aligned to shared dispatch progress",
            "output_path": None,
            "samples": rows,
        }

    def _write_summary(self, summary: dict[str, Any]) -> Path | None:
        if self._output_dir is None:
            return None
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self._output_dir / f"ccd_sync_{stamp}.json"
        payload = dict(summary)
        payload["output_path"] = str(path)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    # ---------------------------------------------------------------- 内部
    def _make_source(self) -> FrameSource:
        return self._source_factory(
            camera_type=self._camera_type,
            cam_id=self._cam_id,
            exposure_ms=self._exposure_ms,
        )

    def _emit_status(self, text: str) -> None:
        self.status_changed.emit(text)
