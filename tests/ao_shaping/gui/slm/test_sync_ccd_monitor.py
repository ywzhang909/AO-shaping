"""``sync_ccd_monitor`` 的离线契约测试（假帧源, 无相机硬件）。

覆盖:
* 三种策略选帧口径: ``every`` / ``periodic`` / ``at-end``;
* ``at-end`` 只保留最新一帧（有界, 不随相机帧数增长）;
* 派发 done 后收尾、``stop()`` 幂等、线程异常安全关闭;
* ``DispatchSnapshot`` 的并发读写一致性;
* 与 ``CameraFrameSource`` 租约联动: 采样期间持有, 退出后释放;
* ``frame_stats`` / ``plan_progress_t`` 纯函数口径。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.ccd.beam_quality import ccd_lease
from ao_shaping.gui.ccd.beam_quality.frame_source import CameraFrameSource
from ao_shaping.gui.slm.sync_ccd_monitor import (
    CCDSyncSampler,
    DispatchSnapshot,
    STRATEGIES,
    frame_stats,
    plan_progress_t,
)
from ao_shaping.utils.slm.transition_timeline import TransitionPlan


@pytest.fixture(scope="module")
def qt_app():
    """离屏 QApplication（一个进程只能有一个）。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _clean_lease():
    ccd_lease.reset_for_tests()
    yield
    ccd_lease.reset_for_tests()


class _Collector(QObject):
    """主线程接收端: 信号跨线程排队, 由 ``processEvents`` 投递。"""

    def __init__(self) -> None:
        super().__init__()
        self.samples: list[dict[str, Any]] = []
        self.statuses: list[str] = []
        self.summaries: list[dict[str, Any]] = []
        self.errors: list[str] = []

    @Slot(object)
    def on_frame(self, payload: object) -> None:
        self.samples.append(payload)  # type: ignore[arg-type]

    @Slot(str)
    def on_status(self, text: str) -> None:
        self.statuses.append(text)

    @Slot(object)
    def on_summary(self, summary: object) -> None:
        self.summaries.append(summary)  # type: ignore[arg-type]

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.errors.append(message)


class _FakeSource:
    """每次 ``drain_latest`` 都产一帧最新图; 帧峰值编码序号, 便于断言「最新」。"""

    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
        size: int = 8,
    ) -> None:
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.size = int(size)
        self.exposure_ms = 6.0
        self.started = False
        self.stop_calls = 0
        self.frames_served = 0
        self.on_drain = None

    def start(self, on_frame=None) -> None:
        if self.fail_start:
            raise RuntimeError("start boom")
        self.started = True

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("stop boom")

    def drain_latest(self) -> tuple[np.ndarray, float] | None:
        if self.on_drain is not None:
            self.on_drain()
        self.frames_served += 1
        frame = np.zeros((self.size, self.size), dtype=np.uint16)
        frame[2:4, 3:5] = self.frames_served
        return frame, time.monotonic()


class _FakeBackend:
    """供真实 ``CameraFrameSource`` 使用的假相机后端。"""

    def __init__(self) -> None:
        self.exposure_ms = 6.0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def start_stream(self, on_frame) -> None:
        on_frame(np.full((4, 4), 3, dtype=np.uint8))

    def stop_stream(self) -> None:
        pass

    def get_exposure_time_ms(self) -> float:
        return self.exposure_ms


def _make_sampler(
    fake: _FakeSource,
    state: DispatchSnapshot,
    strategy: str,
    **kwargs: Any,
) -> tuple[CCDSyncSampler, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(**factory_kwargs: Any) -> _FakeSource:
        captured.update(factory_kwargs)
        return fake

    sampler = CCDSyncSampler(
        camera_type="daheng",
        cam_id=0,
        exposure_ms=5.0,
        dispatch_state=state,
        strategy=strategy,
        source_factory=factory,
        **kwargs,
    )
    return sampler, captured


def _run_to_finish(
    app: QApplication, sampler: CCDSyncSampler, timeout_s: float = 5.0
) -> None:
    assert sampler.wait(int(timeout_s * 1000)), "采样线程未在超时内退出"
    app.processEvents()  # 投递排队中的 frame/summary 信号
    app.processEvents()


def _advance_dispatch(state: DispatchSnapshot, total: int):
    """模拟 ``DynamicShapingWorker.progress``: 1 基帧号, 末帧 mark_done。"""

    def tick() -> None:
        nxt = max(1, state.frame_index + 1)
        elapsed = (nxt - 1) / max(1, total)
        state.update(frame_index=nxt, elapsed_s=elapsed)
        if nxt >= total:
            state.mark_done(elapsed_s=elapsed)

    return tick


# ---------------------------------------------------------------- DispatchSnapshot
def test_dispatch_snapshot_transitions() -> None:
    state = DispatchSnapshot()
    assert state.snapshot() == (-1, 0.0, False)

    state.update(frame_index=3, elapsed_s=0.4)
    assert state.snapshot() == (3, 0.4, False)

    state.mark_done(elapsed_s=-2.0)  # 负秒被钳位, done 置位
    assert state.snapshot() == (3, 0.0, True)


def test_dispatch_snapshot_concurrent_reads_are_consistent() -> None:
    state = DispatchSnapshot()
    stop = threading.Event()
    seen: list[tuple[int, float, bool]] = []

    def reader() -> None:
        while not stop.is_set():
            seen.append(state.snapshot())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(500):
            state.update(frame_index=i, elapsed_s=i * 0.001)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert seen, "读线程没有取到任何快照"
    assert all(elapsed >= 0.0 for _index, elapsed, _done in seen)
    assert state.snapshot()[0] == 499


# ---------------------------------------------------------------- 纯函数
def test_frame_stats_peak_and_centroid() -> None:
    img = np.zeros((20, 20), dtype=np.uint16)
    img[5:8, 10:13] = 100
    stats = frame_stats(img)
    assert stats["peak"] == 100.0
    assert stats["centroid"][0] == pytest.approx(11.0)
    assert stats["centroid"][1] == pytest.approx(6.0)
    assert stats["shape"] == (20, 20)

    # 全黑帧退化为画面中心, 不假装找到光斑
    flat = frame_stats(np.zeros((10, 10), dtype=np.uint8))
    assert flat["peak"] == 0.0
    assert flat["centroid"] == pytest.approx((4.5, 4.5))


def test_plan_progress_t_matches_wall_clock_window() -> None:
    plan = TransitionPlan(duration_s=2.0, fps=10)
    assert plan_progress_t(plan, 0.5) == pytest.approx(0.25)
    assert plan_progress_t(plan, 5.0) == pytest.approx(1.0)
    assert plan_progress_t(plan, -1.0) == pytest.approx(0.0)
    assert plan_progress_t(None, 1.0) is None


# ---------------------------------------------------------------- every
def test_every_strategy_samples_one_per_dispatch_frame(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource()
    fake.on_drain = _advance_dispatch(state, total=10)

    sampler, captured = _make_sampler(
        fake, state, "every", every_n=1, interval_s=0.01
    )
    collector = _Collector()
    sampler.frame_ready.connect(collector.on_frame)
    sampler.status_changed.connect(collector.on_status)
    sampler.finished_summary.connect(collector.on_summary)
    sampler.failed.connect(collector.on_failed)

    sampler.start()
    _run_to_finish(qt_app, sampler)

    assert collector.errors == []
    assert captured["camera_type"] == "daheng"
    assert captured["cam_id"] == 0
    assert captured["exposure_ms"] == 5.0

    indices = [row["dispatch_frame_index"] for row in collector.samples]
    assert indices == list(range(1, 11))
    assert all(isinstance(row["frame"], np.ndarray) for row in collector.samples)
    # 采样元数据: 派发帧号/时间 + 相机时间 + 峰值/质心
    for row in collector.samples:
        assert {"dispatch_elapsed_s", "camera_timestamp", "camera_elapsed_s"} <= set(row)
        assert row["peak"] > 0.0
        assert len(row["centroid"]) == 2
    assert collector.samples[-1]["dispatch_done"] is True
    assert fake.started is True and fake.stop_calls == 1

    assert len(collector.summaries) == 1
    summary = collector.summaries[0]
    assert summary["status"] == "finished"
    assert summary["sample_count"] == 10
    assert "parallel" in summary["alignment"]
    assert summary["dispatch_done"] is True
    assert summary["output_path"] is None


def test_every_n_thins_dispatch_buckets(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource()
    fake.on_drain = _advance_dispatch(state, total=10)

    sampler, _captured = _make_sampler(
        fake, state, "every", every_n=3, interval_s=0.01
    )
    collector = _Collector()
    sampler.frame_ready.connect(collector.on_frame)
    sampler.start()
    _run_to_finish(qt_app, sampler)

    assert [row["dispatch_frame_index"] for row in collector.samples] == [1, 4, 7, 10]


# ---------------------------------------------------------------- periodic
def test_periodic_strategy_respects_interval(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource()
    fake.on_drain = _advance_dispatch(state, total=10_000)  # 长时间不 done

    sampler, _captured = _make_sampler(
        fake, state, "periodic", interval_s=0.05
    )
    collector = _Collector()
    sampler.frame_ready.connect(collector.on_frame)
    sampler.finished_summary.connect(collector.on_summary)

    sampler.start()
    time.sleep(0.28)
    assert sampler.stop(3000) is True
    _run_to_finish(qt_app, sampler, timeout_s=1.0)

    assert len(collector.samples) >= 3
    gaps = [
        b["camera_elapsed_s"] - a["camera_elapsed_s"]
        for a, b in zip(collector.samples, collector.samples[1:])
    ]
    assert all(gap >= 0.04 for gap in gaps), gaps
    assert collector.summaries[0]["status"] == "stopped"
    assert collector.summaries[0]["sample_count"] == len(collector.samples)


# ---------------------------------------------------------------- at-end
def test_at_end_keeps_only_latest_frame(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource()
    total = 50
    fake.on_drain = _advance_dispatch(state, total=total)

    sampler, _captured = _make_sampler(fake, state, "at-end", interval_s=0.005)
    collector = _Collector()
    sampler.frame_ready.connect(collector.on_frame)
    sampler.start()
    _run_to_finish(qt_app, sampler)

    assert len(collector.samples) == 1, "at-end 必须是单帧, 不得随相机帧数增长"
    row = collector.samples[0]
    assert row["dispatch_frame_index"] == total
    assert row["dispatch_done"] is True
    # 帧值编码序号: 收到的必须是最新一帧, 不是最早缓存的
    assert int(row["frame"].max()) == fake.frames_served == total
    assert fake.frames_served == total, "done 后不应继续向相机要帧"


# ---------------------------------------------------------------- 生命周期/异常
def test_start_failure_reports_failed_and_closes(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource(fail_start=True)
    sampler, _captured = _make_sampler(fake, state, "every")

    collector = _Collector()
    sampler.failed.connect(collector.on_failed)
    sampler.finished_summary.connect(collector.on_summary)

    sampler.start()
    _run_to_finish(qt_app, sampler)

    assert len(collector.errors) == 1
    assert "start boom" in collector.errors[0]
    assert collector.summaries[0]["status"] == "failed"
    assert "start boom" in collector.summaries[0]["error"]
    assert fake.stop_calls == 1  # 收尾仍尝试关闭


def test_stop_error_does_not_hide_finished_status(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource(fail_stop=True)
    fake.on_drain = _advance_dispatch(state, total=3)

    sampler, _captured = _make_sampler(fake, state, "at-end", interval_s=0.005)
    collector = _Collector()
    sampler.finished_summary.connect(collector.on_summary)
    sampler.failed.connect(collector.on_failed)

    sampler.start()
    _run_to_finish(qt_app, sampler)

    assert collector.errors == []
    assert collector.summaries[0]["status"] == "finished"
    assert fake.stop_calls == 1


def test_stop_before_run_is_idempotent(qt_app) -> None:
    state = DispatchSnapshot()
    fake = _FakeSource()
    sampler, _captured = _make_sampler(fake, state, "at-end")
    collector = _Collector()
    sampler.finished_summary.connect(collector.on_summary)

    assert sampler.stop(0) is True  # 未启动时立即返回 True
    sampler.start()
    _run_to_finish(qt_app, sampler)

    assert collector.summaries[0]["status"] == "stopped"
    assert collector.summaries[0]["sample_count"] == 0
    assert fake.stop_calls == 1


def test_invalid_arguments_are_rejected() -> None:
    state = DispatchSnapshot()
    base = dict(
        camera_type="daheng",
        cam_id=0,
        exposure_ms=5.0,
        dispatch_state=state,
        strategy="every",
    )
    assert "every" in STRATEGIES
    with pytest.raises(ValueError):
        CCDSyncSampler(**{**base, "strategy": "bogus"})
    with pytest.raises(ValueError):
        CCDSyncSampler(**{**base, "every_n": 0})
    with pytest.raises(ValueError):
        CCDSyncSampler(**{**base, "interval_s": 0.0})
    with pytest.raises(ValueError):
        CCDSyncSampler(**{**base, "exposure_ms": 0.0})
    with pytest.raises(TypeError):
        CCDSyncSampler(**{**base, "dispatch_state": object()})


def test_sampler_holds_and_releases_camera_lease(qt_app) -> None:
    state = DispatchSnapshot()
    backend = _FakeBackend()
    source = CameraFrameSource(camera_type="daheng", cam_id=0, camera=backend)

    sampler = CCDSyncSampler(
        camera_type="daheng",
        cam_id=0,
        exposure_ms=5.0,
        dispatch_state=state,
        strategy="periodic",
        interval_s=0.05,
        source_factory=lambda **_kwargs: source,
    )
    collector = _Collector()
    sampler.finished_summary.connect(collector.on_summary)
    sampler.start()

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and ccd_lease.current() is None:
        qt_app.processEvents()
        time.sleep(0.005)
    assert ccd_lease.current() == ("daheng", 0)

    assert sampler.stop(3000) is True
    qt_app.processEvents()
    assert ccd_lease.current() is None
    assert backend.closed is True
    assert collector.summaries[0]["status"] == "stopped"
