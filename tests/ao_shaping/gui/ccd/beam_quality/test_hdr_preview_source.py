"""「HDR 获取」预览开关的帧源契约测试 (全程离线, 不打开任何设备)。

界面侧的每条承诺最后都落到帧源的几个数上, 所以这里锁的就是这几个数:

1. **序列化**: ``capture_hdr`` 与取流抢的是同一个相机句柄。bracket 在飞期间
   ``start_stream`` / ``set_exposure_ms`` 一次也不许被碰 —— 假相机里埋了绊线,
   一旦被碰就写下 ``ILLEGAL:*``; HDR 开着期间取流是停着的, 退出后恰好恢复一次。
2. **默认关 = 老行为**: 不碰 ``capture_hdr``, 帧来历是 ``"stream"``, fps 口径不变;
   HDR 预览帧**不进** fps EMA (那是 bracket 节拍, 不是相机流健康度)。
3. **诚实降级**: bracket 没出帧 / 后端没有 ``capture_hdr`` / 阶梯非法, 一律只记
   ``last_hdr_error`` + 把 ``hdr_enabled`` 落回 ``False`` + 退回单曝光取流, **绝不抛出**。
4. **一轮只有一个采图人**: 改阶梯与"重拍一轮"只动节拍序号, 不起第二个线程。

假相机经 ``CameraFrameSource(camera=...)`` 注入 (与 :mod:`test_daheng_source` 同一
手法): 真 SDK、真相机、真 Qt 事件循环一律不加载。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ao_shaping.gui.ccd.beam_quality.frame_source import (
    CameraFrameSource,
    SimBeamSource,
)
from ao_shaping.gui.ccd.beam_quality.hdr_preview import (
    DEFAULT_HDR_LADDER_MS,
    MAX_LADDER_RUNGS,
    default_ladder_text,
    format_ladder_ms,
    ladder_to_text,
    parse_exposure_ladder,
    preview_mode_text,
    validate_exposure_ladder,
)

#: 被测阶梯 (ms): 与界面上的默认值一致, 好让测试失败时读起来就是操作员看到的那组数。
LADDER = (1.0, 4.0, 16.0)

#: 融合结果必须是**有起伏**的斜坡: ``tone_map_hdr`` 先扣 1% 分位再拉满量程,
#: 平坦场扣完只剩 0 → 全黑, 断言就再也分不清「映射过了」和「映射成黑色」。
FUSED = np.linspace(40.0, 120.0, 16, dtype=np.float32).reshape(4, 4)


class _HdrResult:
    """只带 ``fused`` 字段的假 :class:`~ao_shaping.utils.image.hdr.HdrResult`。

    ``capture_hdr_display`` 只 ``getattr(result, "fused", None)``, 所以不必把
    ``valid`` / ``coverage`` 那一整套都编出来 —— 编了也没人读。
    """

    def __init__(self, fused: np.ndarray | None = FUSED) -> None:
        self.fused = fused


class _HdrFakeCamera:
    """会记账的假后端: 取流与 bracket 共用一个句柄, 抢用即留下 ``ILLEGAL:*``。

    - ``calls``: 设备动作流水 (``open`` / ``start_stream`` / ``stop_stream`` /
      ``capture_hdr`` / ``close``), 绊线写成 ``ILLEGAL:<动作>_during_capture_hdr``。
    - ``ladders``: 每轮 bracket 实际收到的阶梯, 用来验"改阶梯在下一轮生效"。
    - ``block``: 非 ``None`` 时**第一轮** bracket 卡在 ``block`` 上等测试放行,
      同时置 ``bracket_started`` —— 给"采集期间不许碰设备"这类断言一个可控的现场。
    """

    def __init__(
        self,
        *,
        result: _HdrResult | None = None,
        hdr_raises: bool = False,
        block: threading.Event | None = None,
        exposure_range: tuple[float, float] = (0.011, 200.0),
    ) -> None:
        self.calls: list[str] = []
        self.ladders: list[list[float]] = []
        self.n_samples: list[int] = []
        self.exposure_requests: list[float] = []
        self.block = block
        self.bracket_started = threading.Event()
        self._released = False
        self.result = result if result is not None else _HdrResult()
        self.hdr_raises = hdr_raises
        self.exposure_range = exposure_range
        self.in_capture = False
        self.streaming = False
        self.exposure_ms = 20.0
        self.pixel_size_um = 2.9
        self.last_frame_info = None
        self._on_frame: object | None = None

    # ---------------------------------------------------------------- 生命周期
    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")

    # ---------------------------------------------------------------- 取流
    def start_stream(self, on_frame) -> None:
        if self.in_capture:
            self.calls.append("ILLEGAL:start_stream_during_capture_hdr")
        self._on_frame = on_frame
        self.calls.append("start_stream")
        self.streaming = True
        # 两帧才够让 fps EMA 起步 (信箱只在第二帧才有时间差)
        for value in (7, 9):
            self._emit(value)

    def stop_stream(self) -> None:
        self.calls.append("stop_stream")
        self.streaming = False

    def _emit(self, value: int) -> None:
        if self._on_frame is not None:
            self._on_frame(np.full((4, 4), value, dtype=np.uint8))

    def get_numpy_image(self, n: int = 1, skip_first: bool = False) -> np.ndarray:
        return np.full((4, 4), 11, dtype=np.uint8)

    # ---------------------------------------------------------------- 曝光
    def get_exposure_range(self) -> tuple[float, float]:
        return self.exposure_range

    def get_exposure_time_ms(self) -> float:
        return self.exposure_ms

    def set_exposure_ms(self, ms: float) -> float:
        if self.in_capture:
            self.calls.append("ILLEGAL:set_exposure_during_capture_hdr")
        self.exposure_requests.append(float(ms))
        self.exposure_ms = float(ms)
        return self.exposure_ms

    # ---------------------------------------------------------------- HDR
    def capture_hdr(self, exposures_ms, *, n_sample: int = 1):
        self.calls.append("capture_hdr")
        self.ladders.append([float(value) for value in exposures_ms])
        self.n_samples.append(int(n_sample))
        self.in_capture = True
        try:
            if self.block is not None and not self._released:
                self.bracket_started.set()
                if not self.block.wait(5.0):  # pragma: no cover - 放行是测试的责任
                    raise AssertionError("测试没有放行被卡住的 bracket")
                self._released = True
            if self.hdr_raises:
                raise RuntimeError("SDK 掉了")
            return self.result
        finally:
            self.in_capture = False


class _NoHdrCamera(_HdrFakeCamera):
    """没有 ``capture_hdr`` 的后端 (属性置 ``None`` 即为"不可调用")。

    刻意用子类覆盖而不是 ``del type(cam).capture_hdr``: 后者会改到父类上,
    把同一次 pytest 里别的测试的假相机一起弄坏。
    """

    capture_hdr = None  # type: ignore[assignment]


# ---------------------------------------------------------------- 测试助手
def _wait_until(predicate, timeout: float = 5.0, why: str = "") -> None:
    """等到条件成立, 否则带着"在等什么"失败 —— 线程测试的失败必须能读懂。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"等超时 ({timeout}s): {why}")


def _next_frame(source, kind: str) -> np.ndarray:
    """取回第一帧来历为 ``kind`` 的画面 (信箱只留最新, 积压的旧来历直接被丢)。"""
    found: np.ndarray | None = None

    def _got() -> bool:
        nonlocal found
        drained = source.drain_latest()
        if drained is None:
            return False
        if source.last_frame_kind != kind:
            return False
        found = drained[0]
        return True

    _wait_until(_got, why=f"信箱里没有出现来历为 {kind!r} 的帧")
    assert found is not None
    return found


def _hdr_threads() -> int:
    """当前活着的 HDR 节拍线程数 (契约: 同一时刻至多一个)。"""
    return sum(1 for t in threading.enumerate() if t.name == "beam-quality-hdr")


@pytest.fixture()
def camera(request: pytest.FixtureRequest) -> _HdrFakeCamera:
    """默认造一台好相机; 参数化用例可 ``indirect=True`` 直接塞一台坏进来的。"""
    cam: _HdrFakeCamera | None = getattr(request, "param", None)
    if cam is None:
        cam = _HdrFakeCamera()
    yield cam
    assert not [call for call in cam.calls if call.startswith("ILLEGAL:")], cam.calls


@pytest.fixture()
def source(camera):
    src = CameraFrameSource(camera_type="fake", camera=camera, exposure_ms=20.0)
    yield src
    src.stop()


# ---------------------------------------------------------------- 1. 默认关 = 老行为
def test_default_off_is_the_old_single_exposure_path(source, camera) -> None:
    """开关不动时: 只有取流, ``capture_hdr`` 一次也不被调, HDR 状态全冷。"""
    source.start()

    frame = _next_frame(source, "stream")

    assert frame.dtype == np.uint8
    assert source.last_frame_kind == "stream"
    assert camera.calls == ["open", "start_stream"]
    assert source.hdr_enabled is False
    assert source.hdr_active is False
    assert source.hdr_ladder_ms == ()
    assert source.last_hdr_error is None
    # 没打开时 hdr_supported 只看得到"打算用的后端", 不碰设备 -> 乐观为 True
    assert source.hdr_supported is True


def test_sim_source_default_off_never_touches_hdr_path() -> None:
    """仿真源同样: 默认关时发的是常规单曝光合成帧, 来历标 ``stream``。"""
    src = SimBeamSource(width=64, height=48, fps=30.0)
    try:
        src.start()
        frame = _next_frame(src, "stream")
        assert frame.shape == (48, 64)
        assert src.hdr_enabled is False and src.hdr_active is False
        assert src.last_hdr_error is None
    finally:
        src.stop()


# ---------------------------------------------------------------- 2. 序列化
def test_hdr_bracket_never_overlaps_stream_and_restores_once(source, camera) -> None:
    """开 HDR 必须停流后独占; 关 HDR 必须把取流恢复回来, 且各只发生一次。"""
    source.start()
    _next_frame(source, "stream")

    source.set_hdr(True, LADDER)
    frame = _next_frame(source, "hdr")

    assert frame.dtype == np.uint8 and int(frame.max()) > 0
    assert source.hdr_enabled is True and source.hdr_active is True
    assert source.hdr_ladder_ms == LADDER
    assert source.last_hdr_error is None
    assert camera.calls.count("stop_stream") == 1
    assert camera.calls.count("start_stream") == 1  # 期间没有偷偷重起流
    assert camera.streaming is False  # 信箱里只有 bracket 节拍
    assert camera.ladders[0] == list(LADDER)
    assert _hdr_threads() == 1

    source.set_hdr(False)
    _wait_until(
        lambda: camera.calls.count("start_stream") == 2,
        why="关闭 HDR 后取流没有被恢复",
    )
    _wait_until(lambda: _hdr_threads() == 0, why="HDR 节拍线程没有退出")

    assert source.hdr_enabled is False and source.hdr_active is False
    assert camera.calls.count("stop_stream") == 1  # 恢复取流不再停第二次
    assert camera.streaming is True
    assert _next_frame(source, "stream") is not None


def test_hdr_before_start_skips_streaming_entirely(camera) -> None:
    """开机就带 HDR 意图时, 连"先起流再停流"这一趟都不必走 (MiiCam 代价是重建管线)。"""
    src = CameraFrameSource(camera_type="fake", camera=camera, exposure_ms=20.0)
    try:
        src.set_hdr(True, LADDER)
        assert camera.calls == []  # set_hdr 在未运行时只登记意图, 不碰设备

        src.start()
        _next_frame(src, "hdr")

        assert "start_stream" not in camera.calls
        assert "stop_stream" not in camera.calls
        assert src.hdr_active is True
        assert src.last_frame_kind == "hdr"

        src.set_hdr(False)
        _wait_until(lambda: "start_stream" in camera.calls, why="退出 HDR 后没恢复取流")
        assert camera.calls.count("start_stream") == 1
        assert _next_frame(src, "stream") is not None
    finally:
        src.stop()


def test_exposure_change_during_bracket_is_queued_until_next_round() -> None:
    """bracket 正在逐档积分时写曝光会污染那一档: 请求必须排队到下一轮开头。"""
    gate = threading.Event()
    cam = _HdrFakeCamera(block=gate)
    src = CameraFrameSource(camera_type="fake", camera=cam, exposure_ms=20.0)
    try:
        src.start()
        src.set_hdr(True, LADDER)
        assert cam.bracket_started.wait(5.0), "第一轮 bracket 没有按计划卡住"

        assert src.set_exposure_ms(35.0) == 35.0
        assert cam.exposure_requests == []  # 设备此刻归 bracket 所有
        assert src.exposure_ms == 35.0  # 界面显示的仍是自己发出的值

        gate.set()
        _wait_until(lambda: len(cam.ladders) >= 2, why="卡住的那轮没有收尾")
        _wait_until(lambda: 35.0 in cam.exposure_requests, why="排队曝光没有在轮间生效")

        assert cam.exposure_requests[0] == 35.0  # 只下发一次
        assert not [call for call in cam.calls if call.startswith("ILLEGAL:")]
    finally:
        gate.set()
        src.stop()


def test_ladder_change_applies_on_next_bracket_without_restart(source, camera) -> None:
    """改阶梯 = 下一轮换阶梯, 不是重起线程、也不是再停一次流。"""
    source.start()
    source.set_hdr(True, LADDER)
    _next_frame(source, "hdr")

    source.set_hdr(True, (2.0, 8.0))
    _wait_until(lambda: camera.ladders[-1] == [2.0, 8.0], why="新阶梯没有生效")

    assert source.hdr_ladder_ms == (2.0, 8.0)
    assert source.hdr_active is True
    assert camera.calls.count("stop_stream") == 1
    assert camera.calls.count("start_stream") == 1
    assert _hdr_threads() == 1


# ---------------------------------------------------------------- 3. 诚实降级
@pytest.mark.parametrize(
    ("ladder", "reason"),
    [
        ((), "至少 2 档"),
        ((5.0,), "至少 2 档"),
        ((0.0, 5.0), "有限正数"),
        ((-3.0, 5.0), "有限正数"),
        ((float("nan"), 5.0), "有限正数"),
        ((float("inf"), 5.0), "有限正数"),
        ((4.0, 4.0), "去重后只剩 1 档"),
        (tuple(range(1, MAX_LADDER_RUNGS + 2)), "最多 10 档"),
    ],
    ids=["空", "一档", "零", "负", "nan", "inf", "重复", "超上限"],
)
def test_invalid_ladder_is_refused_without_touching_the_stream(
    source, camera, ladder, reason
) -> None:
    """非法阶梯在 ``set_hdr`` 里就被挡下: 记原因、开关落回、画面继续走取流。

    这条承诺的意义是"界面不许骗人": 操作员填错数字, 看到的是屏幕上明明白白的
    单曝光画面, 而不是一个亮着的 HDR 勾。
    """
    source.start()
    before = list(camera.calls)

    source.set_hdr(True, ladder)  # 不抛

    assert source.hdr_enabled is False
    assert source.hdr_active is False
    assert reason in (source.last_hdr_error or "")
    assert camera.calls == before  # 停流/重起流一次也没发生
    assert camera.streaming is True
    assert _hdr_threads() == 0
    assert _next_frame(source, "stream") is not None


@pytest.mark.parametrize(
    "camera",
    [
        pytest.param(_HdrFakeCamera(result=_HdrResult(None)), id="无融合帧"),
        pytest.param(_HdrFakeCamera(hdr_raises=True), id="bracket 抛错"),
    ],
    indirect=True,
)
def test_hdr_failure_falls_back_to_stream_without_raising(source, camera) -> None:
    """bracket 失败 = 退回单曝光 + 留下一句原因, 绝不把异常冒到界面/线程外。"""
    source.start()
    source.set_hdr(True, LADDER)

    _wait_until(lambda: source.hdr_enabled is False, why="失败后开关没有落回")
    _wait_until(lambda: camera.calls.count("start_stream") == 2, why="失败后没恢复取流")

    assert source.hdr_active is False
    assert "未取到融合帧" in (source.last_hdr_error or "")
    assert camera.streaming is True
    assert _next_frame(source, "stream") is not None
    # 只试一轮: 降级后不许砸着相机无限重试
    assert len(camera.ladders) <= 2


def test_backend_without_capture_hdr_reverts_to_stream() -> None:
    """后端根本没有 ``capture_hdr`` 时: 如实报不支持, 画面停在单曝光取流上。"""
    cam = _NoHdrCamera()
    src = CameraFrameSource(camera_type="fake", camera=cam, exposure_ms=20.0)
    try:
        src.start()
        assert src.hdr_supported is False

        src.set_hdr(True, LADDER)
        _wait_until(lambda: src.hdr_enabled is False, why="不支持的后端没有退回")
        _wait_until(lambda: cam.calls.count("start_stream") == 2, why="没有恢复取流")

        assert src.hdr_active is False
        assert "未取到融合帧" in (src.last_hdr_error or "")
        assert cam.calls.count("stop_stream") == 1
        assert _next_frame(src, "stream") is not None
    finally:
        src.stop()
        assert not [call for call in cam.calls if call.startswith("ILLEGAL:")]


def test_stop_during_hdr_does_not_resume_stream(source, camera) -> None:
    """关闭源时不该把取流"恢复"回来: 设备马上就要关了, 恢复等于边关边起流。"""
    source.start()
    source.set_hdr(True, LADDER)
    _next_frame(source, "hdr")

    source.stop()

    assert camera.calls.count("start_stream") == 1  # 只有最初那一次
    assert "close" in camera.calls
    assert source.hdr_active is False
    assert _hdr_threads() == 0
    assert camera.streaming is False


# ---------------------------------------------------------------- 4. 一轮一个采图人
def test_retake_while_bracket_in_flight_keeps_one_capturer() -> None:
    """bracket 在飞时点"重拍一轮": 只把节拍序号 +1, 绝不起第二个抓相机的人。"""
    gate = threading.Event()
    cam = _HdrFakeCamera(block=gate)
    src = CameraFrameSource(camera_type="fake", camera=cam, exposure_ms=20.0)
    try:
        src.start()
        src.set_hdr(True, LADDER)
        assert cam.bracket_started.wait(5.0), "第一轮 bracket 没有按计划卡住"

        assert src.run_one_hdr_cycle() is True
        assert _hdr_threads() == 1  # 补拍请求没有派生第二个线程

        gate.set()
        _wait_until(lambda: len(cam.ladders) >= 2, why="补拍请求没有被下一轮接住")
        _next_frame(src, "hdr")
        assert _hdr_threads() == 1
    finally:
        gate.set()
        src.stop()
        assert not [call for call in cam.calls if call.startswith("ILLEGAL:")]


def test_retake_without_running_loop_reports_honestly(source) -> None:
    """回路没跑 (源未开 / 已退回) 时补拍返回 ``False`` —— 界面据此不该显示在拍。"""
    assert source.run_one_hdr_cycle() is False
    assert source.last_hdr_error == "HDR 未开启"

    source.set_hdr(True, LADDER)  # 源还没 start, 只登记意图
    assert source.run_one_hdr_cycle() is False
    assert _hdr_threads() == 0


def test_hdr_frames_stay_out_of_the_stream_fps(source, camera) -> None:
    """HDR 预览是 bracket 节拍, 不许混进"相机流健康度"那个 fps 读数。"""
    source.start()
    _next_frame(source, "stream")
    stream_fps = source.fps

    source.set_hdr(True, LADDER)
    _next_frame(source, "hdr")
    _wait_until(lambda: len(camera.ladders) >= 2, why="没有跑到第二轮 bracket")

    assert source.fps == stream_fps


def test_latest_frame_queue_tags_and_keeps_fps_for_stream_only() -> None:
    """信箱单元: 来历随帧一起交出, 派生帧不动 fps EMA。"""
    from ao_shaping.gui.ccd.beam_quality.frame_source import _LatestFrameQueue

    box = _LatestFrameQueue()
    box.push(np.zeros((2, 2), dtype=np.uint8), 100.0)
    box.push(np.full((2, 2), 7, dtype=np.uint8), 100.1)
    assert box.fps == pytest.approx(10.0)

    box.push_display(np.full((2, 2), 9, dtype=np.uint8), 105.0)
    assert box.fps == pytest.approx(10.0), "派生帧被算进了流帧率"

    latest = box.drain_with_kind()
    assert latest is not None and latest[2] == "hdr"
    assert box.drain_with_kind() is None


# ---------------------------------------------------------------- 5. 仿真源走同一条通路
def test_sim_source_hdr_goes_through_the_real_fuse_path() -> None:
    """仿真的 HDR 也是真算法 (``hdr_fuse`` + Reinhard), 不是换个颜色糊弄界面。"""
    src = SimBeamSource(width=64, height=48, fps=30.0)
    try:
        src.start()
        _next_frame(src, "stream")

        src.set_hdr(True, LADDER)
        hdr_frame = _next_frame(src, "hdr")

        assert src.hdr_supported is True
        assert src.hdr_active is True
        assert src.hdr_ladder_ms == LADDER
        assert hdr_frame.shape == (48, 64)
        assert hdr_frame.dtype == np.uint8
        assert int(hdr_frame.max()) == 255, "没经过色调映射的满量程拉伸"
        assert src.run_one_hdr_cycle() is True

        src.set_hdr(False)
        assert _next_frame(src, "stream") is not None
        assert src.hdr_active is False
    finally:
        src.stop()


def test_sim_source_rejects_bad_ladder_like_the_real_one() -> None:
    """仿真源与真源共用一份阶梯规则: 非法值同样只记原因、不改画面。"""
    src = SimBeamSource(width=64, height=48, fps=30.0)
    try:
        src.start()
        src.set_hdr(True, (5.0,))
        assert src.hdr_enabled is False
        assert "至少 2 档" in (src.last_hdr_error or "")
        assert _next_frame(src, "stream") is not None
    finally:
        src.stop()


# ---------------------------------------------------------------- 6. 阶梯规则 (界面上那行字)
def test_default_ladder_is_the_probe_default() -> None:
    """"预览默认阶梯"与"探测默认阶梯"必须是同一个数, 不许长成两套。"""
    from ao_shaping.tools.slm.transfer.hdr_probe import DEFAULT_HDR_EXPOSURES_MS

    assert DEFAULT_HDR_LADDER_MS == (1.0, 4.0, 16.0)
    assert DEFAULT_HDR_LADDER_MS == tuple(DEFAULT_HDR_EXPOSURES_MS)
    assert default_ladder_text() == "1, 4, 16"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1, 4, 16", (1.0, 4.0, 16.0)),
        ("1，4、16", (1.0, 4.0, 16.0)),
        (" 1;4 16 ", (1.0, 4.0, 16.0)),
        ("1,4,16,abc", (1.0, 4.0, 16.0)),
        ("", ()),
        (",,;", ()),
    ],
    ids=["逗号", "中文标点", "分号空格", "混进杂字", "空", "只有分隔符"],
)
def test_parse_exposure_ladder(raw: str, expected: tuple[float, ...]) -> None:
    assert parse_exposure_ladder(raw) == expected


def test_validate_sorts_and_dedups() -> None:
    assert validate_exposure_ladder((16.0, 1.0, 4.0, 4.0)) == (1.0, 4.0, 16.0)
    assert validate_exposure_ladder((1.0, 4.0), max_ms=200.0) == (1.0, 4.0)


def test_validate_refuses_ladder_beyond_camera_range() -> None:
    with pytest.raises(ValueError, match="超出相机量程上限 200 ms: 400"):
        validate_exposure_ladder((1.0, 400.0), max_ms=200.0)


def test_preview_mode_text_tells_the_truth() -> None:
    """模式行: 单曝光时绝不提阶梯, HDR 时说清阶梯/参考/色调映射, 降级要留痕。"""
    assert preview_mode_text(False, (1.0, 4.0, 16.0)) == "预览模式: 单曝光取流"

    text = preview_mode_text(
        True, (1.0, 4.0, 16.0), tone_map="Reinhard", reference_ms=16.0
    )
    assert "HDR 多曝光融合" in text
    assert "阶梯 1/4/16 ms" in text
    assert "参考 16 ms" in text
    assert "显示 Reinhard" in text

    degraded = preview_mode_text(True, (1.0, 4.0), stale="退回单曝光取流")
    assert degraded.endswith("· 退回单曝光取流")
    assert format_ladder_ms((1.0, 4.0, 16.0)) == "1/4/16"
    assert ladder_to_text((1.0, 4.0, 16.0)) == "1, 4, 16"
