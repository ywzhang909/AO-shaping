"""「HDR 获取」开关在界面上的行为 (离屏 Qt, 全程不打开任何设备)。

这里锁的是**界面不许骗人**这一条: 操作员看见的三样东西 —— 勾上的开关、
"预览模式"那一行、统计信息里的读数口径 —— 必须与帧源那边的实际情况一致。
帧源侧的序列化与降级契约在 ``beam_quality/test_hdr_preview_source.py`` 里锁,
本文件只用假帧源检查界面接线。

所有用例都把假源直接挂到 ``window._source`` 上 (不走 ``open_source`` 的后台打开),
收尾一律先 ``window._source = None`` 再 ``close()``, 免得窗口关掉时还在轮询假源。
"""

from __future__ import annotations

import os
import time
from collections import deque

# 必须在任何 QApplication 之前设置: CI / 无显示环境下用离屏后端
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ao_shaping.gui.ccd.beam_quality import app as app_module
from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp
from ao_shaping.gui.ccd.beam_quality.hdr_preview import (
    DEFAULT_HDR_LADDER_MS,
    default_ladder_text,
    validate_exposure_ladder,
)

LADDER = (1.0, 4.0, 16.0)


def _pump_until(qt_app, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qt_app.processEvents()
    return bool(predicate())


@pytest.fixture(scope="module")
def qt_app():
    """离屏 QApplication (一个进程只能有一个)。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _PlainSource:
    """只有界面用得上的最小帧源表面, **没有**任何 HDR 成员。

    界面的约定是 ``getattr`` 逐个探, 所以这种源必须被认出"没有 HDR 通路"并禁用开关,
    而不是抛 AttributeError 把预览弄死。
    """

    is_running = True
    fps = 50.0
    last_frame_info = None
    exposure_range_ms = (0.011, 200.0)

    def __init__(self) -> None:
        self.exposure_ms = 20.0
        self.requests: list[float] = []
        self.last_frame_kind = "stream"
        self._mailbox: deque[tuple[np.ndarray, float, str]] = deque()

    def set_exposure_ms(self, ms: float) -> float:
        self.requests.append(float(ms))
        self.exposure_ms = float(ms)
        return self.exposure_ms

    def offer(self, img: np.ndarray, kind: str) -> None:
        self._mailbox.appendleft((img, time.time(), kind))

    def drain_latest(self):
        if not self._mailbox:
            return None
        img, timestamp, kind = self._mailbox.pop()  # 最新优先 (信箱语义)
        self.last_frame_kind = kind
        return img, timestamp


class _FakeHdrSource(_PlainSource):
    """带 HDR 表面的假源: 那边怎么回答, 由构造参数决定。

    - ``accept=False``: 模拟"开了 HDR 但源侧失败退回单曝光" (bracket 没出帧 /
      后端不支持), 界面必须把开关回弹并留下原因。
    - ``hdr_supported=False``: 有 ``set_hdr`` 但自称没有多曝光通路。
    - 阶梯合法性用**同一份** :func:`validate_exposure_ladder` 判, 与真源一致,
      免得假源变成一个比真源更宽松的口径。
    """

    def __init__(
        self,
        *,
        accept: bool = True,
        supported: bool = True,
        error: str | None = None,
        exposure_range: tuple[float, float] = (0.011, 200.0),
    ) -> None:
        super().__init__()
        self.exposure_range_ms = exposure_range
        self.hdr_calls: list[tuple[bool, tuple[float, ...]]] = []
        self.retakes = 0
        self.hdr_enabled = False
        self.hdr_active = False
        self.hdr_supported = supported
        self.last_hdr_error: str | None = None
        self._accept = accept
        self._error = error or "HDR 未取到融合帧 (相机不支持 / 无有效像素), 详见日志"

    def set_hdr(self, enabled, ladder_ms=None, *, n_sample: int = 1) -> None:
        del n_sample
        want = bool(enabled)
        ladder = tuple(float(value) for value in (ladder_ms or ()))
        self.hdr_calls.append((want, ladder))
        if want:
            try:
                validate_exposure_ladder(ladder)
            except ValueError as exc:
                self.hdr_enabled = self.hdr_active = False
                self.last_hdr_error = str(exc)
                return
            self.hdr_enabled = self.hdr_active = self._accept
            self.last_hdr_error = None if self._accept else self._error
        else:
            self.hdr_enabled = self.hdr_active = False
            self.last_hdr_error = None

    def run_one_hdr_cycle(self) -> bool:
        self.retakes += 1
        return self.hdr_active


@pytest.fixture()
def make_window(qt_app):
    """造一台"源已打开"的窗口: 挂假源 → 探 HDR 能力 → 刷曝光量程 → 放行控件。"""
    built: list[BeamQualityApp] = []

    def _factory(source) -> BeamQualityApp:
        window = BeamQualityApp()
        window._source = source
        window._sync_hdr_capability(source)
        window._sync_exposure_range(source)
        window._set_controls_enabled(True)
        built.append(window)
        return window

    yield _factory

    for window in built:
        window._source = None
        window.close()


def _beam(frame: np.ndarray) -> np.ndarray:
    """给一帧"有东西可析"的画面: 中心亮斑 + 背景, uint8。"""
    size = frame.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r2 = (xx - size / 2) ** 2 + (yy - size / 2) ** 2
    return (5.0 + 200.0 * np.exp(-r2 / 18.0)).astype(np.uint8)


# ---------------------------------------------------------------- 控件默认态
def test_switch_defaults_off_and_ladder_is_gated(qt_app) -> None:
    """默认必须与改动前完全一致: 勾灭着、阶梯与补拍都灰着、文本是默认阶梯。"""
    window = BeamQualityApp()
    try:
        assert window._hdr_check.text() == "HDR 获取"
        assert window._hdr_check.isChecked() is False
        assert window._hdr_enabled is False
        assert window._view_is_hdr is False
        assert window._hdr_seq == 0
        assert window._hdr_ladder_ms == DEFAULT_HDR_LADDER_MS
        assert window._hdr_ladder_edit.text() == default_ladder_text()
        assert window._hdr_ladder_edit.isEnabled() is False
        assert window._hdr_retake_btn.isEnabled() is False
        assert "单曝光" in window._hdr_check.toolTip()
        assert window._preview_mode_label.text() == "预览模式: 单曝光取流"
    finally:
        window.close()


def test_switch_disabled_for_source_without_hdr_path(make_window) -> None:
    """没有 HDR 成员的源: 开关禁用并解释原因, 不显示一个按了不亮的勾。"""
    source = _PlainSource()
    window = make_window(source)

    assert window._hdr_supported is False
    assert window._hdr_check.isEnabled() is False
    assert "没有多曝光采集通路" in window._hdr_check.toolTip()

    window._hdr_ladder_edit.setText("1, 4")
    window._on_hdr_ladder_edited()  # 记着但不生效
    assert window._hdr_ladder_ms == (1.0, 4.0)
    assert window._preview_mode_label.text() == "预览模式: 单曝光取流"


# ---------------------------------------------------------------- 开关 → 帧源
def test_toggling_pushes_the_choice_to_the_source(make_window, qt_app) -> None:
    source = _FakeHdrSource()
    window = make_window(source)
    assert window._hdr_supported is True and window._hdr_check.isEnabled() is True

    # 阶梯与补拍是 HDR 的从属控件: 开关灭着时整组灰掉, 免得像在生效
    assert window._hdr_ladder_edit.isEnabled() is False
    assert window._hdr_retake_btn.isEnabled() is False

    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    assert source.hdr_calls == [(True, LADDER)]
    assert window._hdr_enabled is True
    assert window._hdr_ladder_edit.isEnabled() is True
    assert window._hdr_retake_btn.isEnabled() is True
    assert "HDR 多曝光融合" in window._preview_mode_label.text()
    assert "阶梯 1/4/16 ms" in window._preview_mode_label.text()
    assert "参考 16 ms" in window._preview_mode_label.text()
    assert "显示 Reinhard" in window._preview_mode_label.text()
    assert "逐档 bracket" in window.statusBar().currentMessage()

    window._hdr_check.setChecked(False)
    assert _pump_until(qt_app, lambda: not window._busy)

    assert source.hdr_calls[-1] == (False, LADDER)
    assert window._hdr_enabled is False
    assert window._hdr_ladder_edit.isEnabled() is False
    assert window._preview_mode_label.text() == "预览模式: 单曝光取流"


def test_switch_rolls_back_when_the_source_falls_back(make_window, qt_app) -> None:
    """源侧失败退回单曝光后, 界面上的勾必须跟着灭 —— 亮着比报错更难查。"""
    source = _FakeHdrSource(accept=False)
    window = make_window(source)

    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    assert source.hdr_calls == [(True, LADDER)]
    assert window._hdr_check.isChecked() is False
    assert window._hdr_enabled is False
    assert window._last_hdr_error == source.last_hdr_error
    # 开关灭着时模式行只说"单曝光取流" (这时候提阶梯反而误导), 原因走状态栏那一句
    assert window._preview_mode_label.text() == "预览模式: 单曝光取流"
    assert "HDR 预览已退回单曝光: HDR 未取到融合帧" in window.statusBar().currentMessage()
    assert window._hdr_ladder_edit.isEnabled() is False


def test_auto_exposure_and_switch_are_mutually_exclusive(make_window, qt_app) -> None:
    """闭环与 bracket 抢同一个曝光旋钮: 自动曝光优先, HDR 如实让位。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)
    assert window._hdr_enabled is True

    window._auto_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    assert window._hdr_check.isChecked() is False
    assert window._hdr_enabled is False
    assert source.hdr_calls[-1] == (False, LADDER)
    assert window._last_hdr_error == "自动曝光已接管曝光, HDR 预览让位"
    # 让位那句话必须留在最后一条状态里, 不能被"闭环已开启"覆盖掉
    assert "自动曝光已开启" in window.statusBar().currentMessage()
    assert "已关闭 HDR 预览" in window.statusBar().currentMessage()
    # 自动曝光开着期间开关整组禁用, 免得操作员再点一次去砸相机
    assert window._hdr_check.isEnabled() is False
    assert window._hdr_ladder_edit.isEnabled() is False


def test_switch_disabled_while_auto_exposure_is_on(make_window) -> None:
    """先开闭环再想开 HDR: 开关是灰的, 走不到动相机那一步。"""
    window = make_window(_FakeHdrSource())
    window._auto_check.setChecked(True)

    assert window._hdr_supported is True
    assert window._hdr_check.isEnabled() is False


# ---------------------------------------------------------------- 阶梯文本框
def test_invalid_ladder_never_reaches_the_camera(make_window, qt_app) -> None:
    """阶梯先过 UI 侧校验: 非法就回弹开关 + 说清原因, 一次设备操作都不发起。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_ladder_edit.setText("5")

    window._hdr_check.setChecked(True)

    assert window._hdr_check.isChecked() is False
    assert window._hdr_enabled is False
    assert source.hdr_calls == []
    assert window._hdr_ladder_ms == DEFAULT_HDR_LADDER_MS
    assert "至少 2 档" in window.statusBar().currentMessage()
    assert "预览模式: 单曝光取流" == window._preview_mode_label.text()


def test_ladder_edit_bad_value_keeps_the_running_ladder(make_window, qt_app) -> None:
    """HDR 开着时填坏阶梯: 文本框回写成现用值, 帧源仍按老阶梯跑。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    window._hdr_ladder_edit.setText("1, 400")  # 源量程上限 200 ms
    window._on_hdr_ladder_edited()

    assert window._hdr_ladder_ms == LADDER
    assert window._hdr_ladder_edit.text() == "1, 4, 16"
    assert source.hdr_calls[-1] == (True, LADDER)
    assert "超出相机量程上限 200 ms: 400" in window.statusBar().currentMessage()
    assert "已退回单曝光" in window._preview_mode_label.text()


def test_ladder_edit_good_value_is_confirmed_and_pushed(make_window, qt_app) -> None:
    """改对阶梯: 文本按升序去重回写, 并说清"下一轮 bracket 起用"。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    window._hdr_ladder_edit.setText("16，4、4 2")
    window._on_hdr_ladder_edited()
    assert _pump_until(qt_app, lambda: not window._busy)  # 交阶梯也是后台作业

    assert window._hdr_ladder_ms == (2.0, 4.0, 16.0)
    assert window._hdr_ladder_edit.text() == "2, 4, 16"
    assert source.hdr_calls[-1] == (True, (2.0, 4.0, 16.0))
    assert "下一轮 bracket 起用" in window.statusBar().currentMessage()
    assert "阶梯 2/4/16 ms" in window._preview_mode_label.text()
    assert window._last_hdr_error == ""


# ---------------------------------------------------------------- 重拍一轮
def test_retake_goes_through_the_source_loop(make_window, qt_app) -> None:
    source = _FakeHdrSource()
    window = make_window(source)

    # HDR 灭着时补拍按钮整组是灰的 (点不动), 处理函数自己也有一道同样的闸
    assert window._hdr_retake_btn.isEnabled() is False
    window._on_hdr_retake()
    assert source.retakes == 0
    assert "先打开" in window.statusBar().currentMessage()

    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    window._hdr_retake_btn.click()
    assert _pump_until(qt_app, lambda: not window._busy)

    assert source.retakes == 1
    assert "补拍已发出" in window.statusBar().currentMessage()
    assert window._hdr_check.isChecked() is True


def test_retake_says_so_when_the_loop_is_not_running(make_window, qt_app) -> None:
    """回路没跑还点补拍: 如实说"没发出去", 不假装在拍。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)
    source.hdr_active = False  # 节拍线程已经退了 (源侧失败/还没起来)

    window._hdr_retake_btn.click()
    assert _pump_until(qt_app, lambda: not window._busy)

    assert source.retakes == 1
    assert "补拍没有发出去" in window.statusBar().currentMessage()
    assert "HDR 回路未运行" in window._preview_mode_label.text()


# ---------------------------------------------------------------- 只上屏的帧
class _CountingAverage:
    """替身平均窗口: 记住每一张进过平均的帧。

    界面在 ``_on_poll`` 里把取流帧交给 ``_averager.push()``, HDR 显示帧必须绕过它 ——
    显示图不是任何一个曝光下的原始计数, 混进平均窗口就等于把假数据喂给特征计算。
    """

    def __init__(self) -> None:
        self.size = 1
        self.pushed: list[np.ndarray] = []

    @property
    def filled(self) -> int:
        return len(self.pushed)

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.pushed.append(frame)
        return frame

    def reset(self) -> None:
        self.pushed.clear()


def test_hdr_frame_only_touches_the_screen(make_window, qt_app) -> None:
    """信箱标成 hdr 的那一帧: 只换屏, 原始帧/平均/读数/趋势缓冲/已处理都不许动。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._averager = _CountingAverage()
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    stream = _beam(np.zeros((64, 64), dtype=np.uint8))
    source.offer(stream, "stream")
    window._on_poll()
    baseline = window._last_metrics
    assert baseline is not None
    assert window._processed == 1
    assert len(window._averager.pushed) == 1
    assert len(window._buffer) == 1  # 第一条趋势采样 (初始 _last_append_ts = 0)
    assert window._view_is_hdr is False and window._hdr_seq == 0

    display = _beam(np.zeros((80, 80), dtype=np.uint8))
    stamp = time.time()
    source.offer(display, "hdr")
    window._on_poll()

    # 上屏 + 记数: 操作员要能看出屏上是预览
    assert window._view_is_hdr is True
    assert window._hdr_seq == 1
    assert window._hdr_frame_ts >= stamp
    assert window._last_view is display
    # 其余一律不动: 这些才是"测量口径"
    assert window._last_frame is stream
    assert window._last_metrics is baseline
    assert window._processed == 1
    assert window._averager.pushed == [stream]
    assert len(window._buffer) == 1


def test_a_stream_frame_after_the_hdr_one_puts_the_readout_back(make_window, qt_app) -> None:
    """取流帧回来后口径立刻翻回单曝光: _view_is_hdr 是"这一帧的来历", 不是设置。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    source.offer(_beam(np.zeros((64, 64), dtype=np.uint8)), "hdr")
    window._on_poll()
    assert window._view_is_hdr is True

    stream = _beam(np.zeros((80, 80), dtype=np.uint8))
    source.offer(stream, "stream")
    window._on_poll()

    assert window._view_is_hdr is False
    assert window._last_frame is stream
    assert window._hdr_seq == 1  # 预览帧数是历史, 不跟着口径清零


# ---------------------------------------------------------------- 信息栏口径
def test_stats_name_the_ladder_in_use_and_the_reading_basis(make_window, qt_app) -> None:
    """开着 HDR 时信息栏得说清两件事: 现用哪一版阶梯、屏上这帧是什么口径。"""
    source = _FakeHdrSource()
    window = make_window(source)
    frame = _beam(np.zeros((64, 64), dtype=np.uint8))

    # 开关灭着: 与改动前逐字一致, 一个字都不该提 HDR
    window._update_stats(frame)
    assert "曝光阶梯" not in window._info_label.text()
    assert "HDR" not in window._info_label.text()
    assert "HDR" not in window._stats_label.text()

    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    # 预览还没出第一帧: 屏上仍是取流画面, 读数口径不许冒充 HDR 派生
    window._update_stats(frame)
    assert "曝光阶梯: 1/4/16 ms (等待第一帧…)" in window._info_label.text()
    assert "读数: 单曝光原始帧" in window._info_label.text()
    assert "HDR" not in window._stats_label.text()

    source.offer(_beam(np.zeros((80, 80), dtype=np.uint8)), "hdr")
    window._on_poll()
    assert window._hdr_seq == 1
    window._update_stats(window._last_view)
    info = window._info_label.text()
    assert "曝光阶梯: 1/4/16 ms (已画 1 帧)" in info
    assert "读数: HDR 预览显示派生 · 第 1 帧" in info
    assert window._stats_label.text().endswith("| HDR 预览 1 帧")

    # 阶梯改成什么, 那行就说什么: 不能继续挂着构造时的默认值
    window._hdr_ladder_ms = (2.0, 8.0)
    window._update_stats(window._last_view)
    assert "曝光阶梯: 2/8 ms" in window._info_label.text()

    # 降级原因得同时出现在阶梯那行: 只写在模式行里, 看读数的人照样会误会
    window._last_hdr_error = "融合后没有有效像素"
    window._update_stats(window._last_view)
    assert "已退回单曝光: 融合后没有有效像素" in window._info_label.text()


def test_preview_frames_do_not_count_as_processed(make_window, qt_app) -> None:
    """状态栏的"已处理"只算测量过的帧, 预览帧单独一列。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)

    for _ in range(3):
        source.offer(_beam(np.zeros((64, 64), dtype=np.uint8)), "hdr")
        window._on_poll()
    window._update_stats(window._last_view)

    assert window._hdr_seq == 3
    assert window._processed == 0
    assert window._stats_label.text().startswith("已处理 0 帧")
    assert window._stats_label.text().endswith("| HDR 预览 3 帧")


# ---------------------------------------------------------------- 存快照
def _patch_dialog(monkeypatch, calls: list, result):
    """把保存对话框换成记录器: 存快照这条路上唯一会碰盘的地方就是它。"""

    def _fake(*args, **kwargs):
        calls.append(args)
        return result

    monkeypatch.setattr(app_module.QFileDialog, "getSaveFileName", _fake)


def test_snapshot_is_refused_while_the_screen_is_hdr(make_window, qt_app, monkeypatch) -> None:
    """屏上是 HDR 显示图时存快照: 直接拒, 连对话框都不该弹 —— 存哪一张口径都不符。"""
    calls: list = []
    _patch_dialog(monkeypatch, calls, ("/tmp/never.png", "PNG 图像 (*.png)"))

    source = _FakeHdrSource()
    window = make_window(source)
    stream = _beam(np.zeros((64, 64), dtype=np.uint8))
    source.offer(stream, "stream")
    window._on_poll()  # 手里先有一帧可存的单曝光原始画面
    assert window._last_frame is stream

    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)
    source.offer(_beam(np.zeros((80, 80), dtype=np.uint8)), "hdr")
    window._on_poll()
    assert window._view_is_hdr is True

    window.save_snapshot()

    assert calls == []  # 弹了对话框就等于允许操作员存出一张假数据
    message = window.statusBar().currentMessage()
    assert "不存快照" in message
    assert "关闭 HDR 获取" in message  # 还得说清怎么才能存
    assert window._last_frame is stream  # 拒绝不等于顺手改状态


def test_snapshot_still_asks_for_a_path_on_single_exposure(make_window, monkeypatch) -> None:
    """反向对照: 灭着 HDR 时那道闸不拦路 —— 拒的是 HDR 预览, 不是存快照本身。"""
    calls: list = []
    _patch_dialog(monkeypatch, calls, ("", ""))  # 操作员取消, 不写盘

    source = _FakeHdrSource()
    window = make_window(source)
    source.offer(_beam(np.zeros((64, 64), dtype=np.uint8)), "stream")
    window._on_poll()

    window.save_snapshot()

    assert len(calls) == 1
    default_path = str(calls[0][2])
    assert "snapshots" in default_path and default_path.endswith(".png")
    assert "不存快照" not in window.statusBar().currentMessage()


# ---------------------------------------------------------------- 与帧源对账
def test_a_failure_left_by_the_source_is_mirrored_onto_the_switch(make_window) -> None:
    """帧源自己退回单曝光 (open 时注入的 set_hdr 失败就是这种), 界面一同步就得灭。"""
    source = _FakeHdrSource(accept=False)
    source.set_hdr(True, LADDER)  # 不经界面: 模拟打开帧源那一趟就失败了
    assert source.hdr_enabled is False and source.last_hdr_error

    window = make_window(source)
    window._hdr_enabled = True  # 界面这一侧还停留在操作员刚按下的状态
    window._sync_hdr_check()
    assert window._hdr_check.isChecked() is True

    window._sync_source_hdr_state()
    window._refresh_preview_mode()

    assert window._hdr_enabled is False
    assert window._hdr_check.isChecked() is False
    assert window._last_hdr_error == source.last_hdr_error
    assert window._preview_mode_label.text() == "预览模式: 单曝光取流"
    assert "HDR 预览已退回单曝光" in window.statusBar().currentMessage()
    assert window._hdr_ladder_edit.isEnabled() is False
    assert window._hdr_retake_btn.isEnabled() is False


def test_the_mirror_does_not_invent_a_failure(make_window) -> None:
    """源那边好着呢: 同步既不能灭开关, 也不能留着上一轮的旧原因。"""
    source = _FakeHdrSource()
    source.set_hdr(True, LADDER)
    assert source.hdr_enabled is True

    window = make_window(source)
    window._hdr_enabled = True
    window._sync_hdr_check()
    window._last_hdr_error = "上一轮留下的旧原因"

    window._sync_source_hdr_state()

    assert window._hdr_enabled is True
    assert window._hdr_check.isChecked() is True
    assert window._last_hdr_error == ""
    assert source.hdr_calls == [(True, LADDER)]  # 同步只读不写, 不该再动一次源


def test_poll_re_reads_the_switch_once_the_bracket_fails_mid_run(make_window, qt_app) -> None:
    """节拍线程在后台某一轮挂了: 到 1 Hz 对账点界面跟着灭, 没到点则不去搅。"""
    source = _FakeHdrSource()
    window = make_window(source)
    window._hdr_check.setChecked(True)
    assert _pump_until(qt_app, lambda: not window._busy)
    assert window._hdr_enabled is True

    # 源侧自己退回单曝光并记下原因 (约定: 失败不抛)
    source.hdr_enabled = source.hdr_active = False
    source.last_hdr_error = "第 2 档取帧超时"

    window._ui_fps_since = time.monotonic()  # 还没到 1 Hz
    source.offer(_beam(np.zeros((64, 64), dtype=np.uint8)), "stream")
    window._on_poll()
    assert window._hdr_enabled is True
    assert window._hdr_check.isChecked() is True

    window._ui_fps_since = time.monotonic() - 2.0  # 到点
    source.offer(_beam(np.zeros((80, 80), dtype=np.uint8)), "stream")
    window._on_poll()

    assert window._hdr_enabled is False
    assert window._hdr_check.isChecked() is False
    assert window._last_hdr_error == "第 2 档取帧超时"
    assert window._preview_mode_label.text() == "预览模式: 单曝光取流"
    assert "第 2 档取帧超时" in window.statusBar().currentMessage()
    assert window._hdr_ladder_edit.isEnabled() is False
    assert source.hdr_calls == [(True, LADDER)]  # 对账是读状态, 不是又切一次
