"""⑥ 实时画面的离屏契约: 后台帧信号 → 主线程上屏。

钉住线程边界与刷新语义:

* ``_SessionTask._on_frame`` 只 ``emit``, 不碰控件 (QRunnable 不是 QObject);
* ``frame`` 信号经事件队列排队到 ``LiveView.apply_payload``, 跨线程不直接改 UI;
* 一帧上屏会同时更新 CCD 图、块进度条与帧计数条;
* 大靶面先降采样再归一化 (``display_preview``), 小图原样分析;
* ⑤ 阶段进度面板仍然存在 (实时画面是新增列, 不是替换);
* 开跑前清空实时画面, 避免上一场会话的残帧。
"""

from __future__ import annotations

import os
import threading
import types

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop.live_view import (
    PREVIEW_STEP,
    LiveView,
    display_frame,
    display_phase,
    display_preview,
    preview_step,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    widget = page_mod.TransferPage()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def _payload(**overrides) -> dict:
    yy, xx = np.mgrid[0:40, 0:48]
    frame = (
        2000 * np.exp(-(((xx - 24) ** 2) / 60.0 + ((yy - 20) ** 2) / 40.0))
    ).astype(np.float64)
    payload = dict(
        frame=frame,
        xy=None,
        shape="triangle",
        block=1,
        block_index=2,
        blocks_total=4,
        iteration=12,
        iterations_total=12,
        frame_index=2,
        frames_total=3,
        phase="validate",
        label="验收采集",
    )
    payload.update(overrides)
    return payload


def test_display_frame_normalises_arbitrary_raw_frames():
    """float 计数帧不能 clip 成全黑; uint16 / 3 通道也都要能显示。"""
    float_frame = np.linspace(0, 4321.0, 64, dtype=np.float64).reshape(8, 8)
    shown = display_frame(float_frame)
    assert shown.dtype == np.uint8
    assert shown.max() == 255
    assert display_frame(np.zeros((0, 0))).shape == (1, 1)
    assert display_frame(np.full((4, 4), 7.0)).max() == 0
    rgb = np.zeros((4, 4, 3), dtype=np.uint16)
    rgb[..., 0] = 500
    assert display_frame(rgb).ndim == 2
    nan_frame = np.array([[np.nan, np.inf], [-np.inf, 10.0]])
    assert display_frame(nan_frame).max() == 255


def test_apply_payload_updates_frame_and_both_bars(app: QApplication):
    view = LiveView()
    view.show()
    app.processEvents()
    view.apply_payload(_payload())
    app.processEvents()

    assert view._video._array is not None
    assert (view._block_bar.value(), view._block_bar.maximum()) == (2, 4)
    assert "2 / 4" in view._block_label.text()
    assert "triangle" in view._block_label.text()
    assert (view._remaining_bar.value(), view._remaining_bar.maximum()) == (2, 3)
    # 第二根条不再是「距下次刷新还剩 N iter」: 刷新本来就是逐帧的。
    assert "帧 2/3" in view._info.text()
    assert "块内 iter 12/12" in view._info.text()
    view.close()


def test_bad_or_empty_payloads_do_not_touch_the_view(app: QApplication):
    view = LiveView()
    for bad in (None, "not-a-dict", {}, {"frame": None}):
        view.apply_payload(bad)  # type: ignore[arg-type]
    assert view._last_payload is None
    assert view._video._array is None
    view.close()


def test_capture_phase_has_no_block_progress_but_counts_frames(app: QApplication):
    """采集段没有块: 第一根条标阶段名, 第二根条数这一次采集的第几帧。

    ``frame_index`` 是整段单调的帧号, 所以 ``frames_total`` 未知时 (曝光阶梯的档数
    到读完才知道) 只显示「第 N 帧」而不是一个假的比例。
    """
    view = LiveView()
    view.apply_payload(
        _payload(
            phase="capture",
            blocks_total=0,
            block_index=0,
            shape="",
            frames_total=0,
            frame_index=2,
            label="探针 12/34",
        )
    )
    assert view._block_bar.maximum() == 1
    assert "采集段" in view._block_bar.format()
    assert "采集" in view._block_label.text()
    assert view._remaining_bar.format() == "第 2 帧"
    assert "第 2 帧" in view._info.text()
    assert "探针 12/34" in view._info.text()
    view.close()


def test_display_preview_decimates_only_large_sensor_frames():
    """大靶面按 ``PREVIEW_STEP`` 抽稀; 仿真/测试的小图必须原样分析。

    降采样是显示端的成本取舍: 2592x1944 一帧的归一化 + 分析是 ~220-337 ms,
    抽稀 4 倍后只要 ~16 ms。小图不降采样, 否则 8x10 会变成 2x3, 连剖面都画不出来。
    """
    big = np.zeros((1944, 2592), dtype=np.uint8)
    big[900:1000, 1200:1300] = 200
    shown, step = display_preview(big)
    assert step == PREVIEW_STEP
    assert shown.shape == (1944 // PREVIEW_STEP, 2592 // PREVIEW_STEP)
    assert shown.dtype == np.uint8
    # 归一化按峰值缩放到 255.0, ``astype(uint8)`` 截断会掉到 254 (浮点 200*1.275).
    assert shown.max() >= 254

    shown, step = display_preview(np.zeros((40, 48), dtype=np.uint8))
    assert step == 1
    assert shown.shape == (40, 48)

    # 3 通道与小图一样: 先降到 2-D 再判定。
    assert preview_step((1944, 2592, 3)) == PREVIEW_STEP
    assert preview_step((10,)) == 1


def test_page_exposes_live_view_and_keeps_the_progress_panel(page):
    assert isinstance(page._live_view, LiveView)
    # ⑤ 进度面板与它的阶段表都还在, 实时画面是右栏新增的一块。
    assert page._table.columnCount() == 5
    assert page._bar.format().startswith("已结束")


def test_session_task_on_frame_only_emits(page):
    """后台回调经信号入队, 主线程泵完事件才上屏 —— 不是直接改控件。

    必须从**真的后台线程** emit: 同线程 emit 会被 Qt 优化成 DirectConnection,
    那样测到的「没上屏」只是错觉, 而不是队列在起作用。
    """
    task = page_mod._SessionTask(page.form_values(), page_mod.params.plan_session_dir())
    task.signals.frame.connect(page._live_view.apply_payload)
    assert page._live_view._last_payload is None

    worker = threading.Thread(target=task._on_frame, args=(_payload(),))
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    # 后台线程已经 emit 完毕, 但主线程还没泵事件 → 控件必须仍然是空态。
    assert page._live_view._last_payload is None
    for _ in range(10):
        QCoreApplication.processEvents()
    assert page._live_view._last_payload is not None
    assert page._live_view._block_bar.value() == 2


def test_start_run_clears_the_previous_frame(page, monkeypatch: pytest.MonkeyPatch):
    page._live_view.apply_payload(_payload())
    assert page._live_view._last_payload is not None
    # 不真正开会话: 钉住的只是「开跑前清屏」这一步。
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._start_run()
    assert page._live_view._last_payload is None
    assert page._live_view._video._array is None
    assert page._live_view._block_bar.value() == 0
    page._poll.stop()


def test_display_phase_wraps_2pi_into_uint8_greyscale():
    """相位转显示: mod(2π) → 线性映射, wrap 线性 (2π≈0) 必须叠在一起。"""
    phase = np.array([-np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi])
    shown = display_phase(phase)
    assert shown.dtype == np.uint8
    assert shown.shape == (5,)
    # 0 与 2π 同值 (mod 2π); −π 与 π、3π 同值。
    assert shown[1] == shown[3]
    assert shown[0] == shown[2] == shown[4]
    # π(wrap 到 127) 高于 0; 最大值恰好是半满 (π/2π · 255 ≈ 127)。
    assert shown[0] > shown[1]
    assert shown.max() == 127
    # 二维网格也工作, 中分点 (π/2) 落在 ~64 (1/4 圈) 处。
    grid = np.linspace(0.0, np.pi, 3).reshape(3, 1)
    wrapped = display_phase(grid)
    assert wrapped.shape == (3, 1)
    assert wrapped[1, 0] == pytest.approx(64, abs=2)
    # nan/inf 平 0, 空数组给 1x1 占位; 1.0 rad → 1/(2π)·255 ≈ 40。
    sanitised = display_phase(np.array([[np.nan, np.inf], [-np.inf, 1.0]]))
    assert sanitised.shape == (2, 2)
    assert sanitised[0, 0] == 0
    assert sanitised[1, 1] == pytest.approx(40, abs=2)
    assert display_phase(np.zeros((0, 0))).shape == (1, 1)


def test_apply_payload_renders_phase_and_predicted_panels(app: QApplication):
    """新增的 phase_grid / predicted 键被刷进两张灰度面板, 旧键路径不受影响。"""
    view = LiveView()
    view.show()
    app.processEvents()
    phase = np.mod(np.linspace(0.0, 3.0 * np.pi, 128 * 128), 2 * np.pi).reshape(
        128, 128
    )
    predicted = np.zeros((48, 64), dtype=np.float64)
    predicted[20:30, 26:38] = 500.0
    view.apply_payload(_payload(phase_grid=phase, predicted=predicted))
    app.processEvents()

    assert view._video._array is not None  # CCD 栏照旧
    # 相位面板: 数组已上屏, 尺寸与 phase 一致。
    assert view._phase.image._array is not None
    assert view._phase.image._array.shape == (128, 128)
    # 预测面板: 灰度值已落满 (非零处有亮度)。
    assert view._predicted.image._array is not None
    assert view._predicted.image._array.shape == (48, 64)
    assert view._predicted.image._array.max() > 0
    # 标题标签存在。
    assert "SLM 相位" in view._phase._caption.text()
    assert "预测远场" in view._predicted._caption.text()
    view.close()


def test_phase_or_predicted_absent_or_none_clears_but_does_not_crash(
    app: QApplication,
):
    """采集段没有相位/预测时 (缺失或 None), 面板回空态, 其他键照常刷新。"""
    view = LiveView()
    view.show()
    app.processEvents()
    view.apply_payload(_payload())
    assert view._video._array is not None
    assert view._phase.image._array is None
    assert view._predicted.image._array is None

    # 上一帧有数据后, 新帧两个键为 None → 面板清空而不是残留旧帧。
    phase = np.zeros((16, 16))
    view.apply_payload(_payload(phase_grid=phase, predicted=np.ones((8, 8))))
    assert view._phase.image._array is not None
    # 显式 None 复原。
    view.apply_payload(_payload(phase_grid=None, predicted=None))
    assert view._phase.image._array is None
    assert view._predicted.image._array is None
    assert view._last_payload is not None
    view.close()


def test_clear_resets_phase_and_predicted_panels(app: QApplication):
    view = LiveView()
    view.apply_payload(_payload(phase_grid=np.zeros((8, 8)), predicted=np.ones((4, 4))))
    assert view._phase.image._array is not None
    assert view._predicted.image._array is not None
    view.clear()
    assert view._phase.image._array is None
    assert view._predicted.image._array is None
    view.close()


def test_predicted_absent_with_live_phase_shows_initialization(app: QApplication):
    """相位已驱动而预测尚为 None → 显示短暂的平场/几何初始化提示。

    无 checkpoint 时也会用本次平场与 signed-ramp 构造零校正理想模型，因此
    这里不再表示「模型未就绪」或「冷启动永远没有预测」。提示必须说明当前
    只是在初始化，而不是丢一个像坏了/没数据的黑框。
    """
    view = LiveView()
    view.show()
    app.processEvents()
    view.apply_payload(_payload(phase_grid=np.zeros((16, 16)), predicted=None))
    app.processEvents()
    # SLM 相位已上屏 (采集段确实在驱动相位)。
    assert view._phase.image._array is not None
    # 预测面板没有可显示的远场, 但状态行说清了缘由而不是空黑框。
    assert view._predicted.image._array is None
    assert view._predicted._state.isVisible()
    assert "初始化中" in view._predicted._state.text()
    assert "平场" in view._predicted._state.text()
    view.close()


def test_predicted_present_clears_the_initialization_state(app: QApplication):
    """模型可用后: predicted 非 None → 画远场，初始化状态行立即收起。"""
    view = LiveView()
    view.show()
    app.processEvents()
    view.apply_payload(_payload(phase_grid=np.zeros((16, 16)), predicted=None))
    app.processEvents()
    assert view._predicted._state.isVisible()

    view.apply_payload(
        _payload(phase_grid=np.zeros((16, 16)), predicted=np.ones((8, 8)))
    )
    app.processEvents()
    assert view._predicted.image._array is not None
    assert not view._predicted._state.isVisible()
    view.close()


def test_predicted_absent_without_phase_stays_plain_empty(app: QApplication):
    """会话还没驱动任何相位 (phase_grid 也 None): 面板回"无画面", 不提"模型"。

    双 None 是干净的复位语义, 搬出"初始化中"会误导 —— 此时连相位都还没有。
    """
    view = LiveView()
    view.show()
    app.processEvents()
    view.apply_payload(_payload(phase_grid=None, predicted=None))
    app.processEvents()
    assert view._phase.image._array is None
    assert view._predicted.image._array is None
    assert not view._predicted._state.isVisible()
    view.close()


def test_completed_run_shows_export_buttons_and_next_run_hides_them(
    app: QApplication,
    page,
    monkeypatch,
    tmp_path,
):
    """⑤ 进度: 完成后导出按钮行替换进度条; 失败 / 下次运行 → 还原进度条。"""
    # 完成 → 导出按钮行可见, 进度条与当前阶段标签隐藏。
    page._on_finished("out")
    assert page._export_row.isVisible()
    assert not page._bar.isVisible()
    assert not page._current.isVisible()

    # 失败 → 不显示导出, 还原进度条 / 当前阶段标签。
    page._on_failed("boom")
    assert not page._export_row.isVisible()
    assert page._bar.isVisible()
    assert page._current.isVisible()

    # 再完成 → 导出; 下一次启动 (reset 路径) → 还原进度条。
    page._on_finished("out")
    assert page._export_row.isVisible()

    class _StubSignal:
        def connect(self, *args, **kwargs):
            pass

    class _StubTask(page_mod.QRunnable):
        def __init__(self, values, out):
            super().__init__()
            self.signals = types.SimpleNamespace(
                stage=_StubSignal(),
                frame=_StubSignal(),
                finished=_StubSignal(),
                failed=_StubSignal(),
            )

        def run(self):
            pass

    monkeypatch.setattr(page_mod, "_SessionTask", _StubTask)
    monkeypatch.setattr(page_mod.params, "plan_session_dir", lambda: tmp_path)
    page._start_run()
    assert not page._export_row.isVisible()
    assert page._bar.isVisible()
    assert page._current.isVisible()

    # 收尾: 让 ``_settle`` 跑一遍, 免得 fixture 关窗时等 5 s。
    page._on_failed("cleanup")
