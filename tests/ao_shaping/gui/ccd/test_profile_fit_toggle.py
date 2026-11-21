"""剖面图"显示高斯拟合曲线"开关的离屏测试。

开关的语义很窄: 只切两条高斯拟合虚线, 散点、坐标轴、四格读数 (σ / FWHM /
中心 / R²) 与拟合失败提示都必须照旧。这里用离屏 ``QApplication`` 直接点控件,
覆盖默认态、关掉、再打开、以及 ``clear()`` 之后的语义。
"""

from __future__ import annotations

import os

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.ccd.beam_quality.analyzer import (
    FrameMetrics,
    GaussianFit,
    gaussian_1d,
)
from ao_shaping.gui.ccd.beam_quality.profile_widget import ProfileWidget

PROFILE_LEN = 120


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _fit(center: float = 60.0, sigma: float = 12.0) -> GaussianFit:
    """一条能画出曲线的合成拟合结果。"""
    return GaussianFit(
        amplitude=180.0,
        center=center,
        sigma=sigma,
        offset=4.0,
        r_squared=0.9987,
    )


def _metrics(
    x_fit: GaussianFit | None,
    y_fit: GaussianFit | None,
) -> FrameMetrics:
    """剖面数据固定, 拟合结果可按用例置 None。"""
    grid = np.arange(PROFILE_LEN, dtype=np.float64)
    profile = gaussian_1d(grid, 180.0, 60.0, 12.0, 4.0)
    return FrameMetrics(
        max_value=float(profile.max()),
        mean_value=float(profile.mean()),
        std_value=float(profile.std()),
        sat_ratio=0.0,
        cx=60.0,
        cy=60.0,
        ellipse=None,
        x_profile=profile,
        y_profile=profile,
        x_fit=x_fit,
        y_fit=y_fit,
    )


def _shown(widget: ProfileWidget, app: QApplication) -> ProfileWidget:
    """显示并处理一次事件, 让 isVisible() 与布局尺寸生效。"""
    widget.resize(720, 320)
    widget.show()
    app.processEvents()
    return widget


def test_fit_curve_is_shown_by_default(app: QApplication) -> None:
    widget = _shown(ProfileWidget(), app)
    try:
        assert widget.fit_toggle.isChecked()
        assert widget.fit_toggle.text() == "显示高斯拟合曲线"
        assert widget.fit_toggle.toolTip()
        assert widget.fit_curve_visible()
        for panel in (widget._x_panel, widget._y_panel):
            assert panel.fit_visible()

        widget.update_metrics(_metrics(_fit(), _fit()))
        app.processEvents()
        # 开关只占顶部一行, 两条剖面依旧平分宽度
        widths = (widget._x_panel.width(), widget._y_panel.width())
        assert abs(widths[0] - widths[1]) <= 4
        assert min(widths) > 0
    finally:
        widget.close()


def test_unchecking_hides_only_fit_curves(app: QApplication) -> None:
    widget = _shown(ProfileWidget(), app)
    try:
        widget.update_metrics(_metrics(_fit(), _fit()))
        app.processEvents()
        fit_counts = [p._fit.count() for p in (widget._x_panel, widget._y_panel)]
        assert min(fit_counts) > 0

        QTest.mouseClick(
            widget.fit_toggle,
            Qt.MouseButton.LeftButton,
            pos=widget.fit_toggle.rect().center(),
        )
        app.processEvents()

        assert not widget.fit_toggle.isChecked()
        assert not widget.fit_curve_visible()
        for panel in (widget._x_panel, widget._y_panel):
            assert not panel.fit_visible()
            # 数据散点仍在, 曲线数据也没丢: 只是不显示
            assert panel._data.count() > 0
            assert panel._data.isVisible()
            assert panel._fit.count() > 0
            # 读数不随开关变化
            assert panel._fit_values["sigma"].text().endswith("px")
            assert panel._fit_values["r2"].text() != "—"
    finally:
        widget.close()


def test_rechecking_restores_latest_curve_without_new_frame(app: QApplication) -> None:
    widget = _shown(ProfileWidget(), app)
    try:
        widget.update_metrics(_metrics(_fit(), _fit()))
        app.processEvents()
        counts_before = [p._fit.count() for p in (widget._x_panel, widget._y_panel)]

        widget.set_fit_curve_visible(False)
        app.processEvents()
        assert not widget.fit_curve_visible()

        widget.set_fit_curve_visible(True)
        app.processEvents()
        assert widget.fit_toggle.isChecked()
        assert widget.fit_curve_visible()
        for panel, expected in zip((widget._x_panel, widget._y_panel), counts_before):
            assert panel.fit_visible()
            assert panel._fit.count() == expected
    finally:
        widget.close()


def test_fit_failure_note_ignores_toggle(app: QApplication) -> None:
    widget = _shown(ProfileWidget(), app)
    try:
        widget.update_metrics(_metrics(None, _fit()))
        app.processEvents()
        x_panel = widget._x_panel
        note = x_panel._note.text()
        assert note
        assert x_panel._note.isVisible()
        assert x_panel._fit.count() == 0

        widget.set_fit_curve_visible(False)
        app.processEvents()
        assert x_panel._note.text() == note
        assert x_panel._note.isVisible()
        assert x_panel._fit_values["sigma"].text() == "—"
    finally:
        widget.close()


def test_clear_keeps_toggle_state_and_resets_readouts(app: QApplication) -> None:
    widget = _shown(ProfileWidget(), app)
    try:
        widget.update_metrics(_metrics(_fit(), _fit()))
        widget.set_fit_curve_visible(False)
        app.processEvents()

        widget.clear()
        app.processEvents()

        assert not widget.fit_toggle.isChecked()
        assert not widget.fit_curve_visible()
        for panel in (widget._x_panel, widget._y_panel):
            assert panel._data.count() == 0
            assert panel._fit.count() == 0
            assert panel._fit_values["sigma"].text() == "—"
            assert panel._note.text() == ""

        # 重新打开后照常显示, 下一次 update_metrics 即恢复曲线
        widget.set_fit_curve_visible(True)
        widget.update_metrics(_metrics(_fit(), _fit()))
        app.processEvents()
        assert widget.fit_curve_visible()
        assert widget._x_panel._fit.count() > 0
    finally:
        widget.close()
