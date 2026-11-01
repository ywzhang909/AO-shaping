"""桌面外壳 (多页导航) 的离屏交互测试。

外壳的价值全在「能切页」: 页 1 (迁移学习闭环整形) 是惰性创建的, 只有真的点到
导航项才会 import torch 并实例化页面。这里用离屏 ``QApplication`` 直接模拟点击,
把「导航项只是装饰」这类回归挡在 CI 里。
"""

from __future__ import annotations

import os

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.ccd.beam_quality.shell import DesktopShellApp


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def shell(app: QApplication):
    """新建外壳并显示; 关闭时走 closeEvent 收尾, 不留下在飞作业。"""
    window = DesktopShellApp()
    window.show()
    app.processEvents()
    yield window
    window.close()
    app.processEvents()


def _click_item(window: DesktopShellApp, index: int, app: QApplication) -> None:
    """在导航项中心模拟一次左键点击 (与真实鼠标事件同一条路径)。"""
    item = window._rail._items[index]
    assert item.isVisible(), f"导航项 {index} 应可见"
    QTest.mouseClick(item, Qt.MouseButton.LeftButton, pos=item.rect().center())
    app.processEvents()


def test_first_page_is_beam_quality_and_transfer_page_is_lazy(app, shell) -> None:
    assert shell._stack.currentIndex() == 0
    assert shell._transfer_page is None  # torch 不该在冷启动时被拉起


def test_clicking_rail_switches_to_transfer_page(app, shell) -> None:
    _click_item(shell, 1, app)

    assert shell._stack.currentIndex() == 1
    assert shell._transfer_page is not None
    assert shell._stack.currentWidget() is shell._transfer_page


def test_clicking_back_keeps_the_transfer_page_instance(app, shell) -> None:
    _click_item(shell, 1, app)
    created = shell._transfer_page
    _click_item(shell, 0, app)

    assert shell._stack.currentIndex() == 0
    # 切回去不能重建页面: 否则正在跑的会话与已填参数都会丢。
    assert shell._transfer_page is created

    _click_item(shell, 1, app)
    assert shell._stack.currentWidget() is created


def test_keyboard_activation_switches_page(app, shell) -> None:
    """没有鼠标时 (远程桌面 / 键盘操作) 也得能切页。"""
    item = shell._rail._items[1]
    item.setFocus()
    app.processEvents()
    QTest.keyClick(item, Qt.Key.Key_Return)
    app.processEvents()

    assert shell._stack.currentIndex() == 1
    assert shell._transfer_page is not None


def test_select_page_is_idempotent_and_bounds_checked(app, shell) -> None:
    shell.select_page(1)
    assert shell._stack.currentIndex() == 1
    shell.select_page(1)
    assert shell._stack.currentIndex() == 1
    # 越界索引被钳到合法范围, 不抛异常 (键盘/程序调用都可能传进来)。
    shell.select_page(9)
    assert shell._stack.currentIndex() == 1
