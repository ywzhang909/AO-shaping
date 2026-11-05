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


def _wait_for_transfer_page(app: QApplication, window: DesktopShellApp) -> None:
    """等后台预热完成并在主线程替换占位页 (超时则失败, 不挂住测试)。"""
    import time

    deadline = time.monotonic() + 120
    while window._transfer_page is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert window._transfer_page is not None, "后台预热未在限时内完成"


def test_clicking_rail_shows_placeholder_then_real_transfer_page(
    app, shell, monkeypatch
) -> None:
    """点到页 1 时先放占位 (不同步 import torch), 预热完成后换成真页面。

    预热被换成一个可控的空任务, 占位状态因此确定可见 (不依赖 torch 是否已在进程里缓存)。
    """
    from ao_shaping.gui.ccd.beam_quality import shell as shell_module

    class _HeldWarm(shell_module._WarmTask):
        def run(self) -> None:  # 不做任何导入: 占位保持到测试手动放行
            return

    monkeypatch.setattr(shell_module, "_WarmTask", _HeldWarm)
    _click_item(shell, 1, app)

    assert shell._stack.currentIndex() == 1
    assert shell._stack.currentWidget() is shell._transfer_placeholder
    assert shell._transfer_page is None

    # 放行: 真预热完成后, 占位被真页面替换。
    shell._on_transfer_warm_ready()
    app.processEvents()
    assert shell._transfer_placeholder is None
    assert shell._stack.currentWidget() is shell._transfer_page


def test_clicking_back_keeps_the_transfer_page_instance(app, shell) -> None:
    _click_item(shell, 1, app)
    _wait_for_transfer_page(app, shell)
    created = shell._transfer_page
    _click_item(shell, 0, app)

    assert shell._stack.currentIndex() == 0
    # 切回去不能重建页面: 否则正在跑的会话与已填参数都会丢。
    assert shell._transfer_page is created

    _click_item(shell, 1, app)
    _wait_for_transfer_page(app, shell)
    assert shell._stack.currentWidget() is created


def test_keyboard_activation_switches_page(app, shell) -> None:
    """没有鼠标时 (远程桌面 / 键盘操作) 也得能切页。"""
    item = shell._rail._items[1]
    item.setFocus()
    app.processEvents()
    QTest.keyClick(item, Qt.Key.Key_Return)
    app.processEvents()

    assert shell._stack.currentIndex() == 1
    _wait_for_transfer_page(app, shell)
    assert shell._transfer_page is not None


def test_warmup_runs_once_even_when_clicked_repeatedly(app, shell, monkeypatch) -> None:
    """重复点页 1 只启动一次后台预热, 不会排队多份重依赖导入。"""
    starts: list[object] = []
    original = shell._warm_pool.start

    def _record(task, *a, **k):
        starts.append(task)
        return original(task, *a, **k)

    monkeypatch.setattr(shell._warm_pool, "start", _record)
    _click_item(shell, 1, app)
    _click_item(shell, 0, app)
    _click_item(shell, 1, app)
    _wait_for_transfer_page(app, shell)

    assert len(starts) == 1


def test_select_page_is_idempotent_and_bounds_checked(app, shell) -> None:
    shell.select_page(1)
    assert shell._stack.currentIndex() == 1
    shell.select_page(1)
    assert shell._stack.currentIndex() == 1
    # 越界索引被钳到最后一页, 不抛异常 (键盘/程序调用都可能传进来)。
    # 页面栈只保存已创建的惰性页, 所以用当前控件而非物理栈索引判断逻辑页。
    shell.select_page(9)
    assert shell._static_page is not None
    assert shell._stack.currentWidget() is shell._static_page


# ---------------------------------------------------------------------------
# Lane C: 压缩导航栏 (用户要求「压缩宽度、用竖排」, 去掉副标题与底部逐页说明)。
# ---------------------------------------------------------------------------


def test_rail_is_compressed_and_single_line(app, shell) -> None:
    """导航栏必须显著窄于旧的 220 px, 且每项只剩单行标题 (无副标题)。

    压缩是这次迭代的核心诉求, 钉死数值: 一旦有人把它改宽回去, 这条测试会拦住。
    """
    from ao_shaping.gui.ccd.beam_quality.shell import _RAIL_WIDTH, _RailItem

    assert _RAIL_WIDTH < 220
    item = shell._rail._items[0]
    assert isinstance(item, _RailItem)
    # 无副标题控件: 单行即可容纳。
    assert not hasattr(item, "_subtitle")
    # 单行高度显著小于原来两行的 56 px。
    assert item.height() < 50


def test_rail_has_three_nav_items(app, shell) -> None:
    """导航栏三页齐在: 光束质量 / 迁移学习闭环整形 / SLM 静态整形。"""
    from ao_shaping.gui.ccd.beam_quality.shell import _NAV_ITEMS

    assert len(_NAV_ITEMS) == 3
    assert len(shell._rail._items) == 3
    assert shell._rail._items[2].accessibleName() == "SLM静态整形"


def test_rail_labels_fit_without_clipping(app, shell) -> None:
    """导航短标签必须完整落在分配宽度内, 不裁切。

    120 px 窄栏下, 标题可用宽度只有 ~86 px; 「迁移学习闭环整形」是 8 个汉字
    (≈96 px), 单行必然被 QLabel 裁掉。所以导航栏只能放短标签, 全名由页面自身
    标题承担 (transfer_desktop 页头仍是全称)。这条测试直接量每个导航项的实际
    文本宽度 vs 分配宽度, 谁再塞长标签或把栏改窄, 会被当场拦下。
    """
    for item in shell._rail._items:
        title = item._title
        advance = title.fontMetrics().horizontalAdvance(title.text())
        assert advance <= title.width(), (
            f"导航标签 {title.text()!r} 文本宽 {advance}px 超出其分配宽度 "
            f"{title.width()}px —— 会被裁切, 请改用更短的标签或加宽可用区"
        )


def test_no_bottom_hint_label_in_rail(app, shell) -> None:
    """底部逐页提示已移出导航栏 (它重复了页面内的空状态文案)。"""
    assert not hasattr(shell._rail, "_hint")


def test_static_shaping_page_is_lazy_and_switches(app, shell) -> None:
    """页 2 (SLM 静态整形) 惰性创建: 冷启动不建, 点到才实例化并成为当前页。"""
    assert shell._static_page is None  # 冷启动不建静态整形页
    _click_item(shell, 2, app)

    assert shell._static_page is not None
    assert shell._stack.currentWidget() is shell._static_page


def test_static_shaping_page_stays_cached(app, shell) -> None:
    """切走再切回, 静态整形页实例不重建 (后台任务与已填参数都不丢)。"""
    _click_item(shell, 2, app)
    created = shell._static_page
    _click_item(shell, 0, app)
    _click_item(shell, 2, app)
    assert shell._stack.currentWidget() is created
