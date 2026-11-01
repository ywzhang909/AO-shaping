"""AO-Shaping 桌面外壳: 左侧导航 + 右侧页面栈。

为什么要有这一层
----------------
仓库里有两套互不相干的桌面界面: 光束质量分析 (CCD 侧) 与 SLM 迁移学习闭环整形
(SLM 侧)。它们各有各的设备生命周期 (CCD 相机 / SLM 句柄), 硬塞进同一个窗口会
互相抢设备; 但把它们做成两个独立进程, 操作员又要在两个窗口之间来回切, 看不到
「当前光路到底在跑什么」。这一层给两者一个共同的门面: 一条窄导航栏 + 一个页面栈。

设计取舍
--------
* **页 0 直接嵌入既有的 :class:`BeamQualityApp`** (整个窗口, 不只是它的内容区)。
  它自带菜单栏与状态栏, 而状态栏上有「已处理 N 帧 | UI x fps」——那是操作员判断
  画面流是否正常的唯一读数, 丢了就没法用。所以外壳**不设自己的状态栏**, 否则两条
  状态栏上下叠在一起, 像两个应用挤在一个框里。切页提示改放在导航栏底部。
* **页 1 惰性创建**。迁移学习闭环整形那一页在导入时会拉起 torch (物理预测模型的
  拟合与可微合成都在 torch 上), 冷启动要多花好几秒; 而看光束质量页的人不该为此
  付这个代价。所以只有当操作员真的切到第 1 页时, 才 import 并实例化。
* **窄导航栏**: 220 px 一列, 每项两行 —— 一行标题、一行副标题, 说明那一页管什么。
  副标题用小号灰字, 是这一页的「一句话说明书」。

配色
----
导航栏**不写死颜色**: 标题、副标题、状态点都从当前调色板派生, 这样应用在暗色与
亮色系统主题下都读得清 —— 写死浅色字会在另一种主题里变成「浅底浅字」。

线程约定
--------
外壳本身不做任何阻塞工作: 两个页面各自管理自己的后台线程 (光束质量页有单线程
``QThreadPool`` 跑相机的秒级操作, 迁移学习页有单线程池跑整场会话)。切页不打断
后台作业 —— 页 0 的相机流会继续跑, 页 1 的会话也会继续跑。
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp

__all__ = ["DesktopShellApp", "main"]

#: 导航项: (标题, 副标题)。副标题写「这一页管什么」, 不是「这一页是什么模块」。
_NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("光束质量", "实时画面 · 峰值趋势 · 剖面拟合"),
    ("迁移学习闭环整形", "物理预测模型 · 探针采集 · CCD 锚定验收"),
)

#: 切页时导航栏底部显示的一句话。
_NAV_HINTS: tuple[str, ...] = (
    "光束质量: 选信号源后点「打开」; 相机操作在后台线程, 不冻界面",
    "迁移学习闭环整形: 需要 torch; 切页不会打断正在跑的会话",
)

#: 导航栏宽度 (px)。窄到只放标题 + 副标题, 不抢页面。
_RAIL_WIDTH = 220

#: 状态点 / 标题高亮的强调色。蓝在暗色与亮色主题下都够醒目。
_ACCENT = QColor("#4a9eff")


def _dim(color: QColor, factor: float = 0.62) -> QColor:
    """把调色板颜色压暗一档 (亮色主题) —— 用来做次级文字。

    ``QColor.darker`` 在亮色主题下会把字压深, 在暗色主题下会把字压得更暗 (几乎
    看不见), 所以这里先按亮度判断方向: 底色暗就提亮, 底色亮就压暗。
    """
    if color.lightness() > 128:
        return color.darker(int(100 / factor))
    return color.lighter(int(100 / factor))


class _Dot(QWidget):
    """导航项左侧的状态点: 选中是实心亮蓝, 未选中是空心灰环。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active = False
        self.setFixedSize(8, 8)

    def set_active(self, active: bool) -> None:
        self._active = active
        self.update()

    def paintEvent(self, event) -> None:
        """Qt 覆写点 (驼峰命名由 Qt 决定)。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._active:
            painter.setBrush(_ACCENT)
            painter.setPen(QPen(_ACCENT, 1.0))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(_dim(self.palette().color(self.backgroundRole())), 1.0))
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))


class _RailItem(QWidget):
    """导航项: 状态点 + 标题 + 副标题。

    选中态靠状态点与标题提亮表达, 不靠整块底色 —— 底色在暗色主题下容易和页面
    自己的分组框撞色。

    导航项是**可激活控件**, 不是装饰: 鼠标点击与键盘 (Tab 聚焦 + Enter/空格) 都
    发出 :attr:`activated`, 由 :class:`_Rail` 汇总成带页索引的 ``pageRequested``。
    少了这层接线, 整个外壳只剩页 0 能看 —— 页 1 会被惰性创建逻辑永远藏着。
    """

    #: 用户要求切到本项 (鼠标或键盘)。
    activated = Signal()

    def __init__(
        self, title: str, subtitle: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 9, 10, 9)
        layout.setSpacing(1)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(7)

        self._dot = _Dot()
        head.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)

        self._title = QLabel(title)
        head.addWidget(self._title, 1)
        layout.addLayout(head)

        self._subtitle = QLabel(subtitle)
        self._subtitle.setWordWrap(True)
        layout.addWidget(self._subtitle)

        self.setFixedHeight(56)
        self._restyle()

    def _restyle(self) -> None:
        """按当前调色板刷一遍文字颜色 (换主题时也要重刷)。"""
        text = self.palette().color(self.foregroundRole())
        dim = _dim(text)
        self._title.setStyleSheet(
            f"color: {text.name()}; font-weight: {'600' if self._selected else '400'};"
        )
        self._subtitle.setStyleSheet(f"color: {dim.name()};")

    def set_selected(self, selected: bool) -> None:
        """切换选中态。"""
        if self._selected == selected:
            return
        self._selected = selected
        self._dot.set_active(selected)
        self._restyle()

    def mouseReleaseEvent(self, event) -> None:
        """在项内松开左键 = 激活 (按下、拖出再松开不算, 与按钮一致)。"""
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.activated.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        """键盘也能切页: Tab 聚焦后 Enter / 空格激活。"""
        if event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Space,
        ):
            self.activated.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _Rail(QWidget):
    """左侧导航栏: 品牌字样 + 一列导航项 + 底部提示。"""

    #: 用户点了第几页 (索引与 ``_NAV_ITEMS`` 一致)。
    pageRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(_RAIL_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._brand = QLabel("AO-Shaping")
        self._brand.setFont(QFont("", 12, QFont.Weight.DemiBold))
        layout.addWidget(self._brand)

        self._subtitle = QLabel("实验系统控制台")
        layout.addWidget(self._subtitle)

        self._rule = QFrame()
        self._rule.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(self._rule)

        self._items: list[_RailItem] = []
        for position, (title, note) in enumerate(_NAV_ITEMS):
            item = _RailItem(title, note)
            # 绑死当前索引: 直接引用循环变量会让所有项都指向最后一页。
            item.activated.connect(
                lambda position=position: self.pageRequested.emit(position)
            )
            self._items.append(item)
            layout.addWidget(item)

        layout.addStretch(1)

        self._hint = QLabel(_NAV_HINTS[0])
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self._restyle()

    def _restyle(self) -> None:
        """品牌字样与提示按调色板着色。"""
        text = self.palette().color(self.foregroundRole())
        dim = _dim(text)
        self._brand.setStyleSheet(f"color: {text.name()}; padding: 18px 14px 2px 14px;")
        self._subtitle.setStyleSheet(f"color: {dim.name()}; padding: 0 14px 14px 14px;")
        self._rule.setStyleSheet(f"color: {_dim(text, 0.5).name()};")
        self._hint.setStyleSheet(f"color: {dim.name()}; padding: 10px 14px 16px 14px;")

    def set_current(self, index: int) -> None:
        for position, item in enumerate(self._items):
            item.set_selected(position == index)
        if 0 <= index < len(_NAV_HINTS):
            self._hint.setText(_NAV_HINTS[index])


class DesktopShellApp(QMainWindow):
    """AO-Shaping 桌面外壳: 窄导航 + 页面栈。

    页 0 是既有的光束质量分析 (整个窗口嵌进来), 页 1 是 SLM 迁移学习闭环整形
    (惰性创建)。窗口标题由外壳统一持有, 切页不会换标题 —— 操作员在两页之间移动
    时, 标题栏始终说明「这是哪个应用」。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AO-Shaping 实验系统控制台")
        self.resize(1560, 940)

        self._rail = _Rail(self)
        self._rail.pageRequested.connect(self.select_page)
        self._stack = QStackedWidget(self)
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # 页 0: 光束质量 —— 立刻建好 (它是默认页, 而且不需要 torch)。
        self._beam_app = BeamQualityApp()
        self._stack.addWidget(self._beam_app)

        # 页 1: 迁移学习闭环整形 —— 惰性, 切过去才建。
        self._transfer_page: QWidget | None = None

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._rail)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._rail.set_current(0)

    # ------------------------------------------------------------------ 页面
    def _transfer(self) -> QWidget:
        """惰性创建页 1; 建好之后缓存, 切回来不重建 (重建会丢掉运行进度)。"""
        if self._transfer_page is None:
            from ao_shaping.gui.slm.transfer_desktop import TransferPage

            self._transfer_page = TransferPage()
        return self._transfer_page

    def _sync_stack(self, index: int) -> None:
        """把导航栏的选中态与页面栈对齐。"""
        self._rail.set_current(index)
        if index == 1:
            widget = self._transfer()
            if self._stack.indexOf(widget) < 0:
                self._stack.addWidget(widget)
            self._stack.setCurrentWidget(widget)
        else:
            self._stack.setCurrentIndex(0)

    def select_page(self, index: int) -> None:
        """切页 (0 = 光束质量, 1 = 迁移学习闭环整形)。"""
        self._sync_stack(max(0, min(index, len(_NAV_ITEMS) - 1)))

    # ------------------------------------------------------------------ 事件
    def closeEvent(self, event) -> None:
        """关窗: 先让页面自己收尾 (停相机 / 停后台作业), 再关。

        子页面的 ``closeEvent`` 不会被 Qt 自动转发, 所以这里显式转发 ——
        光束质量页的相机流不关就会占着设备。
        """
        for widget in (self._beam_app, self._transfer_page):
            if widget is None:
                continue
            widget.close()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    """启动桌面外壳 (``python -m ao_shaping.gui.ccd.beam_quality.shell``)。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(sys.argv) if argv is None else list(argv))
    # Fusion: 跨平台的统一样式, 避免同一套界面在两台机器上长得不一样。
    app.setStyle("Fusion")
    app.setApplicationName("ao-shaping-console")
    window = DesktopShellApp()
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # 也允许 python -m ao_shaping.gui.ccd.beam_quality.shell
    raise SystemExit(main())
