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
  状态栏上下叠在一起, 像两个应用挤在一个框里。切页只更新导航项的选中态,
  不再显示额外提示。
* **页 1 / 页 2 惰性创建**。迁移学习闭环整形页在导入时会拉起 torch (物理预测模型
  的拟合与可微合成都在 torch 上), 冷启动要多花好几秒; SLM 静态整形页也只在实际
  切到时创建。看光束质量页的人不该为这些页面的初始化代价买单。
* **窄导航栏**: 120 px 一列, 每项只有一行短标题与状态点, 不放副标题或底部提示,
  把尽可能多的横向空间留给页面内容。超过可用宽度的长名 (如「迁移学习闭环整形」)
  在导航栏缩成短标签, 全名由页面自身标题承担, 悬停可看全名提示。

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

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
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

#: 导航项: (导航短标签, 页面全名)。副标题 (页面一句话说明书) 已按用户要求移除 ——
#: 导航栏只做页面切换, 说明交给页面内部的空状态文案。
#: 导航栏只放**短标签** (120 px 单行放得下, 不裁切): 「迁移学习闭环整形」是 8 个
#: 汉字 (≈96 px), 在 86 px 的可用标题宽度里必然被切; 所以第 2 页的导航标签缩成
#: 「闭环整形」, 全名由页面自身的标题承担 (transfer_desktop 页头仍是全称), 悬停
#: 时用全名做提示。
_NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("光束质量", "光束质量"),
    ("闭环整形", "迁移学习闭环整形"),
    ("SLM静态整形", "SLM静态整形"),
)

#: 切页说明（保留给程序性索引访问，导航栏本身不再显示 —— 见模块 docstring）。
_NAV_HINTS: tuple[str, ...] = (
    "光束质量: 选信号源后点「打开」; 相机操作在后台线程, 不冻界面",
    "迁移学习闭环整形: 需要 torch; 切页不会打断正在跑的会话",
    "SLM静态整形: 选图案类型、调参数、生成预览, 再下发到所选 SLM",
)

#: 导航栏宽度 (px)。压缩到只放单行标题 + 状态点, 不抢页面。
_RAIL_WIDTH = 120

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
    """导航项: 状态点 + 单行标题。

    用户要求「压缩宽度、不显示说明」: 副标题已移除, 每项只占一行 (状态点 + 标题)。
    选中态靠状态点与标题提亮表达, 不靠整块底色 —— 底色在暗色主题下容易和页面
    自己的分组框撞色。

    导航项是**可激活控件**, 不是装饰: 鼠标点击与键盘 (Tab 聚焦 + Enter/空格) 都
    发出 :attr:`activated`, 由 :class:`_Rail` 汇总成带页索引的 ``pageRequested``。
    少了这层接线, 整个外壳只剩页 0 能看 —— 页 1 / 页 2 会被惰性创建逻辑永远藏着。
    """

    #: 用户要求切到本项 (鼠标或键盘)。
    activated = Signal()

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        tooltip: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(title)
        if tooltip:
            self.setToolTip(tooltip)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 8, 7)
        layout.setSpacing(6)

        self._dot = _Dot()
        layout.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)

        self._title = QLabel(title)
        layout.addWidget(self._title, 1)

        # 单行: 高度只容纳标题, 压缩纵向占用。
        self.setFixedHeight(34)
        self._restyle()

    def _restyle(self) -> None:
        """按当前调色板刷一遍文字颜色 (换主题时也要重刷)。"""
        text = self.palette().color(self.foregroundRole())
        self._title.setStyleSheet(
            f"color: {text.name()}; font-weight: {'600' if self._selected else '400'};"
        )

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
        for position, (title, full_name) in enumerate(_NAV_ITEMS):
            item = _RailItem(title, tooltip=full_name)
            # 绑死当前索引: 直接引用循环变量会让所有项都指向最后一页。
            item.activated.connect(
                lambda position=position: self.pageRequested.emit(position)
            )
            self._items.append(item)
            layout.addWidget(item)

        layout.addStretch(1)
        self._restyle()

    def _restyle(self) -> None:
        """品牌字样按调色板着色 (导航栏不再有逐页提示文本)。"""
        text = self.palette().color(self.foregroundRole())
        dim = _dim(text)
        self._brand.setStyleSheet(f"color: {text.name()}; padding: 18px 12px 2px 12px;")
        self._subtitle.setStyleSheet(f"color: {dim.name()}; padding: 0 12px 10px 12px;")
        self._rule.setStyleSheet(f"color: {_dim(text, 0.5).name()};")

    def set_current(self, index: int) -> None:
        for position, item in enumerate(self._items):
            item.set_selected(position == index)


class _WarmSignals(QObject):
    """后台预热的结果信号 (只经信号回主线程, 不碰控件)。"""

    ready = Signal()
    failed = Signal(str)


class _WarmTask(QRunnable):
    """后台线程预热迁移学习页的重依赖 (torch 与页面模块)。

    只做 import: 实测 ``import torch`` 约 3.5 s、页面模块导入约 4.5 s, 都是 GUI
    线程上最大的卡顿来源。控件构造留在主线程 (Qt 控件不能跨线程创建)。
    """

    def __init__(self) -> None:
        super().__init__()
        self.signals = _WarmSignals()

    def run(self) -> None:
        try:
            import torch  # noqa: F401
            import ao_shaping.gui.slm.transfer_desktop.page  # noqa: F401
        except Exception as exc:  # pragma: no cover - 环境相关
            self.signals.failed.emit(str(exc))
            return
        self.signals.ready.emit()


class DesktopShellApp(QMainWindow):
    """AO-Shaping 桌面外壳: 窄导航 + 页面栈。

    页 0 是既有的光束质量分析 (整个窗口嵌进来), 页 1 是 SLM 迁移学习闭环整形
    (惰性创建), 页 2 是 SLM 静态整形 (惰性创建, 会拉 SLM 驱动只在下发时)。
    窗口标题由外壳统一持有, 切页不会换标题 —— 操作员在各页之间移动时, 标题栏始终
    说明「这是哪个应用」。
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

        # 页 1: 迁移学习闭环整形 —— 惰性, 切过去才建。切过去时先放占位页,
        # 后台预热依赖, 预热完成后在主线程构造真页面并替换占位 (见 _start_transfer_warmup)。
        self._transfer_page: QWidget | None = None
        self._transfer_placeholder: QLabel | None = None
        self._warm_pool = QThreadPool(self)
        self._warm_pool.setMaxThreadCount(1)
        self._warm_started = False
        self._closing = False

        # 页 2: SLM 静态整形 —— 惰性, 切过去才建 (不会拖慢冷启动)。
        self._static_page: QWidget | None = None

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
        """页 1 的真页面; 同步建好并缓存 (只在预热完成后调用, 切回来不重建)。"""
        if self._transfer_page is None:
            from ao_shaping.gui.slm.transfer_desktop import TransferPage

            self._transfer_page = TransferPage()
        return self._transfer_page

    def _show_transfer_placeholder(self) -> None:
        """页 1 还没准备好: 栈里放一个占位页, 并启动后台预热 (只启动一次)。"""
        if self._transfer_placeholder is None:
            placeholder = QLabel("正在准备迁移学习闭环整形页…")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setObjectName("transferPlaceholder")
            self._transfer_placeholder = placeholder
            self._stack.addWidget(placeholder)
        self._stack.setCurrentWidget(self._transfer_placeholder)
        self._start_transfer_warmup()

    def _start_transfer_warmup(self) -> None:
        if self._warm_started or self._transfer_page is not None:
            return
        self._warm_started = True
        task = _WarmTask()
        task.signals.ready.connect(self._on_transfer_warm_ready)
        task.signals.failed.connect(self._on_transfer_warm_failed)
        self._warm_pool.start(task)

    def _on_transfer_warm_ready(self) -> None:
        """预热完成 (主线程): 构造真页面, 替换占位。窗口已关则不再建。"""
        if self._closing or self._transfer_page is not None:
            return
        page = self._transfer()
        placeholder = self._transfer_placeholder
        if placeholder is None:
            return
        was_current = self._stack.currentWidget() is placeholder
        self._stack.insertWidget(self._stack.indexOf(placeholder), page)
        self._stack.removeWidget(placeholder)
        placeholder.deleteLater()
        self._transfer_placeholder = None
        if was_current:
            self._stack.setCurrentWidget(page)

    def _on_transfer_warm_failed(self, message: str) -> None:
        """预热失败: 占位改成错误提示; 允许再次点击重试。"""
        self._warm_started = False
        if self._transfer_placeholder is not None:
            self._transfer_placeholder.setText(
                f"迁移学习闭环整形页加载失败：{message}\n请重新点击导航重试。"
            )

    def _static(self) -> QWidget:
        """惰性创建页 2 (SLM 静态整形); 建好之后缓存, 切回来不重建。

        静态整形页本身不 import SLM SDK, 慢的东西在下发时才在后台线程拉起; 惰性
        创建只是为了不在冷启动时多建一个页面对象。
        """
        if self._static_page is None:
            from ao_shaping.gui.slm.static_shaping_page import StaticShapingPage

            self._static_page = StaticShapingPage()
        return self._static_page

    def _page_for(self, index: int) -> QWidget:
        """按页索引取页面实例 (惰性建页 1 / 页 2)。"""
        if index == 1:
            return self._transfer()
        if index == 2:
            return self._static()
        return self._beam_app

    def _sync_stack(self, index: int) -> None:
        """把导航栏的选中态与页面栈对齐。"""
        self._rail.set_current(index)
        if index == 0:
            self._stack.setCurrentIndex(0)
            return
        if index == 1 and self._transfer_page is None:
            self._show_transfer_placeholder()
            return
        widget = self._page_for(index)
        if self._stack.indexOf(widget) < 0:
            self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)

    def select_page(self, index: int) -> None:
        """切页 (0 = 光束质量, 1 = 迁移学习闭环整形, 2 = SLM 静态整形)。"""
        self._sync_stack(max(0, min(index, len(_NAV_ITEMS) - 1)))

    # ------------------------------------------------------------------ 事件
    def closeEvent(self, event) -> None:
        """关窗: 先让页面自己收尾 (停相机 / 停后台作业), 再关。

        子页面的 ``closeEvent`` 不会被 Qt 自动转发, 所以这里显式转发 ——
        光束质量页的相机流不关就会占着设备。
        """
        self._closing = True
        for widget in (self._beam_app, self._transfer_page, self._static_page):
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
