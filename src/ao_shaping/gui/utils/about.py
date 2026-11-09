"""「关于」对话框：版本号（git commit id）＋帮助文档入口。

为什么版本号取 git commit id
-----------------------------
本项目还没有发版号体系（没有 ``pyproject`` 的 ``version`` 字段, 也没有 tag 流程）,
而实验室里跑的机器往往直接对着一个工作副本改代码。用 ``git rev-parse --short HEAD``
当版本号, 操作员报故障时只要念一串短 hash, 就能定位到当时的代码状态 —— 这比一个
永远不变的 "v1.0" 有用得多。

这个读取**必须永不抛异常**: 「关于」是一个纯信息弹窗, 它挂在「帮助」菜单上, 任何
用户都会点。没有 ``.git`` 目录（打包成 exe）、没装 git、git 进程超时, 都只应该让
版本号显示成「未知」, 而不该把弹窗卡在构造阶段。因此本模块只捕获
:class:`OSError` 与 :class:`subprocess.SubprocessError` 两种确切类型。

帮助文档
--------
帮助文档放在 GitHub 上（仓库 ``ywzhang909/AO-shaping`` 的
``docs/GUI/slm/transfer_desktop/``), 所以「帮助文档」按钮用系统默认浏览器打开
GitHub 链接, 而不是打包一份 HTML 进程序 —— 文档改了, 链接指过去就是最新的。

模块边界
--------
本模块只依赖 ``PySide6`` 与标准库, 不碰任何设备, 因此可以离屏测试。
"""

from __future__ import annotations

import subprocess

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.utils.io.file import ROOT_DIR

__all__ = ["HELP_DOC_URL", "UNKNOWN_COMMIT", "AboutDialog", "git_commit_id"]

#: 读不到 commit id 时的占位文案（中文）。
UNKNOWN_COMMIT = "未知"

#: 帮助文档（迁移桌面端用户手册）在 GitHub 上的地址。
HELP_DOC_URL = (
    "https://github.com/ywzhang909/AO-shaping/blob/master/docs/GUI/slm/"
    "transfer_desktop/transfer_desktop_user_manual.md"
)

#: 应用名（与窗口标题一致, 让「关于」与标题栏互相印证）。
APP_NAME = "AO-Shaping 实验系统控制台"

#: ``git`` 读取超时（秒）。git 读 HEAD 是纯本地操作, 正常在毫秒级完成; 超时只可能是
#: 仓库被网络盘卡住之类, 到点就放弃并显示「未知」。
_GIT_TIMEOUT_S = 5.0


def git_commit_id() -> str:
    """当前工作副本的短 commit id; 读不到时返回 :data:`UNKNOWN_COMMIT`。

    Returns:
        形如 ``4c8c7470`` 的短 hash, 或 ``"未知"``。

    本函数**永不抛异常**: 没有 git、没有 ``.git`` 目录（冻结成 exe）、git 非零退出、
    读取超时, 一律降级为 ``"未知"``。只捕获确切类型
    (:class:`OSError` / :class:`subprocess.SubprocessError`), 免得把真正的编程错误
    一起吞掉。
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_COMMIT
    if completed.returncode != 0:
        return UNKNOWN_COMMIT
    return completed.stdout.strip() or UNKNOWN_COMMIT


class AboutDialog(QDialog):
    """「关于」弹窗: 应用名 + 版本号（commit id）+ 帮助文档入口。

    三个读数, 各有各的用处:

    * **应用名** —— 与窗口标题一致, 操作员截图报问题时不必口述。
    * **版本: <commit id>** —— 定位当时的代码状态（见模块文档）。
    * **帮助文档** —— 按钮, 用系统默认浏览器打开 GitHub 上的用户手册。

    纯信息弹窗, 不改任何设置, 所以只有「帮助文档」与「关闭」两个出口。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setObjectName("aboutDialog")

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._app_label = QLabel(APP_NAME)
        self._app_label.setObjectName("aboutAppName")
        layout.addWidget(self._app_label)

        self._version_label = QLabel(f"版本：{git_commit_id()}")
        self._version_label.setObjectName("aboutVersion")
        layout.addWidget(self._version_label)

        self._help_button = QPushButton("帮助文档")
        self._help_button.setObjectName("aboutHelpButton")
        self._help_button.setToolTip("在浏览器中打开迁移桌面端用户手册")
        self._help_button.clicked.connect(self.open_help_doc)
        layout.addWidget(self._help_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.setObjectName("aboutButtons")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def open_help_doc(self) -> None:
        """用系统默认浏览器打开 :data:`HELP_DOC_URL`。

        单列出来是为了让测试能直接调它, 也为了让「按钮 → 打开浏览器」这条路径只有
        一份实现。浏览器打不开（无默认程序）时 Qt 会自己忽略, 这里不额外报错 ——
        「关于」弹窗不该因为一个链接而弹错误框。
        """
        QDesktopServices.openUrl(QUrl(HELP_DOC_URL))
