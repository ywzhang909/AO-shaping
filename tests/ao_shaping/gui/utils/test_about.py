"""「关于」对话框 (gui/utils/about) 的离屏测试。

三件事, 各有各的失败方式, 所以各测各的:

* **版本号**: ``git_commit_id()`` 必须在任何环境下都返回一个非空字符串 —— 有 git
  时是短 hash, 没有时是「未知」。它挂在「帮助」菜单上, 抛异常就会把弹窗卡死。
* **帮助文档链接**: 必须是 GitHub 地址 (离线可用), 而不是本地路径。
* **打开链接**: 「帮助文档」按钮必须真的调 :class:`QDesktopServices` 的 ``openUrl``
  —— 这是「点了没反应」的唯一可测痕迹, 所以把 ``openUrl`` 换成记录器。

``QDesktopServices.openUrl`` 是 Shiboken 绑定的静态方法, 直接给类属性赋值即可替换
(测试里已验证可替换且可还原)。
"""

from __future__ import annotations

import os
import re

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QLabel

from ao_shaping.gui.utils.about import (
    HELP_DOC_URL,
    UNKNOWN_COMMIT,
    AboutDialog,
    git_commit_id,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def dialog(app: QApplication):
    """新建「关于」弹窗并显示; 用完关掉, 不留悬空的模态窗口。"""
    window = AboutDialog()
    window.show()
    app.processEvents()
    yield window
    window.close()
    app.processEvents()


def test_git_commit_id_is_never_empty() -> None:
    """版本号永远返回一个非空字符串 (短 hash 或「未知」)。"""
    value = git_commit_id()
    assert isinstance(value, str)
    assert value.strip(), "版本号不允许为空字符串"
    assert value == UNKNOWN_COMMIT or re.fullmatch(r"[0-9a-f]{4,40}", value), (
        f"版本号应当是短 hash 或「未知」, 实得 {value!r}"
    )


def test_git_commit_id_falls_back_to_unknown_when_git_is_missing(monkeypatch) -> None:
    """git 不可用（没装 git / 没 .git 目录 / 冻结成 exe）时降级为「未知」。"""

    def _boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr("subprocess.run", _boom)
    assert git_commit_id() == UNKNOWN_COMMIT


def test_help_doc_url_points_at_github() -> None:
    """帮助文档链接必须指向 GitHub 上的用户手册。"""
    assert HELP_DOC_URL.startswith("https://github.com/")
    assert "transfer_desktop_user_manual.md" in HELP_DOC_URL


def test_dialog_shows_the_commit_id(dialog, app) -> None:
    """弹窗里能读到版本号（与 :func:`git_commit_id` 一致）。"""
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    assert any("版本：" in text for text in texts), "版本行缺失"
    assert any(git_commit_id() in text for text in texts), (
        f"版本行未显示 commit id {git_commit_id()!r}"
    )


def test_help_button_opens_the_doc_url(dialog, app, monkeypatch) -> None:
    """点「帮助文档」会用系统浏览器打开 GitHub 链接。"""
    opened: list[str] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True),
    )

    dialog._help_button.click()
    app.processEvents()

    assert opened == [HELP_DOC_URL]
