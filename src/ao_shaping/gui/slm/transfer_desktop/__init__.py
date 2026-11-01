"""SLM 迁移学习闭环整形的 PySide6 桌面页。

模块划分:

* :mod:`~.params`  表单值 → 校验 → 会话参数 (纯逻辑, 不 import Qt)
* :mod:`~.page`    桌面页 (PySide6): 设备 → 目标与预算 → 校验 → 运行 → 进度

为什么按 PEP 562 惰性导出
------------------------
``page`` 会 import PySide6, 而 ``ao_shaping.gui.slm`` 本身可能被 streamlit 页面
import (``shaping_controls`` 等)。如果 ``transfer_desktop/__init__.py`` 在 import
时就拉起 Qt, 那么任何 ``import ao_shaping.gui.slm`` 都会拖起一整套 Qt —— 在
streamlit 进程里这是纯浪费, 在无显示环境里甚至会失败。所以这里只在**真正访问**
``TransferPage`` 时才 import 子模块。
"""

from __future__ import annotations

from typing import Any

#: 惰性导出表: 子模块引入 Qt, 不能在 ``import ao_shaping.gui.slm.transfer_desktop``
#: 时就加载。
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TransferPage": ("ao_shaping.gui.slm.transfer_desktop.page", "TransferPage"),
}

__all__ = ["TransferPage"]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
