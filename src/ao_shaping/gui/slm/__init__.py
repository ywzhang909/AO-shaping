"""空间光调制器 (SLM) UI (Streamlit + PySide6).

``transfer_desktop`` 是迁移学习闭环整形的 PySide6 桌面页 (与
``gui/ccd/beam_quality`` 同一个应用的两页)。它会引入 Qt, 因此按
PEP 562 惰性导出 —— 在这里 import 会让任何 ``import ao_shaping.gui.slm``
都拖起 Qt。
"""

from __future__ import annotations

from typing import Any

from ao_shaping.gui.slm import pyarrow_probe as _pyarrow_probe_mod

#: 惰性导出表: 子包引入 Qt, 不能在 ``import ao_shaping.gui.slm`` 时就加载。
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TransferPage": ("ao_shaping.gui.slm.transfer_desktop", "TransferPage"),
}

__all__ = ["TransferPage", "pyarrow_probe"]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
