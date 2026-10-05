"""驱动包共用的 PEP 562 惰性属性安装器。

导入驱动包**不得**加载原生 SDK。这些副作用是真实存在且不明显的 ——
``ccd/miicam/driver.py`` 在模块作用域调用 ``_setup_miicam_sdk()``, 会改写
``sys.path`` 并通过 ``ctypes.CDLL`` 预加载 ``MIIUSB.dll``; ``ccd/daheng/driver.py``
则导入 ``gxipy`` 绑定。由于 Python 会先初始化父包再导入其子模块, 一次 eager 导入
就会让兄弟注册表的*每一个*使用者为此买单。

相机家族原先用模块级 ``__getattr__`` 解决了这个问题。随后那段实现体被复制到第二个
包里, 于是解析规则 (首次访问绑定、``globals()`` 缓存使失败的后端不被重试、优雅降级为
``None``) 同时存在于两处并可能漂移。本模块是唯一实现; 驱动包调用
:func:`install_lazy_attrs`。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from loguru import logger


def install_lazy_attrs(
    module_globals: dict[str, Any],
    backends: dict[str, tuple[str, str]],
    module_name: str,
    on_error: Callable[[str, BaseException], None] | None = None,
) -> Callable[[str], Any]:
    """构造一个 PEP 562 ``__getattr__``, 在首次访问时解析 ``backends``。

    Args:
        module_globals: 调用方包的 ``globals()``。解析成功的名字会缓存在其中,
            因此每个名字至多导入一次, 之后对该子模块的直接 import 也无法遮蔽该绑定。
        backends: 公共属性名 -> ``(模块路径, 属性)``。
        module_name: 点分包名, 用于错误消息。
        on_error: 可选钩子, 以 ``on_error(name, exception)`` 形式调用。
            默认为一条 debug 日志。从中返回 ``False`` 则重新抛出异常,
            而不是降级为 ``None``。

    Returns:
        一个适用于调用方包的 ``__getattr__`` 函数。
    """

    def __getattr__(name: str) -> Any:
        try:
            module_path, attr = backends[name]
        except KeyError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None

        try:
            value = getattr(import_module(module_path), attr)
        except Exception as exc:  # 缺失的 SDK 不应拖垮整个包
            if on_error is not None and on_error(name, exc) is False:
                raise
            logger.debug(f"{name} driver not available: {exc}")
            value = None

        module_globals[name] = value
        return value

    return __getattr__


__all__ = ["install_lazy_attrs"]
