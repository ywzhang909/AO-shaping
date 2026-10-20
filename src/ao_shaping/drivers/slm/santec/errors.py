"""SLM 驱动错误 → 可执行的中文提示。

驱动层抛出的异常只带一个原始错误码 (例如 ``-10003``), 对操作员没有指导意义:
光看到「USB设备已打开」并不知道**下一步该做什么**。本模块把捕获到的异常翻译成
「发生了什么 + 该怎么处理」的中文提示, 供 GUI (Streamlit) 与离线脚本共用。

设计约束:

- **不依赖 Streamlit**: 本模块只依赖 :mod:`ao_shaping.drivers.slm.santec.constants`,
  因此可以在没有 GUI 运行时的环境里直接单元测试。
- **复用既有码表**: 错误码 → 中文短描述取自
  :func:`ao_shaping.drivers.slm.santec.constants.get_slm_error_message`,
  不在这里重复抄写数字。
- **不丢信息**: 拿不到错误码时原样返回异常文本; 拿得到时也会附上「原始错误」。
"""

from __future__ import annotations

import re

from ao_shaping.drivers.slm.santec.constants import (
    SLMErrorCode,
    get_slm_error_message,
)

__all__ = [
    "USB_ALREADY_OPEN_CODE",
    "USB_LINK_BROKEN_CODES",
    "describe_santec_error",
    "extract_error_code",
]

#: 设备已被别的进程打开 (USB 句柄被独占)。
USB_ALREADY_OPEN_CODE = SLMErrorCode.FT_DEVICE_NOT_OPENED

#: USB 链路 / 句柄本身出了问题, 需要重新上电才能恢复的一组错误码。
USB_LINK_BROKEN_CODES = frozenset(
    {
        SLMErrorCode.FT_INVALID_HANDLE,  # -10001
        SLMErrorCode.FT_OTHER_ERROR,  # -10032
        SLMErrorCode.SLM_NOT_OPEN_USB,  # -200
    }
)

# 异常文本里的「错误码: -10003」——用于异常对象没带 ``code`` 属性时的回退解析。
_CODE_PATTERN = re.compile(r"错误码[:：]\s*(-?\d+)")

_ALREADY_OPEN_TEMPLATE = (
    "SLM 的 USB 设备已被其他进程占用（错误码 {code}，{detail}）："
    "另一个程序正持有该 SLM 的 USB 句柄，同一设备无法被二次打开。\n\n"
    "处理办法：\n"
    "1. 关闭正在使用该 SLM 的脚本，或关闭另一个打开本控制页面的浏览器标签；\n"
    "2. 若找不到明显的占用者，在任务管理器中结束残留的 python 进程"
    "（上一次运行异常退出会留下 USB 句柄）；\n"
    "3. 必要时给 SLM 控制器断电重连，然后重新点击「连接」。"
)

_USB_LINK_BROKEN_TEMPLATE = (
    "SLM 的 USB 链路或句柄已失效（错误码 {code}，{detail}）："
    "USB 连接已断开或句柄不可用，设备端可能需要重新上电。\n\n"
    "处理办法：\n"
    "1. 检查 USB 线缆与 SLM 控制器供电，确认 USB 已正确连接；\n"
    "2. 确认没有其他程序占用该设备（若有，按「设备被占用」的提示处理）；\n"
    "3. 给 SLM 控制器断电重启后重试。"
)


def extract_error_code(exc: BaseException) -> int | None:
    """取出异常携带的 SLM SDK 错误码。

    优先读异常对象的 ``code`` 属性（:class:`SantecError` 提供）, 缺失时从异常
    文本里回退解析 ``错误码: -10003``。两者都没有则返回 ``None``。

    Args:
        exc: 捕获到的异常对象。

    Returns:
        错误码整数, 或 ``None``（无法判定时）。
    """
    code = getattr(exc, "code", None)
    if isinstance(code, bool):
        code = None  # bool 是 int 的子类, 但显然不是错误码
    if isinstance(code, int):
        return code
    match = _CODE_PATTERN.search(str(exc))
    if match is not None:
        return int(match.group(1))
    return None


def describe_santec_error(exc: BaseException) -> str:
    """把捕获到的异常转成可执行的中文提示。

    - 错误码 :data:`USB_ALREADY_OPEN_CODE` (``-10003``): 说明设备被别的进程
      独占, 并给出「关脚本 / 关页面 / 结束残留进程 / 断电重连」的处理办法;
    - 错误码 :data:`USB_LINK_BROKEN_CODES`: 说明 USB 链路或句柄失效, 设备
      可能需要断电重启;
    - 其他带码错误: 给出码表里的中文短描述 + 原始信息;
    - 拿不到错误码: 原样返回异常文本（绝不丢信息）。

    Args:
        exc: 捕获到的异常对象。

    Returns:
        面向操作员的中文提示文本。
    """
    raw = str(exc).strip()
    code = extract_error_code(exc)
    if code is None:
        return raw or exc.__class__.__name__

    detail = get_slm_error_message(code)
    if code == USB_ALREADY_OPEN_CODE:
        head = _ALREADY_OPEN_TEMPLATE.format(code=code, detail=detail)
    elif code in USB_LINK_BROKEN_CODES:
        head = _USB_LINK_BROKEN_TEMPLATE.format(code=code, detail=detail)
    else:
        head = f"SLM 设备返回错误（错误码 {code}，{detail}）。"

    if not raw:
        return head
    return f"{head}\n\n原始错误：{raw}"
