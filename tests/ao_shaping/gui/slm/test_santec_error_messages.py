"""SLM 连接失败提示的离线单元测试 (无需 Streamlit 运行时, 无需硬件)。

覆盖 :mod:`ao_shaping.drivers.slm.santec.errors`:

- ``-10003`` (USB 设备已被别的进程打开): 必须说清「被占用」并给出处理办法;
- 另一组 USB 错误码 (``-10001`` / ``-10032`` / ``-200``): 必须说 USB 链路或句柄
  失效、设备可能需要重新上电;
- 取不到错误码的普通异常: 原样返回异常文本, 绝不丢信息。

本模块在 import 期只引入驱动层辅助函数, **不引入 streamlit**。
"""

from __future__ import annotations

from ao_shaping.drivers.slm.santec import SantecError
from ao_shaping.drivers.slm.santec.constants import SLMErrorCode
from ao_shaping.drivers.slm.santec.errors import (
    USB_ALREADY_OPEN_CODE,
    USB_LINK_BROKEN_CODES,
    describe_santec_error,
    extract_error_code,
)


def _santec_error(code: int, message: str = "无法打开SLM #1") -> SantecError:
    """构造一个带错误码的驱动异常 (与 ``Santec.open()`` 抛出的一致)。"""
    return SantecError(message, code=code)


class TestAlreadyOpen:
    """-10003: 设备被别的进程独占。"""

    def test_code_constant(self) -> None:
        assert USB_ALREADY_OPEN_CODE == -10003
        assert USB_ALREADY_OPEN_CODE is SLMErrorCode.FT_DEVICE_NOT_OPENED

    def test_message_explains_occupancy_and_next_steps(self) -> None:
        text = describe_santec_error(_santec_error(-10003))

        assert "已被其他进程占用" in text
        assert "-10003" in text
        # 复用码表里的中文短描述, 而不是自己另抄一份数字。
        assert "USB设备已打开" in text
        # 必须给出下一步动作。
        assert "处理办法" in text
        assert "关闭" in text
        assert "任务管理器" in text
        assert "断电" in text

    def test_raw_error_is_preserved(self) -> None:
        text = describe_santec_error(_santec_error(-10003))

        assert "无法打开SLM #1" in text

    def test_extract_code_from_attribute(self) -> None:
        assert extract_error_code(_santec_error(-10003)) == -10003

    def test_extract_code_from_text(self) -> None:
        exc = RuntimeError("无法打开SLM #1 (错误码: -10003, USB设备已打开)")

        assert extract_error_code(exc) == -10003
        assert "已被其他进程占用" in describe_santec_error(exc)


class TestUsbLinkBroken:
    """-10001 / -10032 / -200: USB 链路或句柄失效。"""

    def test_code_set(self) -> None:
        assert set(USB_LINK_BROKEN_CODES) == {-10001, -10032, -200}

    def test_invalid_handle(self) -> None:
        text = describe_santec_error(_santec_error(-10001))

        assert "USB 链路或句柄已失效" in text
        assert "-10001" in text
        assert "USB驱动句柄无效" in text
        assert "断电" in text

    def test_other_usb_error(self) -> None:
        text = describe_santec_error(_santec_error(-10032))

        assert "USB 链路或句柄已失效" in text
        assert "-10032" in text
        assert "USB其他错误" in text

    def test_usb_not_connected(self) -> None:
        text = describe_santec_error(_santec_error(-200))

        assert "USB 链路或句柄已失效" in text
        assert "-200" in text
        assert "USB未连接或未打开" in text
        assert "断电" in text


class TestGenericException:
    """取不到错误码时不能丢信息。"""

    def test_plain_exception_text_is_returned(self) -> None:
        exc = RuntimeError("面板上找不到 0 级光斑")

        assert describe_santec_error(exc) == "面板上找不到 0 级光斑"

    def test_plain_exception_keeps_message(self) -> None:
        exc = ValueError("波长必须是正整数")

        assert "波长必须是正整数" in describe_santec_error(exc)

    def test_unknown_code_uses_code_table_and_raw_text(self) -> None:
        # 码表里没有的错误码: 给出码 + 原始信息, 仍然不丢信息。
        exc = _santec_error(-99999, "未预期的驱动返回")

        text = describe_santec_error(exc)
        assert "-99999" in text
        assert "未预期的驱动返回" in text

    def test_known_other_code_uses_code_table(self) -> None:
        # 超时 (FT_TIMEOUT=-10019) 不属于特判的两组, 但仍要给出码表描述。
        text = describe_santec_error(_santec_error(-10019))

        assert "-10019" in text
        assert "操作超时" in text

    def test_empty_message_falls_back_to_class_name(self) -> None:
        assert describe_santec_error(RuntimeError()) == "RuntimeError"
