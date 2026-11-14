"""大恒位深 → PixelFormat 单一映射与回读归一化 (零导入叶子)。

位深决定满量程 (8/12/16 bit = 255/4095/65535), 映射集中在本模块一处;
``DahengCamera`` 的 ``_BIT_DEPTH_PIXEL_FORMATS`` 类属性引用这里的表,
加一档只改这一处。回读归一化只依赖本地 ``constants`` (不 import gxipy),
因此离线可测。
"""

from __future__ import annotations

from typing import Any

from ao_shaping.drivers.ccd.daheng import constants

#: 位深 → GenICam ``PixelFormat`` 名。12 bit 必须落到 MONO12, **不允许静默回落**
#: 成 MONO8: 满量程从 4095 变成 255, 一切按满量程折算的判据都会错量纲。
_BIT_DEPTH_PIXEL_FORMATS: dict[int, str] = {
    8: "MONO8",
    12: "MONO12",
    16: "MONO16",
}


def bit_depth_of_format(value: Any) -> int | None:
    """把设备回读的 ``PixelFormat`` 归一化成位深 (bit); 认不出来返回 ``None``。

    gxipy 枚举 ``get()`` 形态不统一: 名称串 (``"Mono12"``) / PFNC 代码
    (``0x01100005``) / 带 ``.name`` 的枚举成员 / ``(代码, 名称)`` 元组
    (本驱动 ``StreamBufferHandlingMode.get()[0]`` 即此形态), 故对四种形态容错。
    """
    candidates = list(value) if isinstance(value, (tuple, list)) else [value]
    for candidate in candidates:
        member_name = getattr(candidate, "name", None)
        if isinstance(member_name, str):  # Enum 成员 → 用它的名字
            candidate = member_name
        if isinstance(candidate, str):
            text = candidate.strip().upper()
            if text.startswith("MONO") and text[4:].isdigit():
                return int(text[4:])
            continue
        code = getattr(candidate, "value", candidate)
        try:
            code_int = int(code)
        except (TypeError, ValueError):
            continue
        for bits, fmt_name in _BIT_DEPTH_PIXEL_FORMATS.items():
            entry = getattr(constants.GxPixelFormatEntry, fmt_name, None)
            if entry is not None and int(getattr(entry, "value", entry)) == code_int:
                return bits
    return None
