"""把会话里填的"点"输入解析成 ``(float, float)`` (bench_settings 的叶子依赖)。

自 :mod:`ao_shaping.gui.slm.bench_settings` 拆出: ``"960,690"`` / ``(960, 690)`` /
``[960, 690]`` 这类输入在这里统一归一化。纯函数, 不读写 session, 也不区分
widget/session 值来源。
"""

from __future__ import annotations

from typing import Any


def _coerce_point(value: Any) -> tuple[float, float] | None:
    """``"960,690"`` / ``(960, 690)`` / ``[960, 690]`` → ``(960.0, 690.0)``。"""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace("，", ",").split(",") if p.strip() != ""]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None
    try:
        seq = list(value)
    except TypeError:
        return None
    if len(seq) != 2:
        return None
    try:
        return float(seq[0]), float(seq[1])
    except (TypeError, ValueError):
        return None


def _parse_point_text(text: str) -> tuple[float, float] | None:
    """``"960,690"`` → ``(960.0, 690.0)``；不合法返回 ``None``。"""
    parts = [p for p in str(text).replace("，", ",").split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None
