"""Canonical target-shape vocabulary for the closed-loop SLM shaping chain.

This module is the single source of truth for the shape names the closed-loop
shaping GUI (``gui/slm/shaping_controls.py``) and the model-in-loop adapter pass
around.  It promotes the former plain ``MIL_TARGET_SHAPE_LABELS`` dict into a
real :class:`TargetShape` enum so a shape can be threaded through the optimizers
and validated in one place.

Scope note: :data:`ao_shaping.utils.image.target.metrics.TargetShape` is a
separate ``Literal`` that belongs to the slm-pib objective system (its own
``TARGET_SHAPE_CHOICES`` also carries ``annular`` / ``grid`` / ``cross`` /
``gaussian`` / ``pentagon``).  The two vocabularies are intentionally **not**
merged in this change; this module only covers the five shapes the closed-loop
shaping chain supports.
"""

from __future__ import annotations

from enum import Enum


class TargetShape(str, Enum):
    """The five target shapes supported by the closed-loop shaping chain.

    Subclasses :class:`str` so members compare equal to their raw string values
    (existing callers pass bare strings such as ``"square"`` around).
    """

    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    STAR = "star"
    CIRCLE = "circle"


#: Chinese GUI labels for each shape, in declaration order.  Preserved verbatim
#: from the former ``shaping_controls.MIL_TARGET_SHAPE_LABELS``.
TARGET_SHAPE_LABELS: dict[str, str] = {
    "square": "正方形",
    "rectangle": "长方形（2:1）",
    "triangle": "三角形",
    "star": "五角星",
    "circle": "圆形",
}


def target_shape_options() -> list[str]:
    """Return the selectable shape option strings in declaration order.

    Returns:
        The :data:`TARGET_SHAPE_LABELS` keys as a new list, ready to feed a
        Streamlit ``selectbox``.
    """
    return list(TARGET_SHAPE_LABELS)


def coerce_target_shape(value: str | TargetShape) -> TargetShape:
    """Coerce ``value`` to a :class:`TargetShape`.

    Args:
        value: A :class:`TargetShape` member (returned unchanged) or a raw
            shape string such as ``"square"``.

    Returns:
        The matching :class:`TargetShape` member.

    Raises:
        ValueError: If ``value`` is not one of the supported shapes.  The
            message lists the valid values.
    """
    if isinstance(value, TargetShape):
        return value
    key = str(value).lower()
    try:
        return TargetShape(key)
    except ValueError:
        valid = ", ".join(member.value for member in TargetShape)
        raise ValueError(
            f"unknown target shape {value!r}; valid values: {valid}"
        ) from None
