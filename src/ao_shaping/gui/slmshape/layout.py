"""页面渲染目标 + 侧栏「页面布局」选择器。

Streamlit 的渲染位置由调用点的**容器**决定，所以「在谁里面画」是一个显式参数
而不是全局状态 —— 否则测试替身与其它页面会互相串味。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

__all__ = [
    "PLAIN_LAYOUT",
    "Layout",
    "layout",
    "layout_options",
    "render_layout_selector",
]

#: 「默认主区域」选项的文案。
PLAIN_LAYOUT = "（无，默认主区域）"


class Layout:
    """页面渲染目标：普通模式 = 主区 ``st``；折叠面板模式 = ``st.status`` 内部。

    本类只做两件事：``with layout:`` 进入该容器，以及把属性访问转给容器
    （``layout.columns`` / ``layout.button`` … 与 ``st`` 一致）。
    """

    __slots__ = ("_container", "_target")

    def __init__(self, target: Any, container: Any = None) -> None:
        self._target = target
        self._container = container

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __enter__(self) -> "Layout":
        if self._container is not None:
            self._container.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._container is not None:
            self._container.__exit__(*exc)


def layout(box_key: str) -> Layout:
    """当前页面的渲染目标（普通主区，或侧栏选中的专用子标题面板）。

    Args:
        box_key: 该页自己的折叠面板句柄名。``None``（未选折叠面板）即主区。
    """
    box = st.session_state.get(box_key)
    if box is not None:
        return Layout(box)
    return Layout(st)


def layout_options() -> list[str]:
    """「页面布局」的选项：默认主区，或挂到专用子标题折叠面板。"""
    return [PLAIN_LAYOUT, "专用子标题（折叠面板）"]


def render_layout_selector(
    option_key: str, box_key: str, title: str, *, help: str | None = None
) -> None:
    """侧栏「页面布局」：把主体挂进 ``st.status`` 折叠面板，或留在主区。

    Args:
        option_key: 选择器自己的 widget key（各页独立，不串页）。
        box_key: 折叠面板句柄在 ``st.session_state`` 里的键；选默认时写 ``None``。
        title: 折叠面板的标题（通常就是页面标题）。
        help: 选择器的说明文案，缺省用通用说明。
    """
    options = layout_options()
    current = st.session_state.get(option_key, options[0])
    index = options.index(current) if current in options else 0
    st.selectbox(
        "页面布局",
        options=options,
        index=index,
        key=option_key,
        help=help
        or "折叠面板模式把主体收进一块可折叠的「专用子标题」容器；"
        "需要屏幕空间时用默认主区域",
    )
    if st.session_state[option_key] == PLAIN_LAYOUT:
        st.session_state[box_key] = None
        return
    st.session_state[box_key] = st.status(f"{title}（专用子标题）", expanded=True)
