"""设备枚举 / 连接判定 / 无设备提示 —— 与 ``multi_slm_controller`` 同一入口。

首次进入时枚举设备（每轮 rerun 重扫等于每次点按钮白等十几秒，所以只做一次）。
"""

from __future__ import annotations

import streamlit as st

from ao_shaping.gui.slm.multi_slm_controller import (
    _initialize_slm_state,
    _refresh_device_list,
)


def ensure_state() -> None:
    """首次进入时枚举设备（与 ``multi_slm_controller.main`` 同一入口）。"""
    if "available_slms" not in st.session_state:
        _initialize_slm_state()
        _refresh_device_list()


def connected_slm_numbers() -> list[int]:
    """已连接（有 handle 且状态为已连接）的 SLM 编号。"""
    return [
        n
        for n in (1, 2)
        if st.session_state.get(f"slm{n}_connected")
        and st.session_state.get(f"slm{n}") is not None
    ]


def render_connection_hint(page_title: str) -> None:
    """没有任何已连接 SLM 时的页面主体。"""
    st.title(page_title)
    st.info("请在左侧连接至少一个 SLM")
    st.caption(
        "本页只需一台 SLM 与一台相机：光路参数在左侧「CCD参数设置」，"
        "标定按钮同在那里。"
    )
