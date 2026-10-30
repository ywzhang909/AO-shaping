"""campaign 的模型复用面板（⑦ 那一块）：这条光路训练出来的模型还能不能拿来用。

为什么单独成模块
----------------
⑦ 里其实是两件事：**逐段/逐块的复盘读数**（跑到哪、CV 走到哪、花了多少）与
**模型复用判断**（同一条光路才成立：窗口 / 零级位置 / 曝光 / 相机格式 / 控制分辨率
任一变了，模型映射就不再适用）。前者的渲染体是 ``multishape_campaign_ui.render_progress_panel``，
被 ⑥ 运行中 / ⑥ 空闲 / ⑦ 历史三处共用；后者只服务 ⑦，却要接
:mod:`ao_shaping.gui.slm.history_browser` 与 :mod:`ao_shaping.tools.slm.transfer.optical_session`。
两者放同一个文件时主文件越过模块规模棘轮线（``scripts/check_module_size.py``：>800 行受管、
只许变小），所以把后者整块搬到这里 —— 主文件只 re-export 这几个名字，对外 API 一字未改，
行为也不变。

阈值与打分只有一份
------------------
兼容分与阈值全部在 ``optical_session``（``COMPAT_THRESHOLD`` / ``DIMENSION_WEIGHTS``），
候选检查点的扫描、排序与打分在 ``history_browser.render_history_browser`` 内部。本模块
**不复制**这两件事：它只把「当前光路指纹」投进 ``st.session_state[current_fp_key(key_prefix)]``，
再把画卡 / 画列表的活转交给历史浏览器。

离线可用
--------
不需要设备：指纹优先读目录里已落盘的 ``fingerprint.json``（采集侧写的），没有再按目录里
已有产物现场算，且**只读不落盘**。``history_browser`` 或 ``optical_session`` 缺失时各自
降级成一行 caption，其余读数不受影响。

导入层级
--------
模块作用域只 import ``streamlit`` / ``loguru`` / 标准库；设备与上层工具
（``history_browser``、``optical_session``）一律函数内惰性 import，与拆分前一致。
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import streamlit as st
from loguru import logger

__all__ = [
    "HISTORY_PREFIX",
    "campaign_fingerprint",
    "has_fingerprint_readings",
    "render_reuse_panel",
]

#: ⑦ 交给 ``history_browser`` 的键前缀：它的「当前光路指纹」state 键由它自己拼
#: （``current_fp_key(prefix)``），页面不另编一套键名。
HISTORY_PREFIX = "msc_hist"


def _history_browser() -> "ModuleType | None":
    """惰性取历史浏览器；模块没就绪时返回 ``None``（⑦ 的模型复用降级为一行说明）。

    写法与 ``campaign_monitor._campaign_control`` 同款（惰性 import ＋ ``logger`` ＋
    ``None``，由渲染侧印一行）。缺这块只丢「模型复用」，计划 / 检查卡 / 逐段读数照常 ——
    只装了 GUI 依赖的机器不该整页崩。
    """
    try:
        from ao_shaping.gui.slm import history_browser
    except ImportError:
        logger.debug("history_browser 尚未就绪：⑦ 的模型复用降级为一行说明")
        return None
    missing = [name for name in ("current_fp_key", "render_fingerprint_card",
                                 "render_history_browser")
               if not hasattr(history_browser, name)]
    if missing:
        logger.warning("history_browser 缺少 {}：⑦ 的模型复用降级为一行说明", missing)
        return None
    return history_browser


def campaign_fingerprint(directory: Path) -> "dict[str, Any] | None":
    """选中目录的光路指纹：已落盘的 ``fingerprint.json`` 优先，没有再按产物现场算。

    2026-10-29 现场：⑦ 看得出「这条 campaign 跑到哪」，看不出「这条光路的模型还能不能
    复用」—— 复用判断散在别处、靠人看目录名。指纹的字段来源与阈值只有
    ``optical_session`` 一份（``load_fingerprint`` / ``compute_optical_fingerprint``），
    本条路径**只读**：只是回看历史，不在这里落盘写 ``fingerprint.json``（写盘属采集侧）。
    指纹模块缺失 ⇒ ``None``，由调用方印一行说明。字段全缺时返回的是一份全 ``None``
    的字典（``compute_optical_fingerprint`` 只现场补 ``created_at``），不是异常。
    """
    try:
        from ao_shaping.tools.slm.transfer.optical_session import (
            compute_optical_fingerprint, load_fingerprint)
    except ImportError:
        logger.debug("optical_session 尚未就绪：⑦ 的光路指纹降级为一行说明")
        return None
    stored = load_fingerprint(directory)
    return dict(stored) if stored else compute_optical_fingerprint(directory)


def has_fingerprint_readings(fingerprint: "dict[str, Any] | None") -> bool:
    """指纹里有没有**真读数**（``created_at`` 由 ``compute_optical_fingerprint`` 现场补，
    单独一条时间戳不构成可比对的指纹）。"""
    return bool(fingerprint) and any(
        value is not None for key, value in fingerprint.items() if key != "created_at")


def render_reuse_panel(directory: Path, *, key_prefix: str = HISTORY_PREFIX) -> None:
    """这条光路的模型还能不能复用：指纹卡 ＋ 候选检查点（排序 / 就绪 / 手选都在历史浏览器）。

    候选与兼容分**不在本页算**：``render_history_browser`` 内部扫
    ``scan_checkpoint_index`` 并调 ``compatibility_score``，阈值与权重只有
    ``optical_session`` 一份（``COMPAT_THRESHOLD`` / ``DIMENSION_WEIGHTS``）；页面只负责把
    「当前光路指纹」投进 ``st.session_state[current_fp_key(key_prefix)]`` —— 不投时它的
    兼容过滤是禁用的。
    """
    st.markdown("**模型复用（这条光路）**")
    browser = _history_browser()
    if browser is None:
        st.caption("历史浏览器模块尚未就绪：这条光路能否复用只能人工看目录名，其余读数不受影响。")
        return
    fingerprint = campaign_fingerprint(directory)
    if fingerprint is None:
        st.caption("光路指纹模块尚未就绪：这条光路没有指纹可比对，其余读数不受影响。")
        return
    st.session_state[browser.current_fp_key(key_prefix)] = fingerprint
    if has_fingerprint_readings(fingerprint):
        browser.render_fingerprint_card(fingerprint, key_prefix=key_prefix)
    else:
        st.caption("未记录：这条目录的产物里没有窗口 / 曝光 / 零级读数（早于指纹功能，"
                   "或产物已被清过），指纹无从比对。")
    browser.render_history_browser(root=directory, key_prefix=key_prefix)
