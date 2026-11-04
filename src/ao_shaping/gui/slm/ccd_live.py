"""「实时 CCD 画面」: 复用实验系统相机配置的 streamlit 直播页 (fragment 自动刷新)。

设计契约 (父级 ``multi_slm_controller`` 依赖这些名字)
------------------------------------------------------
* :func:`render_ccd_live` —— 本 tab 的入口, 复用 :func:`bench_camera` 的配置。
* :func:`close_bench_camera` —— 父级 ``main()`` 在关页/切 tab 时可调用, 释放句柄。

为什么复用 ``bench_camera`` 而不是新加选择器
--------------------------------------------
相机型号/ID 是**一次实验内固定**的量 (侧栏「CCD参数设置」区); 本页只消费它,
不重复造一个 `cam_type`/`cam_id` 的 widget —— 否则两处各持一套选择, 换光路
直接失效。每当 ``bench_camera()`` 返回的键 ``(cam_type, cam_id)`` 变化, 句柄缓存
自动重建 (先关旧句柄再开新的)。

句柄缓存
--------
Streamlit 每次脚本运行都从零渲染, 但相机 `open()` 是昂贵的硬件操作。因此把已 open
的句柄存进 ``session_state`` (键 ``ccd_live_cam``), 配套记它对应的 ``(cam_type,
cam_id)`` 键 ``ccd_live_cache_key``; 键没变就复用句柄, 变化才重建。用户点「关闭/
释放相机」或父级 ``main()`` 显式关闭时, 才真正 ``close()`` 并把句柄从缓存清掉。

刷新节奏
--------
直播用 ``@st.fragment(run_every=...)`` 自动重跑, 每轮只抓**最新一帧**再 ``st.image``
—— 单帧 grab, 不做有界缓冲队列。抓帧异常 (无相机 / 句柄被其它进程占用) 只提示
修补办法, 绝不把 fragment 打崩。

离线可测性
----------
纯/离线逻辑 (缓存键、句柄生命周期、帧归一化、曝光写入入口) 与 ``st.*`` 调用分离,
测试经 `monkeypatch.setattr(module, "st", stub)` 驱动, 不碰真相机。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera
from ao_shaping.drivers.ccd.common import (
    create_camera,
    get_camera_exposure_ms,
    get_camera_exposure_range,
)
from ao_shaping.gui.slm.bench_settings import bench_camera

__all__ = [
    "BENCH_DEFAULT_EXPOSURE_MS",
    "REFRESH_INTERVAL",
    "cache_key_for",
    "cached_handle",
    "close_bench_camera",
    "ensure_camera_handle",
    "normalize_frame",
    "release_camera_handle",
    "render_ccd_live",
]

#: fragment 自动重跑间隔 (s) —— 每轮抓一帧, 平衡刷新率与抓帧耗时。
REFRESH_INTERVAL = 0.15
#: 新建句柄的初始曝光 (ms)。侧栏会按光路给推荐值, 这里只作 fallback。
BENCH_DEFAULT_EXPOSURE_MS = 5.0

#: session_state 缓存键。
_HANDLE_KEY = "ccd_live_cam"
_CACHE_KEY_KEY = "ccd_live_cache_key"
#: 直播运行状态键。
_RUNNING_KEY = "ccd_live_running"

#: 供离线测试关闭真实 fragment 装饰器调用的开关 (测试置 False 走 ``_render_live_frame``)。
_FRAGMENT_ENABLED = True


def cache_key_for(cam_type: str, cam_id: int) -> str:
    """把 ``(cam_type, cam_id)`` 压成 session_state 缓存键。"""
    return f"{cam_type}::{int(cam_id)}"


def cached_handle(st: Any) -> BaseCamera | None:
    """读取缓存的已 open 句柄; 无则 ``None``。"""
    return st.session_state.get(_HANDLE_KEY)


def release_camera_handle(st: Any) -> None:
    """关闭缓存句柄 (幂等) 并清缓存。关页/换键/点释放都用它。"""
    cam = st.session_state.pop(_HANDLE_KEY, None)
    st.session_state.pop(_CACHE_KEY_KEY, None)
    if cam is None:
        return
    # 先停流再关 —— start_stream 起的取流线程不能被 close 打断。
    try:
        stop = getattr(cam, "stop_stream", None)
        if callable(stop):
            stop()
    except Exception:  # policy: 后端差异不可枚举, 释放路径不能因单步失败中断
        logger.exception("release_camera_handle: stop_stream 失败, 继续 close")
    try:
        cam.close()
    except Exception:  # policy: 同上 —— 释放是尽力而为
        logger.exception("release_camera_handle: close 失败, 句柄已从缓存清除")


def ensure_camera_handle(
    st: Any,
    cam_type: str,
    cam_id: int,
    *,
    create: Callable[..., BaseCamera] | None = None,
    exposure_time_ms: float = BENCH_DEFAULT_EXPOSURE_MS,
) -> BaseCamera | None:
    """为 ``(cam_type, cam_id)`` 取一个已 open 的缓存句柄。

    - 缓存键与请求一致 → 直接返回缓存句柄 (不复开);
    - 缓存键变了 → 先 :func:`release_camera_handle` 再建新句柄;
    - 创建或 ``open()`` 失败 → 记录日志、清缓存, 返回 ``None`` (调用方渲染空态)。

    ``create`` 可注入以便离线测试; 缺省时在**调用点**取模块级 :data:`create_camera`
    (不绑定默认值, 便于测试换替身)。
    """
    factory = create if create is not None else create_camera
    key = cache_key_for(cam_type, cam_id)
    current = st.session_state.get(_HANDLE_KEY)
    if current is not None and st.session_state.get(_CACHE_KEY_KEY) == key:
        return current

    release_camera_handle(st)
    try:
        cam = factory(cam_type, cam_id, exposure_time_ms=float(exposure_time_ms))
    except Exception:  # policy: 后端 SDK 缺失/类型未知/枚举失败, 构造即抛
        logger.exception("ensure_camera_handle: create({}, {}) 失败", cam_type, cam_id)
        return None
    try:
        cam.open()
    except Exception:  # policy: 设备被占用 / 未找到, open 即抛
        logger.exception("ensure_camera_handle: open({}, {}) 失败", cam_type, cam_id)
        try:
            cam.close()
        except Exception:  # policy: 释放尽力而为
            logger.exception("ensure_camera_handle: open 失败后 close 也失败")
        return None
    st.session_state[_HANDLE_KEY] = cam
    st.session_state[_CACHE_KEY_KEY] = key
    return cam


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """把任意灰度 raw 帧压成 0..255 uint8 供 ``st.image`` 直接显示。

    与 ``transfer_desktop/live_view.display_frame`` 同规则 (减 1% 分位背景再按峰值
    归一): 直接 ``clip`` 会让 counts 口径的一帧几乎全黑。这里**本地复制**而不 import
    ``transfer_desktop.live_view``, 因为那模块顶层拉 Qt 而本页是纯 streamlit fragment。
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = np.clip(arr - float(np.percentile(arr, 1)), 0.0, None)
    peak = float(arr.max())
    if peak > 0:
        arr = arr * (255.0 / peak)
    return arr.astype(np.uint8)


def apply_exposure_ms(cam: BaseCamera, exposure_ms: float) -> float:
    """改曝光的安全入口; 委托 ``set_exposure_ms`` (返回生效值)。"""
    return cam.set_exposure_ms(float(exposure_ms))


def _render_live_frame(cam: BaseCamera) -> None:
    """抓最新一帧并 ``st.image`` 显示 (fragment 的实质内容, 无 ``st.fragment``)。

    抓帧异常只提示修补办法, 不抛。单独拆出是为了离线测试可直用 ``_render_live_frame``
    驱动 stub, 而不触发真 ``@st.fragment`` 运行时。
    """
    try:
        frame = cam.get_numpy_image(1, skip_first=False)
    except Exception as exc:  # policy: 后端异常类型不可枚举
        logger.exception("实时抓帧失败")
        st.warning(
            f"实时抓帧失败: {exc!r} —— 请确认实验系统相机在侧栏「CCD参数设置」配置正确,"
            " 且句柄未被其他进程占用 (点「关闭/释放相机」后重试, 或换台机器再拉流)。"
        )
        return
    st.image(normalize_frame(frame), caption="实时 CCD 画面")


@st.fragment(run_every=REFRESH_INTERVAL)
def _live_fragment(cam: BaseCamera) -> None:
    """自动刷新的直播 fragment: 每 ``REFRESH_INTERVAL`` 秒重跑抓一帧。"""
    _render_live_frame(cam)


def _mount_live_view(cam: BaseCamera) -> None:
    """挂载直播视图; 测试关掉 ``_FRAGMENT_ENABLED`` 时直接走实质渲染。"""
    if _FRAGMENT_ENABLED:
        _live_fragment(cam)
    else:
        _render_live_frame(cam)


def close_bench_camera() -> None:
    """父级 ``main()`` 在切 tab / 关页时调用, 释放当前缓存的相机句柄。"""
    release_camera_handle(st)


def render_ccd_live() -> None:
    """「实时 CCD 画面」tab 入口: 复用实验系统相机配置的连续抓帧直播。

    空态/错误一律点名具体操作 (侧栏「CCD参数设置」、开始时按钮、释放按钮),
    不出现无指引的一句话提示; 全程无 ``st.divider``, 无 ``type="primary"`` 按钮。
    """
    cam_type, cam_id = bench_camera()
    st.caption(f"实验系统相机: {cam_type} (id={cam_id}) —— 在侧栏「CCD参数设置」修改")

    running = bool(st.session_state.get(_RUNNING_KEY, False))

    if st.button("❚❚ 暂停" if running else "▶ 开始实时采集", key="ccd_live_toggle"):
        st.session_state[_RUNNING_KEY] = not running
        st.rerun()
        return

    # 未运行: 空态引导点名开始按钮; 有残留句柄才给释放入口。
    if not running:
        if cached_handle(st) is not None:
            if st.button("关闭/释放相机", key="ccd_live_release"):
                close_bench_camera()
                st.rerun()
                return
        st.info(
            "采集已暂停 —— 点上方「▶ 开始实时采集」开始连续抓帧;"
            " 实验系统相机来自侧栏「CCD参数设置」的「bench_camera」配置."
        )
        return

    # 运行中: 取/建句柄。失败则空态点名侧栏与释放入口。
    cam = ensure_camera_handle(st, cam_type, cam_id)
    if cam is None:
        st.warning(
            f"无法打开实验系统相机 {cam_type} (id={cam_id}) —— 请在侧栏「CCD参数设置」"
            "核对型号与 ID; 若句柄被其他进程占用, 先释放再点「▶ 开始实时采集」重试。"
        )
        return

    # 曝光控制: 读生效值占位, 改完走 set_exposure_ms, 再回读显示。
    eff = get_camera_exposure_ms(cam)
    lo, hi = get_camera_exposure_range(cam)
    exposure_val = st.number_input(
        "曝光 (ms)",
        value=eff,
        min_value=lo,
        max_value=hi,
        step=0.1,
        key="ccd_live_exposure",
    )
    if st.button("应用曝光", key="ccd_live_apply_exposure"):
        apply_exposure_ms(cam, exposure_val)
    st.caption(
        f"当前生效曝光: {get_camera_exposure_ms(cam):.4g} ms；可设范围 [{lo:g}, {hi:g}] ms"
    )

    if st.button("关闭/释放相机", key="ccd_live_release"):
        close_bench_camera()
        st.rerun()
        return

    _mount_live_view(cam)
